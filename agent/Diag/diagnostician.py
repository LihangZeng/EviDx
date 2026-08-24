# Diag/diagnostician.py 
import json
from typing import List, Dict, Any, Optional, Union
import re
import mcp_lite  
import copy
from .diagnostician_prompts import DIAGNOSTICIAN_SYSTEM_PROMPT

class Diagnostician:

    def __init__(
        self,
        llm_client: Any,  
        model_name: str,  
        tool_definitions: List[Dict[str, Any]],
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_retries = 5
        self.system_prompt = DIAGNOSTICIAN_SYSTEM_PROMPT
        self._original_tool_definitions = tool_definitions
        self.update_available_tools(tool_definitions)

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.temporary_retry_message: Optional[Dict[str, Any]] = None

    def _normalize_tool_calls(self, tool_calls):
        out = []
        for c in tool_calls or []:
            if hasattr(c, "model_dump"):
                out.append(c.model_dump())
                continue
            if isinstance(c, dict):
                out.append(c)
                continue
            out.append({
                "id": getattr(c, "id", None),
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments
                }
            })
        return out

    def _call_llm(self) -> Any:
        messages_to_send = list(self.messages)
        if self.temporary_retry_message:
            messages_to_send.append(self.temporary_retry_message)

        params = {
            "model": self.model_name,
            "messages": messages_to_send,
            "temperature": 0.0, 
        }
        if self.tool_definitions:
            params["tools"] = self.tool_definitions
            params["tool_choice"] = "auto" 

        params["extra_body"] = {"enable_thinking": False}
        
        try:
            response = self.llm_client.chat.completions.create(**params)
            return response.choices[0].message
        except Exception as e:
            if "extra_body" in str(e) or "parameter" in str(e) or "thinking" in str(e):
                print("[Diagnostician] API does not support 'enable_thinking', retrying without it.")
                del params["extra_body"]
                response = self.llm_client.chat.completions.create(**params)
                return response.choices[0].message
            else:
                raise e    

    def _process_incoming_mcp_response(self, mcp_response_str: str) -> List[Dict[str, Any]]:
        try:
            parsed_data = json.loads(mcp_response_str)

            if isinstance(parsed_data, dict):
                response_list = [parsed_data]
            elif isinstance(parsed_data, list):
                response_list = parsed_data
            else:
                return [{"role": "user", "content": f"System Error: Invalid MCP response structure. Got {type(parsed_data)}"}]

            processed_messages = []

            for response_data in response_list:
                if not isinstance(response_data, dict):
                    raise ValueError(f"Invalid MCP response item: {response_data!r}")
                if "jsonrpc" not in response_data or "id" not in response_data:
                    raise ValueError(f"Invalid MCP JSON-RPC response: {response_data!r}")

                tool_call_id = response_data["id"]

                if "error" in response_data:
                    err = response_data["error"]
                    if isinstance(err, dict):
                        content = f"Tool Execution Error: {err.get('message', str(err))}"
                    else:
                        content = f"Tool Execution Error: {str(err)}"

                elif "result" in response_data:
                    result_block = response_data["result"]

                    if isinstance(result_block, dict) and "content" in result_block and isinstance(result_block["content"], list):
                        texts = []
                        for c in result_block["content"]:
                            if isinstance(c, dict) and "text" in c:
                                texts.append(c["text"])
                            else:
                                texts.append(str(c))
                        content = "\n".join(texts).strip()
                    elif isinstance(result_block, dict) and "text" in result_block:
                        content = str(result_block["text"])
                    else:
                        content = str(result_block)
                else:
                    raise ValueError("MCP response must contain either 'result' or 'error'.")

                processed_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content
                })

            return processed_messages
        except json.JSONDecodeError as exc:
            raise ValueError("MCP response is not valid JSON.") from exc

    def ingest_mcp_response(self, mcp_response_str: str) -> None:
        tool_result_msgs = self._process_incoming_mcp_response(mcp_response_str)
        self.messages.extend(tool_result_msgs)

    def start_new_case(self, initial_narrative: str, tracker: Any = None) -> str:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.messages.append({"role": "user", "content": initial_narrative})
        return self.step(tracker=tracker)

    def step(self, incoming_mcp_message: Optional[str] = None, tracker: Any = None) -> str:
        if incoming_mcp_message:
            try:
                tool_result_msgs = self._process_incoming_mcp_response(incoming_mcp_message)
                self.messages.extend(tool_result_msgs)
            except Exception as exc:
                raise RuntimeError("Failed to process the incoming MCP response.") from exc

        retry_count = 0
        self.temporary_retry_message = None
        
        while retry_count < self.max_retries:
            llm_response_msg = self._call_llm()

            content = llm_response_msg.content or ""
            tool_calls = getattr(llm_response_msg, 'tool_calls', None)

            if tool_calls and len(tool_calls) > 0:
                self.messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": self._normalize_tool_calls(tool_calls)
                })
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": content
                })
            
            display_content = content
            if "<think>" in content:
                 display_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                 
            print(f"\n--- [Diagnostician Step {len(self.messages)//2}] ---")
            if display_content:
                print(f"💭 Thought: {display_content}")

            if tool_calls and len(tool_calls) > 0:
                mcp_requests = []
                bad_args = False

                for call in tool_calls:
                    func_name = call.function.name
                    call_id = call.id
                    args_str = call.function.arguments

                    try:
                        args_dict = json.loads(args_str)
                        req = mcp_lite.create_tool_call_request(
                            tool_name=func_name,
                            arguments=args_dict,
                            request_id=call_id
                        )
                        mcp_requests.append(req)
                        print(f"🛠️  Call: {func_name}")

                    except json.JSONDecodeError as e:
                        bad_args = True
                        print(f"❌ JSON Error in arguments for {func_name}")
                        print(f"   [Error Details]: {e}")
                        
                        if tracker:
                            tracker.record_syntax_error()

                        self.messages.pop() 
                        
                        if self.messages and self.messages[-1]["role"] == "tool":
                            self.messages.append({
                                "role": "assistant",
                                "content": "[System] Tool result acknowledged. Retrying due to JSON parse error."
                            })
                            
                        self.temporary_retry_message = {
                            "role": "user",
                            "content": f"Error: Invalid JSON arguments for {func_name}. Please output valid JSON arguments."
                        }
                        break   

                if bad_args:
                    retry_count += 1
                    continue

                if mcp_requests:
                    return json.dumps(mcp_requests)    
            
            else:
                print("⚠️  No tool call detected. Retrying with strict tool requirement...")
                
                if tracker:
                    tracker.record_schema_error() 
                    
                self.messages.pop() 
                
                if self.messages and self.messages[-1]["role"] == "tool":
                    self.messages.append({
                        "role": "assistant",
                        "content": "[System] Tool result acknowledged. Retrying with strict tool requirement."
                    })
                    
                self.temporary_retry_message = {
                    "role": "user",
                    "content": (
                        "SYSTEM: You MUST respond ONLY with MCP tool calls (method='tools/call'). "
                        "Do NOT output free text. If you are ready to conclude, call the `finish` tool."
                    )
                }
                retry_count += 1
                continue

        return "[]"

    def answer_multiple_choice(
        self,
        final_exam_prompt: str,
        examiner: Any = None,
        consultant: Any = None,
        max_steps: int = 6
    ) -> str:
        exam_messages = list(self.messages)
        exam_messages.append({"role": "user", "content": final_exam_prompt})
        tools_enabled = (examiner is not None) or (consultant is not None)

        for step_i in range(max_steps):
            params = {
                "model": self.model_name,
                "messages": exam_messages,
                "temperature": 0.0,
                "extra_body": {"enable_thinking": False}
            }
            if tools_enabled and self.tool_definitions:
                params["tools"] = self.tool_definitions
                params["tool_choice"] = "auto"

            try:
                resp = self.llm_client.chat.completions.create(**params)
            except Exception as e:
                if "extra_body" in str(e) or "parameter" in str(e):
                    if "extra_body" in params:
                        del params["extra_body"]
                    resp = self.llm_client.chat.completions.create(**params)
                else:
                    return f"Error answering multiple choice question: {e}"

            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls and len(tool_calls) > 0 and content:
                exam_messages.append({
                    "role": "user",
                    "content": "SYSTEM: When using tool calls, you MUST NOT output any free text. Put reasoning into _observer_metadata."
                })
                continue

            if tool_calls and len(tool_calls) > 0:
                exam_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": self._normalize_tool_calls(tool_calls)
                })

                for call in tool_calls:
                    name = call.function.name
                    call_id = call.id
                    args_str = call.function.arguments or "{}"
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}

                    if name in ("initialize_environment", "get_available_data_menu", "finish"):
                        tool_text = f"SYSTEM INTERLOCK: Tool `{name}` is not allowed in exam stage."
                        exam_messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_text})
                        continue

                    tool_text = None
                    if examiner is not None and any(t["function"]["name"] == name for t in examiner.TOOL_DEFINITIONS):
                        tool_text = examiner.execute_tool(name, args)
                    elif consultant is not None and any(t["function"]["name"] == name for t in consultant.TOOL_DEFINITIONS):
                        tool_text = consultant.execute_tool(name, args)
                    else:
                        tool_text = f"Error: Unknown/Unavailable tool `{name}` in exam stage."

                    exam_messages.append({"role": "tool", "tool_call_id": call_id, "content": str(tool_text)})
                continue

            if content:
                return content

            exam_messages.append({"role": "user", "content": "SYSTEM: Please output the final answer JSON now."})

        return "Error: Exam stage exceeded max_steps without final answer."

    def update_available_tools(self, new_tool_definitions: List[Dict[str, Any]]):
        injected_tools = []
        thought_schema = {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "description": "Current leading hypothesis (e.g., 'Suspect Acute MI'). Used for Faithfulness check."
                },
                "differential_probs": {
                    "type": "object",
                    "description": "Dictionary of top diagnoses and their estimated probabilities (e.g., {'MI': 0.8, 'GERD': 0.2}). Used for Entropy check.",
                    "additionalProperties": {"type": "number"}
                },
                "reasoning": {
                    "type": "string", 
                    "description": "Brief reasoning for why this specific tool/action is chosen."
                }
            },
            "required": ["hypothesis", "differential_probs", "reasoning"]
        }

        for tool in new_tool_definitions:
            new_tool = copy.deepcopy(tool)
            func_def = new_tool.get("function", {})
            func_name = func_def.get("name", "")
            if func_name in ["initialize_environment", "finish", "get_available_data_menu"]:
                injected_tools.append(new_tool)
                continue

            params = func_def.get("parameters", {})
            if not params:
                params = {"type": "object", "properties": {}, "required": []}
                func_def["parameters"] = params

            props = params.get("properties", {})
            required = params.get("required", [])

            props["_observer_metadata"] = thought_schema
            if "_observer_metadata" not in required:
                required.append("_observer_metadata")
            
            new_tool["function"]["parameters"]["properties"] = props
            new_tool["function"]["parameters"]["required"] = required
            injected_tools.append(new_tool)

        self.tool_definitions = injected_tools
