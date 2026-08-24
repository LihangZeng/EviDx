# Consultant/consultant.py
from typing import Dict, Any, List
import json
import re
from openai import OpenAI

from .rag_engine import RAGEngine
from .consultant_prompts import (
    DIFFERENTIAL_DIAGNOSIS_PROMPT, 
    CLINICAL_RISK_PROMPT,
    QUERY_DECOMPOSITION_PROMPT 
)
from .consultant_tools import CONSULTANT_TOOL_DEFINITIONS

class Consultant:
    def __init__(self, llm_client: OpenAI, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
        self.rag = RAGEngine()
        self.TOOL_DEFINITIONS = CONSULTANT_TOOL_DEFINITIONS

    def _safe_chat_completion(self, messages: List[Dict], temperature: float = 0.3) -> str:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e)
            if "enable_thinking" in error_str and "400" in error_str:
                print(f"  [Consultant Fix] Detected 'enable_thinking' error. Retrying with enable_thinking=False...")
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        stream=False,
                        extra_body={"enable_thinking": False} 
                    )
                    return response.choices[0].message.content
                except Exception as retry_e:
                    raise retry_e
            else:
                raise e

    def _generate_sub_queries(self, complex_query: str) -> List[str]:
        try:
            content = self._safe_chat_completion(
                messages=[{"role": "user", "content": QUERY_DECOMPOSITION_PROMPT.format(query=complex_query)}],
                temperature=0.1
            )
            content = re.sub(r"```json|```", "", content).strip()
            return json.loads(content)
        except Exception as e:
            print(f"  [Consultant Warning] Decomposition failed ({e}), using original query.")
            return [complex_query]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        query = arguments.get("query")
        if not query: return "Error: Query missing."

        print(f"  [Consultant] Processing Task: '{query}'")

        sub_queries = self._generate_sub_queries(query)
        print(f"  [Consultant] Thoughts: I need to search for -> {sub_queries}")

        all_retrieved_contexts = []
        
        for sub_q in sub_queries:
            context = self.rag.search_hybrid(sub_q, top_k=3) 
            if context:
                all_retrieved_contexts.append(f"=== Search Result for '{sub_q}' ===\n{context}")
        
        full_context = "\n\n".join(all_retrieved_contexts)
        
        if tool_name == "get_differential_diagnosis_criteria":
            system_prompt = DIFFERENTIAL_DIAGNOSIS_PROMPT
        elif tool_name == "analyze_clinical_risk":
            system_prompt = CLINICAL_RISK_PROMPT
        else:
            return f"Error: Unknown tool '{tool_name}'"

        full_user_message = system_prompt.format(
            query=query,
            context=full_context
        )

        try:
            answer = self._safe_chat_completion(
                messages=[{"role": "user", "content": full_user_message}],
                temperature=0.3
            )
            return f"【Consultant Report】\n{answer}"
            
        except Exception as e:
            return f"Error generating report: {e}"