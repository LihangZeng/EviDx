import json
from typing import Dict, Any, Union
import re

try:
    from .refiner_prompts import REFINER_SYSTEM_PROMPT
except ImportError:
    from refiner_prompts import REFINER_SYSTEM_PROMPT

class Refiner:
    def __init__(self, llm_client: Any, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
        self.system_prompt = REFINER_SYSTEM_PROMPT
        self.max_retries = 3

    def _clean_json_string(self, json_string: str) -> str:
        if "```" in json_string:
            pattern = r"```(?:json)?\s*(.*?)\s*```"
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                json_string = match.group(1)
            else:
                pattern = r"```(?:json)?\s*(.*?)\s*```"
                match = re.search(pattern, json_string, re.DOTALL)
                if match: json_string = match.group(1)
        return json_string.strip()    

    def refine_ehr(self, raw_text: str, draft_ehr_data: Union[Dict, str]) -> Dict[str, Any]:
        if isinstance(draft_ehr_data, dict):
            draft_json_str = json.dumps(draft_ehr_data, indent=2, ensure_ascii=False)
        else:
            draft_json_str = str(draft_ehr_data)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": f"Here is the data for the audit:\n\n"
                           f"--- RAW CLINICAL TEXT ---\n{raw_text}\n\n"
                           f"--- DRAFT EHR JSON ---\n{draft_json_str}"
            }
        ]

        # print(f"  [Refiner] Starting audit based on {self.model_name}...")
        current_try = 0
        last_error = None

        while current_try < self.max_retries:
            try:
                # print(f"  [Refiner] Attempt {current_try + 1}...")
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "response_format": {"type": "json_object"}, 
                    "temperature": 0.0,
                    "extra_body": {"enable_thinking": False}
                }

                try:
                    response = self.llm_client.chat.completions.create(**params)
                except Exception as e:
                    if "extra_body" in str(e) or "parameter" in str(e) or "thinking" in str(e):
                        print("[System] API does not support 'enable_thinking', retrying without it.")
                        if "extra_body" in params:
                            del params["extra_body"]
                        response = self.llm_client.chat.completions.create(**params)
                    else:
                        raise e
                
                content = response.choices[0].message.content

                cleaned_content = self._clean_json_string(content)

                refined_data = json.loads(cleaned_content)

                if isinstance(refined_data, dict) and "ehr_db" in refined_data and isinstance(refined_data["ehr_db"], dict):
                    refined_data = refined_data["ehr_db"]
                
                return refined_data

            except json.JSONDecodeError as e:
                current_try += 1
                last_error = e
                print(f"  [Refiner ⚠️] JSON Parse Error on attempt {current_try}: {e}")
                
                messages.append({"role": "assistant", "content": content}) 
                messages.append({
                    "role": "user", 
                    "content": f"SYSTEM ERROR: The extraction failed due to invalid JSON format. \nError details: {e}\n\nPlease output the JSON again, ensuring strict syntax compliance."
                })
            
            except Exception as e:
                print(f"  [Refiner ❌] Unknown Error: {e}")
                current_try += 1
                last_error = e

        print(f"  [Refiner 🚨] Failed after {self.max_retries} retries. Returning original draft.")
        if isinstance(draft_ehr_data, dict):
            return draft_ehr_data
        else:
            try:
                return json.loads(draft_json_str)
            except:
                return {"error": "Refiner failed and Draft was invalid JSON", "raw_draft": draft_json_str}