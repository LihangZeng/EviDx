import json
import os
import re
from typing import Tuple, Dict, Any, Optional

try:
    from .synthesizer_prompts import SYNTHESIZER_SYSTEM_PROMPT
    from .refiner import Refiner
except ImportError:
    from synthesizer_prompts import SYNTHESIZER_SYSTEM_PROMPT
    from refiner import Refiner

def extract_think_and_json(text: str) -> tuple[str, str]:
    reasoning = ""
    json_content = text

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        json_content = text.replace(think_match.group(0), "")

    code_match = re.search(r"```(?:json)?\s*(.*)\s*```", json_content, re.DOTALL)
    if code_match:
        json_content = code_match.group(1)
    
    json_content = json_content.strip()
    
    return reasoning, json_content

class NarrativeSynthesizer:
    
    def __init__(self, llm_client, model_name, max_retries: int = 3):
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_retries = max_retries

        print(f"[Synthesizer] Initializing internal Refiner submodule...")
        self.refiner = Refiner(llm_client, model_name)

    def _generate_draft_with_retry(self, raw_data_content: str, dataset_hints: str) -> Tuple[Dict[str, Any], str]:
        messages = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"--- Raw Data Input ({dataset_hints}) ---\n{raw_data_content}"}
        ]

        current_try = 0
        
        while current_try < self.max_retries:
            try:
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
                
                raw_content = response.choices[0].message.content
                
                reasoning_trace, cleaned_json_str = extract_think_and_json(raw_content)
                
                draft_data = json.loads(cleaned_json_str)
                
                return draft_data, reasoning_trace

            except json.JSONDecodeError as e:
                current_try += 1
                print(f"  [Synthesizer ⚠️] JSON Parse Error (Drafting) on attempt {current_try}: {e}")
                
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user", 
                    "content": f"SYSTEM ERROR: Your previous output was not valid JSON.\nError details: {e}\n\nPlease fix the JSON syntax and output ONLY the JSON object."
                })
                
            except Exception as e:
                print(f"  [Synthesizer ❌] Unexpected Error during drafting: {e}")
                return {}, ""

        print(f"  [Synthesizer 🚨] Max retries reached for Drafting. Returning empty structure.")
        return {}, ""

    def synthesize_case(self, raw_data_content: str, dataset_hints: str = "") -> Tuple[str, Dict, str, Dict, str, str]:
        print(f"\n[Synthesizer] Processing Case... (Model: {self.model_name})")
        
        draft_json, reasoning_trace = self._generate_draft_with_retry(raw_data_content, dataset_hints)
        
        if not draft_json:
            return "Error generating narrative.", {}, None, {}, ""

        draft_ehr = draft_json.get("ehr_db", {})
        
        if draft_ehr:
            print(f"  [Synthesizer] Draft generated. Invoking internal Refiner for audit...")
            refined_ehr = self.refiner.refine_ehr(raw_data_content, draft_ehr)
        else:
            print(f"  [Synthesizer] Draft EHR is empty, skipping Refiner.")
            refined_ehr = {}

        initial_narrative = draft_json.get("initial_narrative_summary", "Patient presents for evaluation.")
        clinical_question = draft_json.get("clinical_question", None)
        answer_choices = draft_json.get("answer_choices", {})
        question_focus = draft_json.get("question_focus", "diagnosis")

        if "ehr_db" in refined_ehr and isinstance(refined_ehr["ehr_db"], dict):
            refined_ehr = refined_ehr["ehr_db"]

        if reasoning_trace:
            print(f"  [Synthesizer] CoT captured ({len(reasoning_trace)} chars).")

        return initial_narrative, refined_ehr, clinical_question, answer_choices, reasoning_trace, question_focus
        