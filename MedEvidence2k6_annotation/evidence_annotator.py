from __future__ import annotations

import argparse
import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from rag_engine import RAGEngine

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
MODEL = os.getenv("MODEL") or os.getenv("LLM_MODEL_NAME")

TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))
RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
RESP_URL = f"{BASE_URL}/responses"

class EvidenceAnnotator:
    def __init__(self):
        print("🤖 [Init] Initializing EvidenceAnnotator with requests...")
        
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        
        self.rag = RAGEngine()
        print(f"✅ [Init] System Ready. Target Model: {MODEL}")

    def _post_responses(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None

        for i in range(RETRIES):
            try:
                r = self.session.post(RESP_URL, headers=self.headers, json=payload, timeout=TIMEOUT)

                if r.status_code < 200 or r.status_code >= 300:
                    try:
                        err_json = r.json()
                    except Exception:
                        err_json = {"raw": r.text}

                    if r.status_code >= 500:
                        wait = 0.8 * (2 ** i)
                        print(f"[WARN] HTTP {r.status_code}: retry in {wait:.1f}s ...")
                        time.sleep(wait)
                        continue

                    raise RuntimeError(f"HTTP {r.status_code} error: {err_json}")

                return r.json()

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_err = e
                wait = 0.8 * (2 ** i)
                print(f"[WARN] {type(e).__name__}: retry in {wait:.1f}s ...")
                time.sleep(wait)

        raise RuntimeError(f"Request failed after {RETRIES} retries. Last error: {last_err}")

    def _extract_output_text(self, resp: Dict[str, Any]) -> str:
        if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
            return resp["output_text"].strip()

        out = []
        for item in resp.get("output", []) or []:
            if item.get("type") == "message" and item.get("role") == "assistant":
                for part in item.get("content", []) or []:
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        out.append(text.strip())
        return "\n".join(out).strip()

    def _parse_json_from_text(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            cleaned = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
            return json.loads(cleaned)

    def _step1_plan_search(self, case_text: str, ground_truth: str) -> List[str]:
        system_prompt = """You are a Senior Medical Researcher.
Analyze the Clinical Case and the Diagnosis (Ground Truth).
Generate 2-3 targeted search queries to:
1. Find diagnostic criteria for the Ground Truth.
2. Verify connections between specific case symptoms and the diagnosis.
3. Distinguish from other likely conditions.

Output strictly valid JSON format: {"queries": ["query1", "query2"]}"""

        user_content = f"=== CLINICAL CASE ===\n{case_text[:1500]}...\n\n=== DIAGNOSIS ===\n{ground_truth}"

        payload = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_output_tokens": 512
        }

        try:
            resp = self._post_responses(payload)
            text = self._extract_output_text(resp)
            data = self._parse_json_from_text(text)
            queries = data.get("queries", [])
            if not queries: raise ValueError("Empty queries list")
            return queries
        except Exception as e:
            return [f"Diagnostic criteria and clinical presentation of {ground_truth}"]

    def _step2_execute_search(self, queries: List[str]) -> str:
        aggregated_context = []
        unique_queries = list(set(queries))
        
        for q in unique_queries:
            result_text = self.rag.search_hybrid(q, top_k=2)
            aggregated_context.append(f"### Search Query: '{q}'\n{result_text}")
        
        return "\n\n".join(aggregated_context)

    def _step3_annotate_evidence(self, case_text: str, ground_truth: str, context: str) -> Dict:
        system_prompt = """You are an Expert Medical Annotator.
Identify "Golden Evidence" in the clinical case that supports the Ground Truth Diagnosis.
You MUST base your reasoning on the provided Reference Context.

### OUTPUT SCHEMA (JSON)
{
  "golden_evidence": [
    {
      "original_text": "Exact substring from case text",
      "feature_name": "Standardized medical term",
      "type": "Inclusion" | "Exclusion" | "Differentiation",
      "textbook_reference": "Quote or summary from context",
      "reasoning": "Why this evidence matters."
    }
  ]
}

### RULES
1. `original_text` MUST be an EXACT copy (substring) from the Clinical Case.
2. Output strictly valid JSON.
"""
        user_content = f"=== GROUND TRUTH ===\n{ground_truth}\n\n=== REFERENCE CONTEXT ===\n{context}\n\n=== CLINICAL CASE ===\n{case_text}"
        
        payload = {
            "model": MODEL,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_output_tokens": 2048
        }
        
        try:
            resp = self._post_responses(payload)
            text = self._extract_output_text(resp)
            return self._parse_json_from_text(text)
        except Exception as e:
            print(f"⚠️ [Annotate Error] {e}")
            return {"golden_evidence": []}

    def process_file(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"\n📂 Processing: {input_path} -> {output_path}")

        output_parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_parent, exist_ok=True)
        
        error_path = output_path.replace(".jsonl", "_errors.jsonl")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = len(lines)
        dataset_name = os.path.basename(input_path).split('.')[0]

        for line in tqdm(lines, desc=f"Annotating {dataset_name}", total=total):
            try:
                item = json.loads(line)

                case_text = item.get("question") or item.get("case_text") or item.get("text")
                gt = item.get("answer") or item.get("ground_truth") or item.get("correct_answer")

                if not gt and "options" in item and "label" in item:
                     gt = item["options"].get(item["label"], item["label"])

                if not case_text or not gt:
                    continue

                queries = self._step1_plan_search(case_text, gt)

                context = self._step2_execute_search(queries)

                evidence = self._step3_annotate_evidence(case_text, gt, context)

                result_item = {
                    "original_id": item.get("id"),
                    "dataset": dataset_name,
                    "ground_truth_text": gt,
                    "question_text": case_text,
                    "annotation_meta": {
                        "search_queries": queries,
                        "rag_context_snippet": context[:200] + "..." 
                    },
                    "golden_evidence": evidence.get("golden_evidence", [])
                }
                
                with open(output_path, 'a', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"❌ Error processing line (saved to error log): {e}")
                
                try:
                    error_record = {
                        "error_message": str(e),
                        "timestamp": time.time(),
                        "original_line": line.strip()
                    }
                    with open(error_path, 'a', encoding='utf-8') as err_f:
                        err_f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                except Exception as write_err:
                    print(f"❌ Failed to write to error log: {write_err}")
                
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate diagnostic evidence in a JSONL dataset.")
    parser.add_argument("--input-file", required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output-file", required=True, help="Path to the annotated JSONL file.")
    args = parser.parse_args()

    missing_config = [
        name
        for name, value in {
            "OPENAI_API_KEY, API_KEY, or LLM_API_KEY": API_KEY,
            "OPENAI_BASE_URL or LLM_BASE_URL": BASE_URL,
            "MODEL or LLM_MODEL_NAME": MODEL,
        }.items()
        if not value
    ]
    if missing_config:
        parser.error(f"Missing required configuration: {', '.join(missing_config)}")

    annotator = EvidenceAnnotator()

    print("🚀 Starting Batch Annotation (using requests)...")
    annotator.process_file(args.input_file, args.output_file)
    
    print("\n✅ All tasks completed.")
