import os
import json
import re
import asyncio
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("LLM_API_KEY") 
BASE_URL = os.getenv("LLM_BASE_URL")
JUDGE_MODEL = os.getenv("EVIDENCE_JUDGE_MODEL") or os.getenv("LLM_MODEL_NAME")
ACQ_JUDGE_MODEL = os.getenv("ACQUISITION_JUDGE_MODEL") or JUDGE_MODEL

client: Optional[AsyncOpenAI] = None

def normalize_text(text: str) -> str:
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def extract_json_content(text: str) -> str:
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
        if match: return match.group(1)
    return text

def normalize_id(raw_id: Any) -> str:
    if raw_id is None:
        return ""
    if isinstance(raw_id, float) and raw_id.is_integer():
        return str(int(raw_id))
    s = str(raw_id).strip()
    return s

def normalize_dataset_name(raw_name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(raw_name or "").lower())

def datasets_match(trace_dataset: Any, candidate_names: List[str]) -> bool:
    trace_name = normalize_dataset_name(trace_dataset)
    if not trace_name:
        return False
    for candidate in candidate_names:
        candidate_name = normalize_dataset_name(candidate)
        if candidate_name and (
            trace_name == candidate_name
            or trace_name in candidate_name
            or candidate_name in trace_name
        ):
            return True
    return False

class TraceParser:
    @staticmethod
    def extract_agent_thoughts(trace_steps: List[Dict]) -> List[str]:
        thoughts = []
        for step in trace_steps:
            step_idx = step.get("step_index", "?")
            raw_reasoning = step.get("agent_reasoning", "")
            
            parsed_reasoning = None
            if isinstance(raw_reasoning, str) and (raw_reasoning.strip().startswith("[") or raw_reasoning.strip().startswith("{")):
                try:
                    parsed = json.loads(raw_reasoning)
                    if isinstance(parsed, list): parsed_reasoning = parsed
                    elif isinstance(parsed, dict): parsed_reasoning = [parsed]
                except json.JSONDecodeError: pass

            found_structured_thought = False
            
            if parsed_reasoning:
                for call in parsed_reasoning:
                    args = {}
                    if "params" in call: args = call["params"].get("arguments", {})
                    elif "arguments" in call: args = call["arguments"]
                    
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except: args = {}

                    meta_content = args.get("_observer_metadata") if isinstance(args, dict) else None
                    if meta_content:
                        meta_dict = {}
                        if isinstance(meta_content, str):
                            try: meta_dict = json.loads(meta_content)
                            except: pass
                        elif isinstance(meta_content, dict):
                            meta_dict = meta_content
                        
                        if "reasoning" in meta_dict:
                            thoughts.append(f"[Step {step_idx} Reasoning] {meta_dict['reasoning']}")
                            found_structured_thought = True
                        elif "hypothesis" in meta_dict:
                            thoughts.append(f"[Step {step_idx} Hypothesis] {meta_dict['hypothesis']}")
                            found_structured_thought = True

            if not found_structured_thought and raw_reasoning:
                if isinstance(raw_reasoning, str) and not raw_reasoning.strip().startswith("[{"):
                     thoughts.append(f"[Step {step_idx} Raw] {str(raw_reasoning)}")

        return thoughts

    @staticmethod
    def aggregate_tool_outputs(trace_steps: List[Dict]) -> str:
        outputs = []
        for step in trace_steps:
            out = step.get("tool_output")
            if out: outputs.append(str(out))
        return "\n".join(outputs)

class EvidenceEvaluator:
    
    @staticmethod
    async def check_acquisition_advanced(evidence: Dict, all_inputs_str: str) -> Dict:
        original_text = evidence.get("original_text", "")
        feature_name = evidence.get("feature_name", "")
        
        norm_inputs = normalize_text(all_inputs_str)
        norm_text = normalize_text(original_text)
        norm_feature = normalize_text(feature_name)
        
        if norm_text in norm_inputs:
            return {"acquired": True, "method": "Rule-Exact"}
            
        if len(feature_name.split()) > 1 or len(feature_name) > 6:
            if norm_feature in norm_inputs:
                return {"acquired": True, "method": "Rule-Feature"}

        truncated_context = all_inputs_str[:20000] 
        
        prompt = f"""
### Task
Determine if the specific **Medical Evidence** described below is present in the provided **Text Data** (which contains patient history, tool outputs, etc.).
Be flexible with phrasing, formatting, and abbreviations (e.g., "NG tube" = "nasogastric tube").

### Target Evidence
- Raw Text: "{original_text}"
- Clinical Concept: "{feature_name}"

### Text Data (What the Agent saw)
{truncated_context}

### Question
Does the Text Data contain the information described in the Target Evidence?
Answer JSON: {{"present": true, "reason": "short explanation"}} or {{"present": false}}
"""
        try:
            response = await client.chat.completions.create(
                model=ACQ_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res = json.loads(extract_json_content(response.choices[0].message.content))
            is_present = res.get("present", False)
            return {"acquired": is_present, "method": "LLM-Semantic" if is_present else "Missed"}
        except Exception as e:
            # print(f"Acq LLM Error: {e}")
            return {"acquired": False, "method": "Error"}

    @staticmethod
    async def check_cognition_llm(evidence: Dict, agent_thoughts: str, final_report: str) -> Dict:
        original_text = evidence.get("original_text", "")
        feature_name = evidence.get("feature_name", "")
        ev_type = evidence.get("type", "Evidence")
        
        prompt = f"""
### Role
You are a strict Medical Logic Evaluator.

### Task
Determine if the AI Physician explicitly **considered and used** a specific piece of "Golden Evidence" during its diagnostic reasoning process.

### Golden Evidence
- Raw Text: "{original_text}"
- Concept: "{feature_name}"
- Role: {ev_type}

### Agent's Thinking Process
{agent_thoughts}

### Agent's Final Report
{final_report}

### Evaluation Criteria
1. **Hit**: The agent explicitly mentions the evidence (or semantic equivalent) AND uses it to support/refute a hypothesis.
2. **Miss**: The agent ignores this evidence or only mentions it without linking it to the diagnosis.

### Output Format
Return ONLY a JSON object:
{{
  "recalled": true, 
  "reason": "Brief explanation."
}}
"""
        try:
            response = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(extract_json_content(response.choices[0].message.content))
        except Exception as e:
            return {"recalled": False, "reason": f"Judge Error: {str(e)}"}

async def evaluate_single_case(trace: Dict, golden: Dict) -> Dict:
    case_id = trace.get("case_id")
    
    steps = trace.get("steps", [])
    agent_thoughts_list = TraceParser.extract_agent_thoughts(steps)
    agent_thoughts_str = "\n".join(agent_thoughts_list)
    
    acq_context_parts = []
    if trace.get("initial_narrative"):
        acq_context_parts.append(f"=== INITIAL NARRATIVE ===\n{trace['initial_narrative']}")
    acq_context_parts.append(f"=== TOOL OUTPUTS ===\n{TraceParser.aggregate_tool_outputs(steps)}")
    all_inputs_str = "\n\n".join(acq_context_parts)
    
    final_report = json.dumps(trace.get("openqa_report", {}), ensure_ascii=False)
    if not final_report or len(final_report) < 10:
        final_report = str(trace.get("final_reasoning", ""))

    golden_ev_list = golden.get("golden_evidence", [])
    if not golden_ev_list:
        return {
            "case_id": case_id, "dataset": trace.get("dataset"),
            "total_evidence": 0, "acquired_count": 0, "cognition_hit_count": 0,
            "recall_score": 0.0, "acquisition_score": 0.0, "details": []
        }

    acq_tasks = [EvidenceEvaluator.check_acquisition_advanced(ev, all_inputs_str) for ev in golden_ev_list]
    cog_tasks = [EvidenceEvaluator.check_cognition_llm(ev, agent_thoughts_str, final_report) for ev in golden_ev_list]
    
    acq_results = await asyncio.gather(*acq_tasks)
    cog_results = await asyncio.gather(*cog_tasks)

    evidence_results = []
    hit_count = 0
    acquired_count = 0
    
    for i, ev in enumerate(golden_ev_list):
        acq_res = acq_results[i]
        cog_res = cog_results[i]
        
        is_acquired = acq_res["acquired"]
        is_used = cog_res.get("recalled", False)
        
        if is_acquired: acquired_count += 1
        if is_used: hit_count += 1
        
        evidence_results.append({
            "feature_name": ev.get("feature_name"),
            "original_text": ev.get("original_text"),
            "type": ev.get("type"),
            "status_acquired": is_acquired,
            "status_used": is_used,
            "acq_method": acq_res.get("method"), 
            "judge_reason": cog_res.get("reason")
        })

    total_ev = len(golden_ev_list)
    return {
        "case_id": case_id,
        "dataset": trace.get("dataset"),
        "total_evidence": total_ev,
        "acquired_count": acquired_count,
        "cognition_hit_count": hit_count,
        "recall_score": (hit_count / total_ev) if total_ev > 0 else 0.0,
        "acquisition_score": (acquired_count / total_ev) if total_ev > 0 else 0.0,
        "details": evidence_results
    }

async def main():
    global client
    parser = argparse.ArgumentParser(description="Offline Evidence Recall Evaluator V2")
    parser.add_argument("--trace_file", type=str, required=True, help="Path to inference_traces.jsonl")
    parser.add_argument("--golden_files", type=str, nargs='+', required=True, help="Path(s) to annotated golden .jsonl files")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save results")
    args = parser.parse_args()

    missing_config = [
        name
        for name, value in {
            "LLM_API_KEY": API_KEY,
            "EVIDENCE_JUDGE_MODEL or LLM_MODEL_NAME": JUDGE_MODEL,
        }.items()
        if not value
    ]
    if missing_config:
        parser.error(f"Missing required configuration: {', '.join(missing_config)}")

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Loading Golden Data from {len(args.golden_files)} files...")
    golden_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for g_file in args.golden_files:
        source_path = Path(g_file)
        with open(g_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    raw_id = item.get("original_id") or item.get("id")
                    norm_id = normalize_id(raw_id)
                    if not norm_id:
                        continue
                    candidate = {
                        "item": item,
                        "dataset_names": [
                            item.get("dataset", ""),
                            source_path.parent.name,
                            source_path.stem,
                        ],
                    }
                    golden_by_id.setdefault(norm_id, []).append(candidate)
                except (json.JSONDecodeError, TypeError):
                    continue
    golden_count = sum(len(candidates) for candidates in golden_by_id.values())
    collision_count = sum(1 for candidates in golden_by_id.values() if len(candidates) > 1)
    print(f"Loaded {golden_count} golden cases across {len(golden_by_id)} unique IDs.")
    if collision_count:
        print(f"Detected {collision_count} IDs shared by multiple golden cases; dataset-aware matching is enabled.")

    print(f"Loading Traces from {args.trace_file}...")
    traces = []
    with open(args.trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, str): data = json.loads(data)
                traces.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
    print(f"Loaded {len(traces)} traces.")

    eval_tasks = []
    matched_traces = []
    ambiguous_count = 0
    
    for trace in traces:
        tid_raw = trace.get("case_id")
        tid_norm = normalize_id(tid_raw)
        candidates = golden_by_id.get(tid_norm, [])
        if not candidates:
            continue

        if len(candidates) == 1:
            selected = candidates[0]
        else:
            dataset_candidates = [
                candidate
                for candidate in candidates
                if datasets_match(trace.get("dataset"), candidate["dataset_names"])
            ]
            if len(dataset_candidates) != 1:
                ambiguous_count += 1
                continue
            selected = dataset_candidates[0]

        matched_traces.append(trace)
        eval_tasks.append(evaluate_single_case(trace, selected["item"]))

    if ambiguous_count:
        print(f"Skipped {ambiguous_count} traces with ambiguous cross-dataset IDs.")

    print(f"Starting V2 Evaluation for {len(matched_traces)} matched cases...")
    
    if not matched_traces:
        print("No matches found. Check IDs.")
        return

    results = await tqdm_asyncio.gather(*eval_tasks)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    detail_path = os.path.join(args.output_dir, f"eval_details_v2_{timestamp}.jsonl")
    with open(detail_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    summary_data = []
    for res in results:
        summary_data.append({
            "CaseID": res["case_id"],
            "Total_Evidence": res["total_evidence"],
            "Acquired": res["acquired_count"],
            "Cognition_Hit": res["cognition_hit_count"],
            "Acq_Rate": res["acquisition_score"],
            "Cog_Recall": res["recall_score"]
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(args.output_dir, f"eval_summary_v2_{timestamp}.csv")
    
    avg_recall = df['Cog_Recall'].mean() if not df.empty else 0
    avg_acq = df['Acq_Rate'].mean() if not df.empty else 0

    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*40)
    print(f"V2 Evaluation Complete!")
    print(f"Global Average Recall (Cognition): {avg_recall:.2%}")
    print(f"Global Average Acquisition (Physical): {avg_acq:.2%}")
    print(f"Details saved to: {detail_path}")
    print(f"Summary saved to: {csv_path}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
