import os
import argparse
import json
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import mcp_lite
from Synth.synthesizer import NarrativeSynthesizer
from Exam.examiner import Examiner
from Diag.diagnostician import Diagnostician
from Diag.diagnostician_tools import (
    INIT_TOOL_DEF,
    FINISH_TOOL_DEF,
    SUBMIT_MCQ_TOOL_DEF,
    TOOL_DEFINITIONS as DIAG_TOOL_DEFINITIONS,
)
from Consult.consultant import Consultant
from Judge.judge import Judge
from Observe.observer import ObserverAgent
from Observe.metrics import MetricTracker 

class CaseTrace:
    def __init__(self, case_id, ground_truth, dataset):
        self.case_id = case_id
        self.ground_truth = ground_truth
        self.dataset = dataset
        self.steps = []
        self.final_prediction = None
        self.final_reasoning = None
        self.openqa_report = None

    def add_step(self, step_idx, tool_name, tool_params, tool_result, reasoning):
        self.steps.append({
            "step_index": step_idx,
            "action_type": "tool_call",
            "tool_name": tool_name,
            "tool_params": tool_params,
            "tool_output": tool_result, 
            "agent_reasoning": reasoning, 
            "timestamp": datetime.now().isoformat()
        })

    def record_openqa(self, report):
        self.openqa_report = report

    def finalize(self, prediction, reasoning):
        self.final_prediction = prediction
        self.final_reasoning = reasoning
    
    def to_dict(self):
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "ground_truth": self.ground_truth,
            "final_prediction": self.final_prediction,
            "final_reasoning": self.final_reasoning,
            "openqa_report": self.openqa_report,
            "steps": self.steps
        }

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL_NAME")

judge_api_key = os.getenv("JUDGE_API_KEY")
judge_base_url = os.getenv("JUDGE_BASE_URL")
judge_model_name = os.getenv("JUDGE_MODEL_NAME")

judge_api_key = judge_api_key or api_key
judge_base_url = judge_base_url or base_url
judge_model_name = judge_model_name or MODEL_NAME

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8000/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "EMPTY")
EMBEDDING_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "qwen3-embedding")

llm_client = None
judge_client = None
embed_client = None


def extract_json_content(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()


def _is_mcp_call(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and x.get("method") == "tools/call"
        and isinstance(x.get("params"), dict)
        and "name" in x["params"]
    )

def validate_tool_schema(tool_name, tool_args, allowed_definitions):
    target_def = next((t for t in allowed_definitions if t["function"]["name"] == tool_name), None)
    
    if not target_def:
        return False, f"Tool '{tool_name}' is not in the allowed list."
    
    required_params = target_def["function"]["parameters"].get("required", [])
    missing = [p for p in required_params if p not in tool_args]
    if missing:
        return False, f"Missing required arguments for '{tool_name}': {missing}"
        
    return True, None


def run_clinical_case_orchestration(
    raw_data_str: str,
    dataset_source: str,
    diagnostician: Diagnostician,
    synthesizer: NarrativeSynthesizer,
    examiner: Examiner,
    consultant: Consultant,
    judge_agent: Judge,
    case_data: Dict[str, Any],
    trace_file_path: str 
):
    
    record = {
        "id": case_data.get("id"),
        "dataset": dataset_source,
        "label": case_data.get("label") or case_data.get("answer_idx"),
        "status": "running",
    }
    
    current_trace = CaseTrace(
        case_id=record["id"],
        ground_truth=record["label"],
        dataset=dataset_source
    )

    tracker = MetricTracker(
        case_id=record["id"], 
        ground_truth_evidence=(
            case_data.get("evidence_list")
            or case_data.get("golden_evidence")
            or []
        )
    )

    tool_call_count = 0
    observer_blocks = 0
    print(
        f"\n{'='*60}\n[Orchestrator] >>> START (Source: {dataset_source}) <<<\n{'='*60}"
    )

    TOOLS_PHASE_0 = [INIT_TOOL_DEF]

    menu_tool = next(
        (t for t in examiner.TOOL_DEFINITIONS if t["function"]["name"] == "get_available_data_menu"),
        None,
    )
    if menu_tool is None:
        raise RuntimeError("Examiner missing tool: get_available_data_menu")

    TOOLS_PHASE_1 = [menu_tool]
    TOOLS_PHASE_2 = [FINISH_TOOL_DEF] + examiner.TOOL_DEFINITIONS + consultant.TOOL_DEFINITIONS

    print("[Orchestrator] 🔒 Phase 0: Init...")
    diagnostician.update_available_tools(TOOLS_PHASE_0)

    observer_agent: Optional[ObserverAgent] = None

    start_signal = (
        "SYSTEM_READY_FOR_NEW_CASE\n"
        "ACTION REQUIRED: You MUST execute the tool `initialize_environment` IMMEDIATELY to decrypt and load the Initial Narrative.\n"
        "DO NOT output a diagnosis yet. DO NOT finish. Initialize first."
    )

    current_step_output = diagnostician.start_new_case(start_signal, tracker=tracker)

    step_count = 0
    max_steps = 50
    extracted_choices = {}
    recent_tool_calls = []

    current_probs: Dict[str, float] = {}
    current_hypothesis = "Initial Hypothesis"

    initialized = False
    menu_called = False

    def finalize_metrics_and_record(final_status, openqa_steps, mcqa_steps=0):
        tracker.set_steps(openqa_steps, mcqa_steps)
        if observer_agent:
            tracker.calculate_evidence_recall(observer_agent.s_acquired)
        final_metrics = tracker.finalize()
        record["metrics"] = final_metrics
        record["status"] = final_status
        
        record.update({
             "tool_call_count": tool_call_count,
             "observer_blocks": observer_blocks,
        })
        if observer_agent:
            record["observer_debug"] = {
                "s_acquired_count": len(getattr(observer_agent, "s_acquired", [])),
                "H_last": observer_agent.history_h[-1] if getattr(observer_agent, "history_h", []) else None
            }
            
        append_jsonl(trace_file_path, current_trace.to_dict())
        
        return record

    while step_count < max_steps:
        step_count += 1
        print(f"\n--- [Step {step_count}] ---")
        
        tracker.record_attempt()

        step_reasoning_raw = current_step_output or ""
        if "<think>" in (current_step_output or ""):
            display_output = re.sub(r"<think>.*?</think>", "", current_step_output, flags=re.DOTALL).strip()
        else:
            display_output = current_step_output

        print("[Diagnostician Reasoning]:")
        print("-" * 40)
        print((display_output or "").strip())
        print("-" * 40 + "\n")

        request_list = []
        parsing_error = None
        
        try:
            cleaned_output = extract_json_content(current_step_output)
            parsed_json = json.loads(cleaned_output)

            if isinstance(parsed_json, dict) and "action" in parsed_json:
                 parsed_json = [mcp_lite.create_tool_call_request(
                        tool_name=parsed_json["action"],
                        arguments=parsed_json.get("parameters", {}),
                    )]
            
            if isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
                
            if isinstance(parsed_json, list):
                for x in parsed_json:
                    if _is_mcp_call(x):
                        args = x["params"].get("arguments", {})
                        if isinstance(args, str):
                            try:
                                x["params"]["arguments"] = json.loads(args)
                            except json.JSONDecodeError:
                                pass 
                        request_list.append(x)
                        
        except Exception as e:
            parsing_error = e

        if parsing_error and not request_list:
            tracker.record_syntax_error() 
            
            diagnostician.messages.append({
                "role": "user",
                "content": "SYSTEM ERROR: Invalid JSON format. Output ONLY a valid JSON array of MCP tool calls."
            })
            current_step_output = diagnostician.step(tracker=tracker)
            continue

        if request_list:
            def _tool_name(r):
                return (r.get("params") or {}).get("name", "")

            inits = [r for r in request_list if _tool_name(r) == "initialize_environment"]
            menus = [r for r in request_list if _tool_name(r) == "get_available_data_menu"]
            finishes = [r for r in request_list if _tool_name(r) == "finish"]
            others = [r for r in request_list if _tool_name(r) not in ("initialize_environment", "get_available_data_menu", "finish")]

            ordered = []
            ordered += inits[:1]
            ordered += menus
            ordered += others
            ordered += finishes[:1]
            ordered += inits[1:] + finishes[1:]
            request_list = ordered

            mcp_responses = []
            final_diagnosis_text = ""
            has_finished_successfully = False

            step_phase_lock = "need_init" if not initialized else None

            current_allowed_defs = diagnostician.tool_definitions 

            for request_data in request_list:
                tool_call_count += 1
                req_id = request_data.get("id", "0")
                tool_name = request_data["params"]["name"]
                tool_args = dict(request_data["params"].get("arguments", {}) or {})

                is_valid_schema, schema_err_msg = validate_tool_schema(tool_name, tool_args, current_allowed_defs)
                
                if not is_valid_schema:
                    print(f"  [L1 Schema Error] {schema_err_msg}")
                    tracker.record_schema_error() 
                    
                    tool_result_str = f"SYSTEM ERROR: {schema_err_msg}"
                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    current_trace.add_step(step_count, tool_name, tool_args, tool_result_str, step_reasoning_raw)
                    continue
                
                if step_phase_lock == "need_init" and tool_name != "initialize_environment":
                    tool_result_str = "SYSTEM INTERLOCK: You must call initialize_environment first."
                    print("[L2 Intervention] Init Interlock triggered.")
                    tracker.record_intervention()
                    
                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    current_trace.add_step(step_count, tool_name, tool_args, tool_result_str, step_reasoning_raw)
                    continue

                if has_finished_successfully:
                    tool_result_str = "SKIPPED: Case already closed by finish."
                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    continue

                # ---------------- init ----------------
                if tool_name == "initialize_environment":
                    if initialized:
                        tool_result_str = "ERROR: Already initialized."
                        print("[L2 Intervention] Double Init blocked.")
                        tracker.record_intervention()
                        
                        mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                        current_trace.add_step(step_count, tool_name, tool_args, tool_result_str, step_reasoning_raw)
                        continue

                    init_narrative, ehr_db, q, a, trace, question_focus = synthesizer.synthesize_case(
                        raw_data_str, dataset_hints=dataset_source
                    )

                    question_focus = (question_focus or "diagnosis").strip()
                    case_data["question_focus"] = question_focus
                    case_data["clinical_question"] = q
                    case_data["initial_narrative"] = init_narrative

                    if a:
                        extracted_choices = a
                    if (not case_data.get("options")) and extracted_choices:
                        case_data["options"] = extracted_choices

                    record.update(
                        {
                            "question_focus": case_data.get("question_focus"),
                            "clinical_question": case_data.get("clinical_question"),
                            "initial_narrative": case_data.get("initial_narrative"),
                            "options": case_data.get("options") or extracted_choices,
                        }
                    )

                    examiner.load_ehr(ehr_db)

                    try:
                        print("  [Router] -> Initializing Observer Agent...")
                        observer_agent = ObserverAgent(embed_client, EMBEDDING_MODEL_NAME, ehr_db)
                    except Exception as e:
                        observer_agent = None
                        tool_result_str = f"ERROR: Observer init failed: {e}"
                        mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                        continue

                    initialized = True
                    diagnostician.update_available_tools(TOOLS_PHASE_1)
                    tracker.record_tool_call()

                    tool_result_str = (
                        "Environment Initialized.\n"
                        f'Narrative: "{init_narrative}"\n'
                        f"QuestionFocus: {question_focus}\n"
                        f"ClinicalQuestion: {q}"
                    )
                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    current_trace.add_step(step_count, tool_name, tool_args, tool_result_str, step_reasoning_raw)
                    continue

                # ---------------- menu ----------------
                if tool_name == "get_available_data_menu":
                    if not initialized:
                        tool_result_str = "SYSTEM INTERLOCK: You must call initialize_environment before get_available_data_menu."
                        print("[L2 Intervention] Pre-Init Menu blocked.")
                        tracker.record_intervention()
                        
                        mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                        current_trace.add_step(step_count, tool_name, tool_args, tool_result_str, step_reasoning_raw)
                        continue

                    menu_called = True
                    diagnostician.update_available_tools(TOOLS_PHASE_2)
                    tracker.record_tool_call()

                    tool_args_copy = tool_args.copy() 
                    tool_args.pop("_observer_metadata", None)
                    tool_result_str = examiner.execute_tool(tool_name, tool_args)

                    if observer_agent:
                        try:
                            observer_agent.ingest_available_data_menu(tool_result_str)
                        except Exception as e:
                            print(f"  [Observer] ⚠️ ingest_available_data_menu failed: {e}")

                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    current_trace.add_step(step_count, tool_name, tool_args_copy, tool_result_str, step_reasoning_raw)
                    continue

                if (not menu_called) and tool_name not in ("initialize_environment", "get_available_data_menu"):
                    tool_result_str = "SYSTEM INTERLOCK: You must call get_available_data_menu before using other tools (including finish)."
                    print("[L2 Intervention] Pre-Menu tool blocked.")
                    tracker.record_intervention() 
                    
                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    current_trace.add_step(step_count, tool_name, tool_args, tool_result_str, step_reasoning_raw)
                    continue

                tool_args_for_trace = tool_args.copy() 
                
                observer_meta = tool_args.pop("_observer_metadata", {})
                if isinstance(observer_meta, str):
                    try:
                        observer_meta = json.loads(observer_meta)
                    except Exception:
                        observer_meta = {}

                if observer_meta:
                    current_hypothesis = observer_meta.get("hypothesis", current_hypothesis)
                    new_probs = (
                        observer_meta.get("candidate_probs")
                        or observer_meta.get("probs")
                        or observer_meta.get("differential_probs")
                    )

                    if isinstance(new_probs, dict) and new_probs:
                        current_probs = new_probs
                    elif not current_probs:
                        current_probs = {"Uncertain": 1.0}
                    print(f"  [Thought Injection] Hyp: {current_hypothesis}")
                else:
                    if not current_probs:
                        current_probs = {"Uncertain": 1.0}

                tool_result_str = ""
                is_rejected = False

                # ---------------- audit (L2) ----------------
                if observer_agent and tool_name != "finish":
                    is_approved, reason, metrics = observer_agent.audit_proposal(
                        hypothesis_text=current_hypothesis,
                        action_name=tool_name,
                        action_args=tool_args,
                        current_probs=current_probs,
                    )
                    tracker.record_observer_audit(entropy=metrics['H'], is_blocked=(not is_approved))

                    if not is_approved:
                        print(f"  [Observer 🛑] BLOCK: {reason}")
                        tool_result_str = f"SYSTEM REJECTION: {reason}"
                        is_rejected = True
                        observer_blocks += 1
                    else:
                        print(f"  [Observer ✅] PASS (H={metrics['H']:.2f}, f={metrics['f']:.2f}, U={metrics['U']:.2f})")

                # ---------------- finish ----------------
                if tool_name == "finish":
                    if not observer_agent:
                        guidance = "Observer not initialized. You must call initialize_environment first."
                        tool_result_str = f"SYSTEM INTERLOCK: {guidance}"
                        print("[L2 Intervention] Finish blocked (No Observer).")
                        tracker.record_intervention() 
                        
                        mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                        current_trace.add_step(step_count, tool_name, tool_args_for_trace, tool_result_str, step_reasoning_raw)
                        continue

                    should_terminate, guidance = observer_agent.check_termination(current_probs)
                    if not should_terminate:
                        print(f"  [Observer 🛑] Termination Denied. {guidance}")
                        tool_result_str = f"SYSTEM INTERLOCK: You cannot finish yet.\n{guidance}\nPlease continue."
                        observer_blocks += 1
                        tracker.record_intervention() 
                        
                        mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                        current_trace.add_step(step_count, tool_name, tool_args_for_trace, tool_result_str, step_reasoning_raw)
                        continue

                    print("  [Observer ✅] Termination Approved.")
                    has_finished_successfully = True

                    raw_report = tool_args.get("diagnosis_report", {})
                    if isinstance(raw_report, str):
                        try:
                            final_report = json.loads(raw_report)
                        except Exception:
                            final_report = {"raw_text": raw_report}
                    else:
                        final_report = raw_report

                    final_diagnosis_text = json.dumps(final_report, indent=2, ensure_ascii=False)
                    case_data["openqa_report"] = final_report
                    case_data["openqa_report_text"] = final_diagnosis_text

                    tool_result_str = "Diagnosis Report Accepted. Case Closed."
                    record.update(
                        {
                            "status": "openqa_done",
                            "question_focus": case_data.get("question_focus"),
                            "clinical_question": case_data.get("clinical_question"),
                            "openqa_report": final_report,
                        }
                    )
                    
                    current_trace.record_openqa(final_report)
                    mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                    current_trace.add_step(step_count, tool_name, tool_args_for_trace, tool_result_str, step_reasoning_raw)
                    continue

                if not is_rejected:
                    tracker.record_tool_call()

                    CONSULTANT_TOOLS = ["get_differential_diagnosis_criteria", "analyze_clinical_risk"]
                    if tool_name in CONSULTANT_TOOLS:
                        args_str = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
                        call_signature = f"{tool_name}:{args_str}"
                        if call_signature in recent_tool_calls:
                            tool_result_str = "SYSTEM NOTIFICATION: Duplicate query. Use internal knowledge."
                            print("[L2 Intervention] Duplicate Call Blocked.")
                            tracker.record_intervention() 
                            
                            mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                            current_trace.add_step(step_count, tool_name, tool_args_for_trace, tool_result_str, step_reasoning_raw)
                            continue
                        recent_tool_calls.append(call_signature)

                    if any(t["function"]["name"] == tool_name for t in examiner.TOOL_DEFINITIONS):
                        print(f"  [Router] -> Examiner: {tool_name}")
                        print(f"    Params: {json.dumps(tool_args, ensure_ascii=False)}")
                        tool_result_str = examiner.execute_tool(tool_name, tool_args)

                    elif any(t["function"]["name"] == tool_name for t in consultant.TOOL_DEFINITIONS):
                        print(f"  [Router] -> Consultant: {tool_name}")
                        print(f"    Params: {json.dumps(tool_args, ensure_ascii=False)}")
                        tool_result_str = consultant.execute_tool(tool_name, tool_args)

                    else:
                        tool_result_str = f"Error: Unknown tool {tool_name}"

                    if observer_agent:
                        observer_agent.update_evidence_state(tool_name, tool_args, tool_result_str)

                print(f"  > [Tool Result] ({tool_name}):")
                print(f"    {tool_result_str}")
                mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                
                current_trace.add_step(step_count, tool_name, tool_args_for_trace, tool_result_str, step_reasoning_raw)

            if has_finished_successfully:
                print(f"\n{'='*60}\n[Orchestrator] >>> FINISHED <<<\n{'='*60}")
                print(final_diagnosis_text)
                
                steps_openqa = step_count
                diagnostician.ingest_mcp_response(json.dumps(mcp_responses))

                options = case_data.get("options", {}) or extracted_choices
                correct_label = case_data.get("label") or case_data.get("answer_idx")

                if correct_label:
                    print("[Orchestrator] Running OpenQA report...")
                    openqa_eval = judge_agent.evaluate_openqa(
                        question=case_data.get("clinical_question") or raw_data_str,
                        options=options,
                        openqa_report=case_data.get("openqa_report") or {},
                        ground_truth=correct_label,
                    )
                    case_data["openqa_eval"] = openqa_eval
                    print("[OpenQA Judge]:", json.dumps(openqa_eval, ensure_ascii=False))
                    tracker.set_outcome(openqa_eval=openqa_eval)
                else:
                    case_data["openqa_eval"] = {
                        "skipped": True,
                        "reason": "no_options_or_no_ground_truth",
                        "has_options": bool(options),
                        "has_ground_truth": bool(correct_label),
                    }

                if not options:
                    return finalize_metrics_and_record("no_options", steps_openqa)

                options_str = "\n".join([f"({k}) {v}" for k, v in options.items()])

                # EXAM MODE
                EXAM_TOOLS = [SUBMIT_MCQ_TOOL_DEF] + consultant.TOOL_DEFINITIONS
                diagnostician.update_available_tools(EXAM_TOOLS)

                final_exam_prompt = f"""
EXAM MODE.

You have already produced an OpenQA clinical report.
Now map the report to the SINGLE BEST answer choice.

--- TASK DEFINITION (MUST FOLLOW) ---
QuestionFocus = {case_data.get("question_focus")}
ClinicalQuestion = {case_data.get("clinical_question")}

Rules:
- Your answer MUST address the ClinicalQuestion, NOT necessarily the primary diagnosis.
- If QuestionFocus is "workup_associated_anomaly" / "associated_condition":
  you MUST choose the option that is the associated condition to investigate (e.g., VSD),
  NOT the already-established diagnosis (e.g., EA/TEF), unless the question explicitly asks diagnosis.
- Use Consultant tools if you need to decide WHICH associated condition best matches the options.
- When ready, MUST call tool `submit_mcq_answer` with:
  {{"answer":"A","reasoning":"..."}}

--- QUESTION (RAW) ---
{raw_data_str}

--- YOUR OPENQA REPORT ---
{final_diagnosis_text}

--- OPTIONS ---
{options_str}
""".strip()

                if diagnostician.messages and diagnostician.messages[-1]["role"] == "tool":
                    diagnostician.messages.append({
                        "role": "assistant", 
                        "content": "Tool result acknowledged. Now proceeding to exam mode."
                    })
                diagnostician.messages.append({"role": "user", "content": final_exam_prompt})
                exam_step_output = diagnostician.step(tracker=tracker)
                max_exam_steps = 8
                final_answer_payload = None
                mcqa_loop_steps = 0

                for _ in range(max_exam_steps):
                    mcqa_loop_steps += 1
                    
                    try:
                        cleaned = extract_json_content(exam_step_output)
                        parsed = json.loads(cleaned)
                        reqs = [x for x in parsed] if isinstance(parsed, list) else []
                        reqs = [x for x in reqs if _is_mcp_call(x)]
                    except Exception as e:
                        reqs = []

                    if not reqs:
                        diagnostician.messages.append({"role": "user", "content": "SYSTEM ERROR: Invalid format. Output JSON array only."})
                        exam_step_output = diagnostician.step(tracker=tracker)
                        continue

                    mcp_responses = []
                    for r in reqs:
                        req_id = r.get("id", "0")
                        tool_name = (r.get("params") or {}).get("name", "")
                        tool_args = dict((r.get("params") or {}).get("arguments", {}) or {})
                        tool_call_count += 1
                        tool_args.pop("_observer_metadata", None)

                        if tool_name == "submit_mcq_answer":
                            final_answer_payload = {
                                "answer": tool_args.get("answer", ""),
                                "reasoning": tool_args.get("reasoning", ""),
                            }
                            tool_result_str = "MCQ answer received."
                            mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                            
                            current_trace.finalize(prediction=final_answer_payload["answer"], reasoning=final_answer_payload["reasoning"])
                            continue

                        if any(t["function"]["name"] == tool_name for t in consultant.TOOL_DEFINITIONS):
                            print(f"  [EXAM Router] -> Consultant: {tool_name}")
                            tool_result_str = consultant.execute_tool(tool_name, tool_args)
                            mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))
                            continue

                        tool_result_str = f"ERROR: Tool '{tool_name}' is not allowed in EXAM MODE."
                        mcp_responses.append(mcp_lite.create_tool_call_response(req_id, tool_result_str))

                    if final_answer_payload:
                        diagnostician.ingest_mcp_response(json.dumps(mcp_responses))
                        break

                    exam_step_output = diagnostician.step(json.dumps(mcp_responses), tracker=tracker)

                print(f"\n{'='*30} [FINAL ANSWER] {'='*30}")
                print(json.dumps(final_answer_payload or {"answer": "", "reasoning": "NO_SUBMISSION"}, ensure_ascii=False))

                if correct_label and final_answer_payload:
                    print(f"\n[Ground Truth]: ({correct_label})")
                    mcq_eval = judge_agent.evaluate_mcq(
                        question=raw_data_str,
                        options=options,
                        student_answer=final_answer_payload,
                        ground_truth=correct_label,
                    )
                    case_data["mcq_eval"] = mcq_eval
                    tracker.set_outcome(mcqa_eval=mcq_eval)

                    s_choice = mcq_eval.get("student_choice", "Unknown")
                    is_correct = mcq_eval.get("is_correct", False)
                    print(f"  > {s_choice}")
                    print("✅ " if is_correct else "❌ ")

                final_status = "mcq_no_submission"
                if final_answer_payload:
                    final_status = "mcq_submitted"
                if correct_label and case_data.get("mcq_eval") is not None:
                    final_status = "done"
                    
                record.update({
                    "mcq_answer": final_answer_payload,
                    "mcq_eval": case_data.get("mcq_eval"),
                })

                return finalize_metrics_and_record(final_status, steps_openqa, mcqa_loop_steps)

            current_step_output = diagnostician.step(json.dumps(mcp_responses), tracker=tracker)
            continue

        print(current_step_output)

        diagnostician.messages.append(
            {
                "role": "user",
                "content": (
                    "SYSTEM ERROR: Invalid format. You MUST output a JSON array of MCP tool calls "
                    "(jsonrpc + method=tools/call + params{name,arguments}). Do NOT output free text. "
                    "If done, call the `finish` tool."
                ),
            }
        )
        current_step_output = diagnostician.step(tracker=tracker)
        continue

    print(f"\n[Orchestrator] Stopped after reaching the {max_steps}-step limit.")
    return finalize_metrics_and_record("max_steps_reached", step_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EviDx clinical-agent evaluation pipeline.")
    parser.add_argument("--input-file", required=True, help="Path to an input JSONL dataset.")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of cases to process.")
    parser.add_argument("--output-dir", default="logs", help="Directory for logs, traces, and results.")
    args = parser.parse_args()

    missing_config = [
        name
        for name, value in {
            "LLM_API_KEY": api_key,
            "LLM_MODEL_NAME": MODEL_NAME,
            "JUDGE_API_KEY or LLM_API_KEY": judge_api_key,
            "JUDGE_MODEL_NAME or LLM_MODEL_NAME": judge_model_name,
        }.items()
        if not value
    ]
    if missing_config:
        parser.error(f"Missing required configuration: {', '.join(missing_config)}")

    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    judge_client = OpenAI(api_key=judge_api_key, base_url=judge_base_url)
    embed_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output_dir, f"run_{timestamp}.txt")
    sys.stdout = DualLogger(log_filename)

    print("=" * 60)
    print(f"[Config] LLM model: {MODEL_NAME}; base URL: {base_url or 'OpenAI default'}")
    print(f"[Config] Judge model: {judge_model_name}; base URL: {judge_base_url or 'OpenAI default'}")
    print(f"[Config] Embedding model: {EMBEDDING_MODEL_NAME}; base URL: {EMBED_BASE_URL}")

    synthesizer = NarrativeSynthesizer(llm_client=llm_client, model_name=MODEL_NAME)
    examiner = Examiner()
    consultant = Consultant(llm_client=llm_client, model_name=MODEL_NAME)

    full_tool_definitions = DIAG_TOOL_DEFINITIONS + consultant.TOOL_DEFINITIONS + examiner.TOOL_DEFINITIONS

    diagnostician = Diagnostician(
        llm_client=llm_client,
        model_name=MODEL_NAME,
        tool_definitions=full_tool_definitions,
    )

    judge_agent = Judge(llm_client=judge_client, model_name=judge_model_name)


    DATA_FILE_PATH = args.input_file
    MAX_TEST_CASES = args.max_cases

    RESULTS_PATH = os.path.join(args.output_dir, f"results_{timestamp}.jsonl")
    TRACE_PATH = os.path.join(args.output_dir, f"inference_traces_{timestamp}.jsonl")

    print(f"[Main] Results JSONL: {RESULTS_PATH}")
    print(f"[Main] Traces JSONL: {TRACE_PATH}")
    print(f"[Main] Reading input data from: {DATA_FILE_PATH}")

    try:
        with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if MAX_TEST_CASES is not None and i >= MAX_TEST_CASES:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data_point = json.loads(line)
                    raw_data_str = data_point.get("question", "")
                    
                    case_id = data_point.get("id", f"case_{i+1}")
                    
                    actual_dataset_source = data_point.get("dataset")
                    if not actual_dataset_source:
                        if "MedXpertQA" in DATA_FILE_PATH:
                            actual_dataset_source = "MedXpertQA"
                        elif "DiagnosisArena" in DATA_FILE_PATH:
                            actual_dataset_source = "DiagnosisArena"
                        else:
                            actual_dataset_source = "MedQA-File"

                    if not raw_data_str:
                        continue

                    print(f"\n{'#'*60}")
                    print(f"### Start Case (ID: {case_id} | Dataset: {actual_dataset_source}) ###")
                    print(f"{'#'*60}")
                    
                    data_point["id"] = case_id 

                    record = run_clinical_case_orchestration(
                        raw_data_str=raw_data_str,
                        dataset_source=actual_dataset_source, 
                        diagnostician=diagnostician,
                        synthesizer=synthesizer,
                        examiner=examiner,
                        consultant=consultant,
                        judge_agent=judge_agent,
                        case_data=data_point,
                        trace_file_path=TRACE_PATH
                    )
                    append_jsonl(RESULTS_PATH, record or {"id": case_id, "status": "no_record"})
                    print(f"--- NO. {i+1} Finished ---")

                except json.JSONDecodeError:
                    print(f"[Main] Skipping invalid JSON on input line {i + 1}.")
                    continue
                except Exception as e:
                    print(f"[Main] Case {i + 1} failed: {e}")
                    append_jsonl(
                        RESULTS_PATH,
                        {
                            "id": case_id if 'case_id' in locals() else f"unknown_{i}",
                            "dataset": actual_dataset_source if 'actual_dataset_source' in locals() else "Unknown",
                            "label": data_point.get("label") or data_point.get("answer_idx"),
                            "status": "exception",
                            "error": str(e),
                        },
                    )
                    continue
                    
        print("\n" + "="*60)
        print("[System] Generating CSV Metrics Report...")
        print("="*60)
        
        all_metrics_list = []
        try:
            with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "metrics" in data:
                            all_metrics_list.append(data["metrics"])
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            if all_metrics_list:
                df = pd.DataFrame(all_metrics_list)
                target_cols = [
                    'case_id', 
                    'L1_syntax_errors', 'L1_schema_errors', 'L1_total_attempts',
                    'L2_NEA', 'L2_intervention_count', 'L2_tool_calls_count', 
                    'L2_steps_openqa', 'L2_steps_mcqa',
                    'L3_evidence_recall', 'L3_openqa_eval', 'L3_mcqa_eval'
                ]
                existing = [c for c in target_cols if c in df.columns]
                others = [c for c in df.columns if c not in target_cols]
                df = df[existing + others]
                
                csv_path = RESULTS_PATH.replace(".jsonl", "_metrics.csv")
                df.to_csv(csv_path, index=False)
                print(f"[Success] CSV saved to: {csv_path}")
                print(f"[Summary] Total Cases: {len(df)}")
            else:
                print("[Warning] No metrics found in results file.")
                
        except Exception as e:
            print(f"[Error] Failed to generate CSV: {e}")

    except FileNotFoundError as exc:
        print(f"[Fatal] Input file not found: {DATA_FILE_PATH}")
        raise SystemExit(1) from exc
    except Exception as e:
        print(f"[Fatal] Failed to read or process the input file: {e}")
        raise SystemExit(1) from e

    print("\n[Main] All Finished!!")
