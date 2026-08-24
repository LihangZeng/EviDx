import numpy as np
import json
from typing import List, Dict, Set, Any, Optional

class MetricTracker:
    def __init__(self, case_id: str, ground_truth_evidence: List[Any] = None):
        self.case_id = case_id
        self.gt_evidence = set()
        for evidence in ground_truth_evidence or []:
            if isinstance(evidence, str):
                text = evidence
            elif isinstance(evidence, dict):
                text = evidence.get("feature_name") or evidence.get("original_text") or ""
            else:
                text = str(evidence)
            if text.strip():
                self.gt_evidence.add(text.lower().strip())
        
        self.total_attempts = 0    
        self.syntax_errors = 0   
        self.schema_errors = 0    
        
        self.entropy_history: List[float] = [] 
        
        self.intervention_count = 0        
        
        self.total_actions_audited = 0   
        
        self.steps_openqa = 0     
        self.steps_mcqa = 0       
        self.total_tool_calls = 0   
        
        self.openqa_eval_result = {} 
        self.mcqa_eval_result = {}   
        self.final_evidence_recall: Optional[float] = None

    def record_attempt(self):
        self.total_attempts += 1

    def record_intervention(self):
        self.intervention_count += 1

    def record_syntax_error(self):
        self.syntax_errors += 1

    def record_schema_error(self):
        self.schema_errors += 1

    def record_tool_call(self):
        self.total_tool_calls += 1

    def record_observer_audit(self, entropy: float, is_blocked: bool):
        self.entropy_history.append(entropy)
        self.total_actions_audited += 1
        
        if is_blocked:
            self.record_intervention() 
    def set_steps(self, openqa_steps: int, mcqa_steps: int):
        self.steps_openqa = openqa_steps
        self.steps_mcqa = mcqa_steps

    def set_outcome(self, openqa_eval: Dict = None, mcqa_eval: Dict = None):
        if openqa_eval:
            self.openqa_eval_result = openqa_eval
        if mcqa_eval:
            self.mcqa_eval_result = mcqa_eval

    def calculate_nea(self) -> float:
        if not self.entropy_history:
            return 0.0
        
        first_h = self.entropy_history[0]
        h_init = first_h if first_h > 1e-6 else 1.0
        
        normalized_sum = sum(h / h_init for h in self.entropy_history)
        t = len(self.entropy_history)
        
        return normalized_sum / t if t > 0 else 0.0

    def calculate_evidence_recall(self, acquired_evidence_keys: Set[str]) -> Optional[float]:
        if not self.gt_evidence:
            self.final_evidence_recall = None
            return None
        
        hits = 0
        acquired_text = " ".join(list(acquired_evidence_keys)).lower()
        
        for gt_item in self.gt_evidence:
            if gt_item in acquired_text: 
                hits += 1
        
        if len(self.gt_evidence) > 0:
            self.final_evidence_recall = hits / len(self.gt_evidence)
        else:
            self.final_evidence_recall = 0.0
            
        return self.final_evidence_recall

    def finalize(self) -> Dict[str, Any]:
        nea = self.calculate_nea()

        return {
            "case_id": self.case_id,
            
            # --- L1: Execution (Counts) ---
            "L1_syntax_errors": self.syntax_errors,
            "L1_schema_errors": self.schema_errors,
            "L1_total_attempts": self.total_attempts,
            
            # --- L2: Reasoning (Counts & Steps) ---
            "L2_NEA": round(nea, 4),
     
            "L2_intervention_count": self.intervention_count, 
            
            "L2_tool_calls_count": self.total_tool_calls,
            "L2_steps_openqa": self.steps_openqa,
            "L2_steps_mcqa": self.steps_mcqa,
            "L2_entropy_trace": str([round(x, 2) for x in self.entropy_history]),
            
            # --- L3: Outcome (Judge Results) ---
            "L3_openqa_eval": json.dumps(self.openqa_eval_result, ensure_ascii=False),
            "L3_mcqa_eval": json.dumps(self.mcqa_eval_result, ensure_ascii=False),
            "L3_evidence_recall": round(self.final_evidence_recall, 4) if self.final_evidence_recall is not None else "N/A"
        }
