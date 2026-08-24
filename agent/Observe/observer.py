# Observe/observer.py
import json
import math
import re
import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional

class ObserverAgent:

    def __init__(self, llm_client: Any, embedding_model_name: str, ehr_data: Dict[str, Any]):
        self.llm_client = llm_client
        self.embed_model = embedding_model_name

        self.s_reachable: Set[str] = set()  
        self.s_effective: Set[str] = set() 
        self.s_acquired: Set[str] = set()   

        self.menu_parsed: bool = False
        self.menu_available_roots: Set[str] = set()
        self.menu_unavailable_roots: Set[str] = set()

        self._init_knowledge_space(ehr_data)
        self.s_effective = set(self.s_reachable)

        self.history_h: List[float] = []
        self.history_v: List[float] = []

        self.BASE_TAU_E = 0.9
        self.BASE_TAU_V = 0.4
        self.ETA = 2.0
        self.LIMIT = 0.1
        self.step_counter = 0


    def _init_knowledge_space(self, data: Dict[str, Any], prefix: str = ""):
        if not isinstance(data, dict):
            return

        SKIP_KEYS = {"flag", "unit", "details", "source", "status", "reference_range"}
        INVALID_VALUES = {
            "none", "not provided", "not reported", "n/a", "not applicable",
            "unremarkable", "negative", "false", "", None
        }

        for k, v in data.items():
            if k in SKIP_KEYS:
                continue

            new_key = f"{prefix}.{k}" if prefix else k

            if isinstance(v, (str, int, float, bool)):
                str_v = str(v).lower().strip()
                if str_v not in INVALID_VALUES:
                    self.s_reachable.add(new_key)

            elif isinstance(v, list) and v:
                self.s_reachable.add(new_key)
                for item in v:
                    if isinstance(item, dict):
                        item_name = item.get("test_name") or item.get("modality") or item.get("finding")
                        if item_name:
                            self.s_reachable.add(f"{new_key}.{item_name}")
                        else:
                            self._init_knowledge_space(item, new_key)

            else:
                self._init_knowledge_space(v, new_key)

    # ----------------------- menu ingestion -----------------------

    def ingest_available_data_menu(self, menu_text: str):
        if not menu_text:
            return

        avail_roots: Set[str] = set()
        unavail_roots: Set[str] = set()

        lines = [ln.strip() for ln in str(menu_text).splitlines() if ln.strip()]
        for ln in lines:
            m = re.match(r"^-?\s*([^:]+)\s*:\s*(.+)$", ln)
            if not m:
                continue
            label = m.group(1).strip()
            status = m.group(2).strip().lower()

            label_norm = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")

            is_avail = any(x in status for x in ["available", "present", "yes"])
            is_none  = any(x in status for x in ["none", "not recorded", "not provided", "missing", "n/a", "unavailable"])

            alias = {
                "imaging": "imaging_reports",
                "imaging_reports": "imaging_reports",
                "labs": "lab_results",
                "lab": "lab_results",
                "lab_results": "lab_results",
                "history": "history",
                "physical_exam": "physical_exam",
                "exam": "physical_exam",
                "review_of_systems": "review_of_systems",
                "ros": "review_of_systems",
                "vitals": "vital_signs",
                "vital_signs": "vital_signs",
                "medications": "medications",
            }
            root_guess = alias.get(label_norm, label_norm)

            if is_avail:
                avail_roots.add(root_guess)
            elif is_none:
                unavail_roots.add(root_guess)

        self.menu_parsed = True
        self.menu_available_roots = avail_roots
        self.menu_unavailable_roots = unavail_roots

        if avail_roots:
            eff = set()
            for k in self.s_reachable:
                root = k.split(".")[0].lower()
                if root in avail_roots:
                    eff.add(k)
            if eff:
                self.s_effective = eff

    # ----------------------- embedding + similarity -----------------------

    def _get_embedding(self, text: str) -> np.ndarray:
        text = text.replace("\n", " ")
        try:
            resp = self.llm_client.embeddings.create(model=self.embed_model, input=[text])
            return np.array(resp.data[0].embedding)
        except Exception as e:
            print(f"[Observer] Embedding Error: {e}")
            return np.array([], dtype=float)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1.size == 0 or vec2.size == 0 or vec1.shape != vec2.shape:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _calculate_entropy(self, probs: Dict[str, float]) -> float:
        if not probs:
            return 10.0

        total = sum(probs.values())
        if total == 0:
            return 10.0

        entropy = 0.0
        for p in probs.values():
            p_norm = p / total
            if p_norm > 0:
                entropy -= p_norm * math.log2(p_norm)
        return entropy

    def audit_proposal(
        self,
        hypothesis_text: str,
        action_name: str,
        action_args: Dict,
        current_probs: Dict[str, float],
    ) -> Tuple[bool, str, Dict]:

        self.step_counter += 1

        h_t = self._calculate_entropy(current_probs)
        action_desc = f"{action_name}: {json.dumps(action_args, ensure_ascii=False)}"
        vec_h = self._get_embedding(hypothesis_text)
        vec_a = self._get_embedding(action_desc)
        f_t = self._cosine_similarity(vec_h, vec_a)

        decay_factor = max(0.0, (self.step_counter - 10) * 0.01)
        limit_dynamic = max(0.05, self.LIMIT - decay_factor)

        smoothed_h = max(h_t, 0.5)
        u_t = smoothed_h * (0.2 + self.ETA * f_t)

        metrics = {"H": h_t, "f": f_t, "U": u_t}

        if action_name in ["initialize_environment", "get_available_data_menu", "finish"]:
            return True, "System Action", metrics

        if u_t < limit_dynamic:
            reason = (
                f"Proposal Rejected (Low Utility {u_t:.2f} < {limit_dynamic:.2f}). "
                f"Action '{action_name}' aligns poorly with hypothesis '{hypothesis_text}' (Faithfulness={f_t:.2f})."
            )
            return False, reason, metrics

        return True, "Approved", metrics

    def update_evidence_state(self, action_name: str, action_args: Dict, tool_result_str: str):
        relevant_keywords = []
        for v in action_args.values():
            if isinstance(v, str):
                relevant_keywords.append(v.lower())

        if action_name == "review_system":
            sys_name = action_args.get("system_name", "").lower()
            relevant_keywords.append(f"review_of_systems.{sys_name}")
        elif action_name == "perform_physical_exam":
            sys_name = action_args.get("system_name", "").lower()
            relevant_keywords.append(f"physical_exam.{sys_name}")

        newly_acquired = 0
        for key in self.s_effective:
            if key in self.s_acquired:
                continue

            key_lower = key.lower()
            for kw in relevant_keywords:
                if kw and kw in key_lower:
                    self.s_acquired.add(key)
                    newly_acquired += 1
                    break

            key_leaf = key.split(".")[-1].lower()
            if len(key_leaf) > 3 and key_leaf in str(tool_result_str).lower():
                self.s_acquired.add(key)
                newly_acquired += 1

    def check_termination(self, current_probs: Dict[str, float]) -> Tuple[bool, str]:
        h_t = self._calculate_entropy(current_probs)

        denom = len(self.s_effective)
        v_t = 1.0 if denom == 0 else (len(self.s_acquired) / denom)

        steps = self.step_counter
        dynamic_tau_e = self.BASE_TAU_E + (max(0, steps - 5) * 0.05)
        dynamic_tau_v = max(0.1, self.BASE_TAU_V - (max(0, steps - 10) * 0.01))

        self.history_h.append(h_t)
        self.history_v.append(v_t)

        metrics_info = (f"[Observer Metrics] H(t)={h_t:.2f} (Target < {dynamic_tau_e:.2f}), "
                        f"V(t)={v_t:.2f} (Target > {dynamic_tau_v:.2f})")

        if (h_t < dynamic_tau_e and v_t > dynamic_tau_v) or steps > 25:
            return True, f"Conditions Met (Dynamic). {metrics_info}. Proceed."

        guidance = f"OBSERVER FEEDBACK ({metrics_info}):\n"
        if h_t >= dynamic_tau_e:
            guidance += f"- High Uncertainty: H({h_t:.2f}) >= {dynamic_tau_e:.2f}. Provide clearer candidate_probs.\n"

        if v_t <= dynamic_tau_v:
            missing = list(self.s_effective - self.s_acquired)
            examples = missing[:3] if missing else []
            guidance += f"- Low Evidence Saturation: V({v_t:.2f}) <= {dynamic_tau_v:.2f}. Check: {examples}...\n"

        return False, guidance
