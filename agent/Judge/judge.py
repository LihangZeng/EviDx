# Judge/judge.py
import json
import re
from typing import Dict, Any, Union, Optional

try:
    from .judge_prompts import JUDGE_MCQ_SYSTEM_PROMPT, JUDGE_OPENQA_SYSTEM_PROMPT
except ImportError:
    from judge_prompts import JUDGE_MCQ_SYSTEM_PROMPT, JUDGE_OPENQA_SYSTEM_PROMPT


class Judge:
    """
    - MCQ: deterministic grading if student_answer is structured (dict with "answer")
    - OpenQA: semantic relation grading vs Ground Truth option text:
        entailed / partial / unrelated / contradicts
    """

    def __init__(
        self,
        llm_client,
        model_name: str,
        force_llm_for_mcq: bool = False,
        require_focus_match_openqa: bool = False, 
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.force_llm_for_mcq = force_llm_for_mcq
        self.require_focus_match_openqa = require_focus_match_openqa


    @staticmethod
    def _best_effort_extract_json(txt: str) -> Optional[Dict[str, Any]]:
        if not txt:
            return None
        txt = txt.strip()

        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        l = txt.find("{")
        r = txt.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(txt[l : r + 1])
            except Exception:
                return None

        return None

    @staticmethod
    def _format_options(options: Dict[str, str]) -> str:
        return "\n".join([f"{k}: {v}" for k, v in options.items()])

    @staticmethod
    def _normalize_choice(raw: Any, options: Dict[str, str]) -> str:
        if raw is None:
            return "Unclear"

        s = str(raw).strip()
        if not s:
            return "Unclear"

        opt_keys_upper = {str(k).strip().upper() for k in options.keys()}
        opt_keys_raw = {str(k).strip() for k in options.keys()}

        m = re.search(r"\b([A-Z])\b", s.upper())
        if m:
            cand = m.group(1)
            if cand in opt_keys_upper:
                return cand

        if s in opt_keys_raw:
            return s

        if re.fullmatch(r"-?\d+", s):
            idx = int(s)
            keys_upper = [str(k).strip().upper() for k in options.keys()]
            if all(re.fullmatch(r"[A-Z]", k) for k in keys_upper):
                ordered = sorted(set(keys_upper))
                n = len(ordered)
                if 0 <= idx < n:
                    return ordered[idx]
                if 1 <= idx <= n:
                    return ordered[idx - 1]

        return "Unclear"

    @staticmethod
    def _normalize_relation(x: Any) -> str:
        s = (str(x or "")).strip().lower().replace("-", "_")
        alias = {
            "entails": "entailed",
            "entailment": "entailed",
            "supported": "entailed",
            "support": "entailed",
            "partially_related": "partial",
            "weak_related": "partial",
            "related": "partial",
            "not_related": "unrelated",
            "irrelevant": "unrelated",
            "contradiction": "contradicts",
            "contradict": "contradicts",
        }
        s = alias.get(s, s)
        if s not in ("entailed", "partial", "unrelated", "contradicts"):
            return "unrelated"
        return s

    @staticmethod
    def _normalize_focus_label(x: str) -> str:
        s = (x or "").strip().lower().replace("-", "_")
        alias = {
            "workup": "workup_associated_anomaly",
            "associated_anomaly": "workup_associated_anomaly",
            "workup_associated": "workup_associated_anomaly",
            "next_step": "next_step_test",
            "best_test": "next_step_test",
            "gold_standard": "next_step_test",
            "management": "treatment",
            "therapy": "treatment",
            "pathophysiology": "mechanism",
            "moa": "mechanism",
        }
        return alias.get(s, s)

    @staticmethod
    def _infer_intent_from_question(q: str) -> str:
        s = (q or "").lower()

        if re.search(r"\b(next step|best next step|most appropriate next step|initial management|best test|gold standard)\b", s):
            return "next_step_test"

        if re.search(r"\b(treat|treatment|therapy|management|antibiotic|dose|dosage)\b", s):
            return "treatment"

        if re.search(r"\b(mechanism|pathophysiology|moa)\b", s):
            return "mechanism"

        if re.search(r"\b(prognosis|complication|risk)\b", s):
            return "prognosis"

        if re.search(r"\b(interpret|interpretation)\b", s):
            return "interpretation"

        if re.search(
            r"\b(investigat(e|ed|ing|ion|ions)|screen(ed|ing)?|evaluat(e|ed|ing|ion)|work[\s\-]?up|associated)\b",
            s
        ):
            return "workup_associated_anomaly"

        if re.search(r"\b(diagnosis|most likely diagnosis|what is the diagnosis)\b", s):
            return "diagnosis"

        return "other"

    def _call_llm_json(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        params = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "extra_body": {"enable_thinking": False},
        }

        try:
            try:
                resp = self.llm_client.chat.completions.create(**params)
            except Exception as e:
                emsg = str(e)

                if "extra_body" in emsg or "enable_thinking" in emsg or "thinking" in emsg:
                    params.pop("extra_body", None)
                    resp = self.llm_client.chat.completions.create(**params)

                elif "response_format" in emsg or "json_object" in emsg:
                    params.pop("response_format", None)
                    resp = self.llm_client.chat.completions.create(**params)
                else:
                    raise

            txt = (resp.choices[0].message.content or "").strip()
            obj = self._best_effort_extract_json(txt)
            if isinstance(obj, dict):
                return obj

            return {
                "student_choice": "Error",
                "is_correct": False,
                "reasoning": f"Judge could not parse JSON. Raw: {txt[:200]}",
            }

        except Exception as e:
            return {
                "student_choice": "Error",
                "is_correct": False,
                "reasoning": f"Judge exception: {str(e)}",
            }

    def evaluate_mcq(
        self,
        question: str,
        options: Dict[str, str],
        student_answer: Union[str, Dict[str, Any]],
        ground_truth: str,
    ) -> Dict[str, Any]:
        if not self.force_llm_for_mcq and isinstance(student_answer, dict):
            ans = self._normalize_choice(student_answer.get("answer", ""), options)
            gt = self._normalize_choice(ground_truth, options)
            if ans != "Unclear" and gt != "Unclear":
                return {
                    "student_choice": ans,
                    "is_correct": (ans == gt),
                    "reasoning": "Deterministic grading from structured submit_mcq_answer payload (normalized).",
                }

        user_content = f"""
--- QUESTION ---
{question}

--- OPTIONS ---
{self._format_options(options)}

--- GROUND TRUTH ---
{ground_truth}

--- STUDENT ANSWER ---
{json.dumps(student_answer, ensure_ascii=False) if isinstance(student_answer, dict) else str(student_answer)}
"""
        res = self._call_llm_json(JUDGE_MCQ_SYSTEM_PROMPT, user_content)

        choice = self._normalize_choice(res.get("student_choice", "Unclear"), options)
        gt = self._normalize_choice(ground_truth, options)
        res["student_choice"] = choice
        res["is_correct"] = (choice != "Unclear" and gt != "Unclear" and choice == gt)
        return res

    def evaluate_openqa(
        self,
        question: str,
        options: Dict[str, str],
        openqa_report: Union[str, Dict[str, Any]],
        ground_truth: str,
    ) -> Dict[str, Any]:

        if isinstance(openqa_report, str):
            try:
                parsed = json.loads(openqa_report)
                if isinstance(parsed, dict):
                    openqa_report = parsed
            except Exception:
                pass

        gt_norm = self._normalize_choice(ground_truth, options)
        gt_text = options.get(gt_norm, "")

        intent = self._infer_intent_from_question(question)

        student_focus = "unknown"
        best_answer_text = ""
        reasoning = ""

        if isinstance(openqa_report, dict):
            student_focus = openqa_report.get("question_focus", "unknown")
            best_answer_text = openqa_report.get("best_answer_text", "") or ""
            reasoning = openqa_report.get("reasoning", "") or ""

        norm_intent = self._normalize_focus_label(intent)
        norm_focus = self._normalize_focus_label(student_focus)
        focus_match = False
        if norm_focus != "unknown":
            focus_match = (norm_focus == norm_intent) or (norm_intent == "other")

        answer_core = {
            "question_focus": student_focus,
            "best_answer_text": best_answer_text,
        }
        if reasoning:
            answer_core["reasoning"] = reasoning

        user_content = f"""
--- QUESTION ---
{question}

--- QUESTION_INTENT ---
{intent}

--- OPTIONS ---
{self._format_options(options)}

--- GROUND TRUTH KEY ---
{ground_truth}

--- GROUND TRUTH TEXT ---
{gt_text}

--- STUDENT OPENQA ANSWER CORE ---
{json.dumps(answer_core, ensure_ascii=False, indent=2)}
"""

        out = self._call_llm_json(JUDGE_OPENQA_SYSTEM_PROMPT, user_content) or {}

        out.setdefault("short_answer", "")
        out.setdefault("relation_to_gt", "unrelated")
        out.setdefault("confidence", 0.0)
        out.setdefault("why", "")
        out.setdefault("evidence_spans", [])
        out.setdefault("best_matching_choice", "Unclear")
        out.setdefault("candidates", [])

        relation = self._normalize_relation(out.get("relation_to_gt", "unrelated"))
        out["relation_to_gt"] = relation

        out["ground_truth_norm"] = gt_norm
        out["ground_truth_text"] = gt_text

        out["best_matching_choice"] = self._normalize_choice(out.get("best_matching_choice", "Unclear"), options)

        norm_cands = []
        for c in (out.get("candidates") or [])[:3]:
            ch = self._normalize_choice(c.get("choice", "Unclear"), options)
            norm_cands.append(
                {
                    "choice": ch,
                    "confidence": c.get("confidence", None),
                    "why": c.get("why", ""),
                }
            )
        out["candidates"] = norm_cands

        out["question_intent"] = intent
        out["question_focus_extracted"] = student_focus
        out["focus_match"] = focus_match

        strict = (relation == "entailed")
        weak = (relation in ("entailed", "partial"))

        if self.require_focus_match_openqa and not focus_match:
            strict = False
            weak = False

        out["openqa_score_strict"] = strict
        out["openqa_score_weak"] = weak

        if relation == "contradicts":
            out["openqa_grade"] = "contradicts"
        elif strict:
            out["openqa_grade"] = "correct"
        elif weak:
            out["openqa_grade"] = "weak_correct"
        else:
            out["openqa_grade"] = "incorrect"

        out["openqa_score"] = weak

        return out
