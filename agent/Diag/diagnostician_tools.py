INIT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "initialize_environment",
        "description": "CRITICAL: This MUST be the VERY FIRST action you take when a new case starts. Calling this tool instructs the system to process the raw patient data and build the clinical environment (EHR). You will receive the 'Initial Narrative (Chief Complaint)' as the result of this call.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_action": {
                    "type": "string",
                    "enum": ["start_case"],
                    "description": "Confirmation to start the case. Set to 'start_case'."
                }
            },
            "required": ["case_action"]
        }
    }
}

FINISH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call this ONLY when you have sufficient evidence to form a final diagnosis. This submits your report.",
        "parameters": {
            "type": "object",
            "properties": {
                "diagnosis_report": {
                    "type": "object",
                    "properties": {
                        "question_focus": {"type": "string"},
                        "best_answer_text": {"type": "string", "description": "Direct natural-language answer to the question focus. Should be mappable to one option for MCQ."},
                        "primary_diagnosis": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "reasoning": {"type": "string"},
                        "critical_differentials": {"type": "array", "items": {"type": "string"}},
                        "next_steps": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["question_focus", "best_answer_text",
                                 "primary_diagnosis", "confidence", "reasoning",
                                "critical_differentials", "next_steps"]
                }
            },
            "required": ["diagnosis_report"]
        }
    }
}

SUBMIT_MCQ_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "submit_mcq_answer",
        "description": "Submit final multiple-choice answer as JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Single option letter like 'A'."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Short explanation for why this option is best."
                }
            },
            "required": ["answer", "reasoning"]
        }
    }
}

TOOL_DEFINITIONS = [INIT_TOOL_DEF, FINISH_TOOL_DEF]
