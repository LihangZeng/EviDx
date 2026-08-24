# agent/Examiner/examiner_tools.py
EXAMINER_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_data_menu",
            "description": "CRITICAL FIRST STEP: Returns a summary list (index) of what data exists in this patient's record. Always call this first to avoid guessing test names.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_demographics",
            "description": "Extract basic demographics (age, sex, race) and general notes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vital_signs",
            "description": "Extract all recorded vital signs (BP, HR, Temp, RR, O2 sat).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": "Retrieve patient history categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "past_medical_history",
                            "surgical_history",
                            "social_history",
                            "family_history",
                            "allergies",
                            "current_medications",
                        ],
                        "description": "The specific history category to retrieve.",
                    }
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_system",
            "description": "Check 'Review of Systems' (Subjective symptoms reported by patient).",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_name": {
                        "type": "string",
                        "enum": ["general", "HEENT", "cardiovascular", "respiratory", "gastrointestinal", "genitourinary", "musculoskeletal", "neurologic", "dermatologic", "psychiatric", "endocrine", "reproductive"],
                        "description": "The body system to query.",
                    }
                },
                "required": ["system_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_physical_exam",
            "description": "Check 'Physical Exam' findings (Objective signs observed by doctor).",
            "parameters": {
                "type": "object",
                "properties": {
                    "system_name": {
                        "type": "string",
                        "enum": ["general_appearance", "HEENT", "neck", "cardiovascular", "lungs", "abdomen", "extremities", "neurologic", "skin", "musculoskeletal", "breast", "gynecologic", "other_significant_findings"],
                        "description": "The body system/area to examine.",
                    }
                },
                "required": ["system_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lab_results",
            "description": "Retrieve laboratory test results. Returns a specific test or a list of all available labs if no name provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "Optional: Specific test name (e.g., 'WBC', 'Creatinine'). Matches loosely (substring).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_imaging_reports",
            "description": "Retrieve imaging reports. Returns a specific modality or all reports.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "description": "Optional: Specific modality (e.g., 'CT', 'X-Ray'). Matches loosely.",
                    }
                },
                "required": [],
            },
        },
    },
]