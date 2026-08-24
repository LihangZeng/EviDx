# consultant/tools.py

CONSULTANT_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_differential_diagnosis_criteria",
            "description": "Retrieve authoritative diagnostic criteria, distinguishing features, or confounding factors for specific diseases from medical guidelines. Use this when you are stuck or want to verify a hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The disease name, symptom, or clinical scenario to research (e.g., 'epiglottitis causes in vaccinated children')."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_clinical_risk",
            "description": "Consult clinical guidelines to identify red flags, high-risk differentials, and standard-of-care procedures. Use this before finalizing a diagnosis to ensure safety.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The clinical situation or provisional diagnosis to assess (e.g., 'neonate with tachypnea risk assessment')."
                    }
                },
                "required": ["query"],
            },
        },
    },
]