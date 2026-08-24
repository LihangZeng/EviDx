EXAMINER_USAGE_INSTRUCTION = """
## TOOL USAGE PROTOCOL (THE EXAMINER)
You have access to a digital patient environment via the **Examiner** tools.
Follow these strict rules to access patient data:

1.  **MENU FIRST POLICY:** You MUST call `get_available_data_menu` as your very first action. 
    * Do NOT guess that "CT Head" exists. Check the menu first.
    * Do NOT ask for "Troponin" if the menu says "Labs: None".

2.  **HYPOTHESIS-DRIVEN SEARCH:**
    After seeing the menu and Initial Narrative, form a differential diagnosis.
    * Then, specifically search for findings that confirm or rule out your hypotheses.
    * Use `get_lab_results(test_name="...")` for precision.
    * Use `perform_physical_exam(system_name="...")` to check specific signs.

3.  **DATA INTEGRITY:**
    The Examiner returns raw data from the record. Trust it over your internal knowledge base regarding the specific patient.
"""