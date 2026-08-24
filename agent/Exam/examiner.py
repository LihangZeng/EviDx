# agent/Examiner/examiner.py

import json
import ast
from typing import Any, Dict, List, Optional, Union

try:
    from .examiner_tools import EXAMINER_TOOL_DEFINITIONS
except ImportError:
    from examiner_tools import EXAMINER_TOOL_DEFINITIONS

class Examiner:
    def __init__(self):
        self.ehr_db: Optional[Dict[str, Any]] = None
        self.TOOL_DEFINITIONS = EXAMINER_TOOL_DEFINITIONS

    def load_ehr(self, ehr_data: Dict[str, Any]):
        self.ehr_db = ehr_data
        
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self.ehr_db:
            return "System Error: No Patient Record Loaded."

        try:
            if tool_name == "get_available_data_menu":
                return self._get_menu()
            elif tool_name == "get_patient_demographics":
                return self._get_demographics()
            elif tool_name == "get_vital_signs":
                return self._get_vitals()
            elif tool_name == "get_history":
                return self._get_history(arguments.get("category"))
            elif tool_name == "review_system":
                return self._review_system(arguments.get("system_name"))
            elif tool_name == "perform_physical_exam":
                return self._physical_exam(arguments.get("system_name"))
            elif tool_name == "get_lab_results":
                return self._get_labs(arguments.get("test_name"))
            elif tool_name == "get_imaging_reports":
                return self._get_imaging(arguments.get("modality"))
            else:
                return f"Error: Unknown tool '{tool_name}'"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _format_complex_finding(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        
        if isinstance(item, dict):
            finding = item.get("finding") or item.get("name") or "Unspecified Finding"
            details = item.get("details") or item.get("detail") or item.get("value") or ""
            status = item.get("status")
            
            desc = finding
            extras = []
            if status and status.lower() not in ["present", "normal"]: 
                extras.append(status)
            if details:
                extras.append(details)
            
            if extras:
                desc += f" ({', '.join(extras)})"
            return desc
            
        return str(item)

    def _summarize_list(self, items: List[str], max_show: int = 15) -> str:
        if not items:
            return "None"
        
        unique_items = []
        seen = set()
        for i in items:
            if i not in seen:
                unique_items.append(i)
                seen.add(i)
                
        if len(unique_items) <= max_show:
            return ", ".join(unique_items)
        else:
            shown = ", ".join(unique_items[:max_show])
            remaining = len(unique_items) - max_show
            return f"{shown} ... (and {remaining} more)"

    def _get_menu(self) -> str:
        menu = ["===  PATIENT DATA INDEX (Available Data) ==="]
        phys_exam = self.ehr_db.get("physical_exam") or {}
        vitals = phys_exam.get("vital_signs") or {}

        valid_vitals = [k.replace('_', ' ').title() for k, v in vitals.items() if v and "not" not in str(v).lower()]
        menu.append(f"• Vital Signs: {', '.join(valid_vitals) if valid_vitals else 'None'}")


        hist_cats = []
        target_categories = [
            "past_medical_history", "surgical_history", "social_history", 
            "family_history", "allergies", "current_medications"
        ]
        for cat in target_categories:
            data = self.ehr_db.get(cat)

            if data and (isinstance(data, list) and len(data) > 0) or (isinstance(data, str) and len(data) > 5):
                hist_cats.append(cat.replace("_", " ").title())
        menu.append(f"• History Sections: {', '.join(hist_cats) if hist_cats else 'None'}")

        labs = self.ehr_db.get("lab_results", [])
        lab_names = [item.get("test_name", "Unknown") for item in labs]
        menu.append(f"• Labs ({len(labs)}): {self._summarize_list(lab_names)}")

        imgs = self.ehr_db.get("imaging_reports", [])
        img_modes = [item.get("modality", "Unknown Report") for item in imgs]
        menu.append(f"• Imaging ({len(imgs)}): {self._summarize_list(img_modes)}")

        ros = self.ehr_db.get("review_of_systems", {})
        pe = self.ehr_db.get("physical_exam", {})
        
        valid_ros = [k for k, v in ros.items() if v]

        valid_pe = [k for k, v in pe.items() if v and k != "vital_signs"]
        
        menu.append(f"• Review of Systems (Patient Reported): {', '.join(valid_ros) if valid_ros else 'None'}")
        menu.append(f"• Physical Exam (Doctor Observed): {', '.join(valid_pe) if valid_pe else 'None'}")

        return "\n".join(menu)

    def _get_demographics(self) -> str:
        d = self.ehr_db.get("demographics", {})
        return f"Age: {d.get('age')}, Sex: {d.get('sex')}\nNotes: {d.get('other_notes', 'None')}"

    def _get_vitals(self) -> str:
        v = self.ehr_db.get("physical_exam", {}).get("vital_signs", {})
        if not v: return "No vital signs recorded."
        
        lines = []
        for key, val in v.items():
            if val and "not" not in str(val).lower():
                lines.append(f"{key.replace('_', ' ').title()}: {val}")
        return "\n".join(lines) if lines else "Vital signs requested but none recorded in chart."

    def _get_history(self, category: str) -> str:
        if not category: return "Error: Category required."
        data = self.ehr_db.get(category, [])
        
        if isinstance(data, list):
            if not data: return "None recorded."
            return "\n".join([f"- {self._format_complex_finding(item)}" for item in data])
        return str(data)

    def _review_system(self, system: str) -> str:
        if not system: return "Error: System required."
        data = self.ehr_db.get("review_of_systems", {}).get(system, [])
        
        if not data: return f"Review of {system}: Unremarkable / Not recorded."
        
        if isinstance(data, list):
            return f"Review of {system}:\n" + "\n".join([f"- {self._format_complex_finding(item)}" for item in data])
        return str(data)

    def _physical_exam(self, system: str) -> str:
        if not system: return "Error: System required."
        
        if system == "general_appearance":
            return self.ehr_db.get("physical_exam", {}).get("general_appearance", "Not recorded")

        data = self.ehr_db.get("physical_exam", {}).get(system, [])
        if not data: return f"Exam of {system}: Unremarkable / Not recorded."
        
        if isinstance(data, list):
            return f"Exam of {system}:\n" + "\n".join([f"- {self._format_complex_finding(item)}" for item in data])
        return str(data)

    def _get_labs(self, query: str = None) -> str:
        labs = self.ehr_db.get("lab_results", [])
        if not labs: return "No labs recorded."

        if not query:
            lines = []
            for lab in labs:
                name = lab.get("test_name", "Unknown")
                val = lab.get("value", "")
                unit = lab.get("unit", "")
                flag = lab.get("flag", "")
                
                line = f"{name}: {val} {unit}"
                if flag and flag.lower() != "normal": line += f" [{flag}]"
                lines.append(line)
            return "All Lab Results:\n" + "\n".join(lines)

        hits = []
        for lab in labs:
            name = lab.get("test_name", "Unknown")
            if query.lower() in name.lower():
                val = lab.get("value", "")
                unit = lab.get("unit", "")
                flag = lab.get("flag", "")
                details = lab.get("details") or lab.get("detail") or lab.get("reference_range") or "" 
                
                res = f"**{name}**: {val} {unit}"
                if flag: res += f" (Flag: {flag})"
                if details: res += f"\n   Note: {details}"
                hits.append(res)
        
        if not hits:
            available_labs = [l.get("test_name", "") for l in self.ehr_db.get("lab_results", [])]
            return f"No lab results found matching '{query}'. Available labs are: {', '.join(available_labs)}"
        return "\n".join(hits)

    def _get_imaging(self, query: str = None) -> str:
        imgs = self.ehr_db.get("imaging_reports", [])
        if not imgs: return "No imaging reports."

        if not query:
            lines = []
            for img in imgs:
                modality = img.get("modality", "Unknown")
                impression = img.get("impression", "No impression")
                lines.append(f"- {modality}: {impression}")
            return "Imaging Reports Available:\n" + "\n".join(lines)

        hits = []
        for img in imgs:
            modality = img.get("modality", "Unknown")
            if query.lower() in modality.lower():
                findings = img.get("findings", "")
                impression = img.get("impression", "")
                hits.append(f"--- {modality} ---\nFindings: {findings}\nImpression: {impression}")
        
        if not hits:
            return f"No imaging reports found matching '{query}'."
        return "\n\n".join(hits)