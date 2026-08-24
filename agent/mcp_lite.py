# mcp_lite.py
import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional


def create_tool_call_request(tool_name: str, arguments: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    if request_id is None:
        request_id = str(uuid.uuid4())

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

def create_tool_call_response(request_id: str, content: Any, is_error: bool = False) -> Dict[str, Any]:
    response = {
        "jsonrpc": "2.0",
        "id": request_id
    }
    if is_error:
        response["error"] = {"code": -32000, "message": str(content)}
    else:
        response["result"] = {"content": [{"type": "text", "text": str(content)}]}
    return response
