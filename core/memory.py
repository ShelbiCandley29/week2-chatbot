from __future__ import annotations
from typing import List, Dict, Any

def window_messages(messages: List[Dict[str, Any]], max_messages: int = 16) -> List[Dict[str, Any]]:
    """Keep system + last N non-system messages."""
    if not messages:
        return []
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return system[:1] + rest[-max_messages:]
