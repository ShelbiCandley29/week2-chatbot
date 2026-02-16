from __future__ import annotations
from typing import Dict, List, TypedDict, Optional

class Msg(TypedDict):
    role: str
    content: str
    name: Optional[str]  # tool name when role="tool"

class ConversationStore:
    """Naive in-memory store keyed by conversation_id."""
    def __init__(self) -> None:
        self._db: Dict[str, List[Msg]] = {}

    def get(self, conversation_id: str) -> List[Msg]:
        return self._db.setdefault(conversation_id, [])

    def set(self, conversation_id: str, messages: List[Msg]) -> None:
        self._db[conversation_id] = messages

    def append(self, conversation_id: str, msg: Msg) -> None:
        self._db.setdefault(conversation_id, []).append(msg)
