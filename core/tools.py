from __future__ import annotations
from typing import Any, Dict
import time

KB = {
    "office hours": "Mon–Thu 2–4pm, Room 301",
    "grading": "Projects 60%, Exams 30%, Participation 10%",
    "late policy": "Late within 3 days: partial credit; after that: see syllabus.",
    "contact": "Email instructor via LMS messages; typical response within 24–48 hours.",
}

def get_weather(city: str) -> Dict[str, Any]:
    time.sleep(0.05)
    return {"city": city, "forecast": "Sunny", "temp_c": 27}

def lookup_kb(query: str) -> Dict[str, Any]:
    q = query.strip().lower()
    hits = {k: v for k, v in KB.items() if q in k.lower()}
    return {"query": query, "results": hits or {"note": "no match"}}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get a weather forecast for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_kb",
            "description": "Look up info in the local KB.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TOOL_FN = {
    "get_weather": get_weather,
    "lookup_kb": lookup_kb,
}
