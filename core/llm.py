from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Tuple

from core.tools import TOOL_FN


# ---------------------------
# Helpers (mock streaming)
# ---------------------------

def _chunk_text(text: str, chunk_size: int = 24) -> List[str]:
    """Split text into small chunks to mimic streaming."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def _previous_user_message(messages: List[Dict[str, Any]]) -> str:
    found = 0
    for m in reversed(messages):
        if m.get("role") == "user":
            found += 1
            if found == 2:
                return m.get("content", "") or ""
    return ""


def _extract_city(text: str) -> str:
    """
    Very simple city extractor:
      'weather in Dallas' -> 'Dallas'
      'weather for New York' -> 'New York'
    """
    m = re.search(r"weather\s+(in|for)\s+([A-Za-z .'-]{2,})", text, re.IGNORECASE)
    if m:
        return m.group(2).strip()
    parts = text.strip().split()
    return parts[-1] if parts else "Unknown"


def _mock_reason_and_plan(user_text: str, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Decide whether to call tools and build a final response.
    Returns (final_answer_text, tool_calls_made_list)
    """
    tool_calls_made: List[Dict[str, Any]] = []
    t = (user_text or "").strip()

    # Memory demo: "Repeat what I asked earlier"
    if re.search(r"\brepeat\b|\bearlier\b|\bwhat did i ask\b", t, re.IGNORECASE):
        prev = _previous_user_message(messages)
        answer = f"You previously asked: {prev}" if prev else "I don't see an earlier user message yet."
        return answer, tool_calls_made

    # Tool: KB triggers
    kb_triggers = ["office hours", "grading", "late policy", "contact"]
    for trig in kb_triggers:
        if trig in t.lower():
            result = TOOL_FN["lookup_kb"](query=trig)
            tool_calls_made.append({"name": "lookup_kb", "args": {"query": trig}, "result": result})

            results = result.get("results", {})
            if isinstance(results, dict) and results and "note" not in results:
                first_key = list(results.keys())[0]
                answer = results[first_key]
            else:
                answer = "I checked the KB but didn’t find a match."
            return answer, tool_calls_made

    # Tool: weather
    if "weather" in t.lower():
        city = _extract_city(t)
        result = TOOL_FN["get_weather"](city=city)
        tool_calls_made.append({"name": "get_weather", "args": {"city": city}, "result": result})
        answer = f"Weather for {result['city']}: {result['forecast']}, {result['temp_c']}°C."
        return answer, tool_calls_made

    # Simple “task” behavior
    if re.search(r"\bsummarize\b", t, re.IGNORECASE):
        last_user = _last_user_message(messages)
        answer = f"Summary: {last_user[:120]}{'...' if len(last_user) > 120 else ''}"
        return answer, tool_calls_made

    # Default
    return (
        "Mock mode is ON. I can demonstrate memory and tool use.\n"
        "Try: 'What are the office hours?' or 'What is the weather in Dallas?' or 'Repeat what I asked earlier.'",
        tool_calls_made,
    )


def _mock_stream_chat_with_tools(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
) -> Iterable[Tuple[str, List[Dict[str, Any]], Tuple[int, int]]]:
    """
    Mock streaming+tool-calling loop.
    Yields: (text_chunk, tool_calls_made_so_far, (prompt_tokens, completion_tokens))
    """
    user_text = _last_user_message(messages)
    final_text, tool_calls_made = _mock_reason_and_plan(user_text, messages)

    # append assistant message to conversation state
    messages.append({"role": "assistant", "content": final_text})

    # crude token estimates (good enough for mock metrics)
    prompt_tokens = max(1, sum(len((m.get("content") or "").split()) for m in messages[:-1]))
    completion_tokens = max(1, len(final_text.split()))

    # stream small chunks
    for ch in _chunk_text(final_text):
        time.sleep(0.01)  # tiny delay to mimic streaming
        yield (ch, tool_calls_made, (prompt_tokens, completion_tokens))


# ---------------------------
# Public entrypoint
# ---------------------------

def stream_chat_with_tools(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
) -> Iterable[Tuple[str, List[Dict[str, Any]], Tuple[int, int]]]:
    """
    Switch between REAL OpenAI mode and MOCK mode.
    Set MOCK_MODE=1 to avoid calling OpenAI.
    """
    if os.getenv("MOCK_MODE", "0") == "1":
        yield from _mock_stream_chat_with_tools(model=model, messages=messages, temperature=temperature)
        return

    # ---- REAL MODE (requires OpenAI quota) ----
    from openai import OpenAI  # imported only if needed
    from core.tools import TOOL_SPECS

    client = OpenAI()
    tool_calls_made: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    while True:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SPECS,
            temperature=temperature,
            stream=True,
        )

        pending: Dict[str, Dict[str, Any]] = {}
        assistant_text = ""
        finish_reason = None

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            if getattr(delta, "content", None):
                assistant_text += delta.content
                yield (delta.content, tool_calls_made, (total_prompt_tokens, total_completion_tokens))

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    tc_id = tc.id
                    entry = pending.setdefault(tc_id, {"id": tc_id, "name": None, "arguments": ""})
                    if tc.function and tc.function.name:
                        entry["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        entry["arguments"] += tc.function.arguments

            if finish_reason in ("tool_calls", "stop"):
                break

        if not pending:
            if assistant_text.strip():
                messages.append({"role": "assistant", "content": assistant_text})
            return

        # Add the assistant tool-call message
        messages.append({
            "role": "assistant",
            "content": assistant_text or "",
            "tool_calls": [
                {"id": v["id"], "type": "function", "function": {"name": v["name"], "arguments": v["arguments"]}}
                for v in pending.values()
            ],
        })

        # Execute tools and add tool outputs
        for call in pending.values():
            name = call["name"]
            raw_args = call["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}

            fn = TOOL_FN.get(name)
            if not fn:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": f"Tool {name} failed", "detail": str(e)}

            tool_calls_made.append({"name": name, "args": args, "result": result})
            messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
