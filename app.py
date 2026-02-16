from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from core.llm import stream_chat_with_tools
from core.memory import window_messages
from core.metrics import MetricsLogger, Timer, TurnMetrics, estimate_cost_usd
from core.ratelimit import RateLimiter
from core.safety import redact_secrets, safety_check
from core.store import ConversationStore

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a helpful, safe task-oriented assistant. "
    "Use tools when needed. Keep responses concise and factual. "
    "If you don't know, say so."
)

app = FastAPI()
store = ConversationStore()
limiter = RateLimiter(max_requests=30, window_s=60)
metrics = MetricsLogger("results/metrics.jsonl")


class ChatIn(BaseModel):
    conversation_id: str
    user_message: str


@app.get("/")
def home():
    return {"status": "ok", "endpoint": "/chat"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(inp: ChatIn, request: Request):
    # --- Rate limiting ---
    ip = request.client.host if request.client else "unknown"
    key = f"{ip}:{inp.conversation_id}"
    if not limiter.allow(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again soon.")

    # --- Safety + secret redaction ---
    user_text = redact_secrets(inp.user_message)
    ok, reason = safety_check(user_text)

    # --- Conversation state ---
    convo = store.get(inp.conversation_id)
    if not convo:
        store.append(inp.conversation_id, {"role": "system", "content": SYSTEM_PROMPT, "name": None})

    # If blocked, return JSON that ALWAYS includes the word "BLOCKED"
    if not ok:
        metrics.log(
            TurnMetrics(
                conversation_id=inp.conversation_id,
                ts=time.time(),
                model=MODEL,
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd_est=0.0,
                tool_calls=[],
                blocked=True,
                block_reason=reason,
            )
        )
        return {"blocked": True, "reason": reason, "message": "BLOCKED"}

    store.append(inp.conversation_id, {"role": "user", "content": user_text, "name": None})

    # Convert stored msgs into OpenAI message dicts, apply short-term memory window
    raw_msgs: List[Dict[str, Any]] = []
    for m in convo:
        d: Dict[str, Any] = {"role": m["role"], "content": m["content"]}
        if m.get("name"):
            d["name"] = m["name"]
        raw_msgs.append(d)

    msgs = window_messages(raw_msgs, max_messages=int(os.getenv("MEMORY_MESSAGES", "16")))

    # Windows has a built-in TEMP env var (a folder). Do NOT use TEMP as config.
    raw = os.getenv("CHAT_TEMP", "0.3")
    try:
        chat_temp = float(raw)
    except Exception:
        chat_temp = 0.3

    def _stream():
        tool_calls: List[Dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0

        with Timer() as t:
            for text, tools_so_far, (pt, ct) in stream_chat_with_tools(
                model=MODEL,
                messages=msgs,
                temperature=chat_temp,
            ):
                tool_calls = tools_so_far
                prompt_tokens = pt
                completion_tokens = ct
                yield text

        # Persist updated messages back to store
        store.set(
            inp.conversation_id,
            [{"role": m.get("role"), "content": m.get("content", ""), "name": m.get("name")} for m in msgs],
        )

        cost_est = estimate_cost_usd(MODEL, prompt_tokens, completion_tokens)
        metrics.log(
            TurnMetrics(
                conversation_id=inp.conversation_id,
                ts=time.time(),
                model=MODEL,
                latency_ms=t.ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd_est=cost_est,
                tool_calls=tool_calls,
            )
        )

        yield f"\n\n[latency_ms={t.ms}]\n"

    return StreamingResponse(_stream(), media_type="text/plain")
