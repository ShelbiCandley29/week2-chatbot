from __future__ import annotations

import re
from typing import Tuple


# ---- Secret redaction patterns ----
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
]


# ---- Strong blocking regex ----
SELF_HARM_RE = re.compile(
    r"\bkill\s*myself\b|\bsuicid(e|al)\b|\bend\s*my\s*life\b|\bharm\s*myself\b|\bself[-\s]*harm\b",
    re.IGNORECASE,
)

HACKING_RE = re.compile(
    r"\bhack(ing)?\b|\bbreak\s*into\b|\bsteal\s*password\b|\bphish(ing)?\b|\bkeylogger\b|\bddos\b|\bdoxx?\b",
    re.IGNORECASE,
)

WEAPONS_RE = re.compile(
    r"\bmake\s*a\s*bomb\b|\bbuild\s*a\s*bomb\b|\bpipe\s*bomb\b|\bexplosive(s)?\b",
    re.IGNORECASE,
)


# ---- Redact secrets before logging ----
def redact_secrets(text: str) -> str:
    if not text:
        return text
    out = text
    for p in SECRET_PATTERNS:
        out = p.sub("[REDACTED_SECRET]", out)
    return out


# ---- Safety decision ----
def safety_check(user_text: str) -> Tuple[bool, str]:
    """
    Returns:
        (True, "") if allowed
        (False, reason) if blocked
    """
    t = user_text or ""

    if SELF_HARM_RE.search(t):
        return False, "unsafe: self-harm content"

    if HACKING_RE.search(t):
        return False, "unsafe: hacking / cyber abuse"

    if WEAPONS_RE.search(t):
        return False, "unsafe: weapons / explosives"

    return True, ""
