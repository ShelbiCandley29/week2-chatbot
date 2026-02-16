from __future__ import annotations
import json, os, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# Pricing table (USD per 1M tokens). Update in paper with a citation to OpenAI pricing page.
PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.30, "output": 1.20},
    "gpt-4o": {"input": 3.75, "output": 15.00},
}

def _base_model_name(model: str) -> str:
    # e.g. gpt-4o-mini-2024-07-18 -> gpt-4o-mini
    return model.split("-20")[0]

def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    base = _base_model_name(model)
    rates = PRICING_PER_1M.get(base)
    if not rates:
        return 0.0
    return (prompt_tokens / 1_000_000) * rates["input"] + (completion_tokens / 1_000_000) * rates["output"]

@dataclass
class TurnMetrics:
    conversation_id: str
    ts: float
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd_est: float
    tool_calls: List[Dict[str, Any]]
    blocked: bool = False
    block_reason: str = ""

class MetricsLogger:
    def __init__(self, path: str = "results/metrics.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, m: TurnMetrics) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(m)) + "\n")

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
    @property
    def ms(self) -> int:
        return int((self.end - self.start) * 1000)
