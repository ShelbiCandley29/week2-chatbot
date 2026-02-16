from __future__ import annotations
import time
from collections import defaultdict, deque

class RateLimiter:
    def __init__(self, max_requests: int = 30, window_s: int = 60) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self._hits[key]
        while q and (now - q[0]) > self.window_s:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True
