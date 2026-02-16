import json
import os
import time
import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

def score_contains(output: str, expect: str) -> int:
    if not expect:
        return 1
    return 1 if expect.lower() in (output or "").lower() else 0

def main():
    print(f"[eval] BASE_URL={BASE_URL}")

    with open("eval/tasks.json", "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"[eval] loaded {len(tasks)} tasks")

    os.makedirs("results/transcripts", exist_ok=True)

    total = 0
    passed = 0
    latencies = []

    with httpx.Client(timeout=60) as client:
        for t in tasks:
            total += 1
            payload = {"conversation_id": t["conversation_id"], "user_message": t["prompt"]}

            start = time.time()
            r = client.post(f"{BASE_URL}/chat", json=payload)
            dur_ms = int((time.time() - start) * 1000)

            text = r.text
            latencies.append(dur_ms)

            ok = score_contains(text, t.get("expect", ""))
            passed += ok

            out_path = f"results/transcripts/{t['id']}.txt"
            with open(out_path, "w", encoding="utf-8") as out:
                out.write(f"PROMPT: {t['prompt']}\n\nRESPONSE:\n{text}\n")

            print(f"{t['id']}: {'PASS' if ok else 'FAIL'} | {dur_ms}ms")

    summary = {
        "task_success_pct": round(100.0 * passed / max(1, total), 2),
        "avg_latency_ms": int(sum(latencies) / max(1, len(latencies))),
        "num_tasks": total,
        "passed": passed,
        "mode": "mock" if os.getenv("MOCK_MODE", "0") == "1" else "real",
    }

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:", summary)
    print("[eval] wrote results/metrics.json and results/transcripts/*.txt")

if __name__ == "__main__":
    main()
