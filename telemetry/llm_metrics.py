"""In-process LLM observability: per-call latency + real token usage + cost,
aggregated for a metrics readout. Deliberately dependency-light (stdlib only)
and thread-safe, because Groq calls run inside the chat request's worker thread.

This measures the ACTUAL production path (agents.llm._call_groq_json records
here), unlike the enterprise gateway whose telemetry the live Groq path bypassed
— and it uses the provider's real usage object, not a len()/4 estimate."""
from __future__ import annotations

import os
import threading
import time
from collections import deque

# Approximate USD per 1M tokens (input, output). Override the whole map via env
# if your Groq tier differs; unknown models cost 0 but are still counted.
PRICING: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.10, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}
_DEFAULT_PRICE = (0.0, 0.0)

_LOCK = threading.Lock()
_SPANS: deque[dict] = deque(maxlen=2000)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICING.get(model, _DEFAULT_PRICE)
    return (prompt_tokens / 1_000_000) * price_in + (completion_tokens / 1_000_000) * price_out


def record_llm_call(
    model: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    ok: bool = True,
) -> None:
    if os.environ.get("LLM_METRICS_ENABLED", "1") == "0":
        return
    span = {
        "model": model,
        "latency_ms": float(latency_ms),
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "cost_usd": cost_usd(model, prompt_tokens or 0, completion_tokens or 0),
        "ok": bool(ok),
        "ts": time.time(),
    }
    with _LOCK:
        _SPANS.append(span)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((q / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def snapshot() -> dict:
    """Aggregate recent spans: call count, latency p50/p95/mean, token totals,
    total cost, and a per-model breakdown."""
    with _LOCK:
        spans = list(_SPANS)

    n = len(spans)
    if n == 0:
        return {"calls": 0, "errors": 0, "tokens": {"prompt": 0, "completion": 0, "total": 0},
                "cost_usd": 0.0, "latency_ms": {"p50": 0.0, "p95": 0.0, "mean": 0.0}, "by_model": {}}

    latencies = sorted(s["latency_ms"] for s in spans)
    prompt = sum(s["prompt_tokens"] for s in spans)
    completion = sum(s["completion_tokens"] for s in spans)
    cost = sum(s["cost_usd"] for s in spans)
    errors = sum(1 for s in spans if not s["ok"])

    by_model: dict[str, dict] = {}
    for s in spans:
        m = by_model.setdefault(s["model"], {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["tokens"] += s["prompt_tokens"] + s["completion_tokens"]
        m["cost_usd"] += s["cost_usd"]

    return {
        "calls": n,
        "errors": errors,
        "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion},
        "cost_usd": round(cost, 6),
        "latency_ms": {
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
            "mean": round(sum(latencies) / n, 2),
        },
        "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_model.items()},
    }


def reset() -> None:
    with _LOCK:
        _SPANS.clear()


if __name__ == "__main__":
    reset()
    record_llm_call("openai/gpt-oss-120b", 800, 1000, 500)
    record_llm_call("openai/gpt-oss-120b", 1200, 2000, 300)
    snap = snapshot()
    assert snap["calls"] == 2, snap
    assert snap["tokens"]["total"] == 3800, snap
    assert snap["latency_ms"]["p50"] > 0, snap
    print("OK:", snap)
