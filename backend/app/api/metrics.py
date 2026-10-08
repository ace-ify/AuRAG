"""GET /api/metrics — aggregate LLM observability (call count, latency
p50/p95, token totals, cost, per-model breakdown) from telemetry.llm_metrics.
Read-only operational aggregates (no per-user data), same exposure tier as
/health."""
from fastapi import APIRouter

from telemetry.llm_metrics import snapshot

router = APIRouter()


@router.get("/metrics")
def llm_metrics() -> dict:
    return {"llm": snapshot()}
