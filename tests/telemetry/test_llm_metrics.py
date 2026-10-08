"""LLM observability: metrics core (latency/token/cost aggregation), the live
Groq path recording real usage, and the /api/metrics endpoint."""
from telemetry import llm_metrics


def setup_function(_fn):
    llm_metrics.reset()


def test_cost_uses_pricing_table():
    # gpt-oss-120b = (0.15 in, 0.60 out) per 1M tokens
    assert abs(llm_metrics.cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000) - 0.75) < 1e-9
    assert llm_metrics.cost_usd("unknown-model", 5000, 5000) == 0.0  # unknown => free but counted


def test_snapshot_aggregates_calls_tokens_latency_cost():
    llm_metrics.record_llm_call("openai/gpt-oss-120b", 800, 1000, 500)
    llm_metrics.record_llm_call("openai/gpt-oss-120b", 1200, 2000, 300)
    llm_metrics.record_llm_call("llama-3.3-70b-versatile", 400, 100, 50, ok=False)

    snap = llm_metrics.snapshot()
    assert snap["calls"] == 3
    assert snap["errors"] == 1
    assert snap["tokens"]["total"] == 1000 + 500 + 2000 + 300 + 100 + 50
    assert snap["latency_ms"]["mean"] == round((800 + 1200 + 400) / 3, 2)
    assert snap["cost_usd"] > 0
    assert snap["by_model"]["openai/gpt-oss-120b"]["calls"] == 2


def test_empty_snapshot_is_zeroed():
    snap = llm_metrics.snapshot()
    assert snap["calls"] == 0 and snap["cost_usd"] == 0.0


def test_recording_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LLM_METRICS_ENABLED", "0")
    llm_metrics.record_llm_call("openai/gpt-oss-120b", 100, 10, 10)
    assert llm_metrics.snapshot()["calls"] == 0


def test_call_groq_json_records_real_token_usage(monkeypatch):
    from agents import llm

    class _Usage:
        prompt_tokens = 123
        completion_tokens = 45

    class _Msg:
        content = '{"answer": "ok", "citations": []}'

    class _Resp:
        usage = _Usage()
        choices = [type("C", (), {"message": _Msg()})()]

    class _Client:
        chat = type("Chat", (), {"completions": type("Comp", (), {"create": staticmethod(lambda **kw: _Resp())})()})()

    monkeypatch.setattr(llm, "get_client", lambda: _Client())

    out = llm._call_groq_json("sys", "user", "openai/gpt-oss-120b")
    assert out == {"answer": "ok", "citations": []}

    snap = llm_metrics.snapshot()
    assert snap["calls"] == 1
    assert snap["tokens"]["prompt"] == 123
    assert snap["tokens"]["completion"] == 45
    assert snap["latency_ms"]["mean"] >= 0


def test_metrics_endpoint_returns_snapshot():
    from starlette.testclient import TestClient
    from backend.app.main import app

    llm_metrics.record_llm_call("openai/gpt-oss-120b", 500, 100, 50)
    client = TestClient(app)
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "llm" in body and body["llm"]["calls"] >= 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
