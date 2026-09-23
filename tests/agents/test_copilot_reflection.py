"""Copilot's plan-act-observe loop: it must re-retrieve and re-answer when its
first answer isn't grounded in retrieved evidence, and stop once grounded (or
after max_reflections). Deterministic — no live LLM/retrieval."""
from agents import copilot


class _Retrieve:
    def __init__(self, script):
        self.script = script  # list of per-call results: [(key, text, score), ...]
        self.top_ks = []

    def __call__(self, _session, _query, top_k=5):
        self.top_ks.append(top_k)
        return self.script[min(len(self.top_ks) - 1, len(self.script) - 1)]


class _AskJson:
    def __init__(self, answers):
        self.answers = answers
        self.n = 0

    def __call__(self, _system, _user, context_keys=None, **_kw):
        answer = self.answers[min(self.n, len(self.answers) - 1)]
        self.n += 1
        return answer


def _wire(monkeypatch, retrieve, ask):
    monkeypatch.setattr(copilot, "retrieve", retrieve)
    monkeypatch.setattr(copilot, "ask_json", ask)
    monkeypatch.setattr(copilot, "graph_paths", lambda citations: [{"id": c} for c in citations])


def test_grounded_first_attempt_does_not_reflect(monkeypatch):
    retrieve = _Retrieve([[("FE-001", "bearing failure", 0.9)]])
    ask = _AskJson([{"answer": "P-101 bearing wear.", "citations": ["FE-001"]}])
    _wire(monkeypatch, retrieve, ask)

    result = copilot.answer(object(), "Why did P-101 fail?")

    assert result["citations"] == ["FE-001"]
    assert result["reflected"] is False
    assert result["reflections"] == 0
    assert retrieve.top_ks == [5]  # retrieved once, never broadened


def test_ungrounded_answer_triggers_one_reflection_then_succeeds(monkeypatch):
    retrieve = _Retrieve([
        [("C1", "vaguely related chunk", 0.5)],                       # attempt 0
        [("FE-004", "PSV missed calibration", 0.6), ("C1", "x", 0.4)],  # attempt 1 (broadened)
    ])
    ask = _AskJson([
        {"answer": "I'm not sure.", "citations": []},                 # ungrounded -> reflect
        {"answer": "PSV-701 calibration was missed.", "citations": ["FE-004"]},
    ])
    _wire(monkeypatch, retrieve, ask)

    result = copilot.answer(object(), "Compliance status of V-301?")

    assert result["citations"] == ["FE-004"]
    assert result["reflected"] is True
    assert result["reflections"] == 1
    assert retrieve.top_ks == [5, 10]  # broadened retrieval on the retry


def test_still_ungrounded_after_max_reflections_returns_honest_best(monkeypatch):
    retrieve = _Retrieve([[("C1", "unrelated", 0.5)]])
    ask = _AskJson([{"answer": "The context does not answer this.", "citations": []}])
    _wire(monkeypatch, retrieve, ask)

    result = copilot.answer(object(), "Totally unrelated question?")

    assert result["citations"] == []          # honest: no fabricated grounding
    assert result["reflected"] is True         # it did try to self-correct
    assert retrieve.top_ks == [5, 10]


def test_engine_env_routes_to_tool_calling_agent(monkeypatch):
    monkeypatch.setenv("COPILOT_ENGINE", "tools")
    monkeypatch.setattr(
        copilot, "answer_agentic",
        lambda session, query, memory_context=None: {"routed": "agentic", "q": query},
    )
    assert copilot.answer(object(), "hello")["routed"] == "agentic"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
