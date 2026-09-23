"""Native tool-calling loop: the model drives tool calls, observes results, and
answers — with citations whitelisted to gathered evidence and an honest
abstention when the tool-call budget is exhausted. Deterministic fake client."""
from agents import agent_loop


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, msg):
        self.message = msg


class _Resp:
    def __init__(self, msg):
        self.choices = [_Choice(msg)]


class _Completions:
    def __init__(self, scripted):
        self.scripted = scripted
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.scripted[min(len(self.calls) - 1, len(self.scripted) - 1)]


class _Client:
    def __init__(self, scripted):
        self.completions = _Completions(scripted)
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_model_calls_tool_then_answers_with_whitelisted_citations():
    scripted = [
        _Resp(_Msg(tool_calls=[_ToolCall("c1", "get_equipment_graph", '{"equipment_tag": "V-301"}')])),
        _Resp(_Msg(content='{"answer": "PSV-701 calibration was missed.", "citations": ["FE-004", "HALLUCINATED-9"]}')),
    ]
    client = _Client(scripted)
    executors = {
        "get_equipment_graph": lambda equipment_tag: [
            ("FE-004", "PSV-701 lifted early; missed calibration."),
            ("OISD-1", "Calibration history must be provided."),
        ],
    }

    result = agent_loop.run_tool_loop("sys", "Compliance of V-301?", executors, client=client, model="m")

    assert result["answer"] == "PSV-701 calibration was missed."
    assert result["citations"] == ["FE-004"]              # HALLUCINATED-9 dropped (not in evidence)
    assert set(result["evidence"]) == {"FE-004", "OISD-1"}
    assert result["tool_trace"] == [{"name": "get_equipment_graph", "args": {"equipment_tag": "V-301"}}]
    assert len(client.completions.calls) == 2
    # the tool result was fed back to the model as a tool-role message
    second_msgs = client.completions.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "FE-004" in m["content"] for m in second_msgs)


def test_unknown_tool_is_reported_not_crashed():
    scripted = [
        _Resp(_Msg(tool_calls=[_ToolCall("c1", "no_such_tool", "{}")])),
        _Resp(_Msg(content='{"answer": "done", "citations": []}')),
    ]
    client = _Client(scripted)
    result = agent_loop.run_tool_loop("sys", "q", {}, client=client, model="m")
    assert result["answer"] == "done"
    assert result["tool_trace"] == [{"name": "no_such_tool", "args": {}}]


def test_budget_exhaustion_abstains_instead_of_fabricating():
    # Model keeps calling tools forever; loop must stop and abstain.
    loop_call = _Resp(_Msg(tool_calls=[_ToolCall("c1", "search_knowledge_base", '{"query": "x"}')]))
    client = _Client([loop_call])
    executors = {"search_knowledge_base": lambda query: [("K1", "some text")]}

    result = agent_loop.run_tool_loop("sys", "q", executors, client=client, model="m", max_iters=3)

    assert result["citations"] == []
    assert "could not complete" in result["answer"].lower()
    assert len(client.completions.calls) == 3            # respected the iteration cap
    assert "K1" in result["evidence"]                    # evidence still gathered, just not fabricated into a claim


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
