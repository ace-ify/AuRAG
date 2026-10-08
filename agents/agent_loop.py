"""Native tool-calling agent loop (plan-act-observe with function calling).

Unlike agents/copilot.py's reflection loop — which pre-decides retrieval and
only re-retrieves on an ungrounded answer — here the LLM itself decides which
tools to call and when, iterating tool_call -> observe result -> reason until
it can answer. This is genuine agentic control flow, not a fixed pipeline.

The loop is provider-thin: it takes a Groq-compatible client and a dict of
executor callables, so it runs deterministically under test with a fake client
that emits scripted tool calls. Executors are the only thing that touches the
graph/retrieval layer."""
import json

from agents.llm import REASONING_MODEL, get_client

# Tool schemas advertised to the model. Executors (below, built per request)
# are keyed by these names.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Hybrid semantic + keyword + graph search over the plant "
                "knowledge base. Use for any question about failures, "
                "procedures, clauses, or work orders. Returns evidence "
                "passages, each prefixed with a [KEY] to cite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_graph",
            "description": (
                "Fetch structured records directly and indirectly connected to "
                "one equipment tag (its failure events, work orders, governing "
                "procedures, applicable clauses, and connected sub-equipment via "
                "the assembly graph). Use when the question names a specific tag "
                "like P-101 or V-301."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_tag": {"type": "string", "description": "Equipment tag, e.g. P-101."}
                },
                "required": ["equipment_tag"],
            },
        },
    },
]

_SYSTEM = (
    "You are a plant-operations Copilot with tools. Plan which tool(s) to call "
    "to gather evidence, call them, then answer using ONLY the returned "
    "evidence passages. Each passage is prefixed with a [KEY]; cite every KEY "
    "you rely on. Do not invent facts or keys. When you have enough evidence, "
    "stop calling tools and reply with ONLY a JSON object: "
    '{"answer": string, "citations": [string, ...]}. If the evidence does not '
    "answer the question, say so in `answer` and return an empty citations list."
)

_MAX_ITERS = 4


def _extract_json(content: str) -> dict:
    """Lenient parse of the final answer JSON. Tool-calling models occasionally
    wrap the JSON in prose; fall back to treating the whole content as the
    answer with no citations rather than crashing the turn."""
    if not content:
        return {"answer": "", "citations": []}
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        pass
    start, end = content.find("{"), content.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(content[start : end + 1])
        except (ValueError, TypeError):
            pass
    return {"answer": content, "citations": []}


def run_tool_loop(system, user, executors, *, client=None, model=None, max_iters=_MAX_ITERS):
    """Drive the model through tool calls until it produces a final answer.

    executors: {tool_name: callable(**args) -> list[(key, text)]}. Returns
    {answer, citations (whitelisted to gathered evidence keys), evidence,
    tool_trace}."""
    client = client or get_client()
    model = model or REASONING_MODEL
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    evidence: dict[str, str] = {}
    tool_trace: list[dict] = []

    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, tool_choice="auto"
        )
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)

        if not calls:
            data = _extract_json(msg.content or "")
            # Whitelist citations to keys we actually retrieved — the model
            # cannot cite evidence it never saw.
            citations = [c for c in (data.get("citations") or []) if c in evidence]
            return {
                "answer": data.get("answer", ""),
                "citations": citations,
                "evidence": evidence,
                "tool_trace": tool_trace,
            }

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in calls
            ],
        })

        for tc in calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            tool_trace.append({"name": name, "args": args})
            executor = executors.get(name)
            if executor is None:
                result_text = f"Unknown tool: {name}"
            else:
                items = executor(**args) or []
                for key, text in items:
                    evidence[key] = text
                result_text = "\n".join(f"[{key}] {text}" for key, text in items) or "No matching evidence found."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    # Ran out of tool-call budget without a final answer — abstain honestly
    # rather than fabricate one.
    return {
        "answer": "I could not complete the analysis within the tool-call budget.",
        "citations": [],
        "evidence": evidence,
        "tool_trace": tool_trace,
    }


def build_executors(session, site_id: str | None = None) -> dict:
    """Bind the retrieval tools to a live session (and optional tenant scope).
    Imported lazily so importing this module doesn't drag in the
    embedding/Qdrant stack."""
    from retrieval.hybrid import retrieve
    from retrieval.graph_traversal import traverse

    def search_knowledge_base(query: str):
        return [(key, text) for key, text, _ in retrieve(session, query, top_k=5, site_id=site_id)]

    def get_equipment_graph(equipment_tag: str):
        return traverse(session, equipment_tag, top_k=10, depth=2, site_id=site_id)

    return {
        "search_knowledge_base": search_knowledge_base,
        "get_equipment_graph": get_equipment_graph,
    }
