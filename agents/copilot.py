"""Copilot sub-agent: general Q&A with citations, grounded in the fused
hybrid retrieval layer (Neo4j vector + Qdrant dense + BM25 + graph traversal,
cross-encoder reranked) per PRD Section 8 item 3 — not plain vector RAG.

Runs a plan-act-observe loop: retrieve -> answer -> OBSERVE whether the answer
is grounded in retrieved evidence -> if not, ACT by re-retrieving with a
broader strategy and answering again. This is the self-correction the
grounding/faithfulness signal previously computed but never triggered."""
from agents.llm import ask_json
from agents.util import format_context, format_memory_context, graph_paths
from retrieval.hybrid import retrieve
import os

_INITIAL_TOP_K = 5
_EXPANDED_TOP_K = 10

_SYSTEM = (
    'You are a plant-operations Copilot. Answer the question using ONLY the '
    "numbered context passages given, each tagged with its source key like "
    '[FE-001]. Cite every key you actually relied on. State findings in your '
    "own words — don't copy sentences directly from retrieved context. "
    "When the question asks what a requirement or standard requires, "
    "prioritize the matching RegulatoryClause passage and distinguish that "
    "requirement from incident history or a work-order schedule. "
    'Respond only as JSON: {"answer": string, "citations": [string, ...]}. '
    "If the context does not answer the question, say so in `answer` and "
    "return an empty citations list."
)


def _is_grounded(result: dict, items: list[tuple[str, str]]) -> bool:
    """Observed grounding signal. agents.llm.ask_json already whitelists
    citations to the supplied context keys, so a non-empty citation list means
    the answer is anchored in at least one retrieved passage; an empty list
    means the model could not ground its answer in what was retrieved."""
    cited = result.get("citations") or []
    if not cited:
        return False
    keys = {key for key, _ in items}
    return all(citation in keys for citation in cited)


def _answer_once(query: str, items: list[tuple[str, str]], memory_context: list[str] | None) -> dict:
    memory_note = format_memory_context(memory_context or [])
    user_prompt = f"Context:\n{format_context(items)}"
    if memory_note:
        user_prompt += (
            "\n\nNon-authoritative memory from earlier sessions "
            "(do not cite it or treat it as plant evidence):\n"
            f"{memory_note}"
        )
    user_prompt += f"\n\nQuestion: {query}"
    result = ask_json(_SYSTEM, user_prompt, context_keys=[k for k, _ in items])
    return {
        "user_query": query,
        "agent_response": result["answer"],
        "citations": result["citations"],
        "retrieved_context": items,
        "graph_paths": graph_paths(result["citations"]),
    }


def _abstain(query: str, attempt: int) -> dict:
    return {
        "user_query": query,
        "agent_response": "No relevant context found.",
        "citations": [],
        "retrieved_context": [],
        "graph_paths": [],
        "reflections": attempt,
        "reflected": attempt > 0,
    }


def answer(session, query: str, memory_context: list[str] | None = None, *, max_reflections: int = 1) -> dict:
    # Engine select: the native tool-calling agent (model decides what to fetch)
    # or the default reflection loop (fixed retrieve, self-correct on grounding).
    if os.environ.get("COPILOT_ENGINE", "reflection").lower() == "tools":
        return answer_agentic(session, query, memory_context)

    best: dict | None = None
    top_k = _INITIAL_TOP_K

    for attempt in range(max_reflections + 1):
        context = retrieve(session, query, top_k=top_k)
        items = [(key, text) for key, text, _ in context]

        if not items:
            # Nothing retrieved. Broaden once more if we still can, else abstain.
            if attempt < max_reflections:
                top_k = _EXPANDED_TOP_K
                continue
            return _abstain(query, attempt)

        result = _answer_once(query, items, memory_context)
        result["reflections"] = attempt
        result["reflected"] = attempt > 0
        best = result

        if _is_grounded(result, items):
            return result

        # Observed: answer not grounded in retrieved evidence. Act: broaden
        # retrieval and try once more (up to max_reflections).
        top_k = _EXPANDED_TOP_K

    # Retries exhausted without a grounded answer — return the honest best we
    # produced (its empty citations already signal low confidence downstream).
    return best


def answer_agentic(session, query: str, memory_context: list[str] | None = None) -> dict:
    """Native tool-calling variant: the model plans and calls retrieval tools
    itself (see agents/agent_loop.py), rather than running a fixed pipeline."""
    from agents.agent_loop import _SYSTEM, build_executors, run_tool_loop

    user = query
    memory_note = format_memory_context(memory_context or [])
    if memory_note:
        user += (
            "\n\nNon-authoritative memory from earlier sessions "
            "(do not cite it or treat it as plant evidence):\n" + memory_note
        )

    result = run_tool_loop(_SYSTEM, user, build_executors(session))
    return {
        "user_query": query,
        "agent_response": result["answer"],
        "citations": result["citations"],
        "retrieved_context": list(result["evidence"].items()),
        "graph_paths": graph_paths(result["citations"]),
        "tool_trace": result["tool_trace"],
    }


if __name__ == "__main__":
    import truststore
    truststore.inject_into_ssl()
    from retrieval.index_chunks import get_driver, get_database

    driver, db = get_driver(), get_database()
    with driver.session(database=db) as session:
        result = answer(session, "Why did P-101 fail in March 2025?")
    driver.close()

    assert result["citations"], "expected non-empty citations"
    print(result["agent_response"])
    print("citations:", result["citations"])
    print("OK: agents.copilot self-check passed")
