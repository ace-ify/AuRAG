"""Retrieval ablation harness: run the SAME ground-truth queries through each
retrieval configuration (dense-only -> +BM25 -> +graph -> full hybrid+rerank)
and report hit@k / recall@k / MRR per config, so each component has to EARN its
place with a measured lift instead of being asserted to help.

The orchestration + table formatting here are pure and unit-tested with fake
retrievers; the __main__ wires the real source-toggled retrievers against live
Neo4j/Qdrant using retrieval/ground_truth.json."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.retrieval_metrics import aggregate

REPO_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = REPO_ROOT / "retrieval" / "ground_truth.json"


def run_ablation(cases: list[dict], retrievers: dict, k: int = 5) -> dict:
    """cases: [{query, expected_any}]. retrievers: {name: fn(query)->[keys]}.
    Returns {config_name: aggregate_metrics}. A retriever that raises counts as
    an empty (missed) retrieval rather than crashing the whole run."""
    results: dict[str, dict] = {}
    for name, retrieve_fn in retrievers.items():
        per_case: list[tuple[list[str], set[str]]] = []
        for case in cases:
            relevant = set(case["expected_any"])
            try:
                retrieved = list(retrieve_fn(case["query"]) or [])
            except Exception:
                retrieved = []
            per_case.append((retrieved, relevant))
        results[name] = aggregate(per_case, k=k)
    return results


def format_table(results: dict) -> str:
    """Render the ablation results as a fixed-width table, ordered by hit@k."""
    header = f"{'configuration':<28} {'n':>3} {'hit@k':>7} {'recall@k':>9} {'mrr':>6}"
    lines = [header, "-" * len(header)]
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["hit_at_k"], reverse=True):
        lines.append(
            f"{name:<28} {m['n']:>3} {m['hit_at_k']:>7.3f} {m['recall_at_k']:>9.3f} {m['mrr']:>6.3f}"
        )
    return "\n".join(lines)


def load_cases(path: Path = GROUND_TRUTH) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [{"query": v["query"], "expected_any": v["expected_any"]} for v in raw.values()]


# --- live wiring (needs Neo4j + Qdrant + embeddings) ------------------------
def _live_retrievers(session) -> dict:
    """Source-toggled retrievers sharing the same primitives hybrid.retrieve
    uses, so the only variable across configs is which sources/rerank are on."""
    from retrieval import qdrant_store
    from retrieval.bm25_index import search_bm25
    from retrieval.embeddings import embed_texts
    from retrieval.graph_traversal import traverse
    from retrieval.hybrid import _neo4j_vector_search, retrieve as full_retrieve
    from retrieval.rerank import rerank

    N = 10

    def dense(query):
        vec = embed_texts([query])[0]
        keys = {}
        for key, text in _neo4j_vector_search(session, vec, N):
            keys[key] = text
        for key, text, _ in qdrant_store.search(vec, top_k=N):
            keys.setdefault(key, text)
        return list(keys)[:5]

    def dense_bm25(query):
        vec = embed_texts([query])[0]
        keys = {}
        for key, text in _neo4j_vector_search(session, vec, N):
            keys[key] = text
        for key, text, _ in qdrant_store.search(vec, top_k=N):
            keys.setdefault(key, text)
        for key, text, _ in search_bm25(session, query, top_k=N):
            keys.setdefault(key, text)
        return list(keys)[:5]

    def dense_bm25_graph(query):
        vec = embed_texts([query])[0]
        keys = {}
        for key, text in _neo4j_vector_search(session, vec, N):
            keys[key] = text
        for key, text, _ in qdrant_store.search(vec, top_k=N):
            keys.setdefault(key, text)
        for key, text, _ in search_bm25(session, query, top_k=N):
            keys.setdefault(key, text)
        for key, text in traverse(session, query, top_k=N, depth=2):
            keys.setdefault(key, text)
        return list(keys)[:5]

    def full(query):
        return [key for key, _, _ in full_retrieve(session, query, top_k=5)]

    return {
        "dense-only": dense,
        "dense+bm25": dense_bm25,
        "dense+bm25+graph": dense_bm25_graph,
        "full (hybrid+rerank+boost)": full,
    }


if __name__ == "__main__":
    import truststore
    truststore.inject_into_ssl()
    from retrieval.index_chunks import get_database, get_driver

    cases = load_cases()
    driver, db = get_driver(), get_database()
    with driver.session(database=db) as session:
        results = run_ablation(cases, _live_retrievers(session), k=5)
    driver.close()

    print(format_table(results))
    # The whole point: the full config must not be worse than dense-only.
    assert results["full (hybrid+rerank+boost)"]["hit_at_k"] >= results["dense-only"]["hit_at_k"], results
    print("\nOK: full hybrid >= dense-only on hit@k")
