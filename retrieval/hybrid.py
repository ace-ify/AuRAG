"""Fuses the four retrieval sources (Neo4j native vector, Qdrant dense, BM25
keyword, graph traversal) and reranks with a local cross-encoder. Per PRD
Section 14: each source is validated independently (see each module's
__main__ self-check) before being combined here.

Usage: retrieve(session, query) -> [(key, text, score), ...]
"""
from retrieval import qdrant_store
from retrieval.bm25_index import search_bm25
from retrieval.candidate_filter import (
    filter_to_explicit_anchors,
    filter_to_explicit_years,
)
from retrieval.embeddings import embed_texts
from retrieval.graph_traversal import extract_query_entities, traverse
from ingestion.pipeline import load_known_entities
from retrieval.rerank import rerank

_CANDIDATES_PER_SOURCE = 10

# How much a graph-anchored candidate's normalized rerank score is lifted.
# The cross-encoder scores synthesized passage TEXT, so a structured record
# that exactly answers the query (a FailureEvent, a clause) can otherwise be
# out-ranked by a chattier chunk. This lets the graph's topological relevance
# influence final order instead of being discarded — without letting an
# irrelevant graph hit leapfrog a clearly-better semantic match.
_GRAPH_BOOST = 0.15

# ponytail: db.index.vector.queryNodes is deprecated in favor of a newer
# SEARCH syntax on some Neo4j 5.x builds, but still functions (warning only,
# not an error) — not worth chasing a moving-target syntax mid-hackathon.
_NEO4J_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $vec)
YIELD node, score
RETURN node.id AS id, node.text AS text, score
"""


def _neo4j_vector_search(session, query_vec: list[float], top_k: int) -> list[tuple[str, str]]:
    rows = session.run(_NEO4J_VECTOR_CYPHER, k=top_k, vec=query_vec).data()
    return [(r["id"], r["text"]) for r in rows if r["text"]]


def _blend_graph_provenance(
    reranked: list[tuple[str, str, float]],
    graph_keys: set[str],
    top_k: int,
) -> list[tuple[str, str, float]]:
    """Min-max normalize rerank scores to [0,1] (scale-independent across the
    local cross-encoder's logits and Cohere's 0..1), add a fixed boost to
    graph-anchored candidates, then re-sort and truncate to top_k. Returns the
    blended score so ordering reflects both semantic and topological relevance."""
    if not reranked:
        return []
    scores = [s for _, _, s in reranked]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    blended = []
    for key, text, score in reranked:
        norm = (score - lo) / span
        if key in graph_keys:
            norm += _GRAPH_BOOST
        blended.append((key, text, norm))
    blended.sort(key=lambda item: item[2], reverse=True)
    return blended[:top_k]


def retrieve(session, query: str, top_k: int = 5, site_id: str | None = None) -> list[tuple[str, str, float]]:
    # Dense sources need a query embedding. If embedding is unavailable, degrade
    # to BM25 + graph rather than fabricating a vector — never silently retrieve
    # over garbage coordinates.
    query_vec = None
    try:
        query_vec = embed_texts([query])[0]
    except Exception:
        pass

    candidates: dict[str, str] = {}

    if query_vec is not None:
        try:
            for key, text in _neo4j_vector_search(session, query_vec, _CANDIDATES_PER_SOURCE):
                candidates[key] = text
        except Exception:
            pass

        try:
            for key, text, _ in qdrant_store.search(query_vec, top_k=_CANDIDATES_PER_SOURCE):
                candidates[key] = text
        except Exception:
            pass

    try:
        for key, text, _ in search_bm25(session, query, top_k=_CANDIDATES_PER_SOURCE):
            candidates[key] = text
    except Exception:
        pass

    graph_candidates = []
    try:
        # depth=2: walk the HAS_PART assembly graph so connected-equipment
        # evidence (a vessel's relief valve, a pump's drive motor) surfaces —
        # the multi-hop reasoning a plain-RAG retriever structurally cannot do.
        graph_candidates = traverse(session, query, top_k=_CANDIDATES_PER_SOURCE, depth=2, site_id=site_id)
        for key, text in graph_candidates:
            candidates[key] = text
    except Exception:
        pass

    graph_keys = {key for key, _ in graph_candidates}

    try:
        known_tags, known_names = load_known_entities(session)
        tags, names = extract_query_entities(query, known_tags, known_names)
        candidates = filter_to_explicit_anchors(
            candidates,
            graph_candidates,
            [*tags, *names],
        )
    except Exception:
        pass

    candidates = filter_to_explicit_years(candidates, query)

    if not candidates:
        return []

    # Score every candidate, then blend in graph provenance before truncating —
    # so a directly-anchored structured record isn't buried by the reranker
    # alone (the previous behavior discarded all graph signal at ranking time).
    reranked = rerank(query, list(candidates.items()), top_n=len(candidates))
    return _blend_graph_provenance(reranked, graph_keys, top_k)


if __name__ == "__main__":
    import truststore
    truststore.inject_into_ssl()
    from retrieval.index_chunks import get_driver, get_database

    driver, db = get_driver(), get_database()
    with driver.session(database=db) as session:
        results = retrieve(session, "Why did P-101 fail in March 2025?")
    assert results, "expected fused hits for a P-101 query"
    for key, text, score in results:
        print(f"{score:.2f}  {key}  {text[:80]!r}")
