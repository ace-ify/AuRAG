"""Local dense embeddings, shared by Chunk indexing and query-time search.
sentence-transformers, not a hosted embed API — no per-query external call,
no quota risk (same reasoning as the Groq/local-Qdrant/local-rerank choices,
see NOTES.md).
"""
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = None


import logging

logger = logging.getLogger(__name__)


def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            logger.warning("Could not load SentenceTransformer: %s", e)
            return None
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_model()
    if model is None:
        # Fail loud. A hash-based pseudo-vector would return real-looking
        # cosine scores over meaningless coordinates, so nothing downstream
        # (rerank, RAGAS, grounding) could tell fabricated retrieval from real.
        # Callers (retrieval.hybrid) must drop the dense source, not fabricate.
        raise RuntimeError(
            "Embedding model unavailable; refusing to fabricate pseudo-vectors."
        )
    return model.encode(texts, normalize_embeddings=True).tolist()



if __name__ == "__main__":
    vecs = embed_texts(["pump bearing failure", "unrelated text about weather"])
    assert len(vecs) == 2 and len(vecs[0]) == EMBED_DIM
    print(f"OK: {len(vecs)} vectors, dim={len(vecs[0])}")
