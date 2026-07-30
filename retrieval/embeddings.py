"""Local dense embeddings, shared by Chunk indexing and query-time search.
sentence-transformers, not a hosted embed API — no per-query external call,
no quota risk (same reasoning as the Groq/local-Qdrant/local-rerank choices,
see NOTES.md).
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_model().encode(texts, normalize_embeddings=True).tolist()


if __name__ == "__main__":
    vecs = embed_texts(["pump bearing failure", "unrelated text about weather"])
    assert len(vecs) == 2 and len(vecs[0]) == EMBED_DIM
    print(f"OK: {len(vecs)} vectors, dim={len(vecs[0])}")
