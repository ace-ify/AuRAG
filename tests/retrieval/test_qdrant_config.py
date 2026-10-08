def test_qdrant_client_receives_optional_api_key(monkeypatch):
    from retrieval import qdrant_store

    calls = {}

    class Client:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY", "secret")
    monkeypatch.setattr(qdrant_store, "QdrantClient", Client)
    monkeypatch.setattr(qdrant_store, "_client", None)

    qdrant_store.get_client()

    # url + api_key must be forwarded; a bounded timeout is a deliberate
    # reliability guard (don't let a hung Qdrant block retrieval), so assert the
    # invariants rather than brittle exact-equality that would forbid it.
    assert calls["url"] == "https://qdrant.example"
    assert calls["api_key"] == "secret"
    assert 0 < calls["timeout"] <= 10
