"""Tenant isolation in graph traversal: records tagged for another site are
never returned; untagged (single-tenant/legacy) records always pass; no filter
means no change. Deterministic — fake session returns mixed-site rows."""
from retrieval import graph_traversal
from retrieval.graph_traversal import _site_ok, traverse


class _Result:
    def __init__(self, single):
        self._single = single

    def single(self):
        return self._single

    def data(self):
        return []


class _Session:
    """1-hop equipment row with a same-site failure, a cross-site failure, and
    an untagged clause."""

    def run(self, cypher, **_kwargs):
        return _Result({
            "failure_events": [
                {"id": "FE-MUM", "date": "2025-03-14", "symptom": "s", "root_cause": "r", "site": "plant-mumbai-01"},
                {"id": "FE-JAM", "date": "2025-04-01", "symptom": "s", "root_cause": "r", "site": "plant-jamnagar-02"},
            ],
            "work_orders": [],
            "clauses": [{"id": "C-LEGACY", "source": "x", "text": "t", "site": None}],
            "procedures": [],
            "chunks": [],
        })


def _keys(monkeypatch, site_id):
    monkeypatch.setattr(graph_traversal, "load_known_entities", lambda _s: ({"P-101"}, set()))
    monkeypatch.setattr(graph_traversal, "extract_query_entities", lambda *_a: (["P-101"], []))
    return dict(traverse(_Session(), "status of P-101?", top_k=None, site_id=site_id))


def test_site_ok_helper():
    assert _site_ok({"site": None}, "plant-a") is True          # untagged passes
    assert _site_ok({"site": "plant-a"}, "plant-a") is True     # same site
    assert _site_ok({"site": "plant-b"}, "plant-a") is False    # cross-site blocked
    assert _site_ok({"site": "plant-b"}, None) is True          # no filter requested


def test_no_site_filter_returns_all(monkeypatch):
    keys = _keys(monkeypatch, site_id=None)
    assert {"FE-MUM", "FE-JAM", "C-LEGACY"} <= set(keys)


def test_mumbai_caller_cannot_see_jamnagar_records(monkeypatch):
    keys = _keys(monkeypatch, site_id="plant-mumbai-01")
    assert "FE-MUM" in keys
    assert "C-LEGACY" in keys        # untagged legacy data still visible
    assert "FE-JAM" not in keys      # cross-tenant record blocked


def test_jamnagar_caller_cannot_see_mumbai_records(monkeypatch):
    keys = _keys(monkeypatch, site_id="plant-jamnagar-02")
    assert "FE-JAM" in keys
    assert "FE-MUM" not in keys


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
