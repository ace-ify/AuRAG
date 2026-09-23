"""Multi-hop GraphRAG: depth=2 must surface connected-equipment evidence that
a 1-hop lookup (and dense/BM25, which never see structured records) cannot, and
the hybrid ranker must let that graph provenance influence order."""
from retrieval import graph_traversal
from retrieval.hybrid import _blend_graph_provenance


class _FakeResult:
    def __init__(self, single=None, data=None):
        self._single = single
        self._data = data or []

    def single(self):
        return self._single

    def data(self):
        return self._data


class _FakeSession:
    """V-301 (pressure vessel) HAS_PART PSV-701 (relief valve). V-301's own
    1-hop neighbourhood carries only the Factories-Act clause; the missed-
    calibration failure + OISD clause live on PSV-701, one hop further."""

    def run(self, cypher, **_kwargs):
        if "HAS_PART" in cypher:
            return _FakeResult(data=[{
                "nbr_tag": "PSV-701",
                "failure_events": [{
                    "id": "FE-004", "date": "2025-08-11",
                    "symptom": "PSV-701 lifted at ~90% of set pressure",
                    "root_cause": "spring fatigue + missed annual calibration (WO-1007)",
                }],
                "work_orders": [],
                "clauses": [{
                    "id": "OISD-STD-132-10.2ii", "source": "OISD-STD-132",
                    "text": "Testing and maintenance history must be provided before calibration.",
                }],
                "procedures": [],
            }])
        return _FakeResult(single={
            "failure_events": [],
            "work_orders": [],
            "clauses": [{
                "id": "FACT1948-S31", "source": "Factories Act 1948",
                "text": "Safe working pressure of pressure plant must not be exceeded.",
            }],
            "procedures": [],
            "chunks": [],
        })


def _traverse(monkeypatch, depth):
    monkeypatch.setattr(graph_traversal, "load_known_entities", lambda _s: ({"V-301", "PSV-701"}, set()))
    monkeypatch.setattr(graph_traversal, "extract_query_entities", lambda *_a: (["V-301"], []))
    return dict(graph_traversal.traverse(_FakeSession(), "compliance status of V-301?", top_k=None, depth=depth))


def test_one_hop_misses_connected_relief_valve_evidence(monkeypatch):
    hits = _traverse(monkeypatch, depth=1)
    assert "FACT1948-S31" in hits          # V-301's own clause
    assert "FE-004" not in hits            # lives on the part, 2 hops away
    assert "OISD-STD-132-10.2ii" not in hits


def test_two_hop_surfaces_part_evidence_with_provenance(monkeypatch):
    hits = _traverse(monkeypatch, depth=2)
    assert "FACT1948-S31" in hits          # still keeps the direct hit
    assert "FE-004" in hits                # reached via HAS_PART
    assert "OISD-STD-132-10.2ii" in hits
    assert "HAS_PART" in hits["FE-004"] and "PSV-701" in hits["FE-004"]  # traversed edge is attributed


def test_blend_promotes_close_graph_record_over_marginally_better_chunk():
    reranked = [("CHUNK-A", "chatty chunk", 0.80), ("FE-004", "structured record", 0.75), ("X", "junk", 0.0)]
    out = _blend_graph_provenance(reranked, graph_keys={"FE-004"}, top_k=2)
    assert [k for k, _, _ in out] == ["FE-004", "CHUNK-A"]


def test_blend_does_not_override_a_decisively_better_semantic_hit():
    reranked = [("CHUNK-A", "clearly on-topic", 0.90), ("FE-004", "off-topic record", 0.0)]
    out = _blend_graph_provenance(reranked, graph_keys={"FE-004"}, top_k=2)
    assert out[0][0] == "CHUNK-A"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
