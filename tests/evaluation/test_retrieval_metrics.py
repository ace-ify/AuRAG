"""Retrieval-quality metrics, the ablation harness, and the de-tautologized
benchmark runner."""
from evaluation import ablation, run_benchmark
from evaluation.retrieval_metrics import (
    aggregate,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_and_recall_and_rr():
    retrieved = ["A", "FE-001", "C"]
    relevant = {"FE-001", "WO-9"}
    assert hit_at_k(retrieved, relevant, k=5) == 1.0
    assert hit_at_k(retrieved, relevant, k=1) == 0.0          # FE-001 is rank 2
    assert recall_at_k(retrieved, relevant, k=5) == 0.5       # found 1 of 2
    assert reciprocal_rank(retrieved, relevant) == 0.5        # first hit at rank 2
    assert reciprocal_rank(["X", "Y"], relevant) == 0.0


def test_aggregate_means():
    cases = [(["FE-001"], {"FE-001"}), (["X", "WO-3"], {"WO-3"}), (["X"], {"Z"})]
    agg = aggregate(cases, k=5)
    assert agg["n"] == 3
    assert agg["hit_at_k"] == 2 / 3
    assert abs(agg["mrr"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9


def test_ablation_measures_per_config_lift():
    cases = [
        {"query": "q1", "expected_any": ["FE-001"]},
        {"query": "q2", "expected_any": ["WO-1003"]},
    ]
    retrievers = {
        "dense-only": lambda q: ["IRRELEVANT"],                       # misses both
        "full": lambda q: ["FE-001"] if q == "q1" else ["WO-1003"],    # hits both
    }
    results = ablation.run_ablation(cases, retrievers, k=5)
    assert results["dense-only"]["hit_at_k"] == 0.0
    assert results["full"]["hit_at_k"] == 1.0                          # the lift is measured, not assumed


def test_ablation_retriever_error_counts_as_miss_not_crash():
    def boom(_q):
        raise RuntimeError("retriever down")

    results = ablation.run_ablation(
        [{"query": "q", "expected_any": ["A"]}], {"broken": boom}, k=5
    )
    assert results["broken"]["hit_at_k"] == 0.0


def test_format_table_orders_by_hit_and_lists_configs():
    results = {
        "dense-only": {"n": 2, "k": 5, "hit_at_k": 0.5, "recall_at_k": 0.25, "mrr": 0.3},
        "full": {"n": 2, "k": 5, "hit_at_k": 1.0, "recall_at_k": 0.8, "mrr": 0.9},
    }
    table = ablation.format_table(results)
    lines = table.splitlines()
    assert "full" in lines[2] and "dense-only" in lines[3]  # full ranks first


def test_load_cases_reads_real_ground_truth():
    cases = ablation.load_cases()
    assert len(cases) >= 5
    assert all(c["query"] and c["expected_any"] for c in cases)


def test_run_benchmark_offline_does_not_fabricate_accuracy():
    summary = run_benchmark.run_benchmark(mode="offline")
    assert summary["mode"] == "offline"
    assert summary["passed"] is None      # no model ran -> no pass count
    assert summary["accuracy"] is None    # the old 100% tautology is gone
    assert summary["total"] > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
