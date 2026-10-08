"""Pure retrieval-quality metrics for the ablation/correctness benchmark.

The ground-truth files use `expected_any` — a case is "hit" if ANY expected key
is retrieved — so the primary metric is hit@k, complemented by recall@k (share
of expected keys found) and MRR (how highly the first relevant key ranks).
These measure whether retrieval actually surfaced the right evidence, which
faithfulness/precision/relevancy (LLM-judged over whatever WAS retrieved) do
not — a confidently-grounded answer over incomplete context still scores well
on those but fails here."""
from __future__ import annotations


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant key appears in the top-k retrieved, else 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant keys found in the top-k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant key (1-indexed), else 0.0."""
    for idx, key in enumerate(retrieved, start=1):
        if key in relevant:
            return 1.0 / idx
    return 0.0


def aggregate(per_case: list[tuple[list[str], set[str]]], k: int = 5) -> dict:
    """Mean hit@k, recall@k, and MRR over cases. Each case is
    (retrieved_keys_in_rank_order, relevant_keys)."""
    n = len(per_case)
    if n == 0:
        return {"n": 0, "hit_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "k": k}
    return {
        "n": n,
        "k": k,
        "hit_at_k": sum(hit_at_k(r, rel, k) for r, rel in per_case) / n,
        "recall_at_k": sum(recall_at_k(r, rel, k) for r, rel in per_case) / n,
        "mrr": sum(reciprocal_rank(r, rel) for r, rel in per_case) / n,
    }


if __name__ == "__main__":
    cases = [(["FE-001", "X"], {"FE-001"}), (["A", "WO-1003"], {"WO-1003"}), (["A", "B"], {"Z"})]
    agg = aggregate(cases, k=5)
    assert agg["hit_at_k"] == 2 / 3, agg
    assert abs(agg["mrr"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9, agg
    print("OK:", agg)
