"""Per-query retrieval metrics with deterministic tie order."""

from math import isfinite, log2
from statistics import mean


def ranking_metrics(ranked: list[str], judgments: dict[str, int], *, cutoff: int = 10) -> dict:
    if cutoff < 1:
        raise ValueError("Cutoff must be positive")
    positives = {key: score for key, score in judgments.items() if score > 0}
    if not positives:
        raise ValueError("Metrics require at least one positive judgment")
    if len(ranked) != len(set(ranked)):
        raise ValueError("Rankings must not contain duplicate document IDs")
    top = ranked[:cutoff]
    gains = [(2 ** positives.get(key, 0) - 1) / log2(rank + 2) for rank, key in enumerate(top)]
    ideal = sorted(positives.values(), reverse=True)[:cutoff]
    idcg = sum((2**score - 1) / log2(rank + 2) for rank, score in enumerate(ideal))
    first = next((rank + 1 for rank, key in enumerate(top) if key in positives), None)
    return {
        f"ndcg@{cutoff}": sum(gains) / idcg,
        f"recall@{cutoff}": sum(key in positives for key in top) / len(positives),
        f"mrr@{cutoff}": 1 / first if first is not None else 0.0,
    }


def evaluate_rankings(
    rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], *, cutoff: int = 10
) -> dict:
    if not qrels:
        raise ValueError("Evaluation needs a nonempty query set")
    if set(rankings) != set(qrels):
        raise ValueError("Rankings must match the evaluated query set exactly")
    per_query = {
        key: ranking_metrics(rankings[key], qrels[key], cutoff=cutoff) for key in sorted(qrels)
    }
    keys = (f"ndcg@{cutoff}", f"recall@{cutoff}", f"mrr@{cutoff}")
    return {
        "query_count": len(qrels),
        "metrics": {key: mean(row[key] for row in per_query.values()) for key in keys},
        "per_query": per_query,
    }


def rank_scores(document_ids: list[str], scores: list[float], *, cutoff: int = 10) -> list[str]:
    if len(document_ids) != len(scores):
        raise ValueError("Scores and document IDs must have the same length")
    if cutoff < 1:
        raise ValueError("Cutoff must be positive")
    if any(not isfinite(score) for score in scores):
        raise ValueError("Ranking scores must be finite")
    return [
        key
        for key, _ in sorted(
            zip(document_ids, scores, strict=True), key=lambda item: (-item[1], item[0])
        )[:cutoff]
    ]
