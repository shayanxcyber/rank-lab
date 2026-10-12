from math import log2

import pytest

from rank_lab.metrics import evaluate_rankings, rank_scores, ranking_metrics


def test_graded_relevance_and_multiple_positives():
    result = ranking_metrics(["weak", "irrelevant", "strong"], {"strong": 3, "weak": 1}, cutoff=2)
    assert result == {"ndcg@2": 1 / (7 + 1 / log2(3)), "recall@2": 0.5, "mrr@2": 1.0}


def test_metrics_average_each_query_equally():
    result = evaluate_rankings({"a": ["hit"], "b": ["miss"]}, {"a": {"hit": 1}, "b": {"other": 1}})
    assert result["metrics"] == {"ndcg@10": 0.5, "recall@10": 0.5, "mrr@10": 0.5}


def test_score_ties_use_document_identity():
    assert rank_scores(["b", "a", "c"], [0.8, 0.8, 0.1]) == ["a", "b", "c"]


def test_missing_query_output_is_rejected():
    with pytest.raises(ValueError, match="query set exactly"):
        evaluate_rankings({}, {"q": {"d": 1}})


def test_duplicate_documents_cannot_inflate_metrics():
    with pytest.raises(ValueError, match="duplicate document"):
        ranking_metrics(["d", "d"], {"d": 1})
