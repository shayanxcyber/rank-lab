"""Full-corpus lexical and normalized dense ranking."""

import re
import time
from collections import Counter
from math import log

import numpy as np

from rank_lab.data import RetrievalData
from rank_lab.metrics import evaluate_rankings, rank_scores


class BM25:
    def __init__(self, documents: dict[str, str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.document_ids = sorted(documents)
        self.k1 = k1
        self.b = b
        self.lengths = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, key in enumerate(self.document_ids):
            tokens = re.findall(r"\w+", documents[key].casefold())
            self.lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings.setdefault(term, []).append((index, count))
        self.average_length = sum(self.lengths) / len(self.lengths)

    def rank(self, query: str, *, cutoff: int = 10) -> list[str]:
        scores = np.zeros(len(self.document_ids), dtype=np.float64)
        for term in re.findall(r"\w+", query.casefold()):
            postings = self.postings.get(term, [])
            idf = log(1 + (len(self.document_ids) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, count in postings:
                denominator = count + self.k1 * (
                    1 - self.b + self.b * self.lengths[index] / self.average_length
                )
                scores[index] += idf * count * (self.k1 + 1) / denominator
        return rank_scores(self.document_ids, scores.tolist(), cutoff=cutoff)


def lexical_evaluation(data: RetrievalData, query_ids: list[str]) -> dict:
    started = time.perf_counter()
    index = BM25(data.documents)
    rankings = {key: index.rank(data.queries[key]) for key in query_ids}
    qrels = {
        key: data.test_qrels.get(
            key, (data.validation_qrels or {}).get(key, data.train_qrels.get(key, {}))
        )
        for key in query_ids
    }
    report = evaluate_rankings(rankings, qrels)
    report.update(
        {
            "rankings": rankings,
            "elapsed_seconds": time.perf_counter() - started,
            "method": "BM25",
            "k1": 1.5,
            "b": 0.75,
            "document_count": len(data.documents),
        }
    )
    return report


def dense_evaluation(
    encoder, data: RetrievalData, query_ids: list[str], *, batch_size: int = 32
) -> dict:
    started = time.perf_counter()
    document_ids = sorted(data.documents)
    document_vectors = encoder.encode(
        [data.documents[key] for key in document_ids], batch_size=batch_size
    )
    query_vectors = encoder.encode([data.queries[key] for key in query_ids], batch_size=batch_size)
    scores = query_vectors @ document_vectors.T
    rankings = {
        key: rank_scores(document_ids, scores[row].tolist()) for row, key in enumerate(query_ids)
    }
    qrels = {
        key: data.test_qrels.get(
            key, (data.validation_qrels or {}).get(key, data.train_qrels.get(key, {}))
        )
        for key in query_ids
    }
    report = evaluate_rankings(rankings, qrels)
    report.update(
        {
            "rankings": rankings,
            "elapsed_seconds": time.perf_counter() - started,
            "method": "normalized dense cosine",
            "max_length": encoder.max_length,
            "document_count": len(data.documents),
        }
    )
    return report
