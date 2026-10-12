"""Validate retrieval inputs and keep query groups out of other splits."""

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path


def query_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RetrievalData:
    documents: dict[str, str]
    queries: dict[str, str]
    train_qrels: dict[str, dict[str, int]]
    test_qrels: dict[str, dict[str, int]]

    def validate(self) -> None:
        self._validate_references()
        train_keys = {query_key(self.queries[key]) for key in self.train_qrels}
        test_keys = {query_key(self.queries[key]) for key in self.test_qrels}
        if train_keys & test_keys:
            raise ValueError("Training and test query text groups overlap")

    def _validate_references(self) -> None:
        if not self.documents or not self.queries or not self.train_qrels or not self.test_qrels:
            raise ValueError("Corpus, queries and both judgment splits must be nonempty")
        if self.train_qrels.keys() & self.test_qrels.keys():
            raise ValueError("Training and test query IDs overlap")
        for qrels in (self.train_qrels, self.test_qrels):
            for query_id, judgments in qrels.items():
                if query_id not in self.queries:
                    raise ValueError(f"Unknown query in judgments: {query_id}")
                if not judgments or not any(score > 0 for score in judgments.values()):
                    raise ValueError(f"Query has no positive judgment: {query_id}")
                for document_id, score in judgments.items():
                    if document_id not in self.documents:
                        raise ValueError(f"Unknown document in judgments: {document_id}")
                    if score < 0:
                        raise ValueError("Judgment scores cannot be negative")

    def remove_test_duplicates(self) -> tuple["RetrievalData", dict]:
        self._validate_references()
        test_groups = {}
        for key in self.test_qrels:
            test_groups.setdefault(query_key(self.queries[key]), []).append(key)
        excluded = [
            {"train_id": key, "test_ids": sorted(test_groups[query_key(self.queries[key])])}
            for key in sorted(self.train_qrels)
            if query_key(self.queries[key]) in test_groups
        ]
        excluded_ids = {row["train_id"] for row in excluded}
        cleaned = replace(
            self,
            train_qrels={
                key: value for key, value in self.train_qrels.items() if key not in excluded_ids
            },
        )
        cleaned.validate()
        return cleaned, {
            "policy": "Remove training queries with exact normalized test text; keep test intact",
            "raw_fingerprint": self.fingerprint,
            "clean_fingerprint": cleaned.fingerprint,
            "excluded_training_queries": excluded,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "documents": self.documents,
                "queries": self.queries,
                "train_qrels": self.train_qrels,
                "test_qrels": self.test_qrels,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def split(self, *, seed: int = 1729, validation_fraction: float = 0.1) -> dict:
        self.validate()
        if not 0 < validation_fraction < 1:
            raise ValueError("Validation fraction must be between zero and one")
        groups: dict[str, list[str]] = {}
        for query_id in sorted(self.train_qrels):
            groups.setdefault(query_key(self.queries[query_id]), []).append(query_id)
        if len(groups) < 2:
            raise ValueError("At least two training query groups are required")
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        count = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
        validation_keys = set(keys[:count])
        train = sorted(
            query for key, group in groups.items() if key not in validation_keys for query in group
        )
        validation = sorted(
            query for key, group in groups.items() if key in validation_keys for query in group
        )
        return {
            "schema_version": 1,
            "dataset_fingerprint": self.fingerprint,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "train": train,
            "validation": validation,
            "test": sorted(self.test_qrels),
            "train_groups": len(keys) - count,
            "validation_groups": count,
            "query_grouping": "casefold and collapsed whitespace; exact normalized text",
        }


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            query_id, document_id, score = row["query-id"], row["corpus-id"], int(row["score"])
            judgments = result.setdefault(query_id, {})
            if document_id in judgments:
                raise ValueError(f"Duplicate judgment: {query_id}/{document_id}")
            judgments[document_id] = score
    return result


def load_scifact(directory: Path) -> RetrievalData:
    import pyarrow.parquet as parquet

    documents = {}
    for row in parquet.read_table(directory / "corpus.parquet").to_pylist():
        key = str(row["_id"])
        if key in documents:
            raise ValueError(f"Duplicate document ID: {key}")
        documents[key] = f"{row['title']}\n{row['text']}".strip()
    queries = {}
    for row in parquet.read_table(directory / "queries.parquet").to_pylist():
        key = str(row["_id"])
        if key in queries:
            raise ValueError(f"Duplicate query ID: {key}")
        queries[key] = row["text"]
    result = RetrievalData(
        documents, queries, read_qrels(directory / "train.tsv"), read_qrels(directory / "test.tsv")
    )
    result._validate_references()
    return result
