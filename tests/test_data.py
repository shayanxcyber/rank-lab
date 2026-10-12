from dataclasses import replace

import pytest

from rank_lab.data import RetrievalData


@pytest.fixture
def dataset():
    return RetrievalData(
        {"a": "Retry operations", "b": "Inspect checkpoints", "c": "Compile modules"},
        {
            "q1": "Retry safely",
            "q2": "  RETRY  safely  ",
            "q3": "Resume a checkpoint",
            "q4": "Compile a project",
            "test": "Find checkpoint metadata",
        },
        {"q1": {"a": 1}, "q2": {"a": 1}, "q3": {"b": 1}, "q4": {"c": 1}},
        {"test": {"b": 1}},
    )


def test_split_keeps_text_duplicates_together_and_is_reproducible(dataset):
    split = dataset.split(validation_fraction=0.4)
    assert split == dataset.split(validation_fraction=0.4)
    assert ("q1" in split["train"]) == ("q2" in split["train"])
    assert set(split["train"]) | set(split["validation"]) == set(dataset.train_qrels)
    assert not set(split["train"]) & set(split["validation"])
    assert split["test"] == ["test"]


def test_test_text_leakage_is_rejected(dataset):
    leaking = replace(dataset, queries={**dataset.queries, "test": "retry safely"})
    with pytest.raises(ValueError, match="query text groups overlap"):
        leaking.split()


def test_unknown_judged_document_is_rejected(dataset):
    broken = replace(dataset, test_qrels={"test": {"missing": 1}})
    with pytest.raises(ValueError, match="Unknown document"):
        broken.validate()


def test_fingerprint_changes_with_judgments(dataset):
    changed = replace(dataset, test_qrels={"test": {"a": 1}})
    assert changed.fingerprint != dataset.fingerprint


def test_training_text_duplicates_removed_without_changing_test(dataset):
    raw = replace(dataset, queries={**dataset.queries, "test": "retry safely"})
    cleaned, report = raw.remove_test_duplicates()

    assert set(cleaned.train_qrels) == {"q3", "q4"}
    assert cleaned.test_qrels == raw.test_qrels
    assert raw.train_qrels == dataset.train_qrels
    assert report["excluded_training_queries"] == [
        {"train_id": "q1", "test_ids": ["test"]},
        {"train_id": "q2", "test_ids": ["test"]},
    ]
    assert report["raw_fingerprint"] != report["clean_fingerprint"]
    cleaned.validate()
