import json

import pytest

from rank_lab.artifacts import atomic_json, sha256
from rank_lab.data import RetrievalData
from rank_lab.metrics import evaluate_rankings
from rank_lab.presentation import export_presentation


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    documents = {str(i): f"Document {i}\nPublic fixture abstract {i}" for i in range(10)}
    data = RetrievalData(
        documents,
        {"1": "Train", "2": "Validation", "3": "Reserved"},
        {"1": {"1": 1}, "2": {"2": 1}},
        {"3": {"3": 1}},
    )
    ranking = {"3": list(documents)}
    run = {**evaluate_rankings(ranking, data.test_qrels), "rankings": ranking}
    atomic_json(
        tmp_path / "comparison.json",
        {
            "dataset_fingerprint": data.fingerprint,
            "query_count": 1,
            "selected_epoch": 1,
            "runs": {m: run for m in ("lexical", "pretrained", "fine_tuned")},
        },
    )
    atomic_json(
        tmp_path / "split.json",
        {
            "dataset_fingerprint": data.fingerprint,
            "train": ["1"],
            "validation": ["2"],
            "test": ["3"],
        },
    )
    atomic_json(
        tmp_path / "training/training.json",
        {
            "status": "completed",
            "step": 3,
            "trainable_parameters": 12,
            "peak_cuda_allocated_bytes": 0,
            "weights": "final",
            "initial_weights": "base",
            "history": [{"step": 1, "loss": 0.5}],
            "elapsed_seconds_this_invocation": 1,
        },
    )
    atomic_json(
        tmp_path / "experiment.json",
        {
            "base_model": "fixture",
            "base_revision": "revision",
            "config": {"epochs": 3},
            "platform": "test",
            "device": "CPU",
            "torch": "test",
        },
    )
    for path in (
        "validation-baseline.json",
        "training/epoch-1/validation.json",
        "training/epoch-2/validation.json",
        "training/epoch-3/validation.json",
    ):
        atomic_json(tmp_path / path, {"metrics": {"ndcg@10": 0.5}})
    monkeypatch.setattr("rank_lab.presentation.load_scifact", lambda _: data)
    return tmp_path


def test_export_is_offline_and_preserves_rankings_without_runtime_paths(recorded, tmp_path):
    destination = tmp_path / "presentation"
    result = export_presentation(recorded, destination, resumed_after_step=2)
    study = json.loads((destination / "study.json").read_text())
    abstracts = json.loads((destination / "document-text.json").read_text())
    assert result["queries"] == 1
    assert study["queries"][0]["rankings"]["fine_tuned"] == [str(i) for i in range(10)]
    assert study["document_text_sha256"] == sha256(destination / "document-text.json")
    assert study["documents"]["3"] == {"title": "Document 3"}
    assert abstracts["3"] == "Public fixture abstract 3"
    assert str(recorded) not in (destination / "study.json").read_text()
    assert study["training"]["resumed_after_step"] == 2


def test_export_rejects_data_identity_drift(recorded, tmp_path):
    path = recorded / "comparison.json"
    report = json.loads(path.read_text())
    report["dataset_fingerprint"] = "different"
    atomic_json(path, report)
    with pytest.raises(ValueError, match="differ"):
        export_presentation(recorded, tmp_path / "presentation")


def test_export_rejects_incomplete_reserved_query_results(recorded, tmp_path):
    path = recorded / "comparison.json"
    report = json.loads(path.read_text())
    report["runs"]["fine_tuned"]["rankings"] = {}
    atomic_json(path, report)
    with pytest.raises(ValueError, match="every reserved query"):
        export_presentation(recorded, tmp_path / "presentation")


def test_resume_annotation_is_explicit_and_cannot_follow_completion(recorded, tmp_path):
    destination = tmp_path / "presentation"
    export_presentation(recorded, destination)
    assert (
        json.loads((destination / "study.json").read_text())["training"]["resumed_after_step"]
        is None
    )
    with pytest.raises(ValueError, match="before completion"):
        export_presentation(recorded, destination, resumed_after_step=3)


def test_export_rejects_scores_detached_from_rankings(recorded, tmp_path):
    path = recorded / "comparison.json"
    report = json.loads(path.read_text())
    report["runs"]["fine_tuned"]["metrics"]["ndcg@10"] = 0.99
    atomic_json(path, report)
    with pytest.raises(ValueError, match="scores differ"):
        export_presentation(recorded, tmp_path / "presentation")
