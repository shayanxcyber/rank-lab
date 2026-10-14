import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from rank_lab.cli import run_command
from rank_lab.experiments import (
    load_custom_inputs,
    load_prepared,
    prepare_custom,
    read_config,
    training_config,
)
from rank_lab.retrieval import lexical_evaluation


@pytest.fixture
def inputs(tmp_path):
    documents = [
        {"id": "retry", "title": "Delivery retries", "text": "Retry a failed delivery safely."},
        {"id": "restore", "title": "Restoring data", "text": "Restore a backup snapshot."},
    ]
    queries = [
        {"id": "train-a", "text": "How do I retry a delivery?", "split": "train"},
        {"id": "train-b", "text": "How do I restore data?", "split": "train"},
        {"id": "valid", "text": "Recover a snapshot", "split": "validation"},
        {"id": "held-out", "text": "Resend after failure", "split": "test"},
    ]
    for name, rows in (("documents", documents), ("queries", queries)):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    (tmp_path / "judgments.tsv").write_text(
        "query_id\tdocument_id\trelevance\ntrain-a\tretry\t2\ntrain-b\trestore\t1\n"
        "valid\trestore\t2\nheld-out\tretry\t2\nheld-out\trestore\t0\n",
        encoding="utf-8",
    )
    return {
        name: str(tmp_path / filename)
        for name, filename in (
            ("documents", "documents.jsonl"),
            ("queries", "queries.jsonl"),
            ("judgments", "judgments.tsv"),
        )
    }


def test_explicit_splits_preserve_graded_judgments_and_validation(inputs):
    data, hashes = load_custom_inputs(inputs)
    split = data.split()
    assert split["train"] == ["train-a", "train-b"]
    assert split["validation"] == ["valid"]
    assert split["test"] == ["held-out"]
    assert data.test_qrels["held-out"] == {"retry": 2, "restore": 0}
    assert lexical_evaluation(data, split["validation"])["metrics"]["ndcg@10"] == 1
    assert len(hashes) == 3 and all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate", "Duplicate document"),
        ("empty", "text must be nonempty"),
        ("missing_query", "Unknown query"),
        ("missing_document", "Unknown document"),
        ("missing_positive", "no positive judgment"),
        ("fractional_relevance", "integer"),
        ("duplicate_judgment", "Duplicate judgment"),
        ("leaked_text", "query text groups overlap"),
    ],
)
def test_invalid_inputs_fail_before_training(inputs, mutation, message):
    documents = Path(inputs["documents"])
    queries = Path(inputs["queries"])
    judgments = Path(inputs["judgments"])
    if mutation == "duplicate":
        documents.write_text(documents.read_text() + "\n" + documents.read_text().splitlines()[0])
    elif mutation == "empty":
        documents.write_text(documents.read_text().replace("Retry a failed delivery safely.", ""))
    elif mutation == "leaked_text":
        queries.write_text(
            queries.read_text().replace("Resend after failure", "How do I retry a delivery?")
        )
    else:
        text = judgments.read_text()
        text = {
            "missing_query": lambda: text.replace("held-out", "unknown"),
            "missing_document": lambda: text.replace("retry", "missing"),
            "missing_positive": lambda: text.replace("valid\trestore\t2", "valid\trestore\t0"),
            "fractional_relevance": lambda: text.replace(
                "valid\trestore\t2", "valid\trestore\t1.5"
            ),
            "duplicate_judgment": lambda: text + "held-out\tretry\t2\n",
        }[mutation]()
        judgments.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_custom_inputs(inputs)


def test_config_paths_are_relative_and_overrides_are_typed(inputs, tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text("""[experiment]
name = "Support experiments"
workspace = "runs/a"
[data]
documents = "documents.jsonl"
queries = "queries.jsonl"
judgments = "judgments.tsv"
[training]
epochs = 2
device = "cpu"
""")
    config = read_config(path)
    assert config["data"]["documents"] == inputs["documents"]
    assert config["experiment"]["workspace"] == str(tmp_path / "runs/a")
    assert training_config(config["training"], {"epochs": 4, "seed": None}).epochs == 4
    with pytest.raises(ValueError, match="batch_size"):
        training_config({"batch_size": True})
    with pytest.raises(ValueError, match="learning_rate"):
        training_config({"learning_rate": float("nan")})
    path.write_text(path.read_text() + "unknown = true\n")
    with pytest.raises(ValueError, match="Unknown"):
        read_config(path)


def test_prepared_snapshot_survives_source_removal_and_rejects_tampering(inputs, tmp_path):
    config = {
        "experiment": {"name": "Fixture"},
        "data": {**inputs, "validation_fraction": 0.2},
        "training": asdict(training_config({"device": "cpu"})),
    }
    workspace = tmp_path / "run"
    result = prepare_custom(config, workspace, training_config(config["training"]))
    for filename in inputs.values():
        Path(filename).unlink()
    data, _ = load_prepared(workspace)
    assert data.fingerprint == result["dataset_fingerprint"]
    snapshot = workspace / "data/custom.json"
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    with pytest.raises(ValueError, match="snapshot changed"):
        load_prepared(workspace)


def test_prepare_does_not_replace_a_different_run(inputs, tmp_path):
    config = {
        "experiment": {"name": "Fixture"},
        "data": {**inputs, "validation_fraction": 0.2},
        "training": asdict(training_config({"device": "cpu"})),
    }
    workspace = tmp_path / "run"
    prepare_custom(config, workspace, training_config(config["training"]))
    path = Path(inputs["documents"])
    path.write_text(path.read_text().replace("backup snapshot", "different snapshot"))
    with pytest.raises(ValueError, match="different data"):
        prepare_custom(config, workspace, training_config(config["training"]))


def test_train_cannot_silently_fall_back_to_scifact(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text("""[experiment]
workspace = "runs/new"
[data]
documents = "missing.jsonl"
queries = "missing-queries.jsonl"
judgments = "missing.tsv"
""")
    args = SimpleNamespace(command="train", workspace=None, config=path)
    with pytest.raises(ValueError, match="not prepared"):
        run_command(args)


def test_builtin_preparation_preserves_cli_settings_for_later_commands(tmp_path, monkeypatch):
    received = {}

    def prepare(workspace, *, seed, validation_fraction):
        received.update(seed=seed, fraction=validation_fraction)
        return {"prepared": True}

    monkeypatch.setattr("rank_lab.cli.prepare", prepare)
    args = SimpleNamespace(
        command="prepare",
        config=None,
        workspace=tmp_path,
        device="cpu",
        epochs=2,
        batch_size=4,
        learning_rate=None,
        seed=31,
        max_length=None,
        mixed_precision=None,
        validation_fraction=0.25,
        stop_after_steps=None,
        resume=False,
    )
    assert run_command(args) == {"prepared": True}
    assert received == {"seed": 31, "fraction": 0.25}

    def train(workspace, config, **kwargs):
        return {
            "epochs": config.epochs,
            "seed": config.seed,
            "device": config.device,
            "validation_fraction": kwargs["validation_fraction"],
        }

    monkeypatch.setattr("rank_lab.study.train_study", train)
    args.command = "train"
    for key in ("device", "epochs", "batch_size", "seed", "validation_fraction"):
        setattr(args, key, None)
    assert run_command(args) == {
        "epochs": 2,
        "seed": 31,
        "device": "cpu",
        "validation_fraction": 0.25,
    }


def test_preparation_cannot_replace_started_run_metadata(inputs, tmp_path):
    config = {
        "experiment": {"name": "Original name"},
        "data": {**inputs, "validation_fraction": 0.2},
        "training": asdict(training_config({"device": "cpu"})),
    }
    workspace = tmp_path / "run"
    prepare_custom(config, workspace, training_config(config["training"]))
    original = {
        name: (workspace / name).read_bytes()
        for name in ("preparation.json", "run-config.json", "split.json")
    }
    checkpoint = workspace / "training/training-state.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"existing checkpoint")
    config["experiment"]["name"] = "A different experiment name"
    with pytest.raises(ValueError, match="already started"):
        prepare_custom(config, workspace, training_config(config["training"]))
    for name, content in original.items():
        assert (workspace / name).read_bytes() == content
