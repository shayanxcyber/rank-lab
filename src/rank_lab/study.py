"""Validation-led training and the reserved final benchmark comparison."""

import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

from rank_lab.artifacts import (
    DATA_ASSETS,
    MODEL_ASSETS,
    MODEL_ID,
    MODEL_REVISION,
    atomic_json,
    fetch_assets,
    sha256,
)
from rank_lab.data import load_scifact


def train_study(
    workspace: Path, config, *, resume: bool = False, stop_after_steps: int | None = None
) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    from rank_lab.encoder import Encoder
    from rank_lab.retrieval import dense_evaluation
    from rank_lab.training import train, weights_fingerprint

    fetch_assets(DATA_ASSETS, workspace / "data")
    fetch_assets(MODEL_ASSETS, workspace / "base-model")
    data, exclusions = load_scifact(workspace / "data").remove_test_duplicates()
    split = data.split(seed=config.seed)
    output = workspace / "training"
    experiment_path = workspace / "experiment.json"
    if resume:
        if not (output / "training-state.pt").exists():
            raise ValueError("No checkpoint exists to resume")
        previous = json.loads(experiment_path.read_text(encoding="utf-8"))
        if (
            previous["config"] != asdict(config)
            or previous["dataset_fingerprint"] != data.fingerprint
            or previous["base_revision"] != MODEL_REVISION
            or previous["torch"] != str(torch.__version__)
        ):
            raise ValueError(
                "Resume configuration/data contract does not match the existing experiment"
            )
    elif (output / "training-state.pt").exists():
        raise ValueError("Training state already exists; use resume or choose a new workspace")
    atomic_json(workspace / "split.json", split)
    atomic_json(workspace / "split-exclusions.json", exclusions)
    torch.manual_seed(config.seed)
    encoder = Encoder(workspace / "base-model", device=config.device, max_length=config.max_length)
    atomic_json(
        workspace / "experiment.json",
        {
            "dataset": "BeIR/scifact",
            "dataset_fingerprint": data.fingerprint,
            "documents": len(data.documents),
            "train_queries": len(split["train"]),
            "validation_queries": len(split["validation"]),
            "reserved_test_queries": len(split["test"]),
            "excluded_training_queries": len(exclusions["excluded_training_queries"]),
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "config": asdict(config),
            "selection_metric": "validation ndcg@10; earlier epoch wins exact ties",
            "python": platform.python_version(),
            "platform": platform.system(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if config.device == "cuda" else "CPU",
        },
    )
    baseline_path = workspace / "validation-baseline.json"
    if not baseline_path.exists():
        baseline = dense_evaluation(encoder, data, split["validation"])
        atomic_json(baseline_path, baseline)
        print(
            json.dumps({"event": "validation_baseline", "metrics": baseline["metrics"]}), flush=True
        )

    def evaluate_epoch(model, epoch):
        report = dense_evaluation(model, data, split["validation"])
        destination = output / f"epoch-{epoch}"
        model.save(destination / "model")
        model_manifest = {
            path.name: sha256(path) for path in (destination / "model").iterdir() if path.is_file()
        }
        atomic_json(destination / "model-manifest.json", model_manifest)
        atomic_json(destination / "validation.json", report)
        selection_path = output / "selection.json"
        selected = (
            json.loads(selection_path.read_text(encoding="utf-8"))
            if selection_path.exists()
            else None
        )
        if (
            selected is None
            or report["metrics"]["ndcg@10"] > selected["validation_metrics"]["ndcg@10"]
        ):
            atomic_json(
                selection_path,
                {
                    "epoch": epoch,
                    "checkpoint": f"epoch-{epoch}/model",
                    "validation_metrics": report["metrics"],
                    "weights_fingerprint": weights_fingerprint(model),
                },
            )
        print(
            json.dumps({"event": "validation_epoch", "epoch": epoch, "metrics": report["metrics"]}),
            flush=True,
        )

    result = train(
        encoder,
        data,
        split,
        config,
        output,
        resume=resume,
        stop_after_steps=stop_after_steps,
        evaluate_epoch=evaluate_epoch,
    )
    return {
        "status": result["status"],
        "step": result["step"],
        "weights_changed": result["weights"] != result["initial_weights"],
        "initial_weights": result["initial_weights"],
        "weights": result["weights"],
    }


def final_comparison(workspace: Path, *, device: str = "cuda") -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from rank_lab.encoder import Encoder
    from rank_lab.retrieval import dense_evaluation, lexical_evaluation
    from rank_lab.training import weights_fingerprint

    output = workspace / "training"
    training = json.loads((output / "training.json").read_text(encoding="utf-8"))
    if training["status"] != "completed":
        raise ValueError(
            "Complete training and validation selection before evaluating the reserved test set"
        )
    result_path = workspace / "comparison.json"
    if result_path.exists():
        raise ValueError(
            "Final comparison already exists; preserve this run rather than overwriting it"
        )
    data, _ = load_scifact(workspace / "data").remove_test_duplicates()
    split = json.loads((workspace / "split.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    experiment = json.loads((workspace / "experiment.json").read_text(encoding="utf-8"))
    if data.fingerprint != split["dataset_fingerprint"]:
        raise ValueError("Data changed after split preparation")
    reports = {"lexical": lexical_evaluation(data, split["test"])}
    for name, directory in (
        ("pretrained", workspace / "base-model"),
        ("fine_tuned", output / selection["checkpoint"]),
    ):
        if name == "fine_tuned":
            manifest_path = output / f"epoch-{selection['epoch']}" / "model-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if set(manifest) != {path.name for path in directory.iterdir() if path.is_file()}:
                raise ValueError("Checkpoint file set changed after validation selection")
            if any(sha256(directory / filename) != digest for filename, digest in manifest.items()):
                raise ValueError("Checkpoint bytes changed after validation selection")
        encoder = Encoder(directory, device=device, max_length=experiment["config"]["max_length"])
        if (
            name == "fine_tuned"
            and weights_fingerprint(encoder) != selection["weights_fingerprint"]
        ):
            raise ValueError("Checkpoint weights changed after validation selection")
        reports[name] = dense_evaluation(encoder, data, split["test"])
        del encoder
    report = {
        "schema_version": 1,
        "dataset_fingerprint": data.fingerprint,
        "query_count": len(split["test"]),
        "selected_epoch": selection["epoch"],
        "runs": reports,
        "judgment_scope": "BEIR SciFact relevance judgments; unjudged documents receive zero gain",
        "text_scope": "All methods receive title plus abstract; dense encoders truncate at the recorded wordpiece limit",
    }
    atomic_json(result_path, report)
    return {
        "query_count": report["query_count"],
        "selected_epoch": selection["epoch"],
        "metrics": {name: value["metrics"] for name, value in reports.items()},
    }
