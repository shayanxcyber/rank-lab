"""Local preparation and device verification for the retrieval study."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rank_lab.artifacts import DATA_ASSETS, MODEL_ASSETS, atomic_json, fetch_assets
from rank_lab.data import load_scifact


def prepare(workspace: Path, *, seed: int = 1729, validation_fraction: float = 0.1) -> dict:
    data_manifest = fetch_assets(DATA_ASSETS, workspace / "data")
    model_manifest = fetch_assets(MODEL_ASSETS, workspace / "base-model")
    raw_data = load_scifact(workspace / "data")
    data, exclusions = raw_data.remove_test_duplicates()
    atomic_json(workspace / "split-exclusions.json", exclusions)
    split = data.split(seed=seed, validation_fraction=validation_fraction)
    atomic_json(workspace / "split.json", split)
    result = {
        "dataset_fingerprint": data.fingerprint,
        "documents": len(data.documents),
        "queries": len(data.queries),
        "excluded_training_queries": len(exclusions["excluded_training_queries"]),
        "train_queries": len(split["train"]),
        "validation_queries": len(split["validation"]),
        "test_queries": len(split["test"]),
        "verified_data_files": len(data_manifest["assets"]),
        "verified_model_files": len(model_manifest["assets"]),
    }
    atomic_json(workspace / "preparation.json", result)
    return result


def device_probe() -> dict:
    import torch

    available = torch.cuda.is_available()
    result = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
    }
    if available:
        tensor = torch.ones((16, 16), device="cuda")
        result.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "matrix_product_sum": float((tensor @ tensor).sum().item()),
                "compiled_architectures": torch.cuda.get_arch_list(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable retrieval experiments and result export"
    )
    parser.add_argument("command", choices=("prepare", "device", "train", "compare", "export"))
    parser.add_argument("--config", type=Path, help="Custom-data TOML experiment configuration")
    parser.add_argument("--workspace", type=Path, help="Override the run workspace")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument(
        "--destination", type=Path, help="Export directory; defaults to WORKSPACE/export"
    )
    parser.add_argument("--resumed-after-step", type=int)
    args = parser.parse_args()
    try:
        result = run_command(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


def run_command(args) -> dict:
    from rank_lab.experiments import prepare_custom, read_config, training_config

    workspace = args.workspace or Path("runs/scifact")
    spec = read_config(args.config) if args.config else None
    if spec is not None and args.workspace is None:
        workspace = Path(spec["experiment"]["workspace"])
    if spec is None and (workspace / "run-config.json").exists():
        spec = json.loads((workspace / "run-config.json").read_text(encoding="utf-8"))
    if args.command == "device":
        result = device_probe()
        atomic_json(workspace / "device.json", result)
        return result
    if (
        spec is not None
        and spec.get("kind", "custom") == "custom"
        and args.command != "prepare"
        and not (workspace / "data/custom.json").exists()
    ):
        raise ValueError(
            "Custom data is not prepared; run prepare with the experiment config first"
        )
    overrides = {
        key: getattr(args, key)
        for key in (
            "device",
            "epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "max_length",
            "mixed_precision",
        )
    }
    config = training_config(spec.get("training", {}) if spec else {}, overrides)
    fraction = args.validation_fraction
    if fraction is None:
        fraction = spec["data"].get("validation_fraction", 0.1) if spec else 0.1
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("Stop-after-steps must be positive")
    if args.command == "prepare":
        if spec is not None:
            spec["experiment"]["workspace"] = str(workspace.resolve())
        if spec is None or spec.get("kind") == "scifact":
            if (workspace / "training/training-state.pt").exists():
                raise ValueError(
                    "An experiment has already started; resume it or choose a new workspace"
                )
            result = prepare(workspace, seed=config.seed, validation_fraction=fraction)
            atomic_json(
                workspace / "run-config.json",
                {
                    "kind": "scifact",
                    "experiment": {"name": "BeIR/scifact", "workspace": str(workspace.resolve())},
                    "data": {"validation_fraction": fraction},
                    "training": asdict(config),
                },
            )
            return result
        spec["training"] = asdict(config)
        spec["data"]["validation_fraction"] = fraction
        result = prepare_custom(spec, workspace, config)
        manifest = fetch_assets(MODEL_ASSETS, workspace / "base-model")
        return {**result, "verified_model_files": len(manifest["assets"])}
    if args.command == "train":
        from rank_lab.study import train_study

        if spec is None:
            spec = {
                "kind": "scifact",
                "experiment": {"name": "BeIR/scifact", "workspace": str(workspace.resolve())},
                "data": {"validation_fraction": fraction},
            }
        if not (workspace / "training/training-state.pt").exists():
            spec["training"] = asdict(config)
            spec["data"]["validation_fraction"] = fraction
            atomic_json(workspace / "run-config.json", spec)
        return train_study(
            workspace,
            config,
            resume=args.resume,
            stop_after_steps=args.stop_after_steps,
            validation_fraction=fraction,
            name=spec["experiment"]["name"] if spec else None,
        )
    if args.command == "export":
        from rank_lab.presentation import export_presentation

        return export_presentation(
            workspace,
            args.destination or workspace / "export",
            resumed_after_step=args.resumed_after_step,
        )
    from rank_lab.study import final_comparison

    return final_comparison(workspace, device=config.device)


if __name__ == "__main__":
    main()
