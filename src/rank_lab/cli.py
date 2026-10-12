"""Local preparation and device verification for the retrieval study."""

import argparse
import json
from pathlib import Path

from rank_lab.artifacts import DATA_ASSETS, MODEL_ASSETS, atomic_json, fetch_assets
from rank_lab.data import load_scifact


def prepare(workspace: Path) -> dict:
    data_manifest = fetch_assets(DATA_ASSETS, workspace / "data")
    model_manifest = fetch_assets(MODEL_ASSETS, workspace / "base-model")
    raw_data = load_scifact(workspace / "data")
    data, exclusions = raw_data.remove_test_duplicates()
    atomic_json(workspace / "split-exclusions.json", exclusions)
    split = data.split()
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
    parser = argparse.ArgumentParser(description="Pinned retrieval training study")
    parser.add_argument("command", choices=("prepare", "device", "train", "compare", "export"))
    parser.add_argument("--workspace", type=Path, default=Path("runs/scifact"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--destination", type=Path, default=Path("web/public/data"))
    parser.add_argument("--resumed-after-step", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.workspace)
    elif args.command == "device":
        result = device_probe()
        atomic_json(args.workspace / "device.json", result)
    elif args.command == "train":
        from rank_lab.study import train_study
        from rank_lab.training import TrainingConfig

        result = train_study(
            args.workspace,
            TrainingConfig(device=args.device),
            resume=args.resume,
            stop_after_steps=args.stop_after_steps,
        )
    elif args.command == "export":
        from rank_lab.presentation import export_presentation

        result = export_presentation(
            args.workspace, args.destination, resumed_after_step=args.resumed_after_step
        )
    else:
        from rank_lab.study import final_comparison

        result = final_comparison(args.workspace, device=args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
