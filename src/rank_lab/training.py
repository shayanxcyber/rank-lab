"""Gradient training with atomic model/optimizer/RNG checkpoints."""

import hashlib
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

from rank_lab.artifacts import MODEL_REVISION, atomic_json
from rank_lab.data import RetrievalData


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    temperature: float = 0.05
    max_length: int = 256
    seed: int = 1729
    device: str = "cuda"
    mixed_precision: bool = True

    def validate(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 2
            or self.learning_rate <= 0
            or self.temperature <= 0
        ):
            raise ValueError(
                "Epochs, learning rate and temperature must be positive; batch size at least two"
            )
        if not 1 <= self.max_length <= 512:
            raise ValueError(
                "Sequence limit must be between 1 and 512 for the supported MiniLM model"
            )
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Device must be cpu or cuda")


def weights_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def contrastive_loss(
    query_vectors: torch.Tensor,
    document_vectors: torch.Tensor,
    pairs: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    *,
    temperature: float,
) -> torch.Tensor:
    logits = query_vectors.float() @ document_vectors.float().T / temperature
    mask = torch.tensor(
        [
            [
                column == row or qrels[query_id].get(document_id, 0) <= 0
                for column, (_, document_id) in enumerate(pairs)
            ]
            for row, (query_id, _) in enumerate(pairs)
        ],
        dtype=torch.bool,
        device=logits.device,
    )
    logits = logits.masked_fill(~mask, -torch.inf)
    return functional.cross_entropy(logits, torch.arange(len(pairs), device=logits.device))


def train(
    model: nn.Module,
    data: RetrievalData,
    split: dict,
    config: TrainingConfig,
    output: Path,
    *,
    resume: bool = False,
    stop_after_steps: int | None = None,
    evaluate_epoch=None,
) -> dict:
    config.validate()
    data.validate()
    if split["dataset_fingerprint"] != data.fingerprint:
        raise ValueError("Split fingerprint does not match the dataset")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model.to(config.device)
    pairs = [
        (query_id, document_id)
        for query_id in split["train"]
        for document_id, score in sorted(data.train_qrels[query_id].items())
        if score > 0
    ]
    if len(pairs) < 2:
        raise ValueError("Training needs at least two positive pairs")
    batches_per_epoch = (len(pairs) + config.batch_size - 1) // config.batch_size
    total_steps = batches_per_epoch * config.epochs
    warmup = min(max(1, total_steps // 10), total_steps - 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def schedule(step):
        return (
            step / warmup
            if step < warmup
            else max(0, (total_steps - step) / (total_steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    contract = {
        "format": 1,
        "config": asdict(config),
        "dataset_fingerprint": data.fingerprint,
        "train_query_ids": split["train"],
        "base_revision": MODEL_REVISION,
        "torch": torch.__version__,
    }
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint = output / "training-state.pt"
    output.mkdir(parents=True, exist_ok=True)
    if not resume and checkpoint.exists():
        raise ValueError("Training state already exists; use resume or choose a new output")
    epoch, next_batch, step, history = 0, 0, 0, []
    initial_weights = weights_fingerprint(model)
    if resume:
        state = torch.load(checkpoint, map_location=config.device, weights_only=True)
        if state["contract_hash"] != contract_hash:
            raise ValueError("Checkpoint configuration/data contract does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, next_batch, step, history = (
            state["epoch"],
            state["next_batch"],
            state["step"],
            state["history"],
        )
        initial_weights = state["initial_weights"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        if config.device == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng_cuda"]])
    atomic_json(output / "config.json", contract)
    started = time.perf_counter()

    def persist(current_epoch, cursor):
        save_checkpoint(
            checkpoint,
            {
                "format": 1,
                "contract_hash": contract_hash,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": current_epoch,
                "next_batch": cursor,
                "step": step,
                "history": history,
                "initial_weights": initial_weights,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
            },
        )

    for current_epoch in range(epoch, config.epochs):
        ordered = pairs.copy()
        random.Random(config.seed + current_epoch).shuffle(ordered)
        batches = [
            ordered[start : start + config.batch_size]
            for start in range(0, len(ordered), config.batch_size)
        ]
        cursor_start = next_batch if current_epoch == epoch else 0
        for cursor in range(cursor_start, len(batches)):
            batch = batches[cursor]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=config.device == "cuda" and config.mixed_precision,
            ):
                query_vectors = model([data.queries[key] for key, _ in batch])
                document_vectors = model([data.documents[key] for _, key in batch])
            loss = contrastive_loss(
                query_vectors,
                document_vectors,
                batch,
                data.train_qrels,
                temperature=config.temperature,
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; optimizer step was not applied")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step += 1
            history.append(
                {
                    "step": step,
                    "epoch": current_epoch + 1,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )
            persist(current_epoch, cursor + 1)
            atomic_json(
                output / "progress.json",
                {
                    "status": "training",
                    "step": step,
                    "total_steps": total_steps,
                    "latest": history[-1],
                },
            )
            if stop_after_steps is not None and step >= stop_after_steps:
                result = {
                    "status": "paused",
                    "step": step,
                    "history": history,
                    "initial_weights": initial_weights,
                    "weights": weights_fingerprint(model),
                }
                atomic_json(output / "training.json", result)
                return result
        if evaluate_epoch is not None:
            evaluate_epoch(model, current_epoch + 1)
        next_batch = 0
        persist(current_epoch + 1, 0)
    result = {
        "status": "completed",
        "step": step,
        "history": history,
        "initial_weights": initial_weights,
        "weights": weights_fingerprint(model),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if config.device == "cuda"
        else None,
    }
    atomic_json(output / "training.json", result)
    atomic_json(
        output / "progress.json", {"status": "completed", "step": step, "total_steps": total_steps}
    )
    return result
