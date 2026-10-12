"""Export a recorded study for the static result explorer; never run inference."""

import csv
import io
import json
from pathlib import Path

from rank_lab.artifacts import atomic_json, sha256
from rank_lab.data import load_scifact
from rank_lab.metrics import evaluate_rankings


def export_presentation(
    workspace: Path, destination: Path, *, resumed_after_step: int | None = None
) -> dict:
    comparison_path = workspace / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    split = json.loads((workspace / "split.json").read_text(encoding="utf-8"))
    training = json.loads((workspace / "training/training.json").read_text(encoding="utf-8"))
    data, exclusions = load_scifact(workspace / "data").remove_test_duplicates()
    if (
        data.fingerprint != comparison["dataset_fingerprint"]
        or data.fingerprint != split["dataset_fingerprint"]
    ):
        raise ValueError("Presentation inputs differ from the recorded comparison")
    if training["status"] != "completed":
        raise ValueError("Presentation requires a completed study")
    if resumed_after_step is not None and not 0 < resumed_after_step < training["step"]:
        raise ValueError("Resume annotation must identify a step before completion")
    query_ids = split["test"]
    if (
        comparison["query_count"] != len(query_ids)
        or len(set(query_ids)) != len(query_ids)
        or set(query_ids) != set(data.test_qrels)
    ):
        raise ValueError("Comparison query count differs from the reserved split")
    methods = ("lexical", "pretrained", "fine_tuned")
    documents = set()
    queries = []
    for method in methods:
        run = comparison["runs"][method]
        if set(run["rankings"]) != set(query_ids) or set(run["per_query"]) != set(query_ids):
            raise ValueError("Comparison must cover every reserved query exactly")
        recomputed = evaluate_rankings(run["rankings"], data.test_qrels)
        if recomputed["metrics"] != run["metrics"] or recomputed["per_query"] != run["per_query"]:
            raise ValueError("Recorded scores differ from the rankings and judgments")
    for query_id in sorted(query_ids, key=int):
        judgments = data.test_qrels[query_id]
        rankings = {method: comparison["runs"][method]["rankings"][query_id] for method in methods}
        documents.update(judgments)
        documents.update(doc for ranking in rankings.values() for doc in ranking)
        queries.append(
            {
                "id": query_id,
                "text": data.queries[query_id],
                "judgments": judgments,
                "rankings": rankings,
                "metrics": {
                    method: comparison["runs"][method]["per_query"][query_id] for method in methods
                },
            }
        )
    experiment = json.loads((workspace / "experiment.json").read_text(encoding="utf-8"))
    summary = {
        "dataset": "BeIR/scifact",
        "documents": len(data.documents),
        "test_queries": len(query_ids),
        "train_queries": len(split["train"]),
        "validation_queries": len(split["validation"]),
        "excluded_train_queries": len(exclusions["excluded_training_queries"]),
        "selected_epoch": comparison["selected_epoch"],
        "optimizer_steps": training["step"],
        "trainable_parameters": training["trainable_parameters"],
        "metrics": {method: comparison["runs"][method]["metrics"] for method in methods},
        "base_model": experiment["base_model"],
        "base_revision": experiment["base_revision"],
        "config": experiment["config"],
        "platform": experiment["platform"],
        "gpu": experiment["device"],
        "torch": experiment["torch"],
        "peak_cuda_allocated_bytes": training["peak_cuda_allocated_bytes"],
        "weights_fingerprint": training["weights"],
        "limits": [
            "One seed and benchmark; incomplete relevance judgments",
            "BM25 uses full text; dense encoders truncate at 256 wordpieces",
            "CPU restart equivalence checked; no uninterrupted GPU comparison",
        ],
    }
    atomic_json(
        destination / "document-text.json",
        {key: data.documents[key].partition("\n")[2] for key in sorted(documents)},
    )
    payload = {
        "schema_version": 1,
        "kind": "recorded-study",
        "summary": summary,
        "comparison_sha256": sha256(comparison_path),
        "dataset_fingerprint": data.fingerprint,
        "queries": queries,
        "documents": {
            key: {"title": data.documents[key].partition("\n")[0]} for key in sorted(documents)
        },
        "document_text_sha256": sha256(destination / "document-text.json"),
        "training": {
            "history": training["history"],
            "initial_weights": training["initial_weights"],
            "final_weights": training["weights"],
            "resumed_after_step": resumed_after_step,
            "elapsed_seconds_resumed_invocation": training["elapsed_seconds_this_invocation"],
        },
        "validation": [
            {
                "epoch": epoch,
                "ndcg": json.loads(
                    (
                        workspace
                        / (
                            "validation-baseline.json"
                            if epoch == 0
                            else f"training/epoch-{epoch}/validation.json"
                        )
                    ).read_text(encoding="utf-8")
                )["metrics"]["ndcg@10"],
            }
            for epoch in range(experiment["config"]["epochs"] + 1)
        ],
        "exclusions": exclusions["excluded_training_queries"],
    }
    atomic_json(destination / "study.json", payload)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "query_id",
            "query",
            *[
                f"{method}/{metric}"
                for method in methods
                for metric in ("ndcg@10", "recall@10", "mrr@10")
            ],
        ]
    )
    for query in queries:
        writer.writerow(
            [
                query["id"],
                query["text"],
                *[
                    query["metrics"][method][metric]
                    for method in methods
                    for metric in ("ndcg@10", "recall@10", "mrr@10")
                ],
            ]
        )
    (destination / "queries.csv").write_text(stream.getvalue(), encoding="utf-8", newline="")
    return {
        "queries": len(queries),
        "documents": len(documents),
        "study_sha256": sha256(destination / "study.json"),
    }
