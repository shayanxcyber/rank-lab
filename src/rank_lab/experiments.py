"""Portable experiment configuration and validated local input snapshots."""

import csv
import hashlib
import io
import json
import math
import re
import tomllib
from dataclasses import asdict, fields
from pathlib import Path

from rank_lab.artifacts import atomic_json
from rank_lab.data import RetrievalData

SECTIONS = {
    "experiment": {"name", "workspace"},
    "data": {"documents", "queries", "judgments", "validation_fraction"},
    "training": {
        "epochs",
        "batch_size",
        "learning_rate",
        "temperature",
        "max_length",
        "seed",
        "device",
        "mixed_precision",
    },
}


def read_config(path: Path) -> dict:
    """Resolve input/workspace paths relative to the TOML file, not the shell."""
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Cannot read experiment config {path.name}: {error}") from error
    if set(value) - SECTIONS.keys():
        raise ValueError("Unknown experiment configuration section")
    for section, allowed in SECTIONS.items():
        table = value.setdefault(section, {})
        if not isinstance(table, dict) or set(table) - allowed:
            raise ValueError(f"Unknown or invalid settings in [{section}]")
    name = value["experiment"].get("name", "Custom experiment")
    if not isinstance(name, str) or not name.strip() or len(name) > 160:
        raise ValueError("Experiment name must contain 1–160 characters")
    value["experiment"]["name"] = name.strip()
    for section, keys in (
        ("data", ("documents", "queries", "judgments")),
        ("experiment", ("workspace",)),
    ):
        for key in keys:
            raw = value[section].get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"Missing {section}.{key} path")
            value[section][key] = str((path.parent / raw).resolve())
    fraction = value["data"].get("validation_fraction", 0.1)
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    value["data"]["validation_fraction"] = fraction
    return value


def training_config(settings: dict, overrides: dict | None = None):
    from rank_lab.training import TrainingConfig

    values = {
        **settings,
        **{key: value for key, value in (overrides or {}).items() if value is not None},
    }
    allowed = {field.name for field in fields(TrainingConfig)}
    if set(values) - allowed:
        raise ValueError("Unknown training setting")
    for key, value in values.items():
        if key in ("epochs", "batch_size", "max_length", "seed"):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif key in ("learning_rate", "temperature"):
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            )
        elif key == "mixed_precision":
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid training setting: {key}")
    result = TrainingConfig(**values)
    result.validate()
    return result


def _identity(value, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context}: IDs must be nonempty strings without surrounding whitespace")
    return value


def _records(raw: bytes, kind: str) -> dict[str, dict]:
    records = {}
    try:
        for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{kind} line {number}: expected an object")
            key = _identity(row.get("id"), f"{kind} line {number}")
            if key in records:
                raise ValueError(f"Duplicate {kind} ID: {key}")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{kind} {key}: text must be nonempty")
            records[key] = row
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSONL in {kind}: {error}") from error
    if not records:
        raise ValueError(f"{kind} must not be empty")
    return records


def load_custom_inputs(paths: dict) -> tuple[RetrievalData, dict]:
    raw = {}
    for key in ("documents", "queries", "judgments"):
        try:
            raw[key] = Path(paths[key]).read_bytes()
        except (OSError, KeyError) as error:
            raise ValueError(f"Cannot read {key} input: {error}") from error
    documents = _records(raw["documents"], "document")
    queries = _records(raw["queries"], "query")
    for key, row in documents.items():
        if not isinstance(row.get("title", ""), str):
            raise ValueError(f"Document {key}: title must be a string")
    partitions = {"train": {}, "validation": {}, "test": {}}
    for key, row in queries.items():
        if row.get("split") not in partitions:
            raise ValueError(f"Query {key}: split must be train, validation or test")
        partitions[row["split"]][key] = {}
    try:
        stream = io.StringIO(raw["judgments"].decode("utf-8-sig"), newline="")
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query_id", "document_id", "relevance"]:
            raise ValueError("Judgments need TSV columns: query_id, document_id, relevance")
        for row in reader:
            if None in row:
                raise ValueError("Judgment rows must have exactly three TSV columns")
            query = _identity(row.get("query_id"), "Judgment query")
            document = _identity(row.get("document_id"), "Judgment document")
            if query not in queries:
                raise ValueError(f"Unknown query in judgments: {query}")
            if document not in documents:
                raise ValueError(f"Unknown document in judgments: {document}")
            raw_score = row.get("relevance")
            if not isinstance(raw_score, str) or not re.fullmatch(r"\d+", raw_score):
                raise ValueError("Relevance must be an integer between 0 and 30")
            score = int(raw_score)
            if score > 30:
                raise ValueError("Relevance must be an integer between 0 and 30")
            judgments = partitions[queries[query]["split"]][query]
            if document in judgments:
                raise ValueError(f"Duplicate judgment: {query}/{document}")
            judgments[document] = score
    except UnicodeDecodeError as error:
        raise ValueError("Judgments must be UTF-8 TSV") from error
    result = RetrievalData(
        {key: f"{row.get('title') or key}\n{row['text']}" for key, row in documents.items()},
        {key: row["text"] for key, row in queries.items()},
        partitions["train"],
        partitions["test"],
        partitions["validation"] or None,
    )
    result.validate()
    return result, {key: hashlib.sha256(value).hexdigest() for key, value in raw.items()}


def prepare_custom(config: dict, workspace: Path, training) -> dict:
    data, input_hashes = load_custom_inputs(config["data"])
    split = data.split(
        seed=training.seed, validation_fraction=config["data"]["validation_fraction"]
    )
    pairs = sum(score > 0 for key in split["train"] for score in data.train_qrels[key].values())
    if pairs < 2:
        raise ValueError("Training split needs at least two positive query/document pairs")
    previous = workspace / "preparation.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old["dataset_fingerprint"] != data.fingerprint:
            raise ValueError("Workspace belongs to different data; choose a new workspace")
        if (workspace / "training/training-state.pt").exists():
            raise ValueError("Experiment has already started; resume it or choose a new workspace")
    snapshot = workspace / "data/custom.json"
    atomic_json(snapshot, asdict(data))
    result = {
        "kind": "custom",
        "name": config["experiment"]["name"],
        "dataset_fingerprint": data.fingerprint,
        "input_sha256": input_hashes,
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "documents": len(data.documents),
        "queries": len(data.queries),
        "train_queries": len(split["train"]),
        "validation_queries": len(split["validation"]),
        "test_queries": len(split["test"]),
        "excluded_training_queries": 0,
    }
    atomic_json(workspace / "split.json", split)
    atomic_json(workspace / "split-exclusions.json", {"excluded_training_queries": []})
    atomic_json(workspace / "preparation.json", result)
    atomic_json(workspace / "run-config.json", config)
    return result


def load_prepared(workspace: Path) -> tuple[RetrievalData, dict]:
    from rank_lab.data import load_scifact

    snapshot = workspace / "data/custom.json"
    if snapshot.exists():
        preparation = json.loads((workspace / "preparation.json").read_text(encoding="utf-8"))
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != preparation["snapshot_sha256"]:
            raise ValueError("Prepared data snapshot changed; use a fresh workspace")
        data = RetrievalData(**json.loads(snapshot.read_text(encoding="utf-8")))
        data.validate()
        if data.fingerprint != preparation["dataset_fingerprint"]:
            raise ValueError("Prepared data fingerprint changed")
        return data, {"excluded_training_queries": []}
    return load_scifact(workspace / "data").remove_test_duplicates()
