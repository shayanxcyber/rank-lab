"""Pinned public inputs and atomic experiment records."""

import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"


@dataclass(frozen=True)
class Asset:
    repository: str
    revision: str
    filename: str
    local_name: str
    kind: str = "datasets"
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    expected_git_oid: str | None = None

    @property
    def url(self) -> str:
        prefix = "datasets/" if self.kind == "datasets" else ""
        return (
            f"https://huggingface.co/{prefix}{self.repository}/resolve/"
            f"{self.revision}/{self.filename}"
        )


DATA_ASSETS = (
    Asset(
        "BeIR/scifact",
        DATA_REVISION,
        "corpus/corpus-00000-of-00001.parquet",
        "corpus.parquet",
        expected_bytes=4469916,
        expected_sha256="243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
    ),
    Asset(
        "BeIR/scifact",
        DATA_REVISION,
        "queries/queries-00000-of-00001.parquet",
        "queries.parquet",
        expected_bytes=64982,
        expected_sha256="1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034",
    ),
    Asset(
        "BeIR/scifact-qrels",
        QRELS_REVISION,
        "train.tsv",
        "train.tsv",
        expected_bytes=14502,
        expected_git_oid="3d7882aacef0cbe3befa238357e4bb165165d8ba",
    ),
    Asset(
        "BeIR/scifact-qrels",
        QRELS_REVISION,
        "test.tsv",
        "test.tsv",
        expected_bytes=5389,
        expected_git_oid="c62f58c75f10c0cee23483b7269bff2902ae4080",
    ),
)
MODEL_ASSETS = tuple(
    Asset(
        MODEL_ID,
        MODEL_REVISION,
        name,
        name,
        kind="models",
        expected_bytes=size,
        expected_sha256=digest,
        expected_git_oid=oid,
    )
    for name, size, digest, oid in (
        ("config.json", 612, None, "72b987fd805cfa2b58c4c8c952b274a11bfd5a00"),
        (
            "model.safetensors",
            90868376,
            "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
            None,
        ),
        ("special_tokens_map.json", 112, None, "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
        ("tokenizer.json", 466247, None, "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
        ("tokenizer_config.json", 350, None, "c79f2b6a0cea6f4b564fed1938984bace9d30ff0"),
        ("vocab.txt", 231508, None, "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def _check(asset: Asset, path: Path) -> bool:
    if not path.is_file():
        return False
    if asset.expected_bytes is not None and path.stat().st_size != asset.expected_bytes:
        return False
    if asset.expected_sha256 is not None and sha256(path) != asset.expected_sha256:
        return False
    if asset.expected_git_oid is not None:
        data = path.read_bytes()
        oid = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if oid != asset.expected_git_oid:
            return False
    return True


def fetch_assets(assets: tuple[Asset, ...], destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    records = []
    for asset in assets:
        target = (root / asset.local_name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Asset path escapes the destination")
        if not _check(asset, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            request = urllib.request.Request(asset.url, headers={"User-Agent": "rank-lab/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if not _check(asset, partial):
                raise ValueError(f"Downloaded asset does not match its manifest: {asset.filename}")
            partial.replace(target)
        records.append(
            {
                **asdict(asset),
                "url": asset.url,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    manifest = {"schema_version": 1, "assets": records}
    atomic_json(root / "manifest.json", manifest)
    return manifest
