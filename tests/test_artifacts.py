import hashlib
import io
import json

import pytest

from rank_lab.artifacts import Asset, atomic_json, fetch_assets


def test_hash_mismatch_cannot_replace_existing_artifact(tmp_path, monkeypatch):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"old artifact")
    asset = Asset(
        "example/model",
        "revision",
        "model.safetensors",
        target.name,
        expected_sha256=hashlib.sha256(b"expected model").hexdigest(),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong model")
    )

    with pytest.raises(ValueError, match="does not match its manifest"):
        fetch_assets((asset,), tmp_path)

    assert target.read_bytes() == b"old artifact"
    assert not (tmp_path / "manifest.json").exists()


def test_pinned_text_blob_is_verified_and_reused(tmp_path, monkeypatch):
    data = b"query-id\tcorpus-id\tscore\nq\td\t1\n"
    oid = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
    asset = Asset("example/data", "revision", "train.tsv", "train.tsv", expected_git_oid=oid)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(data))
    first = fetch_assets((asset,), tmp_path)

    def unexpected_network(*args, **kwargs):
        raise AssertionError("A verified cached artifact should be reused")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_network)
    assert fetch_assets((asset,), tmp_path) == first


def test_asset_cannot_escape_destination(tmp_path):
    asset = Asset("example/data", "revision", "data", "../outside")
    with pytest.raises(ValueError, match="escapes the destination"):
        fetch_assets((asset,), tmp_path)


def test_failed_json_write_preserves_previous_record(tmp_path):
    path = tmp_path / "results.json"
    atomic_json(path, {"status": "completed"})
    with pytest.raises(ValueError):
        atomic_json(path, {"score": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed"}
