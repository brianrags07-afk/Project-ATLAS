"""
Fixture tests for the repository dev-data-bundle bootstrap script.

All GitHub network access is monkeypatched -- these tests never make real
HTTP requests and never require a real token or private repository.
"""

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.dev_data_bundle import bootstrap_dev_data_bundle as bootstrap_mod
from scripts.dev_data_bundle.bootstrap_dev_data_bundle import (
    AssetNotFoundError,
    AuthenticationError,
    BootstrapError,
    ChecksumMismatchError,
    MissingDataError,
    bootstrap,
    find_asset,
    resolve_token,
    verify_extracted_artifacts,
    write_env_file,
)
from scripts.dev_data_bundle.manifest import sha256_of_bytes


def test_resolve_token_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")

    assert resolve_token("explicit-token") == "explicit-token"


def test_resolve_token_falls_back_to_github_token_env(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")

    assert resolve_token(None) == "env-token"


def test_resolve_token_falls_back_to_gh_token_env(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "gh-cli-token")

    assert resolve_token(None) == "gh-cli-token"


def test_resolve_token_raises_authentication_error_when_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)

    with pytest.raises(AuthenticationError):
        resolve_token(None)


def test_find_asset_returns_matching_asset():
    release = {
        "tag_name": "v1.0.0",
        "assets": [
            {"name": "release_manifest.json", "url": "https://example/1"},
            {"name": "bundle.tar.gz", "url": "https://example/2"},
        ],
    }

    asset = find_asset(release, "bundle.tar.gz")

    assert asset["url"] == "https://example/2"


def test_find_asset_raises_asset_not_found_error():
    release = {"tag_name": "v1.0.0", "assets": []}

    with pytest.raises(AssetNotFoundError):
        find_asset(release, "missing.tar.gz")


def _build_archive_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()

    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    return buffer.getvalue()


def _manifest_for(artifacts: list[dict], **overrides) -> dict:
    manifest = {
        "bundle_name": "atlas-dev-data-bundle",
        "bundle_version": "1.0.0",
        "created_utc": "2026-07-18T22:00:00+00:00",
        "colab_project_root": "/content/drive/MyDrive/Project_Atlas",
        "source_repository": "brianrags07-afk/Project-ATLAS",
        "packaging_engine_version": "1.0.0",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "part_files": [],
    }
    manifest.update(overrides)
    return manifest


def test_verify_extracted_artifacts_passes_for_correct_files(tmp_path):
    (tmp_path / "data" / "example").mkdir(parents=True)
    file_path = tmp_path / "data" / "example" / "example.parquet"
    file_path.write_bytes(b"fixture bytes")

    manifest = _manifest_for([
        {
            "artifact_id": "example",
            "bundled_relative_path": "data/example/example.parquet",
            "sha256": sha256_of_bytes(b"fixture bytes"),
        }
    ])

    verify_extracted_artifacts(manifest, tmp_path)


def test_verify_extracted_artifacts_raises_missing_data_error(tmp_path):
    manifest = _manifest_for([
        {
            "artifact_id": "example",
            "bundled_relative_path": "data/example/example.parquet",
            "sha256": sha256_of_bytes(b"fixture bytes"),
        }
    ])

    with pytest.raises(MissingDataError):
        verify_extracted_artifacts(manifest, tmp_path)


def test_verify_extracted_artifacts_raises_checksum_mismatch_error(tmp_path):
    (tmp_path / "data" / "example").mkdir(parents=True)
    file_path = tmp_path / "data" / "example" / "example.parquet"
    file_path.write_bytes(b"corrupted bytes")

    manifest = _manifest_for([
        {
            "artifact_id": "example",
            "bundled_relative_path": "data/example/example.parquet",
            "sha256": sha256_of_bytes(b"fixture bytes"),
        }
    ])

    with pytest.raises(ChecksumMismatchError):
        verify_extracted_artifacts(manifest, tmp_path)


def test_write_env_file_writes_atlas_data_root(tmp_path):
    data_root = tmp_path / "extracted" / "data"

    env_path = write_env_file(tmp_path, data_root)

    content = env_path.read_text()
    assert f"ATLAS_DATA_ROOT={data_root}" in content


def test_bootstrap_end_to_end_with_mocked_github(tmp_path, monkeypatch):
    archive_bytes = _build_archive_bytes({
        "data/example/example.parquet": b"fixture bytes",
    })
    archive_sha256 = sha256_of_bytes(archive_bytes)

    manifest = _manifest_for(
        [
            {
                "artifact_id": "example",
                "original_production_path": "data/example/example.parquet",
                "bundled_relative_path": "data/example/example.parquet",
                "file_size_bytes": len(b"fixture bytes"),
                "row_count": 1,
                "column_count": 1,
                "primary_key": ["game_pk"],
                "sha256": sha256_of_bytes(b"fixture bytes"),
                "season": 2025,
                "purpose": "fixture",
            }
        ],
        archive_sha256=archive_sha256,
    )

    manifest_bytes = json.dumps(manifest).encode("utf-8")

    fake_release = {
        "tag_name": "v1.0.0",
        "assets": [
            {"name": "release_manifest.json", "url": "https://example/manifest"},
            {
                "name": "atlas-dev-data-bundle-1.0.0.tar.gz",
                "url": "https://example/archive",
            },
        ],
    }

    def fake_get_release(repo, tag, token):
        assert repo == "brianrags07-afk/Project-ATLAS"
        assert token == "fake-token"
        return fake_release

    def fake_download_asset(asset, token):
        assert token == "fake-token"
        if asset["url"] == "https://example/manifest":
            return manifest_bytes
        if asset["url"] == "https://example/archive":
            return archive_bytes
        raise AssertionError(f"unexpected asset url {asset['url']}")

    monkeypatch.setattr(bootstrap_mod, "get_release", fake_get_release)
    monkeypatch.setattr(bootstrap_mod, "download_asset", fake_download_asset)

    dest = tmp_path / "dest"

    result_manifest = bootstrap(
        repo="brianrags07-afk/Project-ATLAS",
        tag="v1.0.0",
        dest=dest,
        token="fake-token",
    )

    assert result_manifest["artifact_count"] == 1

    extracted_file = dest / "1.0.0" / "data" / "example" / "example.parquet"
    assert extracted_file.exists()
    assert extracted_file.read_bytes() == b"fixture bytes"

    assert (dest / "atlas_dev_data.env").exists()


def test_bootstrap_raises_checksum_mismatch_when_archive_corrupted(tmp_path, monkeypatch):
    archive_bytes = _build_archive_bytes({
        "data/example/example.parquet": b"fixture bytes",
    })

    manifest = _manifest_for(
        [
            {
                "artifact_id": "example",
                "original_production_path": "data/example/example.parquet",
                "bundled_relative_path": "data/example/example.parquet",
                "file_size_bytes": len(b"fixture bytes"),
                "row_count": 1,
                "column_count": 1,
                "primary_key": ["game_pk"],
                "sha256": sha256_of_bytes(b"fixture bytes"),
                "season": 2025,
                "purpose": "fixture",
            }
        ],
        archive_sha256="0" * 64,
    )

    manifest_bytes = json.dumps(manifest).encode("utf-8")

    fake_release = {
        "tag_name": "v1.0.0",
        "assets": [
            {"name": "release_manifest.json", "url": "https://example/manifest"},
            {
                "name": "atlas-dev-data-bundle-1.0.0.tar.gz",
                "url": "https://example/archive",
            },
        ],
    }

    monkeypatch.setattr(bootstrap_mod, "get_release", lambda repo, tag, token: fake_release)
    monkeypatch.setattr(
        bootstrap_mod,
        "download_asset",
        lambda asset, token: (
            manifest_bytes if "manifest" in asset["url"] else archive_bytes
        ),
    )

    with pytest.raises(ChecksumMismatchError):
        bootstrap(
            repo="brianrags07-afk/Project-ATLAS",
            tag="v1.0.0",
            dest=tmp_path / "dest",
            token="fake-token",
        )


def test_safe_extract_all_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "malicious.tar.gz"

    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(name="../escaped.txt")
        payload = b"escape"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    destination = tmp_path / "safe_dest"
    destination.mkdir()

    with tarfile.open(archive_path, "r:gz") as archive:
        with pytest.raises(BootstrapError, match="outside destination"):
            bootstrap_mod._safe_extract_all(archive, destination)
