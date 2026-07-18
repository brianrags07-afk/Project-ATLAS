"""
Fixture tests for the Colab dev-data-bundle packaging script.

These tests build a small synthetic "Drive project root" under a pytest
tmp_path (never touching real production data) with a custom, minimal
required-artifacts registry, then exercise packaging end to end: staging,
grain verification, manifest construction/validation, compression, and
splitting.
"""

import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from scripts.dev_data_bundle.colab_package_dev_data_bundle import (
    PackagingError,
    package_bundle,
    split_archive,
)
from scripts.dev_data_bundle.manifest import sha256_of_file


def _write_registry(path: Path, artifacts: list[dict]) -> Path:
    registry_path = path / "required_artifacts.json"

    with registry_path.open("w") as handle:
        json.dump({"artifacts": artifacts}, handle)

    return registry_path


def _make_project_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "Project_Atlas"
    example_dir = project_root / "data" / "example"
    example_dir.mkdir(parents=True)

    frame = pd.DataFrame({
        "game_pk": [1, 2, 3],
        "team": ["ATL", "NYM", "PHI"],
        "value": [1.0, 2.0, 3.0],
    })
    frame.to_parquet(example_dir / "example.parquet")

    return project_root


def _example_artifact() -> dict:
    return {
        "artifact_id": "example",
        "original_production_path": "data/example/example.parquet",
        "season": 2025,
        "purpose": "fixture test artifact",
        "primary_key": ["game_pk", "team"],
        "catalog_status": "path_unconfirmed_requires_colab_verification",
    }


def test_package_bundle_stages_and_writes_manifest(tmp_path):
    project_root = _make_project_root(tmp_path)
    registry_path = _write_registry(tmp_path, [_example_artifact()])
    output_dir = tmp_path / "build"

    manifest = package_bundle(
        project_root=project_root,
        output_dir=output_dir,
        bundle_version="1.0.0",
        source_repository="brianrags07-afk/Project-ATLAS",
        max_part_size_bytes=1_900_000_000,
        required_artifacts_path=registry_path,
    )

    assert manifest["artifact_count"] == 1
    artifact = manifest["artifacts"][0]
    assert artifact["artifact_id"] == "example"
    assert artifact["row_count"] == 3
    assert artifact["column_count"] == 3
    assert artifact["bundled_relative_path"] == "data/example/example.parquet"

    staged_file = output_dir / "staging" / "data" / "example" / "example.parquet"
    assert staged_file.exists()
    assert sha256_of_file(staged_file) == artifact["sha256"]

    archive_path = output_dir / "atlas-dev-data-bundle-1.0.0.tar.gz"
    assert archive_path.exists()

    manifest_path = output_dir / "release_manifest.json"
    assert manifest_path.exists()
    on_disk_manifest = json.loads(manifest_path.read_text())
    assert on_disk_manifest["artifact_count"] == 1


def test_package_bundle_archive_contains_staged_data(tmp_path):
    project_root = _make_project_root(tmp_path)
    registry_path = _write_registry(tmp_path, [_example_artifact()])
    output_dir = tmp_path / "build"

    package_bundle(
        project_root=project_root,
        output_dir=output_dir,
        bundle_version="1.0.0",
        source_repository="brianrags07-afk/Project-ATLAS",
        max_part_size_bytes=1_900_000_000,
        required_artifacts_path=registry_path,
    )

    archive_path = output_dir / "atlas-dev-data-bundle-1.0.0.tar.gz"

    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()

    assert "data/example/example.parquet" in names


def test_package_bundle_fails_clearly_when_artifact_missing(tmp_path):
    project_root = tmp_path / "Project_Atlas"
    (project_root / "data").mkdir(parents=True)

    registry_path = _write_registry(tmp_path, [_example_artifact()])
    output_dir = tmp_path / "build"

    with pytest.raises(PackagingError, match="was not found under"):
        package_bundle(
            project_root=project_root,
            output_dir=output_dir,
            bundle_version="1.0.0",
            source_repository="brianrags07-afk/Project-ATLAS",
            max_part_size_bytes=1_900_000_000,
            required_artifacts_path=registry_path,
        )


def test_package_bundle_fails_when_project_root_missing(tmp_path):
    registry_path = _write_registry(tmp_path, [_example_artifact()])

    with pytest.raises(PackagingError, match="Project root does not exist"):
        package_bundle(
            project_root=tmp_path / "does_not_exist",
            output_dir=tmp_path / "build",
            bundle_version="1.0.0",
            source_repository="brianrags07-afk/Project-ATLAS",
            max_part_size_bytes=1_900_000_000,
            required_artifacts_path=registry_path,
        )


def test_package_bundle_fails_clearly_on_non_unique_grain(tmp_path):
    project_root = tmp_path / "Project_Atlas"
    example_dir = project_root / "data" / "example"
    example_dir.mkdir(parents=True)

    frame = pd.DataFrame({
        "game_pk": [1, 1, 2],
        "team": ["ATL", "ATL", "NYM"],
        "value": [1.0, 2.0, 3.0],
    })
    frame.to_parquet(example_dir / "example.parquet")

    registry_path = _write_registry(tmp_path, [_example_artifact()])

    with pytest.raises(PackagingError, match="not unique at its declared grain"):
        package_bundle(
            project_root=project_root,
            output_dir=tmp_path / "build",
            bundle_version="1.0.0",
            source_repository="brianrags07-afk/Project-ATLAS",
            max_part_size_bytes=1_900_000_000,
            required_artifacts_path=registry_path,
        )


def test_package_bundle_fails_clearly_on_missing_primary_key_column(tmp_path):
    project_root = tmp_path / "Project_Atlas"
    example_dir = project_root / "data" / "example"
    example_dir.mkdir(parents=True)

    frame = pd.DataFrame({
        "game_pk": [1, 2, 3],
        "value": [1.0, 2.0, 3.0],
    })
    frame.to_parquet(example_dir / "example.parquet")

    registry_path = _write_registry(tmp_path, [_example_artifact()])

    with pytest.raises(PackagingError, match="missing declared primary-key"):
        package_bundle(
            project_root=project_root,
            output_dir=tmp_path / "build",
            bundle_version="1.0.0",
            source_repository="brianrags07-afk/Project-ATLAS",
            max_part_size_bytes=1_900_000_000,
            required_artifacts_path=registry_path,
        )


def test_split_archive_returns_empty_when_under_limit(tmp_path):
    archive_path = tmp_path / "archive.tar.gz"
    archive_path.write_bytes(b"x" * 100)

    parts = split_archive(archive_path, max_part_size_bytes=1_000)

    assert parts == []


def test_split_archive_splits_and_checksums_parts(tmp_path):
    archive_path = tmp_path / "archive.tar.gz"
    payload = b"a" * 250
    archive_path.write_bytes(payload)

    parts = split_archive(archive_path, max_part_size_bytes=100)

    assert len(parts) == 3
    assert [part["part_index"] for part in parts] == [0, 1, 2]
    assert sum(part["size_bytes"] for part in parts) == len(payload)

    for part in parts:
        part_path = archive_path.parent / part["filename"]
        assert part_path.exists()
        assert sha256_of_file(part_path) == part["sha256"]


def test_candidate_paths_are_tried_in_order(tmp_path):
    project_root = tmp_path / "Project_Atlas"
    fallback_dir = project_root / "data" / "fallback_location"
    fallback_dir.mkdir(parents=True)

    frame = pd.DataFrame({"game_pk": [1], "team": ["ATL"]})
    frame.to_parquet(fallback_dir / "example.parquet")

    artifact = _example_artifact()
    artifact["original_production_path"] = "data/primary_location/example.parquet"
    artifact["candidate_paths"] = [
        "data/primary_location/example.parquet",
        "data/fallback_location/example.parquet",
    ]

    registry_path = _write_registry(tmp_path, [artifact])

    manifest = package_bundle(
        project_root=project_root,
        output_dir=tmp_path / "build",
        bundle_version="1.0.0",
        source_repository="brianrags07-afk/Project-ATLAS",
        max_part_size_bytes=1_900_000_000,
        required_artifacts_path=registry_path,
    )

    assert manifest["artifacts"][0]["original_production_path"] == (
        "data/fallback_location/example.parquet"
    )
