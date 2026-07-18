"""
Fixture tests for the development-data bundle manifest specification and
the required-artifacts registry.

These tests validate structure only -- they never touch production Google
Drive paths and never fabricate artifact content.
"""

import json
from pathlib import Path

import pytest

from scripts.dev_data_bundle.manifest import (
    MANIFEST_SCHEMA_PATH,
    REQUIRED_ARTIFACTS_PATH,
    ManifestValidationError,
    load_manifest_schema,
    load_required_artifacts,
    sha256_of_bytes,
    sha256_of_file,
    validate_manifest,
    validate_manifest_or_raise,
    validate_required_artifacts,
)


def _valid_manifest() -> dict:
    return {
        "bundle_name": "atlas-dev-data-bundle",
        "bundle_version": "1.0.0",
        "created_utc": "2026-07-18T22:00:00+00:00",
        "colab_project_root": "/content/drive/MyDrive/Project_Atlas",
        "source_repository": "brianrags07-afk/Project-ATLAS",
        "packaging_engine_version": "1.0.0",
        "artifact_count": 1,
        "artifacts": [
            {
                "artifact_id": "example",
                "original_production_path": "data/example/example.parquet",
                "bundled_relative_path": "data/example/example.parquet",
                "file_size_bytes": 100,
                "row_count": 10,
                "column_count": 3,
                "primary_key": ["game_pk"],
                "sha256": "a" * 64,
                "season": 2025,
                "purpose": "fixture test",
            }
        ],
    }


def test_required_artifacts_registry_file_exists():
    assert REQUIRED_ARTIFACTS_PATH.exists()


def test_manifest_schema_file_exists():
    assert MANIFEST_SCHEMA_PATH.exists()


def test_manifest_schema_is_valid_json_schema_document():
    schema = load_manifest_schema()

    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert "artifacts" in schema["properties"]
    assert "artifact" in schema["definitions"]


def test_required_artifacts_registry_loads_and_validates():
    registry = load_required_artifacts()

    assert registry["artifacts"], "registry must list at least one artifact"

    for artifact in registry["artifacts"]:
        assert artifact["original_production_path"].startswith("data/")
        assert artifact["catalog_status"] in (
            "confirmed_in_schema_catalog",
            "path_unconfirmed_requires_colab_verification",
        )
        assert isinstance(artifact["primary_key"], list)
        assert artifact["primary_key"]

        # Governance rule 7: never use player names as durable keys when
        # player IDs exist.
        assert "player_name" not in artifact["primary_key"]


def test_required_artifacts_registry_has_unique_artifact_ids():
    registry = load_required_artifacts()

    artifact_ids = [a["artifact_id"] for a in registry["artifacts"]]

    assert len(artifact_ids) == len(set(artifact_ids))


def test_confirmed_artifacts_carry_known_values_from_schema_catalog():
    registry = load_required_artifacts()

    for artifact in registry["artifacts"]:
        if artifact["catalog_status"] != "confirmed_in_schema_catalog":
            continue

        assert artifact["schema_reference"] is not None
        schema_path = Path(artifact["schema_reference"])
        assert schema_path.exists(), (
            f"{artifact['artifact_id']} references a schema file that "
            f"does not exist: {schema_path}"
        )

        with schema_path.open() as handle:
            schema_doc = json.load(handle)

        assert schema_doc["row_count"] == artifact["known_row_count"]
        assert schema_doc["column_count"] == artifact["known_column_count"]
        assert schema_doc["source_sha256"] == artifact["known_sha256"]


def test_unconfirmed_artifacts_carry_no_known_values():
    registry = load_required_artifacts()

    for artifact in registry["artifacts"]:
        if artifact["catalog_status"] != "path_unconfirmed_requires_colab_verification":
            continue

        assert artifact.get("known_row_count") is None
        assert artifact.get("known_column_count") is None
        assert artifact.get("known_sha256") is None


def test_validate_required_artifacts_rejects_missing_keys():
    errors = validate_required_artifacts({"artifacts": [{"artifact_id": "x"}]})

    assert any("missing required key" in error for error in errors)


def test_validate_required_artifacts_rejects_non_data_path():
    registry = {
        "artifacts": [
            {
                "artifact_id": "x",
                "original_production_path": "not_data/x.parquet",
                "season": 2025,
                "purpose": "test",
                "primary_key": ["game_pk"],
                "catalog_status": "confirmed_in_schema_catalog",
            }
        ]
    }

    errors = validate_required_artifacts(registry)

    assert any("must start with 'data/'" in error for error in errors)


def test_validate_manifest_accepts_well_formed_manifest():
    assert validate_manifest(_valid_manifest()) == []
    validate_manifest_or_raise(_valid_manifest())


def test_validate_manifest_rejects_missing_top_level_key():
    manifest = _valid_manifest()
    del manifest["bundle_version"]

    errors = validate_manifest(manifest)

    assert any("bundle_version" in error for error in errors)


def test_validate_manifest_rejects_artifact_count_mismatch():
    manifest = _valid_manifest()
    manifest["artifact_count"] = 99

    errors = validate_manifest(manifest)

    assert any("artifact_count" in error for error in errors)


def test_validate_manifest_rejects_bad_sha256():
    manifest = _valid_manifest()
    manifest["artifacts"][0]["sha256"] = "not-a-real-checksum"

    errors = validate_manifest(manifest)

    assert any("sha256" in error for error in errors)


def test_validate_manifest_rejects_negative_row_count():
    manifest = _valid_manifest()
    manifest["artifacts"][0]["row_count"] = -1

    errors = validate_manifest(manifest)

    assert any("row_count" in error for error in errors)


def test_validate_manifest_rejects_bundle_version_not_semver():
    manifest = _valid_manifest()
    manifest["bundle_version"] = "v1"

    errors = validate_manifest(manifest)

    assert any("bundle_version" in error for error in errors)


def test_validate_manifest_rejects_part_size_at_or_over_2gib():
    manifest = _valid_manifest()
    manifest["max_part_size_bytes"] = 2 * 1024**3

    errors = validate_manifest(manifest)

    assert any("2 GiB" in error for error in errors)


def test_validate_manifest_or_raise_raises_on_invalid_manifest():
    manifest = _valid_manifest()
    del manifest["artifacts"]

    with pytest.raises(ManifestValidationError):
        validate_manifest_or_raise(manifest)


def test_sha256_of_file_matches_hashlib(tmp_path):
    import hashlib

    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"atlas dev data bundle fixture bytes")

    expected = hashlib.sha256(file_path.read_bytes()).hexdigest()

    assert sha256_of_file(file_path) == expected


def test_sha256_of_bytes_matches_hashlib():
    import hashlib

    data = b"another fixture payload"

    assert sha256_of_bytes(data) == hashlib.sha256(data).hexdigest()
