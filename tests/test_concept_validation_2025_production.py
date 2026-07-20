import json

import numpy as np
import pandas as pd
import pytest

import atlas.validation.concept_validation_2025 as validation_module
import atlas.validation.concept_validation_2025_production as production_module
from atlas.learning.concept_definition_freeze import (
    build_frozen_member_registry,
    build_frozen_registry,
    dataframe_registry_fingerprint,
)
from atlas.validation.concept_validation_2025_certification import (
    certify_production_run,
)


def _raw_frozen_source() -> pd.DataFrame:
    return pd.DataFrame({
        "concept_id": ["concept_a", "concept_b"],
        "target_name": ["target_team_win", "target_team_win"],
        "concept_status": [
            "STRONG_CONCEPT_CANDIDATE",
            "CONCEPT_CANDIDATE",
        ],
        "broad_domain_pair": [
            "BULLPEN + IDENTITY",
            "IDENTITY + IDENTITY",
        ],
        "member_1_feature": [
            "bullpen__workload",
            "identity__missing_feature",
        ],
        "member_1_threshold_operator": [">=", ">="],
        "member_1_threshold_value": [70.0, 0.10],
        "member_1_semantic_classification": [
            "VALID_RECENT_WORKLOAD_FACT",
            "VALID_CONTINUOUS_BASEBALL_FACT",
        ],
        "member_1_governance_action": [
            "KEEP_SEMANTICALLY_VALID",
            "KEEP_SEMANTICALLY_VALID",
        ],
        "member_1_transformation_family_root": [
            "workload",
            "missing_feature",
        ],
        "member_1_base_metric_root": ["workload", "missing_feature"],
        "member_1_effect_direction": ["SUPPORTS_TARGET", "SUPPORTS_TARGET"],
        "member_2_feature": [
            "identity__run_differential",
            "identity__run_differential",
        ],
        "member_2_threshold_operator": [">=", ">="],
        "member_2_threshold_value": [0.50, 0.50],
        "member_2_semantic_classification": [
            "VALID_CONTINUOUS_BASEBALL_FACT",
            "VALID_CONTINUOUS_BASEBALL_FACT",
        ],
        "member_2_governance_action": [
            "KEEP_SEMANTICALLY_VALID",
            "KEEP_SEMANTICALLY_VALID",
        ],
        "member_2_transformation_family_root": [
            "run_differential",
            "run_differential",
        ],
        "member_2_base_metric_root": [
            "run_differential",
            "run_differential",
        ],
        "member_2_effect_direction": ["SUPPORTS_TARGET", "SUPPORTS_TARGET"],
        "same_effect_direction": [True, True],
        "semantic_family_pair_key": [
            "workload + run_differential",
            "missing_feature + run_differential",
        ],
        "base_metric_pair_key": [
            "workload + run_differential",
            "missing_feature + run_differential",
        ],
        "concept_semantic_status": [
            "SEMANTICALLY_VALID_FREEZE_CANDIDATE",
            "SEMANTICALLY_VALID_FREEZE_CANDIDATE",
        ],
    })


def _frozen_registries() -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = build_frozen_registry(_raw_frozen_source(), discovery_season=2024)
    members = build_frozen_member_registry(frozen)

    registry_sha256 = dataframe_registry_fingerprint(frozen, "definition_sha256")
    member_registry_sha256 = dataframe_registry_fingerprint(
        members, "member_definition_sha256"
    )

    frozen["registry_sha256"] = registry_sha256
    frozen["member_registry_sha256"] = member_registry_sha256
    members["registry_sha256"] = registry_sha256
    members["member_registry_sha256"] = member_registry_sha256

    return frozen, members


def _validation_frame_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n_games = 200

    interactions = pd.DataFrame({
        "game_pk": np.arange(n_games),
        "game_date": pd.Timestamp("2025-04-01")
        + pd.to_timedelta(np.arange(n_games) % 150, unit="D"),
        "atlas_season": 2025,
        "team": ["NYY" if i % 2 == 0 else "BOS" for i in range(n_games)],
        "bullpen__workload": rng.uniform(0, 140, size=n_games),
        "identity__run_differential": rng.uniform(-3, 3, size=n_games),
    })

    high_workload = interactions["bullpen__workload"].ge(70)
    high_diff = interactions["identity__run_differential"].ge(0.50)
    joint = high_workload & high_diff

    win_probability = np.where(joint, 0.75, 0.40)
    wins = (rng.uniform(size=n_games) < win_probability).astype(int)

    targets = pd.DataFrame({
        "game_pk": interactions["game_pk"],
        "game_date": interactions["game_date"],
        "atlas_season": interactions["atlas_season"],
        "team": interactions["team"],
        "target_team_win": wins,
    })

    return interactions, targets


@pytest.fixture()
def rigged_module(tmp_path, monkeypatch):
    frozen, members = _frozen_registries()
    interactions, targets = _validation_frame_sources()

    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()

    definition_path = frozen_dir / "frozen_concept_definition_registry.parquet"
    member_path = frozen_dir / "frozen_concept_member_registry.parquet"

    frozen.to_parquet(definition_path, index=False)
    members.to_parquet(member_path, index=False)

    interaction_path = tmp_path / "interactions.parquet"
    target_path = tmp_path / "targets.parquet"

    interactions.to_parquet(interaction_path, index=False)
    targets.to_parquet(target_path, index=False)

    output_dir = tmp_path / "output"

    monkeypatch.setattr(validation_module, "FROZEN_DEFINITION_REGISTRY_PATH", definition_path)
    monkeypatch.setattr(validation_module, "FROZEN_MEMBER_REGISTRY_PATH", member_path)
    monkeypatch.setattr(validation_module, "INTERACTION_PATH", interaction_path)
    monkeypatch.setattr(validation_module, "TEAM_TARGET_PATH", target_path)
    monkeypatch.setattr(validation_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        validation_module,
        "VALIDATION_REGISTRY_PATH",
        output_dir / "concept_validation_registry.parquet",
    )
    monkeypatch.setattr(
        validation_module,
        "VALIDATION_SUMMARY_PATH",
        output_dir / "concept_validation_summary.parquet",
    )
    monkeypatch.setattr(
        validation_module,
        "METADATA_PATH",
        output_dir / "concept_validation_metadata.json",
    )
    monkeypatch.setattr(
        validation_module,
        "LINEAGE_AUDIT_PATH",
        output_dir / "concept_validation_lineage_audit.json",
    )

    return validation_module


def test_certified_production_run_returns_zero_exit_code(rigged_module, tmp_path):
    manifest_dir = tmp_path / "manifest"

    exit_code, manifest = production_module.run_production_validation(
        manifest_dir=manifest_dir,
        expected_frozen_definition_count=2,
    )

    assert exit_code == 0
    assert manifest["certification"]["passed"] is True
    assert manifest["execution_error"] is None
    assert manifest["frozen_definition_count"] == 2
    assert manifest["validation_record_count"] == 2

    manifest_path = manifest_dir / production_module.MANIFEST_FILENAME
    report_path = manifest_dir / production_module.CERTIFICATION_REPORT_FILENAME

    assert manifest_path.exists()
    assert report_path.exists()

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["certification"]["passed"] is True

    report_text = report_path.read_text(encoding="utf-8")
    assert "PASS" in report_text


def test_failed_lineage_audit_returns_nonzero_exit_code(rigged_module, tmp_path):
    # Corrupt one frozen definition's stored hash so the lineage audit's
    # recomputed-fingerprint cross-check fails, while immutability flags
    # (definitions_frozen, thresholds_mutable, ...) remain valid.
    frozen = pd.read_parquet(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH)
    frozen.loc[0, "definition_sha256"] = "0" * 64
    frozen.to_parquet(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH, index=False)

    manifest_dir = tmp_path / "manifest"

    exit_code, manifest = production_module.run_production_validation(
        manifest_dir=manifest_dir,
        expected_frozen_definition_count=2,
    )

    assert exit_code != 0
    assert manifest["execution_error"] is not None
    assert "Lineage certification failure" in manifest["execution_error"]

    # Canonical outputs must never have been published.
    assert not rigged_module.VALIDATION_REGISTRY_PATH.exists()

    # No certified manifest should be written on failure.
    manifest_path = manifest_dir / production_module.MANIFEST_FILENAME
    assert not manifest_path.exists()


def test_immutability_failure_refuses_to_run(rigged_module, tmp_path):
    frozen = pd.read_parquet(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH)
    frozen["definitions_frozen"] = False
    frozen.to_parquet(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH, index=False)

    manifest_dir = tmp_path / "manifest"

    exit_code, manifest = production_module.run_production_validation(
        manifest_dir=manifest_dir,
        expected_frozen_definition_count=2,
    )

    assert exit_code != 0
    assert "immutability" in manifest["execution_error"].lower()
    assert not rigged_module.VALIDATION_REGISTRY_PATH.exists()


def test_failed_certification_does_not_overwrite_existing_certified_manifest(
    rigged_module, tmp_path
):
    manifest_dir = tmp_path / "manifest"

    exit_code, manifest = production_module.run_production_validation(
        manifest_dir=manifest_dir,
        expected_frozen_definition_count=2,
    )
    assert exit_code == 0

    manifest_path = manifest_dir / production_module.MANIFEST_FILENAME
    original_contents = manifest_path.read_text(encoding="utf-8")

    # Now corrupt the frozen registry so a second run fails certification.
    frozen = pd.read_parquet(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH)
    frozen.loc[0, "definition_sha256"] = "0" * 64
    frozen.to_parquet(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH, index=False)

    exit_code_2, manifest_2 = production_module.run_production_validation(
        manifest_dir=manifest_dir,
        expected_frozen_definition_count=2,
    )

    assert exit_code_2 != 0
    assert manifest_path.read_text(encoding="utf-8") == original_contents


def test_missing_required_output_fails_certification(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # Only create some of the required files.
    (output_dir / "concept_validation_registry.parquet").write_bytes(b"")

    result = certify_production_run(output_dir, expected_frozen_definition_count=2)

    assert result["passed"] is False
    assert "concept_validation_summary.parquet" in result["missing_outputs"]
    assert "concept_validation_metadata.json" in result["missing_outputs"]
    assert "concept_validation_lineage_audit.json" in result["missing_outputs"]


def _write_valid_outputs(output_dir, registry, summary, metadata, lineage_audit):
    output_dir.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(output_dir / "concept_validation_registry.parquet", index=False)
    summary.to_parquet(output_dir / "concept_validation_summary.parquet", index=False)
    (output_dir / "concept_validation_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (output_dir / "concept_validation_lineage_audit.json").write_text(
        json.dumps(lineage_audit), encoding="utf-8"
    )


def _base_valid_registry_summary_metadata_audit():
    registry = pd.DataFrame({
        "frozen_definition_id": ["def_a", "def_b"],
        "discovery_season": [2024, 2024],
        "validation_season": [2025, 2025],
        "prediction_weight_assigned": [False, False],
    })
    summary = pd.DataFrame({
        "target_name": ["__all_targets__"],
        "prediction_weights_assigned": [False],
    })
    metadata = {
        "frozen_definitions_evaluated": 2,
        "concepts_tested": 2,
        "discovery_season": 2024,
        "validation_season": 2025,
        "prediction_weights_assigned": False,
        "2026_used": False,
        "certified_fully_reproducible": True,
    }
    lineage_audit = {
        "total_frozen_definitions_evaluated": 2,
        "total_validation_records_produced": 2,
        "orphan_validation_record_count": 0,
        "frozen_definitions_missing_validation_count": 0,
        "definition_sha256_mismatch_count": 0,
        "registry_sha256_mismatches": {
            "definition_registry_hash_consistent": True,
            "member_registry_hash_consistent": True,
        },
        "unexpected_duplicate_frozen_definition_id_mappings": {
            "in_frozen_definition_registry": [],
            "in_validation_registry": [],
        },
        "used_2026_data": False,
        "validation_frame_2026_row_count": 0,
        "reproducibility": {
            "discovery_season": 2024,
            "validation_season": 2025,
        },
        "certified_fully_reproducible": True,
    }
    return registry, summary, metadata, lineage_audit


def test_valid_synthetic_outputs_pass_certification(tmp_path):
    registry, summary, metadata, lineage_audit = (
        _base_valid_registry_summary_metadata_audit()
    )
    output_dir = tmp_path / "outputs"
    _write_valid_outputs(output_dir, registry, summary, metadata, lineage_audit)

    result = certify_production_run(output_dir, expected_frozen_definition_count=2)

    assert result["passed"] is True
    assert result["errors"] == []


def test_incorrect_record_count_fails_certification(tmp_path):
    registry, summary, metadata, lineage_audit = (
        _base_valid_registry_summary_metadata_audit()
    )
    # Metadata claims 3 records tested, but the registry only has 2.
    metadata["concepts_tested"] = 3
    lineage_audit["total_validation_records_produced"] = 3

    output_dir = tmp_path / "outputs"
    _write_valid_outputs(output_dir, registry, summary, metadata, lineage_audit)

    result = certify_production_run(output_dir, expected_frozen_definition_count=2)

    assert result["passed"] is False
    assert result["checks"]["validation_records_produced_consistent"] is False


def test_duplicate_frozen_definition_id_fails_certification(tmp_path):
    registry, summary, metadata, lineage_audit = (
        _base_valid_registry_summary_metadata_audit()
    )
    # Duplicate a frozen_definition_id row.
    registry = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)
    metadata["concepts_tested"] = 3
    lineage_audit["total_validation_records_produced"] = 3

    output_dir = tmp_path / "outputs"
    _write_valid_outputs(output_dir, registry, summary, metadata, lineage_audit)

    result = certify_production_run(output_dir, expected_frozen_definition_count=2)

    assert result["passed"] is False
    assert (
        result["checks"]["zero_unexpected_duplicate_frozen_definition_id_rows"]
        is False
    )


def test_input_and_output_hashes_recorded(rigged_module, tmp_path):
    manifest_dir = tmp_path / "manifest"

    exit_code, manifest = production_module.run_production_validation(
        manifest_dir=manifest_dir,
        expected_frozen_definition_count=2,
    )

    assert exit_code == 0

    for name, info in manifest["inputs"].items():
        assert info["sha256"], f"missing input hash for {name}"

    for name, info in manifest["outputs"].items():
        assert info["sha256"], f"missing output hash for {name}"
