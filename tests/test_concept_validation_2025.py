import json

import numpy as np
import pandas as pd
import pytest

import atlas.validation.concept_validation_2025 as validation_module
from atlas.learning.concept_definition_freeze import (
    build_frozen_member_registry,
    build_frozen_registry,
    dataframe_registry_fingerprint,
)


def _raw_frozen_source() -> pd.DataFrame:
    return pd.DataFrame({
        "concept_id": [
            "concept_a",
            "concept_b",
        ],
        "target_name": [
            "target_team_win",
            "target_team_win",
        ],
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
        "member_1_threshold_operator": [
            ">=",
            ">=",
        ],
        "member_1_threshold_value": [
            70.0,
            0.10,
        ],
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
        "member_1_base_metric_root": [
            "workload",
            "missing_feature",
        ],
        "member_1_effect_direction": [
            "SUPPORTS_TARGET",
            "SUPPORTS_TARGET",
        ],
        "member_2_feature": [
            "identity__run_differential",
            "identity__run_differential",
        ],
        "member_2_threshold_operator": [
            ">=",
            ">=",
        ],
        "member_2_threshold_value": [
            0.50,
            0.50,
        ],
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
        "member_2_effect_direction": [
            "SUPPORTS_TARGET",
            "SUPPORTS_TARGET",
        ],
        "same_effect_direction": [
            True,
            True,
        ],
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
    frozen = build_frozen_registry(
        _raw_frozen_source(),
        discovery_season=2024,
    )

    members = build_frozen_member_registry(frozen)

    registry_sha256 = dataframe_registry_fingerprint(
        frozen,
        "definition_sha256",
    )

    member_registry_sha256 = dataframe_registry_fingerprint(
        members,
        "member_definition_sha256",
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
        "game_date": pd.Timestamp("2025-04-01") + pd.to_timedelta(
            np.arange(n_games) % 150, unit="D"
        ),
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

    # The canonical team-game target artifact only ever physically
    # contains the factual columns `won` and `run_differential`; it
    # never contains the frozen `target_team_win` column. The margin
    # magnitude is unrelated to the `wins` outcome direction, but the
    # sign always agrees with `wins` so source integrity checks pass.
    margin_magnitude = rng.integers(1, 7, size=n_games)
    run_differential = np.where(
        wins == 1,
        margin_magnitude,
        -margin_magnitude,
    )

    targets = pd.DataFrame({
        "game_pk": interactions["game_pk"],
        "game_date": interactions["game_date"],
        "atlas_season": interactions["atlas_season"],
        "team": interactions["team"],
        "won": wins.astype(bool),
        "run_differential": run_differential,
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

    monkeypatch.setattr(
        validation_module,
        "FROZEN_DEFINITION_REGISTRY_PATH",
        definition_path,
    )
    monkeypatch.setattr(
        validation_module,
        "FROZEN_MEMBER_REGISTRY_PATH",
        member_path,
    )
    monkeypatch.setattr(
        validation_module,
        "INTERACTION_PATH",
        interaction_path,
    )
    monkeypatch.setattr(
        validation_module,
        "TEAM_TARGET_PATH",
        target_path,
    )
    monkeypatch.setattr(
        validation_module,
        "OUTPUT_DIR",
        output_dir,
    )
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


def test_validation_registry_keyed_by_frozen_definition_id(rigged_module):
    result = rigged_module.run_concept_validation_2025()

    registry = pd.read_parquet(
        rigged_module.VALIDATION_REGISTRY_PATH
    )

    assert result["frozen_definitions_evaluated"] == 2
    assert len(registry) == 2
    assert not registry["frozen_definition_id"].isna().any()
    assert not registry["frozen_definition_id"].duplicated().any()


def test_every_record_has_required_lineage_fields(rigged_module):
    rigged_module.run_concept_validation_2025()

    registry = pd.read_parquet(
        rigged_module.VALIDATION_REGISTRY_PATH
    )

    required_fields = (
        "frozen_definition_id",
        "definition_sha256",
        "member_registry_sha256",
        "source_definition_registry_sha256",
        "source_member_registry_sha256",
        "discovery_season",
        "validation_season",
        "validation_engine_version",
        "validation_timestamp_utc",
    )

    for field in required_fields:
        assert field in registry.columns
        assert not registry[field].isna().any()

    assert (registry["discovery_season"] == 2024).all()
    assert (registry["validation_season"] == 2025).all()

    expected_definition_registry_sha256 = rigged_module.file_sha256(
        str(rigged_module.FROZEN_DEFINITION_REGISTRY_PATH)
    )
    expected_member_registry_sha256 = rigged_module.file_sha256(
        str(rigged_module.FROZEN_MEMBER_REGISTRY_PATH)
    )

    assert (
        registry["source_definition_registry_sha256"]
        == expected_definition_registry_sha256
    ).all()
    assert (
        registry["source_member_registry_sha256"]
        == expected_member_registry_sha256
    ).all()


def test_available_concept_is_validated_from_features(rigged_module):
    rigged_module.run_concept_validation_2025()

    registry = pd.read_parquet(
        rigged_module.VALIDATION_REGISTRY_PATH
    )

    available_row = registry.loc[
        registry["concept_id"] == "concept_a"
    ].iloc[0]

    assert available_row["feature_availability_status"] == (
        "both_features_available"
    )
    assert available_row["active_2025_sample"] > 0
    assert available_row["validation_status"] in {
        "validated_strong",
        "validated",
        "direction_retained_weak",
        "not_confirmed",
        "reversed",
        "reversed_strong",
        "insufficient_2025_sample",
    }


def test_regression_canonical_targets_without_frozen_column_are_resolved(
    rigged_module,
):
    """
    Regression test for the real production failure: the canonical
    team-game target artifact only physically contains `won` and
    `run_differential`, never the frozen `target_team_win` column
    directly. The engine must resolve the frozen target column before
    checking target availability, and must never classify a concept as
    `target_unavailable_2025` purely because the frozen column is
    absent from the canonical source.
    """

    targets = pd.read_parquet(rigged_module.TEAM_TARGET_PATH)

    assert "target_team_win" not in targets.columns
    assert "won" in targets.columns
    assert "run_differential" in targets.columns

    result = rigged_module.run_concept_validation_2025()

    assert result["target_unavailable_2025"] == 0

    registry = pd.read_parquet(rigged_module.VALIDATION_REGISTRY_PATH)

    assert not (
        registry["validation_status"] == "target_unavailable_2025"
    ).any()
    assert not (
        registry["feature_availability_status"] == "target_unavailable_2025"
    ).any()


def test_target_resolution_lineage_recorded_in_metadata_and_audit(
    rigged_module,
):
    result = rigged_module.run_concept_validation_2025()

    target_resolution = result["target_resolution"]

    assert target_resolution["rule_matches_manifest"] is True

    win_stats = target_resolution["resolved_targets"]["target_team_win"]
    win_by_2_stats = target_resolution["resolved_targets"][
        "target_team_win_by_2_plus"
    ]

    assert win_stats["non_null_resolved_rows"] == 200
    assert win_stats["positive_2025"] + win_stats["negative_2025"] == 200
    assert win_by_2_stats["non_null_resolved_rows"] == 200
    assert (
        win_by_2_stats["positive_2025"] + win_by_2_stats["negative_2025"]
        == 200
    )

    audit = json.loads(rigged_module.LINEAGE_AUDIT_PATH.read_text())

    assert audit["target_resolution"]["rule_matches_manifest"] is True
    assert audit["certified_fully_reproducible"] is True


def test_missing_feature_concept_still_produces_a_record(rigged_module):
    rigged_module.run_concept_validation_2025()

    registry = pd.read_parquet(
        rigged_module.VALIDATION_REGISTRY_PATH
    )

    missing_row = registry.loc[
        registry["concept_id"] == "concept_b"
    ].iloc[0]

    assert missing_row["feature_availability_status"] == (
        "member_feature_unavailable_2025"
    )
    assert missing_row["validation_status"] == (
        "member_feature_unavailable_2025"
    )


def test_lineage_audit_reports_zero_orphans_and_zero_2026_usage(rigged_module):
    rigged_module.run_concept_validation_2025()

    audit = json.loads(
        rigged_module.LINEAGE_AUDIT_PATH.read_text()
    )

    assert audit["total_frozen_definitions_evaluated"] == 2
    assert audit["total_validation_records_produced"] == 2
    assert audit["frozen_definitions_missing_validation_count"] == 0
    assert audit["validation_rows_missing_frozen_definition_id"] == 0
    assert audit["orphan_validation_record_count"] == 0
    assert audit["definition_sha256_mismatch_count"] == 0
    assert audit["registry_sha256_mismatches"]["definition_registry_hash_consistent"]
    assert audit["registry_sha256_mismatches"]["member_registry_hash_consistent"]
    assert audit["used_2026_data"] is False
    assert audit["detected_2026_rows_in_source"] == 0
    assert audit["certified_fully_reproducible"] is True


def test_outputs_are_all_written(rigged_module):
    rigged_module.run_concept_validation_2025()

    assert rigged_module.VALIDATION_REGISTRY_PATH.exists()
    assert rigged_module.VALIDATION_SUMMARY_PATH.exists()
    assert rigged_module.METADATA_PATH.exists()
    assert rigged_module.LINEAGE_AUDIT_PATH.exists()


def test_2026_source_rows_are_never_used(rigged_module):
    interactions = pd.read_parquet(
        rigged_module.INTERACTION_PATH
    )
    targets = pd.read_parquet(
        rigged_module.TEAM_TARGET_PATH
    )

    poisoned_interactions = pd.concat(
        [
            interactions,
            interactions.iloc[[0]].assign(
                atlas_season=2026,
                game_pk=999999,
            ),
        ],
        ignore_index=True,
    )

    poisoned_targets = pd.concat(
        [
            targets,
            targets.iloc[[0]].assign(
                atlas_season=2026,
                game_pk=999999,
            ),
        ],
        ignore_index=True,
    )

    poisoned_interactions.to_parquet(
        rigged_module.INTERACTION_PATH,
        index=False,
    )
    poisoned_targets.to_parquet(
        rigged_module.TEAM_TARGET_PATH,
        index=False,
    )

    result = rigged_module.run_concept_validation_2025()

    assert result["2026_used"] is False
    assert result["2026_rows_detected_in_source"] == 2

    audit = json.loads(
        rigged_module.LINEAGE_AUDIT_PATH.read_text()
    )

    assert audit["detected_2026_rows_in_source"] == 2
    assert audit["used_2026_data"] is False


def test_source_2026_rows_are_excluded_but_still_certify(rigged_module):
    """
    Shared upstream source files may legitimately contain 2026 rows (for
    example, because other consumers read the same files). Merely
    detecting those rows in the source must not block certification as
    long as this engine never consumes them: the validation frame it
    actually evaluates must contain zero 2026 rows and used_2026_data
    must remain False.
    """

    interactions = pd.read_parquet(
        rigged_module.INTERACTION_PATH
    )
    targets = pd.read_parquet(
        rigged_module.TEAM_TARGET_PATH
    )

    poisoned_interactions = pd.concat(
        [
            interactions,
            interactions.iloc[[0]].assign(
                atlas_season=2026,
                game_pk=999999,
            ),
        ],
        ignore_index=True,
    )

    poisoned_targets = pd.concat(
        [
            targets,
            targets.iloc[[0]].assign(
                atlas_season=2026,
                game_pk=999999,
            ),
        ],
        ignore_index=True,
    )

    poisoned_interactions.to_parquet(
        rigged_module.INTERACTION_PATH,
        index=False,
    )
    poisoned_targets.to_parquet(
        rigged_module.TEAM_TARGET_PATH,
        index=False,
    )

    result = rigged_module.run_concept_validation_2025()

    assert result["2026_rows_detected_in_source"] == 2
    assert result["2026_used"] is False
    assert result["certified_fully_reproducible"] is True

    audit = json.loads(
        rigged_module.LINEAGE_AUDIT_PATH.read_text()
    )

    assert audit["detected_2026_rows_in_source"] == 2
    assert audit["used_2026_data"] is False
    assert audit["validation_frame_2026_row_count"] == 0
    assert audit["certified_fully_reproducible"] is True


def test_mutated_frozen_registry_is_rejected(rigged_module):
    definitions = pd.read_parquet(
        rigged_module.FROZEN_DEFINITION_REGISTRY_PATH
    )
    definitions.loc[0, "thresholds_mutable"] = True
    definitions.to_parquet(
        rigged_module.FROZEN_DEFINITION_REGISTRY_PATH,
        index=False,
    )

    with pytest.raises(AssertionError):
        rigged_module.run_concept_validation_2025()


def test_frozen_source_files_are_not_modified(rigged_module):
    before = rigged_module.FROZEN_DEFINITION_REGISTRY_PATH.read_bytes()
    before_members = rigged_module.FROZEN_MEMBER_REGISTRY_PATH.read_bytes()

    rigged_module.run_concept_validation_2025()

    after = rigged_module.FROZEN_DEFINITION_REGISTRY_PATH.read_bytes()
    after_members = rigged_module.FROZEN_MEMBER_REGISTRY_PATH.read_bytes()

    assert before == after
    assert before_members == after_members


def test_failed_lineage_audit_does_not_overwrite_canonical_outputs(
    rigged_module,
):
    # First run succeeds and publishes canonical outputs.
    rigged_module.run_concept_validation_2025()

    registry_before = rigged_module.VALIDATION_REGISTRY_PATH.read_bytes()
    summary_before = rigged_module.VALIDATION_SUMMARY_PATH.read_bytes()
    metadata_before = rigged_module.METADATA_PATH.read_bytes()
    lineage_audit_before = rigged_module.LINEAGE_AUDIT_PATH.read_bytes()

    # Corrupt a definition's content hash. This does not violate any of
    # the immutability flags checked while loading the frozen registries,
    # so loading still succeeds, but the lineage audit's recomputed-hash
    # check will fail certification.
    definitions = pd.read_parquet(
        rigged_module.FROZEN_DEFINITION_REGISTRY_PATH
    )
    definitions.loc[0, "definition_sha256"] = "corrupted-hash"
    definitions.to_parquet(
        rigged_module.FROZEN_DEFINITION_REGISTRY_PATH,
        index=False,
    )

    with pytest.raises(
        validation_module.LineageAuditCertificationError
    ):
        rigged_module.run_concept_validation_2025()

    assert (
        rigged_module.VALIDATION_REGISTRY_PATH.read_bytes()
        == registry_before
    )
    assert (
        rigged_module.VALIDATION_SUMMARY_PATH.read_bytes()
        == summary_before
    )
    assert (
        rigged_module.METADATA_PATH.read_bytes()
        == metadata_before
    )
    assert (
        rigged_module.LINEAGE_AUDIT_PATH.read_bytes()
        == lineage_audit_before
    )


def test_bullpen_validation_module_imports_cleanly():
    """
    Regression test for docs/ATLAS_KNOWN_ISSUES.md OPEN-1.

    `bullpen_concept_validation_2025.run_bullpen_concept_validation_2025`
    monkey-patches globals on this module that no longer exist after the
    lineage-complete rewrite, and is tracked as a documented follow-up
    rather than fixed here. This test only asserts that importing the
    bullpen validation module, and this module, together stays healthy so
    that unrelated code paths (and the rest of the test suite) are never
    broken by that known, isolated issue.
    """

    import atlas.validation.bullpen_concept_validation_2025 as bullpen_module

    assert hasattr(
        bullpen_module,
        "run_bullpen_concept_validation_2025",
    )
    assert hasattr(
        validation_module,
        "run_concept_validation_2025",
    )
