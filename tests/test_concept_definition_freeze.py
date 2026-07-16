import pandas as pd
import pytest

from atlas.learning.concept_definition_freeze import (
    build_frozen_member_registry,
    build_frozen_registry,
    canonical_json,
    dataframe_registry_fingerprint,
    frozen_definition_fingerprint,
    frozen_definition_id,
    validate_freeze_source,
)


def valid_source():
    return pd.DataFrame({
        "concept_id": [
            "concept_a",
        ],
        "target_name": [
            "target_team_win",
        ],
        "concept_status": [
            "STRONG_CONCEPT_CANDIDATE",
        ],
        "broad_domain_pair": [
            "BULLPEN + IDENTITY",
        ],
        "member_1_feature": [
            "bullpen__bullpen_pitches_prior_1_dates",
        ],
        "member_1_threshold_operator": [
            ">=",
        ],
        "member_1_threshold_value": [
            70.0,
        ],
        "member_1_semantic_classification": [
            "VALID_RECENT_WORKLOAD_FACT",
        ],
        "member_1_governance_action": [
            "KEEP_SEMANTICALLY_VALID",
        ],
        "member_1_transformation_family_root": [
            "bullpen_pitches_1_dates",
        ],
        "member_1_base_metric_root": [
            "bullpen_pitches_1_dates",
        ],
        "member_2_feature": [
            "identity__identity_edge__run_differential",
        ],
        "member_2_threshold_operator": [
            ">=",
        ],
        "member_2_threshold_value": [
            0.50,
        ],
        "member_2_semantic_classification": [
            "VALID_CONTINUOUS_BASEBALL_FACT",
        ],
        "member_2_governance_action": [
            "KEEP_SEMANTICALLY_VALID",
        ],
        "member_2_transformation_family_root": [
            "run_differential",
        ],
        "member_2_base_metric_root": [
            "run_differential",
        ],
        "semantic_family_pair_key": [
            "bullpen_pitches_1_dates + run_differential",
        ],
        "base_metric_pair_key": [
            "bullpen_pitches_1_dates + run_differential",
        ],
        "concept_semantic_status": [
            "SEMANTICALLY_VALID_FREEZE_CANDIDATE",
        ],
    })


def test_canonical_json_is_deterministic():
    row = valid_source().iloc[0].to_dict()
    row["discovery_season"] = 2024

    first = canonical_json(row)
    second = canonical_json(dict(reversed(list(row.items()))))

    assert first == second


def test_definition_hash_is_deterministic():
    row = valid_source().iloc[0].to_dict()
    row["discovery_season"] = 2024

    assert frozen_definition_fingerprint(row) == frozen_definition_fingerprint(row)


def test_definition_id_is_deterministic():
    row = valid_source().iloc[0].to_dict()
    row["discovery_season"] = 2024

    assert frozen_definition_id(row) == frozen_definition_id(row)


def test_valid_source_passes_audit():
    audit = validate_freeze_source(valid_source())

    assert audit["row_valid_for_freeze"].all()


def test_identifier_action_fails_audit():
    source = valid_source()
    source.loc[
        0,
        "member_1_governance_action",
    ] = "BLOCK_IDENTIFIER_THRESHOLD"

    audit = validate_freeze_source(source)

    assert not audit["row_valid_for_freeze"].all()


def test_missing_threshold_fails_audit():
    source = valid_source()
    source.loc[
        0,
        "member_1_threshold_value",
    ] = None

    audit = validate_freeze_source(source)

    assert not audit["row_valid_for_freeze"].all()


def test_invalid_operator_fails_audit():
    source = valid_source()
    source.loc[
        0,
        "member_1_threshold_operator",
    ] = "approximately"

    audit = validate_freeze_source(source)

    assert not audit["row_valid_for_freeze"].all()


def test_frozen_registry_has_immutable_flags():
    frozen = build_frozen_registry(
        valid_source(),
        discovery_season=2024,
    )

    assert frozen["definitions_frozen"].all()
    assert not frozen["thresholds_mutable"].any()
    assert not frozen["member_features_mutable"].any()
    assert not frozen["target_mutable"].any()


def test_frozen_registry_uses_no_future_validation():
    frozen = build_frozen_registry(
        valid_source(),
        discovery_season=2024,
    )

    assert not frozen["2025_validation_used"].any()
    assert not frozen["2026_results_used"].any()


def test_member_registry_has_two_rows():
    frozen = build_frozen_registry(
        valid_source(),
        discovery_season=2024,
    )

    members = build_frozen_member_registry(
        frozen
    )

    assert len(members) == 2
    assert set(members["member_order"]) == {1, 2}


def test_registry_fingerprint_is_order_independent():
    dataframe = pd.DataFrame({
        "definition_sha256": [
            "b",
            "a",
        ],
    })

    reversed_dataframe = dataframe.iloc[
        ::-1
    ].reset_index(
        drop=True
    )

    assert (
        dataframe_registry_fingerprint(dataframe)
        == dataframe_registry_fingerprint(reversed_dataframe)
    )
