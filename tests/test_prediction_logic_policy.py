from pathlib import Path

import pandas as pd
import pytest

from atlas.governance.prediction_logic_policy import (
    MODULE_POLICIES,
    POLICY_VERSION,
    approved_phase_2e_modules,
    approved_prediction_runtime_modules,
    assert_phase_2e_allowed,
    assert_prediction_runtime_allowed,
    quarantined_modules,
)


PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)


def test_policy_version():
    assert POLICY_VERSION == "1.0.0"


def test_legacy_prediction_engine_is_quarantined():
    module = "atlas/predictions/pregame_prediction_engine.py"

    assert (
        MODULE_POLICIES[module].status
        == "QUARANTINED_LEGACY_DECISION_ENGINE"
    )

    assert module in quarantined_modules()

    with pytest.raises(PermissionError):
        assert_phase_2e_allowed(module)

    with pytest.raises(PermissionError):
        assert_prediction_runtime_allowed(module)


def test_prediction_fusion_engine_is_quarantined():
    module = "atlas/predictions/prediction_fusion_engine.py"

    assert module in quarantined_modules()

    with pytest.raises(PermissionError):
        assert_prediction_runtime_allowed(module)


def test_weighted_backtest_engine_is_quarantined():
    module = "atlas/backtest/weighted_state_backtest_engine.py"

    assert module in quarantined_modules()

    with pytest.raises(PermissionError):
        assert_prediction_runtime_allowed(module)


def test_factual_target_builder_is_allowed_for_phase_2e_only():
    module = "atlas/learning/backtest_target_builder.py"

    assert module in approved_phase_2e_modules()

    assert_phase_2e_allowed(module)

    with pytest.raises(PermissionError):
        assert_prediction_runtime_allowed(module)


def test_walk_forward_raw_facts_are_runtime_approved():
    module = "atlas/interactions/walk_forward_snapshot_engine.py"

    assert module in approved_phase_2e_modules()
    assert module in approved_prediction_runtime_modules()

    assert_phase_2e_allowed(module)
    assert_prediction_runtime_allowed(module)


def test_bullpen_fatigue_engine_is_raw_facts_only():
    module = (
        "atlas/identities/"
        "bullpen_availability_fatigue_engine.py"
    )

    assert MODULE_POLICIES[module].status == "RAW_FACTS_ONLY"

    assert_phase_2e_allowed(module)

    with pytest.raises(PermissionError):
        assert_prediction_runtime_allowed(module)


def test_evidence_discovery_is_reliability_only():
    for module in [
        "atlas/learning/team_evidence_discovery.py",
        "atlas/learning/league_evidence_discovery.py",
    ]:
        assert MODULE_POLICIES[module].status == "RELIABILITY_ONLY"

        assert_phase_2e_allowed(module)

        with pytest.raises(PermissionError):
            assert_prediction_runtime_allowed(module)


def test_governance_artifacts_exist():
    base = (
        PROJECT_ROOT
        / "data"
        / "governance"
        / "prediction_logic"
    )

    required = [
        base / "prediction_module_policy.csv",
        base / "prediction_logic_adjudication_registry.csv",
        base / "prediction_logic_policy_metadata.json",
    ]

    for path in required:
        assert path.exists(), path


def test_module_policy_artifact_matches_python_policy():
    path = (
        PROJECT_ROOT
        / "data"
        / "governance"
        / "prediction_logic"
        / "prediction_module_policy.csv"
    )

    policy = pd.read_csv(path)

    assert len(policy) == len(MODULE_POLICIES)
    assert policy["relative_path"].is_unique

    expected_quarantined = set(quarantined_modules())

    saved_quarantined = set(
        policy.loc[
            policy["status"].isin(
                [
                    "QUARANTINED_LEGACY_DECISION_ENGINE",
                    "ANALYSIS_ONLY",
                    "HUMAN_REVIEW_ONLY",
                ]
            ),
            "relative_path",
        ]
    )

    assert saved_quarantined == expected_quarantined


def test_no_quarantined_module_is_runtime_approved():
    quarantined = set(quarantined_modules())
    runtime = set(approved_prediction_runtime_modules())

    assert quarantined.isdisjoint(runtime)
