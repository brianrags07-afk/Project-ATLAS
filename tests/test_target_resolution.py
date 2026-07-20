import numpy as np
import pandas as pd
import pytest

from atlas.validation.target_resolution import (
    FROZEN_TARGET_RESOLUTION_RULES,
    TARGET_TEAM_WIN,
    TARGET_TEAM_WIN_BY_2_PLUS,
    TargetResolutionIntegrityError,
    certify_target_resolution_matches_rules,
    resolve_frozen_targets,
    target_resolution_rules_fingerprint,
)


def _canonical_frame(won, run_differential) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_pk": np.arange(len(won)),
            "won": won,
            "run_differential": run_differential,
        }
    )


def test_win_maps_directly_from_won():
    frame = _canonical_frame([True, False], [3, -1])

    resolved, _ = resolve_frozen_targets(frame)

    assert list(resolved[TARGET_TEAM_WIN]) == pytest.approx([1.0, 0.0])


def test_margin_plus_2_maps_to_win_by_2_equals_1():
    frame = _canonical_frame([True], [2])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN_BY_2_PLUS].iloc[0] == pytest.approx(1.0)


def test_margin_plus_1_maps_to_win_by_2_equals_0():
    frame = _canonical_frame([True], [1])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN_BY_2_PLUS].iloc[0] == pytest.approx(0.0)


def test_margin_0_maps_to_win_by_2_equals_0():
    # `won` must be `False` here to satisfy the won/run_differential
    # agreement check, which treats `won` as equivalent to
    # `run_differential > 0`; a zero margin is not `> 0`.
    frame = _canonical_frame([False], [0])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN_BY_2_PLUS].iloc[0] == pytest.approx(0.0)


def test_negative_margin_maps_to_win_by_2_equals_0():
    frame = _canonical_frame([False], [-5])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN_BY_2_PLUS].iloc[0] == pytest.approx(0.0)


def test_null_run_differential_remains_null():
    # `won` is null (unavailable) for row 1, together with
    # `run_differential`, so agreement cannot be violated; row 0 has an
    # available `won` but a null `run_differential`, which is also
    # permitted (nothing to disagree with).
    frame = _canonical_frame([True, np.nan], [np.nan, np.nan])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN_BY_2_PLUS].isna().all()


def test_null_run_differential_with_won_true_remains_null():
    # `won=True` with an unavailable `run_differential` is permitted:
    # there is no other value to disagree with, and the win-by-2 target
    # simply stays null because its only source column is unavailable.
    frame = _canonical_frame([True], [np.nan])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN].iloc[0] == pytest.approx(1.0)
    assert pd.isna(resolved[TARGET_TEAM_WIN_BY_2_PLUS].iloc[0])


def test_null_run_differential_with_won_false_remains_null():
    frame = _canonical_frame([False], [np.nan])

    resolved, _ = resolve_frozen_targets(frame)

    assert resolved[TARGET_TEAM_WIN].iloc[0] == pytest.approx(0.0)
    assert pd.isna(resolved[TARGET_TEAM_WIN_BY_2_PLUS].iloc[0])


def test_missing_required_source_columns_fail():
    frame = pd.DataFrame({"won": [True, False]})

    with pytest.raises(TargetResolutionIntegrityError):
        resolve_frozen_targets(frame)


def test_non_binary_won_values_fail():
    frame = _canonical_frame([1, 2], [1, 5])

    with pytest.raises(TargetResolutionIntegrityError):
        resolve_frozen_targets(frame)


def test_won_run_differential_disagreement_fails():
    frame = _canonical_frame([True], [-3])

    with pytest.raises(TargetResolutionIntegrityError):
        resolve_frozen_targets(frame)


def test_all_frozen_target_names_become_available_after_resolution():
    frame = _canonical_frame([True, False, True], [4, -2, 1])

    resolved, _ = resolve_frozen_targets(frame)

    for target_name in FROZEN_TARGET_RESOLUTION_RULES:
        assert target_name in resolved.columns
        assert resolved[target_name].notna().all()


def test_resolution_never_touches_2026_rows():
    # `resolve_frozen_targets` operates purely on the dataframe handed
    # to it; validation season filtering happens upstream. This test
    # confirms resolution does not introduce or drop any rows, so a
    # caller that has already excluded 2026 rows keeps them excluded.
    frame = _canonical_frame([True, False], [3, -3])
    frame["atlas_season"] = [2025, 2026]

    frame_2025_only = frame[frame["atlas_season"] == 2025].copy()

    resolved, _ = resolve_frozen_targets(frame_2025_only)

    assert (resolved["atlas_season"] == 2025).all()
    assert len(resolved) == 1


def test_no_prediction_weights_assigned():
    frame = _canonical_frame([True, False], [3, -3])

    resolved, stats = resolve_frozen_targets(frame)

    assert "prediction_weight" not in "".join(resolved.columns).lower()
    assert "weight" not in stats


def test_resolution_stats_record_non_null_and_pos_neg_counts():
    frame = _canonical_frame(
        [True, False, True, np.nan],
        [3, -1, 1, np.nan],
    )

    _, stats = resolve_frozen_targets(frame)

    win_stats = stats["resolved_targets"][TARGET_TEAM_WIN]
    assert win_stats["non_null_resolved_rows"] == 3
    assert win_stats["positive_2025"] == 2
    assert win_stats["negative_2025"] == 1
    assert win_stats["source_columns"] == ["won"]
    assert win_stats["rule"] == "won"

    win_by_2_stats = stats["resolved_targets"][TARGET_TEAM_WIN_BY_2_PLUS]
    assert win_by_2_stats["non_null_resolved_rows"] == 3
    assert win_by_2_stats["positive_2025"] == 1
    assert win_by_2_stats["negative_2025"] == 2


def test_certify_target_resolution_matches_rules():
    frame = _canonical_frame([True, False], [3, -3])

    _, stats = resolve_frozen_targets(frame)

    assert certify_target_resolution_matches_rules(stats) is True


def test_certify_target_resolution_rejects_tampered_stats():
    frame = _canonical_frame([True, False], [3, -3])

    _, stats = resolve_frozen_targets(frame)

    tampered = {
        **stats,
        "resolved_targets": {
            **stats["resolved_targets"],
            TARGET_TEAM_WIN: {
                **stats["resolved_targets"][TARGET_TEAM_WIN],
                "rule": "won == 1 and something_else",
            },
        },
    }

    assert certify_target_resolution_matches_rules(tampered) is False


def test_rules_fingerprint_is_deterministic():
    assert (
        target_resolution_rules_fingerprint()
        == target_resolution_rules_fingerprint()
    )


def test_resolve_frozen_targets_does_not_mutate_input():
    frame = _canonical_frame([True, False], [3, -3])
    before = frame.copy()

    resolve_frozen_targets(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_regression_canonical_inputs_without_frozen_target_columns_resolve():
    """
    Regression test for the real production failure: canonical inputs
    contain `won` and `run_differential` but not the frozen target-name
    columns (`target_team_win`, `target_team_win_by_2_plus`). Resolution
    must materialize both frozen columns rather than leaving them
    unavailable.
    """

    frame = _canonical_frame([True, False, True], [2, -4, 5])

    assert TARGET_TEAM_WIN not in frame.columns
    assert TARGET_TEAM_WIN_BY_2_PLUS not in frame.columns

    resolved, stats = resolve_frozen_targets(frame)

    assert TARGET_TEAM_WIN in resolved.columns
    assert TARGET_TEAM_WIN_BY_2_PLUS in resolved.columns
    assert resolved[TARGET_TEAM_WIN].notna().all()
    assert resolved[TARGET_TEAM_WIN_BY_2_PLUS].notna().all()
    assert certify_target_resolution_matches_rules(stats) is True
