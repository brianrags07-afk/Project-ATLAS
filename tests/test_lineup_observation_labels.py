from __future__ import annotations

import pandas as pd
import pytest

from atlas.lineups.historical_lineup_engine import build_historical_starting_lineups
from atlas.lineups.lineup_observation_labels import (
    BATTING_ORDER_COLUMNS,
    build_reconstructed_lineup_observation_labels,
    certify_reconstructed_lineup_observation_labels,
)
from scripts.build_lineup_observation_labels import (
    build_player_start_summary,
    build_team_pattern_summary,
)


def team_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 10,
                "game_start_at": "2024-04-01T20:00:00Z",
                "official_date": "2024-04-01",
                "season": 2024,
                "team": "HME",
                "team_id": 1,
                "opponent": "AWY",
                "opponent_team_id": 2,
                "home_away": "HOME",
            },
            {
                "game_pk": 10,
                "game_start_at": "2024-04-01T20:00:00Z",
                "official_date": "2024-04-01",
                "season": 2024,
                "team": "AWY",
                "team_id": 2,
                "opponent": "HME",
                "opponent_team_id": 1,
                "home_away": "AWAY",
            },
        ]
    )


def pitch_events(*, game_type: str = "R", omit_home_nine: bool = False) -> pd.DataFrame:
    rows = []
    for slot in range(1, 10):
        rows.append(
            {
                "game_pk": 10,
                "game_date": "2024-04-01",
                "atlas_season": 2024,
                "game_type": game_type,
                "home_team": "HME",
                "away_team": "AWY",
                "inning": 1,
                "inning_topbot": "Top",
                "at_bat_number": slot,
                "pitch_number": 1,
                "batter": 100 + slot,
                "pitcher": 501,
            }
        )
    for slot in range(1, 10 if not omit_home_nine else 9):
        rows.append(
            {
                "game_pk": 10,
                "game_date": "2024-04-01",
                "atlas_season": 2024,
                "game_type": game_type,
                "home_team": "HME",
                "away_team": "AWY",
                "inning": 1,
                "inning_topbot": "Bot",
                "at_bat_number": 9 + slot,
                "pitch_number": 1,
                "batter": 200 + slot,
                "pitcher": 601,
            }
        )
    # A later plate appearance must not change the reconstructed first nine.
    rows.append(
        {
            "game_pk": 10,
            "game_date": "2024-04-01",
            "atlas_season": 2024,
            "game_type": game_type,
            "home_team": "HME",
            "away_team": "AWY",
            "inning": 2,
            "inning_topbot": "Top",
            "at_bat_number": 19,
            "pitch_number": 1,
            "batter": 101,
            "pitcher": 501,
        }
    )
    return pd.DataFrame(rows)


def build(**kwargs) -> pd.DataFrame:
    return build_reconstructed_lineup_observation_labels(
        pitch_events(**kwargs), team_games(), season=2024
    )


def test_reconstructs_first_nine_and_mirrored_starting_pitchers():
    result = build()
    away = result.loc[result["team"].eq("AWY")].iloc[0]
    home = result.loc[result["team"].eq("HME")].iloc[0]
    assert [int(away[column]) for column in BATTING_ORDER_COLUMNS] == list(
        range(101, 110)
    )
    assert [int(home[column]) for column in BATTING_ORDER_COLUMNS] == list(
        range(201, 210)
    )
    assert int(away["starting_pitcher_id"]) == 601
    assert int(away["opposing_starting_pitcher_id"]) == 501
    assert int(home["starting_pitcher_id"]) == 501
    assert int(home["opposing_starting_pitcher_id"]) == 601


def test_labels_are_postgame_targets_never_same_game_features():
    result = build()
    assert result["reconstruction_is_observed_lineup_proxy"].all()
    assert not result["official_starting_lineup_confirmed"].any()
    assert result["postgame_observation_label"].all()
    assert not result["published_lineup_confirmed"].any()
    assert not result["same_game_pregame_eligible"].any()
    assert result["eligible_for_training_target"].all()
    assert result["eligible_for_future_game_feature"].all()
    assert not result["direct_feature_use_allowed"].any()
    assert set(result["label_available_at"].astype(str)) == {
        "2024-04-02 20:00:00+00:00"
    }
    assert result["observed_at"].equals(result["game_start_at"])
    assert set(result["observation_time_semantics"]) == {"game_start_lower_bound"}
    assert set(result["future_feature_eligibility_rule"]) == {
        "label_available_at_must_precede_target_game_start_at"
    }


def test_incomplete_first_nine_is_preserved_and_counted():
    result = build(omit_home_nine=True)
    home = result.loc[result["team"].eq("HME")].iloc[0]
    assert int(home["starting_lineup_size"]) == 8
    assert not home["starting_lineup_complete"]
    assert pd.isna(home["batting_order_9_player_id"])
    report = certify_reconstructed_lineup_observation_labels(
        result, team_games(), season=2024
    )
    assert report["verdict"] == "certified"
    assert report["incomplete_lineups"] == 1


def test_rejects_non_regular_events_and_schedule_coverage_mismatch():
    with pytest.raises(ValueError, match="regular-season-only"):
        build(game_type="S")
    mismatched = team_games().copy()
    mismatched["game_pk"] = 11
    with pytest.raises(ValueError, match="game coverage differs"):
        build_reconstructed_lineup_observation_labels(
            pitch_events(), mismatched, season=2024
        )


def test_certifier_rejects_same_game_eligibility_and_early_availability():
    result = build()
    unsafe = result.copy()
    unsafe["same_game_pregame_eligible"] = True
    unsafe["label_available_at"] = unsafe["game_start_at"]
    report = certify_reconstructed_lineup_observation_labels(
        unsafe, team_games(), season=2024
    )
    assert report["verdict"] == "quarantine_required"
    assert "same_game_pregame_eligible" in "; ".join(report["errors"])
    assert "less than 24 hours" in "; ".join(report["errors"])


def test_certifier_allows_mirrored_missing_starters_without_false_mismatch():
    result = build()
    result["starting_pitcher_id"] = pd.Series([pd.NA, pd.NA], dtype="Int64")
    result["opposing_starting_pitcher_id"] = pd.Series(
        [pd.NA, pd.NA], dtype="Int64"
    )
    report = certify_reconstructed_lineup_observation_labels(
        result, team_games(), season=2024
    )
    assert report["verdict"] == "certified"
    assert report["starter_mirror_errors"] == 0
    assert report["missing_starting_pitchers"] == 2


def test_certifier_rejects_tampered_lineup_content():
    result = build()
    result.loc[result.index[0], "lineup_order_signature"] = "tampered"
    report = certify_reconstructed_lineup_observation_labels(
        result, team_games(), season=2024
    )
    assert report["verdict"] == "quarantine_required"
    assert report["lineup_content_errors"] == 1


def test_build_summaries_preserve_team_player_and_order_detail():
    result = build()
    team_summary = build_team_pattern_summary(result)
    assert len(team_summary) == 2
    assert team_summary["team_games"].eq(1).all()
    assert team_summary["complete_lineups"].eq(1).all()

    player_summary = build_player_start_summary(result)
    leadoff = player_summary.loc[
        player_summary["team"].eq("AWY") & player_summary["player_id"].eq(101)
    ].iloc[0]
    home_starter = player_summary.loc[
        player_summary["team"].eq("HME") & player_summary["player_id"].eq(501)
    ].iloc[0]
    assert int(leadoff["batting_starts"]) == 1
    assert int(leadoff["batting_order_1_starts"]) == 1
    assert int(home_starter["starting_pitcher_starts"]) == 1


def test_legacy_engine_no_longer_calls_reconstruction_pregame_evidence():
    result = build_historical_starting_lineups(pitch_events())
    assert set(result["pregame_information_class"]) == {
        "postgame_reconstructed_truth_label_not_pregame_evidence"
    }
    assert result["postgame_truth_label"].all()
    assert not result["published_lineup_confirmed"].any()
    assert not result["same_game_pregame_eligible"].any()
