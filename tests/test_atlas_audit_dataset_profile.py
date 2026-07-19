"""
Tests for atlas/audit/dataset_profile.py.

Uses small synthetic DataFrames only -- never downloads or reads any
production Cloud Storage data.
"""

from __future__ import annotations

import pandas as pd

from atlas.audit.dataset_profile import (
    classify_column,
    detect_likely_primary_key,
    duplicate_columns,
    null_percentages,
    profile_master_game_database,
    profile_master_pitch_database,
    profile_metadata_json,
    profile_team_game_state,
    rows_by_season,
    unique_games_by_season,
)


def _game_db_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "game_pk": [1, 1, 2, 3, 4],
        "game_date": ["2024-04-01", "2024-04-01", "2024-04-02", "2025-04-01", "2026-03-30"],
        "home_team": ["NYY", "NYY", "BOS", "LAD", "SF"],
        "away_team": ["BOS", "BOS", "NYY", "SF", "LAD"],
        "home_score": [3, 3, 5, 2, None],
    })


def test_classify_column_identifier():
    assert classify_column("game_pk") == "identifier"
    assert classify_column("player_id") == "identifier"


def test_classify_column_schedule_safe():
    assert classify_column("game_date") == "schedule_safe"
    assert classify_column("home_team") == "schedule_safe"


def test_classify_column_postgame_fact():
    assert classify_column("home_score") in ("postgame_fact", "schedule_safe", "identifier")
    assert classify_column("final_score") == "postgame_fact"


def test_classify_column_needs_timestamp_proof():
    assert classify_column("starter_name") == "pregame_possible_but_needs_timestamp_proof"
    assert classify_column("bullpen_rest_days") == "pregame_possible_but_needs_timestamp_proof"


def test_classify_column_unknown_when_no_evidence():
    assert classify_column("zzz_totally_unrecognized_field") == "unknown"


def test_detect_likely_primary_key_uses_first_present_candidate():
    df = _game_db_fixture()
    key, dup_count = detect_likely_primary_key(df, [["game_pk"], ["game_id"]])
    assert key == ["game_pk"]
    assert dup_count == 1  # game_pk=1 appears twice


def test_detect_likely_primary_key_returns_none_when_no_candidate_present():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    key, dup_count = detect_likely_primary_key(df, [["game_pk"], ["game_id"]])
    assert key is None
    assert dup_count == -1


def test_duplicate_columns_detects_repeated_names():
    df = _game_db_fixture()
    df.columns = ["game_pk", "game_date", "home_team", "home_team", "home_score"]
    assert "home_team" in duplicate_columns(df)


def test_null_percentages_reports_actual_nulls():
    df = _game_db_fixture()
    pcts = null_percentages(df)
    assert pcts["home_score"] == 20.0  # 1 of 5 rows null
    assert pcts["game_pk"] == 0.0


def test_rows_by_season_derives_season_from_game_date():
    df = _game_db_fixture()
    counts = rows_by_season(df)
    assert counts == {"2024": 3, "2025": 1, "2026": 1}


def test_unique_games_by_season_counts_distinct_game_pk():
    df = _game_db_fixture()
    counts = unique_games_by_season(df)
    assert counts["2024"] == 2  # game_pk 1 (dup row) + 2


def test_profile_master_game_database_reports_expected_fields():
    df = _game_db_fixture()
    profile = profile_master_game_database(df, "data/master/master_game_database.parquet", 1000)
    assert profile["cloud_path"] == "data/master/master_game_database.parquet"
    assert profile["row_count"] == 5
    assert profile["likely_primary_key"] == ["game_pk"]
    assert profile["duplicate_key_count"] == 1
    assert profile["seasons_present"] == ["2024", "2025", "2026"]
    assert profile["feature_presence"]["game_pk"] == "game_pk"
    assert profile["feature_presence"]["home_away_teams"] in ("home_team", "away_team")
    assert profile["feature_presence"]["starter_information"] is None


def test_profile_team_game_state_uses_game_pk_and_team_grain():
    df = pd.DataFrame({
        "game_pk": [1, 1, 2, 2],
        "team": ["NYY", "BOS", "LAD", "SF"],
        "game_date": ["2025-04-01"] * 4,
    })
    profile = profile_team_game_state(df, "data/master/team_game_state.parquet", 500)
    assert profile["likely_primary_key"] == ["game_pk", "team"]
    assert profile["duplicate_key_count"] == 0
    assert "game_pk + team" in profile["grain"]


def test_profile_master_pitch_database_chronology_reconstructable_true():
    df = pd.DataFrame({
        "game_pk": [1, 1, 1],
        "game_date": ["2025-04-01"] * 3,
        "inning": [1, 1, 1],
        "at_bat_number": [1, 1, 2],
        "pitch_number": [1, 2, 1],
    })
    profile = profile_master_pitch_database(df, "data/master/master_pitch_database.parquet", 700)
    assert profile["chronology_reconstructable"] is True
    assert profile["pitches_by_season"] == {"2025": 3}


def test_profile_master_pitch_database_chronology_reconstructable_false_when_missing_fields():
    df = pd.DataFrame({
        "game_pk": [1, 1],
        "game_date": ["2025-04-01"] * 2,
    })
    profile = profile_master_pitch_database(df, "data/master/master_pitch_database.parquet", 700)
    assert profile["chronology_reconstructable"] is False


def test_profile_metadata_json_compares_claimed_vs_actual_row_counts():
    profiles = {
        "master_game_database": {"row_count": 5},
    }
    metadata = {
        "master_game_database": {"row_count": 5},
        "unknown_dataset": {"row_count": 10},
    }
    comparison = profile_metadata_json(metadata, profiles)
    entries = {c["dataset"]: c for c in comparison["comparisons"]}
    assert entries["master_game_database"]["row_count_match"] is True
    assert entries["unknown_dataset"]["status"] == "no_matching_profiled_dataset"


def test_profile_metadata_json_flags_mismatch():
    profiles = {"master_game_database": {"row_count": 4}}
    metadata = {"master_game_database": {"row_count": 5}}
    comparison = profile_metadata_json(metadata, profiles)
    assert comparison["comparisons"][0]["row_count_match"] is False
