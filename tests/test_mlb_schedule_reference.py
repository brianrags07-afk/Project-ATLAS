"""
Tests for atlas/schedule/mlb_schedule_reference.py: the authoritative MLB
schedule reference layer built from the published MLB Stats API
``/schedule`` endpoint.
"""

from __future__ import annotations

from atlas.schedule.mlb_schedule_reference import (
    SCHEDULE_ENDPOINT,
    build_season_schedule_profile,
    compare_against_master_datasets,
    extract_raw_games,
    fetch_schedule_raw,
    normalize_game_row,
    normalize_schedule,
    postseason_rows,
    regular_season_rows,
    schedule_game_ids,
)


def _status(detailed_state, status_code="F", coded_game_state="F", abstract="Final"):
    return {
        "abstractGameState": abstract,
        "codedGameState": coded_game_state,
        "detailedState": detailed_state,
        "statusCode": status_code,
    }


def _raw_game(
    game_pk,
    *,
    game_type="R",
    season="2024",
    game_date="2024-04-01T17:10:00Z",
    official_date="2024-04-01",
    detailed_state="Final",
    status_code="F",
    double_header="N",
    game_number=1,
    series_game_number=1,
    games_in_series=3,
    series_description="Regular Season",
    home_id=133,
    home_name="Athletics",
    away_id=147,
    away_name="Yankees",
    venue_id=10,
    venue_name="Oakland Coliseum",
):
    return {
        "gamePk": game_pk,
        "gameGuid": f"guid-{game_pk}",
        "season": season,
        "gameType": game_type,
        "gameDate": game_date,
        "officialDate": official_date,
        "status": _status(detailed_state, status_code=status_code),
        "doubleHeader": double_header,
        "gameNumber": game_number,
        "seriesGameNumber": series_game_number,
        "gamesInSeries": games_in_series,
        "seriesDescription": series_description,
        "teams": {
            "home": {"team": {"id": home_id, "name": home_name}},
            "away": {"team": {"id": away_id, "name": away_name}},
        },
        "venue": {"id": venue_id, "name": venue_name},
    }


def _payload(games):
    return {"dates": [{"date": "2024-04-01", "games": games}]}


# --------------------------------------------------------------------------
# extract_raw_games / normalize_game_row
# --------------------------------------------------------------------------


def test_extract_raw_games_flattens_dates_without_altering_fields():
    raw = {
        "dates": [
            {"date": "2024-04-01", "games": [_raw_game(1)]},
            {"date": "2024-04-02", "games": [_raw_game(2)]},
        ]
    }
    games = extract_raw_games(raw)
    assert [g["gamePk"] for g in games] == [1, 2]


def test_extract_raw_games_handles_missing_dates_key():
    assert extract_raw_games({}) == []


def test_normalize_game_row_uses_only_published_fields():
    raw = _raw_game(717896, series_game_number=2, games_in_series=4, series_description="Regular Season")
    row = normalize_game_row(raw, retrieved_at_utc="2024-04-01T00:00:00+00:00", source_url=SCHEDULE_ENDPOINT)

    assert row["game_pk"] == 717896
    assert row["season"] == "2024"
    assert row["season_segment"] == "regular_season"
    assert row["series_game_number"] == 2
    assert row["games_in_series"] == 4
    assert row["series_description"] == "Regular Season"
    assert row["home_team_id"] == 133
    assert row["away_team_id"] == 147
    assert row["source"] == "mlb_stats_api_schedule"
    assert row["counted_in_expected_games"] is True
    assert row["is_final"] is True
    assert isinstance(row["content_hash"], str) and len(row["content_hash"]) == 64


def test_normalize_game_row_content_hash_is_stable_for_identical_content():
    raw = _raw_game(717896)
    row_a = normalize_game_row(raw, retrieved_at_utc="2024-04-01T00:00:00Z", source_url=SCHEDULE_ENDPOINT)
    row_b = normalize_game_row(raw, retrieved_at_utc="2099-01-01T00:00:00Z", source_url="different-url")
    assert row_a["content_hash"] == row_b["content_hash"]


def test_normalize_game_row_content_hash_changes_when_status_changes():
    scheduled = _raw_game(717896, detailed_state="Scheduled", status_code="S")
    final = _raw_game(717896, detailed_state="Final", status_code="F")
    row_scheduled = normalize_game_row(scheduled, retrieved_at_utc="t", source_url=SCHEDULE_ENDPOINT)
    row_final = normalize_game_row(final, retrieved_at_utc="t", source_url=SCHEDULE_ENDPOINT)
    assert row_scheduled["content_hash"] != row_final["content_hash"]


def test_unknown_game_type_and_detailed_state_are_never_guessed():
    raw = _raw_game(1, game_type="Z", detailed_state="Something New")
    row = normalize_game_row(raw, retrieved_at_utc="t", source_url=SCHEDULE_ENDPOINT)
    assert row["season_segment"] == "unknown"
    assert row["game_state_category"] == "unknown"
    # Unknown state must not be assumed cancelled -- it still counts.
    assert row["counted_in_expected_games"] is True


# --------------------------------------------------------------------------
# Regular season vs. postseason separation.
# --------------------------------------------------------------------------


def test_regular_season_and_postseason_rows_are_separated_by_published_game_type():
    rows = normalize_schedule(
        _payload(
            [
                _raw_game(1, game_type="R"),
                _raw_game(2, game_type="W", series_description="World Series"),
                _raw_game(3, game_type="D", series_description="Division Series"),
                _raw_game(4, game_type="A", series_description="All Star Game"),
            ]
        )
    )
    regular = regular_season_rows(rows)
    postseason = postseason_rows(rows)
    assert [r["game_pk"] for r in regular] == [1]
    assert sorted(r["game_pk"] for r in postseason) == [2, 3]
    # All-Star (gameType "A") is neither regular season nor postseason.
    assert all(r["game_pk"] != 4 for r in regular + postseason)


# --------------------------------------------------------------------------
# Postponements / cancellations / suspensions / doubleheaders.
# --------------------------------------------------------------------------


def test_postponed_then_replayed_game_with_same_game_pk_is_counted_once():
    """MLB keeps the same gamePk when a postponed game is rescheduled.
    Merging two raw payloads (one from the original postponed snapshot,
    one from after the makeup date is played) must yield exactly one row."""
    postponed_snapshot = _raw_game(
        555, official_date="2024-04-10", detailed_state="Postponed", status_code="PPD"
    )
    replayed_snapshot = _raw_game(
        555, official_date="2024-04-11", detailed_state="Final", status_code="F"
    )
    rows = normalize_schedule([_payload([postponed_snapshot]), _payload([replayed_snapshot])])
    assert len(rows) == 1
    assert rows[0]["game_pk"] == 555
    # The more authoritative "Final" snapshot wins over "Postponed".
    assert rows[0]["detailed_state"] == "Final"
    assert rows[0]["official_date"] == "2024-04-11"
    assert rows[0]["counted_in_expected_games"] is True


def test_cancelled_game_with_no_makeup_is_excluded_from_expected_games():
    cancelled = _raw_game(556, detailed_state="Cancelled", status_code="CPD")
    rows = normalize_schedule(_payload([cancelled]))
    assert len(rows) == 1
    assert rows[0]["counted_in_expected_games"] is False
    profile = build_season_schedule_profile(rows)
    assert profile["expected_games_by_season"] == {}


def test_suspended_game_resumed_later_same_game_pk_counts_once():
    suspended_snapshot = _raw_game(600, detailed_state="Suspended", status_code="S")
    resumed_final_snapshot = _raw_game(600, detailed_state="Final", status_code="F")
    rows = normalize_schedule([_payload([suspended_snapshot]), _payload([resumed_final_snapshot])])
    assert len(rows) == 1
    assert rows[0]["detailed_state"] == "Final"
    assert rows[0]["counted_in_expected_games"] is True


def test_doubleheader_games_are_two_distinct_counted_games():
    game_one = _raw_game(701, double_header="Y", game_number=1)
    game_two = _raw_game(702, double_header="Y", game_number=2)
    rows = normalize_schedule(_payload([game_one, game_two]))
    assert len(rows) == 2
    assert {r["game_number"] for r in rows} == {1, 2}
    profile = build_season_schedule_profile(rows)
    assert profile["expected_games_by_season"]["2024"] == 2


def test_repeated_raw_entries_for_same_game_pk_never_double_count():
    """Even without any postponement, duplicate raw entries for the same
    gamePk (e.g. overlapping date-range queries) must collapse to one row."""
    raw = _raw_game(999)
    rows = normalize_schedule([_payload([raw]), _payload([raw]), _payload([raw])])
    assert len(rows) == 1


# --------------------------------------------------------------------------
# season_schedule audit profile / expected_games_by_season.
# --------------------------------------------------------------------------


def test_build_season_schedule_profile_shape_matches_audit_contract():
    rows = normalize_schedule(
        _payload([_raw_game(1, season="2024"), _raw_game(2, season="2024"), _raw_game(3, season="2025")])
    )
    profile = build_season_schedule_profile(rows)
    assert profile["expected_games_by_season"] == {"2024": 2, "2025": 1}
    assert profile["source"] == "mlb_stats_api_schedule"
    assert profile["source_endpoint"] == SCHEDULE_ENDPOINT
    assert isinstance(profile["content_hash"], str) and len(profile["content_hash"]) == 64


def test_expected_games_by_season_excludes_postseason_games():
    rows = normalize_schedule(
        _payload(
            [
                _raw_game(1, game_type="R", season="2024"),
                _raw_game(2, game_type="W", season="2024", series_description="World Series"),
            ]
        )
    )
    profile = build_season_schedule_profile(rows)
    assert profile["expected_games_by_season"] == {"2024": 1}
    assert profile["postseason_games_by_season"] == {"2024": 1}


def test_season_schedule_profile_is_consumable_by_coverage_matrix():
    """The season_schedule profile produced here must plug directly into
    atlas.audit.coverage_matrix's independent expected-game-count check."""
    from atlas.audit.coverage_matrix import build_coverage_matrix

    rows = normalize_schedule(_payload([_raw_game(i, season="2024") for i in range(1, 101)]))
    schedule_profile = build_season_schedule_profile(rows)

    dataset_profiles = {
        "master_game_database": {
            "cloud_path": "data/master/master_game_database.parquet",
            "rows_by_season": {"2024": 100},
            "unique_games_by_season": {"2024": 100},
            "feature_presence": {"final_outcomes": "home_score"},
            "column_classification": {"home_score": "postgame_fact"},
            "null_percentages": {"home_score": 0.0},
            "schema_fingerprint": "game-fp-1",
            "data_layer": "normalized_master",
        },
        "season_schedule": schedule_profile,
    }
    matrix = build_coverage_matrix(dataset_profiles, {"focus_area_index": {}})
    row = next(r for r in matrix if r["row"] == "final_scores" and r["season"] == 2024)
    assert row["data_presence"] == "present"
    assert row["source_completeness"] == "complete"


# --------------------------------------------------------------------------
# Comparison against caller-supplied master-game / master-pitch ID sets.
# --------------------------------------------------------------------------


def test_schedule_game_ids_excludes_uncounted_cancelled_games():
    rows = normalize_schedule(
        _payload([_raw_game(1), _raw_game(2, detailed_state="Cancelled", status_code="CPD")])
    )
    assert schedule_game_ids(rows) == {1}


def test_compare_against_master_datasets_finds_missing_and_unexpected_ids():
    rows = normalize_schedule(_payload([_raw_game(1), _raw_game(2), _raw_game(3)]))
    comparison = compare_against_master_datasets(
        rows,
        master_game_ids=[1, 2, 999],
        master_pitch_game_ids=[1],
    )
    assert comparison["expected_game_ids"] == [1, 2, 3]
    assert comparison["missing_from_master_game_database"] == [3]
    assert comparison["unexpected_in_master_game_database"] == [999]
    assert comparison["missing_from_master_pitch_database"] == [2, 3]
    assert comparison["unexpected_in_master_pitch_database"] == []


def test_compare_against_master_datasets_does_not_mutate_inputs():
    rows = normalize_schedule(_payload([_raw_game(1)]))
    master_game_ids = [1]
    master_pitch_game_ids = [1]
    compare_against_master_datasets(rows, master_game_ids, master_pitch_game_ids)
    assert master_game_ids == [1]
    assert master_pitch_game_ids == [1]


# --------------------------------------------------------------------------
# fetch_schedule_raw: network call wiring only (no live network access).
# --------------------------------------------------------------------------


def test_fetch_schedule_raw_calls_published_schedule_endpoint(monkeypatch):
    captured = {}

    def fake_get_json(url, params=None, timeout=30, retries=3):
        captured["url"] = url
        captured["params"] = params
        return _payload([_raw_game(1)])

    monkeypatch.setattr(
        "atlas.schedule.mlb_schedule_reference.get_json", fake_get_json
    )
    result = fetch_schedule_raw("2024-04-01", "2024-04-02", game_types=["R"])
    assert captured["url"] == SCHEDULE_ENDPOINT
    assert captured["params"]["startDate"] == "2024-04-01"
    assert captured["params"]["endDate"] == "2024-04-02"
    assert captured["params"]["gameType"] == "R"
    assert extract_raw_games(result)[0]["gamePk"] == 1


def test_fetch_schedule_raw_omits_game_type_filter_when_not_supplied(monkeypatch):
    captured = {}

    def fake_get_json(url, params=None, timeout=30, retries=3):
        captured["params"] = params
        return _payload([])

    monkeypatch.setattr(
        "atlas.schedule.mlb_schedule_reference.get_json", fake_get_json
    )
    fetch_schedule_raw("2024-04-01", "2024-04-02")
    assert "gameType" not in captured["params"]
