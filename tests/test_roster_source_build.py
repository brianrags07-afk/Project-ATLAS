from __future__ import annotations

import pytest

from atlas.rosters.roster_source_build import build_roster_source_bundle, schedule_team_windows


SCHEDULE = [
    {"season": 2024, "game_type_code": "R", "game_date_utc": "2024-03-29T01:00:00Z", "official_date": "2024-03-28", "home_team_id": 1, "away_team_id": 2},
    {"season": 2024, "game_type_code": "R", "game_date_utc": "2024-09-30T01:00:00Z", "official_date": "2024-09-29", "home_team_id": 2, "away_team_id": 1},
]


def test_team_windows_follow_published_regular_schedule():
    windows = schedule_team_windows(SCHEDULE + [{"season": 2024, "game_type_code": "S", "game_date_utc": "2024-02-20T00:00:00Z", "official_date": "2024-02-19", "home_team_id": 1, "away_team_id": 2}], 2024)
    assert set(windows["first_game_date"]) == {"2024-03-28"}
    assert set(windows["last_game_date"]) == {"2024-09-29"}


def test_missing_official_date_is_rejected_instead_of_using_utc_date():
    row = dict(SCHEDULE[0])
    row["official_date"] = None
    with pytest.raises(ValueError, match="official_date"):
        schedule_team_windows([row], 2024)


def test_bundle_is_season_isolated_and_preserves_raw_payloads():
    def teams(season):
        return {"teams": [{"id": 1}, {"id": 2}]}
    def roster(team_id, date, roster_type):
        return {"roster": [{"person": {"id": team_id * 10 + (1 if roster_type == "active" else 2)}}]}
    def transactions(team_id, start, end):
        return {"transactions": []}
    bundle = build_roster_source_bundle(SCHEDULE, season=2024, retrieved_at_utc="2026-07-22T00:00:00Z", team_fetch=teams, roster_fetch=roster, transaction_fetch=transactions)
    assert len(bundle["team_windows"]) == 2
    assert len(bundle["rosters"]) == 4
    assert set(bundle["raw_payloads"]["clubs"]) == {"1", "2"}


def test_directory_must_exactly_match_scheduled_clubs():
    with pytest.raises(ValueError, match="mismatch"):
        build_roster_source_bundle(SCHEDULE, season=2024, team_fetch=lambda season: {"teams": [{"id": 1}]}, roster_fetch=lambda *args: {}, transaction_fetch=lambda *args: {})
