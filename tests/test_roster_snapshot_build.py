from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.roster_snapshot_build import (
    build_regular_team_games,
    certify_pregame_roster_snapshots,
)


def schedule(**overrides) -> pd.DataFrame:
    row = {
        "game_pk": 10,
        "season": 2024,
        "game_date_utc": "2024-04-01T20:00:00Z",
        "official_date": "2024-04-01",
        "game_type_code": "R",
        "is_final": True,
        "home_team_id": 1,
        "away_team_id": 2,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2024, "team_id": 1, "abbreviation": "HME"},
            {"season": 2024, "team_id": 2, "abbreviation": "AWY"},
        ]
    )


def snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_pk": 10,
                "game_start_at": "2024-04-01T20:00:00Z",
                "season": 2024,
                "team": team,
                "player_id": player,
                "organization_member": True,
                "last_event_at": "2024-03-28T00:00:00Z",
                "last_knowledge_available_at": "2024-03-28T00:00:00Z",
                "pregame_safe": True,
            }
            for team, player in (("HME", 1), ("AWY", 2))
        ]
    )


def test_builds_exact_home_and_away_team_rows_from_numeric_ids():
    result = build_regular_team_games(schedule(), teams(), season=2024)
    assert len(result) == 2
    home = result.loc[result["home_away"].eq("HOME")].iloc[0]
    away = result.loc[result["home_away"].eq("AWAY")].iloc[0]
    assert (home["team"], home["team_id"], home["opponent"]) == ("HME", 1, "AWY")
    assert (away["team"], away["team_id"], away["opponent"]) == ("AWY", 2, "HME")
    assert str(home["game_start_at"]) == "2024-04-01 20:00:00+00:00"


def test_excludes_cancelled_and_non_regular_games():
    cancelled = schedule(is_final=False)
    with pytest.raises(ValueError, match="no completed regular-season games"):
        build_regular_team_games(cancelled, teams(), season=2024)
    spring = schedule(game_type_code="S")
    with pytest.raises(ValueError, match="no completed regular-season games"):
        build_regular_team_games(spring, teams(), season=2024)


def test_rejects_unknown_scheduled_team_and_duplicate_game():
    with pytest.raises(ValueError, match="missing from the team directory"):
        build_regular_team_games(
            schedule(home_team_id=3), teams(), season=2024
        )
    duplicate = pd.concat([schedule(), schedule()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate completed game_pk"):
        build_regular_team_games(duplicate, teams(), season=2024)


def test_certifies_complete_pregame_safe_snapshot_coverage():
    team_games = build_regular_team_games(schedule(), teams(), season=2024)
    report = certify_pregame_roster_snapshots(
        snapshots(), team_games, season=2024
    )
    assert report["verdict"] == "certified"
    assert report["games"] == 1
    assert report["team_games"] == 2
    assert report["missing_team_game_count"] == 0
    assert report["post_first_pitch_knowledge_rows"] == 0


def test_certifier_rejects_missing_team_game_and_future_knowledge():
    team_games = build_regular_team_games(schedule(), teams(), season=2024)
    unsafe = snapshots().iloc[[0]].copy()
    unsafe["last_knowledge_available_at"] = "2024-04-02T00:00:00Z"
    report = certify_pregame_roster_snapshots(unsafe, team_games, season=2024)
    assert report["verdict"] == "quarantine_required"
    assert report["missing_team_game_count"] == 1
    assert report["post_first_pitch_knowledge_rows"] == 1


def test_certifier_rejects_duplicate_player_rows():
    team_games = build_regular_team_games(schedule(), teams(), season=2024)
    duplicate = pd.concat([snapshots(), snapshots().iloc[[0]]], ignore_index=True)
    report = certify_pregame_roster_snapshots(
        duplicate, team_games, season=2024
    )
    assert "duplicate game/team/player" in "; ".join(report["errors"])
