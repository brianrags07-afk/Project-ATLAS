from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.player_presence_signals import (
    build_pregame_player_presence_signals,
    certify_pregame_player_presence_signals,
)


GAME_STARTS = {
    10: "2024-04-01T20:00:00Z",
    11: "2024-04-03T20:00:00Z",
    12: "2024-04-05T20:00:00Z",
}


def team_games() -> pd.DataFrame:
    rows = []
    for game_pk, start in GAME_STARTS.items():
        rows.extend(
            [
                {
                    "game_pk": game_pk,
                    "game_start_at": start,
                    "official_date": start[:10],
                    "season": 2024,
                    "team": "HME",
                    "team_id": 1,
                    "opponent": "AWY",
                    "opponent_team_id": 2,
                    "home_away": "HOME",
                },
                {
                    "game_pk": game_pk,
                    "game_start_at": start,
                    "official_date": start[:10],
                    "season": 2024,
                    "team": "AWY",
                    "team_id": 2,
                    "opponent": "HME",
                    "opponent_team_id": 1,
                    "home_away": "AWAY",
                },
            ]
        )
    return pd.DataFrame(rows)


def roster_snapshots() -> pd.DataFrame:
    rows = []
    for game_pk, start in GAME_STARTS.items():
        for team, player in (("HME", 1), ("AWY", 2)):
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_start_at": start,
                    "season": 2024,
                    "team": team,
                    "player_id": player,
                    "organization_member": True,
                    "active_roster": True,
                    "available": True,
                    "injury_status": None,
                    "roster_status": "active",
                    "last_event_id": f"opening-{team}-{player}",
                    "last_event_type": "opening_roster",
                    "last_event_at": "2024-03-28T00:00:00Z",
                    "last_source": "MLB",
                    "last_source_retrieved_at": "2026-07-22T00:00:00Z",
                    "last_knowledge_available_at": "2024-03-28T00:00:00Z",
                    "pregame_safe": True,
                }
            )
    return pd.DataFrame(rows)


def observed(
    *,
    player_id: int,
    team_id: int,
    team: str,
    game_pk: int,
    observed_at: str,
    known_at: str,
) -> dict:
    return {
        "player_id": player_id,
        "team_id": team_id,
        "team_abbreviation": team,
        "game_pk": game_pk,
        "atlas_season": 2024,
        "observed_at": observed_at,
        "knowledge_available_at": known_at,
        "evidence_type": "postgame_game_appearance",
        "roles_observed": "batter",
        "prospective_only": True,
        "retroactive_backfill_allowed": False,
    }


def observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            observed(
                player_id=1,
                team_id=1,
                team="HME",
                game_pk=10,
                observed_at=GAME_STARTS[10],
                known_at="2024-04-02T20:00:00Z",
            ),
            observed(
                player_id=3,
                team_id=1,
                team="HME",
                game_pk=10,
                observed_at=GAME_STARTS[10],
                known_at="2024-04-02T20:00:00Z",
            ),
            observed(
                player_id=1,
                team_id=2,
                team="AWY",
                game_pk=11,
                observed_at=GAME_STARTS[11],
                known_at="2024-04-04T20:00:00Z",
            ),
            observed(
                player_id=2,
                team_id=2,
                team="AWY",
                game_pk=10,
                observed_at=GAME_STARTS[10],
                known_at="2024-04-02T20:00:00Z",
            ),
        ]
    )


def build() -> pd.DataFrame:
    return build_pregame_player_presence_signals(
        roster_snapshots(), observations(), team_games(), season=2024
    )


def test_observation_only_player_appears_only_after_evidence_is_known():
    result = build()
    player = result.loc[result["player_id"].eq(3)]
    assert set(player["game_pk"]) == {11, 12}
    assert not player["roster_row_present"].any()
    assert set(player["presence_evidence_class"]) == {
        "OBSERVATION_ONLY_SAME_TEAM"
    }
    assert player["latest_observation_matches_team"].all()


def test_active_roster_and_prior_appearance_are_kept_separate():
    result = build()
    game_10 = result.loc[
        result["game_pk"].eq(10)
        & result["team"].eq("HME")
        & result["player_id"].eq(1)
    ].iloc[0]
    assert game_10["presence_evidence_class"] == "ACTIVE_ROSTER_ONLY"
    assert int(game_10["prior_team_observed_games"]) == 0

    game_11 = result.loc[
        result["game_pk"].eq(11)
        & result["team"].eq("HME")
        & result["player_id"].eq(1)
    ].iloc[0]
    assert (
        game_11["presence_evidence_class"]
        == "ACTIVE_ROSTER_AND_PRIOR_TEAM_APPEARANCE"
    )
    assert int(game_11["prior_team_observed_games"]) == 1
    assert int(game_11["team_observed_games_last_7d"]) == 1


def test_latest_other_team_observation_is_an_explicit_conflict():
    result = build()
    row = result.loc[
        result["game_pk"].eq(12)
        & result["team"].eq("HME")
        & result["player_id"].eq(1)
    ].iloc[0]
    assert row["latest_observation_other_team"]
    assert row["last_league_observation_team_id"] == 2
    assert (
        row["presence_evidence_class"]
        == "ACTIVE_ROSTER_CONFLICT_LAST_OBS_OTHER_TEAM"
    )


def test_no_published_lineup_is_invented_and_certification_passes():
    result = build()
    assert not result["published_lineup_confirmed"].any()
    assert not result["same_game_postgame_used"].any()
    assert not result["future_games_used"].any()
    report = certify_pregame_player_presence_signals(result, season=2024)
    assert report["verdict"] == "certified"
    assert report["published_lineup_rows"] == 0
    assert report["observation_only_rows"] == 3


def test_future_known_observation_is_rejected_by_certifier():
    result = build()
    unsafe = result.copy()
    unsafe.loc[unsafe.index[0], "last_team_observation_knowledge_available_at"] = (
        unsafe.loc[unsafe.index[0], "game_start_at"] + pd.Timedelta(hours=1)
    )
    report = certify_pregame_player_presence_signals(unsafe, season=2024)
    assert report["verdict"] == "quarantine_required"
    assert "after first pitch" in "; ".join(report["errors"])


def test_rejects_observation_that_allows_retroactive_backfill():
    evidence = observations()
    evidence["retroactive_backfill_allowed"] = True
    with pytest.raises(ValueError, match="retroactive backfill"):
        build_pregame_player_presence_signals(
            roster_snapshots(), evidence, team_games(), season=2024
        )
