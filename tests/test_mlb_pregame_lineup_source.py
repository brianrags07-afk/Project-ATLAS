from __future__ import annotations

from copy import deepcopy

import pandas as pd

from atlas.lineups.mlb_pregame_lineup_source import (
    GAME_SNAPSHOT_COLUMNS,
    LINEUP_COLUMNS,
    STARTER_COLUMNS,
    certify_timecoded_pregame_bundle,
    format_mlb_timecode,
    normalize_timecoded_pregame_feed,
    partition_timecoded_pregame_bundle,
    prepare_completed_regular_games,
)


def schedule_row(**overrides):
    row = {
        "game_pk": 10,
        "season": 2024,
        "game_date_utc": "2024-04-01T20:00:00Z",
        "game_start_at": "2024-04-01T20:00:00Z",
        "official_date": "2024-04-01",
        "game_type_code": "R",
        "is_final": True,
        "home_team_id": 1,
        "home_team_name": "Home",
        "away_team_id": 2,
        "away_team_name": "Away",
    }
    row.update(overrides)
    return row


def feed() -> dict:
    players = {}
    away_players = {}
    home_players = {}
    for player_id in range(101, 110):
        key = f"ID{player_id}"
        players[key] = {
            "id": player_id,
            "fullName": f"Away {player_id}",
            "batSide": {"code": "R"},
            "pitchHand": {"code": "R"},
        }
        away_players[key] = {
            "person": {"id": player_id, "fullName": f"Away {player_id}"},
            "position": {"abbreviation": "OF"},
            "gameStatus": {"isSubstitute": False},
        }
    for player_id in range(201, 210):
        key = f"ID{player_id}"
        players[key] = {
            "id": player_id,
            "fullName": f"Home {player_id}",
            "batSide": {"code": "L"},
            "pitchHand": {"code": "R"},
        }
        home_players[key] = {
            "person": {"id": player_id, "fullName": f"Home {player_id}"},
            "position": {"abbreviation": "IF"},
            "gameStatus": {"isSubstitute": False},
        }
    return {
        "gamePk": 10,
        "metaData": {"timeStamp": "20240401_194000"},
        "gameData": {
            "status": {"statusCode": "PW", "detailedState": "Warmup"},
            "probablePitchers": {
                "away": {"id": 501, "fullName": "Away Starter"},
                "home": {"id": 601, "fullName": "Home Starter"},
            },
            "players": players,
        },
        "liveData": {
            "plays": {"allPlays": []},
            "boxscore": {
                "teams": {
                    "away": {
                        "battingOrder": list(range(101, 110)),
                        "players": away_players,
                    },
                    "home": {
                        "battingOrder": list(range(201, 210)),
                        "players": home_players,
                    },
                }
            },
        },
    }


def normalize(payload=None):
    return normalize_timecoded_pregame_feed(
        payload or feed(),
        schedule_row(),
        cutoff_minutes=15,
        archive_retrieved_at="2026-07-22T20:00:00Z",
    )


def frames(payload=None):
    snapshot, lineups, starters = normalize(payload)
    return (
        pd.DataFrame([snapshot], columns=GAME_SNAPSHOT_COLUMNS),
        pd.DataFrame(lineups, columns=LINEUP_COLUMNS),
        pd.DataFrame(starters, columns=STARTER_COLUMNS),
    )


def expected_games():
    return prepare_completed_regular_games([schedule_row()], season=2024)


def test_formats_exact_utc_timecode():
    assert format_mlb_timecode("2024-04-01T19:45:00Z") == "20240401_194500"


def test_schedule_adapter_keeps_only_completed_regular_games():
    games = prepare_completed_regular_games(
        [
            schedule_row(),
            schedule_row(game_pk=11, game_type_code="S"),
            schedule_row(game_pk=12, is_final=False),
        ],
        season=2024,
    )
    assert games["game_pk"].tolist() == [10]


def test_normalizes_official_pregame_lineups_and_probable_starters():
    game_snapshots, lineups, starters = frames()
    assert game_snapshots.iloc[0]["requested_timecode"] == "20240401_194500"
    assert str(game_snapshots.iloc[0]["source_snapshot_at"]) == (
        "2024-04-01 19:40:00+00:00"
    )
    assert bool(game_snapshots.iloc[0]["pregame_content_safe"])
    assert bool(game_snapshots.iloc[0]["published_lineups_confirmed"])
    assert not bool(game_snapshots.iloc[0]["actual_live_capture"])
    assert len(lineups) == 18
    assert lineups.groupby("team_id")["batting_order"].apply(list).tolist() == [
        list(range(1, 10)),
        list(range(1, 10)),
    ]
    assert set(starters["pitcher_id"]) == {501, 601}
    assert set(starters["confirmation_status"]) == {"probable"}
    report = certify_timecoded_pregame_bundle(
        game_snapshots, lineups, starters, expected_games(), season=2024
    )
    assert report["verdict"] == "certified"
    assert report["complete_team_lineups"] == 2
    assert report["outcome_fields_extracted"] == 0


def test_incomplete_published_lineup_is_preserved_as_missing_not_invented():
    payload = feed()
    payload["liveData"]["boxscore"]["teams"]["home"]["battingOrder"].pop()
    game_snapshots, lineups, starters = frames(payload)
    assert len(lineups) == 17
    assert not bool(game_snapshots.iloc[0]["published_lineups_confirmed"])
    home = lineups.loc[lineups["home_away"].eq("HOME")]
    assert not home["published_lineup_confirmed"].any()
    report = certify_timecoded_pregame_bundle(
        game_snapshots, lineups, starters, expected_games(), season=2024
    )
    assert report["verdict"] == "certified"
    assert report["incomplete_or_missing_team_lineups"] == 1


def test_source_snapshot_after_cutoff_is_quarantined():
    payload = feed()
    payload["metaData"]["timeStamp"] = "20240401_194600"
    game_snapshots, lineups, starters = frames(payload)
    assert not bool(game_snapshots.iloc[0]["pregame_content_safe"])
    report = certify_timecoded_pregame_bundle(
        game_snapshots, lineups, starters, expected_games(), season=2024
    )
    assert report["verdict"] == "quarantine_required"
    assert "not pregame safe" in "; ".join(report["errors"])


def test_feed_with_a_play_is_not_pregame_safe():
    payload = deepcopy(feed())
    payload["liveData"]["plays"]["allPlays"] = [
        {
            "result": {"eventType": "single"},
            "about": {"isComplete": True},
            "playEvents": [{"isPitch": True}],
        }
    ]
    game_snapshots, lineups, starters = frames(payload)
    assert bool(game_snapshots.iloc[0]["game_had_started_at_snapshot"])
    report = certify_timecoded_pregame_bundle(
        game_snapshots, lineups, starters, expected_games(), season=2024
    )
    assert report["verdict"] == "quarantine_required"
    assert "already contain game action" in "; ".join(report["errors"])


def test_unsafe_snapshot_becomes_documented_gap_not_model_input():
    payload = feed()
    payload["metaData"]["timeStamp"] = "20240401_194600"
    all_games, all_lineups, all_starters = frames(payload)
    (
        safe_games,
        safe_lineups,
        safe_starters,
        quarantined_games,
    ) = partition_timecoded_pregame_bundle(
        all_games,
        all_lineups,
        all_starters,
    )

    assert safe_games.empty
    assert safe_lineups.empty
    assert safe_starters.empty
    assert quarantined_games["game_pk"].tolist() == [10]
    report = certify_timecoded_pregame_bundle(
        all_games,
        safe_lineups,
        safe_starters,
        expected_games(),
        season=2024,
    )
    assert report["verdict"] == "certified_with_documented_gaps"
    assert report["pregame_safe_games"] == 0
    assert report["quarantined_games"] == 1
    assert report["quarantined_game_ids"] == [10]
    assert report["snapshot_after_cutoff_game_ids"] == [10]
    assert report["game_action_at_snapshot_game_ids"] == []
    assert report["incomplete_or_missing_team_lineups"] == 2
    assert report["missing_probable_starter_rows"] == 2
    assert report["errors"] == []
