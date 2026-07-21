from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.roster_timeline import (
    build_pregame_roster_snapshots,
    certify_roster_events,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "opening-1",
                "effective_at": "2024-03-27T12:00:00Z",
                "season": 2024,
                "team": "CIN",
                "player_id": 1,
                "event_type": "opening_roster",
                "source": "fixture",
                "source_retrieved_at": "2024-03-27T12:00:00Z",
                "organization_member": True,
                "active_roster": True,
                "available": True,
                "roster_status": "active",
            },
            {
                "event_id": "il-1",
                "effective_at": "2024-04-02T14:00:00Z",
                "season": 2024,
                "team": "CIN",
                "player_id": 1,
                "event_type": "injured_list",
                "source": "fixture",
                "source_retrieved_at": "2024-04-02T14:00:00Z",
                "active_roster": False,
                "available": False,
                "injury_status": "10-day IL",
                "roster_status": "injured_list",
            },
            {
                "event_id": "trade-out-1",
                "effective_at": "2024-07-30T18:00:00Z",
                "season": 2024,
                "team": "CIN",
                "player_id": 1,
                "event_type": "trade_out",
                "source": "fixture",
                "source_retrieved_at": "2024-07-30T18:00:00Z",
                "organization_member": False,
                "active_roster": False,
                "available": False,
                "roster_status": "traded",
            },
        ]
    )


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_pk": 10, "game_start_at": "2024-03-28T20:00:00Z", "season": 2024, "team": "CIN"},
            {"game_pk": 11, "game_start_at": "2024-04-03T20:00:00Z", "season": 2024, "team": "CIN"},
            {"game_pk": 12, "game_start_at": "2024-07-31T20:00:00Z", "season": 2024, "team": "CIN"},
        ]
    )


def test_certifies_complete_normalized_events():
    report = certify_roster_events(_events())
    assert report["verdict"] == "certified"
    assert report["unique_events"] == 3


def test_builds_chronological_pregame_snapshots_without_future_backfill():
    snapshots = build_pregame_roster_snapshots(_events(), _games())
    assert snapshots["game_pk"].tolist() == [10, 11]
    opening = snapshots.loc[snapshots["game_pk"] == 10].iloc[0]
    injured = snapshots.loc[snapshots["game_pk"] == 11].iloc[0]
    assert bool(opening["active_roster"]) is True
    assert bool(opening["available"]) is True
    assert bool(injured["active_roster"]) is False
    assert injured["injury_status"] == "10-day IL"
    assert snapshots["pregame_safe"].all()


def test_duplicate_events_require_quarantine():
    events = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)
    report = certify_roster_events(events)
    assert report["verdict"] == "quarantine_required"
    assert any("duplicate event_id" in error for error in report["errors"])


def test_first_player_event_must_establish_organization_membership():
    events = _events().iloc[[0]].drop(columns=["organization_member"])
    report = certify_roster_events(events)
    assert report["verdict"] == "quarantine_required"
    assert any("organization_member is required" in error for error in report["errors"])


def test_later_status_event_cannot_substitute_for_missing_opening_membership():
    events = _events().copy()
    events.loc[events["event_id"] == "opening-1", "organization_member"] = None
    report = certify_roster_events(events)
    assert report["verdict"] == "quarantine_required"
    assert any("first event" in error for error in report["errors"])


def test_missing_team_history_is_not_inferred_from_game_appearances():
    games = _games()
    games.loc[:, "team"] = "PIT"
    with pytest.raises(ValueError, match="no roster event history"):
        build_pregame_roster_snapshots(_events(), games)


def test_event_retrieved_after_first_pitch_is_deferred_until_later_game():
    events = _events().iloc[[0]].copy()
    delayed = events.iloc[0].copy()
    delayed["event_id"] = "delayed-il"
    delayed["event_type"] = "injured_list"
    delayed["effective_at"] = "2024-04-02T14:00:00Z"
    delayed["source_retrieved_at"] = "2024-04-03T22:00:00Z"
    delayed["active_roster"] = False
    delayed["available"] = False
    delayed["injury_status"] = "10-day IL"
    delayed["roster_status"] = "injured_list"
    events = pd.concat([events, delayed.to_frame().T], ignore_index=True)
    games = pd.DataFrame(
        [
            {"game_pk": 20, "game_start_at": "2024-04-03T20:00:00Z", "season": 2024, "team": "CIN"},
            {"game_pk": 21, "game_start_at": "2024-04-04T20:00:00Z", "season": 2024, "team": "CIN"},
        ]
    )

    snapshots = build_pregame_roster_snapshots(events, games)

    before_source = snapshots.loc[snapshots["game_pk"] == 20].iloc[0]
    after_source = snapshots.loc[snapshots["game_pk"] == 21].iloc[0]
    assert bool(before_source["active_roster"]) is True
    assert before_source["last_event_id"] == "opening-1"
    assert bool(after_source["active_roster"]) is False
    assert after_source["last_event_id"] == "delayed-il"
    assert snapshots["pregame_safe"].all()


def test_game_before_known_opening_roster_fails_with_explicit_history_error():
    games = pd.DataFrame(
        [{"game_pk": 30, "game_start_at": "2024-03-26T20:00:00Z", "season": 2024, "team": "CIN"}]
    )
    with pytest.raises(ValueError, match="no known organization members"):
        build_pregame_roster_snapshots(_events(), games)


def test_empty_game_request_returns_stable_empty_schema():
    games = pd.DataFrame(columns=["game_pk", "game_start_at", "season", "team"])
    snapshots = build_pregame_roster_snapshots(_events(), games)
    assert snapshots.empty
    assert {"game_pk", "player_id", "pregame_safe"}.issubset(snapshots.columns)
