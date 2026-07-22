from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.mlb_roster_source import normalize_roster, normalize_teams, normalize_transactions


def test_normalizes_team_identity_and_venue():
    frame = normalize_teams({"teams": [{"id": 113, "name": "Cincinnati Reds", "abbreviation": "CIN", "active": True, "venue": {"id": 2602, "name": "Great American Ball Park"}}]}, 2024)
    assert frame.iloc[0]["abbreviation"] == "CIN"
    assert frame.iloc[0]["venue_id"] == 2602


def test_rejects_duplicate_team_ids():
    with pytest.raises(ValueError, match="unique"):
        normalize_teams({"teams": [{"id": 1}, {"id": 1}]}, 2024)


def test_roster_preserves_status_position_and_source_lineage():
    frame = normalize_roster({"roster": [{"person": {"id": 7, "fullName": "A Player"}, "position": {"code": "1", "name": "Pitcher"}, "status": {"code": "A", "description": "Active"}}]}, season=2024, team_id=113, as_of_date="2024-03-28", roster_type="active", retrieved_at_utc="2026-07-22T00:00:00Z")
    assert frame.iloc[0]["status_code"] == "A"
    assert frame.iloc[0]["source_record_sha256"]


def test_rejects_timezone_naive_retrieval_time():
    with pytest.raises(ValueError, match="timezone"):
        normalize_roster({"roster": [{"person": {"id": 7}}]}, season=2024, team_id=113, as_of_date="2024-03-28", roster_type="active", retrieved_at_utc="2026-07-22T00:00:00")


def test_transactions_preserve_both_teams_without_claiming_pregame_knowledge():
    frame = normalize_transactions({"transactions": [{"id": 9, "date": "2024-07-30", "effectiveDate": "2024-07-30", "person": {"id": 7}, "fromTeam": {"id": 112, "name": "Chicago Cubs"}, "toTeam": {"id": 113, "name": "Cincinnati Reds"}, "typeCode": "TR", "typeDesc": "Trade"}]}, season=2024, requested_team_id=113, retrieved_at_utc="2026-07-22T00:00:00Z")
    assert bool(frame.iloc[0]["pregame_time_known"]) is False
    assert "day" in frame.iloc[0]["source_time_precision"]
    assert frame.iloc[0]["from_team_id"] == 112
    assert frame.iloc[0]["to_team_id"] == 113


def test_repeated_transaction_ids_preserve_every_source_row_with_unique_keys():
    item = {"id": 9, "date": "2024-07-30", "person": {"id": 7}}
    frame = normalize_transactions({"transactions": [item, item]}, season=2024, requested_team_id=113)
    assert len(frame) == 2
    assert frame["transaction_id"].tolist() == [9, 9]
    assert frame["transaction_key"].is_unique
    assert frame["source_occurrence"].tolist() == [1, 2]


def test_empty_transaction_window_has_stable_schema():
    frame = normalize_transactions({"transactions": []}, season=2024, requested_team_id=113)
    assert frame.empty
    assert {"transaction_id", "pregame_time_known", "source_record_sha256"}.issubset(frame.columns)
