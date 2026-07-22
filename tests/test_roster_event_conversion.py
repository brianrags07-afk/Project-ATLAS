from __future__ import annotations

import pandas as pd

from atlas.rosters.roster_event_conversion import directional_transaction_events, opening_roster_events


TEAMS = pd.DataFrame([{"team_id": 1, "abbreviation": "AAA"}, {"team_id": 2, "abbreviation": "BBB"}])


def test_opening_rows_collapse_semantically_and_retain_lineage():
    rosters = pd.DataFrame([
        {"season": 2024, "team_id": 1, "as_of_date": "2024-03-27", "roster_type": "40Man", "player_id": 7, "player_identity_known": True, "source": "MLB", "source_retrieved_at": "2026-07-22T00:00:00Z", "source_record_sha256": "a"},
        {"season": 2024, "team_id": 1, "as_of_date": "2024-03-27", "roster_type": "active", "player_id": 7, "player_identity_known": True, "source": "MLB", "source_retrieved_at": "2026-07-22T00:00:00Z", "source_record_sha256": "b"},
    ])
    events, quarantine = opening_roster_events(rosters, TEAMS)
    assert len(events) == 1 and quarantine.empty
    assert bool(events.iloc[0]["active_roster"]) is True
    assert events.iloc[0]["source_row_count"] == 2
    assert str(events.iloc[0]["knowledge_available_at"]) == "2024-03-28 00:00:00+00:00"
    assert events.iloc[0]["source_retrieved_at"].year == 2026


def test_unknown_opening_identity_is_quarantined():
    row = {"season": 2024, "team_id": 1, "as_of_date": "2024-03-27", "roster_type": "40Man", "player_id": None, "player_identity_known": False, "source": "MLB", "source_retrieved_at": "2026-07-22T00:00:00Z", "source_record_sha256": "a"}
    events, quarantine = opening_roster_events(pd.DataFrame([row]), TEAMS)
    assert events.empty and len(quarantine) == 1


def test_trade_creates_both_directions_next_day_and_deduplicates_semantically():
    row = {"season": 2024, "transaction_id": 9, "player_id": 7, "from_team_id": 1, "to_team_id": 2, "effective_date": "2024-07-30", "transaction_date": "2024-07-30", "source_retrieved_at": "2026-07-22T00:00:00Z", "source_record_sha256": "a"}
    events, quarantine = directional_transaction_events(pd.DataFrame([row, row]), TEAMS)
    assert quarantine.empty and len(events) == 2
    assert set(events["event_type"]) == {"structured_transfer_in", "structured_transfer_out"}
    assert set(events["source_row_count"]) == {2}
    assert set(events["knowledge_available_at"].astype(str)) == {"2024-07-31 00:00:00+00:00"}


def test_nondirectional_status_is_quarantined_not_parsed_from_prose():
    row = {"season": 2024, "transaction_id": 10, "player_id": 7, "from_team_id": None, "to_team_id": None, "effective_date": "2024-04-01", "transaction_date": "2024-04-01", "source_retrieved_at": "2026-07-22T00:00:00Z", "source_record_sha256": "x", "description": "Placed on injured list"}
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert events.empty and len(quarantine) == 1
    assert quarantine.iloc[0]["quarantine_reason"] == "no explicit inter-team direction"
