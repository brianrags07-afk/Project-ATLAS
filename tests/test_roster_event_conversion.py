from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.roster_event_conversion import directional_transaction_events, opening_roster_events


TEAMS = pd.DataFrame([{"team_id": 1, "abbreviation": "AAA"}, {"team_id": 2, "abbreviation": "BBB"}])


def transaction(**overrides):
    row = {
        "season": 2024, "transaction_id": 10, "player_id": 7,
        "requested_team_id": 1, "team_id": None,
        "from_team_id": 1, "to_team_id": None,
        "effective_date": "2024-04-01", "transaction_date": "2024-04-01",
        "type_code": "SC", "type_description": "Status Change",
        "source_retrieved_at": "2026-07-22T00:00:00Z",
        "source_record_sha256": "x",
    }
    row.update(overrides)
    return row


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
    row = transaction(
        transaction_id=9, from_team_id=1, to_team_id=2,
        effective_date="2024-07-30", transaction_date="2024-07-30",
        type_code="TR", type_description="Trade", source_record_sha256="a",
    )
    events, quarantine = directional_transaction_events(pd.DataFrame([row, row]), TEAMS)
    assert quarantine.empty and len(events) == 2
    assert set(events["event_type"]) == {"structured_transfer_in", "structured_transfer_out"}
    assert set(events["source_row_count"]) == {2}
    assert set(events["knowledge_available_at"].astype(str)) == {"2024-07-31 00:00:00+00:00"}


@pytest.mark.parametrize(
    ("code", "description", "side", "event_type", "member", "active", "available"),
    [
        ("CU", "Recalled", "to", "recalled", True, True, True),
        ("OPT", "Optioned", "from", "optioned", True, False, False),
        ("DES", "Designated for Assignment", "from", "designated_for_assignment", True, False, False),
        ("DFA", "Declared Free Agency", "from", "declared_free_agency", False, False, False),
        ("OUT", "Outrighted", "from", "outrighted", True, False, False),
        ("REL", "Released", "from", "released", False, False, False),
        ("RET", "Retired", "from", "retired", False, False, False),
        ("SFA", "Signed as Free Agent", "to", "signed_free_agent", True, None, None),
        ("SGN", "Signed", "to", "signed", True, None, None),
        ("SU", "Suspension", "from", "suspended", True, False, False),
    ],
)
def test_allowlisted_status_codes_update_only_bounded_state(
    code, description, side, event_type, member, active, available
):
    row = transaction(
        type_code=code, type_description=description,
        from_team_id=1 if side == "from" else 9001,
        to_team_id=1 if side == "to" else 9001,
    )
    events, quarantine = directional_transaction_events(pd.DataFrame([row, row]), TEAMS)
    assert quarantine.empty and len(events) == 1
    event = events.iloc[0]
    assert event["event_type"] == event_type
    assert bool(event["organization_member"]) is member
    assert (None if pd.isna(event["active_roster"]) else bool(event["active_roster"])) is active
    assert (None if pd.isna(event["available"]) else bool(event["available"])) is available
    assert event["injury_status"] is None
    assert event["source_row_count"] == 2
    assert str(event["knowledge_available_at"]) == "2024-04-02 00:00:00+00:00"


def test_generic_status_is_quarantined_not_parsed_from_prose():
    row = transaction(description="Placed on injured list")
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert events.empty and len(quarantine) == 1
    assert quarantine.iloc[0]["quarantine_reason"] == "type code not approved for roster state conversion"


def test_assignment_direction_does_not_imply_organization_change():
    row = transaction(
        transaction_id=11, from_team_id=1, to_team_id=2,
        type_code="ASG", type_description="Assigned", source_record_sha256="y",
    )
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert events.empty and len(quarantine) == 1
    assert quarantine.iloc[0]["quarantine_reason"] == "type code not approved for roster state conversion"


def test_code_description_mismatch_is_quarantined():
    row = transaction(type_code="REL", type_description="Recalled")
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert events.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "type description does not match approved status meaning"


def test_status_requires_team_scoped_source_club():
    row = transaction(
        type_code="CU", type_description="Recalled", requested_team_id=9001, team_id=None,
        from_team_id=1, to_team_id=2,
    )
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert events.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "status transaction missing team-scoped source club"


def test_backdated_effective_date_does_not_backdate_source_knowledge():
    row = transaction(
        type_code="CU", type_description="Recalled",
        from_team_id=9001, to_team_id=1,
        effective_date="2024-04-08", transaction_date="2024-04-10",
    )
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert quarantine.empty and len(events) == 1
    assert str(events.iloc[0]["effective_at"]) == "2024-04-09 00:00:00+00:00"
    assert str(events.iloc[0]["knowledge_available_at"]) == "2024-04-11 00:00:00+00:00"


def test_missing_transaction_posting_date_is_quarantined():
    row = transaction(
        type_code="CU", type_description="Recalled",
        from_team_id=9001, to_team_id=1, transaction_date=None,
    )
    events, quarantine = directional_transaction_events(pd.DataFrame([row]), TEAMS)
    assert events.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "transaction posting date unknown"
