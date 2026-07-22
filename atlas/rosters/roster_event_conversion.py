"""Conservative conversion of official MLB source facts into roster events."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


EVENT_COLUMNS = [
    "event_id", "effective_at", "knowledge_available_at", "season", "team",
    "team_id", "player_id", "event_type", "source", "source_retrieved_at",
    "organization_member", "active_roster", "available", "injury_status",
    "roster_status", "source_row_count", "source_record_sha256s",
    "knowledge_time_method", "source_type_code", "source_type_description",
]

ORGANIZATION_CHANGE_CODES = {"TR", "CLW"}

# Only codes whose official type description establishes a bounded state change
# are admitted. A code/description mismatch is quarantined rather than guessed.
STATUS_CODE_RULES = {
    "CU": {
        "description": "Recalled", "event_type": "recalled",
        "organization_member": True, "active_roster": True,
        "available": True, "roster_status": "active", "team_side": "to",
    },
    "OPT": {
        "description": "Optioned", "event_type": "optioned",
        "organization_member": True, "active_roster": False,
        "available": False, "roster_status": "optioned", "team_side": "from",
    },
    "DES": {
        "description": "Designated for Assignment",
        "event_type": "designated_for_assignment",
        "organization_member": True, "active_roster": False,
        "available": False, "roster_status": "designated_for_assignment",
        "team_side": "from",
    },
    "DFA": {
        "description": "Declared Free Agency",
        "event_type": "declared_free_agency",
        "organization_member": False, "active_roster": False,
        "available": False, "roster_status": "free_agent", "team_side": "from",
    },
    "OUT": {
        "description": "Outrighted", "event_type": "outrighted",
        "organization_member": True, "active_roster": False,
        "available": False, "roster_status": "outrighted", "team_side": "from",
    },
    "REL": {
        "description": "Released", "event_type": "released",
        "organization_member": False, "active_roster": False,
        "available": False, "roster_status": "released", "team_side": "from",
    },
    "RET": {
        "description": "Retired", "event_type": "retired",
        "organization_member": False, "active_roster": False,
        "available": False, "roster_status": "retired", "team_side": "from",
    },
    "SFA": {
        "description": "Signed as Free Agent", "event_type": "signed_free_agent",
        "organization_member": True, "active_roster": None,
        "available": None, "roster_status": "signed", "team_side": "to",
    },
    "SGN": {
        "description": "Signed", "event_type": "signed",
        "organization_member": True, "active_roster": None,
        "available": None, "roster_status": "signed", "team_side": "to",
    },
    "SU": {
        "description": "Suspension", "event_type": "suspended",
        "organization_member": True, "active_roster": False,
        "available": False, "roster_status": "suspended", "team_side": "from",
    },
}


def _midnight_after(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("source date is missing or invalid")
    return stamp.tz_localize("UTC") + pd.Timedelta(days=1)


def _event_id(parts: list[Any]) -> str:
    value = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(value.encode()).hexdigest()


def _team_lookup(teams: pd.DataFrame) -> dict[int, str]:
    required = {"team_id", "abbreviation"}
    if not required.issubset(teams.columns):
        raise ValueError(f"teams missing columns: {sorted(required-set(teams.columns))}")
    if teams["team_id"].isna().any() or teams["abbreviation"].isna().any():
        raise ValueError("team identities must be complete")
    if teams["team_id"].duplicated().any():
        raise ValueError("team_id must be unique")
    return dict(zip(teams["team_id"].astype(int), teams["abbreviation"]))


def _status_team_id(row: dict[str, Any], lookup: dict[int, str], side: str) -> int | None:
    preferred = row.get(f"{side}_team_id")
    fallback_side = "from" if side == "to" else "to"
    fallback = row.get(f"{fallback_side}_team_id")
    valid = []
    for value in (preferred, fallback):
        if pd.notna(value) and int(value) in lookup and int(value) not in valid:
            valid.append(int(value))
    # A status code cannot safely choose between two different MLB clubs.
    return valid[0] if len(valid) == 1 else None


def opening_roster_events(rosters: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one opening membership event per known player/team baseline."""
    lookup = _team_lookup(teams)
    required = {"season", "team_id", "as_of_date", "roster_type", "player_id",
                "player_identity_known", "source", "source_retrieved_at", "source_record_sha256"}
    missing = required-set(rosters.columns)
    if missing:
        raise ValueError(f"rosters missing columns: {sorted(missing)}")
    quarantine = rosters.loc[~rosters["player_identity_known"].fillna(False)].copy()
    quarantine["quarantine_reason"] = "opening roster player identity unknown"
    known = rosters.loc[rosters["player_identity_known"].fillna(False)].copy()
    records = []
    for (season, team_id, player_id, as_of_date), group in known.groupby(
        ["season", "team_id", "player_id", "as_of_date"], sort=True
    ):
        team_id = int(team_id)
        if team_id not in lookup:
            raise ValueError(f"unknown team_id in roster source: {team_id}")
        types = set(group["roster_type"])
        active = "active" in types
        hashes = sorted(set(group["source_record_sha256"].astype(str)))
        available_at = _midnight_after(as_of_date)
        records.append({
            "event_id": _event_id(["opening", season, team_id, player_id, as_of_date]),
            "effective_at": available_at, "knowledge_available_at": available_at,
            "season": int(season), "team": lookup[team_id], "team_id": team_id,
            "player_id": int(player_id), "event_type": "opening_roster",
            "source": "MLB Stats API roster snapshot",
            "source_retrieved_at": pd.to_datetime(group["source_retrieved_at"], utc=True).max(),
            "organization_member": True, "active_roster": active,
            "available": True if active else None, "injury_status": None,
            "roster_status": "active" if active else "40Man",
            "source_row_count": int(len(group)),
            "source_record_sha256s": json.dumps(hashes),
            "knowledge_time_method": "prior_day_snapshot_available_next_midnight_utc",
            "source_type_code": None, "source_type_description": None,
        })
    return pd.DataFrame(records, columns=EVENT_COLUMNS), quarantine.reset_index(drop=True)


def directional_transaction_events(transactions: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert explicit transfers and allowlisted status facts; quarantine the rest."""
    lookup = _team_lookup(teams)
    required = {"season", "transaction_id", "player_id", "from_team_id", "to_team_id",
                "effective_date", "transaction_date", "type_code", "type_description",
                "source_retrieved_at", "source_record_sha256"}
    missing = required-set(transactions.columns)
    if missing:
        raise ValueError(f"transactions missing columns: {sorted(missing)}")
    transfer_candidates, status_candidates, quarantine_rows = [], [], []
    for row in transactions.to_dict("records"):
        if pd.isna(row.get("player_id")):
            quarantine_rows.append({**row, "quarantine_reason": "transaction player identity unknown"})
            continue
        source_date = row.get("effective_date")
        if pd.isna(source_date):
            source_date = row.get("transaction_date")
        if pd.isna(source_date):
            quarantine_rows.append({**row, "quarantine_reason": "transaction effective date unknown"})
            continue

        type_code = row.get("type_code")
        if type_code in ORGANIZATION_CHANGE_CODES:
            directions = []
            from_id, to_id = row.get("from_team_id"), row.get("to_team_id")
            if pd.notna(from_id) and int(from_id) in lookup and (pd.isna(to_id) or int(to_id) != int(from_id)):
                directions.append(("out", int(from_id), False))
            if pd.notna(to_id) and int(to_id) in lookup and (pd.isna(from_id) or int(from_id) != int(to_id)):
                directions.append(("in", int(to_id), True))
            if not directions:
                quarantine_rows.append({**row, "quarantine_reason": "no explicit inter-team direction"})
                continue
            for direction, team_id, member in directions:
                transfer_candidates.append({
                    **row, "direction": direction, "event_team_id": team_id,
                    "organization_member": member, "source_date": str(source_date),
                })
            continue

        rule = STATUS_CODE_RULES.get(type_code)
        if rule is None:
            quarantine_rows.append({
                **row, "quarantine_reason": "type code not approved for roster state conversion"
            })
            continue
        if row.get("type_description") != rule["description"]:
            quarantine_rows.append({
                **row, "quarantine_reason": "type description does not match approved status meaning"
            })
            continue
        team_id = _status_team_id(row, lookup, rule["team_side"])
        if team_id is None:
            quarantine_rows.append({
                **row, "quarantine_reason": "status transaction has missing or ambiguous MLB team"
            })
            continue
        status_candidates.append({
            **row, "event_team_id": team_id, "source_date": str(source_date), **rule,
        })

    records = []
    transfer_frame = pd.DataFrame(transfer_candidates)
    if not transfer_frame.empty:
        keys = ["season", "transaction_id", "player_id", "direction", "event_team_id", "source_date", "type_code"]
        for key, group in transfer_frame.groupby(keys, sort=True, dropna=False):
            season, transaction_id, player_id, direction, team_id, source_date, type_code = key
            hashes = sorted(set(group["source_record_sha256"].astype(str)))
            available_at = _midnight_after(source_date)
            records.append({
                "event_id": _event_id(["transaction", *key]),
                "effective_at": available_at, "knowledge_available_at": available_at,
                "season": int(season), "team": lookup[int(team_id)], "team_id": int(team_id),
                "player_id": int(player_id), "event_type": f"structured_transfer_{direction}",
                "source": "MLB Stats API transaction",
                "source_retrieved_at": pd.to_datetime(group["source_retrieved_at"], utc=True).max(),
                "organization_member": bool(group["organization_member"].iloc[0]),
                "active_roster": False if direction == "out" else None,
                "available": False if direction == "out" else None,
                "injury_status": None,
                "roster_status": "transferred_out" if direction == "out" else "transferred_in",
                "source_row_count": int(len(group)), "source_record_sha256s": json.dumps(hashes),
                "knowledge_time_method": "date_only_transaction_available_next_midnight_utc",
                "source_type_code": type_code,
                "source_type_description": group["type_description"].iloc[0],
            })

    status_frame = pd.DataFrame(status_candidates)
    if not status_frame.empty:
        keys = ["season", "transaction_id", "player_id", "event_team_id", "source_date", "type_code"]
        for key, group in status_frame.groupby(keys, sort=True, dropna=False):
            season, transaction_id, player_id, team_id, source_date, type_code = key
            first = group.iloc[0]
            hashes = sorted(set(group["source_record_sha256"].astype(str)))
            available_at = _midnight_after(source_date)
            records.append({
                "event_id": _event_id(["status", *key]),
                "effective_at": available_at, "knowledge_available_at": available_at,
                "season": int(season), "team": lookup[int(team_id)], "team_id": int(team_id),
                "player_id": int(player_id), "event_type": first["event_type"],
                "source": "MLB Stats API transaction",
                "source_retrieved_at": pd.to_datetime(group["source_retrieved_at"], utc=True).max(),
                "organization_member": first["organization_member"],
                "active_roster": first["active_roster"], "available": first["available"],
                "injury_status": None, "roster_status": first["roster_status"],
                "source_row_count": int(len(group)), "source_record_sha256s": json.dumps(hashes),
                "knowledge_time_method": "date_only_transaction_available_next_midnight_utc",
                "source_type_code": type_code,
                "source_type_description": first["type_description"],
            })
    return pd.DataFrame(records, columns=EVENT_COLUMNS), pd.DataFrame(quarantine_rows).reset_index(drop=True)
