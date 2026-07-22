"""Official MLB roster-source acquisition and lossless normalization."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping

import pandas as pd

from atlas.schedule.mlb_schedule_reference import MLB_API
from atlas.utils.api import get_json

SOURCE = "MLB Stats API"
TEAMS_URL = f"{MLB_API}/teams"
TRANSACTIONS_URL = f"{MLB_API}/transactions"


def _retrieved_at(value: str | None) -> str:
    stamp = pd.Timestamp(value or datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        raise ValueError("retrieved_at_utc must include a timezone")
    return stamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fetch_team_directory(season: int, *, fetch: Callable[..., Mapping[str, Any]] = get_json):
    return fetch(TEAMS_URL, params={"sportId": 1, "season": int(season)})


def fetch_roster(team_id: int, as_of_date: str | date, roster_type: str = "40Man", *, fetch=get_json):
    url = f"{MLB_API}/teams/{int(team_id)}/roster"
    return fetch(url, params={"rosterType": roster_type, "date": str(as_of_date)})


def fetch_transactions(team_id: int, start_date: str | date, end_date: str | date, *, fetch=get_json):
    return fetch(
        TRANSACTIONS_URL,
        params={"teamId": int(team_id), "startDate": str(start_date), "endDate": str(end_date)},
    )


def normalize_teams(payload: Mapping[str, Any], season: int) -> pd.DataFrame:
    rows = []
    for team in payload.get("teams", []):
        venue = team.get("venue") or {}
        rows.append({
            "season": int(season), "team_id": team.get("id"), "team_name": team.get("name"),
            "abbreviation": team.get("abbreviation"), "venue_id": venue.get("id"),
            "venue_name": venue.get("name"), "active": team.get("active"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty or frame["team_id"].isna().any() or frame["team_id"].duplicated().any():
        raise ValueError("team directory must be nonempty with unique, non-null team IDs")
    return frame.sort_values("team_id").reset_index(drop=True)


def normalize_roster(
    payload: Mapping[str, Any], *, season: int, team_id: int, as_of_date: str,
    roster_type: str, retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    retrieved = _retrieved_at(retrieved_at_utc)
    rows = []
    for entry in payload.get("roster", []):
        person, position, status = entry.get("person") or {}, entry.get("position") or {}, entry.get("status") or {}
        raw_hash = _hash(entry)
        rows.append({
            "season": int(season), "team_id": int(team_id), "as_of_date": str(as_of_date),
            "roster_type": roster_type, "player_id": person.get("id"), "player_name": person.get("fullName"),
            "position_code": position.get("code"), "position_name": position.get("name"),
            "status_code": status.get("code"), "status_description": status.get("description"),
            "jersey_number": entry.get("jerseyNumber"), "source": SOURCE,
            "source_retrieved_at": retrieved, "source_time_precision": "retrieval_exact; roster_date_day",
            "source_record_sha256": raw_hash,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"empty {roster_type} roster for team {team_id} on {as_of_date}")
    if frame["player_id"].isna().any() or frame.duplicated(["team_id", "as_of_date", "roster_type", "player_id"]).any():
        raise ValueError("roster contains null player IDs or duplicate player keys")
    return frame.sort_values("player_id").reset_index(drop=True)


def normalize_transactions(
    payload: Mapping[str, Any], *, season: int, requested_team_id: int,
    retrieved_at_utc: str | None = None,
) -> pd.DataFrame:
    retrieved = _retrieved_at(retrieved_at_utc)
    rows = []
    occurrences: dict[tuple[Any, str], int] = {}
    for item in payload.get("transactions", []):
        person = item.get("person") or {}
        from_team = item.get("fromTeam") or {}
        to_team = item.get("toTeam") or {}
        team = item.get("team") or {}
        record_hash = _hash(item)
        occurrence_key = (item.get("id"), record_hash)
        occurrence = occurrences.get(occurrence_key, 0) + 1
        occurrences[occurrence_key] = occurrence
        rows.append({
            "season": int(season), "requested_team_id": int(requested_team_id),
            "transaction_id": item.get("id"),
            "transaction_key": f"{item.get('id')}:{record_hash}:{occurrence}",
            "source_occurrence": occurrence, "player_id": person.get("id"),
            "player_name": person.get("fullName"),
            "team_id": team.get("id"), "team_name": team.get("name"),
            "from_team_id": from_team.get("id"), "from_team_name": from_team.get("name"),
            "to_team_id": to_team.get("id"), "to_team_name": to_team.get("name"),
            "transaction_date": item.get("date"), "effective_date": item.get("effectiveDate"),
            "resolution_date": item.get("resolutionDate"), "type_code": item.get("typeCode"),
            "type_description": item.get("typeDesc"), "description": item.get("description"),
            "source": SOURCE, "source_retrieved_at": retrieved,
            "source_time_precision": "retrieval_exact; transaction_dates_day",
            "pregame_time_known": False, "source_record_sha256": record_hash,
        })
    columns = ["season", "requested_team_id", "transaction_id", "transaction_key", "source_occurrence", "player_id", "player_name", "team_id", "team_name",
               "from_team_id", "from_team_name", "to_team_id", "to_team_name",
               "transaction_date", "effective_date", "resolution_date", "type_code", "type_description", "description",
               "source", "source_retrieved_at", "source_time_precision", "pregame_time_known", "source_record_sha256"]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty and frame["transaction_id"].isna().any():
        raise ValueError("transactions contain null transaction IDs")
    if not frame.empty and frame["transaction_key"].duplicated().any():
        raise AssertionError("transaction row keys are not unique")
    return frame.sort_values(["transaction_date", "transaction_id"], na_position="last").reset_index(drop=True)
