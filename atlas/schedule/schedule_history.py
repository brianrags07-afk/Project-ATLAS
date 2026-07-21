"""Schedule-history preservation and reschedule/resumption auditing."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from atlas.schedule.mlb_schedule_reference import (
    SCHEDULE_ENDPOINT,
    extract_raw_games,
    normalize_game_row,
)

SCHEDULE_CHANGE_FIELDS = (
    "rescheduleDate",
    "rescheduleGameDate",
    "rescheduledFrom",
    "rescheduledFromDate",
    "resumeDate",
    "resumeGameDate",
    "resumedFrom",
    "resumedFromDate",
)

HISTORY_EXTRA_FIELDS = (
    "schedule_record_index",
    "schedule_history_content_hash",
) + SCHEDULE_CHANGE_FIELDS


def _history_hash(row: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"retrieved_at_utc", "schedule_record_index"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_schedule_history(
    raw_payloads: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    retrieved_at_utc: str,
    source_url: str = SCHEDULE_ENDPOINT,
) -> list[dict[str, Any]]:
    """Preserve every published schedule record before canonical deduplication."""
    payloads: Iterable[Mapping[str, Any]]
    if isinstance(raw_payloads, Mapping):
        payloads = [raw_payloads]
    else:
        payloads = raw_payloads

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for raw_game in extract_raw_games(payload):
            row = normalize_game_row(
                raw_game,
                retrieved_at_utc=retrieved_at_utc,
                source_url=source_url,
            )
            for field in SCHEDULE_CHANGE_FIELDS:
                row[field] = raw_game.get(field)
            row["schedule_record_index"] = len(rows)
            row["schedule_history_content_hash"] = _history_hash(row)
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("game_date_utc") or "",
            row.get("game_pk") if row.get("game_pk") is not None else float("inf"),
            row.get("schedule_record_index", 0),
        )
    )
    return rows


def _joined_dates(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    values = sorted(
        {
            str(row[field])
            for row in rows
            for field in fields
            if row.get(field) not in (None, "")
        }
    )
    return "|".join(values)


def build_schedule_change_audit(
    history_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one audit row per game with a reschedule or resume field."""
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in history_rows:
        if any(row.get(field) not in (None, "") for field in SCHEDULE_CHANGE_FIELDS):
            grouped[row.get("game_pk")].append(row)

    audit: list[dict[str, Any]] = []
    for game_pk, rows in grouped.items():
        final_rows = [row for row in rows if row.get("is_final")]
        representative = final_rows[-1] if final_rows else rows[-1]
        states = sorted({str(row.get("detailed_state")) for row in rows})
        audit.append(
            {
                "game_pk": game_pk,
                "season": representative.get("season"),
                "game_type_code": representative.get("game_type_code"),
                "away_team_id": representative.get("away_team_id"),
                "away_team_name": representative.get("away_team_name"),
                "home_team_id": representative.get("home_team_id"),
                "home_team_name": representative.get("home_team_name"),
                "original_scheduled_dates": _joined_dates(
                    rows, ("rescheduledFromDate", "resumedFromDate")
                ),
                "rescheduled_dates": _joined_dates(
                    rows, ("rescheduleGameDate", "resumeGameDate")
                ),
                "published_official_date": representative.get("official_date"),
                "was_postponed": any(
                    row.get("game_state_category") == "postponed" for row in rows
                ),
                "was_rescheduled": any(
                    row.get("rescheduledFromDate") or row.get("rescheduleGameDate")
                    for row in rows
                ),
                "was_suspended_or_resumed": any(
                    row.get("resumedFromDate") or row.get("resumeGameDate")
                    for row in rows
                ),
                "schedule_record_count": len(rows),
                "published_states": "|".join(states),
                "history_content_hashes": "|".join(
                    sorted(
                        {
                            str(row.get("schedule_history_content_hash"))
                            for row in rows
                            if row.get("schedule_history_content_hash")
                        }
                    )
                ),
            }
        )
    audit.sort(key=lambda row: (row.get("season") or "", row.get("game_pk") or 0))
    return audit


def schedule_history_metrics(
    history_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    postponed = [
        row for row in history_rows if row.get("game_state_category") == "postponed"
    ]
    return {
        "schedule_history_rows": len(history_rows),
        "unique_game_pks": len(
            {row.get("game_pk") for row in history_rows if row.get("game_pk") is not None}
        ),
        "postponed_rows": len(postponed),
        "unique_postponed_game_pks": len(
            {row.get("game_pk") for row in postponed if row.get("game_pk") is not None}
        ),
        "suspended_or_resumed_game_pks": sum(
            bool(row.get("was_suspended_or_resumed")) for row in audit_rows
        ),
        "schedule_affected_game_pks": len(audit_rows),
    }
