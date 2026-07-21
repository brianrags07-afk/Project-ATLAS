"""Checksum-bound certification for the saved 2024 MLB schedule snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from atlas.schedule.mlb_schedule_reference import extract_raw_games
from atlas.schedule.schedule_history import SCHEDULE_CHANGE_FIELDS

SAVED_2024_SCHEDULE_SHA256 = (
    "04232fe371adfd92402dbb8c4ed6ee9a97e6117cad27811d7df41381013bace7"
)

SAVED_2024_EXPECTATIONS: dict[str, Any] = {
    "raw_history_rows": 2469,
    "unique_game_pks": 2430,
    "postponed_rows": 36,
    "unique_postponed_game_pks": 34,
    "suspended_or_resumed_game_pks": 3,
    "schedule_affected_game_pks": 37,
    "non_regular_rows": 0,
}

SPECIAL_GAME_EXPECTATIONS: dict[int, dict[str, Any]] = {
    745180: {
        "record_count": 2,
        "resume_game_dates": ["2024-05-22"],
        "resumed_from_dates": ["2024-05-21"],
    },
    746755: {
        "record_count": 2,
        "resume_game_dates": ["2024-08-28"],
        "resumed_from_dates": ["2024-08-27"],
    },
    746942: {
        "record_count": 2,
        "resume_game_dates": ["2024-08-26"],
        "resumed_from_dates": ["2024-06-26"],
    },
    747090: {
        "record_count": 3,
        "postponed_rows": 2,
        "reschedule_game_dates": ["2024-07-24", "2024-09-09"],
        "rescheduled_from_dates": ["2024-07-23", "2024-07-24"],
    },
    747139: {
        "record_count": 3,
        "postponed_rows": 2,
        "reschedule_game_dates": ["2024-09-26", "2024-09-30"],
        "rescheduled_from_dates": ["2024-04-10", "2024-09-26"],
    },
}


def _values(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    return sorted(
        {
            str(row[field])
            for row in rows
            if row.get(field) not in (None, "")
        }
    )


def _is_schedule_affected(game: Mapping[str, Any]) -> bool:
    state = (game.get("status") or {}).get("detailedState")
    return state in {"Postponed", "Suspended", "Suspended: Rain"} or any(
        game.get(field) not in (None, "") for field in SCHEDULE_CHANGE_FIELDS
    )


def schedule_snapshot_metrics(payload: Mapping[str, Any]) -> dict[str, int]:
    games = extract_raw_games(payload)
    postponed = [
        game
        for game in games
        if (game.get("status") or {}).get("detailedState") == "Postponed"
    ]
    resumed_ids = {
        game.get("gamePk")
        for game in games
        if any(
            game.get(field) not in (None, "")
            for field in ("resumeDate", "resumeGameDate", "resumedFrom", "resumedFromDate")
        )
        and game.get("gamePk") is not None
    }
    affected_ids = {
        game.get("gamePk")
        for game in games
        if game.get("gamePk") is not None and _is_schedule_affected(game)
    }
    return {
        "raw_history_rows": len(games),
        "unique_game_pks": len(
            {game.get("gamePk") for game in games if game.get("gamePk") is not None}
        ),
        "postponed_rows": len(postponed),
        "unique_postponed_game_pks": len(
            {game.get("gamePk") for game in postponed if game.get("gamePk") is not None}
        ),
        "suspended_or_resumed_game_pks": len(resumed_ids),
        "schedule_affected_game_pks": len(affected_ids),
        "non_regular_rows": sum(game.get("gameType") != "R" for game in games),
    }


def special_game_observations(payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    games = extract_raw_games(payload)
    observations: dict[int, dict[str, Any]] = {}
    for game_pk in SPECIAL_GAME_EXPECTATIONS:
        rows = [game for game in games if game.get("gamePk") == game_pk]
        observations[game_pk] = {
            "record_count": len(rows),
            "postponed_rows": sum(
                (row.get("status") or {}).get("detailedState") == "Postponed"
                for row in rows
            ),
            "resume_game_dates": _values(rows, "resumeGameDate"),
            "resumed_from_dates": _values(rows, "resumedFromDate"),
            "reschedule_game_dates": _values(rows, "rescheduleGameDate"),
            "rescheduled_from_dates": _values(rows, "rescheduledFromDate"),
        }
    return observations


def certify_schedule_snapshot(
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    expected_sha256: str = SAVED_2024_SCHEDULE_SHA256,
    expected_metrics: Mapping[str, Any] = SAVED_2024_EXPECTATIONS,
    special_expectations: Mapping[int, Mapping[str, Any]] = SPECIAL_GAME_EXPECTATIONS,
) -> dict[str, Any]:
    """Certify one immutable schedule snapshot without mutating source data."""
    metrics = schedule_snapshot_metrics(payload)
    observations = special_game_observations(payload)
    errors: list[str] = []

    if source_sha256 != expected_sha256:
        errors.append(
            f"source checksum mismatch: expected {expected_sha256}, observed {source_sha256}"
        )

    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            errors.append(
                f"metric {key} mismatch: expected {expected}, observed {metrics.get(key)}"
            )

    for game_pk, expected in special_expectations.items():
        observed = observations.get(game_pk, {})
        for key, expected_value in expected.items():
            if observed.get(key) != expected_value:
                errors.append(
                    f"game_pk {game_pk} {key} mismatch: "
                    f"expected {expected_value}, observed {observed.get(key)}"
                )

    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "source_sha256": source_sha256,
        "expected_source_sha256": expected_sha256,
        "metrics": metrics,
        "expected_metrics": dict(expected_metrics),
        "special_games": {str(key): value for key, value in observations.items()},
        "errors": errors,
    }


def certify_schedule_file(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = source.read_bytes()
    payload = json.loads(data)
    report = certify_schedule_snapshot(
        payload,
        source_sha256=hashlib.sha256(data).hexdigest(),
    )
    report["source_path"] = str(source)
    report["source_size_bytes"] = len(data)
    return report
