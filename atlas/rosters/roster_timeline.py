"""Pregame-safe roster and player-team timeline construction.

This module is intentionally source-agnostic. Source adapters must first emit
the canonical event columns below. The builder never infers a transaction,
injury, or roster state from a player's later game appearance.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "effective_at",
    "season",
    "team",
    "player_id",
    "event_type",
    "source",
    "source_retrieved_at",
}

STATE_COLUMNS = (
    "organization_member",
    "active_roster",
    "available",
    "injury_status",
    "roster_status",
)

REQUIRED_GAME_COLUMNS = {
    "game_pk",
    "game_start_at",
    "season",
    "team",
}


def _missing(columns: Iterable[str], required: set[str]) -> list[str]:
    return sorted(required.difference(columns))


def certify_roster_events(events: pd.DataFrame) -> dict[str, Any]:
    """Return a deterministic readiness verdict for normalized roster events."""
    errors: list[str] = []
    missing = _missing(events.columns, REQUIRED_EVENT_COLUMNS)
    if missing:
        return {
            "verdict": "not_ready",
            "rows": int(len(events)),
            "errors": [f"missing required columns: {missing}"],
        }

    normalized = events.copy()
    normalized["effective_at"] = pd.to_datetime(
        normalized["effective_at"], utc=True, errors="coerce"
    )
    normalized["source_retrieved_at"] = pd.to_datetime(
        normalized["source_retrieved_at"], utc=True, errors="coerce"
    )

    if normalized["event_id"].isna().any():
        errors.append("event_id contains null values")
    duplicate_ids = normalized.loc[
        normalized["event_id"].duplicated(keep=False), "event_id"
    ].dropna()
    if not duplicate_ids.empty:
        errors.append(
            "duplicate event_id values: "
            + ", ".join(sorted(duplicate_ids.astype(str).unique()))
        )
    for column in ("effective_at", "source_retrieved_at"):
        if normalized[column].isna().any():
            errors.append(f"{column} contains missing or invalid timestamps")
    for column in ("season", "team", "player_id", "event_type", "source"):
        if normalized[column].isna().any():
            errors.append(f"{column} contains null values")

    state_columns_present = [c for c in STATE_COLUMNS if c in normalized.columns]
    if not state_columns_present:
        errors.append("no roster state columns are present")
    elif normalized[state_columns_present].isna().all(axis=1).any():
        errors.append("one or more events change no roster state")

    future_sourced = normalized["source_retrieved_at"] < normalized["effective_at"]
    if future_sourced.any():
        errors.append(
            "source_retrieved_at precedes effective_at; source chronology is invalid"
        )

    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "rows": int(len(normalized)),
        "unique_events": int(normalized["event_id"].nunique(dropna=True)),
        "players": int(normalized["player_id"].nunique(dropna=True)),
        "teams": int(normalized["team"].nunique(dropna=True)),
        "seasons": sorted(
            int(value) for value in normalized["season"].dropna().unique()
        ),
        "errors": errors,
    }


def build_pregame_roster_snapshots(
    events: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Apply only known roster events at or before each scheduled first pitch.

    The output contains one row per game/team/player currently known to be an
    organization member. `last_event_at` and `last_source_retrieved_at` make
    the temporal lineage auditable. Empty or incomplete event histories are
    rejected rather than backfilled from future appearances.
    """
    report = certify_roster_events(events)
    if report["verdict"] != "certified":
        raise ValueError("roster events are not certified: " + "; ".join(report["errors"]))

    missing_games = _missing(team_games.columns, REQUIRED_GAME_COLUMNS)
    if missing_games:
        raise ValueError(f"missing required game columns: {missing_games}")

    event_rows = events.copy()
    event_rows["effective_at"] = pd.to_datetime(event_rows["effective_at"], utc=True)
    event_rows["source_retrieved_at"] = pd.to_datetime(
        event_rows["source_retrieved_at"], utc=True
    )
    games = team_games.copy()
    games["game_start_at"] = pd.to_datetime(games["game_start_at"], utc=True)

    event_rows = event_rows.sort_values(
        ["season", "team", "effective_at", "source_retrieved_at", "event_id"],
        kind="stable",
    )
    games = games.sort_values(
        ["season", "team", "game_start_at", "game_pk"], kind="stable"
    )

    records: list[dict[str, Any]] = []
    grouped_events = {
        key: frame for key, frame in event_rows.groupby(["season", "team"], sort=False)
    }

    for (season, team), game_group in games.groupby(["season", "team"], sort=False):
        relevant = grouped_events.get((season, team))
        if relevant is None or relevant.empty:
            raise ValueError(f"no roster event history for season={season}, team={team}")

        state: dict[Any, dict[str, Any]] = {}
        event_list = list(relevant.to_dict("records"))
        cursor = 0

        for game in game_group.to_dict("records"):
            game_start = game["game_start_at"]
            while cursor < len(event_list) and event_list[cursor]["effective_at"] <= game_start:
                event = event_list[cursor]
                player_state = state.setdefault(
                    event["player_id"], {column: None for column in STATE_COLUMNS}
                )
                for column in STATE_COLUMNS:
                    if column in event and pd.notna(event[column]):
                        player_state[column] = event[column]
                player_state.update(
                    {
                        "last_event_id": event["event_id"],
                        "last_event_type": event["event_type"],
                        "last_event_at": event["effective_at"],
                        "last_source": event["source"],
                        "last_source_retrieved_at": event["source_retrieved_at"],
                    }
                )
                cursor += 1

            for player_id, player_state in state.items():
                if player_state.get("organization_member") is not True:
                    continue
                records.append(
                    {
                        "game_pk": game["game_pk"],
                        "game_start_at": game_start,
                        "season": season,
                        "team": team,
                        "player_id": player_id,
                        **player_state,
                        "pregame_safe": (
                            player_state["last_event_at"] <= game_start
                            and player_state["last_source_retrieved_at"] <= game_start
                        ),
                    }
                )

    output = pd.DataFrame.from_records(records)
    if not output.empty and not output["pregame_safe"].all():
        raise AssertionError("a roster snapshot contains post-first-pitch information")
    return output.sort_values(
        ["game_start_at", "game_pk", "team", "player_id"], kind="stable"
    ).reset_index(drop=True)
