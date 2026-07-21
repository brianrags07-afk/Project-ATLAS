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

OUTPUT_COLUMNS = (
    "game_pk",
    "game_start_at",
    "season",
    "team",
    "player_id",
    *STATE_COLUMNS,
    "last_event_id",
    "last_event_type",
    "last_event_at",
    "last_source",
    "last_source_retrieved_at",
    "pregame_safe",
)


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

    if normalized.empty:
        errors.append("roster event ledger is empty")

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

    if "organization_member" not in normalized.columns:
        errors.append("organization_member is required to establish roster membership")
    else:
        ordered = normalized.sort_values(
            ["season", "team", "player_id", "effective_at", "source_retrieved_at", "event_id"],
            kind="stable",
        )
        first_events = ordered.groupby(
            ["season", "team", "player_id"], sort=False, dropna=False
        ).head(1)
        missing_opening_membership = first_events["organization_member"].isna()
        if missing_opening_membership.any():
            errors.append(
                "the first event for every season/team/player must establish "
                "organization_member"
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
    games["game_start_at"] = pd.to_datetime(
        games["game_start_at"], utc=True, errors="coerce"
    )
    if games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    for column in ("game_pk", "game_start_at", "season", "team"):
        if games[column].isna().any():
            raise ValueError(f"team_games {column} contains null or invalid values")

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

        for game in game_group.to_dict("records"):
            game_start = game["game_start_at"]
            # Reconstruct the as-known state at each first pitch. An event is
            # unusable until both the real-world change and its source record
            # were available. Rebuilding from the eligible subset also lets a
            # delayed source become usable for later games without leaking it
            # into earlier snapshots.
            eligible = relevant.loc[
                (relevant["effective_at"] <= game_start)
                & (relevant["source_retrieved_at"] <= game_start)
            ]
            state: dict[Any, dict[str, Any]] = {}
            for event in eligible.to_dict("records"):
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
            emitted_for_game = 0
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
                emitted_for_game += 1
            if emitted_for_game == 0:
                raise ValueError(
                    "no known organization members at first pitch for "
                    f"game_pk={game['game_pk']}, season={season}, team={team}"
                )

    output = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    if not output.empty and not output["pregame_safe"].all():
        raise AssertionError("a roster snapshot contains post-first-pitch information")
    return output.sort_values(
        ["game_start_at", "game_pk", "team", "player_id"], kind="stable"
    ).reset_index(drop=True)
