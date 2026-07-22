"""Season-isolated schedule adapter and roster-snapshot certification."""

from __future__ import annotations

from typing import Any

import pandas as pd


SCHEDULE_COLUMNS = {
    "game_pk",
    "season",
    "game_date_utc",
    "official_date",
    "game_type_code",
    "is_final",
    "home_team_id",
    "away_team_id",
}

TEAM_COLUMNS = {"season", "team_id", "abbreviation"}

TEAM_GAME_COLUMNS = [
    "game_pk",
    "game_start_at",
    "official_date",
    "season",
    "team",
    "team_id",
    "opponent",
    "opponent_team_id",
    "home_away",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _integer(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or (values % 1).ne(0).any():
        raise ValueError(f"{label} contains missing or non-integer values")
    return values.astype("int64")


def build_regular_team_games(
    schedule: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Return two official team rows for every completed regular-season game."""
    _require_columns(schedule, SCHEDULE_COLUMNS, "schedule")
    _require_columns(teams, TEAM_COLUMNS, "team directory")

    games = schedule.copy()
    games["game_pk"] = _integer(games["game_pk"], "schedule.game_pk")
    games["season"] = _integer(games["season"], "schedule.season")
    games["game_type_code"] = (
        games["game_type_code"].astype("string").str.upper().str.strip()
    )
    games = games.loc[
        games["season"].eq(int(season))
        & games["game_type_code"].eq("R")
        & games["is_final"].fillna(False).astype(bool)
    ].copy()
    if games.empty:
        raise ValueError(f"schedule has no completed regular-season games for {season}")
    if games["game_pk"].duplicated().any():
        raise ValueError("schedule contains duplicate completed game_pk values")
    games["game_start_at"] = pd.to_datetime(
        games["game_date_utc"], utc=True, errors="coerce"
    )
    if games["game_start_at"].isna().any():
        raise ValueError("schedule contains an invalid game_date_utc")
    for column in ("home_team_id", "away_team_id"):
        games[column] = _integer(games[column], f"schedule.{column}")
    if games["home_team_id"].eq(games["away_team_id"]).any():
        raise ValueError("schedule contains identical home and away team IDs")

    directory = teams.copy()
    directory["season"] = _integer(directory["season"], "team directory.season")
    directory = directory.loc[directory["season"].eq(int(season))].copy()
    directory["team_id"] = _integer(directory["team_id"], "team directory.team_id")
    directory["abbreviation"] = (
        directory["abbreviation"].astype("string").str.upper().str.strip()
    )
    if directory.empty or directory["team_id"].duplicated().any():
        raise ValueError("team directory must contain unique teams for the season")
    if directory["abbreviation"].isna().any():
        raise ValueError("team directory contains a missing abbreviation")
    lookup = dict(zip(directory["team_id"], directory["abbreviation"]))

    scheduled_ids = set(games["home_team_id"]).union(games["away_team_id"])
    missing_ids = sorted(scheduled_ids.difference(lookup))
    if missing_ids:
        raise ValueError(
            "scheduled team IDs are missing from the team directory: "
            f"{missing_ids}"
        )

    common = {
        "game_pk": games["game_pk"],
        "game_start_at": games["game_start_at"],
        "official_date": games["official_date"],
        "season": games["season"],
    }
    home = pd.DataFrame(
        {
            **common,
            "team_id": games["home_team_id"],
            "opponent_team_id": games["away_team_id"],
            "home_away": "HOME",
        }
    )
    away = pd.DataFrame(
        {
            **common,
            "team_id": games["away_team_id"],
            "opponent_team_id": games["home_team_id"],
            "home_away": "AWAY",
        }
    )
    output = pd.concat([home, away], ignore_index=True)
    output["team"] = output["team_id"].map(lookup)
    output["opponent"] = output["opponent_team_id"].map(lookup)
    output = output[TEAM_GAME_COLUMNS].sort_values(
        ["game_start_at", "game_pk", "home_away"], kind="stable"
    ).reset_index(drop=True)

    if len(output) != len(games) * 2:
        raise AssertionError("team-game adapter did not emit exactly two rows per game")
    if output.duplicated(["game_pk", "team"], keep=False).any():
        raise AssertionError("team-game adapter emitted duplicate game/team rows")
    counts = output.groupby("game_pk").size()
    if not counts.eq(2).all():
        raise AssertionError("one or more games do not have exactly two team rows")
    return output


def certify_pregame_roster_snapshots(
    snapshots: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    season: int,
) -> dict[str, Any]:
    """Certify coverage, uniqueness, and first-pitch chronology."""
    required = {
        "game_pk",
        "game_start_at",
        "season",
        "team",
        "player_id",
        "organization_member",
        "last_event_at",
        "last_knowledge_available_at",
        "pregame_safe",
    }
    errors: list[str] = []
    missing = sorted(required.difference(snapshots.columns))
    if missing:
        return {
            "verdict": "not_ready",
            "rows": int(len(snapshots)),
            "errors": [f"snapshots missing columns: {missing}"],
        }
    if snapshots.empty:
        errors.append("pregame roster snapshots are empty")
    if snapshots.duplicated(["game_pk", "team", "player_id"], keep=False).any():
        errors.append("duplicate game/team/player snapshot rows detected")
    if not snapshots["season"].eq(int(season)).all():
        errors.append(f"snapshot rows exist outside season {season}")
    if not snapshots["organization_member"].fillna(False).astype(bool).all():
        errors.append("snapshot contains a player not known as an organization member")
    if not snapshots["pregame_safe"].fillna(False).astype(bool).all():
        errors.append("snapshot contains a row not marked pregame safe")

    start = pd.to_datetime(snapshots["game_start_at"], utc=True, errors="coerce")
    effective = pd.to_datetime(snapshots["last_event_at"], utc=True, errors="coerce")
    known = pd.to_datetime(
        snapshots["last_knowledge_available_at"], utc=True, errors="coerce"
    )
    if start.isna().any() or effective.isna().any() or known.isna().any():
        errors.append("snapshot chronology contains a missing or invalid timestamp")
    if effective.gt(start).any():
        errors.append("snapshot uses an event effective after first pitch")
    if known.gt(start).any():
        errors.append("snapshot uses evidence known after first pitch")

    expected_keys = set(
        map(tuple, team_games[["game_pk", "team"]].itertuples(index=False, name=None))
    )
    actual_keys = set(
        map(tuple, snapshots[["game_pk", "team"]].itertuples(index=False, name=None))
    )
    missing_team_games = sorted(expected_keys.difference(actual_keys))
    unexpected_team_games = sorted(actual_keys.difference(expected_keys))
    if missing_team_games:
        errors.append(f"missing team-game snapshots: {missing_team_games[:20]}")
    if unexpected_team_games:
        errors.append(f"unexpected team-game snapshots: {unexpected_team_games[:20]}")

    sizes = snapshots.groupby(["game_pk", "team"], sort=False).size()
    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "rows": int(len(snapshots)),
        "games": int(snapshots["game_pk"].nunique(dropna=True)),
        "team_games": int(len(actual_keys)),
        "players": int(snapshots["player_id"].nunique(dropna=True)),
        "teams": int(snapshots["team"].nunique(dropna=True)),
        "minimum_players_per_team_game": int(sizes.min()) if len(sizes) else 0,
        "maximum_players_per_team_game": int(sizes.max()) if len(sizes) else 0,
        "mean_players_per_team_game": float(sizes.mean()) if len(sizes) else 0.0,
        "missing_team_game_count": int(len(missing_team_games)),
        "unexpected_team_game_count": int(len(unexpected_team_games)),
        "post_first_pitch_effective_rows": int(effective.gt(start).sum()),
        "post_first_pitch_knowledge_rows": int(known.gt(start).sum()),
        "errors": errors,
    }
