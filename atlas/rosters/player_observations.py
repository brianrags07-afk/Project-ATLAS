"""Timestamp-safe player observations derived from completed game events.

Historical pitch events prove that a player appeared for a club, but they do
not prove that the appearance was knowable before that game.  This adapter
therefore emits postgame evidence with a conservative 24-hour availability
delay.  The evidence may reconcile a quarantined transaction prospectively;
it must never rewrite an earlier roster snapshot or become a same-game
pregame feature.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from atlas.rosters.roster_reconciliation import certify_player_observations


OBSERVATION_POLICY_VERSION = "postgame_game_appearance_v1"
POSTGAME_AVAILABILITY_DELAY = pd.Timedelta(hours=24)

PITCH_COLUMNS = {
    "game_pk",
    "atlas_season",
    "game_type",
    "inning_topbot",
    "batter",
    "pitcher",
}

SCHEDULE_COLUMNS = {
    "game_pk",
    "season",
    "game_date_utc",
    "game_type_code",
    "is_final",
    "home_team_id",
    "away_team_id",
}

TEAM_COLUMNS = {
    "season",
    "team_id",
    "abbreviation",
}

OUTPUT_COLUMNS = [
    "player_id",
    "team_id",
    "team_abbreviation",
    "game_pk",
    "atlas_season",
    "observed_at",
    "knowledge_available_at",
    "evidence_type",
    "roles_observed",
    "source_pitch_rows",
    "source_game_start_at",
    "source",
    "observation_time_semantics",
    "knowledge_policy",
    "eligible_for_same_game_pregame",
    "prospective_only",
    "retroactive_backfill_allowed",
]


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _integer_series(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{label} contains missing or non-numeric values")
    if (values % 1).ne(0).any():
        raise ValueError(f"{label} contains non-integer values")
    return values.astype("int64")


def _certified_schedule(
    schedule: pd.DataFrame,
    season: int,
    game_pks: set[int],
) -> pd.DataFrame:
    _require_columns(schedule, SCHEDULE_COLUMNS, "schedule")
    normalized = schedule.copy()
    normalized["game_pk"] = _integer_series(
        normalized["game_pk"], "schedule.game_pk"
    )
    normalized["season"] = _integer_series(
        normalized["season"], "schedule.season"
    )
    normalized["game_type_code"] = (
        normalized["game_type_code"].astype("string").str.upper().str.strip()
    )
    regular = normalized.loc[
        normalized["season"].eq(int(season))
        & normalized["game_type_code"].eq("R")
    ].copy()
    if regular["game_pk"].duplicated().any():
        raise ValueError("schedule contains duplicate regular-season game_pk values")

    relevant = regular.loc[regular["game_pk"].isin(game_pks)].copy()
    missing_games = sorted(game_pks.difference(set(relevant["game_pk"])))
    if missing_games:
        raise ValueError(
            "pitch events are missing from the certified schedule: "
            f"{missing_games[:20]}"
        )
    if not relevant["is_final"].fillna(False).astype(bool).all():
        raise ValueError("pitch events include a schedule game that is not final")

    relevant["game_date_utc"] = pd.to_datetime(
        relevant["game_date_utc"], utc=True, errors="coerce"
    )
    if relevant["game_date_utc"].isna().any():
        raise ValueError("schedule.game_date_utc contains missing or invalid values")
    for column in ("home_team_id", "away_team_id"):
        relevant[column] = _integer_series(relevant[column], f"schedule.{column}")
    if relevant["home_team_id"].eq(relevant["away_team_id"]).any():
        raise ValueError("schedule contains a game with identical home and away teams")
    return relevant


def _team_directory(teams: pd.DataFrame, season: int) -> pd.DataFrame:
    _require_columns(teams, TEAM_COLUMNS, "team directory")
    normalized = teams.copy()
    normalized["season"] = _integer_series(
        normalized["season"], "team directory.season"
    )
    normalized = normalized.loc[normalized["season"].eq(int(season))].copy()
    normalized["team_id"] = _integer_series(
        normalized["team_id"], "team directory.team_id"
    )
    normalized["abbreviation"] = (
        normalized["abbreviation"].astype("string").str.upper().str.strip()
    )
    if normalized.empty:
        raise ValueError(f"team directory has no rows for season {season}")
    if normalized["team_id"].duplicated().any():
        raise ValueError("team directory contains duplicate team IDs")
    if normalized["abbreviation"].isna().any():
        raise ValueError("team directory contains a missing abbreviation")
    return normalized[["team_id", "abbreviation"]]


def build_postgame_player_observations(
    pitches: pd.DataFrame,
    schedule: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Build one conservative postgame observation per player/team/game.

    Team identity comes from the certified schedule's numeric home/away IDs,
    selected by ``inning_topbot``.  Source abbreviations are never used to
    infer identity.
    """
    _require_columns(pitches, PITCH_COLUMNS, "pitch events")
    if pitches.empty:
        raise ValueError("pitch events are empty")

    events = pitches.copy()
    events["game_pk"] = _integer_series(events["game_pk"], "pitch events.game_pk")
    events["atlas_season"] = _integer_series(
        events["atlas_season"], "pitch events.atlas_season"
    )
    if not events["atlas_season"].eq(int(season)).all():
        raise ValueError(f"pitch events contain rows outside season {season}")
    game_types = set(
        events["game_type"].dropna().astype(str).str.upper().str.strip().unique()
    )
    if game_types != {"R"}:
        raise ValueError(
            "player observations require regular-season-only pitch events; "
            f"found {sorted(game_types)}"
        )

    half = events["inning_topbot"].astype("string").str.strip()
    invalid_halves = sorted(set(half.dropna().unique()).difference({"Top", "Bot"}))
    if invalid_halves or half.isna().any():
        raise ValueError(
            "pitch events contain invalid inning_topbot values: "
            f"{invalid_halves}"
        )
    events["_top"] = half.eq("Top")

    game_pks = set(events["game_pk"].unique())
    games = _certified_schedule(schedule, season, game_pks)
    directory = _team_directory(teams, season)
    scheduled_team_ids = set(games["home_team_id"]).union(games["away_team_id"])
    missing_team_ids = sorted(scheduled_team_ids.difference(set(directory["team_id"])))
    if missing_team_ids:
        raise ValueError(
            "scheduled team IDs are missing from the official team directory: "
            f"{missing_team_ids}"
        )

    events = events.merge(
        games[
            [
                "game_pk",
                "game_date_utc",
                "home_team_id",
                "away_team_id",
            ]
        ],
        on="game_pk",
        how="left",
        validate="many_to_one",
    )

    batter = pd.DataFrame(
        {
            "game_pk": events["game_pk"],
            "atlas_season": events["atlas_season"],
            "player_id": events["batter"],
            "team_id": events["away_team_id"].where(
                events["_top"], events["home_team_id"]
            ),
            "source_game_start_at": events["game_date_utc"],
            "role": "batter",
        }
    )
    pitcher = pd.DataFrame(
        {
            "game_pk": events["game_pk"],
            "atlas_season": events["atlas_season"],
            "player_id": events["pitcher"],
            "team_id": events["home_team_id"].where(
                events["_top"], events["away_team_id"]
            ),
            "source_game_start_at": events["game_date_utc"],
            "role": "pitcher",
        }
    )
    appearances = pd.concat([batter, pitcher], ignore_index=True)
    appearances["player_id"] = _integer_series(
        appearances["player_id"], "pitch events player identity"
    )
    appearances["team_id"] = _integer_series(
        appearances["team_id"], "derived player team identity"
    )

    keys = ["player_id", "team_id", "game_pk", "atlas_season"]
    observations = (
        appearances.groupby(keys, as_index=False, sort=True)
        .agg(
            roles_observed=("role", lambda values: ",".join(sorted(set(values)))),
            source_pitch_rows=("role", "size"),
            source_game_start_at=("source_game_start_at", "first"),
        )
    )
    observations = observations.merge(
        directory.rename(columns={"abbreviation": "team_abbreviation"}),
        on="team_id",
        how="left",
        validate="many_to_one",
    )
    observations["observed_at"] = observations["source_game_start_at"]
    observations["knowledge_available_at"] = (
        observations["source_game_start_at"] + POSTGAME_AVAILABILITY_DELAY
    )
    observations["evidence_type"] = "postgame_game_appearance"
    observations["source"] = "ATLAS certified regular-season pitch events"
    observations["observation_time_semantics"] = "game_start_lower_bound"
    observations["knowledge_policy"] = OBSERVATION_POLICY_VERSION
    observations["eligible_for_same_game_pregame"] = False
    observations["prospective_only"] = True
    observations["retroactive_backfill_allowed"] = False
    observations = observations[OUTPUT_COLUMNS].sort_values(
        ["knowledge_available_at", "game_pk", "team_id", "player_id"],
        kind="stable",
    ).reset_index(drop=True)

    report = certify_postgame_player_observations(observations, season=season)
    if report["verdict"] != "certified":
        raise ValueError(
            "postgame player observations failed certification: "
            + "; ".join(report["errors"])
        )
    return observations


def certify_postgame_player_observations(
    observations: pd.DataFrame,
    *,
    season: int,
) -> dict[str, Any]:
    """Certify chronology and non-retroactivity of postgame observations."""
    errors: list[str] = []
    missing = sorted(set(OUTPUT_COLUMNS).difference(observations.columns))
    if missing:
        return {
            "verdict": "not_ready",
            "rows": int(len(observations)),
            "errors": [f"missing postgame observation columns: {missing}"],
        }

    base = certify_player_observations(observations)
    errors.extend(base["errors"])
    if observations.duplicated(
        subset=["player_id", "team_id", "game_pk"], keep=False
    ).any():
        errors.append("duplicate player/team/game observations detected")
    if not observations["atlas_season"].eq(int(season)).all():
        errors.append(f"observations contain rows outside season {season}")
    if not observations["evidence_type"].eq("postgame_game_appearance").all():
        errors.append("unexpected evidence_type in postgame observations")
    if not observations["knowledge_policy"].eq(OBSERVATION_POLICY_VERSION).all():
        errors.append("unexpected postgame knowledge policy")

    start = pd.to_datetime(observations["source_game_start_at"], utc=True, errors="coerce")
    observed = pd.to_datetime(observations["observed_at"], utc=True, errors="coerce")
    known = pd.to_datetime(
        observations["knowledge_available_at"], utc=True, errors="coerce"
    )
    if start.isna().any():
        errors.append("source_game_start_at contains null or invalid values")
    if not observed.eq(start).all():
        errors.append("observed_at must equal the conservative game-start lower bound")
    if not known.ge(start + POSTGAME_AVAILABILITY_DELAY).all():
        errors.append("postgame evidence became available less than 24 hours after start")
    if observations["eligible_for_same_game_pregame"].fillna(True).astype(bool).any():
        errors.append("postgame evidence cannot be eligible for same-game pregame use")
    if not observations["prospective_only"].fillna(False).astype(bool).all():
        errors.append("postgame evidence must be prospective only")
    if observations["retroactive_backfill_allowed"].fillna(True).astype(bool).any():
        errors.append("postgame evidence cannot permit retroactive backfill")

    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "rows": int(len(observations)),
        "players": int(observations["player_id"].nunique(dropna=True)),
        "teams": int(observations["team_id"].nunique(dropna=True)),
        "games": int(observations["game_pk"].nunique(dropna=True)),
        "errors": errors,
        "same_game_pregame_rows": int(
            observations["eligible_for_same_game_pregame"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "retroactive_backfill_rows": int(
            observations["retroactive_backfill_allowed"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
    }
