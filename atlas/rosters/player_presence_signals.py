"""Pregame player-presence evidence without lineup or roster overclaims.

The output keeps three concepts separate:

* certified roster state at first pitch;
* player appearances known strictly before first pitch; and
* published lineup confirmation (reserved here, never inferred).

Prior appearances may add an observation-only candidate or identify a stale
roster conflict, but they never mutate the certified roster ledger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ENGINE_VERSION = "pregame_player_presence_v1"

SNAPSHOT_KEYS = ["game_pk", "team", "player_id"]

SNAPSHOT_REQUIRED = {
    "game_pk",
    "game_start_at",
    "season",
    "team",
    "player_id",
    "organization_member",
    "active_roster",
    "available",
    "injury_status",
    "roster_status",
    "last_event_id",
    "last_event_type",
    "last_event_at",
    "last_knowledge_available_at",
    "pregame_safe",
}

OBSERVATION_REQUIRED = {
    "player_id",
    "team_id",
    "team_abbreviation",
    "game_pk",
    "atlas_season",
    "observed_at",
    "knowledge_available_at",
    "evidence_type",
    "roles_observed",
    "prospective_only",
    "retroactive_backfill_allowed",
}

TEAM_GAME_REQUIRED = {
    "game_pk",
    "game_start_at",
    "official_date",
    "season",
    "team",
    "team_id",
    "opponent",
    "opponent_team_id",
    "home_away",
}

ROSTER_VALUE_COLUMNS = [
    "organization_member",
    "active_roster",
    "available",
    "injury_status",
    "roster_status",
    "last_event_id",
    "last_event_type",
    "last_event_at",
    "last_knowledge_available_at",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalize_inputs(
    snapshots: pd.DataFrame,
    observations: pd.DataFrame,
    team_games: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(snapshots, SNAPSHOT_REQUIRED, "roster snapshots")
    _require_columns(observations, OBSERVATION_REQUIRED, "player observations")
    _require_columns(team_games, TEAM_GAME_REQUIRED, "team games")
    if snapshots.empty or observations.empty or team_games.empty:
        raise ValueError("snapshots, observations, and team games must be nonempty")

    roster = snapshots.copy()
    roster["season"] = pd.to_numeric(roster["season"], errors="raise").astype("int64")
    roster["player_id"] = pd.to_numeric(
        roster["player_id"], errors="raise"
    ).astype("int64")
    roster["game_pk"] = pd.to_numeric(roster["game_pk"], errors="raise").astype("int64")
    roster["game_start_at"] = pd.to_datetime(
        roster["game_start_at"], utc=True, errors="coerce"
    )
    if not roster["season"].eq(int(season)).all():
        raise ValueError(f"roster snapshots contain rows outside season {season}")
    if roster["game_start_at"].isna().any():
        raise ValueError("roster snapshots contain an invalid game_start_at")
    if not roster["pregame_safe"].fillna(False).astype(bool).all():
        raise ValueError("roster snapshots contain a row not marked pregame safe")
    if roster.duplicated(SNAPSHOT_KEYS, keep=False).any():
        raise ValueError("roster snapshots contain duplicate game/team/player rows")

    evidence = observations.copy()
    for column in ("player_id", "team_id", "game_pk", "atlas_season"):
        evidence[column] = pd.to_numeric(evidence[column], errors="raise").astype("int64")
    evidence["observed_at"] = pd.to_datetime(
        evidence["observed_at"], utc=True, errors="coerce"
    )
    evidence["knowledge_available_at"] = pd.to_datetime(
        evidence["knowledge_available_at"], utc=True, errors="coerce"
    )
    if not evidence["atlas_season"].eq(int(season)).all():
        raise ValueError(f"player observations contain rows outside season {season}")
    if evidence[["observed_at", "knowledge_available_at"]].isna().any().any():
        raise ValueError("player observations contain an invalid timestamp")
    if evidence["knowledge_available_at"].lt(evidence["observed_at"]).any():
        raise ValueError("player observation knowledge predates the observation")
    if not evidence["prospective_only"].fillna(False).astype(bool).all():
        raise ValueError("player observations must be prospective only")
    if evidence["retroactive_backfill_allowed"].fillna(True).astype(bool).any():
        raise ValueError("player observations permit retroactive backfill")
    if evidence.duplicated(["game_pk", "team_id", "player_id"], keep=False).any():
        raise ValueError("player observations contain duplicate game/team/player rows")

    games = team_games.copy()
    for column in ("game_pk", "season", "team_id", "opponent_team_id"):
        games[column] = pd.to_numeric(games[column], errors="raise").astype("int64")
    games["game_start_at"] = pd.to_datetime(
        games["game_start_at"], utc=True, errors="coerce"
    )
    if not games["season"].eq(int(season)).all():
        raise ValueError(f"team games contain rows outside season {season}")
    if games["game_start_at"].isna().any():
        raise ValueError("team games contain an invalid game_start_at")
    if games.duplicated(["game_pk", "team"], keep=False).any():
        raise ValueError("team games contain duplicate game/team rows")

    roster_keys = set(map(tuple, roster[["game_pk", "team"]].drop_duplicates().to_numpy()))
    game_keys = set(map(tuple, games[["game_pk", "team"]].to_numpy()))
    if roster_keys != game_keys:
        raise ValueError("roster snapshot team-game coverage does not match team games")
    return roster, evidence, games


def _latest_asof(
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    by: list[str],
    prefix: str,
) -> pd.DataFrame:
    right_columns = [*by]
    if "team_id" not in by:
        right_columns.append("team_id")
    right_columns.extend(
        [
            "game_pk",
            "observed_at",
            "knowledge_available_at",
            "roles_observed",
            "evidence_type",
        ]
    )
    right = observations[right_columns].copy()
    rename = {
        "game_pk": f"{prefix}_game_pk",
        "observed_at": f"{prefix}_observed_at",
        "knowledge_available_at": f"{prefix}_knowledge_available_at",
        "roles_observed": f"{prefix}_roles_observed",
        "evidence_type": f"{prefix}_evidence_type",
    }
    if "team_id" not in by:
        rename["team_id"] = f"{prefix}_team_id"
    right = right.rename(columns=rename).sort_values(
        [f"{prefix}_knowledge_available_at", *by], kind="stable"
    )
    left = candidates.copy()
    left["_candidate_order"] = np.arange(len(left), dtype="int64")
    left = left.sort_values(["game_start_at", *by], kind="stable")
    merged = pd.merge_asof(
        left,
        right,
        left_on="game_start_at",
        right_on=f"{prefix}_knowledge_available_at",
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.sort_values("_candidate_order", kind="stable").drop(
        columns="_candidate_order"
    ).reset_index(drop=True)


def _prior_observation_counts(
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(index=candidates.index)
    for column in (
        "prior_team_observed_games",
        "team_observed_games_last_7d",
        "team_observed_games_last_14d",
        "team_observed_games_last_30d",
    ):
        output[column] = 0

    histories = {
        key: np.sort(
            group["knowledge_available_at"].astype("int64").to_numpy()
        )
        for key, group in observations.groupby(["team_id", "player_id"], sort=False)
    }
    day_ns = int(pd.Timedelta(days=1).value)
    for key, group in candidates.groupby(["team_id", "player_id"], sort=False):
        history = histories.get(key)
        if history is None or not len(history):
            continue
        starts = group["game_start_at"].astype("int64").to_numpy()
        end = np.searchsorted(history, starts, side="right")
        output.loc[group.index, "prior_team_observed_games"] = end
        for days in (7, 14, 30):
            begin = np.searchsorted(history, starts - days * day_ns, side="left")
            output.loc[group.index, f"team_observed_games_last_{days}d"] = end - begin
    return output.astype("int64")


def build_pregame_player_presence_signals(
    snapshots: pd.DataFrame,
    observations: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Combine roster state and strictly prior observations without mutation."""
    roster, evidence, games = _normalize_inputs(
        snapshots, observations, team_games, season
    )
    context_columns = [
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

    roster_rows = roster.merge(
        games[context_columns],
        on=["game_pk", "game_start_at", "season", "team"],
        how="left",
        validate="many_to_one",
    )
    if roster_rows["team_id"].isna().any():
        raise ValueError("roster rows failed to join official team-game identity")
    roster_rows["roster_row_present"] = True

    observed_grids = []
    for team_id, team_observations in evidence.groupby("team_id", sort=True):
        team_games_subset = games.loc[games["team_id"].eq(team_id), context_columns]
        if team_games_subset.empty:
            raise ValueError(f"observed team_id {team_id} is absent from team games")
        players = pd.DataFrame(
            {"player_id": sorted(team_observations["player_id"].unique())}
        )
        observed_grids.append(
            team_games_subset.assign(_join_key=1).merge(
                players.assign(_join_key=1), on="_join_key", how="inner"
            ).drop(columns="_join_key")
        )
    observation_candidates = pd.concat(observed_grids, ignore_index=True)

    candidates = pd.concat(
        [
            roster_rows[context_columns + ["player_id"]],
            observation_candidates[context_columns + ["player_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["game_pk", "team", "player_id"], keep="first")
    candidates = candidates.merge(
        roster_rows[SNAPSHOT_KEYS + ROSTER_VALUE_COLUMNS + ["roster_row_present"]],
        on=SNAPSHOT_KEYS,
        how="left",
        validate="one_to_one",
    )
    candidates["roster_row_present"] = candidates["roster_row_present"].eq(True)

    candidates = _latest_asof(
        candidates,
        evidence,
        by=["team_id", "player_id"],
        prefix="last_team_observation",
    )
    candidates = candidates.loc[
        candidates["roster_row_present"]
        | candidates["last_team_observation_knowledge_available_at"].notna()
    ].reset_index(drop=True)

    candidates = _latest_asof(
        candidates,
        evidence,
        by=["player_id"],
        prefix="last_league_observation",
    )
    count_frame = _prior_observation_counts(candidates, evidence)
    for column in count_frame.columns:
        candidates[column] = count_frame[column]

    candidates["prior_team_appearance_known"] = candidates[
        "last_team_observation_knowledge_available_at"
    ].notna()
    candidates["latest_observation_matches_team"] = (
        candidates["last_league_observation_team_id"].notna()
        & candidates["last_league_observation_team_id"].eq(candidates["team_id"])
    )
    candidates["latest_observation_other_team"] = (
        candidates["last_league_observation_team_id"].notna()
        & candidates["last_league_observation_team_id"].ne(candidates["team_id"])
    )
    candidates["days_since_last_team_observation_known"] = (
        candidates["game_start_at"]
        - candidates["last_team_observation_knowledge_available_at"]
    ).dt.total_seconds() / 86400.0
    candidates["active_roster_known_true"] = candidates["active_roster"].eq(True)
    candidates["available_known_true"] = candidates["available"].eq(True)

    active = candidates["active_roster_known_true"]
    member = candidates["organization_member"].eq(True)
    same_team = candidates["latest_observation_matches_team"]
    other_team = candidates["latest_observation_other_team"]
    candidates["presence_evidence_class"] = np.select(
        [
            active & same_team,
            active & other_team,
            active,
            member & same_team,
            member & other_team,
            member,
            ~candidates["roster_row_present"] & other_team,
            same_team,
        ],
        [
            "ACTIVE_ROSTER_AND_PRIOR_TEAM_APPEARANCE",
            "ACTIVE_ROSTER_CONFLICT_LAST_OBS_OTHER_TEAM",
            "ACTIVE_ROSTER_ONLY",
            "ORG_MEMBER_AND_PRIOR_TEAM_APPEARANCE",
            "ORG_MEMBER_CONFLICT_LAST_OBS_OTHER_TEAM",
            "ORG_MEMBER_ONLY",
            "OBSERVATION_ONLY_CONFLICT_LAST_OBS_OTHER_TEAM",
            "OBSERVATION_ONLY_SAME_TEAM",
        ],
        default="INSUFFICIENT_EVIDENCE",
    )

    candidates["published_lineup_confirmed"] = False
    candidates["published_lineup_known_at"] = pd.Series(
        pd.NaT, index=candidates.index, dtype="datetime64[ns, UTC]"
    )
    candidates["pregame_safe"] = True
    candidates["same_game_postgame_used"] = False
    candidates["future_games_used"] = False
    candidates["roster_ledger_mutated"] = False
    candidates["presence_signal_version"] = ENGINE_VERSION

    output = candidates.sort_values(
        ["game_start_at", "game_pk", "team", "player_id"], kind="stable"
    ).reset_index(drop=True)
    report = certify_pregame_player_presence_signals(output, season=season)
    if report["verdict"] != "certified":
        raise ValueError(
            "pregame player-presence signals failed certification: "
            + "; ".join(report["errors"])
        )
    return output


def certify_pregame_player_presence_signals(
    signals: pd.DataFrame,
    *,
    season: int,
) -> dict[str, Any]:
    required = {
        "game_pk",
        "game_start_at",
        "season",
        "team",
        "team_id",
        "player_id",
        "roster_row_present",
        "last_team_observation_knowledge_available_at",
        "last_league_observation_knowledge_available_at",
        "presence_evidence_class",
        "published_lineup_confirmed",
        "pregame_safe",
        "same_game_postgame_used",
        "future_games_used",
        "roster_ledger_mutated",
    }
    missing = sorted(required.difference(signals.columns))
    if missing:
        return {
            "verdict": "not_ready",
            "rows": int(len(signals)),
            "errors": [f"presence signals missing columns: {missing}"],
        }
    errors: list[str] = []
    if signals.empty:
        errors.append("pregame player-presence signals are empty")
    if signals.duplicated(SNAPSHOT_KEYS, keep=False).any():
        errors.append("duplicate game/team/player presence signals detected")
    if not signals["season"].eq(int(season)).all():
        errors.append(f"presence signals contain rows outside season {season}")
    if not signals["pregame_safe"].fillna(False).astype(bool).all():
        errors.append("one or more presence signals are not pregame safe")
    for column in (
        "published_lineup_confirmed",
        "same_game_postgame_used",
        "future_games_used",
        "roster_ledger_mutated",
    ):
        if signals[column].fillna(False).astype(bool).any():
            errors.append(f"{column} must remain false in this historical evidence build")

    start = pd.to_datetime(signals["game_start_at"], utc=True, errors="coerce")
    for column in (
        "last_team_observation_knowledge_available_at",
        "last_league_observation_knowledge_available_at",
    ):
        known = pd.to_datetime(signals[column], utc=True, errors="coerce")
        if known.gt(start).fillna(False).any():
            errors.append(f"{column} contains evidence known after first pitch")
    observation_only_same_team = signals["presence_evidence_class"].eq(
        "OBSERVATION_ONLY_SAME_TEAM"
    )
    observation_only_conflict = signals["presence_evidence_class"].eq(
        "OBSERVATION_ONLY_CONFLICT_LAST_OBS_OTHER_TEAM"
    )
    observation_only = ~signals["roster_row_present"].eq(True)
    if (
        (observation_only_same_team | observation_only_conflict)
        & signals["roster_row_present"]
    ).any():
        errors.append("observation-only classification contains a roster row")
    if (
        observation_only_same_team
        & ~signals["latest_observation_matches_team"].fillna(False).astype(bool)
    ).any():
        errors.append("observation-only classification lacks same-team evidence")
    if (
        observation_only_conflict
        & ~signals["latest_observation_other_team"].fillna(False).astype(bool)
    ).any():
        errors.append("observation-only conflict lacks other-team evidence")
    count_columns = [
        "prior_team_observed_games",
        "team_observed_games_last_7d",
        "team_observed_games_last_14d",
        "team_observed_games_last_30d",
    ]
    for column in count_columns:
        if signals[column].isna().any() or signals[column].lt(0).any():
            errors.append(f"{column} contains missing or negative values")
    if (
        signals["team_observed_games_last_7d"]
        .gt(signals["team_observed_games_last_14d"])
        .any()
        or signals["team_observed_games_last_14d"]
        .gt(signals["team_observed_games_last_30d"])
        .any()
        or signals["team_observed_games_last_30d"]
        .gt(signals["prior_team_observed_games"])
        .any()
    ):
        errors.append("prior observation windows are not monotonically nested")

    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "rows": int(len(signals)),
        "games": int(signals["game_pk"].nunique(dropna=True)),
        "team_games": int(
            signals[["game_pk", "team"]].drop_duplicates().shape[0]
        ),
        "players": int(signals["player_id"].nunique(dropna=True)),
        "teams": int(signals["team"].nunique(dropna=True)),
        "roster_rows": int(signals["roster_row_present"].sum()),
        "observation_only_rows": int(observation_only.sum()),
        "published_lineup_rows": int(signals["published_lineup_confirmed"].sum()),
        "post_first_pitch_observation_rows": 0,
        "future_games_used": False,
        "roster_ledger_mutated": False,
        "errors": errors,
    }
