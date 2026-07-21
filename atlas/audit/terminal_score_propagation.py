"""Deterministic repair and validation of terminal-score-derived fields."""

from __future__ import annotations

from typing import Any

import pandas as pd


MASTER_REPAIRED_FIELDS = ("run_differential",)
TEAM_REPAIRED_FIELDS = ("runs_scored", "runs_allowed", "run_differential", "won")


class TerminalScoreRepairError(RuntimeError):
    """Raised when repair prerequisites or postconditions fail."""


def _changes_by_season(
    before: pd.Series, after: pd.Series, seasons: pd.Series
) -> dict[str, int]:
    changed = before.ne(after)
    return {
        str(int(season)): int((changed & seasons.eq(season)).sum())
        for season in sorted(seasons.dropna().unique())
        if int((changed & seasons.eq(season)).sum())
    }


def repair_terminal_score_propagation(
    master: pd.DataFrame,
    team_state: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return repaired copies; never mutate caller-owned dataframes."""
    required_master = {
        "game_pk",
        "atlas_season",
        "home_score",
        "away_score",
        "run_differential",
    }
    required_team = {
        "game_pk",
        "atlas_season",
        "home_away",
        "runs_scored",
        "runs_allowed",
        "run_differential",
        "won",
    }
    missing_master = sorted(required_master - set(master.columns))
    missing_team = sorted(required_team - set(team_state.columns))
    if missing_master or missing_team:
        raise TerminalScoreRepairError(
            f"missing repair columns: master={missing_master}, team={missing_team}"
        )
    if master["game_pk"].duplicated().any():
        raise TerminalScoreRepairError("master game_pk must be unique")

    repaired_master = master.copy(deep=True)
    repaired_team = team_state.copy(deep=True)
    before_master = master.copy(deep=True)
    before_team = team_state.copy(deep=True)

    repaired_master["run_differential"] = (
        repaired_master["home_score"] - repaired_master["away_score"]
    )
    lookup = repaired_master.set_index("game_pk")
    home_scores = repaired_team["game_pk"].map(lookup["home_score"])
    away_scores = repaired_team["game_pk"].map(lookup["away_score"])
    if home_scores.isna().any() or away_scores.isna().any():
        raise TerminalScoreRepairError("team row lacks a matching master game_pk")

    home = repaired_team["home_away"].astype(str).str.lower().eq("home")
    away = repaired_team["home_away"].astype(str).str.lower().eq("away")
    if not (home | away).all():
        raise TerminalScoreRepairError("home_away must contain only home or away")

    repaired_team.loc[home, "runs_scored"] = home_scores.loc[home]
    repaired_team.loc[home, "runs_allowed"] = away_scores.loc[home]
    repaired_team.loc[away, "runs_scored"] = away_scores.loc[away]
    repaired_team.loc[away, "runs_allowed"] = home_scores.loc[away]
    repaired_team["run_differential"] = (
        repaired_team["runs_scored"] - repaired_team["runs_allowed"]
    )
    repaired_team["won"] = (
        repaired_team["runs_scored"] > repaired_team["runs_allowed"]
    )

    for column in master.columns:
        if column not in MASTER_REPAIRED_FIELDS and not master[column].equals(
            repaired_master[column]
        ):
            raise TerminalScoreRepairError(f"unauthorized master change: {column}")
    for column in team_state.columns:
        if column not in TEAM_REPAIRED_FIELDS and not team_state[column].equals(
            repaired_team[column]
        ):
            raise TerminalScoreRepairError(f"unauthorized team change: {column}")

    team_pairs = repaired_team.groupby("game_pk").agg(
        rows=("game_pk", "size"),
        runs_scored=("runs_scored", "sum"),
        runs_allowed=("runs_allowed", "sum"),
        differential=("run_differential", "sum"),
        winners=("won", "sum"),
    )
    postconditions = {
        "master_derived_errors": int(
            (
                repaired_master["run_differential"]
                != repaired_master["home_score"] - repaired_master["away_score"]
            ).sum()
        ),
        "team_run_differential_errors": int(
            (
                repaired_team["run_differential"]
                != repaired_team["runs_scored"] - repaired_team["runs_allowed"]
            ).sum()
        ),
        "team_won_errors": int(
            (
                repaired_team["won"]
                != (repaired_team["runs_scored"] > repaired_team["runs_allowed"])
            ).sum()
        ),
        "team_pair_row_count_errors": int((team_pairs["rows"] != 2).sum()),
        "team_pair_score_symmetry_errors": int(
            (team_pairs["runs_scored"] != team_pairs["runs_allowed"]).sum()
        ),
        "team_pair_differential_errors": int(
            (team_pairs["differential"] != 0).sum()
        ),
        "team_pair_winner_errors": int((team_pairs["winners"] != 1).sum()),
    }
    if any(postconditions.values()):
        raise TerminalScoreRepairError(f"repair postconditions failed: {postconditions}")

    audit = {
        "master": {
            field: {
                "rows_changed": int(before_master[field].ne(repaired_master[field]).sum()),
                "rows_changed_by_season": _changes_by_season(
                    before_master[field],
                    repaired_master[field],
                    repaired_master["atlas_season"],
                ),
            }
            for field in MASTER_REPAIRED_FIELDS
        },
        "team_state": {
            field: {
                "rows_changed": int(before_team[field].ne(repaired_team[field]).sum()),
                "rows_changed_by_season": _changes_by_season(
                    before_team[field],
                    repaired_team[field],
                    repaired_team["atlas_season"],
                ),
            }
            for field in TEAM_REPAIRED_FIELDS
        },
        "verification": postconditions,
    }
    return repaired_master, repaired_team, audit
