"""
Factual "totals" (scoring-shape) target construction for Project ATLAS.

This module makes totals learning a first-class, independent Baseball
Brain target family, alongside (and never merged with) the moneyline
(``target_team_win``/``target_team_win_by_2_plus``) and run-margin
targets already produced by ``atlas.learning.factual_target_builder``.

Independence is structural, not just semantic:

- This module never mutates ``game_targets`` or ``team_game_targets``.
  It only reads their already-frozen factual columns
  (``home_score``/``away_score``/``game_total_runs`` and
  ``home_away``/``team_runs``/``target_team_scored_3_or_less``/
  ``target_team_scored_exactly_4``/``target_team_scored_5_plus``) and
  returns a brand-new game-level table.
- It never renames or overwrites a frozen production column; every
  column produced here is new.
- It carries no moneyline (``target_team_win*``) or run-margin
  (``run_margin``/``home_margin``/``margin_*_plus``) columns at all.

Trace (pregame identity -> totals learning -> postgame explanation):

    pregame baseball state
    -> projected_home_runs / projected_away_runs (future model output)
    -> projected_total_runs (future model output)
    -> scoring-shape probabilities (future model output)
    -> over/under probabilities (future model output)
    -> actual_total_runs (this module, factual/postgame)
    -> postgame Game Story explanation and learning

This module produces only the factual/postgame side of that trace. It
never reads or writes a sportsbook total line; ``market_line_used`` is
always ``False`` here, matching the "keep sportsbook total lines
outside the baseball model" requirement -- any market-total comparison
happens strictly downstream of, and after, this factual/projection
layer.

Total-run bucket boundaries below are not arbitrary. They are derived
from the canonical 2024 completed-game total-runs distribution (2,428
completed games; see
``atlas_reference/samples/games/data__game_intelligence__factual_learning_targets__2024__factual_game_learning_targets.parquet.games.parquet``,
column ``game_total_runs``):

    25th percentile = 5 runs, median = 8 runs, 75th percentile = 11
    runs, 90th percentile = 15 runs.

``LOW_SCORING_MAX_RUNS`` (5) is the bottom quartile cut point.
``HIGH_SCORING_MIN_RUNS`` (12) is one run above the 75th-percentile cut
point (11), i.e. the top quartile -- matching this family's explicit
requirement to separate "approximately 4-run games" (bottom-quartile
territory) from "12+-run games" (top-quartile territory).
``EXTREME_HIGH_SCORING_MIN_RUNS`` (15) is the 90th-percentile cut
point, isolating the deep high-scoring tail.
"""

from __future__ import annotations

from typing import Final

import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

TOTALS_TARGET_FAMILY: Final[str] = "totals"

# Derived from the canonical 2024 completed-game total-runs distribution.
# See module docstring for the exact percentile evidence.
LOW_SCORING_MAX_RUNS: Final[int] = 5
HIGH_SCORING_MIN_RUNS: Final[int] = 12
EXTREME_HIGH_SCORING_MIN_RUNS: Final[int] = 15

TOTAL_RUN_BUCKET_LOW: Final[str] = "low"
TOTAL_RUN_BUCKET_AVERAGE: Final[str] = "average"
TOTAL_RUN_BUCKET_HIGH: Final[str] = "high"
TOTAL_RUN_BUCKET_EXTREME_HIGH: Final[str] = "extreme_high"

REQUIRED_GAME_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "game_date",
    "atlas_season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "game_total_runs",
)

REQUIRED_TEAM_GAME_TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "home_away",
    "team_runs",
    "target_team_scored_3_or_less",
    "target_team_scored_exactly_4",
    "target_team_scored_5_plus",
)


def _classify_total_run_bucket(
    actual_total_runs: pd.Series,
) -> pd.Series:
    return pd.Series(
        [
            TOTAL_RUN_BUCKET_LOW
            if runs <= LOW_SCORING_MAX_RUNS
            else TOTAL_RUN_BUCKET_EXTREME_HIGH
            if runs >= EXTREME_HIGH_SCORING_MIN_RUNS
            else TOTAL_RUN_BUCKET_HIGH
            if runs >= HIGH_SCORING_MIN_RUNS
            else TOTAL_RUN_BUCKET_AVERAGE
            for runs in actual_total_runs
        ],
        index=actual_total_runs.index,
        dtype="string",
    )


def _side_scoring_shape(
    team_game_targets: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    side_rows = team_game_targets[
        team_game_targets["home_away"].eq(side)
    ]

    prefix = side.lower()

    return pd.DataFrame(
        {
            "game_pk": side_rows["game_pk"].to_numpy(),
            f"{prefix}_team_scored_3_or_less": side_rows[
                "target_team_scored_3_or_less"
            ].to_numpy(),
            f"{prefix}_team_scored_exactly_4": side_rows[
                "target_team_scored_exactly_4"
            ].to_numpy(),
            f"{prefix}_team_scored_5_plus": side_rows[
                "target_team_scored_5_plus"
            ].to_numpy(),
        }
    )


def build_total_runs_targets(
    game_targets: pd.DataFrame,
    team_game_targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the independent, game-level totals/scoring-shape target table.

    ``game_targets`` must be the output of
    ``atlas.learning.factual_target_builder.build_game_targets``.
    ``team_game_targets`` must be the output of
    ``atlas.learning.factual_target_builder.build_team_game_targets``
    built from that same ``game_targets`` frame.

    Returns a new game-level dataframe. Neither input is mutated.
    """

    missing_game_columns = sorted(
        set(REQUIRED_GAME_TARGET_COLUMNS).difference(
            game_targets.columns
        )
    )

    if missing_game_columns:
        raise KeyError(
            f"game_targets is missing required columns: {missing_game_columns}"
        )

    missing_team_columns = sorted(
        set(REQUIRED_TEAM_GAME_TARGET_COLUMNS).difference(
            team_game_targets.columns
        )
    )

    if missing_team_columns:
        raise KeyError(
            "team_game_targets is missing required columns: "
            f"{missing_team_columns}"
        )

    totals = pd.DataFrame(
        {
            "game_pk": game_targets["game_pk"],
            "game_date": game_targets["game_date"],
            "atlas_season": game_targets["atlas_season"],
            "home_team": game_targets["home_team"],
            "away_team": game_targets["away_team"],
            "home_runs_scored": game_targets["home_score"],
            "away_runs_scored": game_targets["away_score"],
            "actual_total_runs": game_targets["game_total_runs"],
        }
    )

    home_shape = _side_scoring_shape(
        team_game_targets,
        "HOME",
    )

    away_shape = _side_scoring_shape(
        team_game_targets,
        "AWAY",
    )

    totals = totals.merge(
        home_shape,
        on="game_pk",
        how="left",
        validate="one_to_one",
    ).merge(
        away_shape,
        on="game_pk",
        how="left",
        validate="one_to_one",
    )

    totals["low_scoring_game"] = totals["actual_total_runs"].le(
        LOW_SCORING_MAX_RUNS
    )

    totals["high_scoring_game"] = totals["actual_total_runs"].ge(
        HIGH_SCORING_MIN_RUNS
    )

    totals["extreme_high_scoring_game"] = totals["actual_total_runs"].ge(
        EXTREME_HIGH_SCORING_MIN_RUNS
    )

    totals["total_run_bucket"] = _classify_total_run_bucket(
        totals["actual_total_runs"]
    )

    totals["totals_target_family"] = TOTALS_TARGET_FAMILY

    # Structural independence markers: totals never depend on, and never
    # produce, a moneyline or run-margin column.
    totals["moneyline_independent"] = True
    totals["run_margin_independent"] = True

    # Sportsbook total lines stay outside the baseball model: this
    # factual/projection layer never reads or uses a market total.
    totals["market_line_used"] = False

    totals["target_builder_version"] = ENGINE_VERSION

    duplicate_games = int(
        totals.duplicated(
            subset=["game_pk"]
        ).sum()
    )

    if duplicate_games:
        raise AssertionError(
            f"Duplicate game-level totals targets: {duplicate_games:,}"
        )

    return totals.reset_index(drop=True)
