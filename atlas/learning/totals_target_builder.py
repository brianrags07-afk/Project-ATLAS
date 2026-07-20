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

import json
from pathlib import Path
from typing import Final

import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.1"

TOTALS_TARGET_FAMILY: Final[str] = "totals"

# The total-run bucket boundaries are governed by a versioned, frozen
# contract file, not recomputed here. See
# ``atlas_reference/manifests/frozen_scoring_bucket_contract_2024_v1.json``
# for the discovery season, source dataset, sample size, percentile
# method, percentile values, and immutability rules. This module only
# *reads* that frozen contract at import time; it never derives
# percentiles from its own runtime inputs, and the 2024 boundaries are
# reused unchanged for 2025+ blind validation.
FROZEN_SCORING_BUCKET_CONTRACT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "atlas_reference"
    / "manifests"
    / "frozen_scoring_bucket_contract_2024_v1.json"
)


def _load_frozen_scoring_bucket_contract() -> dict:
    with FROZEN_SCORING_BUCKET_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


_FROZEN_SCORING_BUCKET_CONTRACT: Final[dict] = (
    _load_frozen_scoring_bucket_contract()
)

FROZEN_SCORING_BUCKET_CONTRACT_VERSION: Final[str] = (
    _FROZEN_SCORING_BUCKET_CONTRACT["contract_version"]
)

_FROZEN_BOUNDARIES: Final[dict] = _FROZEN_SCORING_BUCKET_CONTRACT[
    "bucket_boundaries"
]

# Derived from the canonical 2024 completed-game total-runs distribution,
# and loaded from the frozen contract above -- not recomputed here. See
# the module docstring for the exact percentile evidence.
LOW_SCORING_MAX_RUNS: Final[int] = _FROZEN_BOUNDARIES[
    "low_scoring_max_runs"
]
HIGH_SCORING_MIN_RUNS: Final[int] = _FROZEN_BOUNDARIES[
    "high_scoring_min_runs"
]
EXTREME_HIGH_SCORING_MIN_RUNS: Final[int] = _FROZEN_BOUNDARIES[
    "extreme_high_scoring_min_runs"
]

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

# Optional per-game status column. When present on ``game_targets``, it
# is asserted to contain only "final" games; this module never labels a
# postponed, cancelled, suspended-incomplete, or otherwise-invalid game.
# When absent, the caller's ``completed_games`` contract
# (``atlas.learning.factual_target_builder.build_game_targets``) is the
# sole authority for completeness, matching existing production
# behavior -- this column is optional so as not to silently change that
# frozen input contract.
OPTIONAL_GAME_STATE_COLUMN: Final[str] = "game_state_category"

VALID_FINAL_GAME_STATES: Final[frozenset[str]] = frozenset({"final"})

# Regulation-vs-extra-innings fields are *reserved*, not fabricated.
# Splitting a game's total into a regulation-innings (1-9) total and an
# extra-innings total requires a genuine per-inning line score. No such
# per-inning source exists in this repository today (the closest
# available data, ``atlas/game_intelligence/outcome_classifier.py``'s
# ``game_outcomes`` table, only carries innings-bucketed run totals for
# innings 1-3 / 4-6 / 7+, and its 7+ bucket conflates innings 7-9 with
# any extra innings actually played). Inventing a split from that data
# would silently misclassify extra-innings games. These columns are
# therefore always reserved as null until a genuine per-inning line
# score becomes an authoritative ATLAS source.
RESERVED_REGULATION_EXTRA_INNING_COLUMNS: Final[tuple[str, ...]] = (
    "regulation_total_runs",
    "extra_inning_runs",
    "scoring_shape_regulation",
    "scoring_shape_final",
)

# ``went_extra_innings`` is the one field in this family that *is*
# computable today, from the existing, authoritative
# ``game_outcomes.extra_innings`` column (produced by
# ``atlas/game_intelligence/outcome_classifier.py``). It is only
# populated when an optional ``game_outcomes`` frame is supplied to
# ``build_total_runs_targets``; otherwise it is reserved as null, same
# as the columns above.
WENT_EXTRA_INNINGS_COLUMN: Final[str] = "went_extra_innings"

REQUIRED_GAME_OUTCOMES_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "extra_innings",
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


def _assert_data_quality(game_targets: pd.DataFrame) -> None:
    """
    Enforce the factual-only, completed-games-only data-quality
    contract for totals/scoring-shape targets. Raises rather than
    silently repairing bad input -- this module never coerces a
    missing score to zero and never labels an incomplete game.
    """

    if OPTIONAL_GAME_STATE_COLUMN in game_targets.columns:
        invalid_states = sorted(
            set(
                game_targets[OPTIONAL_GAME_STATE_COLUMN]
                .dropna()
                .unique()
            ).difference(VALID_FINAL_GAME_STATES)
        )

        if invalid_states or game_targets[OPTIONAL_GAME_STATE_COLUMN].isna().any():
            raise ValueError(
                "game_targets contains non-final game states "
                f"({invalid_states}) or missing state labels; totals "
                "targets may only label completed ('final') games -- "
                "postponed, cancelled, and suspended-incomplete games "
                "must be excluded before calling build_total_runs_targets."
            )

    for column in ("home_score", "away_score", "game_total_runs"):
        if game_targets[column].isna().any():
            raise ValueError(
                f"game_targets['{column}'] contains missing values; "
                "totals targets never coerce a missing final score to "
                "zero."
            )

    mismatched = (
        game_targets["home_score"] + game_targets["away_score"]
    ).ne(
        game_targets["game_total_runs"]
    )

    if mismatched.any():
        raise AssertionError(
            "game_targets['home_score'] + game_targets['away_score'] != "
            "game_targets['game_total_runs'] for "
            f"{int(mismatched.sum()):,} row(s); refusing to build "
            "totals targets from an inconsistent final score."
        )


def _one_vs_two_team_scoring(totals: pd.DataFrame) -> pd.DataFrame:
    """
    Classify high-scoring games by *where* the runs came from: one
    team's offensive explosion versus two-sided scoring, and a lopsided
    blowout total versus a competitive high total. A 12-1 result and a
    7-6 result both total 13 runs but must not receive identical
    explanatory treatment.
    """

    home = totals["home_runs_scored"]
    away = totals["away_runs_scored"]
    total = totals["actual_total_runs"]
    high_scoring = totals["high_scoring_game"]

    largest_team_run_total = home.where(home.ge(away), away)
    smallest_team_run_total = total - largest_team_run_total

    home_scoring_share = (home / total).where(total.gt(0), 0.0)
    away_scoring_share = (away / total).where(total.gt(0), 0.0)

    # One side alone reaching the frozen high-scoring threshold means
    # that team's offense, not the two teams combined, produced the
    # high total (e.g. 12-1).
    one_team_offensive_explosion = high_scoring & largest_team_run_total.ge(
        HIGH_SCORING_MIN_RUNS
    )
    two_sided_high_scoring_game = high_scoring & ~one_team_offensive_explosion

    # A margin larger than the frozen bottom-quartile (low-scoring) cut
    # point means the high total was driven by a lopsided blowout
    # rather than two competitive offenses (e.g. 12-1 vs. 7-6).
    run_margin_for_shape_classification = (
        largest_team_run_total - smallest_team_run_total
    )
    blowout_high_total = high_scoring & run_margin_for_shape_classification.gt(
        LOW_SCORING_MAX_RUNS
    )
    competitive_high_total = high_scoring & ~blowout_high_total

    return pd.DataFrame(
        {
            "home_scoring_share": home_scoring_share,
            "away_scoring_share": away_scoring_share,
            "largest_team_run_total": largest_team_run_total,
            "one_team_offensive_explosion": one_team_offensive_explosion,
            "two_sided_high_scoring_game": two_sided_high_scoring_game,
            "blowout_high_total": blowout_high_total,
            "competitive_high_total": competitive_high_total,
        },
        index=totals.index,
    )


def _reserved_regulation_extra_inning_columns(
    totals: pd.DataFrame,
    game_outcomes: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Reserve (never fabricate) regulation-vs-extra-innings fields. See
    ``RESERVED_REGULATION_EXTRA_INNING_COLUMNS`` for why a true
    regulation/extra-innings run split cannot be computed from any
    source currently in this repository. ``went_extra_innings`` is the
    one field in this group that *is* computable, from the existing
    ``game_outcomes.extra_innings`` column, when ``game_outcomes`` is
    supplied.
    """

    reserved = pd.DataFrame(
        {
            "regulation_total_runs": pd.array(
                [pd.NA] * len(totals), dtype="Int64"
            ),
            "extra_inning_runs": pd.array(
                [pd.NA] * len(totals), dtype="Int64"
            ),
            "scoring_shape_regulation": pd.array(
                [pd.NA] * len(totals), dtype="string"
            ),
            "scoring_shape_final": pd.array(
                [pd.NA] * len(totals), dtype="string"
            ),
        },
        index=totals.index,
    )

    if game_outcomes is None:
        reserved[WENT_EXTRA_INNINGS_COLUMN] = pd.array(
            [pd.NA] * len(totals), dtype="boolean"
        )
        return reserved

    missing_outcome_columns = sorted(
        set(REQUIRED_GAME_OUTCOMES_COLUMNS).difference(
            game_outcomes.columns
        )
    )

    if missing_outcome_columns:
        raise KeyError(
            "game_outcomes is missing required columns: "
            f"{missing_outcome_columns}"
        )

    extra_innings_by_game = game_outcomes.set_index("game_pk")[
        "extra_innings"
    ]

    reserved[WENT_EXTRA_INNINGS_COLUMN] = (
        totals["game_pk"]
        .map(extra_innings_by_game)
        .astype("boolean")
    )

    return reserved


def build_total_runs_targets(
    game_targets: pd.DataFrame,
    team_game_targets: pd.DataFrame,
    game_outcomes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the independent, game-level totals/scoring-shape target table.

    ``game_targets`` must be the output of
    ``atlas.learning.factual_target_builder.build_game_targets``.
    ``team_game_targets`` must be the output of
    ``atlas.learning.factual_target_builder.build_team_game_targets``
    built from that same ``game_targets`` frame. ``game_outcomes`` is
    optional; when supplied (the output of
    ``atlas.game_intelligence.outcome_classifier``), it populates
    ``went_extra_innings`` -- see
    ``RESERVED_REGULATION_EXTRA_INNING_COLUMNS`` for the other
    regulation/extra-innings fields, which remain reserved (null)
    regardless, because no per-inning line score exists yet to compute
    them accurately.

    Returns a new game-level dataframe. None of the inputs are mutated.
    Only factual, observed outcomes are produced here -- no projected
    runs, probabilities, market comparisons, or over/under selections.
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

    _assert_data_quality(game_targets)

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

    totals = pd.concat(
        [
            totals,
            _one_vs_two_team_scoring(totals),
            _reserved_regulation_extra_inning_columns(
                totals,
                game_outcomes,
            ),
        ],
        axis=1,
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
    totals["scoring_bucket_contract_version"] = (
        FROZEN_SCORING_BUCKET_CONTRACT_VERSION
    )

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
