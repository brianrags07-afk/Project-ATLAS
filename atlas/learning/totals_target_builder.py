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

# The total-run bucket boundaries are governed by a versioned contract
# file, not recomputed here. See
# ``atlas_reference/manifests/frozen_scoring_bucket_contract_2024_v1.json``
# for the discovery season, source dataset, sample size, percentile
# method, percentile values, and immutability rules. This module only
# *reads* that contract at import time; it never derives percentiles
# from its own runtime inputs. Whether the 2024 boundaries may be
# reused for 2025+ blind validation or production depends on the
# contract's provenance -- see ``_validate_contract_provenance`` and
# ``SCORING_BUCKET_CONTRACT_PRODUCTION_READY`` below: a contract is
# only production-ready once it is ``status == "frozen"`` *and* carries
# a resolved, authoritative full-source-population hash, not merely a
# checked-in sample-file hash.
FROZEN_SCORING_BUCKET_CONTRACT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "atlas_reference"
    / "manifests"
    / "frozen_scoring_bucket_contract_2024_v1.json"
)

# Statuses that represent a canonically frozen, production-ready
# contract. A contract declaring one of these statuses must carry a
# resolved authoritative full-source hash -- if it does not, that is a
# provenance contradiction and must fail loudly rather than silently
# be treated as trustworthy.
_FROZEN_PRODUCTION_READY_STATUSES: Final[frozenset[str]] = frozenset(
    {"frozen"}
)


def _load_frozen_scoring_bucket_contract() -> dict:
    with FROZEN_SCORING_BUCKET_CONTRACT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _validate_contract_provenance(contract: dict) -> bool:
    """
    Determine whether ``contract`` is genuinely production-ready, and
    refuse to load a contract that misrepresents itself.

    A contract that declares a canonically frozen/production-ready
    ``status`` (see ``_FROZEN_PRODUCTION_READY_STATUSES``) but has no
    resolved authoritative full-source-population hash is a provenance
    contradiction: percentile boundaries were derived from "the full
    2024 source population", yet nothing here proves the input used
    to derive them matches that population. This must fail loudly at
    import time rather than let a frozen/production-ready contract with
    an absent or explicitly pending hash load silently.

    Returns ``True`` when the contract carries a resolved authoritative
    full-source hash (i.e. it is safe to use for 2025+ blind validation
    or production), ``False`` otherwise (e.g. a provisional /
    source-unverified contract, which may still be loaded -- to expose
    its boundaries for inspection and 2024-only, non-production use --
    but must not be usable for 2025+ blind validation or production).
    """

    status = contract.get("status")
    source_dataset = contract.get("source_dataset", {})
    authoritative_hash = source_dataset.get(
        "authoritative_full_source_sha256"
    )
    hash_status = source_dataset.get(
        "authoritative_full_source_hash_status"
    )

    hash_resolved = bool(authoritative_hash) and hash_status == "resolved"

    if status in _FROZEN_PRODUCTION_READY_STATUSES and not hash_resolved:
        raise ValueError(
            f"Scoring bucket contract declares status={status!r} "
            "(canonically frozen/production-ready) but "
            "source_dataset.authoritative_full_source_sha256/"
            "authoritative_full_source_hash_status is absent or not "
            "'resolved'. A frozen discovery contract must be tied to "
            "the exact full source population used to derive its "
            "percentile values -- refusing to load a contract that "
            "represents itself as frozen while its authoritative "
            "source hash remains unresolved. Either attach and verify "
            "the authoritative full-source hash, or mark this "
            "contract's status as provisional/source_unverified."
        )

    return hash_resolved


_FROZEN_SCORING_BUCKET_CONTRACT: Final[dict] = (
    _load_frozen_scoring_bucket_contract()
)

# Whether the loaded contract is verified against its authoritative
# full-source population and therefore safe to use for 2025+ blind
# validation or production. See ``_validate_contract_provenance``.
SCORING_BUCKET_CONTRACT_PRODUCTION_READY: Final[bool] = (
    _validate_contract_provenance(_FROZEN_SCORING_BUCKET_CONTRACT)
)

FROZEN_SCORING_BUCKET_CONTRACT_VERSION: Final[str] = (
    _FROZEN_SCORING_BUCKET_CONTRACT["contract_version"]
)

_CONTRACT_DISCOVERY_SEASON: Final[int] = _FROZEN_SCORING_BUCKET_CONTRACT[
    "discovery_season"
]

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


def _assert_scoring_bucket_contract_may_be_used(
    game_targets: pd.DataFrame,
) -> None:
    """
    Enforce the frozen scoring-bucket contract's usage restrictions.
    The 2024-derived boundaries may always be used to label 2024
    (the discovery season) itself, but reusing them unchanged for
    2025+ blind validation or production requires the contract to be
    verified against its authoritative full-source population (see
    ``SCORING_BUCKET_CONTRACT_PRODUCTION_READY``). Refuses -- rather
    than silently proceeding -- to label a post-discovery season with
    an unverified contract.
    """

    if SCORING_BUCKET_CONTRACT_PRODUCTION_READY:
        return

    post_discovery_seasons = sorted(
        set(
            game_targets["atlas_season"][
                game_targets["atlas_season"].gt(_CONTRACT_DISCOVERY_SEASON)
            ]
        )
    )

    if post_discovery_seasons:
        raise ValueError(
            "The frozen scoring bucket contract "
            f"({FROZEN_SCORING_BUCKET_CONTRACT_PATH.name}) is not "
            "production-ready: its authoritative full-source hash is "
            "absent or pending, so it may not be used for 2025+ blind "
            "validation or production. Refusing to build totals "
            f"targets for season(s) {post_discovery_seasons} until the "
            "authoritative full-source hash is attached and verified "
            "(or the contract's status is otherwise re-marked "
            "'frozen')."
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


def _assert_team_game_scoring_consistency(
    game_targets: pd.DataFrame,
    team_game_targets: pd.DataFrame,
) -> None:
    """
    Verify, before any side-level classification is merged onto the
    game-level totals table, that ``team_game_targets`` genuinely
    agrees with ``game_targets`` on scoring. A left join against a
    malformed ``team_game_targets`` (a missing HOME/AWAY row, a
    duplicate side row, or a ``team_runs`` value that disagrees with
    the frozen ``home_score``/``away_score``) would otherwise silently
    surface as null or wrong scoring-shape columns downstream. This
    function fails loudly instead.
    """

    game_pks = game_targets["game_pk"]

    game_pk_target_pks = set(game_pks)
    team_game_pks = set(team_game_targets["game_pk"])

    missing_from_team_game_targets = sorted(
        game_pk_target_pks.difference(team_game_pks)
    )
    extra_in_team_game_targets = sorted(
        team_game_pks.difference(game_pk_target_pks)
    )

    if missing_from_team_game_targets or extra_in_team_game_targets:
        raise AssertionError(
            "team_game_targets['game_pk'] coverage does not match "
            "game_targets['game_pk'] coverage; "
            f"missing from team_game_targets: {missing_from_team_game_targets}; "
            f"extra in team_game_targets: {extra_in_team_game_targets}."
        )

    for side in ("HOME", "AWAY"):
        side_rows = team_game_targets[
            team_game_targets["home_away"].eq(side)
        ]

        side_pk_counts = side_rows["game_pk"].value_counts()

        missing_side = sorted(
            game_pk_target_pks.difference(set(side_pk_counts.index))
        )

        if missing_side:
            raise AssertionError(
                f"team_game_targets is missing a {side} row for "
                f"game_pk(s): {missing_side}."
            )

        duplicated_side = sorted(
            side_pk_counts[side_pk_counts.gt(1)].index
        )

        if duplicated_side:
            raise AssertionError(
                f"team_game_targets has more than one {side} row for "
                f"game_pk(s): {duplicated_side}."
            )

    total_side_rows = len(
        team_game_targets[
            team_game_targets["home_away"].isin(("HOME", "AWAY"))
        ]
    )
    expected_side_rows = 2 * len(game_pk_target_pks)

    if total_side_rows != len(team_game_targets) or (
        total_side_rows != expected_side_rows
    ):
        raise AssertionError(
            "team_game_targets contains extra team-game rows beyond "
            "exactly one HOME row and one AWAY row per game_pk; "
            f"expected {expected_side_rows:,} rows, found "
            f"{len(team_game_targets):,} row(s)."
        )

    expected_runs_by_side = {
        "HOME": game_targets.set_index("game_pk")["home_score"],
        "AWAY": game_targets.set_index("game_pk")["away_score"],
    }

    for side, expected_runs in expected_runs_by_side.items():
        side_rows = team_game_targets[
            team_game_targets["home_away"].eq(side)
        ].set_index("game_pk")

        actual_runs = side_rows["team_runs"].reindex(expected_runs.index)

        mismatched_runs = expected_runs.ne(actual_runs)

        if mismatched_runs.any():
            raise AssertionError(
                f"team_game_targets['team_runs'] for {side} rows does "
                "not match game_targets['"
                f"{'home_score' if side == 'HOME' else 'away_score'}"
                "'] for game_pk(s): "
                f"{sorted(expected_runs.index[mismatched_runs])}."
            )

        team_runs = side_rows["team_runs"].reindex(expected_runs.index)

        expected_flags = {
            "target_team_scored_3_or_less": team_runs.le(3),
            "target_team_scored_exactly_4": team_runs.eq(4),
            "target_team_scored_5_plus": team_runs.ge(5),
        }

        for flag_column, expected_flag in expected_flags.items():
            actual_flag = side_rows[flag_column].reindex(
                expected_runs.index
            )

            mismatched_flag = expected_flag.astype("boolean").ne(
                actual_flag.astype("boolean")
            )

            if mismatched_flag.any():
                raise AssertionError(
                    f"team_game_targets['{flag_column}'] for {side} rows "
                    "does not match team_runs-derived expectations for "
                    f"game_pk(s): "
                    f"{sorted(expected_runs.index[mismatched_flag])}."
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
    _assert_scoring_bucket_contract_may_be_used(game_targets)
    _assert_team_game_scoring_consistency(game_targets, team_game_targets)

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
