"""
Canonical consolidated team-game flow fact table for Project ATLAS.

This module joins the frozen Phase 2B and Phase 2D factual layers:

- team outcomes
- team game flow
- lead protection and separation
- response and recovery

The resulting table contains exactly two rows per verified game.

This remains postgame factual data. It is not directly safe as a
pregame feature because it describes the game represented by the row.
Pregame models may later consume lagged or historical aggregates built
strictly from prior games.

This module does not:
- create predictions
- calculate sportsbook value
- update identities
- create explanations
- use future games
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile

import pandas as pd

from .contracts import BRAIN_ENGINE_VERSION


GAME_FLOW_FACT_TABLE_VERSION = "1.0.0"

REPO_ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT = REPO_ROOT / "data"

TEAM_FLOW_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "team_game_flow"
    / "{season}"
    / "team_game_flow.parquet"
)

LEAD_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "lead_protection"
    / "{season}"
    / "team_lead_protection.parquet"
)

RESPONSE_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "response_recovery"
    / "{season}"
    / "team_response_recovery.parquet"
)

TEAM_OUTCOME_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "team_outcomes"
    / "{season}"
    / "team_game_outcomes.parquet"
)

ROLE_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "scoring_event_roles"
    / "{season}"
    / "scoring_event_roles.parquet"
)

OUTPUT_TEMPLATE = (
    DATA_ROOT
    / "game_intelligence"
    / "game_flow_facts"
    / "{season}"
)


KEY_COLUMNS = [
    "game_pk",
    "team",
]


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        destination,
    )


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=str,
        )

        temporary_name = handle.name

    os.replace(
        temporary_name,
        destination,
    )


def _normalize_team_table(
    dataframe: pd.DataFrame,
    *,
    table_name: str,
    season: int,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "team",
        "opponent",
        "home_away",
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise KeyError(
            f"{table_name} missing required columns: "
            f"{sorted(missing)}"
        )

    normalized = dataframe.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    normalized["game_date"] = pd.to_datetime(
        normalized["game_date"],
        errors="raise",
    ).dt.normalize()

    normalized = normalized[
        normalized["atlas_season"].eq(
            int(season)
        )
    ].copy()

    if normalized.duplicated(
        subset=KEY_COLUMNS
    ).any():
        duplicates = normalized[
            normalized.duplicated(
                subset=KEY_COLUMNS,
                keep=False,
            )
        ][KEY_COLUMNS]

        raise AssertionError(
            f"{table_name} contains duplicate team-games: "
            f"{duplicates.head(10).to_dict('records')}"
        )

    return normalized.sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _rename_nonkeys(
    dataframe: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    preserved = {
        "game_pk",
        "team",
    }

    rename_map = {
        column:
            f"{prefix}{column}"

        for column in dataframe.columns
        if column not in preserved
    }

    return dataframe.rename(
        columns=rename_map
    )


def _series_equal(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    left_values = left.copy()
    right_values = right.copy()

    both_missing = (
        left_values.isna()
        & right_values.isna()
    )

    equal = left_values.eq(
        right_values
    )

    return (
        equal
        | both_missing
    )


def _build_role_game_summary(
    roles: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "winner_team",
        "scoring_event_number",
        "opening_score",
        "tying_score",
        "go_ahead_score",
        "lead_extension",
        "deficit_reduction",
        "decisive_scoring_event",
        "late_inning_scoring_event",
    }

    missing = required - set(
        roles.columns
    )

    if missing:
        raise KeyError(
            "Scoring-event roles missing required columns: "
            f"{sorted(missing)}"
        )

    normalized = roles.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    normalized["game_date"] = pd.to_datetime(
        normalized["game_date"],
        errors="raise",
    ).dt.normalize()

    summary = (
        normalized.groupby(
            "game_pk",
            sort=False,
        )
        .agg(
            role_game_date=(
                "game_date",
                "first",
            ),

            role_atlas_season=(
                "atlas_season",
                "first",
            ),

            role_home_team=(
                "home_team",
                "first",
            ),

            role_away_team=(
                "away_team",
                "first",
            ),

            role_winner_team=(
                "winner_team",
                "first",
            ),

            role_scoring_events=(
                "scoring_event_number",
                "size",
            ),

            role_opening_scores=(
                "opening_score",
                "sum",
            ),

            role_tying_scores=(
                "tying_score",
                "sum",
            ),

            role_go_ahead_scores=(
                "go_ahead_score",
                "sum",
            ),

            role_lead_extensions=(
                "lead_extension",
                "sum",
            ),

            role_deficit_reductions=(
                "deficit_reduction",
                "sum",
            ),

            role_decisive_events=(
                "decisive_scoring_event",
                "sum",
            ),

            role_late_scoring_events=(
                "late_inning_scoring_event",
                "sum",
            ),
        )
        .reset_index()
    )

    return summary


def build_team_game_flow_fact_table(
    *,
    team_flow: pd.DataFrame,
    lead_facts: pd.DataFrame,
    response_facts: pd.DataFrame,
    team_outcomes: pd.DataFrame,
    roles: pd.DataFrame,
    season: int = 2024,
) -> pd.DataFrame:
    season = int(season)

    team_flow = _normalize_team_table(
        team_flow,
        table_name="team game flow",
        season=season,
    )

    lead_facts = _normalize_team_table(
        lead_facts,
        table_name="lead protection",
        season=season,
    )

    response_facts = _normalize_team_table(
        response_facts,
        table_name="response recovery",
        season=season,
    )

    team_outcomes = _normalize_team_table(
        team_outcomes,
        table_name="team outcomes",
        season=season,
    )

    base_keys = set(
        map(
            tuple,
            team_flow[
                KEY_COLUMNS
            ].to_numpy(),
        )
    )

    for name, dataframe in [
        (
            "lead protection",
            lead_facts,
        ),
        (
            "response recovery",
            response_facts,
        ),
        (
            "team outcomes",
            team_outcomes,
        ),
    ]:
        keys = set(
            map(
                tuple,
                dataframe[
                    KEY_COLUMNS
                ].to_numpy(),
            )
        )

        if keys != base_keys:
            missing_from_layer = sorted(
                base_keys - keys
            )[:10]

            extra_in_layer = sorted(
                keys - base_keys
            )[:10]

            raise AssertionError(
                f"{name} key universe differs from team flow. "
                f"Missing examples={missing_from_layer}; "
                f"extra examples={extra_in_layer}"
            )

    flow_prefixed = _rename_nonkeys(
        team_flow,
        "flow__",
    )

    lead_prefixed = _rename_nonkeys(
        lead_facts,
        "lead__",
    )

    response_prefixed = _rename_nonkeys(
        response_facts,
        "response__",
    )

    outcome_prefixed = _rename_nonkeys(
        team_outcomes,
        "outcome__",
    )

    consolidated = (
        flow_prefixed
        .merge(
            lead_prefixed,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            response_prefixed,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            outcome_prefixed,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    )

    role_summary = _build_role_game_summary(
        roles
    )

    consolidated = consolidated.merge(
        role_summary,
        on="game_pk",
        how="left",
        validate="many_to_one",
    )

    if consolidated[
        "role_scoring_events"
    ].isna().any():
        raise AssertionError(
            "One or more consolidated rows are missing "
            "game-level role summaries."
        )

    # --------------------------------------------------------
    # Canonical shared identity fields
    # --------------------------------------------------------

    consolidated[
        "game_date"
    ] = consolidated[
        "flow__game_date"
    ]

    consolidated[
        "atlas_season"
    ] = consolidated[
        "flow__atlas_season"
    ].astype("int64")

    consolidated[
        "opponent"
    ] = consolidated[
        "flow__opponent"
    ]

    consolidated[
        "home_away"
    ] = consolidated[
        "flow__home_away"
    ]

    consolidated[
        "team_score"
    ] = consolidated[
        "flow__team_score"
    ].astype("int64")

    consolidated[
        "opponent_score"
    ] = consolidated[
        "flow__opponent_score"
    ].astype("int64")

    consolidated[
        "run_differential"
    ] = consolidated[
        "flow__run_differential"
    ].astype("int64")

    consolidated[
        "won"
    ] = consolidated[
        "flow__won"
    ].astype(bool)

    consolidated[
        "lost"
    ] = consolidated[
        "flow__lost"
    ].astype(bool)

    consolidated[
        "covered_minus_1_5"
    ] = consolidated[
        "flow__covered_minus_1_5"
    ].astype(bool)

    consolidated[
        "covered_plus_1_5"
    ] = consolidated[
        "flow__covered_plus_1_5"
    ].astype(bool)

    # --------------------------------------------------------
    # Cross-layer equality flags
    # --------------------------------------------------------

    consolidated[
        "shared_game_date_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__game_date"
            ],
            consolidated[
                "lead__game_date"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__game_date"
            ],
            consolidated[
                "response__game_date"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__game_date"
            ],
            consolidated[
                "outcome__game_date"
            ],
        )
    )

    consolidated[
        "shared_season_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__atlas_season"
            ],
            consolidated[
                "lead__atlas_season"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__atlas_season"
            ],
            consolidated[
                "response__atlas_season"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__atlas_season"
            ],
            consolidated[
                "outcome__atlas_season"
            ],
        )
    )

    consolidated[
        "shared_opponent_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__opponent"
            ],
            consolidated[
                "lead__opponent"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__opponent"
            ],
            consolidated[
                "response__opponent"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__opponent"
            ],
            consolidated[
                "outcome__opponent"
            ],
        )
    )

    consolidated[
        "shared_home_away_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__home_away"
            ],
            consolidated[
                "lead__home_away"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__home_away"
            ],
            consolidated[
                "response__home_away"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__home_away"
            ],
            consolidated[
                "outcome__home_away"
            ],
        )
    )

    consolidated[
        "shared_score_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__team_score"
            ],
            consolidated[
                "lead__team_score"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__team_score"
            ],
            consolidated[
                "response__team_score"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__team_score"
            ],
            consolidated[
                "outcome__team_score"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__opponent_score"
            ],
            consolidated[
                "lead__opponent_score"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__opponent_score"
            ],
            consolidated[
                "response__opponent_score"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__opponent_score"
            ],
            consolidated[
                "outcome__opponent_score"
            ],
        )
    )

    consolidated[
        "shared_margin_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__run_differential"
            ],
            consolidated[
                "lead__final_run_differential"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__run_differential"
            ],
            consolidated[
                "response__run_differential"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__run_differential"
            ],
            consolidated[
                "outcome__run_differential"
            ],
        )
    )

    consolidated[
        "shared_win_loss_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__won"
            ],
            consolidated[
                "lead__won"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__won"
            ],
            consolidated[
                "response__won"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__won"
            ],
            consolidated[
                "outcome__won"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__lost"
            ],
            consolidated[
                "lead__lost"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__lost"
            ],
            consolidated[
                "response__lost"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__lost"
            ],
            consolidated[
                "outcome__lost"
            ],
        )
    )

    consolidated[
        "shared_run_line_match"
    ] = (
        _series_equal(
            consolidated[
                "flow__covered_minus_1_5"
            ],
            consolidated[
                "lead__covered_minus_1_5"
            ],
        )
        & _series_equal(
            consolidated[
                "flow__covered_plus_1_5"
            ],
            consolidated[
                "lead__covered_plus_1_5"
            ],
        )
    )

    consolidated[
        "score_math_match"
    ] = consolidated[
        "run_differential"
    ].eq(
        consolidated[
            "team_score"
        ]
        - consolidated[
            "opponent_score"
        ]
    )

    consolidated[
        "winner_margin_match"
    ] = consolidated[
        "won"
    ].eq(
        consolidated[
            "run_differential"
        ].gt(0)
    )

    consolidated[
        "loser_margin_match"
    ] = consolidated[
        "lost"
    ].eq(
        consolidated[
            "run_differential"
        ].lt(0)
    )

    consolidated[
        "minus_1_5_math_match"
    ] = consolidated[
        "covered_minus_1_5"
    ].eq(
        consolidated[
            "run_differential"
        ].ge(2)
    )

    consolidated[
        "plus_1_5_math_match"
    ] = consolidated[
        "covered_plus_1_5"
    ].eq(
        consolidated[
            "run_differential"
        ].ge(-1)
    )

    # --------------------------------------------------------
    # Phase 2B outcome compatibility aliases
    # --------------------------------------------------------
    #
    # The frozen Phase 2B team-outcome artifact does not expose
    # every margin threshold under the exact names originally
    # expected by Phase 2D.6.
    #
    # These aliases are derived only from the already verified
    # team score, opponent score, run differential, won, and lost
    # fields. No definitions or source facts are changed.
    # --------------------------------------------------------

    consolidated[
        "outcome__one_run_win"
    ] = (
        consolidated[
            "outcome__won"
        ].astype(bool)
        & consolidated[
            "outcome__run_differential"
        ].eq(1)
    )

    consolidated[
        "outcome__one_run_loss"
    ] = (
        consolidated[
            "outcome__lost"
        ].astype(bool)
        & consolidated[
            "outcome__run_differential"
        ].eq(-1)
    )

    consolidated[
        "outcome__win_by_2_plus"
    ] = (
        consolidated[
            "outcome__won"
        ].astype(bool)
        & consolidated[
            "outcome__run_differential"
        ].ge(2)
    )

    consolidated[
        "outcome__loss_by_2_plus"
    ] = (
        consolidated[
            "outcome__lost"
        ].astype(bool)
        & consolidated[
            "outcome__run_differential"
        ].le(-2)
    )

    consolidated[
        "outcome__win_by_4_plus"
    ] = (
        consolidated[
            "outcome__won"
        ].astype(bool)
        & consolidated[
            "outcome__run_differential"
        ].ge(4)
    )

    consolidated[
        "outcome__loss_by_4_plus"
    ] = (
        consolidated[
            "outcome__lost"
        ].astype(bool)
        & consolidated[
            "outcome__run_differential"
        ].le(-4)
    )

    # --------------------------------------------------------
    # Compact canonical feature families
    # --------------------------------------------------------

    canonical_columns = [
        "game_pk",
        "game_date",
        "atlas_season",
        "team",
        "opponent",
        "home_away",
        "team_score",
        "opponent_score",
        "run_differential",
        "won",
        "lost",

        # Run-line facts
        "flow__won_by_2_plus",
        "flow__won_by_3_plus",
        "flow__lost_by_2_plus",
        "flow__lost_by_3_plus",
        "covered_minus_1_5",
        "covered_plus_1_5",
        "flow__failed_minus_1_5_as_winner",

        # Scoring-role facts
        "flow__scored_first",
        "flow__allowed_first_score",
        "flow__team_scoring_events",
        "flow__opponent_scoring_events",
        "flow__team_tying_scores",
        "flow__team_go_ahead_scores",
        "flow__team_lead_extensions",
        "flow__team_deficit_reductions",
        "flow__team_late_scoring_events",
        "flow__team_late_runs",
        "flow__team_late_lead_extensions",
        "flow__decisive_score_for",
        "flow__decisive_score_against",
        "flow__decisive_inning",
        "flow__decisive_lead_size",
        "flow__winner_additional_runs_after_decisive",

        # Lead/separation facts
        "lead__ever_led",
        "lead__ever_led_by_2",
        "lead__ever_led_by_3",
        "lead__ever_led_by_4",
        "lead__maximum_lead",
        "lead__maximum_deficit",
        "lead__first_lead_inning",
        "lead__first_two_run_lead_inning",
        "lead__first_three_run_lead_inning",
        "lead__surrendered_lead",
        "lead__regained_lead_after_surrender",
        "lead__led_but_lost",
        "lead__led_by_2_but_lost",
        "lead__led_by_3_but_lost",
        "lead__two_run_lead_held_to_final",
        "lead__three_run_lead_held_to_final",
        "lead__led_by_2_but_failed_minus_1_5",
        "lead__led_by_3_but_failed_minus_1_5",
        "lead__dropped_below_two_after_reaching_two",
        "lead__winner_failed_to_separate",
        "lead__winner_maintained_two_plus_after_first_two_run_lead",
        "lead__gave_back_runs_after_maximum_lead",
        "lead__winner_created_two_plus_after_decisive",

        # Response/recovery facts
        "response__won_after_allowing_first_score",
        "response__lost_after_scoring_first",
        "response__opponent_scoring_events",
        "response__team_scored_next_after_opponent_event",
        "response__team_eventually_responded",
        "response__unanswered_opponent_scoring_events",
        "response__same_inning_responses",
        "response__within_one_inning_responses",
        "response__tying_responses",
        "response__go_ahead_responses",
        "response__late_responses",
        "response__immediate_response_rate",
        "response__eventual_response_rate",
        "response__average_response_event_gap",
        "response__average_response_inning_gap",
        "response__longest_opponent_unanswered_event_streak",
        "response__longest_opponent_unanswered_run_streak",
        "response__answered_after_falling_behind",
        "response__tied_after_falling_behind",
        "response__took_lead_after_falling_behind",

        # Existing Phase 2B outcome facts
        "outcome__one_run_win",
        "outcome__one_run_loss",
        "outcome__win_by_2_plus",
        "outcome__loss_by_2_plus",
        "outcome__win_by_4_plus",
        "outcome__loss_by_4_plus",
        "outcome__shutout_win",
        "outcome__shutout_loss",
        "outcome__walkoff_win",
        "outcome__walkoff_loss",
        "outcome__comeback_win",
        "outcome__comeback_loss",
        "outcome__largest_deficit_overcome",
        "outcome__largest_lead_lost",

        # Game-level role summary
        "role_scoring_events",
        "role_opening_scores",
        "role_tying_scores",
        "role_go_ahead_scores",
        "role_lead_extensions",
        "role_deficit_reductions",
        "role_decisive_events",
        "role_late_scoring_events",

        # Cross-layer validation
        "shared_game_date_match",
        "shared_season_match",
        "shared_opponent_match",
        "shared_home_away_match",
        "shared_score_match",
        "shared_margin_match",
        "shared_win_loss_match",
        "shared_run_line_match",
        "score_math_match",
        "winner_margin_match",
        "loser_margin_match",
        "minus_1_5_math_match",
        "plus_1_5_math_match",
    ]

    missing_canonical = [
        column
        for column in canonical_columns
        if column not in consolidated.columns
    ]

    if missing_canonical:
        raise KeyError(
            "Consolidated table is missing canonical columns: "
            f"{missing_canonical}"
        )

    result = consolidated[
        canonical_columns
    ].copy()

    result[
        "all_cross_layer_checks_pass"
    ] = result[
        [
            "shared_game_date_match",
            "shared_season_match",
            "shared_opponent_match",
            "shared_home_away_match",
            "shared_score_match",
            "shared_margin_match",
            "shared_win_loss_match",
            "shared_run_line_match",
            "score_math_match",
            "winner_margin_match",
            "loser_margin_match",
            "minus_1_5_math_match",
            "plus_1_5_math_match",
        ]
    ].all(
        axis=1
    )

    result[
        "postgame_factual_only"
    ] = True

    result[
        "pregame_feature_safe"
    ] = False

    result[
        "prediction_created"
    ] = False

    result[
        "identity_updated"
    ] = False

    result[
        "explanation_created"
    ] = False

    result[
        "future_games_used"
    ] = False

    result[
        "brain_engine_version"
    ] = BRAIN_ENGINE_VERSION

    result[
        "game_flow_fact_table_version"
    ] = GAME_FLOW_FACT_TABLE_VERSION

    return result.sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)


def audit_team_game_flow_fact_table(
    facts: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows = []

    for game_pk, game in facts.groupby(
        "game_pk",
        sort=False,
    ):
        exactly_two_rows = bool(
            len(game) == 2
        )

        unique_teams = bool(
            game[
                "team"
            ].nunique() == 2
        )

        one_home_one_away = bool(
            set(
                game[
                    "home_away"
                ].tolist()
            ) == {
                "HOME",
                "AWAY",
            }
        )

        opponent_mirror = bool(
            len(game) == 2
            and game[
                "team"
            ].iloc[0]
            == game[
                "opponent"
            ].iloc[1]
            and game[
                "opponent"
            ].iloc[0]
            == game[
                "team"
            ].iloc[1]
        )

        score_mirror = bool(
            len(game) == 2
            and int(
                game[
                    "team_score"
                ].iloc[0]
            )
            == int(
                game[
                    "opponent_score"
                ].iloc[1]
            )
            and int(
                game[
                    "opponent_score"
                ].iloc[0]
            )
            == int(
                game[
                    "team_score"
                ].iloc[1]
            )
        )

        margin_mirror = bool(
            len(game) == 2
            and int(
                game[
                    "run_differential"
                ].sum()
            ) == 0
        )

        one_winner = bool(
            int(
                game[
                    "won"
                ].sum()
            ) == 1
        )

        one_loser = bool(
            int(
                game[
                    "lost"
                ].sum()
            ) == 1
        )

        one_decisive_owner = bool(
            int(
                game[
                    "flow__decisive_score_for"
                ].sum()
            ) == 1
        )

        one_scored_first = bool(
            int(
                game[
                    "flow__scored_first"
                ].sum()
            ) == 1
        )

        one_opening_score = bool(
            int(
                game[
                    "role_opening_scores"
                ].iloc[0]
            ) == 1
            and game[
                "role_opening_scores"
            ].nunique() == 1
        )

        one_decisive_role = bool(
            int(
                game[
                    "role_decisive_events"
                ].iloc[0]
            ) == 1
            and game[
                "role_decisive_events"
            ].nunique() == 1
        )

        role_summary_shared = bool(
            game[
                [
                    "role_scoring_events",
                    "role_opening_scores",
                    "role_tying_scores",
                    "role_go_ahead_scores",
                    "role_lead_extensions",
                    "role_deficit_reductions",
                    "role_decisive_events",
                    "role_late_scoring_events",
                ]
            ].nunique(
                dropna=False
            ).le(1).all()
        )

        all_row_checks = bool(
            game[
                "all_cross_layer_checks_pass"
            ].all()
        )

        run_line_complement = bool(
            int(
                game[
                    "covered_minus_1_5"
                ].sum()
            )
            in {
                0,
                1,
            }
        )

        plus_line_logic = bool(
            int(
                game[
                    "covered_plus_1_5"
                ].sum()
            )
            in {
                1,
                2,
            }
        )

        one_run_game_plus_line = True

        winner_margin = int(
            game.loc[
                game[
                    "won"
                ],
                "run_differential",
            ].iloc[0]
        )

        if winner_margin == 1:
            one_run_game_plus_line = bool(
                game[
                    "covered_plus_1_5"
                ].all()
                and not game[
                    "covered_minus_1_5"
                ].any()
            )

        provenance_pass = bool(
            game[
                "postgame_factual_only"
            ].all()
            and (
                ~game[
                    "pregame_feature_safe"
                ]
            ).all()
            and (
                ~game[
                    "prediction_created"
                ]
            ).all()
            and (
                ~game[
                    "identity_updated"
                ]
            ).all()
            and (
                ~game[
                    "explanation_created"
                ]
            ).all()
            and (
                ~game[
                    "future_games_used"
                ]
            ).all()
        )

        audit_pass = bool(
            exactly_two_rows
            and unique_teams
            and one_home_one_away
            and opponent_mirror
            and score_mirror
            and margin_mirror
            and one_winner
            and one_loser
            and one_decisive_owner
            and one_scored_first
            and one_opening_score
            and one_decisive_role
            and role_summary_shared
            and all_row_checks
            and run_line_complement
            and plus_line_logic
            and one_run_game_plus_line
            and provenance_pass
        )

        audit_rows.append({
            "game_pk":
                int(game_pk),

            "game_date":
                game[
                    "game_date"
                ].iloc[0],

            "rows":
                int(len(game)),

            "exactly_two_rows":
                exactly_two_rows,

            "unique_teams":
                unique_teams,

            "one_home_one_away":
                one_home_one_away,

            "opponent_mirror":
                opponent_mirror,

            "score_mirror":
                score_mirror,

            "margin_mirror":
                margin_mirror,

            "one_winner":
                one_winner,

            "one_loser":
                one_loser,

            "one_decisive_owner":
                one_decisive_owner,

            "one_scored_first":
                one_scored_first,

            "one_opening_score":
                one_opening_score,

            "one_decisive_role":
                one_decisive_role,

            "role_summary_shared":
                role_summary_shared,

            "all_row_checks":
                all_row_checks,

            "run_line_complement":
                run_line_complement,

            "plus_line_logic":
                plus_line_logic,

            "one_run_game_plus_line":
                one_run_game_plus_line,

            "provenance_pass":
                provenance_pass,

            "audit_pass":
                audit_pass,
        })

    return pd.DataFrame(
        audit_rows
    ).sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_game_flow_fact_table_build(
    season: int = 2024,
) -> dict[str, Any]:
    season = int(season)

    team_flow_path = Path(
        str(
            TEAM_FLOW_TEMPLATE
        ).format(
            season=season
        )
    )

    lead_path = Path(
        str(
            LEAD_TEMPLATE
        ).format(
            season=season
        )
    )

    response_path = Path(
        str(
            RESPONSE_TEMPLATE
        ).format(
            season=season
        )
    )

    team_outcome_path = Path(
        str(
            TEAM_OUTCOME_TEMPLATE
        ).format(
            season=season
        )
    )

    role_path = Path(
        str(
            ROLE_TEMPLATE
        ).format(
            season=season
        )
    )

    output_dir = Path(
        str(
            OUTPUT_TEMPLATE
        ).format(
            season=season
        )
    )

    fact_path = (
        output_dir
        / "team_game_flow_facts.parquet"
    )

    audit_path = (
        output_dir
        / "team_game_flow_fact_audit.parquet"
    )

    failure_path = (
        output_dir
        / "team_game_flow_fact_failures.parquet"
    )

    metadata_path = (
        output_dir
        / "team_game_flow_fact_metadata.json"
    )

    failures = pd.DataFrame(
        columns=[
            "season",
            "error_type",
            "error_message",
        ]
    )

    try:
        team_flow = pd.read_parquet(
            team_flow_path
        )

        lead_facts = pd.read_parquet(
            lead_path
        )

        response_facts = pd.read_parquet(
            response_path
        )

        team_outcomes = pd.read_parquet(
            team_outcome_path
        )

        roles = pd.read_parquet(
            role_path
        )

        facts = build_team_game_flow_fact_table(
            team_flow=team_flow,
            lead_facts=lead_facts,
            response_facts=response_facts,
            team_outcomes=team_outcomes,
            roles=roles,
            season=season,
        )

        audit = audit_team_game_flow_fact_table(
            facts
        )

    except Exception as exc:
        failures = pd.DataFrame([
            {
                "season":
                    season,

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            }
        ])

        facts = pd.DataFrame()
        audit = pd.DataFrame()

    duplicate_team_games = int(
        facts.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
        if not facts.empty
        else 0
    )

    audit_failures = (
        audit[
            ~audit[
                "audit_pass"
            ]
        ]
        if not audit.empty
        else pd.DataFrame()
    )

    cross_layer_failures = int(
        (
            ~facts[
                "all_cross_layer_checks_pass"
            ]
        ).sum()
        if not facts.empty
        else 0
    )

    games = int(
        facts[
            "game_pk"
        ].nunique()
        if not facts.empty
        else 0
    )

    teams = int(
        facts[
            "team"
        ].nunique()
        if not facts.empty
        else 0
    )

    phase_pass = bool(
        len(facts) == 4_856
        and games == 2_428
        and teams == 30
        and failures.empty
        and audit_failures.empty
        and duplicate_team_games == 0
        and cross_layer_failures == 0
    )

    metadata = {
        "engine":
            "ATLAS Consolidated Team Game-Flow Fact Table",

        "season":
            season,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "game_flow_fact_table_version":
            GAME_FLOW_FACT_TABLE_VERSION,

        "team_game_rows":
            int(
                len(facts)
            ),

        "columns":
            int(
                len(
                    facts.columns
                )
            ),

        "games":
            games,

        "teams":
            teams,

        "build_failures":
            int(
                len(failures)
            ),

        "audit_failures":
            int(
                len(audit_failures)
            ),

        "cross_layer_failures":
            cross_layer_failures,

        "duplicate_team_games":
            duplicate_team_games,

        "winners_by_2_plus":
            int(
                facts[
                    "flow__won_by_2_plus"
                ].sum()
                if not facts.empty
                else 0
            ),

        "one_run_winners":
            int(
                facts[
                    "flow__failed_minus_1_5_as_winner"
                ].sum()
                if not facts.empty
                else 0
            ),

        "teams_reaching_two_run_lead":
            int(
                facts[
                    "lead__ever_led_by_2"
                ].sum()
                if not facts.empty
                else 0
            ),

        "teams_reaching_two_but_failing_minus_1_5":
            int(
                facts[
                    "lead__led_by_2_but_failed_minus_1_5"
                ].sum()
                if not facts.empty
                else 0
            ),

        "wins_after_allowing_first":
            int(
                facts[
                    "response__won_after_allowing_first_score"
                ].sum()
                if not facts.empty
                else 0
            ),

        "phase_2d6_pass":
            phase_pass,

        "postgame_factual_only":
            True,

        "pregame_feature_safe":
            False,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "future_games_used":
            False,

        "built_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": {
            "facts":
                str(fact_path),

            "audit":
                str(audit_path),

            "failures":
                str(failure_path),

            "metadata":
                str(metadata_path),
        },
    }

    _atomic_parquet_write(
        facts,
        fact_path,
    )

    _atomic_parquet_write(
        audit,
        audit_path,
    )

    _atomic_parquet_write(
        failures,
        failure_path,
    )

    _atomic_json_write(
        metadata,
        metadata_path,
    )

    return {
        "facts":
            facts,

        "audit":
            audit,

        "failures":
            failures,

        "metadata":
            metadata,
    }
