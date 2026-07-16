"""
Feature semantic governance for Project ATLAS.

This module distinguishes baseball measurements from identifiers, bookkeeping
fields, exposure proxies, safety flags, target analogues, and ambiguous fields.

The classification is governance-only. It does not assign prediction weights,
confidence, probabilities, or predictions.
"""

from __future__ import annotations

from typing import Final
import re

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.1"


IDENTIFIER_EXACT_TOKENS: Final[set[str]] = {
    "game_pk",
    "game_id",
    "player_id",
    "pitcher_id",
    "batter_id",
    "team_id",
    "opponent_id",
    "home_team_id",
    "away_team_id",
    "starter_id",
    "mlbam_id",
    "person_id",
    "roster_id",
    "event_id",
    "play_id",
    "at_bat_number",
    "pitch_number",
}


IDENTIFIER_SUFFIXES: Final[tuple[str, ...]] = (
    "_player_id",
    "_pitcher_id",
    "_batter_id",
    "_team_id",
    "_game_id",
    "_roster_id",
    "_person_id",
    "_mlbam_id",
)


PROVENANCE_TOKENS: Final[tuple[str, ...]] = (
    "strict_backtest_safe",
    "same_date_games_used",
    "future_games_used",
    "prediction_created",
    "prediction_weight",
    "handcrafted",
    "market_used",
    "target_included",
    "source_path",
    "engine_version",
    "provenance",
    "available__",
    "is_available",
    "availability_indicator",
)


TARGET_ANALOGUE_TOKENS: Final[tuple[str, ...]] = (
    "target_",
    "actual_",
    "final_score",
    "winner",
    "loser",
    "won",
    "lost",
    "win_by_",
    "loss_by_",
    "covered_minus",
    "covered_plus",
    "failed_minus",
    "shutout_win",
    "shutout_loss",
)


RATE_TOKENS: Final[tuple[str, ...]] = (
    "_pct",
    "_rate",
    "_per_pa",
    "_per_pitch",
    "_per_game",
    "_per_bf",
    "_per_inning",
    "_ratio",
    "_average",
    "_avg_",
    "_mean",
    "whip",
    "era",
    "fip",
    "woba",
    "ops",
    "slug",
    "obp",
)


COUNT_TOKENS: Final[tuple[str, ...]] = (
    "runs",
    "hits",
    "walks",
    "strikeouts",
    "home_runs",
    "pitches",
    "whiffs",
    "swings",
    "balls",
    "called_strikes",
    "plate_appearances",
    "batters_faced",
    "innings",
    "outs",
    "games_used",
    "appearances",
    "scoring_events",
    "lead_changes",
)


EXPOSURE_TOKENS: Final[tuple[str, ...]] = (
    "plate_appearances",
    "career_pa",
    "season_pa",
    "prior_pa",
    "batters_faced",
    "innings_pitched",
    "pitches_thrown",
    "total_pitches",
    "games_played",
    "games_used",
    "appearances",
    "sample_size",
    "available_sample",
    "minimum_identity_games",
    "identity_games",
    "career_games",
    "season_games",
    "prior_games",
)


CONTEXT_TOKENS: Final[tuple[str, ...]] = (
    "home_away",
    "batting_order_slot",
    "days_rest",
    "days_since",
    "game_number",
    "series_game",
    "doubleheader",
    "park",
    "venue",
    "handedness",
    "throws",
    "stand",
)


ENTITY_NAME_TOKENS: Final[tuple[str, ...]] = (
    "_name",
    "player_name",
    "pitcher_name",
    "team_name",
    "opponent_name",
)


def normalized_name(
    feature_name: str,
) -> str:
    value = str(
        feature_name
    ).strip().lower()

    value = value.replace(
        "home__",
        "",
        1,
    ) if value.startswith(
        "home__"
    ) else value

    value = value.replace(
        "away__",
        "",
        1,
    ) if value.startswith(
        "away__"
    ) else value

    return value


def terminal_name(
    feature_name: str,
) -> str:
    value = normalized_name(
        feature_name
    )

    if "__" in value:
        return value.split(
            "__"
        )[-1]

    return value


def is_identifier_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    terminal = terminal_name(
        feature_name
    )

    if terminal in IDENTIFIER_EXACT_TOKENS:
        return True

    if any(
        terminal.endswith(
            suffix
        )
        for suffix in IDENTIFIER_SUFFIXES
    ):
        return True

    if re.search(
        r"(^|_)batting_order_[1-9]_player_id($|_)",
        value,
    ):
        return True

    if re.search(
        r"(^|_)slot_[1-9]_player_id($|_)",
        value,
    ):
        return True

    return False


def is_entity_name_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    return any(
        token in value
        for token in ENTITY_NAME_TOKENS
    )


def is_provenance_or_safety_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    return any(
        token in value
        for token in PROVENANCE_TOKENS
    )


def is_target_analogue_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    return any(
        token in value
        for token in TARGET_ANALOGUE_TOKENS
    )


def is_recent_workload_feature(
    feature_name: str,
) -> bool:
    """
    Recognize recent pregame workload and usage facts.

    These fields measure recent operational demand, not historical sample size.
    """

    value = normalized_name(
        feature_name
    )

    recent_workload_patterns = (
        "bullpen_games_used_prior_1_dates",
        "bullpen_games_used_prior_2_dates",
        "bullpen_games_used_prior_3_dates",
        "bullpen_games_used_prior_5_dates",
        "bullpen_pitches_prior_1_dates",
        "bullpen_pitches_prior_2_dates",
        "bullpen_pitches_prior_3_dates",
        "bullpen_pitches_prior_5_dates",
        "days_since_prior_bullpen_date",
        "relievers_used_prior_",
        "appearances_prior_1_dates",
        "appearances_prior_2_dates",
        "appearances_prior_3_dates",
        "appearances_prior_5_dates",
    )

    return any(
        pattern in value
        for pattern in recent_workload_patterns
    )


def is_exposure_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    if is_recent_workload_feature(
        feature_name
    ):
        return False

    return any(
        token in value
        for token in EXPOSURE_TOKENS
    )


def is_rate_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    return any(
        token in value
        for token in RATE_TOKENS
    )


def is_count_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    return any(
        token in value
        for token in COUNT_TOKENS
    )


def is_context_feature(
    feature_name: str,
) -> bool:
    value = normalized_name(
        feature_name
    )

    return any(
        token in value
        for token in CONTEXT_TOKENS
    )


def transformation_family_root(
    feature_name: str,
) -> str:
    """
    Collapse common season/career, lineup aggregation, and side wrappers while
    retaining the underlying baseball metric and batting-order slot.
    """

    value = normalized_name(
        feature_name
    )

    prefixes = (
        "identity__identity_edge__",
        "identity__team_identity__",
        "identity__opponent_identity__",
        "identity__",
        "bullpen__",
        "lineup_starter__",
    )

    for prefix in prefixes:
        if value.startswith(
            prefix
        ):
            value = value[
                len(prefix):
            ]

            break

    replacements = {
        "career_prior_": "",
        "season_prior_": "",
        "historical_": "",
        "prior_": "",
        "_season_prior": "",
        "_career_prior": "",
        "_historical": "",
        "_prior_mean": "",
        "_prior_std": "_std",
        "_mean": "",
        "_median": "",
        "_minimum": "_min",
        "_maximum": "_max",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip(
        "_"
    )

    return value


def base_metric_root(
    feature_name: str,
) -> str:
    """
    Collapse aggregation operators while preserving batting-order slot identity.
    """

    value = transformation_family_root(
        feature_name
    )

    value = re.sub(
        r"_(mean|median|min|max|std|sum)$",
        "",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    ).strip(
        "_"
    )

    return value


def infer_value_profile(
    series: pd.Series,
) -> dict:
    non_null = series.dropna()

    row_count = int(
        len(series)
    )

    non_null_count = int(
        len(non_null)
    )

    missing_count = int(
        row_count
        - non_null_count
    )

    if non_null_count == 0:
        return {
            "row_count":
                row_count,

            "non_null_count":
                0,

            "missing_count":
                missing_count,

            "missing_rate":
                1.0,

            "unique_values":
                0,

            "numeric_compatible":
                False,

            "integer_like":
                False,

            "binary_like":
                False,

            "minimum":
                np.nan,

            "maximum":
                np.nan,

            "median":
                np.nan,

            "mean":
                np.nan,
        }

    numeric = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    numeric_compatible = bool(
        numeric.notna().all()
    )

    if numeric_compatible:
        numeric_values = numeric.astype(
            float
        )

        integer_like = bool(
            np.isclose(
                numeric_values,
                np.round(
                    numeric_values
                ),
                equal_nan=False,
            ).all()
        )

        unique_numeric = pd.unique(
            numeric_values
        )

        binary_like = bool(
            set(
                unique_numeric.tolist()
            ).issubset(
                {
                    0.0,
                    1.0,
                }
            )
        )

        minimum = float(
            numeric_values.min()
        )

        maximum = float(
            numeric_values.max()
        )

        median = float(
            numeric_values.median()
        )

        mean = float(
            numeric_values.mean()
        )

    else:
        integer_like = False
        binary_like = False
        minimum = np.nan
        maximum = np.nan
        median = np.nan
        mean = np.nan

    try:
        unique_values = int(
            non_null.nunique(
                dropna=True
            )
        )

    except TypeError:
        unique_values = int(
            non_null.astype(
                str
            ).nunique(
                dropna=True
            )
        )

    return {
        "row_count":
            row_count,

        "non_null_count":
            non_null_count,

        "missing_count":
            missing_count,

        "missing_rate":
            float(
                missing_count
                / row_count
            )
            if row_count
            else np.nan,

        "unique_values":
            unique_values,

        "numeric_compatible":
            numeric_compatible,

        "integer_like":
            integer_like,

        "binary_like":
            binary_like,

        "minimum":
            minimum,

        "maximum":
            maximum,

        "median":
            median,

        "mean":
            mean,
    }


def classify_feature(
    feature_name: str,
    dtype_name: str,
    value_profile: dict,
) -> tuple[str, str, str]:
    """
    Returns:
        semantic_classification,
        governance_action,
        governance_reason
    """

    value = normalized_name(
        feature_name
    )

    numeric_compatible = bool(
        value_profile.get(
            "numeric_compatible",
            False,
        )
    )

    integer_like = bool(
        value_profile.get(
            "integer_like",
            False,
        )
    )

    binary_like = bool(
        value_profile.get(
            "binary_like",
            False,
        )
    )

    unique_values = int(
        value_profile.get(
            "unique_values",
            0,
        )
    )

    if is_identifier_feature(
        feature_name
    ):
        return (
            "IDENTIFIER_NOT_MEASUREMENT",
            "BLOCK_IDENTIFIER_THRESHOLD",
            "Entity or database identifier cannot be interpreted as a continuous baseball measurement.",
        )

    if is_entity_name_feature(
        feature_name
    ):
        return (
            "ENTITY_LABEL_NOT_MEASUREMENT",
            "BLOCK_ENTITY_LABEL",
            "Entity label is descriptive identity, not an ordered baseball measurement.",
        )

    if is_provenance_or_safety_feature(
        feature_name
    ):
        return (
            "PROVENANCE_OR_SAFETY_FIELD",
            "BLOCK_NON_BASEBALL_PROVENANCE",
            "Safety, provenance, availability or engine-control field cannot be treated as baseball evidence.",
        )

    if is_target_analogue_feature(
        feature_name
    ):
        return (
            "POTENTIAL_TARGET_ANALOGUE",
            "BLOCK_TARGET_ANALOGUE",
            "Feature name directly describes a result or a close analogue of the learning target.",
        )

    if is_recent_workload_feature(
        feature_name
    ):
        return (
            "VALID_RECENT_WORKLOAD_FACT",
            "KEEP_SEMANTICALLY_VALID",
            "Chronologically safe recent pregame workload or usage measurement.",
        )

    if is_exposure_feature(
        feature_name
    ):
        return (
            "SAMPLE_SIZE_OR_EXPOSURE_PROXY",
            "REVIEW_EXPOSURE_PROXY",
            "Field primarily describes historical opportunity or sample size rather than standalone skill.",
        )

    if is_context_feature(
        feature_name
    ):
        return (
            "VALID_CONTEXT_FACT",
            "KEEP_SEMANTICALLY_VALID",
            "Pregame context fact with interpretable baseball meaning.",
        )

    if not numeric_compatible:
        return (
            "NON_NUMERIC_UNSUPPORTED_FOR_THRESHOLD",
            "BLOCK_NON_NUMERIC_THRESHOLD",
            "Feature values are not consistently numeric but were used with a numeric threshold.",
        )

    if binary_like:
        return (
            "VALID_BINARY_BASEBALL_FACT",
            "KEEP_SEMANTICALLY_VALID",
            "Binary factual baseball state or condition.",
        )

    if is_rate_feature(
        feature_name
    ):
        return (
            "VALID_RATE_BASEBALL_FACT",
            "KEEP_SEMANTICALLY_VALID",
            "Continuous baseball rate, percentage, ratio or normalized measurement.",
        )

    if is_count_feature(
        feature_name
    ):
        if integer_like:
            return (
                "VALID_COUNT_BASEBALL_FACT",
                "KEEP_SEMANTICALLY_VALID",
                "Interpretable baseball event or workload count.",
            )

        return (
            "VALID_CONTINUOUS_BASEBALL_FACT",
            "KEEP_SEMANTICALLY_VALID",
            "Continuous historical baseball measurement derived from factual events.",
        )

    if (
        integer_like
        and unique_values > 100
        and (
            value.endswith(
                "_id"
            )
            or "identifier" in value
        )
    ):
        return (
            "POSSIBLE_UNRECOGNIZED_IDENTIFIER",
            "MANUAL_REVIEW_REQUIRED",
            "High-cardinality integer-like feature resembles an unrecognized identifier.",
        )

    if unique_values <= 1:
        return (
            "CONSTANT_OR_DEGENERATE_FEATURE",
            "BLOCK_DEGENERATE_FEATURE",
            "Feature does not vary meaningfully in the audited discovery universe.",
        )

    if integer_like:
        return (
            "VALID_COUNT_OR_ORDINAL_FACT",
            "KEEP_SEMANTICALLY_VALID",
            "Integer-valued factual baseball measurement with interpretable ordering.",
        )

    return (
        "VALID_CONTINUOUS_BASEBALL_FACT",
        "KEEP_SEMANTICALLY_VALID",
        "Continuous factual baseball measurement.",
    )


def combine_member_actions(
    member_1_action: str,
    member_2_action: str,
) -> tuple[str, str]:
    actions = {
        str(
            member_1_action
        ),
        str(
            member_2_action
        ),
    }

    blocking_actions = {
        "BLOCK_IDENTIFIER_THRESHOLD",
        "BLOCK_ENTITY_LABEL",
        "BLOCK_NON_BASEBALL_PROVENANCE",
        "BLOCK_TARGET_ANALOGUE",
        "BLOCK_NON_NUMERIC_THRESHOLD",
        "BLOCK_DEGENERATE_FEATURE",
    }

    review_actions = {
        "REVIEW_EXPOSURE_PROXY",
        "MANUAL_REVIEW_REQUIRED",
    }

    if actions.intersection(
        blocking_actions
    ):
        return (
            "BLOCKED_INVALID_MEMBER",
            "At least one concept member is not a valid ordered baseball measurement.",
        )

    if actions.intersection(
        review_actions
    ):
        return (
            "REVIEW_REQUIRED_MEMBER_SEMANTICS",
            "At least one concept member is an exposure proxy or requires semantic review.",
        )

    return (
        "SEMANTICALLY_VALID_FREEZE_CANDIDATE",
        "Both members passed semantic measurement governance.",
    )


def representative_sort_frame(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    status_rank = {
        "STRONG_CONCEPT_CANDIDATE": 0,
        "CONCEPT_CANDIDATE": 1,
        "WEAK_CONCEPT_CANDIDATE": 2,
    }

    output[
        "_rank_status"
    ] = output[
        "concept_status"
    ].map(
        status_rank
    ).fillna(
        9
    )

    output[
        "_rank_q"
    ] = pd.to_numeric(
        output[
            "q_value"
        ],
        errors="coerce",
    ).fillna(
        np.inf
    )

    output[
        "_rank_incremental"
    ] = -pd.to_numeric(
        output[
            "incremental_lift_over_strongest_member"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_rank_lift"
    ] = -pd.to_numeric(
        output[
            "absolute_joint_lift"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_rank_sample"
    ] = -pd.to_numeric(
        output[
            "joint_active_sample"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    return output


def nominate_transformation_family_representatives(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = representative_sort_frame(
        dataframe
    )

    output = output.sort_values(
        [
            "target_name",
            "semantic_family_pair_key",
            "_rank_status",
            "_rank_q",
            "_rank_incremental",
            "_rank_lift",
            "_rank_sample",
            "concept_id",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
    )

    output[
        "semantic_family_rank"
    ] = (
        output.groupby(
            [
                "target_name",
                "semantic_family_pair_key",
            ],
            sort=False,
        )
        .cumcount()
        + 1
    )

    output[
        "semantic_family_representative"
    ] = output[
        "semantic_family_rank"
    ].eq(
        1
    )

    return output.drop(
        columns=[
            "_rank_status",
            "_rank_q",
            "_rank_incremental",
            "_rank_lift",
            "_rank_sample",
        ],
        errors="ignore",
    )
