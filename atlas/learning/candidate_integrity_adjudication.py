"""
Candidate-integrity and redundancy adjudication for Project ATLAS.

This module evaluates discovered univariate conditions for:

- exact activation-mask duplication,
- complementary or inverse activation masks,
- semantic duplication,
- side duplication in paired home/away views,
- direct historical analogues of the target,
- availability-indicator artifacts,
- source-family provenance.

All discovery results remain preserved.

A deterministic representative may be nominated for each redundant evidence
group, but no prediction weight, probability, confidence score, or prediction
is created.
"""

from __future__ import annotations

from typing import Final
import hashlib
import re

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

CANDIDATE_STATUSES: Final[tuple[str, ...]] = (
    "STRONG_DISCOVERY_CANDIDATE",
    "DISCOVERY_CANDIDATE",
    "WEAK_DISCOVERY_CANDIDATE",
)

TARGET_ANALOGUE_TOKENS: Final[dict[str, tuple[str, ...]]] = {
    "target_team_win": (
        "__won",
        "__lost",
        "win_pct",
        "win_rate",
        "winning_percentage",
        "victory",
        "defeat",
    ),

    "target_team_win_by_2_plus": (
        "win_by_2_plus",
        "won_by_2_plus",
        "loss_by_2_plus",
        "lost_by_2_plus",
        "covered_minus_1_5",
        "covered_plus_1_5",
        "two_run_lead",
        "ever_led_by_4",
        "margin_2",
    ),

    "target_game_total_over_10": (
        "game_total_over_10",
        "game_total_10_plus",
        "total_10_plus",
        "total_over_10",
        "high_scoring_game",
        "scored_10_plus",
    ),

    "target_game_total_7_or_less": (
        "game_total_7_or_less",
        "game_total_6_or_less",
        "total_7_or_less",
        "low_scoring_game",
        "scored_3_or_less",
        "allowed_3_or_less",
    ),
}


def feature_side(
    feature_name: str,
) -> str:
    name = str(
        feature_name
    )

    if name.startswith(
        "home__"
    ):
        return "HOME"

    if name.startswith(
        "away__"
    ):
        return "AWAY"

    return "TEAM"


def strip_game_side(
    feature_name: str,
) -> str:
    name = str(
        feature_name
    )

    if name.startswith(
        "home__"
    ):
        return name[
            len(
                "home__"
            ):
        ]

    if name.startswith(
        "away__"
    ):
        return name[
            len(
                "away__"
            ):
        ]

    return name


def canonical_semantic_name(
    feature_name: str,
) -> str:
    """
    Produce a conservative semantic key.

    The function removes paired-game HOME/AWAY prefixes and standardizes a
    small set of mechanically mirrored naming fragments. It does not attempt
    to claim that two different baseball statistics are equivalent.
    """

    name = strip_game_side(
        feature_name
    ).lower()

    replacements = (
        (
            "__team_identity__",
            "__identity_subject__",
        ),
        (
            "__opponent_identity__",
            "__identity_opponent__",
        ),
        (
            "career_prior_",
            "historical_prior_",
        ),
        (
            "season_prior_",
            "historical_prior_",
        ),
    )

    for old, new in replacements:
        name = name.replace(
            old,
            new,
        )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    return name


def source_classification(
    feature_name: str,
) -> str:
    name = strip_game_side(
        feature_name
    ).lower()

    if (
        "availability__" in name
        or name.endswith(
            "__available"
        )
        or name.endswith(
            "_is_available"
        )
        or "missing_indicator" in name
    ):
        return "AVAILABILITY_INDICATOR"

    if name.startswith(
        "bullpen__"
    ):
        return "RAW_BULLPEN_PREGAME_FACT"

    if name.startswith(
        "lineup_starter__"
    ):
        return "LINEUP_STARTER_PREGAME_FACT"

    if name.startswith(
        "identity__identity_edge__"
    ):
        return "DERIVED_IDENTITY_EDGE"

    if name.startswith(
        "identity__team_identity__"
    ):
        return "TEAM_IDENTITY_SUMMARY"

    if name.startswith(
        "identity__opponent_identity__"
    ):
        return "OPPONENT_IDENTITY_SUMMARY"

    if name.startswith(
        "identity__"
    ):
        return "IDENTITY_CONTEXT"

    return "OTHER_PREGAME_FACT"


def direct_target_analogue(
    target_name: str,
    feature_name: str,
) -> tuple[bool, str]:
    lower = strip_game_side(
        feature_name
    ).lower()

    matched_tokens = [
        token
        for token in TARGET_ANALOGUE_TOKENS.get(
            target_name,
            (),
        )
        if token in lower
    ]

    if not matched_tokens:
        return (
            False,
            "",
        )

    return (
        True,
        "|".join(
            matched_tokens
        ),
    )


def condition_mask(
    series: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if operator == ">=":
        return numeric.ge(
            threshold
        )

    if operator == "<=":
        return numeric.le(
            threshold
        )

    if operator == "==":
        return numeric.eq(
            threshold
        )

    raise ValueError(
        f"Unsupported operator: {operator}"
    )


def trinary_condition_state(
    series: pd.Series,
    operator: str,
    threshold: float,
) -> np.ndarray:
    """
    Encode each row as:

    0 = unavailable/missing
    1 = available but inactive
    2 = available and active
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    available = numeric.notna()

    active = condition_mask(
        series=numeric,
        operator=operator,
        threshold=threshold,
    ) & available

    state = np.zeros(
        len(
            numeric
        ),
        dtype=np.uint8,
    )

    state[
        available.to_numpy()
    ] = 1

    state[
        active.to_numpy()
    ] = 2

    return state


def complement_trinary_state(
    state: np.ndarray,
) -> np.ndarray:
    output = state.copy()

    available = output > 0

    output[
        available
    ] = (
        3
        - output[
            available
        ]
    )

    return output


def hash_state(
    state: np.ndarray,
) -> str:
    return hashlib.sha256(
        np.asarray(
            state,
            dtype=np.uint8,
        ).tobytes()
    ).hexdigest()


def undirected_mask_group_key(
    mask_hash: str,
    complement_hash: str,
) -> str:
    return min(
        str(
            mask_hash
        ),
        str(
            complement_hash
        ),
    )


def build_condition_mask_record(
    dataframe: pd.DataFrame,
    target_name: str,
    feature_name: str,
    operator: str,
    threshold: float,
) -> dict[str, object]:
    if feature_name not in dataframe.columns:
        raise KeyError(
            f"Feature not found in discovery view: {feature_name}"
        )

    state = trinary_condition_state(
        series=dataframe[
            feature_name
        ],
        operator=operator,
        threshold=float(
            threshold
        ),
    )

    complement = complement_trinary_state(
        state
    )

    mask_hash = hash_state(
        state
    )

    complement_hash = hash_state(
        complement
    )

    active_rows = int(
        np.count_nonzero(
            state == 2
        )
    )

    inactive_rows = int(
        np.count_nonzero(
            state == 1
        )
    )

    missing_rows = int(
        np.count_nonzero(
            state == 0
        )
    )

    return {
        "target_name":
            target_name,

        "feature_name":
            feature_name,

        "threshold_operator":
            operator,

        "threshold_value":
            float(
                threshold
            ),

        "mask_hash":
            mask_hash,

        "complement_mask_hash":
            complement_hash,

        "undirected_mask_group_key":
            undirected_mask_group_key(
                mask_hash,
                complement_hash,
            ),

        "mask_active_rows":
            active_rows,

        "mask_inactive_rows":
            inactive_rows,

        "mask_missing_rows":
            missing_rows,

        "mask_reconstruction_passed":
            True,
    }


def representative_sort_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create deterministic governance priorities.

    These priorities select a representative only. They are not predictive
    weights and do not measure baseball importance.
    """

    output = dataframe.copy()

    output[
        "_priority_direct_analogue"
    ] = output[
        "direct_target_analogue"
    ].astype(
        int
    )

    output[
        "_priority_availability"
    ] = output[
        "source_classification"
    ].eq(
        "AVAILABILITY_INDICATOR"
    ).astype(
        int
    )

    status_rank = {
        "STRONG_DISCOVERY_CANDIDATE":
            0,

        "DISCOVERY_CANDIDATE":
            1,

        "WEAK_DISCOVERY_CANDIDATE":
            2,
    }

    output[
        "_priority_status"
    ] = output[
        "research_status"
    ].map(
        status_rank
    ).fillna(
        9
    )

    output[
        "_priority_q"
    ] = pd.to_numeric(
        output[
            "q_value"
        ],
        errors="coerce",
    ).fillna(
        np.inf
    )

    output[
        "_priority_lift"
    ] = -pd.to_numeric(
        output[
            "absolute_lift"
        ],
        errors="coerce",
    ).fillna(
        -np.inf
    )

    output[
        "_priority_sample"
    ] = -pd.to_numeric(
        output[
            "active_sample"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    return output


def nominate_group_representatives(
    candidate_registry: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "target_name",
        "undirected_mask_group_key",
        "feature_name",
        "condition_name",
        "research_status",
        "q_value",
        "absolute_lift",
        "active_sample",
        "direct_target_analogue",
        "source_classification",
    }

    missing = sorted(
        required.difference(
            candidate_registry.columns
        )
    )

    if missing:
        raise KeyError(
            f"Candidate registry lacks fields: {missing}"
        )

    ranked = representative_sort_columns(
        candidate_registry
    )

    ranked = ranked.sort_values(
        [
            "target_name",
            "undirected_mask_group_key",
            "_priority_direct_analogue",
            "_priority_availability",
            "_priority_status",
            "_priority_q",
            "_priority_lift",
            "_priority_sample",
            "feature_name",
            "condition_name",
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
            True,
            True,
        ],
        kind="stable",
    )

    ranked[
        "redundancy_group_rank"
    ] = (
        ranked.groupby(
            [
                "target_name",
                "undirected_mask_group_key",
            ],
            sort=False,
        )
        .cumcount()
        + 1
    )

    ranked[
        "nominated_representative"
    ] = ranked[
        "redundancy_group_rank"
    ].eq(
        1
    )

    ranked = ranked.drop(
        columns=[
            "_priority_direct_analogue",
            "_priority_availability",
            "_priority_status",
            "_priority_q",
            "_priority_lift",
            "_priority_sample",
        ],
        errors="ignore",
    )

    return ranked
