"""
Governed starter, pitcher and lineup pregame adapters for Project ATLAS.

The adapters preserve raw or historical pregame measurements and remove:

- explicit same-game outcomes,
- historical targets,
- handcrafted scores,
- ratings,
- tiers,
- labels,
- probability fields,
- prediction fields,
- ambiguous outcome-like fields without clear lagged provenance.

The module does not assign baseball meaning or predictive direction.
"""

from __future__ import annotations

from typing import Final
import re

import pandas as pd


ADAPTER_VERSION: Final[str] = "1.0.0"

KEY_COLUMNS: Final[frozenset[str]] = frozenset({
    "game_pk",
    "game_date",
    "official_date",
    "date",
    "team",
    "opponent",
    "home_away",
    "atlas_season",
    "season",
    "batter",
    "batter_id",
    "pitcher",
    "pitcher_id",
    "starter_id",
    "starting_pitcher_id",
    "player_id",
    "batting_order",
    "lineup_slot",
})

SAFETY_COLUMNS: Final[frozenset[str]] = frozenset({
    "strict_backtest_safe",
    "same_date_games_used",
    "future_games_used",
    "future_games_excluded",
    "current_game_used",
    "market_used",
    "prediction_used",
})

TRUE_LEAKAGE_PATTERNS: Final[tuple[str, ...]] = (
    r"(^|_)actual_",
    r"(^|_)final_",
    r"(^|_)winner($|_)",
    r"(^|_)game_result($|_)",
    r"(^|_)target($|_)",
    r"(^|_)target_",
    r"(^|_)outcome($|_)",
    r"(^|_)covered($|_)",
    r"(^|_)home_win($|_)",
    r"(^|_)away_win($|_)",
    r"(^|_)team_win($|_)",
    r"(^|_)team_loss($|_)",
    r"(^|_)total_runs($|_)",
    r"(^|_)final_score($|_)",
)

OUTCOME_WORD_PATTERNS: Final[tuple[str, ...]] = (
    r"run(?:s)?_allowed",
    r"run(?:s)?_scored",
    r"hit(?:s)?_allowed",
    r"strikeout(?:s)?",
    r"walk(?:s)?",
    r"home_run(?:s)?",
    r"inning(?:s)?_pitched",
    r"batter(?:s)?_faced",
    r"win(?:s)?",
    r"loss(?:es)?",
)

HISTORICAL_MARKERS: Final[tuple[str, ...]] = (
    "prior",
    "previous",
    "pregame",
    "before",
    "rolling",
    "expanding",
    "career",
    "season_to_date",
    "last_",
    "lag",
    "history",
    "historical",
    "avg",
    "mean",
    "rate",
    "pct",
    "per_",
)

HANDCRAFTED_PATTERNS: Final[tuple[str, ...]] = (
    r"(^|_)confidence_score($|_)",
    r"(^|_)matchup_score($|_)",
    r"(^|_)starter_score($|_)",
    r"(^|_)lineup_score($|_)",
    r"(^|_)edge_score($|_)",
    r"(^|_)quality_score($|_)",
    r"(^|_)strength_score($|_)",
    r"(^|_)rating($|_)",
    r"(^|_)grade($|_)",
    r"(^|_)tier($|_)",
    r"(^|_)label($|_)",
    r"(^|_)probability($|_)",
    r"(^|_)predicted($|_)",
    r"(^|_)prediction($|_)",
    r"(^|_)adjustment($|_)",
    r"(^|_)bonus($|_)",
    r"(^|_)penalty($|_)",
    r"(^|_)multiplier($|_)",
    r"(^|_)weighted_score($|_)",
    r"(^|_)composite_score($|_)",
)


def _matches_any(
    value: str,
    patterns: tuple[str, ...],
) -> bool:
    value = str(
        value
    ).lower()

    return any(
        re.search(
            pattern,
            value,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def _has_historical_marker(
    column_name: str,
) -> bool:
    lower = str(
        column_name
    ).lower()

    return any(
        marker in lower
        for marker in HISTORICAL_MARKERS
    )


def classify_column(
    column_name: str,
) -> tuple[str, str]:
    name = str(
        column_name
    )

    lower = name.lower()

    if lower in KEY_COLUMNS:
        return (
            "KEEP_KEY",
            "Canonical identifier.",
        )

    if lower in SAFETY_COLUMNS:
        return (
            "KEEP_SAFETY",
            "Chronology or provenance control.",
        )

    if _matches_any(
        lower,
        TRUE_LEAKAGE_PATTERNS,
    ):
        return (
            "BLOCK_POSTGAME",
            "Same-game outcome or target.",
        )

    if _matches_any(
        lower,
        HANDCRAFTED_PATTERNS,
    ):
        return (
            "BLOCK_HANDCRAFTED",
            "Handcrafted decision field.",
        )

    if _matches_any(
        lower,
        OUTCOME_WORD_PATTERNS,
    ):
        if _has_historical_marker(
            lower
        ):
            return (
                "KEEP_HISTORICAL_FACT",
                "Lagged or historical pregame measurement.",
            )

        return (
            "MANUAL_REVIEW",
            "Outcome-like field lacks explicit historical provenance.",
        )

    return (
        "KEEP_RAW_OR_NEUTRAL",
        "No prohibited marker detected.",
    )


def clean_pregame_artifact(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe.empty:
        raise ValueError(
            "Pregame dataframe is empty."
        )

    registry_rows = []

    for column in dataframe.columns:
        action, reason = classify_column(
            str(
                column
            )
        )

        registry_rows.append({
            "column":
                str(
                    column
                ),

            "dtype":
                str(
                    dataframe[
                        column
                    ].dtype
                ),

            "governance_action":
                action,

            "reason":
                reason,
        })

    registry = pd.DataFrame(
        registry_rows
    )

    blocked = {
        "BLOCK_POSTGAME",
        "BLOCK_HANDCRAFTED",
        "MANUAL_REVIEW",
    }

    keep_columns = registry.loc[
        ~registry[
            "governance_action"
        ].isin(
            blocked
        ),
        "column",
    ].tolist()

    clean = dataframe[
        keep_columns
    ].copy()

    if "game_pk" in clean.columns:
        clean[
            "game_pk"
        ] = pd.to_numeric(
            clean[
                "game_pk"
            ],
            errors="raise",
        ).astype(
            "int64"
        )

    if "game_date" in clean.columns:
        clean[
            "game_date"
        ] = pd.to_datetime(
            clean[
                "game_date"
            ],
            errors="raise",
        ).dt.normalize()

    if "atlas_season" not in clean.columns:
        if "game_date" not in clean.columns:
            raise KeyError(
                "Cannot derive atlas_season without game_date."
            )

        clean[
            "atlas_season"
        ] = (
            clean[
                "game_date"
            ]
            .dt.year
            .astype(
                "int64"
            )
        )

    clean[
        "strict_backtest_safe"
    ] = True

    clean[
        "same_date_games_used"
    ] = False

    clean[
        "future_games_used"
    ] = False

    clean[
        "handcrafted_scores_included"
    ] = False

    clean[
        "postgame_columns_included"
    ] = False

    clean[
        "clean_adapter_version"
    ] = ADAPTER_VERSION

    return (
        clean,
        registry,
    )
