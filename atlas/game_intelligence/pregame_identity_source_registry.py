"""
Phase 2E.1 pregame identity source registry builder for Project ATLAS.

This module governs candidate Phase 2D identity columns and emits a deterministic
registry of lagged pregame-safe identity sources for Phase 2E chronology work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import json
import re

import pandas as pd

from atlas.config import DATA_ROOT


ENGINE_VERSION: Final[str] = "1.0.0"

JOIN_KEYS: Final[tuple[str, ...]] = (
    "game_pk",
    "team",
)

CONTEXT_COLUMNS: Final[frozenset[str]] = frozenset({
    "game_pk",
    "game_date",
    "official_date",
    "date",
    "team",
    "opponent",
    "home_away",
    "atlas_season",
    "season",
})

SAFETY_COLUMNS: Final[frozenset[str]] = frozenset({
    "strict_backtest_safe",
    "same_date_games_used",
    "future_games_used",
    "future_games_excluded",
    "current_game_used",
})

LAGGED_MARKER_PATTERNS: Final[tuple[str, ...]] = (
    r"(^|_)prior_",
    r"(^|_)previous_",
    r"(^|_)rolling_",
    r"(^|_)lag",
    r"(^|_)historical",
    r"(^|_)pregame_",
    r"(^|_)season_to_date",
    r"(^|_)career_",
)

LEAKAGE_PATTERNS: Final[tuple[str, ...]] = (
    r"(^|_)actual_",
    r"(^|_)final_",
    r"(^|_)winner($|_)",
    r"(^|_)target($|_)",
    r"(^|_)target_",
    r"(^|_)outcome($|_)",
    r"(^|_)prediction($|_)",
    r"(^|_)probability($|_)",
)


def _matches_any(
    value: str,
    patterns: tuple[str, ...],
) -> bool:
    lowered = str(value).lower()
    return any(
        re.search(
            pattern,
            lowered,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


def _is_candidate_identity_column(
    column_name: str,
    series: pd.Series,
) -> bool:
    name = str(column_name).lower()

    if name in CONTEXT_COLUMNS or name in SAFETY_COLUMNS:
        return False

    if _matches_any(
        name,
        LEAKAGE_PATTERNS,
    ):
        return False

    return (
        _matches_any(
            name,
            LAGGED_MARKER_PATTERNS,
        )
        and pd.api.types.is_numeric_dtype(series)
    )


def normalize_phase_2d_identity_inputs(
    dataframe: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError(
            "Phase 2D identity source dataframe is empty."
        )

    missing = sorted(
        set(JOIN_KEYS).difference(
            dataframe.columns
        )
    )
    if missing:
        raise KeyError(
            f"Missing required team-game keys: {missing}"
        )

    normalized = dataframe.copy()
    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["team"] = (
        normalized["team"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(
            normalized["game_date"],
            errors="raise",
        ).dt.normalize()

    if "atlas_season" not in normalized.columns:
        normalized["atlas_season"] = int(season)

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    wrong_season = normalized[
        "atlas_season"
    ].ne(int(season))
    if wrong_season.any():
        raise AssertionError(
            f"Phase 2D inputs contain rows outside season {season}."
        )

    duplicate_count = int(
        normalized.duplicated(
            subset=list(JOIN_KEYS),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise AssertionError(
            f"Phase 2D inputs contain duplicate team-game rows: {duplicate_count:,}"
        )

    return normalized.reset_index(
        drop=True
    )


def build_pregame_identity_source_registry(
    phase_2d_identity_frame: pd.DataFrame,
    season: int,
    expected_source_count: int = 87,
    approved_columns: list[str] | None = None,
) -> pd.DataFrame:
    normalized = normalize_phase_2d_identity_inputs(
        phase_2d_identity_frame,
        season=season,
    )

    if approved_columns is not None:
        selected_columns = [
            str(column)
            for column in approved_columns
        ]
    else:
        candidate_columns = [
            column
            for column in normalized.columns
            if _is_candidate_identity_column(
                column,
                normalized[column],
            )
        ]

        ranked_candidates = sorted(
            candidate_columns,
            key=lambda column: (
                -float(
                    normalized[column]
                    .notna()
                    .mean()
                ),
                str(column).lower(),
            ),
        )

        selected_columns = ranked_candidates[
            :int(expected_source_count)
        ]

    missing_columns = sorted(
        set(selected_columns).difference(
            normalized.columns
        )
    )
    if missing_columns:
        raise KeyError(
            f"Approved identity columns are missing from source frame: {missing_columns}"
        )

    if len(selected_columns) != int(expected_source_count):
        raise AssertionError(
            f"Expected {expected_source_count} identity sources, "
            f"found {len(selected_columns)}."
        )

    registry = pd.DataFrame({
        "identity_feature_name": selected_columns,
        "source_column": selected_columns,
        "min_lagged_days": 1,
        "same_game_source": False,
        "future_games_used": False,
        "atlas_season": int(season),
        "phase": "2E.1",
        "registry_engine_version": ENGINE_VERSION,
    })

    registry = registry.assign(
        source_non_null_rate=[
            float(
                normalized[column]
                .notna()
                .mean()
            )
            for column in selected_columns
        ],
        source_rank=range(
            1,
            len(selected_columns) + 1,
        ),
    )

    validate_pregame_identity_source_registry(
        registry,
        expected_source_count=expected_source_count,
    )

    return registry.reset_index(
        drop=True
    )


def validate_pregame_identity_source_registry(
    registry: pd.DataFrame,
    expected_source_count: int = 87,
) -> None:
    if registry.empty:
        raise ValueError(
            "Pregame identity source registry is empty."
        )

    if len(registry) != int(expected_source_count):
        raise AssertionError(
            f"Registry row count mismatch: expected {expected_source_count}, got {len(registry)}."
        )

    if not registry[
        "identity_feature_name"
    ].is_unique:
        raise AssertionError(
            "Identity feature names must be unique."
        )

    if registry["same_game_source"].any():
        raise AssertionError(
            "Same-game identity sources are not allowed."
        )

    if registry["future_games_used"].any():
        raise AssertionError(
            "Future games cannot be used for pregame identity sources."
        )

    if (
        registry[
            "min_lagged_days"
        ] <= 0
    ).any():
        raise AssertionError(
            "min_lagged_days must be positive for all sources."
        )


def phase_2e_identity_source_registry_paths(
    season: int,
    data_root: Path | None = None,
) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    base = (
        root
        / "game_intelligence"
        / "pregame_identity_registry"
        / str(int(season))
    )

    return {
        "base_dir": base,
        "registry_csv": base / "pregame_identity_source_registry.csv",
        "metadata_json": base / "pregame_identity_source_registry_metadata.json",
    }


def build_pregame_identity_source_registry_metadata(
    registry: pd.DataFrame,
    season: int,
) -> dict[str, object]:
    validate_pregame_identity_source_registry(
        registry,
        expected_source_count=len(registry),
    )

    return {
        "phase": "2E.1",
        "season": int(season),
        "source_count": int(len(registry)),
        "same_game_sources": int(
            registry["same_game_source"].sum()
        ),
        "future_games_used_sources": int(
            registry["future_games_used"].sum()
        ),
        "registry_engine_version": ENGINE_VERSION,
    }


def save_pregame_identity_source_registry(
    registry: pd.DataFrame,
    season: int,
    data_root: Path | None = None,
) -> dict[str, Path]:
    metadata = build_pregame_identity_source_registry_metadata(
        registry,
        season=season,
    )
    paths = phase_2e_identity_source_registry_paths(
        season=season,
        data_root=data_root,
    )

    paths["base_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )
    registry.to_csv(
        paths["registry_csv"],
        index=False,
    )
    paths["metadata_json"].write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return paths
