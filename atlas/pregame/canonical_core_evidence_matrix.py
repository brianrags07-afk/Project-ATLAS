"""
Canonical core pregame evidence matrix for Project ATLAS.

This module joins governed team-game pregame sources without assigning
baseball meaning, predictive direction, weights, confidence, grades, or
probabilities.

Core sources:

- team-versus-opponent identity matchup facts,
- raw governed bullpen pregame facts,
- governed lineup-starter interaction facts.

Join grain:

    game_pk + team

Every source must be unique at that grain before joining.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

JOIN_KEYS: Final[tuple[str, ...]] = (
    "game_pk",
    "team",
)

CANONICAL_CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
)


def normalize_team_code(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.upper()
        .str.strip()
        .replace({
            "OAK": "ATH",
            "ARI": "AZ",
        })
    )

    return normalized


def normalize_source(
    dataframe: pd.DataFrame,
    source_name: str,
    season: int,
) -> pd.DataFrame:
    if dataframe.empty:
        raise ValueError(
            f"{source_name} source is empty."
        )

    required = {
        "game_pk",
        "team",
    }

    missing = sorted(
        required.difference(
            dataframe.columns
        )
    )

    if missing:
        raise KeyError(
            f"{source_name} is missing required keys: {missing}"
        )

    normalized = dataframe.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype(
        "int64"
    )

    normalized["team"] = normalize_team_code(
        normalized["team"]
    )

    if "opponent" in normalized.columns:
        normalized["opponent"] = normalize_team_code(
            normalized["opponent"]
        )

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(
            normalized["game_date"],
            errors="raise",
        ).dt.normalize()

    if "atlas_season" not in normalized.columns:
        normalized["atlas_season"] = int(
            season
        )

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype(
        "int64"
    )

    wrong_season = normalized[
        "atlas_season"
    ].ne(
        int(season)
    )

    if wrong_season.any():
        examples = normalized.loc[
            wrong_season,
            [
                column
                for column in [
                    "game_pk",
                    "game_date",
                    "atlas_season",
                    "team",
                ]
                if column in normalized.columns
            ],
        ].head(
            10
        )

        raise AssertionError(
            f"{source_name} contains rows outside season {season}:\n"
            + examples.to_string(
                index=False
            )
        )

    duplicate_count = int(
        normalized.duplicated(
            subset=list(
                JOIN_KEYS
            ),
            keep=False,
        ).sum()
    )

    if duplicate_count:
        duplicate_examples = normalized.loc[
            normalized.duplicated(
                subset=list(
                    JOIN_KEYS
                ),
                keep=False,
            ),
            [
                column
                for column in [
                    "game_pk",
                    "game_date",
                    "team",
                    "opponent",
                    "home_away",
                ]
                if column in normalized.columns
            ],
        ].sort_values(
            list(
                JOIN_KEYS
            ),
            kind="stable",
        ).head(
            30
        )

        raise AssertionError(
            f"{source_name} has duplicate team-game rows: "
            f"{duplicate_count:,}\n"
            + duplicate_examples.to_string(
                index=False
            )
        )

    return normalized.reset_index(
        drop=True
    )


def prefixed_feature_frame(
    dataframe: pd.DataFrame,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_columns = [
        column
        for column in CANONICAL_CONTEXT_COLUMNS
        if column in dataframe.columns
    ]

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in CANONICAL_CONTEXT_COLUMNS
    ]

    rename_map = {
        column:
            f"{prefix}__{column}"
        for column in feature_columns
    }

    prefixed = dataframe[
        context_columns
        + feature_columns
    ].rename(
        columns=rename_map
    )

    registry_rows = []

    for column in context_columns:
        registry_rows.append({
            "output_column":
                column,
            "source_column":
                column,
            "source_family":
                "canonical_context",
            "dtype":
                str(
                    dataframe[column].dtype
                ),
            "column_role":
                "CONTEXT",
        })

    for column in feature_columns:
        registry_rows.append({
            "output_column":
                rename_map[column],
            "source_column":
                column,
            "source_family":
                prefix,
            "dtype":
                str(
                    dataframe[column].dtype
                ),
            "column_role":
                "PREGAME_FACT",
        })

    registry = pd.DataFrame(
        registry_rows
    )

    return (
        prefixed,
        registry,
    )


def assert_context_agreement(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_source_name: str,
) -> pd.DataFrame:
    comparison_columns = [
        column
        for column in [
            "game_date",
            "atlas_season",
            "opponent",
            "home_away",
        ]
        if (
            column in left.columns
            and column in right.columns
        )
    ]

    right_context = right[
        list(
            JOIN_KEYS
        )
        + comparison_columns
    ].copy()

    rename_map = {
        column:
            f"__right_{column}"
        for column in comparison_columns
    }

    right_context = right_context.rename(
        columns=rename_map
    )

    comparison = left[
        list(
            JOIN_KEYS
        )
        + comparison_columns
    ].merge(
        right_context,
        on=list(
            JOIN_KEYS
        ),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    rows = []

    left_only = int(
        comparison[
            "_merge"
        ].eq(
            "left_only"
        ).sum()
    )

    right_only = int(
        comparison[
            "_merge"
        ].eq(
            "right_only"
        ).sum()
    )

    rows.append({
        "source":
            right_source_name,
        "check":
            "left-only team-games",
        "failures":
            left_only,
        "passed":
            left_only == 0,
    })

    rows.append({
        "source":
            right_source_name,
        "check":
            "right-only team-games",
        "failures":
            right_only,
        "passed":
            right_only == 0,
    })

    matched = comparison[
        comparison[
            "_merge"
        ].eq(
            "both"
        )
    ]

    for column in comparison_columns:
        left_values = matched[
            column
        ]

        right_values = matched[
            f"__right_{column}"
        ]

        both_missing = (
            left_values.isna()
            & right_values.isna()
        )

        equal = (
            left_values.eq(
                right_values
            )
            | both_missing
        )

        failures = int(
            (
                ~equal
            ).sum()
        )

        rows.append({
            "source":
                right_source_name,
            "check":
                f"context agreement: {column}",
            "failures":
                failures,
            "passed":
                failures == 0,
        })

    return pd.DataFrame(
        rows
    )


def add_missingness_indicators(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indicator_data = {}
    registry_rows = []

    for column in feature_columns:
        series = dataframe[
            column
        ]

        missing_rows = int(
            series.isna().sum()
        )

        if missing_rows == 0:
            continue

        indicator_column = (
            f"{column}__available"
        )

        indicator_data[
            indicator_column
        ] = series.notna().astype(
            "boolean"
        )

        registry_rows.append({
            "output_column":
                indicator_column,
            "source_column":
                column,
            "source_family":
                "missingness_contract",
            "dtype":
                "boolean",
            "column_role":
                "AVAILABILITY_INDICATOR",
        })

    if indicator_data:
        indicator_frame = pd.DataFrame(
            indicator_data,
            index=dataframe.index,
        )

        output = pd.concat(
            [
                dataframe,
                indicator_frame,
            ],
            axis=1,
        ).copy()

    else:
        output = dataframe.copy()

    return (
        output,
        pd.DataFrame(
            registry_rows
        ),
    )


def build_canonical_core_evidence(
    identity: pd.DataFrame,
    bullpen: pd.DataFrame,
    lineup_starter: pd.DataFrame,
    season: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    identity = normalize_source(
        identity,
        source_name="identity",
        season=season,
    )

    bullpen = normalize_source(
        bullpen,
        source_name="bullpen",
        season=season,
    )

    lineup_starter = normalize_source(
        lineup_starter,
        source_name="lineup_starter",
        season=season,
    )

    expected_rows = len(
        identity
    )

    expected_games = int(
        identity[
            "game_pk"
        ].nunique()
    )

    context_audits = [
        assert_context_agreement(
            identity,
            bullpen,
            right_source_name="bullpen",
        ),
        assert_context_agreement(
            identity,
            lineup_starter,
            right_source_name="lineup_starter",
        ),
    ]

    join_audit = pd.concat(
        context_audits,
        ignore_index=True,
    )

    if not join_audit[
        "passed"
    ].all():
        failed = join_audit[
            ~join_audit[
                "passed"
            ]
        ]

        raise AssertionError(
            "Source context or coverage mismatch:\n"
            + failed.to_string(
                index=False
            )
        )

    identity_prefixed, identity_registry = prefixed_feature_frame(
        identity,
        prefix="identity",
    )

    bullpen_prefixed, bullpen_registry = prefixed_feature_frame(
        bullpen,
        prefix="bullpen",
    )

    lineup_prefixed, lineup_registry = prefixed_feature_frame(
        lineup_starter,
        prefix="lineup_starter",
    )

    base_context = identity[
        [
            column
            for column in CANONICAL_CONTEXT_COLUMNS
            if column in identity.columns
        ]
    ].copy()

    identity_features = identity_prefixed.drop(
        columns=[
            column
            for column in CANONICAL_CONTEXT_COLUMNS
            if column in identity_prefixed.columns
        ]
    )

    identity_features = pd.concat(
        [
            identity[
                list(
                    JOIN_KEYS
                )
            ].reset_index(
                drop=True
            ),
            identity_features.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    bullpen_features = bullpen_prefixed.drop(
        columns=[
            column
            for column in CANONICAL_CONTEXT_COLUMNS
            if column in bullpen_prefixed.columns
            and column not in JOIN_KEYS
        ],
        errors="ignore",
    )

    lineup_features = lineup_prefixed.drop(
        columns=[
            column
            for column in CANONICAL_CONTEXT_COLUMNS
            if column in lineup_prefixed.columns
            and column not in JOIN_KEYS
        ],
        errors="ignore",
    )

    matrix = base_context.merge(
        identity_features,
        on=list(
            JOIN_KEYS
        ),
        how="left",
        validate="one_to_one",
    )

    matrix = matrix.merge(
        bullpen_features,
        on=list(
            JOIN_KEYS
        ),
        how="left",
        validate="one_to_one",
    )

    matrix = matrix.merge(
        lineup_features,
        on=list(
            JOIN_KEYS
        ),
        how="left",
        validate="one_to_one",
    )

    if len(
        matrix
    ) != expected_rows:
        raise AssertionError(
            f"Evidence matrix row count changed: "
            f"{len(matrix):,} != {expected_rows:,}"
        )

    if int(
        matrix[
            "game_pk"
        ].nunique()
    ) != expected_games:
        raise AssertionError(
            "Evidence matrix unique-game count changed."
        )

    duplicate_team_games = int(
        matrix.duplicated(
            subset=list(
                JOIN_KEYS
            )
        ).sum()
    )

    if duplicate_team_games:
        raise AssertionError(
            f"Evidence matrix duplicate team-games: "
            f"{duplicate_team_games:,}"
        )

    rows_per_game = matrix.groupby(
        "game_pk"
    ).size()

    invalid_game_rows = int(
        rows_per_game.ne(
            2
        ).sum()
    )

    if invalid_game_rows:
        raise AssertionError(
            f"Games without exactly two team rows: "
            f"{invalid_game_rows:,}"
        )

    feature_columns = [
        column
        for column in matrix.columns
        if column not in CANONICAL_CONTEXT_COLUMNS
    ]

    matrix, missingness_registry = add_missingness_indicators(
        matrix,
        feature_columns=feature_columns,
    )

    matrix[
        "evidence_matrix_version"
    ] = ENGINE_VERSION

    matrix[
        "strict_backtest_safe"
    ] = True

    matrix[
        "same_date_games_used"
    ] = False

    matrix[
        "future_games_used"
    ] = False

    matrix[
        "handcrafted_scores_included"
    ] = False

    matrix[
        "prediction_values_created"
    ] = False

    matrix[
        "market_used"
    ] = False

    matrix[
        "target_columns_included"
    ] = False

    matrix = matrix.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    context_registry = pd.DataFrame([
        {
            "output_column":
                column,
            "source_column":
                column,
            "source_family":
                "canonical_context",
            "dtype":
                str(
                    matrix[column].dtype
                ),
            "column_role":
                "CONTEXT",
        }
        for column in CANONICAL_CONTEXT_COLUMNS
        if column in matrix.columns
    ])

    source_registry = pd.concat(
        [
            identity_registry[
                identity_registry[
                    "column_role"
                ].eq(
                    "PREGAME_FACT"
                )
            ],
            bullpen_registry[
                bullpen_registry[
                    "column_role"
                ].eq(
                    "PREGAME_FACT"
                )
            ],
            lineup_registry[
                lineup_registry[
                    "column_role"
                ].eq(
                    "PREGAME_FACT"
                )
            ],
        ],
        ignore_index=True,
    )

    safety_registry = pd.DataFrame([
        {
            "output_column":
                "evidence_matrix_version",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "object",
            "column_role":
                "PROVENANCE",
        },
        {
            "output_column":
                "strict_backtest_safe",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
        {
            "output_column":
                "same_date_games_used",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
        {
            "output_column":
                "future_games_used",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
        {
            "output_column":
                "handcrafted_scores_included",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
        {
            "output_column":
                "prediction_values_created",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
        {
            "output_column":
                "market_used",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
        {
            "output_column":
                "target_columns_included",
            "source_column":
                None,
            "source_family":
                "matrix_governance",
            "dtype":
                "bool",
            "column_role":
                "SAFETY",
        },
    ])

    column_registry = pd.concat(
        [
            context_registry,
            source_registry,
            missingness_registry,
            safety_registry,
        ],
        ignore_index=True,
    )

    column_registry = column_registry.drop_duplicates(
        subset=[
            "output_column",
        ],
        keep="last",
    ).reset_index(
        drop=True
    )

    output_columns = set(
        matrix.columns
    )

    registry_columns = set(
        column_registry[
            "output_column"
        ]
    )

    missing_registry_columns = sorted(
        output_columns.difference(
            registry_columns
        )
    )

    if missing_registry_columns:
        extra_registry = pd.DataFrame([
            {
                "output_column":
                    column,
                "source_column":
                    None,
                "source_family":
                    "derived_or_existing_safety",
                "dtype":
                    str(
                        matrix[column].dtype
                    ),
                "column_role":
                    "REVIEWED_OUTPUT",
            }
            for column in missing_registry_columns
        ])

        column_registry = pd.concat(
            [
                column_registry,
                extra_registry,
            ],
            ignore_index=True,
        )

    row_audit = matrix[
        [
            column
            for column in CANONICAL_CONTEXT_COLUMNS
            if column in matrix.columns
        ]
    ].copy()

    row_audit[
        "identity_joined"
    ] = True

    row_audit[
        "bullpen_joined"
    ] = True

    row_audit[
        "lineup_starter_joined"
    ] = True

    row_audit[
        "evidence_feature_count"
    ] = len(
        feature_columns
    )

    row_audit[
        "missing_evidence_values"
    ] = matrix[
        feature_columns
    ].isna().sum(
        axis=1
    )

    row_audit[
        "complete_evidence_values"
    ] = (
        len(
            feature_columns
        )
        - row_audit[
            "missing_evidence_values"
        ]
    )

    return (
        matrix,
        column_registry,
        join_audit,
        row_audit,
    )
