"""
Target-isolated controlled discovery views for Project ATLAS.

Two discovery grains are supported:

1. Team-game grain for team-perspective outcomes such as:
   - team win,
   - team win by two or more,
   - team win by four or more.

2. Game grain for shared game outcomes such as:
   - total over ten,
   - total ten or more,
   - total seven or fewer.

Game-level views pair the HOME and AWAY evidence into one row so shared
game-total outcomes are not duplicated once for each team perspective.

This module creates discovery datasets only. It does not discover rules,
assign weights, create predictions, or modify canonical evidence.
"""

from __future__ import annotations

from typing import Final

import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
)

ALLOWED_FEATURE_ROLES: Final[tuple[str, ...]] = (
    "PREGAME_FACT",
    "AVAILABILITY_INDICATOR",
)


def normalize_team_code(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
        .replace({
            "OAK": "ATH",
            "ARI": "AZ",
        })
    )


def select_governed_feature_columns(
    evidence: pd.DataFrame,
    column_registry: pd.DataFrame,
) -> list[str]:
    required_registry_columns = {
        "output_column",
        "column_role",
    }

    missing_registry_columns = sorted(
        required_registry_columns.difference(
            column_registry.columns
        )
    )

    if missing_registry_columns:
        raise KeyError(
            "Evidence column registry is missing fields: "
            f"{missing_registry_columns}"
        )

    allowed = column_registry[
        column_registry[
            "column_role"
        ].isin(
            ALLOWED_FEATURE_ROLES
        )
    ][
        "output_column"
    ].astype(
        "string"
    ).tolist()

    selected = [
        column
        for column in allowed
        if column in evidence.columns
    ]

    selected = list(
        dict.fromkeys(
            selected
        )
    )

    if not selected:
        raise AssertionError(
            "No governed pregame features were selected."
        )

    forbidden_tokens = (
        "target_",
        "actual_",
        "final_score",
        "prediction_probability",
        "prediction_value",
    )

    forbidden = [
        column
        for column in selected
        if any(
            token in column.lower()
            for token in forbidden_tokens
        )
    ]

    if forbidden:
        raise AssertionError(
            "Forbidden outcome or prediction fields entered "
            f"the feature set: {forbidden[:30]}"
        )

    return selected


def normalize_inputs(
    evidence: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_evidence = set(
        CONTEXT_COLUMNS
    )

    required_targets = set(
        CONTEXT_COLUMNS
    )

    missing_evidence = sorted(
        required_evidence.difference(
            evidence.columns
        )
    )

    missing_targets = sorted(
        required_targets.difference(
            targets.columns
        )
    )

    if missing_evidence:
        raise KeyError(
            f"Evidence is missing context fields: {missing_evidence}"
        )

    if missing_targets:
        raise KeyError(
            f"Targets are missing context fields: {missing_targets}"
        )

    evidence = evidence.copy()
    targets = targets.copy()

    for dataframe in [
        evidence,
        targets,
    ]:
        dataframe[
            "game_pk"
        ] = pd.to_numeric(
            dataframe[
                "game_pk"
            ],
            errors="raise",
        ).astype(
            "int64"
        )

        dataframe[
            "game_date"
        ] = pd.to_datetime(
            dataframe[
                "game_date"
            ],
            errors="raise",
        ).dt.normalize()

        dataframe[
            "atlas_season"
        ] = pd.to_numeric(
            dataframe[
                "atlas_season"
            ],
            errors="raise",
        ).astype(
            "int64"
        )

        dataframe[
            "team"
        ] = normalize_team_code(
            dataframe[
                "team"
            ]
        )

        dataframe[
            "opponent"
        ] = normalize_team_code(
            dataframe[
                "opponent"
            ]
        )

        dataframe[
            "home_away"
        ] = (
            dataframe[
                "home_away"
            ]
            .astype("string")
            .str.upper()
            .str.strip()
        )

    for name, dataframe in [
        (
            "evidence",
            evidence,
        ),
        (
            "targets",
            targets,
        ),
    ]:
        duplicate_count = int(
            dataframe.duplicated(
                subset=[
                    "game_pk",
                    "team",
                ]
            ).sum()
        )

        if duplicate_count:
            raise AssertionError(
                f"{name} contains duplicate team-games: "
                f"{duplicate_count:,}"
            )

    return (
        evidence,
        targets,
    )


def add_row_coverage(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    output = dataframe.copy()

    output[
        "discovery_feature_count"
    ] = len(
        feature_columns
    )

    output[
        "discovery_available_feature_count"
    ] = output[
        feature_columns
    ].notna().sum(
        axis=1
    )

    output[
        "discovery_missing_feature_count"
    ] = output[
        feature_columns
    ].isna().sum(
        axis=1
    )

    output[
        "discovery_feature_coverage_rate"
    ] = (
        output[
            "discovery_available_feature_count"
        ]
        / float(
            len(
                feature_columns
            )
        )
    )

    return output


def build_team_game_discovery_view(
    evidence: pd.DataFrame,
    targets: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if target_column not in targets.columns:
        raise KeyError(
            f"Target column not found: {target_column}"
        )

    if not pd.api.types.is_bool_dtype(
        targets[
            target_column
        ]
    ):
        raise TypeError(
            f"Target must be boolean: {target_column}"
        )

    target_context = targets[
        list(
            CONTEXT_COLUMNS
        )
        + [
            target_column,
        ]
    ].copy()

    view = evidence[
        list(
            CONTEXT_COLUMNS
        )
        + feature_columns
    ].merge(
        target_context,
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
            "opponent",
            "home_away",
        ],
        how="inner",
        validate="one_to_one",
    )

    view = view.rename(
        columns={
            target_column:
                "target_label",
        }
    )

    view = add_row_coverage(
        view,
        feature_columns=feature_columns,
    )

    view[
        "target_name"
    ] = target_column

    view[
        "discovery_grain"
    ] = "TEAM_GAME"

    view[
        "strict_backtest_safe"
    ] = True

    view[
        "canonical_evidence_modified"
    ] = False

    view[
        "prediction_created"
    ] = False

    view[
        "weight_assigned"
    ] = False

    view[
        "discovery_view_version"
    ] = ENGINE_VERSION

    view = view.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    if view.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).any():
        raise AssertionError(
            "Team-game discovery view contains duplicate rows."
        )

    return view


def build_game_discovery_view(
    evidence: pd.DataFrame,
    targets: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    if target_column not in targets.columns:
        raise KeyError(
            f"Target column not found: {target_column}"
        )

    if not pd.api.types.is_bool_dtype(
        targets[
            target_column
        ]
    ):
        raise TypeError(
            f"Target must be boolean: {target_column}"
        )

    target_consistency = (
        targets.groupby(
            "game_pk"
        )[
            target_column
        ]
        .nunique(
            dropna=False
        )
    )

    inconsistent_games = int(
        target_consistency.ne(
            1
        ).sum()
    )

    if inconsistent_games:
        raise AssertionError(
            f"Game target is inconsistent across team rows: "
            f"{inconsistent_games:,}"
        )

    home = evidence[
        evidence[
            "home_away"
        ].eq(
            "HOME"
        )
    ][
        list(
            CONTEXT_COLUMNS
        )
        + feature_columns
    ].copy()

    away = evidence[
        evidence[
            "home_away"
        ].eq(
            "AWAY"
        )
    ][
        list(
            CONTEXT_COLUMNS
        )
        + feature_columns
    ].copy()

    home_rename = {
        "team":
            "home_team",

        "opponent":
            "away_team_from_home_row",

        "home_away":
            "home_row_home_away",
    }

    away_rename = {
        "team":
            "away_team",

        "opponent":
            "home_team_from_away_row",

        "home_away":
            "away_row_home_away",
    }

    for column in feature_columns:
        home_rename[
            column
        ] = (
            f"home__{column}"
        )

        away_rename[
            column
        ] = (
            f"away__{column}"
        )

    home = home.rename(
        columns=home_rename
    )

    away = away.rename(
        columns=away_rename
    )

    away = away.drop(
        columns=[
            "game_date",
            "atlas_season",
        ]
    )

    paired = home.merge(
        away,
        on=[
            "game_pk",
        ],
        how="inner",
        validate="one_to_one",
    )

    opponent_mismatch = (
        paired[
            "away_team_from_home_row"
        ].ne(
            paired[
                "away_team"
            ]
        )
        | paired[
            "home_team_from_away_row"
        ].ne(
            paired[
                "home_team"
            ]
        )
    )

    if opponent_mismatch.any():
        raise AssertionError(
            "Home/away opponent pairing mismatch detected."
        )

    game_targets = (
        targets[
            [
                "game_pk",
                target_column,
            ]
        ]
        .drop_duplicates()
    )

    if game_targets.duplicated(
        subset=[
            "game_pk",
        ]
    ).any():
        raise AssertionError(
            "Game target table remains duplicated after deduplication."
        )

    view = paired.merge(
        game_targets,
        on=[
            "game_pk",
        ],
        how="inner",
        validate="one_to_one",
    )

    view = view.rename(
        columns={
            target_column:
                "target_label",
        }
    )

    home_features = [
        f"home__{column}"
        for column in feature_columns
    ]

    away_features = [
        f"away__{column}"
        for column in feature_columns
    ]

    paired_features = (
        home_features
        + away_features
    )

    view = add_row_coverage(
        view,
        feature_columns=paired_features,
    )

    view[
        "target_name"
    ] = target_column

    view[
        "discovery_grain"
    ] = "GAME"

    view[
        "strict_backtest_safe"
    ] = True

    view[
        "shared_game_target_counted_once"
    ] = True

    view[
        "canonical_evidence_modified"
    ] = False

    view[
        "prediction_created"
    ] = False

    view[
        "weight_assigned"
    ] = False

    view[
        "discovery_view_version"
    ] = ENGINE_VERSION

    view = view.drop(
        columns=[
            "away_team_from_home_row",
            "home_team_from_away_row",
            "home_row_home_away",
            "away_row_home_away",
        ],
        errors="ignore",
    )

    view = view.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    if view.duplicated(
        subset=[
            "game_pk",
        ]
    ).any():
        raise AssertionError(
            "Game discovery view contains duplicate games."
        )

    return view
