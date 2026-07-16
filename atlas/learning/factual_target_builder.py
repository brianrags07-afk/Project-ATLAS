"""
Factual historical target construction for Project ATLAS.

This module converts completed game results into deterministic game-level and
team-game-level learning targets.

Targets are historical outcomes. They are not pregame evidence and must never
be written into the canonical pregame evidence artifact.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


ENGINE_VERSION: Final[str] = "1.0.0"

TEAM_ALIASES: Final[dict[str, str]] = {
    "OAK": "ATH",
    "ARI": "AZ",
}

GAME_ID_CANDIDATES: Final[tuple[str, ...]] = (
    "game_pk",
    "game_id",
    "gamepk",
)

GAME_DATE_CANDIDATES: Final[tuple[str, ...]] = (
    "game_date",
    "official_date",
    "date",
)

SEASON_CANDIDATES: Final[tuple[str, ...]] = (
    "atlas_season",
    "season",
    "game_season",
)

HOME_TEAM_CANDIDATES: Final[tuple[str, ...]] = (
    "home_team",
    "home_team_abbreviation",
    "home_abbreviation",
    "home",
)

AWAY_TEAM_CANDIDATES: Final[tuple[str, ...]] = (
    "away_team",
    "away_team_abbreviation",
    "away_abbreviation",
    "away",
)

HOME_SCORE_CANDIDATES: Final[tuple[str, ...]] = (
    "home_score",
    "home_runs",
    "home_team_runs",
    "home_final_score",
    "home_r",
)

AWAY_SCORE_CANDIDATES: Final[tuple[str, ...]] = (
    "away_score",
    "away_runs",
    "away_team_runs",
    "away_final_score",
    "away_r",
)


def normalize_team_code(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
        .replace(
            TEAM_ALIASES
        )
    )


def resolve_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    field_name: str,
    required: bool = True,
) -> str | None:
    lower_map = {
        str(column).lower():
            str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    if required:
        raise KeyError(
            f"Could not resolve {field_name}. "
            f"Candidates: {list(candidates)}"
        )

    return None


def standardize_completed_games(
    dataframe: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe.empty:
        raise ValueError(
            "Master game dataframe is empty."
        )

    game_pk_column = resolve_column(
        dataframe,
        GAME_ID_CANDIDATES,
        "game identifier",
    )

    game_date_column = resolve_column(
        dataframe,
        GAME_DATE_CANDIDATES,
        "game date",
    )

    season_column = resolve_column(
        dataframe,
        SEASON_CANDIDATES,
        "season",
        required=False,
    )

    home_team_column = resolve_column(
        dataframe,
        HOME_TEAM_CANDIDATES,
        "home team",
    )

    away_team_column = resolve_column(
        dataframe,
        AWAY_TEAM_CANDIDATES,
        "away team",
    )

    home_score_column = resolve_column(
        dataframe,
        HOME_SCORE_CANDIDATES,
        "home score",
    )

    away_score_column = resolve_column(
        dataframe,
        AWAY_SCORE_CANDIDATES,
        "away score",
    )

    standardized = pd.DataFrame({
        "game_pk":
            pd.to_numeric(
                dataframe[
                    game_pk_column
                ],
                errors="coerce",
            ),

        "game_date":
            pd.to_datetime(
                dataframe[
                    game_date_column
                ],
                errors="coerce",
            ).dt.normalize(),

        "home_team":
            normalize_team_code(
                dataframe[
                    home_team_column
                ]
            ),

        "away_team":
            normalize_team_code(
                dataframe[
                    away_team_column
                ]
            ),

        "home_score":
            pd.to_numeric(
                dataframe[
                    home_score_column
                ],
                errors="coerce",
            ),

        "away_score":
            pd.to_numeric(
                dataframe[
                    away_score_column
                ],
                errors="coerce",
            ),
    })

    if season_column is not None:
        standardized[
            "atlas_season"
        ] = pd.to_numeric(
            dataframe[
                season_column
            ],
            errors="coerce",
        )

    else:
        standardized[
            "atlas_season"
        ] = standardized[
            "game_date"
        ].dt.year

    standardized = standardized[
        standardized[
            "atlas_season"
        ].eq(
            int(season)
        )
    ].copy()

    required_columns = [
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    missing_required = standardized[
        required_columns
    ].isna().any(
        axis=1
    )

    excluded_rows = standardized[
        missing_required
    ].copy()

    standardized = standardized[
        ~missing_required
    ].copy()

    standardized[
        "game_pk"
    ] = standardized[
        "game_pk"
    ].astype(
        "int64"
    )

    standardized[
        "atlas_season"
    ] = standardized[
        "atlas_season"
    ].astype(
        "int64"
    )

    standardized[
        "home_score"
    ] = standardized[
        "home_score"
    ].astype(
        "int64"
    )

    standardized[
        "away_score"
    ] = standardized[
        "away_score"
    ].astype(
        "int64"
    )

    negative_scores = (
        standardized[
            "home_score"
        ].lt(
            0
        )
        | standardized[
            "away_score"
        ].lt(
            0
        )
    )

    if negative_scores.any():
        raise AssertionError(
            "Negative final scores were found."
        )

    duplicate_games = int(
        standardized.duplicated(
            subset=[
                "game_pk",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_games:
        duplicate_examples = standardized[
            standardized.duplicated(
                subset=[
                    "game_pk",
                ],
                keep=False,
            )
        ].sort_values(
            "game_pk",
            kind="stable",
        ).head(
            30
        )

        raise AssertionError(
            f"Duplicate completed-game rows: {duplicate_games:,}\n"
            + duplicate_examples.to_string(
                index=False
            )
        )

    standardized = standardized.sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    source_audit = pd.DataFrame([
        {
            "canonical_field":
                "game_pk",
            "source_column":
                game_pk_column,
        },
        {
            "canonical_field":
                "game_date",
            "source_column":
                game_date_column,
        },
        {
            "canonical_field":
                "atlas_season",
            "source_column":
                (
                    season_column
                    if season_column is not None
                    else "derived_from_game_date"
                ),
        },
        {
            "canonical_field":
                "home_team",
            "source_column":
                home_team_column,
        },
        {
            "canonical_field":
                "away_team",
            "source_column":
                away_team_column,
        },
        {
            "canonical_field":
                "home_score",
            "source_column":
                home_score_column,
        },
        {
            "canonical_field":
                "away_score",
            "source_column":
                away_score_column,
        },
    ])

    return (
        standardized,
        source_audit,
    )


def build_game_targets(
    completed_games: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    }

    missing = sorted(
        required.difference(
            completed_games.columns
        )
    )

    if missing:
        raise KeyError(
            f"Completed-game table lacks fields: {missing}"
        )

    targets = completed_games.copy()

    targets[
        "game_total_runs"
    ] = (
        targets[
            "home_score"
        ]
        + targets[
            "away_score"
        ]
    )

    targets[
        "home_margin"
    ] = (
        targets[
            "home_score"
        ]
        - targets[
            "away_score"
        ]
    )

    targets[
        "away_margin"
    ] = (
        -targets[
            "home_margin"
        ]
    )

    targets[
        "home_win"
    ] = targets[
        "home_margin"
    ].gt(
        0
    )

    targets[
        "away_win"
    ] = targets[
        "away_margin"
    ].gt(
        0
    )

    targets[
        "tie_game"
    ] = targets[
        "home_score"
    ].eq(
        targets[
            "away_score"
        ]
    )

    if targets[
        "tie_game"
    ].any():
        tie_examples = targets[
            targets[
                "tie_game"
            ]
        ].head(
            20
        )

        raise AssertionError(
            "Completed MLB games cannot remain tied:\n"
            + tie_examples.to_string(
                index=False
            )
        )

    targets[
        "one_run_game"
    ] = targets[
        "home_margin"
    ].abs().eq(
        1
    )

    targets[
        "margin_2_plus"
    ] = targets[
        "home_margin"
    ].abs().ge(
        2
    )

    targets[
        "margin_4_plus"
    ] = targets[
        "home_margin"
    ].abs().ge(
        4
    )

    targets[
        "margin_6_plus"
    ] = targets[
        "home_margin"
    ].abs().ge(
        6
    )

    targets[
        "game_total_6_or_less"
    ] = targets[
        "game_total_runs"
    ].le(
        6
    )

    targets[
        "game_total_7_or_less"
    ] = targets[
        "game_total_runs"
    ].le(
        7
    )

    targets[
        "game_total_8_or_less"
    ] = targets[
        "game_total_runs"
    ].le(
        8
    )

    targets[
        "game_total_9_plus"
    ] = targets[
        "game_total_runs"
    ].ge(
        9
    )

    targets[
        "game_total_10_plus"
    ] = targets[
        "game_total_runs"
    ].ge(
        10
    )

    targets[
        "game_total_over_10"
    ] = targets[
        "game_total_runs"
    ].gt(
        10
    )

    targets[
        "game_total_12_plus"
    ] = targets[
        "game_total_runs"
    ].ge(
        12
    )

    targets[
        "both_teams_scored"
    ] = (
        targets[
            "home_score"
        ].gt(
            0
        )
        & targets[
            "away_score"
        ].gt(
            0
        )
    )

    targets[
        "either_team_shutout"
    ] = (
        targets[
            "home_score"
        ].eq(
            0
        )
        | targets[
            "away_score"
        ].eq(
            0
        )
    )

    targets[
        "both_teams_scored_4_plus"
    ] = (
        targets[
            "home_score"
        ].ge(
            4
        )
        & targets[
            "away_score"
        ].ge(
            4
        )
    )

    targets[
        "both_teams_scored_5_plus"
    ] = (
        targets[
            "home_score"
        ].ge(
            5
        )
        & targets[
            "away_score"
        ].ge(
            5
        )
    )

    targets[
        "either_team_scored_8_plus"
    ] = (
        targets[
            "home_score"
        ].ge(
            8
        )
        | targets[
            "away_score"
        ].ge(
            8
        )
    )

    targets[
        "either_team_scored_10_plus"
    ] = (
        targets[
            "home_score"
        ].ge(
            10
        )
        | targets[
            "away_score"
        ].ge(
            10
        )
    )

    targets[
        "strict_factual_target"
    ] = True

    targets[
        "market_line_used"
    ] = False

    targets[
        "pregame_evidence_included"
    ] = False

    targets[
        "prediction_created"
    ] = False

    targets[
        "target_builder_version"
    ] = ENGINE_VERSION

    return targets


def build_team_game_targets(
    game_targets: pd.DataFrame,
) -> pd.DataFrame:
    home = pd.DataFrame({
        "game_pk":
            game_targets[
                "game_pk"
            ],

        "game_date":
            game_targets[
                "game_date"
            ],

        "atlas_season":
            game_targets[
                "atlas_season"
            ],

        "team":
            game_targets[
                "home_team"
            ],

        "opponent":
            game_targets[
                "away_team"
            ],

        "home_away":
            "HOME",

        "team_runs":
            game_targets[
                "home_score"
            ],

        "opponent_runs":
            game_targets[
                "away_score"
            ],
    })

    away = pd.DataFrame({
        "game_pk":
            game_targets[
                "game_pk"
            ],

        "game_date":
            game_targets[
                "game_date"
            ],

        "atlas_season":
            game_targets[
                "atlas_season"
            ],

        "team":
            game_targets[
                "away_team"
            ],

        "opponent":
            game_targets[
                "home_team"
            ],

        "home_away":
            "AWAY",

        "team_runs":
            game_targets[
                "away_score"
            ],

        "opponent_runs":
            game_targets[
                "home_score"
            ],
    })

    targets = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    targets[
        "run_margin"
    ] = (
        targets[
            "team_runs"
        ]
        - targets[
            "opponent_runs"
        ]
    )

    targets[
        "game_total_runs"
    ] = (
        targets[
            "team_runs"
        ]
        + targets[
            "opponent_runs"
        ]
    )

    targets[
        "target_team_win"
    ] = targets[
        "run_margin"
    ].gt(
        0
    )

    targets[
        "target_team_loss"
    ] = targets[
        "run_margin"
    ].lt(
        0
    )

    targets[
        "target_team_win_by_2_plus"
    ] = targets[
        "run_margin"
    ].ge(
        2
    )

    targets[
        "target_team_loss_by_2_plus"
    ] = targets[
        "run_margin"
    ].le(
        -2
    )

    targets[
        "target_team_win_by_4_plus"
    ] = targets[
        "run_margin"
    ].ge(
        4
    )

    targets[
        "target_team_loss_by_4_plus"
    ] = targets[
        "run_margin"
    ].le(
        -4
    )

    targets[
        "target_team_win_by_6_plus"
    ] = targets[
        "run_margin"
    ].ge(
        6
    )

    targets[
        "target_team_loss_by_6_plus"
    ] = targets[
        "run_margin"
    ].le(
        -6
    )

    targets[
        "target_one_run_game"
    ] = targets[
        "run_margin"
    ].abs().eq(
        1
    )

    targets[
        "target_margin_2_plus_game"
    ] = targets[
        "run_margin"
    ].abs().ge(
        2
    )

    targets[
        "target_margin_4_plus_game"
    ] = targets[
        "run_margin"
    ].abs().ge(
        4
    )

    targets[
        "target_game_total_6_or_less"
    ] = targets[
        "game_total_runs"
    ].le(
        6
    )

    targets[
        "target_game_total_7_or_less"
    ] = targets[
        "game_total_runs"
    ].le(
        7
    )

    targets[
        "target_game_total_8_or_less"
    ] = targets[
        "game_total_runs"
    ].le(
        8
    )

    targets[
        "target_game_total_9_plus"
    ] = targets[
        "game_total_runs"
    ].ge(
        9
    )

    targets[
        "target_game_total_10_plus"
    ] = targets[
        "game_total_runs"
    ].ge(
        10
    )

    targets[
        "target_game_total_over_10"
    ] = targets[
        "game_total_runs"
    ].gt(
        10
    )

    targets[
        "target_game_total_12_plus"
    ] = targets[
        "game_total_runs"
    ].ge(
        12
    )

    targets[
        "target_team_scored_0"
    ] = targets[
        "team_runs"
    ].eq(
        0
    )

    targets[
        "target_team_scored_2_or_less"
    ] = targets[
        "team_runs"
    ].le(
        2
    )

    targets[
        "target_team_scored_3_or_less"
    ] = targets[
        "team_runs"
    ].le(
        3
    )

    targets[
        "target_team_scored_exactly_4"
    ] = targets[
        "team_runs"
    ].eq(
        4
    )

    targets[
        "target_team_scored_5_plus"
    ] = targets[
        "team_runs"
    ].ge(
        5
    )

    targets[
        "target_team_scored_6_plus"
    ] = targets[
        "team_runs"
    ].ge(
        6
    )

    targets[
        "target_team_scored_8_plus"
    ] = targets[
        "team_runs"
    ].ge(
        8
    )

    targets[
        "target_team_scored_10_plus"
    ] = targets[
        "team_runs"
    ].ge(
        10
    )

    targets[
        "target_team_allowed_0"
    ] = targets[
        "opponent_runs"
    ].eq(
        0
    )

    targets[
        "target_team_allowed_2_or_less"
    ] = targets[
        "opponent_runs"
    ].le(
        2
    )

    targets[
        "target_team_allowed_3_or_less"
    ] = targets[
        "opponent_runs"
    ].le(
        3
    )

    targets[
        "target_team_allowed_5_plus"
    ] = targets[
        "opponent_runs"
    ].ge(
        5
    )

    targets[
        "target_team_allowed_6_plus"
    ] = targets[
        "opponent_runs"
    ].ge(
        6
    )

    targets[
        "target_team_shutout_win"
    ] = (
        targets[
            "target_team_win"
        ]
        & targets[
            "opponent_runs"
        ].eq(
            0
        )
    )

    targets[
        "target_team_shutout_loss"
    ] = (
        targets[
            "target_team_loss"
        ]
        & targets[
            "team_runs"
        ].eq(
            0
        )
    )

    targets[
        "strict_factual_target"
    ] = True

    targets[
        "market_line_used"
    ] = False

    targets[
        "pregame_evidence_included"
    ] = False

    targets[
        "prediction_created"
    ] = False

    targets[
        "same_date_pregame_feature_used"
    ] = False

    targets[
        "future_game_used"
    ] = False

    targets[
        "target_builder_version"
    ] = ENGINE_VERSION

    targets = targets.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    duplicate_team_games = int(
        targets.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if duplicate_team_games:
        raise AssertionError(
            f"Duplicate team-game targets: "
            f"{duplicate_team_games:,}"
        )

    return targets


def validate_target_symmetry(
    team_targets: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "team",
        "opponent",
        "team_runs",
        "opponent_runs",
        "run_margin",
        "target_team_win",
        "target_team_loss",
        "target_team_win_by_2_plus",
        "target_team_loss_by_2_plus",
        "target_team_win_by_4_plus",
        "target_team_loss_by_4_plus",
        "game_total_runs",
    }

    missing = sorted(
        required.difference(
            team_targets.columns
        )
    )

    if missing:
        raise KeyError(
            f"Target symmetry input lacks fields: {missing}"
        )

    rows = []

    counts = team_targets.groupby(
        "game_pk"
    ).size()

    invalid_two_rows = int(
        counts.ne(
            2
        ).sum()
    )

    rows.append({
        "check":
            "exactly two team rows per game",
        "failures":
            invalid_two_rows,
        "passed":
            invalid_two_rows == 0,
    })

    paired = team_targets.merge(
        team_targets,
        left_on=[
            "game_pk",
            "team",
            "opponent",
        ],
        right_on=[
            "game_pk",
            "opponent",
            "team",
        ],
        suffixes=(
            "_team",
            "_mirror",
        ),
        how="left",
        validate="one_to_one",
    )

    mirror_missing = int(
        paired[
            "team_mirror"
        ].isna().sum()
    )

    rows.append({
        "check":
            "mirror team row exists",
        "failures":
            mirror_missing,
        "passed":
            mirror_missing == 0,
    })

    checks = {
        "team runs mirror opponent runs":
            paired[
                "team_runs_team"
            ].eq(
                paired[
                    "opponent_runs_mirror"
                ]
            ),

        "opponent runs mirror team runs":
            paired[
                "opponent_runs_team"
            ].eq(
                paired[
                    "team_runs_mirror"
                ]
            ),

        "run margins are inverse":
            paired[
                "run_margin_team"
            ].eq(
                -paired[
                    "run_margin_mirror"
                ]
            ),

        "game totals match":
            paired[
                "game_total_runs_team"
            ].eq(
                paired[
                    "game_total_runs_mirror"
                ]
            ),

        "team win mirrors opponent loss":
            paired[
                "target_team_win_team"
            ].eq(
                paired[
                    "target_team_loss_mirror"
                ]
            ),

        "team loss mirrors opponent win":
            paired[
                "target_team_loss_team"
            ].eq(
                paired[
                    "target_team_win_mirror"
                ]
            ),

        "win by 2 mirrors loss by 2":
            paired[
                "target_team_win_by_2_plus_team"
            ].eq(
                paired[
                    "target_team_loss_by_2_plus_mirror"
                ]
            ),

        "win by 4 mirrors loss by 4":
            paired[
                "target_team_win_by_4_plus_team"
            ].eq(
                paired[
                    "target_team_loss_by_4_plus_mirror"
                ]
            ),
    }

    for name, passed_mask in checks.items():
        failures = int(
            (
                ~passed_mask.fillna(
                    False
                )
            ).sum()
        )

        rows.append({
            "check":
                name,
            "failures":
                failures,
            "passed":
                failures == 0,
        })

    return pd.DataFrame(
        rows
    )
