"""
Phase 2E.3A pregame team-versus-opponent identity matchup builder.

Joins each Phase 2E.2 team-game timeline row to its opponent's row for the
same ``game_pk`` and derives paired identity features (team vs. opponent
edges) plus sample-size and confidence diagnostics.

Contract source
----------------

Output schema, column naming, and column ordering are transcribed from the
authoritative schema shipped in the repository at:

- ``atlas_reference/schemas/data__game_intelligence__pregame_identity_matchups
  __2024__pregame_identity_matchups.parquet.schema.json``

No fixture sample exists for this artifact in ``atlas_reference/samples/``
(schema-profile-only ground truth). The summary-diagnostic formulas below
(``identity_sample_balance``, ``identity_sample_confidence_score``,
``identity_sample_confidence_label``) are reconstructed from the schema's
aggregate ``numeric_profile``/``sample_values`` statistics rather than
verified row-by-row against a real fixture; see
``docs/AUTOPILOT_EXECUTION_LEDGER.md`` for the exact confidence level and
the minimum redacted fixture that would allow full row-level verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import json

import pandas as pd

from atlas.config import DATA_ROOT
from atlas.game_intelligence.pregame_identity_source_registry import (
    approved_lagged_identity_columns,
)


ENGINE_VERSION: Final[str] = "1.0.0"

JOIN_KEYS: Final[tuple[str, ...]] = (
    "game_pk",
    "team",
)

CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
    "identity_games_before_date",
    "identity_dates_before_date",
)

# (lower_bound_inclusive, label), evaluated against minimum_identity_games
# using the same 1/5/10/20/40 thresholds as the timeline's
# ``identity_sample_{N}_plus`` flags.
CONFIDENCE_LABEL_BANDS: Final[tuple[tuple[int, str], ...]] = (
    (40, "VERY_HIGH"),
    (20, "HIGH"),
    (10, "MODERATE"),
    (5, "LOW"),
    (1, "VERY_LOW"),
    (0, "NO_HISTORY"),
)

SAMPLE_THRESHOLDS: Final[tuple[int, ...]] = (1, 5, 10, 20, 40)
CONFIDENCE_SCORE_CAP: Final[int] = 40


def _confidence_label(minimum_identity_games: pd.Series) -> pd.Series:
    labels = pd.Series(
        "NO_HISTORY",
        index=minimum_identity_games.index,
        dtype="object",
    )
    for lower_bound, label in sorted(CONFIDENCE_LABEL_BANDS):
        labels = labels.mask(
            minimum_identity_games.ge(lower_bound),
            label,
        )
    return pd.Categorical(
        labels,
        categories=[
            "NO_HISTORY",
            "VERY_LOW",
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH",
        ],
        ordered=True,
    )


def build_pregame_identity_matchups(
    timeline: pd.DataFrame,
    source_registry: pd.DataFrame,
    season: int,
    expected_source_count: int = 87,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if timeline.empty:
        raise ValueError(
            "Pregame team identity timeline is empty."
        )

    source_columns = approved_lagged_identity_columns(
        source_registry
    )

    if len(source_columns) != int(expected_source_count):
        raise AssertionError(
            f"Expected {expected_source_count} identity sources, "
            f"found {len(source_columns)}."
        )

    feature_columns = [
        f"identity__expanding_mean__{column}"
        for column in source_columns
    ]

    missing_features = sorted(
        set(feature_columns).difference(
            timeline.columns
        )
    )
    if missing_features:
        raise KeyError(
            "Timeline is missing identity feature columns: "
            f"{missing_features}"
        )

    normalized = timeline.copy()

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["atlas_season"] = pd.to_numeric(
        normalized["atlas_season"],
        errors="raise",
    ).astype("int64")

    wrong_season = normalized[
        "atlas_season"
    ].ne(int(season))
    if wrong_season.any():
        raise AssertionError(
            f"Timeline contains rows outside season {season}."
        )

    normalized["team"] = (
        normalized["team"]
        .astype("string")
        .str.upper()
        .str.strip()
    )
    normalized["opponent"] = (
        normalized["opponent"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    duplicates = int(
        normalized.duplicated(
            subset=list(JOIN_KEYS),
            keep=False,
        ).sum()
    )
    if duplicates:
        raise AssertionError(
            "Timeline contains duplicate team-game rows: "
            f"{duplicates:,}"
        )

    left = normalized[
        [
            *CONTEXT_COLUMNS,
            *feature_columns,
        ]
    ].copy()

    right_columns = {
        "identity_games_before_date": "opponent_identity_games_before_date",
        "identity_dates_before_date": "opponent_identity_dates_before_date",
        "home_away": "opponent_home_away",
        **{
            feature: f"opponent_identity__{source_column}"
            for feature, source_column in zip(
                feature_columns,
                source_columns,
            )
        },
    }

    right = normalized[
        [
            "game_pk",
            "team",
            "home_away",
            "identity_games_before_date",
            "identity_dates_before_date",
            *feature_columns,
        ]
    ].rename(
        columns={
            "team": "opponent",
            **right_columns,
        }
    )

    matchups = left.rename(
        columns={
            feature: f"team_identity__{source_column}"
            for feature, source_column in zip(
                feature_columns,
                source_columns,
            )
        }
    ).merge(
        right,
        on=["game_pk", "opponent"],
        how="left",
        validate="one_to_one",
    )

    team_columns = [
        f"team_identity__{column}"
        for column in source_columns
    ]
    opponent_columns = [
        f"opponent_identity__{column}"
        for column in source_columns
    ]

    missing_opponent_rows = int(
        matchups[opponent_columns[0]].isna().sum()
        - matchups[team_columns[0]].isna().sum()
    )
    if missing_opponent_rows < 0:
        missing_opponent_rows = 0

    edge_columns = []
    abs_edge_columns = []
    interleaved: dict[str, pd.Series] = {}
    for source_column, team_col, opponent_col in zip(
        source_columns,
        team_columns,
        opponent_columns,
    ):
        edge_col = f"identity_edge__{source_column}"
        abs_edge_col = f"identity_abs_edge__{source_column}"
        edge_columns.append(edge_col)
        abs_edge_columns.append(abs_edge_col)
        interleaved[edge_col] = (
            matchups[team_col] - matchups[opponent_col]
        )
        interleaved[abs_edge_col] = interleaved[edge_col].abs()

    for column, series in interleaved.items():
        matchups[column] = series

    # Reorder feature block to team / opponent / edge / abs_edge per
    # feature, matching the authoritative schema exactly.
    ordered_feature_columns = []
    for source_column in source_columns:
        ordered_feature_columns.extend([
            f"team_identity__{source_column}",
            f"opponent_identity__{source_column}",
            f"identity_edge__{source_column}",
            f"identity_abs_edge__{source_column}",
        ])

    matchups = matchups.copy()

    matchups["minimum_identity_games"] = matchups[
        [
            "identity_games_before_date",
            "opponent_identity_games_before_date",
        ]
    ].min(axis=1)
    matchups["maximum_identity_games"] = matchups[
        [
            "identity_games_before_date",
            "opponent_identity_games_before_date",
        ]
    ].max(axis=1)
    matchups["identity_game_sample_gap"] = (
        matchups["identity_games_before_date"]
        - matchups["opponent_identity_games_before_date"]
    )
    matchups["identity_sample_balance"] = (
        matchups["minimum_identity_games"]
        / matchups["maximum_identity_games"].replace(0, float("nan"))
    ).fillna(1.0)
    matchups["identity_sample_confidence_score"] = (
        matchups["minimum_identity_games"].clip(
            upper=CONFIDENCE_SCORE_CAP
        )
        / CONFIDENCE_SCORE_CAP
    )
    matchups["identity_sample_confidence_label"] = _confidence_label(
        matchups["minimum_identity_games"]
    )

    for threshold in SAMPLE_THRESHOLDS:
        matchups[f"both_teams_sample_{threshold}_plus"] = (
            matchups["identity_games_before_date"].ge(threshold)
            & matchups["opponent_identity_games_before_date"].ge(threshold)
        )

    available_mask = matchups[edge_columns].notna()
    matchups["available_identity_edges"] = available_mask.sum(axis=1)
    matchups["missing_identity_edges"] = (
        len(edge_columns) - matchups["available_identity_edges"]
    )
    matchups["positive_identity_edges"] = (
        matchups[edge_columns].gt(0) & available_mask
    ).sum(axis=1)
    matchups["negative_identity_edges"] = (
        matchups[edge_columns].lt(0) & available_mask
    ).sum(axis=1)
    matchups["neutral_identity_edges"] = (
        matchups[edge_columns].eq(0) & available_mask
    ).sum(axis=1)
    matchups["mean_absolute_identity_edge"] = matchups[
        abs_edge_columns
    ].mean(axis=1, skipna=True)
    matchups["maximum_absolute_identity_edge"] = matchups[
        abs_edge_columns
    ].max(axis=1, skipna=True)

    matchups["pregame_feature_safe"] = True
    matchups["same_date_games_used"] = False
    matchups["future_games_used"] = False
    matchups["prediction_created"] = False
    matchups["identity_matchup_version"] = ENGINE_VERSION

    mirror_audit = _build_mirror_audit(
        matchups,
        edge_columns=edge_columns,
    )

    matchups = matchups.merge(
        mirror_audit[["game_pk", "team", "all_identity_edges_mirror"]],
        on=["game_pk", "team"],
        how="left",
        validate="one_to_one",
    )

    matchups = matchups[
        [
            *CONTEXT_COLUMNS,
            "opponent_home_away",
            "opponent_identity_games_before_date",
            "opponent_identity_dates_before_date",
            *ordered_feature_columns,
            "minimum_identity_games",
            "maximum_identity_games",
            "identity_game_sample_gap",
            "identity_sample_balance",
            "identity_sample_confidence_score",
            "identity_sample_confidence_label",
            *[
                f"both_teams_sample_{threshold}_plus"
                for threshold in SAMPLE_THRESHOLDS
            ],
            "available_identity_edges",
            "missing_identity_edges",
            "positive_identity_edges",
            "negative_identity_edges",
            "neutral_identity_edges",
            "mean_absolute_identity_edge",
            "maximum_absolute_identity_edge",
            "pregame_feature_safe",
            "same_date_games_used",
            "future_games_used",
            "prediction_created",
            "identity_matchup_version",
            "all_identity_edges_mirror",
        ]
    ]

    matchups = matchups.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    validate_pregame_identity_matchups(
        matchups,
        mirror_audit=mirror_audit,
        expected_source_count=expected_source_count,
    )

    return (
        matchups,
        mirror_audit,
    )


def _build_mirror_audit(
    matchups: pd.DataFrame,
    edge_columns: list[str],
) -> pd.DataFrame:
    mirror_merge = matchups[
        [
            "game_pk",
            "team",
            "opponent",
            *edge_columns,
        ]
    ].merge(
        matchups[
            [
                "game_pk",
                "team",
                "opponent",
                *edge_columns,
            ]
        ].rename(
            columns={
                "team": "opponent",
                "opponent": "team",
                **{
                    column: f"mirror__{column}"
                    for column in edge_columns
                },
            }
        ),
        on=["game_pk", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    mismatches = pd.Series(
        False,
        index=mirror_merge.index,
    )
    for column in edge_columns:
        mirror_column = f"mirror__{column}"
        both_null = (
            mirror_merge[column].isna()
            & mirror_merge[mirror_column].isna()
        )
        matches = mirror_merge[column].eq(
            -mirror_merge[mirror_column]
        )
        mismatches |= ~(matches | both_null)

    mirror_merge["all_identity_edges_mirror"] = ~mismatches
    mirror_merge["mirror_failures"] = mismatches.astype("int64")
    mirror_merge["audit_pass"] = ~mismatches

    return mirror_merge[
        [
            "game_pk",
            "team",
            "opponent",
            "all_identity_edges_mirror",
            "mirror_failures",
            "audit_pass",
        ]
    ].sort_values(
        ["game_pk", "team"],
        kind="stable",
    ).reset_index(drop=True)


def validate_pregame_identity_matchups(
    matchups: pd.DataFrame,
    mirror_audit: pd.DataFrame,
    expected_source_count: int = 87,
) -> None:
    if matchups.empty:
        raise ValueError(
            "Pregame identity matchup frame is empty."
        )

    duplicates = int(
        matchups.duplicated(
            subset=list(JOIN_KEYS),
            keep=False,
        ).sum()
    )
    if duplicates:
        raise AssertionError(
            "Identity matchup frame has duplicate team-game rows: "
            f"{duplicates:,}"
        )

    team_columns = [
        column
        for column in matchups.columns
        if column.startswith("team_identity__")
    ]
    opponent_columns = [
        column
        for column in matchups.columns
        if column.startswith("opponent_identity__")
        and column
        not in {
            "opponent_identity_games_before_date",
            "opponent_identity_dates_before_date",
        }
    ]
    raw_edge_columns = [
        column
        for column in matchups.columns
        if column.startswith("identity_edge__")
    ]
    abs_edge_columns = [
        column
        for column in matchups.columns
        if column.startswith("identity_abs_edge__")
    ]

    for family_columns in (
        team_columns,
        opponent_columns,
        raw_edge_columns,
        abs_edge_columns,
    ):
        if len(family_columns) != int(expected_source_count):
            raise AssertionError(
                "Identity matchup feature family count mismatch: "
                f"{len(family_columns)} != {expected_source_count}"
            )

    if not matchups["pregame_feature_safe"].all():
        raise AssertionError(
            "Identity matchups must be marked pregame feature safe."
        )

    if matchups["same_date_games_used"].any():
        raise AssertionError(
            "Identity matchups cannot use same-date games."
        )

    if matchups["future_games_used"].any():
        raise AssertionError(
            "Identity matchups cannot use future games."
        )

    if mirror_audit.empty:
        raise ValueError(
            "Identity matchup mirror audit is empty."
        )

    if not mirror_audit["audit_pass"].all():
        raise AssertionError(
            "Identity matchup mirror audit failed."
        )


def assert_reproduces_reference_matchups(
    matchups: pd.DataFrame,
    reference_matchups: pd.DataFrame,
    key_columns: tuple[str, ...] = JOIN_KEYS,
) -> None:
    if list(matchups.columns) != list(reference_matchups.columns):
        raise AssertionError(
            "Matchup schema mismatch against reference artifact."
        )

    if len(matchups) != len(reference_matchups):
        raise AssertionError(
            "Matchup row-count mismatch against reference artifact."
        )

    left_duplicates = int(
        matchups.duplicated(
            subset=list(key_columns),
            keep=False,
        ).sum()
    )
    right_duplicates = int(
        reference_matchups.duplicated(
            subset=list(key_columns),
            keep=False,
        ).sum()
    )

    if left_duplicates or right_duplicates:
        raise AssertionError(
            "Matchup key uniqueness mismatch against reference artifact."
        )

    left = matchups.sort_values(
        list(key_columns),
        kind="stable",
    ).reset_index(drop=True)

    right = reference_matchups.sort_values(
        list(key_columns),
        kind="stable",
    ).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise AssertionError(
            "Matchup value mismatch against reference artifact."
        ) from exc


def phase_2e_identity_matchup_paths(
    season: int,
    data_root: Path | None = None,
) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DATA_ROOT

    base = (
        root
        / "game_intelligence"
        / "pregame_identity_matchups"
        / str(int(season))
    )

    return {
        "base_dir": base,
        "matchups_parquet": base / "pregame_identity_matchups.parquet",
        "metadata_json": base / "pregame_identity_matchups_metadata.json",
    }


def build_pregame_identity_matchups_metadata(
    matchups: pd.DataFrame,
    mirror_audit: pd.DataFrame,
    season: int,
    expected_source_count: int = 87,
) -> dict[str, object]:
    validate_pregame_identity_matchups(
        matchups,
        mirror_audit=mirror_audit,
        expected_source_count=expected_source_count,
    )

    return {
        "phase": "2E.3A",
        "season": int(season),
        "matchup_rows": int(len(matchups)),
        "unique_games": int(matchups["game_pk"].nunique()),
        "teams": int(matchups["team"].nunique()),
        "team_identity_columns": int(expected_source_count),
        "opponent_identity_columns": int(expected_source_count),
        "raw_identity_edges": int(expected_source_count),
        "absolute_identity_edges": int(expected_source_count),
        "mirror_failures": int(mirror_audit["mirror_failures"].sum()),
        "same_date_games_used": False,
        "future_games_used": False,
        "identity_matchup_version": ENGINE_VERSION,
    }


def save_pregame_identity_matchups(
    matchups: pd.DataFrame,
    mirror_audit: pd.DataFrame,
    season: int,
    data_root: Path | None = None,
    expected_source_count: int = 87,
) -> dict[str, Path]:
    metadata = build_pregame_identity_matchups_metadata(
        matchups=matchups,
        mirror_audit=mirror_audit,
        season=season,
        expected_source_count=expected_source_count,
    )

    paths = phase_2e_identity_matchup_paths(
        season=season,
        data_root=data_root,
    )

    paths["base_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    matchups.to_parquet(
        paths["matchups_parquet"],
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
