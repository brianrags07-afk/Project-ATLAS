"""
Phase 2E.3A pregame team-versus-opponent identity matchup builder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import json

import pandas as pd

from atlas.config import DATA_ROOT


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
)

PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "identity_source_game_pk",
    "identity_source_game_date",
    "strict_backtest_safe",
    "same_date_games_used",
    "future_games_used",
)


def _registry_feature_names(
    source_registry: pd.DataFrame,
) -> list[str]:
    required = {
        "identity_feature_name",
    }

    missing = sorted(
        required.difference(
            source_registry.columns
        )
    )
    if missing:
        raise KeyError(
            f"Registry missing required columns: {missing}"
        )

    feature_names = [
        str(value)
        for value in source_registry[
            "identity_feature_name"
        ].tolist()
    ]

    if len(feature_names) != len(set(feature_names)):
        raise AssertionError(
            "Registry identity feature names must be unique."
        )

    return feature_names


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

    feature_names = _registry_feature_names(
        source_registry
    )

    if len(feature_names) != int(expected_source_count):
        raise AssertionError(
            f"Expected {expected_source_count} identity features, "
            f"found {len(feature_names)}."
        )

    missing_features = sorted(
        set(feature_names).difference(
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

    team_feature_columns = {
        feature: f"team_identity__{feature}"
        for feature in feature_names
    }
    opponent_feature_columns = {
        feature: f"opponent_identity__{feature}"
        for feature in feature_names
    }

    left = normalized[
        [
            *CONTEXT_COLUMNS,
            *PROVENANCE_COLUMNS,
            *feature_names,
        ]
    ].copy()

    left = left.rename(
        columns=team_feature_columns
    )

    right = normalized[
        [
            "game_pk",
            "team",
            *feature_names,
        ]
    ].copy()

    right = right.rename(
        columns={
            "team": "opponent",
            **opponent_feature_columns,
        }
    )

    matchups = left.merge(
        right,
        on=["game_pk", "opponent"],
        how="left",
        validate="one_to_one",
    )

    missing_opponent_rows = int(
        matchups[
            "opponent_identity__" + feature_names[0]
        ].isna().sum()
    )

    if missing_opponent_rows:
        raise AssertionError(
            f"Missing opponent identity rows: {missing_opponent_rows:,}"
        )

    for feature in feature_names:
        team_col = team_feature_columns[feature]
        opponent_col = opponent_feature_columns[feature]
        edge_col = f"identity_edge__{feature}"
        abs_edge_col = f"identity_edge_abs__{feature}"

        matchups[edge_col] = (
            matchups[team_col]
            - matchups[opponent_col]
        )

        matchups[abs_edge_col] = matchups[edge_col].abs()

    matchups["raw_identity_edges"] = int(
        len(feature_names)
    )
    matchups["absolute_identity_edges"] = int(
        len(feature_names)
    )
    matchups["strict_backtest_safe"] = True
    matchups["same_date_games_used"] = False
    matchups["future_games_used"] = False
    matchups["phase"] = "2E.3A"
    matchups["matchup_engine_version"] = ENGINE_VERSION

    edge_columns = [
        f"identity_edge__{feature}"
        for feature in feature_names
    ]
    abs_edge_columns = [
        f"identity_edge_abs__{feature}"
        for feature in feature_names
    ]

    mirror_merge = matchups[
        [
            "game_pk",
            "team",
            "opponent",
            *edge_columns,
            *abs_edge_columns,
        ]
    ].merge(
        matchups[
            [
                "game_pk",
                "team",
                "opponent",
                *edge_columns,
                *abs_edge_columns,
            ]
        ].rename(
            columns={
                "team": "opponent",
                "opponent": "team",
                **{
                    column: f"mirror__{column}"
                    for column in [
                        *edge_columns,
                        *abs_edge_columns,
                    ]
                },
            }
        ),
        on=["game_pk", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    edge_failures = pd.Series(
        data=0,
        index=mirror_merge.index,
        dtype="int64",
    )

    for column in edge_columns:
        mirror_column = f"mirror__{column}"
        edge_failures += (
            ~mirror_merge[column].eq(
                -mirror_merge[mirror_column]
            )
        ).astype("int64")

    for column in abs_edge_columns:
        mirror_column = f"mirror__{column}"
        edge_failures += (
            ~mirror_merge[column].eq(
                mirror_merge[mirror_column]
            )
        ).astype("int64")

    mirror_merge["mirror_failures"] = edge_failures
    mirror_merge["audit_pass"] = mirror_merge[
        "mirror_failures"
    ].eq(0)

    mirror_audit = (
        mirror_merge.groupby(
            "game_pk",
            as_index=False,
            sort=False,
        )[
            [
                "mirror_failures",
                "audit_pass",
            ]
        ]
        .agg({
            "mirror_failures": "max",
            "audit_pass": "all",
        })
        .sort_values(
            "game_pk",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if not mirror_audit["audit_pass"].all():
        raise AssertionError(
            "Identity matchup mirror audit failed."
        )

    validate_pregame_identity_matchups(
        matchups,
        mirror_audit=mirror_audit,
        expected_source_count=expected_source_count,
    )

    matchups = matchups.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return (
        matchups,
        mirror_audit,
    )


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
    ]
    raw_edge_columns = [
        column
        for column in matchups.columns
        if column.startswith("identity_edge__")
        and not column.startswith("identity_edge_abs__")
    ]
    abs_edge_columns = [
        column
        for column in matchups.columns
        if column.startswith("identity_edge_abs__")
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

    if not matchups["strict_backtest_safe"].all():
        raise AssertionError(
            "Identity matchups must be strict backtest safe."
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
        "matchup_engine_version": ENGINE_VERSION,
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
