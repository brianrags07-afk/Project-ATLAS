"""
Phase 2E.2 strict prior-date pregame team identity timeline builder.
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

TEAM_CODE_NORMALIZATION: Final[dict[str, str]] = {
    "OAK": "ATH",
    "ARI": "AZ",
}

DATE_ALIASES: Final[tuple[str, ...]] = (
    "game_date",
    "official_date",
    "date",
)

CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
)


def _normalize_team_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
        .replace(TEAM_CODE_NORMALIZATION)
    )


def _resolve_date_column(frame: pd.DataFrame) -> str:
    lower_map = {
        str(column).lower(): str(column)
        for column in frame.columns
    }

    for alias in DATE_ALIASES:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    raise KeyError(
        "Missing required game date column. "
        f"Expected one of: {DATE_ALIASES}"
    )


def normalize_team_identity_source(
    frame: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(
            "Team identity source dataframe is empty."
        )

    missing = sorted(
        set(JOIN_KEYS).difference(frame.columns)
    )
    if missing:
        raise KeyError(
            f"Missing required team-game keys: {missing}"
        )

    normalized = frame.copy()

    date_column = _resolve_date_column(
        normalized
    )

    normalized["game_pk"] = pd.to_numeric(
        normalized["game_pk"],
        errors="raise",
    ).astype("int64")

    normalized["team"] = _normalize_team_code(
        normalized["team"]
    )

    if "opponent" in normalized.columns:
        normalized["opponent"] = _normalize_team_code(
            normalized["opponent"]
        )

    if "home_away" in normalized.columns:
        normalized["home_away"] = (
            normalized["home_away"]
            .astype("string")
            .str.upper()
            .str.strip()
        )

    normalized["game_date"] = pd.to_datetime(
        normalized[date_column],
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
            f"Source contains rows outside season {season}."
        )

    duplicate_count = int(
        normalized.duplicated(
            subset=list(JOIN_KEYS),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise AssertionError(
            "Source contains duplicate team-game rows: "
            f"{duplicate_count:,}"
        )

    normalized = normalized.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    )

    return normalized.reset_index(
        drop=True
    )


def _registry_feature_pairs(
    registry: pd.DataFrame,
) -> list[tuple[str, str]]:
    required_columns = {
        "identity_feature_name",
        "source_column",
    }

    missing = sorted(
        required_columns.difference(
            registry.columns
        )
    )
    if missing:
        raise KeyError(
            f"Registry missing required columns: {missing}"
        )

    rows = []
    for _, row in registry.iterrows():
        feature_name = str(
            row["identity_feature_name"]
        )
        source_column = str(
            row["source_column"]
        )
        rows.append((
            source_column,
            feature_name,
        ))

    if len(rows) != len(set(rows)):
        raise AssertionError(
            "Registry contains duplicate source-to-feature mappings."
        )

    return rows


def build_pregame_team_identity_timeline(
    phase_2d_identity_frame: pd.DataFrame,
    source_registry: pd.DataFrame,
    season: int,
    expected_source_count: int = 87,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = normalize_team_identity_source(
        phase_2d_identity_frame,
        season=season,
    )

    feature_pairs = _registry_feature_pairs(
        source_registry
    )

    if len(feature_pairs) != int(expected_source_count):
        raise AssertionError(
            f"Expected {expected_source_count} identity sources, "
            f"found {len(feature_pairs)}."
        )

    source_columns = [
        source_column
        for source_column, _ in feature_pairs
    ]

    missing_source_columns = sorted(
        set(source_columns).difference(
            source.columns
        )
    )
    if missing_source_columns:
        raise KeyError(
            "Source columns required by registry are missing: "
            f"{missing_source_columns}"
        )

    mapping = dict(feature_pairs)

    date_team_last = (
        source.sort_values(
            [
                "team",
                "game_date",
                "game_pk",
            ],
            kind="stable",
        )
        .groupby(
            ["team", "game_date"],
            as_index=False,
            sort=False,
        )
        .tail(1)
        .sort_values(
            [
                "team",
                "game_date",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    prior_snapshot = date_team_last[
        [
            "team",
            "game_date",
            "game_pk",
            *source_columns,
        ]
    ].copy()

    prior_snapshot = prior_snapshot.rename(
        columns={
            "game_pk": "identity_source_game_pk",
        }
    )

    prior_snapshot["identity_source_game_date"] = prior_snapshot[
        "game_date"
    ]

    shifted_columns = [
        "identity_source_game_pk",
        "identity_source_game_date",
        *source_columns,
    ]

    prior_snapshot[shifted_columns] = (
        prior_snapshot.groupby(
            "team",
            sort=False,
        )[shifted_columns]
        .shift(1)
    )

    merge_columns = [
        "team",
        "game_date",
        *shifted_columns,
    ]

    timeline_base = source.drop(
        columns=source_columns,
        errors="ignore",
    )

    timeline = timeline_base.merge(
        prior_snapshot[merge_columns],
        on=["team", "game_date"],
        how="left",
        validate="many_to_one",
    )

    rename_map = {
        source_column: mapping[source_column]
        for source_column in source_columns
    }

    timeline = timeline.rename(
        columns=rename_map
    )

    feature_columns = [
        feature_name
        for _, feature_name in feature_pairs
    ]

    timeline["strict_backtest_safe"] = True
    timeline["same_date_games_used"] = False
    timeline["future_games_used"] = False
    timeline["phase"] = "2E.2"
    timeline["timeline_engine_version"] = ENGINE_VERSION

    timeline = timeline.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    audit = (
        timeline[
            [
                "team",
                "game_date",
                "identity_source_game_pk",
                "identity_source_game_date",
            ]
        ]
        .drop_duplicates(
            subset=["team", "game_date"],
            keep="first",
        )
        .sort_values(
            [
                "game_date",
                "team",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    team_date_game_counts = (
        timeline.groupby(
            ["team", "game_date"],
            sort=False,
        )[
            "game_pk"
        ]
        .size()
        .rename("team_games_on_date")
        .reset_index()
    )

    audit = audit.merge(
        team_date_game_counts,
        on=["team", "game_date"],
        how="left",
        validate="one_to_one",
    )

    has_source = audit[
        "identity_source_game_date"
    ].notna()

    strict_prior = (
        ~has_source
        | audit[
            "identity_source_game_date"
        ].lt(audit["game_date"])
    )

    audit["strict_prior_date_pass"] = strict_prior
    audit["same_date_games_used"] = False
    audit["future_games_used"] = False
    audit["audit_pass"] = (
        audit["strict_prior_date_pass"]
        & ~audit["same_date_games_used"]
        & ~audit["future_games_used"]
    )
    audit["timeline_engine_version"] = ENGINE_VERSION

    lag_days = (
        audit["game_date"]
        - audit["identity_source_game_date"]
    ).dt.days
    audit["lagged_days"] = lag_days

    failures = audit.loc[
        ~audit["audit_pass"]
    ].copy()

    validate_pregame_team_identity_timeline(
        timeline,
        audit=audit,
        expected_source_count=expected_source_count,
    )

    if feature_columns:
        if timeline[feature_columns].shape[1] != expected_source_count:
            raise AssertionError(
                "Timeline feature count does not match registry count."
            )

    return (
        timeline,
        audit,
        failures,
    )


def validate_pregame_team_identity_timeline(
    timeline: pd.DataFrame,
    audit: pd.DataFrame,
    expected_source_count: int = 87,
) -> None:
    if timeline.empty:
        raise ValueError(
            "Pregame team identity timeline is empty."
        )

    duplicate_count = int(
        timeline.duplicated(
            subset=list(JOIN_KEYS),
            keep=False,
        ).sum()
    )
    if duplicate_count:
        raise AssertionError(
            "Timeline contains duplicate team-game rows: "
            f"{duplicate_count:,}"
        )

    feature_columns = [
        column
        for column in timeline.columns
        if column not in set(CONTEXT_COLUMNS)
        and column
        not in {
            "identity_source_game_pk",
            "identity_source_game_date",
            "strict_backtest_safe",
            "same_date_games_used",
            "future_games_used",
            "phase",
            "timeline_engine_version",
        }
    ]

    if len(feature_columns) < int(expected_source_count):
        raise AssertionError(
            "Timeline identity feature count is lower than expected: "
            f"{len(feature_columns)} < {expected_source_count}"
        )

    if not timeline["strict_backtest_safe"].all():
        raise AssertionError(
            "Timeline must be strict backtest safe."
        )

    if timeline["same_date_games_used"].any():
        raise AssertionError(
            "Timeline cannot use same-date games."
        )

    if timeline["future_games_used"].any():
        raise AssertionError(
            "Timeline cannot use future games."
        )

    if audit.empty:
        raise ValueError(
            "Timeline audit is empty."
        )

    if not audit["audit_pass"].all():
        raise AssertionError(
            "Timeline chronology audit failed."
        )


def assert_reproduces_reference_timeline(
    timeline: pd.DataFrame,
    reference_timeline: pd.DataFrame,
    key_columns: tuple[str, ...] = JOIN_KEYS,
) -> None:
    if list(timeline.columns) != list(reference_timeline.columns):
        raise AssertionError(
            "Timeline schema mismatch against reference artifact."
        )

    if len(timeline) != len(reference_timeline):
        raise AssertionError(
            "Timeline row-count mismatch against reference artifact."
        )

    timeline_duplicates = int(
        timeline.duplicated(
            subset=list(key_columns),
            keep=False,
        ).sum()
    )
    reference_duplicates = int(
        reference_timeline.duplicated(
            subset=list(key_columns),
            keep=False,
        ).sum()
    )

    if timeline_duplicates or reference_duplicates:
        raise AssertionError(
            "Timeline key uniqueness mismatch against reference artifact."
        )

    left = timeline.sort_values(
        list(key_columns),
        kind="stable",
    ).reset_index(drop=True)

    right = reference_timeline.sort_values(
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
            "Timeline value mismatch against reference artifact."
        ) from exc


def phase_2e_team_identity_timeline_paths(
    season: int,
    data_root: Path | None = None,
) -> dict[str, Path]:
    root = Path(data_root) if data_root is not None else DATA_ROOT

    base = (
        root
        / "game_intelligence"
        / "pregame_team_identities"
        / str(int(season))
    )

    return {
        "base_dir": base,
        "timeline_parquet": base / "pregame_team_identity_timeline.parquet",
        "audit_parquet": base / "pregame_team_identity_timeline_audit.parquet",
        "failure_parquet": base / "pregame_team_identity_timeline_failures.parquet",
        "metadata_json": base / "pregame_team_identity_timeline_metadata.json",
    }


def build_pregame_team_identity_timeline_metadata(
    timeline: pd.DataFrame,
    audit: pd.DataFrame,
    failures: pd.DataFrame,
    season: int,
    expected_source_count: int = 87,
) -> dict[str, object]:
    validate_pregame_team_identity_timeline(
        timeline,
        audit=audit,
        expected_source_count=expected_source_count,
    )

    feature_columns = [
        column
        for column in timeline.columns
        if column not in set(CONTEXT_COLUMNS)
        and column
        not in {
            "identity_source_game_pk",
            "identity_source_game_date",
            "strict_backtest_safe",
            "same_date_games_used",
            "future_games_used",
            "phase",
            "timeline_engine_version",
        }
    ]

    return {
        "phase": "2E.2",
        "season": int(season),
        "team_game_rows": int(len(timeline)),
        "unique_games": int(timeline["game_pk"].nunique()),
        "teams": int(timeline["team"].nunique()),
        "identity_features": int(len(feature_columns)),
        "date_team_audit_rows": int(len(audit)),
        "audit_failures": int(len(failures)),
        "same_date_games_used": False,
        "future_games_used": False,
        "timeline_engine_version": ENGINE_VERSION,
    }


def save_pregame_team_identity_timeline(
    timeline: pd.DataFrame,
    audit: pd.DataFrame,
    failures: pd.DataFrame,
    season: int,
    data_root: Path | None = None,
    expected_source_count: int = 87,
) -> dict[str, Path]:
    metadata = build_pregame_team_identity_timeline_metadata(
        timeline=timeline,
        audit=audit,
        failures=failures,
        season=season,
        expected_source_count=expected_source_count,
    )

    paths = phase_2e_team_identity_timeline_paths(
        season=season,
        data_root=data_root,
    )

    paths["base_dir"].mkdir(
        parents=True,
        exist_ok=True,
    )

    timeline.to_parquet(
        paths["timeline_parquet"],
        index=False,
    )

    audit.to_parquet(
        paths["audit_parquet"],
        index=False,
    )

    failures.to_parquet(
        paths["failure_parquet"],
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
