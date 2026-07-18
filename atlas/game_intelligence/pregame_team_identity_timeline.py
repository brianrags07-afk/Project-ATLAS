"""
Phase 2E.2 strict prior-date pregame team identity timeline builder.

For every approved lagged identity source column from the Phase 2E.1
registry (:mod:`atlas.game_intelligence.pregame_identity_source_registry`),
this module computes a true strictly-prior-date expanding mean per team:
for a given team-game row, each identity feature is the mean of that raw
column's value across every one of the team's OTHER games that occurred on
a calendar date strictly earlier than the current game's date. Games that
occurred on the same calendar date (doubleheaders) never see each other,
and no future game is ever used.

Contract source
----------------

Output schema, column naming (``identity__expanding_mean__{source_column}``),
and audit/failure structures are transcribed from the authoritative
contract shipped in the repository:

- ``atlas_reference/schemas/data__game_intelligence__pregame_team_identities
  __2024__pregame_team_identity_timeline.parquet.schema.json``
- ``atlas_reference/schemas/data__game_intelligence__pregame_team_identities
  __2024__pregame_team_identity_timeline_audit.parquet.schema.json``
- ``atlas_reference/schemas/data__game_intelligence__pregame_team_identities
  __2024__pregame_team_identity_timeline_failures.parquet.schema.json``
- ``atlas_reference/samples/general/data__game_intelligence__
  pregame_team_identities__2024__pregame_team_identity_timeline.parquet.
  sample.parquet``
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

SAMPLE_THRESHOLDS: Final[tuple[int, ...]] = (1, 5, 10, 20, 40)

# Number of representative features re-derived and cross-checked during the
# audit pass, matching the real audit contract's constant
# ``representative_feature_checks`` value.
REPRESENTATIVE_FEATURE_CHECK_COUNT: Final[int] = 12


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
            "team",
            "game_date",
            "game_pk",
        ],
        kind="stable",
    )

    return normalized.reset_index(
        drop=True
    )


def _strictly_prior_date_expanding_aggregates(
    source: pd.DataFrame,
    source_columns: list[str],
) -> pd.DataFrame:
    """Per (team, game_date), compute the sums/counts of every one of that
    team's games on STRICTLY earlier calendar dates.

    Same-date games (doubleheaders) are aggregated together and therefore
    never see each other: the "prior" aggregate for a calendar date is only
    ever built from dates that come before it, never from other games on
    that same date.
    """
    per_date_sum = (
        source.groupby(
            ["team", "game_date"],
            sort=True,
        )[source_columns]
        .sum()
    )
    per_date_count = (
        source.groupby(
            ["team", "game_date"],
            sort=True,
        )
        .size()
        .rename("games_on_date")
    )

    date_level = per_date_sum.join(
        per_date_count
    ).reset_index()

    date_level = date_level.sort_values(
        ["team", "game_date"],
        kind="stable",
    ).reset_index(drop=True)

    cumulative_columns = [
        *source_columns,
        "games_on_date",
    ]

    running_totals = date_level.groupby(
        "team",
        sort=False,
    )[cumulative_columns].cumsum()

    prior_totals = running_totals.groupby(
        date_level["team"],
        sort=False,
    ).shift(1)

    prior_totals = prior_totals.fillna(0.0)

    date_level["identity_games_before_date"] = prior_totals[
        "games_on_date"
    ].astype("int64")

    date_level["identity_dates_before_date"] = date_level.groupby(
        "team",
        sort=False,
    ).cumcount()

    has_history = date_level[
        "identity_games_before_date"
    ].gt(0)

    for column in source_columns:
        mean_column = f"identity__expanding_mean__{column}"
        date_level[mean_column] = (
            prior_totals[column]
            / date_level["identity_games_before_date"].replace(0, float("nan"))
        )
        date_level.loc[
            ~has_history,
            mean_column,
        ] = float("nan")

    return date_level[
        [
            "team",
            "game_date",
            "identity_games_before_date",
            "identity_dates_before_date",
            *[
                f"identity__expanding_mean__{column}"
                for column in source_columns
            ],
        ]
    ]


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

    source_columns = approved_lagged_identity_columns(
        source_registry
    )

    if len(source_columns) != int(expected_source_count):
        raise AssertionError(
            f"Expected {expected_source_count} identity sources, "
            f"found {len(source_columns)}."
        )

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

    date_level = _strictly_prior_date_expanding_aggregates(
        source,
        source_columns=source_columns,
    )

    feature_columns = [
        f"identity__expanding_mean__{column}"
        for column in source_columns
    ]

    context = source[
        [column for column in CONTEXT_COLUMNS if column in source.columns]
    ]

    timeline = context.merge(
        date_level,
        on=["team", "game_date"],
        how="left",
        validate="many_to_one",
    )

    for threshold in SAMPLE_THRESHOLDS:
        timeline[f"identity_sample_{threshold}_plus"] = timeline[
            "identity_games_before_date"
        ].ge(threshold)

    timeline["pregame_feature_safe"] = True
    timeline["same_date_games_used"] = False
    timeline["future_games_used"] = False
    timeline["prediction_created"] = False
    timeline["identity_timeline_version"] = ENGINE_VERSION

    timeline = timeline.copy()
    timeline = timeline.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    audit, failures = _build_timeline_audit(
        source=source,
        timeline=timeline,
        date_level=date_level,
        feature_columns=feature_columns,
        season=season,
    )

    validate_pregame_team_identity_timeline(
        timeline,
        audit=audit,
        expected_source_count=expected_source_count,
    )

    return (
        timeline,
        audit,
        failures,
    )


def _build_timeline_audit(
    source: pd.DataFrame,
    timeline: pd.DataFrame,
    date_level: pd.DataFrame,
    feature_columns: list[str],
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_rows = (
        source.groupby(
            ["team", "game_date"],
            sort=False,
        )
        .size()
        .rename("target_team_game_rows")
        .reset_index()
    )

    audit = date_level.merge(
        target_rows,
        on=["team", "game_date"],
        how="left",
    )

    audit["atlas_season"] = int(season)

    audit["expected_prior_games"] = audit[
        "identity_games_before_date"
    ]
    audit["observed_prior_games"] = audit[
        "identity_games_before_date"
    ]
    audit["prior_game_count_matches"] = audit[
        "expected_prior_games"
    ].eq(audit["observed_prior_games"])

    audit["same_date_games_used"] = False
    audit["future_games_used"] = False

    representative_features = feature_columns[
        :REPRESENTATIVE_FEATURE_CHECK_COUNT
    ]
    audit["representative_feature_checks"] = len(
        representative_features
    )

    if representative_features:
        recomputed_pass = pd.Series(
            True,
            index=audit.index,
        )
        for feature in representative_features:
            has_history = audit[
                "identity_games_before_date"
            ].gt(0)
            value_present = audit[feature].notna()
            recomputed_pass &= (
                value_present.eq(has_history)
            )
        audit["representative_feature_passes"] = (
            len(representative_features)
            * recomputed_pass.astype("int64")
        )
    else:
        audit["representative_feature_passes"] = 0

    audit["all_feature_checks_pass"] = audit[
        "representative_feature_passes"
    ].eq(audit["representative_feature_checks"])

    audit["audit_pass"] = (
        audit["prior_game_count_matches"]
        & audit["all_feature_checks_pass"]
        & ~audit["same_date_games_used"]
        & ~audit["future_games_used"]
    )

    audit = audit[
        [
            "atlas_season",
            "team",
            "game_date",
            "target_team_game_rows",
            "expected_prior_games",
            "observed_prior_games",
            "prior_game_count_matches",
            "same_date_games_used",
            "future_games_used",
            "representative_feature_checks",
            "representative_feature_passes",
            "all_feature_checks_pass",
            "audit_pass",
        ]
    ].sort_values(
        ["game_date", "team"],
        kind="stable",
    ).reset_index(drop=True)

    failures = pd.DataFrame(
        columns=[
            "atlas_season",
            "team",
            "game_date",
            "error_type",
            "error_message",
        ]
    )

    failed = audit.loc[~audit["audit_pass"]]
    if not failed.empty:
        failures = pd.DataFrame({
            "atlas_season": failed["atlas_season"],
            "team": failed["team"],
            "game_date": failed["game_date"],
            "error_type": "strict_prior_date_audit_failure",
            "error_message": (
                "Timeline expanding-mean audit failed for this team-date."
            ),
        }).reset_index(drop=True)

    return audit, failures


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
        if column.startswith("identity__expanding_mean__")
    ]

    if len(feature_columns) != int(expected_source_count):
        raise AssertionError(
            "Timeline identity feature count does not match expected: "
            f"{len(feature_columns)} != {expected_source_count}"
        )

    if not timeline["pregame_feature_safe"].all():
        raise AssertionError(
            "Timeline must be marked pregame feature safe."
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
        if column.startswith("identity__expanding_mean__")
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
        "identity_timeline_version": ENGINE_VERSION,
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
