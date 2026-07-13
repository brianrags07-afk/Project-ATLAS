
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"

TEAM_GAME_STATE_PATH = (
    DATA_DIR
    / "master"
    / "team_game_state.parquet"
)

PREGAME_INTERACTIONS_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "pregame"
    / "bullpen"
)

BULLPEN_STATE_PATH = (
    OUTPUT_DIR
    / "bullpen_pregame_state.parquet"
)

BULLPEN_DAILY_HISTORY_PATH = (
    OUTPUT_DIR
    / "bullpen_daily_history.parquet"
)

BULLPEN_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bullpen_state_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "bullpen_state_metadata.json"
)


REQUIRED_HISTORY_COLUMNS = [
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
    "bullpen_pitches",
    "bullpen_whiffs",
    "bullpen_strikeouts",
    "bullpen_walks",
    "bullpen_hits_allowed",
    "bullpen_runs_allowed",
]

REQUIRED_PREGAME_COLUMNS = [
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
]


def _load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_parquet(path)


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(destination)


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary.replace(destination)


def _require_columns(
    dataframe: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{label} is missing columns: {missing}"
        )


def _normalize_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    history = history.copy()

    _require_columns(
        history,
        REQUIRED_HISTORY_COLUMNS,
        "team_game_state",
    )

    history["game_date"] = pd.to_datetime(
        history["game_date"],
        errors="raise",
    ).dt.normalize()

    history["team"] = (
        history["team"]
        .astype(str)
        .str.upper()
    )

    numeric_columns = [
        "bullpen_pitches",
        "bullpen_whiffs",
        "bullpen_strikeouts",
        "bullpen_walks",
        "bullpen_hits_allowed",
        "bullpen_runs_allowed",
    ]

    for column in numeric_columns:
        history[column] = (
            pd.to_numeric(
                history[column],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

    history["bullpen_used"] = (
        history["bullpen_pitches"].gt(0)
    ).astype("int8")

    return history.sort_values(
        [
            "team",
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _build_daily_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    # Same-date games are aggregated before lagging.
    # This prevents one game of a doubleheader from leaking
    # into another same-date pregame snapshot.
    daily = (
        history.groupby(
            [
                "team",
                "atlas_season",
                "game_date",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            games_played_that_date=(
                "game_pk",
                "nunique",
            ),
            bullpen_pitches=(
                "bullpen_pitches",
                "sum",
            ),
            bullpen_whiffs=(
                "bullpen_whiffs",
                "sum",
            ),
            bullpen_strikeouts=(
                "bullpen_strikeouts",
                "sum",
            ),
            bullpen_walks=(
                "bullpen_walks",
                "sum",
            ),
            bullpen_hits_allowed=(
                "bullpen_hits_allowed",
                "sum",
            ),
            bullpen_runs_allowed=(
                "bullpen_runs_allowed",
                "sum",
            ),
            bullpen_used=(
                "bullpen_used",
                "max",
            ),
        )
    )

    daily = daily.sort_values(
        [
            "team",
            "game_date",
        ],
        kind="stable",
    ).reset_index(drop=True)

    daily["bullpen_whiff_per_pitch"] = np.where(
        daily["bullpen_pitches"].gt(0),
        daily["bullpen_whiffs"]
        / daily["bullpen_pitches"],
        np.nan,
    )

    daily["bullpen_strikeout_per_pitch"] = np.where(
        daily["bullpen_pitches"].gt(0),
        daily["bullpen_strikeouts"]
        / daily["bullpen_pitches"],
        np.nan,
    )

    daily["bullpen_walk_per_pitch"] = np.where(
        daily["bullpen_pitches"].gt(0),
        daily["bullpen_walks"]
        / daily["bullpen_pitches"],
        np.nan,
    )

    daily["bullpen_hits_per_pitch"] = np.where(
        daily["bullpen_pitches"].gt(0),
        daily["bullpen_hits_allowed"]
        / daily["bullpen_pitches"],
        np.nan,
    )

    daily["bullpen_runs_per_pitch"] = np.where(
        daily["bullpen_pitches"].gt(0),
        daily["bullpen_runs_allowed"]
        / daily["bullpen_pitches"],
        np.nan,
    )

    return daily


def _rolling_prior_sum(
    series: pd.Series,
    window: int,
) -> pd.Series:
    return (
        series.shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .sum()
    )


def _rolling_prior_mean(
    series: pd.Series,
    window: int,
) -> pd.Series:
    return (
        series.shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )


def _expanding_prior_mean(
    series: pd.Series,
) -> pd.Series:
    return (
        series.shift(1)
        .expanding(min_periods=1)
        .mean()
    )


def _expanding_prior_std(
    series: pd.Series,
) -> pd.Series:
    return (
        series.shift(1)
        .expanding(min_periods=3)
        .std()
    )



def _build_team_date_states(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create strictly pregame bullpen states using prior
    CALENDAR days, not prior game rows.

    Doubleheaders were already aggregated in daily history,
    so no same-date result can enter another same-date snapshot.
    """
    records: list[dict[str, Any]] = []

    count_columns = [
        "bullpen_pitches",
        "bullpen_whiffs",
        "bullpen_strikeouts",
        "bullpen_walks",
        "bullpen_hits_allowed",
        "bullpen_runs_allowed",
    ]

    rate_names = {
        "bullpen_whiff_per_pitch":
            "bullpen_whiffs",
        "bullpen_strikeout_per_pitch":
            "bullpen_strikeouts",
        "bullpen_walk_per_pitch":
            "bullpen_walks",
        "bullpen_hits_per_pitch":
            "bullpen_hits_allowed",
        "bullpen_runs_per_pitch":
            "bullpen_runs_allowed",
    }

    grouped = daily.groupby(
        [
            "team",
            "atlas_season",
        ],
        sort=True,
    )

    for (
        team,
        season,
    ), group in grouped:
        group = (
            group.sort_values(
                "game_date",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        usage_by_date = {
            row.game_date: int(
                row.bullpen_used
            )
            for row in group.itertuples(
                index=False
            )
        }

        for position, row in group.iterrows():
            current_date = row["game_date"]

            prior = group.loc[
                group["game_date"].lt(
                    current_date
                )
            ].copy()

            record = row.to_dict()

            # ------------------------------------------------
            # Most recent prior bullpen-use date
            # ------------------------------------------------
            prior_used = prior.loc[
                prior["bullpen_used"].eq(1)
            ]

            if prior_used.empty:
                prior_bullpen_date = pd.NaT
                days_since_prior = np.nan

            else:
                prior_bullpen_date = (
                    prior_used["game_date"].max()
                )

                days_since_prior = int(
                    (
                        current_date
                        - prior_bullpen_date
                    ).days
                )

            record[
                "prior_bullpen_date"
            ] = prior_bullpen_date

            record[
                "days_since_prior_bullpen_date"
            ] = days_since_prior

            # ------------------------------------------------
            # True calendar-day workload windows
            # ------------------------------------------------
            for window in [
                1,
                2,
                3,
                5,
                7,
            ]:
                window_start = (
                    current_date
                    - pd.Timedelta(
                        days=window
                    )
                )

                recent = prior.loc[
                    prior["game_date"].ge(
                        window_start
                    )
                ]

                record[
                    f"bullpen_pitches_prior_{window}_dates"
                ] = float(
                    recent[
                        "bullpen_pitches"
                    ].sum()
                )

                record[
                    f"bullpen_games_used_prior_{window}_dates"
                ] = float(
                    recent[
                        "bullpen_used"
                    ].sum()
                )

            for column in count_columns[1:]:
                for window in [3, 5]:
                    window_start = (
                        current_date
                        - pd.Timedelta(
                            days=window
                        )
                    )

                    recent = prior.loc[
                        prior["game_date"].ge(
                            window_start
                        )
                    ]

                    record[
                        f"{column}_prior_{window}_dates"
                    ] = float(
                        recent[column].sum()
                    )

            # ------------------------------------------------
            # Pitch-weighted recent rates
            # ------------------------------------------------
            prior_5_start = (
                current_date
                - pd.Timedelta(days=5)
            )

            recent_5 = prior.loc[
                prior["game_date"].ge(
                    prior_5_start
                )
            ]

            recent_5_pitches = float(
                recent_5[
                    "bullpen_pitches"
                ].sum()
            )

            season_prior_pitches = float(
                prior[
                    "bullpen_pitches"
                ].sum()
            )

            for rate_name, numerator in (
                rate_names.items()
            ):
                recent_numerator = float(
                    recent_5[
                        numerator
                    ].sum()
                )

                season_numerator = float(
                    prior[
                        numerator
                    ].sum()
                )

                record[
                    f"{rate_name}_prior_5_dates"
                ] = (
                    recent_numerator
                    / recent_5_pitches
                    if recent_5_pitches > 0
                    else np.nan
                )

                record[
                    f"{rate_name}_season_prior_mean"
                ] = (
                    season_numerator
                    / season_prior_pitches
                    if season_prior_pitches > 0
                    else np.nan
                )

            # ------------------------------------------------
            # Season-prior workload baselines
            # ------------------------------------------------
            record[
                "bullpen_pitches_season_prior_mean"
            ] = (
                float(
                    prior[
                        "bullpen_pitches"
                    ].mean()
                )
                if not prior.empty
                else np.nan
            )

            record[
                "bullpen_pitches_season_prior_std"
            ] = (
                float(
                    prior[
                        "bullpen_pitches"
                    ].std()
                )
                if len(prior) >= 3
                else np.nan
            )

            for column in [
                "bullpen_runs_allowed",
                "bullpen_walks",
                "bullpen_hits_allowed",
            ]:
                record[
                    f"{column}_season_prior_mean"
                ] = (
                    float(
                        prior[column].mean()
                    )
                    if not prior.empty
                    else np.nan
                )

            # ------------------------------------------------
            # True immediately preceding calendar-day streak
            # ------------------------------------------------
            consecutive_usage_days = 0
            check_date = (
                current_date
                - pd.Timedelta(days=1)
            )

            while usage_by_date.get(
                check_date,
                0,
            ) == 1:
                consecutive_usage_days += 1

                check_date = (
                    check_date
                    - pd.Timedelta(days=1)
                )

            record[
                "bullpen_consecutive_prior_usage_dates"
            ] = consecutive_usage_days

            # ------------------------------------------------
            # Workload z-score
            # ------------------------------------------------
            prior_mean = record[
                "bullpen_pitches_season_prior_mean"
            ]

            prior_std = record[
                "bullpen_pitches_season_prior_std"
            ]

            recent_three_average = (
                record[
                    "bullpen_pitches_prior_3_dates"
                ]
                / 3.0
            )

            if (
                pd.notna(prior_mean)
                and pd.notna(prior_std)
                and prior_std > 0
            ):
                workload_zscore = (
                    recent_three_average
                    - prior_mean
                ) / prior_std

            else:
                workload_zscore = 0.0

            record[
                "bullpen_recent_workload_zscore"
            ] = float(
                np.clip(
                    workload_zscore,
                    -4.0,
                    4.0,
                )
            )

            # ------------------------------------------------
            # Rest and workload pressure
            # ------------------------------------------------
            if pd.isna(days_since_prior):
                recovery_score = 1.00
            elif days_since_prior >= 3:
                recovery_score = 1.00
            elif days_since_prior == 2:
                recovery_score = 0.78
            elif days_since_prior == 1:
                recovery_score = 0.38
            else:
                recovery_score = 0.20

            record[
                "bullpen_rest_recovery_score"
            ] = recovery_score

            pitches_1 = record[
                "bullpen_pitches_prior_1_dates"
            ]

            pitches_3 = record[
                "bullpen_pitches_prior_3_dates"
            ]

            pitches_5 = record[
                "bullpen_pitches_prior_5_dates"
            ]

            pressure_score = (
                0.40
                * np.clip(
                    pitches_1 / 110.0,
                    0.0,
                    1.25,
                )
                + 0.35
                * np.clip(
                    pitches_3 / 250.0,
                    0.0,
                    1.25,
                )
                + 0.15
                * np.clip(
                    pitches_5 / 400.0,
                    0.0,
                    1.25,
                )
                + 0.10
                * np.clip(
                    consecutive_usage_days
                    / 3.0,
                    0.0,
                    1.25,
                )
            )

            pressure_score = float(
                np.clip(
                    pressure_score,
                    0.0,
                    1.0,
                )
            )

            record[
                "bullpen_workload_pressure_score"
            ] = pressure_score

            fatigue_score = (
                0.85
                * pressure_score
                + 0.15
                * (
                    1.0
                    - recovery_score
                )
            )

            fatigue_score = float(
                np.clip(
                    fatigue_score,
                    0.0,
                    1.0,
                )
            )

            record[
                "bullpen_fatigue_score"
            ] = fatigue_score

            record[
                "bullpen_availability_proxy_score"
            ] = float(
                1.0
                - fatigue_score
            )

            # ------------------------------------------------
            # Recent effectiveness
            # ------------------------------------------------
            runs_rate = record[
                "bullpen_runs_per_pitch_prior_5_dates"
            ]

            walks_rate = record[
                "bullpen_walk_per_pitch_prior_5_dates"
            ]

            hits_rate = record[
                "bullpen_hits_per_pitch_prior_5_dates"
            ]

            whiff_rate = record[
                "bullpen_whiff_per_pitch_prior_5_dates"
            ]

            effectiveness_parts = []

            if pd.notna(runs_rate):
                effectiveness_parts.append(
                    (
                        1.0
                        - np.clip(
                            runs_rate / 0.08,
                            0.0,
                            1.0,
                        ),
                        0.35,
                    )
                )

            if pd.notna(walks_rate):
                effectiveness_parts.append(
                    (
                        1.0
                        - np.clip(
                            walks_rate / 0.06,
                            0.0,
                            1.0,
                        ),
                        0.25,
                    )
                )

            if pd.notna(hits_rate):
                effectiveness_parts.append(
                    (
                        1.0
                        - np.clip(
                            hits_rate / 0.12,
                            0.0,
                            1.0,
                        ),
                        0.20,
                    )
                )

            if pd.notna(whiff_rate):
                effectiveness_parts.append(
                    (
                        np.clip(
                            whiff_rate / 0.15,
                            0.0,
                            1.0,
                        ),
                        0.20,
                    )
                )

            if effectiveness_parts:
                effectiveness_score = (
                    sum(
                        value * weight
                        for value, weight
                        in effectiveness_parts
                    )
                    / sum(
                        weight
                        for _, weight
                        in effectiveness_parts
                    )
                )
            else:
                effectiveness_score = np.nan

            record[
                "bullpen_recent_effectiveness_score"
            ] = effectiveness_score

            # ------------------------------------------------
            # State label
            # ------------------------------------------------
            if fatigue_score >= 0.78:
                state_label = "OVERWORKED"

            elif fatigue_score >= 0.58:
                state_label = "FATIGUED"

            elif (
                fatigue_score < 0.30
                and pd.notna(
                    effectiveness_score
                )
                and effectiveness_score >= 0.65
            ):
                state_label = "FRESH_EFFECTIVE"

            elif fatigue_score < 0.30:
                state_label = "FRESH"

            else:
                state_label = "NORMAL"

            record[
                "bullpen_state_label"
            ] = state_label

            records.append(record)

    return pd.DataFrame(records)

def _prepare_pregame_rows(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    interactions = interactions.copy()

    _require_columns(
        interactions,
        REQUIRED_PREGAME_COLUMNS,
        "pregame interactions",
    )

    interactions["game_date"] = pd.to_datetime(
        interactions["game_date"],
        errors="raise",
    ).dt.normalize()

    interactions["team"] = (
        interactions["team"]
        .astype(str)
        .str.upper()
    )

    return (
        interactions[
            REQUIRED_PREGAME_COLUMNS
        ]
        .drop_duplicates(
            subset=[
                "game_pk",
                "team",
            ]
        )
        .sort_values(
            [
                "game_date",
                "game_pk",
                "team",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _attach_states_to_games(
    pregame_rows: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    state_columns = [
        column
        for column in states.columns
        if column not in {
            "games_played_that_date",
            "bullpen_pitches",
            "bullpen_whiffs",
            "bullpen_strikeouts",
            "bullpen_walks",
            "bullpen_hits_allowed",
            "bullpen_runs_allowed",
            "bullpen_used",
            "bullpen_whiff_per_pitch",
            "bullpen_strikeout_per_pitch",
            "bullpen_walk_per_pitch",
            "bullpen_hits_per_pitch",
            "bullpen_runs_per_pitch",
        }
    ]

    state_table = states[
        state_columns
    ].copy()

    merged = pregame_rows.merge(
        state_table,
        on=[
            "team",
            "atlas_season",
            "game_date",
        ],
        how="left",
        validate="many_to_one",
        suffixes=("", "_state"),
    )

    merged[
        "bullpen_snapshot_available"
    ] = (
        merged[
            "bullpen_pitches_prior_1_dates"
        ].notna()
    )

    merged[
        "strict_pregame_safe"
    ] = True

    merged[
        "current_game_outcome_used"
    ] = False

    merged[
        "same_date_games_used"
    ] = False

    merged[
        "future_games_used"
    ] = False

    merged[
        "specific_reliever_availability_known"
    ] = False

    merged[
        "availability_is_team_level_proxy"
    ] = True

    merged[
        "bullpen_engine_version"
    ] = ENGINE_VERSION

    merged[
        "built_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    return merged.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)


def run_bullpen_availability_fatigue_engine() -> dict[str, Any]:
    history = _normalize_history(
        _load_parquet(
            TEAM_GAME_STATE_PATH,
            "team game state",
        )
    )

    interactions = _prepare_pregame_rows(
        _load_parquet(
            PREGAME_INTERACTIONS_PATH,
            "pregame interaction rows",
        )
    )

    daily_history = _build_daily_history(
        history
    )

    states = _build_team_date_states(
        daily_history
    )

    snapshots = _attach_states_to_games(
        pregame_rows=interactions,
        states=states,
    )

    duplicate_rows = int(
        snapshots.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if duplicate_rows:
        raise AssertionError(
            f"Duplicate team-game bullpen states: "
            f"{duplicate_rows}"
        )

    if snapshots[
        "current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Current-game outcomes were used."
        )

    if snapshots[
        "same_date_games_used"
    ].any():
        raise AssertionError(
            "Same-date games were used."
        )

    if snapshots[
        "future_games_used"
    ].any():
        raise AssertionError(
            "Future games were used."
        )

    summary = (
        snapshots.groupby(
            [
                "atlas_season",
                "bullpen_state_label",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            team_game_rows=(
                "game_pk",
                "size",
            ),
            teams=(
                "team",
                "nunique",
            ),
            snapshot_coverage=(
                "bullpen_snapshot_available",
                "mean",
            ),
            mean_fatigue_score=(
                "bullpen_fatigue_score",
                "mean",
            ),
            mean_availability_proxy=(
                "bullpen_availability_proxy_score",
                "mean",
            ),
            mean_effectiveness_score=(
                "bullpen_recent_effectiveness_score",
                "mean",
            ),
        )
        .reset_index()
    )

    _atomic_parquet_write(
        daily_history,
        BULLPEN_DAILY_HISTORY_PATH,
    )

    _atomic_parquet_write(
        snapshots,
        BULLPEN_STATE_PATH,
    )

    _atomic_parquet_write(
        summary,
        BULLPEN_SUMMARY_PATH,
    )

    result = {
        "engine":
            "ATLAS Bullpen Availability and Fatigue Engine",
        "engine_version":
            ENGINE_VERSION,
        "history_rows":
            int(len(history)),
        "daily_history_rows":
            int(len(daily_history)),
        "pregame_team_game_rows":
            int(len(snapshots)),
        "unique_games":
            int(
                snapshots["game_pk"].nunique()
            ),
        "teams":
            int(
                snapshots["team"].nunique()
            ),
        "snapshot_rows_available":
            int(
                snapshots[
                    "bullpen_snapshot_available"
                ].sum()
            ),
        "snapshot_coverage":
            float(
                snapshots[
                    "bullpen_snapshot_available"
                ].mean()
            ),
        "duplicate_team_games":
            duplicate_rows,
        "specific_reliever_availability_known":
            False,
        "outputs": {
            "daily_history":
                str(BULLPEN_DAILY_HISTORY_PATH),
            "pregame_states":
                str(BULLPEN_STATE_PATH),
            "summary":
                str(BULLPEN_SUMMARY_PATH),
        },
        "pregame_safety": {
            "current_game_outcome_used":
                False,
            "same_date_games_used":
                False,
            "future_games_used":
                False,
            "daily_history_shifted_before_features":
                True,
        },
        "limitations": {
            "team_level_only":
                True,
            "closer_identity_available":
                False,
            "setup_identity_available":
                False,
            "specific_reliever_availability_available":
                False,
            "availability_is_proxy":
                True,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS BULLPEN AVAILABILITY & FATIGUE ENGINE")
    print("=" * 78)
    print(
        f"Historical Team-Games....... "
        f"{len(history):,}"
    )
    print(
        f"Team-Date History Rows....... "
        f"{len(daily_history):,}"
    )
    print(
        f"Pregame Team-Game Rows....... "
        f"{len(snapshots):,}"
    )
    print(
        f"Unique Games................. "
        f"{snapshots['game_pk'].nunique():,}"
    )
    print(
        f"Teams........................ "
        f"{snapshots['team'].nunique():,}"
    )
    print(
        f"Snapshot Coverage............ "
        f"{result['snapshot_coverage']:.2%}"
    )
    print(
        f"Duplicate Team-Games......... "
        f"{duplicate_rows:,}"
    )
    print(
        "Current-Game Outcomes Used... False"
    )
    print(
        "Same-Date Games Used......... False"
    )
    print(
        "Future Games Used............ False"
    )
    print(
        "Specific Reliever Status..... Not Yet Available"
    )
    print(
        f"Saved To..................... "
        f"{OUTPUT_DIR}"
    )
    print("=" * 78)

    return result
