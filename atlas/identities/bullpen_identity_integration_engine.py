
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"

BASE_INTERACTIONS_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
)

BULLPEN_STATES_PATH = (
    DATA_DIR
    / "pregame"
    / "bullpen"
    / "bullpen_pregame_state.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "pregame"
    / "interactions"
)

ENRICHED_INTERACTIONS_PATH = (
    OUTPUT_DIR
    / "lineup_starter_bullpen_inputs.parquet"
)

FEATURE_REGISTRY_DIR = (
    DATA_DIR
    / "pregame"
    / "feature_registry"
)

FEATURE_REGISTRY_PATH = (
    FEATURE_REGISTRY_DIR
    / "bullpen_identity_feature_registry.parquet"
)

INTEGRATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bullpen_identity_integration_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "bullpen_identity_integration_metadata.json"
)


KEY_COLUMNS = [
    "game_pk",
    "game_date",
    "atlas_season",
    "team",
    "opponent",
    "home_away",
]


# These are approved as pregame-safe identity features.
BASE_BULLPEN_FEATURES = [
    "days_since_prior_bullpen_date",

    "bullpen_pitches_prior_1_dates",
    "bullpen_pitches_prior_2_dates",
    "bullpen_pitches_prior_3_dates",
    "bullpen_pitches_prior_5_dates",
    "bullpen_pitches_prior_7_dates",

    "bullpen_games_used_prior_1_dates",
    "bullpen_games_used_prior_2_dates",
    "bullpen_games_used_prior_3_dates",
    "bullpen_games_used_prior_5_dates",
    "bullpen_games_used_prior_7_dates",

    "bullpen_whiffs_prior_3_dates",
    "bullpen_whiffs_prior_5_dates",
    "bullpen_strikeouts_prior_3_dates",
    "bullpen_strikeouts_prior_5_dates",
    "bullpen_walks_prior_3_dates",
    "bullpen_walks_prior_5_dates",
    "bullpen_hits_allowed_prior_3_dates",
    "bullpen_hits_allowed_prior_5_dates",
    "bullpen_runs_allowed_prior_3_dates",
    "bullpen_runs_allowed_prior_5_dates",

    "bullpen_whiff_per_pitch_prior_5_dates",
    "bullpen_strikeout_per_pitch_prior_5_dates",
    "bullpen_walk_per_pitch_prior_5_dates",
    "bullpen_hits_per_pitch_prior_5_dates",
    "bullpen_runs_per_pitch_prior_5_dates",

    "bullpen_whiff_per_pitch_season_prior_mean",
    "bullpen_strikeout_per_pitch_season_prior_mean",
    "bullpen_walk_per_pitch_season_prior_mean",
    "bullpen_hits_per_pitch_season_prior_mean",
    "bullpen_runs_per_pitch_season_prior_mean",

    "bullpen_pitches_season_prior_mean",
    "bullpen_pitches_season_prior_std",
    "bullpen_runs_allowed_season_prior_mean",
    "bullpen_walks_season_prior_mean",
    "bullpen_hits_allowed_season_prior_mean",

    "bullpen_consecutive_prior_usage_dates",
    "bullpen_recent_workload_zscore",
    "bullpen_workload_pressure_score",
    "bullpen_rest_recovery_score",
    "bullpen_fatigue_score",
    "bullpen_availability_proxy_score",
    "bullpen_recent_effectiveness_score",

    "bullpen_state_label",
    "bullpen_snapshot_available",
]


MATCHUP_FEATURES = [
    "bullpen_fatigue_edge",
    "bullpen_availability_edge",
    "bullpen_effectiveness_edge",
    "bullpen_workload_pressure_edge",
    "bullpen_rest_recovery_edge",
    "bullpen_pitches_prior_1_edge",
    "bullpen_pitches_prior_3_edge",
    "bullpen_pitches_prior_5_edge",
    "bullpen_usage_streak_edge",
    "bullpen_state_matchup",
    "both_bullpens_overworked",
    "both_bullpens_fatigued_or_worse",
    "team_bullpen_fresher",
    "opponent_bullpen_fresher",
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
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"{label} missing columns: {missing}"
        )


def _normalize_keys(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="raise",
    ).dt.normalize()

    for column in [
        "team",
        "opponent",
        "home_away",
    ]:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.upper()
        )

    return dataframe


def _prepare_bullpen_table(
    bullpen_states: pd.DataFrame,
) -> pd.DataFrame:
    bullpen_states = _normalize_keys(
        bullpen_states
    )

    required = (
        KEY_COLUMNS
        + BASE_BULLPEN_FEATURES
        + [
            "strict_pregame_safe",
            "current_game_outcome_used",
            "same_date_games_used",
            "future_games_used",
            "specific_reliever_availability_known",
            "availability_is_team_level_proxy",
            "bullpen_engine_version",
        ]
    )

    _require_columns(
        bullpen_states,
        required,
        "bullpen states",
    )

    duplicates = int(
        bullpen_states.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if duplicates:
        raise AssertionError(
            f"Duplicate bullpen team-games: {duplicates}"
        )

    if not bullpen_states[
        "strict_pregame_safe"
    ].all():
        raise AssertionError(
            "Some bullpen states are not marked pregame-safe."
        )

    if bullpen_states[
        "current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Current-game bullpen outcomes were used."
        )

    if bullpen_states[
        "same_date_games_used"
    ].any():
        raise AssertionError(
            "Same-date bullpen outcomes were used."
        )

    if bullpen_states[
        "future_games_used"
    ].any():
        raise AssertionError(
            "Future bullpen outcomes were used."
        )

    return bullpen_states[required].copy()


def _rename_identity_columns(
    bullpen_table: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    identity_columns = (
        BASE_BULLPEN_FEATURES
        + [
            "specific_reliever_availability_known",
            "availability_is_team_level_proxy",
            "bullpen_engine_version",
        ]
    )

    rename_map = {
        column: f"{prefix}{column}"
        for column in identity_columns
    }

    return bullpen_table.rename(
        columns=rename_map
    )


def _attach_team_bullpen(
    base: pd.DataFrame,
    bullpen_table: pd.DataFrame,
) -> pd.DataFrame:
    team_table = _rename_identity_columns(
        bullpen_table,
        "team_",
    )

    merge_columns = [
        "game_pk",
        "game_date",
        "atlas_season",
        "team",
    ]

    keep_columns = (
        merge_columns
        + [
            column
            for column in team_table.columns
            if column.startswith("team_")
        ]
    )

    return base.merge(
        team_table[keep_columns],
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )


def _attach_opponent_bullpen(
    base: pd.DataFrame,
    bullpen_table: pd.DataFrame,
) -> pd.DataFrame:
    opponent_table = bullpen_table.copy()

    opponent_table = opponent_table.rename(
        columns={
            "team": "opponent",
            "opponent": "team",
        }
    )

    opponent_table = _rename_identity_columns(
        opponent_table,
        "opponent_",
    )

    merge_columns = [
        "game_pk",
        "game_date",
        "atlas_season",
        "opponent",
    ]

    keep_columns = (
        merge_columns
        + [
            column
            for column in opponent_table.columns
            if column.startswith("opponent_")
        ]
    )

    return base.merge(
        opponent_table[keep_columns],
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )


def _numeric_difference(
    dataframe: pd.DataFrame,
    team_column: str,
    opponent_column: str,
) -> pd.Series:
    team_value = pd.to_numeric(
        dataframe[team_column],
        errors="coerce",
    )

    opponent_value = pd.to_numeric(
        dataframe[opponent_column],
        errors="coerce",
    )

    return team_value - opponent_value


def _create_matchup_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["bullpen_fatigue_edge"] = (
        _numeric_difference(
            dataframe,
            "opponent_bullpen_fatigue_score",
            "team_bullpen_fatigue_score",
        )
    )

    dataframe["bullpen_availability_edge"] = (
        _numeric_difference(
            dataframe,
            "team_bullpen_availability_proxy_score",
            "opponent_bullpen_availability_proxy_score",
        )
    )

    dataframe["bullpen_effectiveness_edge"] = (
        _numeric_difference(
            dataframe,
            "team_bullpen_recent_effectiveness_score",
            "opponent_bullpen_recent_effectiveness_score",
        )
    )

    dataframe[
        "bullpen_workload_pressure_edge"
    ] = _numeric_difference(
        dataframe,
        "opponent_bullpen_workload_pressure_score",
        "team_bullpen_workload_pressure_score",
    )

    dataframe["bullpen_rest_recovery_edge"] = (
        _numeric_difference(
            dataframe,
            "team_bullpen_rest_recovery_score",
            "opponent_bullpen_rest_recovery_score",
        )
    )

    dataframe["bullpen_pitches_prior_1_edge"] = (
        _numeric_difference(
            dataframe,
            "opponent_bullpen_pitches_prior_1_dates",
            "team_bullpen_pitches_prior_1_dates",
        )
    )

    dataframe["bullpen_pitches_prior_3_edge"] = (
        _numeric_difference(
            dataframe,
            "opponent_bullpen_pitches_prior_3_dates",
            "team_bullpen_pitches_prior_3_dates",
        )
    )

    dataframe["bullpen_pitches_prior_5_edge"] = (
        _numeric_difference(
            dataframe,
            "opponent_bullpen_pitches_prior_5_dates",
            "team_bullpen_pitches_prior_5_dates",
        )
    )

    dataframe["bullpen_usage_streak_edge"] = (
        _numeric_difference(
            dataframe,
            "opponent_bullpen_consecutive_prior_usage_dates",
            "team_bullpen_consecutive_prior_usage_dates",
        )
    )

    team_state = (
        dataframe["team_bullpen_state_label"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    opponent_state = (
        dataframe["opponent_bullpen_state_label"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    dataframe["bullpen_state_matchup"] = (
        team_state
        + "_VS_"
        + opponent_state
    )

    dataframe["both_bullpens_overworked"] = (
        team_state.eq("OVERWORKED")
        & opponent_state.eq("OVERWORKED")
    )

    fatigue_states = {
        "FATIGUED",
        "OVERWORKED",
    }

    dataframe[
        "both_bullpens_fatigued_or_worse"
    ] = (
        team_state.isin(fatigue_states)
        & opponent_state.isin(fatigue_states)
    )

    dataframe["team_bullpen_fresher"] = (
        dataframe[
            "bullpen_availability_edge"
        ].gt(0.10)
    )

    dataframe["opponent_bullpen_fresher"] = (
        dataframe[
            "bullpen_availability_edge"
        ].lt(-0.10)
    )

    return dataframe


def _build_feature_registry(
    enriched: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    numeric_domains = {
        "fatigue": [
            "fatigue",
            "workload",
            "pitches",
            "games_used",
            "consecutive",
            "rest_recovery",
            "availability",
            "days_since",
        ],
        "effectiveness": [
            "whiff",
            "strikeout",
            "walk",
            "hits",
            "runs",
            "effectiveness",
        ],
    }

    feature_columns = (
        [
            f"team_{column}"
            for column in BASE_BULLPEN_FEATURES
        ]
        + [
            f"opponent_{column}"
            for column in BASE_BULLPEN_FEATURES
        ]
        + MATCHUP_FEATURES
    )

    for feature in feature_columns:
        if feature not in enriched.columns:
            raise KeyError(
                f"Expected integrated feature missing: {feature}"
            )

        if feature.startswith("team_"):
            feature_scope = "team_bullpen"
        elif feature.startswith("opponent_"):
            feature_scope = "opponent_bullpen"
        else:
            feature_scope = "bullpen_matchup"

        if (
            feature.endswith("_label")
            or feature == "bullpen_state_matchup"
        ):
            feature_type = "categorical"
        elif pd.api.types.is_bool_dtype(
            enriched[feature]
        ):
            feature_type = "boolean"
        else:
            feature_type = "numeric"

        feature_domain = "bullpen_identity"

        lowered = feature.lower()

        for domain, keywords in (
            numeric_domains.items()
        ):
            if any(
                keyword in lowered
                for keyword in keywords
            ):
                feature_domain = (
                    f"bullpen_{domain}"
                )
                break

        records.append({
            "feature_name": feature,
            "feature_domain": feature_domain,
            "feature_scope": feature_scope,
            "feature_type": feature_type,
            "source_engine":
                "ATLAS Bullpen Availability and Fatigue Engine",
            "source_engine_version":
                ENGINE_VERSION,
            "source_table":
                str(BULLPEN_STATES_PATH),
            "strict_pregame_safe": True,
            "current_game_outcome_used": False,
            "same_date_games_used": False,
            "future_games_used": False,
            "discovery_eligible": True,
            "validation_required": True,
            "specific_reliever_identity_required": False,
            "availability_is_team_level_proxy": True,
            "automatic_prediction_weight": False,
        })

    registry = pd.DataFrame(records)

    if registry["feature_name"].duplicated().any():
        raise AssertionError(
            "Duplicate bullpen feature registry entries."
        )

    return registry


def run_bullpen_identity_integration_engine() -> dict[str, Any]:
    base = _normalize_keys(
        _load_parquet(
            BASE_INTERACTIONS_PATH,
            "base pregame interaction matrix",
        )
    )

    _require_columns(
        base,
        KEY_COLUMNS,
        "base interactions",
    )

    original_rows = len(base)
    original_columns = len(base.columns)

    original_duplicate_rows = int(
        base.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if original_duplicate_rows:
        raise AssertionError(
            f"Base interactions contain "
            f"{original_duplicate_rows} duplicate team-games."
        )

    bullpen_table = _prepare_bullpen_table(
        _load_parquet(
            BULLPEN_STATES_PATH,
            "pregame bullpen states",
        )
    )

    enriched = _attach_team_bullpen(
        base=base,
        bullpen_table=bullpen_table,
    )

    enriched = _attach_opponent_bullpen(
        base=enriched,
        bullpen_table=bullpen_table,
    )

    enriched = _create_matchup_features(
        enriched
    )

    enriched[
        "team_bullpen_identity_joined"
    ] = enriched[
        "team_bullpen_snapshot_available"
    ].fillna(False).astype(bool)

    enriched[
        "opponent_bullpen_identity_joined"
    ] = enriched[
        "opponent_bullpen_snapshot_available"
    ].fillna(False).astype(bool)

    enriched[
        "complete_bullpen_identity_join"
    ] = (
        enriched[
            "team_bullpen_identity_joined"
        ]
        & enriched[
            "opponent_bullpen_identity_joined"
        ]
    )

    enriched[
        "bullpen_features_strict_pregame_safe"
    ] = True

    enriched[
        "bullpen_current_game_outcome_used"
    ] = False

    enriched[
        "bullpen_same_date_games_used"
    ] = False

    enriched[
        "bullpen_future_games_used"
    ] = False

    enriched[
        "bullpen_identity_integration_version"
    ] = ENGINE_VERSION

    enriched[
        "bullpen_identity_integrated_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    enriched = enriched.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_rows = int(
        enriched.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if duplicate_rows:
        raise AssertionError(
            f"Enriched interactions contain "
            f"{duplicate_rows} duplicate team-games."
        )

    if len(enriched) != original_rows:
        raise AssertionError(
            "Bullpen integration changed the interaction row count."
        )

    if enriched[
        "bullpen_current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Current-game outcomes entered bullpen integration."
        )

    if enriched[
        "bullpen_same_date_games_used"
    ].any():
        raise AssertionError(
            "Same-date outcomes entered bullpen integration."
        )

    if enriched[
        "bullpen_future_games_used"
    ].any():
        raise AssertionError(
            "Future outcomes entered bullpen integration."
        )

    registry = _build_feature_registry(
        enriched
    )

    summary = (
        enriched.groupby(
            "atlas_season",
            sort=True,
        )
        .agg(
            team_game_rows=(
                "game_pk",
                "size",
            ),
            unique_games=(
                "game_pk",
                "nunique",
            ),
            teams=(
                "team",
                "nunique",
            ),
            team_bullpen_join_rate=(
                "team_bullpen_identity_joined",
                "mean",
            ),
            opponent_bullpen_join_rate=(
                "opponent_bullpen_identity_joined",
                "mean",
            ),
            complete_join_rate=(
                "complete_bullpen_identity_join",
                "mean",
            ),
            mean_team_fatigue=(
                "team_bullpen_fatigue_score",
                "mean",
            ),
            mean_opponent_fatigue=(
                "opponent_bullpen_fatigue_score",
                "mean",
            ),
            mean_availability_edge=(
                "bullpen_availability_edge",
                "mean",
            ),
        )
        .reset_index()
    )

    _atomic_parquet_write(
        enriched,
        ENRICHED_INTERACTIONS_PATH,
    )

    _atomic_parquet_write(
        registry,
        FEATURE_REGISTRY_PATH,
    )

    _atomic_parquet_write(
        summary,
        INTEGRATION_SUMMARY_PATH,
    )

    new_columns = (
        len(enriched.columns)
        - original_columns
    )

    result = {
        "engine":
            "ATLAS Bullpen Identity Integration Engine",
        "engine_version":
            ENGINE_VERSION,
        "base_rows":
            int(original_rows),
        "base_columns":
            int(original_columns),
        "enriched_rows":
            int(len(enriched)),
        "enriched_columns":
            int(len(enriched.columns)),
        "new_columns":
            int(new_columns),
        "registered_discovery_features":
            int(len(registry)),
        "unique_games":
            int(
                enriched["game_pk"].nunique()
            ),
        "teams":
            int(
                enriched["team"].nunique()
            ),
        "team_join_rate":
            float(
                enriched[
                    "team_bullpen_identity_joined"
                ].mean()
            ),
        "opponent_join_rate":
            float(
                enriched[
                    "opponent_bullpen_identity_joined"
                ].mean()
            ),
        "complete_join_rate":
            float(
                enriched[
                    "complete_bullpen_identity_join"
                ].mean()
            ),
        "duplicate_team_games":
            duplicate_rows,
        "original_source_modified":
            False,
        "outputs": {
            "enriched_interactions":
                str(ENRICHED_INTERACTIONS_PATH),
            "feature_registry":
                str(FEATURE_REGISTRY_PATH),
            "integration_summary":
                str(INTEGRATION_SUMMARY_PATH),
        },
        "pregame_safety": {
            "current_game_outcome_used":
                False,
            "same_date_games_used":
                False,
            "future_games_used":
                False,
            "specific_reliever_availability_claimed":
                False,
        },
        "policy": {
            "original_interaction_table_immutable":
                True,
            "bullpen_features_discovery_eligible":
                True,
            "prediction_weights_assigned":
                False,
            "validation_required":
                True,
            "2026_outcomes_used_to_build_features":
                False,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS BULLPEN IDENTITY INTEGRATION ENGINE")
    print("=" * 78)
    print(
        f"Base Rows.................... "
        f"{original_rows:,}"
    )
    print(
        f"Base Columns................. "
        f"{original_columns:,}"
    )
    print(
        f"Enriched Rows................ "
        f"{len(enriched):,}"
    )
    print(
        f"Enriched Columns............. "
        f"{len(enriched.columns):,}"
    )
    print(
        f"New Columns.................. "
        f"{new_columns:,}"
    )
    print(
        f"Registered Features.......... "
        f"{len(registry):,}"
    )
    print(
        f"Team Bullpen Join Rate....... "
        f"{result['team_join_rate']:.2%}"
    )
    print(
        f"Opponent Bullpen Join Rate... "
        f"{result['opponent_join_rate']:.2%}"
    )
    print(
        f"Complete Bullpen Join Rate... "
        f"{result['complete_join_rate']:.2%}"
    )
    print(
        f"Duplicate Team-Games......... "
        f"{duplicate_rows:,}"
    )
    print(
        "Original Source Modified..... False"
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
        f"Saved To..................... "
        f"{ENRICHED_INTERACTIONS_PATH}"
    )
    print("=" * 78)

    return result
