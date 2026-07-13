
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
DISCOVERY_SEASON = 2024

ENRICHED_INTERACTIONS_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_bullpen_inputs.parquet"
)

BULLPEN_FEATURE_REGISTRY_PATH = (
    DATA_DIR
    / "pregame"
    / "feature_registry"
    / "bullpen_identity_feature_registry.parquet"
)

FROZEN_ORIGINAL_EVIDENCE_PATH = (
    DATA_DIR
    / "learning"
    / "team_evidence"
    / str(DISCOVERY_SEASON)
    / "team_evidence_registry.parquet"
)

OUTPUT_ROOT = (
    DATA_DIR
    / "learning"
    / "bullpen_evidence"
    / str(DISCOVERY_SEASON)
)

DISCOVERY_INPUT_PATH = (
    OUTPUT_ROOT
    / "bullpen_discovery_inputs.parquet"
)

TEAM_CHECKPOINT_DIR = (
    OUTPUT_ROOT
    / "team_checkpoints"
)

MASTER_REGISTRY_PATH = (
    OUTPUT_ROOT
    / "bullpen_team_evidence_registry.parquet"
)

MASTER_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "bullpen_team_evidence_summary.parquet"
)

RUN_METADATA_PATH = (
    OUTPUT_ROOT
    / "bullpen_evidence_runner_metadata.json"
)

WRAPPER_METADATA_PATH = (
    OUTPUT_ROOT
    / "bullpen_discovery_wrapper_metadata.json"
)


KEY_COLUMNS = [
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


def _sha256(
    path: Path,
) -> str | None:
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _build_bullpen_only_input() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    enriched = _load_parquet(
        ENRICHED_INTERACTIONS_PATH,
        "bullpen-enriched interaction matrix",
    )

    feature_registry = _load_parquet(
        BULLPEN_FEATURE_REGISTRY_PATH,
        "bullpen feature registry",
    )

    required_registry_columns = [
        "feature_name",
        "strict_pregame_safe",
        "current_game_outcome_used",
        "same_date_games_used",
        "future_games_used",
        "discovery_eligible",
        "validation_required",
        "automatic_prediction_weight",
    ]

    missing_registry_columns = [
        column
        for column in required_registry_columns
        if column not in feature_registry.columns
    ]

    if missing_registry_columns:
        raise KeyError(
            "Bullpen registry missing columns: "
            f"{missing_registry_columns}"
        )

    if not feature_registry[
        "strict_pregame_safe"
    ].all():
        raise AssertionError(
            "Registry contains non-pregame-safe features."
        )

    if feature_registry[
        "current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Current-game outcome leakage in registry."
        )

    if feature_registry[
        "same_date_games_used"
    ].any():
        raise AssertionError(
            "Same-date leakage in registry."
        )

    if feature_registry[
        "future_games_used"
    ].any():
        raise AssertionError(
            "Future-game leakage in registry."
        )

    if not feature_registry[
        "discovery_eligible"
    ].all():
        raise AssertionError(
            "Registry contains discovery-ineligible features."
        )

    if not feature_registry[
        "validation_required"
    ].all():
        raise AssertionError(
            "Registry contains features not requiring validation."
        )

    if feature_registry[
        "automatic_prediction_weight"
    ].any():
        raise AssertionError(
            "Bullpen features already have automatic weights."
        )

    feature_names = (
        feature_registry[
            "feature_name"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in enriched.columns
    ]

    if missing_features:
        raise KeyError(
            "Registered bullpen features missing from "
            f"enriched inputs: {missing_features[:20]}"
        )

    missing_keys = [
        column
        for column in KEY_COLUMNS
        if column not in enriched.columns
    ]

    if missing_keys:
        raise KeyError(
            f"Enriched inputs missing keys: {missing_keys}"
        )

    bullpen_input = enriched[
        KEY_COLUMNS + feature_names
    ].copy()

    bullpen_input["game_date"] = pd.to_datetime(
        bullpen_input["game_date"],
        errors="raise",
    ).dt.normalize()

    bullpen_input = bullpen_input[
        bullpen_input["atlas_season"].eq(
            DISCOVERY_SEASON
        )
    ].copy()

    bullpen_input = bullpen_input.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_rows = int(
        bullpen_input.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    if duplicate_rows:
        raise AssertionError(
            f"Duplicate bullpen discovery rows: "
            f"{duplicate_rows}"
        )

    unexpected_columns = sorted(
        set(bullpen_input.columns)
        - set(KEY_COLUMNS)
        - set(feature_names)
    )

    if unexpected_columns:
        raise AssertionError(
            "Unexpected columns entered bullpen discovery: "
            f"{unexpected_columns}"
        )

    return bullpen_input, feature_registry


def _redirect_runner_paths(
    runner_module: Any,
) -> dict[str, Any]:
    original_values = {}

    replacements = {
        "INTERACTION_INPUT_PATH":
            DISCOVERY_INPUT_PATH,
        "OUTPUT_ROOT":
            OUTPUT_ROOT,
        "TEAM_CHECKPOINT_DIR":
            TEAM_CHECKPOINT_DIR,
        "MASTER_REGISTRY_PATH":
            MASTER_REGISTRY_PATH,
        "MASTER_SUMMARY_PATH":
            MASTER_SUMMARY_PATH,
        "RUN_METADATA_PATH":
            RUN_METADATA_PATH,
    }

    for attribute, replacement in replacements.items():
        if not hasattr(runner_module, attribute):
            raise AttributeError(
                "team_evidence_runner_v2 missing expected "
                f"global: {attribute}"
            )

        original_values[attribute] = getattr(
            runner_module,
            attribute,
        )

        setattr(
            runner_module,
            attribute,
            replacement,
        )

    return original_values


def _restore_runner_paths(
    runner_module: Any,
    original_values: dict[str, Any],
) -> None:
    for attribute, original_value in (
        original_values.items()
    ):
        setattr(
            runner_module,
            attribute,
            original_value,
        )


def run_bullpen_evidence_discovery_2024(
    limit: int | None = None,
    resume: bool = True,
    only_team: str | None = None,
) -> dict[str, Any]:
    original_hash_before = _sha256(
        FROZEN_ORIGINAL_EVIDENCE_PATH
    )

    bullpen_input, feature_registry = (
        _build_bullpen_only_input()
    )

    _atomic_parquet_write(
        bullpen_input,
        DISCOVERY_INPUT_PATH,
    )

    runner_module = importlib.import_module(
        "atlas.learning.team_evidence_runner_v2"
    )

    original_values = _redirect_runner_paths(
        runner_module
    )

    try:
        runner_result = (
            runner_module
            .run_team_evidence_discovery_v2(
                limit=limit,
                resume=resume,
                only_team=only_team,
            )
        )

    finally:
        _restore_runner_paths(
            runner_module,
            original_values,
        )

    original_hash_after = _sha256(
        FROZEN_ORIGINAL_EVIDENCE_PATH
    )

    original_unchanged = (
        original_hash_before
        == original_hash_after
    )

    if not original_unchanged:
        raise AssertionError(
            "Frozen original 2024 evidence registry changed."
        )

    result = {
        "engine":
            "ATLAS 2024 Bullpen Evidence Discovery Engine",
        "engine_version":
            ENGINE_VERSION,
        "discovery_season":
            DISCOVERY_SEASON,
        "bullpen_input_rows":
            int(len(bullpen_input)),
        "bullpen_input_columns":
            int(len(bullpen_input.columns)),
        "registered_bullpen_features":
            int(len(feature_registry)),
        "target_teams": (
            runner_result.get("target_teams")
            if isinstance(runner_result, dict)
            else None
        ),
        "runner_result":
            runner_result,
        "frozen_original_registry_unchanged":
            original_unchanged,
        "prediction_weights_assigned":
            False,
        "2025_used":
            False,
        "2026_used":
            False,
        "outputs": {
            "discovery_input":
                str(DISCOVERY_INPUT_PATH),
            "evidence_registry":
                str(MASTER_REGISTRY_PATH),
            "evidence_summary":
                str(MASTER_SUMMARY_PATH),
            "checkpoint_directory":
                str(TEAM_CHECKPOINT_DIR),
        },
        "policy": {
            "bullpen_only_features":
                True,
            "original_2024_evidence_frozen":
                True,
            "discovery_uses_2024_only":
                True,
            "blind_2025_validation_required":
                True,
            "prediction_weights_assigned":
                False,
        },
    }

    _atomic_json_write(
        result,
        WRAPPER_METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS 2024 BULLPEN EVIDENCE DISCOVERY")
    print("=" * 78)
    print(
        f"Discovery Input Rows....... "
        f"{len(bullpen_input):,}"
    )
    print(
        f"Discovery Input Columns.... "
        f"{len(bullpen_input.columns):,}"
    )
    print(
        f"Registered Features........ "
        f"{len(feature_registry):,}"
    )
    print(
        "Discovery Season........... 2024"
    )
    print(
        "2025 Used.................. False"
    )
    print(
        "2026 Used.................. False"
    )
    print(
        "Prediction Weights......... False"
    )
    print(
        "Original Registry Changed.. False"
    )
    print(
        f"Saved To................... "
        f"{OUTPUT_ROOT}"
    )
    print("=" * 78)

    return result
