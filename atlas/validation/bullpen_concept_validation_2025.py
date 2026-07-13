
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
VALIDATION_SEASON = 2025

INTERACTION_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_bullpen_inputs.parquet"
)

TEAM_TARGET_PATH = (
    DATA_DIR
    / "backtest"
    / "targets"
    / "team_game_targets.parquet"
)

BULLPEN_CONCEPT_REGISTRY_PATH = (
    DATA_DIR
    / "learning"
    / "bullpen_concepts"
    / str(DISCOVERY_SEASON)
    / "bullpen_concept_registry.parquet"
)

BULLPEN_CONCEPT_MEMBER_MAP_PATH = (
    DATA_DIR
    / "learning"
    / "bullpen_concepts"
    / str(DISCOVERY_SEASON)
    / "bullpen_concept_member_map.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "validation"
    / "bullpen_concepts"
    / str(VALIDATION_SEASON)
)

TEAM_CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "team_checkpoints"
)

VALIDATION_REGISTRY_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_validation_registry.parquet"
)

VALIDATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_validation_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_validation_metadata.json"
)

FROZEN_ORIGINAL_VALIDATION_PATH = (
    DATA_DIR
    / "validation"
    / "concepts"
    / str(VALIDATION_SEASON)
    / "concept_validation_registry.parquet"
)


def _load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_parquet(path)


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


def _redirect_validation_globals(
    module: Any,
) -> dict[str, Any]:
    replacements = {
        "INTERACTION_PATH":
            INTERACTION_PATH,
        "TEAM_TARGET_PATH":
            TEAM_TARGET_PATH,
        "CONCEPT_REGISTRY_PATH":
            BULLPEN_CONCEPT_REGISTRY_PATH,
        "CONCEPT_MEMBER_MAP_PATH":
            BULLPEN_CONCEPT_MEMBER_MAP_PATH,
        "OUTPUT_DIR":
            OUTPUT_DIR,
        "TEAM_CHECKPOINT_DIR":
            TEAM_CHECKPOINT_DIR,
        "VALIDATION_REGISTRY_PATH":
            VALIDATION_REGISTRY_PATH,
        "VALIDATION_SUMMARY_PATH":
            VALIDATION_SUMMARY_PATH,
        "METADATA_PATH":
            METADATA_PATH,
    }

    original_values: dict[str, Any] = {}

    for name, replacement in replacements.items():
        if not hasattr(module, name):
            raise AttributeError(
                "Existing validation engine is missing "
                f"expected global: {name}"
            )

        original_values[name] = getattr(
            module,
            name,
        )

        setattr(
            module,
            name,
            replacement,
        )

    return original_values


def _restore_validation_globals(
    module: Any,
    original_values: dict[str, Any],
) -> None:
    for name, value in original_values.items():
        setattr(
            module,
            name,
            value,
        )


def run_bullpen_concept_validation_2025(
    only_team: str | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    interactions = _load_parquet(
        INTERACTION_PATH,
        "lineup-starter-bullpen interaction matrix",
    )

    targets = _load_parquet(
        TEAM_TARGET_PATH,
        "team-game targets",
    )

    concepts = _load_parquet(
        BULLPEN_CONCEPT_REGISTRY_PATH,
        "frozen 2024 bullpen concept registry",
    )

    member_map = _load_parquet(
        BULLPEN_CONCEPT_MEMBER_MAP_PATH,
        "frozen 2024 bullpen concept member map",
    )

    if not concepts[
        "learning_season"
    ].eq(DISCOVERY_SEASON).all():
        raise AssertionError(
            "Non-2024 concept entered bullpen validation."
        )

    if concepts[
        "concept_id"
    ].duplicated().any():
        raise AssertionError(
            "Duplicate bullpen concept IDs."
        )

    if member_map[
        "evidence_id"
    ].duplicated().any():
        raise AssertionError(
            "Duplicate bullpen member evidence IDs."
        )

    validation_interactions = interactions[
        interactions["atlas_season"].eq(
            VALIDATION_SEASON
        )
    ].copy()

    validation_targets = targets[
        targets["atlas_season"].eq(
            VALIDATION_SEASON
        )
    ].copy()

    if validation_interactions.empty:
        raise AssertionError(
            "No 2025 bullpen interaction rows found."
        )

    if validation_targets.empty:
        raise AssertionError(
            "No 2025 team targets found."
        )

    if validation_interactions[
        "game_date"
    ].isna().any():
        raise AssertionError(
            "Missing 2025 interaction dates."
        )

    frozen_concept_hash_before = _sha256(
        BULLPEN_CONCEPT_REGISTRY_PATH
    )

    frozen_member_hash_before = _sha256(
        BULLPEN_CONCEPT_MEMBER_MAP_PATH
    )

    original_validation_hash_before = _sha256(
        FROZEN_ORIGINAL_VALIDATION_PATH
    )

    validation_module = importlib.import_module(
        "atlas.validation.concept_validation_2025"
    )

    original_globals = (
        _redirect_validation_globals(
            validation_module
        )
    )

    try:
        base_result = (
            validation_module
            .run_concept_validation_2025(
                only_team=only_team,
                limit=limit,
                resume=resume,
            )
        )

    finally:
        _restore_validation_globals(
            validation_module,
            original_globals,
        )

    registry = _load_parquet(
        VALIDATION_REGISTRY_PATH,
        "bullpen concept validation registry",
    )

    summary = _load_parquet(
        VALIDATION_SUMMARY_PATH,
        "bullpen concept validation summary",
    )

    frozen_concept_hash_after = _sha256(
        BULLPEN_CONCEPT_REGISTRY_PATH
    )

    frozen_member_hash_after = _sha256(
        BULLPEN_CONCEPT_MEMBER_MAP_PATH
    )

    original_validation_hash_after = _sha256(
        FROZEN_ORIGINAL_VALIDATION_PATH
    )

    frozen_bullpen_concepts_unchanged = (
        frozen_concept_hash_before
        == frozen_concept_hash_after
        and frozen_member_hash_before
        == frozen_member_hash_after
    )

    original_validation_unchanged = (
        original_validation_hash_before
        == original_validation_hash_after
    )

    if not frozen_bullpen_concepts_unchanged:
        raise AssertionError(
            "Frozen 2024 bullpen concept artifacts changed."
        )

    if not original_validation_unchanged:
        raise AssertionError(
            "Original 2025 validation registry changed."
        )

    if registry[
        "concept_id"
    ].duplicated().any():
        raise AssertionError(
            "Duplicate bullpen validation concept IDs."
        )

    if not registry[
        "discovery_season"
    ].eq(DISCOVERY_SEASON).all():
        raise AssertionError(
            "Incorrect bullpen discovery season."
        )

    if not registry[
        "validation_season"
    ].eq(VALIDATION_SEASON).all():
        raise AssertionError(
            "Incorrect bullpen validation season."
        )

    result = {
        "engine":
            "ATLAS 2025 Blind Bullpen Concept Validation Engine",
        "engine_version":
            ENGINE_VERSION,
        "discovery_season":
            DISCOVERY_SEASON,
        "validation_season":
            VALIDATION_SEASON,
        "2025_interaction_rows":
            int(len(validation_interactions)),
        "2025_target_rows":
            int(len(validation_targets)),
        "frozen_concepts":
            int(len(concepts)),
        "concepts_validated":
            int(len(registry)),
        "validation_summary_rows":
            int(len(summary)),
        "teams_validated":
            int(registry["team"].nunique()),
        "targets_validated":
            int(registry["target"].nunique()),
        "duplicate_validation_rows":
            int(
                registry[
                    "concept_id"
                ].duplicated().sum()
            ),
        "frozen_bullpen_concepts_unchanged":
            frozen_bullpen_concepts_unchanged,
        "original_validation_unchanged":
            original_validation_unchanged,
        "prediction_weights_assigned":
            False,
        "2026_outcomes_used":
            False,
        "base_engine_result":
            base_result,
        "outputs": {
            "validation_registry":
                str(VALIDATION_REGISTRY_PATH),
            "validation_summary":
                str(VALIDATION_SUMMARY_PATH),
            "team_checkpoints":
                str(TEAM_CHECKPOINT_DIR),
        },
        "policy": {
            "2024_concepts_frozen":
                True,
            "2025_blind_validation_only":
                True,
            "new_concepts_discovered":
                False,
            "prediction_weights_assigned":
                False,
            "2026_outcomes_used":
                False,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print(
        "ATLAS 2025 BLIND BULLPEN CONCEPT VALIDATION"
    )
    print("=" * 78)
    print(
        f"Frozen 2024 Concepts....... "
        f"{len(concepts):,}"
    )
    print(
        f"2025 Interaction Rows...... "
        f"{len(validation_interactions):,}"
    )
    print(
        f"Concepts Validated......... "
        f"{len(registry):,}"
    )
    print(
        f"Teams Validated............ "
        f"{registry['team'].nunique():,}"
    )
    print(
        f"Targets Validated.......... "
        f"{registry['target'].nunique():,}"
    )
    print(
        "New Concepts Discovered.... False"
    )
    print(
        "Prediction Weights......... False"
    )
    print(
        "2026 Outcomes Used......... False"
    )
    print(
        "Frozen Concepts Changed.... False"
    )
    print(
        f"Saved To................... "
        f"{OUTPUT_DIR}"
    )
    print("=" * 78)

    return result
