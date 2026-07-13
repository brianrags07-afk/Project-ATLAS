
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

RAW_EVIDENCE_PATH = (
    DATA_DIR
    / "learning"
    / "bullpen_evidence"
    / str(DISCOVERY_SEASON)
    / "bullpen_team_evidence_registry.parquet"
)

RAW_SUMMARY_PATH = (
    DATA_DIR
    / "learning"
    / "bullpen_evidence"
    / str(DISCOVERY_SEASON)
    / "bullpen_team_evidence_summary.parquet"
)

FEATURE_REGISTRY_PATH = (
    DATA_DIR
    / "pregame"
    / "feature_registry"
    / "bullpen_identity_feature_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "bullpen_concepts"
    / str(DISCOVERY_SEASON)
)

CONCEPT_REGISTRY_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_registry.parquet"
)

CONCEPT_MEMBER_MAP_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_member_map.parquet"
)

CONCEPT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "bullpen_concept_consolidation_metadata.json"
)

FROZEN_ORIGINAL_CONCEPT_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(DISCOVERY_SEASON)
    / "team_concept_registry.parquet"
)

FROZEN_ORIGINAL_MEMBER_MAP_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(DISCOVERY_SEASON)
    / "team_concept_member_map.parquet"
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


def _load_feature_metadata() -> pd.DataFrame:
    registry = _load_parquet(
        FEATURE_REGISTRY_PATH,
        "bullpen feature registry",
    )

    required_columns = [
        "feature_name",
        "feature_domain",
        "feature_scope",
        "feature_type",
        "strict_pregame_safe",
        "current_game_outcome_used",
        "same_date_games_used",
        "future_games_used",
        "validation_required",
        "automatic_prediction_weight",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in registry.columns
    ]

    if missing_columns:
        raise KeyError(
            "Bullpen feature registry missing columns: "
            f"{missing_columns}"
        )

    if not registry[
        "strict_pregame_safe"
    ].fillna(False).astype(bool).all():
        raise AssertionError(
            "Non-pregame-safe bullpen feature detected."
        )

    for leakage_column in [
        "current_game_outcome_used",
        "same_date_games_used",
        "future_games_used",
    ]:
        if registry[
            leakage_column
        ].fillna(False).astype(bool).any():
            raise AssertionError(
                f"Leakage flag detected: {leakage_column}"
            )

    if not registry[
        "validation_required"
    ].fillna(False).astype(bool).all():
        raise AssertionError(
            "All bullpen features must require validation."
        )

    if registry[
        "automatic_prediction_weight"
    ].fillna(False).astype(bool).any():
        raise AssertionError(
            "Bullpen feature already has automatic weight."
        )

    return (
        registry
        .drop_duplicates(
            subset=["feature_name"]
        )
        .reset_index(drop=True)
    )


def _classify_bullpen_feature(
    feature: str,
    metadata_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    feature = str(feature)
    lowered = feature.lower()

    metadata = metadata_lookup.get(
        feature,
        {},
    )

    registered_scope = str(
        metadata.get(
            "feature_scope",
            "bullpen_matchup",
        )
    )

    if feature.startswith("team_"):
        concept_scope = "team_bullpen"
    elif feature.startswith("opponent_"):
        concept_scope = "opponent_bullpen"
    else:
        concept_scope = "bullpen_matchup"

    if any(
        token in lowered
        for token in [
            "fatigue",
            "workload",
            "pitches_prior",
            "games_used",
            "consecutive_prior_usage",
        ]
    ):
        concept_name = (
            "bullpen_workload_and_fatigue"
        )
        concept_domain = "bullpen_fatigue"

    elif any(
        token in lowered
        for token in [
            "availability",
            "rest_recovery",
            "days_since_prior",
            "fresher",
        ]
    ):
        concept_name = (
            "bullpen_availability_and_rest"
        )
        concept_domain = (
            "bullpen_availability"
        )

    elif any(
        token in lowered
        for token in [
            "whiff",
            "strikeout",
        ]
    ):
        concept_name = (
            "bullpen_bat_missing_ability"
        )
        concept_domain = (
            "bullpen_effectiveness"
        )

    elif "walk" in lowered:
        concept_name = "bullpen_command"
        concept_domain = (
            "bullpen_effectiveness"
        )

    elif any(
        token in lowered
        for token in [
            "hits_allowed",
            "hits_per_pitch",
        ]
    ):
        concept_name = (
            "bullpen_contact_allowed"
        )
        concept_domain = (
            "bullpen_effectiveness"
        )

    elif any(
        token in lowered
        for token in [
            "runs_allowed",
            "runs_per_pitch",
            "effectiveness",
        ]
    ):
        concept_name = (
            "bullpen_run_prevention"
        )
        concept_domain = (
            "bullpen_effectiveness"
        )

    elif "edge" in lowered:
        concept_name = "bullpen_matchup_edge"
        concept_domain = "bullpen_matchup"

    else:
        concept_name = "bullpen_identity"
        concept_domain = str(
            metadata.get(
                "feature_domain",
                "bullpen_identity",
            )
        )

    return {
        "concept_scope": concept_scope,
        "batting_order_slot": None,
        "normalized_metric": feature,
        "concept_name": concept_name,
        "concept_domain": concept_domain,
        "registered_feature_scope":
            registered_scope,
    }


def _redirect_globals(
    module: Any,
) -> dict[str, Any]:
    replacements = {
        "RAW_EVIDENCE_PATH":
            RAW_EVIDENCE_PATH,
        "RAW_TEAM_SUMMARY_PATH":
            RAW_SUMMARY_PATH,
        "OUTPUT_DIR":
            OUTPUT_DIR,
        "CONCEPT_REGISTRY_PATH":
            CONCEPT_REGISTRY_PATH,
        "CONCEPT_MEMBER_MAP_PATH":
            CONCEPT_MEMBER_MAP_PATH,
        "CONCEPT_SUMMARY_PATH":
            CONCEPT_SUMMARY_PATH,
        "METADATA_PATH":
            METADATA_PATH,
    }

    original_values: dict[str, Any] = {}

    for name, value in replacements.items():
        if not hasattr(module, name):
            raise AttributeError(
                "Existing consolidation engine missing "
                f"expected global: {name}"
            )

        original_values[name] = getattr(
            module,
            name,
        )

        setattr(
            module,
            name,
            value,
        )

    return original_values


def _restore_globals(
    module: Any,
    original_values: dict[str, Any],
) -> None:
    for name, value in original_values.items():
        setattr(
            module,
            name,
            value,
        )


def run_bullpen_concept_consolidation_2024(
) -> dict[str, Any]:
    original_concept_hash_before = _sha256(
        FROZEN_ORIGINAL_CONCEPT_PATH
    )

    original_member_hash_before = _sha256(
        FROZEN_ORIGINAL_MEMBER_MAP_PATH
    )

    evidence = _load_parquet(
        RAW_EVIDENCE_PATH,
        "bullpen evidence registry",
    )

    feature_registry = (
        _load_feature_metadata()
    )

    if not evidence[
        "learning_season"
    ].eq(DISCOVERY_SEASON).all():
        raise AssertionError(
            "Non-2024 evidence entered consolidation."
        )

    if evidence[
        "evidence_id"
    ].duplicated().any():
        raise AssertionError(
            "Duplicate bullpen evidence IDs."
        )

    metadata_lookup = (
        feature_registry
        .set_index("feature_name")
        .to_dict(orient="index")
    )

    registered_features = set(
        feature_registry[
            "feature_name"
        ]
        .dropna()
        .astype(str)
    )

    evidence_features = set(
        evidence[
            "feature"
        ]
        .dropna()
        .astype(str)
    )

    unexpected_features = sorted(
        evidence_features
        - registered_features
    )

    if unexpected_features:
        raise AssertionError(
            "Unregistered bullpen evidence features: "
            f"{unexpected_features[:20]}"
        )

    consolidation_module = (
        importlib.import_module(
            "atlas.learning."
            "evidence_consolidation_engine"
        )
    )

    if not hasattr(
        consolidation_module,
        "_classify_feature",
    ):
        raise AttributeError(
            "Existing consolidation engine does not "
            "expose _classify_feature."
        )

    original_classifier = (
        consolidation_module
        ._classify_feature
    )

    original_globals = _redirect_globals(
        consolidation_module
    )

    def bullpen_classifier(
        feature: str,
    ) -> dict[str, Any]:
        return _classify_bullpen_feature(
            feature,
            metadata_lookup,
        )

    consolidation_module._classify_feature = (
        bullpen_classifier
    )

    try:
        base_result = (
            consolidation_module
            .run_evidence_consolidation()
        )

    finally:
        consolidation_module._classify_feature = (
            original_classifier
        )

        _restore_globals(
            consolidation_module,
            original_globals,
        )

    concepts = _load_parquet(
        CONCEPT_REGISTRY_PATH,
        "bullpen concept registry",
    )

    member_map = _load_parquet(
        CONCEPT_MEMBER_MAP_PATH,
        "bullpen concept member map",
    )

    summary = _load_parquet(
        CONCEPT_SUMMARY_PATH,
        "bullpen concept summary",
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
            "Bullpen evidence assigned to multiple concepts."
        )

    if not member_map[
        "evidence_id"
    ].astype(str).isin(
        evidence[
            "evidence_id"
        ].astype(str)
    ).all():
        raise AssertionError(
            "Member map contains unknown evidence IDs."
        )

    original_concept_hash_after = _sha256(
        FROZEN_ORIGINAL_CONCEPT_PATH
    )

    original_member_hash_after = _sha256(
        FROZEN_ORIGINAL_MEMBER_MAP_PATH
    )

    original_artifacts_unchanged = (
        original_concept_hash_before
        == original_concept_hash_after
        and original_member_hash_before
        == original_member_hash_after
    )

    if not original_artifacts_unchanged:
        raise AssertionError(
            "Frozen original 2024 concepts changed."
        )

    result = {
        "engine":
            "ATLAS 2024 Bullpen Concept Consolidation Engine",
        "engine_version":
            ENGINE_VERSION,
        "discovery_season":
            DISCOVERY_SEASON,
        "raw_evidence_rows":
            int(len(evidence)),
        "concept_rows":
            int(len(concepts)),
        "member_map_rows":
            int(len(member_map)),
        "summary_rows":
            int(len(summary)),
        "teams":
            int(concepts["team"].nunique()),
        "targets":
            int(concepts["target"].nunique()),
        "concept_domains":
            int(
                concepts[
                    "concept_domain"
                ].nunique()
            ),
        "duplicate_concept_ids":
            int(
                concepts[
                    "concept_id"
                ].duplicated().sum()
            ),
        "duplicate_member_evidence":
            int(
                member_map[
                    "evidence_id"
                ].duplicated().sum()
            ),
        "original_2024_concepts_unchanged":
            original_artifacts_unchanged,
        "prediction_weights_assigned":
            False,
        "validated_out_of_sample":
            False,
        "2025_used":
            False,
        "2026_used":
            False,
        "base_engine_result":
            base_result,
        "outputs": {
            "concept_registry":
                str(CONCEPT_REGISTRY_PATH),
            "concept_member_map":
                str(CONCEPT_MEMBER_MAP_PATH),
            "concept_summary":
                str(CONCEPT_SUMMARY_PATH),
        },
        "policy": {
            "2024_discovery_only":
                True,
            "raw_evidence_preserved":
                True,
            "original_concepts_frozen":
                True,
            "blind_2025_validation_required":
                True,
            "prediction_weights_assigned":
                False,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print(
        "ATLAS 2024 BULLPEN CONCEPT CONSOLIDATION"
    )
    print("=" * 78)
    print(
        f"Raw Evidence Rows.......... "
        f"{len(evidence):,}"
    )
    print(
        f"Concept Rows............... "
        f"{len(concepts):,}"
    )
    print(
        f"Member Map Rows............ "
        f"{len(member_map):,}"
    )
    print(
        f"Teams...................... "
        f"{concepts['team'].nunique():,}"
    )
    print(
        f"Targets.................... "
        f"{concepts['target'].nunique():,}"
    )
    print(
        f"Concept Domains............ "
        f"{concepts['concept_domain'].nunique():,}"
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
        "Original Concepts Changed.. False"
    )
    print(
        f"Saved To................... "
        f"{OUTPUT_DIR}"
    )
    print("=" * 78)

    return result
