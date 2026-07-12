
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR
from atlas.learning.evidence_consolidation_engine import (
    _classify_feature,
)


ENGINE_VERSION = "1.0.0"
DISCOVERY_SEASON = 2024
VALIDATION_SEASON = 2025

BELIEF_REGISTRY_PATH = (
    DATA_DIR
    / "learning"
    / "concept_beliefs"
    / "concept_belief_registry.parquet"
)

LEAGUE_TEAM_PATH = (
    DATA_DIR
    / "learning"
    / "league_evidence"
    / str(DISCOVERY_SEASON)
    / "league_team_evidence_registry.parquet"
)

LEAGUE_GAME_PATH = (
    DATA_DIR
    / "learning"
    / "league_evidence"
    / str(DISCOVERY_SEASON)
    / "league_game_environment_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "integrated_beliefs"
)

INTEGRATED_REGISTRY_PATH = (
    OUTPUT_DIR
    / "integrated_concept_belief_registry.parquet"
)

LEAGUE_CONCEPT_PRIOR_PATH = (
    OUTPUT_DIR
    / "league_concept_prior_registry.parquet"
)

INTEGRATED_SUMMARY_PATH = (
    OUTPUT_DIR
    / "integrated_belief_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "league_prior_integration_metadata.json"
)


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


def _load_parquet(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    return pd.read_parquet(path)


def _normalize_league_feature(
    feature: str,
) -> tuple[str, str | None]:
    feature = str(feature)

    side = None

    if feature.startswith("home_"):
        side = "home"
        feature = feature[5:]

    elif feature.startswith("away_"):
        side = "away"
        feature = feature[5:]

    return feature, side


def _effect_direction(
    lift: float,
) -> str:
    if pd.isna(lift):
        return "unknown"

    if lift > 0:
        return "supports_target"

    if lift < 0:
        return "suppresses_target"

    return "neutral"


def _league_status_strength(
    status: str,
) -> float:
    mapping = {
        "strong_candidate": 1.00,
        "candidate": 0.72,
        "weak_candidate": 0.42,
    }

    return mapping.get(
        str(status),
        0.25,
    )


def _prepare_league_evidence(
    team_evidence: pd.DataFrame,
    game_evidence: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    if not team_evidence.empty:
        team = team_evidence.copy()
        team["league_scope"] = "team_game_baseline"
        frames.append(team)

    if not game_evidence.empty:
        game = game_evidence.copy()
        game["league_scope"] = "full_game_environment"
        frames.append(game)

    if not frames:
        raise ValueError(
            "No league evidence was available."
        )

    league = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    normalized_features = []
    source_sides = []
    classifications = []

    for feature in league["feature"].astype(str):
        normalized, side = _normalize_league_feature(
            feature
        )

        normalized_features.append(normalized)
        source_sides.append(side)
        classifications.append(
            _classify_feature(normalized)
        )

    classification_frame = pd.DataFrame(
        classifications
    )

    league = pd.concat(
        [
            league.reset_index(drop=True),
            classification_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    league["normalized_league_feature"] = (
        normalized_features
    )

    league["league_source_side"] = (
        source_sides
    )

    league["effect_direction"] = (
        pd.to_numeric(
            league["lift"],
            errors="coerce",
        ).apply(_effect_direction)
    )

    league["status_strength"] = (
        league["lifecycle_status"]
        .apply(_league_status_strength)
    )

    league["league_weight"] = (
        pd.to_numeric(
            league["condition_sample_size"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=1.0)
        *
        pd.to_numeric(
            league["confidence_score"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.05)
        *
        league["status_strength"]
    )

    return league


def build_league_concept_priors(
    league: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "target",
        "concept_domain",
        "concept_scope",
        "batting_order_slot",
        "concept_name",
        "effect_direction",
    ]

    records: list[dict[str, Any]] = []

    for keys, group in league.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            target,
            concept_domain,
            concept_scope,
            batting_order_slot,
            concept_name,
            effect_direction,
        ) = keys

        weights = pd.to_numeric(
            group["league_weight"],
            errors="coerce",
        ).fillna(1.0)

        confidence = pd.to_numeric(
            group["confidence_score"],
            errors="coerce",
        ).fillna(0.0)

        lift = pd.to_numeric(
            group["lift"],
            errors="coerce",
        ).fillna(0.0)

        absolute_lift = pd.to_numeric(
            group["absolute_lift"],
            errors="coerce",
        ).fillna(0.0)

        sample = pd.to_numeric(
            group["condition_sample_size"],
            errors="coerce",
        ).fillna(0.0)

        weighted_confidence = float(
            np.average(
                confidence,
                weights=weights,
            )
        )

        weighted_lift = float(
            np.average(
                lift,
                weights=weights,
            )
        )

        weighted_absolute_lift = float(
            np.average(
                absolute_lift,
                weights=weights,
            )
        )

        sample_component = min(
            1.0,
            float(sample.sum()) / 2500.0,
        )

        effect_component = min(
            1.0,
            weighted_absolute_lift / 0.10,
        )

        league_prior_strength = float(
            np.clip(
                0.50 * weighted_confidence
                + 0.30 * effect_component
                + 0.20 * sample_component,
                0.0,
                1.0,
            )
        )

        records.append({
            "target":
                str(target),
            "concept_domain":
                str(concept_domain),
            "concept_scope":
                str(concept_scope),
            "batting_order_slot": (
                None
                if pd.isna(batting_order_slot)
                else str(batting_order_slot)
            ),
            "concept_name":
                str(concept_name),
            "effect_direction":
                str(effect_direction),
            "league_evidence_objects":
                int(len(group)),
            "league_unique_features":
                int(
                    group["feature"].nunique()
                ),
            "league_strong_candidates":
                int(
                    group["lifecycle_status"]
                    .eq("strong_candidate")
                    .sum()
                ),
            "league_candidates":
                int(
                    group["lifecycle_status"]
                    .eq("candidate")
                    .sum()
                ),
            "league_weak_candidates":
                int(
                    group["lifecycle_status"]
                    .eq("weak_candidate")
                    .sum()
                ),
            "league_total_condition_sample":
                int(sample.sum()),
            "league_weighted_confidence":
                weighted_confidence,
            "league_weighted_lift":
                weighted_lift,
            "league_weighted_absolute_lift":
                weighted_absolute_lift,
            "league_prior_strength":
                league_prior_strength,
            "league_scopes": "|".join(
                sorted(
                    group["league_scope"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            ),
            "league_prior_engine_version":
                ENGINE_VERSION,
        })

    return pd.DataFrame(records)


def _signature_columns() -> list[str]:
    return [
        "target",
        "concept_domain",
        "concept_scope",
        "batting_order_slot",
        "concept_name",
    ]


def _standardize_slot(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["batting_order_slot"] = (
        dataframe["batting_order_slot"]
        .astype("string")
        .replace("<NA>", pd.NA)
    )

    return dataframe


def integrate_team_and_league(
    belief: pd.DataFrame,
    league_priors: pd.DataFrame,
) -> pd.DataFrame:
    belief = _standardize_slot(belief)
    league_priors = _standardize_slot(
        league_priors
    )

    signature = _signature_columns()

    supporting_prior = league_priors[
        league_priors["effect_direction"].eq(
            "supports_target"
        )
    ].copy()

    suppressing_prior = league_priors[
        league_priors["effect_direction"].eq(
            "suppresses_target"
        )
    ].copy()

    supporting_prior = supporting_prior.rename(
        columns={
            column: f"league_support_{column}"
            for column in supporting_prior.columns
            if column not in signature
        }
    )

    suppressing_prior = suppressing_prior.rename(
        columns={
            column: f"league_suppress_{column}"
            for column in suppressing_prior.columns
            if column not in signature
        }
    )

    integrated = belief.merge(
        supporting_prior,
        on=signature,
        how="left",
        validate="many_to_one",
    )

    integrated = integrated.merge(
        suppressing_prior,
        on=signature,
        how="left",
        validate="many_to_one",
    )

    support_strength = pd.to_numeric(
        integrated.get(
            "league_support_league_prior_strength",
            0.0,
        ),
        errors="coerce",
    ).fillna(0.0)

    suppress_strength = pd.to_numeric(
        integrated.get(
            "league_suppress_league_prior_strength",
            0.0,
        ),
        errors="coerce",
    ).fillna(0.0)

    team_supports = integrated[
        "effect_direction"
    ].eq("supports_target")

    integrated["league_agreement_strength"] = np.where(
        team_supports,
        support_strength,
        suppress_strength,
    )

    integrated["league_conflict_strength"] = np.where(
        team_supports,
        suppress_strength,
        support_strength,
    )

    agreement = integrated[
        "league_agreement_strength"
    ]

    conflict = integrated[
        "league_conflict_strength"
    ]

    has_agreement = agreement.gt(0.0)
    has_conflict = conflict.gt(0.0)

    integrated["league_relationship"] = np.select(
        [
            has_agreement & ~has_conflict,
            ~has_agreement & has_conflict,
            has_agreement & has_conflict
            & agreement.ge(conflict),
            has_agreement & has_conflict
            & conflict.gt(agreement),
        ],
        [
            "reinforced_by_league",
            "conflicts_with_league",
            "mixed_league_evidence_agreement_leads",
            "mixed_league_evidence_conflict_leads",
        ],
        default="team_specific",
    )

    net_prior = (
        agreement - conflict
    ).clip(
        lower=-1.0,
        upper=1.0,
    )

    integrated[
        "league_net_prior_adjustment"
    ] = net_prior

    # League priors can move belief modestly but may not
    # erase strong team-specific knowledge.
    belief_adjustment = np.where(
        net_prior >= 0,
        0.08 * net_prior,
        0.10 * net_prior,
    )

    integrated[
        "league_adjusted_belief_score"
    ] = (
        pd.to_numeric(
            integrated["belief_score"],
            errors="coerce",
        ).fillna(0.0)
        + belief_adjustment
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    relationship_multiplier = (
        integrated["league_relationship"]
        .map({
            "reinforced_by_league": 1.10,
            "team_specific": 1.00,
            "mixed_league_evidence_agreement_leads": 1.03,
            "mixed_league_evidence_conflict_leads": 0.82,
            "conflicts_with_league": 0.72,
        })
        .fillna(1.00)
    )

    integrated[
        "league_adjusted_prediction_weight"
    ] = (
        pd.to_numeric(
            integrated[
                "provisional_prediction_weight"
            ],
            errors="coerce",
        ).fillna(0.0)
        * relationship_multiplier
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    validation_allowed = ~integrated[
        "validation_status"
    ].isin(
        [
            "not_tested",
            "reversed",
            "reversed_strong",
            "insufficient_2025_sample",
        ]
    )

    integrated[
        "integrated_prediction_weight_ready"
    ] = (
        validation_allowed
        & integrated[
            "league_adjusted_belief_score"
        ].ge(0.76)
        & integrated[
            "league_adjusted_prediction_weight"
        ].gt(0.0)
    )

    integrated[
        "integrated_signed_prediction_weight"
    ] = (
        integrated[
            "league_adjusted_prediction_weight"
        ]
        * np.where(
            integrated["effect_direction"].eq(
                "supports_target"
            ),
            1.0,
            -1.0,
        )
    )

    integrated["league_prior_integrated"] = True
    integrated["team_specific_signal_preserved"] = True
    integrated["2026_outcomes_used"] = False
    integrated["walk_forward_update_required"] = True
    integrated["integration_engine_version"] = (
        ENGINE_VERSION
    )
    integrated["integrated_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    return integrated


def build_summary(
    integrated: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        integrated.groupby(
            [
                "team",
                "target",
            ],
            sort=True,
        )
        .agg(
            concepts=(
                "concept_id",
                "nunique",
            ),
            reinforced_by_league=(
                "league_relationship",
                lambda values: int(
                    values.eq(
                        "reinforced_by_league"
                    ).sum()
                ),
            ),
            team_specific=(
                "league_relationship",
                lambda values: int(
                    values.eq(
                        "team_specific"
                    ).sum()
                ),
            ),
            league_conflicts=(
                "league_relationship",
                lambda values: int(
                    values.isin(
                        [
                            "conflicts_with_league",
                            "mixed_league_evidence_conflict_leads",
                        ]
                    ).sum()
                ),
            ),
            integrated_ready_concepts=(
                "integrated_prediction_weight_ready",
                "sum",
            ),
            mean_original_belief=(
                "belief_score",
                "mean",
            ),
            mean_integrated_belief=(
                "league_adjusted_belief_score",
                "mean",
            ),
            maximum_integrated_belief=(
                "league_adjusted_belief_score",
                "max",
            ),
            mean_integrated_weight=(
                "league_adjusted_prediction_weight",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        "integrated_ready_concepts"
    ] = (
        summary[
            "integrated_ready_concepts"
        ].astype("int64")
    )

    summary["discovery_season"] = (
        DISCOVERY_SEASON
    )

    summary["validation_season"] = (
        VALIDATION_SEASON
    )

    summary["2026_outcomes_used"] = False
    summary["engine_version"] = ENGINE_VERSION

    return summary


def run_league_prior_integration() -> dict[str, Any]:
    belief = _load_parquet(
        BELIEF_REGISTRY_PATH,
        "concept belief registry",
    )

    team_evidence = _load_parquet(
        LEAGUE_TEAM_PATH,
        "league team evidence",
    )

    game_evidence = _load_parquet(
        LEAGUE_GAME_PATH,
        "league game-environment evidence",
    )

    league = _prepare_league_evidence(
        team_evidence=team_evidence,
        game_evidence=game_evidence,
    )

    league_priors = build_league_concept_priors(
        league
    )

    integrated = integrate_team_and_league(
        belief=belief,
        league_priors=league_priors,
    )

    summary = build_summary(integrated)

    duplicate_ids = int(
        integrated["concept_id"]
        .duplicated().sum()
    )

    if duplicate_ids:
        raise AssertionError(
            f"Duplicate integrated concept IDs: "
            f"{duplicate_ids}"
        )

    if len(integrated) != len(belief):
        raise AssertionError(
            "Integrated registry row count changed."
        )

    if integrated[
        "2026_outcomes_used"
    ].any():
        raise AssertionError(
            "2026 outcomes were used."
        )

    _atomic_parquet_write(
        league_priors,
        LEAGUE_CONCEPT_PRIOR_PATH,
    )

    _atomic_parquet_write(
        integrated,
        INTEGRATED_REGISTRY_PATH,
    )

    _atomic_parquet_write(
        summary,
        INTEGRATED_SUMMARY_PATH,
    )

    relationship_counts = (
        integrated["league_relationship"]
        .value_counts()
    )

    result = {
        "engine":
            "ATLAS League-Prior Integration Engine",
        "engine_version":
            ENGINE_VERSION,
        "team_concepts_integrated":
            int(len(integrated)),
        "league_raw_evidence":
            int(len(league)),
        "league_concept_priors":
            int(len(league_priors)),
        "teams":
            int(
                integrated["team"].nunique()
            ),
        "targets":
            int(
                integrated["target"].nunique()
            ),
        "reinforced_by_league":
            int(
                relationship_counts.get(
                    "reinforced_by_league",
                    0,
                )
            ),
        "team_specific":
            int(
                relationship_counts.get(
                    "team_specific",
                    0,
                )
            ),
        "conflicts_with_league":
            int(
                relationship_counts.get(
                    "conflicts_with_league",
                    0,
                )
                + relationship_counts.get(
                    "mixed_league_evidence_conflict_leads",
                    0,
                )
            ),
        "mixed_agreement_leads":
            int(
                relationship_counts.get(
                    "mixed_league_evidence_agreement_leads",
                    0,
                )
            ),
        "integrated_ready_concepts":
            int(
                integrated[
                    "integrated_prediction_weight_ready"
                ].sum()
            ),
        "duplicate_concept_ids":
            duplicate_ids,
        "2026_outcomes_used":
            False,
        "outputs": {
            "league_concept_priors":
                str(LEAGUE_CONCEPT_PRIOR_PATH),
            "integrated_registry":
                str(INTEGRATED_REGISTRY_PATH),
            "integrated_summary":
                str(INTEGRATED_SUMMARY_PATH),
        },
        "policy": {
            "team_specific_signals_preserved":
                True,
            "league_priors_can_erase_team_signal":
                False,
            "league_conflicts_flagged":
                True,
            "league_adjustment_is_bounded":
                True,
            "weights_remain_provisional":
                True,
            "2026_walk_forward_required":
                True,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print(
        "ATLAS LEAGUE-PRIOR INTEGRATION ENGINE"
    )
    print("=" * 78)
    print(
        f"Team Concepts Integrated... "
        f"{len(integrated):,}"
    )
    print(
        f"League Raw Evidence........ "
        f"{len(league):,}"
    )
    print(
        f"League Concept Priors...... "
        f"{len(league_priors):,}"
    )
    print(
        f"Reinforced by League....... "
        f"{result['reinforced_by_league']:,}"
    )
    print(
        f"Team-Specific.............. "
        f"{result['team_specific']:,}"
    )
    print(
        f"League Conflicts........... "
        f"{result['conflicts_with_league']:,}"
    )
    print(
        f"Mixed — Agreement Leads.... "
        f"{result['mixed_agreement_leads']:,}"
    )
    print(
        f"Integrated Ready Concepts.. "
        f"{result['integrated_ready_concepts']:,}"
    )
    print(
        f"Duplicate Concept IDs...... "
        f"{duplicate_ids:,}"
    )
    print(
        f"2026 Outcomes Used......... "
        f"{result['2026_outcomes_used']}"
    )
    print(
        f"Saved To................... "
        f"{INTEGRATED_REGISTRY_PATH}"
    )
    print("=" * 78)

    return result
