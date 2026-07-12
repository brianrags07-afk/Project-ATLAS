
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
LEARNING_SEASON = 2024

RAW_EVIDENCE_PATH = (
    DATA_DIR
    / "learning"
    / "team_evidence"
    / str(LEARNING_SEASON)
    / "team_evidence_registry.parquet"
)

RAW_TEAM_SUMMARY_PATH = (
    DATA_DIR
    / "learning"
    / "team_evidence"
    / str(LEARNING_SEASON)
    / "team_evidence_summary.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(LEARNING_SEASON)
)

CONCEPT_REGISTRY_PATH = (
    OUTPUT_DIR
    / "team_concept_registry.parquet"
)

CONCEPT_MEMBER_MAP_PATH = (
    OUTPUT_DIR
    / "team_concept_member_map.parquet"
)

CONCEPT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "team_concept_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "evidence_consolidation_metadata.json"
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


def _load_raw_evidence() -> pd.DataFrame:
    if not RAW_EVIDENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw evidence registry: {RAW_EVIDENCE_PATH}"
        )

    evidence = pd.read_parquet(
        RAW_EVIDENCE_PATH
    )

    required = {
        "evidence_id",
        "team",
        "target",
        "feature",
        "feature_family",
        "direction",
        "threshold_value",
        "condition_sample_size",
        "target_base_rate",
        "condition_observed_rate",
        "lift",
        "absolute_lift",
        "confidence_score",
        "lifecycle_status",
        "chronological_direction_consistent",
    }

    missing = required - set(
        evidence.columns
    )

    if missing:
        raise KeyError(
            f"Raw evidence registry missing columns: "
            f"{sorted(missing)}"
        )

    return evidence


def _scope_from_feature(
    feature: str,
) -> tuple[str, str | None]:
    feature = str(feature)

    slot_match = re.match(
        r"slot_(\d+)_",
        feature,
    )

    if slot_match:
        position = slot_match.group(1)

        return (
            "individual_batting_order_slot",
            position,
        )

    if feature.startswith("lineup_"):
        return (
            "lineup_aggregate",
            None,
        )

    if feature.startswith("starter_"):
        return (
            "opposing_starter",
            None,
        )

    return (
        "other_pregame_context",
        None,
    )


def _strip_structural_tokens(
    feature: str,
) -> str:
    normalized = str(feature).lower()

    normalized = re.sub(
        r"^slot_\d+_",
        "",
        normalized,
    )

    normalized = re.sub(
        r"^(lineup_|starter_)",
        "",
        normalized,
    )

    normalized = normalized.replace(
        "career_prior_",
        "",
    )

    normalized = normalized.replace(
        "season_prior_",
        "",
    )

    normalized = re.sub(
        r"_(mean|min|max|sum)$",
        "",
        normalized,
    )

    return normalized


def _concept_from_metric(
    normalized_metric: str,
) -> tuple[str, str]:
    metric = str(
        normalized_metric
    ).lower()

    # Run creation / overall offensive productivity
    if any(token in metric for token in [
        "wrc",
        "woba",
        "runs_created",
        "weighted_runs",
        "ops",
        "on_base_plus_slugging",
    ]):
        return (
            "run_creation",
            "offense",
        )

    # Power
    if any(token in metric for token in [
        "home_run",
        "slug",
        "isolated_power",
        "iso",
        "barrel",
        "extra_base",
        "max_exit_velocity",
    ]):
        return (
            "power",
            "offense",
        )

    # Contact quality
    if any(token in metric for token in [
        "hard_hit",
        "exit_velocity",
        "launch_angle",
        "batted_ball",
        "expected_woba",
        "expected_ba",
        "expected_slg",
        "sweet_spot",
    ]):
        return (
            "contact_quality",
            "offense",
        )

    # Bat-to-ball / strikeout avoidance
    if any(token in metric for token in [
        "whiff_pct",
        "whiffs",
        "strikeout_rate",
        "strikeouts",
        "contact_pct",
        "swinging_strike",
    ]):
        return (
            "bat_to_ball",
            "plate_discipline",
        )

    # Patience / zone judgment
    if any(token in metric for token in [
        "walk_rate",
        "walks",
        "ball_pct",
        "balls_seen",
        "chase_pct",
        "chase_swings",
        "out_zone",
        "called_strike",
        "swing_pct",
    ]):
        return (
            "plate_discipline",
            "plate_discipline",
        )

    # General offensive volume / lineup depth
    if any(token in metric for token in [
        "plate_appearances",
        "games",
        "pitches_seen",
        "swings",
        "hits",
    ]):
        return (
            "experience_and_lineup_depth",
            "lineup_structure",
        )

    # Pitcher bat-missing ability
    if any(token in metric for token in [
        "csw_pct",
        "strikeout_rate_per_pa",
        "whiff_pct_per_swing",
        "whiffs",
        "swinging_strike",
    ]):
        return (
            "starter_bat_missing_ability",
            "opposing_pitcher",
        )

    # Pitcher command / strike throwing
    if any(token in metric for token in [
        "strike_pct",
        "ball_pct",
        "walk_rate_per_pa",
        "called_strike_pct",
        "zone_pct",
        "first_pitch_strike",
    ]):
        return (
            "starter_command",
            "opposing_pitcher",
        )

    # Pitcher location risk
    if any(token in metric for token in [
        "heart_pct",
        "middle_middle",
        "chase_pct",
        "out_zone_pct",
    ]):
        return (
            "starter_location_profile",
            "opposing_pitcher",
        )

    # Pitcher contact allowed
    if any(token in metric for token in [
        "hit_rate_per_pa",
        "home_run_rate_per_pa",
        "hard_hit_pct",
        "avg_exit_velocity_allowed",
        "avg_launch_angle_allowed",
        "hits_allowed",
        "home_runs_allowed",
    ]):
        return (
            "starter_contact_allowed",
            "opposing_pitcher",
        )

    # Pitcher velocity
    if any(token in metric for token in [
        "velocity",
        "effective_speed",
    ]):
        return (
            "starter_velocity",
            "opposing_pitcher",
        )

    # Workload / experience
    if any(token in metric for token in [
        "pitches_thrown",
        "plate_appearances",
        "games",
    ]):
        return (
            "starter_workload_and_experience",
            "opposing_pitcher",
        )

    return (
        "other_measured_signal",
        "other",
    )


def _classify_feature(
    feature: str,
) -> dict[str, Any]:
    scope, batting_order_slot = (
        _scope_from_feature(
            feature
        )
    )

    normalized_metric = (
        _strip_structural_tokens(
            feature
        )
    )

    concept_name, concept_domain = (
        _concept_from_metric(
            normalized_metric
        )
    )

    # Batter and lineup evidence can never be labeled as
    # opposing-pitcher evidence merely because a metric name
    # overlaps a pitcher-allowed statistic.
    if scope in {
        "individual_batting_order_slot",
        "lineup_aggregate",
    }:
        if any(token in normalized_metric for token in [
            "wrc",
            "woba",
            "runs_created",
            "weighted_runs",
            "ops",
        ]):
            concept_name = "run_creation"
            concept_domain = "offense"

        elif any(token in normalized_metric for token in [
            "home_run",
            "slug",
            "isolated_power",
            "iso",
            "barrel",
            "extra_base",
            "max_exit_velocity",
        ]):
            concept_name = "power"
            concept_domain = "offense"

        elif any(token in normalized_metric for token in [
            "hard_hit",
            "exit_velocity",
            "launch_angle",
            "batted_ball",
            "expected_woba",
            "expected_ba",
            "expected_slg",
            "sweet_spot",
            "hit_rate_per_pa",
            "hits",
        ]):
            concept_name = "contact_quality"
            concept_domain = "offense"

        elif any(token in normalized_metric for token in [
            "whiff",
            "strikeout",
            "contact_pct",
            "swinging_strike",
        ]):
            concept_name = "bat_to_ball"
            concept_domain = "plate_discipline"

        elif any(token in normalized_metric for token in [
            "walk",
            "ball_pct",
            "balls_seen",
            "chase",
            "out_zone",
            "called_strike",
            "swing_pct",
        ]):
            concept_name = "plate_discipline"
            concept_domain = "plate_discipline"

        elif any(token in normalized_metric for token in [
            "plate_appearances",
            "games",
            "pitches_seen",
            "swings",
        ]):
            concept_name = "experience_and_lineup_depth"
            concept_domain = "lineup_structure"

    # Pitcher features must use pitcher concepts even when
    # the metric name overlaps offensive concepts.
    if scope == "opposing_starter":
        if any(token in normalized_metric for token in [
            "whiff",
            "strikeout",
            "csw",
            "swinging_strike",
        ]):
            concept_name = (
                "starter_bat_missing_ability"
            )
            concept_domain = (
                "opposing_pitcher"
            )

        elif any(token in normalized_metric for token in [
            "strike_pct",
            "ball_pct",
            "walk_rate",
            "called_strike",
            "zone_pct",
        ]):
            concept_name = "starter_command"
            concept_domain = (
                "opposing_pitcher"
            )

        elif any(token in normalized_metric for token in [
            "heart",
            "middle_middle",
            "chase",
            "out_zone",
        ]):
            concept_name = (
                "starter_location_profile"
            )
            concept_domain = (
                "opposing_pitcher"
            )

        elif any(token in normalized_metric for token in [
            "hit_rate",
            "home_run_rate",
            "hard_hit",
            "exit_velocity",
            "launch_angle",
            "hits_allowed",
            "home_runs_allowed",
        ]):
            concept_name = (
                "starter_contact_allowed"
            )
            concept_domain = (
                "opposing_pitcher"
            )

        elif "velocity" in normalized_metric:
            concept_name = (
                "starter_velocity"
            )
            concept_domain = (
                "opposing_pitcher"
            )

        elif any(token in normalized_metric for token in [
            "pitches_thrown",
            "plate_appearances",
            "games",
        ]):
            concept_name = (
                "starter_workload_and_experience"
            )
            concept_domain = (
                "opposing_pitcher"
            )

    return {
        "concept_scope": scope,
        "batting_order_slot":
            batting_order_slot,
        "normalized_metric":
            normalized_metric,
        "concept_name":
            concept_name,
        "concept_domain":
            concept_domain,
    }


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


def _concept_id(
    team: str,
    target: str,
    concept_scope: str,
    batting_order_slot: str | None,
    concept_name: str,
    effect_direction: str,
) -> str:
    raw = "|".join([
        str(LEARNING_SEASON),
        str(team),
        str(target),
        str(concept_scope),
        str(batting_order_slot or "ALL"),
        str(concept_name),
        str(effect_direction),
    ])

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{team}_{target}_{concept_name}_"
        f"{effect_direction}_{digest}"
    ).upper()


def _status_rank(
    status: str,
) -> int:
    mapping = {
        "strong_candidate": 3,
        "candidate": 2,
        "weak_candidate": 1,
    }

    return mapping.get(
        str(status),
        0,
    )


def _concept_status(
    member_count: int,
    strong_members: int,
    candidate_members: int,
    chronological_consistency_rate: float,
    median_absolute_lift: float,
    concept_confidence: float,
) -> str:
    if (
        member_count >= 3
        and strong_members >= 1
        and chronological_consistency_rate >= 0.60
        and median_absolute_lift >= 0.10
        and concept_confidence >= 0.65
    ):
        return "strong_concept_candidate"

    if (
        member_count >= 2
        and (
            strong_members >= 1
            or candidate_members >= 2
        )
        and median_absolute_lift >= 0.08
        and concept_confidence >= 0.50
    ):
        return "concept_candidate"

    return "weak_concept_candidate"


def build_concept_tables(
    raw_evidence: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    evidence = raw_evidence.copy()

    classifications = (
        evidence["feature"]
        .apply(_classify_feature)
        .apply(pd.Series)
    )

    evidence = pd.concat(
        [
            evidence.reset_index(drop=True),
            classifications.reset_index(drop=True),
        ],
        axis=1,
    )

    evidence["effect_direction"] = (
        evidence["lift"]
        .apply(_effect_direction)
    )

    evidence["status_rank"] = (
        evidence[
            "lifecycle_status"
        ].apply(_status_rank)
    )

    evidence["concept_id"] = [
        _concept_id(
            team=row.team,
            target=row.target,
            concept_scope=row.concept_scope,
            batting_order_slot=(
                row.batting_order_slot
                if pd.notna(
                    row.batting_order_slot
                )
                else None
            ),
            concept_name=row.concept_name,
            effect_direction=row.effect_direction,
        )
        for row in evidence.itertuples(
            index=False
        )
    ]

    member_map_columns = [
        "concept_id",
        "evidence_id",
        "learning_season",
        "team",
        "target",
        "concept_domain",
        "concept_scope",
        "batting_order_slot",
        "concept_name",
        "effect_direction",
        "feature_family",
        "feature",
        "normalized_metric",
        "direction",
        "threshold_operator",
        "threshold_value",
        "condition_sample_size",
        "target_base_rate",
        "condition_observed_rate",
        "lift",
        "absolute_lift",
        "confidence_score",
        "chronological_direction_consistent",
        "lifecycle_status",
        "validated_out_of_sample",
        "requires_2025_validation",
    ]

    available_member_columns = [
        column
        for column in member_map_columns
        if column in evidence.columns
    ]

    member_map = evidence[
        available_member_columns
    ].copy()

    group_columns = [
        "concept_id",
        "learning_season",
        "team",
        "target",
        "concept_domain",
        "concept_scope",
        "batting_order_slot",
        "concept_name",
        "effect_direction",
    ]

    records: list[dict[str, Any]] = []

    for keys, group in evidence.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            concept_id,
            learning_season,
            team,
            target,
            concept_domain,
            concept_scope,
            batting_order_slot,
            concept_name,
            effect_direction,
        ) = keys

        weights = (
            pd.to_numeric(
                group["condition_sample_size"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=1.0)
        )

        confidence_values = (
            pd.to_numeric(
                group["confidence_score"],
                errors="coerce",
            )
            .fillna(0.0)
        )

        absolute_lifts = pd.to_numeric(
            group["absolute_lift"],
            errors="coerce",
        )

        lifts = pd.to_numeric(
            group["lift"],
            errors="coerce",
        )

        weighted_confidence = float(
            np.average(
                confidence_values,
                weights=weights,
            )
        )

        weighted_lift = float(
            np.average(
                lifts.fillna(0.0),
                weights=weights,
            )
        )

        chronological_rate = float(
            group[
                "chronological_direction_consistent"
            ]
            .fillna(False)
            .astype(bool)
            .mean()
        )

        member_count = int(
            len(group)
        )

        strong_members = int(
            group[
                "lifecycle_status"
            ].eq(
                "strong_candidate"
            ).sum()
        )

        candidate_members = int(
            group[
                "lifecycle_status"
            ].eq(
                "candidate"
            ).sum()
        )

        weak_members = int(
            group[
                "lifecycle_status"
            ].eq(
                "weak_candidate"
            ).sum()
        )

        median_absolute_lift = float(
            absolute_lifts.median()
        )

        maximum_absolute_lift = float(
            absolute_lifts.max()
        )

        concept_status = _concept_status(
            member_count=member_count,
            strong_members=strong_members,
            candidate_members=candidate_members,
            chronological_consistency_rate=
                chronological_rate,
            median_absolute_lift=
                median_absolute_lift,
            concept_confidence=
                weighted_confidence,
        )

        career_members = int(
            group["feature"]
            .astype(str)
            .str.contains(
                "career_prior",
                regex=False,
            )
            .sum()
        )

        season_members = int(
            group["feature"]
            .astype(str)
            .str.contains(
                "season_prior",
                regex=False,
            )
            .sum()
        )

        career_season_agreement = bool(
            career_members > 0
            and season_members > 0
        )

        records.append({
            "concept_id":
                concept_id,
            "learning_season":
                int(learning_season),
            "team":
                str(team),
            "target":
                str(target),
            "concept_domain":
                str(concept_domain),
            "concept_scope":
                str(concept_scope),
            "batting_order_slot": (
                None
                if pd.isna(
                    batting_order_slot
                )
                else str(
                    batting_order_slot
                )
            ),
            "concept_name":
                str(concept_name),
            "effect_direction":
                str(effect_direction),
            "member_count":
                member_count,
            "unique_features":
                int(
                    group[
                        "feature"
                    ].nunique()
                ),
            "unique_normalized_metrics":
                int(
                    group[
                        "normalized_metric"
                    ].nunique()
                ),
            "strong_members":
                strong_members,
            "candidate_members":
                candidate_members,
            "weak_members":
                weak_members,
            "career_members":
                career_members,
            "season_members":
                season_members,
            "career_season_agreement":
                career_season_agreement,
            "total_condition_sample":
                int(
                    pd.to_numeric(
                        group[
                            "condition_sample_size"
                        ],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
            "median_condition_sample":
                float(
                    pd.to_numeric(
                        group[
                            "condition_sample_size"
                        ],
                        errors="coerce",
                    ).median()
                ),
            "weighted_lift":
                weighted_lift,
            "median_absolute_lift":
                median_absolute_lift,
            "maximum_absolute_lift":
                maximum_absolute_lift,
            "chronological_consistency_rate":
                chronological_rate,
            "concept_confidence_score":
                weighted_confidence,
            "concept_lifecycle_status":
                concept_status,
            "raw_evidence_preserved":
                True,
            "predictive_weight_assigned":
                False,
            "validated_out_of_sample":
                False,
            "requires_2025_validation":
                True,
            "engine_version":
                ENGINE_VERSION,
        })

    concept_registry = pd.DataFrame(
        records
    )

    concept_registry = (
        concept_registry.sort_values(
            [
                "team",
                "target",
                "concept_confidence_score",
                "median_absolute_lift",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    summary = (
        concept_registry.groupby(
            [
                "team",
                "target",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            concepts_found=(
                "concept_id",
                "nunique",
            ),
            strong_concepts=(
                "concept_lifecycle_status",
                lambda values: int(
                    (
                        values
                        == "strong_concept_candidate"
                    ).sum()
                ),
            ),
            concept_candidates=(
                "concept_lifecycle_status",
                lambda values: int(
                    (
                        values
                        == "concept_candidate"
                    ).sum()
                ),
            ),
            weak_concepts=(
                "concept_lifecycle_status",
                lambda values: int(
                    (
                        values
                        == "weak_concept_candidate"
                    ).sum()
                ),
            ),
            raw_evidence_members=(
                "member_count",
                "sum",
            ),
            average_concept_confidence=(
                "concept_confidence_score",
                "mean",
            ),
        )
        .reset_index()
    )

    # Preserve every team-target pair, including targets that
    # generated zero qualifying concepts.
    if not RAW_TEAM_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing team evidence summary: "
            f"{RAW_TEAM_SUMMARY_PATH}"
        )

    expected_pairs = (
        pd.read_parquet(
            RAW_TEAM_SUMMARY_PATH
        )[
            ["team", "target"]
        ]
        .drop_duplicates()
    )

    summary = expected_pairs.merge(
        summary,
        on=["team", "target"],
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "concepts_found",
        "strong_concepts",
        "concept_candidates",
        "weak_concepts",
        "raw_evidence_members",
    ]

    for column in count_columns:
        summary[column] = (
            summary[column]
            .fillna(0)
            .astype("int64")
        )

    summary["average_concept_confidence"] = (
        pd.to_numeric(
            summary[
                "average_concept_confidence"
            ],
            errors="coerce",
        )
    )

    summary[
        "learning_season"
    ] = LEARNING_SEASON

    summary[
        "requires_2025_validation"
    ] = True

    summary[
        "engine_version"
    ] = ENGINE_VERSION

    summary = summary.sort_values(
        ["team", "target"],
        kind="stable",
    ).reset_index(drop=True)

    return (
        concept_registry,
        member_map,
        summary,
    )


def validate_consolidation(
    raw_evidence: pd.DataFrame,
    concept_registry: pd.DataFrame,
    member_map: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    raw_ids = set(
        raw_evidence[
            "evidence_id"
        ].astype(str)
    )

    mapped_ids = set(
        member_map[
            "evidence_id"
        ].astype(str)
    )

    missing_ids = sorted(
        raw_ids - mapped_ids
    )

    extra_ids = sorted(
        mapped_ids - raw_ids
    )

    duplicate_member_ids = int(
        member_map[
            "evidence_id"
        ].duplicated().sum()
    )

    duplicate_concept_ids = int(
        concept_registry[
            "concept_id"
        ].duplicated().sum()
    )

    teams = int(
        concept_registry[
            "team"
        ].nunique()
    )

    targets = int(
        concept_registry[
            "target"
        ].nunique()
    )

    weights_assigned = int(
        concept_registry[
            "predictive_weight_assigned"
        ].fillna(False).sum()
    )

    if missing_ids:
        raise AssertionError(
            f"{len(missing_ids):,} raw evidence IDs "
            "were not mapped to a concept."
        )

    if extra_ids:
        raise AssertionError(
            f"{len(extra_ids):,} unknown evidence IDs "
            "appeared in the member map."
        )

    if duplicate_member_ids:
        raise AssertionError(
            f"{duplicate_member_ids:,} evidence IDs "
            "were mapped more than once."
        )

    if duplicate_concept_ids:
        raise AssertionError(
            f"{duplicate_concept_ids:,} duplicate "
            "concept IDs were created."
        )

    if teams != 30:
        raise AssertionError(
            f"Expected 30 teams; found {teams}."
        )

    if weights_assigned:
        raise AssertionError(
            "Concept engine assigned predictive weights."
        )

    return {
        "learning_season":
            LEARNING_SEASON,
        "raw_evidence_objects":
            int(len(raw_evidence)),
        "mapped_evidence_objects":
            int(len(member_map)),
        "concept_objects":
            int(len(concept_registry)),
        "teams":
            teams,
        "targets":
            targets,
        "team_target_summary_rows":
            int(len(summary)),
        "strong_concepts":
            int(
                concept_registry[
                    "concept_lifecycle_status"
                ].eq(
                    "strong_concept_candidate"
                ).sum()
            ),
        "concept_candidates":
            int(
                concept_registry[
                    "concept_lifecycle_status"
                ].eq(
                    "concept_candidate"
                ).sum()
            ),
        "weak_concepts":
            int(
                concept_registry[
                    "concept_lifecycle_status"
                ].eq(
                    "weak_concept_candidate"
                ).sum()
            ),
        "unmapped_evidence_ids":
            int(len(missing_ids)),
        "duplicate_member_ids":
            duplicate_member_ids,
        "duplicate_concept_ids":
            duplicate_concept_ids,
        "predictive_weights_assigned":
            weights_assigned,
    }


def run_evidence_consolidation() -> dict[str, Any]:
    raw_evidence = _load_raw_evidence()

    (
        concept_registry,
        member_map,
        summary,
    ) = build_concept_tables(
        raw_evidence
    )

    validation = validate_consolidation(
        raw_evidence=raw_evidence,
        concept_registry=concept_registry,
        member_map=member_map,
        summary=summary,
    )

    _atomic_parquet_write(
        concept_registry,
        CONCEPT_REGISTRY_PATH,
    )

    _atomic_parquet_write(
        member_map,
        CONCEPT_MEMBER_MAP_PATH,
    )

    _atomic_parquet_write(
        summary,
        CONCEPT_SUMMARY_PATH,
    )

    metadata = {
        "engine": (
            "ATLAS Evidence Consolidation Engine"
        ),
        "engine_version":
            ENGINE_VERSION,
        "learning_season":
            LEARNING_SEASON,
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "validation":
            validation,
        "outputs": {
            "concept_registry": str(
                CONCEPT_REGISTRY_PATH
            ),
            "concept_member_map": str(
                CONCEPT_MEMBER_MAP_PATH
            ),
            "concept_summary": str(
                CONCEPT_SUMMARY_PATH
            ),
        },
        "policy": {
            "raw_evidence_deleted":
                False,
            "every_raw_evidence_id_preserved":
                True,
            "career_and_season_signals_grouped":
                True,
            "batting_order_scope_preserved":
                True,
            "prediction_weights_assigned":
                False,
            "2025_validation_required":
                True,
        },
    }

    _atomic_json_write(
        metadata,
        METADATA_PATH,
    )

    print("=" * 78)
    print(
        "ATLAS EVIDENCE CONSOLIDATION ENGINE"
    )
    print("=" * 78)
    print(
        f"Raw Evidence Objects...... "
        f"{validation['raw_evidence_objects']:,}"
    )
    print(
        f"Mapped Evidence Objects... "
        f"{validation['mapped_evidence_objects']:,}"
    )
    print(
        f"Concept Objects........... "
        f"{validation['concept_objects']:,}"
    )
    print(
        f"Teams..................... "
        f"{validation['teams']:,}"
    )
    print(
        f"Targets................... "
        f"{validation['targets']:,}"
    )
    print(
        f"Strong Concepts........... "
        f"{validation['strong_concepts']:,}"
    )
    print(
        f"Concept Candidates........ "
        f"{validation['concept_candidates']:,}"
    )
    print(
        f"Weak Concepts............. "
        f"{validation['weak_concepts']:,}"
    )
    print(
        f"Unmapped Evidence......... "
        f"{validation['unmapped_evidence_ids']:,}"
    )
    print(
        f"Duplicate Member IDs...... "
        f"{validation['duplicate_member_ids']:,}"
    )
    print(
        f"Prediction Weights........ "
        f"{validation['predictive_weights_assigned']:,}"
    )
    print(
        f"Saved To.................. "
        f"{CONCEPT_REGISTRY_PATH}"
    )
    print("=" * 78)

    return metadata
