
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
EVALUATION_SEASON = 2026

FAILURE_DIR = (
    DATA_DIR
    / "history"
    / "failure_analysis"
    / str(EVALUATION_SEASON)
)

TARGET_PERFORMANCE_PATH = (
    FAILURE_DIR
    / "target_performance.parquet"
)

TEAM_REPORTS_PATH = (
    FAILURE_DIR
    / "team_report_cards.parquet"
)

CONCEPT_REPORTS_PATH = (
    FAILURE_DIR
    / "concept_report_cards.parquet"
)

VALIDATION_AUDIT_PATH = (
    FAILURE_DIR
    / "validation_audit.parquet"
)

FAILURE_CLUSTERS_PATH = (
    FAILURE_DIR
    / "failure_clusters.parquet"
)

LEARNING_QUEUE_PATH = (
    FAILURE_DIR
    / "learning_queue.parquet"
)

FAILURE_SUMMARY_PATH = (
    FAILURE_DIR
    / "failure_summary.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "history"
    / "model_repair_plans"
    / str(EVALUATION_SEASON)
)

TARGET_REPAIR_PLAN_PATH = (
    OUTPUT_DIR
    / "target_repair_plan.parquet"
)

CONCEPT_REPAIR_PLAN_PATH = (
    OUTPUT_DIR
    / "concept_repair_plan.parquet"
)

TEAM_REPAIR_PLAN_PATH = (
    OUTPUT_DIR
    / "team_repair_plan.parquet"
)

IDENTITY_GAP_PLAN_PATH = (
    OUTPUT_DIR
    / "identity_gap_plan.parquet"
)

DOMAIN_BALANCE_PLAN_PATH = (
    OUTPUT_DIR
    / "domain_balance_plan.parquet"
)

BUILD_QUEUE_PATH = (
    OUTPUT_DIR
    / "controlled_build_queue.parquet"
)

EXECUTIVE_PLAN_PATH = (
    OUTPUT_DIR
    / "executive_repair_plan.parquet"
)

REPAIR_SUMMARY_PATH = (
    OUTPUT_DIR
    / "repair_plan_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "model_repair_plan_metadata.json"
)


def _load(
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


def _priority_tier(
    score: float,
) -> str:
    if score >= 120:
        return "P0_CRITICAL"

    if score >= 105:
        return "P1_HIGH"

    if score >= 90:
        return "P2_MEDIUM"

    return "P3_RESEARCH"


def _build_target_plan(
    target_performance: pd.DataFrame,
    failure_clusters: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in target_performance.itertuples(
        index=False
    ):
        target = str(row.target)

        target_clusters = failure_clusters[
            failure_clusters["target"].eq(target)
        ].copy()

        sparse_failures = int(
            target_clusters.loc[
                target_clusters[
                    "root_cause"
                ].eq("failure_sparse_evidence"),
                "failures",
            ].sum()
        )

        contradictory_failures = int(
            target_clusters.loc[
                target_clusters[
                    "root_cause"
                ].eq(
                    "failure_contradictory_evidence"
                ),
                "failures",
            ].sum()
        )

        overconfident_failures = int(
            target_clusters.loc[
                target_clusters[
                    "root_cause"
                ].eq("failure_overconfident"),
                "failures",
            ].sum()
        )

        brier_gain = (
            float(row.brier_improvement)
            if pd.notna(row.brier_improvement)
            else np.nan
        )

        coverage = float(row.coverage_rate)

        performance_penalty = (
            abs(min(brier_gain, 0.0)) * 1000
            if pd.notna(brier_gain)
            else 15.0
        )

        coverage_penalty = (
            max(0.0, 0.30 - coverage)
            * 100
        )

        priority_score = (
            70.0
            + performance_penalty
            + coverage_penalty
            + min(sparse_failures, 100) * 0.10
            + min(contradictory_failures, 50) * 0.20
            + min(overconfident_failures, 50) * 0.25
        )

        if row.target_action == "REBUILD":
            repair_action = (
                "REBUILD_FEATURES_WEIGHTS_AND_CALIBRATION"
            )
        elif row.target_action == "REDUCE":
            repair_action = (
                "REDUCE_TRUST_AND_EXPAND_EVIDENCE"
            )
        else:
            repair_action = (
                "HOLD_AND_GATHER_MORE_DATA"
            )

        if sparse_failures >= max(
            contradictory_failures,
            overconfident_failures,
        ):
            primary_failure_mode = (
                "SPARSE_EVIDENCE"
            )
        elif contradictory_failures >= (
            overconfident_failures
        ):
            primary_failure_mode = (
                "CONTRADICTORY_EVIDENCE"
            )
        else:
            primary_failure_mode = (
                "OVERCONFIDENCE"
            )

        records.append({
            "target": target,
            "current_action":
                row.target_action,
            "repair_action":
                repair_action,
            "primary_failure_mode":
                primary_failure_mode,
            "predictions":
                int(row.predictions),
            "coverage_rate":
                coverage,
            "brier_improvement":
                brier_gain,
            "log_loss_improvement":
                (
                    float(
                        row.log_loss_improvement
                    )
                    if pd.notna(
                        row.log_loss_improvement
                    )
                    else np.nan
                ),
            "calibration_bias":
                (
                    float(row.calibration_bias)
                    if pd.notna(
                        row.calibration_bias
                    )
                    else np.nan
                ),
            "sparse_failures":
                sparse_failures,
            "contradictory_failures":
                contradictory_failures,
            "overconfident_failures":
                overconfident_failures,
            "priority_score":
                priority_score,
            "priority_tier":
                _priority_tier(
                    priority_score
                ),
            "required_work": (
                "add richer pregame-safe concepts; "
                "rerun 2024 discovery; rerun blind 2025 "
                "validation; rebuild belief scores; "
                "recalibrate chronologically"
            ),
            "automatic_change_allowed":
                False,
            "human_review_required":
                True,
        })

    return (
        pd.DataFrame(records)
        .sort_values(
            "priority_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _build_concept_plan(
    concept_reports: pd.DataFrame,
) -> pd.DataFrame:
    plan = concept_reports.copy()

    plan["repair_action"] = np.select(
        [
            plan[
                "weight_recommendation"
            ].eq("PROMOTE"),
            plan[
                "weight_recommendation"
            ].eq("KEEP"),
            plan[
                "weight_recommendation"
            ].eq("RETIRE_CANDIDATE"),
            plan[
                "weight_recommendation"
            ].eq("GATHER_MORE_DATA"),
        ],
        [
            "HOLD_FOR_PROMOTION_REVIEW",
            "FREEZE_CURRENT_WEIGHT",
            "QUARANTINE_AND_RETEST",
            "FREEZE_PENDING_MORE_SAMPLE",
        ],
        default="REDUCE_AND_REVALIDATE",
    )

    plan["provisional_weight_multiplier"] = (
        np.select(
            [
                plan["repair_action"].eq(
                    "HOLD_FOR_PROMOTION_REVIEW"
                ),
                plan["repair_action"].eq(
                    "FREEZE_CURRENT_WEIGHT"
                ),
                plan["repair_action"].eq(
                    "FREEZE_PENDING_MORE_SAMPLE"
                ),
                plan["repair_action"].eq(
                    "REDUCE_AND_REVALIDATE"
                ),
                plan["repair_action"].eq(
                    "QUARANTINE_AND_RETEST"
                ),
            ],
            [
                1.10,
                1.00,
                1.00,
                0.65,
                0.00,
            ],
            default=1.00,
        )
    )

    # This is only a recommendation, never applied here.
    plan["provisional_weight_multiplier_applied"] = (
        False
    )

    plan["repair_priority_score"] = (
        plan["activations"]
        .clip(upper=100)
        * np.where(
            plan["direction_accuracy"].lt(0.50),
            1.25,
            0.50,
        )
        + plan["mean_prediction_weight"]
        .fillna(0.0)
        * 20
    )

    plan["priority_tier"] = (
        plan["repair_priority_score"]
        .apply(_priority_tier)
    )

    plan["automatic_weight_change"] = False
    plan["automatic_retirement"] = False
    plan["human_review_required"] = True

    selected_columns = [
        "concept_id",
        "team",
        "target",
        "concept_domain",
        "concept_scope",
        "concept_name",
        "effect_direction",
        "validation_status",
        "league_relationship",
        "activations",
        "direction_accuracy",
        "mean_prediction_weight",
        "weight_recommendation",
        "repair_action",
        "provisional_weight_multiplier",
        "provisional_weight_multiplier_applied",
        "repair_priority_score",
        "priority_tier",
        "automatic_weight_change",
        "automatic_retirement",
        "human_review_required",
    ]

    return (
        plan[selected_columns]
        .sort_values(
            "repair_priority_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _build_team_plan(
    team_reports: pd.DataFrame,
) -> pd.DataFrame:
    plan = team_reports.copy()

    plan["repair_action"] = np.select(
        [
            plan["team_grade"].eq("WEAK"),
            plan["team_grade"].eq("LOW_SAMPLE"),
            plan["team_grade"].eq("MIXED"),
            plan["team_grade"].isin(
                ["RELIABLE", "STRONG"]
            ),
        ],
        [
            "BUILD_TEAM_IDENTITY_EXPANSION",
            "EXPAND_TEAM_COVERAGE",
            "REVIEW_TEAM_SPECIFIC_CONCEPTS",
            "PRESERVE_AND_MONITOR",
        ],
        default="REVIEW",
    )

    plan["priority_score"] = (
        (1.0 - plan["coverage_rate"])
        * 60
        + np.where(
            plan["team_grade"].eq("WEAK"),
            50,
            0,
        )
        + np.where(
            plan["team_grade"].eq("MIXED"),
            20,
            0,
        )
        + np.where(
            plan["team_grade"].eq(
                "LOW_SAMPLE"
            ),
            30,
            0,
        )
    )

    plan["priority_tier"] = (
        plan["priority_score"]
        .apply(_priority_tier)
    )

    plan["required_identity_families"] = np.where(
        plan["repair_action"].eq(
            "PRESERVE_AND_MONITOR"
        ),
        "monitor existing team-local concepts",
        (
            "starter matchup; bullpen availability; "
            "offensive identity; home-road identity; "
            "series-rest-travel context"
        ),
    )

    plan["automatic_change_allowed"] = False
    plan["human_review_required"] = True

    return (
        plan.sort_values(
            "priority_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _build_domain_balance_plan(
    concept_reports: pd.DataFrame,
    failure_clusters: pd.DataFrame,
) -> pd.DataFrame:
    concept_domain = (
        concept_reports.groupby(
            "concept_domain",
            dropna=False,
            sort=True,
        )
        .agg(
            concepts=("concept_id", "nunique"),
            activations=("activations", "sum"),
            mean_direction_accuracy=(
                "direction_accuracy",
                "mean",
            ),
            promote_or_keep=(
                "weight_recommendation",
                lambda values: int(
                    values.isin(
                        ["PROMOTE", "KEEP"]
                    ).sum()
                ),
            ),
            reduce_or_retire=(
                "weight_recommendation",
                lambda values: int(
                    values.isin(
                        [
                            "RETIRE_CANDIDATE",
                            "REDUCE_OR_INVESTIGATE",
                        ]
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    failures = (
        failure_clusters.groupby(
            "dominant_concept_domain",
            dropna=False,
            sort=True,
        )
        .agg(
            clustered_failures=(
                "failures",
                "sum",
            ),
            failure_clusters=(
                "failures",
                "size",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "dominant_concept_domain":
                    "concept_domain"
            }
        )
    )

    plan = concept_domain.merge(
        failures,
        on="concept_domain",
        how="outer",
    )

    numeric_columns = [
        "concepts",
        "activations",
        "promote_or_keep",
        "reduce_or_retire",
        "clustered_failures",
        "failure_clusters",
    ]

    for column in numeric_columns:
        plan[column] = (
            plan[column]
            .fillna(0)
            .astype("int64")
        )

    plan["failure_per_concept"] = np.where(
        plan["concepts"].gt(0),
        plan["clustered_failures"]
        / plan["concepts"],
        np.nan,
    )

    plan["domain_action"] = np.select(
        [
            (
                plan["reduce_or_retire"]
                > plan["promote_or_keep"]
            ),
            (
                plan["clustered_failures"]
                >= 100
            ),
            (
                plan["concepts"] <= 5
            ),
        ],
        [
            "PRUNE_AND_REBUILD_DOMAIN",
            "EXPAND_WITH_RICHER_FEATURES",
            "UNDERREPRESENTED_EXPAND_DISCOVERY",
        ],
        default="MONITOR",
    )

    plan["automatic_change_allowed"] = False
    plan["human_review_required"] = True

    return (
        plan.sort_values(
            [
                "clustered_failures",
                "reduce_or_retire",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _build_identity_gap_plan(
    learning_queue: pd.DataFrame,
    target_plan: pd.DataFrame,
) -> pd.DataFrame:
    identity_areas = {
        "bullpen_availability_and_fatigue": {
            "engine_name":
                "Bullpen Availability and Fatigue Engine",
            "feature_scope":
                "reliever usage, rest, leverage, handedness, availability",
        },
        "starter_arsenal_and_matchup_identity": {
            "engine_name":
                "Starter Arsenal Matchup Engine",
            "feature_scope":
                "pitch mix, velocity, movement, platoon, opponent matchup",
        },
        "team_offensive_identity_v2": {
            "engine_name":
                "Team Offensive Identity Engine v2",
            "feature_scope":
                "contact, power, patience, sequencing, early-late scoring",
        },
        "series_rest_and_travel_context": {
            "engine_name":
                "Series Rest and Travel Engine",
            "feature_scope":
                "published series position, rest, travel, day-night transition",
        },
        "park_weather_umpire_environment": {
            "engine_name":
                "Park Weather and Umpire Engine",
            "feature_scope":
                "park, roof, temperature, wind, humidity, umpire zone",
        },
    }

    records: list[dict[str, Any]] = []

    for area, specification in (
        identity_areas.items()
    ):
        queue_match = learning_queue[
            learning_queue[
                "research_area"
            ].eq(area)
        ]

        queue_priority = (
            float(
                queue_match[
                    "priority_score"
                ].iloc[0]
            )
            if not queue_match.empty
            else 75.0
        )

        affected_targets = (
            target_plan.loc[
                target_plan[
                    "primary_failure_mode"
                ].eq("SPARSE_EVIDENCE"),
                "target",
            ]
            .astype(str)
            .tolist()
        )

        records.append({
            "identity_gap":
                area,
            "recommended_engine":
                specification["engine_name"],
            "feature_scope":
                specification["feature_scope"],
            "affected_targets":
                "|".join(affected_targets),
            "priority_score":
                queue_priority,
            "priority_tier":
                _priority_tier(
                    queue_priority
                ),
            "required_training_flow":
                (
                    "build pregame-safe features → "
                    "discover in 2024 → validate blindly in 2025 → "
                    "score beliefs → integrate league priors → "
                    "backtest and recalibrate → test on frozen 2026 ledger"
                ),
            "automatic_build":
                False,
            "human_approval_required":
                True,
        })

    return (
        pd.DataFrame(records)
        .sort_values(
            "priority_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _build_controlled_queue(
    target_plan: pd.DataFrame,
    concept_plan: pd.DataFrame,
    team_plan: pd.DataFrame,
    identity_plan: pd.DataFrame,
    domain_plan: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in target_plan.itertuples(
        index=False
    ):
        records.append({
            "work_type": "TARGET_REPAIR",
            "work_item": row.target,
            "priority_score":
                float(row.priority_score),
            "priority_tier":
                row.priority_tier,
            "required_action":
                row.repair_action,
            "source":
                "target_repair_plan",
        })

    for row in identity_plan.itertuples(
        index=False
    ):
        records.append({
            "work_type": "IDENTITY_ENGINE",
            "work_item": row.identity_gap,
            "priority_score":
                float(row.priority_score),
            "priority_tier":
                row.priority_tier,
            "required_action":
                row.recommended_engine,
            "source":
                "identity_gap_plan",
        })

    critical_concepts = concept_plan[
        concept_plan["repair_action"].isin(
            [
                "QUARANTINE_AND_RETEST",
                "REDUCE_AND_REVALIDATE",
            ]
        )
    ].head(25)

    for row in critical_concepts.itertuples(
        index=False
    ):
        records.append({
            "work_type": "CONCEPT_REPAIR",
            "work_item": str(row.concept_id),
            "priority_score":
                float(row.repair_priority_score),
            "priority_tier":
                row.priority_tier,
            "required_action":
                row.repair_action,
            "source":
                "concept_repair_plan",
        })

    for row in domain_plan.itertuples(
        index=False
    ):
        if row.domain_action == "MONITOR":
            continue

        records.append({
            "work_type": "DOMAIN_REPAIR",
            "work_item":
                str(row.concept_domain),
            "priority_score":
                float(
                    80
                    + min(
                        row.clustered_failures,
                        200,
                    ) * 0.10
                ),
            "priority_tier":
                _priority_tier(
                    80
                    + min(
                        row.clustered_failures,
                        200,
                    ) * 0.10
                ),
            "required_action":
                row.domain_action,
            "source":
                "domain_balance_plan",
        })

    queue = pd.DataFrame(records)

    queue = (
        queue.sort_values(
            "priority_score",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    queue.insert(
        0,
        "queue_rank",
        np.arange(
            1,
            len(queue) + 1,
        ),
    )

    queue["status"] = "PROPOSED"
    queue["approved"] = False
    queue["executed"] = False
    queue["historical_predictions_modified"] = False
    queue["current_weights_modified"] = False
    queue["human_review_required"] = True

    return queue


def _build_executive_plan(
    target_plan: pd.DataFrame,
    concept_plan: pd.DataFrame,
    team_plan: pd.DataFrame,
    identity_plan: pd.DataFrame,
    controlled_queue: pd.DataFrame,
) -> pd.DataFrame:
    quarantined = int(
        concept_plan["repair_action"]
        .eq("QUARANTINE_AND_RETEST")
        .sum()
    )

    reduced = int(
        concept_plan["repair_action"]
        .eq("REDUCE_AND_REVALIDATE")
        .sum()
    )

    top_target = target_plan.iloc[0]
    top_identity = identity_plan.iloc[0]

    rows = [
        {
            "section": "primary_target_repair",
            "finding": (
                f"{top_target['target']} is the highest-priority "
                f"target repair ({top_target['priority_tier']})."
            ),
        },
        {
            "section": "primary_identity_gap",
            "finding": (
                f"{top_identity['recommended_engine']} is the "
                "highest-priority new identity engine."
            ),
        },
        {
            "section": "concept_control",
            "finding": (
                f"{quarantined} concepts are quarantine candidates; "
                f"{reduced} require reduction and revalidation."
            ),
        },
        {
            "section": "team_expansion",
            "finding": (
                f"{int(team_plan['repair_action'].ne('PRESERVE_AND_MONITOR').sum())} "
                "teams require additional identity or coverage work."
            ),
        },
        {
            "section": "execution_control",
            "finding": (
                f"{len(controlled_queue)} repair tasks were proposed. "
                "None were approved or executed automatically."
            ),
        },
    ]

    report = pd.DataFrame(rows)

    report["evaluation_season"] = (
        EVALUATION_SEASON
    )

    report["generated_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    report["human_review_required"] = True

    return report


def run_model_repair_planning_engine() -> dict[str, Any]:
    target_performance = _load(
        TARGET_PERFORMANCE_PATH,
        "target performance",
    )

    team_reports = _load(
        TEAM_REPORTS_PATH,
        "team report cards",
    )

    concept_reports = _load(
        CONCEPT_REPORTS_PATH,
        "concept report cards",
    )

    validation_audit = _load(
        VALIDATION_AUDIT_PATH,
        "validation audit",
    )

    failure_clusters = _load(
        FAILURE_CLUSTERS_PATH,
        "failure clusters",
    )

    learning_queue = _load(
        LEARNING_QUEUE_PATH,
        "learning queue",
    )

    failure_summary = _load(
        FAILURE_SUMMARY_PATH,
        "failure summary",
    )

    target_plan = _build_target_plan(
        target_performance=target_performance,
        failure_clusters=failure_clusters,
    )

    concept_plan = _build_concept_plan(
        concept_reports=concept_reports,
    )

    team_plan = _build_team_plan(
        team_reports=team_reports,
    )

    domain_plan = _build_domain_balance_plan(
        concept_reports=concept_reports,
        failure_clusters=failure_clusters,
    )

    identity_plan = _build_identity_gap_plan(
        learning_queue=learning_queue,
        target_plan=target_plan,
    )

    controlled_queue = _build_controlled_queue(
        target_plan=target_plan,
        concept_plan=concept_plan,
        team_plan=team_plan,
        identity_plan=identity_plan,
        domain_plan=domain_plan,
    )

    executive_plan = _build_executive_plan(
        target_plan=target_plan,
        concept_plan=concept_plan,
        team_plan=team_plan,
        identity_plan=identity_plan,
        controlled_queue=controlled_queue,
    )

    summary = pd.DataFrame([{
        "evaluation_season":
            EVALUATION_SEASON,
        "targets_planned":
            int(len(target_plan)),
        "concepts_planned":
            int(len(concept_plan)),
        "teams_planned":
            int(len(team_plan)),
        "identity_gaps":
            int(len(identity_plan)),
        "concept_domains":
            int(len(domain_plan)),
        "controlled_queue_items":
            int(len(controlled_queue)),
        "approved_items":
            int(
                controlled_queue[
                    "approved"
                ].sum()
            ),
        "executed_items":
            int(
                controlled_queue[
                    "executed"
                ].sum()
            ),
        "historical_predictions_modified":
            False,
        "current_weights_modified":
            False,
        "automatic_retraining":
            False,
        "engine_version":
            ENGINE_VERSION,
    }])

    outputs = [
        (
            target_plan,
            TARGET_REPAIR_PLAN_PATH,
        ),
        (
            concept_plan,
            CONCEPT_REPAIR_PLAN_PATH,
        ),
        (
            team_plan,
            TEAM_REPAIR_PLAN_PATH,
        ),
        (
            identity_plan,
            IDENTITY_GAP_PLAN_PATH,
        ),
        (
            domain_plan,
            DOMAIN_BALANCE_PLAN_PATH,
        ),
        (
            controlled_queue,
            BUILD_QUEUE_PATH,
        ),
        (
            executive_plan,
            EXECUTIVE_PLAN_PATH,
        ),
        (
            summary,
            REPAIR_SUMMARY_PATH,
        ),
    ]

    for dataframe, path in outputs:
        _atomic_parquet_write(
            dataframe,
            path,
        )

    result = {
        "engine":
            "ATLAS Model Repair Planning Engine",
        "engine_version":
            ENGINE_VERSION,
        "evaluation_season":
            EVALUATION_SEASON,
        "targets_planned":
            int(len(target_plan)),
        "concepts_planned":
            int(len(concept_plan)),
        "teams_planned":
            int(len(team_plan)),
        "identity_gaps":
            int(len(identity_plan)),
        "domain_plans":
            int(len(domain_plan)),
        "controlled_queue_items":
            int(len(controlled_queue)),
        "approved_items":
            0,
        "executed_items":
            0,
        "historical_predictions_modified":
            False,
        "current_weights_modified":
            False,
        "automatic_retraining":
            False,
        "outputs": {
            "target_repair_plan":
                str(TARGET_REPAIR_PLAN_PATH),
            "concept_repair_plan":
                str(CONCEPT_REPAIR_PLAN_PATH),
            "team_repair_plan":
                str(TEAM_REPAIR_PLAN_PATH),
            "identity_gap_plan":
                str(IDENTITY_GAP_PLAN_PATH),
            "domain_balance_plan":
                str(DOMAIN_BALANCE_PLAN_PATH),
            "controlled_build_queue":
                str(BUILD_QUEUE_PATH),
            "executive_repair_plan":
                str(EXECUTIVE_PLAN_PATH),
            "repair_plan_summary":
                str(REPAIR_SUMMARY_PATH),
        },
        "policy": {
            "planning_only":
                True,
            "historical_predictions_immutable":
                True,
            "automatic_weight_changes":
                False,
            "automatic_concept_retirement":
                False,
            "automatic_retraining":
                False,
            "human_approval_required":
                True,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS MODEL REPAIR PLANNING ENGINE")
    print("=" * 78)
    print(
        f"Targets Planned............. "
        f"{len(target_plan):,}"
    )
    print(
        f"Concepts Planned............ "
        f"{len(concept_plan):,}"
    )
    print(
        f"Teams Planned............... "
        f"{len(team_plan):,}"
    )
    print(
        f"Identity Gaps............... "
        f"{len(identity_plan):,}"
    )
    print(
        f"Domain Plans................ "
        f"{len(domain_plan):,}"
    )
    print(
        f"Controlled Queue Items...... "
        f"{len(controlled_queue):,}"
    )
    print(
        "Approved Items.............. 0"
    )
    print(
        "Executed Items.............. 0"
    )
    print(
        "Historical Predictions Changed False"
    )
    print(
        "Current Weights Changed..... False"
    )
    print(
        f"Saved To.................... "
        f"{OUTPUT_DIR}"
    )
    print("=" * 78)

    return result
