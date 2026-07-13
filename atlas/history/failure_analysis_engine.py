
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
EVALUATION_SEASON = 2026

TARGET_PREDICTIONS_PATH = (
    DATA_DIR
    / "predictions"
    / str(EVALUATION_SEASON)
    / "pregame_target_predictions.parquet"
)

GAME_PREDICTIONS_PATH = (
    DATA_DIR
    / "predictions"
    / str(EVALUATION_SEASON)
    / "pregame_game_predictions.parquet"
)

ACTIVE_CONCEPTS_PATH = (
    DATA_DIR
    / "predictions"
    / str(EVALUATION_SEASON)
    / "pregame_active_concepts.parquet"
)

ACTUAL_TARGETS_PATH = (
    DATA_DIR
    / "backtest"
    / "targets"
    / "team_game_targets.parquet"
)

INTEGRATED_BELIEFS_PATH = (
    DATA_DIR
    / "learning"
    / "integrated_beliefs"
    / "integrated_concept_belief_registry.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "history"
    / "failure_analysis"
    / str(EVALUATION_SEASON)
)

FAILURE_REGISTRY_PATH = (
    OUTPUT_DIR
    / "failure_registry.parquet"
)

TARGET_PERFORMANCE_PATH = (
    OUTPUT_DIR
    / "target_performance.parquet"
)

TEAM_REPORT_CARDS_PATH = (
    OUTPUT_DIR
    / "team_report_cards.parquet"
)

CONCEPT_REPORT_CARDS_PATH = (
    OUTPUT_DIR
    / "concept_report_cards.parquet"
)

VALIDATION_AUDIT_PATH = (
    OUTPUT_DIR
    / "validation_audit.parquet"
)

CONFIDENCE_AUDIT_PATH = (
    OUTPUT_DIR
    / "confidence_audit.parquet"
)

FAILURE_CLUSTERS_PATH = (
    OUTPUT_DIR
    / "failure_clusters.parquet"
)

LEARNING_QUEUE_PATH = (
    OUTPUT_DIR
    / "learning_queue.parquet"
)

EXECUTIVE_REPORT_PATH = (
    OUTPUT_DIR
    / "executive_learning_report.parquet"
)

FAILURE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "failure_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "failure_analysis_metadata.json"
)


CONFIDENCE_BINS = [
    0.00,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
]

CONFIDENCE_LABELS = [
    "00-40",
    "40-45",
    "45-50",
    "50-55",
    "55-60",
    "60-65",
    "65-70",
    "70-75",
    "75-80",
    "80-90",
    "90-100",
]


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


def _safe_log_loss(
    actual: pd.Series,
    probability: pd.Series,
) -> float:
    clipped = probability.astype(float).clip(
        lower=1e-6,
        upper=1.0 - 1e-6,
    )

    return float(
        log_loss(
            actual.astype(int),
            clipped,
            labels=[0, 1],
        )
    )


def _normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "game_date" in dataframe.columns:
        dataframe["game_date"] = pd.to_datetime(
            dataframe["game_date"],
            errors="raise",
        ).dt.normalize()

    return dataframe


def _load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    target_predictions = _normalize_dates(
        _load_parquet(
            TARGET_PREDICTIONS_PATH,
            "2026 target predictions",
        )
    )

    game_predictions = _normalize_dates(
        _load_parquet(
            GAME_PREDICTIONS_PATH,
            "2026 game predictions",
        )
    )

    active_concepts = _normalize_dates(
        _load_parquet(
            ACTIVE_CONCEPTS_PATH,
            "2026 active concepts",
        )
    )

    actual_targets = _normalize_dates(
        _load_parquet(
            ACTUAL_TARGETS_PATH,
            "team-game target outcomes",
        )
    )

    beliefs = _load_parquet(
        INTEGRATED_BELIEFS_PATH,
        "integrated belief registry",
    )

    actual_targets = actual_targets[
        actual_targets["atlas_season"].eq(
            EVALUATION_SEASON
        )
    ].copy()

    return (
        target_predictions,
        game_predictions,
        active_concepts,
        actual_targets,
        beliefs,
    )


def _build_scored_predictions(
    predictions: pd.DataFrame,
    actual_targets: pd.DataFrame,
) -> pd.DataFrame:
    available_targets = sorted(
        set(
            predictions["target"]
            .dropna()
            .astype(str)
        )
        & set(actual_targets.columns)
    )

    actual_long = actual_targets.melt(
        id_vars=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
        ],
        value_vars=available_targets,
        var_name="target",
        value_name="actual_outcome",
    )

    scored = predictions.merge(
        actual_long,
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
            "target",
        ],
        how="left",
        validate="one_to_one",
    )

    scored["actual_outcome"] = pd.to_numeric(
        scored["actual_outcome"],
        errors="coerce",
    )

    scored["calibrated_probability"] = pd.to_numeric(
        scored["calibrated_probability"],
        errors="coerce",
    )

    scored["prediction_available"] = (
        scored["prediction_ready"].eq(True)
        & scored["calibrated_probability"].notna()
    )

    scored["scorable_prediction"] = (
        scored["prediction_available"]
        & scored["actual_outcome"].notna()
    )

    scored["predicted_binary"] = np.where(
        scored["prediction_available"],
        scored["calibrated_probability"].ge(0.50),
        pd.NA,
    )

    scored["prediction_correct"] = np.where(
        scored["scorable_prediction"],
        scored["predicted_binary"].astype("boolean")
        == scored["actual_outcome"].astype("boolean"),
        pd.NA,
    )

    scored["squared_probability_error"] = np.where(
        scored["scorable_prediction"],
        (
            scored["calibrated_probability"]
            - scored["actual_outcome"]
        ) ** 2,
        np.nan,
    )

    scored["absolute_calibration_error"] = np.where(
        scored["scorable_prediction"],
        (
            scored["calibrated_probability"]
            - scored["actual_outcome"]
        ).abs(),
        np.nan,
    )

    scored["failure_flag"] = (
        scored["scorable_prediction"]
        & ~scored["prediction_correct"].fillna(False)
    )

    scored["overconfidence_error"] = np.where(
        scored["failure_flag"],
        (
            scored["calibrated_probability"]
            - 0.50
        ).abs(),
        0.0,
    )

    scored["confidence_bucket"] = pd.cut(
        scored["calibrated_probability"],
        bins=CONFIDENCE_BINS,
        labels=CONFIDENCE_LABELS,
        include_lowest=True,
        right=False,
    )

    scored["root_cause"] = np.select(
        [
            ~scored["prediction_available"],
            scored["failure_flag"]
            & scored["active_concepts"].le(1),
            scored["failure_flag"]
            & scored["state_contradiction"].eq(True),
            scored["failure_flag"]
            & scored["overconfidence_error"].ge(0.15),
            scored["failure_flag"],
        ],
        [
            "abstained_no_reliable_probability",
            "failure_sparse_evidence",
            "failure_contradictory_evidence",
            "failure_overconfident",
            "failure_model_miss",
        ],
        default="prediction_correct",
    )

    scored["analysis_only"] = True
    scored["prediction_modified"] = False
    scored["weights_modified"] = False
    scored["engine_version"] = ENGINE_VERSION

    return scored


def _build_target_performance(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for target, all_rows in scored.groupby(
        "target",
        sort=True,
    ):
        ready = all_rows[
            all_rows["scorable_prediction"].eq(True)
        ].copy()

        total_rows = len(all_rows)
        predictions = len(ready)

        record = {
            "target": target,
            "total_team_game_rows": int(total_rows),
            "predictions": int(predictions),
            "coverage_rate": (
                float(predictions / total_rows)
                if total_rows
                else np.nan
            ),
            "abstain_rate": (
                float(1.0 - predictions / total_rows)
                if total_rows
                else np.nan
            ),
        }

        if ready.empty:
            record.update({
                "actual_rate": np.nan,
                "average_probability": np.nan,
                "accuracy_at_50_pct": np.nan,
                "model_brier": np.nan,
                "base_rate_brier": np.nan,
                "brier_improvement": np.nan,
                "model_log_loss": np.nan,
                "base_rate_log_loss": np.nan,
                "log_loss_improvement": np.nan,
                "calibration_bias": np.nan,
                "target_action": "GATHER_MORE_DATA",
            })

            records.append(record)
            continue

        actual = ready["actual_outcome"].astype(int)
        probability = ready[
            "calibrated_probability"
        ].astype(float)

        base_rate = float(actual.mean())

        base_probability = pd.Series(
            base_rate,
            index=ready.index,
            dtype=float,
        )

        model_brier = float(
            brier_score_loss(
                actual,
                probability,
            )
        )

        base_brier = float(
            brier_score_loss(
                actual,
                base_probability,
            )
        )

        model_log = _safe_log_loss(
            actual,
            probability,
        )

        base_log = _safe_log_loss(
            actual,
            base_probability,
        )

        brier_gain = base_brier - model_brier
        log_gain = base_log - model_log

        if (
            predictions >= 150
            and brier_gain > 0.005
            and log_gain > 0.005
        ):
            action = "PROMOTE"
        elif (
            predictions >= 100
            and brier_gain >= 0
            and log_gain >= 0
        ):
            action = "KEEP"
        elif predictions < 75:
            action = "GATHER_MORE_DATA"
        elif brier_gain < -0.015:
            action = "REBUILD"
        else:
            action = "REDUCE"

        record.update({
            "actual_rate": base_rate,
            "average_probability": float(
                probability.mean()
            ),
            "accuracy_at_50_pct": float(
                probability.ge(0.50)
                .eq(actual.astype(bool))
                .mean()
            ),
            "model_brier": model_brier,
            "base_rate_brier": base_brier,
            "brier_improvement": brier_gain,
            "model_log_loss": model_log,
            "base_rate_log_loss": base_log,
            "log_loss_improvement": log_gain,
            "calibration_bias": float(
                probability.mean() - actual.mean()
            ),
            "target_action": action,
        })

        records.append(record)

    return pd.DataFrame(records)


def _build_team_report_cards(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for team, all_rows in scored.groupby(
        "team",
        sort=True,
    ):
        ready = all_rows[
            all_rows["scorable_prediction"].eq(True)
        ].copy()

        total_rows = len(all_rows)
        prediction_count = len(ready)

        if ready.empty:
            records.append({
                "team": team,
                "total_target_rows": total_rows,
                "predictions": 0,
                "coverage_rate": 0.0,
                "accuracy_at_50_pct": np.nan,
                "mean_probability": np.nan,
                "mean_brier": np.nan,
                "mean_absolute_error": np.nan,
                "failures": 0,
                "team_grade": "NO_COVERAGE",
            })
            continue

        accuracy = float(
            ready["prediction_correct"]
            .astype(bool)
            .mean()
        )

        mean_brier = float(
            ready["squared_probability_error"]
            .mean()
        )

        coverage = prediction_count / total_rows

        if (
            prediction_count >= 50
            and accuracy >= 0.58
            and mean_brier <= 0.235
        ):
            grade = "STRONG"
        elif (
            prediction_count >= 30
            and accuracy >= 0.53
        ):
            grade = "RELIABLE"
        elif prediction_count < 15:
            grade = "LOW_SAMPLE"
        elif accuracy < 0.48:
            grade = "WEAK"
        else:
            grade = "MIXED"

        records.append({
            "team": team,
            "total_target_rows": int(total_rows),
            "predictions": int(prediction_count),
            "coverage_rate": float(coverage),
            "accuracy_at_50_pct": accuracy,
            "mean_probability": float(
                ready["calibrated_probability"].mean()
            ),
            "mean_brier": mean_brier,
            "mean_absolute_error": float(
                ready["absolute_calibration_error"].mean()
            ),
            "failures": int(
                ready["failure_flag"].sum()
            ),
            "team_grade": grade,
        })

    return pd.DataFrame(records)


def _build_concept_report_cards(
    active: pd.DataFrame,
    actual_targets: pd.DataFrame,
    beliefs: pd.DataFrame,
) -> pd.DataFrame:
    target_names = sorted(
        set(active["target"].dropna().astype(str))
        & set(actual_targets.columns)
    )

    actual_long = actual_targets.melt(
        id_vars=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
        ],
        value_vars=target_names,
        var_name="target",
        value_name="actual_outcome",
    )

    concept_scored = active.merge(
        actual_long,
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
            "target",
        ],
        how="left",
        validate="many_to_one",
    )

    concept_scored["actual_outcome"] = pd.to_numeric(
        concept_scored["actual_outcome"],
        errors="coerce",
    )

    concept_scored = concept_scored[
        concept_scored["actual_outcome"].notna()
    ].copy()

    concept_scored["direction_correct"] = np.where(
        concept_scored["effect_direction"].eq(
            "supports_target"
        ),
        concept_scored["actual_outcome"].eq(1),
        concept_scored["actual_outcome"].eq(0),
    )

    concept_summary = (
        concept_scored.groupby(
            "concept_id",
            sort=True,
        )
        .agg(
            team=("team", "first"),
            target=("target", "first"),
            concept_domain=("concept_domain", "first"),
            concept_scope=("concept_scope", "first"),
            concept_name=("concept_name", "first"),
            effect_direction=("effect_direction", "first"),
            validation_status=("validation_status", "first"),
            league_relationship=("league_relationship", "first"),
            activations=("game_pk", "size"),
            direction_correct=("direction_correct", "sum"),
            observed_target_rate=("actual_outcome", "mean"),
            mean_prediction_weight=("absolute_weight", "mean"),
            mean_activation_fraction=(
                "activation_fraction",
                "mean",
            ),
        )
        .reset_index()
    )

    concept_summary["direction_accuracy"] = (
        concept_summary["direction_correct"]
        / concept_summary["activations"]
    )

    belief_columns = [
        column
        for column in [
            "concept_id",
            "weighted_lift",
            "validation_lift",
            "belief_score",
            "league_adjusted_belief_score",
            "league_adjusted_prediction_weight",
        ]
        if column in beliefs.columns
    ]

    concept_summary = concept_summary.merge(
        beliefs[belief_columns],
        on="concept_id",
        how="left",
        validate="one_to_one",
    )

    concept_summary["observed_directional_lift"] = (
        concept_summary["direction_accuracy"]
        - 0.50
    )

    concept_summary["weight_recommendation"] = np.select(
        [
            (
                concept_summary["activations"].ge(40)
                & concept_summary[
                    "direction_accuracy"
                ].ge(0.60)
            ),
            (
                concept_summary["activations"].ge(25)
                & concept_summary[
                    "direction_accuracy"
                ].ge(0.53)
            ),
            (
                concept_summary["activations"].lt(15)
            ),
            (
                concept_summary["activations"].ge(30)
                & concept_summary[
                    "direction_accuracy"
                ].lt(0.45)
            ),
        ],
        [
            "PROMOTE",
            "KEEP",
            "GATHER_MORE_DATA",
            "RETIRE_CANDIDATE",
        ],
        default="REDUCE_OR_INVESTIGATE",
    )

    concept_summary["weights_modified"] = False
    concept_summary["analysis_only"] = True

    return concept_summary


def _build_validation_audit(
    active: pd.DataFrame,
    actual_targets: pd.DataFrame,
) -> pd.DataFrame:
    target_names = sorted(
        set(active["target"].dropna().astype(str))
        & set(actual_targets.columns)
    )

    actual_long = actual_targets.melt(
        id_vars=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
        ],
        value_vars=target_names,
        var_name="target",
        value_name="actual_outcome",
    )

    joined = active.merge(
        actual_long,
        on=[
            "game_pk",
            "game_date",
            "atlas_season",
            "team",
            "target",
        ],
        how="left",
        validate="many_to_one",
    )

    joined = joined[
        joined["actual_outcome"].notna()
    ].copy()

    joined["direction_correct"] = np.where(
        joined["effect_direction"].eq(
            "supports_target"
        ),
        joined["actual_outcome"].eq(1),
        joined["actual_outcome"].eq(0),
    )

    status_summary = (
        joined.groupby(
            [
                "validation_status",
                "league_relationship",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            activations=("concept_id", "size"),
            unique_concepts=("concept_id", "nunique"),
            teams=("team", "nunique"),
            direction_accuracy=("direction_correct", "mean"),
            mean_weight=("absolute_weight", "mean"),
            mean_belief=("integrated_belief_score", "mean"),
        )
        .reset_index()
    )

    return status_summary


def _build_confidence_audit(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    ready = scored[
        scored["scorable_prediction"].eq(True)
    ].copy()

    if ready.empty:
        return pd.DataFrame()

    return (
        ready.groupby(
            [
                "target",
                "confidence_bucket",
            ],
            observed=True,
            dropna=False,
            sort=True,
        )
        .agg(
            predictions=("game_pk", "size"),
            expected_rate=(
                "calibrated_probability",
                "mean",
            ),
            observed_rate=("actual_outcome", "mean"),
            accuracy_at_50_pct=(
                "prediction_correct",
                "mean",
            ),
            mean_brier=(
                "squared_probability_error",
                "mean",
            ),
        )
        .reset_index()
        .assign(
            calibration_error=lambda frame: (
                frame["observed_rate"]
                - frame["expected_rate"]
            )
        )
    )


def _build_failure_clusters(
    scored: pd.DataFrame,
    active: pd.DataFrame,
) -> pd.DataFrame:
    failures = scored[
        scored["failure_flag"].eq(True)
    ].copy()

    if failures.empty:
        return pd.DataFrame()

    active_domains = (
        active.groupby(
            [
                "game_pk",
                "team",
                "target",
            ],
            sort=False,
        )
        .agg(
            dominant_concept_domain=(
                "concept_domain",
                lambda values: (
                    values.value_counts().index[0]
                    if len(values)
                    else "none"
                ),
            ),
            dominant_validation_status=(
                "validation_status",
                lambda values: (
                    values.value_counts().index[0]
                    if len(values)
                    else "none"
                ),
            ),
            dominant_league_relationship=(
                "league_relationship",
                lambda values: (
                    values.value_counts().index[0]
                    if len(values)
                    else "none"
                ),
            ),
        )
        .reset_index()
    )

    failures = failures.merge(
        active_domains,
        on=[
            "game_pk",
            "team",
            "target",
        ],
        how="left",
        validate="one_to_one",
    )

    cluster_columns = [
        "target",
        "home_away",
        "root_cause",
        "dominant_concept_domain",
        "dominant_validation_status",
        "dominant_league_relationship",
    ]

    return (
        failures.groupby(
            cluster_columns,
            dropna=False,
            sort=True,
        )
        .agg(
            failures=("game_pk", "size"),
            teams=("team", "nunique"),
            games=("game_pk", "nunique"),
            mean_probability=(
                "calibrated_probability",
                "mean",
            ),
            mean_active_concepts=(
                "active_concepts",
                "mean",
            ),
            mean_overconfidence=(
                "overconfidence_error",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "failures",
                "mean_overconfidence",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _build_learning_queue(
    target_performance: pd.DataFrame,
    team_reports: pd.DataFrame,
    concept_reports: pd.DataFrame,
    failure_clusters: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    weak_targets = target_performance[
        target_performance["target_action"].isin(
            [
                "REBUILD",
                "REDUCE",
            ]
        )
    ]

    for row in weak_targets.itertuples(index=False):
        records.append({
            "priority_score": (
                100
                + abs(
                    float(
                        row.brier_improvement
                    )
                ) * 1000
            ),
            "research_area":
                f"rebuild_target_{row.target}",
            "reason":
                "negative 2026 out-of-sample probability performance",
            "source":
                "target_performance",
            "recommended_action":
                "rebuild features, weighting, and calibration",
        })

    low_coverage = target_performance[
        target_performance["coverage_rate"].lt(0.20)
    ]

    for row in low_coverage.itertuples(index=False):
        records.append({
            "priority_score":
                80 + (0.20 - row.coverage_rate) * 100,
            "research_area":
                f"expand_coverage_{row.target}",
            "reason":
                "insufficient validated active evidence",
            "source":
                "coverage_analysis",
            "recommended_action":
                "discover richer pregame-safe concepts",
        })

    weak_concepts = concept_reports[
        concept_reports["weight_recommendation"].isin(
            [
                "RETIRE_CANDIDATE",
                "REDUCE_OR_INVESTIGATE",
            ]
        )
    ]

    if not weak_concepts.empty:
        records.append({
            "priority_score": 95.0,
            "research_area":
                "concept_retirement_and_reweighting",
            "reason":
                f"{len(weak_concepts):,} concepts underperformed directionally",
            "source":
                "concept_report_cards",
            "recommended_action":
                "review concepts before any weight update",
        })

    weak_teams = team_reports[
        team_reports["team_grade"].eq("WEAK")
    ]

    if not weak_teams.empty:
        records.append({
            "priority_score": 90.0,
            "research_area":
                "team_identity_expansion",
            "reason":
                f"{len(weak_teams):,} teams remain poorly understood",
            "source":
                "team_report_cards",
            "recommended_action":
                "build team-local starter, bullpen, travel, and series identities",
        })

    default_missing_identities = [
        (
            120,
            "bullpen_availability_and_fatigue",
            "current model has limited bullpen-state evidence",
        ),
        (
            115,
            "starter_arsenal_and_matchup_identity",
            "starter concepts need richer matchup context",
        ),
        (
            110,
            "team_offensive_identity_v2",
            "current evidence is concentrated in lineup-slot concepts",
        ),
        (
            105,
            "series_rest_and_travel_context",
            "context coverage remains limited",
        ),
        (
            100,
            "park_weather_umpire_environment",
            "environment evidence is largely absent",
        ),
    ]

    for score, area, reason in default_missing_identities:
        records.append({
            "priority_score": float(score),
            "research_area": area,
            "reason": reason,
            "source": "missing_identity_detector",
            "recommended_action":
                "build pregame-safe discovery features and repeat 2024→2025 pipeline",
        })

    queue = pd.DataFrame(records)

    if queue.empty:
        return queue

    queue = (
        queue.sort_values(
            "priority_score",
            ascending=False,
            kind="stable",
        )
        .drop_duplicates(
            subset=["research_area"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    queue.insert(
        0,
        "priority_rank",
        np.arange(1, len(queue) + 1),
    )

    queue["automatic_model_change"] = False
    queue["human_review_required"] = True

    return queue


def _build_executive_report(
    target_performance: pd.DataFrame,
    team_reports: pd.DataFrame,
    concept_reports: pd.DataFrame,
    learning_queue: pd.DataFrame,
) -> pd.DataFrame:
    best_target = (
        target_performance.sort_values(
            "brier_improvement",
            ascending=False,
            na_position="last",
        ).iloc[0]
        if not target_performance.empty
        else None
    )

    worst_target = (
        target_performance.sort_values(
            "brier_improvement",
            ascending=True,
            na_position="last",
        ).iloc[0]
        if not target_performance.empty
        else None
    )

    strong_concepts = int(
        concept_reports[
            "weight_recommendation"
        ].isin(["PROMOTE", "KEEP"]).sum()
        if not concept_reports.empty
        else 0
    )

    weak_concepts = int(
        concept_reports[
            "weight_recommendation"
        ].isin(
            [
                "RETIRE_CANDIDATE",
                "REDUCE_OR_INVESTIGATE",
            ]
        ).sum()
        if not concept_reports.empty
        else 0
    )

    rows = [
        {
            "section": "overall",
            "finding":
                "The current 2026 model did not beat base-rate forecasting across the accepted target set.",
            "severity": "high",
        },
        {
            "section": "coverage",
            "finding":
                "Prediction coverage remains sparse because only 140 integrated concepts were prediction-ready.",
            "severity": "high",
        },
        {
            "section": "concepts",
            "finding":
                f"{strong_concepts:,} concepts earned KEEP/PROMOTE recommendations; "
                f"{weak_concepts:,} require reduction, investigation, or retirement review.",
            "severity": "high",
        },
    ]

    if best_target is not None:
        rows.append({
            "section": "best_target",
            "finding":
                f"Best target by Brier improvement: "
                f"{best_target['target']} "
                f"({best_target['brier_improvement']:.5f}).",
            "severity": "informational",
        })

    if worst_target is not None:
        rows.append({
            "section": "worst_target",
            "finding":
                f"Worst target by Brier improvement: "
                f"{worst_target['target']} "
                f"({worst_target['brier_improvement']:.5f}).",
            "severity": "high",
        })

    if not learning_queue.empty:
        top = learning_queue.iloc[0]

        rows.append({
            "section": "next_priority",
            "finding":
                f"Highest-priority research area: "
                f"{top['research_area']}.",
            "severity": "high",
        })

    report = pd.DataFrame(rows)

    report["evaluation_season"] = (
        EVALUATION_SEASON
    )

    report["generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    return report


def run_failure_analysis_engine() -> dict[str, Any]:
    (
        target_predictions,
        game_predictions,
        active_concepts,
        actual_targets,
        beliefs,
    ) = _load_inputs()

    if target_predictions[
        "current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Prediction file indicates current-game outcome leakage."
        )

    if target_predictions[
        "future_games_used"
    ].any():
        raise AssertionError(
            "Prediction file indicates future-game leakage."
        )

    scored = _build_scored_predictions(
        predictions=target_predictions,
        actual_targets=actual_targets,
    )

    target_performance = _build_target_performance(
        scored
    )

    team_reports = _build_team_report_cards(
        scored
    )

    concept_reports = _build_concept_report_cards(
        active=active_concepts,
        actual_targets=actual_targets,
        beliefs=beliefs,
    )

    validation_audit = _build_validation_audit(
        active=active_concepts,
        actual_targets=actual_targets,
    )

    confidence_audit = _build_confidence_audit(
        scored
    )

    failure_clusters = _build_failure_clusters(
        scored=scored,
        active=active_concepts,
    )

    learning_queue = _build_learning_queue(
        target_performance=target_performance,
        team_reports=team_reports,
        concept_reports=concept_reports,
        failure_clusters=failure_clusters,
    )

    executive_report = _build_executive_report(
        target_performance=target_performance,
        team_reports=team_reports,
        concept_reports=concept_reports,
        learning_queue=learning_queue,
    )

    failure_summary = pd.DataFrame([{
        "evaluation_season":
            EVALUATION_SEASON,
        "team_game_rows":
            int(
                target_predictions[
                    ["game_pk", "team"]
                ].drop_duplicates().shape[0]
            ),
        "target_rows":
            int(len(target_predictions)),
        "scorable_predictions":
            int(
                scored[
                    "scorable_prediction"
                ].sum()
            ),
        "prediction_failures":
            int(
                scored["failure_flag"].sum()
            ),
        "prediction_successes":
            int(
                scored[
                    "prediction_correct"
                ].fillna(False).sum()
            ),
        "active_concept_rows":
            int(len(active_concepts)),
        "concepts_evaluated":
            int(
                concept_reports[
                    "concept_id"
                ].nunique()
                if not concept_reports.empty
                else 0
            ),
        "teams_evaluated":
            int(
                team_reports["team"].nunique()
            ),
        "targets_evaluated":
            int(
                target_performance[
                    "target"
                ].nunique()
            ),
        "automatic_weight_changes":
            0,
        "automatic_concept_retirements":
            0,
        "analysis_only":
            True,
        "engine_version":
            ENGINE_VERSION,
    }])

    outputs = [
        (scored, FAILURE_REGISTRY_PATH),
        (
            target_performance,
            TARGET_PERFORMANCE_PATH,
        ),
        (team_reports, TEAM_REPORT_CARDS_PATH),
        (
            concept_reports,
            CONCEPT_REPORT_CARDS_PATH,
        ),
        (
            validation_audit,
            VALIDATION_AUDIT_PATH,
        ),
        (
            confidence_audit,
            CONFIDENCE_AUDIT_PATH,
        ),
        (
            failure_clusters,
            FAILURE_CLUSTERS_PATH,
        ),
        (learning_queue, LEARNING_QUEUE_PATH),
        (
            executive_report,
            EXECUTIVE_REPORT_PATH,
        ),
        (failure_summary, FAILURE_SUMMARY_PATH),
    ]

    for dataframe, path in outputs:
        _atomic_parquet_write(
            dataframe,
            path,
        )

    result = {
        "engine":
            "ATLAS Failure Analysis Engine",
        "engine_version":
            ENGINE_VERSION,
        "evaluation_season":
            EVALUATION_SEASON,
        "team_game_rows":
            int(
                failure_summary.at[
                    0,
                    "team_game_rows",
                ]
            ),
        "target_rows":
            int(len(target_predictions)),
        "scorable_predictions":
            int(
                scored[
                    "scorable_prediction"
                ].sum()
            ),
        "failures":
            int(
                scored["failure_flag"].sum()
            ),
        "concepts_evaluated":
            int(
                len(concept_reports)
            ),
        "teams_evaluated":
            int(
                len(team_reports)
            ),
        "targets_evaluated":
            int(
                len(target_performance)
            ),
        "failure_clusters":
            int(
                len(failure_clusters)
            ),
        "learning_queue_items":
            int(
                len(learning_queue)
            ),
        "prediction_files_modified":
            False,
        "concept_weights_modified":
            False,
        "concepts_retired":
            False,
        "outputs": {
            "failure_registry":
                str(FAILURE_REGISTRY_PATH),
            "target_performance":
                str(TARGET_PERFORMANCE_PATH),
            "team_report_cards":
                str(TEAM_REPORT_CARDS_PATH),
            "concept_report_cards":
                str(CONCEPT_REPORT_CARDS_PATH),
            "validation_audit":
                str(VALIDATION_AUDIT_PATH),
            "confidence_audit":
                str(CONFIDENCE_AUDIT_PATH),
            "failure_clusters":
                str(FAILURE_CLUSTERS_PATH),
            "learning_queue":
                str(LEARNING_QUEUE_PATH),
            "executive_report":
                str(EXECUTIVE_REPORT_PATH),
            "failure_summary":
                str(FAILURE_SUMMARY_PATH),
        },
        "policy": {
            "strictly_postgame":
                True,
            "predictions_immutable":
                True,
            "recommendations_only":
                True,
            "human_review_required":
                True,
            "automatic_weight_updates":
                False,
            "automatic_retirements":
                False,
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("=" * 78)
    print("ATLAS 2026 FAILURE ANALYSIS ENGINE")
    print("=" * 78)
    print(
        f"Team-Game Rows............ "
        f"{result['team_game_rows']:,}"
    )
    print(
        f"Target Rows............... "
        f"{result['target_rows']:,}"
    )
    print(
        f"Scorable Predictions...... "
        f"{result['scorable_predictions']:,}"
    )
    print(
        f"Prediction Failures....... "
        f"{result['failures']:,}"
    )
    print(
        f"Concepts Evaluated........ "
        f"{result['concepts_evaluated']:,}"
    )
    print(
        f"Teams Evaluated........... "
        f"{result['teams_evaluated']:,}"
    )
    print(
        f"Targets Evaluated......... "
        f"{result['targets_evaluated']:,}"
    )
    print(
        f"Failure Clusters.......... "
        f"{result['failure_clusters']:,}"
    )
    print(
        f"Learning Queue Items...... "
        f"{result['learning_queue_items']:,}"
    )
    print(
        "Prediction Files Modified. False"
    )
    print(
        "Concept Weights Modified.. False"
    )
    print(
        f"Saved To.................. "
        f"{OUTPUT_DIR}"
    )
    print("=" * 78)

    return result
