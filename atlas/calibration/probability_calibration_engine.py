
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
CALIBRATION_SEASON = 2025

MIN_ROWS = 150
MIN_POSITIVES = 20
MIN_NEGATIVES = 20
TRAIN_FRACTION = 0.70

STATE_PATH = (
    DATA_DIR
    / "backtest"
    / "weighted_states"
    / str(CALIBRATION_SEASON)
    / "weighted_target_states.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "calibration"
    / "probabilities"
    / str(CALIBRATION_SEASON)
)

REGISTRY_PATH = (
    OUTPUT_DIR
    / "target_probability_calibration_registry.parquet"
)

KNOTS_PATH = (
    OUTPUT_DIR
    / "target_probability_calibration_knots.parquet"
)

HOLDOUT_PATH = (
    OUTPUT_DIR
    / "target_probability_holdout_predictions.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "probability_calibration_metadata.json"
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


def _load_states() -> pd.DataFrame:
    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"Missing weighted states: {STATE_PATH}"
        )

    state = pd.read_parquet(STATE_PATH)

    state["game_date"] = pd.to_datetime(
        state["game_date"],
        errors="raise",
    ).dt.normalize()

    state["actual_outcome"] = pd.to_numeric(
        state["actual_outcome"],
        errors="coerce",
    )

    state["net_weighted_state_score"] = pd.to_numeric(
        state["net_weighted_state_score"],
        errors="coerce",
    )

    state["total_absolute_weight"] = pd.to_numeric(
        state["total_absolute_weight"],
        errors="coerce",
    ).fillna(0.0)

    return state


def _safe_log_loss(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    clipped = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )

    return float(
        log_loss(
            actual,
            clipped,
            labels=[0, 1],
        )
    )


def _chronological_split(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Timestamp,
]:
    ordered = dataframe.sort_values(
        [
            "game_date",
            "game_pk",
            "team",
        ],
        kind="stable",
    ).reset_index(drop=True)

    dates = np.array(
        sorted(
            ordered["game_date"]
            .dropna()
            .unique()
        )
    )

    if len(dates) < 10:
        raise ValueError(
            "Too few unique dates for chronological validation."
        )

    split_index = int(
        np.floor(
            len(dates)
            * TRAIN_FRACTION
        )
    )

    split_index = max(
        1,
        min(
            split_index,
            len(dates) - 1,
        ),
    )

    split_date = pd.Timestamp(
        dates[split_index]
    )

    train = ordered[
        ordered["game_date"] < split_date
    ].copy()

    holdout = ordered[
        ordered["game_date"] >= split_date
    ].copy()

    return train, holdout, split_date


def _fit_model(
    dataframe: pd.DataFrame,
) -> IsotonicRegression:
    model = IsotonicRegression(
        y_min=0.01,
        y_max=0.99,
        increasing=True,
        out_of_bounds="clip",
    )

    model.fit(
        dataframe[
            "net_weighted_state_score"
        ].to_numpy(dtype=float),
        dataframe[
            "actual_outcome"
        ].to_numpy(dtype=float),
        sample_weight=np.maximum(
            1.0,
            dataframe[
                "total_absolute_weight"
            ].to_numpy(dtype=float),
        ),
    )

    return model


def _insufficient_record(
    target: str,
    evidence: pd.DataFrame,
    reason: str,
) -> dict[str, Any]:
    rows = len(evidence)

    positives = int(
        evidence["actual_outcome"].sum()
        if rows
        else 0
    )

    return {
        "target": target,
        "calibration_status": "insufficient_sample",
        "insufficient_reason": reason,
        "evidence_rows": int(rows),
        "positive_outcomes": positives,
        "negative_outcomes": int(rows - positives),
        "outcome_base_rate": (
            float(
                evidence[
                    "actual_outcome"
                ].mean()
            )
            if rows
            else np.nan
        ),
        "train_rows": 0,
        "holdout_rows": 0,
        "split_date": pd.NaT,
        "holdout_base_brier": np.nan,
        "holdout_model_brier": np.nan,
        "holdout_brier_improvement": np.nan,
        "holdout_base_log_loss": np.nan,
        "holdout_model_log_loss": np.nan,
        "holdout_log_loss_improvement": np.nan,
        "holdout_score_rate_spread": np.nan,
        "calibration_accepted": False,
        "final_model_fit": False,
        "2026_outcomes_used": False,
        "engine_version": ENGINE_VERSION,
    }


def _calibrate_target(
    target: str,
    target_rows: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    evidence = target_rows[
        target_rows["state_has_evidence"].eq(True)
        & target_rows["actual_outcome"].notna()
        & target_rows[
            "net_weighted_state_score"
        ].notna()
    ].copy()

    rows = len(evidence)

    positives = int(
        evidence["actual_outcome"].sum()
        if rows
        else 0
    )

    negatives = rows - positives

    if rows < MIN_ROWS:
        return (
            _insufficient_record(
                target,
                evidence,
                "too_few_evidence_rows",
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if positives < MIN_POSITIVES:
        return (
            _insufficient_record(
                target,
                evidence,
                "too_few_positive_outcomes",
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if negatives < MIN_NEGATIVES:
        return (
            _insufficient_record(
                target,
                evidence,
                "too_few_negative_outcomes",
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    if (
        evidence[
            "net_weighted_state_score"
        ].nunique()
        < 4
    ):
        return (
            _insufficient_record(
                target,
                evidence,
                "too_few_unique_scores",
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    train, holdout, split_date = (
        _chronological_split(evidence)
    )

    if (
        len(train) < 75
        or len(holdout) < 40
        or train["actual_outcome"].sum() < 10
        or holdout["actual_outcome"].sum() < 5
    ):
        return (
            _insufficient_record(
                target,
                evidence,
                "chronological_split_too_sparse",
            ),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    model = _fit_model(train)

    holdout_scores = holdout[
        "net_weighted_state_score"
    ].to_numpy(dtype=float)

    actual = holdout[
        "actual_outcome"
    ].to_numpy(dtype=int)

    calibrated_probability = model.predict(
        holdout_scores
    )

    train_base_rate = float(
        train["actual_outcome"].mean()
    )

    base_probability = np.full(
        len(holdout),
        train_base_rate,
        dtype=float,
    )

    base_brier = float(
        brier_score_loss(
            actual,
            base_probability,
        )
    )

    model_brier = float(
        brier_score_loss(
            actual,
            calibrated_probability,
        )
    )

    base_log = _safe_log_loss(
        actual,
        base_probability,
    )

    model_log = _safe_log_loss(
        actual,
        calibrated_probability,
    )

    median_score = float(
        holdout[
            "net_weighted_state_score"
        ].median()
    )

    low = holdout[
        holdout[
            "net_weighted_state_score"
        ] <= median_score
    ]

    high = holdout[
        holdout[
            "net_weighted_state_score"
        ] > median_score
    ]

    low_rate = float(
        low["actual_outcome"].mean()
    )

    high_rate = float(
        high["actual_outcome"].mean()
    )

    spread = high_rate - low_rate

    accepted = bool(
        model_brier < base_brier
        and model_log <= base_log
        and spread > 0
    )

    record = {
        "target": target,
        "calibration_status": (
            "accepted"
            if accepted
            else "holdout_failed"
        ),
        "insufficient_reason": None,
        "evidence_rows": int(rows),
        "positive_outcomes": int(positives),
        "negative_outcomes": int(negatives),
        "outcome_base_rate": float(
            evidence[
                "actual_outcome"
            ].mean()
        ),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "split_date": split_date,
        "holdout_base_brier": base_brier,
        "holdout_model_brier": model_brier,
        "holdout_brier_improvement": (
            base_brier - model_brier
        ),
        "holdout_base_log_loss": base_log,
        "holdout_model_log_loss": model_log,
        "holdout_log_loss_improvement": (
            base_log - model_log
        ),
        "holdout_score_rate_spread": spread,
        "calibration_accepted": accepted,
        "final_model_fit": accepted,
        "2026_outcomes_used": False,
        "engine_version": ENGINE_VERSION,
    }

    holdout_output = holdout[
        [
            "game_pk",
            "game_date",
            "team",
            "opponent",
            "home_away",
            "target",
            "net_weighted_state_score",
            "active_concepts",
            "actual_outcome",
        ]
    ].copy()

    holdout_output[
        "base_probability"
    ] = base_probability

    holdout_output[
        "calibrated_probability"
    ] = calibrated_probability

    holdout_output[
        "engine_version"
    ] = ENGINE_VERSION

    if not accepted:
        return (
            record,
            pd.DataFrame(),
            holdout_output,
        )

    final_model = _fit_model(evidence)

    knots = pd.DataFrame({
        "target": target,
        "score_threshold":
            final_model.X_thresholds_,
        "calibrated_probability":
            final_model.y_thresholds_,
        "calibration_season":
            CALIBRATION_SEASON,
        "engine_version":
            ENGINE_VERSION,
    })

    return record, knots, holdout_output


def run_probability_calibration() -> dict[str, Any]:
    state = _load_states()

    registry_records = []
    knot_frames = []
    holdout_frames = []

    targets = sorted(
        state["target"]
        .dropna()
        .astype(str)
        .unique()
    )

    for target in targets:
        record, knots, holdout = (
            _calibrate_target(
                target=target,
                target_rows=state[
                    state["target"].eq(target)
                ].copy(),
            )
        )

        registry_records.append(record)

        if not knots.empty:
            knot_frames.append(knots)

        if not holdout.empty:
            holdout_frames.append(holdout)

        gain = record[
            "holdout_brier_improvement"
        ]

        gain_text = (
            "nan"
            if pd.isna(gain)
            else f"{gain:.5f}"
        )

        print(
            f"{target:<28} "
            f"status="
            f"{record['calibration_status']:<19} "
            f"rows={record['evidence_rows']:>4,} "
            f"brier_gain={gain_text}"
        )

    registry = pd.DataFrame(
        registry_records
    )

    knots = (
        pd.concat(
            knot_frames,
            ignore_index=True,
        )
        if knot_frames
        else pd.DataFrame(
            columns=[
                "target",
                "score_threshold",
                "calibrated_probability",
                "calibration_season",
                "engine_version",
            ]
        )
    )

    holdout_predictions = (
        pd.concat(
            holdout_frames,
            ignore_index=True,
        )
        if holdout_frames
        else pd.DataFrame()
    )

    duplicate_targets = int(
        registry["target"]
        .duplicated()
        .sum()
    )

    if duplicate_targets:
        raise AssertionError(
            f"Duplicate calibration targets: "
            f"{duplicate_targets}"
        )

    if not knots.empty:
        violations = int(
            knots.sort_values(
                [
                    "target",
                    "score_threshold",
                ]
            )
            .groupby("target")[
                "calibrated_probability"
            ]
            .diff()
            .lt(-1e-12)
            .sum()
        )

        if violations:
            raise AssertionError(
                f"Calibration monotonicity violations: "
                f"{violations}"
            )

    _atomic_parquet_write(
        registry,
        REGISTRY_PATH,
    )

    _atomic_parquet_write(
        knots,
        KNOTS_PATH,
    )

    _atomic_parquet_write(
        holdout_predictions,
        HOLDOUT_PATH,
    )

    accepted = int(
        registry[
            "calibration_accepted"
        ].sum()
    )

    status_counts = (
        registry[
            "calibration_status"
        ].value_counts()
    )

    result = {
        "engine":
            "ATLAS Probability Calibration Core",
        "engine_version":
            ENGINE_VERSION,
        "calibration_season":
            CALIBRATION_SEASON,
        "targets_evaluated":
            int(len(registry)),
        "accepted_targets":
            accepted,
        "holdout_failed_targets":
            int(
                status_counts.get(
                    "holdout_failed",
                    0,
                )
            ),
        "insufficient_sample_targets":
            int(
                status_counts.get(
                    "insufficient_sample",
                    0,
                )
            ),
        "calibration_knots":
            int(len(knots)),
        "holdout_prediction_rows":
            int(len(holdout_predictions)),
        "duplicate_target_rows":
            duplicate_targets,
        "2026_outcomes_used":
            False,
        "outputs": {
            "registry":
                str(REGISTRY_PATH),
            "knots":
                str(KNOTS_PATH),
            "holdout_predictions":
                str(HOLDOUT_PATH),
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("\n" + "=" * 78)
    print("ATLAS PROBABILITY CALIBRATION CORE")
    print("=" * 78)
    print(
        f"Targets Evaluated.......... "
        f"{len(registry):,}"
    )
    print(
        f"Accepted Targets........... "
        f"{accepted:,}"
    )
    print(
        f"Holdout Failed............. "
        f"{result['holdout_failed_targets']:,}"
    )
    print(
        f"Insufficient Sample........ "
        f"{result['insufficient_sample_targets']:,}"
    )
    print(
        f"Calibration Knots.......... "
        f"{len(knots):,}"
    )
    print(
        f"Holdout Prediction Rows.... "
        f"{len(holdout_predictions):,}"
    )
    print(
        "2026 Outcomes Used......... False"
    )
    print("=" * 78)

    return result
