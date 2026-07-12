
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
PREDICTION_SEASON = 2026

PREDICTION_ROOT = (
    DATA_DIR
    / "predictions"
    / str(PREDICTION_SEASON)
)

FUSION_ROOT = (
    DATA_DIR
    / "predictions"
    / str(PREDICTION_SEASON)
    / "fusion"
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


def _load_run(
    run_directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    game_path = (
        run_directory
        / "game_predictions.parquet"
    )

    target_path = (
        run_directory
        / "target_predictions.parquet"
    )

    active_path = (
        run_directory
        / "active_concepts.parquet"
    )

    for path, label in [
        (game_path, "game predictions"),
        (target_path, "target predictions"),
        (active_path, "active concepts"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    return (
        pd.read_parquet(game_path),
        pd.read_parquet(target_path),
        pd.read_parquet(active_path),
    )


def _safe_probability(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(
            np.nan,
            index=dataframe.index,
            dtype="float64",
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).clip(
        lower=0.0,
        upper=1.0,
    )


def _weighted_available_mean(
    values: list[tuple[float, float]],
) -> tuple[float, float]:
    valid = [
        (float(value), float(weight))
        for value, weight in values
        if pd.notna(value)
        and pd.notna(weight)
        and weight > 0
    ]

    if not valid:
        return np.nan, 0.0

    numerator = sum(
        value * weight
        for value, weight in valid
    )

    denominator = sum(
        weight
        for _, weight in valid
    )

    return (
        float(numerator / denominator),
        float(denominator),
    )


def _coverage_grade(
    available_targets: int,
    active_concepts: int,
) -> str:
    if (
        available_targets >= 6
        and active_concepts >= 5
    ):
        return "excellent"

    if (
        available_targets >= 4
        and active_concepts >= 3
    ):
        return "good"

    if (
        available_targets >= 2
        and active_concepts >= 2
    ):
        return "limited"

    if available_targets >= 1:
        return "minimal"

    return "none"


def _confidence_grade(
    score: float,
) -> str:
    if score >= 85:
        return "high"

    if score >= 70:
        return "moderate_high"

    if score >= 55:
        return "moderate"

    if score >= 40:
        return "low"

    return "insufficient"


def _team_outlook(
    win_probability: float,
    offense_probability: float,
    prevention_probability: float,
) -> str:
    if pd.isna(win_probability):
        return "insufficient_moneyline_evidence"

    if win_probability >= 0.62:
        strength = "strong_favorite"
    elif win_probability >= 0.55:
        strength = "lean_favorite"
    elif win_probability <= 0.38:
        strength = "strong_underdog"
    elif win_probability <= 0.45:
        strength = "lean_underdog"
    else:
        strength = "near_coin_flip"

    parts = [strength]

    if pd.notna(offense_probability):
        if offense_probability >= 0.55:
            parts.append("offense_positive")
        elif offense_probability <= 0.35:
            parts.append("offense_limited")

    if pd.notna(prevention_probability):
        if prevention_probability >= 0.55:
            parts.append("run_prevention_positive")
        elif prevention_probability <= 0.35:
            parts.append("run_prevention_concern")

    return "|".join(parts)


def _build_reason_table(
    active: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    if active.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "team",
                "top_reasons",
            ]
        )

    active = active.copy()

    active["reason_text"] = (
        active["target"].astype(str)
        + ":"
        + active["concept_name"].astype(str)
        + ":"
        + active["effect_direction"].astype(str)
        + ":w="
        + active["absolute_weight"]
        .round(3)
        .astype(str)
    )

    records = []

    for (
        game_pk,
        team,
    ), group in active.groupby(
        [
            "game_pk",
            "team",
        ],
        sort=False,
    ):
        ordered = group.sort_values(
            [
                "absolute_weight",
                "integrated_belief_score",
            ],
            ascending=[
                False,
                False,
            ],
            kind="stable",
        )

        reasons = (
            ordered["reason_text"]
            .head(top_n)
            .tolist()
        )

        records.append({
            "game_pk": game_pk,
            "team": team,
            "top_reasons": " | ".join(reasons),
            "top_reason_count": len(reasons),
        })

    return pd.DataFrame(records)


def _build_fused_team_predictions(
    games: pd.DataFrame,
    targets: pd.DataFrame,
    active: pd.DataFrame,
) -> pd.DataFrame:
    fused = games.copy()

    fused[
        "moneyline_probability"
    ] = _safe_probability(
        fused,
        "reconciled_win_probability",
    )

    score_5 = _safe_probability(
        fused,
        "team_scored_5_plus_probability",
    )

    score_3_or_less = _safe_probability(
        fused,
        "team_scored_3_or_less_probability",
    )

    score_8 = _safe_probability(
        fused,
        "team_scored_8_plus_probability",
    )

    allow_3_or_less = _safe_probability(
        fused,
        "team_allowed_3_or_less_probability",
    )

    allow_5 = _safe_probability(
        fused,
        "team_allowed_5_plus_probability",
    )

    over_10_5 = _safe_probability(
        fused,
        "game_total_10_5_plus_probability",
    )

    over_12 = _safe_probability(
        fused,
        "game_total_12_plus_probability",
    )

    offense_scores = []
    prevention_scores = []
    total_environment_scores = []
    available_target_counts = []

    for index in fused.index:
        offense, _ = _weighted_available_mean([
            (score_5.loc[index], 1.00),
            (
                1.0 - score_3_or_less.loc[index]
                if pd.notna(
                    score_3_or_less.loc[index]
                )
                else np.nan,
                0.85,
            ),
            (score_8.loc[index], 0.55),
        ])

        prevention, _ = _weighted_available_mean([
            (allow_3_or_less.loc[index], 1.00),
            (
                1.0 - allow_5.loc[index]
                if pd.notna(
                    allow_5.loc[index]
                )
                else np.nan,
                0.90,
            ),
        ])

        total_environment, _ = (
            _weighted_available_mean([
                (over_10_5.loc[index], 1.00),
                (over_12.loc[index], 0.75),
            ])
        )

        probability_values = [
            fused.at[
                index,
                "moneyline_probability",
            ],
            score_5.loc[index],
            score_3_or_less.loc[index],
            score_8.loc[index],
            allow_3_or_less.loc[index],
            allow_5.loc[index],
            over_10_5.loc[index],
            over_12.loc[index],
        ]

        available_target_counts.append(
            int(
                sum(
                    pd.notna(value)
                    for value in probability_values
                )
            )
        )

        offense_scores.append(offense)
        prevention_scores.append(prevention)
        total_environment_scores.append(
            total_environment
        )

    fused[
        "offensive_outlook_score"
    ] = offense_scores

    fused[
        "run_prevention_outlook_score"
    ] = prevention_scores

    fused[
        "high_total_environment_score"
    ] = total_environment_scores

    fused[
        "available_probability_targets"
    ] = available_target_counts

    # --------------------------------------------------------
    # Cross-target contradiction checks
    # --------------------------------------------------------
    contradiction_count = []

    for index in fused.index:
        contradictions = 0

        win_probability = fused.at[
            index,
            "moneyline_probability",
        ]

        offense = fused.at[
            index,
            "offensive_outlook_score",
        ]

        prevention = fused.at[
            index,
            "run_prevention_outlook_score",
        ]

        high_total = fused.at[
            index,
            "high_total_environment_score",
        ]

        if (
            pd.notna(win_probability)
            and pd.notna(offense)
            and pd.notna(prevention)
            and win_probability >= 0.60
            and offense <= 0.40
            and prevention <= 0.40
        ):
            contradictions += 1

        if (
            pd.notna(score_5.loc[index])
            and pd.notna(
                score_3_or_less.loc[index]
            )
            and score_5.loc[index] >= 0.55
            and score_3_or_less.loc[index] >= 0.55
        ):
            contradictions += 1

        if (
            pd.notna(allow_3_or_less.loc[index])
            and pd.notna(allow_5.loc[index])
            and allow_3_or_less.loc[index] >= 0.55
            and allow_5.loc[index] >= 0.55
        ):
            contradictions += 1

        if (
            pd.notna(high_total)
            and pd.notna(offense)
            and pd.notna(prevention)
            and high_total >= 0.55
            and offense <= 0.35
            and prevention >= 0.60
        ):
            contradictions += 1

        contradiction_count.append(
            contradictions
        )

    fused[
        "fusion_contradiction_count"
    ] = contradiction_count

    active_counts = (
        active.groupby(
            [
                "game_pk",
                "team",
            ],
            sort=False,
        )
        .agg(
            unique_active_concepts=(
                "concept_id",
                "nunique",
            ),
            strong_validation_concepts=(
                "validation_status",
                lambda values: int(
                    values.isin(
                        [
                            "validated",
                            "validated_strong",
                        ]
                    ).sum()
                ),
            ),
            league_reinforced_concepts=(
                "league_relationship",
                lambda values: int(
                    values.eq(
                        "reinforced_by_league"
                    ).sum()
                ),
            ),
            mean_active_belief=(
                "integrated_belief_score",
                "mean",
            ),
            total_active_weight=(
                "absolute_weight",
                "sum",
            ),
        )
        .reset_index()
        if not active.empty
        else pd.DataFrame(
            columns=[
                "game_pk",
                "team",
                "unique_active_concepts",
                "strong_validation_concepts",
                "league_reinforced_concepts",
                "mean_active_belief",
                "total_active_weight",
            ]
        )
    )

    fused = fused.merge(
        active_counts,
        on=[
            "game_pk",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    integer_columns = [
        "unique_active_concepts",
        "strong_validation_concepts",
        "league_reinforced_concepts",
    ]

    for column in integer_columns:
        fused[column] = (
            fused[column]
            .fillna(0)
            .astype("int64")
        )

    fused[
        "mean_active_belief"
    ] = pd.to_numeric(
        fused["mean_active_belief"],
        errors="coerce",
    )

    fused[
        "total_active_weight"
    ] = pd.to_numeric(
        fused["total_active_weight"],
        errors="coerce",
    ).fillna(0.0)

    reasons = _build_reason_table(
        active=active,
        top_n=5,
    )

    fused = fused.merge(
        reasons,
        on=[
            "game_pk",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    fused["top_reasons"] = (
        fused["top_reasons"]
        .fillna("")
    )

    fused["top_reason_count"] = (
        fused["top_reason_count"]
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # Coverage and decision-quality score
    # --------------------------------------------------------
    fused["coverage_grade"] = [
        _coverage_grade(
            available_targets=int(
                row.available_probability_targets
            ),
            active_concepts=int(
                row.unique_active_concepts
            ),
        )
        for row in fused.itertuples(
            index=False
        )
    ]

    target_coverage_component = (
        fused[
            "available_probability_targets"
        ]
        .div(8.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    concept_coverage_component = (
        fused[
            "unique_active_concepts"
        ]
        .div(6.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    validation_component = (
        fused[
            "strong_validation_concepts"
        ]
        .div(3.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    league_component = (
        fused[
            "league_reinforced_concepts"
        ]
        .div(2.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    belief_component = (
        fused[
            "mean_active_belief"
        ]
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    contradiction_penalty = (
        fused[
            "fusion_contradiction_count"
        ]
        .mul(0.15)
        .clip(
            lower=0.0,
            upper=0.45,
        )
    )

    moneyline_bonus = (
        fused[
            "moneyline_probability"
        ]
        .notna()
        .astype(float)
        * 0.10
    )

    decision_score = (
        0.28 * target_coverage_component
        + 0.24 * concept_coverage_component
        + 0.16 * validation_component
        + 0.10 * league_component
        + 0.12 * belief_component
        + moneyline_bonus
        - contradiction_penalty
    )

    fused[
        "decision_quality_score"
    ] = (
        decision_score
        .clip(
            lower=0.0,
            upper=1.0,
        )
        .mul(100.0)
    )

    fused["confidence_grade"] = (
        fused[
            "decision_quality_score"
        ]
        .apply(_confidence_grade)
    )

    fused["team_outlook"] = [
        _team_outlook(
            win_probability=
                row.moneyline_probability,
            offense_probability=
                row.offensive_outlook_score,
            prevention_probability=
                row.run_prevention_outlook_score,
        )
        for row in fused.itertuples(
            index=False
        )
    ]

    fused[
        "fusion_prediction_ready"
    ] = (
        fused[
            "available_probability_targets"
        ].ge(2)
        & fused[
            "unique_active_concepts"
        ].ge(2)
        & fused[
            "decision_quality_score"
        ].ge(40.0)
    )

    fused[
        "fusion_status"
    ] = np.where(
        fused[
            "fusion_prediction_ready"
        ],
        "fused_prediction_ready",
        "abstain_insufficient_fusion_evidence",
    )

    fused[
        "sportsbook_used"
    ] = False

    fused[
        "current_game_outcome_used"
    ] = False

    fused["future_games_used"] = False
    fused["fusion_engine_version"] = (
        ENGINE_VERSION
    )

    fused["fused_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    return fused


def _build_game_fusion(
    fused_team: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for game_pk, group in fused_team.groupby(
        "game_pk",
        sort=False,
    ):
        if len(group) != 2:
            continue

        group = group.sort_values(
            "team",
            kind="stable",
        )

        first = group.iloc[0]
        second = group.iloc[1]

        first_probability = first[
            "moneyline_probability"
        ]

        second_probability = second[
            "moneyline_probability"
        ]

        if (
            pd.notna(first_probability)
            and pd.notna(second_probability)
        ):
            if first_probability >= second_probability:
                predicted_winner = first["team"]
                predicted_loser = second["team"]
                winner_probability = float(
                    first_probability
                )
            else:
                predicted_winner = second["team"]
                predicted_loser = first["team"]
                winner_probability = float(
                    second_probability
                )
        else:
            predicted_winner = None
            predicted_loser = None
            winner_probability = np.nan

        team_ready_count = int(
            group[
                "fusion_prediction_ready"
            ].sum()
        )

        game_decision_score = float(
            group[
                "decision_quality_score"
            ].mean()
        )

        game_contradictions = int(
            group[
                "fusion_contradiction_count"
            ].sum()
        )

        records.append({
            "game_pk": game_pk,
            "game_date": first["game_date"],
            "home_team": (
                first.get("home_team")
                if pd.notna(
                    first.get(
                        "home_team",
                        np.nan,
                    )
                )
                else second.get("home_team")
            ),
            "away_team": (
                first.get("away_team")
                if pd.notna(
                    first.get(
                        "away_team",
                        np.nan,
                    )
                )
                else second.get("away_team")
            ),
            "predicted_winner": predicted_winner,
            "predicted_loser": predicted_loser,
            "predicted_winner_probability":
                winner_probability,
            "moneyline_prediction_ready":
                predicted_winner is not None,
            "fusion_ready_team_rows":
                team_ready_count,
            "game_decision_quality_score":
                game_decision_score,
            "game_contradiction_count":
                game_contradictions,
            "game_fusion_ready": bool(
                team_ready_count >= 1
            ),
            "sportsbook_used": False,
            "current_game_outcome_used": False,
            "future_games_used": False,
            "fusion_engine_version":
                ENGINE_VERSION,
        })

    return pd.DataFrame(records)


def run_prediction_fusion_engine(
    run_directory: str | Path,
) -> dict[str, Any]:
    run_directory = Path(
        run_directory
    )

    games, targets, active = _load_run(
        run_directory
    )

    fused_team = _build_fused_team_predictions(
        games=games,
        targets=targets,
        active=active,
    )

    fused_games = _build_game_fusion(
        fused_team
    )

    duplicate_team_games = int(
        fused_team.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    duplicate_games = int(
        fused_games["game_pk"]
        .duplicated()
        .sum()
    )

    if duplicate_team_games:
        raise AssertionError(
            f"Duplicate fused team-games: "
            f"{duplicate_team_games}"
        )

    if duplicate_games:
        raise AssertionError(
            f"Duplicate fused games: "
            f"{duplicate_games}"
        )

    if fused_team[
        "current_game_outcome_used"
    ].any():
        raise AssertionError(
            "Current-game outcomes were used."
        )

    if fused_team[
        "future_games_used"
    ].any():
        raise AssertionError(
            "Future games were used."
        )

    run_name = run_directory.name

    output_directory = (
        FUSION_ROOT
        / run_name
    )

    team_output_path = (
        output_directory
        / "fused_team_predictions.parquet"
    )

    game_output_path = (
        output_directory
        / "fused_game_predictions.parquet"
    )

    metadata_path = (
        output_directory
        / "fusion_metadata.json"
    )

    _atomic_parquet_write(
        fused_team,
        team_output_path,
    )

    _atomic_parquet_write(
        fused_games,
        game_output_path,
    )

    ready_team_rows = int(
        fused_team[
            "fusion_prediction_ready"
        ].sum()
    )

    ready_games = int(
        fused_games[
            "game_fusion_ready"
        ].sum()
    )

    moneyline_ready_games = int(
        fused_games[
            "moneyline_prediction_ready"
        ].sum()
    )

    result = {
        "engine":
            "ATLAS Prediction Fusion Engine",
        "engine_version":
            ENGINE_VERSION,
        "source_run_directory":
            str(run_directory),
        "team_rows":
            int(len(fused_team)),
        "game_rows":
            int(len(fused_games)),
        "fusion_ready_team_rows":
            ready_team_rows,
        "fusion_ready_games":
            ready_games,
        "moneyline_ready_games":
            moneyline_ready_games,
        "duplicate_team_games":
            duplicate_team_games,
        "duplicate_games":
            duplicate_games,
        "sportsbook_used":
            False,
        "current_game_outcomes_used":
            False,
        "future_games_used":
            False,
        "outputs": {
            "fused_team_predictions":
                str(team_output_path),
            "fused_game_predictions":
                str(game_output_path),
        },
        "policy": {
            "fusion_does_not_create_new_evidence":
                True,
            "missing_probabilities_not_imputed":
                True,
            "contradictions_penalized":
                True,
            "explanations_preserved":
                True,
            "market_used":
                False,
            "2026_outcomes_used":
                False,
        },
    }

    _atomic_json_write(
        result,
        metadata_path,
    )

    print("=" * 78)
    print("ATLAS PREDICTION FUSION ENGINE")
    print("=" * 78)
    print(
        f"Team Rows.................. "
        f"{len(fused_team):,}"
    )
    print(
        f"Game Rows.................. "
        f"{len(fused_games):,}"
    )
    print(
        f"Fusion-Ready Team Rows..... "
        f"{ready_team_rows:,}"
    )
    print(
        f"Fusion-Ready Games......... "
        f"{ready_games:,}"
    )
    print(
        f"Moneyline-Ready Games...... "
        f"{moneyline_ready_games:,}"
    )
    print(
        f"Duplicate Team-Games....... "
        f"{duplicate_team_games:,}"
    )
    print(
        f"Duplicate Games............ "
        f"{duplicate_games:,}"
    )
    print(
        "2026 Outcomes Used......... False"
    )
    print(
        "Sportsbook Used............ False"
    )
    print(
        f"Saved To................... "
        f"{output_directory}"
    )
    print("=" * 78)

    return result
