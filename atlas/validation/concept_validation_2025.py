
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR


ENGINE_VERSION = "1.0.0"
DISCOVERY_SEASON = 2024
VALIDATION_SEASON = 2025

INTERACTION_PATH = (
    DATA_DIR
    / "pregame"
    / "interactions"
    / "lineup_starter_inputs.parquet"
)

TEAM_TARGET_PATH = (
    DATA_DIR
    / "backtest"
    / "targets"
    / "team_game_targets.parquet"
)

CONCEPT_REGISTRY_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(DISCOVERY_SEASON)
    / "team_concept_registry.parquet"
)

CONCEPT_MEMBER_MAP_PATH = (
    DATA_DIR
    / "learning"
    / "team_concepts"
    / str(DISCOVERY_SEASON)
    / "team_concept_member_map.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "validation"
    / "concepts"
    / str(VALIDATION_SEASON)
)

TEAM_CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "team_checkpoints"
)

VALIDATION_REGISTRY_PATH = (
    OUTPUT_DIR
    / "concept_validation_registry.parquet"
)

VALIDATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "concept_validation_summary.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "concept_validation_metadata.json"
)


ALLOWED_CONCEPT_STATUSES = {
    "strong_concept_candidate",
    "concept_candidate",
}


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

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
    destination.parent.mkdir(parents=True, exist_ok=True)

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


def _normalize_dates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["game_date"] = pd.to_datetime(
        dataframe["game_date"],
        errors="raise",
    ).dt.normalize()

    return dataframe


def _condition_is_active(
    values: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if operator == "<=":
        return numeric.le(threshold)

    if operator == ">=":
        return numeric.ge(threshold)

    if operator == "<":
        return numeric.lt(threshold)

    if operator == ">":
        return numeric.gt(threshold)

    if operator == "==":
        return numeric.eq(threshold)

    raise ValueError(
        f"Unsupported threshold operator: {operator}"
    )


def _required_active_members(
    total_members: int,
) -> int:
    if total_members <= 2:
        return 1

    return max(
        2,
        int(
            np.ceil(total_members * 0.50)
        ),
    )


def _two_proportion_p_value(
    successes_a: int,
    sample_a: int,
    successes_b: int,
    sample_b: int,
) -> float | None:
    if sample_a <= 0 or sample_b <= 0:
        return None

    pooled = (
        successes_a + successes_b
    ) / (
        sample_a + sample_b
    )

    variance = (
        pooled
        * (1.0 - pooled)
        * (
            (1.0 / sample_a)
            + (1.0 / sample_b)
        )
    )

    if variance <= 0:
        return None

    rate_a = successes_a / sample_a
    rate_b = successes_b / sample_b

    z_score = (
        rate_a - rate_b
    ) / math.sqrt(variance)

    return float(
        math.erfc(
            abs(z_score)
            / math.sqrt(2.0)
        )
    )



def _benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        p_values,
        errors="coerce",
    )

    valid = numeric.dropna()

    output = pd.Series(
        np.nan,
        index=p_values.index,
        dtype="float64",
    )

    if valid.empty:
        return output

    ordered = valid.sort_values(
        kind="stable",
    )

    total = len(ordered)

    adjusted = (
        ordered
        * total
        / np.arange(
            1,
            total + 1,
            dtype="float64",
        )
    )

    adjusted = (
        adjusted.iloc[::-1]
        .cummin()
        .iloc[::-1]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    output.loc[
        ordered.index
    ] = adjusted.values

    return output


def _validation_status(
    discovery_effect_direction: str,
    validation_lift: float | None,
    validation_sample: int,
    q_value: float | None,
    active_successes: int,
    inactive_successes: int,
) -> str:
    if (
        validation_lift is None
        or pd.isna(validation_lift)
        or validation_sample < 10
    ):
        return "insufficient_2025_sample"

    expected_sign = (
        1
        if discovery_effect_direction == "supports_target"
        else -1
    )

    observed_sign = int(
        np.sign(validation_lift)
    )

    direction_retained = (
        observed_sign == expected_sign
    )

    absolute_lift = abs(validation_lift)

    total_successes = (
        int(active_successes)
        + int(inactive_successes)
    )

    # Rare outcomes require enough observed positive events
    # before receiving a confirmed or reversed classification.
    event_sample_sufficient = (
        total_successes >= 8
    )

    statistically_supported = (
        q_value is not None
        and not pd.isna(q_value)
        and q_value <= 0.10
    )

    statistically_strong = (
        q_value is not None
        and not pd.isna(q_value)
        and q_value <= 0.05
    )

    if (
        direction_retained
        and validation_sample >= 25
        and absolute_lift >= 0.08
        and statistically_strong
        and event_sample_sufficient
    ):
        return "validated_strong"

    if (
        direction_retained
        and validation_sample >= 20
        and absolute_lift >= 0.05
        and statistically_supported
        and event_sample_sufficient
    ):
        return "validated"

    if (
        direction_retained
        and validation_sample >= 15
        and absolute_lift >= 0.025
    ):
        return "direction_retained_weak"

    if (
        not direction_retained
        and validation_sample >= 25
        and absolute_lift >= 0.08
        and statistically_strong
        and event_sample_sufficient
    ):
        return "reversed_strong"

    if (
        not direction_retained
        and validation_sample >= 20
        and absolute_lift >= 0.05
        and statistically_supported
        and event_sample_sufficient
    ):
        return "reversed"

    return "not_confirmed"


def _team_paths(
    team: str,
) -> tuple[Path, Path]:
    return (
        TEAM_CHECKPOINT_DIR
        / f"{team}_concept_validation.parquet",
        TEAM_CHECKPOINT_DIR
        / f"{team}_validation_summary.parquet",
    )


def _team_complete(
    team: str,
) -> bool:
    registry_path, summary_path = _team_paths(team)

    return (
        registry_path.exists()
        and summary_path.exists()
    )


def _prepare_validation_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    interactions = _normalize_dates(
        _load_parquet(
            INTERACTION_PATH,
            "pregame interaction inputs",
        )
    )

    targets = _normalize_dates(
        _load_parquet(
            TEAM_TARGET_PATH,
            "team-game targets",
        )
    )

    concept_registry = _load_parquet(
        CONCEPT_REGISTRY_PATH,
        "2024 concept registry",
    )

    member_map = _load_parquet(
        CONCEPT_MEMBER_MAP_PATH,
        "2024 concept member map",
    )

    interactions = interactions[
        interactions["atlas_season"].eq(
            VALIDATION_SEASON
        )
    ].copy()

    target_columns = sorted(
        set(
            concept_registry["target"]
            .dropna()
            .astype(str)
        )
        & set(targets.columns)
    )

    join_keys = [
        "game_pk",
        "game_date",
        "atlas_season",
        "team",
    ]

    validation = interactions.merge(
        targets[
            join_keys + target_columns
        ],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )

    concept_registry = concept_registry[
        concept_registry[
            "concept_lifecycle_status"
        ].isin(ALLOWED_CONCEPT_STATUSES)
    ].copy()

    selected_ids = set(
        concept_registry["concept_id"]
        .astype(str)
    )

    member_map = member_map[
        member_map["concept_id"]
        .astype(str)
        .isin(selected_ids)
    ].copy()

    validation = validation.sort_values(
        [
            "team",
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return (
        validation,
        concept_registry,
        member_map,
    )


def _validate_one_team(
    team_rows: pd.DataFrame,
    team_concepts: pd.DataFrame,
    team_members: pd.DataFrame,
    team: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []

    concept_lookup = (
        team_concepts
        .set_index("concept_id", drop=False)
    )

    for concept_id, members in team_members.groupby(
        "concept_id",
        sort=True,
    ):
        concept_id = str(concept_id)

        if concept_id not in concept_lookup.index:
            continue

        concept = concept_lookup.loc[concept_id]

        target = str(concept["target"])

        if target not in team_rows.columns:
            continue

        member_flags = []
        available_members = 0

        for member in members.itertuples(
            index=False
        ):
            feature = str(member.feature)

            if feature not in team_rows.columns:
                continue

            available_members += 1

            flag = _condition_is_active(
                values=team_rows[feature],
                operator=str(
                    member.threshold_operator
                ),
                threshold=float(
                    member.threshold_value
                ),
            )

            member_flags.append(
                flag.fillna(False)
            )

        if not member_flags:
            continue

        flag_matrix = pd.concat(
            member_flags,
            axis=1,
        )

        active_count = (
            flag_matrix.sum(axis=1)
            .astype("int16")
        )

        required_count = _required_active_members(
            available_members
        )

        active_mask = (
            active_count >= required_count
        )

        target_values = pd.to_numeric(
            team_rows[target],
            errors="coerce",
        )

        valid_mask = target_values.notna()

        active_valid = (
            active_mask & valid_mask
        )

        inactive_valid = (
            (~active_mask) & valid_mask
        )

        active_sample = int(
            active_valid.sum()
        )

        inactive_sample = int(
            inactive_valid.sum()
        )

        active_successes = int(
            target_values[
                active_valid
            ].sum()
        )

        inactive_successes = int(
            target_values[
                inactive_valid
            ].sum()
        )

        active_rate = (
            active_successes / active_sample
            if active_sample > 0
            else None
        )

        inactive_rate = (
            inactive_successes / inactive_sample
            if inactive_sample > 0
            else None
        )

        validation_lift = (
            active_rate - inactive_rate
            if (
                active_rate is not None
                and inactive_rate is not None
            )
            else None
        )

        p_value = _two_proportion_p_value(
            successes_a=active_successes,
            sample_a=active_sample,
            successes_b=inactive_successes,
            sample_b=inactive_sample,
        )

        effect_direction = str(
            concept["effect_direction"]
        )

        expected_sign = (
            1
            if effect_direction == "supports_target"
            else -1
        )

        direction_retained = bool(
            validation_lift is not None
            and np.sign(validation_lift)
            == expected_sign
        )

        # Status is assigned after within-target
        # multiple-testing correction.
        status = None

        records.append({
            "concept_id":
                concept_id,
            "discovery_season":
                DISCOVERY_SEASON,
            "validation_season":
                VALIDATION_SEASON,
            "team":
                team,
            "target":
                target,
            "concept_domain":
                str(
                    concept["concept_domain"]
                ),
            "concept_scope":
                str(
                    concept["concept_scope"]
                ),
            "batting_order_slot": (
                None
                if pd.isna(
                    concept[
                        "batting_order_slot"
                    ]
                )
                else str(
                    concept[
                        "batting_order_slot"
                    ]
                )
            ),
            "concept_name":
                str(
                    concept["concept_name"]
                ),
            "effect_direction":
                effect_direction,
            "discovery_status":
                str(
                    concept[
                        "concept_lifecycle_status"
                    ]
                ),
            "discovery_confidence":
                float(
                    concept[
                        "concept_confidence_score"
                    ]
                ),
            "discovery_weighted_lift":
                float(
                    concept["weighted_lift"]
                ),
            "total_concept_members":
                int(
                    concept["member_count"]
                ),
            "available_2025_members":
                int(available_members),
            "required_active_members":
                int(required_count),
            "validation_games":
                int(valid_mask.sum()),
            "active_2025_sample":
                active_sample,
            "inactive_2025_sample":
                inactive_sample,
            "active_2025_successes":
                active_successes,
            "inactive_2025_successes":
                inactive_successes,
            "active_2025_rate":
                active_rate,
            "inactive_2025_rate":
                inactive_rate,
            "validation_lift":
                validation_lift,
            "validation_absolute_lift": (
                abs(validation_lift)
                if validation_lift is not None
                else None
            ),
            "direction_retained":
                direction_retained,
            "validation_p_value":
                p_value,
            "validation_q_value":
                None,
            "validation_status":
                status,
            "prediction_weight_assigned":
                False,
            "2026_used":
                False,
            "engine_version":
                ENGINE_VERSION,
        })

    registry = pd.DataFrame(records)

    if not registry.empty:
        registry["validation_q_value"] = (
            registry.groupby(
                "target",
                sort=False,
            )["validation_p_value"]
            .transform(
                _benjamini_hochberg
            )
        )

        registry["validation_status"] = [
            _validation_status(
                discovery_effect_direction=
                    row.effect_direction,
                validation_lift=
                    row.validation_lift,
                validation_sample=
                    row.active_2025_sample,
                q_value=
                    row.validation_q_value,
                active_successes=
                    row.active_2025_successes,
                inactive_successes=
                    row.inactive_2025_successes,
            )
            for row in registry.itertuples(
                index=False
            )
        ]

        registry = registry.sort_values(
            [
                "validation_status",
                "validation_absolute_lift",
                "discovery_confidence",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    if registry.empty:
        summary = pd.DataFrame(
            [{
                "team": team,
                "concepts_tested": 0,
                "validated_strong": 0,
                "validated": 0,
                "direction_retained_weak": 0,
                "not_confirmed": 0,
                "reversed": 0,
                "insufficient_2025_sample": 0,
            }]
        )

    else:
        counts = (
            registry["validation_status"]
            .value_counts()
        )

        summary = pd.DataFrame(
            [{
                "team":
                    team,
                "concepts_tested":
                    int(len(registry)),
                "validated_strong":
                    int(
                        counts.get(
                            "validated_strong",
                            0,
                        )
                    ),
                "validated":
                    int(
                        counts.get(
                            "validated",
                            0,
                        )
                    ),
                "direction_retained_weak":
                    int(
                        counts.get(
                            "direction_retained_weak",
                            0,
                        )
                    ),
                "not_confirmed":
                    int(
                        counts.get(
                            "not_confirmed",
                            0,
                        )
                    ),
                "reversed":
                    int(
                        counts.get(
                            "reversed",
                            0,
                        )
                    ),
                "reversed_strong":
                    int(
                        counts.get(
                            "reversed_strong",
                            0,
                        )
                    ),
                "insufficient_2025_sample":
                    int(
                        counts.get(
                            "insufficient_2025_sample",
                            0,
                        )
                    ),
            }]
        )

    summary["discovery_season"] = DISCOVERY_SEASON
    summary["validation_season"] = VALIDATION_SEASON
    summary["prediction_weights_assigned"] = False
    summary["engine_version"] = ENGINE_VERSION

    return registry, summary


def _assemble_master(
    teams: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry_frames = []
    summary_frames = []

    for team in teams:
        registry_path, summary_path = _team_paths(team)

        if registry_path.exists():
            registry_frames.append(
                pd.read_parquet(
                    registry_path
                )
            )

        if summary_path.exists():
            summary_frames.append(
                pd.read_parquet(
                    summary_path
                )
            )

    registry = (
        pd.concat(
            registry_frames,
            ignore_index=True,
        )
        if registry_frames
        else pd.DataFrame()
    )

    summary = (
        pd.concat(
            summary_frames,
            ignore_index=True,
        )
        if summary_frames
        else pd.DataFrame()
    )

    return registry, summary


def run_concept_validation_2025(
    only_team: str | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    started = time.time()

    (
        validation_rows,
        concept_registry,
        member_map,
    ) = _prepare_validation_data()

    all_teams = sorted(
        validation_rows["team"]
        .dropna()
        .astype(str)
        .unique()
    )

    alias_map = {
        "ARI": "AZ",
        "OAK": "ATH",
    }

    if only_team is not None:
        requested = alias_map.get(
            str(only_team).upper(),
            str(only_team).upper(),
        )

        if requested not in all_teams:
            raise ValueError(
                f"Unknown 2025 team code: {requested}"
            )

        target_teams = [requested]

    elif limit is not None:
        target_teams = all_teams[:limit]

    else:
        target_teams = all_teams

    TEAM_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete = [
        team
        for team in target_teams
        if resume and _team_complete(team)
    ]

    remaining = [
        team
        for team in target_teams
        if team not in complete
    ]

    print("=" * 78)
    print("ATLAS 2025 BLIND CONCEPT VALIDATION")
    print("=" * 78)
    print(
        f"Discovery Season......... {DISCOVERY_SEASON}"
    )
    print(
        f"Validation Season........ {VALIDATION_SEASON}"
    )
    print(
        f"2025 Team-Game Rows...... {len(validation_rows):,}"
    )
    print(
        f"Frozen Concepts.......... {len(concept_registry):,}"
    )
    print(
        f"Frozen Members........... {len(member_map):,}"
    )
    print(
        f"Target Teams............. {len(target_teams):,}"
    )
    print(
        f"Already Complete......... {len(complete):,}"
    )
    print(
        f"Remaining................ {len(remaining):,}"
    )
    print("=" * 78)

    newly_built = 0

    for team in remaining:
        team_started = time.time()

        team_rows = validation_rows[
            validation_rows["team"].eq(team)
        ].copy()

        team_concepts = concept_registry[
            concept_registry["team"].eq(team)
        ].copy()

        concept_ids = set(
            team_concepts["concept_id"]
            .astype(str)
        )

        team_members = member_map[
            member_map["concept_id"]
            .astype(str)
            .isin(concept_ids)
        ].copy()

        registry, summary = _validate_one_team(
            team_rows=team_rows,
            team_concepts=team_concepts,
            team_members=team_members,
            team=team,
        )

        registry_path, summary_path = _team_paths(team)

        _atomic_parquet_write(
            registry,
            registry_path,
        )

        _atomic_parquet_write(
            summary,
            summary_path,
        )

        newly_built += 1

        elapsed = time.time() - team_started

        validated_count = int(
            registry[
                "validation_status"
            ].isin(
                [
                    "validated_strong",
                    "validated",
                ]
            ).sum()
            if not registry.empty
            else 0
        )

        print(
            f"Completed {team:<4} | "
            f"{len(complete) + newly_built:>2}/"
            f"{len(target_teams):<2} | "
            f"tested={len(registry):>4,} | "
            f"validated={validated_count:>3,} | "
            f"time={elapsed:>5.1f}s"
        )

    registry, summary = _assemble_master(
        target_teams
    )

    complete_teams = int(
        summary["team"].nunique()
        if not summary.empty
        else 0
    )

    full_run_complete = bool(
        only_team is None
        and limit is None
        and complete_teams == len(all_teams)
    )

    if full_run_complete:
        _atomic_parquet_write(
            registry,
            VALIDATION_REGISTRY_PATH,
        )

        _atomic_parquet_write(
            summary,
            VALIDATION_SUMMARY_PATH,
        )

    duplicate_ids = int(
        registry["concept_id"]
        .duplicated()
        .sum()
        if not registry.empty
        else 0
    )

    if duplicate_ids:
        raise AssertionError(
            f"Duplicate validated concept IDs: {duplicate_ids}"
        )

    status_counts = (
        registry["validation_status"]
        .value_counts()
        if not registry.empty
        else pd.Series(dtype="int64")
    )

    elapsed = time.time() - started

    result = {
        "engine":
            "ATLAS 2025 Blind Concept Validation Engine",
        "engine_version":
            ENGINE_VERSION,
        "discovery_season":
            DISCOVERY_SEASON,
        "validation_season":
            VALIDATION_SEASON,
        "teams_complete":
            complete_teams,
        "target_teams":
            int(len(target_teams)),
        "newly_built":
            int(newly_built),
        "concepts_tested":
            int(len(registry)),
        "validated_strong":
            int(
                status_counts.get(
                    "validated_strong",
                    0,
                )
            ),
        "validated":
            int(
                status_counts.get(
                    "validated",
                    0,
                )
            ),
        "direction_retained_weak":
            int(
                status_counts.get(
                    "direction_retained_weak",
                    0,
                )
            ),
        "not_confirmed":
            int(
                status_counts.get(
                    "not_confirmed",
                    0,
                )
            ),
        "reversed":
            int(
                status_counts.get(
                    "reversed",
                    0,
                )
            ),
        "insufficient_2025_sample":
            int(
                status_counts.get(
                    "insufficient_2025_sample",
                    0,
                )
            ),
        "duplicate_concept_ids":
            duplicate_ids,
        "prediction_weights_assigned":
            False,
        "2026_used":
            False,
        "full_run_complete":
            full_run_complete,
        "elapsed_seconds":
            float(elapsed),
        "outputs": {
            "validation_registry": (
                str(VALIDATION_REGISTRY_PATH)
                if full_run_complete
                else None
            ),
            "validation_summary": (
                str(VALIDATION_SUMMARY_PATH)
                if full_run_complete
                else None
            ),
            "checkpoint_directory":
                str(TEAM_CHECKPOINT_DIR),
        },
    }

    _atomic_json_write(
        result,
        METADATA_PATH,
    )

    print("\n" + "=" * 78)
    print("2025 BLIND CONCEPT VALIDATION COMPLETE")
    print("=" * 78)
    print(
        f"Teams Complete........... "
        f"{complete_teams:,}/{len(target_teams):,}"
    )
    print(
        f"Concepts Tested.......... "
        f"{len(registry):,}"
    )
    print(
        f"Validated Strong......... "
        f"{result['validated_strong']:,}"
    )
    print(
        f"Validated................ "
        f"{result['validated']:,}"
    )
    print(
        f"Direction Retained Weak.. "
        f"{result['direction_retained_weak']:,}"
    )
    print(
        f"Not Confirmed............ "
        f"{result['not_confirmed']:,}"
    )
    print(
        f"Reversed................. "
        f"{result['reversed']:,}"
    )
    print(
        f"Insufficient Sample...... "
        f"{result['insufficient_2025_sample']:,}"
    )
    print(
        f"Prediction Weights....... "
        f"{result['prediction_weights_assigned']}"
    )
    print(
        f"Full Master Built........ "
        f"{full_run_complete}"
    )
    print("=" * 78)

    return result
