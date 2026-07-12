
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.config import DATA_DIR
from atlas.learning.team_evidence_discovery import (
    ENGINE_VERSION as DISCOVERY_ENGINE_VERSION,
    LEARNING_SEASON,
    TARGET_COLUMNS,
    _numeric_feature_columns,
    _safe_rate,
    discover_team_target_evidence,
)


RUNNER_VERSION = "2.0.0"

INTERACTION_INPUT_PATH = (
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

OUTPUT_ROOT = (
    DATA_DIR
    / "learning"
    / "team_evidence"
    / str(LEARNING_SEASON)
)

TEAM_CHECKPOINT_DIR = (
    OUTPUT_ROOT
    / "team_checkpoints"
)

MASTER_REGISTRY_PATH = (
    OUTPUT_ROOT
    / "team_evidence_registry.parquet"
)

MASTER_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "team_evidence_summary.parquet"
)

RUN_METADATA_PATH = (
    OUTPUT_ROOT
    / "team_evidence_runner_metadata.json"
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


def _load_learning_table() -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    if not INTERACTION_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing interaction inputs: {INTERACTION_INPUT_PATH}"
        )

    if not TEAM_TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Missing team targets: {TEAM_TARGET_PATH}"
        )

    inputs = pd.read_parquet(
        INTERACTION_INPUT_PATH
    )

    targets = pd.read_parquet(
        TEAM_TARGET_PATH
    )

    inputs["game_date"] = pd.to_datetime(
        inputs["game_date"],
        errors="raise",
    ).dt.normalize()

    targets["game_date"] = pd.to_datetime(
        targets["game_date"],
        errors="raise",
    ).dt.normalize()

    available_targets = [
        column
        for column in TARGET_COLUMNS
        if column in targets.columns
    ]

    if not available_targets:
        raise KeyError(
            "No configured discovery targets were found."
        )

    join_columns = [
        "game_pk",
        "team",
        "atlas_season",
        "game_date",
    ]

    combined = inputs.merge(
        targets[
            join_columns
            + available_targets
        ],
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    learning = combined[
        combined["atlas_season"].eq(
            LEARNING_SEASON
        )
    ].copy()

    if learning.empty:
        raise ValueError(
            f"No learning rows found for {LEARNING_SEASON}."
        )

    learning = learning.sort_values(
        [
            "team",
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    feature_columns = _numeric_feature_columns(
        learning
    )

    leaked_targets = sorted(
        set(feature_columns)
        & set(available_targets)
    )

    if leaked_targets:
        raise AssertionError(
            f"Target leakage detected: {leaked_targets}"
        )

    teams = sorted(
        str(value)
        for value in learning[
            "team"
        ].dropna().unique()
    )

    return (
        learning,
        feature_columns,
        available_targets,
    )


def _team_paths(
    team: str,
) -> tuple[Path, Path]:
    return (
        TEAM_CHECKPOINT_DIR
        / f"{team}_evidence.parquet",
        TEAM_CHECKPOINT_DIR
        / f"{team}_summary.parquet",
    )


def _team_complete(
    team: str,
) -> bool:
    evidence_path, summary_path = (
        _team_paths(team)
    )

    return (
        evidence_path.exists()
        and summary_path.exists()
    )


def _build_one_team(
    team_df: pd.DataFrame,
    team: str,
    feature_columns: list[str],
    target_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    evidence_records: list[
        dict[str, Any]
    ] = []

    summary_records: list[
        dict[str, Any]
    ] = []

    total_targets = len(
        target_columns
    )

    print(
        f"\n[{team}] Games: {len(team_df):,} | "
        f"Features: {len(feature_columns):,}"
    )

    for target_index, target in enumerate(
        target_columns,
        start=1,
    ):
        started = time.time()

        target_records = (
            discover_team_target_evidence(
                team_df=team_df,
                team=team,
                target=target,
                feature_columns=feature_columns,
            )
        )

        evidence_records.extend(
            target_records
        )

        base_rate = _safe_rate(
            team_df[target]
        )

        summary_records.append({
            "learning_season":
                LEARNING_SEASON,
            "team":
                team,
            "target":
                target,
            "games":
                int(len(team_df)),
            "target_base_rate":
                base_rate,
            "features_screened":
                int(len(feature_columns)),
            "evidence_objects_found":
                int(len(target_records)),
            "strong_candidates":
                int(sum(
                    record[
                        "lifecycle_status"
                    ] == "strong_candidate"
                    for record in target_records
                )),
            "candidates":
                int(sum(
                    record[
                        "lifecycle_status"
                    ] == "candidate"
                    for record in target_records
                )),
            "weak_candidates":
                int(sum(
                    record[
                        "lifecycle_status"
                    ] == "weak_candidate"
                    for record in target_records
                )),
            "validated_out_of_sample":
                False,
            "requires_2025_validation":
                True,
            "engine_version":
                DISCOVERY_ENGINE_VERSION,
            "runner_version":
                RUNNER_VERSION,
        })

        elapsed = time.time() - started

        print(
            f"  Target {target_index:>2}/"
            f"{total_targets}: "
            f"{target:<28} "
            f"objects={len(target_records):>5,} "
            f"time={elapsed:>6.1f}s"
        )

    evidence_df = pd.DataFrame(
        evidence_records
    )

    summary_df = pd.DataFrame(
        summary_records
    )

    if not evidence_df.empty:
        evidence_df = evidence_df.sort_values(
            [
                "target",
                "confidence_score",
                "absolute_lift",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    summary_df = summary_df.sort_values(
        "target",
        kind="stable",
    ).reset_index(drop=True)

    return evidence_df, summary_df


def _assemble_master_registry(
    teams: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    evidence_frames = []
    summary_frames = []

    for team in teams:
        evidence_path, summary_path = (
            _team_paths(team)
        )

        if not (
            evidence_path.exists()
            and summary_path.exists()
        ):
            continue

        evidence_frames.append(
            pd.read_parquet(
                evidence_path
            )
        )

        summary_frames.append(
            pd.read_parquet(
                summary_path
            )
        )

    evidence = (
        pd.concat(
            evidence_frames,
            ignore_index=True,
        )
        if evidence_frames
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

    if not evidence.empty:
        duplicate_ids = int(
            evidence[
                "evidence_id"
            ].duplicated().sum()
        )

        if duplicate_ids:
            raise AssertionError(
                f"Duplicate evidence IDs: {duplicate_ids}"
            )

        evidence = evidence.sort_values(
            [
                "team",
                "target",
                "confidence_score",
                "absolute_lift",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
            kind="stable",
        ).reset_index(drop=True)

    if not summary.empty:
        summary = summary.sort_values(
            [
                "team",
                "target",
            ],
            kind="stable",
        ).reset_index(drop=True)

    return evidence, summary


def run_team_evidence_discovery_v2(
    limit: int | None = None,
    resume: bool = True,
    only_team: str | None = None,
) -> dict[str, Any]:
    started = time.time()

    (
        learning,
        feature_columns,
        target_columns,
    ) = _load_learning_table()

    all_teams = sorted(
        str(value)
        for value in learning[
            "team"
        ].dropna().unique()
    )

    if only_team is not None:
        only_team = str(
            only_team
        ).upper()

        if only_team not in all_teams:
            raise ValueError(
                f"Unknown team: {only_team}"
            )

        target_teams = [
            only_team
        ]
    elif limit is not None:
        target_teams = all_teams[
            :limit
        ]
    else:
        target_teams = all_teams

    TEAM_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = [
        team
        for team in target_teams
        if (
            resume
            and _team_complete(team)
        )
    ]

    remaining = [
        team
        for team in target_teams
        if team not in completed
    ]

    print("=" * 78)
    print(
        "ATLAS TEAM EVIDENCE DISCOVERY V2"
    )
    print("=" * 78)
    print(
        f"Learning Season.......... "
        f"{LEARNING_SEASON}"
    )
    print(
        f"Learning Rows............ "
        f"{len(learning):,}"
    )
    print(
        f"Features Screened........ "
        f"{len(feature_columns):,}"
    )
    print(
        f"Targets per Team......... "
        f"{len(target_columns):,}"
    )
    print(
        f"Target Teams............. "
        f"{len(target_teams):,}"
    )
    print(
        f"Already Complete......... "
        f"{len(completed):,}"
    )
    print(
        f"Remaining................ "
        f"{len(remaining):,}"
    )
    print(
        f"Checkpoint Directory..... "
        f"{TEAM_CHECKPOINT_DIR}"
    )
    print("=" * 78)

    newly_built = 0

    for team_number, team in enumerate(
        remaining,
        start=1,
    ):
        team_started = time.time()

        team_df = learning[
            learning["team"].eq(team)
        ].copy()

        evidence_df, summary_df = (
            _build_one_team(
                team_df=team_df,
                team=team,
                feature_columns=feature_columns,
                target_columns=target_columns,
            )
        )

        evidence_path, summary_path = (
            _team_paths(team)
        )

        _atomic_parquet_write(
            evidence_df,
            evidence_path,
        )

        _atomic_parquet_write(
            summary_df,
            summary_path,
        )

        newly_built += 1

        team_elapsed = (
            time.time()
            - team_started
        )

        complete_now = (
            len(completed)
            + newly_built
        )

        print(
            f"\nCompleted {team} | "
            f"{complete_now}/{len(target_teams)} teams | "
            f"evidence={len(evidence_df):,} | "
            f"time={team_elapsed:.1f}s"
        )

    evidence, summary = (
        _assemble_master_registry(
            target_teams
        )
    )

    complete_team_count = int(
        summary["team"].nunique()
        if not summary.empty
        else 0
    )

    full_run_complete = (
        only_team is None
        and limit is None
        and complete_team_count
        == len(all_teams)
    )

    if full_run_complete:
        _atomic_parquet_write(
            evidence,
            MASTER_REGISTRY_PATH,
        )

        _atomic_parquet_write(
            summary,
            MASTER_SUMMARY_PATH,
        )

    elapsed = (
        time.time()
        - started
    )

    result = {
        "engine":
            "ATLAS Team Evidence Discovery",
        "runner_version":
            RUNNER_VERSION,
        "discovery_engine_version":
            DISCOVERY_ENGINE_VERSION,
        "learning_season":
            LEARNING_SEASON,
        "all_teams":
            int(len(all_teams)),
        "target_teams":
            int(len(target_teams)),
        "teams_complete":
            complete_team_count,
        "newly_built":
            int(newly_built),
        "features_screened":
            int(len(feature_columns)),
        "targets_per_team":
            int(len(target_columns)),
        "evidence_objects":
            int(len(evidence)),
        "full_run_complete":
            full_run_complete,
        "elapsed_seconds":
            float(elapsed),
        "checkpoint_directory":
            str(TEAM_CHECKPOINT_DIR),
        "master_registry":
            (
                str(MASTER_REGISTRY_PATH)
                if full_run_complete
                else None
            ),
        "master_summary":
            (
                str(MASTER_SUMMARY_PATH)
                if full_run_complete
                else None
            ),
        "prediction_weights_assigned":
            False,
        "requires_2025_validation":
            True,
    }

    _atomic_json_write(
        result,
        RUN_METADATA_PATH,
    )

    print("\n" + "=" * 78)
    print(
        "TEAM EVIDENCE DISCOVERY V2 RUN COMPLETE"
    )
    print("=" * 78)
    print(
        f"Teams Complete........... "
        f"{complete_team_count:,}/"
        f"{len(target_teams):,}"
    )
    print(
        f"Newly Built.............. "
        f"{newly_built:,}"
    )
    print(
        f"Evidence Objects......... "
        f"{len(evidence):,}"
    )
    print(
        f"Elapsed.................. "
        f"{elapsed / 60:.1f} minutes"
    )
    print(
        f"Full Registry Built...... "
        f"{full_run_complete}"
    )
    print("=" * 78)

    return result
