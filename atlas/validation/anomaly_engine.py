
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR, MASTER_DIR


ANOMALY_ENGINE_VERSION = "1.0.0"

MASTER_PITCH_PATH = (
    MASTER_DIR
    / "master_pitch_database.parquet"
)

MASTER_GAME_PATH = (
    MASTER_DIR
    / "master_game_database.parquet"
)

HISTORICAL_LINEUP_PATH = (
    DATA_DIR
    / "history"
    / "lineups"
    / "historical_starting_lineups.parquet"
)

OUTPUT_DIR = (
    DATA_DIR
    / "validation"
    / "anomalies"
)

GAME_ANOMALY_PATH = (
    OUTPUT_DIR
    / "game_anomaly_registry.parquet"
)

ENTITY_ANOMALY_PATH = (
    OUTPUT_DIR
    / "entity_game_anomalies.parquet"
)

DOUBLEHEADER_PATH = (
    OUTPUT_DIR
    / "same_day_multi_game_registry.parquet"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "anomaly_engine_metadata.json"
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


def _load_sources() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    if not MASTER_PITCH_PATH.exists():
        raise FileNotFoundError(
            f"Missing master pitches: {MASTER_PITCH_PATH}"
        )

    if not MASTER_GAME_PATH.exists():
        raise FileNotFoundError(
            f"Missing master games: {MASTER_GAME_PATH}"
        )

    pitches = pd.read_parquet(
        MASTER_PITCH_PATH
    )

    games = pd.read_parquet(
        MASTER_GAME_PATH
    )

    lineups = (
        pd.read_parquet(
            HISTORICAL_LINEUP_PATH
        )
        if HISTORICAL_LINEUP_PATH.exists()
        else None
    )

    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"],
        errors="coerce",
    ).dt.normalize()

    games["game_date"] = pd.to_datetime(
        games["game_date"],
        errors="coerce",
    ).dt.normalize()

    if lineups is not None:
        lineups["game_date"] = pd.to_datetime(
            lineups["game_date"],
            errors="coerce",
        ).dt.normalize()

    return pitches, games, lineups


def _prepare_pitch_membership(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
    }

    missing = required - set(pitches.columns)

    if missing:
        raise KeyError(
            f"Pitch data missing: {sorted(missing)}"
        )

    events = pitches[
        [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
            "inning_topbot",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
        ]
    ].copy()

    top = (
        events["inning_topbot"]
        .astype("string")
        .eq("Top")
    )

    events["batting_team"] = np.where(
        top,
        events["away_team"],
        events["home_team"],
    )

    events["pitching_team"] = np.where(
        top,
        events["home_team"],
        events["away_team"],
    )

    events["pitch_event_key"] = (
        events["game_pk"].astype(str)
        + "_"
        + events["at_bat_number"].astype(str)
        + "_"
        + events["pitch_number"].astype(str)
    )

    return events


def _entity_team_anomalies(
    events: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    batter_membership = (
        events.dropna(subset=["batter"])
        .groupby(
            ["game_pk", "batter"],
            sort=True,
        )["batting_team"]
        .agg(
            lambda values: sorted(
                set(
                    str(value)
                    for value in values.dropna()
                )
            )
        )
    )

    for (
        game_pk,
        batter_id,
    ), teams in batter_membership.items():
        if len(teams) > 1:
            records.append({
                "game_pk": int(game_pk),
                "entity_type": "batter",
                "entity_id": int(batter_id),
                "teams_observed": teams,
                "team_count": int(len(teams)),
                "anomaly_type": (
                    "entity_appears_for_multiple_teams_in_game"
                ),
                "severity": "critical",
                "pregame_model_safe": False,
                "strict_backtest_safe": False,
                "preserve_historical_data": True,
            })

    pitcher_membership = (
        events.dropna(subset=["pitcher"])
        .groupby(
            ["game_pk", "pitcher"],
            sort=True,
        )["pitching_team"]
        .agg(
            lambda values: sorted(
                set(
                    str(value)
                    for value in values.dropna()
                )
            )
        )
    )

    for (
        game_pk,
        pitcher_id,
    ), teams in pitcher_membership.items():
        if len(teams) > 1:
            records.append({
                "game_pk": int(game_pk),
                "entity_type": "pitcher",
                "entity_id": int(pitcher_id),
                "teams_observed": teams,
                "team_count": int(len(teams)),
                "anomaly_type": (
                    "entity_appears_for_multiple_teams_in_game"
                ),
                "severity": "critical",
                "pregame_model_safe": False,
                "strict_backtest_safe": False,
                "preserve_historical_data": True,
            })

    return pd.DataFrame(records)


def _same_day_multi_games(
    games: pd.DataFrame,
) -> pd.DataFrame:
    home = games[
        [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
        ]
    ].rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
        }
    )

    away = games[
        [
            "game_pk",
            "game_date",
            "away_team",
            "home_team",
        ]
    ].rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
        }
    )

    team_games = pd.concat(
        [home, away],
        ignore_index=True,
    )

    counts = (
        team_games.groupby(
            ["game_date", "team"],
            sort=True,
        )
        .agg(
            games_on_date=(
                "game_pk",
                "nunique",
            ),
            game_pks=(
                "game_pk",
                lambda values: sorted(
                    int(value)
                    for value in set(values)
                ),
            ),
            opponents=(
                "opponent",
                lambda values: sorted(
                    str(value)
                    for value in set(values)
                ),
            ),
        )
        .reset_index()
    )

    counts = counts[
        counts["games_on_date"] > 1
    ].copy()

    counts["anomaly_type"] = (
        "same_day_multiple_games"
    )

    counts["severity"] = "special_handling"

    counts["pregame_model_safe"] = True

    counts["same_day_prior_results_safe"] = False

    counts["recommended_snapshot_rule"] = (
        "exclude_all_current_date_games_without_verified_start_times"
    )

    counts["preserve_historical_data"] = True

    return counts


def _lineup_anomalies(
    lineups: pd.DataFrame | None,
) -> pd.DataFrame:
    if lineups is None:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    duplicate_mask = lineups.duplicated(
        subset=["game_pk", "team"],
        keep=False,
    )

    for _, row in lineups.loc[
        duplicate_mask
    ].iterrows():
        records.append({
            "game_pk": int(row["game_pk"]),
            "entity_type": "lineup",
            "entity_id": str(row["team"]),
            "teams_observed": [str(row["team"])],
            "team_count": 1,
            "anomaly_type": (
                "duplicate_team_lineup_record"
            ),
            "severity": "critical",
            "pregame_model_safe": False,
            "strict_backtest_safe": False,
            "preserve_historical_data": True,
        })

    incomplete = lineups[
        ~lineups[
            "starting_lineup_complete"
        ].fillna(False)
    ]

    for _, row in incomplete.iterrows():
        records.append({
            "game_pk": int(row["game_pk"]),
            "entity_type": "lineup",
            "entity_id": str(row["team"]),
            "teams_observed": [str(row["team"])],
            "team_count": 1,
            "anomaly_type": (
                "incomplete_reconstructed_starting_lineup"
            ),
            "severity": "high",
            "pregame_model_safe": False,
            "strict_backtest_safe": False,
            "preserve_historical_data": True,
        })

    return pd.DataFrame(records)


def build_game_anomaly_registry(
    pitches: pd.DataFrame,
    games: pd.DataFrame,
    lineups: pd.DataFrame | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    events = _prepare_pitch_membership(
        pitches
    )

    entity_anomalies = (
        _entity_team_anomalies(
            events
        )
    )

    lineup_anomalies = (
        _lineup_anomalies(
            lineups
        )
    )

    entity_frames = [
        frame
        for frame in [
            entity_anomalies,
            lineup_anomalies,
        ]
        if not frame.empty
    ]

    if entity_frames:
        all_entity_anomalies = pd.concat(
            entity_frames,
            ignore_index=True,
        )
    else:
        all_entity_anomalies = pd.DataFrame(
            columns=[
                "game_pk",
                "entity_type",
                "entity_id",
                "teams_observed",
                "team_count",
                "anomaly_type",
                "severity",
                "pregame_model_safe",
                "strict_backtest_safe",
                "preserve_historical_data",
            ]
        )

    doubleheaders = _same_day_multi_games(
        games
    )

    base = games[
        [
            "game_pk",
            "game_date",
            "home_team",
            "away_team",
        ]
    ].drop_duplicates(
        subset=["game_pk"]
    ).copy()

    pitch_counts = (
        events.groupby(
            "game_pk",
            sort=True,
        )
        .agg(
            pitch_rows=(
                "pitch_event_key",
                "size",
            ),
            duplicate_pitch_event_keys=(
                "pitch_event_key",
                lambda values: int(
                    values.duplicated().sum()
                ),
            ),
            unique_batting_teams=(
                "batting_team",
                "nunique",
            ),
            unique_pitching_teams=(
                "pitching_team",
                "nunique",
            ),
        )
        .reset_index()
    )

    base = base.merge(
        pitch_counts,
        on="game_pk",
        how="left",
        validate="one_to_one",
    )

    anomaly_counts = (
        all_entity_anomalies.groupby(
            "game_pk",
            sort=True,
        )
        .agg(
            entity_anomaly_count=(
                "anomaly_type",
                "size",
            ),
            anomaly_types=(
                "anomaly_type",
                lambda values: sorted(
                    set(
                        str(value)
                        for value in values
                    )
                ),
            ),
            highest_severity=(
                "severity",
                lambda values: (
                    "critical"
                    if "critical" in set(values)
                    else (
                        "high"
                        if "high" in set(values)
                        else "none"
                    )
                ),
            ),
        )
        .reset_index()
    )

    base = base.merge(
        anomaly_counts,
        on="game_pk",
        how="left",
        validate="one_to_one",
    )

    base["entity_anomaly_count"] = (
        base["entity_anomaly_count"]
        .fillna(0)
        .astype("int64")
    )

    base["anomaly_types"] = base[
        "anomaly_types"
    ].apply(
        lambda value: (
            value
            if isinstance(value, list)
            else []
        )
    )

    base["highest_severity"] = (
        base["highest_severity"]
        .fillna("none")
    )

    base["structural_team_count_valid"] = (
        base["unique_batting_teams"].eq(2)
        & base["unique_pitching_teams"].eq(2)
    )

    base["pitch_event_identity_valid"] = (
        base[
            "duplicate_pitch_event_keys"
        ].fillna(0).eq(0)
    )

    base["critical_entity_anomaly"] = (
        base["highest_severity"].eq(
            "critical"
        )
    )

    base["strict_backtest_safe"] = (
        base[
            "structural_team_count_valid"
        ]
        & base[
            "pitch_event_identity_valid"
        ]
        & ~base[
            "critical_entity_anomaly"
        ]
    )

    base["pregame_model_safe"] = (
        base["strict_backtest_safe"]
    )

    base["quarantine_from_strict_walk_forward"] = (
        ~base["strict_backtest_safe"]
    )

    base["preserve_historical_data"] = True

    base["recommended_action"] = np.where(
        base[
            "quarantine_from_strict_walk_forward"
        ],
        (
            "preserve_in_master_and_game_cards;"
            "exclude_from_initial_strict_backtest;"
            "review_with_verified_chronology"
        ),
        "retain_for_strict_walk_forward",
    )

    base["anomaly_engine_version"] = (
        ANOMALY_ENGINE_VERSION
    )

    return (
        base.sort_values(
            ["game_date", "game_pk"],
            kind="stable",
        ).reset_index(drop=True),
        all_entity_anomalies.sort_values(
            ["game_pk", "entity_type"],
            kind="stable",
        ).reset_index(drop=True),
        doubleheaders.sort_values(
            ["game_date", "team"],
            kind="stable",
        ).reset_index(drop=True),
    )


def run_anomaly_engine() -> dict[str, Any]:
    pitches, games, lineups = (
        _load_sources()
    )

    (
        game_registry,
        entity_anomalies,
        doubleheaders,
    ) = build_game_anomaly_registry(
        pitches=pitches,
        games=games,
        lineups=lineups,
    )

    _atomic_parquet_write(
        game_registry,
        GAME_ANOMALY_PATH,
    )

    _atomic_parquet_write(
        entity_anomalies,
        ENTITY_ANOMALY_PATH,
    )

    _atomic_parquet_write(
        doubleheaders,
        DOUBLEHEADER_PATH,
    )

    quarantined = game_registry[
        game_registry[
            "quarantine_from_strict_walk_forward"
        ]
    ]

    summary = {
        "engine": (
            "ATLAS Data Validation & Anomaly Engine"
        ),
        "engine_version": (
            ANOMALY_ENGINE_VERSION
        ),
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "games_audited": int(
            len(game_registry)
        ),
        "strict_backtest_safe_games": int(
            game_registry[
                "strict_backtest_safe"
            ].sum()
        ),
        "quarantined_games": int(
            len(quarantined)
        ),
        "quarantined_game_pks": [
            int(value)
            for value in quarantined[
                "game_pk"
            ].tolist()
        ],
        "entity_anomaly_rows": int(
            len(entity_anomalies)
        ),
        "same_day_multi_game_team_dates": int(
            len(doubleheaders)
        ),
        "outputs": {
            "game_registry": str(
                GAME_ANOMALY_PATH
            ),
            "entity_anomalies": str(
                ENTITY_ANOMALY_PATH
            ),
            "same_day_multi_games": str(
                DOUBLEHEADER_PATH
            ),
        },
        "policy": {
            "raw_data_deleted": False,
            "master_data_modified": False,
            "game_cards_modified": False,
            "unsafe_games_quarantined_only": True,
            "same_day_games_require_conservative_snapshots": True,
        },
    }

    _atomic_json_write(
        summary,
        METADATA_PATH,
    )

    print("=" * 76)
    print("ATLAS DATA VALIDATION & ANOMALY ENGINE")
    print("=" * 76)
    print(
        f"Games Audited................. "
        f"{summary['games_audited']:,}"
    )
    print(
        f"Strict Backtest Safe.......... "
        f"{summary['strict_backtest_safe_games']:,}"
    )
    print(
        f"Quarantined Games............. "
        f"{summary['quarantined_games']:,}"
    )
    print(
        f"Entity Anomaly Rows........... "
        f"{summary['entity_anomaly_rows']:,}"
    )
    print(
        f"Same-Day Multi-Game Team Dates "
        f"{summary['same_day_multi_game_team_dates']:,}"
    )
    print(
        f"Game Registry................. "
        f"{GAME_ANOMALY_PATH}"
    )
    print("=" * 76)

    return summary
