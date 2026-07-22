
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.config import DATA_DIR, MASTER_DIR


LINEUP_ENGINE_VERSION = "1.1.0"

MASTER_PITCH_PATH = (
    MASTER_DIR
    / "master_pitch_database.parquet"
)

LINEUP_DATA_DIR = (
    DATA_DIR
    / "history"
    / "lineups"
)

HISTORICAL_LINEUP_PATH = (
    LINEUP_DATA_DIR
    / "historical_starting_lineups.parquet"
)

LINEUP_METADATA_PATH = (
    LINEUP_DATA_DIR
    / "historical_starting_lineups_metadata.json"
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


def load_regular_pitch_events() -> pd.DataFrame:
    if not MASTER_PITCH_PATH.exists():
        raise FileNotFoundError(
            f"Missing master pitch database: {MASTER_PITCH_PATH}"
        )

    pitches = pd.read_parquet(
        MASTER_PITCH_PATH
    )

    required = {
        "game_pk",
        "game_date",
        "atlas_season",
        "game_type",
        "home_team",
        "away_team",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "batter",
        "pitcher",
    }

    missing = required - set(pitches.columns)

    if missing:
        raise KeyError(
            f"Pitch database missing columns: {sorted(missing)}"
        )

    game_types = set(
        pitches["game_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    if game_types != {"R"}:
        raise ValueError(
            "Historical Lineup Engine requires the cleaned "
            f"regular-season table. Found: {sorted(game_types)}"
        )

    return pitches


def _prepare_team_batting_events(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    events = pitches[
        [
            "game_pk",
            "game_date",
            "atlas_season",
            "home_team",
            "away_team",
            "inning",
            "inning_topbot",
            "at_bat_number",
            "pitch_number",
            "batter",
            "pitcher",
        ]
    ].copy()

    events["game_date"] = pd.to_datetime(
        events["game_date"],
        errors="coerce",
    )

    top = (
        events["inning_topbot"]
        .astype("string")
        .eq("Top")
    )

    events["team"] = events["away_team"].where(
        top,
        events["home_team"],
    )

    events["opponent"] = events["home_team"].where(
        top,
        events["away_team"],
    )

    events["home_away"] = top.map(
        {
            True: "AWAY",
            False: "HOME",
        }
    )

    events["_half_order"] = (
        events["inning_topbot"]
        .astype("string")
        .map(
            {
                "Top": 0,
                "Bot": 1,
            }
        )
        .fillna(2)
        .astype("int8")
    )

    events = events.sort_values(
        [
            "game_pk",
            "inning",
            "_half_order",
            "at_bat_number",
            "pitch_number",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return events


def _first_pitcher_faced(
    team_events: pd.DataFrame,
) -> int | None:
    values = (
        team_events["pitcher"]
        .dropna()
        .astype("int64")
    )

    if values.empty:
        return None

    return int(values.iloc[0])


def _starting_batter_order(
    team_events: pd.DataFrame,
) -> list[int]:
    plate_appearances = (
        team_events[
            [
                "at_bat_number",
                "batter",
            ]
        ]
        .dropna(subset=["batter"])
        .drop_duplicates(
            subset=["at_bat_number"],
            keep="first",
        )
        .sort_values(
            "at_bat_number",
            kind="stable",
        )
    )

    ordered_unique = []
    seen = set()

    for batter in plate_appearances["batter"]:
        batter_id = int(batter)

        if batter_id in seen:
            continue

        ordered_unique.append(batter_id)
        seen.add(batter_id)

        if len(ordered_unique) == 9:
            break

    return ordered_unique


def build_historical_starting_lineups(
    pitches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pitches is None:
        pitches = load_regular_pitch_events()

    events = _prepare_team_batting_events(
        pitches
    )

    records = []

    for (
        game_pk,
        team,
    ), team_events in events.groupby(
        [
            "game_pk",
            "team",
        ],
        sort=True,
        dropna=False,
    ):
        first = team_events.iloc[0]

        batting_order = _starting_batter_order(
            team_events
        )

        record = {
            "game_pk": int(game_pk),
            "game_date": first["game_date"],
            "atlas_season": int(
                first["atlas_season"]
            ),
            "team": str(team),
            "opponent": str(
                first["opponent"]
            ),
            "home_away": str(
                first["home_away"]
            ),
            "opposing_starting_pitcher_id":
                _first_pitcher_faced(
                    team_events
                ),
            "starting_lineup_size": int(
                len(batting_order)
            ),
            "starting_lineup_complete": (
                len(batting_order) == 9
            ),
            "starting_lineup_ids": batting_order,
            "source": (
                "historical_first_plate_appearance_reconstruction"
            ),
            "pregame_information_class": (
                "postgame_reconstructed_truth_label_not_pregame_evidence"
            ),
            "postgame_truth_label": True,
            "published_lineup_confirmed": False,
            "same_game_pregame_eligible": False,
            "eligible_for_future_game_feature": False,
            "uses_outcome_statistics": False,
            "uses_final_score": False,
            "uses_future_games": False,
            "safe_for_historical_interaction_reconstruction": True,
            "live_prediction_requires_published_lineup": True,
            "lineup_engine_version": LINEUP_ENGINE_VERSION,
        }

        for position in range(1, 10):
            record[
                f"batting_order_{position}_player_id"
            ] = (
                batting_order[position - 1]
                if len(batting_order) >= position
                else pd.NA
            )

        records.append(record)

    lineup_df = pd.DataFrame(
        records
    )

    lineup_df = lineup_df.sort_values(
        [
            "game_date",
            "game_pk",
            "home_away",
        ],
        kind="stable",
    ).reset_index(drop=True)

    integer_columns = [
        "game_pk",
        "atlas_season",
        "opposing_starting_pitcher_id",
        "starting_lineup_size",
    ] + [
        f"batting_order_{position}_player_id"
        for position in range(1, 10)
    ]

    for column in integer_columns:
        lineup_df[column] = pd.to_numeric(
            lineup_df[column],
            errors="coerce",
        ).astype("Int64")

    return lineup_df


def validate_historical_lineups(
    lineup_df: pd.DataFrame,
    pitches: pd.DataFrame,
) -> dict[str, Any]:
    expected_games = int(
        pitches["game_pk"].nunique()
    )

    expected_team_games = (
        expected_games * 2
    )

    duplicate_team_games = int(
        lineup_df.duplicated(
            subset=[
                "game_pk",
                "team",
            ]
        ).sum()
    )

    complete_lineups = int(
        lineup_df[
            "starting_lineup_complete"
        ].sum()
    )

    incomplete_lineups = int(
        (
            ~lineup_df[
                "starting_lineup_complete"
            ]
        ).sum()
    )

    missing_starters = int(
        lineup_df[
            "opposing_starting_pitcher_id"
        ].isna().sum()
    )

    if len(lineup_df) != expected_team_games:
        raise AssertionError(
            f"Expected {expected_team_games:,} team-lineups; "
            f"found {len(lineup_df):,}."
        )

    if duplicate_team_games:
        raise AssertionError(
            f"Found {duplicate_team_games} duplicate team-lineups."
        )

    return {
        "games": expected_games,
        "expected_team_lineups": expected_team_games,
        "actual_team_lineups": int(
            len(lineup_df)
        ),
        "complete_lineups": complete_lineups,
        "incomplete_lineups": incomplete_lineups,
        "complete_pct": (
            complete_lineups
            / len(lineup_df)
            if len(lineup_df)
            else None
        ),
        "missing_opposing_starters": missing_starters,
        "duplicate_team_games": duplicate_team_games,
    }


def run_historical_lineup_engine() -> dict[str, Any]:
    pitches = load_regular_pitch_events()

    lineup_df = build_historical_starting_lineups(
        pitches
    )

    validation = validate_historical_lineups(
        lineup_df,
        pitches,
    )

    _atomic_parquet_write(
        lineup_df,
        HISTORICAL_LINEUP_PATH,
    )

    metadata = {
        "engine": "ATLAS Historical Lineup Engine",
        "engine_version": LINEUP_ENGINE_VERSION,
        "built_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source_dataset": str(
            MASTER_PITCH_PATH
        ),
        "output_path": str(
            HISTORICAL_LINEUP_PATH
        ),
        "validation": validation,
        "pregame_safety": {
            "starting_lineup_was_known_before_game": False,
            "historical_source_is_postgame_event_reconstruction": True,
            "published_source_timestamp_available": False,
            "same_game_pregame_eligible": False,
            "outcome_statistics_used": False,
            "final_score_used": False,
            "future_games_used": False,
            "live_predictions_require_published_lineups": True,
        },
    }

    _atomic_json_write(
        metadata,
        LINEUP_METADATA_PATH,
    )

    print("=" * 72)
    print("ATLAS HISTORICAL LINEUP ENGINE")
    print("=" * 72)
    print(
        f"Games.................... "
        f"{validation['games']:,}"
    )
    print(
        f"Team Lineups............. "
        f"{validation['actual_team_lineups']:,}"
    )
    print(
        f"Complete Starting Nines.. "
        f"{validation['complete_lineups']:,}"
    )
    print(
        f"Incomplete Lineups....... "
        f"{validation['incomplete_lineups']:,}"
    )
    print(
        f"Missing Opposing Starters "
        f"{validation['missing_opposing_starters']:,}"
    )
    print(
        f"Duplicate Team-Games..... "
        f"{validation['duplicate_team_games']:,}"
    )
    print(
        f"Saved To................. "
        f"{HISTORICAL_LINEUP_PATH}"
    )
    print("=" * 72)

    return metadata
