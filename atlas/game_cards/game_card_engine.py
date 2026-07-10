
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.config.paths import (
    MASTER_GAME_DATABASE,
    MASTER_PITCH_DATABASE,
    DATA_ROOT,
)


GAME_CARD_ENGINE_VERSION = "1.0.0"

GAME_CARD_ROOT = DATA_ROOT / "history" / "game_cards"
CORE_PATH = GAME_CARD_ROOT / "game_card_core.parquet"
MANIFEST_PATH = GAME_CARD_ROOT / "game_card_manifest.parquet"
METADATA_PATH = GAME_CARD_ROOT / "game_card_recorder_metadata.json"
EVENT_STORE_DIR = GAME_CARD_ROOT / "events"


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination = Path(destination)
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
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )

    temporary.replace(destination)


def _validate_regular_data(
    games: pd.DataFrame,
    pitches: pd.DataFrame,
) -> None:
    required_game_columns = {
        "game_pk",
        "game_date",
        "atlas_season",
        "game_type",
    }

    required_pitch_columns = {
        "game_pk",
        "game_date",
        "atlas_season",
        "game_type",
    }

    missing_games = (
        required_game_columns - set(games.columns)
    )
    missing_pitches = (
        required_pitch_columns - set(pitches.columns)
    )

    if missing_games:
        raise KeyError(
            f"Master game database missing columns: "
            f"{sorted(missing_games)}"
        )

    if missing_pitches:
        raise KeyError(
            f"Master pitch database missing columns: "
            f"{sorted(missing_pitches)}"
        )

    game_types = set(
        games["game_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    pitch_types = set(
        pitches["game_type"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    if game_types != {"R"}:
        raise ValueError(
            f"Game Card Recorder requires regular-season games only. "
            f"Found game types: {sorted(game_types)}"
        )

    if pitch_types != {"R"}:
        raise ValueError(
            f"Game Card Recorder requires regular-season pitches only. "
            f"Found game types: {sorted(pitch_types)}"
        )

    if games["game_pk"].duplicated().any():
        raise ValueError(
            "Duplicate game_pk values found in master games."
        )

    game_ids = set(
        games["game_pk"]
        .dropna()
        .astype(int)
    )

    pitch_game_ids = set(
        pitches["game_pk"]
        .dropna()
        .astype(int)
    )

    orphan_pitch_games = pitch_game_ids - game_ids

    if orphan_pitch_games:
        raise ValueError(
            f"Pitch rows reference {len(orphan_pitch_games)} "
            "games absent from the master game database."
        )


def _event_sort_columns(
    pitches: pd.DataFrame,
) -> list[str]:
    candidates = [
        "game_pk",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "sv_id",
    ]

    return [
        column
        for column in candidates
        if column in pitches.columns
    ]


def _add_event_order(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    pitches = pitches.copy()

    if "inning_topbot" in pitches.columns:
        pitches["_atlas_half_order"] = (
            pitches["inning_topbot"]
            .astype("string")
            .map({"Top": 0, "Bot": 1})
            .fillna(2)
            .astype("int8")
        )

    sort_columns = [
        column
        for column in [
            "game_pk",
            "inning",
            "_atlas_half_order",
            "at_bat_number",
            "pitch_number",
            "sv_id",
        ]
        if column in pitches.columns
    ]

    if sort_columns:
        pitches = pitches.sort_values(
            sort_columns,
            kind="stable",
        )

    pitches["atlas_event_order"] = (
        pitches.groupby("game_pk")
        .cumcount()
        .add(1)
        .astype("int32")
    )

    pitches["atlas_event_id"] = (
        pitches["game_pk"].astype("Int64").astype(str)
        + "_"
        + pitches["atlas_event_order"].astype(str)
    )

    if "_atlas_half_order" in pitches.columns:
        pitches = pitches.drop(columns=["_atlas_half_order"])

    return pitches


def _build_game_counts(
    pitches: pd.DataFrame,
) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "event_row_count": (
            "atlas_event_order",
            "count",
        ),
    }

    if "at_bat_number" in pitches.columns:
        aggregations["plate_appearance_count"] = (
            "at_bat_number",
            "nunique",
        )

    if "pitcher" in pitches.columns:
        aggregations["pitcher_count"] = (
            "pitcher",
            "nunique",
        )

    if "batter" in pitches.columns:
        aggregations["batter_count"] = (
            "batter",
            "nunique",
        )

    counts = (
        pitches.groupby("game_pk")
        .agg(**aggregations)
        .reset_index()
    )

    for column in counts.columns:
        if column != "game_pk":
            counts[column] = (
                counts[column]
                .fillna(0)
                .astype("int64")
            )

    return counts


def _write_event_stores(
    pitches: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    EVENT_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    season_outputs: dict[int, dict[str, Any]] = {}

    for season in sorted(
        pitches["atlas_season"]
        .dropna()
        .astype(int)
        .unique()
    ):
        season_events = pitches[
            pitches["atlas_season"].astype(int) == season
        ].copy()

        destination = (
            EVENT_STORE_DIR
            / f"game_events_{season}_regular.parquet"
        )

        _atomic_parquet_write(
            season_events,
            destination,
        )

        season_outputs[int(season)] = {
            "path": str(destination),
            "rows": int(len(season_events)),
            "games": int(
                season_events["game_pk"].nunique()
            ),
            "columns": int(
                len(season_events.columns)
            ),
        }

    return season_outputs


def build_game_card_recorder(
    game_database_path: Path | str = MASTER_GAME_DATABASE,
    pitch_database_path: Path | str = MASTER_PITCH_DATABASE,
) -> dict[str, Any]:
    game_database_path = Path(game_database_path)
    pitch_database_path = Path(pitch_database_path)

    GAME_CARD_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    games = pd.read_parquet(
        game_database_path
    )

    pitches = pd.read_parquet(
        pitch_database_path
    )

    _validate_regular_data(
        games,
        pitches,
    )

    pitches = _add_event_order(
        pitches
    )

    game_counts = _build_game_counts(
        pitches
    )

    core = games.merge(
        game_counts,
        on="game_pk",
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        "event_row_count",
        "plate_appearance_count",
        "pitcher_count",
        "batter_count",
    ]

    for column in count_columns:
        if column not in core.columns:
            core[column] = 0

        core[column] = (
            core[column]
            .fillna(0)
            .astype("int64")
        )

    core["game_card_schema_version"] = "1.0.0"
    core["game_card_engine_version"] = (
        GAME_CARD_ENGINE_VERSION
    )
    core["pregame_safe_section_available"] = True
    core["postgame_section_available"] = True
    core["event_store_available"] = (
        core["event_row_count"] > 0
    )

    season_outputs = _write_event_stores(
        pitches
    )

    core["event_store_path"] = core[
        "atlas_season"
    ].astype(int).map(
        {
            season: output["path"]
            for season, output
            in season_outputs.items()
        }
    )

    core["event_lookup_key"] = (
        core["game_pk"]
        .astype("Int64")
        .astype(str)
    )

    manifest_columns = [
        "game_pk",
        "game_date",
        "atlas_season",
        "game_type",
        "atlas_game_type",
        "event_row_count",
        "plate_appearance_count",
        "pitcher_count",
        "batter_count",
        "event_store_path",
        "event_lookup_key",
        "game_card_schema_version",
        "game_card_engine_version",
    ]

    manifest_columns = [
        column
        for column in manifest_columns
        if column in core.columns
    ]

    manifest = core[
        manifest_columns
    ].copy()

    # --------------------------------------------------------
    # Final integrity checks
    # --------------------------------------------------------

    if len(core) != len(games):
        raise AssertionError(
            "Game-card core row count does not equal "
            "master game count."
        )

    if core["game_pk"].duplicated().any():
        raise AssertionError(
            "Duplicate game cards were created."
        )

    if manifest["game_pk"].nunique() != len(games):
        raise AssertionError(
            "Manifest does not contain every game."
        )

    if int(core["event_row_count"].sum()) != len(pitches):
        raise AssertionError(
            "Game event counts do not reconcile with "
            "master pitch database rows."
        )

    if pitches["atlas_event_id"].duplicated().any():
        raise AssertionError(
            "Duplicate event IDs were generated."
        )

    _atomic_parquet_write(
        core,
        CORE_PATH,
    )

    _atomic_parquet_write(
        manifest,
        MANIFEST_PATH,
    )

    metadata = {
        "engine": "ATLAS Game Card Recorder",
        "engine_version": GAME_CARD_ENGINE_VERSION,
        "schema_version": "1.0.0",
        "built_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "source": {
            "master_game_database": str(
                game_database_path
            ),
            "master_pitch_database": str(
                pitch_database_path
            ),
        },
        "regular_season_only": True,
        "game_type_filter": "R",
        "games": int(len(core)),
        "event_rows": int(len(pitches)),
        "event_columns_preserved": int(
            len(pitches.columns)
        ),
        "core_columns_preserved": int(
            len(core.columns)
        ),
        "seasons": season_outputs,
        "outputs": {
            "core": str(CORE_PATH),
            "manifest": str(MANIFEST_PATH),
            "event_store_directory": str(
                EVENT_STORE_DIR
            ),
        },
        "raw_statcast_modified": False,
        "event_data_reduced": False,
        "all_master_pitch_columns_preserved": True,
    }

    _atomic_json_write(
        metadata,
        METADATA_PATH,
    )

    print("=" * 72)
    print("ATLAS GAME CARD RECORDER")
    print("=" * 72)
    print(f"Games Preserved       : {len(core):,}")
    print(f"Event Rows Preserved  : {len(pitches):,}")
    print(f"Event Columns         : {len(pitches.columns):,}")
    print(f"Core Columns          : {len(core.columns):,}")
    print(f"Core Path             : {CORE_PATH}")
    print(f"Manifest Path         : {MANIFEST_PATH}")
    print(f"Event Store Directory : {EVENT_STORE_DIR}")
    print("=" * 72)

    return metadata


def load_game_card(
    game_pk: int,
    include_events: bool = True,
) -> dict[str, Any]:
    game_pk = int(game_pk)

    if not CORE_PATH.exists():
        raise FileNotFoundError(
            f"Game-card core does not exist: {CORE_PATH}"
        )

    core = pd.read_parquet(
        CORE_PATH,
        filters=[
            ("game_pk", "==", game_pk),
        ],
    )

    if core.empty:
        raise KeyError(
            f"No Game Card found for game_pk={game_pk}"
        )

    if len(core) != 1:
        raise ValueError(
            f"Expected one Game Card for game_pk={game_pk}; "
            f"found {len(core)}."
        )

    core_record = core.iloc[0].to_dict()

    card: dict[str, Any] = {
        "game_pk": game_pk,
        "core": core_record,
        "events": None,
    }

    if include_events:
        event_store_path = Path(
            core_record["event_store_path"]
        )

        if not event_store_path.exists():
            raise FileNotFoundError(
                f"Event store missing: {event_store_path}"
            )

        events = pd.read_parquet(
            event_store_path,
            filters=[
                ("game_pk", "==", game_pk),
            ],
        )

        events = events.sort_values(
            "atlas_event_order",
            kind="stable",
        ).reset_index(drop=True)

        expected_events = int(
            core_record["event_row_count"]
        )

        if len(events) != expected_events:
            raise AssertionError(
                f"Event-count mismatch for game_pk={game_pk}: "
                f"expected {expected_events}, found {len(events)}."
            )

        card["events"] = events

    return card
