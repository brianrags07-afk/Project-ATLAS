
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas.config import DATA_DIR
from atlas.pitchers.v2.cards import build_pitcher_card
from atlas.pitchers.v2.definitions import (
    PITCHER_CARD_VERSION,
    PITCHER_ENGINE_VERSION,
)
from atlas.pitchers.v2.pitch_table import (
    build_pitcher_pitch_table,
    load_master_pitches,
)


PITCHER_DIR = (
    DATA_DIR
    / "history"
    / "pitchers"
)

BUILD_PREFIX = "pitchers_build_"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def save_pitcher_card(
    card: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pitcher_id = int(
        card["metadata"]["pitcher_id"]
    )

    destination = (
        output_dir
        / f"{pitcher_id}.json"
    )

    temporary = destination.with_suffix(
        ".json.tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _json_safe(card),
            file,
            indent=2,
        )

    temporary.replace(destination)

    return destination


def _valid_card_ids(
    directory: Path,
) -> set[int]:
    if not directory.exists():
        return set()

    ids = set()

    for path in directory.glob("*.json"):
        if path.stem.isdigit():
            ids.add(int(path.stem))

    return ids


def _new_build_directory() -> Path:
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    directory = (
        PITCHER_DIR.parent
        / f"{BUILD_PREFIX}{timestamp}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    return directory


def _latest_build_directory() -> Path | None:
    candidates = sorted(
        PITCHER_DIR.parent.glob(
            f"{BUILD_PREFIX}*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates[0] if candidates else None


def _validate_card_file(
    path: Path,
    expected_pitcher_id: int,
) -> None:
    with open(
        path,
        encoding="utf-8",
    ) as file:
        card = json.load(file)

    actual_id = int(
        card["metadata"]["pitcher_id"]
    )

    if actual_id != expected_pitcher_id:
        raise AssertionError(
            f"Pitcher ID mismatch in {path}: "
            f"expected {expected_pitcher_id}, "
            f"found {actual_id}"
        )

    if (
        card["metadata"]["pitcher_card_version"]
        != PITCHER_CARD_VERSION
    ):
        raise AssertionError(
            f"Unexpected card version in {path}"
        )

    if not card["metadata"].get(
        "regular_season_only"
    ):
        raise AssertionError(
            f"Non-regular card detected: {path}"
        )

    if (
        card["traceability"]["source_event_rows"]
        != card["overall"]["pitches"]
    ):
        raise AssertionError(
            f"Traceability mismatch in {path}"
        )


def run_pitcher_engine_v2(
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    pitches = load_master_pitches()

    pitcher_table = (
        build_pitcher_pitch_table(
            pitches
        )
    )

    all_pitcher_ids = sorted(
        int(value)
        for value in pitcher_table[
            "pitcher_id"
        ].dropna().unique()
    )

    target_ids = (
        all_pitcher_ids[:limit]
        if limit is not None
        else all_pitcher_ids
    )

    if limit is not None:
        staging_dir = _new_build_directory()
    else:
        latest = (
            _latest_build_directory()
            if resume
            else None
        )

        staging_dir = (
            latest
            if latest is not None
            else _new_build_directory()
        )

    completed_ids = (
        _valid_card_ids(staging_dir)
        & set(target_ids)
    )

    remaining_ids = [
        pitcher_id
        for pitcher_id in target_ids
        if pitcher_id not in completed_ids
    ]

    grouped = pitcher_table.groupby(
        "pitcher_id",
        sort=True,
    )

    print("=" * 72)
    print("ATLAS PITCHER ENGINE V2")
    print("=" * 72)
    print(
        f"Master Pitch Rows....... "
        f"{len(pitches):,}"
    )
    print(
        f"Pitcher Event Rows...... "
        f"{len(pitcher_table):,}"
    )
    print(
        f"Unique Pitchers......... "
        f"{len(all_pitcher_ids):,}"
    )
    print(
        f"Target Pitchers......... "
        f"{len(target_ids):,}"
    )
    print(
        f"Already Complete........ "
        f"{len(completed_ids):,}"
    )
    print(
        f"Remaining............... "
        f"{len(remaining_ids):,}"
    )
    print(
        f"Staging Directory....... "
        f"{staging_dir}"
    )
    print("=" * 72)

    newly_built = 0

    for pitcher_id in remaining_ids:
        pitcher_df = grouped.get_group(
            pitcher_id
        )

        card = build_pitcher_card(
            pitcher_df
        )

        path = save_pitcher_card(
            card,
            output_dir=staging_dir,
        )

        _validate_card_file(
            path,
            expected_pitcher_id=pitcher_id,
        )

        newly_built += 1

        if (
            newly_built % 25 == 0
            or newly_built == len(remaining_ids)
        ):
            complete_now = len(
                _valid_card_ids(
                    staging_dir
                )
                & set(target_ids)
            )

            print(
                f"Progress: "
                f"{complete_now:,}/"
                f"{len(target_ids):,}"
            )

    final_ids = (
        _valid_card_ids(staging_dir)
        & set(target_ids)
    )

    missing_ids = sorted(
        set(target_ids) - final_ids
    )

    if missing_ids:
        raise AssertionError(
            f"Pitcher build incomplete. "
            f"Missing {len(missing_ids)} cards. "
            f"First missing IDs: {missing_ids[:10]}"
        )

    final_output_dir = staging_dir
    backup_dir = None

    if limit is None:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        backup_dir = (
            PITCHER_DIR.parent
            / "backups"
            / f"pitchers_pre_v2_{timestamp}"
        )

        if (
            PITCHER_DIR.exists()
            and PITCHER_DIR.resolve()
            != staging_dir.resolve()
        ):
            backup_dir.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            PITCHER_DIR.replace(
                backup_dir
            )

        if (
            staging_dir.resolve()
            != PITCHER_DIR.resolve()
        ):
            staging_dir.replace(
                PITCHER_DIR
            )

        final_output_dir = PITCHER_DIR

    summary = {
        "engine": "ATLAS Pitcher Engine",
        "engine_version": (
            PITCHER_ENGINE_VERSION
        ),
        "card_version": (
            PITCHER_CARD_VERSION
        ),
        "master_pitch_rows": int(
            len(pitches)
        ),
        "pitcher_event_rows": int(
            len(pitcher_table)
        ),
        "unique_pitchers": int(
            len(all_pitcher_ids)
        ),
        "cards_built": int(
            len(target_ids)
        ),
        "newly_built": int(
            newly_built
        ),
        "regular_season_only": True,
        "output_directory": str(
            final_output_dir
        ),
        "backup_directory": (
            str(backup_dir)
            if (
                backup_dir is not None
                and backup_dir.exists()
            )
            else None
        ),
    }

    print("\n" + "=" * 72)
    print("PITCHER ENGINE V2 COMPLETE")
    print("=" * 72)
    print(
        f"Pitcher Cards Built..... "
        f"{summary['cards_built']:,}"
    )
    print(
        f"Newly Built............. "
        f"{summary['newly_built']:,}"
    )
    print(
        f"Saved To................ "
        f"{summary['output_directory']}"
    )
    print("=" * 72)

    return summary
