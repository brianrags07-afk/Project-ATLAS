"""
Canonical data contracts for the ATLAS Baseball Brain.

This module contains paths and source-level invariants only.
It does not reconstruct games, create predictions or update
identities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


BRAIN_ENGINE_VERSION: Final[str] = "1.0.0"

REPO_ROOT: Final[Path] = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

DATA_ROOT: Final[Path] = (
    REPO_ROOT / "data"
)

GAME_CARD_ROOT: Final[Path] = (
    DATA_ROOT / "history" / "game_cards"
)

GAME_CARD_CORE_PATH: Final[Path] = (
    GAME_CARD_ROOT / "game_card_core.parquet"
)

GAME_CARD_MANIFEST_PATH: Final[Path] = (
    GAME_CARD_ROOT / "game_card_manifest.parquet"
)

GAME_EVENT_DIR: Final[Path] = (
    GAME_CARD_ROOT / "events"
)

INTERACTION_PATH: Final[Path] = (
    DATA_ROOT
    / "pregame"
    / "interactions"
    / "lineup_starter_bullpen_inputs.parquet"
)

LINEUP_PATH: Final[Path] = (
    DATA_ROOT
    / "history"
    / "lineups"
    / "historical_starting_lineups.parquet"
)

BULLPEN_STATE_PATH: Final[Path] = (
    DATA_ROOT
    / "pregame"
    / "bullpen"
    / "bullpen_pregame_state.parquet"
)

TARGET_PATH: Final[Path] = (
    DATA_ROOT
    / "backtest"
    / "targets"
    / "game_targets.parquet"
)

PITCHER_PREGAME_SNAPSHOT_PATH: Final[Path] = (
    DATA_ROOT
    / "pregame"
    / "snapshots"
    / "pitcher_pregame_snapshots.parquet"
)

PITCHER_GAME_FACT_PATH: Final[Path] = (
    DATA_ROOT
    / "pregame"
    / "snapshots"
    / "pitcher_game_facts.parquet"
)

BRAIN_OUTPUT_ROOT: Final[Path] = (
    DATA_ROOT / "game_intelligence"
)

RECONSTRUCTION_OUTPUT_DIR: Final[Path] = (
    BRAIN_OUTPUT_ROOT / "reconstruction"
)

RECONSTRUCTION_AUDIT_DIR: Final[Path] = (
    BRAIN_OUTPUT_ROOT / "audits"
)


def game_event_path(
    season: int,
    game_type: str = "regular",
) -> Path:
    """
    Return the canonical event-store path for a season.
    """
    normalized_game_type = (
        str(game_type)
        .strip()
        .lower()
    )

    if normalized_game_type != "regular":
        raise ValueError(
            "Phase 1 currently supports regular-season "
            "event stores only."
        )

    return (
        GAME_EVENT_DIR
        / f"game_events_{int(season)}_regular.parquet"
    )


def canonical_source_paths(
    season: int,
) -> dict[str, Path]:
    """
    Return all canonical Phase 1 source paths.
    """
    return {
        "game_card_core":
            GAME_CARD_CORE_PATH,
        "game_card_manifest":
            GAME_CARD_MANIFEST_PATH,
        "game_events":
            game_event_path(season),
        "pregame_interactions":
            INTERACTION_PATH,
        "historical_lineups":
            LINEUP_PATH,
        "bullpen_pregame_state":
            BULLPEN_STATE_PATH,
        "game_targets":
            TARGET_PATH,
        "pitcher_pregame_snapshots":
            PITCHER_PREGAME_SNAPSHOT_PATH,
        "pitcher_game_facts":
            PITCHER_GAME_FACT_PATH,
    }


def assert_canonical_sources_exist(
    season: int,
    require_pitcher_sources: bool = False,
) -> dict[str, Path]:
    """
    Validate that required canonical sources exist.

    Pitcher snapshot products are optional during the first
    reconstruction milestone because the 859-column interaction
    table already contains starter information.
    """
    paths = canonical_source_paths(
        season=season,
    )

    required_names = {
        "game_card_core",
        "game_card_manifest",
        "game_events",
        "pregame_interactions",
        "historical_lineups",
        "bullpen_pregame_state",
        "game_targets",
    }

    if require_pitcher_sources:
        required_names.update({
            "pitcher_pregame_snapshots",
            "pitcher_game_facts",
        })

    missing = {
        name: path
        for name, path in paths.items()
        if (
            name in required_names
            and not path.exists()
        )
    }

    if missing:
        formatted = "\n".join(
            f"- {name}: {path}"
            for name, path in missing.items()
        )

        raise FileNotFoundError(
            "Missing canonical ATLAS sources:\n"
            f"{formatted}"
        )

    return paths


PHASE_1_INVARIANTS: Final[dict[str, object]] = {
    "learning_season": 2024,
    "normal_game_team_rows": 2,
    "normal_game_lineup_rows": 2,
    "normal_game_bullpen_rows": 2,
    "normal_game_core_rows": 1,
    "normal_game_target_rows": 1,
    "score_sources_must_agree": True,
    "pregame_current_game_outcome_used": False,
    "pregame_future_games_used": False,
    "identity_updates_allowed": False,
    "predictions_allowed": False,
    "known_exception_game_pks": [746942],
}
