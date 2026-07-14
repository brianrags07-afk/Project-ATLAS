"""
ATLAS Game Intelligence package.

This package implements the central game-by-game baseball
learning pipeline defined in ATLAS_BRAIN_BUILD_CONTRACT.md.
"""

from .contracts import (
    BRAIN_ENGINE_VERSION,
    DATA_ROOT,
    GAME_CARD_CORE_PATH,
    GAME_CARD_MANIFEST_PATH,
    game_event_path,
    INTERACTION_PATH,
    LINEUP_PATH,
    BULLPEN_STATE_PATH,
    TARGET_PATH,
    PITCHER_PREGAME_SNAPSHOT_PATH,
    PITCHER_GAME_FACT_PATH,
    RECONSTRUCTION_OUTPUT_DIR,
    assert_canonical_sources_exist,
)

__all__ = [
    "BRAIN_ENGINE_VERSION",
    "DATA_ROOT",
    "GAME_CARD_CORE_PATH",
    "GAME_CARD_MANIFEST_PATH",
    "game_event_path",
    "INTERACTION_PATH",
    "LINEUP_PATH",
    "BULLPEN_STATE_PATH",
    "TARGET_PATH",
    "PITCHER_PREGAME_SNAPSHOT_PATH",
    "PITCHER_GAME_FACT_PATH",
    "RECONSTRUCTION_OUTPUT_DIR",
    "assert_canonical_sources_exist",
]
