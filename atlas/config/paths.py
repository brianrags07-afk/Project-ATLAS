
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Production default. The ATLAS Google Drive workspace remains a fully
# supported runtime option and is never removed as a default.
_DEFAULT_CODE_ROOT = "/content/Project_ATLAS"
_DEFAULT_DATA_ROOT = "/content/drive/MyDrive/Project_Atlas/data"

# Both roots can be overridden with environment variables so that modules
# can be imported, unit-tested, and run against local or CI fixtures without
# requiring a mounted Google Drive workspace. When the environment variables
# are unset, behavior is unchanged from the original hard-coded production
# paths.
CODE_ROOT = Path(
    os.environ.get(
        "ATLAS_CODE_ROOT",
        _DEFAULT_CODE_ROOT,
    )
)
DATA_ROOT = Path(
    os.environ.get(
        "ATLAS_DATA_ROOT",
        _DEFAULT_DATA_ROOT,
    )
)

MASTER_DIR = DATA_ROOT / "master"

MASTER_GAME_DATABASE = MASTER_DIR / "master_game_database.parquet"
MASTER_PITCH_DATABASE = MASTER_DIR / "master_pitch_database.parquet"
TEAM_GAME_STATE = MASTER_DIR / "team_game_state.parquet"

QUESTIONS_DIR = DATA_ROOT / "questions"
TAXONOMY_DIR = DATA_ROOT / "taxonomy"
EVIDENCE_DIR = DATA_ROOT / "evidence"

HISTORY_DIR = DATA_ROOT / "history"
TEAM_CARDS_DIR = HISTORY_DIR / "teams"
PITCHER_CARDS_DIR = HISTORY_DIR / "pitchers"
PLAYER_CARDS_DIR = HISTORY_DIR / "players"
GAME_CARDS_DIR = HISTORY_DIR / "game_cards"

RAW_DIR = DATA_ROOT / "raw"
DAILY_DIR = DATA_ROOT / "daily"
SNAPSHOT_DIR = DAILY_DIR / "snapshots"
GAMECARD_DIR = DAILY_DIR / "game_cards"

LOCAL_TZ = "America/Chicago"
MLB_API = "https://statsapi.mlb.com/api/v1"


def today_str() -> str:
    return datetime.now(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d")


def ensure_dirs() -> None:
    for path in [
        DATA_ROOT,
        RAW_DIR,
        MASTER_DIR,
        DAILY_DIR,
        SNAPSHOT_DIR,
        GAMECARD_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
