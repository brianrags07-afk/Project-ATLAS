from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ATLAS_VERSION = "3.0.0"

PROJECT_ROOT = Path("/content/drive/MyDrive/Project_Atlas")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MASTER_DIR = DATA_DIR / "master"
DAILY_DIR = DATA_DIR / "daily"

SNAPSHOT_DIR = DAILY_DIR / "snapshots"
GAMECARD_DIR = DAILY_DIR / "game_cards"

LOCAL_TZ = "America/Chicago"
MLB_API = "https://statsapi.mlb.com/api/v1"

def today_str():
    return datetime.now(ZoneInfo(LOCAL_TZ)).strftime("%Y-%m-%d")

def ensure_dirs():
    for path in [
        DATA_DIR,
        RAW_DIR,
        MASTER_DIR,
        DAILY_DIR,
        SNAPSHOT_DIR,
        GAMECARD_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
