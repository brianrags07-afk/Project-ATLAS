
from pathlib import Path

CODE_ROOT = Path("/content/Project_ATLAS")
DATA_ROOT = Path("/content/drive/MyDrive/Project_Atlas/data")

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
