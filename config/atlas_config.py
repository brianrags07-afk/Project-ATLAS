"""
Project Atlas Configuration
Version: 0.1.0
"""

from pathlib import Path

PROJECT_NAME = "Project Atlas"
PROJECT_VERSION = "0.1.0"

ATLAS_ROOT = Path("/content/drive/MyDrive/Project_Atlas")

# ----------------------------
# Data Directories
# ----------------------------

DATA_DIR = ATLAS_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
STATCAST_DIR = RAW_DIR / "statcast"
WEATHER_DIR = RAW_DIR / "weather"
ODDS_DIR = RAW_DIR / "odds"
LINEUPS_DIR = RAW_DIR / "lineups"
UMPIRES_DIR = RAW_DIR / "umpires"
INJURIES_DIR = RAW_DIR / "injuries"

PROCESSED_DIR = DATA_DIR / "processed"
MASTER_DIR = DATA_DIR / "master"
CACHE_DIR = DATA_DIR / "cache"
BACKUP_DIR = DATA_DIR / "backups"

# ----------------------------
# Project Directories
# ----------------------------

CONFIG_DIR = ATLAS_ROOT / "config"
DOCS_DIR = ATLAS_ROOT / "docs"
MODELS_DIR = ATLAS_ROOT / "models"
REPORTS_DIR = ATLAS_ROOT / "reports"
RESEARCH_DIR = ATLAS_ROOT / "research"

# ----------------------------
# Master Databases
# ----------------------------

MASTER_PITCH_DATABASE = MASTER_DIR / "master_pitch_database.parquet"
MASTER_GAME_DATABASE = MASTER_DIR / "master_game_database.parquet"
MASTER_FEATURE_DATABASE = MASTER_DIR / "master_feature_database.parquet"

# ----------------------------
# Defaults
# ----------------------------

SUPPORTED_SEASONS = [2024, 2025, 2026]

RANDOM_SEED = 42
