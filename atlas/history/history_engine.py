from pathlib import Path
import pandas as pd

from atlas.config import MASTER_DIR

HISTORY_ENGINE_VERSION = "1.0.0"


def load_master_games():
    """
    Load the master game database.
    """
    path = MASTER_DIR / "master_game_database.parquet"

    games = pd.read_parquet(path)

    print("=" * 60)
    print("ATLAS HISTORY ENGINE")
    print("=" * 60)
    print(f"Rows.............. {len(games):,}")
    print(f"Columns........... {len(games.columns)}")
    print("=" * 60)

    return games


def preview_games(limit=5):
    """
    Preview the first few historical games.
    """
    games = load_master_games()

    return games.head(limit)
