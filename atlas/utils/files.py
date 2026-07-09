
from pathlib import Path
import json
import pandas as pd


def ensure_parent(path):
    """
    Create parent directories if they don't exist.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    """
    Save a JSON file.
    """
    ensure_parent(path)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return Path(path)


def load_json(path):
    """
    Load a JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def save_csv(df, path):
    """
    Save a pandas DataFrame to CSV.
    """
    ensure_parent(path)

    df.to_csv(path, index=False)

    return Path(path)


def load_csv(path):
    """
    Load a CSV into a DataFrame.
    """
    return pd.read_csv(path)


def save_parquet(df, path):
    """
    Save a DataFrame as parquet.
    """
    ensure_parent(path)

    df.to_parquet(path, index=False)

    return Path(path)


def load_parquet(path):
    """
    Load a parquet file.
    """
    return pd.read_parquet(path)
