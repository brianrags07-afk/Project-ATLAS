from pathlib import Path
from datetime import datetime
import json


MEMORY_ENGINE_VERSION = "1.0.0"


def timestamp():
    return datetime.now().isoformat()


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json_card(card, path):
    path = ensure_parent(path)

    with open(path, "w") as f:
        json.dump(card, f, indent=2)

    return path


def load_json_card(path):
    with open(path, "r") as f:
        return json.load(f)


def count_json_cards(folder):
    folder = Path(folder)

    if not folder.exists():
        return 0

    return len(list(folder.glob("*.json")))
