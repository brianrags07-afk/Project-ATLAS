from pathlib import Path
from datetime import datetime
import json

REGISTRY_PATH = Path("/content/drive/MyDrive/Project_Atlas/config/atlas_registry.json")


def load_registry():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry):
    registry["updated_at"] = datetime.now().isoformat()

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def update_counts(**kwargs):
    registry = load_registry()

    for key, value in kwargs.items():
        registry["counts"][key] = value

    save_registry(registry)


def update_engine_status(engine_name, status):
    registry = load_registry()

    registry["engines"][engine_name] = status

    save_registry(registry)


def registry_summary():
    registry = load_registry()

    print("=" * 60)
    print("ATLAS REGISTRY")
    print("=" * 60)

    print("\nCounts")
    for k, v in registry["counts"].items():
        print(f"{k:25} : {v}")

    print("\nEngines")
    for k, v in registry["engines"].items():
        print(f"{k:25} : {v}")

    print("=" * 60)

    return registry
