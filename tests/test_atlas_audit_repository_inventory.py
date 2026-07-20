"""
Tests for atlas/audit/repository_inventory.py.

Runs against small synthetic repo trees under tmp_path -- never mutates
the real checked-out repository.
"""

from __future__ import annotations

import json
from pathlib import Path

from atlas.audit.repository_inventory import (
    build_repository_inventory,
    write_repository_inventory,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_repository_inventory_detects_hardcoded_bucket_and_season(tmp_path: Path):
    _write(
        tmp_path / "atlas" / "cloud_sync" / "uploader.py",
        '''
"""Uploads season 2024 data to the bucket."""

BUCKET = "gs://atlas-mlb-data-brian-4817/data/master"


def upload_season(season=2024):
    return BUCKET
''',
    )
    inventory = build_repository_inventory(tmp_path)
    modules = {m["path"]: m for m in inventory["atlas_modules"]}
    mod = modules["atlas/cloud_sync/uploader.py"]
    assert "gs://atlas-mlb-data-brian-4817/data/master" in mod["hardcoded_bucket_names"]
    assert "2024" in mod["hardcoded_seasons"]
    assert mod["season_parameterized"] == "season_parameterized"
    assert "cloud_sync" in mod["focus_areas"]


def test_build_repository_inventory_detects_colab_drive_dependency(tmp_path: Path):
    _write(
        tmp_path / "atlas" / "learning" / "notebook_helper.py",
        '''
from google.colab import drive

def mount():
    drive.mount("/content/drive")
''',
    )
    inventory = build_repository_inventory(tmp_path)
    modules = {m["path"]: m for m in inventory["atlas_modules"]}
    mod = modules["atlas/learning/notebook_helper.py"]
    assert mod["colab_or_drive_dependency"] is True
    assert mod["status"] == "notebook_only"


def test_build_repository_inventory_detects_missing_import():
    from atlas.audit.repository_inventory import _analyze_python_module
    import ast
    import tempfile

    source = "def build():\n    return pd.DataFrame()\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source)
        f.flush()
        report = _analyze_python_module(Path(f.name), "fake/module.py")
    assert any("pd" in m for m in report.missing_imports)


def test_build_repository_inventory_detects_duplicate_symbols(tmp_path: Path):
    _write(tmp_path / "atlas" / "game_cards" / "engine.py", "def build_card():\n    pass\n")
    _write(tmp_path / "atlas" / "gamecards" / "engine.py", "def build_card():\n    pass\n")
    inventory = build_repository_inventory(tmp_path)
    assert "build_card" in inventory["duplicate_symbols"]
    assert len(inventory["duplicate_symbols"]["build_card"]) == 2


def test_build_repository_inventory_handles_syntax_errors_gracefully(tmp_path: Path):
    _write(tmp_path / "atlas" / "broken" / "bad.py", "def broken(:\n    pass\n")
    inventory = build_repository_inventory(tmp_path)
    modules = {m["path"]: m for m in inventory["atlas_modules"]}
    mod = modules["atlas/broken/bad.py"]
    assert mod["parse_error"] is not None


def test_write_repository_inventory_writes_json_and_markdown(tmp_path: Path):
    _write(tmp_path / "atlas" / "pregame" / "sample.py", '"""Doc."""\n\ndef build_card():\n    pass\n')
    output_dir = tmp_path / "artifacts" / "audits"
    json_path, md_path = write_repository_inventory(tmp_path, output_dir)
    assert json_path.exists()
    assert md_path.exists()
    data = json.loads(json_path.read_text())
    assert data["counts"]["atlas_modules"] == 1
    assert "ATLAS Repository Inventory" in md_path.read_text()


def test_real_repository_inventory_does_not_invent_modules():
    """Sanity check against the real repo: every path reported must exist."""
    repo_root = Path(__file__).resolve().parents[1]
    inventory = build_repository_inventory(repo_root)
    for mod in inventory["atlas_modules"]:
        assert (repo_root / mod["path"]).exists()
    for workflow in inventory["workflows"]:
        assert (repo_root / workflow).exists()
