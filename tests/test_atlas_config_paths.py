"""
Fixture tests for atlas/config/paths.py's ATLAS_DATA_ROOT resolution.

These tests confirm every production module that imports from
atlas.config resolves successfully (guarding against the previously
dead/shadowed atlas/config.py module reintroducing an ImportError for
gamecard_engine.py / daily/data_engine.py), and that ATLAS_DATA_ROOT /
ATLAS_CODE_ROOT env vars are honored while the Google Drive path remains
the default when unset.
"""

import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_data_root_is_google_drive_path(monkeypatch):
    monkeypatch.delenv("ATLAS_DATA_ROOT", raising=False)
    monkeypatch.delenv("ATLAS_CODE_ROOT", raising=False)

    import atlas.config.paths as paths

    importlib.reload(paths)

    assert str(paths.DATA_ROOT) == "/content/drive/MyDrive/Project_Atlas/data"
    assert str(paths.CODE_ROOT) == "/content/Project_ATLAS"


def test_atlas_data_root_env_var_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_DATA_ROOT", str(tmp_path))

    import atlas.config.paths as paths

    importlib.reload(paths)

    try:
        assert paths.DATA_ROOT == tmp_path
        assert paths.MASTER_DIR == tmp_path / "master"
        assert paths.GAMECARD_DIR == tmp_path / "daily" / "game_cards"
    finally:
        monkeypatch.delenv("ATLAS_DATA_ROOT", raising=False)
        importlib.reload(paths)


def test_atlas_config_package_exports_all_names_used_by_production_modules():
    """
    Every name any atlas/ production module imports via
    ``from atlas.config import ...`` must actually be exported by the
    atlas.config package (not just by the dead, shadowed legacy
    atlas/config.py module that this repair removed).
    """

    import atlas.config as config

    for name in (
        "DATA_DIR",
        "DATA_ROOT",
        "MASTER_DIR",
        "GAMECARD_DIR",
        "MLB_API",
        "today_str",
    ):
        assert hasattr(config, name), f"atlas.config is missing '{name}'"


def test_dead_shadowed_config_module_was_removed():
    assert not (REPO_ROOT / "atlas" / "config.py").exists()
    assert (REPO_ROOT / "atlas" / "config" / "__init__.py").exists()


def test_gamecard_engine_and_daily_data_engine_import_successfully():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import atlas.gamecards.gamecard_engine; "
            "import atlas.daily.data_engine",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_ensure_dirs_creates_expected_directories(monkeypatch, tmp_path):
    monkeypatch.setenv("ATLAS_DATA_ROOT", str(tmp_path / "atlas_data"))

    import atlas.config.paths as paths

    importlib.reload(paths)

    try:
        paths.ensure_dirs()

        assert paths.DATA_ROOT.exists()
        assert paths.RAW_DIR.exists()
        assert paths.MASTER_DIR.exists()
        assert paths.DAILY_DIR.exists()
        assert paths.SNAPSHOT_DIR.exists()
        assert paths.GAMECARD_DIR.exists()
    finally:
        monkeypatch.delenv("ATLAS_DATA_ROOT", raising=False)
        importlib.reload(paths)


def test_missing_data_root_produces_clear_file_not_found_error(monkeypatch, tmp_path):
    """
    When ATLAS_DATA_ROOT points somewhere with no production artifacts,
    downstream builders must fail with a clear FileNotFoundError rather
    than fabricating data -- the same behavior expected of the bootstrap
    and packaging tooling.
    """

    empty_root = tmp_path / "empty_data_root"
    monkeypatch.setenv("ATLAS_DATA_ROOT", str(empty_root))

    import atlas.config.paths as paths

    importlib.reload(paths)

    try:
        assert not paths.MASTER_GAME_DATABASE.exists()

        import pandas as pd
        import pytest

        with pytest.raises(FileNotFoundError):
            pd.read_parquet(paths.MASTER_GAME_DATABASE)
    finally:
        monkeypatch.delenv("ATLAS_DATA_ROOT", raising=False)
        importlib.reload(paths)
