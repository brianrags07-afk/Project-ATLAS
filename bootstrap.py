
import os
import sys
import importlib
from pathlib import Path


def start_atlas(
    code_root="/content/drive/MyDrive/Project_Atlas",
    data_root="/content/drive/MyDrive/Project_Atlas/data",
):
    code_root = Path(code_root).resolve()
    data_root = Path(data_root).resolve()

    if not code_root.exists():
        raise FileNotFoundError(f"ATLAS code root missing: {code_root}")

    if not data_root.exists():
        raise FileNotFoundError(f"ATLAS data root missing: {data_root}")

    os.chdir(code_root)

    # Remove all known competing ATLAS roots.
    blocked_roots = {
        "/content",
        "/content/Project_ATLAS",
        "/content/Project_ATLAS_Check",
        str(code_root),
    }

    sys.path = [p for p in sys.path if p not in blocked_roots]
    sys.path.insert(0, str(code_root))

    # Clear cached ATLAS modules.
    for module_name in list(sys.modules):
        if module_name == "atlas" or module_name.startswith("atlas."):
            del sys.modules[module_name]

    importlib.invalidate_caches()

    import atlas

    atlas_paths = [str(Path(p).resolve()) for p in atlas.__path__]
    expected_atlas_path = str((code_root / "atlas").resolve())

    if atlas_paths != [expected_atlas_path]:
        raise RuntimeError(
            "ATLAS imported from the wrong location. "
            f"Expected {[expected_atlas_path]}, got {atlas_paths}"
        )

    print("=" * 60)
    print("ATLAS BOOTSTRAP")
    print("=" * 60)
    print("Code Root :", code_root)
    print("Data Root :", data_root)
    print("Atlas Path:", atlas_paths)
    print("Code OK   :", (code_root / "atlas").exists())
    print("Data OK   :", data_root.exists())
    print("=" * 60)

    return {
        "code_root": code_root,
        "data_root": data_root,
        "atlas_path": atlas_paths,
    }
