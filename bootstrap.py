
import os
import sys
from pathlib import Path


def start_atlas(
    code_root="/content/Project_ATLAS",
    data_root="/content/drive/MyDrive/Project_Atlas/data",
):
    code_root = Path(code_root)
    data_root = Path(data_root)

    os.chdir(code_root)

    sys.path = [p for p in sys.path if p != "/content"]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    for module in list(sys.modules):
        if module == "atlas" or module.startswith("atlas."):
            del sys.modules[module]

    import atlas

    print("=" * 60)
    print("ATLAS BOOTSTRAP")
    print("=" * 60)
    print("Code Root :", code_root)
    print("Data Root :", data_root)
    print("Atlas Path:", list(atlas.__path__))
    print("Code OK   :", (code_root / "atlas").exists())
    print("Data OK   :", data_root.exists())
    print("=" * 60)

    return {
        "code_root": code_root,
        "data_root": data_root,
        "atlas_path": list(atlas.__path__),
    }
