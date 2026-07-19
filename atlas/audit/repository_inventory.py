"""
Repository inventory for the ATLAS historical readiness audit.

Walks the checked-out repository (never invents files) and reports, for
every relevant module, its public interface, apparent readiness, season
sensitivity, pregame-safety heuristic, Colab/Drive dependencies, and any
hard-coded seasons/paths/bucket names/filenames found in the source text.

All classifications derived here are heuristic. Where evidence is
insufficient the module is labeled ``"unknown"`` rather than guessed.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Directories that are considered "repository code" for inventory purposes.
CODE_DIRS = ("atlas", "scripts", "config", "atlas_reference")
WORKFLOW_DIR = ".github/workflows"
DOCS_DIR = "docs"
TESTS_DIR = "tests"
NOTEBOOK_GLOB = "**/*.ipynb"

FOCUS_AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "master_game_builder": ("master_game", "game_database"),
    "master_pitch_builder": ("master_pitch", "pitch_database"),
    "chronology": ("chronology", "timeline"),
    "team_game_state": ("team_game_state",),
    "scoring_events": ("scoring", "scoring_state"),
    "game_flow": ("game_flow",),
    "offense": ("offense",),
    "contact": ("contact",),
    "discipline": ("discipline",),
    "starter": ("starter",),
    "bullpen": ("bullpen",),
    "team_pitching": ("team_pitching",),
    "identities": ("identity", "identities"),
    "memories": ("memory", "memories"),
    "concepts": ("concept",),
    "feature_registry": ("feature_registry", "registry"),
    "feature_lineage": ("lineage",),
    "validation": ("validation", "validator"),
    "prediction": ("prediction", "predictions"),
    "moneyline": ("moneyline",),
    "totals": ("totals",),
    "run_line": ("run_line", "runline"),
    "player_props": ("player_props", "props"),
    "pregame_snapshots": ("pregame",),
    "daily_pipeline": ("daily",),
    "cloud_sync": ("cloud", "bucket", "gcs", "upload_to_bucket"),
}

HARDCODED_SEASON_RE = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")
BUCKET_RE = re.compile(r"gs://[A-Za-z0-9._-]+(?:/[A-Za-z0-9._/-]*)?")
ABS_PATH_RE = re.compile(
    r"(?:/content/drive/[^\s\"'\)]+|/(?:Users|home)/[^\s\"'\)]+|[A-Za-z]:\\\\[^\s\"'\)]+)"
)
DATE_LITERAL_RE = re.compile(r"\b20[12]\d-[01]\d-[0-3]\d\b")
COLAB_MARKERS = ("google.colab", "drive.mount", "/content/drive")
DEPRECATED_MARKERS = ("deprecated", "do not use", "legacy - unused")
STUB_MARKERS = ("notimplementederror", "todo", "fixme", "not yet implemented")

LIB_IMPORT_ALIASES = {
    "pd": "pandas",
    "np": "numpy",
    "pa": "pyarrow",
    "pq": "pyarrow.parquet",
    "plt": "matplotlib.pyplot",
    "gcs": "google.cloud.storage",
}


@dataclass
class ModuleReport:
    path: str
    public_functions: list[str] = field(default_factory=list)
    public_classes: list[str] = field(default_factory=list)
    module_docstring: str | None = None
    expected_inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    status: str = "unknown"
    season_parameterized: str = "unknown"
    pregame_safety: str = "unknown"
    colab_or_drive_dependency: bool = False
    missing_imports: list[str] = field(default_factory=list)
    hardcoded_seasons: list[str] = field(default_factory=list)
    hardcoded_paths: list[str] = field(default_factory=list)
    hardcoded_bucket_names: list[str] = field(default_factory=list)
    hardcoded_dates: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "public_functions": self.public_functions,
            "public_classes": self.public_classes,
            "module_docstring": self.module_docstring,
            "expected_inputs": self.expected_inputs,
            "outputs": self.outputs,
            "status": self.status,
            "season_parameterized": self.season_parameterized,
            "pregame_safety": self.pregame_safety,
            "colab_or_drive_dependency": self.colab_or_drive_dependency,
            "missing_imports": self.missing_imports,
            "hardcoded_seasons": self.hardcoded_seasons,
            "hardcoded_paths": self.hardcoded_paths,
            "hardcoded_bucket_names": self.hardcoded_bucket_names,
            "hardcoded_dates": self.hardcoded_dates,
            "focus_areas": self.focus_areas,
            "parse_error": self.parse_error,
        }


def _relpath(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _detect_focus_areas(rel_path: str) -> list[str]:
    lowered = rel_path.lower()
    hits = []
    for area, keywords in FOCUS_AREA_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            hits.append(area)
    return hits


def _extract_docstring_returns(func: ast.AST) -> list[str]:
    doc = ast.get_docstring(func) or ""
    outputs = []
    for line in doc.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("returns") or stripped.startswith(":return"):
            outputs.append(line.strip())
    return outputs


def _guess_season_parameterized(source: str, func_and_class_args: list[str]) -> str:
    if any(name in ("season", "seasons", "year", "years") for name in func_and_class_args):
        return "season_parameterized"
    if HARDCODED_SEASON_RE.search(source):
        return "hard_coded"
    return "unknown"


def _guess_pregame_safety(rel_path: str, source: str) -> str:
    lowered = rel_path.lower()
    text = source.lower()
    if "postgame" in lowered or "outcome" in lowered or "final_score" in text:
        return "postgame_only"
    if "pregame" in lowered or "cutoff" in text or "feature_cutoff" in text:
        return "pregame_safe_claimed"
    return "unknown"


def _find_missing_lib_imports(tree: ast.AST, source: str) -> list[str]:
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
    missing = []
    for alias, real_module in LIB_IMPORT_ALIASES.items():
        pattern = re.compile(rf"\b{re.escape(alias)}\.")
        if pattern.search(source) and alias not in imported_names:
            missing.append(f"{alias} (used but not imported; expected `import {real_module} as {alias}`)")
    return missing


def _analyze_python_module(path: Path, rel_path: str) -> ModuleReport:
    report = ModuleReport(path=rel_path)
    report.focus_areas = _detect_focus_areas(rel_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.parse_error = f"unreadable: {exc}"
        return report

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        report.parse_error = f"SyntaxError: {exc}"
        return report

    report.module_docstring = ast.get_docstring(tree)

    all_arg_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            all_arg_names.extend(a.arg for a in node.args.args)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
            report.public_functions.append(node.name)
            report.outputs.extend(_extract_docstring_returns(node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            report.public_classes.append(node.name)

    report.expected_inputs = sorted(set(all_arg_names) - {"self", "cls"})
    report.season_parameterized = _guess_season_parameterized(source, all_arg_names)
    report.pregame_safety = _guess_pregame_safety(rel_path, source)
    report.colab_or_drive_dependency = any(marker in source for marker in COLAB_MARKERS)
    report.missing_imports = _find_missing_lib_imports(tree, source)
    report.hardcoded_seasons = sorted(set(HARDCODED_SEASON_RE.findall(source)))
    report.hardcoded_paths = sorted(set(ABS_PATH_RE.findall(source)))
    report.hardcoded_bucket_names = sorted(set(BUCKET_RE.findall(source)))
    report.hardcoded_dates = sorted(set(DATE_LITERAL_RE.findall(source)))

    lowered_source = source.lower()
    if any(marker in lowered_source for marker in DEPRECATED_MARKERS):
        report.status = "deprecated"
    elif report.colab_or_drive_dependency:
        report.status = "notebook_only"
    elif any(marker in lowered_source for marker in STUB_MARKERS):
        report.status = "partial"
    elif report.public_functions or report.public_classes:
        report.status = "production_ready"
    else:
        report.status = "unknown"

    return report


def _detect_duplicate_implementations(module_reports: list[ModuleReport]) -> dict[str, list[str]]:
    """Map a public symbol name -> list of module paths that define it, for
    symbols defined in more than one module (a signal of duplicated logic)."""
    symbol_to_paths: dict[str, list[str]] = {}
    for report in module_reports:
        for symbol in report.public_functions + report.public_classes:
            symbol_to_paths.setdefault(symbol, []).append(report.path)
    return {symbol: paths for symbol, paths in symbol_to_paths.items() if len(paths) > 1}


def _list_files(repo_root: Path, subdir: str, suffixes: tuple[str, ...]) -> list[str]:
    base = repo_root / subdir
    if not base.exists():
        return []
    results = []
    for suffix in suffixes:
        for path in sorted(base.rglob(f"*{suffix}")):
            if path.is_file():
                results.append(_relpath(path, repo_root))
    return sorted(set(results))


def build_repository_inventory(repo_root: Path) -> dict[str, Any]:
    """Build the full repository inventory dict (used to write both the
    JSON and Markdown reports). Never fabricates entries -- every path
    listed is a file that exists in the checked-out repository."""
    repo_root = Path(repo_root).resolve()

    python_modules: list[ModuleReport] = []
    for code_dir in ("atlas",):
        base = repo_root / code_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if not path.is_file():
                continue
            rel = _relpath(path, repo_root)
            python_modules.append(_analyze_python_module(path, rel))

    scripts_modules: list[ModuleReport] = []
    for path in sorted((repo_root / "scripts").rglob("*.py")) if (repo_root / "scripts").exists() else []:
        if path.is_file():
            scripts_modules.append(_analyze_python_module(path, _relpath(path, repo_root)))

    duplicate_symbols = _detect_duplicate_implementations(python_modules)
    for report in python_modules:
        dup_here = [s for s in report.public_functions + report.public_classes if s in duplicate_symbols]
        if dup_here and report.status == "production_ready":
            report.status = "duplicated"

    workflows = _list_files(repo_root, WORKFLOW_DIR, (".yml", ".yaml"))
    notebooks = [
        _relpath(p, repo_root) for p in sorted(repo_root.rglob("*.ipynb")) if p.is_file()
    ]
    scripts = [
        _relpath(p, repo_root)
        for p in sorted((repo_root / "scripts").rglob("*.py"))
        if (repo_root / "scripts").exists() and p.is_file()
    ] + [
        _relpath(p, repo_root)
        for p in sorted((repo_root / "scripts").rglob("*.sh"))
        if (repo_root / "scripts").exists() and p.is_file()
    ]
    configs = [
        _relpath(p, repo_root)
        for p in sorted((repo_root / "config").rglob("*"))
        if (repo_root / "config").exists() and p.is_file()
    ]
    schemas = [
        _relpath(p, repo_root)
        for p in sorted(repo_root.rglob("*.schema.json"))
        if p.is_file()
    ]
    manifests = [
        _relpath(p, repo_root)
        for p in sorted(repo_root.rglob("*manifest*.json"))
        if p.is_file()
    ]
    tests = _list_files(repo_root, TESTS_DIR, (".py",))
    validation_modules = [r.path for r in python_modules if "atlas/validation" in r.path]
    documentation = _list_files(repo_root, DOCS_DIR, (".md",)) + (
        [_relpath(repo_root / "README.md", repo_root)] if (repo_root / "README.md").exists() else []
    )

    focus_area_index: dict[str, list[str]] = {area: [] for area in FOCUS_AREA_KEYWORDS}
    for report in python_modules:
        for area in report.focus_areas:
            focus_area_index[area].append(report.path)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "atlas_modules": [r.to_dict() for r in python_modules],
        "script_modules": [r.to_dict() for r in scripts_modules],
        "workflows": workflows,
        "notebooks": notebooks,
        "scripts": sorted(set(scripts)),
        "configs": configs,
        "schemas": schemas,
        "manifests": manifests,
        "tests": tests,
        "validation_modules": validation_modules,
        "documentation": sorted(set(documentation)),
        "focus_area_index": focus_area_index,
        "duplicate_symbols": duplicate_symbols,
        "counts": {
            "atlas_modules": len(python_modules),
            "script_modules": len(scripts_modules),
            "workflows": len(workflows),
            "notebooks": len(notebooks),
            "tests": len(tests),
        },
    }


def render_repository_inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = ["# ATLAS Repository Inventory", ""]
    lines.append(f"Generated: {inventory['generated_at_utc']}")
    lines.append("")
    counts = inventory["counts"]
    lines.append("## Counts")
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Focus Area Coverage")
    for area, paths in inventory["focus_area_index"].items():
        lines.append(f"- **{area}**: {len(paths)} module(s)")
        for p in paths:
            lines.append(f"  - `{p}`")
    lines.append("")

    lines.append("## Duplicate Symbols (defined in more than one module)")
    if not inventory["duplicate_symbols"]:
        lines.append("- none detected")
    for symbol, paths in inventory["duplicate_symbols"].items():
        lines.append(f"- `{symbol}` defined in: {', '.join(f'`{p}`' for p in paths)}")
    lines.append("")

    lines.append("## atlas/ Modules")
    for mod in inventory["atlas_modules"]:
        lines.append(f"### `{mod['path']}`")
        lines.append(f"- status: {mod['status']}")
        lines.append(f"- season_parameterized: {mod['season_parameterized']}")
        lines.append(f"- pregame_safety: {mod['pregame_safety']}")
        lines.append(f"- public_functions: {mod['public_functions']}")
        lines.append(f"- public_classes: {mod['public_classes']}")
        lines.append(f"- colab_or_drive_dependency: {mod['colab_or_drive_dependency']}")
        if mod["missing_imports"]:
            lines.append(f"- missing_imports: {mod['missing_imports']}")
        if mod["hardcoded_seasons"]:
            lines.append(f"- hardcoded_seasons: {mod['hardcoded_seasons']}")
        if mod["hardcoded_bucket_names"]:
            lines.append(f"- hardcoded_bucket_names: {mod['hardcoded_bucket_names']}")
        if mod["hardcoded_paths"]:
            lines.append(f"- hardcoded_paths: {mod['hardcoded_paths']}")
        if mod["parse_error"]:
            lines.append(f"- parse_error: {mod['parse_error']}")
        lines.append("")

    lines.append("## Non-code Inventory")
    for key in ("workflows", "notebooks", "scripts", "configs", "schemas", "manifests", "tests", "documentation"):
        lines.append(f"### {key}")
        for item in inventory[key]:
            lines.append(f"- `{item}`")
        lines.append("")

    return "\n".join(lines)


def write_repository_inventory(repo_root: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_repository_inventory(repo_root)
    json_path = output_dir / "repository_inventory.json"
    md_path = output_dir / "repository_inventory.md"
    json_path.write_text(json.dumps(inventory, indent=2, sort_keys=False), encoding="utf-8")
    md_path.write_text(render_repository_inventory_markdown(inventory), encoding="utf-8")
    return json_path, md_path
