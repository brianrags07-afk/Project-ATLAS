"""
GitHub Actions job summary rendering for the ATLAS historical readiness
audit. Pure string formatting -- no Cloud Storage access.
"""

from __future__ import annotations

from typing import Any


def render_job_summary(
    cloud_inventory: dict[str, Any],
    dataset_profiles: dict[str, dict[str, Any]],
    coverage_matrix: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> str:
    lines = ["# ATLAS Historical Readiness Audit", ""]

    lines.append("## Cloud files found")
    lines.append(f"- bucket: `{cloud_inventory.get('bucket')}`")
    lines.append(f"- objects listed: {cloud_inventory.get('object_count')}")
    lines.append(f"- known master files found: {cloud_inventory.get('known_master_files_found')}")
    missing = cloud_inventory.get("known_master_files_missing") or []
    if missing:
        lines.append(f"- **known master files missing:** {missing}")
    lines.append("")

    lines.append("## Seasons detected / games & pitches by season")
    for name, profile in dataset_profiles.items():
        seasons = profile.get("seasons_present", [])
        rows_by_season = profile.get("rows_by_season", {})
        unique_games = profile.get("unique_games_by_season", {})
        lines.append(f"- **{name}**: seasons={seasons}")
        lines.append(f"  - rows_by_season={rows_by_season}")
        if unique_games:
            lines.append(f"  - unique_games_by_season={unique_games}")
    lines.append("")

    lines.append("## Major missing data")
    missing_rows = [r for r in coverage_matrix if r["status"] == "missing"]
    if not missing_rows:
        lines.append("- none detected among audited rows")
    else:
        for r in missing_rows[:30]:
            lines.append(f"- season {r['season']}: `{r['row']}` -- {r['evidence']}")
        if len(missing_rows) > 30:
            lines.append(f"- ... and {len(missing_rows) - 30} more (see historical_coverage_matrix.csv)")
    lines.append("")

    lines.append("## Major leakage risks")
    leakage_rows = [r for r in coverage_matrix if r["status"] == "present_but_not_pregame_safe"]
    if not leakage_rows:
        lines.append("- none detected among audited rows")
    else:
        for r in leakage_rows[:30]:
            lines.append(f"- season {r['season']}: `{r['row']}` -- {r['evidence']}")
        if len(leakage_rows) > 30:
            lines.append(f"- ... and {len(leakage_rows) - 30} more (see historical_coverage_matrix.csv)")
    lines.append("")

    decisions = readiness.get("decisions", {})

    def _fmt_decision(key: str, label: str) -> str:
        d = decisions.get(key, {})
        return f"- **{label}**: {d.get('decision', 'unknown')} -- next action: {d.get('next_action', 'n/a')}"

    lines.append("## 2024 rebuild readiness")
    lines.append(_fmt_decision("B_rebuild_2024_from_raw", "Rebuild 2024 from raw"))
    lines.append(_fmt_decision("A_exact_2024_reproduction", "Exact 2024 reproduction"))
    lines.append("")

    lines.append("## 2025 walk-forward readiness")
    lines.append(_fmt_decision("E_2025_walk_forward_backtest", "2025 walk-forward backtest"))
    lines.append(_fmt_decision("D_parse_2025_identical_transformations", "Parse 2025 w/ identical transforms"))
    lines.append("")

    lines.append("## Pregame Game Card readiness")
    lines.append(_fmt_decision("F_2025_pregame_game_cards", "2025 pregame Game Cards"))
    lines.append("")

    lines.append("## 2026 forward prediction readiness")
    lines.append(_fmt_decision("G_2026_forward_predictions", "2026 forward predictions"))
    lines.append("")

    lines.append("## Exact recommended next step")
    not_ready = [k for k, d in decisions.items() if d.get("decision") == "not_ready"]
    if not_ready:
        first = not_ready[0]
        lines.append(f"- Resolve blockers for `{first}` first: {decisions[first].get('next_action')}")
    else:
        lines.append(
            "- Proceed to build/verify pipeline manifests and Pregame Game Card timestamp proofs "
            "before running any rebuild or backtest (this audit does not authorize either)."
        )
    lines.append("")
    lines.append(
        "_This audit is read-only. No Cloud Storage object was deleted, overwritten, renamed, "
        "moved, or uploaded. No rebuild or backtest was executed._"
    )

    return "\n".join(lines)
