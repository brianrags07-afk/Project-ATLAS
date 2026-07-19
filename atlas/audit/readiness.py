"""
Readiness decisions for the ATLAS historical readiness audit.

Determines, from the coverage matrix and dataset profiles only, whether
ATLAS is ready for each of the seven historical-readiness questions
(A-G). Every decision carries evidence, missing requirements, risks, and
an exact next action. A fully-populated season table is explicitly *not*
treated as automatically pregame-safe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_DECISIONS = ("ready", "ready_with_warnings", "not_ready", "unknown")

DECISION_KEYS = (
    "A_exact_2024_reproduction",
    "B_rebuild_2024_from_raw",
    "C_freeze_2024_learned_artifacts",
    "D_parse_2025_identical_transformations",
    "E_2025_walk_forward_backtest",
    "F_2025_pregame_game_cards",
    "G_2026_forward_predictions",
)


def _rows_for(matrix: list[dict[str, Any]], season: int, row: str | None = None) -> list[dict[str, Any]]:
    return [
        r for r in matrix
        if r["season"] == season and (row is None or r["row"] == row)
    ]


def _statuses_for(matrix: list[dict[str, Any]], season: int, rows: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for row in rows:
        matches = _rows_for(matrix, season, row)
        result[row] = matches[0]["status"] if matches else "unknown"
    return result


def _decide(statuses: dict[str, str], required_complete: tuple[str, ...]) -> str:
    values = [statuses.get(r, "unknown") for r in required_complete]
    if any(v == "missing" for v in values):
        return "not_ready"
    if all(v == "complete" for v in values):
        return "ready"
    if any(v == "unknown" for v in values):
        return "unknown"
    return "ready_with_warnings"


def build_readiness_decisions(
    coverage_matrix: list[dict[str, Any]],
    dataset_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}

    core_2024 = _statuses_for(
        coverage_matrix, 2024,
        ("published_schedule", "game_identifiers", "final_scores", "pitch_by_pitch"),
    )
    pregame_2024 = _statuses_for(
        coverage_matrix, 2024,
        ("starters", "lineups", "bullpen_usage", "weather", "rest", "travel"),
    )
    core_2025 = _statuses_for(
        coverage_matrix, 2025,
        ("published_schedule", "game_identifiers", "final_scores", "pitch_by_pitch"),
    )
    pregame_2025 = _statuses_for(
        coverage_matrix, 2025,
        ("starters", "lineups", "bullpen_usage", "weather", "rest", "travel"),
    )
    market_2025 = _statuses_for(coverage_matrix, 2025, ("opening_market", "closing_market"))
    core_2026 = _statuses_for(
        coverage_matrix, 2026,
        ("published_schedule", "game_identifiers"),
    )

    # A. Exact 2024 reproduction: requires complete raw + complete
    # reproduction of learned-artifact lineage (concept discovery, model
    # artifacts). We only have evidence of the raw tables here.
    concept_2024 = _statuses_for(coverage_matrix, 2024, ("concept_discovery", "model_artifacts"))
    decisions["A_exact_2024_reproduction"] = {
        "decision": _decide({**core_2024, **concept_2024}, tuple(core_2024) + tuple(concept_2024)),
        "evidence": {**core_2024, **concept_2024},
        "missing_requirements": [
            k for k, v in {**core_2024, **concept_2024}.items() if v in ("missing", "unknown")
        ],
        "risks": [
            "This audit cannot confirm bit-for-bit lineage of previously learned/frozen artifacts "
            "without a pipeline manifest recording source hashes and versions.",
        ],
        "next_action": (
            "Locate or produce pipeline manifests for the original 2024 runs "
            "(schemas/pipeline_manifest.schema.json) before attempting exact reproduction."
        ),
    }

    decisions["B_rebuild_2024_from_raw"] = {
        "decision": _decide(core_2024, tuple(core_2024)),
        "evidence": core_2024,
        "missing_requirements": [k for k, v in core_2024.items() if v in ("missing", "unknown")],
        "risks": [
            "Rebuilding from raw requires the raw data to remain immutable during the rebuild; "
            "this audit does not verify raw-file immutability guarantees beyond current listing.",
        ],
        "next_action": "Confirm raw source objects for 2024 are complete, then rebuild in staging only.",
    }

    decisions["C_freeze_2024_learned_artifacts"] = {
        "decision": _decide(concept_2024, tuple(concept_2024)),
        "evidence": concept_2024,
        "missing_requirements": [k for k, v in concept_2024.items() if v in ("missing", "unknown")],
        "risks": [
            "Freezing artifacts without a pipeline manifest ID risks silently mutating a "
            "'frozen' artifact in a later run.",
        ],
        "next_action": "Attach a pipeline_manifest.schema.json-conformant manifest to any artifact before freezing.",
    }

    decisions["D_parse_2025_identical_transformations"] = {
        "decision": _decide(core_2025, tuple(core_2025)),
        "evidence": core_2025,
        "missing_requirements": [k for k, v in core_2025.items() if v in ("missing", "unknown")],
        "risks": [
            "Any difference in schema between the 2024 and 2025 master tables must be treated as a "
            "compatibility break, not silently renamed.",
        ],
        "next_action": "Diff the 2024 vs 2025 dataset_profile.json column sets before reusing transformations.",
    }

    e_statuses = {**core_2025, **pregame_2025}
    decisions["E_2025_walk_forward_backtest"] = {
        "decision": _decide(e_statuses, tuple(e_statuses)),
        "evidence": e_statuses,
        "missing_requirements": [k for k, v in e_statuses.items() if v != "complete"],
        "risks": [
            "A complete full-season 2025 table is NOT automatically pregame-safe. Backtest features "
            "must be reconstructed as of each game's feature_cutoff_time, not read from the final table.",
            "If starters/lineups/bullpen/weather/rest/travel are 'present_but_not_pregame_safe', "
            "a naive backtest using the full table would leak postgame information.",
        ],
        "next_action": (
            "Do not run the 2025 backtest until every pregame_* field has a source_retrieved_at_utc "
            "timestamp proving it was known before feature_cutoff_time for that game."
        ),
    }

    f_statuses = {**pregame_2025, **market_2025}
    decisions["F_2025_pregame_game_cards"] = {
        "decision": _decide(f_statuses, tuple(pregame_2025)),
        "evidence": f_statuses,
        "missing_requirements": [k for k, v in f_statuses.items() if v != "complete"],
        "risks": [
            "Historical pregame Game Cards for 2025 can only be built if starters/lineups/bullpen/"
            "weather/rest/travel are timestamp-proven pregame, per schemas/pregame_game_card.schema.json.",
        ],
        "next_action": (
            "Build Game Cards only from fields classified 'schedule_safe' or from fields with a "
            "confirmed pre-cutoff source_retrieved_at_utc timestamp."
        ),
    }

    decisions["G_2026_forward_predictions"] = {
        "decision": _decide(core_2026, tuple(core_2026)),
        "evidence": core_2026,
        "missing_requirements": [k for k, v in core_2026.items() if v in ("missing", "unknown")],
        "risks": [
            "2026 forward predictions must never be informed by any 2026 game result; only the "
            "published schedule and pre-game feature snapshots may be used.",
        ],
        "next_action": "Confirm published 2026 schedule coverage before enabling forward prediction runs.",
    }

    for key, value in decisions.items():
        if value["decision"] not in ALLOWED_DECISIONS:
            value["decision"] = "unknown"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decisions": decisions,
        "note": (
            "A completed full-season table is not automatically pregame-safe. "
            "'ready_with_warnings' and 'complete' statuses above reflect data presence only, "
            "not proof of pregame timestamp integrity, unless explicitly stated."
        ),
    }


def write_readiness_decisions(decisions: dict[str, Any], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "readiness_decisions.json"
    path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    return path
