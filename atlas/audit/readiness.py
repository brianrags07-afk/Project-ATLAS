"""
Readiness decisions for the ATLAS historical readiness audit.

Redesigned so that each of the seven historical-readiness questions
(A-G) explicitly selects the evidence dimensions it requires and the
threshold each dimension must meet -- there is no single generic
``_decide()`` function that treats all non-complete states equivalently.

Every decision returns:
  - ``verdict``: ready | ready_with_warnings | not_ready | unknown
  - ``required_dimensions``: the dimensions/thresholds this decision needs
  - ``evidence_used``: the coverage-matrix rows / provenance records consulted
  - ``blockers``: unmet hard requirements (drive ``not_ready``)
  - ``warnings``: unmet soft requirements (drive ``ready_with_warnings``)
  - ``next_action``: exact next action
  - ``does_not_authorize``: explicit statement of what this verdict does
    NOT authorize, even if ``ready``

A fully-populated season table is never treated as automatically
pregame-safe, and postgame-only evidence is never a blocker for
historical reconstruction/learning use cases (only for same-game
pregame use).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_VERDICTS = ("ready", "ready_with_warnings", "not_ready", "unknown")

DECISION_KEYS = (
    "A_exact_2024_reproduction",
    "B_rebuild_2024_from_raw",
    "C_freeze_2024_learned_artifacts",
    "D_parse_2025_identical_transformations",
    "E_2025_walk_forward_backtest",
    "F_2025_pregame_game_cards",
    "G_2026_forward_predictions",
)

# Statements common to every decision: nothing produced by this audit
# authorizes any of these actions on its own.
BASELINE_DOES_NOT_AUTHORIZE = (
    "Does not authorize writing, deleting, renaming, or overwriting any Cloud Storage object.",
    "Does not authorize running a 2024 rebuild.",
    "Does not authorize running a 2025 backtest.",
    "Does not authorize model training.",
    "Does not authorize prediction generation.",
    "Does not authorize mutating existing master data.",
)


def _row(matrix: list[dict[str, Any]], row: str, season: int) -> dict[str, Any]:
    for r in matrix:
        if r["row"] == row and r["season"] == season:
            return r
    return {
        "row": row, "season": season,
        "data_presence": "unknown", "source_completeness": "unknown",
        "provenance_status": "unknown", "temporal_availability": "unknown",
        "pregame_safety": "unknown", "evidence": [], "risks": [], "required_next_evidence": [],
    }


def _verdict_from_checks(hard_ok: list[bool], soft_ok: list[bool], any_evidence: bool) -> str:
    if not any_evidence:
        return "unknown"
    if not all(hard_ok):
        return "not_ready"
    if all(soft_ok):
        return "ready"
    return "ready_with_warnings"


def _decision(
    required_dimensions: dict[str, str],
    evidence_used: list[dict[str, Any]],
    blockers: list[str],
    warnings: list[str],
    next_action: str,
    extra_does_not_authorize: tuple[str, ...] = (),
    any_evidence: bool | None = None,
) -> dict[str, Any]:
    if any_evidence is None:
        any_evidence = bool(evidence_used)
    if blockers:
        verdict = "not_ready"
    elif not any_evidence:
        verdict = "unknown"
    elif warnings:
        verdict = "ready_with_warnings"
    else:
        verdict = "ready"
    return {
        "verdict": verdict,
        "required_dimensions": required_dimensions,
        "evidence_used": evidence_used,
        "blockers": blockers,
        "warnings": warnings,
        "next_action": next_action,
        "does_not_authorize": list(BASELINE_DOES_NOT_AUTHORIZE) + list(extra_does_not_authorize),
    }


def _decide_exact_reproduction(
    matrix: list[dict[str, Any]],
    provenance: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "master_game_database.provenance_status": "verified",
        "master_pitch_database.provenance_status": "verified",
        "concept_discovery(row).provenance_status": ">= partial",
        "model_artifacts(row).provenance_status": ">= partial",
    }
    concept_row = _row(matrix, "concept_discovery", 2024)
    model_row = _row(matrix, "model_artifacts", 2024)
    game_prov = provenance.get("master_game_database", {})
    pitch_prov = provenance.get("master_pitch_database", {})

    evidence_used = [concept_row, model_row, game_prov, pitch_prov]
    blockers = []
    warnings = []

    if game_prov.get("provenance_status") != "verified":
        blockers.append(
            "master_game_database provenance is "
            f"'{game_prov.get('provenance_status', 'unknown')}', not 'verified' (needs a "
            "content hash AND a manifest linkage)."
        )
    if pitch_prov.get("provenance_status") != "verified":
        blockers.append(
            "master_pitch_database provenance is "
            f"'{pitch_prov.get('provenance_status', 'unknown')}', not 'verified'."
        )
    if concept_row.get("provenance_status") == "missing":
        blockers.append("No concept-discovery artifact provenance found for 2024.")
    elif concept_row.get("provenance_status") != "verified":
        warnings.append("concept_discovery provenance is not fully verified (manifest/hash lineage incomplete).")
    if model_row.get("provenance_status") == "missing":
        blockers.append("No model-artifact provenance found for 2024.")
    elif model_row.get("provenance_status") != "verified":
        warnings.append("model_artifacts provenance is not fully verified (manifest/hash lineage incomplete).")

    blockers.append(
        "Processed-table (master_game_database / master_pitch_database) presence alone is "
        "insufficient for exact reproduction: source hashes, code/version lineage, pipeline "
        "manifests, artifact lineage, transformation identity, schema versions, and a "
        "reproducible environment record are all required and none were fully verified by "
        "this audit."
    ) if blockers else None
    blockers = [b for b in blockers if b]

    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Locate or produce a pipeline_manifest.schema.json-conformant manifest recording "
            "source object hashes, code commit SHA, pipeline version, and schema refs for the "
            "original 2024 run before attempting exact reproduction."
        ),
        extra_does_not_authorize=(
            "Does not authorize treating processed-table presence as proof of exact reproducibility.",
        ),
    )


def _decide_rebuild_from_raw(
    matrix: list[dict[str, Any]],
    dataset_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "final_scores.data_presence": "present",
        "pitch_by_pitch.data_presence": "present",
        "raw_source datasets": "data_layer == raw_source with source_completeness == complete",
    }
    core_rows = [_row(matrix, r, 2024) for r in ("final_scores", "pitch_by_pitch")]
    raw_profiles = [p for p in dataset_profiles.values() if p.get("data_layer") == "raw_source"]

    evidence_used = list(core_rows) + raw_profiles
    blockers = []
    warnings = []

    for row in core_rows:
        if row["data_presence"] == "missing":
            blockers.append(f"'{row['row']}' has no data presence for 2024.")

    if not raw_profiles:
        warnings.append(
            "No dataset was classified as data_layer='raw_source'. Only normalized/master "
            "tables (master_game_database/master_pitch_database) were profiled; those "
            "prove neither the existence nor the completeness of the original raw source "
            "objects. This is a warning, not an automatic blocker, because postgame raw "
            "facts remain valid raw inputs for reconstruction once raw provenance is found."
        )
    else:
        for profile in raw_profiles:
            if profile.get("row_count", 0) <= 0:
                blockers.append(f"Raw-source dataset at '{profile.get('cloud_path')}' has zero rows.")

    # Postgame nature of raw pitch/game facts is explicitly NOT a blocker.
    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Confirm raw source objects for 2024 are present, complete, and provenance-verified "
            "(hash + manifest), then rebuild in a staging location only -- never overwrite "
            "existing master data."
        ),
        extra_does_not_authorize=(
            "Does not authorize promoting any rebuild output out of staging.",
            "Does not authorize using the rebuilt tables for pregame prediction without "
            "separate pregame-safety verification.",
        ),
    )


def _decide_freeze_learned_artifacts(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "concept_discovery.provenance_status": "verified",
        "concept_validation.provenance_status": ">= partial",
        "model_artifacts.provenance_status": "verified",
    }
    concept_row = _row(matrix, "concept_discovery", 2024)
    validation_row = _row(matrix, "concept_validation", 2024)
    model_row = _row(matrix, "model_artifacts", 2024)
    evidence_used = [concept_row, validation_row, model_row]

    blockers = []
    warnings = []
    for row, label in ((concept_row, "concept_discovery"), (model_row, "model_artifacts")):
        if row["provenance_status"] == "missing":
            blockers.append(f"No {label} artifact/manifest evidence found for 2024.")
        elif row["provenance_status"] != "verified":
            blockers.append(
                f"{label} provenance is '{row['provenance_status']}', not 'verified' -- a "
                "manifest ID, source hashes, code commit, schema version, and validation "
                "status are all required before freezing."
            )
    if validation_row["provenance_status"] not in ("verified", "partial"):
        warnings.append("concept_validation provenance is not established; freeze without it is risky.")

    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Attach a pipeline_manifest.schema.json-conformant manifest (manifest ID, source "
            "hashes, code commit, schema version, validation status) to any artifact before "
            "marking it frozen, and write it to an immutable/versioned destination."
        ),
        extra_does_not_authorize=(
            "Does not authorize freezing an artifact that lacks a manifest ID.",
        ),
    )


def _decide_parse_2025_identical(
    matrix: list[dict[str, Any]],
    dataset_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from atlas.audit.dataset_profile import compare_schema_compatibility

    required = {
        "master_game_database schema fingerprint": "must match between reference (2024) and candidate profile",
        "master_pitch_database schema fingerprint": "must match between reference (2024) and candidate profile",
    }
    core_rows = [_row(matrix, r, 2025) for r in ("final_scores", "pitch_by_pitch")]
    evidence_used: list[dict[str, Any]] = list(core_rows)
    blockers = []
    warnings = []

    game_profile = dataset_profiles.get("master_game_database")
    pitch_profile = dataset_profiles.get("master_pitch_database")
    if game_profile is None or pitch_profile is None:
        blockers.append("master_game_database or master_pitch_database profile not available for compatibility check.")
    else:
        # Compare the single available profile against itself is a no-op;
        # in real use the caller supplies a reference (prior/2024) profile
        # via dataset_profiles["<name>__2024_reference"] when available.
        reference_game = dataset_profiles.get("master_game_database__2024_reference", game_profile)
        reference_pitch = dataset_profiles.get("master_pitch_database__2024_reference", pitch_profile)
        game_report = compare_schema_compatibility(reference_game, game_profile)
        pitch_report = compare_schema_compatibility(reference_pitch, pitch_profile)
        evidence_used.extend([game_report, pitch_report])
        if not game_report["compatible"]:
            blockers.append(
                f"master_game_database schema incompatible with reference: "
                f"added={game_report['added_columns']}, removed={game_report['removed_columns']}, "
                f"dtype_mismatches={game_report['dtype_mismatches']}."
            )
        if not pitch_report["compatible"]:
            blockers.append(
                f"master_pitch_database schema incompatible with reference: "
                f"added={pitch_report['added_columns']}, removed={pitch_report['removed_columns']}, "
                f"dtype_mismatches={pitch_report['dtype_mismatches']}."
            )

    for row in core_rows:
        if row["data_presence"] == "missing":
            warnings.append(f"'{row['row']}' has no 2025 data presence yet.")

    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Produce an explicit schema-compatibility report (dataset_profile.compare_schema_compatibility) "
            "between the 2024 reference schema and the 2025 candidate schema before reusing any "
            "transformation. Never silently rename or coerce a column to force compatibility."
        ),
        extra_does_not_authorize=(
            "Does not authorize silently renaming or coercing an incompatible column.",
        ),
    )


def _decide_walk_forward_backtest(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    dynamic_rows = ("starters", "lineups", "bullpen_usage", "weather", "rest", "travel", "opening_market")
    required = {f"{r}.pregame_safety": "safe" for r in dynamic_rows}
    required["final_scores.temporal_availability"] = "postgame_only (informational only, not usable as input)"

    rows = [_row(matrix, r, 2025) for r in dynamic_rows]
    evidence_used = list(rows)
    blockers = []
    warnings = []

    for row in rows:
        if row["pregame_safety"] == "unsafe":
            blockers.append(
                f"'{row['row']}' for 2025 is pregame_safety=unsafe (postgame-only evidence); using it "
                "in a walk-forward backtest would leak postgame information into a pregame feature."
            )
        elif row["pregame_safety"] in ("unknown", "conditional"):
            blockers.append(
                f"'{row['row']}' for 2025 has no per-game timestamp proof "
                f"(pregame_safety={row['pregame_safety']}); a full-season table cannot authorize "
                "this backtest."
            )

    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Do not run the 2025 backtest until every dynamic pregame field has a "
            "source_retrieved_at_utc timestamp proven on-or-before each game's "
            "feature_cutoff_time, chronological state updates are enforced, and predictions "
            "are frozen before outcomes are known (leakage-guard tests must pass)."
        ),
        extra_does_not_authorize=(
            "Does not authorize reading pregame features from the final full-season table.",
            "Does not authorize any non-chronological state update.",
        ),
    )


def _decide_pregame_game_cards(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    dynamic_rows = ("starters", "lineups", "bullpen_usage", "weather", "rest", "travel")
    required = {f"{r}.pregame_safety": "safe or conditional (nulls preserved)" for r in dynamic_rows}
    frozen_row = _row(matrix, "frozen_pregame_cards", 2025)

    rows = [_row(matrix, r, 2025) for r in dynamic_rows]
    evidence_used = list(rows) + [frozen_row]
    blockers = []
    warnings = []

    for row in rows:
        if row["pregame_safety"] == "unsafe":
            blockers.append(
                f"'{row['row']}' for 2025 is pregame_safety=unsafe and must never be backfilled "
                "into a historical Game Card using postgame knowledge."
            )
        elif row["pregame_safety"] in ("unknown",):
            warnings.append(
                f"'{row['row']}' for 2025 has no per-game timestamp proof yet; Game Cards for this "
                "field must record null/unknown rather than a guessed value."
            )

    if frozen_row["provenance_status"] == "missing":
        warnings.append("No frozen Game Card artifact evidence found for 2025 yet.")

    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Build 2025 historical Game Cards only from fields with field-level temporal "
            "provenance (source_retrieved_at_utc <= feature_cutoff_time_utc per "
            "schemas/pregame_game_card.schema.json); preserve null/unknown for any field "
            "without that proof instead of backfilling with postgame knowledge."
        ),
        extra_does_not_authorize=(
            "Does not authorize backfilling a null pregame field using postgame knowledge.",
        ),
    )


def _decide_forward_predictions_2026(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "published_schedule.provenance_status": "verified",
        "published_schedule.pregame_safety": "safe",
        "game_identifiers.data_presence": "present",
    }
    schedule_row = _row(matrix, "published_schedule", 2026)
    id_row = _row(matrix, "game_identifiers", 2026)
    evidence_used = [schedule_row, id_row]
    blockers = []
    warnings = []

    if schedule_row["data_presence"] == "missing":
        blockers.append("No published_schedule data presence for 2026.")
    if schedule_row["provenance_status"] != "verified":
        blockers.append(
            f"published_schedule provenance_status is '{schedule_row['provenance_status']}', not "
            "'verified'; a timestamped published-schedule source is required before any forward "
            "prediction run."
        )
    if schedule_row["pregame_safety"] != "safe":
        blockers.append(
            f"published_schedule pregame_safety is '{schedule_row['pregame_safety']}', not 'safe'."
        )
    if id_row["data_presence"] == "missing":
        warnings.append("No game_identifiers data presence for 2026 yet.")

    return _decision(
        required,
        evidence_used,
        blockers,
        warnings,
        next_action=(
            "Confirm a verified, pregame-safe published 2026 schedule and current pregame "
            "snapshots, freeze the model/artifact version to be used, and require complete run "
            "manifesting with no same-game outcome inputs before enabling forward prediction runs."
        ),
        extra_does_not_authorize=(
            "Does not authorize using any 2026 same-game outcome as a model input.",
            "Does not authorize forward prediction without a frozen model/artifact version.",
        ),
    )


def build_readiness_decisions(
    coverage_matrix: list[dict[str, Any]],
    dataset_profiles: dict[str, dict[str, Any]],
    provenance: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    provenance = provenance or {}
    decisions: dict[str, Any] = {
        "A_exact_2024_reproduction": _decide_exact_reproduction(coverage_matrix, provenance),
        "B_rebuild_2024_from_raw": _decide_rebuild_from_raw(coverage_matrix, dataset_profiles),
        "C_freeze_2024_learned_artifacts": _decide_freeze_learned_artifacts(coverage_matrix),
        "D_parse_2025_identical_transformations": _decide_parse_2025_identical(coverage_matrix, dataset_profiles),
        "E_2025_walk_forward_backtest": _decide_walk_forward_backtest(coverage_matrix),
        "F_2025_pregame_game_cards": _decide_pregame_game_cards(coverage_matrix),
        "G_2026_forward_predictions": _decide_forward_predictions_2026(coverage_matrix),
    }

    for value in decisions.values():
        if value["verdict"] not in ALLOWED_VERDICTS:
            value["verdict"] = "unknown"

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decisions": decisions,
        "note": (
            "A completed full-season table is not automatically pregame-safe, and a dataset can "
            "be complete/source_completeness=complete while still pregame_safety=unsafe. Postgame "
            "facts remain valid raw inputs for historical reconstruction/learning but never "
            "authorize same-game pregame prediction. No verdict here authorizes a rebuild, "
            "backtest, model training, prediction generation, or any Cloud Storage mutation."
        ),
    }


def write_readiness_decisions(decisions: dict[str, Any], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "readiness_decisions.json"
    path.write_text(json.dumps(decisions, indent=2, default=str), encoding="utf-8")
    return path
