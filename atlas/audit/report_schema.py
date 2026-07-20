"""
JSON Schema validation helpers for the redesigned ATLAS historical
readiness audit reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
REPORT_SCHEMA_PATH = SCHEMAS_DIR / "historical_readiness_report.schema.json"
EVIDENCE_SCHEMA_PATH = SCHEMAS_DIR / "evidence_record.schema.json"


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report_validator() -> jsonschema.protocols.Validator:
    """Build a validator for the historical_readiness_report schema with a
    resolver that can load the sibling evidence_record.schema.json by
    relative $ref (both files live in ``schemas/``)."""
    report_schema = _load_schema(REPORT_SCHEMA_PATH)
    evidence_schema = _load_schema(EVIDENCE_SCHEMA_PATH)

    store = {
        report_schema["$id"]: report_schema,
        evidence_schema["$id"]: evidence_schema,
        "evidence_record.schema.json": evidence_schema,
    }
    resolver = jsonschema.RefResolver(base_uri=report_schema["$id"], referrer=report_schema, store=store)
    validator_cls = jsonschema.validators.validator_for(report_schema)
    return validator_cls(report_schema, resolver=resolver, format_checker=jsonschema.FormatChecker())


def validate_report(coverage_matrix_rows: list[dict[str, Any]], readiness: dict[str, Any]) -> list[str]:
    """Validate the combined coverage-matrix + readiness report against
    ``schemas/historical_readiness_report.schema.json``. Returns a list of
    human-readable error strings (empty list = valid)."""
    from datetime import datetime, timezone

    document = {
        "coverage_matrix": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": coverage_matrix_rows,
        },
        "readiness": readiness,
    }
    validator = build_report_validator()
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in errors]
