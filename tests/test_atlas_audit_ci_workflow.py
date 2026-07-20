"""
Static safety tests for the new PR/push CI workflow
(.github/workflows/atlas-audit-ci.yml).

This workflow must run the audit/schema test suite on every pull request
and push, using Python 3.12, WITHOUT ever authenticating to Google Cloud
or touching the real bucket. The real-bucket audit stays manual-only in
.github/workflows/atlas-historical-readiness-audit.yml.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "atlas-audit-ci.yml"
REAL_AUDIT_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "atlas-historical-readiness-audit.yml"

GCP_AUTH_MARKERS = (
    "google-github-actions/auth",
    "workload_identity_provider",
    "setup-gcloud",
)
BUCKET_ACCESS_MARKERS = (
    "gcloud storage cp",
    "gcloud storage ls",
    "gcloud storage objects",
    "atlas-mlb-data-brian-4817",
)
CLOUD_WRITE_COMMAND_RE = re.compile(
    r"gcloud\s+storage\s+(rm|mv|rsync)\b|gsutil\s+(rm|mv|rsync|cp\s+.*-[a-zA-Z]*\bD)\b"
)


def _load_ci_workflow() -> dict:
    with CI_WORKFLOW_PATH.open() as fh:
        return yaml.safe_load(fh)


def test_ci_workflow_file_exists_and_parses():
    assert CI_WORKFLOW_PATH.exists()
    doc = _load_ci_workflow()
    assert doc is not None


def test_ci_workflow_triggers_on_pull_request_and_push():
    doc = _load_ci_workflow()
    on_key = True if True in doc else "on"
    triggers = doc[on_key]
    assert "pull_request" in triggers
    assert "push" in triggers
    assert "workflow_dispatch" not in triggers


def test_ci_workflow_uses_python_3_12():
    doc = _load_ci_workflow()
    job = next(iter(doc["jobs"].values()))
    python_steps = [s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python")]
    assert python_steps
    assert python_steps[0]["with"]["python-version"] == "3.12"


def _non_guard_lines(text: str) -> str:
    """Strip out the lines that intentionally *mention* forbidden markers
    as part of this workflow's own self-check guard step, so those
    string literals don't trigger false positives below."""
    return "\n".join(
        line for line in text.splitlines() if "::error::" not in line and "grep -Eq" not in line
    )


def test_ci_workflow_never_authenticates_to_gcp():
    text = _non_guard_lines(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
    for marker in GCP_AUTH_MARKERS:
        assert marker not in text, f"CI workflow must not authenticate to GCP (found {marker!r})"


def test_ci_workflow_never_accesses_the_real_bucket():
    text = _non_guard_lines(CI_WORKFLOW_PATH.read_text(encoding="utf-8"))
    for marker in BUCKET_ACCESS_MARKERS:
        assert marker not in text, f"CI workflow must not access the real bucket (found {marker!r})"


def test_ci_workflow_has_no_cloud_mutation_commands():
    text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert not CLOUD_WRITE_COMMAND_RE.search(text)


def test_ci_workflow_runs_the_audit_test_suite():
    text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "pytest tests/test_atlas_audit_" in text


def test_ci_workflow_runs_syntax_and_import_validation():
    text = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "py_compile" in text
    assert "import atlas.audit" in text


def test_real_bucket_audit_workflow_remains_manual_only():
    with REAL_AUDIT_WORKFLOW_PATH.open() as fh:
        doc = yaml.safe_load(fh)
    on_key = True if True in doc else "on"
    triggers = doc[on_key]
    assert list(triggers.keys()) == ["workflow_dispatch"]
