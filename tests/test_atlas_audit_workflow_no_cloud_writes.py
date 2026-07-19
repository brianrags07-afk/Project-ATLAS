"""
Static safety tests for .github/workflows/atlas-historical-readiness-audit.yml
and scripts/run_historical_readiness_audit.py: the audit must never write
to Cloud Storage.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "atlas-historical-readiness-audit.yml"
AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_historical_readiness_audit.py"

# Commands that would mutate or upload data to Cloud Storage. `cp` alone is
# ambiguous (it is used to *download*), so we specifically look for cp/mv/rm
# invocations whose destination argument is a gs:// URI.
CLOUD_WRITE_COMMAND_RE = re.compile(
    r"gcloud\s+storage\s+(rm|mv|rsync)\b|gsutil\s+(rm|mv|rsync|cp\s+.*-[a-zA-Z]*\bD)\b"
)
CP_TO_BUCKET_RE = re.compile(r"gcloud\s+storage\s+cp\s+\S+\s+(gs://|\"\$\{?ATLAS_BUCKET)")


def test_workflow_file_exists_and_parses():
    assert WORKFLOW_PATH.exists()
    with WORKFLOW_PATH.open() as fh:
        doc = yaml.safe_load(fh)
    assert doc is not None


def test_workflow_is_workflow_dispatch_only():
    with WORKFLOW_PATH.open() as fh:
        doc = yaml.safe_load(fh)
    triggers = doc.get(True, doc.get("on"))
    assert list(triggers.keys()) == ["workflow_dispatch"]


def test_workflow_has_no_cloud_delete_move_or_rsync_commands():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert not CLOUD_WRITE_COMMAND_RE.search(text), "workflow must not delete/move/rsync Cloud Storage objects"


def test_workflow_has_no_cp_that_writes_to_the_bucket():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert not CP_TO_BUCKET_RE.search(text), "workflow must not `cp` anything INTO the bucket"


def test_workflow_uploads_reports_as_actions_artifact_only():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "actions/upload-artifact" in text
    assert "gcloud storage cp artifacts" not in text
    assert "gcloud storage rsync artifacts" not in text


def test_workflow_uses_python_3_12():
    with WORKFLOW_PATH.open() as fh:
        doc = yaml.safe_load(fh)
    job = next(iter(doc["jobs"].values()))
    python_steps = [s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python")]
    assert python_steps
    assert python_steps[0]["with"]["python-version"] == "3.12"


def test_workflow_authenticates_with_workload_identity_federation():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "google-github-actions/auth@v2" in text
    assert "workload_identity_provider" in text
    assert "WORKLOAD_IDENTITY_PROVIDER" in text


def test_audit_script_has_no_upload_or_write_command_to_bucket():
    text = AUDIT_SCRIPT_PATH.read_text(encoding="utf-8")
    assert not CLOUD_WRITE_COMMAND_RE.search(text)
    assert not CP_TO_BUCKET_RE.search(text)
    # Only `cp <bucket_path> <local_path>` (download) invocations are allowed.
    for match in re.finditer(r'\["gcloud", "storage", "cp",\s*([^\]]+)\]', text):
        args_text = match.group(1)
        assert "remote_path" in args_text and "local_path" in args_text
