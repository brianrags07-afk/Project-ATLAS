from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_builds_certified_event_and_quarantine_artifacts(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    teams = pd.DataFrame([{"season": 2024, "team_id": 1, "abbreviation": "AAA"}])
    rosters = pd.DataFrame([
        {"season": 2024, "team_id": 1, "as_of_date": "2024-03-27", "roster_type": "active", "player_id": 7, "player_identity_known": True, "source": "MLB", "source_retrieved_at": "2026-07-22T00:00:00Z", "source_record_sha256": "a"}
    ])
    transactions = pd.DataFrame(columns=["season", "transaction_id", "player_id", "requested_team_id", "team_id", "from_team_id", "to_team_id", "effective_date", "transaction_date", "type_code", "type_description", "source_retrieved_at", "source_record_sha256"])
    frames = {"teams.parquet": teams, "rosters.parquet": rosters, "transactions.parquet": transactions}
    artifacts = {}
    import hashlib
    for name, frame in frames.items():
        path = source / name
        frame.to_parquet(path, index=False)
        artifacts[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (source / "manifest.json").write_text(json.dumps({"season": 2024, "schedule_certification_verdict": "certified", "schedule_source_sha256": "fixture", "artifacts": artifacts}))
    result = subprocess.run([sys.executable, "scripts/build_historical_roster_events.py", "--source", str(source), "--output", str(output), "--season", "2024"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["certification"]["verdict"] == "certified"
    assert manifest["event_rows"] == 1
    assert manifest["promotion_status"] == "build_only_not_canonical"


def test_rejects_source_checksum_mismatch(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    for name in ("teams.parquet", "rosters.parquet", "transactions.parquet"):
        pd.DataFrame().to_parquet(source / name)
    artifacts = {name: {"sha256": "wrong"} for name in ("teams.parquet", "rosters.parquet", "transactions.parquet")}
    (source / "manifest.json").write_text(json.dumps({"season": 2024, "schedule_certification_verdict": "certified", "schedule_source_sha256": "fixture", "artifacts": artifacts}))
    result = subprocess.run([sys.executable, "scripts/build_historical_roster_events.py", "--source", str(source), "--output", str(output), "--season", "2024"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
