from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_profiles_events_and_quarantine_without_semantic_inference(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    events = pd.DataFrame([{"event_type": "opening_roster", "team": "AAA", "player_id": 7}])
    quarantine = pd.DataFrame([{"quarantine_source": "transaction", "quarantine_reason": "no explicit inter-team direction", "type_code": "IL", "type_description": "Injured List", "player_id": 7}])
    artifacts = {}
    for name, frame in (("roster_events.parquet", events), ("roster_event_quarantine.parquet", quarantine)):
        path = source / name
        frame.to_parquet(path, index=False)
        artifacts[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (source / "manifest.json").write_text(json.dumps({"season": 2024, "certification": {"verdict": "certified"}, "artifacts": artifacts}))
    result = subprocess.run([sys.executable, "scripts/profile_historical_roster_events.py", "--source", str(source), "--output", str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    profile = json.loads((output / "profile.json").read_text())
    assert profile["transaction_type_codes"] == {"IL": 1}
    assert profile["semantic_mapping_status"] == "team_scoped_status_semantics_v3"
    assert (output / "quarantine_type_profile.csv").exists()
