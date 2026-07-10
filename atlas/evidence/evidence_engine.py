
import json
from pathlib import Path

from atlas.config.paths import EVIDENCE_DIR
from atlas.evidence.team_evidence import build_team_moneyline_evidence


EVIDENCE_ENGINE_VERSION = "1.0.0"


def run_evidence_engine(team_game_state, output_dir=None):
    output_dir = Path(output_dir) if output_dir else EVIDENCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = []

    for team in sorted(team_game_state["team"].dropna().unique()):
        team_df = team_game_state[team_game_state["team"] == team].copy()
        evidence.append(build_team_moneyline_evidence(team_df))

    outfile = output_dir / "team_moneyline_evidence.json"

    with open(outfile, "w") as f:
        json.dump(evidence, f, indent=2)

    print("=" * 60)
    print("ATLAS EVIDENCE ENGINE")
    print("=" * 60)
    print(f"Evidence Objects : {len(evidence)}")
    print(f"Saved To         : {outfile}")
    print("=" * 60)

    return {
        "engine": "ATLAS Evidence Engine",
        "engine_version": EVIDENCE_ENGINE_VERSION,
        "objects": len(evidence),
        "output_path": str(outfile),
    }
