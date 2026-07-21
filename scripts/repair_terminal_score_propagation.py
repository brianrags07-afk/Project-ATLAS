#!/usr/bin/env python3
"""Create non-destructive terminal-score repair candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

from atlas.audit.terminal_score_propagation import (
    MASTER_REPAIRED_FIELDS,
    TEAM_REPAIRED_FIELDS,
    repair_terminal_score_propagation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace(table: pa.Table, name: str, values) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(
        index,
        field,
        pa.array(values, type=field.type, from_pandas=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--team-state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    master_path = Path(args.master)
    team_path = Path(args.team_state)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    master_table = pq.read_table(master_path)
    team_table = pq.read_table(team_path)
    repaired_master, repaired_team, audit = repair_terminal_score_propagation(
        master_table.to_pandas(),
        team_table.to_pandas(),
    )
    candidate_master = master_table
    for field in MASTER_REPAIRED_FIELDS:
        candidate_master = _replace(candidate_master, field, repaired_master[field])
    candidate_team = team_table
    for field in TEAM_REPAIRED_FIELDS:
        candidate_team = _replace(candidate_team, field, repaired_team[field])

    master_output = output / "master_game_database.parquet"
    team_output = output / "team_game_state.parquet"
    pq.write_table(candidate_master, master_output, compression="snappy")
    pq.write_table(candidate_team, team_output, compression="snappy")

    verified_master = pq.read_table(master_output)
    verified_team = pq.read_table(team_output)
    if not verified_master.schema.equals(master_table.schema, check_metadata=True):
        raise RuntimeError("master schema or metadata changed")
    if not verified_team.schema.equals(team_table.schema, check_metadata=True):
        raise RuntimeError("team schema or metadata changed")
    if verified_master.num_rows != master_table.num_rows:
        raise RuntimeError("master row count changed")
    if verified_team.num_rows != team_table.num_rows:
        raise RuntimeError("team row count changed")

    manifest = {
        "verdict": "candidate_certified",
        "source": {
            "master": {"path": str(master_path), "sha256": _sha256(master_path)},
            "team_state": {"path": str(team_path), "sha256": _sha256(team_path)},
        },
        "candidate": {
            "master": {"path": str(master_output), "sha256": _sha256(master_output)},
            "team_state": {"path": str(team_output), "sha256": _sha256(team_output)},
        },
        "authorized_changes": audit,
        "row_counts_preserved": True,
        "schemas_preserved": True,
    }
    (output / "terminal_score_propagation_repair.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": manifest["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
