"""Read-only certification of schedule, game, pitch, and team-game universes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

import pandas as pd

from atlas.schedule.mlb_schedule_reference import extract_raw_games


def _schedule_index(
    schedule_payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> tuple[dict[int, Mapping[str, Any]], set[int]]:
    """Index either a raw MLB schedule payload or canonical schedule rows."""
    if isinstance(schedule_payload, Mapping) and "dates" in schedule_payload:
        rows = extract_raw_games(schedule_payload)
        regular = [row for row in rows if row.get("gameType") == "R"]
        by_pk = {int(row["gamePk"]): row for row in regular}
        cancelled = {
            game_pk
            for game_pk, row in by_pk.items()
            if (row.get("status") or {}).get("detailedState") == "Cancelled"
        }
        return by_pk, cancelled

    canonical_rows = (
        [schedule_payload]
        if isinstance(schedule_payload, Mapping)
        else list(schedule_payload)
    )
    by_pk: dict[int, Mapping[str, Any]] = {}
    cancelled: set[int] = set()
    for row in canonical_rows:
        game_type = row.get("game_type_code", row.get("gameType"))
        game_pk = row.get("game_pk", row.get("gamePk"))
        if game_type != "R" or game_pk is None:
            continue
        key = int(game_pk)
        by_pk[key] = row
        detailed_state = row.get("detailed_state")
        state_category = row.get("game_state_category")
        counted = row.get("counted_in_expected_games")
        if (
            detailed_state == "Cancelled"
            or state_category == "cancelled"
            or counted is False
        ):
            cancelled.add(key)
    return by_pk, cancelled


def certify_historical_datasets(
    schedule_payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    master: pd.DataFrame,
    pitch: pd.DataFrame,
    team_state: pd.DataFrame,
    *,
    season: int,
) -> dict[str, Any]:
    schedule_by_pk, cancelled = _schedule_index(schedule_payload)
    expected = set(schedule_by_pk) - cancelled

    game = master.loc[master["atlas_season"] == season].copy()
    pitches = pitch.loc[pitch["atlas_season"] == season].copy()
    teams = team_state.loc[team_state["atlas_season"] == season].copy()
    ids = {
        "master": set(game["game_pk"].dropna().astype(int)),
        "pitch": set(pitches["game_pk"].dropna().astype(int)),
        "team": set(teams["game_pk"].dropna().astype(int)),
    }

    coverage = {}
    errors: list[str] = []
    for name, observed in ids.items():
        coverage[f"missing_{name}"] = sorted(expected - observed)
        coverage[f"unexpected_{name}"] = sorted(observed - expected)
        if coverage[f"missing_{name}"]:
            errors.append(f"missing_{name}: {coverage[f'missing_{name}']}")
        if coverage[f"unexpected_{name}"]:
            errors.append(f"unexpected_{name}: {coverage[f'unexpected_{name}']}")

    master_errors = {
        "duplicate_game_pk": int(game.duplicated("game_pk").sum()),
        "non_regular_rows": int((game["game_type"] != "R").sum()),
        "total_runs": int(
            (game["total_runs"] != game["home_score"] + game["away_score"]).sum()
        ),
        "run_differential": int(
            (
                game["run_differential"]
                != game["home_score"] - game["away_score"]
            ).sum()
        ),
        "home_win": int(
            (game["home_win"] != (game["home_score"] > game["away_score"])).sum()
        ),
        "away_win": int(
            (game["away_win"] != (game["away_score"] > game["home_score"])).sum()
        ),
    }
    pitch_errors = {
        "non_regular_rows": int((pitches["game_type"] != "R").sum()),
        "duplicate_pitch_keys": int(
            pitches.duplicated(
                ["game_pk", "at_bat_number", "pitch_number"]
            ).sum()
        ),
    }
    pairs = teams.groupby("game_pk").agg(
        rows=("game_pk", "size"),
        runs_scored=("runs_scored", "sum"),
        runs_allowed=("runs_allowed", "sum"),
        differential=("run_differential", "sum"),
        winners=("won", "sum"),
    )
    team_errors = {
        "non_two_row_games": int((pairs["rows"] != 2).sum()),
        "run_differential": int(
            (
                teams["run_differential"]
                != teams["runs_scored"] - teams["runs_allowed"]
            ).sum()
        ),
        "won": int(
            (teams["won"] != (teams["runs_scored"] > teams["runs_allowed"])).sum()
        ),
        "score_symmetry": int(
            (pairs["runs_scored"] != pairs["runs_allowed"]).sum()
        ),
        "differential_symmetry": int((pairs["differential"] != 0).sum()),
        "winner_count": int((pairs["winners"] != 1).sum()),
    }
    for group, checks in (
        ("master", master_errors),
        ("pitch", pitch_errors),
        ("team", team_errors),
    ):
        for name, count in checks.items():
            if count:
                errors.append(f"{group}.{name}: {count}")

    return {
        "verdict": (
            "certified_with_documented_exceptions"
            if not errors and cancelled
            else "certified"
            if not errors
            else "quarantine_required"
        ),
        "season": season,
        "schedule": {
            "published_regular_games": len(schedule_by_pk),
            "completed_games": len(expected),
            "cancelled_game_pks": sorted(cancelled),
        },
        "datasets": {
            "master_games": len(game),
            "pitch_games": len(ids["pitch"]),
            "pitch_rows": len(pitches),
            "team_games": len(ids["team"]),
            "team_rows": len(teams),
        },
        "coverage": coverage,
        "master_errors": master_errors,
        "pitch_errors": pitch_errors,
        "team_errors": team_errors,
        "errors": errors,
    }


def attach_certification_provenance(
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a self-contained report after validating durable source identity."""
    required_top_level = {
        "schema_version",
        "certified_at_utc",
        "github",
        "transfer_manifest",
        "inputs",
    }
    missing = sorted(required_top_level - set(provenance))
    if missing:
        raise ValueError(f"missing provenance sections: {missing}")

    required_github = {"repository", "commit_sha", "run_id", "workflow", "ref"}
    missing_github = sorted(required_github - set(provenance["github"]))
    if missing_github:
        raise ValueError(f"missing GitHub provenance fields: {missing_github}")

    required_identity = {"gcs_uri", "generation", "sha256"}
    identities = {
        "transfer_manifest": provenance["transfer_manifest"],
        **dict(provenance["inputs"]),
    }
    if not identities:
        raise ValueError("no provenance identities supplied")
    for name, identity in identities.items():
        missing_identity = sorted(required_identity - set(identity))
        if missing_identity:
            raise ValueError(
                f"missing provenance identity fields for {name}: {missing_identity}"
            )
        if not str(identity["gcs_uri"]).startswith("gs://"):
            raise ValueError(f"invalid GCS URI for {name}")
        if len(str(identity["sha256"])) != 64:
            raise ValueError(f"invalid SHA-256 for {name}")

    enriched = dict(report)
    enriched["provenance"] = dict(provenance)
    return enriched
