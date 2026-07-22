#!/usr/bin/env python3
"""Build immutable 2024 batting-order and starter observation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.lineups.lineup_observation_labels import (
    BATTING_ORDER_COLUMNS,
    build_reconstructed_lineup_observation_labels,
    certify_reconstructed_lineup_observation_labels,
)


EVENT_COLUMNS = [
    "game_pk",
    "atlas_season",
    "game_type",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot_manifest(path: Path, *, season: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("season") or 0) != int(season):
        raise ValueError(f"roster snapshot manifest is not for season {season}")
    if manifest.get("game_type") != "R":
        raise ValueError("roster snapshot manifest is not regular-season isolated")
    certification = manifest.get("certification", {})
    if certification.get("verdict") != "certified":
        raise ValueError("roster snapshot manifest is not certified")
    return manifest


def verify_manifest_artifact(
    path: Path,
    manifest: dict[str, Any],
    name: str,
) -> str:
    expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
    if not expected:
        raise ValueError(f"roster snapshot manifest does not declare {name}")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"roster snapshot artifact checksum mismatch: {name}")
    return observed


def build_team_pattern_summary(labels: pd.DataFrame) -> pd.DataFrame:
    records = []
    for team, group in labels.groupby("team", sort=True):
        order_counts = group["lineup_order_signature"].value_counts()
        set_counts = group["lineup_player_set_signature"].value_counts()
        records.append(
            {
                "team": team,
                "team_games": int(len(group)),
                "complete_lineups": int(group["starting_lineup_complete"].sum()),
                "incomplete_lineups": int((~group["starting_lineup_complete"]).sum()),
                "unique_lineup_orders": int(group["lineup_order_signature"].nunique()),
                "unique_lineup_player_sets": int(
                    group["lineup_player_set_signature"].nunique()
                ),
                "most_common_lineup_order_games": int(order_counts.max()),
                "most_common_lineup_order_share": float(order_counts.max() / len(group)),
                "most_common_player_set_games": int(set_counts.max()),
                "most_common_player_set_share": float(set_counts.max() / len(group)),
                "unique_starting_pitchers": int(
                    group["starting_pitcher_id"].nunique(dropna=True)
                ),
            }
        )
    return pd.DataFrame(records)


def build_player_start_summary(labels: pd.DataFrame) -> pd.DataFrame:
    batting_frames = []
    for position, column in enumerate(BATTING_ORDER_COLUMNS, start=1):
        frame = labels[["team", "official_date", column]].rename(
            columns={column: "player_id"}
        )
        frame = frame.dropna(subset=["player_id"]).copy()
        frame["player_id"] = frame["player_id"].astype("int64")
        frame["batting_order_position"] = position
        batting_frames.append(frame)
    batting = pd.concat(batting_frames, ignore_index=True)
    batting_totals = (
        batting.groupby(["team", "player_id"], as_index=False)
        .agg(
            batting_starts=("official_date", "size"),
            first_batting_start_date=("official_date", "min"),
            last_batting_start_date=("official_date", "max"),
            mean_batting_order_position=("batting_order_position", "mean"),
        )
    )
    slot_counts = (
        batting.assign(_count=1)
        .pivot_table(
            index=["team", "player_id"],
            columns="batting_order_position",
            values="_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=lambda value: f"batting_order_{int(value)}_starts")
        .reset_index()
    )
    batting_totals = batting_totals.merge(
        slot_counts, on=["team", "player_id"], how="left", validate="one_to_one"
    )

    pitchers = labels[
        ["team", "official_date", "starting_pitcher_id"]
    ].dropna(subset=["starting_pitcher_id"]).rename(
        columns={"starting_pitcher_id": "player_id"}
    )
    pitchers["player_id"] = pitchers["player_id"].astype("int64")
    pitcher_totals = (
        pitchers.groupby(["team", "player_id"], as_index=False)
        .agg(
            starting_pitcher_starts=("official_date", "size"),
            first_pitching_start_date=("official_date", "min"),
            last_pitching_start_date=("official_date", "max"),
        )
    )
    output = batting_totals.merge(
        pitcher_totals,
        on=["team", "player_id"],
        how="outer",
        validate="one_to_one",
    )
    count_columns = [
        "batting_starts",
        "starting_pitcher_starts",
        *[f"batting_order_{position}_starts" for position in range(1, 10)],
    ]
    for column in count_columns:
        if column not in output.columns:
            output[column] = 0
        output[column] = output[column].fillna(0).astype("int64")
    output["total_starting_roles"] = (
        output["batting_starts"] + output["starting_pitcher_starts"]
    )
    return output.sort_values(
        ["team", "total_starting_roles", "player_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--team-games", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-events-sha256", required=True)
    parser.add_argument("--expected-completed-games", required=True, type=int)
    args = parser.parse_args()

    if args.season != 2024:
        raise ValueError("this certified lineup truth build currently supports 2024 only")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    snapshot_manifest = load_snapshot_manifest(
        args.snapshot_manifest, season=args.season
    )
    team_game_sha = verify_manifest_artifact(
        args.team_games, snapshot_manifest, "team_games.parquet"
    )
    events_sha = sha256(args.events)
    if events_sha != args.expected_events_sha256:
        raise ValueError("certified pitch-event source checksum mismatch")

    events = pd.read_parquet(args.events, columns=EVENT_COLUMNS)
    team_games = pd.read_parquet(args.team_games)
    completed_games = int(team_games["game_pk"].nunique())
    if completed_games != args.expected_completed_games:
        raise ValueError(
            "completed game count mismatch: "
            f"{completed_games} != {args.expected_completed_games}"
        )
    labels = build_reconstructed_lineup_observation_labels(
        events, team_games, season=args.season
    )
    certification = certify_reconstructed_lineup_observation_labels(
        labels, team_games, season=args.season
    )
    if certification["verdict"] != "certified":
        raise ValueError(
            "lineup truth certification failed: "
            + "; ".join(certification["errors"])
        )

    labels_path = args.output / "reconstructed_lineup_observation_labels.parquet"
    team_summary_path = args.output / "team_lineup_pattern_summary.csv"
    player_summary_path = args.output / "player_start_summary.csv"
    labels.to_parquet(labels_path, index=False)
    team_summary = build_team_pattern_summary(labels)
    player_summary = build_player_start_summary(labels)
    team_summary.to_csv(team_summary_path, index=False)
    player_summary.to_csv(player_summary_path, index=False)

    summary = {
        "season": args.season,
        "game_type": "R",
        "certification": certification,
        "completed_games": completed_games,
        "team_games": int(len(labels)),
        "teams": int(labels["team"].nunique()),
        "batting_players": int(
            labels[BATTING_ORDER_COLUMNS].stack().nunique()
        ),
        "starting_pitchers": int(labels["starting_pitcher_id"].nunique()),
        "unique_lineup_orders": int(labels["lineup_order_signature"].nunique()),
        "unique_lineup_player_sets": int(
            labels["lineup_player_set_signature"].nunique()
        ),
        "published_lineup_confirmed_rows": 0,
        "same_game_pregame_eligible_rows": 0,
        "lagged_future_feature_eligible_rows": int(
            labels["eligible_for_future_game_feature"].sum()
        ),
        "direct_feature_use_allowed_rows": 0,
        "future_games_used_rows": 0,
        "prediction_created": False,
    }
    summary_path = args.output / "lineup_observation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    artifacts = {}
    for path, rows in (
        (labels_path, len(labels)),
        (team_summary_path, len(team_summary)),
        (player_summary_path, len(player_summary)),
        (summary_path, None),
    ):
        entry = {"sha256": sha256(path)}
        if rows is not None:
            entry["rows"] = int(rows)
        artifacts[path.name] = entry
    manifest = {
        "season": args.season,
        "game_type": "R",
        "build_class": "immutable_postgame_lineup_observation_labels",
        "certification": certification,
        "source_hashes": {
            "pitch_events": events_sha,
            "team_games": team_game_sha,
            "roster_snapshot_manifest": sha256(args.snapshot_manifest),
        },
        "artifacts": artifacts,
        "published_lineup_inferred": False,
        "same_game_pregame_use_allowed": False,
        "future_game_feature_use_allowed_after_label_available_at": True,
        "direct_feature_use_allowed": False,
        "official_starting_lineup_confirmed": False,
        "reconstruction_scope": "first_nine_observed_batters_and_first_pitcher_faced",
        "future_games_used": False,
        "prediction_created": False,
        "promotion_status": "label_only_not_pregame_feature",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
