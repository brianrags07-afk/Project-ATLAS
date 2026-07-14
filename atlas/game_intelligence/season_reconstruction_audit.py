"""
Full-season reconstruction audit for the ATLAS Baseball Brain.

This module validates whether every eligible historical game can
be reconstructed from the canonical ATLAS source products.

It does not create predictions or update identities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

from .contracts import (
    BRAIN_ENGINE_VERSION,
    PHASE_1_INVARIANTS,
    RECONSTRUCTION_AUDIT_DIR,
    assert_canonical_sources_exist,
)
from .reconstruction import (
    RECONSTRUCTION_ENGINE_VERSION,
    _all_available_safety_checks_pass,
    _extract_game_teams,
    _extract_team_set,
    _pregame_safety_checks,
    _score_sources,
    _scores_agree,
    _season_matches,
)


SEASON_AUDIT_ENGINE_VERSION = "1.0.0"


def _atomic_parquet_write(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_parquet(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        destination,
    )


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=destination.parent,
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            default=str,
        )

        temporary_name = handle.name

    os.replace(
        temporary_name,
        destination,
    )


def _normalize(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    if "game_pk" in result.columns:
        result["game_pk"] = pd.to_numeric(
            result["game_pk"],
            errors="coerce",
        ).astype("Int64")

    if "game_date" in result.columns:
        result["game_date"] = pd.to_datetime(
            result["game_date"],
            errors="coerce",
        ).dt.normalize()

    if "atlas_season" in result.columns:
        result["atlas_season"] = pd.to_numeric(
            result["atlas_season"],
            errors="coerce",
        ).astype("Int64")

    return result


def _filter_season(
    dataframe: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    result = _normalize(
        dataframe
    )

    if "atlas_season" in result.columns:
        result = result[
            result["atlas_season"].eq(
                int(season)
            )
        ].copy()

    elif "game_year" in result.columns:
        years = pd.to_numeric(
            result["game_year"],
            errors="coerce",
        )

        result = result[
            years.eq(int(season))
        ].copy()

    elif "game_date" in result.columns:
        result = result[
            result["game_date"]
            .dt.year
            .eq(int(season))
        ].copy()

    return result.reset_index(
        drop=True
    )


def _group_by_game(
    dataframe: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    if (
        dataframe.empty
        or "game_pk" not in dataframe.columns
    ):
        return {}

    return {
        int(game_pk): group.reset_index(
            drop=True
        )
        for game_pk, group in dataframe.groupby(
            "game_pk",
            sort=False,
            dropna=True,
        )
    }


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _bool_all_false(
    dataframe: pd.DataFrame,
    column: str,
) -> bool | None:
    if column not in dataframe.columns:
        return None

    values = dataframe[
        column
    ].dropna()

    if values.empty:
        return None

    return bool(
        (~values.astype(bool)).all()
    )


def _bool_all_true(
    dataframe: pd.DataFrame,
    column: str,
) -> bool | None:
    if column not in dataframe.columns:
        return None

    values = dataframe[
        column
    ].dropna()

    if values.empty:
        return None

    return bool(
        values.astype(bool).all()
    )


def _classify_exception(
    game_pk: int,
    game_core: pd.DataFrame,
    manifest: pd.DataFrame,
    pregame: pd.DataFrame,
    lineups: pd.DataFrame,
    bullpens: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
) -> str:
    known_exceptions = set(
        PHASE_1_INVARIANTS[
            "known_exception_game_pks"
        ]
    )

    if int(game_pk) in known_exceptions:
        return "known_source_coverage_exception"

    missing = []

    for label, dataframe in [
        ("game_core", game_core),
        ("manifest", manifest),
        ("pregame", pregame),
        ("lineups", lineups),
        ("bullpens", bullpens),
        ("events", events),
        ("targets", targets),
    ]:
        if dataframe.empty:
            missing.append(label)

    if missing:
        return (
            "unexpected_missing_sources:"
            + ",".join(missing)
        )

    return "none"


def _audit_one_game(
    *,
    game_pk: int,
    season: int,
    game_core: pd.DataFrame,
    manifest: pd.DataFrame,
    pregame: pd.DataFrame,
    lineups: pd.DataFrame,
    bullpens: pd.DataFrame,
    events: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, Any]:
    game_teams = _extract_game_teams(
        game_core
    )

    pregame_teams = _extract_team_set(
        pregame
    )

    lineup_teams = _extract_team_set(
        lineups
    )

    bullpen_teams = _extract_team_set(
        bullpens
    )

    scores = _score_sources(
        game_core=game_core,
        events=events,
        targets=targets,
    )

    safety_checks = (
        _pregame_safety_checks(
            pregame_teams=pregame,
            bullpens=bullpens,
        )
    )

    duplicate_pregame = (
        int(
            pregame.duplicated(
                subset=[
                    "game_pk",
                    "team",
                ]
            ).sum()
        )
        if all(
            column in pregame.columns
            for column in [
                "game_pk",
                "team",
            ]
        )
        else None
    )

    duplicate_lineups = (
        int(
            lineups.duplicated(
                subset=[
                    "game_pk",
                    "team",
                ]
            ).sum()
        )
        if all(
            column in lineups.columns
            for column in [
                "game_pk",
                "team",
            ]
        )
        else None
    )

    duplicate_bullpens = (
        int(
            bullpens.duplicated(
                subset=[
                    "game_pk",
                    "team",
                ]
            ).sum()
        )
        if all(
            column in bullpens.columns
            for column in [
                "game_pk",
                "team",
            ]
        )
        else None
    )

    exception_class = _classify_exception(
        game_pk=game_pk,
        game_core=game_core,
        manifest=manifest,
        pregame=pregame,
        lineups=lineups,
        bullpens=bullpens,
        events=events,
        targets=targets,
    )

    known_exception = (
        exception_class
        == "known_source_coverage_exception"
    )

    one_game_core_row = (
        len(game_core) == 1
    )

    one_manifest_row = (
        len(manifest) == 1
    )

    two_pregame_rows = (
        len(pregame) == 2
    )

    two_lineup_rows = (
        len(lineups) == 2
    )

    two_bullpen_rows = (
        len(bullpens) == 2
    )

    events_available = (
        len(events) > 0
    )

    one_target_row = (
        len(targets) == 1
    )

    teams_match = bool(
        game_teams
        and pregame_teams == game_teams
        and lineup_teams == game_teams
        and bullpen_teams == game_teams
    )

    duplicates_pass = bool(
        duplicate_pregame == 0
        and duplicate_lineups == 0
        and duplicate_bullpens == 0
    )

    scores_match = _scores_agree(
        scores
    )

    safety_pass = (
        _all_available_safety_checks_pass(
            safety_checks
        )
    )

    normal_pass = bool(
        one_game_core_row
        and one_manifest_row
        and two_pregame_rows
        and two_lineup_rows
        and two_bullpen_rows
        and events_available
        and one_target_row
        and teams_match
        and duplicates_pass
        and scores_match
        and safety_pass
        and _season_matches(
            game_core,
            season,
        )
        and _season_matches(
            pregame,
            season,
        )
        and _season_matches(
            lineups,
            season,
        )
        and _season_matches(
            bullpens,
            season,
        )
        and _season_matches(
            events,
            season,
        )
        and _season_matches(
            targets,
            season,
        )
    )

    audit_status = (
        "pass"
        if normal_pass
        else (
            "known_exception"
            if known_exception
            else "fail"
        )
    )

    game_date = (
        game_core["game_date"].iloc[0]
        if (
            not game_core.empty
            and "game_date"
            in game_core.columns
        )
        else None
    )

    home_team = (
        str(
            game_core[
                "home_team"
            ].iloc[0]
        )
        if (
            not game_core.empty
            and "home_team"
            in game_core.columns
        )
        else None
    )

    away_team = (
        str(
            game_core[
                "away_team"
            ].iloc[0]
        )
        if (
            not game_core.empty
            and "away_team"
            in game_core.columns
        )
        else None
    )

    return {
        "game_pk":
            int(game_pk),

        "game_date":
            game_date,

        "atlas_season":
            int(season),

        "home_team":
            home_team,

        "away_team":
            away_team,

        "game_core_rows":
            int(len(game_core)),

        "manifest_rows":
            int(len(manifest)),

        "pregame_team_rows":
            int(len(pregame)),

        "lineup_rows":
            int(len(lineups)),

        "bullpen_rows":
            int(len(bullpens)),

        "event_rows":
            int(len(events)),

        "target_rows":
            int(len(targets)),

        "one_game_core_row":
            one_game_core_row,

        "one_manifest_row":
            one_manifest_row,

        "two_pregame_team_rows":
            two_pregame_rows,

        "two_lineup_rows":
            two_lineup_rows,

        "two_bullpen_rows":
            two_bullpen_rows,

        "events_available":
            events_available,

        "one_target_row":
            one_target_row,

        "teams_match":
            teams_match,

        "duplicate_pregame_team_rows":
            duplicate_pregame,

        "duplicate_lineup_team_rows":
            duplicate_lineups,

        "duplicate_bullpen_team_rows":
            duplicate_bullpens,

        "duplicates_pass":
            duplicates_pass,

        "game_core_home_score":
            scores["game_core"][
                "home_score"
            ],

        "game_core_away_score":
            scores["game_core"][
                "away_score"
            ],

        "event_home_score":
            scores["events"][
                "home_score"
            ],

        "event_away_score":
            scores["events"][
                "away_score"
            ],

        "target_home_score":
            scores["targets"][
                "home_score"
            ],

        "target_away_score":
            scores["targets"][
                "away_score"
            ],

        "scores_agree":
            scores_match,

        "pregame_safety_checks_available":
            int(
                sum(
                    value is not None
                    for value
                    in safety_checks.values()
                )
            ),

        "pregame_safety_pass":
            safety_pass,

        "exception_class":
            exception_class,

        "known_exception":
            known_exception,

        "audit_status":
            audit_status,

        "reconstruction_pass":
            normal_pass,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "future_games_used":
            False,

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "reconstruction_engine_version":
            RECONSTRUCTION_ENGINE_VERSION,

        "season_audit_engine_version":
            SEASON_AUDIT_ENGINE_VERSION,
    }


def run_season_reconstruction_audit(
    season: int = 2024,
    save_outputs: bool = True,
) -> dict[str, Any]:
    started = time.time()
    season = int(season)

    paths = assert_canonical_sources_exist(
        season=season,
        require_pitcher_sources=False,
    )

    print("=" * 78)
    print(
        "ATLAS FULL-SEASON RECONSTRUCTION AUDIT"
    )
    print("=" * 78)
    print(
        f"Season.................... {season}"
    )
    print(
        "Loading canonical sources once..."
    )

    game_core = _filter_season(
        pd.read_parquet(
            paths["game_card_core"]
        ),
        season,
    )

    manifest = _filter_season(
        pd.read_parquet(
            paths["game_card_manifest"]
        ),
        season,
    )

    pregame = _filter_season(
        pd.read_parquet(
            paths["pregame_interactions"]
        ),
        season,
    )

    lineups = _filter_season(
        pd.read_parquet(
            paths["historical_lineups"]
        ),
        season,
    )

    bullpens = _filter_season(
        pd.read_parquet(
            paths["bullpen_pregame_state"]
        ),
        season,
    )

    events = _filter_season(
        pd.read_parquet(
            paths["game_events"]
        ),
        season,
    )

    targets = _filter_season(
        pd.read_parquet(
            paths["game_targets"]
        ),
        season,
    )

    grouped = {
        "game_core":
            _group_by_game(game_core),

        "manifest":
            _group_by_game(manifest),

        "pregame":
            _group_by_game(pregame),

        "lineups":
            _group_by_game(lineups),

        "bullpens":
            _group_by_game(bullpens),

        "events":
            _group_by_game(events),

        "targets":
            _group_by_game(targets),
    }

    game_pks = sorted(
        set(
            grouped["game_core"].keys()
        )
    )

    print(
        f"Game-card games........... "
        f"{len(game_pks):,}"
    )
    print(
        f"Pregame games............ "
        f"{len(grouped['pregame']):,}"
    )
    print(
        f"Event-store games........ "
        f"{len(grouped['events']):,}"
    )
    print(
        f"Target games.............. "
        f"{len(grouped['targets']):,}"
    )
    print("=" * 78)

    records = []

    for index, game_pk in enumerate(
        game_pks,
        start=1,
    ):
        records.append(
            _audit_one_game(
                game_pk=game_pk,
                season=season,
                game_core=grouped[
                    "game_core"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
                manifest=grouped[
                    "manifest"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
                pregame=grouped[
                    "pregame"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
                lineups=grouped[
                    "lineups"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
                bullpens=grouped[
                    "bullpens"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
                events=grouped[
                    "events"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
                targets=grouped[
                    "targets"
                ].get(
                    game_pk,
                    _empty_frame(),
                ),
            )
        )

        if (
            index % 250 == 0
            or index == len(game_pks)
        ):
            print(
                f"Audited {index:>4,}/"
                f"{len(game_pks):,} games"
            )

    audit = pd.DataFrame(
        records
    ).sort_values(
        [
            "game_date",
            "game_pk",
        ],
        kind="stable",
    ).reset_index(drop=True)

    status_counts = (
        audit["audit_status"]
        .value_counts()
    )

    failures = audit[
        audit["audit_status"].eq(
            "fail"
        )
    ].copy()

    known_exceptions = audit[
        audit["audit_status"].eq(
            "known_exception"
        )
    ].copy()

    duplicate_game_ids = int(
        audit["game_pk"]
        .duplicated()
        .sum()
    )

    if duplicate_game_ids:
        raise AssertionError(
            "Duplicate game IDs in season audit: "
            f"{duplicate_game_ids}"
        )

    output_dir = (
        RECONSTRUCTION_AUDIT_DIR
        / str(season)
    )

    audit_path = (
        output_dir
        / "season_reconstruction_audit.parquet"
    )

    failure_path = (
        output_dir
        / "season_reconstruction_failures.parquet"
    )

    exception_path = (
        output_dir
        / "season_reconstruction_exceptions.parquet"
    )

    metadata_path = (
        output_dir
        / "season_reconstruction_audit_metadata.json"
    )

    elapsed = time.time() - started

    result = {
        "engine":
            "ATLAS Season Reconstruction Audit",

        "brain_engine_version":
            BRAIN_ENGINE_VERSION,

        "reconstruction_engine_version":
            RECONSTRUCTION_ENGINE_VERSION,

        "season_audit_engine_version":
            SEASON_AUDIT_ENGINE_VERSION,

        "season":
            season,

        "game_card_games":
            int(len(game_pks)),

        "games_audited":
            int(len(audit)),

        "passed":
            int(
                status_counts.get(
                    "pass",
                    0,
                )
            ),

        "known_exceptions":
            int(
                status_counts.get(
                    "known_exception",
                    0,
                )
            ),

        "unexpected_failures":
            int(
                status_counts.get(
                    "fail",
                    0,
                )
            ),

        "score_mismatches":
            int(
                (~audit["scores_agree"])
                .sum()
            ),

        "pregame_safety_failures":
            int(
                (~audit["pregame_safety_pass"])
                .sum()
            ),

        "duplicate_game_ids":
            duplicate_game_ids,

        "prediction_created":
            False,

        "identity_updated":
            False,

        "future_games_used":
            False,

        "phase_1b_pass":
            bool(
                len(failures) == 0
                and duplicate_game_ids == 0
            ),

        "elapsed_seconds":
            float(elapsed),

        "outputs": {
            "audit": (
                str(audit_path)
                if save_outputs
                else None
            ),
            "failures": (
                str(failure_path)
                if save_outputs
                else None
            ),
            "exceptions": (
                str(exception_path)
                if save_outputs
                else None
            ),
            "metadata": (
                str(metadata_path)
                if save_outputs
                else None
            ),
        },

        "built_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    if save_outputs:
        _atomic_parquet_write(
            audit,
            audit_path,
        )

        _atomic_parquet_write(
            failures,
            failure_path,
        )

        _atomic_parquet_write(
            known_exceptions,
            exception_path,
        )

        _atomic_json_write(
            result,
            metadata_path,
        )

    print()
    print("=" * 78)
    print(
        "SEASON RECONSTRUCTION AUDIT COMPLETE"
    )
    print("=" * 78)
    print(
        f"Games Audited............ "
        f"{result['games_audited']:,}"
    )
    print(
        f"Passed................... "
        f"{result['passed']:,}"
    )
    print(
        f"Known Exceptions......... "
        f"{result['known_exceptions']:,}"
    )
    print(
        f"Unexpected Failures...... "
        f"{result['unexpected_failures']:,}"
    )
    print(
        f"Score Mismatches......... "
        f"{result['score_mismatches']:,}"
    )
    print(
        f"Pregame Safety Failures.. "
        f"{result['pregame_safety_failures']:,}"
    )
    print(
        f"Duplicate Game IDs....... "
        f"{result['duplicate_game_ids']:,}"
    )
    print(
        f"Phase 1B Pass............ "
        f"{result['phase_1b_pass']}"
    )
    print(
        f"Elapsed.................. "
        f"{elapsed / 60:.1f} minutes"
    )
    print("=" * 78)

    return {
        "result": result,
        "audit": audit,
        "failures": failures,
        "exceptions": known_exceptions,
    }
