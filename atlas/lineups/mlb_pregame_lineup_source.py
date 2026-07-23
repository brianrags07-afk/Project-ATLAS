"""Official MLB timecoded pregame lineup and probable-starter sources.

The MLB live-feed endpoint can replay the source as it existed at an
explicit UTC ``timecode``.  ATLAS uses that capability to build honest
historical pregame inputs while preserving two separate timestamps:

* ``source_snapshot_at`` -- the official MLB source version at or before
  the prediction cutoff;
* ``archive_retrieved_at`` -- when ATLAS downloaded that historical version.

Historical replay is not represented as a live ATLAS capture.  Normalized
tables whitelist identities, order and source provenance only; postgame
outcomes are never extracted from the feed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

import pandas as pd


SOURCE_VERSION = "mlb_timecoded_pregame_lineup_v1"
SOURCE_NAME = "mlb_stats_api_timecoded_live_feed"
CAPTURE_MODE = "official_historical_replay"
LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

SCHEDULE_COLUMNS = {
    "game_pk",
    "season",
    "game_date_utc",
    "official_date",
    "game_type_code",
    "is_final",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
}

GAME_SNAPSHOT_COLUMNS = [
    "game_pk",
    "season",
    "game_start_at",
    "official_date",
    "prediction_cutoff_at",
    "cutoff_minutes_before_start",
    "requested_timecode",
    "source_snapshot_at",
    "archive_retrieved_at",
    "capture_mode",
    "historical_replay",
    "actual_live_capture",
    "source",
    "source_url",
    "source_status_code",
    "source_detailed_state",
    "source_play_count",
    "source_non_advisory_play_count",
    "source_pitch_count",
    "game_had_started_at_snapshot",
    "snapshot_at_or_before_cutoff",
    "pregame_content_safe",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_lineup_count",
    "away_lineup_count",
    "home_probable_pitcher_id",
    "away_probable_pitcher_id",
    "published_lineups_confirmed",
    "probable_starters_available",
    "outcome_fields_extracted",
    "source_version",
]

LINEUP_COLUMNS = [
    "game_pk",
    "season",
    "game_start_at",
    "official_date",
    "prediction_cutoff_at",
    "source_snapshot_at",
    "team_id",
    "team_name",
    "opponent_team_id",
    "home_away",
    "batting_order",
    "player_id",
    "player_name",
    "bat_side",
    "throw_side",
    "position",
    "source_is_substitute_flag",
    "published_lineup_confirmed",
    "pregame_safe",
    "capture_mode",
    "source",
    "source_url",
    "source_version",
]

STARTER_COLUMNS = [
    "game_pk",
    "season",
    "game_start_at",
    "official_date",
    "prediction_cutoff_at",
    "source_snapshot_at",
    "team_id",
    "team_name",
    "opponent_team_id",
    "home_away",
    "pitcher_id",
    "pitcher_name",
    "confirmation_status",
    "source_field",
    "pregame_safe",
    "capture_mode",
    "source",
    "source_url",
    "source_version",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _integer(value: Any, label: str) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed) or float(parsed) % 1:
        raise ValueError(f"{label} is missing or non-integer: {value!r}")
    return int(parsed)


def format_mlb_timecode(value: Any) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"cannot format invalid MLB timecode: {value!r}")
    return timestamp.strftime("%Y%m%d_%H%M%S")


def parse_mlb_timecode(value: Any) -> pd.Timestamp:
    if value is None or not str(value).strip():
        return pd.NaT
    return pd.to_datetime(
        str(value).strip(), format="%Y%m%d_%H%M%S", utc=True, errors="coerce"
    )


def prepare_completed_regular_games(
    schedule_rows: Iterable[Mapping[str, Any]], *, season: int
) -> pd.DataFrame:
    schedule = pd.DataFrame(list(schedule_rows))
    _require_columns(schedule, SCHEDULE_COLUMNS, "schedule")
    schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
    schedule["game_type_code"] = (
        schedule["game_type_code"].astype("string").str.upper().str.strip()
    )
    games = schedule.loc[
        schedule["season"].eq(int(season))
        & schedule["game_type_code"].eq("R")
        & schedule["is_final"].fillna(False).astype(bool)
    ].copy()
    if games.empty:
        raise ValueError(f"schedule has no completed regular-season games for {season}")
    games["game_pk"] = pd.to_numeric(games["game_pk"], errors="coerce")
    if games["game_pk"].isna().any() or (games["game_pk"] % 1).ne(0).any():
        raise ValueError("schedule contains an invalid game_pk")
    games["game_pk"] = games["game_pk"].astype("int64")
    games["game_start_at"] = pd.to_datetime(
        games["game_date_utc"], utc=True, errors="coerce"
    )
    if games["game_start_at"].isna().any():
        raise ValueError("schedule contains an invalid game_date_utc")
    if games["game_pk"].duplicated().any():
        raise ValueError("schedule contains duplicate completed game_pk values")
    for column in ("home_team_id", "away_team_id"):
        games[column] = pd.to_numeric(games[column], errors="coerce")
        if games[column].isna().any() or (games[column] % 1).ne(0).any():
            raise ValueError(f"schedule contains an invalid {column}")
        games[column] = games[column].astype("int64")
    if games["home_team_id"].eq(games["away_team_id"]).any():
        raise ValueError("schedule contains identical home and away team IDs")
    return games.sort_values(
        ["game_start_at", "game_pk"], kind="stable"
    ).reset_index(drop=True)


def fetch_timecoded_game_feed(
    game_pk: int,
    requested_timecode: str,
    *,
    timeout: int = 45,
    retries: int = 5,
    request_get: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if request_get is None:
        import requests

        request_get = requests.get
    url = LIVE_FEED_URL.format(game_pk=int(game_pk))
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = request_get(
                url,
                params={"timecode": requested_timecode},
                headers={"User-Agent": "Project-ATLAS/1.0 historical-pregame-source"},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if int(payload.get("gamePk") or 0) != int(game_pk):
                raise ValueError(f"MLB feed gamePk mismatch for {game_pk}")
            return payload
        except Exception as exc:  # requests and JSON errors share retry policy
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"could not retrieve MLB timecoded feed for game {game_pk}"
    ) from last_error


def _player_details(
    feed: Mapping[str, Any], team_boxscore: Mapping[str, Any], player_id: int
) -> dict[str, Any]:
    key = f"ID{int(player_id)}"
    team_player = (team_boxscore.get("players") or {}).get(key) or {}
    game_player = ((feed.get("gameData") or {}).get("players") or {}).get(key) or {}
    person = team_player.get("person") or game_player
    return {
        "player_name": person.get("fullName"),
        "bat_side": (game_player.get("batSide") or {}).get("code"),
        "throw_side": (game_player.get("pitchHand") or {}).get("code"),
        "position": (team_player.get("position") or {}).get("abbreviation"),
        "source_is_substitute_flag": bool(
            (team_player.get("gameStatus") or {}).get("isSubstitute", False)
        ),
    }


def normalize_timecoded_pregame_feed(
    feed: Mapping[str, Any],
    schedule_game: Mapping[str, Any],
    *,
    cutoff_minutes: int,
    archive_retrieved_at: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    game_pk = _integer(schedule_game.get("game_pk"), "schedule game_pk")
    if int(feed.get("gamePk") or 0) != game_pk:
        raise ValueError(f"MLB feed gamePk mismatch for {game_pk}")
    if cutoff_minutes <= 0:
        raise ValueError("cutoff_minutes must be positive")

    game_start = pd.to_datetime(
        schedule_game.get("game_start_at", schedule_game.get("game_date_utc")),
        utc=True,
        errors="coerce",
    )
    if pd.isna(game_start):
        raise ValueError(f"game {game_pk} has an invalid start time")
    cutoff = game_start - pd.Timedelta(minutes=int(cutoff_minutes))
    requested_timecode = format_mlb_timecode(cutoff)
    metadata = feed.get("metaData") or {}
    source_snapshot = parse_mlb_timecode(metadata.get("timeStamp"))
    status = (feed.get("gameData") or {}).get("status") or {}
    plays = (feed.get("liveData") or {}).get("plays") or {}
    all_plays = plays.get("allPlays") or []
    play_count = len(all_plays)
    non_advisory_play_count = 0
    pitch_count = 0
    completed_play_count = 0
    for play in all_plays:
        event_type = ((play.get("result") or {}).get("eventType") or "").strip()
        non_advisory_play_count += int(
            bool(event_type) and event_type != "game_advisory"
        )
        completed_play_count += int(bool((play.get("about") or {}).get("isComplete")))
        pitch_count += sum(
            int(bool(event.get("isPitch")))
            for event in (play.get("playEvents") or [])
        )
    game_started = bool(
        non_advisory_play_count or pitch_count or completed_play_count
    )
    snapshot_before_cutoff = bool(
        pd.notna(source_snapshot) and source_snapshot <= cutoff
    )
    pregame_safe = bool(
        snapshot_before_cutoff and cutoff <= game_start and not game_started
    )
    source_url = (
        f"{LIVE_FEED_URL.format(game_pk=game_pk)}?timecode={requested_timecode}"
    )

    game_data = feed.get("gameData") or {}
    boxscore_teams = ((feed.get("liveData") or {}).get("boxscore") or {}).get(
        "teams"
    ) or {}
    probable = game_data.get("probablePitchers") or {}
    lineup_rows: list[dict[str, Any]] = []
    starter_rows: list[dict[str, Any]] = []
    lineup_counts: dict[str, int] = {}
    starter_ids: dict[str, int | None] = {}

    for side, home_away, opponent_side in (
        ("away", "AWAY", "home"),
        ("home", "HOME", "away"),
    ):
        team_id = _integer(schedule_game.get(f"{side}_team_id"), f"{side}_team_id")
        opponent_team_id = _integer(
            schedule_game.get(f"{opponent_side}_team_id"),
            f"{opponent_side}_team_id",
        )
        team_name = schedule_game.get(f"{side}_team_name")
        team_boxscore = boxscore_teams.get(side) or {}
        raw_order = team_boxscore.get("battingOrder") or []
        ordered_ids: list[int] = []
        for raw_player_id in raw_order:
            try:
                ordered_ids.append(_integer(raw_player_id, "battingOrder player_id"))
            except ValueError:
                continue
        lineup_complete = len(ordered_ids) == 9 and len(set(ordered_ids)) == 9
        lineup_counts[side] = len(ordered_ids)
        for position, player_id in enumerate(ordered_ids, start=1):
            details = _player_details(feed, team_boxscore, player_id)
            lineup_rows.append(
                {
                    "game_pk": game_pk,
                    "season": int(schedule_game.get("season")),
                    "game_start_at": game_start,
                    "official_date": schedule_game.get("official_date"),
                    "prediction_cutoff_at": cutoff,
                    "source_snapshot_at": source_snapshot,
                    "team_id": team_id,
                    "team_name": team_name,
                    "opponent_team_id": opponent_team_id,
                    "home_away": home_away,
                    "batting_order": position,
                    "player_id": player_id,
                    **details,
                    "published_lineup_confirmed": lineup_complete and pregame_safe,
                    "pregame_safe": pregame_safe,
                    "capture_mode": CAPTURE_MODE,
                    "source": SOURCE_NAME,
                    "source_url": source_url,
                    "source_version": SOURCE_VERSION,
                }
            )

        pitcher = probable.get(side) or {}
        pitcher_id = pitcher.get("id")
        if pitcher_id is not None:
            pitcher_id = _integer(pitcher_id, "probable pitcher_id")
        starter_ids[side] = pitcher_id
        starter_rows.append(
            {
                "game_pk": game_pk,
                "season": int(schedule_game.get("season")),
                "game_start_at": game_start,
                "official_date": schedule_game.get("official_date"),
                "prediction_cutoff_at": cutoff,
                "source_snapshot_at": source_snapshot,
                "team_id": team_id,
                "team_name": team_name,
                "opponent_team_id": opponent_team_id,
                "home_away": home_away,
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher.get("fullName"),
                "confirmation_status": "probable" if pitcher_id else "unavailable",
                "source_field": "gameData.probablePitchers",
                "pregame_safe": pregame_safe,
                "capture_mode": CAPTURE_MODE,
                "source": SOURCE_NAME,
                "source_url": source_url,
                "source_version": SOURCE_VERSION,
            }
        )

    snapshot = {
        "game_pk": game_pk,
        "season": int(schedule_game.get("season")),
        "game_start_at": game_start,
        "official_date": schedule_game.get("official_date"),
        "prediction_cutoff_at": cutoff,
        "cutoff_minutes_before_start": int(cutoff_minutes),
        "requested_timecode": requested_timecode,
        "source_snapshot_at": source_snapshot,
        "archive_retrieved_at": pd.to_datetime(
            archive_retrieved_at, utc=True, errors="raise"
        ),
        "capture_mode": CAPTURE_MODE,
        "historical_replay": True,
        "actual_live_capture": False,
        "source": SOURCE_NAME,
        "source_url": source_url,
        "source_status_code": status.get("statusCode"),
        "source_detailed_state": status.get("detailedState"),
        "source_play_count": play_count,
        "source_non_advisory_play_count": non_advisory_play_count,
        "source_pitch_count": pitch_count,
        "game_had_started_at_snapshot": game_started,
        "snapshot_at_or_before_cutoff": snapshot_before_cutoff,
        "pregame_content_safe": pregame_safe,
        "home_team_id": _integer(schedule_game.get("home_team_id"), "home_team_id"),
        "home_team_name": schedule_game.get("home_team_name"),
        "away_team_id": _integer(schedule_game.get("away_team_id"), "away_team_id"),
        "away_team_name": schedule_game.get("away_team_name"),
        "home_lineup_count": lineup_counts["home"],
        "away_lineup_count": lineup_counts["away"],
        "home_probable_pitcher_id": starter_ids["home"],
        "away_probable_pitcher_id": starter_ids["away"],
        "published_lineups_confirmed": (
            lineup_counts["home"] == 9
            and lineup_counts["away"] == 9
            and pregame_safe
        ),
        "probable_starters_available": (
            starter_ids["home"] is not None
            and starter_ids["away"] is not None
            and pregame_safe
        ),
        "outcome_fields_extracted": False,
        "source_version": SOURCE_VERSION,
    }
    return snapshot, lineup_rows, starter_rows


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[Any, ...]]:
    if frame.empty:
        return set()
    return set(frame[columns].itertuples(index=False, name=None))


def partition_timecoded_pregame_bundle(
    game_snapshots: pd.DataFrame,
    lineups: pd.DataFrame,
    starters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate model-readable pregame rows from unsafe source snapshots.

    The raw archive remains complete.  A snapshot is eligible for normalized
    pregame use only when the official source version is timestamped at or
    before the requested cutoff and contains no game action.  Games that fail
    either proof are preserved in a quarantine table and contribute explicit
    coverage gaps; their lineup and starter rows never enter model-readable
    outputs.
    """

    for frame, columns, label in (
        (game_snapshots, GAME_SNAPSHOT_COLUMNS, "game snapshots"),
        (lineups, LINEUP_COLUMNS, "lineups"),
        (starters, STARTER_COLUMNS, "starters"),
    ):
        _require_columns(frame, set(columns), label)

    safe_mask = (
        game_snapshots["pregame_content_safe"].fillna(False).astype(bool)
        & ~game_snapshots["game_had_started_at_snapshot"]
        .fillna(True)
        .astype(bool)
    )
    safe_game_snapshots = game_snapshots.loc[safe_mask].copy()
    quarantined_game_snapshots = game_snapshots.loc[~safe_mask].copy()
    safe_game_ids = set(safe_game_snapshots["game_pk"].astype(int))
    safe_lineups = lineups.loc[
        lineups["game_pk"].astype(int).isin(safe_game_ids)
    ].copy()
    safe_starters = starters.loc[
        starters["game_pk"].astype(int).isin(safe_game_ids)
    ].copy()
    return (
        safe_game_snapshots.reset_index(drop=True),
        safe_lineups.reset_index(drop=True),
        safe_starters.reset_index(drop=True),
        quarantined_game_snapshots.reset_index(drop=True),
    )


def certify_timecoded_pregame_bundle(
    game_snapshots: pd.DataFrame,
    lineups: pd.DataFrame,
    starters: pd.DataFrame,
    expected_games: pd.DataFrame,
    *,
    season: int,
) -> dict[str, Any]:
    errors: list[str] = []
    for frame, columns, label in (
        (game_snapshots, GAME_SNAPSHOT_COLUMNS, "game snapshots"),
        (lineups, LINEUP_COLUMNS, "lineups"),
        (starters, STARTER_COLUMNS, "starters"),
    ):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            errors.append(f"{label} missing columns: {missing}")
    if errors:
        return {"verdict": "not_ready", "errors": errors}

    expected_game_ids = set(expected_games["game_pk"].astype(int))
    actual_game_ids = set(game_snapshots["game_pk"].astype(int))
    missing_games = sorted(expected_game_ids.difference(actual_game_ids))
    unexpected_games = sorted(actual_game_ids.difference(expected_game_ids))
    if missing_games:
        errors.append(f"missing game snapshots: {missing_games[:20]}")
    if unexpected_games:
        errors.append(f"unexpected game snapshots: {unexpected_games[:20]}")
    if game_snapshots["game_pk"].duplicated().any():
        errors.append("duplicate game snapshots detected")
    if not game_snapshots["season"].eq(int(season)).all():
        errors.append(f"game snapshots contain rows outside season {season}")
    if game_snapshots["outcome_fields_extracted"].fillna(True).astype(bool).any():
        errors.append("one or more snapshots claim outcome fields were extracted")
    if not game_snapshots["historical_replay"].fillna(False).astype(bool).all():
        errors.append("historical replay flag is missing")
    if game_snapshots["actual_live_capture"].fillna(True).astype(bool).any():
        errors.append("historical replay cannot be marked as an actual live capture")

    lineup_duplicate_keys = int(
        lineups.duplicated(["game_pk", "team_id", "batting_order"]).sum()
    )
    lineup_duplicate_players = int(
        lineups.duplicated(["game_pk", "team_id", "player_id"]).sum()
    )
    if lineup_duplicate_keys:
        errors.append("duplicate game/team/batting-order rows detected")
    if lineup_duplicate_players:
        errors.append("duplicate players within a team lineup detected")
    if not lineups["pregame_safe"].fillna(False).astype(bool).all():
        errors.append("one or more lineup rows are not pregame safe")

    safe_snapshot_mask = (
        game_snapshots["pregame_content_safe"].fillna(False).astype(bool)
        & ~game_snapshots["game_had_started_at_snapshot"]
        .fillna(True)
        .astype(bool)
    )
    safe_game_ids = set(
        game_snapshots.loc[safe_snapshot_mask, "game_pk"].astype(int)
    )
    quarantined_game_ids = sorted(actual_game_ids.difference(safe_game_ids))
    lineup_game_ids = set(lineups["game_pk"].astype(int))
    starter_game_ids = set(starters["game_pk"].astype(int))
    if not lineup_game_ids.issubset(safe_game_ids):
        errors.append("lineup rows include quarantined source snapshots")
    if not starter_game_ids.issubset(safe_game_ids):
        errors.append("starter rows include quarantined source snapshots")
    unsafe_model_game_ids = (
        lineup_game_ids.union(starter_game_ids).difference(safe_game_ids)
    )
    if (
        game_snapshots.loc[
            game_snapshots["game_pk"].astype(int).isin(unsafe_model_game_ids),
            "game_had_started_at_snapshot",
        ]
        .fillna(True)
        .astype(bool)
        .any()
    ):
        errors.append("one or more source snapshots already contain game action")

    expected_team_keys = set()
    safe_expected_team_keys = set()
    for row in expected_games.to_dict("records"):
        game_pk = int(row["game_pk"])
        team_keys = {
            (game_pk, int(row["home_team_id"])),
            (game_pk, int(row["away_team_id"])),
        }
        expected_team_keys.update(team_keys)
        if game_pk in safe_game_ids:
            safe_expected_team_keys.update(team_keys)
    starter_team_keys = _keys(starters, ["game_pk", "team_id"])
    if starter_team_keys != safe_expected_team_keys:
        errors.append(
            "starter rows do not cover exactly two scheduled teams per safe game"
        )
    if starters.duplicated(["game_pk", "team_id"]).any():
        errors.append("duplicate game/team starter rows detected")
    if not starters["pregame_safe"].fillna(False).astype(bool).all():
        errors.append("one or more starter rows are not pregame safe")

    lineup_sizes = lineups.groupby(["game_pk", "team_id"]).size()
    complete_team_lineups = int(lineup_sizes.eq(9).sum())
    expected_team_games = len(expected_team_keys)
    complete_games = int(
        game_snapshots["published_lineups_confirmed"].fillna(False).sum()
    )
    probable_starter_rows = int(starters["pitcher_id"].notna().sum())
    snapshot_after_cutoff_game_ids = sorted(
        game_snapshots.loc[
            ~game_snapshots["snapshot_at_or_before_cutoff"]
            .fillna(False)
            .astype(bool),
            "game_pk",
        ].astype(int)
    )
    game_action_at_snapshot_game_ids = sorted(
        game_snapshots.loc[
            game_snapshots["game_had_started_at_snapshot"]
            .fillna(True)
            .astype(bool),
            "game_pk",
        ].astype(int)
    )
    verdict = (
        "quarantine_required"
        if errors
        else (
            "certified_with_documented_gaps"
            if quarantined_game_ids
            else "certified"
        )
    )
    return {
        "verdict": verdict,
        "season": int(season),
        "games": int(len(actual_game_ids)),
        "expected_games": int(len(expected_game_ids)),
        "pregame_safe_games": int(len(safe_game_ids)),
        "quarantined_games": int(len(quarantined_game_ids)),
        "quarantined_game_ids": quarantined_game_ids,
        "snapshot_after_cutoff_games": int(
            len(snapshot_after_cutoff_game_ids)
        ),
        "snapshot_after_cutoff_game_ids": snapshot_after_cutoff_game_ids,
        "game_action_at_snapshot_games": int(
            len(game_action_at_snapshot_game_ids)
        ),
        "game_action_at_snapshot_game_ids": game_action_at_snapshot_game_ids,
        "team_games": int(expected_team_games),
        "pregame_safe_team_games": int(len(safe_expected_team_keys)),
        "lineup_rows": int(len(lineups)),
        "starter_rows": int(len(starters)),
        "complete_team_lineups": complete_team_lineups,
        "incomplete_or_missing_team_lineups": int(
            expected_team_games - complete_team_lineups
        ),
        "games_with_both_confirmed_lineups": complete_games,
        "probable_starter_rows": probable_starter_rows,
        "missing_probable_starter_rows": int(
            expected_team_games - probable_starter_rows
        ),
        "missing_probable_starter_rows_within_safe_games": int(
            len(safe_expected_team_keys) - probable_starter_rows
        ),
        "lineup_duplicate_keys": lineup_duplicate_keys,
        "lineup_duplicate_players": lineup_duplicate_players,
        "missing_games": missing_games,
        "unexpected_games": unexpected_games,
        "outcome_fields_extracted": 0,
        "errors": errors,
    }
