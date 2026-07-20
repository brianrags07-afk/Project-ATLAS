"""
Authoritative MLB schedule reference layer.

This module builds a canonical, one-row-per-MLB-game reference dataset
from the published MLB Stats API ``/schedule`` endpoint
(``{MLB_API}/schedule``). It is the ONLY source of pregame-safe schedule
and series-context facts recognized by the ATLAS historical readiness
audit (see ``atlas.audit.schedule_source_assessment``).

Hard rules enforced by this module:

  - Every field that describes *what game is scheduled*, *when*, *what
    type of game it is* (regular season vs. postseason), and *series
    context* (``gamesInSeries`` / ``seriesGameNumber`` / ``seriesDescription``)
    comes directly from the MLB Stats API's own published schedule fields.
    Nothing here is inferred from scores, ``status`` completion, historical
    game counts, ``master_game_database``, or pitch-by-pitch history.
  - The MLB Stats API's own durable identifier, ``gamePk``, is the only
    primary key ever used to identify a game. Team/player *names* are
    carried as labels only and are never used as a durable key.
  - Postponements, cancellations, suspensions, and doubleheaders are
    classified using only the API's own published ``status`` fields
    (``status.detailedState`` / ``status.statusCode`` / ``status.codedGameState``)
    and schedule fields (``doubleHeader`` / ``gameNumber``). A postponed
    game that is later replayed keeps the same ``gamePk`` under MLB's own
    scheduling model, so de-duplicating by ``gamePk`` across however many
    raw API payloads are merged together is sufficient to prevent counting
    the same real-world game twice.
  - This module never reads or writes Cloud Storage. It is a pure
    fetch-from-MLB / normalize-in-memory layer; it is read-only with
    respect to any existing Cloud Storage data, exactly like the rest of
    the ATLAS audit tooling.

Per the ATLAS Copilot instructions for data builders: this module does not
touch ``atlas_reference`` schemas/registries/samples at all, because a
published external schedule source is, by design, independent of and
authoritative over anything derived internally (e.g. ``master_game_database``,
``master_pitch_database``). Column names below are either taken verbatim
from the MLB Stats API's own published response fields, or are explicitly
documented new canonical field names -- none are invented registry column
names for an existing ATLAS dataset.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from atlas.config import MLB_API
from atlas.utils.api import get_json, safe_get

SCHEDULE_ENDPOINT = f"{MLB_API}/schedule"
SCHEDULE_SOURCE_NAME = "mlb_stats_api_schedule"

# --------------------------------------------------------------------------
# Official MLB Stats API ``gameType`` codes (published field:
# ``games[].gameType``). These codes are documented/observed values of an
# existing published field -- this mapping classifies known values, it does
# not invent a new column. Any code not present here is left as
# "unknown" rather than guessed.
# --------------------------------------------------------------------------
GAME_TYPE_SEASON_SEGMENT: dict[str, str] = {
    "R": "regular_season",
    "F": "postseason",  # Wild Card
    "D": "postseason",  # Division Series
    "L": "postseason",  # League Championship Series
    "W": "postseason",  # World Series
    "A": "all_star",
    "S": "spring_training",
    "E": "exhibition",
}

# --------------------------------------------------------------------------
# Official MLB Stats API ``status.detailedState`` values (published field:
# ``games[].status.detailedState``). As above, this classifies known,
# observed values of an existing published field; an unrecognized string
# is left as "unknown" rather than guessed.
# --------------------------------------------------------------------------
DETAILED_STATE_CATEGORY: dict[str, str] = {
    "Scheduled": "scheduled",
    "Pre-Game": "scheduled",
    "Warmup": "scheduled",
    "In Progress": "in_progress",
    "Manager Challenge": "in_progress",
    "Delayed": "delayed",
    "Delayed Start": "delayed",
    "Delayed: Rain": "delayed",
    "Suspended": "suspended",
    "Suspended: Rain": "suspended",
    "Postponed": "postponed",
    "Cancelled": "cancelled",
    "Final": "final",
    "Game Over": "final",
    "Completed Early": "final",
    "Completed Early: Rain": "final",
    "Forfeit": "final",
}

# Priority used only to pick a single winning raw record when the same
# ``gamePk`` appears more than once across one or more merged raw API
# payloads (e.g. overlapping date-range calls). Higher priority values win.
# This never changes *which* game is scheduled -- only which duplicate
# published snapshot of that same gamePk is kept.
_STATE_CATEGORY_PRIORITY = {
    "final": 5,
    "in_progress": 4,
    "suspended": 3,
    "delayed": 3,
    "postponed": 2,
    "cancelled": 2,
    "scheduled": 1,
    "unknown": 0,
}

CANONICAL_FIELDS = (
    "game_pk",
    "game_guid",
    "season",
    "game_date_utc",
    "official_date",
    "game_type_code",
    "season_segment",
    "status_code",
    "coded_game_state",
    "abstract_game_state",
    "detailed_state",
    "game_state_category",
    "is_final",
    "counted_in_expected_games",
    "double_header_code",
    "game_number",
    "series_game_number",
    "games_in_series",
    "series_description",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "venue_id",
    "venue_name",
)


def fetch_schedule_raw(
    start_date: str,
    end_date: str,
    *,
    sport_id: int = 1,
    game_types: Sequence[str] | None = None,
    timeout: int = 30,
    retries: int = 3,
) -> dict[str, Any]:
    """Fetch one raw ``/schedule`` payload from the published MLB Stats API.

    ``game_types`` is an optional filter passed straight through as the
    API's own ``gameType`` query parameter (e.g. ``["R"]`` for regular
    season only). When omitted, no filter is applied and the API's default
    set of game types for the date range is returned unmodified.

    This function only reads from the public MLB Stats API. It never
    reads or writes Cloud Storage or any ATLAS dataset.
    """
    params: dict[str, Any] = {
        "sportId": sport_id,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team,venue",
    }
    if game_types:
        params["gameType"] = ",".join(game_types)
    return get_json(SCHEDULE_ENDPOINT, params=params, timeout=timeout, retries=retries)


def extract_raw_games(raw_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a raw ``/schedule`` payload's ``dates[].games[]`` into one
    flat list, exactly as published -- no field is added, removed, or
    reinterpreted here."""
    games: list[dict[str, Any]] = []
    for date_entry in safe_get(raw_payload, ["dates"], []) or []:
        games.extend(date_entry.get("games", []) or [])
    return games


def _row_content_hash(fields: Mapping[str, Any]) -> str:
    """Stable content hash of a canonical row's fields, excluding volatile
    fetch-time provenance (``retrieved_at_utc`` / ``source_url``), so the
    hash reflects only the game's own published content and changes
    exactly when that content changes (e.g. Scheduled -> Final)."""
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_game_row(
    raw_game: Mapping[str, Any],
    *,
    retrieved_at_utc: str,
    source_url: str,
) -> dict[str, Any]:
    """Normalize one raw ``/schedule`` game dict into a canonical row.

    Every value comes directly from a published MLB Stats API field. Series
    context (``games_in_series`` / ``series_game_number`` /
    ``series_description``) is copied verbatim from the API's own fields --
    it is never computed by counting completed games.
    """
    status = raw_game.get("status") or {}
    detailed_state = status.get("detailedState")
    game_state_category = DETAILED_STATE_CATEGORY.get(detailed_state, "unknown")

    game_type_code = raw_game.get("gameType")
    season_segment = GAME_TYPE_SEASON_SEGMENT.get(game_type_code, "unknown")

    home = safe_get(raw_game, ["teams", "home", "team"], {}) or {}
    away = safe_get(raw_game, ["teams", "away", "team"], {}) or {}
    venue = raw_game.get("venue") or {}

    # A game is counted toward the season's expected-game total unless it
    # was cancelled outright with no makeup game to represent it. Games
    # that are postponed-and-later-replayed keep the same gamePk under
    # MLB's own scheduling model and therefore are still counted exactly
    # once via that single gamePk row.
    counted_in_expected_games = game_state_category != "cancelled"

    fields = {
        "game_pk": raw_game.get("gamePk"),
        "game_guid": raw_game.get("gameGuid"),
        "season": raw_game.get("season"),
        "game_date_utc": raw_game.get("gameDate"),
        "official_date": raw_game.get("officialDate"),
        "game_type_code": game_type_code,
        "season_segment": season_segment,
        "status_code": status.get("statusCode"),
        "coded_game_state": status.get("codedGameState"),
        "abstract_game_state": status.get("abstractGameState"),
        "detailed_state": detailed_state,
        "game_state_category": game_state_category,
        "is_final": game_state_category == "final",
        "counted_in_expected_games": counted_in_expected_games,
        "double_header_code": raw_game.get("doubleHeader"),
        "game_number": raw_game.get("gameNumber"),
        "series_game_number": raw_game.get("seriesGameNumber"),
        "games_in_series": raw_game.get("gamesInSeries"),
        "series_description": raw_game.get("seriesDescription"),
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
    }

    row = dict(fields)
    row["content_hash"] = _row_content_hash(fields)
    row["source"] = SCHEDULE_SOURCE_NAME
    row["source_url"] = source_url
    row["retrieved_at_utc"] = retrieved_at_utc
    return row


def _priority(row: Mapping[str, Any]) -> int:
    return _STATE_CATEGORY_PRIORITY.get(row.get("game_state_category"), 0)


def normalize_schedule(
    raw_payloads: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    retrieved_at_utc: str | None = None,
    source_url: str = SCHEDULE_ENDPOINT,
) -> list[dict[str, Any]]:
    """Normalize one or more raw ``/schedule`` payloads into one canonical
    row per MLB game, de-duplicated by ``gamePk``.

    Accepts either a single raw payload dict or an iterable of raw payload
    dicts (e.g. results from several date-range calls covering a full
    season). When the same ``gamePk`` is present in more than one payload
    (for example because of overlapping date-range queries around a
    postponement/reschedule), the most authoritative published status
    snapshot (Final > In Progress > Suspended/Delayed > Postponed/Cancelled
    > Scheduled) is kept and the rest are discarded, so a rescheduled game
    is represented exactly once.
    """
    if retrieved_at_utc is None:
        retrieved_at_utc = datetime.now(timezone.utc).isoformat()

    if isinstance(raw_payloads, Mapping):
        payloads: Iterable[Mapping[str, Any]] = [raw_payloads]
    else:
        payloads = raw_payloads

    by_game_pk: dict[Any, dict[str, Any]] = {}
    for payload in payloads:
        for raw_game in extract_raw_games(payload):
            row = normalize_game_row(
                raw_game, retrieved_at_utc=retrieved_at_utc, source_url=source_url
            )
            game_pk = row["game_pk"]
            existing = by_game_pk.get(game_pk)
            if existing is None or _priority(row) >= _priority(existing):
                by_game_pk[game_pk] = row

    rows = list(by_game_pk.values())
    rows.sort(key=lambda r: (r.get("game_date_utc") or "", r.get("game_pk") or 0))
    return rows


def regular_season_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only rows whose published ``gameType`` classifies as regular
    season (``season_segment == "regular_season"``)."""
    return [dict(r) for r in rows if r.get("season_segment") == "regular_season"]


def postseason_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only rows whose published ``gameType`` classifies as
    postseason (``season_segment == "postseason"``)."""
    return [dict(r) for r in rows if r.get("season_segment") == "postseason"]


def _counts_by_season(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not row.get("counted_in_expected_games"):
            continue
        season = row.get("season")
        if season is None:
            continue
        season_key = str(season)
        counts[season_key] = counts.get(season_key, 0) + 1
    return {season: counts[season] for season in sorted(counts)}


def build_season_schedule_profile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the audit-compatible ``season_schedule`` dataset profile.

    Returns a dict containing ``expected_games_by_season`` -- an
    independently-sourced, published-schedule-derived expected-game-count
    reference keyed by season string -- exactly the shape consumed by
    ``atlas.audit.coverage_matrix._expected_games_for_season``.
    ``expected_games_by_season`` counts regular-season games only (the
    conventional 30-team, ~162-game full-season universe); postseason
    counts are reported separately under ``postseason_games_by_season``
    since postseason game counts are not fixed per season.

    Cancelled games with no makeup game (``counted_in_expected_games`` is
    ``False``) are excluded from both counts so a game that never actually
    happened is never counted, and a postponed-then-replayed game (same
    gamePk, counted once) is never counted twice.
    """
    regular = regular_season_rows(rows)
    postseason = postseason_rows(rows)

    all_hashes = sorted(r["content_hash"] for r in rows if r.get("content_hash"))
    profile_content_hash = hashlib.sha256(
        "|".join(all_hashes).encode("utf-8")
    ).hexdigest()

    return {
        "source": SCHEDULE_SOURCE_NAME,
        "source_endpoint": SCHEDULE_ENDPOINT,
        "expected_games_by_season": _counts_by_season(regular),
        "postseason_games_by_season": _counts_by_season(postseason),
        "regular_season_game_count": len(regular),
        "postseason_game_count": len(postseason),
        "total_game_count": len(rows),
        "content_hash": profile_content_hash,
        "canonical_fields": list(CANONICAL_FIELDS),
    }


def schedule_game_ids(rows: Iterable[Mapping[str, Any]]) -> set[int]:
    """Return the set of expected ``gamePk`` ids from canonical schedule
    rows that are actually counted as scheduled games (excludes outright
    cancellations with no makeup)."""
    return {
        int(row["game_pk"])
        for row in rows
        if row.get("counted_in_expected_games") and row.get("game_pk") is not None
    }


def compare_against_master_datasets(
    rows: Iterable[Mapping[str, Any]],
    master_game_ids: Iterable[Any],
    master_pitch_game_ids: Iterable[Any],
) -> dict[str, Any]:
    """Compare the expected schedule game-ID set against caller-supplied
    ``master_game_database`` and ``master_pitch_database`` game-ID sets.

    This is a pure, read-only comparison: it never reads, writes, or
    otherwise touches Cloud Storage or any ATLAS dataset itself, and it
    never mutates the caller-supplied id sets. The schedule's own game-ID
    universe (from the published MLB Stats API) is always the reference
    side of the comparison -- master datasets are never used to infer or
    correct schedule facts.
    """
    expected_ids = schedule_game_ids(rows)
    master_game_ids_set = {int(x) for x in master_game_ids}
    master_pitch_game_ids_set = {int(x) for x in master_pitch_game_ids}

    return {
        "expected_game_ids": sorted(expected_ids),
        "expected_game_count": len(expected_ids),
        "missing_from_master_game_database": sorted(expected_ids - master_game_ids_set),
        "unexpected_in_master_game_database": sorted(master_game_ids_set - expected_ids),
        "missing_from_master_pitch_database": sorted(expected_ids - master_pitch_game_ids_set),
        "unexpected_in_master_pitch_database": sorted(master_pitch_game_ids_set - expected_ids),
    }
