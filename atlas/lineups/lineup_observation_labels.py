"""Governed postgame observation labels for batting order and starting pitchers.

Pitch events can reconstruct the first nine unique batters and the first
pitcher faced.  They cannot prove that ATLAS possessed a published lineup
before first pitch.  These rows are therefore training/evaluation labels,
never same-game pregame features.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


LABEL_VERSION = "reconstructed_lineup_observation_v1"
LABEL_AVAILABILITY_DELAY = pd.Timedelta(hours=24)

PITCH_COLUMNS = {
    "game_pk",
    "atlas_season",
    "game_type",
    "inning",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
}

TEAM_GAME_COLUMNS = {
    "game_pk",
    "game_start_at",
    "official_date",
    "season",
    "team",
    "team_id",
    "opponent",
    "opponent_team_id",
    "home_away",
}

BATTING_ORDER_COLUMNS = [
    f"batting_order_{position}_player_id" for position in range(1, 10)
]

OUTPUT_COLUMNS = [
    "game_pk",
    "game_start_at",
    "official_date",
    "season",
    "team",
    "team_id",
    "opponent",
    "opponent_team_id",
    "home_away",
    "starting_pitcher_id",
    "opposing_starting_pitcher_id",
    "starting_lineup_size",
    "starting_lineup_complete",
    *BATTING_ORDER_COLUMNS,
    "lineup_order_signature",
    "lineup_player_set_signature",
    "source_pitch_rows",
    "source_plate_appearances",
    "observed_at",
    "label_available_at",
    "source",
    "reconstruction_method",
    "reconstruction_is_observed_lineup_proxy",
    "official_starting_lineup_confirmed",
    "observation_time_semantics",
    "availability_timestamp_type",
    "future_feature_eligibility_rule",
    "postgame_observation_label",
    "published_lineup_confirmed",
    "same_game_pregame_eligible",
    "eligible_for_training_target",
    "eligible_for_future_game_feature",
    "direct_feature_use_allowed",
    "future_games_used",
    "lineup_observation_version",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _integer(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or (values % 1).ne(0).any():
        raise ValueError(f"{label} contains missing or non-integer values")
    return values.astype("int64")


def _first_nine_unique_batters(team_events: pd.DataFrame) -> list[int]:
    plate_appearances = (
        team_events[["at_bat_number", "batter"]]
        .dropna(subset=["batter"])
        .drop_duplicates("at_bat_number", keep="first")
        .sort_values("at_bat_number", kind="stable")
    )
    lineup: list[int] = []
    seen: set[int] = set()
    for value in plate_appearances["batter"]:
        player_id = int(value)
        if player_id in seen:
            continue
        lineup.append(player_id)
        seen.add(player_id)
        if len(lineup) == 9:
            break
    return lineup


def _first_pitcher(team_events: pd.DataFrame) -> int | None:
    pitchers = team_events["pitcher"].dropna()
    return None if pitchers.empty else int(pitchers.iloc[0])


def build_reconstructed_lineup_observation_labels(
    pitches: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    """Reconstruct label-only batting order and starters from pitch events."""
    _require_columns(pitches, PITCH_COLUMNS, "pitch events")
    _require_columns(team_games, TEAM_GAME_COLUMNS, "team games")
    if pitches.empty or team_games.empty:
        raise ValueError("pitch events and team games must be nonempty")

    events = pitches.copy()
    events["game_pk"] = _integer(events["game_pk"], "pitch events.game_pk")
    events["atlas_season"] = _integer(
        events["atlas_season"], "pitch events.atlas_season"
    )
    if not events["atlas_season"].eq(int(season)).all():
        raise ValueError(f"pitch events contain rows outside season {season}")
    game_types = set(
        events["game_type"].dropna().astype(str).str.upper().str.strip().unique()
    )
    if game_types != {"R"}:
        raise ValueError(
            "lineup observation labels require regular-season-only events; "
            f"found {sorted(game_types)}"
        )
    events["inning"] = _integer(events["inning"], "pitch events.inning")
    events["at_bat_number"] = _integer(
        events["at_bat_number"], "pitch events.at_bat_number"
    )
    events["pitch_number"] = _integer(
        events["pitch_number"], "pitch events.pitch_number"
    )
    for column in ("batter", "pitcher"):
        events[column] = _integer(events[column], f"pitch events.{column}")

    half = events["inning_topbot"].astype("string").str.strip()
    invalid = sorted(set(half.dropna().unique()).difference({"Top", "Bot"}))
    if invalid or half.isna().any():
        raise ValueError(f"pitch events contain invalid inning sides: {invalid}")
    events["_top"] = half.eq("Top")
    events["_half_order"] = events["_top"].map({True: 0, False: 1}).astype("int8")

    games = team_games.copy()
    for column in ("game_pk", "season", "team_id", "opponent_team_id"):
        games[column] = _integer(games[column], f"team games.{column}")
    games["game_start_at"] = pd.to_datetime(
        games["game_start_at"], utc=True, errors="coerce"
    )
    if games["game_start_at"].isna().any():
        raise ValueError("team games contain an invalid game_start_at")
    if not games["season"].eq(int(season)).all():
        raise ValueError(f"team games contain rows outside season {season}")
    if games.duplicated(["game_pk", "team_id"], keep=False).any():
        raise ValueError("team games contain duplicate game/team rows")
    games["home_away"] = games["home_away"].astype("string").str.upper().str.strip()
    invalid_directions = sorted(
        set(games["home_away"].dropna().unique()).difference({"HOME", "AWAY"})
    )
    if invalid_directions or games["home_away"].isna().any():
        raise ValueError(
            f"team games contain invalid home_away values: {invalid_directions}"
        )
    direction_counts = games.groupby("game_pk")["home_away"].value_counts().unstack(
        fill_value=0
    )
    for direction in ("HOME", "AWAY"):
        if direction not in direction_counts:
            direction_counts[direction] = 0
    if not (
        direction_counts["HOME"].eq(1) & direction_counts["AWAY"].eq(1)
    ).all():
        raise ValueError("team games do not contain one HOME and one AWAY row per game")
    event_games = set(events["game_pk"].unique())
    scheduled_games = set(games["game_pk"].unique())
    if event_games != scheduled_games:
        raise ValueError(
            "pitch-event and completed-schedule game coverage differs: "
            f"missing_events={sorted(scheduled_games-event_games)[:20]}, "
            f"unexpected_events={sorted(event_games-scheduled_games)[:20]}"
        )

    sides = games[
        ["game_pk", "team_id", "opponent_team_id", "home_away"]
    ].copy()
    home = sides.loc[sides["home_away"].eq("HOME")].rename(
        columns={"team_id": "home_team_id", "opponent_team_id": "away_team_id"}
    )[["game_pk", "home_team_id", "away_team_id"]]
    away = sides.loc[sides["home_away"].eq("AWAY")]
    if len(home) != len(scheduled_games) or len(away) != len(scheduled_games):
        raise ValueError("team games do not contain one HOME and one AWAY row per game")
    events = events.merge(home, on="game_pk", how="left", validate="many_to_one")
    events["team_id"] = events["away_team_id"].where(
        events["_top"], events["home_team_id"]
    )
    events["team_id"] = _integer(events["team_id"], "derived batting team_id")
    events = events.sort_values(
        [
            "game_pk",
            "inning",
            "_half_order",
            "at_bat_number",
            "pitch_number",
        ],
        kind="stable",
    ).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for (game_pk, team_id), group in events.groupby(
        ["game_pk", "team_id"], sort=True
    ):
        lineup = _first_nine_unique_batters(group)
        record: dict[str, Any] = {
            "game_pk": int(game_pk),
            "team_id": int(team_id),
            "opposing_starting_pitcher_id": _first_pitcher(group),
            "starting_lineup_size": len(lineup),
            "starting_lineup_complete": len(lineup) == 9,
            "lineup_order_signature": "-".join(str(value) for value in lineup),
            "lineup_player_set_signature": "-".join(
                str(value) for value in sorted(lineup)
            ),
            "source_pitch_rows": int(len(group)),
            "source_plate_appearances": int(group["at_bat_number"].nunique()),
        }
        for position, column in enumerate(BATTING_ORDER_COLUMNS, start=1):
            record[column] = lineup[position - 1] if len(lineup) >= position else pd.NA
        records.append(record)

    labels = games.merge(
        pd.DataFrame(records),
        on=["game_pk", "team_id"],
        how="left",
        validate="one_to_one",
    )
    starter_mirror = labels[
        ["game_pk", "team_id", "opposing_starting_pitcher_id"]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "opposing_starting_pitcher_id": "starting_pitcher_id",
        }
    )
    labels = labels.merge(
        starter_mirror,
        on=["game_pk", "opponent_team_id"],
        how="left",
        validate="one_to_one",
    )
    labels["label_available_at"] = (
        labels["game_start_at"] + LABEL_AVAILABILITY_DELAY
    )
    # The game start is only the event-time lower bound.  Availability is kept
    # separately and governs when a later game may consume the label.
    labels["observed_at"] = labels["game_start_at"]
    labels["source"] = "ATLAS certified regular-season pitch events"
    labels["reconstruction_method"] = (
        "first_nine_unique_batters_and_first_pitcher_faced"
    )
    labels["reconstruction_is_observed_lineup_proxy"] = True
    labels["official_starting_lineup_confirmed"] = False
    labels["observation_time_semantics"] = "game_start_lower_bound"
    labels["availability_timestamp_type"] = "conservative_game_start_plus_24h"
    labels["future_feature_eligibility_rule"] = (
        "label_available_at_must_precede_target_game_start_at"
    )
    labels["postgame_observation_label"] = True
    labels["published_lineup_confirmed"] = False
    labels["same_game_pregame_eligible"] = False
    labels["eligible_for_training_target"] = True
    labels["eligible_for_future_game_feature"] = True
    labels["direct_feature_use_allowed"] = False
    labels["future_games_used"] = False
    labels["lineup_observation_version"] = LABEL_VERSION

    for column in [
        "starting_pitcher_id",
        "opposing_starting_pitcher_id",
        "starting_lineup_size",
        *BATTING_ORDER_COLUMNS,
    ]:
        labels[column] = pd.to_numeric(labels[column], errors="coerce").astype("Int64")
    labels = labels[OUTPUT_COLUMNS].sort_values(
        ["game_start_at", "game_pk", "home_away"], kind="stable"
    ).reset_index(drop=True)

    report = certify_reconstructed_lineup_observation_labels(
        labels, games, season=season
    )
    if report["verdict"] != "certified":
        raise ValueError(
            "reconstructed lineup observation labels failed certification: "
            + "; ".join(report["errors"])
        )
    return labels


def certify_reconstructed_lineup_observation_labels(
    labels: pd.DataFrame,
    team_games: pd.DataFrame,
    *,
    season: int,
) -> dict[str, Any]:
    missing = sorted(set(OUTPUT_COLUMNS).difference(labels.columns))
    if missing:
        return {
            "verdict": "not_ready",
            "rows": int(len(labels)),
            "errors": [f"lineup observation labels missing columns: {missing}"],
        }
    errors: list[str] = []
    if labels.empty:
        errors.append("lineup observation labels are empty")
    if labels.duplicated(["game_pk", "team_id"], keep=False).any():
        errors.append("duplicate game/team lineup observation labels detected")
    if not labels["season"].eq(int(season)).all():
        errors.append(f"lineup observation labels contain rows outside season {season}")

    expected = set(
        map(tuple, team_games[["game_pk", "team_id"]].to_numpy())
    )
    actual = set(map(tuple, labels[["game_pk", "team_id"]].to_numpy()))
    missing_team_games = sorted(expected.difference(actual))
    unexpected_team_games = sorted(actual.difference(expected))
    if missing_team_games:
        errors.append(f"missing team-game labels: {missing_team_games[:20]}")
    if unexpected_team_games:
        errors.append(f"unexpected team-game labels: {unexpected_team_games[:20]}")

    start = pd.to_datetime(labels["game_start_at"], utc=True, errors="coerce")
    observed = pd.to_datetime(labels["observed_at"], utc=True, errors="coerce")
    available = pd.to_datetime(labels["label_available_at"], utc=True, errors="coerce")
    if start.isna().any() or observed.isna().any() or available.isna().any():
        errors.append("lineup label chronology contains an invalid timestamp")
    if not available.ge(start + LABEL_AVAILABILITY_DELAY).all():
        errors.append("a reconstructed label became available less than 24 hours after start")
    if not observed.eq(start).all():
        errors.append("observed_at must equal the conservative game-start lower bound")
    for column, expected_value in (
        ("reconstruction_is_observed_lineup_proxy", True),
        ("official_starting_lineup_confirmed", False),
        ("postgame_observation_label", True),
        ("published_lineup_confirmed", False),
        ("same_game_pregame_eligible", False),
        ("eligible_for_training_target", True),
        ("eligible_for_future_game_feature", True),
        ("direct_feature_use_allowed", False),
        ("future_games_used", False),
    ):
        if not labels[column].eq(expected_value).all():
            errors.append(f"{column} must be {expected_value}")
    for column, expected_value in (
        ("observation_time_semantics", "game_start_lower_bound"),
        ("availability_timestamp_type", "conservative_game_start_plus_24h"),
        (
            "future_feature_eligibility_rule",
            "label_available_at_must_precede_target_game_start_at",
        ),
    ):
        if not labels[column].eq(expected_value).all():
            errors.append(f"{column} must be {expected_value}")

    lineup_size = pd.to_numeric(labels["starting_lineup_size"], errors="coerce")
    complete = labels["starting_lineup_complete"].fillna(False).astype(bool)
    if lineup_size.isna().any() or lineup_size.lt(0).any() or lineup_size.gt(9).any():
        errors.append("starting_lineup_size must be an integer from zero through nine")
    if not complete.eq(lineup_size.eq(9)).all():
        errors.append("starting_lineup_complete does not match starting_lineup_size")
    lineup_content_errors = 0
    for index, row in labels[BATTING_ORDER_COLUMNS].iterrows():
        values = [int(value) for value in row if pd.notna(value)]
        expected_order = "-".join(str(value) for value in values)
        expected_set = "-".join(str(value) for value in sorted(values))
        lineup_content_errors += int(
            len(values) != len(set(values))
            or len(values) != lineup_size.loc[index]
            or labels.at[index, "lineup_order_signature"] != expected_order
            or labels.at[index, "lineup_player_set_signature"] != expected_set
        )
    if lineup_content_errors:
        errors.append("one or more reconstructed lineup contents are inconsistent")

    mirror = labels[
        ["game_pk", "team_id", "starting_pitcher_id"]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "starting_pitcher_id": "expected_opposing_starting_pitcher_id",
        }
    )
    checked = labels.merge(
        mirror,
        on=["game_pk", "opponent_team_id"],
        how="left",
        validate="one_to_one",
    )
    observed_starter = checked["opposing_starting_pitcher_id"]
    expected_starter = checked["expected_opposing_starting_pitcher_id"]
    starter_mismatch = observed_starter.isna().ne(expected_starter.isna()) | (
        observed_starter.notna()
        & expected_starter.notna()
        & observed_starter.ne(expected_starter).fillna(False)
    )
    starter_mirror_errors = int(starter_mismatch.sum())
    if starter_mirror_errors:
        errors.append("starting pitcher mirror validation failed")

    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "rows": int(len(labels)),
        "games": int(labels["game_pk"].nunique(dropna=True)),
        "team_games": int(len(actual)),
        "teams": int(labels["team_id"].nunique(dropna=True)),
        "complete_lineups": int(complete.sum()),
        "incomplete_lineups": int((~complete).sum()),
        "missing_starting_pitchers": int(labels["starting_pitcher_id"].isna().sum()),
        "missing_opposing_starting_pitchers": int(
            labels["opposing_starting_pitcher_id"].isna().sum()
        ),
        "starter_mirror_errors": starter_mirror_errors,
        "published_lineup_rows": int(labels["published_lineup_confirmed"].sum()),
        "same_game_feature_rows": int(labels["same_game_pregame_eligible"].sum()),
        "lagged_future_feature_rows": int(
            labels["eligible_for_future_game_feature"].sum()
        ),
        "lineup_content_errors": lineup_content_errors,
        "future_games_used": False,
        "errors": errors,
    }
