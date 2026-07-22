from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.player_observations import (
    build_postgame_player_observations,
    certify_postgame_player_observations,
)


def pitch_events(**overrides) -> pd.DataFrame:
    rows = [
        {
            "game_pk": 10,
            "atlas_season": 2024,
            "game_type": "R",
            "inning_topbot": "Top",
            "batter": 101,
            "pitcher": 201,
        },
        {
            "game_pk": 10,
            "atlas_season": 2024,
            "game_type": "R",
            "inning_topbot": "Top",
            "batter": 101,
            "pitcher": 201,
        },
        {
            "game_pk": 10,
            "atlas_season": 2024,
            "game_type": "R",
            "inning_topbot": "Bot",
            "batter": 201,
            "pitcher": 101,
        },
    ]
    for row in rows:
        row.update(overrides)
    return pd.DataFrame(rows)


def schedule(**overrides) -> pd.DataFrame:
    row = {
        "game_pk": 10,
        "season": 2024,
        "game_date_utc": "2024-04-01T20:00:00Z",
        "game_type_code": "R",
        "is_final": True,
        "home_team_id": 1,
        "away_team_id": 2,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2024, "team_id": 1, "abbreviation": "HME"},
            {"season": 2024, "team_id": 2, "abbreviation": "AWY"},
        ]
    )


def test_assigns_numeric_team_identity_from_schedule_side():
    result = build_postgame_player_observations(
        pitch_events(), schedule(), teams(), season=2024
    )

    batter = result.loc[
        result["player_id"].eq(101) & result["team_id"].eq(2)
    ].iloc[0]
    assert batter["team_abbreviation"] == "AWY"
    assert batter["roles_observed"] == "batter,pitcher"
    assert int(batter["source_pitch_rows"]) == 3

    home_pitcher = result.loc[
        result["player_id"].eq(201) & result["team_id"].eq(1)
    ].iloc[0]
    assert home_pitcher["roles_observed"] == "batter,pitcher"
    assert int(home_pitcher["source_pitch_rows"]) == 3


def test_postgame_evidence_is_delayed_and_never_same_game_safe():
    result = build_postgame_player_observations(
        pitch_events(), schedule(), teams(), season=2024
    )
    assert set(result["observed_at"].astype(str)) == {
        "2024-04-01 20:00:00+00:00"
    }
    assert set(result["knowledge_available_at"].astype(str)) == {
        "2024-04-02 20:00:00+00:00"
    }
    assert not result["eligible_for_same_game_pregame"].any()
    assert result["prospective_only"].all()
    assert not result["retroactive_backfill_allowed"].any()

    report = certify_postgame_player_observations(result, season=2024)
    assert report["verdict"] == "certified"
    assert report["same_game_pregame_rows"] == 0
    assert report["retroactive_backfill_rows"] == 0


def test_rejects_non_regular_or_mixed_season_events():
    with pytest.raises(ValueError, match="regular-season-only"):
        build_postgame_player_observations(
            pitch_events(game_type="S"), schedule(), teams(), season=2024
        )
    with pytest.raises(ValueError, match="outside season 2024"):
        build_postgame_player_observations(
            pitch_events(atlas_season=2025), schedule(), teams(), season=2024
        )


def test_rejects_missing_schedule_game_and_nonfinal_game():
    with pytest.raises(ValueError, match="missing from the certified schedule"):
        build_postgame_player_observations(
            pitch_events(), schedule(game_pk=11), teams(), season=2024
        )
    with pytest.raises(ValueError, match="not final"):
        build_postgame_player_observations(
            pitch_events(), schedule(is_final=False), teams(), season=2024
        )


def test_rejects_invalid_inning_side_and_unknown_scheduled_team():
    with pytest.raises(ValueError, match="invalid inning_topbot"):
        build_postgame_player_observations(
            pitch_events(inning_topbot="Middle"), schedule(), teams(), season=2024
        )
    with pytest.raises(ValueError, match="official team directory"):
        build_postgame_player_observations(
            pitch_events(), schedule(home_team_id=3), teams(), season=2024
        )


def test_certifier_detects_unsafe_timestamp_or_retroactive_flag():
    result = build_postgame_player_observations(
        pitch_events(), schedule(), teams(), season=2024
    )
    unsafe = result.copy()
    unsafe["knowledge_available_at"] = unsafe["source_game_start_at"]
    unsafe["retroactive_backfill_allowed"] = True
    report = certify_postgame_player_observations(unsafe, season=2024)
    assert report["verdict"] == "quarantine_required"
    assert "less than 24 hours" in "; ".join(report["errors"])
    assert "retroactive backfill" in "; ".join(report["errors"])
