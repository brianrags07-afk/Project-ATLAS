from __future__ import annotations

from atlas.schedule.schedule_fixture_certification import (
    SAVED_2024_EXPECTATIONS,
    SPECIAL_GAME_EXPECTATIONS,
    certify_schedule_snapshot,
    schedule_snapshot_metrics,
    special_game_observations,
)


def _game(game_pk: int, date: str, state: str = "Final", **changes) -> dict:
    game = {
        "gamePk": game_pk,
        "gameType": "R",
        "gameDate": f"{date}T17:10:00Z",
        "officialDate": date,
        "status": {"detailedState": state},
    }
    game.update(changes)
    return game


def _special_games_payload() -> dict:
    games = [
        _game(745180, "2024-05-21", resumeGameDate="2024-05-22"),
        _game(745180, "2024-05-22", resumedFromDate="2024-05-21"),
        _game(746755, "2024-08-27", resumeGameDate="2024-08-28"),
        _game(746755, "2024-08-28", resumedFromDate="2024-08-27"),
        _game(746942, "2024-06-26", resumeGameDate="2024-08-26"),
        _game(746942, "2024-08-26", resumedFromDate="2024-06-26"),
        _game(
            747090,
            "2024-07-23",
            "Postponed",
            rescheduleGameDate="2024-07-24",
        ),
        _game(
            747090,
            "2024-07-24",
            "Postponed",
            rescheduleGameDate="2024-09-09",
            rescheduledFromDate="2024-07-23",
        ),
        _game(747090, "2024-09-09", rescheduledFromDate="2024-07-24"),
        _game(
            747139,
            "2024-04-10",
            "Postponed",
            rescheduleGameDate="2024-09-26",
        ),
        _game(
            747139,
            "2024-09-26",
            "Postponed",
            rescheduleGameDate="2024-09-30",
            rescheduledFromDate="2024-04-10",
        ),
        _game(747139, "2024-09-30", rescheduledFromDate="2024-09-26"),
    ]
    return {"dates": [{"games": games}]}


def test_saved_2024_expectations_are_checksum_bound_and_regular_only():
    assert len(SAVED_2024_EXPECTATIONS) == 7
    assert SAVED_2024_EXPECTATIONS["raw_history_rows"] == 2469
    assert SAVED_2024_EXPECTATIONS["unique_game_pks"] == 2430
    assert SAVED_2024_EXPECTATIONS["non_regular_rows"] == 0


def test_five_special_games_preserve_resume_and_multi_reschedule_chains():
    observations = special_game_observations(_special_games_payload())
    for game_pk, expected in SPECIAL_GAME_EXPECTATIONS.items():
        for key, value in expected.items():
            assert observations[game_pk][key] == value


def test_certification_passes_matching_snapshot_contract():
    payload = _special_games_payload()
    metrics = schedule_snapshot_metrics(payload)
    report = certify_schedule_snapshot(
        payload,
        source_sha256="fixture-hash",
        expected_sha256="fixture-hash",
        expected_metrics=metrics,
    )
    assert report["verdict"] == "certified"
    assert report["errors"] == []


def test_checksum_or_reschedule_chain_mismatch_requires_quarantine():
    payload = _special_games_payload()
    metrics = schedule_snapshot_metrics(payload)
    payload["dates"][0]["games"][-1]["rescheduledFromDate"] = "2024-09-25"
    report = certify_schedule_snapshot(
        payload,
        source_sha256="wrong-hash",
        expected_sha256="fixture-hash",
        expected_metrics=metrics,
    )
    assert report["verdict"] == "quarantine_required"
    assert any("checksum mismatch" in error for error in report["errors"])
    assert any("game_pk 747139" in error for error in report["errors"])
