from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from atlas.schedule.schedule_builder import ScheduleBuildError
from atlas.schedule.schedule_builder_v2 import ARTIFACT_NAMES, build_historical_schedule_v2
from atlas.schedule.schedule_history import (
    build_schedule_change_audit,
    normalize_schedule_history,
    schedule_history_metrics,
)


def _game(
    game_pk: int,
    *,
    state: str = "Final",
    official_date: str = "2024-04-02",
    **changes,
) -> dict:
    status_code = "PPD" if state == "Postponed" else "F"
    game = {
        "gamePk": game_pk,
        "gameGuid": f"guid-{game_pk}",
        "season": "2024",
        "gameType": "R",
        "gameDate": f"{official_date}T17:10:00Z",
        "officialDate": official_date,
        "status": {
            "statusCode": status_code,
            "codedGameState": status_code,
            "abstractGameState": "Final" if state == "Final" else "Preview",
            "detailedState": state,
        },
        "teams": {
            "home": {"team": {"id": 1, "name": "Home"}},
            "away": {"team": {"id": 2, "name": "Away"}},
        },
        "venue": {"id": 3, "name": "Venue"},
    }
    game.update(changes)
    return game


def _payload() -> dict:
    return {
        "dates": [
            {
                "games": [
                    _game(
                        101,
                        state="Postponed",
                        official_date="2024-04-01",
                        rescheduleGameDate="2024-04-02",
                    ),
                    _game(
                        101,
                        official_date="2024-04-02",
                        rescheduledFromDate="2024-04-01",
                    ),
                    _game(102, official_date="2024-04-03"),
                ]
            }
        ]
    }


def test_history_preserves_duplicate_schedule_snapshots_and_builds_audit():
    history = normalize_schedule_history(
        _payload(), retrieved_at_utc="2024-12-01T00:00:00+00:00"
    )
    assert len(history) == 3
    assert [row["game_pk"] for row in history].count(101) == 2
    assert len({row["schedule_history_content_hash"] for row in history}) == 3

    audit = build_schedule_change_audit(history)
    assert len(audit) == 1
    assert audit[0]["game_pk"] == 101
    assert audit[0]["original_scheduled_dates"] == "2024-04-01"
    assert audit[0]["rescheduled_dates"] == "2024-04-02"
    assert audit[0]["was_postponed"] is True
    assert audit[0]["was_rescheduled"] is True

    metrics = schedule_history_metrics(history, audit)
    assert metrics["postponed_rows"] == 1
    assert metrics["unique_postponed_game_pks"] == 1
    assert metrics["schedule_affected_game_pks"] == 1


def test_resume_fields_are_classified_as_suspended_or_resumed():
    payload = {
        "dates": [
            {
                "games": [
                    _game(
                        201,
                        official_date="2024-06-18",
                        resumedFromDate="2024-06-17",
                        resumeGameDate="2024-06-18",
                    )
                ]
            }
        ]
    }
    history = normalize_schedule_history(
        payload, retrieved_at_utc="2024-12-01T00:00:00+00:00"
    )
    audit = build_schedule_change_audit(history)
    assert audit[0]["was_suspended_or_resumed"] is True
    assert schedule_history_metrics(history, audit)[
        "suspended_or_resumed_game_pks"
    ] == 1


def test_v2_builder_requests_regular_season_and_emits_all_artifacts(
    tmp_path, monkeypatch
):
    calls = []

    def fetcher(*args, **kwargs):
        calls.append((args, kwargs))
        return _payload()

    monkeypatch.setattr(
        "atlas.schedule.schedule_builder_v2._write_parquet",
        lambda rows, path: path.write_text(
            json.dumps(list(rows), sort_keys=True, default=str), encoding="utf-8"
        ),
    )
    monkeypatch.setattr(
        "atlas.schedule.schedule_builder_v2._write_history_parquet",
        lambda rows, path, columns: path.write_text(
            json.dumps(list(rows), sort_keys=True, default=str), encoding="utf-8"
        ),
    )

    summary = build_historical_schedule_v2(
        [2024],
        tmp_path,
        fetcher=fetcher,
        timestamp=datetime(2024, 12, 1, tzinfo=timezone.utc),
    )

    assert calls[0][1]["game_types"] == ["R"]
    assert summary.games_processed == 2
    assert set(path.name for path in tmp_path.iterdir()) == set(ARTIFACT_NAMES)
    manifest = json.loads((tmp_path / "schedule_manifest.json").read_text())
    validation = json.loads((tmp_path / "schedule_validation.json").read_text())
    assert manifest["schedule_history_implemented"] is True
    assert manifest["schedule_history_rows"] == 3
    assert validation["unique_game_pks"] == 2
    assert validation["unique_postponed_game_pks"] == 1


def test_v2_builder_rejects_non_regular_games(tmp_path, monkeypatch):
    payload = _payload()
    payload["dates"][0]["games"][0]["gameType"] = "S"

    monkeypatch.setattr(
        "atlas.schedule.schedule_builder_v2._write_parquet",
        lambda rows, path: path.write_bytes(b"unused"),
    )
    monkeypatch.setattr(
        "atlas.schedule.schedule_builder_v2._write_history_parquet",
        lambda rows, path, columns: path.write_bytes(b"unused"),
    )

    with pytest.raises(ScheduleBuildError, match="non-regular"):
        build_historical_schedule_v2(
            [2024],
            tmp_path,
            fetcher=lambda *_args, **_kwargs: payload,
            timestamp=datetime(2024, 12, 1, tzinfo=timezone.utc),
        )
    assert not (tmp_path / "canonical_schedule.parquet").exists()
