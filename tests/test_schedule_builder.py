from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from atlas.schedule.mlb_schedule_reference import normalize_game_row
from atlas.schedule.schedule_builder import (
    REQUIRED_COLUMNS,
    ScheduleBuildError,
    build_historical_schedule,
    validate_schedule,
)


def _game(game_pk: int, state: str = "Final") -> dict:
    return {
        "gamePk": game_pk,
        "gameGuid": f"guid-{game_pk}",
        "season": "2024",
        "gameType": "R",
        "gameDate": "2024-04-01T17:10:00Z",
        "officialDate": "2024-04-01",
        "status": {
            "statusCode": "F",
            "codedGameState": "F",
            "abstractGameState": "Final",
            "detailedState": state,
        },
        "teams": {
            "home": {"team": {"id": 1, "name": "Home"}},
            "away": {"team": {"id": 2, "name": "Away"}},
        },
        "venue": {"id": 3, "name": "Venue"},
    }


def _fetcher(*_args, **_kwargs):
    return {"dates": [{"games": [_game(1), _game(2)]}]}


def test_validation_requires_schema_and_rejects_unknown_status():
    row = normalize_game_row(
        _game(1), retrieved_at_utc="fixed", source_url="source"
    )
    assert validate_schedule([row])["status"] == "passed"
    row["detailed_state"] = "Not a published state"
    result = validate_schedule([row])
    assert result["status"] == "failed"
    assert "invalid schedule statuses detected" in result["errors"]
    assert set(REQUIRED_COLUMNS) == set(row)


def test_duplicate_game_pk_detection():
    row = normalize_game_row(_game(1), retrieved_at_utc="fixed", source_url="source")
    result = validate_schedule([row, dict(row)], duplicate_count=0)
    assert result["status"] == "failed"
    assert result["duplicate_count"] == 1


def test_build_is_deterministic_and_generates_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atlas.schedule.schedule_builder._write_parquet",
        lambda rows, path: path.write_bytes(
            json.dumps(list(rows), sort_keys=True, default=str).encode()
        ),
    )
    fixed_time = datetime(2024, 12, 1, tzinfo=timezone.utc)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_historical_schedule([2024], first, fetcher=_fetcher, timestamp=fixed_time)
    build_historical_schedule([2024], second, fetcher=_fetcher, timestamp=fixed_time)
    assert (first / "canonical_schedule.parquet").read_bytes() == (
        second / "canonical_schedule.parquet"
    ).read_bytes()
    assert json.loads((first / "schedule_manifest.json").read_text())["seasons"] == [2024]
    assert (first / "schedule_validation.json").exists()


def test_duplicate_raw_game_pk_aborts_before_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atlas.schedule.schedule_builder._write_parquet",
        lambda rows, path: path.write_bytes(b"unused"),
    )

    def duplicate_fetcher(*_args, **_kwargs):
        return {"dates": [{"games": [_game(1), _game(1)]}]}

    with pytest.raises(ScheduleBuildError, match="duplicate"):
        build_historical_schedule([2024], tmp_path, fetcher=duplicate_fetcher)
    assert not (tmp_path / "canonical_schedule.parquet").exists()
