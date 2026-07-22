from __future__ import annotations

import pandas as pd
import pytest

from atlas.rosters.roster_reconciliation import (
    certify_player_observations,
    reconcile_quarantined_transactions,
)


def quarantined(**overrides):
    row = {
        "transaction_key": "10:a:1",
        "transaction_id": 10,
        "player_id": 7,
        "requested_team_id": 1,
        "transaction_date": "2024-04-01",
        "effective_date": "2024-04-01",
        "type_code": "SC",
        "quarantine_reason": "type code not approved for roster state conversion",
        "source_record_sha256": "a",
    }
    row.update(overrides)
    return row


def observed(**overrides):
    row = {
        "player_id": 7,
        "team_id": 1,
        "game_pk": 99,
        "observed_at": "2024-04-03T20:00:00Z",
        "knowledge_available_at": "2024-04-03T20:00:00Z",
        "evidence_type": "official_pregame_lineup",
        "source": "MLB Stats API game feed",
    }
    row.update(overrides)
    return row


def test_same_team_observation_is_prospective_evidence_only():
    result = reconcile_quarantined_transactions(
        pd.DataFrame([quarantined()]),
        pd.DataFrame([observed()]),
    )
    row = result.iloc[0]
    assert row["reconciliation_status"] == "same_scoped_team_observed"
    assert bool(row["same_source_team"]) is True
    assert str(row["usable_prospectively_from"]) == "2024-04-03 20:00:00+00:00"
    assert bool(row["retroactive_backfill_allowed"]) is False


def test_observation_before_transaction_knowledge_is_not_used():
    observations = pd.DataFrame([
        observed(
            observed_at="2024-04-01T18:00:00Z",
            knowledge_available_at="2024-04-01T18:00:00Z",
            game_pk=98,
        ),
        observed(
            observed_at="2024-04-04T18:00:00Z",
            knowledge_available_at="2024-04-05T00:00:00Z",
            evidence_type="postgame_appearance",
            game_pk=100,
        ),
    ])
    result = reconcile_quarantined_transactions(
        pd.DataFrame([quarantined()]), observations
    )
    row = result.iloc[0]
    assert row["observation_game_pk"] == 100
    assert str(row["usable_prospectively_from"]) == "2024-04-05 00:00:00+00:00"


def test_different_team_is_reported_without_semantic_claim():
    result = reconcile_quarantined_transactions(
        pd.DataFrame([quarantined()]),
        pd.DataFrame([observed(team_id=2)]),
    )
    row = result.iloc[0]
    assert row["reconciliation_status"] == "different_team_observed"
    assert bool(row["same_source_team"]) is False
    assert bool(row["retroactive_backfill_allowed"]) is False


def test_unknown_player_identity_remains_preserved():
    result = reconcile_quarantined_transactions(
        pd.DataFrame([quarantined(player_id=None)]),
        pd.DataFrame([observed()]),
    )
    assert result.iloc[0]["reconciliation_status"] == "identity_missing"
    assert result.iloc[0]["source_record_sha256"] == "a"


def test_no_later_observation_is_explicit():
    result = reconcile_quarantined_transactions(
        pd.DataFrame([quarantined(player_id=8)]),
        pd.DataFrame([observed()]),
    )
    assert result.iloc[0]["reconciliation_status"] == "no_later_observation"


def test_naive_observation_timestamps_are_rejected():
    evidence = pd.DataFrame([observed(knowledge_available_at="2024-04-03 20:00:00")])
    report = certify_player_observations(evidence)
    assert report["verdict"] == "quarantine_required"
    with pytest.raises(ValueError, match="not certified"):
        reconcile_quarantined_transactions(
            pd.DataFrame([quarantined()]), evidence
        )


def test_knowledge_cannot_precede_observation():
    evidence = pd.DataFrame([
        observed(
            observed_at="2024-04-03T20:00:00Z",
            knowledge_available_at="2024-04-03T19:00:00Z",
        )
    ])
    report = certify_player_observations(evidence)
    assert "knowledge_available_at cannot precede observed_at" in report["errors"]
