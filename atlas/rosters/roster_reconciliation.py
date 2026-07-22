"""Prospective reconciliation for quarantined roster-source facts.

Reconciliation is evidence, not a roster-state mutation. Later lineups, rosters, or
appearances may explain where a player next surfaced, but never become knowledge
for an earlier game.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


OBSERVATION_COLUMNS = {
    "player_id",
    "team_id",
    "observed_at",
    "knowledge_available_at",
    "evidence_type",
    "source",
}

OUTPUT_COLUMNS = [
    "transaction_key",
    "transaction_id",
    "player_id",
    "source_team_id",
    "type_code",
    "quarantine_reason",
    "transaction_available_at",
    "reconciliation_status",
    "first_observed_team_id",
    "first_observed_at",
    "observation_knowledge_available_at",
    "observation_evidence_type",
    "observation_source",
    "observation_game_pk",
    "same_source_team",
    "usable_prospectively_from",
    "retroactive_backfill_allowed",
    "source_record_sha256",
]


def _has_timezone(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return pd.Timestamp(value).tzinfo is not None
    except (TypeError, ValueError):
        return False


def _next_midnight_utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("transaction date is missing or invalid")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.normalize() + pd.Timedelta(days=1)


def certify_player_observations(observations: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(OBSERVATION_COLUMNS.difference(observations.columns))
    errors: list[str] = []
    if missing:
        return {"verdict": "not_ready", "rows": int(len(observations)),
                "errors": [f"missing observation columns: {missing}"]}
    if observations.empty:
        errors.append("player observation evidence is empty")
    for column in ("observed_at", "knowledge_available_at"):
        if observations[column].isna().any():
            errors.append(f"{column} contains null values")
        if not observations[column].map(_has_timezone).all():
            errors.append(f"{column} contains timezone-naive or invalid values")
    for column in ("player_id", "team_id", "evidence_type", "source"):
        if observations[column].isna().any():
            errors.append(f"{column} contains null values")
    if not errors:
        observed = pd.to_datetime(observations["observed_at"], utc=True)
        known = pd.to_datetime(observations["knowledge_available_at"], utc=True)
        if (known < observed).any():
            errors.append("knowledge_available_at cannot precede observed_at")
    return {
        "verdict": "certified" if not errors else "quarantine_required",
        "rows": int(len(observations)),
        "players": int(observations["player_id"].nunique(dropna=True)),
        "teams": int(observations["team_id"].nunique(dropna=True)),
        "errors": errors,
    }


def reconcile_quarantined_transactions(
    quarantine: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the earliest later player observation without rewriting history."""
    required = {
        "transaction_id", "player_id", "requested_team_id", "transaction_date",
        "effective_date", "type_code", "quarantine_reason", "source_record_sha256",
    }
    missing = sorted(required.difference(quarantine.columns))
    if missing:
        raise ValueError(f"quarantine missing columns: {missing}")
    report = certify_player_observations(observations)
    if report["verdict"] != "certified":
        raise ValueError("player observations are not certified: " + "; ".join(report["errors"]))

    evidence = observations.copy()
    evidence["observed_at"] = pd.to_datetime(evidence["observed_at"], utc=True)
    evidence["knowledge_available_at"] = pd.to_datetime(
        evidence["knowledge_available_at"], utc=True
    )
    evidence = evidence.sort_values(
        ["knowledge_available_at", "observed_at", "team_id", "player_id"],
        kind="stable",
    )

    records = []
    for index, row in quarantine.reset_index(drop=True).iterrows():
        transaction_key = row.get("transaction_key")
        if pd.isna(transaction_key):
            transaction_key = f"quarantine-row-{index}"
        player_id = row.get("player_id")
        source_team_id = row.get("requested_team_id")
        source_date = row.get("transaction_date")
        if pd.isna(source_date):
            source_date = row.get("effective_date")

        base = {
            "transaction_key": transaction_key,
            "transaction_id": row.get("transaction_id"),
            "player_id": player_id,
            "source_team_id": source_team_id,
            "type_code": row.get("type_code"),
            "quarantine_reason": row.get("quarantine_reason"),
            "transaction_available_at": (
                pd.NaT if pd.isna(source_date) else _next_midnight_utc(source_date)
            ),
            "first_observed_team_id": None,
            "first_observed_at": pd.NaT,
            "observation_knowledge_available_at": pd.NaT,
            "observation_evidence_type": None,
            "observation_source": None,
            "observation_game_pk": None,
            "same_source_team": None,
            "usable_prospectively_from": pd.NaT,
            "retroactive_backfill_allowed": False,
            "source_record_sha256": row.get("source_record_sha256"),
        }
        if pd.isna(player_id):
            records.append({**base, "reconciliation_status": "identity_missing"})
            continue
        if pd.isna(source_date):
            records.append({**base, "reconciliation_status": "transaction_date_missing"})
            continue

        effective_date = row.get("effective_date")
        if pd.isna(effective_date):
            effective_date = source_date
        effective_at = _next_midnight_utc(effective_date)
        available_at = base["transaction_available_at"]
        candidates = evidence.loc[
            (evidence["player_id"] == player_id)
            & (evidence["observed_at"] >= effective_at)
            & (evidence["knowledge_available_at"] >= available_at)
        ]
        if candidates.empty:
            records.append({**base, "reconciliation_status": "no_later_observation"})
            continue

        first = candidates.iloc[0]
        same_team = (
            False if pd.isna(source_team_id)
            else int(first["team_id"]) == int(source_team_id)
        )
        records.append({
            **base,
            "reconciliation_status": (
                "same_scoped_team_observed" if same_team
                else "different_team_observed"
            ),
            "first_observed_team_id": int(first["team_id"]),
            "first_observed_at": first["observed_at"],
            "observation_knowledge_available_at": first["knowledge_available_at"],
            "observation_evidence_type": first["evidence_type"],
            "observation_source": first["source"],
            "observation_game_pk": first.get("game_pk"),
            "same_source_team": same_team,
            "usable_prospectively_from": first["knowledge_available_at"],
        })

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)
