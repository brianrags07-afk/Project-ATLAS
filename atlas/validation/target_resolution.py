"""
Shared frozen-target resolution for ATLAS blind concept validation.

The canonical team-game target artifact (``team_game_targets.parquet``)
only ever physically contains the raw factual columns produced by
``atlas.learning.backtest_target_builder``:

- ``won``
- ``run_differential``

It never contains the frozen ``target_name`` columns referenced by frozen
concept definitions (for example ``target_team_win`` or
``target_team_win_by_2_plus``). Those frozen target columns must be
*materialized* from the canonical factual columns before any validation
engine checks target availability.

This module is the single, centralized place that performs that
materialization. It is intentionally narrow:

- It never reads or writes any file. Callers own all I/O.
- It never mutates frozen concept definitions or renames frozen
  ``target_name`` values.
- It never overwrites the canonical ``team_game_targets.parquet``
  artifact; it only returns an in-memory copy with additional columns.
- It never assigns prediction weights.
- It fails fast (``TargetResolutionIntegrityError``) whenever the source
  columns are missing or fail an integrity check, rather than silently
  producing an unavailable or incorrect target.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


WON_SOURCE_COLUMN = "won"
RUN_DIFFERENTIAL_SOURCE_COLUMN = "run_differential"

REQUIRED_TARGET_SOURCE_COLUMNS = (
    WON_SOURCE_COLUMN,
    RUN_DIFFERENTIAL_SOURCE_COLUMN,
)

TARGET_TEAM_WIN = "target_team_win"
TARGET_TEAM_WIN_BY_2_PLUS = "target_team_win_by_2_plus"

# Frozen, declarative target-resolution lineage. This is the single
# source of truth for how each frozen target name is derived from the
# canonical factual target columns. It must never be edited to match
# code drift; code drift must be caught and rejected instead (see
# ``certify_target_resolution_matches_rules`` below).
FROZEN_TARGET_RESOLUTION_RULES: dict[str, dict[str, Any]] = {
    TARGET_TEAM_WIN: {
        "source_columns": [WON_SOURCE_COLUMN],
        "rule": "won",
    },
    TARGET_TEAM_WIN_BY_2_PLUS: {
        "source_columns": [RUN_DIFFERENTIAL_SOURCE_COLUMN],
        "rule": "run_differential >= 2",
    },
}

# Computed once at import time: ``FROZEN_TARGET_RESOLUTION_RULES`` is
# frozen and never mutated at runtime, so recomputing this hash on every
# call to ``target_resolution_rules_fingerprint`` would be wasted work.
_FROZEN_TARGET_RESOLUTION_RULES_FINGERPRINT = hashlib.sha256(
    json.dumps(
        FROZEN_TARGET_RESOLUTION_RULES,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


class TargetResolutionIntegrityError(RuntimeError):
    """
    Raised when the canonical target source columns fail an integrity
    check prior to frozen target materialization.

    Callers must treat this as fatal: never proceed with target
    availability checks, validation masks, or validation status
    computation when this is raised.
    """


def target_resolution_rules_fingerprint() -> str:
    """
    Deterministic fingerprint of ``FROZEN_TARGET_RESOLUTION_RULES``.

    Used to certify that the resolution rules actually applied by
    ``resolve_frozen_targets`` at run time are the same rules recorded
    in lineage/manifest artifacts, guarding against silent code drift.
    """

    return _FROZEN_TARGET_RESOLUTION_RULES_FINGERPRINT


def _require_target_source_columns(dataframe: pd.DataFrame) -> None:
    missing = sorted(
        set(REQUIRED_TARGET_SOURCE_COLUMNS).difference(
            dataframe.columns
        )
    )

    if missing:
        raise TargetResolutionIntegrityError(
            "Canonical team-game target source is missing required "
            f"target-resolution columns: {missing}"
        )


def _coerce_won(won_raw: pd.Series) -> pd.Series:
    won_numeric = pd.to_numeric(
        won_raw,
        errors="coerce",
    )

    unparseable = won_numeric.isna() & won_raw.notna()

    if unparseable.any():
        raise TargetResolutionIntegrityError(
            "Non-binary 'won' values could not be coerced to numeric: "
            f"{sorted(won_raw[unparseable].unique().tolist())}"
        )

    non_null = won_numeric.dropna()

    invalid = sorted(
        set(non_null.unique().tolist()) - {0.0, 1.0}
    )

    if invalid:
        raise TargetResolutionIntegrityError(
            f"'won' contains non-binary values outside {{0, 1}}: {invalid}"
        )

    return won_numeric


def _coerce_run_differential(run_differential_raw: pd.Series) -> pd.Series:
    run_differential_numeric = pd.to_numeric(
        run_differential_raw,
        errors="coerce",
    )

    unparseable = (
        run_differential_numeric.isna()
        & run_differential_raw.notna()
    )

    if unparseable.any():
        raise TargetResolutionIntegrityError(
            "Non-numeric 'run_differential' values could not be coerced: "
            f"{sorted(run_differential_raw[unparseable].unique().tolist())}"
        )

    return run_differential_numeric


def _assert_won_agrees_with_run_differential(
    won_numeric: pd.Series,
    run_differential_numeric: pd.Series,
) -> None:
    both_available = (
        won_numeric.notna()
        & run_differential_numeric.notna()
    )

    if not both_available.any():
        return

    expected_won = (
        run_differential_numeric.gt(0)
        .astype(float)
    )

    disagreement = both_available & won_numeric.ne(expected_won)

    if disagreement.any():
        raise TargetResolutionIntegrityError(
            "'won' disagrees with 'run_differential > 0' for "
            f"{int(disagreement.sum())} row(s). Source integrity "
            "violated; refusing to resolve frozen targets."
        )


def _assert_resolved_target_is_binary_or_null(
    resolved: pd.Series,
    target_name: str,
) -> None:
    invalid = sorted(
        set(resolved.dropna().unique().tolist()) - {0.0, 1.0}
    )

    if invalid:
        raise TargetResolutionIntegrityError(
            f"Resolved target '{target_name}' contains values outside "
            f"{{0, 1, null}}: {invalid}"
        )


def resolve_frozen_targets(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Materialize the frozen ``target_team_win`` and
    ``target_team_win_by_2_plus`` columns from the canonical ``won`` and
    ``run_differential`` factual columns.

    Returns ``(resolved_dataframe, resolution_stats)``. ``resolved_dataframe``
    is a copy of ``dataframe`` with the frozen target columns
    materialized (any pre-existing columns of the same name are
    overwritten so that resolution is always driven from the canonical
    factual columns, never from a stale or foreign value). No input is
    mutated in place and no file is read or written.

    Raises ``TargetResolutionIntegrityError`` if the required source
    columns are missing, contain non-binary/non-numeric values, or if
    ``won`` disagrees with ``run_differential > 0`` for any row where
    both are available.
    """

    _require_target_source_columns(dataframe)

    won_numeric = _coerce_won(dataframe[WON_SOURCE_COLUMN])

    run_differential_numeric = _coerce_run_differential(
        dataframe[RUN_DIFFERENTIAL_SOURCE_COLUMN]
    )

    _assert_won_agrees_with_run_differential(
        won_numeric,
        run_differential_numeric,
    )

    target_team_win = won_numeric

    target_team_win_by_2_plus = pd.Series(
        np.where(
            run_differential_numeric.isna(),
            np.nan,
            run_differential_numeric.ge(2).astype(float),
        ),
        index=dataframe.index,
    )

    _assert_resolved_target_is_binary_or_null(
        target_team_win,
        TARGET_TEAM_WIN,
    )

    _assert_resolved_target_is_binary_or_null(
        target_team_win_by_2_plus,
        TARGET_TEAM_WIN_BY_2_PLUS,
    )

    resolved = dataframe.copy()
    resolved[TARGET_TEAM_WIN] = target_team_win
    resolved[TARGET_TEAM_WIN_BY_2_PLUS] = target_team_win_by_2_plus

    resolved_targets = {
        TARGET_TEAM_WIN: target_team_win,
        TARGET_TEAM_WIN_BY_2_PLUS: target_team_win_by_2_plus,
    }

    resolution_stats: dict[str, Any] = {
        "rules_fingerprint": target_resolution_rules_fingerprint(),
        # The `_2025` suffix indicates these counts reflect the data
        # passed to this function; callers are responsible for any
        # season filtering (in production, only the already
        # season-filtered 2025 validation frame is ever passed in).
        "resolved_targets": {
            target_name: {
                "source_columns": list(
                    FROZEN_TARGET_RESOLUTION_RULES[target_name][
                        "source_columns"
                    ]
                ),
                "rule": FROZEN_TARGET_RESOLUTION_RULES[target_name]["rule"],
                "non_null_resolved_rows": int(values.notna().sum()),
                "positive_2025": int(values.eq(1.0).sum()),
                "negative_2025": int(values.eq(0.0).sum()),
            }
            for target_name, values in resolved_targets.items()
        },
    }

    return resolved, resolution_stats


def certify_target_resolution_matches_rules(
    resolution_stats: dict[str, Any],
) -> bool:
    """
    Certify that the resolution rule metadata recorded for this run
    (typically embedded in lineage/manifest artifacts) matches the
    frozen ``FROZEN_TARGET_RESOLUTION_RULES`` source of truth exactly.

    Returns ``True`` only when every frozen target name is present with
    identical ``source_columns``/``rule`` metadata and the recorded
    fingerprint matches the current fingerprint.
    """

    if resolution_stats.get(
        "rules_fingerprint"
    ) != target_resolution_rules_fingerprint():
        return False

    resolved_targets = resolution_stats.get("resolved_targets", {})

    for target_name, rule in FROZEN_TARGET_RESOLUTION_RULES.items():
        recorded = resolved_targets.get(target_name)

        if recorded is None:
            return False

        if recorded.get("source_columns") != rule["source_columns"]:
            return False

        if recorded.get("rule") != rule["rule"]:
            return False

    return True
