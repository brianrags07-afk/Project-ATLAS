"""
Prediction-logic governance for Project ATLAS.

This module prevents legacy hand-authored decision logic from entering the
Phase 2E Brain. It does not delete legacy engines. It defines which modules
may provide factual targets, raw pregame facts, statistical reliability
metrics, or production prediction behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

POLICY_VERSION: Final[str] = "1.0.0"


@dataclass(frozen=True)
class ModulePolicy:
    status: str
    allowed_use: str
    phase_2e_allowed: bool
    prediction_runtime_allowed: bool
    reason: str


MODULE_POLICIES: Final[dict[str, ModulePolicy]] = {
    "atlas/learning/backtest_target_builder.py": ModulePolicy(
        status='APPROVED_FACTUAL_TARGETS',
        allowed_use='Historical target creation only.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=False,
        reason='Defines factual historical outcomes and target buckets. It must never be used as a same-game pregame feature.',
    ),
    "atlas/interactions/walk_forward_snapshot_engine.py": ModulePolicy(
        status='APPROVED_RAW_PREGAME_FACTS',
        allowed_use='Chronologically safe raw batter and pitcher history.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=True,
        reason='Produces prior-game factual snapshots. Any derived manual scores remain prohibited unless separately approved.',
    ),
    "atlas/interactions/pregame_snapshot_builder.py": ModulePolicy(
        status='APPROVED_RAW_PREGAME_FACTS',
        allowed_use='Chronologically safe player pregame snapshots.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=True,
        reason='Provides historical pregame facts with future-game exclusion.',
    ),
    "atlas/interactions/lineup_starter_input_engine.py": ModulePolicy(
        status='APPROVED_RAW_PREGAME_FACTS',
        allowed_use='Pregame lineup and starter interaction inputs.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=True,
        reason='May supply raw matchup facts. Handcrafted matchup scores remain prohibited until independently adjudicated.',
    ),
    "atlas/identities/bullpen_availability_fatigue_engine.py": ModulePolicy(
        status='RAW_FACTS_ONLY',
        allowed_use='Reliever usage, pitches thrown, days of rest, back-to-back usage, recent workload and availability facts only.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=False,
        reason='Raw bullpen workload facts are valuable. Handcrafted pressure, recovery, fatigue and availability scores are not approved.',
    ),
    "atlas/identities/bullpen_identity_integration_engine.py": ModulePolicy(
        status='RAW_FACTS_ONLY',
        allowed_use='Join approved raw bullpen facts to pregame records.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=False,
        reason='May integrate factual fields, but derived legacy bullpen scores cannot enter predictions until replaced with learned relationships.',
    ),
    "atlas/learning/team_evidence_discovery.py": ModulePolicy(
        status='RELIABILITY_ONLY',
        allowed_use='Discover candidate relationships and report sample size, effect size, significance and stability.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=False,
        reason='Evidence-quality formulas may rank research reliability but cannot determine baseball probability or feature influence.',
    ),
    "atlas/learning/league_evidence_discovery.py": ModulePolicy(
        status='RELIABILITY_ONLY',
        allowed_use='League-level evidence discovery and statistical validation.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=False,
        reason='Manual confidence formulas are restricted to research triage and cannot become predictive weights.',
    ),
    "atlas/learning/bullpen_evidence_discovery_2024.py": ModulePolicy(
        status='RELIABILITY_ONLY',
        allowed_use='Historical discovery of bullpen relationships.',
        phase_2e_allowed=True,
        prediction_runtime_allowed=False,
        reason='May discover relationships but cannot impose manually selected bullpen influence on predictions.',
    ),
    "atlas/predictions/pregame_prediction_engine.py": ModulePolicy(
        status='QUARANTINED_LEGACY_DECISION_ENGINE',
        allowed_use='Historical reference and comparison only.',
        phase_2e_allowed=False,
        prediction_runtime_allowed=False,
        reason='Contains unadjudicated scoring, confidence or probability logic. It cannot power the new Phase 2E Brain.',
    ),
    "atlas/predictions/prediction_fusion_engine.py": ModulePolicy(
        status='QUARANTINED_LEGACY_DECISION_ENGINE',
        allowed_use='Historical reference and comparison only.',
        phase_2e_allowed=False,
        prediction_runtime_allowed=False,
        reason='Contains unadjudicated fusion rules and potential manual weights.',
    ),
    "atlas/backtest/weighted_state_backtest_engine.py": ModulePolicy(
        status='QUARANTINED_LEGACY_DECISION_ENGINE',
        allowed_use='Historical comparison only.',
        phase_2e_allowed=False,
        prediction_runtime_allowed=False,
        reason='Weighted-state behavior must not be treated as canonical unless all weights are proven to be learned strictly from training data.',
    ),
    "atlas/history/failure_analysis_engine.py": ModulePolicy(
        status='ANALYSIS_ONLY',
        allowed_use='Postgame failure analysis only.',
        phase_2e_allowed=False,
        prediction_runtime_allowed=False,
        reason='Postgame diagnostic information cannot enter same-game predictions.',
    ),
    "atlas/history/model_repair_planning_engine.py": ModulePolicy(
        status='HUMAN_REVIEW_ONLY',
        allowed_use='Generate proposed repair plans for human review.',
        phase_2e_allowed=False,
        prediction_runtime_allowed=False,
        reason='Repair recommendations and grading tiers must not automatically change model logic or prediction weights.',
    ),

}


QUARANTINED_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "QUARANTINED_LEGACY_DECISION_ENGINE",
        "ANALYSIS_ONLY",
        "HUMAN_REVIEW_ONLY",
    }
)


def get_module_policy(relative_path: str) -> ModulePolicy:
    """Return the governance policy for one repository-relative module."""
    try:
        return MODULE_POLICIES[relative_path]
    except KeyError as exc:
        raise KeyError(
            f"No prediction-logic policy exists for: {relative_path}"
        ) from exc


def assert_phase_2e_allowed(relative_path: str) -> None:
    """Raise when a module is not approved as a Phase 2E input source."""
    policy = get_module_policy(relative_path)

    if not policy.phase_2e_allowed:
        raise PermissionError(
            f"Module is not approved for Phase 2E: {relative_path}. "
            f"Status={policy.status}. Reason={policy.reason}"
        )


def assert_prediction_runtime_allowed(relative_path: str) -> None:
    """Raise when a module is not approved to influence live predictions."""
    policy = get_module_policy(relative_path)

    if not policy.prediction_runtime_allowed:
        raise PermissionError(
            f"Module is not approved for prediction runtime: {relative_path}. "
            f"Status={policy.status}. Reason={policy.reason}"
        )


def approved_phase_2e_modules() -> tuple[str, ...]:
    """Return modules approved to provide Phase 2E facts or reliability data."""
    return tuple(
        sorted(
            path
            for path, policy in MODULE_POLICIES.items()
            if policy.phase_2e_allowed
        )
    )


def approved_prediction_runtime_modules() -> tuple[str, ...]:
    """Return modules currently approved to influence production predictions."""
    return tuple(
        sorted(
            path
            for path, policy in MODULE_POLICIES.items()
            if policy.prediction_runtime_allowed
        )
    )


def quarantined_modules() -> tuple[str, ...]:
    """Return modules explicitly blocked from the new decision Brain."""
    return tuple(
        sorted(
            path
            for path, policy in MODULE_POLICIES.items()
            if policy.status in QUARANTINED_STATUSES
        )
    )


def policy_artifact_paths(
    project_root: Path,
) -> dict[str, Path]:
    """Return canonical governance artifact paths."""
    base = (
        project_root
        / "data"
        / "governance"
        / "prediction_logic"
    )

    return {
        "module_policy":
            base
            / "prediction_module_policy.csv",

        "adjudication_registry":
            base
            / "prediction_logic_adjudication_registry.csv",

        "metadata":
            base
            / "prediction_logic_policy_metadata.json",
    }
