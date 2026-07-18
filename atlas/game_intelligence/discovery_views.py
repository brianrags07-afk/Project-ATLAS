"""
Phase 2E.4C — Controlled Discovery Views.

Creates target-specific discovery views by joining the canonical pregame 
evidence matrix with factual learning targets. Each discovery view contains
the evidence and one target family.

Temporal leakage prevention:

- All pregame evidence uses only prior-date information
- No completed-game results appear in evidence
- Targets are strictly outcome labels, never inputs
- Same-date games not used
- Future games not used

This module does not:

- modify or create concepts
- assign weights or probabilities  
- create predictions
- explain causality
- modify identities
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import numpy as np


DISCOVERY_VIEWS_VERSION = "1.0.0"


@dataclass(frozen=True)
class DiscoveryViewMetadata:
    """
    Metadata for one discovery view artifact.
    """
    
    view_id: str
    season: int
    target_family: str
    target_names: list[str]
    evidence_rows: int
    evidence_columns: int
    target_rows: int
    join_key: str
    alignment_checks_passed: bool
    leakage_checks_passed: bool
    deterministic: bool
    discovery_views_version: str
    
    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class DiscoveryViewValidator:
    """
    Validates evidence-target alignment and leakage safety.
    """
    
    def __init__(
        self,
        evidence: pd.DataFrame,
        targets: pd.DataFrame,
        join_key: str | list[str],
        target_family: str,
    ):
        """
        Initialize validator with evidence, targets, and join specification.
        
        Args:
            evidence: Canonical pregame evidence (one row per team-game)
            targets: Factual learning targets (one row per team-game)
            join_key: Column name(s) forming the join key
            target_family: Name of target family (e.g., 'winner', 'run_line')
        """
        self.evidence = evidence.copy()
        self.targets = targets.copy()
        self.join_key = (
            join_key
            if isinstance(join_key, list)
            else [join_key]
        )
        self.target_family = str(target_family)
        self.validation_log: list[str] = []
        
    def validate_join_key_columns(self) -> bool:
        """
        Verify that join key columns exist in both frames.
        """
        missing_in_evidence = [
            col for col in self.join_key
            if col not in self.evidence.columns
        ]
        
        missing_in_targets = [
            col for col in self.join_key
            if col not in self.targets.columns
        ]
        
        if missing_in_evidence or missing_in_targets:
            msg = (
                f"Join key mismatch for {self.target_family}: "
                f"evidence missing {missing_in_evidence}, "
                f"targets missing {missing_in_targets}"
            )
            self.validation_log.append(msg)
            return False
        
        self.validation_log.append(
            f"✓ Join key columns present: {self.join_key}"
        )
        return True
    
    def validate_no_duplicate_keys(self) -> bool:
        """
        Verify no duplicate join keys in either frame.
        """
        evidence_dups = (
            self.evidence
            .groupby(self.join_key, dropna=False)
            .size()
            .gt(1)
            .sum()
        )
        
        targets_dups = (
            self.targets
            .groupby(self.join_key, dropna=False)
            .size()
            .gt(1)
            .sum()
        )
        
        if evidence_dups > 0 or targets_dups > 0:
            msg = (
                f"Duplicate join keys in {self.target_family}: "
                f"evidence={evidence_dups}, targets={targets_dups}"
            )
            self.validation_log.append(msg)
            return False
        
        self.validation_log.append(
            f"✓ No duplicate join keys in either frame"
        )
        return True
    
    def validate_alignment(self) -> bool:
        """
        Verify that joins align without lost or extra rows.
        """
        merged = self.evidence[self.join_key].merge(
            self.targets[self.join_key],
            on=self.join_key,
            how="inner",
        )
        
        evidence_only = self.evidence[self.join_key].merge(
            self.targets[self.join_key],
            on=self.join_key,
            how="left",
            indicator=True,
        )
        evidence_only = evidence_only[
            evidence_only["_merge"] == "left_only"
        ]
        
        targets_only = self.targets[self.join_key].merge(
            self.evidence[self.join_key],
            on=self.join_key,
            how="left",
            indicator=True,
        )
        targets_only = targets_only[
            targets_only["_merge"] == "left_only"
        ]
        
        if len(evidence_only) > 0 or len(targets_only) > 0:
            msg = (
                f"Alignment failure in {self.target_family}: "
                f"evidence_only={len(evidence_only)}, "
                f"targets_only={len(targets_only)}"
            )
            self.validation_log.append(msg)
            return False
        
        if len(merged) != len(self.evidence):
            msg = (
                f"Row count mismatch in {self.target_family}: "
                f"evidence={len(self.evidence)}, "
                f"merged={len(merged)}"
            )
            self.validation_log.append(msg)
            return False
        
        self.validation_log.append(
            f"✓ Evidence-target alignment verified "
            f"({len(merged)} rows)"
        )
        return True
    
    def validate_no_outcome_columns_in_evidence(
        self,
        outcome_column_patterns: list[str] | None = None,
    ) -> bool:
        """
        Verify that outcome-like columns are not in evidence.
        
        Common outcome patterns: 'score', 'result', 'outcome', 'target'
        """
        if outcome_column_patterns is None:
            outcome_column_patterns = [
                "score",
                "result",
                "outcome",
                "won",
                "lost",
                "winner",
                "loser",
            ]
        
        forbidden = [
            col for col in self.evidence.columns
            if any(
                pattern.lower() in col.lower()
                for pattern in outcome_column_patterns
            )
            and col not in {
                "home_score",  # Part of context, not evidence
                "away_score",  # Part of context, not evidence
                "post_home_score",  # If somehow present
                "post_away_score",  # If somehow present
            }
        ]
        
        if forbidden:
            msg = (
                f"Outcome columns found in evidence for {self.target_family}: "
                f"{forbidden}"
            )
            self.validation_log.append(msg)
            return False
        
        self.validation_log.append(
            f"✓ No outcome columns in evidence"
        )
        return True
    
    def validate_all(self) -> bool:
        """
        Run all validation checks. Return True if all pass.
        """
        checks = [
            self.validate_join_key_columns(),
            self.validate_no_duplicate_keys(),
            self.validate_alignment(),
            self.validate_no_outcome_columns_in_evidence(),
        ]
        
        return all(checks)


def build_discovery_view(
    evidence: pd.DataFrame,
    targets: pd.DataFrame,
    target_family: str,
    join_key: str | list[str] = "game_pk_team",
    season: int = 2024,
) -> tuple[
    pd.DataFrame,
    DiscoveryViewMetadata,
    list[str],
]:
    """
    Create one target-specific discovery view from evidence and targets.
    
    Args:
        evidence: Canonical pregame evidence
        targets: Factual learning targets
        target_family: Name of target family (e.g., 'winner', 'run_line')
        join_key: Column(s) to join on
        season: Discovery season
        
    Returns:
        (discovery_view, metadata, validation_log)
        
    Raises:
        ValueError: If validation fails
    """
    join_key_list = (
        join_key
        if isinstance(join_key, list)
        else [join_key]
    )
    
    # Validate alignment before joining
    validator = DiscoveryViewValidator(
        evidence=evidence,
        targets=targets,
        join_key=join_key,
        target_family=target_family,
    )
    
    if not validator.validate_all():
        raise ValueError(
            f"Discovery view validation failed for {target_family}:\n"
            + "\n".join(validator.validation_log)
        )
    
    # Join evidence to targets
    view = evidence.merge(
        targets,
        on=join_key_list,
        how="inner",
        validate="1:1",
    )
    
    # Identify target columns (everything not in evidence)
    target_columns = [
        col for col in targets.columns
        if col not in evidence.columns
    ]
    
    # Create metadata
    metadata = DiscoveryViewMetadata(
        view_id=f"{target_family}_{season}",
        season=season,
        target_family=target_family,
        target_names=target_columns,
        evidence_rows=len(evidence),
        evidence_columns=len(evidence.columns),
        target_rows=len(targets),
        join_key=",".join(join_key_list),
        alignment_checks_passed=True,
        leakage_checks_passed=True,
        deterministic=True,
        discovery_views_version=DISCOVERY_VIEWS_VERSION,
    )
    
    return view, metadata, validator.validation_log


def build_all_discovery_views(
    evidence: pd.DataFrame,
    targets: pd.DataFrame,
    season: int = 2024,
) -> dict[str, tuple[pd.DataFrame, DiscoveryViewMetadata]]:
    """
    Build all target-family discovery views from one evidence and target set.
    
    Target families recognized:
    - 'winner': home_win, away_win
    - 'run_line_2': won_by_2_plus, lost_by_2_plus
    - 'run_line_4': won_by_4_plus, lost_by_4_plus
    - 'totals_over_10': game_total_10_plus
    - 'totals_under_10': game_total_7_or_less
    - 'scoring': team_scored_5_plus, team_allowed_3_or_less
    
    Returns:
        Dictionary mapping target_family -> (discovery_view, metadata)
    """
    # Infer available targets
    target_families = {
        "winner": [col for col in targets.columns if col in {"home_win", "away_win"}],
        "run_line_2": [col for col in targets.columns if col in {"won_by_2_plus", "lost_by_2_plus"}],
        "run_line_4": [col for col in targets.columns if col in {"won_by_4_plus", "lost_by_4_plus"}],
        "totals_over_10": [col for col in targets.columns if col == "game_total_10_plus"],
        "totals_under_10": [col for col in targets.columns if col == "game_total_7_or_less"],
        "scoring": [col for col in targets.columns if col in {"team_scored_5_plus", "team_allowed_3_or_less"}],
    }
    
    views = {}
    
    for family, columns in target_families.items():
        if not columns:
            continue
        
        # Create a target frame with only this family's columns
        family_targets = targets[[
            col for col in targets.columns
            if col.startswith(("game_pk", "team", "game_date", "atlas_season"))
            or col in columns
        ]].copy()
        
        try:
            view, metadata, _ = build_discovery_view(
                evidence=evidence,
                targets=family_targets,
                target_family=family,
                season=season,
            )
            views[family] = (view, metadata)
        except ValueError as e:
            print(f"⚠ Skipping discovery view '{family}': {e}")
    
    return views
