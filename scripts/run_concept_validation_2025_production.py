#!/usr/bin/env python3
"""
Production entry point for Phase 2E.5A: execute
``atlas.validation.concept_validation_2025.run_concept_validation_2025``
against the canonical ATLAS artifacts and certify the result.

This script is a thin CLI wrapper. It does not implement any
discovery, freezing, thresholding, or validation logic itself -- all
of that lives in ``atlas.validation.concept_validation_2025`` (merged
in PR #7) and is orchestrated by
``atlas.validation.concept_validation_2025_production``.

It is safe to run repeatedly:

- it never modifies discovery or frozen artifacts,
- it refuses to run if the frozen registries fail immutability checks,
- it refuses to publish canonical validation outputs if lineage
  certification fails,
- it refuses to overwrite a previously *certified* production
  manifest with a failed run's results,
- it returns a non-zero exit code on any failure.

Usage:
    python scripts/run_concept_validation_2025_production.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from atlas.validation.concept_validation_2025_production import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    sys.exit(main())
