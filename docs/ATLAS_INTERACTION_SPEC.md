
# PROJECT ATLAS
# Interaction & Evidence Specification
Version: 1.0.0

## Core Rule

No interaction has a universal weight.

Every interaction must earn importance through evidence.

## Interaction Evaluation

Every interaction is evaluated by:

- Target
- Entity
- Context
- Effect size
- Sample size
- Confidence
- Stability
- Consistency
- Recency
- Validation status

## Target-Specific Knowledge

The same interaction may matter differently for:

- Moneyline
- Runline
- Totals
- Over
- Under
- YRFI / NRFI
- First 5
- Pitcher props
- Batter props

## Entity-Specific Weighting

ATLAS must evaluate interactions separately for:

- Teams
- Pitchers
- Batters
- Bullpens
- Lineups
- Parks
- Umpires

Example:

Wind Out x Fly Ball Offense may be strong for one team and weak for another.

## Valid Interaction Result

An interaction may return:

- Strong positive effect
- Weak positive effect
- No meaningful effect
- Weak negative effect
- Strong negative effect

No meaningful effect is a valid scientific result.

## Required Interaction Fields

Each discovered interaction should store:

- interaction_id
- target
- entities
- context
- sample_size
- effect_size
- confidence
- stability
- evidence
- status
- last_validated

## Bias Control

ATLAS must never assume an interaction matters.

ATLAS measures, validates, weights, or discards.

## Final Principle

Interactions are the bridge between baseball memory and probability.
