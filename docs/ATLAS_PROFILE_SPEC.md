
# PROJECT ATLAS
# Profile Specification
Version: 1.0.0

## Purpose

The Profile Layer turns memory into structured baseball knowledge.

Profiles do not predict outcomes. Profiles organize evidence so later engines can discover what matters.

## Core Questions

Every profile must eventually answer:

- WHEN does this happen?
- WHY does this happen?
- AGAINST WHO does this happen?
- UNDER WHAT CONDITIONS does this happen?
- BY HOW MUCH does it matter?
- HOW CONFIDENT is ATLAS?

## Bias Control Rules

ATLAS begins neutral.

1. No profile is assumed predictive.
2. No context is assumed important.
3. Every effect must be measured.
4. Every effect must include sample size.
5. Every effect must include confidence.
6. Every profile must allow “no meaningful effect” as a valid result.
7. Prediction engines may only use validated signal, not assumptions.
8. ATLAS must reduce or discard weak signals when evidence does not support them.

## Universal Profile Structure

Every profile should contain:

- schema_version
- questions
- facts
- contexts
- interactions
- samples
- confidence
- evidence
- last_updated

## Core Contexts

ATLAS should evaluate pregame-safe context including:

- Home / Away
- Day / Night
- Park
- Weather
- Rest
- Travel
- Series Game
- Opponent
- Pitcher Handedness
- Batter Handedness
- Pitch Type
- Pitch Shape
- Velocity
- Count
- Inning
- Score State
- Bullpen Rest
- Lineup Construction
- Umpire
- Market Context later

## Team Profiles

- Offense Profile
- Contact Profile
- Discipline Profile
- Starting Pitching Profile
- Bullpen Profile
- Defense Profile
- Baserunning Profile
- Moneyline Profile
- Runline Profile
- Totals Profile
- YRFI / NRFI Profile
- First 5 Profile

## Pitcher Profiles

- Pitch Type Profile
- Velocity Profile
- Movement Profile
- Command Profile
- Damage Profile
- Inning Profile
- First Inning Profile
- Times Through Order Profile
- Pitch Count Profile
- Platoon Profile
- Rest Profile
- Park Profile
- Weather Profile
- Opponent Profile
- YRFI / NRFI Profile
- First 5 Profile
- Prop Profile

## Player / Batter Profiles

- Offense Profile
- Contact Profile
- Discipline Profile
- Pitch Type Profile
- Zone Profile
- Count Profile
- Platoon Profile
- Pitcher Matchup Profile
- Home / Road Profile
- Park Profile
- Weather Profile
- Rest Profile
- Late Game Profile
- Prop Profile

## Confidence Rules

Confidence should account for:

- Sample size
- Recency
- Stability
- Context specificity
- Out-of-sample validation
- Volatility

Small samples are allowed, but must be labeled low confidence.

## Final Rule

A profile is not complete because it has averages.

A profile is only useful when it helps ATLAS evaluate conditional baseball evidence without bias.
