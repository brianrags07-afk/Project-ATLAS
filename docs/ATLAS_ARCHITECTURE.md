# Project ATLAS Architecture

Version: 3.0.0

---

# Mission

ATLAS is a baseball intelligence system.

Its purpose is to:

- Learn baseball identities
- Analyze pregame matchups
- Predict MLB outcomes
- Learn from every completed game
- Improve continuously

ATLAS does NOT predict from sportsbook odds.

Markets are used only for comparison after predictions are made.

---

# Core Philosophy

ATLAS provides facts.

Prediction models discover what matters.

Every prediction model learns independently.

Examples:

- Moneyline Model
- Totals Model
- Strikeout Model
- Hits Allowed Model
- Total Bases Model
- NRFI/YRFI Model

No model shares target labels.

---

# Engine Pipeline

Daily Data Engine
↓

Daily Snapshot
↓

Game Card Builder
↓

Identity Engine
↓

Matchup Engine
↓

Pregame Feature Matrix
↓

Prediction Models
↓

Live Engine
↓

Postgame Learning Engine

---

# Game States

SCHEDULED

RAW

PARTIAL

READY

PREDICTED

LIVE

FINAL

LEARNING_COMPLETE

---

# Daily Data Engine

Responsibilities:

- Pull schedule
- Pull starters
- Pull lineups
- Pull venue
- Pull weather
- Pull umpires
- Pull market (comparison only)

Output:

Daily Snapshot

---

# Game Card Builder

Creates one JSON document for every MLB game.

A Game Card becomes the single source of truth for that game.

Every engine reads and updates the Game Card.

---

# Identity Engine

Builds:

Team Identity

Pitcher Identity

Hitter Identity

Bullpen Identity

Series Identity

Environment Identity

---

# Matchup Engine

Compares:

Today's starter

vs

Today's confirmed lineup

using historical pitch-by-pitch data.

Outputs matchup features only.

No predictions.

---

# Prediction Models

Independent models.

Moneyline

Totals

Strikeouts

Hits Allowed

Total Bases

NRFI/YRFI

---

# Learning Engine

After every game:

Compare predictions

Compare outcomes

Discover new features

Update identities

Store learning report

---

# Data Rules

Raw data is immutable.

Derived features are reproducible.

Pregame information never uses future knowledge.

Series information must be pregame-safe.

---

# Repository Structure

atlas/
daily/
identity/
matchup/
models/
learning/
utils/

docs/

notebooks/

data/
