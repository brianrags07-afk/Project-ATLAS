# Project Atlas Charter

Version: 0.1.0  
Created: 2026-07-04

## Mission

Project Atlas exists to discover, validate, and explain the patterns that make baseball predictable.

Every prediction must be earned, every hypothesis must be tested, and every improvement must make the system smarter — not just more complicated.

## Core Principles

1. No data leakage.
2. Baseball first. Machine learning second.
3. Preserve raw data forever.
4. One notebook equals one responsibility.
5. Master databases use Parquet.
6. Reports and exports use CSV.
7. Metadata uses JSON.
8. Documentation uses Markdown.
9. Every feature must be reproducible.
10. Every prediction must be explainable.
11. Every hypothesis must be tested.
12. Every model must be validated out of sample.
13. Atlas should evolve, not be rebuilt.

## Architecture

Atlas is organized into four layers:

### 1. Data Layer
Stores raw data, processed data, master databases, and validation reports.

### 2. Intelligence Layer
Includes the Research Lab, Identity Engine, Discovery Engine, Pattern Engine, Knowledge Library, and Critic Engine.

### 3. Prediction Layer
Includes winner models, totals models, team-run models, confidence scoring, and explainability.

### 4. Presentation Layer
Includes daily reports, CSV exports, dashboards, and visual summaries.

## Storage Rules

- Parquet = master databases
- CSV = reports and human-readable exports
- JSON = manifests and metadata
- Markdown = documentation

## Non-Negotiable Rule

If a feature was not knowable before first pitch, it cannot be used in a pregame prediction model.

## Motto

Trust is earned.
