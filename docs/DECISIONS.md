# Project Atlas Decisions

## Decision 001 — Storage Formats

Status: Accepted

Atlas will use:
- Parquet for master databases
- CSV for reports and exports
- JSON for metadata and manifests
- Markdown for documentation

Reason:
This balances performance, readability, and long-term maintainability.

## Decision 002 — No Data Leakage

Status: Accepted

Any feature used for pregame prediction must be knowable before first pitch.

## Decision 003 — One Notebook, One Responsibility

Status: Accepted

Each Colab notebook has one clear job.

## Decision 004 — Dashboard-Ready Architecture

Status: Accepted

Atlas will eventually support a dashboard, so prediction outputs and reports should be saved in dashboard-friendly formats.
