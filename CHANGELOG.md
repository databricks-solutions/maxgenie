# Changelog

## v0.1.0 - 2026-06-04

Initial public release.

- Published the clone-safe MaxGenie runtime, workspace skill, templates, documentation, and tests.
- Added Databricks workspace skill and local CLI flows for export, clone creation, benchmark splitting, candidate gating, checkpointing, and final reporting.
- Added Databricks job submission support for unattended runs, status checks, cancellation, and audit/replay jobs.
- Added result-based comparison support when a SQL warehouse is available, with diagnostic labeling for non-comparable runs.
- Added hidden-holdout isolation and terminal selection for promotion eligibility.
- Added no-benchmark advisory handling: one conservative best-practices pass on the clone, not benchmark-verified, with no fabricated score.
- Added the public demo GIF, README architecture diagram, and public release guardrails.
