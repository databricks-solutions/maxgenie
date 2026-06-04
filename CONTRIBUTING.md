# Contributing

MaxGenie changes should preserve the clone-safe optimizer contract: the source Genie space is never modified directly, accepted states are checkpointed, hidden holdout payloads stay out of candidate context, and authentication, compute, endpoint, package, permission, and warehouse failures are reported explicitly.

Before proposing a change:

1. Read the project overview in `README.md`.
2. Keep changes scoped to the optimizer behavior or publication hygiene being addressed.
3. Do not add generated workspace exports, benchmark payloads, local reports, transcripts, credentials, or customer data.
4. Use synthetic fixture data for tests and examples.
5. Run the focused test module for the behavior touched. For shared runtime or CLI changes, run `python -m pytest -q`.

Public-facing changes should receive at least one peer review from a repository owner or subject matter expert before publication.
