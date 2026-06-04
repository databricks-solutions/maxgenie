# MaxGenie

<p align="center">
  <img src="docs/assets/maxgenie-demo.gif" alt="MaxGenie demo" width="720">
</p>

MaxGenie is a Databricks workspace skill for optimizing one Genie space at a time. It runs inside the Databricks workspace, creates a disposable optimization clone, reads the space benchmarks, proposes structured improvements through a Databricks serving endpoint, rejects regressions with benchmark gates, and writes a final report with checkpoints and rollback artifacts.

The source Genie space is never modified directly.

The local `maxgenie` command is an operator wrapper for syncing the skill, launching Databricks jobs, checking status, and running the same workflow from a shell when needed. It is not the primary product surface.

## Promotion Evidence

The one promotable MaxGenie path is the clone-safe workspace skill or Databricks job flow with result-based comparison, a fresh canonical baseline, full benchmark coverage, hidden-holdout isolation, checkpointed rollback, and terminal selection that requires full-score improvement while blocking holdout regression.

Runs that use SQL-string comparison, partial benchmark coverage, observed-only baselines, or noncanonical splits are diagnostics. They can be useful for smoke tests and hypothesis generation, but final reports mark them non-comparable and they must be rerun under the strict result-based protocol before they count as leaderboard or release evidence.

## What It Does

- Parses a Genie space URL and validates Databricks authentication.
- Exports space configuration and benchmark questions.
- Creates or reuses a MaxGenie-managed optimization clone.
- Builds train, validation, and hidden holdout splits.
- Reuses the latest completed Genie benchmark run when requested, so the initial score does not need to be rerun on the source space.
- Applies safe metadata curation and benchmark-gated candidate changes.
- Uses a Databricks serving endpoint for strict JSON candidate proposals.
- Replays compatible prior accepted candidate sequences when local run memory is available.
- Repairs stale replay payloads so required pass-guard examples are preserved.
- Runs train and validation gates before accepting a candidate.
- Restores explicit recovery checkpoints if a clone needs to be recreated mid-run.
- Runs terminal selection so a final state is promoted only when full score improves and holdout does not regress.
- Preserves checkpoints, iteration logs, candidate payloads, benchmark summaries, and owner-facing reports.

## Public Release Guardrails

This repository is intended to contain only the reusable MaxGenie runtime, workspace skill files, templates, documentation, and tests. Do not publish generated workspace exports, benchmark payloads, query-history exports, local reports, transcripts, credentials, or customer data.

Examples and tests use synthetic fixture data. Any domain-specific product names, metric names, table names, and SQL snippets in tests are illustrative fixtures used to exercise optimizer behavior, not customer-provided data.

Before making a repository public, complete the checklist in `PUBLIC_RELEASE.md`, including license selection, third-party dependency license review, peer review, owner assignment, and confirmation that no non-public information is tracked.

## How To Get Help

Databricks support does not cover this content. For questions or bugs, open a GitHub issue and the team will review it on a best-effort basis.

## License

Copyright 2026 Databricks, Inc. All rights reserved. The source is provided subject to the Databricks License in `LICENSE.md`. Included or referenced third-party libraries are subject to their own licenses.

| library | description | license | source |
|---|---|---|---|
| `databricks-sdk` | Databricks SDK for Python | Apache-2.0 | <https://github.com/databricks/databricks-sdk-py> |
| `httpx` | HTTP client | BSD-3-Clause | <https://github.com/encode/httpx> |
| `pydantic` | Data validation | MIT | <https://github.com/pydantic/pydantic> |
| `PyYAML` | YAML parser/emitter | MIT | <https://pyyaml.org/> |
| `typer` | CLI framework | MIT | <https://github.com/fastapi/typer> |
| `python-dotenv` | Local env-file loading | BSD-3-Clause | <https://github.com/theskumar/python-dotenv> |
| `pytest` | Test runner | MIT | <https://github.com/pytest-dev/pytest> |
| `mlflow` | Optional tracking integration | Apache-2.0 | <https://github.com/mlflow/mlflow> |

## Primary Use: Databricks Workspace Skill

The workspace skill lives under `genie_skills/maxgenie`.

Sync it to a user-level skill path:

```bash
python genie_skills/maxgenie/scripts/sync_to_workspace.py \
  --profile <profile> \
  --host "https://<workspace>" \
  --destination "/Workspace/Users/<user>@<domain>/.assistant/skills/maxgenie"
```

Or sync it to a workspace-level skill path:

```bash
python genie_skills/maxgenie/scripts/sync_to_workspace.py \
  --profile <profile> \
  --host "https://<workspace>" \
  --destination "/Workspace/.assistant/skills/maxgenie"
```

After sync, open a Genie space and invoke the `MaxGenie` skill. In workspace authoring runtimes, use the in-process Python entrypoint because child shell processes may not inherit the runtime authentication context. Load the synced workspace runner explicitly; some Genie Code runtimes do not put the skill root on `sys.path`:

```python
import importlib.util
from pathlib import Path

from databricks.sdk import WorkspaceClient


def load_maxgenie_run_skill():
    user_name = WorkspaceClient().current_user.me().user_name
    candidate_paths = [
        Path(f"/Workspace/Users/{user_name}/.assistant/skills/maxgenie/scripts/runner.py"),
        Path("/Workspace/.assistant/skills/maxgenie/scripts/runner.py"),
    ]
    for runner_path in candidate_paths:
        if not runner_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("maxgenie_skill_runner", runner_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_skill
    raise FileNotFoundError("Could not find the synced MaxGenie runner.")


run_skill = load_maxgenie_run_skill()

run_skill(
    mode="full",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    strategy="serving",
    serving_endpoint="<candidate-serving-endpoint>",
    workspace_root="/Workspace/Users/<user>@<domain>/.maxgenie/runs/full",
    use_latest_observed_benchmark=True,
    plateau_rounds=4,
    warehouse_id="<sql-warehouse-id>",
)
```

If the run should create clones in a shared folder, pass `clone_parent_path="/Shared/<team-folder>"`. Otherwise MaxGenie creates clones under the Databricks run-as user's home folder.

## Multi-Hour Databricks Job Runs

For unattended optimization, submit the synced runner as a Databricks job. This is the recommended path for long runs because it is durable and does not depend on a foreground chat session:

```python
run_skill(
    mode="submit",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    strategy="serving",
    serving_endpoint="<candidate-serving-endpoint>",
    workspace_root="/Workspace/Users/<user>@<domain>/.maxgenie/runs/job",
    use_latest_observed_benchmark=True,
    plateau_rounds=4,
    timeout_seconds=14400,
    warehouse_id="<sql-warehouse-id>",
)
```

`mode="submit"` is non-blocking by default from the programmatic workspace entrypoint. It prints the run id and canonical Jobs run page URL immediately; call `run_skill(mode="status", run_id="<run_id>", status_format="chat")` for later chat snapshots or final results. In a local shell, pass `wait=True` or use `maxgenie submit-run --wait` when you want the foreground process to poll and print streamed status updates.

The default job compute mode is serverless. If serverless jobs are unavailable, pass `compute_mode="existing_cluster"` plus `existing_cluster_id="<cluster-id>"`, or pass `compute_mode="new_cluster"` plus `new_cluster_json="<json-or-file>"`.

First-time users may need permissions for:

- reading the source Genie space and its benchmarks
- creating Genie spaces under the selected clone parent folder
- running the selected Databricks job compute mode
- calling the configured serving endpoint
- using a SQL warehouse for result-based comparison

Query-history enrichment may additionally need access to `system.query`. That signal is optional; MaxGenie reports the blocker instead of failing the run.

## Lite Smoke Test

Use lite mode only to verify that export, clone creation, a tiny benchmark subset, candidate evaluation, report writing, and optional clone cleanup work end to end:

```python
run_skill(
    mode="lite",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    benchmark_limit=1,
    max_candidates=1,
    trash_clone=True,
)
```

Lite mode is not the optimizer; full or submit mode is the production path.

## Local Operator Setup

Local installation is useful for syncing the skill, submitting jobs, running status checks, and running the same workflow from a shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

MaxGenie resolves Databricks credentials in this order:

1. `--profile <profile>`
2. `DATABRICKS_CONFIG_PROFILE`
3. `DATABRICKS_HOST` plus `DATABRICKS_TOKEN`

When a space URL host is available and `--profile` is omitted, MaxGenie checks `~/.databrickscfg` and prefers a profile whose `host` matches the space URL.

Validate access:

```bash
maxgenie doctor --space-url "https://<workspace>/genie/rooms/<space_id>"
```

For repeated runs, set:

```bash
export MAXGENIE_DEFAULT_SPACE_URL="https://<workspace>/genie/rooms/<space_id>"
export MAXGENIE_SERVING_ENDPOINT="<candidate-serving-endpoint>"
```

## Local Full Run

```bash
maxgenie optimize \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --strategy serving \
  --serving-endpoint <candidate-serving-endpoint> \
  --use-latest-observed-benchmark \
  --plateau-rounds 4
```

`--strategy serving` asks the configured Databricks serving endpoint to return a strict JSON operation plan. MaxGenie applies only supported structured operations, pushes them to the clone, runs benchmark gates, rejects regressions, and checkpoints accepted states.

When no diagnostic candidate mode is forced, serving optimization uses strict `artifact_patch_v2` candidates by default. Strict patch candidates must be hash-checked, compact, and tied to visible train/validation evidence with markers such as `target_train_failure:<question_id>`, `visible_family:<family>`, `failure_family:<family>`, or `guard_preservation:<question_id>`. When structured visible context is available, the prompt lists exact markers, token hints, expected/generated token hints, one-target plans, family-level plans only when repeated visible failures share an edit surface, compact visible target evidence, shared family-member question/SQL evidence, target and family-member question terms from the same overlap logic used by local validation, compact visible pass-guard evidence, concrete existing edit-file hints and excerpts from the decomposed files, target and family guard-risk hints, family-specific causal repair recipes, and the example-question alignment rule before asking for file edits. Family-level token, shared-token, member-question, and member SQL/result evidence is scoped to the visible family members that share the selected edit surface. Mutating default `artifact_patch_v2` patches with structured visible context must also include a top-level `causal_patch_plan` object so final artifacts show which visible plan the candidate intended to execute. Older, non-strict, or no-visible-context responses that omit the plan are normalized to `null` for artifact compatibility, but mutating default visible-context patches are rejected when the plan is missing. Present plans are checked for consistency with the visible marker, per-edit local use of the selected marker, non-empty family, selected target family, visible-evidence-bound root cause, visible-evidence-bound edit-surface rationale, non-empty compatible edited paths, non-empty and recipe-aligned repair and avoid intent, structural tokens copied only from visible hints and present in each changed edit's new content, and a visible-effect claim that names available visible token evidence when such tokens exist; family-level plans with repeated shared token hints must include at least one shared token in `structural_tokens` and the expected visible effect. `selected_marker` must use an exact marker string such as `target_train_failure:<id>` rather than a raw question ID, and family-level selected markers must agree with the plan `family`. Local validation rejects default strict patches that combine multiple target plans, use only guard-preservation markers without an improvement target, target a high guard-risk visible failure or high guard-risk family plan without any visible guard-preservation marker when pass guards exist, use a family marker without a shared edit surface across at least two visible family members, contradict their own causal patch plan, create a new file for metadata/text-style existing-surface plans, replace whole existing files instead of compact edits, rewrite trusted-query routing text, or add invalid executable assets such as full-query snippets, pass-guard-clone examples, examples whose question does not overlap the selected visible target question, family-level examples whose question does not overlap any visible member question in the selected family, target-level literal-mismatch executable examples/snippets that mention only the generated placeholder or column token instead of the expected visible literal, or target-level executable examples/snippets that omit the missing expected token when visible train evidence supplies one. Older structured modes remain available for controlled diagnostics through `--serving-candidate-mode`, but they are not the default production synthesis path.

For family-level target plans, `shared_token_hints` take precedence over unioned `token_hints`: proposal prompts ask the candidate to include one shared token in `structural_tokens` and `expected_visible_effect`, and local validation enforces the same normalized token rule. Family-level executable examples and SQL snippets also distinguish `shared_expected_token_hints` from `shared_generated_token_hints`; when repeated visible family evidence supplies a shared missing expected token, executable patches must carry that token in both `causal_patch_plan.structural_tokens` and changed content.

Strict target-plan prompts also include a visible-only patch archetype line. The archetype is guidance, not a response-schema field: candidates should encode the chosen shape through existing `causal_patch_plan` fields and `file_edits`. This gives the proposer a concrete artifact shape, such as a one-fragment SQL snippet, one executable example, or one compact metadata edit, while keeping validation on the existing schema and gates.

Each strict target/family plan also includes a patch template hint for the selected edit surface. Template hints show the minimal file-edit shape, such as a new SQL-snippet YAML with `display_name` and one `sql: |` fragment, an example YAML with one overlapping question and executable SQL, a compact metadata `find`/`replace`, or an instruction `append_content` edit. These hints are prompt guidance only and do not change the response schema. For full-file executable YAML patches, local validation parses the proposed file before benchmarking: example patches must be YAML mappings with non-empty `question` and `sql`, and SQL-snippet patches must be YAML mappings with non-empty string/list `sql` plus a string or repairable `display_name`. Do not wrap the YAML file body in a top-level `content:` key; `file_edits[].content` is already the file body. Compact `find`/`replace` edits keep the lighter substring checks.

Metadata/table patches may clarify business meaning, synonyms, or entity hints, but they must not encode generated query plans, CTE names, result-schema column lists, or aggregation instructions. Strict local validation rejects those query-plan-shaped metadata edits before benchmark gates.

When strict artifact patch mode has no safe mutation, the no-change contract is `operations: []`, `file_edits: []`, and `causal_patch_plan: null`. MaxGenie normalizes older no-change responses that used a `no_change` operation with null file edits and records them as no-change rather than as artifact-patch risk rejections.

For a skeleton-first artifact-patch experiment, force `--serving-candidate-mode skeleton_artifact_patch_v2`. This mode keeps the strict patch path but moves target, path, hash, edit mode, marker, family, and find-anchor selection into deterministic local code built only from visible evidence and current decomposed files. The serving proposal may fill only bounded replacement text and existing causal fields. Local validation rejects any response that drifts from the selected skeleton, exceeds the replacement budget, uses bind placeholders, references hidden holdout context, encodes metadata query plans, or omits the required visible token or intent label.

Strict patch prompts include a self-checklist before the response artifacts. The checklist asks the proposer to internally verify hidden-holdout exclusion, selected-marker attribution on every changed edit, patch-shape/template fit, expected-hash/path safety, expected-token use in executable assets, high-risk guard support, pass-guard non-clone behavior, and causal specificity. It is prompt guidance only; candidates must not add checklist fields to the JSON response.

Strict target/family plans also include normalized visible SQL/result delta intent labels, such as removing bind placeholders, literalizing expected values, restoring prior-period comparison logic, or restoring expected result columns. These labels are derived only from visible failure metadata and help `expected_visible_effect` name the actual SQL/result behavior the patch intends to change. MaxGenie preserves the available and selected labels in artifact-patch risk/audit summaries, prior-attempt context, and final reports so a regressing candidate still leaves behind what behavior it intended to change.

Prompt context also promotes visible prior-candidate lessons into the compact summary: accepted patterns are called out as shapes to preserve, while train-destructive, guard-regressing, skipped, or risk-rejected shapes are called out as anti-patterns to avoid unless the new causal plan explains the material difference. If prior visible lessons show train-health collapse, the prompt warns the proposer not to repeat broad/global text, metadata, or patch shapes from rejected attempts. Local validation also rejects a repeated default `artifact_patch_v2` candidate that changes only `instructions/text_instruction.md` after prior visible lessons show train-health collapse.

Prior-candidate lessons include artifact-patch file paths and edit modes for benchmark-rejected and risk-refused file edits, so future proposals can distinguish a rejected text-only append from a rejected example or snippet patch. Prompt compaction groups those lessons by artifact surface, such as examples, SQL snippets, text instructions, metadata, and join specs, before preserving the raw file-edit shapes in a prioritized avoid/revise subsection even when many rejected candidates are present. Local validation rejects an exact repeated single-file path/edit-mode shape after prior visible lessons show train-health collapse, and also rejects an exact repeated risk-refused single-file shape when path, edit mode, and reason all match the prior refused shape. The risk assessment records the blocked path/edit mode for audit.

When prior visible lessons show repeated new executable example or SQL-snippet train collapse, strict target plans stop advertising another new executable asset. Executable example and SQL-snippet plans set `new_file_allowed=false`, prefer listed existing example/snippet files with compact `find` plus `replace`, and return no_change when no existing executable edit surface is available.

Use `--warehouse-id <warehouse_id>` when result-based comparison should execute generated SQL and compare result sets. Without a usable warehouse, MaxGenie falls back to SQL-string comparison and records that in the report as diagnostic, non-comparable evidence. When generated SQL is missing or a result-based comparison attempt fails for an individual question, MaxGenie scores that question as a failed result-based comparison instead of falling back to SQL-string similarity or dropping it from coverage.

Useful flags:

- `--bootstrap-only`: export, clone, split, and score the baseline without candidate generation.
- `--use-latest-observed-benchmark`: seed the baseline from the latest completed Genie benchmark eval run.
- `--fresh-lineage`: start a new MaxGenie-managed clone lineage from the input space.
- `--clone-parent-path /Users/<user>` or `/Shared/<team-folder>`: choose where optimization clones are created.
- `--plateau-rounds` and `--timeout-seconds`: control the unattended loop for normal serving runs. Omit `--max-iterations` and `--candidate-budget` unless a diagnostic lane intentionally needs a hard cap; plateau and the job timeout should normally determine when the lane stops.
- `--serving-reasoning-effort none|low|medium|high|xhigh`: raise proposer reasoning effort (default `low`). Example: `--serving-reasoning-effort xhigh`; endpoints that do not support `xhigh` cap at `high`.
- `--serving-candidate-mode benchmark_example_set`: force one structured benchmark-tagged example-bundle lane for diagnostics or controlled comparisons.
- `--serving-candidate-mode typed_template`: force one visible typed-template lane where the serving proposal chooses visible target IDs and MaxGenie materializes executable examples plus pass guards before risk gating.
- `--serving-candidate-mode schema_time_window`: force one reusable SQL-snippet-fragment lane for visible output-schema, metric-column, period-comparison, and time-window failures without adding more benchmark-shaped examples. Serving-proposed snippets are rejected when they introduce bind placeholders, full `SELECT`/`WITH` statements, or filter snippets that are menus of alternative predicates. A missing snippet display name is deterministically repaired from a clear description, question, id, or alias when possible.
- `--serving-candidate-mode deterministic_guidance`: force one concise text-instruction lane for recurring visible result-error families after examples and snippets have overfit or stayed flat. It can target literal values instead of bind placeholders, required output aliases, period-comparison semantics, or time-window labels, but must not add examples, snippets, file edits, or copied benchmark SQL.
- `--serving-candidate-mode schema_metadata_aliases`: force one guarded structural metadata lane for visible schema, alias, and entity-resolution ambiguity after examples, snippets, and text guidance have plateaued. It can touch at most two existing table or column descriptions/synonyms and should prefer metadata clarity over enabling entity matching.
- `--serving-candidate-mode skeleton_artifact_patch_v2`: force one deterministic skeleton-bound file patch where MaxGenie selects the visible marker, family, path, hash, edit mode, and exact find anchor locally, then validates that the returned replacement text stays bound to that skeleton.
- `--serving-candidate-mode composite_artifact_patch_v2`: force a strict coordinated artifact-patch lane after single-artifact modes plateau. It requires two to four compact, hash-checked decomposed file edits across compatible artifact families, rejects one-file generic instruction rewrites, and should combine visible examples, narrow guidance, and metadata only when they address the same visible failure family without hidden holdout content.
- `--serving-candidate-mode executable_artifact_patch_v2`: force the post-composite executable-asset patch lane. It keeps strict artifact-patch validation but rejects instruction-only, metadata-only, instruction-plus-metadata, malformed executable YAML, placeholder-bearing snippet/example, and pass-guard-clone example patches; at least one changed file must be a concrete `instructions/examples/*.yml` or `instructions/sql_snippets/*/*.yml` asset.
- `--serving-candidate-mode freedom_artifact_patch_v1`: force a broad hash-checked decomposed-file patch lane. It may edit up to twelve allowlisted Genie asset files in one candidate when the visible evidence supports one coherent bundle.
- `--serving-candidate-mode freedom_artifact_patch_with_history_v1`: run a separate visible-only historical analyst call, then force the same broad patch lane with that memo included in proposal context.
- `--serving-candidate-mode data_probe_artifact_patch_v1`: run bounded read-only aggregate SQL probes inside Databricks job compute, then force a compact data-aware patch lane. Probe results omit row-level samples and hidden holdout content.
- `--audit-checkpoint-path <path>`: restore a checkpoint into the clone and run a final audit without candidate generation.
- `--audit-patch-path <path>`: replay an artifact patch sequence into the clone and run a final audit without candidate generation.

## Local Job Submission

The same job run can be submitted from a local shell:

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --strategy serving \
  --serving-endpoint <candidate-serving-endpoint> \
  --workspace-root "/Workspace/Users/<user>@<domain>/.maxgenie/runs/job" \
  --use-latest-observed-benchmark \
  --max-iterations 50 \
  --plateau-rounds 4 \
  --timeout-seconds 14400
```

Job submission requires the production-comparable protocol by default. Runs must provide a SQL warehouse for
result-based comparison and use a fresh baseline unless they are explicitly exploratory. To submit a diagnostic
run that should not be treated as promotion evidence, pass `--allow-diagnostic`.

The default job compute mode is serverless. If serverless jobs are unavailable, use `--compute-mode existing_cluster --existing-cluster-id <cluster-id>` or `--compute-mode new_cluster --new-cluster-json '<json-or-file>'`.

First-time users may need permissions for:

- reading the source Genie space and its benchmarks
- creating Genie spaces under the selected clone parent folder
- running the selected Databricks job compute mode
- calling the configured serving endpoint
- using a SQL warehouse for result-based comparison

Query-history enrichment may additionally need access to `system.query`. That signal is optional; MaxGenie reports the blocker instead of failing the run.

Useful job commands:

```bash
maxgenie submit-run ... --dry-run
maxgenie run-status <run_id> --space-url "https://<workspace>/genie/rooms/<space_id>"
maxgenie cancel-run <run_id> --space-url "https://<workspace>/genie/rooms/<space_id>"
```

To audit a local positive-control checkpoint through the workspace job harness, pass
`--positive-control` with a JSON spec path. MaxGenie uploads the resolved checkpoint
under the selected `--workspace-root` and submits a checkpoint-audit job:

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --runner-path "/Workspace/Users/<user>@<domain>/.conversations/space-optimizer/worker-runners/<lane>/scripts/runner.py" \
  --workspace-root "/Workspace/Users/<user>@<domain>/.conversations/space-optimizer/runs/<lane>" \
  --positive-control ./positive_control.json \
  --warehouse-id <sql-warehouse-id> \
  --fresh-baseline
```

To audit a prepared artifact patch sequence instead of a direct checkpoint restore, use
a positive-control JSON spec whose `patch_path` points at the local patch directory:

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --runner-path "/Workspace/Users/<user>@<domain>/.conversations/space-optimizer/worker-runners/<lane>/scripts/runner.py" \
  --workspace-root "/Workspace/Users/<user>@<domain>/.conversations/space-optimizer/runs/<lane>" \
  --positive-control ./positive_control_patch.json \
  --warehouse-id <sql-warehouse-id> \
  --fresh-baseline
```

Patch positive controls replay strict `artifact_patch_v2` sequences and audit the
replayed clone under the same result-based reporting path.

For controlled patch-mode validation, force the serving scheduler to use decomposed artifact patches. Use `artifact_patch_v2` for the stricter path: at most two file edits, required edit reasons, required hashes for existing files, visible-effect markers from structured train/validation context, compact file-delta metadata, and the same local decomposed-space validation before benchmark gates. Use `skeleton_artifact_patch_v2` when you want deterministic local skeleton selection and only bounded replacement text from the serving response. Use `composite_artifact_patch_v2` only after single-artifact lanes plateau; it keeps the same strict validation but requires a coordinated two-to-four-edit patch across compatible decomposed assets.

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --runner-path "/Workspace/Users/<user>@<domain>/.conversations/space-optimizer/worker-runners/<lane>/scripts/runner.py" \
  --workspace-root "/Workspace/Users/<user>@<domain>/.conversations/space-optimizer/runs/<lane>" \
  --strategy serving \
  --serving-endpoint <candidate-serving-endpoint> \
  --serving-reasoning-effort low \
  --serving-candidate-mode artifact_patch_v2 \
  --warehouse-id <sql-warehouse-id> \
  --fresh-baseline \
  --fresh-lineage
```

Set `--serving-reasoning-effort xhigh` when you want a higher-effort proposal pass; endpoints that do not support `xhigh` cap at `high`.

## Artifact Layout

By default, local runs write under `workspace/<space_id>/` and workspace jobs write under the `--workspace-root` path.

```text
workspace/<space_id>/
  space/
    meta.yml
    tables/
    instructions/
    sample_questions.yml
  benchmarks/
    train_set.json
    validation_set.json
    validation_folds.json
    test_set.json
    split_meta.json
  results/
    baseline_train.json
    latest_train.json
    latest_validation_summary.json
    latest_test_score.txt
    latest_full_summary.json
    observed_benchmark_seed.json
    optimization_summary.json
    terminal_selection.json
    generalization_audit.json
    trace_memory.json
    failure_context_pack.json
    cross_run_memory.json
    query_history_summary.json
    query_history_recommendations.json
  history/
    iteration_log.jsonl
    train_runs/
    validation_runs/
    test_runs/
    full_runs/
    serving_candidates/
    recovery_checkpoints/
  checkpoints/
  clone_meta.json
  workspace_meta.json
  OPTIMIZATION_TASK.md
  final_report.md
```

The final holdout is used for audit and reporting, not for candidate proposal generation.

## Manual Commands

After bootstrap, run these from any directory by passing `--workspace-root` and `--space-id`, or from `workspace/<space_id>/` with defaults:

```bash
maxgenie push
maxgenie benchmark --train
maxgenie benchmark --validation
maxgenie benchmark --test
maxgenie benchmark --full
maxgenie query-history --warehouse-id <warehouse_id>
maxgenie status
maxgenie report
```

`maxgenie curate-loop` runs a conservative metadata curation sweep. Each curation mode is pushed and benchmarked independently, and a mode is accepted only when the benchmark gate does not regress.

## Operating Rules

- Never modify the source Genie space directly.
- Create optimization clones under the run-as user's home folder by default.
- Preserve run artifacts, checkpoints, and iteration logs.
- Accept a candidate only when benchmark gates do not regress.
- Keep hidden holdout content out of candidate proposal context.
- Report authentication, permission, package, endpoint, and warehouse blockers explicitly.
