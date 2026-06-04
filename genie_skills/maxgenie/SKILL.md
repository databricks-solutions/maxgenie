---
name: MaxGenie
description: Optimize the current Databricks Genie space with MaxGenie; when invoked inside a Genie room, default to optimizing that room's space.
---

# MaxGenie

Use this skill when the user asks to optimize, benchmark, clone, export, inspect, or improve a Databricks Genie space. When the skill is invoked inside a Genie room with no additional instruction, interpret the invocation as "optimize this current Genie space."

In workspace pickers, choose the skill entry named `MaxGenie`. Keep generated clones, run folders, reports, and smoke-test markers under `Space Optimizer` or another non-MaxGenie runtime name so `MaxGenie` remains reserved for the skill entry.

## Workflow

MaxGenie runs a clone-safe optimization loop:

1. Parse the Genie space URL and workspace host.
2. Validate Databricks authentication.
3. Export the source space and benchmark dataset.
4. Create or reuse an optimization clone while leaving the source untouched.
5. Build train, validation, and hidden holdout splits.
6. Run or reuse the baseline benchmark.
7. Generate structured candidate changes.
8. Push candidates only to the clone.
9. Accept only candidates that pass benchmark gates.
10. Preserve checkpoints and write a final report.

## Conversational Run (Genie Code front door)

First action in a Genie room: set `space_url` to the current room URL and call `run_skill(mode="submit", ...)`. Resolve serving endpoint, model, and warehouse only when absent. Echo the launch note and do not stop to ask.

In a Genie room, the current room is the target and optimization is the default action. Treat `@maxgenie`, `@maxgenie optimize`, and "optimize this space" as launch confirmation after the target space is resolved; do not show a menu unless there is no current-space context.

Front-door intent rule:

- If the skill is invoked from a Genie space screen and the room URL or space id is derivable, start the optimize flow for that same screen's space.
- Do not respond with a list of available actions such as Optimize, Benchmark, Clone, Smoke test, or Status check.
- Do not stop after printing host/user discovery. If environment discovery runs, continue directly to `run_skill(mode="submit", ...)`.
- Ask a follow-up only when the current Genie space cannot be identified or a required launch input cannot be resolved automatically.
- Only choose benchmark, clone, smoke test, or status when the user's message explicitly asks for that non-optimization action.

Launch sequence:

1. Resolve the target space from the current room URL. If it is not derivable, ask once for the Genie space URL.
2. Resolve inputs with this precedence:

| input | explicit | host default | env/config | discover |
|---|---|---|---|---|
| `space_url` | user-provided URL | current Genie room URL | `MAXGENIE_DEFAULT_SPACE_URL` | ask once |
| serving endpoint/model | user-provided values | workspace default `databricks-gpt-5-5` if READY | `MAXGENIE_SERVING_ENDPOINT`, `MAXGENIE_SERVING_MODEL` | READY workspace endpoint |
| serving reasoning effort | user-provided value (`xhigh` raises effort) | `low` | `MAXGENIE_SERVING_REASONING_EFFORT` | runtime default `low` |
| SQL warehouse | user-provided warehouse | none | known/default warehouse values | accessible RUNNING/STARTING/STOPPED warehouse |
| workspace root | user-provided root | user home `.maxgenie/runs/job` | configured default | none |

If no accessible warehouse is available, report the blocker instead of falling back to string-only scoring. Echo a one-line launch note, not a question, when optimization was already requested. Example: "Launching serving optimization on space `<id>` with `databricks-gpt-5-5` at low reasoning effort, fresh baseline, result-based scoring on warehouse `<id>`, results under `<workspace_root>`."

3. Launch the durable job in non-blocking submit mode so the cell returns the run id and run page URL immediately. Do not pass `wait=True` in Genie Code; stdout is buffered until cell completion, so progress comes from `mode="status"` and the Jobs run page.

## Loader Snippet

Do not import the runner with `from scripts.runner import run_skill`; Genie Code cells do not always put the skill root on `sys.path`. Load the synced workspace runner explicitly:

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
    raise FileNotFoundError(
        "Could not find the synced MaxGenie runner. Re-sync the skill to "
        "/Workspace/Users/<user>/.assistant/skills/maxgenie."
    )


run_skill = load_maxgenie_run_skill()
```

For optimization, call:

```python
run_skill(
    mode="submit",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    strategy="serving",
    serving_endpoint="<candidate-serving-endpoint>",
    serving_model="<candidate-serving-model>",
    serving_reasoning_effort="low",
    warehouse_id="<sql-warehouse-id>",
    workspace_root="/Workspace/Users/<user>@<domain>/.maxgenie/runs/job",
    use_latest_observed_benchmark=False,
    max_iterations=30,
    plateau_rounds=3,
    timeout_seconds=14400,
)
```

Omit `serving_endpoint`, `serving_model`, `serving_reasoning_effort`, or `warehouse_id` only when you want submit mode to discover the endpoint/model and use the default low reasoning effort. Set `serving_reasoning_effort="xhigh"` when you want a higher-effort proposal pass; endpoints that do not support `xhigh` cap at `high`. The submit output includes the run id, canonical run page URL, deterministic report landing path, and initial run state. Do not wait in the Genie Code submit cell; use the Jobs run page for live progress and `mode="status"` for snapshots or final clone URL/title, scores, terminal-selection reason, and rendered report link.

Run-link surfacing rule (the job is durable and can take 1-2h, so the run page is where the user watches progress): make the printed canonical run page URL the **first line** of your reply, rendered as a clickable link whose visible text is the full URL. Copy it **verbatim** from the submit output — never shorten, relabel (e.g. to the run name), or reconstruct it. In particular never emit the `#job-run/<run_id>` form: it routes back to Genie Code instead of the run. The only valid Jobs fragment is `#job/<job_id>/run/<run_id>` exactly as printed. Then tell the user the job is durable and the submit cell can finish immediately; if the chat/status cell disconnects, the job keeps running and `mode="status"` recovers the same final summary.

Report state immediately after launch and on request. Use the same loaded `run_skill` object:

```python
run_skill(
    mode="status",
    run_id="<run_id>",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    status_format="chat",
)
```

Use the returned Markdown verbatim for job state, task state, optimized clone URL/title, score movement, terminal-selection reason, and rendered report link. In a shell, the equivalent is `maxgenie run-status <run_id> --space-url <space_url> --chat`. Do not guess a report path such as `<workspace_root>/final_report.md`; status derives the correct `<workspace_root>/<source_space_id>/final_report.ipynb` and `.md` paths from the submitted job parameters. Do not hand-write `workspace.export(...)` calls for report retrieval; status already reads workspace files with `workspace.download(...)` and returns the openable links.

Baseline wording rule:

- If the existing Genie Benchmark UI score differs from the MaxGenie run baseline, report both. Say plainly that "no net improvement" means final versus the fresh MaxGenie result-based baseline from this run, not versus the existing Genie UI eval score.

Mode selection rule:

- Real optimization (multi-hour): always `mode="submit"` (durable Databricks job).
- End-to-end smoke: `mode="lite"`.
- Short interactive debugging only, when explicitly requested: `mode="full"` (runs in-process and holds the session open for the whole run).

Fast testing rule:

- The production default is `databricks-gpt-5-5` with low reasoning effort. Do not switch endpoints just to make a real optimization shorter without labeling the run diagnostic/non-promotion evidence.
- To test launch, links, permissions, or state reporting quickly, use `mode="lite"` or submit with `bootstrap_only=True` instead of changing the endpoint.
- If the user explicitly asks for a one-iteration diagnostic candidate loop, use a faster READY endpoint such as `databricks-gpt-5-4-mini`, pass `max_iterations=1`, and label the run diagnostic/non-promotion evidence.

Do not paste raw `run_skill(...)` keyword arguments as the user interface. Launch once optimization is requested; ask only for inputs that cannot be resolved.

## Preferred Invocation

Whether to run in-process or submit a durable job follows the mode selection rule above (real runs submit; smoke and short debugging run in-process). This section is about the preferred *mechanism* once an in-process run is chosen.

When running in-process inside a workspace authoring runtime, prefer the Python entrypoint over the shell command. Some runtimes do not pass their Databricks authentication context to child shell processes.

Use the Loader Snippet, then call:

```python
run_skill(
    mode="full",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    strategy="serving",
    serving_endpoint="<candidate-serving-endpoint>",
    serving_model="<candidate-serving-model>",
    workspace_root="/Workspace/Users/<user>@<domain>/.maxgenie/runs/full",
    plateau_rounds=4,
)
```

If the run should create clones in a shared folder, pass `clone_parent_path="/Shared/<team-folder>"`. Otherwise MaxGenie creates clones under the Databricks run-as user's home folder.

## Unattended Job Mode

For multi-hour runs, submit the synced runner as a Databricks job:

```python
run_skill(
    mode="submit",
    space_url="https://<workspace>/genie/rooms/<space_id>",
    strategy="serving",
    serving_endpoint="<candidate-serving-endpoint>",
    serving_model="<candidate-serving-model>",
    workspace_root="/Workspace/Users/<user>@<domain>/.maxgenie/runs/job",
    plateau_rounds=4,
    timeout_seconds=14400,
)
```

If serverless jobs are unavailable, pass `compute_mode="existing_cluster"` plus `existing_cluster_id="<cluster-id>"`, or pass `compute_mode="new_cluster"` plus `new_cluster_json="<json-or-file>"`.

## Named Job (Jobs UI button and schedule)

The conversational and submit paths launch one-shot runs. For a permanent "Run now" button, scheduling, and run history in one place, deploy the named gateway job defined in `bundle/databricks.yml`:

1. Sync the skill first so the runner exists in the workspace: `python genie_skills/maxgenie/scripts/sync_to_workspace.py --profile <profile>`.
2. Deploy the job from `genie_skills/maxgenie/bundle`: `databricks bundle deploy --profile <profile>`.
3. In the Jobs UI, open `Space Optimizer Optimization Run`, click Run now, fill the space URL, serving endpoint, warehouse, and iteration knobs, then run. Uncomment the (paused) schedule block in the bundle for recurring runs.

The job runs the same `runner.py --mode full` worker as submit mode against an already-synced runner. It does not re-run submit mode's production-protocol preflight, so set the warehouse and serving endpoint deliberately.

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

## Shell Command

For shells or jobs where subprocesses inherit authentication:

```bash
python scripts/runner.py \
  --mode full \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --strategy serving \
  --serving-endpoint "<candidate-serving-endpoint>" \
  --workspace-root "/Workspace/Users/<user>@<domain>/.maxgenie/runs/full" \
  --use-latest-observed-benchmark \
  --plateau-rounds 4
```

For local use, add `--profile <profile>`.

## Outputs

Full and submit modes write normal optimizer artifacts under the selected workspace root:

- exported space assets
- benchmark splits and observed benchmark seed
- candidate proposal requests and parsed payloads
- train, validation, holdout, and full benchmark summaries
- accepted checkpoints and recovery checkpoints
- `results/optimization_summary.json`
- `results/terminal_selection.json`
- `results/generalization_audit.json`
- `final_report.md`

## Controlled Candidate Modes

By default, unforced serving optimization uses strict `artifact_patch_v2` candidates. Each strict default patch must be compact, hash-checked, and tied to visible train/validation evidence with markers such as `target_train_failure:<question_id>`, `visible_family:<family>`, `failure_family:<family>`, or `guard_preservation:<question_id>`. When structured visible context is available, proposal prompts list exact markers, token hints, expected/generated token hints, one-target plans, family-level plans only when repeated visible failures share an edit surface, compact visible target evidence, shared family-member question/SQL evidence, target and family-member question terms from the same overlap logic used by local validation, compact visible pass-guard evidence, concrete existing edit-file hints and excerpts from the decomposed files, target and family guard-risk hints, family-specific causal repair recipes, and the example-question alignment rule before asking for file edits. Family-level token, shared-token, member-question, and member SQL/result evidence is scoped to the visible family members that share the selected edit surface. Mutating default `artifact_patch_v2` patches with structured visible context must also include a top-level `causal_patch_plan` object so final artifacts show which visible plan the candidate intended to execute. Older, non-strict, or no-visible-context responses that omit the plan are normalized to `null` for artifact compatibility, but mutating default visible-context patches are rejected when the plan is missing. Present plans are checked for consistency with the visible marker, per-edit local use of the selected marker, non-empty family, selected target family, visible-evidence-bound root cause, visible-evidence-bound edit-surface rationale, non-empty compatible edited paths, non-empty and recipe-aligned repair and avoid intent, structural tokens copied only from visible hints and present in each changed edit's new content, and a visible-effect claim that names available visible token evidence when such tokens exist; family-level plans with repeated shared token hints must include at least one shared token in `structural_tokens` and the expected visible effect. `selected_marker` must use an exact marker string such as `target_train_failure:<id>` rather than a raw question ID, and family-level selected markers must agree with the plan `family`. Local validation rejects default strict patches that combine multiple target plans, use only guard-preservation markers without an improvement target, target a high guard-risk visible failure or high guard-risk family plan without any visible guard-preservation marker when pass guards exist, use a family marker without a shared edit surface across at least two visible family members, contradict their own causal patch plan, create a new file for metadata/text-style existing-surface plans, replace whole existing files instead of compact edits, rewrite trusted-query routing text, or add invalid executable assets such as full-query snippets, pass-guard-clone examples, examples whose question does not overlap the selected visible target question, family-level examples whose question does not overlap any visible member question in the selected family, target-level literal-mismatch executable examples/snippets that mention only the generated placeholder or column token instead of the expected visible literal, or target-level executable examples/snippets that omit the missing expected token when visible train evidence supplies one. The modes below are diagnostic controls for isolated validation after a local review, not the default production synthesis sequence.

For family-level target plans, `shared_token_hints` take precedence over unioned `token_hints`: proposal prompts ask the candidate to include one shared token in `structural_tokens` and `expected_visible_effect`, and local validation enforces the same normalized token rule. Family-level executable examples and SQL snippets also distinguish `shared_expected_token_hints` from `shared_generated_token_hints`; when repeated visible family evidence supplies a shared missing expected token, executable patches must carry that token in both `causal_patch_plan.structural_tokens` and changed content.

Strict target-plan prompts also include a visible-only patch archetype line. The archetype is guidance, not a response-schema field: candidates should encode the chosen shape through existing `causal_patch_plan` fields and `file_edits`. This gives the proposer a concrete artifact shape, such as a one-fragment SQL snippet, one executable example, or one compact metadata edit, while keeping validation on the existing schema and gates.

Each strict target/family plan also includes a patch template hint for the selected edit surface. Template hints show the minimal file-edit shape, such as a new SQL-snippet YAML with `display_name` and one `sql: |` fragment, an example YAML with one overlapping question and executable SQL, a compact metadata `find`/`replace`, or an instruction `append_content` edit. These hints are prompt guidance only and do not change the response schema. For full-file executable YAML patches, local validation parses the proposed file before benchmarking: example patches must be YAML mappings with non-empty `question` and `sql`, and SQL-snippet patches must be YAML mappings with non-empty string/list `sql` plus a string or repairable `display_name`. Do not wrap the YAML file body in a top-level `content:` key; `file_edits[].content` is already the file body. Compact `find`/`replace` edits keep the lighter substring checks.

Metadata/table patches may clarify business meaning, synonyms, or entity hints, but they must not encode generated query plans, CTE names, result-schema column lists, or aggregation instructions. Strict local validation rejects those query-plan-shaped metadata edits before benchmark gates.

When strict artifact patch mode has no safe mutation, the no-change contract is `operations: []`, `file_edits: []`, and `causal_patch_plan: null`. MaxGenie normalizes older no-change responses that used a `no_change` operation with null file edits and records them as no-change rather than as artifact-patch risk rejections.

For a locally reviewed skeleton-first patch experiment, pass `serving_candidate_mode="skeleton_artifact_patch_v2"`. This keeps strict decomposed-file patch validation, but MaxGenie selects the visible marker, family, exact existing path, current hash, edit mode, and find anchor locally from visible evidence and current decomposed files. The serving response may fill only bounded replacement text and existing causal fields; local validation rejects path, hash, marker, family, or find-anchor drift, hidden-holdout references, bind placeholders, metadata query-plan guidance, and missing required visible token or intent labels before benchmarking.

Strict patch prompts include a self-checklist before the response artifacts. The checklist asks the proposer to internally verify hidden-holdout exclusion, selected-marker attribution on every changed edit, patch-shape/template fit, expected-hash/path safety, expected-token use in executable assets, high-risk guard support, pass-guard non-clone behavior, and causal specificity. It is prompt guidance only; candidates must not add checklist fields to the JSON response.

Strict target/family plans also include normalized visible SQL/result delta intent labels, such as removing bind placeholders, literalizing expected values, restoring prior-period comparison logic, or restoring expected result columns. These labels are derived only from visible failure metadata and help `expected_visible_effect` name the actual SQL/result behavior the patch intends to change. MaxGenie preserves the available and selected labels in artifact-patch risk/audit summaries, prior-attempt context, and final reports so a regressing candidate still leaves behind what behavior it intended to change.

Prompt context also promotes visible prior-candidate lessons into the compact summary: accepted patterns are called out as shapes to preserve, while train-destructive, guard-regressing, skipped, or risk-rejected shapes are called out as anti-patterns to avoid unless the new causal plan explains the material difference. If prior visible lessons show train-health collapse, the prompt warns the proposer not to repeat broad/global text, metadata, or patch shapes from rejected attempts. Local validation also rejects a repeated default `artifact_patch_v2` candidate that changes only `instructions/text_instruction.md` after prior visible lessons show train-health collapse.

Prior-candidate lessons include artifact-patch file paths and edit modes for benchmark-rejected and risk-refused file edits, so future proposals can distinguish a rejected text-only append from a rejected example or snippet patch. Prompt compaction groups those lessons by artifact surface, such as examples, SQL snippets, text instructions, metadata, and join specs, before preserving the raw file-edit shapes in a prioritized avoid/revise subsection even when many rejected candidates are present. Local validation rejects an exact repeated single-file path/edit-mode shape after prior visible lessons show train-health collapse, and also rejects an exact repeated risk-refused single-file shape when path, edit mode, and reason all match the prior refused shape. The risk assessment records the blocked path/edit mode for audit.

When prior visible lessons show repeated new executable example or SQL-snippet train collapse, strict target plans stop advertising another new executable asset. Executable example and SQL-snippet plans set `new_file_allowed=false`, prefer listed existing example/snippet files with compact `find` plus `replace`, and return no_change when no existing executable edit surface is available.

For controlled validation, pass `serving_candidate_mode="typed_template"` to force one visible typed-template lane. In that mode the serving proposal chooses visible target benchmark IDs, while MaxGenie materializes executable visible SQL examples, injects required visible pass guards, and then uses the normal risk and benchmark gates.

After a typed-template lane overfits visible examples, pass `serving_candidate_mode="schema_time_window"` for the next isolated lane. This mode permits one reusable SQL snippet fragment for visible output-schema, metric-column, period-comparison, or time-window failures and does not add copied benchmark examples. Serving-proposed snippets are rejected when they introduce bind placeholders, full `SELECT`/`WITH` statements, or filter snippets that are menus of alternative predicates. Missing snippet display names are deterministically repaired from a clear description, question, id, or alias when possible.

After snippet lanes stay flat, pass `serving_candidate_mode="deterministic_guidance"` for a conservative text-guidance lane. This mode permits exactly one concise `append_text_instruction` operation for recurring visible result-error families such as bind-placeholder literalization, output aliases, period comparison, or time-window labels. It does not add examples, snippets, file edits, or copied benchmark SQL.

After deterministic text guidance also plateaus, pass `serving_candidate_mode="schema_metadata_aliases"` for a guarded structural metadata lane. This mode permits at most two existing table or column metadata edits for visible schema, alias, and entity-resolution ambiguity. Prefer column descriptions and synonyms; enable entity matching only when visible evidence shows entity resolution is the direct blocker.

After metadata-only aliasing also plateaus, pass `serving_candidate_mode="composite_artifact_patch_v2"` only for a locally reviewed coordinated patch lane. This mode requires two to four compact, hash-checked file edits across compatible decomposed assets, rejects one-file generic instruction rewrites, and should combine visible examples, narrow guidance, and metadata only when the edits address the same visible failure family without hidden holdout content.

For an isolated skeleton-first check, pass `serving_candidate_mode="skeleton_artifact_patch_v2"`. This mode allows at most one compact existing-file `find`/`replace` edit and binds the returned patch to one deterministic local skeleton before any benchmark gate runs.

After composite instruction-plus-metadata patches also plateau, pass `serving_candidate_mode="executable_artifact_patch_v2"` only for a locally reviewed executable-asset patch lane. This mode keeps strict hash-checked artifact patches but requires at least one changed `instructions/examples/*.yml` or `instructions/sql_snippets/*/*.yml` file, so broad prose, metadata-only, malformed executable YAML, placeholder-bearing, and pass-guard-clone example edits are rejected before benchmarking.

For a broad bundled recipe, pass `serving_candidate_mode="freedom_artifact_patch_v1"`. This keeps strict hash, path, reason, and placeholder safeguards, but allows up to twelve allowlisted decomposed Genie asset file edits in one candidate when the visible evidence supports a coherent multi-asset patch.

For a broad recipe with explicit history review, pass `serving_candidate_mode="freedom_artifact_patch_with_history_v1"`. MaxGenie first runs a separate visible-only historical analyst call, stores the memo, and then includes it in the candidate proposal context.

For a data-aware recipe, pass `serving_candidate_mode="data_probe_artifact_patch_v1"`. MaxGenie runs bounded read-only aggregate SQL probes inside Databricks job compute before proposal, then includes compact diagnostics such as trim mismatch counts and raw-vs-trimmed distinct counts. Probe context omits row-level samples and hidden holdout content.

## Operating Rules

- Do not edit the source space directly.
- Use observed benchmark reuse when a completed benchmark run already exists and the user wants to avoid a redundant source-space baseline.
- Use result-based comparison when a SQL warehouse is available.
- Do not use hidden holdout question text or SQL as candidate proposal context.
- Report auth, permission, compute, endpoint, package, and warehouse blockers explicitly.
- If a foreground execution times out, rerun the same command. MaxGenie persists lineage, rejected candidate suppressions, checkpoints, and iteration logs so the run can resume.
