# Operator Reference

This reference covers advanced local commands and Databricks job submission. For first-run setup, start with the README.

## Local Job Submission

Use `submit-run` for unattended optimization. It uploads the synced runner context, starts a Databricks job, prints the run id and Jobs run page URL, and leaves the final report under the selected workspace root.

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --strategy serving \
  --serving-endpoint <candidate-serving-endpoint> \
  --workspace-root "/Workspace/Users/<user>@<domain>/.maxgenie/runs/job" \
  --use-latest-observed-benchmark \
  --plateau-rounds 4 \
  --timeout-seconds 14400 \
  --warehouse-id <sql-warehouse-id>
```

By default, job submission requires the production-comparable protocol. If a run is exploratory and should not be treated as promotion evidence, pass `--allow-diagnostic`.

## Useful Flags

- `--bootstrap-only`: export, clone, split, and score the baseline without candidate generation.
- `--use-latest-observed-benchmark`: seed the baseline from the latest completed Genie benchmark eval run.
- `--fresh-baseline`: run a fresh baseline instead of reusing the latest observed benchmark.
- `--fresh-lineage`: start a new MaxGenie-managed clone lineage from the input space.
- `--clone-parent-path /Users/<user>` or `/Shared/<team-folder>`: choose where optimization clones are created.
- `--warehouse-id <warehouse_id>`: enable result-based SQL comparison.
- `--plateau-rounds` and `--timeout-seconds`: control normal unattended stopping.
- `--max-iterations` and `--candidate-budget`: hard caps for diagnostics and controlled validation.
- `--serving-reasoning-effort none|low|medium|high|xhigh`: raise proposal effort when the endpoint supports it.
- `--adaptive-recipes`: allow fresh iterations to choose a narrow mode from the dominant visible failure family.
- `--confirm-on-beat <n>`: re-run an accepted visible benchmark beat before checkpointing it.
- `--content-match`: use the value-first content comparator; re-baseline before comparing with older runs.
- `--audit-checkpoint-path <path>`: restore a checkpoint into the clone and run a final audit without candidate generation.
- `--audit-patch-path <path>`: replay an artifact patch sequence into the clone and run a final audit without candidate generation.
- `--positive-control <path-or-name>`: resolve a positive-control checkpoint or patch sequence locally, upload it, and submit an audit job.

## Compute Modes

Serverless job compute is the default:

```bash
maxgenie submit-run ... --compute-mode serverless
```

Use an existing cluster when serverless jobs are unavailable or when the workspace requires cluster-scoped permissions:

```bash
maxgenie submit-run ... \
  --compute-mode existing_cluster \
  --existing-cluster-id <cluster-id>
```

Use `new_cluster` when the job should create dedicated compute:

```bash
maxgenie submit-run ... \
  --compute-mode new_cluster \
  --new-cluster-json '<json-or-file>'
```

## Status And Control

```bash
maxgenie submit-run ... --dry-run
maxgenie run-status <run_id> --space-url "https://<workspace>/genie/rooms/<space_id>"
maxgenie cancel-run <run_id> --space-url "https://<workspace>/genie/rooms/<space_id>"
```

Use `--wait` on `submit-run` when a foreground shell should poll until completion.

## Audit And Replay

Checkpoint audit restores a checkpoint into the managed clone and runs the final audit path without candidate generation:

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --workspace-root "/Workspace/Users/<user>@<domain>/.maxgenie/runs/audit" \
  --audit-checkpoint-path "/Workspace/Users/<user>@<domain>/.maxgenie/runs/source/checkpoints/<checkpoint>" \
  --warehouse-id <sql-warehouse-id> \
  --fresh-baseline
```

Patch replay audit replays an artifact patch sequence into the clone before running the same final audit:

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --workspace-root "/Workspace/Users/<user>@<domain>/.maxgenie/runs/audit" \
  --audit-patch-path "/Workspace/Users/<user>@<domain>/.maxgenie/runs/patches/<patch-sequence>" \
  --warehouse-id <sql-warehouse-id> \
  --fresh-baseline
```

Positive controls are a local convenience for resolving a named or JSON-described checkpoint or patch sequence before submission:

```bash
maxgenie submit-run \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --workspace-root "/Workspace/Users/<user>@<domain>/.maxgenie/runs/audit" \
  --positive-control ./positive_control.json \
  --warehouse-id <sql-warehouse-id> \
  --fresh-baseline
```

Use only one audit source per job: `--positive-control`, `--audit-checkpoint-path`, or `--audit-patch-path`.

## Evidence Notes

Promotion evidence requires result-based comparison with a SQL warehouse, canonical benchmark coverage, a comparable baseline, hidden-holdout isolation, checkpointed rollback, and terminal selection. Runs that use SQL-string comparison, partial benchmark coverage, observed-only baselines, or forced diagnostic modes are useful for investigation but should not be treated as promotion evidence without rerunning under the comparable protocol.
