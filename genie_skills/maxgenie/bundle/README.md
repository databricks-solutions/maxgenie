# MaxGenie gateway job bundle

Defines `Space Optimizer Optimization Run`, the named Databricks job that gives the
workspace a permanent **Run now** button, run history, and optional scheduling — the
persistent equivalent of `maxgenie submit-run` / `run_skill(mode="submit", ...)`.

This bundle only creates the job definition. It does **not** sync the skill payload;
the runner is provisioned separately by `scripts/sync_to_workspace.py`.

## Deploy

```bash
# 1. Provision the runner in the workspace (once, and after skill changes).
python genie_skills/maxgenie/scripts/sync_to_workspace.py --profile <profile>

# 2. Create/update the named job.
cd genie_skills/maxgenie/bundle
databricks bundle deploy --profile <profile>
```

Then open **Space Optimizer Optimization Run** in the Jobs UI, click **Run now**, and
adjust the parameters if needed. The default serving config is
`databricks-gpt-5-5` at low reasoning effort. Override
`serving_reasoning_effort` in the Run-now form, for example `xhigh`, when you
want a higher-effort proposal pass; endpoints that do not support `xhigh` cap at
`high`.

## Notes

- The task runs `runner.py --mode full` — the same worker as submit mode — so it does
  not re-run submit mode's production-protocol preflight. Set `warehouse_id` and
  serving settings deliberately when overriding the defaults.
- The serverless environment and dependency pins mirror the `runs/submit` payload in
  `maxgenie/workspace_jobs.py`; keep them in sync if that payload changes. The task's
  parameter list intentionally omits flags that equal `runner.py` defaults
  (`--delay-seconds`, `--match-threshold`, and the serving token/format knobs); if one
  of those defaults changes, add it here so the named job and submit mode stay aligned.
- The conversational front door and mode-selection rules live in `../SKILL.md`.
