# Genie Space Optimization Task

## Objective
Maximize the benchmark pass rate for Genie space `{space_id}`.

## Workspace
- `space/` contains the editable Genie assets. Change only these files.
- `benchmarks/train_set.json` contains `{train_count}` train questions with expected SQL.
- `benchmarks/validation_set.json` contains `{validation_count}` validation IDs.
- `benchmarks/test_set.json` contains `{test_count}` final blind test IDs only.
- `benchmarks/validation_folds.json` captures validation-fold metadata when small-sample mode is active.
- Split strategy: `{split_strategy}`.
- `results/` contains the latest benchmark outputs, diagnostics, curation summary, optimization summary, query-history summary, query-history recommendations, and blocked-failure notes.
- `history/iteration_log.jsonl` is the trace log for every benchmark/checkpoint.
{validation_notes}

## Commands
- `maxgenie curate --mode safe_defaults` applies deterministic metadata enrichment locally without the riskier hide-noise and matching toggles.
- `maxgenie curate-loop` applies narrow curation modes one at a time (descriptions, hide_noise, synonyms, entity_matching, format_assistance, matching), benchmarks each against train and validation, and keeps only modes that improve validation without material train regression.
- `maxgenie push` reassembles `space/` and deploys it to the clone.
- `maxgenie query-history` collects recent SQL history and feedback signals when workspace permissions allow it.
- `maxgenie benchmark --train` runs train failures first, then the prior passes.
- `maxgenie benchmark --validation` runs the blind validation split and stores only a score.
- `maxgenie benchmark --test` runs the final blind test set and writes only a pass-rate score.
- `maxgenie benchmark --full` runs train, validation, and final test and stores only train details plus summary metrics.
- `maxgenie status` shows the current clone, latest scores, and blocked-failure streaks.
- `maxgenie report` writes the final concise report.

## Workflow
1. Read `results/baseline_train.json`, `results/space_diagnostics.json`, and `results/optimization_summary.json` before touching assets.
2. Treat `maxgenie optimize` as the default autonomous path. Use `maxgenie curate-loop` only when you want one manual benchmark-gated sweep over selected curation modes.
3. Fix the highest-leverage remaining asset using this order:
   a. table and column descriptions, hidden/noisy fields, joins, and reusable SQL expressions/snippets
   b. prompt matching or format assistance on the right columns
   c. example SQL and usage guidance for recurring query families
   d. text instructions only when the behavior cannot be expressed more structurally
4. If query-history artifacts are available, inspect them before adding new examples. Use `query_history_recommendations.json` as a ranking signal, not as direct training data.
5. Inspect only the relevant assets under `space/`.
6. Make one logical asset change at a time.
7. Run `maxgenie push`.
8. Run `maxgenie benchmark --train`.
9. Use `maxgenie benchmark --validation` as the visible acceptance signal and treat train as a guardrail, not the primary objective.
10. Use `maxgenie benchmark --test` only at checkpoints or the end.
11. If a failure stays unresolved after 3 distinct attempts, explain it in `results/blocked_failures.md` and move on.
12. If train and validation saturate but the final test still lags, stop adding narrow examples and leave the report with a clear owner handoff.
13. Finish with `maxgenie report`.

## Rules
- Never modify `benchmarks/`.
- Do not copy benchmark SQL into trusted examples.
- Prefer structural fixes before text instructions.
- Treat metadata, joins, SQL expressions/snippets, and prompt matching as higher-priority levers than adding more example SQL.
- Use parameterized examples or trusted assets only for stable recurring patterns, not one-off benchmark questions.
- Treat validation and final test results as blind scores only.
- Once train is saturated, judge changes by validation lift, not by train lift.
