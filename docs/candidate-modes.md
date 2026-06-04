# Candidate Modes

MaxGenie serving runs use `artifact_patch_v2` by default. That mode keeps the production path comparable across iterations: candidates are compact decomposed-file edits, tied to visible train/validation evidence, checked against current file hashes, pushed only to the managed clone, and accepted only after benchmark gates pass.

The hidden holdout is not shown to candidate generation. It is used for terminal audit and promotion eligibility.

## Default Mode

`artifact_patch_v2` is the default and recovery mode for normal serving optimization. It rejects patches that:

- reference hidden holdout content
- lack visible train/validation evidence markers
- edit files without matching expected hashes
- rewrite broad files when a compact edit is required
- add malformed executable YAML assets
- regress pass guards or accepted fixes
- encode query-plan-shaped metadata instead of business meaning

If the mode has no safe mutation, the expected no-change response is empty operations and file edits with no causal plan. MaxGenie records that as no-change rather than as an unsafe patch.

## Controlled Modes

These modes are available through `--serving-candidate-mode` for diagnostics, validation lanes, or targeted experiments. They are not the default production scheduler.

| mode | purpose |
|---|---|
| `artifact_patch_v2` | Strict default decomposed-file patch mode. Allows compact hash-checked file edits grounded in visible evidence. |
| `skeleton_artifact_patch_v2` | Deterministic local skeleton selection. MaxGenie chooses the target, path, hash, edit mode, visible marker, and find anchor; the serving response fills bounded replacement text. |
| `composite_artifact_patch_v2` | Coordinated two-to-four-file patch mode for compatible artifact families. Rejects one-file generic instruction rewrites. |
| `executable_artifact_patch_v2` | One-to-three-file patch mode that must edit at least one executable example or SQL-snippet asset. Rejects prose-only, metadata-only, malformed YAML, placeholders, and pass-guard clones. |
| `benchmark_example_set` | Structured benchmark-tagged example-bundle diagnostic mode. |
| `typed_template` | Materializes visible typed-template examples plus pass guards from selected visible target IDs. |
| `schema_time_window` | One reusable SQL-snippet-fragment mode for visible schema, metric-column, period-comparison, and time-window failures. |
| `deterministic_guidance` | One concise text-instruction mode for recurring visible result-error families. |
| `schema_metadata_aliases` | Guarded structural metadata mode for schema, alias, and entity-resolution ambiguity. |
| `freedom_artifact_patch_v1` | Broad hash-checked file patch mode for one coherent visible-evidence bundle. |
| `freedom_artifact_patch_with_history_v1` | Broad file patch mode with a separate visible-only historical analysis memo. |
| `data_probe_artifact_patch_v1` | File patch mode informed by bounded read-only aggregate SQL probes. Probe results omit row-level samples and hidden holdout content. |

## Adaptive Recipes

`--adaptive-recipes` is opt-in. When enabled, fresh serving iterations can map a dominant visible failure family to a narrower recipe mode. Ambiguous or unmapped families still fall back to `artifact_patch_v2`, and recovery after invalid responses, pass-guard regressions, or accepted-fix regressions also stays on `artifact_patch_v2`.

Current adaptive mappings include:

| visible failure family | mode |
|---|---|
| `missing_generated_sql` | `deterministic_guidance` |
| `literal_or_parameter_mismatch` | `typed_template` |
| `percentage_or_comparison_metric` | `typed_template` |
| `time_filter_mismatch` | `schema_time_window` |
| `table_or_join_path_mismatch` | `schema_metadata_aliases` |
| `selected_metric_or_column_mismatch` | `schema_metadata_aliases` |
| `aggregation_or_grouping_mismatch` | `schema_metadata_aliases` |

## Repeated Collapse Handling

When prior visible lessons show repeated new executable example or SQL-snippet train collapse, strict target plans stop advertising another new executable asset. Executable example and SQL-snippet plans set `new_file_allowed=false`, prefer listed existing example/snippet files with compact `find` plus `replace`, and return no_change when no existing executable edit surface is available.

## Example

```bash
maxgenie optimize \
  --space-url "https://<workspace>/genie/rooms/<space_id>" \
  --profile <profile> \
  --strategy serving \
  --serving-endpoint <candidate-serving-endpoint> \
  --warehouse-id <sql-warehouse-id> \
  --serving-candidate-mode skeleton_artifact_patch_v2
```

Forced modes should be treated as controlled diagnostics unless the run also satisfies the production-comparable evidence protocol: result-based comparison, canonical benchmark coverage, fresh comparable baseline, hidden-holdout isolation, checkpointed rollback, and terminal selection.
