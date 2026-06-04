"""Tests for failure context extraction."""

import json
from pathlib import Path

from maxgenie.failure_context import build_failure_context, build_structured_failure_context
from maxgenie.serving_candidates import baseline_pass_guard_ids


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_structured_failure_context_excludes_hidden_holdout(tmp_path: Path):
    """Proposer context must never contain holdout rows, ids, SQL, or score signal.

    Even when the workspace also holds holdout/full result artifacts and a benchmark
    payload that carries the holdout question, the proposer-facing failure context is
    built from the visible (train + validation) split only.
    """
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    holdout_id = "HOLDOUT-SECRET-QUESTION"
    holdout_sql = "SELECT secret FROM holdout_secret_table"

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "VISIBLE-TRAIN-FAIL",
                "question": "Visible training failure question",
                "status": "failed",
                "expected_sql": "SELECT SUM(sales) FROM visible_table",
                "generated_sql": "SELECT sales FROM visible_table",
                "comparison_score": 0.0,
                "comparison_method": "result_mismatch",
            },
        ]
    })
    _write_json(results_dir / "latest_validation_summary.json", {"pass_rate": 33.3})

    # Holdout + full artifacts that exist in the same workspace but must never be
    # read into proposer context. Uniquely named so any leak is detectable.
    _write_json(results_dir / "latest_test.json", {
        "results": [
            {
                "question_id": holdout_id,
                "question": "Hidden holdout question text",
                "status": "failed",
                "expected_sql": holdout_sql,
                "generated_sql": "SELECT secret FROM wrong",
                "comparison_score": 0.0,
            },
        ]
    })
    # Full summary carries the holdout score signal (test split pass rate).
    _write_json(results_dir / "latest_full_summary.json", {
        "overall": {"pass_rate": 40.0},
        "test": {"pass_rate": 66.67},
    })
    # Benchmark payload carries all questions including the holdout one.
    _write_json(tmp_path / ".benchmark_payload.json", {
        "questions": [
            {"id": "VISIBLE-TRAIN-FAIL", "question": "Visible training failure question"},
            {"id": holdout_id, "question": "Hidden holdout question text", "expected_sql": holdout_sql},
        ]
    })

    result = build_structured_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
        iteration_label="serving_iteration_07",
    )
    serialized = json.dumps(result)

    # Positive control: the visible train failure is present.
    assert "VISIBLE-TRAIN-FAIL" in serialized
    assert result["failed_question_count"] == 1

    # Holdout id, holdout expected SQL/rows, and the holdout score signal are absent.
    assert holdout_id not in serialized
    assert "holdout_secret_table" not in serialized
    assert "66.67" not in serialized
    assert "full_pass_rate" not in serialized
    # No holdout/full split key leaks into the structured context.
    assert "test" not in result
    assert "overall" not in result
    # The omission is explicitly recorded.
    assert "omitted" in result["hidden_holdout_policy"]


def test_build_failure_context_includes_failed_questions(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "q1",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
                "comparison_score": 1.0,
                "comparison_method": "exact_match",
                "comparison_details": "",
            },
            {
                "question_id": "q2",
                "question": "Show sales by region",
                "status": "failed",
                "expected_sql": "SELECT region, SUM(sales) FROM t GROUP BY region",
                "generated_sql": "SELECT area, SUM(revenue) FROM t GROUP BY area",
                "comparison_score": 0.45,
                "comparison_method": "similarity_match",
                "comparison_details": "token_jaccard=0.6, sequence_ratio=0.3",
            },
        ]
    })

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Failed Questions (1)" in result
    assert "q2" in result
    assert "Show sales by region" in result
    assert "SELECT region, SUM(sales)" in result
    assert "SELECT area, SUM(revenue)" in result
    assert "token_jaccard=0.6" in result
    assert "## Passing Questions (1)" in result
    assert "q1" in result


def test_build_failure_context_includes_compact_failure_evidence_and_family_groups(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "q-pct",
                "question": "What is the percentage difference for Lokelma current vs prior?",
                "status": "failed",
                "expected_sql": (
                    "SELECT (SUM(CASE WHEN time_group = 'current' THEN nrx ELSE 0 END) - "
                    "SUM(CASE WHEN time_group = 'prior' THEN nrx ELSE 0 END)) / "
                    "SUM(CASE WHEN time_group = 'prior' THEN nrx ELSE 0 END) AS percentage_difference "
                    "FROM sales WHERE brand_name = 'Lokelma'"
                ),
                "generated_sql": (
                    "SELECT SUM(nrx) AS nrx FROM sales "
                    "WHERE brand_name = 'Lokelma' AND time_group = 'current'"
                ),
                "comparison_score": 0.38,
                "comparison_method": "result_compare",
            },
            {
                "question_id": "q-region",
                "question": "Show sales by region",
                "status": "failed",
                "expected_sql": "SELECT region_name, SUM(sales) FROM fact_sales GROUP BY region_name",
                "generated_sql": "SELECT territory_name, SUM(sales) FROM fact_sales GROUP BY territory_name",
                "comparison_score": 0.62,
                "comparison_method": "similarity_match",
            },
        ]
    })

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Compact Failure Evidence" in result
    assert "Train-visible failures only" in result
    assert "hidden final holdout content is intentionally omitted" in result
    assert "`q-pct` [percentage_or_comparison_metric]" in result
    assert "Expected compact SQL" in result
    assert "Generated compact SQL" in result
    assert "Missing expected tokens" in result
    assert "`percentage_difference`" in result
    assert "## Failure Families" in result
    assert "`percentage_or_comparison_metric` (1): `q-pct`" in result
    assert "`aggregation_or_grouping_mismatch` (1): `q-region`" in result


def test_build_structured_failure_context_is_json_serializable_with_failure_fields(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "q-guard",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
                "comparison_score": 1.0,
            },
            {
                "question_id": "q-pct",
                "question": "What is the percentage difference current vs prior?",
                "status": "failed",
                "expected_sql": (
                    "SELECT current_sales / prior_sales - 1 AS percentage_difference "
                    "FROM metric_view WHERE period = 'current'"
                ),
                "generated_sql": "SELECT current_sales FROM metric_view WHERE period = 'current'",
                "comparison_score": 0.38,
                "comparison_method": "result_compare",
                "comparison_details": "row mismatch",
            },
        ]
    })
    _write_json(results_dir / "baseline_train.json", {
        "results": [
            {
                "question_id": "q-guard",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
            }
        ]
    })

    result = build_structured_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
        iteration_label="serving_iteration_04",
        best_train_rate=50.0,
        best_validation_rate=40.0,
    )

    json.dumps(result)
    assert result["schema_version"] == 1
    assert result["iteration_label"] == "serving_iteration_04"
    assert result["train_pass_rate"] == 50.0
    assert result["failed_question_count"] == 1
    assert result["failure_families"] == {"percentage_or_comparison_metric": ["q-pct"]}

    failure = result["failures"][0]
    assert failure["question_id"] == "q-pct"
    assert failure["question"] == "What is the percentage difference current vs prior?"
    assert failure["observed_status"] == "failed"
    assert failure["passed"] is False
    assert failure["score"] == 0.38
    assert failure["issue_class"] == "result_mismatch"
    assert failure["expected_sql"].startswith("SELECT current_sales")
    assert failure["generated_sql"] == "SELECT current_sales FROM metric_view WHERE period = 'current'"
    assert failure["likely_root_cause_family"] == "percentage_or_comparison_metric"
    assert failure["suggested_operation_families"] == [
        "add_example_sql",
        "add_sql_snippet",
        "append_text_instruction",
    ]
    assert failure["confidence"] == 0.75
    assert failure["guard_regression_risk"] == "medium"
    assert "prefer_localized_example_or_snippet" in failure["guard_regression_risk_reasons"]
    assert "prior_sales" in failure["missing_expected_tokens"]

    assert result["pass_guards"] == [
        {
            "question_id": "q-guard",
            "question": "Show total sales",
            "expected_sql": "SELECT SUM(sales) FROM t",
            "source": "baseline_train",
            "latest_status": "passed",
            "guard_reason": "baseline_train_pass",
            "accepted_candidate": None,
        }
    ]


def test_structured_failure_context_classifies_result_errors_by_actionable_family(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "q-unbound",
                "question": "Show Farxiga sales for current 13 weeks",
                "status": "failed",
                "expected_sql": "SELECT SUM(sales) FROM t WHERE brand_name = 'Farxiga'",
                "generated_sql": "SELECT SUM(sales) FROM t WHERE brand_name = :brand_name",
                "comparison_method": "result_unbound_parameters_error",
                "error_message": (
                    "Generated SQL did not match expected SQL "
                    "(Result comparison failed: SQL execution FAILED: "
                    "[UNBOUND_SQL_PARAMETER] Found the unbound parameter: brand_name.)"
                ),
            },
            {
                "question_id": "q-columns",
                "question": "Show market share by region",
                "status": "failed",
                "expected_sql": "SELECT region, brand_name, nrx_market_share_pct FROM t",
                "generated_sql": "SELECT exists_flag FROM t",
                "comparison_method": "result_column_mismatch",
                "error_message": (
                    "Generated SQL did not match expected SQL "
                    "(Column mismatch: missing={'region','brand_name'}, extra={'exists_flag'})"
                ),
            },
        ]
    })

    result = build_structured_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    failures = {failure["question_id"]: failure for failure in result["failures"]}
    assert failures["q-unbound"]["issue_class"] == "unbound_parameters"
    assert failures["q-unbound"]["likely_root_cause_family"] == "literal_or_parameter_mismatch"
    assert failures["q-columns"]["issue_class"] == "column_mismatch"
    assert failures["q-columns"]["likely_root_cause_family"] == "selected_metric_or_column_mismatch"
    assert result["failure_families"] == {
        "literal_or_parameter_mismatch": ["q-unbound"],
        "selected_metric_or_column_mismatch": ["q-columns"],
    }


def test_build_structured_failure_context_omits_hidden_holdout_payloads(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    benchmarks_dir = tmp_path / "benchmarks"
    results_dir.mkdir()
    history_dir.mkdir()
    benchmarks_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "train-fail",
                "question": "Visible train question",
                "status": "failed",
                "expected_sql": "SELECT SUM(sales) FROM visible_table",
                "generated_sql": "SELECT COUNT(*) FROM visible_table",
                "comparison_score": 0.1,
            }
        ]
    })
    _write_json(benchmarks_dir / "test_set.json", [
        {
            "id": "hidden-1",
            "question": "Secret holdout question that must not be exposed",
            "expected_sql": "SELECT secret FROM hidden_table",
        }
    ])

    result = build_structured_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
        benchmarks_dir=benchmarks_dir,
    )

    serialized = json.dumps(result)
    assert "Visible train question" in serialized
    assert "Secret holdout" not in serialized
    assert "hidden_table" not in serialized


def test_build_failure_context_includes_baseline_pass_guard_patterns(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "q1",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
            }
        ]
    })
    _write_json(results_dir / "baseline_train.json", {
        "results": [
            {
                "question_id": "q1",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
            }
        ]
    })

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Baseline Pass Guard Patterns" in result
    assert "These questions passed at baseline" in result
    assert "- Guard reason: `baseline_train_pass`" in result
    assert "Baseline Expected SQL Pattern" in result
    assert "SELECT SUM(sales) FROM t" in result


def test_build_failure_context_includes_validation_pass_guard_patterns(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    benchmarks_dir = tmp_path / "benchmarks"
    results_dir.mkdir()
    history_dir.mkdir()
    benchmarks_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "train-pass",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
            }
        ]
    })
    _write_json(results_dir / "baseline_train.json", {
        "results": [
            {
                "question_id": "train-pass",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
            }
        ]
    })
    _write_json(results_dir / "baseline_validation_summary.json", {
        "validation_strategy": "small_sample_kfold",
        "folds": [
            {
                "validation_ids": ["train-pass", "validation-pass", "validation-fail"],
                "failed_ids": ["validation-fail"],
                "error_ids": [],
            }
        ],
    })
    _write_json(results_dir / "latest_validation_summary.json", {
        "validation_strategy": "small_sample_kfold",
        "folds": [
            {
                "validation_ids": ["train-pass", "validation-pass", "validation-fail"],
                "failed_ids": ["validation-pass", "validation-fail"],
                "error_ids": [],
            }
        ],
    })
    _write_json(tmp_path / ".benchmark_payload.json", {
        "questions": [
            {
                "id": "validation-pass",
                "question": "Already passing validation query",
                "expected_sql": "SELECT region, SUM(sales) FROM t GROUP BY region",
            },
            {
                "id": "validation-fail",
                "question": "Failed validation query",
                "expected_sql": "SELECT bad FROM t",
            },
        ],
    })

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
        benchmarks_dir=benchmarks_dir,
    )

    assert "validation-pass" in result
    assert "Already passing validation query" in result
    assert "SELECT region, SUM(sales)" in result
    assert "- Source: `baseline_validation`" in result
    assert "- Latest status after rollback/best checkpoint: `failed`" in result
    assert "- Guard reason: `baseline_validation_pass`" in result
    assert "validation-fail" not in result


def test_build_failure_context_does_not_load_hidden_holdout_artifacts(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    benchmarks_dir = tmp_path / "benchmarks"
    results_dir.mkdir()
    history_dir.mkdir()
    benchmarks_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "train-fail",
                "question": "Visible train question",
                "status": "failed",
                "expected_sql": "SELECT SUM(sales) FROM visible_table",
                "generated_sql": "SELECT COUNT(*) FROM visible_table",
            }
        ]
    })
    _write_json(results_dir / "latest_test_summary.json", {
        "test_ids": ["hidden-1"],
        "questions": [
            {
                "id": "hidden-1",
                "question": "Secret holdout question that must not be exposed",
                "expected_sql": "SELECT secret FROM hidden_table",
            }
        ],
    })
    _write_json(benchmarks_dir / "test_set.json", [
        {
            "id": "hidden-1",
            "question": "Secret holdout benchmark text",
            "expected_sql": "SELECT secret FROM hidden_table",
        }
    ])

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
        benchmarks_dir=benchmarks_dir,
    )

    assert "Visible train question" in result
    assert "hidden final holdout content is intentionally omitted" in result
    assert "Secret holdout" not in result
    assert "hidden_table" not in result


def test_build_failure_context_includes_prior_attempts(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {"results": []})
    _write_text(
        history_dir / "iteration_log.jsonl",
        json.dumps({
            "phase": "candidate_evaluated",
            "label": "optimize.sweep01.serving_iteration_01",
            "name": "entity_matching",
            "outcome": "rejected",
            "reason": "train regressed 44.4% -> 33.3%",
            "train_rate": 33.3,
            "validation_rate": None,
            "train_failed_ids": ["q1", "q2"],
            "skipped_validation": True,
            "candidate_summary": {
                "candidate_name": "metadata_region_matching",
                "candidate_mode": "metadata",
                "rationale": "Enable safer region matching.",
                "applied_operations": [{"op": "set_column_config"}],
                "risk_assessment": {
                    "visible_effect_verification": {
                        "causal_patch_plan": {
                            "expected_visible_effect": "restore_expected_result_columns for prior_nrx",
                            "available_sql_delta_intents": [
                                "restore_expected_result_columns",
                                "remove_unrequested_result_columns",
                            ],
                            "selected_sql_delta_intents": [
                                "restore_expected_result_columns"
                            ],
                        }
                    }
                },
            },
        })
        + "\n"
        + json.dumps({
            "phase": "benchmark_train",
            "label": "optimize.sweep01.serving_iteration_01.train",
            "diff": {
                "fixed": ["q3"],
                "regressed": ["q4"],
                "still_failing": ["q1", "q2"],
            },
        })
        + "\n",
    )

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Prior Optimization Attempts" in result
    assert "entity_matching" in result
    assert "rejected" in result
    assert "train regressed" in result
    assert "metadata_region_matching" in result
    assert "Candidate mode: `metadata`" in result
    assert (
        "Visible SQL/result intent audit: selected `restore_expected_result_columns`; "
        "available `restore_expected_result_columns`, `remove_unrequested_result_columns`"
    ) in result
    assert "Expected visible effect: restore_expected_result_columns for prior_nrx" in result
    assert "Fixed in train before rollback: `q3`" in result
    assert "Regressed pass guards before rollback: `q4`" in result
    assert "### Rejected Patterns To Avoid Or Revise" in result
    assert "Rejected candidate modes: `metadata`=1" in result
    assert "`metadata_region_matching`: rejected because train regressed 44.4% -> 33.3%." in result
    assert "Still failing after rejected attempt: `q1`, `q2`" in result


def test_build_failure_context_summarizes_artifact_patch_shapes(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {"results": []})
    _write_text(
        history_dir / "iteration_log.jsonl",
        json.dumps({
            "phase": "candidate_evaluated",
            "label": "optimize.sweep01.serving_iteration_01",
            "name": "serving_iteration_01",
            "outcome": "rejected",
            "reason": "train_health_failed: 33.3% -> 0.0%",
            "train_rate": 0.0,
            "skipped_validation": True,
            "candidate_summary": {
                "candidate_name": "text_patch_after_collapse",
                "candidate_mode": "artifact_patch_v2",
                "applied_operations": [
                    {
                        "path": "instructions/text_instruction.md",
                        "edit_mode": "append_content",
                        "reason": "target_train_failure:q-period",
                    }
                ],
            },
        })
        + "\n",
    )

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    expected = (
        "file_edit path=instructions/text_instruction.md "
        "mode=append_content reason=target_train_failure:q-period"
    )
    assert f"Applied operations: {expected}" in result
    assert f"Operations that failed the gates: {expected}" in result


def test_build_failure_context_includes_prior_accepted_and_skipped_lessons(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {"results": []})
    _write_text(
        history_dir / "iteration_log.jsonl",
        json.dumps({
            "phase": "candidate_evaluated",
            "label": "optimize.sweep01.serving_iteration_01",
            "name": "serving_iteration_01",
            "outcome": "accepted",
            "train_rate": 50.0,
            "validation_rate": 50.0,
            "train_failed_ids": ["q-overestimated-a", "q-overestimated-b"],
            "validation_failed_ids": ["v-overestimated-a"],
            "candidate_summary": {
                "candidate_name": "accepted_percentage_examples",
                "candidate_mode": "example_set",
                "applied_operations": [
                    {
                        "op": "add_example_sql",
                        "id": "q-pct",
                        "reason": "target_train_failure:q-pct",
                    }
                ],
            },
        })
        + "\n"
        + json.dumps({
            "phase": "benchmark_train",
            "label": "optimize.sweep01.serving_iteration_01.train",
            "diff": {
                "fixed": ["q-pct"],
                "still_failing": ["q-overestimated-a", "q-overestimated-b"],
            },
        })
        + "\n"
        + json.dumps({
            "phase": "candidate_evaluated",
            "label": "optimize.sweep01.serving_iteration_02",
            "name": "serving_iteration_02",
            "outcome": "skipped",
            "reason": "serving_candidate_risk_rejected",
            "candidate_summary": {
                "candidate_name": "risky_ratio_snippet",
                "candidate_mode": "sql_snippet",
                "risk_assessment": {
                    "accepted": False,
                    "reasons": ["sql_snippet_touches_unrelated_metric"],
                },
                "applied_operations": [
                    {
                        "path": "instructions/sql_snippets/measures/ratio.yml",
                        "edit_mode": "replace_file",
                        "reason": "target_train_failure:q-ratio",
                    }
                ],
                "skipped_operations": [
                    {
                        "op": "add_sql_snippet",
                        "reason": "unsupported broad ratio rewrite",
                    }
                ],
            },
        })
        + "\n",
    )

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Prior Candidate Lessons" in result
    assert "### Accepted Patterns To Preserve" in result
    assert "`accepted_percentage_examples`: accepted" in result
    assert "Fixed train-visible failures: `q-pct`" in result
    assert "Did not solve train-visible failures: `q-overestimated-a`, `q-overestimated-b`" in result
    assert "Visible validation still failing: `v-overestimated-a`" in result
    assert "Candidate mode that worked: `example_set`" in result
    assert "Operations to preserve as precedent: add_example_sql id=q-pct reason=target_train_failure:q-pct" in result
    assert "### Skipped Or Risk-Rejected Patterns To Avoid" in result
    assert "`risky_ratio_snippet`: skipped because serving_candidate_risk_rejected" in result
    assert "Risk gate: `sql_snippet_touches_unrelated_metric`" in result
    assert (
        "Candidate shape refused before benchmarking: "
        "file_edit path=instructions/sql_snippets/measures/ratio.yml "
        "mode=replace_file reason=target_train_failure:q-ratio"
    ) in result
    assert "Skipped operations: add_sql_snippet reason=unsupported broad ratio rewrite" in result


def test_build_failure_context_promotes_accepted_fixed_ids_to_pass_guards(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {
        "results": [
            {
                "question_id": "cccccccccccccccccccccccccccccccc",
                "question": "What is the percentage difference for Lokelma current vs prior?",
                "status": "passed",
                "expected_sql": "SELECT percentage_difference FROM fixed_visible_train",
                "generated_sql": "SELECT percentage_difference FROM fixed_visible_train",
            },
            {
                "question_id": "still-failing",
                "question": "Show an overestimated target failure",
                "status": "failed",
                "expected_sql": "SELECT target_sales FROM visible_train",
                "generated_sql": "SELECT target_sales * 3 FROM visible_train",
            },
        ]
    })
    _write_text(
        history_dir / "iteration_log.jsonl",
        json.dumps({
            "phase": "candidate_evaluated",
            "label": "optimize.sweep01.serving_iteration_01",
            "name": "serving_iteration_01",
            "outcome": "accepted",
            "train_rate": 50.0,
            "validation_rate": 50.0,
            "candidate_summary": {
                "candidate_name": "accepted_percentage_examples",
                "candidate_mode": "example_set",
            },
        })
        + "\n"
        + json.dumps({
            "phase": "benchmark_train",
            "label": "optimize.sweep01.serving_iteration_01.train",
            "diff": {
                "fixed": ["cccccccccccccccccccccccccccccccc"],
                "still_failing": ["still-failing"],
            },
        })
        + "\n"
        + json.dumps({
            "phase": "candidate_evaluated",
            "label": "optimize.sweep01.serving_iteration_03",
            "name": "serving_iteration_03",
            "outcome": "rejected",
            "reason": "pass guard regressed: cccccccccccccccccccccccccccccccc",
            "train_rate": 37.5,
            "skipped_validation": True,
            "candidate_summary": {
                "candidate_name": "metadata_entity_matching",
                "candidate_mode": "metadata",
            },
        })
        + "\n",
    )

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Baseline Pass Guard Patterns" in result
    assert "were fixed by an accepted visible-train candidate" in result
    assert "### Guard: What is the percentage difference for Lokelma current vs prior?" in result
    assert "- ID: `cccccccccccccccccccccccccccccccc`" in result
    assert "- Source: `accepted_candidate`" in result
    assert "- Accepted candidate: `accepted_percentage_examples`" in result
    assert "- Guard reason: `accepted_candidate_fix`" in result
    assert "Future candidates must preserve this accepted visible-train fix." in result
    assert "SELECT percentage_difference FROM fixed_visible_train" in result
    assert baseline_pass_guard_ids(result) == ("cccccccccccccccccccccccccccccccc",)


def test_build_failure_context_includes_invalid_serving_proposals(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {"results": []})
    _write_text(
        history_dir / "iteration_log.jsonl",
        json.dumps({
            "phase": "serving_candidate_proposal_invalid",
            "label": "serving_iteration_03",
            "reason": "invalid_serving_response_json",
            "content_length": 47201,
            "raw_content_path": "/Workspace/path/raw_content.txt",
        })
        + "\n",
    )

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "serving_iteration_03" in result
    assert "serving proposal invalid" in result
    assert "invalid_serving_response_json" in result
    assert "Response characters: 47201" in result


def test_build_failure_context_includes_diagnostics(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    results_dir.mkdir()
    history_dir.mkdir()

    _write_json(results_dir / "latest_train.json", {"results": []})
    _write_json(results_dir / "space_diagnostics.json", {
        "metadata_coverage": {
            "total_tables": 3,
            "total_columns": 108,
            "described_columns": 108,
        },
        "entity_matching_candidates": [
            {
                "table_identifier": "catalog.schema.table",
                "column_name": "region_name",
                "reason": "Categorical column",
            }
        ],
        "format_assistance_candidates": [],
        "external_table_references": [],
    })

    result = build_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
    )

    assert "## Space Diagnostics" in result
    assert "Tables: 3" in result
    assert "region_name" in result
    assert "Entity matching candidates" in result
