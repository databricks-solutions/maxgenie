import json
from pathlib import Path

from typer.testing import CliRunner

from maxgenie import cli as cli_module
from maxgenie.artifacts import ArtifactManager
from maxgenie.trace_memory import (
    CROSS_RUN_MEMORY_SCHEMA_VERSION,
    FAILURE_CONTEXT_PACK_SCHEMA_VERSION,
    TRACE_MEMORY_SCHEMA_VERSION,
    build_cross_run_memory,
    build_failure_context_pack,
    build_trace_memory_index,
    write_trace_memory_artifacts,
)


def _summary(
    *,
    total: int,
    passed: int,
    failed: int = 0,
    errors: int = 0,
    failed_ids: tuple[str, ...] = (),
) -> dict:
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round((passed / total * 100) if total else 0.0, 2),
        "failed_ids": list(failed_ids),
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _workspace(tmp_path: Path) -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path / "workspace" / "space-123", space_id="space-123")
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id="source-123",
        input_space_id="source-123",
        managed_source_space_id="source-123",
        clone_parent_path="/Users/runner@example.com",
    )
    _write_json(
        artifacts.train_set_path,
        [
            {"id": "q-pass", "question": "Show total sales", "expected_sql": "SELECT SUM(sales) FROM t"},
            {
                "id": "q-fail",
                "question": "Percentage difference current vs prior",
                "expected_sql": "SELECT current_sales / prior_sales - 1 AS percentage_difference FROM t",
            },
        ],
    )
    _write_json(
        artifacts.validation_set_path,
        [{"id": "q-val", "question": "Validation", "expected_sql": "SELECT 1"}],
    )
    _write_json(
        artifacts.test_set_path,
        [{"id": "q-hidden", "question": "Hidden", "expected_sql": "SELECT secret FROM hidden_table"}],
    )
    latest_train = {
        "results": [
            {
                "question_id": "q-pass",
                "question": "Show total sales",
                "status": "passed",
                "expected_sql": "SELECT SUM(sales) FROM t",
                "generated_sql": "SELECT SUM(sales) FROM t",
                "comparison_score": 1.0,
                "comparison_method": "exact_match",
            },
            {
                "question_id": "q-fail",
                "question": "Percentage difference current vs prior",
                "status": "failed",
                "expected_sql": "SELECT current_sales / prior_sales - 1 AS percentage_difference FROM t",
                "generated_sql": "SELECT current_sales FROM t",
                "comparison_score": 0.35,
                "comparison_method": "similarity_match",
            },
        ]
    }
    _write_json(artifacts.baseline_train_path, latest_train)
    _write_json(artifacts.latest_train_path, latest_train)
    _write_json(artifacts.baseline_validation_summary_path, _summary(total=2, passed=1, failed=1))
    _write_json(artifacts.latest_validation_summary_path, _summary(total=2, passed=1, failed=1))
    _write_json(
        artifacts.baseline_full_summary_path,
        {
            "overall": _summary(total=4, passed=2, failed=2, failed_ids=("q-fail", "q-hidden")),
            "test": _summary(total=1, passed=0, failed=1, failed_ids=("q-hidden",)),
        },
    )
    _write_json(
        artifacts.latest_full_summary_path,
        {
            "overall": _summary(total=4, passed=2, failed=2, failed_ids=("q-fail", "q-hidden")),
            "test": _summary(total=1, passed=0, failed=1, failed_ids=("q-hidden",)),
        },
    )
    _write_json(
        artifacts.optimization_summary_path,
        {
            "strategy": "serving_autonomous",
            "best_train_rate": 50.0,
            "best_validation_rate": 50.0,
            "stop_reason": "validation_plateau",
            "final": {
                "overall": _summary(total=4, passed=2, failed=2, failed_ids=("q-fail", "q-hidden")),
                "train": _summary(total=2, passed=1, failed=1),
                "validation": _summary(total=2, passed=1, failed=1),
                "test": _summary(total=1, passed=0, failed=1, failed_ids=("q-hidden",)),
            },
            "passes": [
                {
                    "name": "candidate_01",
                    "family": "serving",
                    "outcome": "rejected",
                    "train_rate": 50.0,
                    "reason": "pass guard regressed: q-pass",
                    "skipped_validation": True,
                    "candidate_summary": {"candidate_mode": "example_set"},
                },
                {
                    "name": "candidate_02",
                    "family": "serving",
                    "outcome": "skipped",
                    "train_rate": 50.0,
                    "validation_rate": 50.0,
                    "reason": "no_changes",
                },
            ],
        },
    )
    return artifacts


def test_trace_memory_indexes_scores_questions_candidates_and_guard_regressions(tmp_path: Path):
    artifacts = _workspace(tmp_path)

    trace_memory = build_trace_memory_index(artifacts)
    payload = trace_memory.to_dict()

    assert payload["schema_version"] == TRACE_MEMORY_SCHEMA_VERSION
    assert payload["clone"]["clone_space_id"] == "clone-123"
    assert payload["scores"]["baseline"]["full"]["pass_rate"] == 50.0
    assert payload["scores"]["current"]["holdout"]["pass_rate"] == 0.0
    assert payload["failure_families"] == {"percentage_or_comparison_metric": ["q-fail"]}
    assert payload["guard_regressions"] == [
        {
            "question_id": "q-pass",
            "candidate_name": "candidate_01",
            "reason": "pass guard regressed: q-pass",
            "source_path": str(artifacts.optimization_summary_path),
        }
    ]

    questions = {question["question_id"]: question for question in payload["questions"]}
    assert questions["q-fail"]["root_cause_family"] == "percentage_or_comparison_metric"
    assert "prior_sales" in questions["q-fail"]["missing_expected_tokens"]
    assert questions["q-fail"]["expected_sql"].startswith("SELECT current_sales")
    assert questions["q-fail"]["generated_sql"] == "SELECT current_sales FROM t"
    assert questions["q-pass"]["split_membership"] == ["train"]


def test_failure_context_pack_is_compact_and_omits_hidden_holdout_payload(tmp_path: Path):
    artifacts = _workspace(tmp_path)
    trace_memory = build_trace_memory_index(artifacts)

    pack = build_failure_context_pack(trace_memory).to_dict()

    assert pack["schema_version"] == FAILURE_CONTEXT_PACK_SCHEMA_VERSION
    assert pack["scorecard"]["baseline"]["full"]["pass_rate"] == 50.0
    assert pack["scorecard"]["baseline"]["full"]["failed_ids"] == []
    assert pack["scorecard"]["baseline"]["full"]["failed_ids_policy"].startswith("omitted")
    assert pack["scorecard"]["current"]["holdout"]["failed_ids"] == []
    assert pack["scorecard"]["current"]["holdout"]["failed_ids_policy"].startswith("omitted")
    assert pack["scorecard"]["current"]["train"]["failed_ids"] == []
    assert pack["target_failures"][0]["question_id"] == "q-fail"
    assert pack["target_failures"][0]["root_cause_family"] == "percentage_or_comparison_metric"
    assert pack["pass_guards"][0]["question_id"] == "q-pass"
    assert pack["guard_regressions"][0]["question_id"] == "q-pass"
    assert pack["candidate_lessons"][0]["candidate_mode"] == "example_set"
    assert "q-hidden" not in json.dumps(pack)
    assert "hidden_table" not in json.dumps(pack)
    assert "Prior attempts regressed pass guards" in pack["recommended_next_focus"]


def test_write_trace_memory_artifacts(tmp_path: Path):
    artifacts = _workspace(tmp_path)

    memory_path, pack_path = write_trace_memory_artifacts(artifacts)

    assert memory_path == artifacts.trace_memory_path
    assert pack_path == artifacts.failure_context_pack_path
    assert json.loads(memory_path.read_text(encoding="utf-8"))["schema_version"] == TRACE_MEMORY_SCHEMA_VERSION
    assert (
        json.loads(pack_path.read_text(encoding="utf-8"))["schema_version"]
        == FAILURE_CONTEXT_PACK_SCHEMA_VERSION
    )
    assert (
        json.loads(artifacts.cross_run_memory_path.read_text(encoding="utf-8"))["schema_version"]
        == CROSS_RUN_MEMORY_SCHEMA_VERSION
    )

def test_cross_run_memory_captures_prior_accepted_patterns_without_hidden_payload(tmp_path: Path):
    artifacts = _workspace(tmp_path)
    previous = ArtifactManager(tmp_path / "s12-worker" / "space-123", space_id="space-123")
    _write_json(
        previous.optimization_summary_path,
        {
            "baseline": {
                "full_pass_rate": 40.0,
            },
            "final": {
                "overall": _summary(total=5, passed=3, failed=2),
                "test": _summary(total=1, passed=0, failed=1),
            },
            "passes": [
                {
                    "name": "serving_iteration_01",
                    "outcome": "accepted",
                    "train_rate": 77.78,
                    "validation_rate": 66.67,
                    "checkpoint_path": "/Workspace/runs/s12/checkpoints/best",
                    "candidate_summary": {
                        "candidate_name": "guarded_percentage_example_set",
                        "candidate_mode": "example_set",
                        "operation_count": 7,
                        "payload_path": "/Workspace/runs/s12/history/candidate_payload.json",
                        "risk_assessment": {
                            "canonicalization": {
                                "canonicalized_example_sql_ids": ["q-pass", "q-fail"],
                            },
                            "reasons": [],
                        },
                    },
                },
                {
                    "name": "serving_iteration_02",
                    "outcome": "accepted",
                    "train_rate": 77.78,
                    "validation_rate": 66.67,
                    "checkpoint_path": "/Workspace/runs/s12/checkpoints/best-2",
                    "candidate_summary": {
                        "candidate_name": "second_guarded_percentage_example_set",
                        "candidate_mode": "example_set",
                        "operation_count": 4,
                        "payload_path": "/Workspace/runs/s12/history/candidate_payload_2.json",
                        "risk_assessment": {
                            "canonicalization": {
                                "canonicalized_example_sql_ids": ["q-next"],
                            },
                            "reasons": [],
                        },
                    },
                },
                {
                    "name": "serving_iteration_03",
                    "outcome": "skipped",
                    "train_rate": 77.78,
                    "candidate_summary": {
                        "candidate_name": "too_broad_example_set",
                        "candidate_mode": "example_set",
                        "operation_count": 8,
                        "risk_assessment": {
                            "reasons": ["multiple_high_guard_risk_targets:q-a,q-b"],
                        },
                    },
                },
            ],
        },
    )

    memory = build_cross_run_memory(artifacts)
    payload = memory.to_dict()

    assert payload["schema_version"] == CROSS_RUN_MEMORY_SCHEMA_VERSION
    assert payload["runs_considered"] == 1
    assert payload["best_prior_lesson"]["candidate_name"] == "guarded_percentage_example_set"
    assert payload["best_prior_lesson"]["canonicalized_example_sql_ids"] == ["q-pass", "q-fail"]
    assert payload["best_prior_lesson"]["final_holdout_rate"] == 0.0
    assert [
        lesson["candidate_name"]
        for lesson in payload["best_prior_sequence"]
    ] == [
        "guarded_percentage_example_set",
        "second_guarded_percentage_example_set",
    ]
    assert payload["rejected_or_skipped_lessons"][0]["risk_reasons"] == [
        "multiple_high_guard_risk_targets:q-a,q-b"
    ]
    assert "Hidden" not in json.dumps(payload)
    assert "hidden_table" not in json.dumps(payload)
