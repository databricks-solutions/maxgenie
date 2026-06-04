"""Tests for observed benchmark seed normalization."""

import json

import pytest

from maxgenie.benchmark_service import BenchmarkQuestion, BenchmarkStatus
from maxgenie.observed_benchmarks import (
    load_observed_benchmark_run,
    normalize_observed_questions,
    normalize_observed_results,
    write_latest_observed_benchmark_seed,
    write_observed_benchmark_seed,
)


def test_normalize_workspace_benchmark_payloads():
    curated = {
        "questions": [
            {
                "id": "q1",
                "question": ["Show sales"],
                "answer": [{"format": "SQL", "content": ["SELECT 1"]}],
            }
        ]
    }
    results = {
        "benchmark_results": [
            {
                "question_id": "q1",
                "question": "Show sales",
                "status": "FAILED",
                "generated_sql": "SELECT 0",
                "comparison_score": 0.2,
                "comparison_method": "similarity_match",
            }
        ]
    }

    questions = normalize_observed_questions(curated)
    observed_results = normalize_observed_results(results)

    assert questions == [
        BenchmarkQuestion(id="q1", question="Show sales", expected_sql="SELECT 1")
    ]
    assert observed_results[0].question_id == "q1"
    assert observed_results[0].status == BenchmarkStatus.FAILED
    assert observed_results[0].generated_sql == "SELECT 0"


def test_write_and_load_observed_seed(tmp_path):
    seed_path = tmp_path / "benchmark_seed.json"
    write_observed_benchmark_seed(
        curated_questions={
            "questions": [
                {"id": "q1", "question": "Q1", "expected_sql": "SELECT 1"},
                {"id": "q2", "question": "Q2", "expected_sql": "SELECT 2"},
            ]
        },
        benchmark_results={
            "results": [
                {"question_id": "q1", "status": "passed", "score": 1.0},
                {"question_id": "q2", "status": "failed", "score": 0.4},
            ]
        },
        path=seed_path,
    )

    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    assert payload["question_count"] == 2
    assert payload["result_count"] == 2

    run, _ = load_observed_benchmark_run(
        seed_path,
        questions=[
            BenchmarkQuestion(id="q1", question="Q1", expected_sql="SELECT 1"),
            BenchmarkQuestion(id="q2", question="Q2", expected_sql="SELECT 2"),
        ],
        space_id="clone",
    )

    assert run.space_id == "clone"
    assert run.passed == 1
    assert run.failed == 1
    assert run.pass_rate == 50.0


def test_observed_seed_requires_complete_result_coverage(tmp_path):
    seed_path = tmp_path / "benchmark_seed.json"
    seed_path.write_text(
        json.dumps({"results": [{"question_id": "q1", "status": "passed"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not contain results"):
        load_observed_benchmark_run(
            seed_path,
            questions=[
                BenchmarkQuestion(id="q1", question="Q1"),
                BenchmarkQuestion(id="q2", question="Q2"),
            ],
            space_id="clone",
        )


def test_normalize_observed_results_combines_separate_pass_fail_lists():
    results = normalize_observed_results(
        {
            "failed_questions": [{"question_id": "q1", "status": "failed"}],
            "passed_questions": [{"question_id": "q2", "status": "passed"}],
        }
    )

    assert {result.question_id for result in results} == {"q1", "q2"}
    assert [result.status for result in results] == [
        BenchmarkStatus.FAILED,
        BenchmarkStatus.PASSED,
    ]


def test_normalize_observed_results_handles_assessments_and_percent_scores():
    results = normalize_observed_results(
        {
            "results": [
                {"question_id": "q1", "assessment": "GOOD"},
                {"question_id": "q2", "assessment": "NEEDS_REVIEW"},
                {"question_id": "q3", "status": "failed", "score": 50},
            ]
        }
    )

    assert [result.status for result in results] == [
        BenchmarkStatus.PASSED,
        BenchmarkStatus.FAILED,
        BenchmarkStatus.FAILED,
    ]
    assert [result.comparison_score for result in results] == [1.0, 0.5, 0.5]
    assert results[1].comparison_method == "observed_assessment_needs_review"


class _FakeApiClient:
    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []

    def do(self, method: str, path: str, query: dict | None = None):
        self.calls.append((method, path, query))
        if path.endswith("/eval-runs"):
            return {
                "eval_runs": [
                    {
                        "eval_run_id": "older",
                        "eval_run_status": "DONE",
                        "num_questions": 3,
                        "created_timestamp": 1,
                    },
                    {
                        "eval_run_id": "latest",
                        "eval_run_status": "DONE",
                        "num_questions": 2,
                        "created_timestamp": 3,
                    },
                ]
            }
        if path.endswith("/eval-runs/latest/results"):
            return {
                "eval_results": [
                    {"result_id": "r1", "benchmark_question_id": "q1"},
                    {"result_id": "r2", "benchmark_question_id": "q2"},
                ]
            }
        if path.endswith("/results/r1"):
            return {
                "result_id": "r1",
                "benchmark_question_id": "q1",
                "question": "Q1",
                "assessment": "GOOD",
                "expected_response": [{"response_type": "SQL", "response": "SELECT 1"}],
                "actual_response": [{"response_type": "SQL", "response": "SELECT 1"}],
            }
        if path.endswith("/results/r2"):
            return {
                "result_id": "r2",
                "benchmark_question_id": "q2",
                "question": "Q2",
                "assessment": "BAD",
                "expected_response": [{"response_type": "SQL", "response": "SELECT 2"}],
                "actual_response": [{"response_type": "SQL", "response": "SELECT 0"}],
            }
        raise AssertionError(f"Unexpected API path: {path}")


class _FakeWorkspaceClient:
    def __init__(self):
        self.api_client = _FakeApiClient()


def test_write_latest_observed_benchmark_seed_fetches_matching_eval_run(tmp_path):
    seed_path = tmp_path / "latest_observed_seed.json"
    write_latest_observed_benchmark_seed(
        wc=_FakeWorkspaceClient(),
        space_id="space-1",
        path=seed_path,
        expected_question_count=2,
    )

    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    assert payload["source"] == "latest_genie_eval_run"
    assert payload["eval_run_id"] == "latest"
    assert payload["assessment_counts"] == {"bad": 1, "good": 1}

    run, _ = load_observed_benchmark_run(
        seed_path,
        questions=[
            BenchmarkQuestion(id="q1", question="Q1"),
            BenchmarkQuestion(id="q2", question="Q2"),
        ],
        space_id="clone",
    )

    assert run.passed == 1
    assert run.failed == 1
    assert run.results[0].expected_sql == "SELECT 1"
    assert run.results[1].generated_sql == "SELECT 0"
