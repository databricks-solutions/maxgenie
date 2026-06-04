"""Regression tests for the two comparability defects the Fair-Eval Stage-0 canary surfaced.

Defect #1: content-comparator methods (result_match_content / result_content_mismatch /
result_content_mandatory_missing) must count as result-based coverage, not string-based.

Defect #2: when a Genie message fails because the generated SQL could not execute, the
question must be scored as a *counted* result-based failure (so coverage stays complete and
the greedy gate can reject the candidate), not dropped as an uncounted runtime error.
Genuine infra/transient failures (auth, timeout, space gone) must stay uncounted errors.
"""
from unittest.mock import MagicMock

import pytest
from databricks.sdk import WorkspaceClient

from maxgenie.benchmark_service import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkService,
    BenchmarkStatus,
)
from maxgenie.benchmark_summaries import _comparison_metrics_from_run
from maxgenie.result_comparator import (
    classify_sql_error_text,
    is_sql_execution_failure_text,
)


def _service() -> BenchmarkService:
    return BenchmarkService(MagicMock(spec=WorkspaceClient))


def _content_result(qid: str, method: str, score: float, status: BenchmarkStatus) -> BenchmarkResult:
    return BenchmarkResult(
        question_id=qid,
        question=f"Question {qid}",
        status=status,
        expected_sql="SELECT 1",
        generated_sql="SELECT 1",
        comparison_score=score,
        comparison_method=method,
    )


# ---- Defect #1: content methods are result-based coverage ----

def test_content_comparison_methods_count_as_result_based():
    run = BenchmarkRun(space_id="space")
    run.results = [
        _content_result("q1", "result_match_content", 1.0, BenchmarkStatus.PASSED),
        _content_result("q2", "result_content_mismatch", 0.4, BenchmarkStatus.FAILED),
        _content_result("q3", "result_content_mandatory_missing", 0.0, BenchmarkStatus.FAILED),
    ]
    assert run.result_based_question_count == 3
    assert run.string_based_question_count == 0
    assert _comparison_metrics_from_run(run)["comparison_mode"] == "result_based"


def test_mixed_byte_exact_and_content_methods_are_all_result_based():
    run = BenchmarkRun(space_id="space")
    run.results = [
        _content_result("q1", "result_unbound_parameters_error", 0.0, BenchmarkStatus.FAILED),
        _content_result("q2", "result_match_content", 1.0, BenchmarkStatus.PASSED),
        _content_result("q3", "result_content_mismatch", 0.0, BenchmarkStatus.FAILED),
    ]
    assert run.result_based_question_count == 3
    assert run.string_based_question_count == 0


# ---- Defect #2: Genie SQL-execution failures are counted result-based failures ----

def _question() -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="q1",
        question="Total trx for lokelma last 13 weeks",
        expected_sql="SELECT SUM(trx) FROM sales WHERE time_grp_type = :time_grp_type",
    )


def _run_with_failed_message(monkeypatch, message: dict) -> BenchmarkResult:
    service = _service()
    monkeypatch.setattr(service, "start_conversation", lambda *a, **kw: ("conv1", "msg1"))
    monkeypatch.setattr(service, "poll_until_complete", lambda *a, **kw: (False, message))
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)
    return service.run_question(
        space_id="space",
        benchmark=_question(),
        match_threshold=0.85,
        result_comparator=MagicMock(),
    )


def test_genie_unbound_parameter_failure_scored_as_result_based(monkeypatch: pytest.MonkeyPatch):
    result = _run_with_failed_message(
        monkeypatch,
        {"error": {"error": "[UNBOUND_SQL_PARAMETER] Found the unbound parameter: brand_name. SQLSTATE: 42P02",
                    "type": "SQL_EXECUTION_EXCEPTION"}},
    )
    assert result.status == BenchmarkStatus.FAILED
    assert result.comparison_method == "result_unbound_parameters_error"
    assert result.comparison_score == pytest.approx(0.0)


def test_genie_syntax_error_failure_scored_as_result_based(monkeypatch: pytest.MonkeyPatch):
    result = _run_with_failed_message(
        monkeypatch,
        {"error": {"error": "[PARSE_SYNTAX_ERROR] Syntax error at or near 'v'. SQLSTATE: 42601",
                    "type": "SQL_EXECUTION_EXCEPTION"}},
    )
    assert result.status == BenchmarkStatus.FAILED
    assert result.comparison_method == "result_execution_error"
    assert result.comparison_score == pytest.approx(0.0)


def test_genie_infra_failure_stays_uncounted_error(monkeypatch: pytest.MonkeyPatch):
    result = _run_with_failed_message(monkeypatch, {"error": "Auth error: token expired"})
    assert result.status == BenchmarkStatus.ERROR
    assert result.comparison_score is None
    assert result.comparison_method is None


def test_genie_polling_timeout_stays_uncounted_error(monkeypatch: pytest.MonkeyPatch):
    result = _run_with_failed_message(monkeypatch, {"error": "Polling timeout exceeded (180 attempts)"})
    assert result.status == BenchmarkStatus.ERROR
    assert result.comparison_score is None


def test_genie_cancelled_message_stays_uncounted_error(monkeypatch: pytest.MonkeyPatch):
    result = _run_with_failed_message(
        monkeypatch,
        {"status": "CANCELLED", "error": "Message execution was cancelled"},
    )
    assert result.status == BenchmarkStatus.ERROR
    assert result.comparison_score is None
    assert result.comparison_method is None


# ---- SQL-execution-failure classifier (unit) ----

def test_is_sql_execution_failure_text_detects_sql_errors():
    assert is_sql_execution_failure_text("[UNBOUND_SQL_PARAMETER] x. SQLSTATE: 42P02")
    assert is_sql_execution_failure_text("[PARSE_SYNTAX_ERROR] near 'v'. SQLSTATE: 42601")
    assert is_sql_execution_failure_text('{"type": "SQL_EXECUTION_EXCEPTION"}')


def test_is_sql_execution_failure_text_ignores_infra_errors():
    assert not is_sql_execution_failure_text("Auth error: token expired")
    assert not is_sql_execution_failure_text("Polling timeout exceeded (180 attempts)")
    assert not is_sql_execution_failure_text("Space gone: does not exist")
    assert not is_sql_execution_failure_text(None)


def test_classify_sql_error_text_kinds():
    assert classify_sql_error_text("[UNBOUND_SQL_PARAMETER] time_grp_type") == "unbound_parameters"
    assert classify_sql_error_text("Only SELECT/WITH queries allowed, got: CREATE") == "non_select"
    assert classify_sql_error_text("[PARSE_SYNTAX_ERROR] near 'v'") == "other"
