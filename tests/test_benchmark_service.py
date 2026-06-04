from pathlib import Path
from unittest.mock import MagicMock

import pytest
from databricks.sdk import WorkspaceClient

from maxgenie.benchmark_cache import BenchmarkCache
from maxgenie.benchmark_service import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkService,
    BenchmarkStatus,
    ResultComparatorRequiredError,
)
from maxgenie.result_comparator import classify_comparator_error


def _service() -> BenchmarkService:
    return BenchmarkService(MagicMock(spec=WorkspaceClient))


def test_compare_sql_uses_high_token_match_gate():
    service = _service()
    expected = """
    SELECT market_name, brand_name, SUM(trx) AS total_trx
    FROM sales_data
    WHERE brand_name = 'lokelma' AND hierarchy_level = 'NATN'
    GROUP BY market_name, brand_name
    ORDER BY total_trx DESC
    """
    generated = """
    SELECT brand_name, market_name, SUM(trx) AS total_trx
    FROM sales_data
    WHERE hierarchy_level = 'NATN' AND brand_name = 'farxiga'
    GROUP BY brand_name, market_name
    ORDER BY total_trx DESC
    """

    is_match, score, method, details = service._compare_sql(generated, expected, match_threshold=0.85)

    assert is_match is True
    assert score == 1.0
    assert method == "high_token_match"
    assert "high_token_match_gate=true" in details


def test_compare_sql_uses_large_query_token_match_gate():
    service = _service()
    expected = (
        "WITH x AS (SELECT a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p FROM sales "
        "WHERE market='x' AND brand='y' AND team='z' GROUP BY a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p), "
        "y AS (SELECT a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p FROM sales2 "
        "WHERE market='x' AND brand='y' AND team='z' GROUP BY a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) "
        "SELECT * FROM x UNION ALL SELECT * FROM y"
    )
    generated = (
        "WITH x AS (SELECT a,b,c,d,e,f,g,h FROM sales "
        "WHERE market='x' AND brand='y' AND team='z' GROUP BY a,b,c,d,e,f,g,h), "
        "y AS (SELECT a,b,c,d,e,f,g,h FROM sales2 "
        "WHERE market='x' AND brand='y' AND team='z' GROUP BY a,b,c,d,e,f,g,h) "
        "SELECT * FROM x UNION ALL SELECT * FROM y"
    )

    expected = " ; ".join([expected] * 5)
    generated = " ; ".join([generated] * 5)

    is_match, score, method, details = service._compare_sql(generated, expected, match_threshold=0.85)

    assert is_match is True
    assert method == "large_query_token_match"
    assert score == 0.95
    assert "large_query_token_match_gate=true" in details


def test_compare_sql_fails_when_overlap_is_low():
    service = _service()
    expected = "SELECT brand_name, SUM(trx) FROM sales GROUP BY brand_name"
    generated = "SELECT region, COUNT(*) FROM territories GROUP BY region"

    is_match, score, method, details = service._compare_sql(generated, expected, match_threshold=0.85)

    assert is_match is False
    assert method == "similarity_match"
    assert score < 0.85
    assert "threshold=0.850" in details


def test_poll_until_complete_retries_transient_get_message_errors(monkeypatch: pytest.MonkeyPatch):
    service = _service()
    calls = {"count": 0}

    def flaky_get_message(space_id: str, conversation_id: str, message_id: str) -> dict[str, str]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("room metadata still propagating")
        return {"status": "COMPLETED"}

    monkeypatch.setattr(service, "get_message", flaky_get_message)
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    success, payload = service.poll_until_complete("space", "conv", "msg")

    assert success is True
    assert payload == {"status": "COMPLETED"}
    assert calls["count"] == 3


def test_poll_until_complete_retries_transient_permission_errors(monkeypatch: pytest.MonkeyPatch):
    service = _service()
    calls = {"count": 0}

    def flaky_get_message(space_id: str, conversation_id: str, message_id: str) -> dict[str, str]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError(
                "Unable to get space [abc]. Caused by User has no read permission "
                "for node with aclPath /workspace/123"
            )
        return {"status": "COMPLETED"}

    monkeypatch.setattr(service, "get_message", flaky_get_message)
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    success, payload = service.poll_until_complete("space", "conv", "msg")

    assert success is True
    assert payload == {"status": "COMPLETED"}
    assert calls["count"] == 3


def test_poll_until_complete_fails_hard_auth_without_retry(monkeypatch: pytest.MonkeyPatch):
    service = _service()
    calls = {"count": 0}

    def unauthorized_get_message(space_id: str, conversation_id: str, message_id: str) -> dict[str, str]:
        calls["count"] += 1
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(service, "get_message", unauthorized_get_message)
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    success, payload = service.poll_until_complete("space", "conv", "msg")

    assert success is False
    assert payload["error"] == "Auth error: 403 Forbidden"
    assert calls["count"] == 1


def test_run_benchmarks_persists_cache_after_each_uncached_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    service = _service()
    questions = [
        BenchmarkQuestion(id="q1", question="Question 1", expected_sql="SELECT 1"),
        BenchmarkQuestion(id="q2", question="Question 2", expected_sql="SELECT 2"),
    ]
    cache = BenchmarkCache()
    cache_path = tmp_path / "benchmark_cache.json"
    saved_sizes: list[int] = []

    def fake_run_question(
        *,
        space_id: str,
        benchmark: BenchmarkQuestion,
        match_threshold: float = 0.85,
        result_comparator: object | None = None,
        require_result_based: bool = False,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            question_id=benchmark.id,
            question=benchmark.question,
            status=BenchmarkStatus.PASSED,
            expected_sql=benchmark.expected_sql,
            generated_sql=benchmark.expected_sql,
        )

    original_save = BenchmarkCache.save

    def tracked_save(self: BenchmarkCache, path: str | Path) -> Path:
        saved_sizes.append(self.size)
        return original_save(self, path)

    monkeypatch.setattr(service, "run_question", fake_run_question)
    monkeypatch.setattr(BenchmarkCache, "save", tracked_save)
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    run = service.run_benchmarks(
        space_id="space",
        questions=questions,
        delay_between_questions=0.0,
        cache=cache,
        config_hash="config-hash",
        cache_save_path=cache_path,
    )

    assert run.total == 2
    assert saved_sizes == [1, 2]
    assert cache_path.exists()
    restored = BenchmarkCache.load(cache_path)
    assert restored.size == 2


def test_run_benchmarks_stops_when_min_pass_rate_unreachable(monkeypatch: pytest.MonkeyPatch):
    service = _service()
    questions = [
        BenchmarkQuestion(id="q1", question="Question 1", expected_sql="SELECT 1"),
        BenchmarkQuestion(id="q2", question="Question 2", expected_sql="SELECT 2"),
        BenchmarkQuestion(id="q3", question="Question 3", expected_sql="SELECT 3"),
    ]
    calls: list[str] = []

    def fake_run_question(
        *,
        space_id: str,
        benchmark: BenchmarkQuestion,
        match_threshold: float = 0.85,
        result_comparator: object | None = None,
        require_result_based: bool = False,
    ) -> BenchmarkResult:
        calls.append(benchmark.id)
        return BenchmarkResult(
            question_id=benchmark.id,
            question=benchmark.question,
            status=BenchmarkStatus.FAILED,
            expected_sql=benchmark.expected_sql,
            generated_sql="SELECT wrong",
        )

    monkeypatch.setattr(service, "run_question", fake_run_question)
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    run = service.run_benchmarks(
        space_id="space",
        questions=questions,
        delay_between_questions=0.0,
        min_pass_rate=100.0,
    )

    assert calls == ["q1"]
    assert [result.question_id for result in run.results] == ["q1"]
    assert run.failed == 1


def test_no_sql_generated_counts_as_result_based_failure(monkeypatch: pytest.MonkeyPatch):
    service = _service()
    question = BenchmarkQuestion(
        id="q1",
        question="Show total revenue",
        expected_sql="SELECT SUM(revenue) FROM sales",
    )

    fake_comparator = MagicMock()
    monkeypatch.setattr(service, "start_conversation", lambda *a, **kw: ("conv1", "msg1"))
    monkeypatch.setattr(
        service,
        "poll_until_complete",
        lambda *a, **kw: (True, {"status": "COMPLETED"}),
    )
    monkeypatch.setattr(service, "extract_generated_sql", lambda msg: None)
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    result = service.run_question(
        space_id="space",
        benchmark=question,
        result_comparator=fake_comparator,
    )

    assert result.status == BenchmarkStatus.FAILED
    assert result.error_message == "No SQL generated"
    assert result.comparison_score == pytest.approx(0.0)
    assert result.comparison_method == "result_no_sql_generated"
    fake_comparator.compare.assert_not_called()


def test_cached_no_sql_result_is_repaired_for_result_based_coverage(monkeypatch: pytest.MonkeyPatch):
    service = _service()
    cache = BenchmarkCache()
    question = BenchmarkQuestion(
        id="q1",
        question="Show total revenue",
        expected_sql="SELECT SUM(revenue) FROM sales",
    )
    cache.put(
        question.id,
        "config-hash",
        BenchmarkResult(
            question_id=question.id,
            question=question.question,
            status=BenchmarkStatus.FAILED,
            expected_sql=question.expected_sql,
            error_message="No SQL generated",
        ),
    )

    monkeypatch.setattr(
        service,
        "run_question",
        lambda **kwargs: pytest.fail("cached result should be used"),
    )
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    run = service.run_benchmarks(
        space_id="space",
        questions=[question],
        delay_between_questions=0.0,
        cache=cache,
        config_hash="config-hash",
        result_comparator=MagicMock(),
    )

    assert run.scored_question_count == 1
    assert run.result_based_question_count == 1
    assert run.string_based_question_count == 0
    assert run.comparison_method_counts == {"result_no_sql_generated": 1}
    assert run.results[0].comparison_score == pytest.approx(0.0)


def test_benchmark_run_reports_comparison_metrics():
    run = BenchmarkRun(space_id="space")
    run.results = [
        BenchmarkResult(
            question_id="q1",
            question="Question 1",
            status=BenchmarkStatus.PASSED,
            expected_sql="SELECT 1",
            generated_sql="SELECT 1",
            comparison_score=1.0,
            comparison_method="result_exact",
        ),
        BenchmarkResult(
            question_id="q2",
            question="Question 2",
            status=BenchmarkStatus.FAILED,
            expected_sql="SELECT 2",
            generated_sql="SELECT 0",
            comparison_score=0.4,
            comparison_method="similarity_match",
        ),
        BenchmarkResult(
            question_id="q3",
            question="Question 3",
            status=BenchmarkStatus.PASSED,
            expected_sql=None,
            generated_sql="SELECT 3",
            comparison_score=None,
            comparison_method="no_expected_sql",
        ),
    ]

    payload = run.to_dict()

    assert run.scored_question_count == 2
    assert run.average_comparison_score == 0.7
    assert run.result_based_question_count == 1
    assert run.string_based_question_count == 1
    assert run.comparison_method_counts == {
        "no_expected_sql": 1,
        "result_exact": 1,
        "similarity_match": 1,
    }
    assert payload["summary"]["scored_question_count"] == 2
    assert payload["summary"]["average_comparison_score"] == 0.7
    assert payload["summary"]["result_based_question_count"] == 1
    assert payload["summary"]["string_based_question_count"] == 1
    assert payload["summary"]["comparison_method_counts"] == {
        "no_expected_sql": 1,
        "result_exact": 1,
        "similarity_match": 1,
    }


# ---------------------------------------------------------------------------
# classify_comparator_error unit tests
# ---------------------------------------------------------------------------


def test_unbound_parameter_result_error_tag_is_returned(monkeypatch: pytest.MonkeyPatch):
    """Comparator raises UNBOUND_SQL_PARAMETER; strict result scoring marks it failed."""
    service = _service()
    question = BenchmarkQuestion(
        id="q1",
        question="How many sales?",
        expected_sql="SELECT SUM(trx) FROM sales WHERE brand_name = :brand_name",
    )

    def fake_compare(**kwargs):
        raise RuntimeError("SQL execution FAILED: [UNBOUND_SQL_PARAMETER] brand_name")

    fake_comparator = MagicMock()
    fake_comparator.compare.side_effect = fake_compare

    monkeypatch.setattr(
        service,
        "_compare_sql",
        lambda **kwargs: pytest.fail("_compare_sql should not run after result error"),
    )
    monkeypatch.setattr(service, "start_conversation", lambda *a, **kw: ("conv1", "msg1"))
    monkeypatch.setattr(
        service,
        "poll_until_complete",
        lambda *a, **kw: (True, {"status": "COMPLETED"}),
    )
    monkeypatch.setattr(
        service,
        "extract_generated_sql",
        lambda msg: "SELECT SUM(trx) FROM sales WHERE brand_name = 'lokelma'",
    )
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    result = service.run_question(
        space_id="space",
        benchmark=question,
        match_threshold=0.85,
        result_comparator=fake_comparator,
    )

    assert result.status == BenchmarkStatus.FAILED
    assert result.comparison_method == "result_unbound_parameters_error"
    assert result.comparison_score == pytest.approx(0.0)
    assert "UNBOUND_SQL_PARAMETER" in str(result.comparison_details)


def test_non_select_result_error_tag_is_returned(monkeypatch: pytest.MonkeyPatch):
    """Comparator raises a non-SELECT error; strict result scoring marks it failed."""
    service = _service()
    question = BenchmarkQuestion(
        id="q2",
        question="Create a table",
        expected_sql="CREATE TABLE foo AS SELECT 1",
    )

    fake_comparator = MagicMock()
    fake_comparator.compare.side_effect = ValueError(
        "Only SELECT/WITH queries allowed, got: CREATE"
    )

    monkeypatch.setattr(
        service,
        "_compare_sql",
        lambda **kwargs: pytest.fail("_compare_sql should not run after result error"),
    )
    monkeypatch.setattr(service, "start_conversation", lambda *a, **kw: ("conv2", "msg2"))
    monkeypatch.setattr(
        service,
        "poll_until_complete",
        lambda *a, **kw: (True, {"status": "COMPLETED"}),
    )
    monkeypatch.setattr(
        service,
        "extract_generated_sql",
        lambda msg: "SELECT 1",
    )
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    result = service.run_question(
        space_id="space",
        benchmark=question,
        match_threshold=0.85,
        result_comparator=fake_comparator,
    )

    assert result.status == BenchmarkStatus.FAILED
    assert result.comparison_method == "result_non_select_error"
    assert result.comparison_score == pytest.approx(0.0)


def test_result_comparator_error_methods_count_as_result_based_and_legacy_fallbacks_as_string_based():
    run = BenchmarkRun(space_id="space")
    run.results = [
        BenchmarkResult(
            question_id="q1",
            question="Question 1",
            status=BenchmarkStatus.FAILED,
            comparison_score=0.0,
            comparison_method="result_unbound_parameters_error",
        ),
        BenchmarkResult(
            question_id="q2",
            question="Question 2",
            status=BenchmarkStatus.PASSED,
            comparison_score=0.9,
            comparison_method="result_unbound_parameters_fallback",
        ),
        BenchmarkResult(
            question_id="q3",
            question="Question 3",
            status=BenchmarkStatus.FAILED,
            comparison_score=0.2,
            comparison_method="result_column_mismatch",
        ),
        BenchmarkResult(
            question_id="q4",
            question="Question 4",
            status=BenchmarkStatus.FAILED,
            comparison_score=0.0,
            comparison_method="result_no_sql_generated",
        ),
    ]

    assert run.result_based_question_count == 3
    assert run.string_based_question_count == 1
    assert run.to_dict()["summary"]["result_based_question_count"] == 3
    assert run.to_dict()["summary"]["string_based_question_count"] == 1


def test_other_result_comparator_error_is_result_execution_failure(monkeypatch: pytest.MonkeyPatch):
    """Comparator raises an unrelated error; strict result scoring marks it failed."""
    service = _service()
    question = BenchmarkQuestion(
        id="q3",
        question="Total revenue?",
        expected_sql="SELECT SUM(revenue) FROM sales",
    )

    fake_comparator = MagicMock()
    fake_comparator.compare.side_effect = RuntimeError("Connection timeout")

    monkeypatch.setattr(
        service,
        "_compare_sql",
        lambda **kwargs: pytest.fail("_compare_sql should not run after result error"),
    )
    monkeypatch.setattr(service, "start_conversation", lambda *a, **kw: ("conv3", "msg3"))
    monkeypatch.setattr(
        service,
        "poll_until_complete",
        lambda *a, **kw: (True, {"status": "COMPLETED"}),
    )
    monkeypatch.setattr(
        service,
        "extract_generated_sql",
        lambda msg: "SELECT SUM(revenue) FROM sales",
    )
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    result = service.run_question(
        space_id="space",
        benchmark=question,
        match_threshold=0.85,
        result_comparator=fake_comparator,
    )

    assert result.status == BenchmarkStatus.FAILED
    assert result.comparison_method == "result_execution_error"
    assert result.comparison_score == pytest.approx(0.0)


def test_run_benchmarks_requires_comparator_when_result_based_required():
    """A run that declares result-based intent must hard-error when no comparator is
    wired, rather than silently degrading to string/token comparison."""
    service = _service()
    questions = [BenchmarkQuestion(id="q1", question="q?", expected_sql="SELECT 1")]

    with pytest.raises(ResultComparatorRequiredError):
        service.run_benchmarks(
            space_id="space",
            questions=questions,
            result_comparator=None,
            require_result_based=True,
        )


def test_run_question_requires_comparator_fails_fast(monkeypatch: pytest.MonkeyPatch):
    """run_question fails before contacting Genie when result-based is required but no
    comparator is provided."""
    service = _service()

    def fail_if_called(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("start_conversation must not run on a hard comparator error")

    monkeypatch.setattr(service, "start_conversation", fail_if_called)
    question = BenchmarkQuestion(id="q1", question="q?", expected_sql="SELECT 1")

    with pytest.raises(ResultComparatorRequiredError):
        service.run_question(
            space_id="space",
            benchmark=question,
            result_comparator=None,
            require_result_based=True,
        )


def test_run_benchmarks_allows_string_path_when_not_required(monkeypatch: pytest.MonkeyPatch):
    """Without the result-based requirement (no warehouse configured) the legacy
    string-comparison path is still permitted: no hard error is raised."""
    service = _service()
    question = BenchmarkQuestion(id="q1", question="q?", expected_sql="SELECT 1")

    monkeypatch.setattr(service, "start_conversation", lambda *a, **k: ("conv", "msg"))
    monkeypatch.setattr(service, "poll_until_complete", lambda *a, **k: (True, {"status": "COMPLETED"}))
    monkeypatch.setattr(service, "extract_generated_sql", lambda *a, **k: "SELECT 1")
    monkeypatch.setattr("maxgenie.benchmark_service.time.sleep", lambda _: None)

    run = service.run_benchmarks(
        space_id="space",
        questions=[question],
        result_comparator=None,
        require_result_based=False,
    )

    assert len(run.results) == 1
    assert run.results[0].comparison_method is not None
