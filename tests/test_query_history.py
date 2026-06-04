"""Tests for query-history collection and summarization."""

from maxgenie.benchmark_service import BenchmarkQuestion
from maxgenie.query_history import (
    QueryHistoryCollector,
    QueryHistorySummary,
    build_expansion_candidates,
    build_query_history_recommendations,
    discover_best_warehouse_id,
)


class _FakeApiClient:
    def __init__(self):
        self.requests: list[tuple[str, str, dict | None]] = []

    def do(self, method: str, path: str, body: dict | None = None):
        self.requests.append((method, path, body))
        if method == "GET" and path == "/api/2.0/sql/warehouses":
            return {
                "warehouses": [
                    {"id": "wh-stopped", "state": "STOPPED"},
                    {"id": "wh-running", "state": "RUNNING"},
                ]
            }
        if method == "POST" and path == "/api/2.0/sql/statements":
            statement = str((body or {}).get("statement", ""))
            if "FROM system.query.history" in statement:
                return {
                    "status": {"state": "SUCCEEDED"},
                    "manifest": {
                        "schema": {
                            "columns": [
                                {"name": "statement_id"},
                                {"name": "genie_space_id"},
                                {"name": "execution_status"},
                                {"name": "statement_type"},
                                {"name": "executed_by"},
                                {"name": "start_time"},
                                {"name": "total_duration_ms"},
                                {"name": "error_message"},
                                {"name": "statement_text"},
                            ]
                        }
                    },
                    "result": {
                        "data_array": [
                            [
                                "s1",
                                "space-1",
                                "FINISHED",
                                "SELECT",
                                "user@example.com",
                                "2026-03-28T12:00:00Z",
                                1200,
                                None,
                                "SELECT brand_name, SUM(trx) FROM sales GROUP BY brand_name",
                            ],
                            [
                                "s2",
                                "space-1",
                                "FAILED",
                                None,
                                "user@example.com",
                                "2026-03-28T12:05:00Z",
                                800,
                                "parse error",
                                "SELECT brand_name, SUM(trx) FROM sales GROUP BY brand_name",
                            ],
                            [
                                "s3",
                                "space-1",
                                "FINISHED",
                                "SELECT",
                                "user@example.com",
                                "2026-03-28T12:10:00Z",
                                900,
                                None,
                                "SELECT market_name, SUM(nrx) FROM sales GROUP BY market_name",
                            ],
                        ]
                    },
                }
            if "FROM system.access.audit" in statement:
                return {
                    "status": {"state": "SUCCEEDED"},
                    "manifest": {
                        "schema": {
                            "columns": [
                                {"name": "feedback_rating"},
                                {"name": "feedback_count"},
                            ]
                        }
                    },
                    "result": {
                        "data_array": [
                            ["THUMBS_UP", 4],
                            ["THUMBS_DOWN", 1],
                        ]
                    },
                }
        raise AssertionError(f"Unexpected request: {method} {path}")


class _FakeWorkspaceClient:
    def __init__(self):
        self.api_client = _FakeApiClient()


def test_discover_best_warehouse_id_prefers_running():
    wc = _FakeWorkspaceClient()

    assert discover_best_warehouse_id(wc) == "wh-running"


def test_query_history_collector_summarizes_queries_and_feedback():
    wc = _FakeWorkspaceClient()
    collector = QueryHistoryCollector(wc, "wh-running")

    entries, summary = collector.collect(space_ids=["space-1"], days=30, limit=50)

    assert len(entries) == 3
    assert summary.status == "available"
    assert summary.query_count == 3
    assert summary.successful_query_count == 2
    assert summary.failed_query_count == 1
    assert summary.feedback_thumbs_up == 4
    assert summary.feedback_thumbs_down == 1
    assert summary.unique_query_family_count == 2
    assert summary.recurring_success_patterns[0].query_count >= 1
    assert summary.recurring_failure_patterns[0].failed_count == 1


def test_build_query_history_recommendations_filters_one_off_noise():
    wc = _FakeWorkspaceClient()
    collector = QueryHistoryCollector(wc, "wh-running")
    entries, summary = collector.collect(space_ids=["space-1"], days=30, limit=50)

    recommendations = build_query_history_recommendations(entries, summary)

    assert recommendations.status == "available"
    assert recommendations.repeated_query_families == 1
    assert recommendations.noisy_one_off_queries == 1
    assert recommendations.recommended_trusted_asset_candidates == []
    assert len(recommendations.recommended_benchmark_candidates) == 1
    assert recommendations.recommended_benchmark_candidates[0].query_count == 2


def test_build_query_history_recommendations_matches_existing_benchmark_family():
    wc = _FakeWorkspaceClient()
    collector = QueryHistoryCollector(wc, "wh-running")
    entries, summary = collector.collect(space_ids=["space-1"], days=30, limit=50)

    benchmark_questions = [
        BenchmarkQuestion(
            id="bench-1",
            question="Show TRx by brand",
            expected_sql="SELECT brand_name, SUM(trx) FROM sales GROUP BY brand_name",
        )
    ]

    recommendations = build_query_history_recommendations(
        entries,
        summary,
        benchmark_questions=benchmark_questions,
    )

    candidate = recommendations.recommended_benchmark_candidates[0]
    assert candidate.expansion_strategy == "expand_existing_family"
    assert candidate.matched_benchmark_ids == ["bench-1"]
    assert candidate.matched_benchmark_questions == ["Show TRx by brand"]
    assert candidate.supports_auto_question_expansion is False


def test_build_query_history_recommendations_handles_unavailable_summary():
    summary = QueryHistorySummary(
        status="unavailable",
        message="permission denied",
        warehouse_id=None,
        space_ids=["space-1"],
        days=30,
        query_count=0,
        successful_query_count=0,
        failed_query_count=0,
        unique_query_family_count=0,
        feedback_thumbs_up=0,
        feedback_thumbs_down=0,
        recurring_success_patterns=[],
        recurring_failure_patterns=[],
    )

    recommendations = build_query_history_recommendations([], summary)

    assert recommendations.status == "unavailable"
    assert recommendations.message == "permission denied"
    assert recommendations.recommended_trusted_asset_candidates == []
    assert recommendations.recommended_benchmark_candidates == []


# ---------------------------------------------------------------------------
# build_expansion_candidates tests
# ---------------------------------------------------------------------------

_SAMPLE_RECS_WITH_MIXED = {
    "recommended_benchmark_candidates": [
        {
            "sample_sql": "SELECT brand_name, SUM(trx) FROM sales GROUP BY brand_name",
            "matched_benchmark_ids": ["bench-1"],
            "matched_benchmark_questions": ["Show TRx by brand"],
            "query_count": 3,
            "last_seen_at": "2026-03-28T12:05:00Z",
        },
        {
            "sample_sql": "SELECT market_name, SUM(nrx) FROM sales GROUP BY market_name",
            "matched_benchmark_ids": [],
            "matched_benchmark_questions": [],
            "query_count": 2,
            "last_seen_at": "2026-03-28T12:10:00Z",
        },
    ]
}


def test_build_expansion_candidates_excludes_matched_benchmarks():
    result = build_expansion_candidates(_SAMPLE_RECS_WITH_MIXED, now_iso="2026-04-01T00:00:00+00:00")

    assert result["candidate_count"] == 1
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert "market_name" in candidate["expected_sql"]
    assert candidate["source"] == "query_history"


def test_build_expansion_candidates_deterministic_id():
    result_a = build_expansion_candidates(_SAMPLE_RECS_WITH_MIXED, now_iso="2026-04-01T00:00:00+00:00")
    result_b = build_expansion_candidates(_SAMPLE_RECS_WITH_MIXED, now_iso="2026-04-02T00:00:00+00:00")

    ids_a = [c["id"] for c in result_a["candidates"]]
    ids_b = [c["id"] for c in result_b["candidates"]]
    assert ids_a == ids_b


def test_build_expansion_candidates_empty_list_yields_zero_count():
    recs = {"recommended_benchmark_candidates": []}
    result = build_expansion_candidates(recs, now_iso="2026-04-01T00:00:00+00:00")

    assert result["candidate_count"] == 0
    assert result["candidates"] == []


def test_build_expansion_candidates_confidence_from_query_count():
    recs = {
        "recommended_benchmark_candidates": [
            {
                "sample_sql": "SELECT a FROM t",
                "matched_benchmark_ids": [],
                "matched_benchmark_questions": ["How many a?"],
                "query_count": 2,
                "last_seen_at": None,
            },
            {
                "sample_sql": "SELECT b FROM t",
                "matched_benchmark_ids": [],
                "matched_benchmark_questions": ["How many b?"],
                "query_count": 10,
                "last_seen_at": None,
            },
        ]
    }
    result = build_expansion_candidates(recs, now_iso="2026-04-01T00:00:00+00:00")

    assert result["candidate_count"] == 2
    confidences = {c["expected_sql"]: c["confidence"] for c in result["candidates"]}
    assert confidences["SELECT a FROM t"] == "medium"
    assert confidences["SELECT b FROM t"] == "high"
