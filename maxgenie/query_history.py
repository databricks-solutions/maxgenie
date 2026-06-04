"""Space-scoped query history collection for Genie workspaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import re
import time
from typing import Any

from databricks.sdk import WorkspaceClient

from maxgenie.benchmark_service import BenchmarkQuestion

POLL_INTERVAL = 2.0
MAX_POLL_ATTEMPTS = 60
INLINE_WAIT_TIMEOUT = "50s"


@dataclass(frozen=True)
class QueryHistoryEntry:
    """One query-history row relevant to the Genie space."""

    statement_id: str
    genie_space_id: str | None
    execution_status: str
    statement_type: str | None
    executed_by: str | None
    start_time: str | None
    total_duration_ms: int | None
    error_message: str | None
    statement_text: str
    normalized_sql: str


@dataclass(frozen=True)
class QueryHistoryPattern:
    """Recurring normalized query family."""

    normalized_sql: str
    query_count: int
    successful_count: int
    failed_count: int
    last_seen_at: str | None
    sample_sql: str


@dataclass(frozen=True)
class QueryHistorySummary:
    """Summary of collected query-history evidence."""

    status: str
    message: str
    warehouse_id: str | None
    space_ids: list[str]
    days: int
    query_count: int
    successful_query_count: int
    failed_query_count: int
    unique_query_family_count: int
    feedback_thumbs_up: int
    feedback_thumbs_down: int
    recurring_success_patterns: list[QueryHistoryPattern]
    recurring_failure_patterns: list[QueryHistoryPattern]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueryHistoryRecommendation:
    """Actionable recommendation derived from query history."""

    recommendation_type: str
    rationale: str
    expansion_strategy: str
    owner_action: str
    query_count: int
    sample_sql: str
    last_seen_at: str | None
    matched_benchmark_ids: list[str]
    matched_benchmark_questions: list[str]
    supports_auto_question_expansion: bool


@dataclass(frozen=True)
class QueryHistoryRecommendations:
    """Ranked recommendations derived from collected query history."""

    status: str
    message: str
    recommended_trusted_asset_candidates: list[QueryHistoryRecommendation]
    recommended_benchmark_candidates: list[QueryHistoryRecommendation]
    noisy_one_off_queries: int
    repeated_query_families: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_best_warehouse_id(wc: WorkspaceClient) -> str | None:
    """Choose a usable warehouse id from the current workspace."""
    try:
        response = wc.api_client.do("GET", "/api/2.0/sql/warehouses")
    except Exception:
        return None

    warehouses = response.get("warehouses", [])
    if not isinstance(warehouses, list):
        return None

    ranked: list[tuple[int, str]] = []
    for warehouse in warehouses:
        if not isinstance(warehouse, dict):
            continue
        warehouse_id = warehouse.get("id")
        if not warehouse_id:
            continue
        state = str(warehouse.get("state", "")).upper()
        if state == "RUNNING":
            rank = 0
        elif state == "STARTING":
            rank = 1
        elif state == "STOPPED":
            rank = 2
        else:
            rank = 3
        ranked.append((rank, str(warehouse_id)))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _normalize_sql_family(sql: str) -> str:
    normalized = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    normalized = re.sub(r"--[^\r\n]*", " ", normalized)
    normalized = normalized.lower()
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"\s+", " ", normalized).strip(" ;\n\r\t")
    return normalized


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class QueryHistoryCollector:
    """Collect and summarize space-scoped query history."""

    def __init__(self, wc: WorkspaceClient, warehouse_id: str):
        self.wc = wc
        self.warehouse_id = warehouse_id

    def _execute_sql(self, sql: str) -> list[dict[str, Any]]:
        response = self.wc.api_client.do(
            "POST",
            "/api/2.0/sql/statements",
            body={
                "warehouse_id": self.warehouse_id,
                "statement": sql,
                "wait_timeout": INLINE_WAIT_TIMEOUT,
                "disposition": "INLINE",
                "format": "JSON_ARRAY",
            },
        )

        status = response.get("status", {}).get("state", "")
        if status == "SUCCEEDED":
            return self._extract_rows(response)

        statement_id = response.get("statement_id")
        if not statement_id:
            raise RuntimeError(f"SQL execution failed: {response}")

        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL)
            poll = self.wc.api_client.do("GET", f"/api/2.0/sql/statements/{statement_id}")
            state = poll.get("status", {}).get("state", "")
            if state == "SUCCEEDED":
                return self._extract_rows(poll)
            if state in ("FAILED", "CANCELED", "CLOSED"):
                error = poll.get("status", {}).get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"SQL execution {state}: {error}")

        raise RuntimeError("SQL execution timed out")

    @staticmethod
    def _extract_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
        manifest = response.get("manifest", {})
        columns = [
            str(column.get("name"))
            for column in manifest.get("schema", {}).get("columns", [])
            if isinstance(column, dict) and column.get("name")
        ]
        rows = response.get("result", {}).get("data_array", [])
        if not isinstance(rows, list):
            return []

        payload: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            payload.append(
                {
                    column_name: row[index] if index < len(row) else None
                    for index, column_name in enumerate(columns)
                }
            )
        return payload

    def collect(
        self,
        *,
        space_ids: list[str],
        days: int = 30,
        limit: int = 200,
    ) -> tuple[list[QueryHistoryEntry], QueryHistorySummary]:
        sanitized_ids = [space_id for space_id in dict.fromkeys(space_ids) if space_id]
        if not sanitized_ids:
            return [], QueryHistorySummary(
                status="unavailable",
                message="No Genie space ids were provided for query-history collection.",
                warehouse_id=self.warehouse_id,
                space_ids=[],
                days=days,
                query_count=0,
                successful_query_count=0,
                failed_query_count=0,
                unique_query_family_count=0,
                feedback_thumbs_up=0,
                feedback_thumbs_down=0,
                recurring_success_patterns=[],
                recurring_failure_patterns=[],
            )

        try:
            entries = self._fetch_query_entries(space_ids=sanitized_ids, days=days, limit=limit)
            feedback = self._fetch_feedback(space_ids=sanitized_ids, days=days)
        except Exception as exc:
            return [], QueryHistorySummary(
                status="unavailable",
                message=f"Query history collection failed: {exc}",
                warehouse_id=self.warehouse_id,
                space_ids=sanitized_ids,
                days=days,
                query_count=0,
                successful_query_count=0,
                failed_query_count=0,
                unique_query_family_count=0,
                feedback_thumbs_up=0,
                feedback_thumbs_down=0,
                recurring_success_patterns=[],
                recurring_failure_patterns=[],
            )

        if not entries:
            return [], QueryHistorySummary(
                status="no_data",
                message="No recent Genie query-history rows were found for this space.",
                warehouse_id=self.warehouse_id,
                space_ids=sanitized_ids,
                days=days,
                query_count=0,
                successful_query_count=0,
                failed_query_count=0,
                unique_query_family_count=0,
                feedback_thumbs_up=feedback.get("THUMBS_UP", 0),
                feedback_thumbs_down=feedback.get("THUMBS_DOWN", 0),
                recurring_success_patterns=[],
                recurring_failure_patterns=[],
            )

        successful = [entry for entry in entries if entry.execution_status.upper() == "FINISHED"]
        failed = [entry for entry in entries if entry.execution_status.upper() != "FINISHED"]
        success_patterns = _aggregate_patterns(successful)
        failure_patterns = _aggregate_patterns(failed)
        summary = QueryHistorySummary(
            status="available",
            message="Collected space-scoped query history and feedback signals.",
            warehouse_id=self.warehouse_id,
            space_ids=sanitized_ids,
            days=days,
            query_count=len(entries),
            successful_query_count=len(successful),
            failed_query_count=len(failed),
            unique_query_family_count=len({entry.normalized_sql for entry in entries}),
            feedback_thumbs_up=feedback.get("THUMBS_UP", 0),
            feedback_thumbs_down=feedback.get("THUMBS_DOWN", 0),
            recurring_success_patterns=success_patterns[:10],
            recurring_failure_patterns=failure_patterns[:10],
        )
        return entries, summary

    def _fetch_query_entries(
        self,
        *,
        space_ids: list[str],
        days: int,
        limit: int,
    ) -> list[QueryHistoryEntry]:
        ids_sql = ", ".join(_sql_string_literal(space_id) for space_id in space_ids)
        query = f"""
SELECT
  statement_id,
  query_source.genie_space_id AS genie_space_id,
  execution_status,
  statement_type,
  executed_by,
  CAST(start_time AS STRING) AS start_time,
  total_duration_ms,
  error_message,
  statement_text
FROM system.query.history
WHERE start_time >= current_timestamp() - INTERVAL {max(days, 1)} DAYS
  AND query_source.genie_space_id IN ({ids_sql})
  AND statement_text IS NOT NULL
  AND (statement_type = 'SELECT' OR statement_type IS NULL)
ORDER BY start_time DESC
LIMIT {max(limit, 1)}
""".strip()
        rows = self._execute_sql(query)
        entries: list[QueryHistoryEntry] = []
        for row in rows:
            statement_text = str(row.get("statement_text") or "").strip()
            if not statement_text:
                continue
            entries.append(
                QueryHistoryEntry(
                    statement_id=str(row.get("statement_id") or ""),
                    genie_space_id=str(row.get("genie_space_id")) if row.get("genie_space_id") is not None else None,
                    execution_status=str(row.get("execution_status") or "UNKNOWN"),
                    statement_type=str(row.get("statement_type")) if row.get("statement_type") is not None else None,
                    executed_by=str(row.get("executed_by")) if row.get("executed_by") is not None else None,
                    start_time=str(row.get("start_time")) if row.get("start_time") is not None else None,
                    total_duration_ms=int(row["total_duration_ms"]) if row.get("total_duration_ms") is not None else None,
                    error_message=str(row.get("error_message")) if row.get("error_message") is not None else None,
                    statement_text=statement_text,
                    normalized_sql=_normalize_sql_family(statement_text),
                )
            )
        return entries

    def _fetch_feedback(self, *, space_ids: list[str], days: int) -> dict[str, int]:
        ids_sql = ", ".join(_sql_string_literal(space_id) for space_id in space_ids)
        query = f"""
SELECT
  request_params['feedback_rating'] AS feedback_rating,
  COUNT(*) AS feedback_count
FROM system.access.audit
WHERE event_date >= current_date() - {max(days, 1)}
  AND service_name = 'aibiGenie'
  AND action_name = 'updateConversationMessageFeedback'
  AND request_params['space_id'] IN ({ids_sql})
GROUP BY 1
""".strip()
        rows = self._execute_sql(query)
        counts: dict[str, int] = {}
        for row in rows:
            rating = str(row.get("feedback_rating") or "")
            if not rating:
                continue
            counts[rating] = int(row.get("feedback_count") or 0)
        return counts


def _aggregate_patterns(entries: list[QueryHistoryEntry]) -> list[QueryHistoryPattern]:
    grouped: dict[str, list[QueryHistoryEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.normalized_sql, []).append(entry)

    patterns: list[QueryHistoryPattern] = []
    for normalized_sql, rows in grouped.items():
        rows.sort(key=lambda item: item.start_time or "", reverse=True)
        successful_count = sum(1 for row in rows if row.execution_status.upper() == "FINISHED")
        failed_count = len(rows) - successful_count
        patterns.append(
            QueryHistoryPattern(
                normalized_sql=normalized_sql,
                query_count=len(rows),
                successful_count=successful_count,
                failed_count=failed_count,
                last_seen_at=rows[0].start_time,
                sample_sql=rows[0].statement_text,
            )
        )
    patterns.sort(
        key=lambda item: (
            item.query_count,
            item.successful_count - item.failed_count,
            item.last_seen_at or "",
        ),
        reverse=True,
    )
    return patterns


def _benchmark_family_index(
    benchmark_questions: list[BenchmarkQuestion] | None,
) -> dict[str, list[BenchmarkQuestion]]:
    index: dict[str, list[BenchmarkQuestion]] = {}
    if not benchmark_questions:
        return index

    for question in benchmark_questions:
        if not question.expected_sql:
            continue
        normalized_sql = _normalize_sql_family(question.expected_sql)
        if not normalized_sql:
            continue
        index.setdefault(normalized_sql, []).append(question)
    return index


def build_query_history_recommendations(
    entries: list[QueryHistoryEntry],
    summary: QueryHistorySummary,
    benchmark_questions: list[BenchmarkQuestion] | None = None,
) -> QueryHistoryRecommendations:
    """Convert raw history into conservative, actionable recommendations."""
    if summary.status != "available":
        return QueryHistoryRecommendations(
            status=summary.status,
            message=summary.message,
            recommended_trusted_asset_candidates=[],
            recommended_benchmark_candidates=[],
            noisy_one_off_queries=0,
            repeated_query_families=0,
        )

    benchmark_index = _benchmark_family_index(benchmark_questions)
    by_family: dict[str, list[QueryHistoryEntry]] = {}
    for entry in entries:
        by_family.setdefault(entry.normalized_sql, []).append(entry)

    trusted_asset_candidates: list[QueryHistoryRecommendation] = []
    benchmark_candidates: list[QueryHistoryRecommendation] = []
    noisy_one_off_queries = 0
    repeated_query_families = 0

    for rows in by_family.values():
        rows.sort(key=lambda item: item.start_time or "", reverse=True)
        total = len(rows)
        successes = sum(1 for row in rows if row.execution_status.upper() == "FINISHED")
        failures = total - successes
        if total == 1:
            noisy_one_off_queries += 1
            continue
        repeated_query_families += 1
        sample_sql = rows[0].statement_text
        last_seen_at = rows[0].start_time
        matched_benchmarks = benchmark_index.get(rows[0].normalized_sql, [])
        matched_benchmark_ids = [question.id for question in matched_benchmarks]
        matched_benchmark_questions = [question.question for question in matched_benchmarks[:3]]

        if successes >= 2 and failures == 0:
            if matched_benchmark_ids:
                expansion_strategy = "promote_existing_family"
                owner_action = (
                    "Review the matched benchmark family, then add two to four real user phrasings from the"
                    " Messages feed before promoting the pattern to a trusted asset or highly scoped example."
                )
                rationale = (
                    "Recurring successful SQL family already matches an existing benchmark family, so the next"
                    " step is broadening real user phrasing coverage rather than adding more synthetic examples."
                )
            else:
                expansion_strategy = "review_new_family"
                owner_action = (
                    "Confirm that this recurring successful SQL family reflects an in-scope business question,"
                    " then add a reviewed benchmark family or trusted asset instead of bulk-importing raw history."
                )
                rationale = (
                    "Recurring successful SQL family could be promoted to a trusted asset or curated example if"
                    " the business wording is stable."
                )
            trusted_asset_candidates.append(
                QueryHistoryRecommendation(
                    recommendation_type="trusted_asset_candidate",
                    rationale=rationale,
                    expansion_strategy=expansion_strategy,
                    owner_action=owner_action,
                    query_count=total,
                    sample_sql=sample_sql,
                    last_seen_at=last_seen_at,
                    matched_benchmark_ids=matched_benchmark_ids,
                    matched_benchmark_questions=matched_benchmark_questions,
                    supports_auto_question_expansion=False,
                )
            )
        elif failures >= 1:
            if matched_benchmark_ids:
                expansion_strategy = "expand_existing_family"
                owner_action = (
                    "Add two to four real user phrasings from the Messages feed to the matched benchmark family"
                    " and rerun the grouped validation folds before touching the final holdout."
                )
                rationale = (
                    "Repeated SQL family already maps to an existing benchmark family but still fails in real"
                    " traffic, so the family needs broader phrasing coverage instead of more benchmark-shaped"
                    " examples."
                )
            else:
                expansion_strategy = "add_new_family"
                owner_action = (
                    "Review the sample SQL against supported scope, then add a new benchmark family with two to"
                    " four real user phrasings and a reviewed SQL answer."
                )
                rationale = (
                    "Repeated SQL family shows at least one execution failure and should become a benchmark or"
                    " owner-reviewed failure cluster before adding more examples."
                )
            benchmark_candidates.append(
                QueryHistoryRecommendation(
                    recommendation_type="benchmark_candidate",
                    rationale=rationale,
                    expansion_strategy=expansion_strategy,
                    owner_action=owner_action,
                    query_count=total,
                    sample_sql=sample_sql,
                    last_seen_at=last_seen_at,
                    matched_benchmark_ids=matched_benchmark_ids,
                    matched_benchmark_questions=matched_benchmark_questions,
                    supports_auto_question_expansion=False,
                )
            )

    trusted_asset_candidates.sort(key=lambda item: (item.query_count, item.last_seen_at or ""), reverse=True)
    benchmark_candidates.sort(key=lambda item: (item.query_count, item.last_seen_at or ""), reverse=True)
    return QueryHistoryRecommendations(
        status="available",
        message="Derived conservative recommendations from repeated query-history families.",
        recommended_trusted_asset_candidates=trusted_asset_candidates[:10],
        recommended_benchmark_candidates=benchmark_candidates[:10],
        noisy_one_off_queries=noisy_one_off_queries,
        repeated_query_families=repeated_query_families,
    )


_HIGH_CONFIDENCE_QUERY_COUNT_THRESHOLD = 5


def build_expansion_candidates(
    recommendations: dict[str, Any],
    *,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Convert a recommendations dict to an expansion-candidates payload (pure).

    Accepts the already-deserialized JSON form of query_history_recommendations.json.
    Returns a dict suitable for writing as expansion_candidates.json.
    """
    generated_at = now_iso or datetime.now(timezone.utc).isoformat()
    raw_candidates: list[dict[str, Any]] = recommendations.get("recommended_benchmark_candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    candidates: list[dict[str, Any]] = []
    for entry in raw_candidates:
        if not isinstance(entry, dict):
            continue
        # Skip entries that already have a matched benchmark — they're already covered.
        matched_ids = entry.get("matched_benchmark_ids") or []
        if matched_ids:
            continue

        sample_sql: str = entry.get("sample_sql") or ""
        matched_questions: list[str] = entry.get("matched_benchmark_questions") or []
        first_phrasing: str = matched_questions[0] if matched_questions else _synthesize_question(sample_sql)

        normalized = _normalize_sql_family(sample_sql)
        candidate_id = hashlib.sha1(
            f"{normalized}|{first_phrasing}".encode()
        ).hexdigest()[:32]

        query_count: int = entry.get("query_count") or 0
        confidence = "high" if query_count >= _HIGH_CONFIDENCE_QUERY_COUNT_THRESHOLD else "medium"

        candidates.append(
            {
                "id": candidate_id,
                "question": first_phrasing,
                "expected_sql": sample_sql,
                "source": "query_history",
                "confidence": confidence,
                "last_seen_at": entry.get("last_seen_at"),
            }
        )

    return {
        "generated_at": generated_at,
        "source_recommendations": recommendations.get("_source_path", ""),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def _synthesize_question(sql: str) -> str:
    """Produce a minimal human-readable question label from a SQL snippet."""
    normalized = _normalize_sql_family(sql)
    return f"Review: {normalized[:120]}"
