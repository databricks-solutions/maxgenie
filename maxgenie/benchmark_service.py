"""Benchmark service for programmatically running Genie Space evaluations.

Uses the Conversation API to run benchmark questions and compare results.
Rate limited to ~5 queries/min/workspace.
"""
import json
import logging
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

from databricks.sdk import WorkspaceClient

from maxgenie.result_comparator import (
    classify_comparator_error,
    classify_sql_error_text,
    is_sql_execution_failure_text,
)

logger = logging.getLogger(__name__)


class ResultComparatorRequiredError(RuntimeError):
    """Raised when result-based comparison is required but no comparator is wired.

    Result-based comparison is mandatory whenever a SQL warehouse is configured for
    a run. Silently degrading to string/token comparison produces a non-comparable
    score, so a missing comparator under that contract is a hard error that stops the
    run rather than a degraded mode it proceeds through.
    """


POLL_INTERVAL_SECONDS = 2
# Genie responses can legitimately take multiple minutes on large spaces.
# Keep the default patient enough for full benchmark runs instead of treating
# slower-but-healthy queries as infrastructure failures.
MAX_POLL_ATTEMPTS = 180  # 6 minutes max wait per question
MAX_TRANSIENT_PERMISSION_POLL_ERRORS = 6

RESULT_BASED_COMPARISON_METHODS = frozenset(
    {
        "result_exact",
        "result_match",
        "result_match_alias_tolerant",
        "result_column_mismatch",
        "result_row_mismatch",
        "result_mismatch",
        "result_execution_error",
        "result_unbound_parameters_error",
        "result_non_select_error",
        "result_no_sql_generated",
        # Content-based comparator methods (value-first scoring). They execute both
        # queries and compare result content, so they are result-based coverage.
        "result_match_content",
        "result_content_mismatch",
        "result_content_mandatory_missing",
    }
)

STRING_BASED_COMPARISON_METHODS = frozenset(
    {
        "high_token_match",
        "large_query_token_match",
        "similarity_match",
        "string_fallback_unbound_parameters",
        "string_fallback_non_select",
        # Legacy persisted names. These were produced by string fallback even
        # though they used a result_* prefix, so never count them as
        # result-based evidence.
        "result_unbound_parameters_fallback",
        "result_non_select_fallback",
    }
)


def _poll_sleep_seconds(attempt: int) -> float:
    """Use a light backoff to reduce API chatter on long-running queries."""
    if attempt < 5:
        return float(POLL_INTERVAL_SECONDS)
    if attempt < 20:
        return max(float(POLL_INTERVAL_SECONDS), 5.0)
    return max(float(POLL_INTERVAL_SECONDS), 10.0)


def _is_hard_auth_error(error: str) -> bool:
    lowered = error.lower()
    return "401" in lowered or "403" in lowered


def _is_permission_poll_error(error: str) -> bool:
    lowered = error.lower()
    return "permission" in lowered or "aclpath" in lowered


def _is_result_based_comparison_method(method: str | None) -> bool:
    return isinstance(method, str) and method in RESULT_BASED_COMPARISON_METHODS


def _is_string_based_comparison_method(method: str | None) -> bool:
    if not isinstance(method, str) or method == "no_expected_sql":
        return False
    if method in STRING_BASED_COMPARISON_METHODS:
        return True
    return not _is_result_based_comparison_method(method)


class BenchmarkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A benchmark question with optional ground-truth SQL."""
    id: str
    question: str
    expected_sql: str | None = None


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark question."""
    question_id: str
    question: str
    status: BenchmarkStatus
    generated_sql: str | None = None
    expected_sql: str | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    conversation_id: str | None = None
    message_id: str | None = None
    comparison_score: float | None = None
    comparison_method: str | None = None
    comparison_details: str | None = None


def _mark_no_sql_as_result_based_failure(
    *,
    result: BenchmarkResult,
    benchmark: BenchmarkQuestion,
    result_comparator: Any | None,
) -> bool:
    """Preserve result-based coverage when Genie produces no SQL."""
    if result_comparator is None:
        return False
    expected_sql = result.expected_sql or benchmark.expected_sql
    if not expected_sql:
        return False
    if result.generated_sql:
        return False
    if result.status != BenchmarkStatus.FAILED:
        return False
    error_message = str(result.error_message or "").lower()
    if "no sql generated" not in error_message:
        return False

    changed = (
        result.comparison_score != 0.0
        or result.comparison_method != "result_no_sql_generated"
        or result.comparison_details != "No SQL generated; scored as failed result-based comparison."
    )
    result.expected_sql = expected_sql
    result.comparison_score = 0.0
    result.comparison_method = "result_no_sql_generated"
    result.comparison_details = "No SQL generated; scored as failed result-based comparison."
    return changed


def _failure_error_text(message: Any) -> str:
    """Flatten a failed Genie message's error payload into searchable text."""
    if not isinstance(message, dict):
        return str(message or "")
    parts: list[str] = []
    err = message.get("error")
    if isinstance(err, dict):
        parts.extend(str(value) for value in err.values() if value not in (None, ""))
    elif err not in (None, ""):
        parts.append(str(err))
    for key in ("type", "status", "message"):
        value = message.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    return " ".join(parts)


def _mark_sql_execution_failure_as_result_based(
    *,
    result: BenchmarkResult,
    benchmark: BenchmarkQuestion,
    result_comparator: Any | None,
    error_text: str,
) -> bool:
    """Score a Genie message failure caused by broken generated SQL as a counted failure.

    Keeps result-based coverage complete so the greedy gate can reject a candidate whose
    generated SQL fails to execute, instead of dropping it as an uncounted runtime error.
    Genuine infra/transient failures (auth, timeout, space gone) are left untouched.
    """
    if result_comparator is None:
        return False
    expected_sql = result.expected_sql or benchmark.expected_sql
    if not expected_sql:
        return False
    if not is_sql_execution_failure_text(error_text):
        return False
    # Map the SQL error kind to a result-based failure method. `non_select` does not
    # arise on this path (its trigger phrase is not a SQL-execution marker, so the guard
    # above filters it out); it is kept for symmetry with classify_sql_error_text.
    method = {
        "unbound_parameters": "result_unbound_parameters_error",
        "non_select": "result_non_select_error",
    }.get(classify_sql_error_text(error_text), "result_execution_error")
    result.expected_sql = expected_sql
    result.comparison_score = 0.0
    result.comparison_method = method
    result.comparison_details = (
        "Genie message failed during SQL execution; scored as failed result-based comparison."
    )
    result.status = BenchmarkStatus.FAILED
    return True


@dataclass
class BenchmarkRun:
    """Complete benchmark run with all results."""
    space_id: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == BenchmarkStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == BenchmarkStatus.FAILED)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == BenchmarkStatus.ERROR)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    @property
    def scored_question_count(self) -> int:
        return sum(1 for r in self.results if r.comparison_score is not None)

    @property
    def average_comparison_score(self) -> float | None:
        scores = [float(r.comparison_score) for r in self.results if r.comparison_score is not None]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 4)

    @property
    def comparison_method_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            if not result.comparison_method:
                continue
            counts[result.comparison_method] = counts.get(result.comparison_method, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def result_based_question_count(self) -> int:
        return sum(
            1
            for result in self.results
            if _is_result_based_comparison_method(result.comparison_method)
        )

    @property
    def string_based_question_count(self) -> int:
        return sum(
            1
            for result in self.results
            if _is_string_based_comparison_method(result.comparison_method)
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRun":
        """Reconstruct a BenchmarkRun from saved JSON."""
        results = []
        for r in data.get("results", []):
            results.append(BenchmarkResult(
                question_id=r["question_id"],
                question=r["question"],
                status=BenchmarkStatus(r["status"]),
                generated_sql=r.get("generated_sql"),
                expected_sql=r.get("expected_sql"),
                error_message=r.get("error_message"),
                duration_seconds=r.get("duration_seconds", 0.0),
                comparison_score=r.get("comparison_score"),
                comparison_method=r.get("comparison_method"),
                comparison_details=r.get("comparison_details"),
            ))
        started = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now()
        completed = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        return cls(
            space_id=data["space_id"],
            started_at=started,
            completed_at=completed,
            results=results,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "pass_rate": round(self.pass_rate, 2),
                "scored_question_count": self.scored_question_count,
                "average_comparison_score": self.average_comparison_score,
                "result_based_question_count": self.result_based_question_count,
                "string_based_question_count": self.string_based_question_count,
                "comparison_method_counts": self.comparison_method_counts,
            },
            "results": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "status": r.status.value,
                    "generated_sql": r.generated_sql,
                    "expected_sql": r.expected_sql,
                    "comparison_score": round(r.comparison_score, 4)
                    if r.comparison_score is not None
                    else None,
                    "comparison_method": r.comparison_method,
                    "comparison_details": r.comparison_details,
                    "error_message": r.error_message,
                    "duration_seconds": round(r.duration_seconds, 2),
                }
                for r in self.results
            ],
        }


class BenchmarkService:
    """Service for running Genie Space benchmarks programmatically."""

    def __init__(self, wc: WorkspaceClient):
        self.wc = wc

    def _api_get(self, path: str, params: dict | None = None) -> dict:
        return self.wc.api_client.do("GET", path, query=params)

    def _api_post(self, path: str, body: dict) -> dict:
        return self.wc.api_client.do("POST", path, body=body)

    def start_conversation(self, space_id: str, question: str) -> tuple[str, str]:
        """Start a new conversation with a question. Returns (conversation_id, message_id)."""
        response = self._api_post(
            f"/api/2.0/genie/spaces/{space_id}/start-conversation",
            {"content": question},
        )
        return response["conversation_id"], response["message_id"]

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> dict:
        """Get message status and content."""
        return self._api_get(
            f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}"
        )

    def poll_until_complete(
        self, space_id: str, conversation_id: str, message_id: str
    ) -> tuple[bool, dict]:
        """Poll message until complete. Returns (success, message_data)."""
        last_error: Exception | None = None
        consecutive_errors = 0
        for attempt in range(MAX_POLL_ATTEMPTS):
            try:
                msg = self.get_message(space_id, conversation_id, message_id)
                consecutive_errors = 0
            except Exception as exc:
                last_error = exc
                consecutive_errors += 1
                error_str = str(exc)
                # Fail fast on permanent errors instead of retrying 180 times
                if "does not exist" in error_str or "not accessible" in error_str:
                    logger.error("Poll failed on permanent error (space gone): %s", exc)
                    return False, {"error": f"Space gone: {exc}"}
                if _is_hard_auth_error(error_str):
                    logger.error("Poll failed on hard auth error: %s", exc)
                    return False, {"error": f"Auth error: {exc}"}
                if (
                    _is_permission_poll_error(error_str)
                    and consecutive_errors >= MAX_TRANSIENT_PERMISSION_POLL_ERRORS
                ):
                    logger.error(
                        "Poll failed after %d permission/ACL errors. Last: %s",
                        consecutive_errors,
                        exc,
                    )
                    return False, {"error": f"Auth error after retries: {exc}"}
                if consecutive_errors % 10 == 0:
                    logger.warning(
                        "Poll attempt %d/%d: %d consecutive errors. Last: %s",
                        attempt + 1, MAX_POLL_ATTEMPTS, consecutive_errors, exc,
                    )
                if consecutive_errors >= 30:
                    logger.error(
                        "Poll giving up after %d consecutive errors. Last: %s", consecutive_errors, exc
                    )
                    return False, {"error": f"Too many consecutive errors ({consecutive_errors}): {exc}"}
                time.sleep(_poll_sleep_seconds(attempt))
                continue
            status = msg.get("status")

            if status == "COMPLETED":
                return True, msg
            elif status in ("FAILED", "CANCELLED"):
                return False, msg

            time.sleep(_poll_sleep_seconds(attempt))

        if last_error is not None:
            return False, {"error": f"Polling failed repeatedly after {MAX_POLL_ATTEMPTS} attempts: {last_error}"}
        return False, {"error": f"Polling timeout exceeded ({MAX_POLL_ATTEMPTS} attempts)"}

    def extract_generated_sql(self, message: dict) -> str | None:
        """Extract the generated SQL from a completed message."""
        attachments = message.get("attachments", [])
        for att in attachments:
            query = att.get("query", {})
            if query.get("query"):
                return query["query"]
        return None

    def _normalize_sql(self, sql: str, normalize_literals: bool) -> str:
        """Normalize SQL for robust text-based matching."""
        normalized = sql
        normalized = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)
        normalized = re.sub(r"--[^\r\n]*", " ", normalized)
        normalized = normalized.lower()

        if normalize_literals:
            normalized = re.sub(r"'(?:''|[^'])*'", "?", normalized)
            normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "?", normalized)

        normalized = normalized.replace("`", "")
        normalized = re.sub(r"\s+", " ", normalized).strip(" ;\n\r\t")
        return normalized

    def _tokenize_sql(self, sql: str) -> set[str]:
        """Tokenize normalized SQL for similarity scoring."""
        tokens = re.findall(r"[a-z_][a-z0-9_]*|\?|<=|>=|<>|!=|[=(),.*+-/]", sql)
        return set(tokens)

    def _compare_sql(
        self,
        generated_sql: str,
        expected_sql: str,
        match_threshold: float,
    ) -> tuple[bool, float, str, str]:
        """Compare generated SQL to expected SQL and return pass/fail + score."""
        generated_exact = self._normalize_sql(generated_sql, normalize_literals=False)
        expected_exact = self._normalize_sql(expected_sql, normalize_literals=False)

        if generated_exact == expected_exact:
            return True, 1.0, "exact_match", "Normalized SQL strings are identical."

        generated_shape = self._normalize_sql(generated_sql, normalize_literals=True)
        expected_shape = self._normalize_sql(expected_sql, normalize_literals=True)

        if generated_shape == expected_shape:
            return True, 1.0, "shape_match", "SQL structure matches after literal normalization."

        generated_tokens = self._tokenize_sql(generated_shape)
        expected_tokens = self._tokenize_sql(expected_shape)

        if not generated_tokens and not expected_tokens:
            return True, 1.0, "empty_match", "Both SQL token sets are empty after normalization."

        union_tokens = generated_tokens | expected_tokens
        token_jaccard = (
            len(generated_tokens & expected_tokens) / len(union_tokens) if union_tokens else 0.0
        )
        sequence_ratio = SequenceMatcher(None, generated_shape, expected_shape).ratio()
        # High token overlap with reasonable sequence alignment typically indicates
        # SQL equivalence despite formatting/ordering drift in large queries.
        if token_jaccard >= 0.95 and sequence_ratio >= 0.65:
            details = (
                f"token_jaccard={token_jaccard:.3f}, "
                f"sequence_ratio={sequence_ratio:.3f}, "
                "high_token_match_gate=true"
            )
            return True, 1.0, "high_token_match", details

        if (
            len(expected_shape) >= 1500
            and len(generated_shape) >= 1000
            and token_jaccard >= 0.75
            and sequence_ratio >= 0.18
        ):
            details = (
                f"token_jaccard={token_jaccard:.3f}, "
                f"sequence_ratio={sequence_ratio:.3f}, "
                "large_query_token_match_gate=true"
            )
            return True, 0.95, "large_query_token_match", details

        score = (token_jaccard + sequence_ratio) / 2
        passed = score >= match_threshold
        details = (
            f"token_jaccard={token_jaccard:.3f}, "
            f"sequence_ratio={sequence_ratio:.3f}, "
            f"threshold={match_threshold:.3f}"
        )
        return passed, score, "similarity_match", details

    def run_question(
        self,
        space_id: str,
        benchmark: BenchmarkQuestion,
        match_threshold: float = 0.85,
        result_comparator: Any | None = None,
        require_result_based: bool = False,
    ) -> BenchmarkResult:
        """Run a single benchmark question and return result.

        When ``require_result_based`` is set (a SQL warehouse is configured for the
        run), a missing ``result_comparator`` is a hard error: the method refuses to
        silently fall back to string/token comparison, which would produce a
        non-comparable score.
        """
        if require_result_based and result_comparator is None:
            raise ResultComparatorRequiredError(
                "Result-based comparison is required for this run (a SQL warehouse is "
                "configured) but no result comparator was provided for question "
                f"{benchmark.id}. Refusing to fall back to string/token comparison."
            )
        start_time = time.time()
        result = BenchmarkResult(
            question_id=benchmark.id,
            question=benchmark.question,
            status=BenchmarkStatus.RUNNING,
            expected_sql=benchmark.expected_sql,
        )

        try:
            conv_id, msg_id = self.start_conversation(space_id, benchmark.question)
            result.conversation_id = conv_id
            result.message_id = msg_id

            success, message = self.poll_until_complete(space_id, conv_id, msg_id)

            if not success:
                error_text = _failure_error_text(message)
                result.error_message = error_text or "Message failed or cancelled"
                # A Genie message can fail because the generated SQL itself could not
                # execute. Score that as a result-based failure so coverage stays
                # complete and the greedy gate can reject the candidate; leave genuine
                # infra/transient failures as uncounted errors.
                if not _mark_sql_execution_failure_as_result_based(
                    result=result,
                    benchmark=benchmark,
                    result_comparator=result_comparator,
                    error_text=error_text,
                ):
                    result.status = BenchmarkStatus.ERROR
            else:
                result.generated_sql = self.extract_generated_sql(message)

                if result.generated_sql:
                    if benchmark.expected_sql:
                        # Try result-based comparison first if comparator available
                        if result_comparator:
                            try:
                                is_match, score, method, details = result_comparator.compare(
                                    generated_sql=result.generated_sql,
                                    expected_sql=benchmark.expected_sql,
                                )
                            except Exception as comparator_exc:
                                logger.warning(
                                    "Result comparator failed for %s; scoring as result-based failure: %s",
                                    result.question_id, comparator_exc,
                                )
                                error_kind = classify_comparator_error(comparator_exc)
                                if error_kind == "unbound_parameters":
                                    method = "result_unbound_parameters_error"
                                elif error_kind == "non_select":
                                    method = "result_non_select_error"
                                else:
                                    method = "result_execution_error"
                                is_match = False
                                score = 0.0
                                details = f"Result comparison failed: {comparator_exc}"
                        else:
                            is_match, score, method, details = self._compare_sql(
                                generated_sql=result.generated_sql,
                                expected_sql=benchmark.expected_sql,
                                match_threshold=match_threshold,
                            )
                        result.comparison_score = score
                        result.comparison_method = method
                        result.comparison_details = details
                        result.status = BenchmarkStatus.PASSED if is_match else BenchmarkStatus.FAILED
                        if not is_match:
                            result.error_message = (
                                "Generated SQL did not match expected SQL "
                                f"({details})"
                            )
                    else:
                        result.comparison_method = "no_expected_sql"
                        result.status = BenchmarkStatus.PASSED
                else:
                    result.status = BenchmarkStatus.FAILED
                    result.error_message = "No SQL generated"
                    _mark_no_sql_as_result_based_failure(
                        result=result,
                        benchmark=benchmark,
                        result_comparator=result_comparator,
                    )

        except Exception as e:
            result.status = BenchmarkStatus.ERROR
            result.error_message = str(e)
            logger.exception(f"Error running benchmark question: {benchmark.question}")

        result.duration_seconds = time.time() - start_time
        return result

    def run_benchmarks(
        self,
        space_id: str,
        questions: list[BenchmarkQuestion],
        delay_between_questions: float = 12.0,  # ~5 queries/min limit
        fail_fast: bool = False,
        match_threshold: float = 0.85,
        cache: Any | None = None,
        config_hash: str | None = None,
        cache_save_path: str | Path | None = None,
        cache_save_every: int | None = 1,
        result_comparator: Any | None = None,
        min_pass_rate: float | None = None,
        require_result_based: bool = False,
    ) -> BenchmarkRun:
        """Run all benchmark questions and return results.

        Args:
            space_id: The Genie Space ID to run benchmarks against.
            questions: List of benchmark questions to run.
            delay_between_questions: Seconds to wait between questions (rate limiting).
            fail_fast: If True, stop on first failure/error and return partial results.
            match_threshold: Similarity threshold in [0.0, 1.0] for SQL comparison.
            cache: Optional BenchmarkCache for reusing results across runs.
            config_hash: Config hash for cache keying (required if cache is provided).
            cache_save_path: Optional path for persisting cache progress.
            cache_save_every: Persist cache after this many uncached benchmark
                results. Set to None or <= 0 to save only at the end.
            min_pass_rate: If set, abort early when the best possible pass rate
                (assuming all remaining questions pass) falls below this threshold.
        """
        if require_result_based and result_comparator is None:
            raise ResultComparatorRequiredError(
                "Result-based comparison is required for this run (a SQL warehouse is "
                "configured) but no result comparator was wired. Refusing to fall back "
                "to string/token comparison, which would produce a non-comparable score."
            )

        run = BenchmarkRun(space_id=space_id)
        total = len(questions)
        passed_so_far = 0
        uncached_results_since_save = 0

        for i, question in enumerate(questions):
            # Check cache first
            if cache is not None and config_hash is not None:
                cached = cache.get(question.id, config_hash)
                if cached is not None:
                    logger.info(f"Cache hit {i + 1}/{total}: {question.question[:50]}...")
                    _mark_no_sql_as_result_based_failure(
                        result=cached,
                        benchmark=question,
                        result_comparator=result_comparator,
                    )
                    run.results.append(cached)
                    if cached.status == BenchmarkStatus.PASSED:
                        passed_so_far += 1
                    continue

            logger.info(f"Running benchmark {i + 1}/{total}: {question.question[:50]}...")

            result = self.run_question(
                space_id=space_id,
                benchmark=question,
                match_threshold=match_threshold,
                result_comparator=result_comparator,
                require_result_based=require_result_based,
            )
            run.results.append(result)
            if result.status == BenchmarkStatus.PASSED:
                passed_so_far += 1

            # Store in cache
            if cache is not None and config_hash is not None:
                cache.put(question.id, config_hash, result)
                uncached_results_since_save += 1
                if (
                    cache_save_path is not None
                    and cache_save_every is not None
                    and cache_save_every > 0
                    and uncached_results_since_save >= cache_save_every
                ):
                    cache.save(cache_save_path)
                    uncached_results_since_save = 0

            logger.info(f"  Status: {result.status.value} ({result.duration_seconds:.1f}s)")

            # Fail-fast: stop on first failure/error
            if fail_fast and result.status in (BenchmarkStatus.FAILED, BenchmarkStatus.ERROR):
                logger.info(f"Fail-fast: stopping after {i + 1} questions due to {result.status.value}")
                break

            # Upper-bound check: even if all remaining questions pass,
            # can we still meet min_pass_rate?
            if min_pass_rate is not None and total > 0:
                remaining = total - (i + 1)
                best_possible_rate = (passed_so_far + remaining) / total * 100
                if best_possible_rate < min_pass_rate:
                    logger.info(
                        f"Early abort: best possible {best_possible_rate:.1f}% < "
                        f"min required {min_pass_rate:.1f}% after {i + 1}/{total} questions"
                    )
                    break

            # Rate limiting delay (skip after last question and cache hits)
            if i < len(questions) - 1:
                time.sleep(delay_between_questions)

        if (
            cache is not None
            and cache_save_path is not None
            and uncached_results_since_save > 0
        ):
            cache.save(cache_save_path)

        run.completed_at = datetime.now()
        return run

    def get_space_benchmarks(self, space_id: str) -> list[BenchmarkQuestion]:
        """Fetch benchmark questions from a Genie Space (if defined)."""
        response = self._api_get(
            f"/api/2.0/genie/spaces/{space_id}",
            params={"include_serialized_space": "true"},
        )

        serialized = json.loads(response.get("serialized_space", "{}"))
        benchmarks_data = serialized.get("benchmarks", {}).get("questions", [])

        questions = []
        for bm in benchmarks_data:
            # Handle question as list or string
            question_raw = bm.get("question", "")
            question = question_raw[0] if isinstance(question_raw, list) else question_raw

            # Extract expected SQL from answer if present
            expected_sql = None
            answers = bm.get("answer", [])
            for ans in answers:
                if ans.get("format") == "SQL":
                    content = ans.get("content", [])
                    expected_sql = "".join(content) if isinstance(content, list) else content
                    break

            questions.append(
                BenchmarkQuestion(
                    id=bm.get("id", ""),
                    question=question,
                    expected_sql=expected_sql,
                )
            )

        return questions


def get_benchmark_service(profile: str | None = None) -> BenchmarkService:
    """Factory function to create a BenchmarkService."""
    wc = WorkspaceClient(profile=profile)
    return BenchmarkService(wc)
