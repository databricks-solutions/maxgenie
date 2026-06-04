"""Normalize externally observed Genie benchmark state."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from maxgenie.benchmark_service import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkStatus,
)


_QUESTION_ROWS_KEYS = (
    "questions",
    "curated_questions",
    "benchmarks",
    "benchmark_questions",
)
_RESULT_ROWS_KEYS = (
    "results",
    "benchmark_results",
    "evaluations",
    "failed_questions",
    "passed_questions",
)


def _as_dict_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("items", "rows", "questions", "results", "evaluations"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _find_rows(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _as_dict_rows(payload)
    if not isinstance(payload, dict):
        return []
    for key in keys:
        rows = _as_dict_rows(payload.get(key))
        if rows:
            return rows
    return []


def _find_all_rows(payload: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _as_dict_rows(payload)
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in keys:
        rows.extend(_as_dict_rows(payload.get(key)))
    return rows


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if isinstance(value, dict):
        for key in ("content", "text", "query", "sql"):
            nested = _normalize_content(value.get(key))
            if nested:
                return nested
        return None
    text = str(value)
    return text if text else None


def _extract_answer_sql(row: dict[str, Any]) -> str | None:
    expected = _first_value(
        row,
        (
            "expected_sql",
            "expectedSql",
            "expected_query",
            "expectedQuery",
            "reference_sql",
            "referenceSql",
            "answer_sql",
        ),
    )
    if expected is not None:
        return _normalize_content(expected)

    answers = row.get("answer") or row.get("answers")
    if isinstance(answers, dict):
        answers = [answers]
    if isinstance(answers, list):
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            answer_format = str(answer.get("format") or answer.get("type") or "").upper()
            if answer_format and "SQL" not in answer_format:
                continue
            sql = _normalize_content(answer.get("content") or answer.get("query") or answer.get("sql"))
            if sql:
                return sql
    return None


def normalize_observed_questions(payload: Any) -> list[BenchmarkQuestion]:
    """Normalize curated benchmark question payloads from workspace tools."""
    rows = _find_rows(payload, _QUESTION_ROWS_KEYS)
    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
    for row in rows:
        question_id = _first_value(row, ("id", "question_id", "questionId", "benchmark_id", "benchmarkId"))
        question_text = _first_value(row, ("question", "question_text", "questionText", "prompt", "content"))
        if not question_id or not question_text:
            continue
        normalized_id = str(question_id)
        if normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        question = question_text[0] if isinstance(question_text, list) and question_text else question_text
        questions.append(
            BenchmarkQuestion(
                id=normalized_id,
                question=str(question),
                expected_sql=_extract_answer_sql(row),
            )
        )
    return questions


def _status_from_row(row: dict[str, Any], score: float | None, match_threshold: float) -> BenchmarkStatus:
    assessment = _first_value(row, ("assessment", "eval_assessment", "evalAssessment"))
    if assessment is not None:
        normalized_assessment = str(assessment).lower()
        if normalized_assessment in {"good", "passed", "pass"}:
            return BenchmarkStatus.PASSED
        if normalized_assessment in {"bad", "failed", "fail", "needs_review", "needs review"}:
            return BenchmarkStatus.FAILED

    passed_value = _first_value(row, ("passed", "is_passed", "isPassed", "success"))
    if isinstance(passed_value, bool):
        return BenchmarkStatus.PASSED if passed_value else BenchmarkStatus.FAILED
    raw_status = _first_value(row, ("status", "result", "outcome", "evaluation_status", "evaluationStatus"))
    if raw_status is not None:
        status = str(raw_status).lower()
        if status in {"pass", "passed", "success", "succeeded"}:
            return BenchmarkStatus.PASSED
        if status in {"fail", "failed", "mismatch", "incorrect", "needs_review", "needs review"}:
            return BenchmarkStatus.FAILED
        if status in {"error", "errored", "cancelled", "canceled", "timeout"}:
            return BenchmarkStatus.ERROR
    if score is not None:
        return BenchmarkStatus.PASSED if score >= match_threshold else BenchmarkStatus.FAILED
    return BenchmarkStatus.ERROR


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_score_or_none(value: Any) -> float | None:
    score = _float_or_none(value)
    if score is None:
        return None
    if 1.0 < score <= 100.0:
        return score / 100.0
    return score


def _score_from_assessment(assessment: Any) -> float | None:
    if assessment is None:
        return None
    normalized = str(assessment).lower()
    if normalized in {"good", "passed", "pass"}:
        return 1.0
    if normalized in {"needs_review", "needs review"}:
        return 0.5
    if normalized in {"bad", "failed", "fail"}:
        return 0.0
    return None


def _extract_response_sql(value: Any) -> str | None:
    if isinstance(value, list):
        for row in value:
            if not isinstance(row, dict):
                continue
            response_type = str(row.get("response_type") or "").upper()
            if response_type and "SQL" not in response_type and "QUERY" not in response_type:
                continue
            response = _normalize_content(row.get("response"))
            if response:
                return response
        return None
    return _normalize_content(value)


def _api_get(wc: Any, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return wc.api_client.do("GET", path, query=params)


def _list_paginated(
    wc: Any,
    path: str,
    *,
    list_key: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {"page_size": page_size}
        if page_token:
            params["page_token"] = page_token
        payload = _api_get(wc, path, params)
        page_rows = payload.get(list_key, [])
        if isinstance(page_rows, list):
            rows.extend(row for row in page_rows if isinstance(row, dict))
        page_token = payload.get("next_page_token")
        if not page_token:
            return rows


def _latest_done_eval_run(
    wc: Any,
    *,
    space_id: str,
    expected_question_count: int | None,
) -> dict[str, Any]:
    eval_runs = _list_paginated(
        wc,
        f"/api/2.0/genie/spaces/{space_id}/eval-runs",
        list_key="eval_runs",
    )
    done_runs = [
        run
        for run in eval_runs
        if str(run.get("eval_run_status") or "").upper() == "DONE"
    ]
    if expected_question_count is not None:
        matching_runs = [
            run
            for run in done_runs
            if int(run.get("num_questions", -1)) == expected_question_count
        ]
        if not matching_runs:
            raise ValueError(
                "No completed Genie benchmark eval run matched the exported "
                f"{expected_question_count} question(s)."
            )
        done_runs = matching_runs
    if not done_runs:
        raise ValueError("No completed Genie benchmark eval run was found for this space.")
    return max(done_runs, key=lambda run: int(run.get("created_timestamp", 0) or 0))


def _eval_result_rows(
    wc: Any,
    *,
    space_id: str,
    eval_run_id: str,
) -> list[dict[str, Any]]:
    return _list_paginated(
        wc,
        f"/api/2.0/genie/spaces/{space_id}/eval-runs/{eval_run_id}/results",
        list_key="eval_results",
    )


def _eval_result_detail(
    wc: Any,
    *,
    space_id: str,
    eval_run_id: str,
    result_id: str,
) -> dict[str, Any]:
    return _api_get(
        wc,
        f"/api/2.0/genie/spaces/{space_id}/eval-runs/{eval_run_id}/results/{result_id}",
    )


def latest_eval_run_observed_summary(
    *,
    wc: Any,
    space_id: str,
    expected_question_count: int | None = None,
) -> dict[str, Any]:
    """Fetch row-level pass/total counts for the latest completed eval run."""
    eval_run = _latest_done_eval_run(
        wc,
        space_id=space_id,
        expected_question_count=expected_question_count,
    )
    eval_run_id = str(eval_run["eval_run_id"])
    rows = _eval_result_rows(wc, space_id=space_id, eval_run_id=eval_run_id)
    results: list[dict[str, Any]] = []
    assessment_counts: dict[str, int] = {}
    passed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        assessment = row.get("assessment") or row.get("status")
        status = str(assessment or row.get("eval_run_status") or "").lower()
        if assessment is not None:
            assessment_counts[status] = assessment_counts.get(status, 0) + 1
        score = _score_from_assessment(assessment)
        if (isinstance(score, (int, float)) and score >= 1.0) or status in {
            "good",
            "passed",
            "pass",
        }:
            passed += 1
        results.append(
            {
                "assessment": assessment,
                "status": status,
                "score": score,
                "result_id": row.get("result_id"),
                "eval_run_id": eval_run_id,
            }
        )

    total = len(results)
    return {
        "schema_version": 1,
        "source": "latest_genie_eval_run",
        "created_at": datetime.now().isoformat(),
        "eval_run": eval_run,
        "eval_run_id": eval_run_id,
        "assessment_counts": assessment_counts,
        "question_count": total,
        "result_count": total,
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) * 100.0 if total else None,
        "results": results,
    }


def latest_eval_run_observed_payload(
    *,
    wc: Any,
    space_id: str,
    expected_question_count: int | None = None,
) -> dict[str, Any]:
    """Fetch the latest completed Genie benchmark eval run as an observed seed payload."""
    eval_run = _latest_done_eval_run(
        wc,
        space_id=space_id,
        expected_question_count=expected_question_count,
    )
    eval_run_id = str(eval_run["eval_run_id"])
    rows = _eval_result_rows(wc, space_id=space_id, eval_run_id=eval_run_id)
    questions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    assessment_counts: dict[str, int] = {}
    for row in rows:
        result_id = row.get("result_id")
        detail = (
            _eval_result_detail(
                wc,
                space_id=space_id,
                eval_run_id=eval_run_id,
                result_id=str(result_id),
            )
            if result_id
            else {}
        )
        question_id = str(
            detail.get("benchmark_question_id")
            or row.get("benchmark_question_id")
            or row.get("question_id")
            or ""
        )
        if not question_id:
            continue
        question = str(detail.get("question") or row.get("question") or "")
        expected_sql = (
            _extract_response_sql(detail.get("expected_response"))
            or _normalize_content(row.get("benchmark_answer"))
        )
        generated_sql = _extract_response_sql(detail.get("actual_response"))
        assessment = detail.get("assessment") or row.get("assessment")
        if assessment is not None:
            normalized_assessment = str(assessment).lower()
            assessment_counts[normalized_assessment] = assessment_counts.get(normalized_assessment, 0) + 1
        questions.append(
            {
                "id": question_id,
                "question": question,
                "expected_sql": expected_sql,
            }
        )
        results.append(
            {
                "question_id": question_id,
                "question": question,
                "assessment": assessment,
                "status": str(assessment).lower() if assessment is not None else detail.get("eval_run_status"),
                "score": _score_from_assessment(assessment),
                "generated_sql": generated_sql,
                "expected_sql": expected_sql,
                "result_id": result_id,
                "eval_run_id": eval_run_id,
                "comparison_method": (
                    f"observed_assessment_{str(assessment).lower()}"
                    if assessment is not None
                    else "observed_eval_result"
                ),
                "comparison_details": (
                    f"Imported from Genie benchmark eval run {eval_run_id}"
                    + (f" with assessment {assessment}" if assessment is not None else "")
                ),
            }
        )
    return {
        "schema_version": 1,
        "source": "latest_genie_eval_run",
        "created_at": datetime.now().isoformat(),
        "eval_run": eval_run,
        "eval_run_id": eval_run_id,
        "assessment_counts": assessment_counts,
        "question_count": len(questions),
        "result_count": len(results),
        "questions": questions,
        "results": results,
    }


def write_latest_observed_benchmark_seed(
    *,
    wc: Any,
    space_id: str,
    path: str | Path,
    expected_question_count: int | None = None,
) -> Path:
    """Write a seed file from the latest completed Genie benchmark eval run."""
    payload = latest_eval_run_observed_payload(
        wc=wc,
        space_id=space_id,
        expected_question_count=expected_question_count,
    )
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def normalize_observed_results(
    payload: Any,
    *,
    match_threshold: float = 0.85,
) -> list[BenchmarkResult]:
    """Normalize benchmark result payloads from workspace tools."""
    rows = _find_all_rows(payload, _RESULT_ROWS_KEYS)
    results: list[BenchmarkResult] = []
    seen_ids: set[str] = set()
    for row in rows:
        question_id = _first_value(
            row,
            ("question_id", "questionId", "benchmark_id", "benchmarkId", "id"),
        )
        if not question_id:
            question = row.get("question")
            if isinstance(question, dict):
                question_id = _first_value(
                    question,
                    ("id", "question_id", "questionId", "benchmark_id", "benchmarkId"),
                )
        if not question_id:
            continue
        normalized_id = str(question_id)
        if normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)

        assessment = _first_value(row, ("assessment", "eval_assessment", "evalAssessment"))
        score = _normalized_score_or_none(
            _first_value(
                row,
                ("comparison_score", "comparisonScore", "score", "similarity", "similarity_score"),
            )
        )
        if score is None:
            score = _score_from_assessment(assessment)
        status = _status_from_row(row, score, match_threshold)
        question_text = _first_value(row, ("question", "question_text", "questionText", "prompt", "content"))
        if isinstance(question_text, dict):
            question_text = _first_value(question_text, ("question", "question_text", "questionText", "prompt", "content"))
        generated_sql = _normalize_content(
            _first_value(row, ("generated_sql", "generatedSql", "actual_sql", "actualSql", "sql", "query"))
        )
        results.append(
            BenchmarkResult(
                question_id=normalized_id,
                question=str(question_text or ""),
                status=status,
                generated_sql=generated_sql,
                expected_sql=_extract_answer_sql(row),
                error_message=_normalize_content(
                    _first_value(row, ("error_message", "errorMessage", "reason", "failure_reason", "failureReason", "details"))
                ),
                duration_seconds=_float_or_none(_first_value(row, ("duration_seconds", "durationSeconds", "duration"))) or 0.0,
                conversation_id=_normalize_content(_first_value(row, ("conversation_id", "conversationId"))),
                message_id=_normalize_content(_first_value(row, ("message_id", "messageId"))),
                comparison_score=score,
                comparison_method=(
                    _normalize_content(_first_value(row, ("comparison_method", "comparisonMethod", "method")))
                    or (
                        f"observed_assessment_{str(assessment).lower()}"
                        if assessment is not None
                        else None
                    )
                ),
                comparison_details=_normalize_content(
                    _first_value(row, ("comparison_details", "comparisonDetails", "details"))
                ),
            )
        )
    return results


def write_observed_benchmark_seed(
    *,
    curated_questions: Any,
    benchmark_results: Any,
    path: str | Path,
    source: str = "workspace_authoring_runtime",
    match_threshold: float = 0.85,
) -> Path:
    """Write a canonical observed benchmark seed file."""
    questions = normalize_observed_questions(curated_questions)
    results = normalize_observed_results(benchmark_results, match_threshold=match_threshold)
    payload = {
        "schema_version": 1,
        "source": source,
        "created_at": datetime.now().isoformat(),
        "match_threshold": match_threshold,
        "question_count": len(questions),
        "result_count": len(results),
        "questions": [
            {
                "id": question.id,
                "question": question.question,
                "expected_sql": question.expected_sql,
            }
            for question in questions
        ],
        "results": [
            {
                "question_id": result.question_id,
                "question": result.question,
                "status": result.status.value,
                "generated_sql": result.generated_sql,
                "expected_sql": result.expected_sql,
                "error_message": result.error_message,
                "duration_seconds": result.duration_seconds,
                "conversation_id": result.conversation_id,
                "message_id": result.message_id,
                "comparison_score": result.comparison_score,
                "comparison_method": result.comparison_method,
                "comparison_details": result.comparison_details,
            }
            for result in results
        ],
    }
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_observed_benchmark_run(
    path: str | Path,
    *,
    questions: list[BenchmarkQuestion],
    space_id: str,
    match_threshold: float = 0.85,
) -> tuple[BenchmarkRun, dict[str, Any]]:
    """Load observed benchmark results and align them to exported questions."""
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    observed_results = normalize_observed_results(payload, match_threshold=match_threshold)
    results_by_id = {result.question_id: result for result in observed_results}
    missing_ids = [question.id for question in questions if question.id not in results_by_id]
    if missing_ids:
        sample = ", ".join(missing_ids[:5])
        raise ValueError(
            "Observed benchmark seed does not contain results for all exported questions: "
            f"{sample}"
        )

    observed_questions = normalize_observed_questions(payload)
    expected_by_id = {
        question.id: question.expected_sql
        for question in observed_questions
        if question.expected_sql
    }
    run = BenchmarkRun(space_id=space_id)
    run.results = []
    for question in questions:
        source = results_by_id[question.id]
        result = BenchmarkResult(
            question_id=question.id,
            question=question.question,
            status=source.status,
            generated_sql=source.generated_sql,
            expected_sql=source.expected_sql or question.expected_sql or expected_by_id.get(question.id),
            error_message=source.error_message,
            duration_seconds=source.duration_seconds,
            conversation_id=source.conversation_id,
            message_id=source.message_id,
            comparison_score=source.comparison_score,
            comparison_method=source.comparison_method or "observed_benchmark",
            comparison_details=source.comparison_details,
        )
        run.results.append(result)
    run.completed_at = run.started_at
    return run, payload
