"""Build a clean failure context document from workspace artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_MAX_BASELINE_PASS_GUARDS_IN_CONTEXT = 8
_MAX_COMPACT_SQL_CHARS = 260
_MAX_SQL_DELTA_TOKENS = 8

_SQL_STOPWORDS = {
    "all",
    "and",
    "as",
    "asc",
    "between",
    "by",
    "case",
    "cast",
    "desc",
    "distinct",
    "else",
    "end",
    "false",
    "from",
    "group",
    "having",
    "in",
    "inner",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "over",
    "right",
    "select",
    "then",
    "true",
    "when",
    "where",
    "with",
}

_COMPARISON_TERMS = {
    "change",
    "compare",
    "comparison",
    "difference",
    "growth",
    "percent",
    "percentage",
    "prior",
    "ratio",
    "share",
    "versus",
    "vs",
}

_TIME_TERMS = {
    "current",
    "date",
    "month",
    "monthly",
    "period",
    "previous",
    "prior",
    "quarter",
    "recent",
    "time",
    "week",
    "year",
}

_FAMILY_HINTS = {
    "runtime_or_generation_error": (
        "Generated SQL did not complete; prefer fixing invalid references or unsupported syntax "
        "before adding examples."
    ),
    "missing_generated_sql": (
        "No generated SQL was captured; prefer no_change unless diagnostics show a clear asset issue."
    ),
    "percentage_or_comparison_metric": (
        "Check numerator, denominator, percent scaling, comparator filters, and current/prior windows."
    ),
    "time_filter_mismatch": "Check date grain, current/prior period filters, and range literals.",
    "literal_or_parameter_mismatch": (
        "Expected SQL uses concrete visible literals while generated SQL uses placeholders or "
        "different entity values."
    ),
    "aggregation_or_grouping_mismatch": (
        "Check aggregate functions, selected dimensions, GROUP BY clauses, and measure definitions."
    ),
    "table_or_join_path_mismatch": (
        "Check table selection and join path before changing examples or text instructions."
    ),
    "selected_metric_or_column_mismatch": "Check selected columns, measure aliases, and metric definitions.",
    "general_sql_mismatch": "Compare the compact SQL deltas and target the smallest shared visible pattern.",
}

_SUGGESTED_OPERATION_FAMILIES_BY_ROOT_CAUSE = {
    "runtime_or_generation_error": (
        "set_column_config",
        "add_sql_snippet",
        "add_join_spec",
        "no_change",
    ),
    "missing_generated_sql": (
        "no_change",
        "append_text_instruction",
    ),
    "percentage_or_comparison_metric": (
        "add_example_sql",
        "add_sql_snippet",
        "append_text_instruction",
    ),
    "time_filter_mismatch": (
        "add_example_sql",
        "add_sql_snippet",
        "set_column_config",
    ),
    "literal_or_parameter_mismatch": (
        "add_example_sql",
        "set_column_config",
    ),
    "aggregation_or_grouping_mismatch": (
        "set_column_config",
        "add_sql_snippet",
        "add_example_sql",
    ),
    "table_or_join_path_mismatch": (
        "add_join_spec",
        "set_table_description",
        "set_column_config",
    ),
    "selected_metric_or_column_mismatch": (
        "set_column_config",
        "add_sql_snippet",
        "add_example_sql",
    ),
    "general_sql_mismatch": (
        "add_example_sql",
        "set_column_config",
        "no_change",
    ),
}

_ROOT_CAUSE_CONFIDENCE = {
    "runtime_or_generation_error": 0.9,
    "missing_generated_sql": 0.85,
    "percentage_or_comparison_metric": 0.75,
    "time_filter_mismatch": 0.7,
    "literal_or_parameter_mismatch": 0.8,
    "aggregation_or_grouping_mismatch": 0.7,
    "table_or_join_path_mismatch": 0.75,
    "selected_metric_or_column_mismatch": 0.65,
    "general_sql_mismatch": 0.45,
}

_STRUCTURAL_ROOT_CAUSES = {
    "aggregation_or_grouping_mismatch",
    "table_or_join_path_mismatch",
    "selected_metric_or_column_mismatch",
}


@dataclass(frozen=True)
class StructuredFailureQuestion:
    """JSON-ready failure evidence for one visible benchmark question."""

    question_id: str
    question: str | None
    observed_status: str
    passed: bool
    score: float | None
    issue_class: str
    expected_sql: str | None
    generated_sql: str | None
    likely_root_cause_family: str
    suggested_operation_families: tuple[str, ...]
    confidence: float
    guard_regression_risk: str
    guard_regression_risk_reasons: tuple[str, ...]
    comparison_method: str | None = None
    comparison_details: str | None = None
    error_message: str | None = None
    missing_expected_tokens: tuple[str, ...] = ()
    extra_generated_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredPassGuardQuestion:
    """JSON-ready pass-guard evidence safe to expose to candidate creation."""

    question_id: str
    question: str | None
    expected_sql: str | None
    source: str
    latest_status: str
    guard_reason: str
    accepted_candidate: str | None = None


@dataclass(frozen=True)
class StructuredFailureContext:
    """Structured companion for the markdown failure context."""

    schema_version: int
    iteration_label: str
    train_pass_rate: float | None
    best_train_rate: float | None
    best_validation_rate: float | None
    train_question_count: int
    failed_question_count: int
    failures: tuple[StructuredFailureQuestion, ...]
    pass_guards: tuple[StructuredPassGuardQuestion, ...]
    failure_families: dict[str, list[str]]
    hidden_holdout_policy: str
    # question_id -> [{"label", "effect": "fixed"|"regressed", "accepted"}]; train-visible only.
    per_question_attribution: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class _FailureEvidence:
    question_id: str
    question: str
    status: str
    score: Any
    method: str
    family: str
    expected_sql: str
    generated_sql: str
    missing_tokens: tuple[str, ...]
    extra_tokens: tuple[str, ...]
    error_message: str | None


@dataclass(frozen=True)
class _IterationLogFacts:
    attempts: list[dict[str, Any]]
    invalid_proposals: list[dict[str, Any]]
    train_diffs_by_label: dict[str, dict[str, Any]]


def _truncate(text: str | None, limit: int = 2000) -> str:
    if not text:
        return "(empty)"
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated, {len(text)} chars total)"


def _format_sql(sql: str | None, *, limit: int = 2000) -> str:
    if not sql:
        return "(no SQL)"
    return _truncate(sql.strip(), limit)


def _compact_text(text: str | None, *, limit: int) -> str:
    if not text:
        return "(empty)"
    compact = re.sub(r"\s+", " ", str(text).strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 18].rstrip() + " ... (truncated)"


def _inline_code(text: str | None, *, limit: int) -> str:
    compact = _compact_text(text, limit=limit).replace("`", "'")
    return f"`{compact}`"


def _compact_sql(sql: str | None) -> str:
    if not sql:
        return "(no SQL)"
    return _compact_text(sql, limit=_MAX_COMPACT_SQL_CHARS)


def _sql_tokens(sql: str | None) -> tuple[str, ...]:
    if not sql:
        return ()
    without_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_comments = re.sub(r"--[^\r\n]*", " ", without_comments)
    raw_tokens = re.findall(
        r":[A-Za-z_][\w$]*|'[^']{1,80}'|\"[^\"]{1,80}\"|[A-Za-z_][\w$]*|\d+(?:\.\d+)?",
        without_comments.lower(),
    )
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_token in raw_tokens:
        token = raw_token.strip()
        if not token or token in _SQL_STOPWORDS:
            continue
        if len(token) <= 1 and not token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tuple(tokens)


def _sql_token_delta(expected_sql: str | None, generated_sql: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_tokens = _sql_tokens(expected_sql)
    generated_tokens = _sql_tokens(generated_sql)
    expected_set = set(expected_tokens)
    generated_set = set(generated_tokens)
    missing = tuple(token for token in expected_tokens if token not in generated_set)
    extra = tuple(token for token in generated_tokens if token not in expected_set)
    return missing[:_MAX_SQL_DELTA_TOKENS], extra[:_MAX_SQL_DELTA_TOKENS]


def _load_iteration_log_facts(iteration_log_path: Path) -> _IterationLogFacts:
    attempts: list[dict[str, Any]] = []
    invalid_proposals: list[dict[str, Any]] = []
    train_diffs_by_label: dict[str, dict[str, Any]] = {}
    if not iteration_log_path.exists():
        return _IterationLogFacts(
            attempts=attempts,
            invalid_proposals=invalid_proposals,
            train_diffs_by_label=train_diffs_by_label,
        )

    for line in iteration_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("phase") == "candidate_evaluated":
            attempts.append(entry)
        elif entry.get("phase") == "serving_candidate_proposal_invalid":
            invalid_proposals.append(entry)
        elif entry.get("phase") == "benchmark_train":
            label = entry.get("label")
            diff = entry.get("diff")
            if isinstance(label, str) and isinstance(diff, dict):
                train_diffs_by_label[label] = diff
    return _IterationLogFacts(
        attempts=attempts,
        invalid_proposals=invalid_proposals,
        train_diffs_by_label=train_diffs_by_label,
    )


def _per_question_edit_attribution(
    attempts: list[dict[str, Any]],
    train_diffs_by_label: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Map each visible (train) question id to the edits that flipped its pass/fail state.

    Returns ``{question_id: [{"label", "effect", "accepted"}]}`` in round order, where
    ``effect`` is ``"fixed"`` (fail -> pass) or ``"regressed"`` (pass -> fail) for each
    candidate label that moved that question. Train-diff ids are train-visible by
    construction, so no hidden holdout id can appear here.
    """
    accepted_by_label: dict[str, bool] = {}
    for attempt in attempts:
        label = attempt.get("label")
        if isinstance(label, str) and label:
            accepted_by_label[label] = attempt.get("outcome") == "accepted"

    attribution: dict[str, list[dict[str, Any]]] = {}
    for diff_key, diff in train_diffs_by_label.items():
        if not isinstance(diff_key, str) or not diff_key.endswith(".train"):
            continue
        if not isinstance(diff, dict):
            continue
        label = diff_key[: -len(".train")]
        if label == "baseline":
            continue
        accepted = accepted_by_label.get(label, False)
        for effect in ("fixed", "regressed"):
            ids = diff.get(effect)
            if not isinstance(ids, list):
                continue
            for raw_id in ids:
                question_id = str(raw_id)
                if not question_id:
                    continue
                attribution.setdefault(question_id, []).append(
                    {"label": label, "effect": effect, "accepted": accepted}
                )
    return attribution


def _sql_clause(sql: str | None, clause: str) -> str:
    if not sql:
        return ""
    normalized = re.sub(r"\s+", " ", sql.lower())
    if clause == "select":
        match = re.search(r"\bselect\b(.*?)(\bfrom\b|$)", normalized)
    elif clause == "from":
        match = re.search(r"\bfrom\b(.*?)(\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", normalized)
    elif clause == "where":
        match = re.search(r"\bwhere\b(.*?)(\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", normalized)
    elif clause == "group_by":
        match = re.search(r"\bgroup\s+by\b(.*?)(\border\s+by\b|\blimit\b|$)", normalized)
    else:
        match = None
    return match.group(1).strip() if match else ""


def _aggregate_functions(sql: str | None) -> tuple[str, ...]:
    if not sql:
        return ()
    return tuple(dict.fromkeys(re.findall(r"\b(avg|count|sum|min|max|median)\s*\(", sql.lower())))


def _has_placeholder(sql: str | None) -> bool:
    return bool(sql and re.search(r":[A-Za-z_][\w$]*|\$\{[^}]+\}|\?", sql))


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _classify_failure_family(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").lower()
    method = str(result.get("comparison_method") or "").lower()
    question = str(result.get("question") or "")
    expected_sql = str(result.get("expected_sql") or "")
    generated_sql = str(result.get("generated_sql") or "")
    error_message = str(result.get("error_message") or "")
    combined = " ".join([question, expected_sql, generated_sql, error_message]).lower()

    if not generated_sql.strip():
        return "missing_generated_sql"
    if "unbound parameter" in combined or (
        _has_placeholder(generated_sql) and not _has_placeholder(expected_sql)
    ):
        return "literal_or_parameter_mismatch"
    if "column mismatch" in combined or "result_column_mismatch" in method:
        return "selected_metric_or_column_mismatch"
    if _contains_any(combined, _COMPARISON_TERMS) or re.search(r"[%/]", expected_sql):
        return "percentage_or_comparison_metric"
    if (
        _contains_any(combined, _TIME_TERMS)
        and _sql_clause(expected_sql, "where") != _sql_clause(generated_sql, "where")
    ):
        return "time_filter_mismatch"
    if status == "error" or error_message:
        return "runtime_or_generation_error"
    if (
        _sql_clause(expected_sql, "group_by") != _sql_clause(generated_sql, "group_by")
        or _aggregate_functions(expected_sql) != _aggregate_functions(generated_sql)
    ):
        return "aggregation_or_grouping_mismatch"
    if _sql_clause(expected_sql, "from") != _sql_clause(generated_sql, "from"):
        return "table_or_join_path_mismatch"
    if _sql_clause(expected_sql, "select") != _sql_clause(generated_sql, "select"):
        return "selected_metric_or_column_mismatch"
    return "general_sql_mismatch"


def _failure_evidence(result: dict[str, Any]) -> _FailureEvidence:
    expected_sql = str(result.get("expected_sql") or "")
    generated_sql = str(result.get("generated_sql") or "")
    missing_tokens, extra_tokens = _sql_token_delta(expected_sql, generated_sql)
    return _FailureEvidence(
        question_id=str(result.get("question_id") or "?"),
        question=str(result.get("question") or "(unknown question)"),
        status=str(result.get("status") or "unknown"),
        score=result.get("comparison_score"),
        method=str(result.get("comparison_method") or "unknown"),
        family=_classify_failure_family(result),
        expected_sql=_compact_sql(expected_sql),
        generated_sql=_compact_sql(generated_sql),
        missing_tokens=missing_tokens,
        extra_tokens=extra_tokens,
        error_message=(
            str(result.get("error_message"))
            if result.get("error_message")
            else None
        ),
    )


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _issue_class(result: dict[str, Any]) -> str:
    status = str(result.get("status") or "").lower()
    method = str(result.get("comparison_method") or "").lower()
    generated_sql = str(result.get("generated_sql") or "").strip()
    error_message = str(result.get("error_message") or "").lower()
    score = _numeric_score(result.get("comparison_score"))

    if not generated_sql:
        return "missing_generated_sql"
    if "unbound parameter" in error_message or "result_unbound_parameters" in method:
        return "unbound_parameters"
    if "column mismatch" in error_message or "result_column_mismatch" in method:
        return "column_mismatch"
    if "non-select" in error_message or "non_select" in method:
        return "non_select_sql"
    if "syntax" in error_message or "parse" in error_message:
        return "sql_syntax_error"
    if status == "error" or result.get("error_message"):
        return "error"
    if method and "result" in method:
        return "result_mismatch"
    if score is not None and score < 0.2:
        return "severe_sql_mismatch"
    if method and ("sql" in method or "similarity" in method or "exact" in method):
        return "sql_mismatch"
    return "observed_failure"


def _guard_regression_risk(
    *,
    family: str,
    score: float | None,
    pass_guard_count: int,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if pass_guard_count == 0:
        return "low", ("no_visible_pass_guards_available",)

    if family in _STRUCTURAL_ROOT_CAUSES:
        reasons.append("structural_change_may_affect_pass_guards")
    if family == "runtime_or_generation_error":
        reasons.append("execution_fix_may_require_broad_asset_change")
    if score is not None and score >= 0.6:
        reasons.append("near_miss_failure_has_regression_risk")
    if family in {
        "percentage_or_comparison_metric",
        "time_filter_mismatch",
        "literal_or_parameter_mismatch",
    }:
        reasons.append("prefer_localized_example_or_snippet")
    if not reasons:
        reasons.append("visible_pass_guards_must_be_preserved")

    if any(
        reason in reasons
        for reason in (
            "structural_change_may_affect_pass_guards",
            "execution_fix_may_require_broad_asset_change",
        )
    ):
        return "high", tuple(reasons)
    return "medium", tuple(reasons)


def _structured_failure_question(
    result: dict[str, Any],
    *,
    pass_guard_count: int,
    question_payload_by_id: dict[str, dict[str, Any]],
) -> StructuredFailureQuestion:
    question_id = str(result.get("question_id") or "?")
    payload = question_payload_by_id.get(question_id, {})
    expected_sql = result.get("expected_sql") or payload.get("expected_sql")
    generated_sql = result.get("generated_sql")
    family = _classify_failure_family(result)
    score = _numeric_score(result.get("comparison_score"))
    missing_tokens, extra_tokens = _sql_token_delta(
        str(expected_sql or ""),
        str(generated_sql or ""),
    )
    risk, risk_reasons = _guard_regression_risk(
        family=family,
        score=score,
        pass_guard_count=pass_guard_count,
    )
    return StructuredFailureQuestion(
        question_id=question_id,
        question=(
            str(result.get("question") or payload.get("question"))
            if result.get("question") or payload.get("question")
            else None
        ),
        observed_status=str(result.get("status") or "unknown").lower(),
        passed=str(result.get("status") or "").lower() == "passed",
        score=score,
        issue_class=_issue_class(result),
        expected_sql=str(expected_sql) if expected_sql is not None else None,
        generated_sql=str(generated_sql) if generated_sql is not None else None,
        likely_root_cause_family=family,
        suggested_operation_families=_SUGGESTED_OPERATION_FAMILIES_BY_ROOT_CAUSE.get(
            family,
            _SUGGESTED_OPERATION_FAMILIES_BY_ROOT_CAUSE["general_sql_mismatch"],
        ),
        confidence=_ROOT_CAUSE_CONFIDENCE.get(family, _ROOT_CAUSE_CONFIDENCE["general_sql_mismatch"]),
        guard_regression_risk=risk,
        guard_regression_risk_reasons=risk_reasons,
        comparison_method=(
            str(result.get("comparison_method"))
            if result.get("comparison_method") is not None
            else None
        ),
        comparison_details=(
            str(result.get("comparison_details"))
            if result.get("comparison_details") is not None
            else None
        ),
        error_message=(
            str(result.get("error_message"))
            if result.get("error_message")
            else None
        ),
        missing_expected_tokens=missing_tokens,
        extra_generated_tokens=extra_tokens,
    )


def _format_token_list(tokens: tuple[str, ...]) -> str:
    if not tokens:
        return "(none)"
    return ", ".join(_inline_code(token, limit=80) for token in tokens)


def _format_compact_failure_evidence(failures: list[_FailureEvidence]) -> list[str]:
    if not failures:
        return []
    lines: list[str] = [
        "## Compact Failure Evidence",
        "",
        (
            "Train-visible failures only. Visible validation appears only as pass guards; "
            "hidden final holdout content is intentionally omitted."
        ),
        "",
    ]
    for failure in failures:
        score_text = f", score={failure.score}" if failure.score is not None else ""
        lines.append(
            f"- `{failure.question_id}` [{failure.family}] status={failure.status}, "
            f"method={failure.method}{score_text}: {_truncate(failure.question, 160)}"
        )
        if failure.error_message:
            lines.append(f"  - Error: {_truncate(failure.error_message, 220)}")
        lines.append(f"  - Expected compact SQL: {_inline_code(failure.expected_sql, limit=_MAX_COMPACT_SQL_CHARS)}")
        lines.append(f"  - Generated compact SQL: {_inline_code(failure.generated_sql, limit=_MAX_COMPACT_SQL_CHARS)}")
        lines.append(f"  - Missing expected tokens: {_format_token_list(failure.missing_tokens)}")
        lines.append(f"  - Extra generated tokens: {_format_token_list(failure.extra_tokens)}")
    lines.append("")
    return lines


def _format_failure_families(failures: list[_FailureEvidence]) -> list[str]:
    if not failures:
        return []
    grouped: dict[str, list[_FailureEvidence]] = {}
    for failure in failures:
        grouped.setdefault(failure.family, []).append(failure)

    lines: list[str] = [
        "## Failure Families",
        "",
        "Use these train-visible groups to target one shared root cause per serving candidate.",
        "",
    ]
    for family, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        ids = ", ".join(f"`{failure.question_id}`" for failure in members)
        hint = _FAMILY_HINTS.get(family, _FAMILY_HINTS["general_sql_mismatch"])
        lines.append(f"- `{family}` ({len(members)}): {ids}")
        lines.append(f"  - Evidence hint: {hint}")
    lines.append("")
    return lines


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _result_by_question_id(data: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for result in data.get("results", []):
        if not isinstance(result, dict):
            continue
        question_id = result.get("question_id")
        if question_id:
            output[str(question_id)] = result
    return output


def _question_payload_by_id(*, results_dir: Path, benchmarks_dir: Path | None) -> dict[str, dict[str, Any]]:
    payload_paths: list[Path] = [results_dir.parent / ".benchmark_payload.json"]
    if benchmarks_dir is not None:
        payload_paths.append(benchmarks_dir / "visible_pool.json")
    output: dict[str, dict[str, Any]] = {}
    for payload_path in payload_paths:
        payload = _load_json(payload_path)
        if not isinstance(payload, dict):
            continue
        questions = payload.get("questions")
        if not isinstance(questions, list):
            continue
        for question in questions:
            if not isinstance(question, dict):
                continue
            question_id = question.get("id") or question.get("question_id")
            if question_id:
                output[str(question_id)] = question
    return output


def _validation_status_by_id(summary: Any) -> dict[str, str]:
    if not isinstance(summary, dict):
        return {}
    folds = summary.get("folds")
    output: dict[str, str] = {}
    if isinstance(folds, list) and folds:
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            failed_ids = {str(value) for value in fold.get("failed_ids", [])}
            error_ids = {str(value) for value in fold.get("error_ids", [])}
            for raw_question_id in fold.get("validation_ids", []):
                question_id = str(raw_question_id)
                if question_id in error_ids:
                    output[question_id] = "error"
                elif question_id in failed_ids:
                    output[question_id] = "failed"
                else:
                    output[question_id] = "passed"
        return output

    failed_ids = {str(value) for value in summary.get("failed_ids", [])}
    error_ids = {str(value) for value in summary.get("error_ids", [])}
    question_ids = [str(value) for value in summary.get("validation_ids", []) or summary.get("ids", [])]
    for question_id in question_ids:
        if question_id in error_ids:
            output[question_id] = "error"
        elif question_id in failed_ids:
            output[question_id] = "failed"
        else:
            output[question_id] = "passed"
    return output


def _format_id_list(values: Any, *, limit: int = 6) -> str | None:
    if not isinstance(values, list) or not values:
        return None
    return ", ".join(f"`{value}`" for value in values[:limit])


def _operation_summary(operation: dict[str, Any]) -> str:
    path = operation.get("path") or operation.get("relative_path")
    op = str(operation.get("op") or ("file_edit" if path else operation.get("reason") or "unknown"))
    question_id = operation.get("id")
    reason = operation.get("reason")
    edit_mode = operation.get("edit_mode")
    parts = [op]
    if question_id:
        parts.append(f"id={question_id}")
    if path:
        parts.append(f"path={_compact_text(str(path), limit=120)}")
    if edit_mode:
        parts.append(f"mode={_compact_text(str(edit_mode), limit=80)}")
    if reason and reason != op:
        parts.append(f"reason={_compact_text(str(reason), limit=80)}")
    return " ".join(parts)


def _operation_summary_list(operations: Any, *, limit: int = 5) -> str | None:
    if not isinstance(operations, list) or not operations:
        return None
    summaries = [
        _operation_summary(operation)
        for operation in operations
        if isinstance(operation, dict)
    ]
    if not summaries:
        return None
    suffix = "" if len(summaries) <= limit else f"; +{len(summaries) - limit} more"
    return "; ".join(summaries[:limit]) + suffix


def _guard_reason_for_source(source: str) -> str:
    if source == "baseline_train":
        return "baseline_train_pass"
    if source == "baseline_validation":
        return "baseline_validation_pass"
    if source == "accepted_candidate":
        return "accepted_candidate_fix"
    return "visible_pass_guard"


def _candidate_summary_lines(candidate_summary: Any) -> list[str]:
    if not isinstance(candidate_summary, dict):
        return []
    lines: list[str] = []
    candidate_name = candidate_summary.get("candidate_name")
    if candidate_name:
        lines.append(f"  - Candidate payload: `{candidate_name}`")
    candidate_mode = candidate_summary.get("candidate_mode")
    if candidate_mode:
        lines.append(f"  - Candidate mode: `{candidate_mode}`")
    rationale = candidate_summary.get("rationale")
    if rationale:
        lines.append(f"  - Rationale: {_truncate(str(rationale), 400)}")
    applied_operations = candidate_summary.get("applied_operations")
    applied_operation_summary = _operation_summary_list(applied_operations)
    if applied_operation_summary:
        lines.append(f"  - Applied operations: {applied_operation_summary}")
    skipped_operations = _operation_summary_list(candidate_summary.get("skipped_operations"))
    if skipped_operations:
        lines.append(f"  - Skipped operations: {skipped_operations}")
    risk_assessment = candidate_summary.get("risk_assessment")
    if isinstance(risk_assessment, dict):
        risk_reasons = risk_assessment.get("reasons")
        if isinstance(risk_reasons, list) and risk_reasons:
            reason_text = ", ".join(f"`{reason}`" for reason in risk_reasons[:5])
            lines.append(f"  - Risk gate reasons: {reason_text}")
        verification = risk_assessment.get("visible_effect_verification")
        causal_plan = (
            verification.get("causal_patch_plan")
            if isinstance(verification, dict)
            else None
        )
        if isinstance(causal_plan, dict):
            available_intents = [
                str(value)
                for value in causal_plan.get("available_sql_delta_intents", [])
                if str(value).strip()
            ]
            selected_intents = [
                str(value)
                for value in causal_plan.get("selected_sql_delta_intents", [])
                if str(value).strip()
            ]
            if available_intents or selected_intents:
                selected_text = (
                    ", ".join(f"`{value}`" for value in selected_intents)
                    if selected_intents
                    else "none"
                )
                available_text = (
                    ", ".join(f"`{value}`" for value in available_intents[:6])
                    if available_intents
                    else "none"
                )
                lines.append(
                    "  - Visible SQL/result intent audit: "
                    f"selected {selected_text}; available {available_text}"
                )
            expected_effect = str(causal_plan.get("expected_visible_effect") or "").strip()
            if expected_effect and (available_intents or selected_intents):
                lines.append(
                    f"  - Expected visible effect: {_truncate(expected_effect, 220)}"
                )
    return lines


def _attempt_train_diff(
    attempt: dict[str, Any],
    train_diffs_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    label = attempt.get("label")
    if not isinstance(label, str):
        return None
    train_diff = train_diffs_by_label.get(f"{label}.train")
    return train_diff if isinstance(train_diff, dict) else None


def _attempt_display_name(attempt: dict[str, Any]) -> str:
    summary = attempt.get("candidate_summary")
    if isinstance(summary, dict) and summary.get("candidate_name"):
        return str(summary["candidate_name"])
    return str(attempt.get("name") or attempt.get("label") or "?")


def _rate_text(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.1f}%"
    return "n/a"


def _format_prior_candidate_lessons(
    *,
    attempts: list[dict[str, Any]],
    invalid_proposals: list[dict[str, Any]],
    train_diffs_by_label: dict[str, dict[str, Any]],
) -> list[str]:
    accepted_attempts = [attempt for attempt in attempts if attempt.get("outcome") == "accepted"]
    rejected_attempts = [attempt for attempt in attempts if attempt.get("outcome") == "rejected"]
    skipped_attempts = [attempt for attempt in attempts if attempt.get("outcome") == "skipped"]
    if not accepted_attempts and not rejected_attempts and not skipped_attempts and not invalid_proposals:
        return []

    lines: list[str] = [
        "## Prior Candidate Lessons",
        "",
        "These lessons come from visible train/validation gates only; hidden final holdout content is not summarized.",
        "",
    ]

    if accepted_attempts:
        lines.append("### Accepted Patterns To Preserve")
        for attempt in accepted_attempts[-5:]:
            name = _attempt_display_name(attempt)
            train_diff = _attempt_train_diff(attempt, train_diffs_by_label) or {}
            fixed = _format_id_list(train_diff.get("fixed"))
            still_failing = _format_id_list(train_diff.get("still_failing"))
            validation_failed = _format_id_list(attempt.get("validation_failed_ids"))
            lines.append(
                f"- `{name}`: accepted at train={_rate_text(attempt.get('train_rate'))}, "
                f"visible_validation={_rate_text(attempt.get('validation_rate'))}."
            )
            if fixed:
                lines.append(f"  - Fixed train-visible failures: {fixed}")
            if still_failing:
                lines.append(f"  - Did not solve train-visible failures: {still_failing}")
            if validation_failed:
                lines.append(f"  - Visible validation still failing: {validation_failed}")
            summary = attempt.get("candidate_summary")
            if isinstance(summary, dict):
                mode = summary.get("candidate_mode")
                if mode:
                    lines.append(f"  - Candidate mode that worked: `{mode}`")
                operations = _operation_summary_list(summary.get("applied_operations"))
                if operations:
                    lines.append(f"  - Operations to preserve as precedent: {operations}")
        lines.append("")

    if rejected_attempts:
        mode_counts: dict[str, int] = {}
        for attempt in rejected_attempts:
            summary = attempt.get("candidate_summary")
            if not isinstance(summary, dict):
                continue
            mode = summary.get("candidate_mode")
            if mode:
                mode_counts[str(mode)] = mode_counts.get(str(mode), 0) + 1
        lines.append("### Rejected Patterns To Avoid Or Revise")
        if mode_counts:
            mode_text = ", ".join(
                f"`{mode}`={count}" for mode, count in sorted(mode_counts.items())
            )
            lines.append(f"- Rejected candidate modes: {mode_text}")
        for attempt in rejected_attempts[-5:]:
            name = _attempt_display_name(attempt)
            reason = _compact_text(str(attempt.get("reason") or "no reason recorded"), limit=220)
            lines.append(f"- `{name}`: rejected because {reason}.")
            summary = attempt.get("candidate_summary")
            if isinstance(summary, dict):
                mode = summary.get("candidate_mode")
                if mode:
                    lines.append(f"  - Candidate mode: `{mode}`")
                operations = _operation_summary_list(summary.get("applied_operations"))
                if operations:
                    lines.append(f"  - Operations that failed the gates: {operations}")
                risk_assessment = summary.get("risk_assessment")
                if isinstance(risk_assessment, dict):
                    risk_reasons = risk_assessment.get("reasons")
                    if isinstance(risk_reasons, list) and risk_reasons:
                        reason_text = ", ".join(f"`{reason}`" for reason in risk_reasons[:5])
                        lines.append(f"  - Risk gate: {reason_text}")
            train_diff = _attempt_train_diff(attempt, train_diffs_by_label) or {}
            fixed = _format_id_list(train_diff.get("fixed"))
            regressed = _format_id_list(train_diff.get("regressed"))
            still_failing = _format_id_list(train_diff.get("still_failing"))
            if fixed:
                lines.append(f"  - Fixed before rejection: {fixed}")
            if regressed:
                lines.append(f"  - Regressed pass guards: {regressed}")
            if still_failing:
                lines.append(f"  - Still failing after rejected attempt: {still_failing}")
        lines.append("")

    if skipped_attempts:
        lines.append("### Skipped Or Risk-Rejected Patterns To Avoid")
        for attempt in skipped_attempts[-5:]:
            name = _attempt_display_name(attempt)
            reason = _compact_text(str(attempt.get("reason") or "no_changes"), limit=220)
            lines.append(f"- `{name}`: skipped because {reason}.")
            summary = attempt.get("candidate_summary")
            if not isinstance(summary, dict):
                continue
            mode = summary.get("candidate_mode")
            if mode:
                lines.append(f"  - Candidate mode: `{mode}`")
            risk_assessment = summary.get("risk_assessment")
            if isinstance(risk_assessment, dict):
                risk_reasons = risk_assessment.get("reasons")
                if isinstance(risk_reasons, list) and risk_reasons:
                    reason_text = ", ".join(f"`{reason}`" for reason in risk_reasons[:5])
                    lines.append(f"  - Risk gate: {reason_text}")
            operations = _operation_summary_list(summary.get("applied_operations"))
            if operations:
                lines.append(f"  - Candidate shape refused before benchmarking: {operations}")
            skipped_operations = _operation_summary_list(summary.get("skipped_operations"))
            if skipped_operations:
                lines.append(f"  - Skipped operations: {skipped_operations}")
        lines.append("")

    if invalid_proposals:
        lines.append("### Malformed Proposal Lessons")
        for proposal in invalid_proposals[-3:]:
            label = str(proposal.get("label") or "?")
            content_length = proposal.get("content_length")
            size_text = f" ({content_length} response characters)" if content_length is not None else ""
            lines.append(
                f"- `{label}`: malformed candidate JSON{size_text}; "
                "keep the next proposal smaller and schema-only."
            )
        lines.append("")

    return lines


def _accepted_candidate_fixed_guards(
    *,
    attempts: list[dict[str, Any]],
    train_diffs_by_label: dict[str, dict[str, Any]],
    latest_by_id: dict[str, dict[str, Any]],
    question_payload_by_id: dict[str, dict[str, Any]],
    seen_guard_ids: set[str],
) -> list[dict[str, Any]]:
    guards: list[dict[str, Any]] = []
    for attempt in attempts:
        if attempt.get("outcome") != "accepted":
            continue
        train_diff = _attempt_train_diff(attempt, train_diffs_by_label)
        if not isinstance(train_diff, dict):
            continue
        fixed_ids = train_diff.get("fixed")
        if not isinstance(fixed_ids, list):
            continue
        for raw_question_id in fixed_ids:
            question_id = str(raw_question_id)
            if not question_id or question_id in seen_guard_ids:
                continue
            latest_result = latest_by_id.get(question_id, {})
            question_payload = question_payload_by_id.get(question_id, {})
            seen_guard_ids.add(question_id)
            guards.append(
                {
                    "question_id": question_id,
                    "question": (
                        latest_result.get("question")
                        or question_payload.get("question")
                        or "(unknown question)"
                    ),
                    "expected_sql": (
                        latest_result.get("expected_sql")
                        or question_payload.get("expected_sql")
                    ),
                    "source": "accepted_candidate",
                    "latest_status": str(latest_result.get("status", "unknown")),
                    "guard_reason": _guard_reason_for_source("accepted_candidate"),
                    "accepted_candidate": _attempt_display_name(attempt),
                }
            )
    return guards


def _baseline_pass_guard_records(
    *,
    baseline_train_data: Any,
    train_data: Any,
    baseline_validation_data: Any,
    validation_data: Any,
    question_payload_by_id: dict[str, dict[str, Any]],
    iteration_facts: _IterationLogFacts,
) -> list[dict[str, Any]]:
    baseline_by_id = _result_by_question_id(baseline_train_data)
    latest_by_id = _result_by_question_id(train_data)
    baseline_pass_guards: list[dict[str, Any]] = []
    seen_guard_ids: set[str] = set()
    for result in baseline_by_id.values():
        if str(result.get("status", "")).lower() != "passed":
            continue
        question_id = str(result.get("question_id", ""))
        if not question_id or question_id in seen_guard_ids:
            continue
        seen_guard_ids.add(question_id)
        baseline_pass_guards.append(
            {
                "question_id": question_id,
                "question": result.get("question", "(unknown question)"),
                "expected_sql": result.get("expected_sql"),
                "source": "baseline_train",
                "latest_status": str(latest_by_id.get(question_id, {}).get("status", "unknown")),
                "guard_reason": _guard_reason_for_source("baseline_train"),
            }
        )

    baseline_validation_statuses = _validation_status_by_id(baseline_validation_data)
    latest_validation_statuses = _validation_status_by_id(validation_data)
    for question_id, baseline_status in baseline_validation_statuses.items():
        if baseline_status != "passed" or question_id in seen_guard_ids:
            continue
        question_payload = question_payload_by_id.get(question_id)
        if not question_payload:
            continue
        seen_guard_ids.add(question_id)
        baseline_pass_guards.append(
            {
                "question_id": question_id,
                "question": question_payload.get("question", "(unknown question)"),
                "expected_sql": question_payload.get("expected_sql"),
                "source": "baseline_validation",
                "latest_status": latest_validation_statuses.get(question_id, "unknown"),
                "guard_reason": _guard_reason_for_source("baseline_validation"),
            }
        )
    baseline_pass_guards.extend(
        _accepted_candidate_fixed_guards(
            attempts=iteration_facts.attempts,
            train_diffs_by_label=iteration_facts.train_diffs_by_label,
            latest_by_id=latest_by_id,
            question_payload_by_id=question_payload_by_id,
            seen_guard_ids=seen_guard_ids,
        )
    )
    return baseline_pass_guards


def build_structured_failure_context_model(
    *,
    results_dir: Path,
    history_dir: Path,
    benchmarks_dir: Path | None = None,
    iteration_label: str = "current",
    best_train_rate: float | None = None,
    best_validation_rate: float | None = None,
) -> StructuredFailureContext:
    """Build structured visible failure evidence from workspace artifacts."""
    train_data = _load_json(results_dir / "latest_train.json")
    baseline_train_data = _load_json(results_dir / "baseline_train.json")
    validation_data = _load_json(results_dir / "latest_validation_summary.json")
    baseline_validation_data = _load_json(results_dir / "baseline_validation_summary.json")
    question_payload_by_id = _question_payload_by_id(
        results_dir=results_dir,
        benchmarks_dir=benchmarks_dir,
    )
    iteration_facts = _load_iteration_log_facts(history_dir / "iteration_log.jsonl")
    baseline_pass_guards = _baseline_pass_guard_records(
        baseline_train_data=baseline_train_data,
        train_data=train_data,
        baseline_validation_data=baseline_validation_data,
        validation_data=validation_data,
        question_payload_by_id=question_payload_by_id,
        iteration_facts=iteration_facts,
    )

    results = train_data.get("results", []) if isinstance(train_data, dict) else []
    typed_results = [result for result in results if isinstance(result, dict)]
    total = len(typed_results)
    passed = sum(1 for result in typed_results if str(result.get("status", "")).lower() == "passed")
    train_pass_rate = round(passed / total * 100, 2) if total else None
    failed_results = [
        result
        for result in typed_results
        if str(result.get("status", "")).lower() != "passed"
    ]
    failures = tuple(
        _structured_failure_question(
            result,
            pass_guard_count=len(baseline_pass_guards),
            question_payload_by_id=question_payload_by_id,
        )
        for result in failed_results
    )
    family_ids: dict[str, list[str]] = {}
    for failure in failures:
        family_ids.setdefault(failure.likely_root_cause_family, []).append(failure.question_id)

    pass_guards = tuple(
        StructuredPassGuardQuestion(
            question_id=str(guard.get("question_id", "?")),
            question=(
                str(guard.get("question"))
                if guard.get("question") is not None
                else None
            ),
            expected_sql=(
                str(guard.get("expected_sql"))
                if guard.get("expected_sql") is not None
                else None
            ),
            source=str(guard.get("source", "baseline")),
            latest_status=str(guard.get("latest_status", "unknown")),
            guard_reason=str(
                guard.get("guard_reason")
                or _guard_reason_for_source(str(guard.get("source", "baseline")))
            ),
            accepted_candidate=(
                str(guard.get("accepted_candidate"))
                if guard.get("accepted_candidate") is not None
                else None
            ),
        )
        for guard in baseline_pass_guards[:_MAX_BASELINE_PASS_GUARDS_IN_CONTEXT]
    )
    per_question_attribution = _per_question_edit_attribution(
        iteration_facts.attempts, iteration_facts.train_diffs_by_label
    )
    return StructuredFailureContext(
        schema_version=1,
        iteration_label=iteration_label,
        train_pass_rate=train_pass_rate,
        best_train_rate=best_train_rate,
        best_validation_rate=best_validation_rate,
        train_question_count=total,
        failed_question_count=len(failures),
        failures=failures,
        pass_guards=pass_guards,
        failure_families=family_ids,
        hidden_holdout_policy="hidden final holdout content is intentionally omitted",
        per_question_attribution=per_question_attribution,
    )


def build_structured_failure_context(
    *,
    results_dir: Path,
    history_dir: Path,
    benchmarks_dir: Path | None = None,
    iteration_label: str = "current",
    best_train_rate: float | None = None,
    best_validation_rate: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable companion for failure context content."""
    return build_structured_failure_context_model(
        results_dir=results_dir,
        history_dir=history_dir,
        benchmarks_dir=benchmarks_dir,
        iteration_label=iteration_label,
        best_train_rate=best_train_rate,
        best_validation_rate=best_validation_rate,
    ).to_dict()


def build_failure_context(
    *,
    results_dir: Path,
    history_dir: Path,
    benchmarks_dir: Path | None = None,
    iteration_label: str = "current",
    best_train_rate: float | None = None,
    best_validation_rate: float | None = None,
) -> str:
    """Build a FAILURE_CONTEXT.md string from workspace artifacts.

    Returns the markdown content as a string. The caller is responsible
    for writing it to the appropriate location.
    """
    lines: list[str] = [f"# Failure Context for {iteration_label}", ""]

    # --- Current state ---
    train_data = _load_json(results_dir / "latest_train.json")
    baseline_train_data = _load_json(results_dir / "baseline_train.json")
    validation_data = _load_json(results_dir / "latest_validation_summary.json")
    baseline_validation_data = _load_json(results_dir / "baseline_validation_summary.json")
    diagnostics_data = _load_json(results_dir / "space_diagnostics.json")
    question_payload_by_id = _question_payload_by_id(
        results_dir=results_dir,
        benchmarks_dir=benchmarks_dir,
    )
    iteration_facts = _load_iteration_log_facts(history_dir / "iteration_log.jsonl")

    if train_data:
        results = train_data.get("results", [])
        total = len(results)
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = total - passed
        train_rate = round(passed / total * 100, 2) if total else 0.0
        lines.append("## Current State")
        lines.append(f"- Train: {train_rate}% ({passed}/{total} pass, {failed} fail)")
        if validation_data:
            lines.append(f"- Validation: {validation_data.get('pass_rate', 'n/a')}%")
        if best_train_rate is not None:
            lines.append(f"- Best train achieved so far: {best_train_rate:.2f}%")
        if best_validation_rate is not None:
            lines.append(f"- Best validation achieved so far: {best_validation_rate:.2f}%")
        lines.append("")

        # --- Failed questions ---
        failed_results = [
            result
            for result in results
            if isinstance(result, dict) and result.get("status") != "passed"
        ]
        failure_evidence = [_failure_evidence(result) for result in failed_results]
        lines.extend(_format_compact_failure_evidence(failure_evidence))
        lines.extend(_format_failure_families(failure_evidence))

        if failed_results:
            lines.append(f"## Failed Questions ({len(failed_results)})")
            lines.append("")
            for i, result in enumerate(failed_results, 1):
                question = result.get("question", "(unknown question)")
                question_id = result.get("question_id", "?")
                score = result.get("comparison_score")
                method = result.get("comparison_method", "unknown")
                details = result.get("comparison_details", "")
                expected = result.get("expected_sql")
                generated = result.get("generated_sql")
                error_msg = result.get("error_message")

                lines.append(f"### Failure {i}: {_truncate(question, 120)}")
                lines.append(f"- ID: `{question_id}`")
                if score is not None:
                    lines.append(f"- Score: {score} ({details})")
                lines.append(f"- Method: {method}")
                if error_msg:
                    lines.append(f"- Error: {error_msg}")
                lines.append("")
                lines.append("**Expected SQL:**")
                lines.append("```sql")
                lines.append(_format_sql(expected))
                lines.append("```")
                lines.append("")
                lines.append("**Generated SQL:**")
                lines.append("```sql")
                lines.append(_format_sql(generated))
                lines.append("```")
                lines.append("")

        # --- Passing questions (summary only) ---
        passed_results = [r for r in results if r.get("status") == "passed"]
        if passed_results:
            lines.append(f"## Passing Questions ({len(passed_results)})")
            lines.append("")
            for result in passed_results:
                question = result.get("question", "(unknown)")
                score = result.get("comparison_score", "?")
                lines.append(f"- `{result.get('question_id', '?')}`: {_truncate(question, 80)} (score={score})")
            lines.append("")

    baseline_by_id = _result_by_question_id(baseline_train_data)
    latest_by_id = _result_by_question_id(train_data)
    baseline_pass_guards: list[dict[str, Any]] = []
    seen_guard_ids: set[str] = set()
    for result in baseline_by_id.values():
        if str(result.get("status", "")).lower() != "passed":
            continue
        question_id = str(result.get("question_id", ""))
        if not question_id or question_id in seen_guard_ids:
            continue
        seen_guard_ids.add(question_id)
        baseline_pass_guards.append(
            {
                "question_id": question_id,
                "question": result.get("question", "(unknown question)"),
                "expected_sql": result.get("expected_sql"),
                "source": "baseline_train",
                "latest_status": str(latest_by_id.get(question_id, {}).get("status", "unknown")),
                "guard_reason": _guard_reason_for_source("baseline_train"),
            }
        )

    baseline_validation_statuses = _validation_status_by_id(baseline_validation_data)
    latest_validation_statuses = _validation_status_by_id(validation_data)
    for question_id, baseline_status in baseline_validation_statuses.items():
        if baseline_status != "passed" or question_id in seen_guard_ids:
            continue
        question_payload = question_payload_by_id.get(question_id)
        if not question_payload:
            continue
        seen_guard_ids.add(question_id)
        baseline_pass_guards.append(
            {
                "question_id": question_id,
                "question": question_payload.get("question", "(unknown question)"),
                "expected_sql": question_payload.get("expected_sql"),
                "source": "baseline_validation",
                "latest_status": latest_validation_statuses.get(question_id, "unknown"),
                "guard_reason": _guard_reason_for_source("baseline_validation"),
            }
        )
    baseline_pass_guards.extend(
        _accepted_candidate_fixed_guards(
            attempts=iteration_facts.attempts,
            train_diffs_by_label=iteration_facts.train_diffs_by_label,
            latest_by_id=latest_by_id,
            question_payload_by_id=question_payload_by_id,
            seen_guard_ids=seen_guard_ids,
        )
    )

    if baseline_pass_guards:
        lines.append("## Baseline Pass Guard Patterns")
        lines.append("")
        lines.append(
            "These questions passed at baseline in train/validation-visible evaluation or were fixed by "
            "an accepted visible-train candidate. Candidate fixes should preserve these SQL patterns "
            "unless they can prove a strictly better equivalent query."
        )
        lines.append(
            "In recovery example-set modes, include one add_example_sql operation for every guard below "
            "before adding failed-train examples, and set each guard operation reason to "
            "guard_preservation:<question_id>."
        )
        lines.append("")
        for guard in baseline_pass_guards[:_MAX_BASELINE_PASS_GUARDS_IN_CONTEXT]:
            question_id = str(guard.get("question_id", "?"))
            latest_status = str(guard.get("latest_status", "unknown"))
            question = guard.get("question", "(unknown question)")
            lines.append(f"### Guard: {_truncate(question, 120)}")
            lines.append(f"- ID: `{question_id}`")
            lines.append(f"- Source: `{guard.get('source', 'baseline')}`")
            lines.append(f"- Latest status after rollback/best checkpoint: `{latest_status}`")
            guard_source = str(guard.get("source", "baseline"))
            guard_reason = guard.get("guard_reason") or _guard_reason_for_source(guard_source)
            lines.append(
                f"- Guard reason: `{guard_reason}`"
            )
            accepted_candidate = guard.get("accepted_candidate")
            if accepted_candidate:
                lines.append(f"- Accepted candidate: `{accepted_candidate}`")
                lines.append("- Future candidates must preserve this accepted visible-train fix.")
            lines.append("")
            lines.append("**Baseline Expected SQL Pattern:**")
            lines.append("```sql")
            lines.append(_format_sql(guard.get("expected_sql"), limit=20000))
            lines.append("```")
            lines.append("")

    # --- Prior attempts ---
    if iteration_facts.attempts or iteration_facts.invalid_proposals or iteration_facts.train_diffs_by_label:
        lines.extend(
            _format_prior_candidate_lessons(
                attempts=iteration_facts.attempts,
                invalid_proposals=iteration_facts.invalid_proposals,
                train_diffs_by_label=iteration_facts.train_diffs_by_label,
            )
        )

        if iteration_facts.attempts or iteration_facts.invalid_proposals:
            lines.append("## Prior Optimization Attempts")
            lines.append("")
            combined_attempts = sorted(
                [
                    *iteration_facts.attempts[-10:],
                    *iteration_facts.invalid_proposals[-5:],
                ],
                key=lambda entry: str(entry.get("recorded_at") or ""),
            )
            for attempt in combined_attempts[-10:]:
                if attempt.get("phase") == "serving_candidate_proposal_invalid":
                    label = attempt.get("label", "?")
                    content_length = attempt.get("content_length")
                    lines.append(f"- **{label}** → skipped (serving proposal invalid)")
                    lines.append("  - Reason: invalid_serving_response_json")
                    if content_length is not None:
                        lines.append(f"  - Response characters: {content_length}")
                    raw_content_path = attempt.get("raw_content_path")
                    if raw_content_path:
                        lines.append(f"  - Raw response artifact: `{raw_content_path}`")
                    continue
                name = attempt.get("name", "?")
                outcome = attempt.get("outcome", "?")
                reason = attempt.get("reason", "")
                train_rate = attempt.get("train_rate")
                val_rate = attempt.get("validation_rate")
                skipped_val = attempt.get("skipped_validation", False)
                train_info = f"train={train_rate:.1f}%" if train_rate is not None else ""
                val_info = f"val={val_rate:.1f}%" if val_rate is not None else ("val=skipped" if skipped_val else "")
                lines.append(f"- **{name}** → {outcome} ({train_info}, {val_info})")
                if reason:
                    lines.append(f"  - Reason: {reason}")
                lines.extend(_candidate_summary_lines(attempt.get("candidate_summary")))
                label = attempt.get("label")
                train_diff = (
                    iteration_facts.train_diffs_by_label.get(f"{label}.train")
                    if isinstance(label, str)
                    else None
                )
                if isinstance(train_diff, dict):
                    fixed = _format_id_list(train_diff.get("fixed"))
                    regressed = _format_id_list(train_diff.get("regressed"))
                    still_failing = _format_id_list(train_diff.get("still_failing"))
                    if fixed:
                        lines.append(f"  - Fixed in train before rollback: {fixed}")
                    if regressed:
                        lines.append(f"  - Regressed pass guards before rollback: {regressed}")
                    if still_failing:
                        lines.append(f"  - Still failing: {still_failing}")
                failed_ids = attempt.get("train_failed_ids", [])
                if failed_ids:
                    lines.append(f"  - Train failures: {', '.join(f'`{fid}`' for fid in failed_ids[:5])}")
            lines.append("")

    # --- Space diagnostics ---
    if diagnostics_data:
        metadata = diagnostics_data.get("metadata_coverage", {})
        entity_candidates = diagnostics_data.get("entity_matching_candidates", [])
        format_candidates = diagnostics_data.get("format_assistance_candidates", [])
        external_refs = diagnostics_data.get("external_table_references", [])

        lines.append("## Space Diagnostics")
        lines.append(f"- Tables: {metadata.get('total_tables', '?')}")
        lines.append(
            f"- Columns: {metadata.get('total_columns', '?')} "
            f"({metadata.get('described_columns', '?')} described)"
        )
        lines.append(f"- External table warnings: {len(external_refs)}")
        lines.append(f"- Entity matching candidates (not yet enabled): {len(entity_candidates)}")
        for candidate in entity_candidates[:5]:
            lines.append(
                f"  - `{candidate.get('table_identifier', '?')}.{candidate.get('column_name', '?')}`: "
                f"{candidate.get('reason', '')}"
            )
        lines.append(f"- Format assistance candidates (not yet enabled): {len(format_candidates)}")
        for candidate in format_candidates[:5]:
            lines.append(
                f"  - `{candidate.get('table_identifier', '?')}.{candidate.get('column_name', '?')}`: "
                f"{candidate.get('reason', '')}"
            )
        lines.append("")

    return "\n".join(lines)
