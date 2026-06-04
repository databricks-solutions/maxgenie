"""Shared semantic helpers for artifact-patch candidate checks."""

from __future__ import annotations

import re
from typing import Any, Mapping


QUESTION_OVERLAP_STOPWORDS = frozenset(
    {
        "and",
        "are",
        "can",
        "for",
        "from",
        "get",
        "give",
        "how",
        "into",
        "list",
        "of",
        "please",
        "show",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


def question_overlap_terms(text: str) -> set[str]:
    """Return non-generic question terms used for prompt anchors and local guards."""

    terms = set(re.findall(r"[a-z0-9][a-z0-9_+]{2,}", text.lower()))
    return {
        term
        for term in terms
        if term not in QUESTION_OVERLAP_STOPWORDS and not term.isdigit()
    }


def sql_delta_intents_from_visible_failure(failure: Mapping[str, Any]) -> list[str]:
    """Return normalized visible SQL/result behavior labels for one failure."""

    family = str(failure.get("likely_root_cause_family") or "").strip().lower()
    issue_class = str(failure.get("issue_class") or "").strip().lower()
    method = str(failure.get("comparison_method") or "").strip().lower()
    expected_tokens = [
        str(value).strip()
        for value in failure.get("missing_expected_tokens", []) or []
        if str(value).strip()
    ]
    generated_tokens = [
        str(value).strip()
        for value in failure.get("extra_generated_tokens", []) or []
        if str(value).strip()
    ]
    expected_text = " ".join(token.lower() for token in expected_tokens)
    generated_text = " ".join(token.lower() for token in generated_tokens)
    signal_text = " ".join([family, issue_class, method, expected_text, generated_text])

    intents: list[str] = []
    if "literal_or_parameter_mismatch" in family:
        if expected_tokens:
            intents.append("literalize_expected_value")
        if any(
            token.startswith(":") or token.startswith("${")
            for token in generated_tokens
        ):
            intents.append("remove_bind_placeholder")
    if "unbound" in signal_text or "runtime" in family:
        intents.append("produce_executable_sql")
    if "percentage_or_comparison_metric" in family:
        intents.append("restore_prior_period_comparison")
        if "percentage" in signal_text or "difference" in signal_text:
            intents.append("restore_percentage_delta_alias")
    if "time_filter_mismatch" in family:
        intents.append("align_time_window_filter")
    if "selected_metric_or_column_mismatch" in family or "column_mismatch" in method:
        if expected_tokens:
            intents.append("restore_expected_result_columns")
        if generated_tokens:
            intents.append("remove_unrequested_result_columns")
    if "value_mismatch" in method:
        intents.append("repair_result_semantics")
    return list(dict.fromkeys(intents))
