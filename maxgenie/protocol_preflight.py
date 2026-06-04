"""Pure protocol checks for production-comparable optimizer runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


PreflightStatus = Literal["comparable", "non_comparable"]
IssueSeverity = Literal["error", "warning"]

RESULT_BASED_COMPARISON_MODES = frozenset(
    {
        "result_based",
        "result-based",
        "result",
        "results",
        "query_result",
        "query-results",
        "query_results",
        "semantic_result",
        "semantic-results",
        "semantic_results",
    }
)
STRING_ONLY_COMPARISON_MODES = frozenset(
    {
        "string",
        "string_only",
        "string-only",
        "sql_string",
        "sql-string",
        "similarity",
        "similarity_match",
    }
)
CANONICAL_SPLIT_VALUES = frozenset(
    {
        "canonical",
        "fixed",
        "fixed_canonical",
        "canonical_fixed",
        "baseline_outcome_balanced",
    }
)
SERVING_STRATEGIES = frozenset(
    {
        "serving",
        "serving_autonomous",
        "serving-autonomous",
    }
)
STRING_FALLBACK_COMPARISON_METHODS = frozenset(
    {
        "string_fallback_unbound_parameters",
        "string_fallback_non_select",
        # Legacy persisted method names. These used a result_* prefix but were
        # produced by string fallback after result comparison failed.
        "result_unbound_parameters_fallback",
        "result_non_select_fallback",
    }
)


@dataclass(frozen=True)
class PreflightIssue:
    """Structured preflight issue."""

    code: str
    message: str
    severity: IssueSeverity = "error"
    field: str | None = None
    expected: Any | None = None
    actual: Any | None = None


@dataclass(frozen=True)
class ProtocolPreflightResult:
    """Comparable/non-comparable verdict plus structured diagnostics."""

    status: PreflightStatus
    errors: tuple[PreflightIssue, ...] = ()
    warnings: tuple[PreflightIssue, ...] = ()

    @property
    def comparable(self) -> bool:
        return self.status == "comparable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "comparable": self.comparable,
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


def validate_protocol_preflight(
    payload: Mapping[str, Any] | object,
    *,
    expected_benchmark_count: int = 15,
) -> ProtocolPreflightResult:
    """Validate whether optimizer settings produce production-comparable evidence.

    The function accepts a plain dict or a settings-like object. It performs no
    I/O and intentionally validates only protocol shape, not external resources.
    """

    if expected_benchmark_count <= 0:
        raise ValueError("expected_benchmark_count must be positive")

    errors: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    mode = _normalized(_get(payload, "mode", "run_mode", "workflow_mode"))

    comparison_mode = _normalized(
        _get(
            payload,
            "comparison_mode",
            "preferred_comparison_mode",
            "actual_full_comparison_mode",
            "full_comparison_mode",
        )
    )
    if not comparison_mode:
        errors.append(
            PreflightIssue(
                code="missing_comparison_mode",
                field="comparison_mode",
                expected="result_based",
                message="Production-comparable runs must record a result-based comparison mode.",
            )
        )
    elif comparison_mode in STRING_ONLY_COMPARISON_MODES:
        errors.append(
            PreflightIssue(
                code="string_only_non_comparable",
                field="comparison_mode",
                expected="result_based",
                actual=comparison_mode,
                message="String-only comparison runs are non-comparable production evidence.",
            )
        )
    elif comparison_mode not in RESULT_BASED_COMPARISON_MODES:
        errors.append(
            PreflightIssue(
                code="unknown_comparison_mode",
                field="comparison_mode",
                expected=sorted(RESULT_BASED_COMPARISON_MODES),
                actual=comparison_mode,
                message="Comparison mode must be explicitly result-based for production comparability.",
            )
        )

    if comparison_mode in RESULT_BASED_COMPARISON_MODES and not _present(_get(payload, "warehouse_id")):
        errors.append(
            PreflightIssue(
                code="missing_warehouse_id",
                field="warehouse_id",
                expected="non-empty warehouse id",
                actual=_get(payload, "warehouse_id"),
                message="Result-based comparison requires a warehouse_id.",
            )
        )

    final_evidence = _boolish(_get(payload, "final_evidence", "actual_evidence"))
    actual_coverage = _coverage_count(payload)
    declared_expected_coverage = _coverage_expected_count(payload)
    if actual_coverage is None:
        errors.append(
            PreflightIssue(
                code="missing_benchmark_coverage",
                field="benchmark_coverage",
                expected=f"{expected_benchmark_count}/{expected_benchmark_count}",
                message="Expected benchmark coverage must be recorded.",
            )
        )
    elif final_evidence is True and declared_expected_coverage is None:
        errors.append(
            PreflightIssue(
                code="missing_expected_benchmark_coverage",
                field="benchmark_coverage",
                expected=f"{expected_benchmark_count}/{expected_benchmark_count}",
                actual=actual_coverage,
                message="Final evidence must record the expected benchmark population.",
            )
        )
    elif declared_expected_coverage is not None and declared_expected_coverage != expected_benchmark_count:
        errors.append(
            PreflightIssue(
                code="benchmark_coverage_expected_mismatch",
                field="benchmark_coverage",
                expected=f"{expected_benchmark_count}/{expected_benchmark_count}",
                actual=f"{actual_coverage}/{declared_expected_coverage}",
                message="Production-comparable runs must declare the expected benchmark population.",
            )
        )
    elif actual_coverage != expected_benchmark_count:
        errors.append(
            PreflightIssue(
                code="incomplete_benchmark_coverage",
                field="benchmark_coverage",
                expected=f"{expected_benchmark_count}/{expected_benchmark_count}",
                actual=f"{actual_coverage}/{expected_benchmark_count}",
                message="Production-comparable runs must score every expected benchmark question.",
            )
        )
    if comparison_mode in RESULT_BASED_COMPARISON_MODES:
        result_based_coverage = _coverage_result_based_count(payload)
        if result_based_coverage is None:
            if final_evidence is True:
                errors.append(
                    PreflightIssue(
                        code="missing_result_based_benchmark_coverage",
                        field="result_based_question_count",
                        expected=expected_benchmark_count,
                        message="Final result-based evidence must record result-based benchmark coverage.",
                    )
                )
        elif result_based_coverage != expected_benchmark_count:
            errors.append(
                PreflightIssue(
                    code="incomplete_result_based_benchmark_coverage",
                    field="result_based_question_count",
                    expected=expected_benchmark_count,
                    actual=result_based_coverage,
                    message="Production-comparable result-based runs must score every benchmark with result comparison.",
                )
            )
        fallback_counts = _string_fallback_method_counts(payload)
        if fallback_counts:
            errors.append(
                PreflightIssue(
                    code="string_fallback_in_result_based_evidence",
                    field="comparison_method_counts",
                    expected="no string fallback comparison methods",
                    actual=fallback_counts,
                    message=(
                        "Result-comparator failures scored by string fallback are "
                        "non-comparable result-based evidence."
                    ),
                )
            )

    split = _normalized(_get(payload, "split_strategy", "split", "validation_split"))
    fixed_split = _boolish(_get(payload, "fixed_split", "canonical_split"))
    if fixed_split is not True and split not in CANONICAL_SPLIT_VALUES:
        errors.append(
            PreflightIssue(
                code="missing_fixed_canonical_split",
                field="split_strategy",
                expected="fixed/canonical split",
                actual=split or None,
                message="Production-comparable runs require a fixed/canonical split.",
            )
        )

    observed_policy = _get(
        payload,
        "observed_benchmark_reuse_policy",
        "observed_reuse_policy",
        "benchmark_reuse_policy",
        "use_latest_observed_benchmark",
    )
    observed_policy_name = _normalized(observed_policy)
    explicit_observed_seed_baseline = observed_policy_name in {
        "explicit_observed_benchmark_seed",
        "observed_benchmark_seed",
        "observed_seed_initial_baseline",
        "shared_observed_benchmark_seed",
    }
    fresh_baseline = _boolish(
        _get(payload, "fresh_baseline", "fresh_baseline_required", "ran_fresh_baseline")
    )
    if (
        mode != "checkpoint_audit"
        and fresh_baseline is not True
        and not explicit_observed_seed_baseline
    ):
        errors.append(
            PreflightIssue(
                code="missing_fresh_baseline",
                field="fresh_baseline",
                expected="fresh baseline or explicit observed benchmark seed",
                actual=_get(payload, "fresh_baseline", "fresh_baseline_required", "ran_fresh_baseline"),
                message=(
                    "Production-comparable runs require a fresh baseline unless mode is "
                    "checkpoint_audit or an explicit observed benchmark seed is recorded."
                ),
            )
        )

    strategy = _normalized(_get(payload, "strategy", "autonomous_strategy"))
    if strategy in SERVING_STRATEGIES and mode not in {"checkpoint_audit", "patch_replay_audit"}:
        _require_present(
            payload,
            errors,
            field="serving_endpoint",
            aliases=("serving_endpoint", "endpoint"),
            code="missing_serving_endpoint",
            message="Serving strategy requires an explicit serving endpoint.",
        )
        _require_present(
            payload,
            errors,
            field="serving_model",
            aliases=("serving_model", "model", "candidate_model"),
            code="missing_serving_model",
            message="Serving strategy requires an explicit serving model.",
        )
        endpoint_name = _normalized(_get(payload, "serving_endpoint", "endpoint")) or ""
        model_name = _normalized(_get(payload, "serving_model", "model", "candidate_model")) or ""
        uses_adaptive_thinking = "claude" in endpoint_name or "claude" in model_name
        is_gpt_5_5_pro = "gpt-5-5-pro" in endpoint_name or "gpt-5-5-pro" in model_name
        reasoning = _normalized(
            _get(payload, "reasoning_effort", "serving_reasoning_effort", "candidate_reasoning_effort")
        )
        if uses_adaptive_thinking:
            # Adaptive-thinking endpoints use low/medium/high; xhigh-style aliases map
            # to high. A moderate effort is preferred to avoid over-engineered broad
            # edits that trigger train-collapse, so xhigh is neither required nor the
            # recommended default for these endpoints.
            valid_thinking_efforts = {
                "low", "medium", "high", "xhigh", "extra_high", "extra-high", "max", "maximum",
            }
            if reasoning not in valid_thinking_efforts:
                errors.append(
                    PreflightIssue(
                        code="serving_thinking_effort_invalid",
                        field="reasoning_effort",
                        expected="one of low/medium/high for adaptive thinking effort",
                        actual=reasoning or None,
                        message=(
                            "Adaptive-thinking serving models must set a valid effort "
                            "(low, medium, or high); xhigh-style aliases map to high."
                        ),
                    )
                )
        elif is_gpt_5_5_pro and reasoning == "high":
            pass
        elif "gpt-5-5" in endpoint_name or "gpt-5-5" in model_name:
            valid_gpt_5_5_efforts = {"low", "medium", "xhigh"}
            if reasoning not in valid_gpt_5_5_efforts:
                errors.append(
                    PreflightIssue(
                        code="serving_reasoning_not_xhigh",
                        field="reasoning_effort",
                        expected="one of low/medium/xhigh",
                        actual=reasoning or None,
                        message=(
                            "Databricks gpt-5-5 serving runs must use a supported "
                            "reasoning effort: low, medium, or xhigh."
                        ),
                    )
                )
        elif reasoning != "xhigh":
            errors.append(
                PreflightIssue(
                    code="serving_reasoning_not_xhigh",
                    field="reasoning_effort",
                    expected="xhigh",
                    actual=reasoning or None,
                    message="Serving strategy must use xhigh reasoning for production-comparable runs.",
                )
            )

    if observed_policy is None:
        errors.append(
            PreflightIssue(
                code="missing_observed_benchmark_reuse_policy",
                field="observed_benchmark_reuse_policy",
                expected="explicitly recorded reuse policy",
                message="Observed benchmark reuse policy must be explicitly recorded.",
            )
        )
    elif isinstance(observed_policy, bool):
        warnings.append(
            PreflightIssue(
                code="boolean_observed_reuse_policy",
                field="observed_benchmark_reuse_policy",
                severity="warning",
                expected="descriptive policy string",
                actual=observed_policy,
                message="Boolean observed benchmark reuse is explicit but less auditable than a named policy.",
            )
        )
    elif not str(observed_policy).strip():
        errors.append(
            PreflightIssue(
                code="empty_observed_benchmark_reuse_policy",
                field="observed_benchmark_reuse_policy",
                expected="non-empty reuse policy",
                actual=observed_policy,
                message="Observed benchmark reuse policy must be non-empty when recorded.",
            )
        )

    status: PreflightStatus = "non_comparable" if errors else "comparable"
    return ProtocolPreflightResult(
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _require_present(
    payload: Mapping[str, Any] | object,
    errors: list[PreflightIssue],
    *,
    field: str,
    aliases: tuple[str, ...],
    code: str,
    message: str,
) -> None:
    value = _get(payload, *aliases)
    if not _present(value):
        errors.append(
            PreflightIssue(
                code=code,
                field=field,
                expected="non-empty value",
                actual=value,
                message=message,
            )
        )


def _coverage_count(payload: Mapping[str, Any] | object) -> int | None:
    coverage = _get(payload, "benchmark_coverage", "coverage")
    if isinstance(coverage, str) and "/" in coverage:
        left, _ = coverage.split("/", 1)
        return _to_int(left)
    if isinstance(coverage, Mapping):
        for key in ("scored", "actual", "covered", "result_count", "question_count"):
            value = _to_int(coverage.get(key))
            if value is not None:
                return value

    for field in (
        "scored_questions",
        "result_based_questions",
        "benchmark_result_count",
        "result_count",
        "question_count",
        "benchmark_question_count",
    ):
        value = _to_int(_get(payload, field))
        if value is not None:
            return value
    return None


def _coverage_expected_count(payload: Mapping[str, Any] | object) -> int | None:
    coverage = _get(payload, "benchmark_coverage", "coverage")
    if isinstance(coverage, str) and "/" in coverage:
        _, right = coverage.split("/", 1)
        return _to_int(right)
    if isinstance(coverage, Mapping):
        for key in ("expected", "total", "benchmark_question_count"):
            value = _to_int(coverage.get(key))
            if value is not None:
                return value
    return _to_int(_get(payload, "expected_benchmark_count", "benchmark_expected_count"))


def _coverage_result_based_count(payload: Mapping[str, Any] | object) -> int | None:
    coverage = _get(payload, "benchmark_coverage", "coverage")
    if isinstance(coverage, Mapping):
        for key in ("result_based", "result_based_question_count", "result_based_questions"):
            value = _to_int(coverage.get(key))
            if value is not None:
                return value
    return _to_int(
        _get(
            payload,
            "result_based_question_count",
            "result_based_questions",
            "result_based_benchmark_count",
        )
    )


def _string_fallback_method_counts(payload: Mapping[str, Any] | object) -> dict[str, int]:
    method_counts = _get(payload, "comparison_method_counts", "method_counts")
    coverage = _get(payload, "benchmark_coverage", "coverage")
    if not isinstance(method_counts, Mapping) and isinstance(coverage, Mapping):
        nested_counts = coverage.get("comparison_method_counts") or coverage.get("method_counts")
        if isinstance(nested_counts, Mapping):
            method_counts = nested_counts
    if not isinstance(method_counts, Mapping):
        return {}

    fallback_counts: dict[str, int] = {}
    for method, count in method_counts.items():
        method_name = _normalized(method)
        if method_name not in STRING_FALLBACK_COMPARISON_METHODS:
            continue
        parsed_count = _to_int(count)
        if parsed_count is None or parsed_count <= 0:
            continue
        fallback_counts[method_name] = parsed_count
    return dict(sorted(fallback_counts.items()))


def _get(payload: Mapping[str, Any] | object, *names: str) -> Any | None:
    for name in names:
        if isinstance(payload, Mapping):
            if name in payload:
                return payload[name]
        elif hasattr(payload, name):
            return getattr(payload, name)
    return None


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _normalized(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = _normalized(value)
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
