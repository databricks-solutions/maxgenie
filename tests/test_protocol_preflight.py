from dataclasses import dataclass

import pytest

from maxgenie.protocol_preflight import validate_protocol_preflight


def _comparable_payload(**overrides):
    payload = {
        "comparison_mode": "result_based",
        "warehouse_id": "wh-123",
        "benchmark_coverage": "15/15",
        "split_strategy": "canonical",
        "fresh_baseline": True,
        "strategy": "serving",
        "serving_endpoint": "databricks-gpt-5-5-pro",
        "serving_model": "databricks-gpt-5-5",
        "reasoning_effort": "xhigh",
        "observed_benchmark_reuse_policy": "fresh_baseline_then_latest_observed_for_seed_only",
    }
    payload.update(overrides)
    return payload


def test_comparable_result_based_serving_payload_passes():
    result = validate_protocol_preflight(_comparable_payload())

    assert result.status == "comparable"
    assert result.comparable is True
    assert result.errors == ()
    assert result.to_dict()["status"] == "comparable"


def test_result_based_requires_warehouse_id():
    result = validate_protocol_preflight(_comparable_payload(warehouse_id=None))

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == ["missing_warehouse_id"]


def test_string_only_run_is_non_comparable():
    result = validate_protocol_preflight(
        _comparable_payload(comparison_mode="string_only", warehouse_id=None)
    )

    assert result.status == "non_comparable"
    assert {issue.code for issue in result.errors} == {"string_only_non_comparable"}


def test_requires_expected_benchmark_coverage():
    result = validate_protocol_preflight(
        _comparable_payload(benchmark_coverage=None, scored_questions=14)
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == ["incomplete_benchmark_coverage"]
    assert result.errors[0].expected == "15/15"
    assert result.errors[0].actual == "14/15"


def test_coverage_denominator_must_match_expected_population():
    result = validate_protocol_preflight(
        _comparable_payload(benchmark_coverage="15/14")
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "benchmark_coverage_expected_mismatch"
    ]
    assert result.errors[0].actual == "15/14"


def test_final_evidence_requires_expected_benchmark_population():
    result = validate_protocol_preflight(
        _comparable_payload(
            benchmark_coverage=None,
            scored_questions=15,
            result_based_question_count=15,
            final_evidence=True,
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "missing_expected_benchmark_coverage"
    ]
    assert result.errors[0].expected == "15/15"
    assert result.errors[0].actual == 15


def test_final_result_based_evidence_requires_result_based_coverage():
    result = validate_protocol_preflight(
        _comparable_payload(
            benchmark_coverage={"scored": 15, "expected": 15},
            final_evidence=True,
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "missing_result_based_benchmark_coverage"
    ]


def test_final_result_based_evidence_rejects_partial_result_based_coverage():
    result = validate_protocol_preflight(
        _comparable_payload(
            benchmark_coverage={"scored": 15, "expected": 15, "result_based": 14},
            result_based_question_count=14,
            final_evidence=True,
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "incomplete_result_based_benchmark_coverage"
    ]
    assert result.errors[0].expected == 15
    assert result.errors[0].actual == 14


def test_result_based_evidence_rejects_string_fallback_methods():
    result = validate_protocol_preflight(
        _comparable_payload(
            benchmark_coverage={
                "scored": 15,
                "expected": 15,
                "result_based": 15,
                "comparison_method_counts": {
                    "result_match": 14,
                    "string_fallback_unbound_parameters": 1,
                },
            },
            result_based_question_count=15,
            comparison_method_counts={
                "result_match": 14,
                "string_fallback_unbound_parameters": 1,
            },
            final_evidence=True,
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "string_fallback_in_result_based_evidence"
    ]
    assert result.errors[0].actual == {"string_fallback_unbound_parameters": 1}


def test_result_based_evidence_rejects_legacy_result_prefixed_fallback_methods():
    result = validate_protocol_preflight(
        _comparable_payload(
            benchmark_coverage={
                "scored": 15,
                "expected": 15,
                "result_based": 15,
                "comparison_method_counts": {
                    "result_match": 14,
                    "result_unbound_parameters_fallback": 1,
                },
            },
            result_based_question_count=15,
            final_evidence=True,
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "string_fallback_in_result_based_evidence"
    ]
    assert result.errors[0].actual == {"result_unbound_parameters_fallback": 1}


def test_allows_caller_specified_expected_benchmark_count():
    result = validate_protocol_preflight(
        _comparable_payload(
            benchmark_coverage={"scored": 2, "expected": 2, "result_based": 2},
            result_based_question_count=2,
            final_evidence=True,
        ),
        expected_benchmark_count=2,
    )

    assert result.status == "comparable"


def test_requires_fixed_or_canonical_split():
    result = validate_protocol_preflight(_comparable_payload(split_strategy="random"))

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == ["missing_fixed_canonical_split"]


def test_checkpoint_audit_does_not_require_fresh_baseline():
    result = validate_protocol_preflight(
        _comparable_payload(mode="checkpoint_audit", fresh_baseline=False)
    )

    assert result.status == "comparable"


def test_patch_replay_audit_does_not_require_serving_endpoint():
    result = validate_protocol_preflight(
        _comparable_payload(
            mode="patch_replay_audit",
            serving_endpoint=None,
            serving_model=None,
        )
    )

    assert result.status == "comparable"


def test_non_checkpoint_mode_requires_fresh_baseline():
    result = validate_protocol_preflight(_comparable_payload(fresh_baseline=False))

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == ["missing_fresh_baseline"]


def test_explicit_observed_benchmark_seed_can_replace_fresh_baseline():
    result = validate_protocol_preflight(
        _comparable_payload(
            fresh_baseline=False,
            observed_benchmark_reuse_policy="observed_seed_initial_baseline",
        )
    )

    assert result.status == "comparable"
    assert result.errors == ()


def test_serving_strategy_requires_endpoint_model_and_xhigh_reasoning():
    result = validate_protocol_preflight(
        _comparable_payload(
            serving_endpoint="",
            serving_model=None,
            reasoning_effort="high",
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "missing_serving_endpoint",
        "missing_serving_model",
        "serving_reasoning_not_xhigh",
    ]


def test_serving_strategy_accepts_gpt_5_5_pro_high_control_effort():
    result = validate_protocol_preflight(
        _comparable_payload(
            serving_endpoint="databricks-gpt-5-5-pro",
            serving_model="databricks-gpt-5-5-pro",
            reasoning_effort="high",
        )
    )

    assert result.status == "comparable"
    assert result.errors == ()


def test_serving_strategy_accepts_gpt_5_5_reasoning_sweep_efforts():
    for effort in ("low", "medium", "xhigh"):
        result = validate_protocol_preflight(
            _comparable_payload(
                serving_endpoint="databricks-gpt-5-5",
                serving_model="databricks-gpt-5-5",
                reasoning_effort=effort,
            )
        )

        assert result.status == "comparable"
        assert result.errors == ()


def test_serving_strategy_rejects_gpt_5_5_high_effort():
    result = validate_protocol_preflight(
        _comparable_payload(
            serving_endpoint="databricks-gpt-5-5",
            serving_model="databricks-gpt-5-5",
            reasoning_effort="high",
        )
    )

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == ["serving_reasoning_not_xhigh"]
    assert result.errors[0].expected == "one of low/medium/xhigh"


def test_serving_strategy_accepts_adaptive_moderate_thinking_effort():
    result = validate_protocol_preflight(
        _comparable_payload(
            serving_endpoint="databricks-claude-opus-4-8",
            serving_model="databricks-claude-opus-4-8",
            reasoning_effort="medium",
        )
    )

    assert result.status == "comparable"
    assert result.errors == ()


def test_serving_strategy_rejects_invalid_adaptive_thinking_effort():
    result = validate_protocol_preflight(
        _comparable_payload(
            serving_endpoint="databricks-claude-opus-4-8",
            serving_model="databricks-claude-opus-4-8",
            reasoning_effort="ludicrous",
        )
    )

    assert result.status == "non_comparable"
    assert "serving_thinking_effort_invalid" in [issue.code for issue in result.errors]


def test_observed_benchmark_reuse_policy_must_be_recorded():
    payload = _comparable_payload()
    payload.pop("observed_benchmark_reuse_policy")

    result = validate_protocol_preflight(payload)

    assert result.status == "non_comparable"
    assert [issue.code for issue in result.errors] == [
        "missing_observed_benchmark_reuse_policy"
    ]


def test_boolean_observed_reuse_policy_is_explicit_but_warned():
    result = validate_protocol_preflight(
        _comparable_payload(observed_benchmark_reuse_policy=True)
    )

    assert result.status == "comparable"
    assert [issue.code for issue in result.warnings] == ["boolean_observed_reuse_policy"]


def test_settings_like_object_is_supported():
    @dataclass
    class Settings:
        comparison_mode: str = "result_based"
        warehouse_id: str = "wh-123"
        question_count: int = 15
        fixed_split: bool = True
        fresh_baseline: bool = True
        strategy: str = "serving_autonomous"
        endpoint: str = "databricks-gpt-5-5-pro"
        model: str = "databricks-gpt-5-5"
        candidate_reasoning_effort: str = "xhigh"
        benchmark_reuse_policy: str = "disabled_for_baseline"

    result = validate_protocol_preflight(Settings())

    assert result.status == "comparable"


def test_expected_benchmark_count_must_be_positive():
    with pytest.raises(ValueError, match="expected_benchmark_count"):
        validate_protocol_preflight(_comparable_payload(), expected_benchmark_count=0)
