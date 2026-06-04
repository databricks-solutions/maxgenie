import json

import pytest
import requests

from maxgenie.schemas import (
    GenieColumnConfig,
    GenieDataSources,
    GenieInstruction,
    GenieInstructions,
    GenieSpaceConfig,
    GenieTableConfig,
)
from maxgenie.serving_candidates import (
    ServingCandidateClient,
    ServingCandidateConfig,
    ServingCandidateResponseError,
    assess_serving_candidate_risk,
    apply_serving_candidate,
    baseline_pass_guard_ids,
    baseline_pass_guard_questions,
    build_artifact_patch_skeletons,
    build_serving_candidate_prompt,
    canonicalize_tagged_example_sql,
    count_prior_accepted_fix_regressions,
    count_prior_invalid_responses,
    count_prior_pass_guard_regressions,
    count_prior_rejected_candidates,
    count_prior_rejected_candidate_modes,
    count_prior_skipped_candidates,
    materialize_typed_template_examples,
    repair_example_set_pass_guards,
    serving_candidate_artifact_patch_requires_executable_asset,
    serving_candidate_artifact_patch_file_edit_limits,
    serving_candidate_allowed_operation_groups,
    serving_candidate_is_broad_artifact_patch_mode,
    serving_candidate_is_executable_artifact_path,
    serving_candidate_is_artifact_patch_mode,
    serving_candidate_is_strict_artifact_patch_mode,
    serving_candidate_uses_data_probe,
    serving_candidate_uses_historical_analysis,
    serving_candidate_max_mutation_operations,
    serving_candidate_mode_for_iteration,
    validate_artifact_patch_skeleton_binding,
)


class _FakeApiClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def do(self, method, path, body=None, query=None):
        self.calls.append({"method": method, "path": path, "body": body, "query": query})
        return self.response


class _FakeWorkspaceClient:
    def __init__(self, response):
        self.api_client = _FakeApiClient(response)


class _FakeBaseClient:
    def __init__(self):
        self._http_timeout_seconds = 60.0
        self._retry_timeout_seconds = 300


class _FakeApiClientWithTimeouts(_FakeApiClient):
    def __init__(self, response):
        super().__init__(response)
        self._api_client = _FakeBaseClient()

    def do(self, method, path, body=None, query=None):
        self.calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "query": query,
                "http_timeout": self._api_client._http_timeout_seconds,
                "retry_timeout": self._api_client._retry_timeout_seconds,
            }
        )
        return self.response


class _FakeWorkspaceClientWithTimeouts:
    def __init__(self, response):
        self.api_client = _FakeApiClientWithTimeouts(response)


def _config() -> GenieSpaceConfig:
    return GenieSpaceConfig(
        title="Demo",
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.schema.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="trx"),
                    ],
                )
            ]
        ),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact brand filters."])]
        ),
    )


def test_apply_serving_candidate_applies_structured_operations():
    payload = {
        "candidate_name": "brand_filter_fix",
        "rationale": "Failures need reusable brand guidance.",
        "expected_impact": "Fixes brand filter selection.",
        "risk": "Low",
        "owner_followups": ["Confirm brand list freshness."],
        "operations": [
            {
                "op": "append_text_instruction",
                "content": "When brand_name is provided, filter by exact uppercase brand name.",
            },
            {
                "op": "set_column_config",
                "table_identifier": "catalog.schema.sales",
                "column_name": "brand_name",
                "description": "Commercial brand name.",
                "synonyms": ["product"],
                "enable_entity_matching": True,
                "enable_format_assistance": True,
            },
            {
                "op": "add_sql_snippet",
                "group": "filters",
                "display_name": "Brand Exact Filter",
                "sql": "upper(brand_name) = 'FARXIGA'",
            },
        ],
    }

    candidate, summary = apply_serving_candidate(_config(), payload)

    assert summary["changed"] is True
    assert summary["operation_count"] == 3
    content = "\n".join(candidate.instructions.text_instructions[0].content)
    assert "exact uppercase brand name" in content
    column = candidate.data_sources.tables[0].column_configs[0]
    assert column.description == ["Commercial brand name."]
    assert column.synonyms == ["product"]
    assert column.enable_entity_matching is True
    assert candidate.instructions.sql_snippets.filters[0].display_name == "Brand Exact Filter"


def test_apply_serving_candidate_skips_invalid_operations_without_changes():
    candidate, summary = apply_serving_candidate(
        _config(),
        {
            "candidate_name": "invalid",
            "rationale": "",
            "expected_impact": "",
            "risk": "",
            "owner_followups": [],
            "operations": [
                {
                    "op": "set_column_config",
                    "table_identifier": "catalog.schema.sales",
                    "column_name": "missing_column",
                    "description": "Nope",
                }
            ],
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "no_valid_serving_operations"
    assert summary["skipped_operations"][0]["reason"] == "unknown_column"
    assert candidate.to_serialized_space()["data_sources"] == _config().to_serialized_space()["data_sources"]


def test_assess_serving_candidate_risk_allows_one_example_sql_operation():
    payload = {
        "candidate_name": "one_example",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Show Farxiga sales by region for NRx",
                "sql": "SELECT region_name, sum(nrx) FROM catalog.schema.sales GROUP BY region_name",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(payload)

    assert assessment["accepted"] is True
    assert assessment["operation_groups"] == ["example_sql"]


def test_assess_serving_candidate_risk_rejects_mixed_broad_operations():
    payload = {
        "candidate_name": "broad_fix",
        "operations": [
            {
                "op": "append_text_instruction",
                "content": "Always rewrite every market share question using this benchmark pattern.",
            },
            {
                "op": "add_example_sql",
                "question": "Show Farxiga sales by region for NRx",
                "sql": "SELECT region_name, sum(nrx) FROM catalog.schema.sales GROUP BY region_name",
            },
            {
                "op": "set_column_config",
                "table_identifier": "catalog.schema.sales",
                "column_name": "brand_name",
                "enable_entity_matching": True,
            },
        ],
    }

    assessment = assess_serving_candidate_risk(payload)

    assert assessment["accepted"] is False
    assert assessment["reason"] == "serving_candidate_risk_rejected"
    assert "mixed_operation_groups:example_sql,metadata,text_instruction" in assessment["reasons"]


def test_serving_candidate_mode_defaults_to_strict_artifact_patch_v2():
    assert serving_candidate_mode_for_iteration(1) == "artifact_patch_v2"
    assert serving_candidate_mode_for_iteration(2) == "artifact_patch_v2"
    assert serving_candidate_mode_for_iteration(3) == "artifact_patch_v2"
    assert serving_candidate_mode_for_iteration(4) == "artifact_patch_v2"
    assert serving_candidate_mode_for_iteration(7) == "artifact_patch_v2"
    assert (
        serving_candidate_mode_for_iteration(4, prior_pass_guard_regressions=1)
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(4, prior_pass_guard_regressions=2)
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            3,
            prior_pass_guard_regressions=1,
            prior_accepted_fix_regressions=1,
            prior_rejected_candidates=1,
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_pass_guard_regressions=2,
            prior_accepted_fix_regressions=1,
            prior_rejected_candidates=2,
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(4, prior_rejected_candidates=1)
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(4, prior_rejected_candidates=2)
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_rejected_candidates=1,
            prior_rejected_mode_counts={"example_set": 1},
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_rejected_candidates=3,
            prior_rejected_mode_counts={"example_set": 1, "example_sql": 1, "metadata": 1},
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(4, prior_skipped_candidates=1)
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_rejected_candidates=1,
            prior_skipped_candidates=1,
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_rejected_candidates=1,
            prior_skipped_candidates=2,
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_pass_guard_regressions=1,
            prior_rejected_mode_counts={"example_set": 1, "example_sql": 1},
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(4, prior_invalid_responses=1)
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(
            4,
            prior_rejected_candidates=2,
            prior_invalid_responses=1,
        )
        == "artifact_patch_v2"
    )
    assert serving_candidate_allowed_operation_groups("example_set") == frozenset({"example_sql"})
    assert serving_candidate_allowed_operation_groups("benchmark_example_set") == frozenset(
        {"example_sql"}
    )
    assert serving_candidate_max_mutation_operations("example_set") == 9
    assert serving_candidate_max_mutation_operations("benchmark_example_set") == 9
    assert serving_candidate_allowed_operation_groups("schema_time_window") == frozenset(
        {"sql_snippet"}
    )
    assert serving_candidate_max_mutation_operations("schema_time_window") == 1
    assert serving_candidate_allowed_operation_groups("artifact_patch") == frozenset({"artifact_patch"})
    assert serving_candidate_max_mutation_operations("artifact_patch") is None

    assessment = assess_serving_candidate_risk(
        {
            "candidate_name": "wrong_mode",
            "operations": [
                {
                    "op": "append_text_instruction",
                    "content": "Literalize values in generated SQL.",
                }
            ],
        },
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_sql"),
    )

    assert assessment["accepted"] is False
    assert "disallowed_operation_groups:text_instruction" in assessment["reasons"]


def test_assess_serving_candidate_risk_allows_example_set_with_explicit_cap():
    payload = {
        "candidate_name": "guard_preserving_examples",
        "risk": "Preserves pass guard SQL patterns before adding failed examples.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": f"Question {index}",
                "sql": f"SELECT {index} AS value",
            }
            for index in range(4)
        ],
    }

    default_assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_sql"),
    )
    assert default_assessment["accepted"] is False
    assert "too_many_example_sql_operations:4" in default_assessment["reasons"]

    example_set_assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
    )
    assert example_set_assessment["accepted"] is True


def test_assess_serving_candidate_risk_rejects_example_sql_placeholders():
    payload = {
        "candidate_name": "placeholder_example",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Show Farxiga sales",
                "sql": "SELECT * FROM sales WHERE brand_name = :brand_name",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(payload)

    assert assessment["accepted"] is False
    assert "example_sql_contains_placeholder" in assessment["reasons"]


def test_assess_serving_candidate_risk_rejects_empty_untagged_example_sql():
    payload = {
        "candidate_name": "empty_example",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Show Farxiga sales",
                "sql": None,
            }
        ],
    }

    assessment = assess_serving_candidate_risk(payload)

    assert assessment["accepted"] is False
    assert "example_sql_missing_sql_or_benchmark_id" in assessment["reasons"]


def test_assess_serving_candidate_risk_rejects_claimed_example_sql_mismatch():
    payload = {
        "candidate_name": "mismatch",
        "operations": [
            {
                "op": "add_example_sql",
                "reason": "target_train_failure:question-a",
                "question": "Show sales",
                "sql": "SELECT region, sales FROM t",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        expected_sql_by_question_id={"question-a": "SELECT region, revenue FROM t"},
    )

    assert assessment["accepted"] is False
    assert "example_sql_mismatch:question-a" in assessment["reasons"]


def test_assess_serving_candidate_risk_accepts_claimed_example_sql_exact_match():
    payload = {
        "candidate_name": "match",
        "operations": [
            {
                "op": "add_example_sql",
                "reason": "target_train_failure:question-a",
                "question": "Show sales",
                "sql": "SELECT\n  region,\n  sales\nFROM t;",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        expected_sql_by_question_id={"question-a": "select region, sales from t"},
    )

    assert assessment["accepted"] is True


def test_count_prior_pass_guard_regressions():
    context = """
## Prior Optimization Attempts
- Fixed in train before rollback: `a`
- Regressed pass guards before rollback: `b`, `c`
- Fixed in train before rollback: `d`
- Regressed pass guards before rollback: `b`, `c`
"""

    assert count_prior_pass_guard_regressions(context) == 2


def test_count_prior_accepted_fix_regressions():
    context = """
## Prior Optimization Attempts
- **serving_iteration_02** → rejected (train=77.8%, val=skipped)
  - Reason: accepted fix regressed: cccccccccccccccccccccccccccccccc
- **serving_iteration_03** → rejected (train=77.8%, val=skipped)
  - Reason: pass guard regressed: 33333333333333333333333333333333
- **serving_iteration_04** → rejected (train=77.8%, val=skipped)
  - Reason: accepted fix regressed: 22222222222222222222222222222222
"""

    assert count_prior_accepted_fix_regressions(context) == 2


def test_count_prior_rejected_candidates():
    context = """
## Prior Optimization Attempts
- **serving_iteration_01** → rejected (train=0.0%, val=skipped)
  - Reason: train_health_failed: 44.4% -> 0.0%
- **serving_iteration_02** → accepted (train=55.6%, val=50.0%)
- **serving_iteration_03** → rejected (train=22.2%, val=skipped)
"""

    assert count_prior_rejected_candidates(context) == 2


def test_count_prior_rejected_candidate_modes():
    context = """
## Prior Optimization Attempts
- **serving_iteration_01** → rejected (train=0.0%, val=skipped)
  - Reason: train_health_failed: 44.4% -> 0.0%
  - Candidate mode: `example_set`
- **serving_iteration_02** → accepted (train=55.6%, val=50.0%)
  - Candidate mode: `metadata`
- **serving_iteration_03** → rejected (train=22.2%, val=skipped)
  - Candidate mode: `example_set`
- **serving_iteration_04** → rejected (train=44.4%, val=41.7%)
  - Candidate mode: `sql_snippet`
"""

    assert count_prior_rejected_candidate_modes(context) == {
        "example_set": 2,
        "sql_snippet": 1,
    }


def test_count_prior_invalid_responses():
    context = """
## Prior Optimization Attempts
- **serving_iteration_03** → skipped (serving proposal invalid)
  - Reason: invalid_serving_response_json
- **serving_iteration_04** → accepted (train=77.8%, val=66.7%)
"""

    assert count_prior_invalid_responses(context) == 1


def test_count_prior_skipped_candidates():
    context = """
## Prior Optimization Attempts
- **serving_iteration_01** → rejected (train=0.0%, val=skipped)
- **serving_iteration_02** → skipped (train=44.4%, val=41.7%)
- **serving_iteration_03** → skipped (serving proposal invalid)
"""

    assert count_prior_skipped_candidates(context) == 2


def test_baseline_pass_guard_ids_extracts_visible_guard_ids():
    context = """
## Baseline Pass Guard Patterns

### Guard: give me sales of Lokelma?
- ID: `guard-one`
- Latest status after rollback/best checkpoint: `passed`

### Guard: TRx for Lokelma
- ID: `guard-two`
- Latest status after rollback/best checkpoint: `passed`
"""

    assert baseline_pass_guard_ids(context) == ("guard-one", "guard-two")


def test_baseline_pass_guard_questions_extracts_question_text():
    context = """
## Baseline Pass Guard Patterns

### Guard: NRx for Farxiga in current 13 weeks
- ID: `guard-one`
- Latest status after rollback/best checkpoint: `passed`

### Guard: TRx for Lokelma
- ID: `guard-two`
- Latest status after rollback/best checkpoint: `passed`
"""

    assert baseline_pass_guard_questions(context) == {
        "guard-one": "NRx for Farxiga in current 13 weeks",
        "guard-two": "TRx for Lokelma",
    }


def test_canonicalize_tagged_example_sql_uses_visible_benchmark_sql():
    payload = {
        "candidate_name": "copy_drift",
        "operations": [
            {
                "op": "add_example_sql",
                "id": "question-a",
                "reason": "target_train_failure:question-a",
                "question": "Wrong wording",
                "sql": None,
            }
        ],
    }

    canonicalized, summary = canonicalize_tagged_example_sql(
        payload,
        benchmark_by_question_id={
            "question-a": {
                "question": "Expected question",
                "expected_sql": "SELECT expected_value FROM t",
            }
        },
    )

    operation = canonicalized["operations"][0]
    assert operation["question"] == "Expected question"
    assert operation["sql"] == "SELECT expected_value FROM t"
    assert summary["canonicalized_example_sql_ids"] == ["question-a"]
    assert payload["operations"][0]["sql"] is None


def test_repair_example_set_pass_guards_adds_visible_missing_guard():
    payload = {
        "candidate_name": "stale_replay",
        "rationale": "Preserve pass guards and fix one train failure.",
        "expected_impact": "Improves one visible failure.",
        "risk": "Preserves pass guard SQL patterns.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Failed question",
                "sql": "SELECT 3",
                "reason": "target_train_failure:failed-one",
            },
        ],
    }

    repaired, repair_summary = repair_example_set_pass_guards(
        payload,
        benchmark_by_question_id={
            "guard-two": {
                "question": "Guard two",
                "expected_sql": "SELECT expected_guard_two",
            }
        },
        required_pass_guard_ids=("guard-one", "guard-two"),
        max_mutation_operations=3,
    )

    assessment = assess_serving_candidate_risk(
        repaired,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
        required_pass_guard_ids=("guard-one", "guard-two"),
    )

    assert repair_summary == {
        "repaired_pass_guard_ids": ["guard-two"],
        "unrepaired_pass_guard_ids": [],
        "pruned_duplicate_operation_count": 0,
    }
    assert repaired["operations"][-1] == {
        "op": "add_example_sql",
        "id": "guard-two",
        "question": "Guard two",
        "sql": "SELECT expected_guard_two",
        "usage_guidance": (
            "Preserve this visible pass-guard benchmark pattern before targeting remaining failures."
        ),
        "reason": "guard_preservation:guard-two",
    }
    assert assessment["accepted"] is True
    assert len(payload["operations"]) == 2


def test_repair_example_set_pass_guards_reports_capacity_blocker():
    payload = {
        "candidate_name": "full_replay",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Failed question",
                "sql": "SELECT 2",
                "reason": "target_train_failure:failed-one",
            },
        ],
    }

    repaired, repair_summary = repair_example_set_pass_guards(
        payload,
        benchmark_by_question_id={
            "guard-two": {
                "question": "Guard two",
                "expected_sql": "SELECT expected_guard_two",
            }
        },
        required_pass_guard_ids=("guard-one", "guard-two"),
        max_mutation_operations=2,
    )

    assert repair_summary == {
        "repaired_pass_guard_ids": [],
        "unrepaired_pass_guard_ids": ["guard-two"],
        "pruned_duplicate_operation_count": 0,
    }
    assert repaired == payload


def test_assess_serving_candidate_risk_rejects_example_set_missing_guard_examples():
    payload = {
        "candidate_name": "partial_guard_bundle",
        "rationale": "Preserve pass guards and fix one train failure.",
        "expected_impact": "Improves one visible failure.",
        "risk": "Preserves pass guard SQL patterns.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Failed question",
                "sql": "SELECT 2",
                "reason": "target_train_failure:failed-one",
            },
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
        required_pass_guard_ids=("guard-one", "guard-two"),
    )

    assert assessment["accepted"] is False
    assert "missing_pass_guard_examples:guard-two" in assessment["reasons"]
    assert "too_few_example_set_operations:2<3" in assessment["reasons"]


def test_assess_serving_candidate_risk_accepts_example_set_covering_all_guards_and_failure():
    payload = {
        "candidate_name": "complete_guard_bundle",
        "rationale": "Preserve pass guards and fix one train failure.",
        "expected_impact": "Improves one visible failure.",
        "risk": "Preserves pass guard SQL patterns.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Guard two",
                "sql": "SELECT 2",
                "reason": "guard_preservation:guard-two",
            },
            {
                "op": "add_example_sql",
                "question": "Failed question",
                "sql": "SELECT 3",
                "reason": "target_train_failure:failed-one",
            },
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
        required_pass_guard_ids=("guard-one", "guard-two"),
    )

    assert assessment["accepted"] is True


def test_assess_serving_candidate_risk_allows_same_family_example_set_targets():
    payload = {
        "candidate_name": "compatible_guard_bundle",
        "rationale": "Preserve pass guards and fix compatible train failures.",
        "expected_impact": "Improves two visible failures from one root cause.",
        "risk": "Preserves pass guard SQL patterns.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Target one",
                "sql": "SELECT 2",
                "reason": "target_train_failure:failed-one",
            },
            {
                "op": "add_example_sql",
                "question": "Target two",
                "sql": "SELECT 3",
                "reason": "target_train_failure:failed-two",
            },
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
        required_pass_guard_ids=("guard-one",),
        failure_family_by_question_id={
            "failed-one": "percentage_or_comparison_metric",
            "failed-two": "percentage_or_comparison_metric",
        },
        guard_regression_risk_by_question_id={
            "failed-one": "medium",
            "failed-two": "medium",
        },
    )

    assert assessment["accepted"] is True


def test_assess_serving_candidate_risk_rejects_mixed_family_example_set_targets():
    payload = {
        "candidate_name": "mixed_family_guard_bundle",
        "rationale": "Preserve pass guards and fix train failures.",
        "expected_impact": "Improves two unrelated visible failures.",
        "risk": "Preserves pass guard SQL patterns.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Target one",
                "sql": "SELECT 2",
                "reason": "target_train_failure:failed-one",
            },
            {
                "op": "add_example_sql",
                "question": "Target two",
                "sql": "SELECT 3",
                "reason": "target_train_failure:failed-two",
            },
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
        required_pass_guard_ids=("guard-one",),
        failure_family_by_question_id={
            "failed-one": "percentage_or_comparison_metric",
            "failed-two": "table_or_join_path_mismatch",
        },
        guard_regression_risk_by_question_id={
            "failed-one": "medium",
            "failed-two": "high",
        },
    )

    assert assessment["accepted"] is False
    assert (
        "mixed_target_failure_families:"
        "failed-one=percentage_or_comparison_metric,failed-two=table_or_join_path_mismatch"
    ) in assessment["reasons"]
    assert "multiple_high_guard_risk_targets:failed-two" in assessment["reasons"]


def test_assess_serving_candidate_risk_rejects_too_many_example_set_targets():
    payload = {
        "candidate_name": "overbroad_guard_bundle",
        "rationale": "Preserve pass guards and fix many train failures.",
        "expected_impact": "Improves several visible failures.",
        "risk": "Preserves pass guard SQL patterns.",
        "operations": [
            {
                "op": "add_example_sql",
                "question": "Guard one",
                "sql": "SELECT 1",
                "reason": "guard_preservation:guard-one",
            },
            {
                "op": "add_example_sql",
                "question": "Target one",
                "sql": "SELECT 2",
                "reason": "target_train_failure:failed-one",
            },
            {
                "op": "add_example_sql",
                "question": "Target two",
                "sql": "SELECT 3",
                "reason": "target_train_failure:failed-two",
            },
            {
                "op": "add_example_sql",
                "question": "Target three",
                "sql": "SELECT 4",
                "reason": "target_train_failure:failed-three",
            },
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("example_set"),
        require_guardrail_acknowledgement=True,
        max_mutation_operations=serving_candidate_max_mutation_operations("example_set"),
        required_pass_guard_ids=("guard-one",),
    )

    assert assessment["accepted"] is False
    assert "too_many_target_failure_examples:3>2" in assessment["reasons"]


def test_assess_serving_candidate_risk_rejects_filter_snippet_overlapping_pass_guard():
    payload = {
        "candidate_name": "farxiga_filter",
        "rationale": "Fix explicit Farxiga filtering.",
        "expected_impact": "Improves Farxiga failures.",
        "risk": "Preserves guards.",
        "operations": [
            {
                "op": "add_sql_snippet",
                "group": "filters",
                "display_name": "Farxiga literal brand filter",
                "column_name": "brand_name",
                "synonyms": ["Farxiga"],
                "sql": "LOWER(brand_name) = 'farxiga'",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("sql_snippet"),
        max_mutation_operations=serving_candidate_max_mutation_operations("sql_snippet"),
        pass_guard_questions_by_id={"guard-one": "NRx for Farxiga in Current 13 weeks"},
    )

    assert assessment["accepted"] is False
    assert "sql_snippet_overlaps_pass_guard:guard-one:farxiga" in assessment["reasons"]


def test_build_serving_candidate_prompt_prioritizes_full_guard_context():
    failure_context = "\n".join(
        [
            "# Failure Context",
            "## Failed Questions (1)",
            "x" * 1000,
            "## Baseline Pass Guard Patterns",
            "### Guard: Guard one question",
            "- ID: `guard-one`",
            "",
            "**Baseline Expected SQL Pattern:**",
            "```sql",
            "SELECT * FROM important_guard_query",
            "```",
            "",
            "## Prior Optimization Attempts",
            "- **serving_iteration_01** \u2192 rejected",
        ]
    )

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context=failure_context,
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="example_set",
    )

    guard_section_index = prompt.index("## Required Baseline Pass Guard Context")
    latest_context_index = prompt.index("## Latest Failure Context")
    assert guard_section_index < latest_context_index
    assert "guard-one" in prompt
    assert "SELECT * FROM important_guard_query" in prompt
    assert "do not claim guard IDs are unavailable" in prompt
    assert "set id to the benchmark question_id and sql to null" in prompt
    assert "remaining compatible failed benchmark IDs" in prompt
    assert "not previously fixed target_train_failure IDs" in prompt
    assert "Do not copy long benchmark SQL into example_set responses" in prompt
    assert "Outside example-set modes, add_example_sql must return exactly one literal example query" in prompt
    assert "Use exactly one operation group per candidate: metadata OR SQL examples" in prompt
    assert "OR one SQL example OR" not in prompt


def test_build_serving_candidate_prompt_includes_artifact_patch_context():
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/text_instruction.md",
                    "sha256": "a" * 64,
                    "content": "Existing instruction",
                }
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch",
    )

    assert "## Decomposed Space Files" in prompt
    assert "Use top-level file_edits only; set operations to an empty array" in prompt
    assert "instructions/text_instruction.md" in prompt
    assert "expected_sha256" in prompt
    assert "Artifact patch mode is the only mode that may use file_edits" in prompt


def test_artifact_patch_v2_mode_is_strict_file_edit_mode():
    assert serving_candidate_allowed_operation_groups("artifact_patch_v2") == frozenset({"artifact_patch"})
    assert serving_candidate_max_mutation_operations("artifact_patch_v2") is None
    assert serving_candidate_is_artifact_patch_mode("artifact_patch_v2") is True
    assert serving_candidate_is_strict_artifact_patch_mode("artifact_patch_v2") is True
    assert serving_candidate_artifact_patch_file_edit_limits("artifact_patch_v2") == (
        None,
        2,
    )

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/text_instruction.md",
                    "sha256": "a" * 64,
                    "content": "Existing instruction",
                }
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "Mode: artifact_patch_v2" in prompt
    assert "return no_change using the required candidate-mode contract" in prompt
    assert "Return at most two decomposed-space file edits" in prompt
    assert "use exactly one edit primitive per file edit" in prompt
    assert "set unused nullable file-edit fields to null" in prompt
    assert "path and relative_path must not disagree" in prompt
    assert "copy expected_sha256 exactly from the current Decomposed Space Files entry" in prompt
    assert "use compact append_content or exact find/replace edits" in prompt
    assert (
        "full-file content rewrites are rejected for visible-context causal-plan patches"
        in prompt
    )
    assert "Each edit must include path" in prompt
    assert "selected visible-effect marker" in prompt
    assert "target_train_failure:<question_id>" in prompt
    assert "visible_family:<family>" in prompt
    assert "guard_preservation:<question_id>" in prompt
    assert "strict artifact_patch_v2 validation rejects trusted-query routing edits" in prompt
    assert "do not add new bind placeholders" in prompt
    assert (
        "Do not introduce colon-prefixed placeholder tokens anywhere in changed file content"
        in prompt
    )
    assert "do not encode generated query plans" in prompt
    assert "For artifact_patch_v2 no_change, set operations to []" in prompt
    assert "Do not include a no_change operation and do not set file_edits to null" in prompt
    assert ":time_grp_type" in prompt
    assert "full content for a new executable YAML file" in prompt
    assert "examples require non-empty question and sql fields" in prompt
    assert "SQL snippets require non-empty string/list sql" in prompt
    assert "Do not put a top-level content: key inside that YAML" in prompt
    assert "the example question must share meaningful non-generic terms" in prompt
    assert "unrelated example questions are rejected before benchmarking" in prompt


def test_broad_artifact_patch_recipe_modes_are_hash_checked_file_edit_modes():
    modes = {
        "freedom_artifact_patch_v1": (None, 12),
        "freedom_artifact_patch_with_history_v1": (None, 12),
        "data_probe_artifact_patch_v1": (None, 6),
    }
    for mode, limits in modes.items():
        assert serving_candidate_allowed_operation_groups(mode) == frozenset({"artifact_patch"})
        assert serving_candidate_max_mutation_operations(mode) is None
        assert serving_candidate_is_artifact_patch_mode(mode) is True
        assert serving_candidate_is_strict_artifact_patch_mode(mode) is True
        assert serving_candidate_is_broad_artifact_patch_mode(mode) is True
        assert serving_candidate_artifact_patch_file_edit_limits(mode) == limits

    assert serving_candidate_uses_historical_analysis(
        "freedom_artifact_patch_with_history_v1"
    )
    assert not serving_candidate_uses_historical_analysis("freedom_artifact_patch_v1")
    assert serving_candidate_uses_data_probe("data_probe_artifact_patch_v1")
    assert not serving_candidate_uses_data_probe("freedom_artifact_patch_v1")


def test_freedom_history_prompt_includes_analyst_memo_and_broad_rules():
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/text_instruction.md",
                    "sha256": "a" * 64,
                    "expected_sha256": "a" * 64,
                    "content": "Existing instruction",
                }
            ],
        },
        historical_analyst_memo="Accepted executable examples helped; text-only rewrites collapsed train.",
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="freedom_artifact_patch_with_history_v1",
    )

    assert "Mode: freedom_artifact_patch_with_history_v1" in prompt
    assert "up to twelve changed files" in prompt
    assert "## Historical Analyst Memo" in prompt
    assert "text-only rewrites collapsed train" in prompt
    assert "Broad artifact-patch modes may bundle multiple compatible edits" in prompt
    assert "hidden holdout text, SQL, and failed-ID lists remain excluded" in prompt


def test_data_probe_prompt_includes_aggregate_probe_context():
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={"schema_version": "maxgenie.decomposed_space_context.v1", "files": []},
        data_probe_payload={
            "schema_version": "maxgenie.data_probe_context.v1",
            "probes": [
                {
                    "table": "catalog.schema.sales",
                    "column": "brand_name",
                    "summary": {"trim_mismatch_count": 12},
                }
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="data_probe_artifact_patch_v1",
    )

    assert "Mode: data_probe_artifact_patch_v1" in prompt
    assert "up to six changed files" in prompt
    assert "## Data Probe Diagnostics" in prompt
    assert "trim_mismatch_count" in prompt
    assert "row-level samples" in prompt


def test_skeleton_artifact_patch_v2_prompt_lists_bound_skeleton():
    target_id = "dddddddddddddddddddddddddddddddd"
    digest = "c" * 64
    structured_context = {
        "schema_version": 1,
        "failure_families": {
            "selected_metric_or_column_mismatch": [target_id],
        },
        "failures": [
            {
                "question_id": target_id,
                "question": "Show current and prior NRX for Farxiga",
                "issue_class": "column_mismatch",
                "comparison_method": "result_column_mismatch",
                "expected_sql": "SELECT current_nrx, prior_nrx FROM sales",
                "generated_sql": "SELECT current_nrx, time_bucket FROM sales",
                "likely_root_cause_family": "selected_metric_or_column_mismatch",
                "suggested_operation_families": ["set_column_config"],
                "guard_regression_risk": "medium",
                "missing_expected_tokens": ["prior_nrx"],
                "extra_generated_tokens": ["time_bucket"],
            }
        ],
        "pass_guards": [
            {
                "question_id": "b" * 32,
                "latest_status": "passed",
                "source": "baseline_validation",
            }
        ],
    }
    decomposed_payload = {
        "schema_version": "maxgenie.decomposed_space_context.v1",
        "files": [
            {
                "path": "tables/sales.yml",
                "sha256": digest,
                "expected_sha256": digest,
                "content": (
                    "identifier: sales\n"
                    "columns:\n"
                    "- column_name: prior_nrx\n"
                    "  description: Prior-period NRX value\n"
                ),
            }
        ],
    }

    skeletons = build_artifact_patch_skeletons(
        structured_failure_context=structured_context,
        decomposed_space_payload=decomposed_payload,
        candidate_mode="skeleton_artifact_patch_v2",
    )
    assert len(skeletons) == 1
    assert skeletons[0].path == "tables/sales.yml"
    assert skeletons[0].selected_marker == f"target_train_failure:{target_id}"
    assert skeletons[0].expected_sha256 == digest
    assert skeletons[0].find_anchor == "- column_name: prior_nrx"
    assert skeletons[0].required_visible_tokens == ("prior_nrx",)

    assert serving_candidate_allowed_operation_groups(
        "skeleton_artifact_patch_v2"
    ) == frozenset({"artifact_patch"})
    assert serving_candidate_is_artifact_patch_mode("skeleton_artifact_patch_v2")
    assert serving_candidate_is_strict_artifact_patch_mode("skeleton_artifact_patch_v2")
    assert serving_candidate_artifact_patch_file_edit_limits(
        "skeleton_artifact_patch_v2"
    ) == (1, 1)

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context=structured_context,
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload=decomposed_payload,
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="skeleton_artifact_patch_v2",
    )

    assert "Mode: skeleton_artifact_patch_v2" in prompt
    assert "### Deterministic Patch Skeleton Contract" in prompt
    assert "Serving may fill only `file_edits[0].replace`" in prompt
    assert "Do not choose a different path, marker, family, edit mode, or find anchor" in prompt
    assert "skeleton_id=`skel_01_dddddddd`" in prompt
    assert f"selected_marker=`target_train_failure:{target_id}`" in prompt
    assert "path=`tables/sales.yml`" in prompt
    assert f"expected_sha256=`{digest}`" in prompt
    assert 'find_anchor_json="- column_name: prior_nrx"' in prompt
    assert "required_visible_tokens=`prior_nrx`" in prompt
    assert "required_visible_intent_labels=`restore_expected_result_columns`" in prompt
    assert "metadata_query_plan_or_result_schema" in prompt
    assert "preserve guard_preservation:" in prompt


def test_validate_artifact_patch_skeleton_binding_rejects_drift():
    skeleton = build_artifact_patch_skeletons(
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "comparison_method": "result_column_mismatch",
                }
            ],
            "pass_guards": [],
        },
        decomposed_space_payload={
            "files": [
                {
                    "path": "tables/sales.yml",
                    "expected_sha256": "d" * 64,
                    "content": "- column_name: prior_nrx\n  description: Old\n",
                }
            ]
        },
        candidate_mode="skeleton_artifact_patch_v2",
    )[0]

    accepted = validate_artifact_patch_skeleton_binding(
        {
            "candidate_name": "bound_patch",
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": skeleton.selected_marker,
                "family": skeleton.family,
                "expected_visible_effect": "restore_expected_result_columns prior_nrx",
                "repair_shape": "metadata column repair",
                "root_cause": "metric column mismatch",
            },
            "file_edits": [
                {
                    "path": skeleton.path,
                    "expected_sha256": skeleton.expected_sha256,
                    "find": skeleton.find_anchor,
                    "replace": "- column_name: prior_nrx\n  description: Prior NRX value",
                    "content": None,
                    "append_content": None,
                }
            ],
        },
        skeletons=[skeleton],
    )
    assert accepted["accepted"] is True
    assert accepted["selected_skeleton_id"] == skeleton.skeleton_id

    rejected = validate_artifact_patch_skeleton_binding(
        {
            "candidate_name": "drifted_patch",
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": skeleton.selected_marker,
                "family": skeleton.family,
                "expected_visible_effect": "restore_expected_result_columns prior_nrx",
            },
            "file_edits": [
                {
                    "path": skeleton.path,
                    "expected_sha256": skeleton.expected_sha256,
                    "find": "description: Old",
                    "replace": "description: Prior NRX value",
                    "content": None,
                    "append_content": None,
                }
            ],
        },
        skeletons=[skeleton],
    )
    assert rejected["accepted"] is False
    assert "skeleton_artifact_patch_no_matching_skeleton" in rejected["reasons"]


def test_artifact_patch_v2_prompt_lists_exact_visible_effect_markers():
    target_id = "dddddddddddddddddddddddddddddddd"
    guard_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context={
            "schema_version": 1,
            "train_pass_rate": 22.22,
            "best_train_rate": 33.33,
            "best_validation_rate": 25.0,
            "failed_question_count": 1,
            "failure_families": {
                "literal_or_parameter_mismatch": [target_id],
            },
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "guard_regression_risk": "high",
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                    "question": "Visible validation guard question",
                    "expected_sql": "SELECT guard_metric FROM visible_guard",
                }
            ],
            "hidden_holdout_diagnostics": {
                "question": "HIDDEN_DO_NOT_LEAK holdout question",
                "sql": "SELECT HIDDEN_DO_NOT_LEAK",
            },
        },
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/text_instruction.md",
                    "sha256": "a" * 64,
                    "expected_sha256": "a" * 64,
                    "content": "Existing instruction",
                }
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "### Artifact Patch Visible-Effect Contract" in prompt
    assert "Every changed strict file edit must cite one exact marker" in prompt
    assert "top-level rationale/expected_impact is not enough" in prompt
    assert "in each changed edit reason, description, or comment" in prompt
    assert "in each edit reason or top-level rationale/expected_impact" not in prompt
    assert "If the chosen target has guard_risk=`high`" in prompt
    assert f"`target_train_failure:{target_id}`" in prompt
    assert "guard_risk=`high`" in prompt
    assert "For text-guidance edits, mention at least one listed structural token" in prompt
    assert "Guard-preservation markers are support markers only" in prompt
    assert "`'farxiga'`" in prompt
    assert "`:brand_name`" in prompt
    assert "`visible_family:literal_or_parameter_mismatch`" in prompt
    assert "`failure_family:literal_or_parameter_mismatch`" in prompt
    assert f"`guard_preservation:{guard_id}`" in prompt
    assert "evidence=(question=\"Visible validation guard question\"" in prompt
    assert "expected_sql=`SELECT guard_metric FROM visible_guard`" in prompt
    assert "Do not invent marker IDs" in prompt
    assert "If no marker below fits the patch, return no_change" in prompt
    assert "### Strict Artifact Patch Target Plan" in prompt
    assert "Choose exactly one plan below before creating file_edits" in prompt
    assert "Mutating default artifact_patch_v2 responses must include" in prompt
    assert "top-level causal_patch_plan" in prompt
    assert (
        "selected_marker, family, root_cause, edit_surface_evidence, repair_shape, avoid"
        in prompt
    )
    assert "selected_marker must be an exact visible marker string" in prompt
    assert "not a raw question ID" in prompt
    assert "family must be non-empty and match the chosen marker or target family" in prompt
    assert "root_cause must be non-empty" in prompt
    assert "root_cause_anchors value" in prompt
    assert "edit_surface_evidence must be non-empty" in prompt
    assert "edit_surface_anchors value" in prompt
    assert "compatible_paths must be non-empty and copied from the chosen plan" in prompt
    assert "repair_shape and avoid must be non-empty" in prompt
    assert (
        "Copy structural_tokens only from the chosen plan's shared_token_hints when shared_token_hints are listed"
        in prompt
    )
    assert (
        "otherwise copy structural_tokens only from that plan's token_hints"
        in prompt
    )
    assert (
        "For target-level executable examples or SQL snippets, prefer expected_token_hints over generated_token_hints"
        in prompt
    )
    assert "include one expected token in structural_tokens and changed content" in prompt
    assert (
        "For target-level literal_or_parameter_mismatch executable examples or SQL snippets"
        in prompt
    )
    assert "using only the generated placeholder or column token is rejected" in prompt
    assert "expected_visible_effect name the concrete visible SQL/result delta" in prompt
    assert "sql_delta_intents are normalized visible SQL/result delta labels" in prompt
    assert "expected_visible_effect should name one chosen intent" in prompt
    assert "including one shared_token_hints value when shared_token_hints are listed" in prompt
    assert "otherwise including one token_hints value when token_hints are listed" in prompt
    assert "include one listed structural_tokens value in the new content" in prompt
    assert "Each changed file_edits[].reason, description, or comment must use" in prompt
    assert "no_change with causal_patch_plan set to null" in prompt
    assert f"marker=`target_train_failure:{target_id}`" in prompt
    assert "compatible_paths=`instructions/sql_snippets/filters/*.yml`" in prompt
    assert "existing_edit_files=(none)" in prompt
    assert "new_file_allowed=`true`" in prompt
    assert "sql_delta_intents=`literalize_expected_value`, `remove_bind_placeholder`" in prompt
    assert "expected_token_hints=`'farxiga'`" in prompt
    assert "generated_token_hints=`:brand_name`" in prompt
    assert "token_hints=`'farxiga'`, `:brand_name`" in prompt
    assert "root_cause_anchors=`literal`, `parameter`, `placeholder`" in prompt
    assert "edit_surface_anchors=`snippet`, `fragment`, `filter`" in prompt
    assert "Causal repair recipe" in prompt
    assert "Add one literal SQL fragment" in prompt
    assert "Avoid: bind placeholders" in prompt
    assert "prefer expected_token_hints and include an expected token" in prompt
    assert "expected visible literal rather than only the generated placeholder" in prompt
    assert (
        "High guard risk: include a visible guard_preservation:<question_id>"
        in prompt
    )
    assert "### Strict Artifact Patch Self-Checklist" in prompt
    assert "Do not add checklist fields to the response" in prompt
    assert "Hidden holdout exclusion" in prompt
    assert "Selected marker attribution" in prompt
    assert "every changed file_edit reason, description, or comment repeats that same marker" in prompt
    assert "Patch shape: file_edits follow the chosen plan's patch archetype and patch template hint" in prompt
    assert "Executable YAML shape" in prompt
    assert "parse as YAML mappings with required non-empty fields before benchmarking" in prompt
    assert "do not wrap the YAML in a top-level content: key" in prompt
    assert "Expected-token safety" in prompt
    assert "shared_expected_token_hints value" in prompt
    assert "Pass-guard safety" in prompt
    assert "Non-clone safety" in prompt
    assert "return no_change with causal_patch_plan set to null" in prompt
    assert "HIDDEN_DO_NOT_LEAK" not in prompt


def test_artifact_patch_v2_prompt_lists_matching_existing_edit_files():
    target_id = "dddddddddddddddddddddddddddddddd"
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context={
            "schema_version": 1,
            "failure_families": {
                "selected_metric_or_column_mismatch": [target_id],
            },
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Show current and prior NRX for Farxiga",
                    "issue_class": "column_mismatch",
                    "comparison_method": "result_column_mismatch",
                    "expected_sql": "SELECT current_nrx, prior_nrx FROM sales",
                    "generated_sql": "SELECT current_nrx, time_bucket FROM sales",
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["time_bucket"],
                }
            ],
            "pass_guards": [],
        },
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "tables/market.yml",
                    "sha256": "b" * 64,
                    "expected_sha256": "b" * 64,
                    "content": "columns:\n- column_name: market_name\n",
                },
                {
                    "path": "tables/sales.yml",
                    "sha256": "a" * 64,
                    "expected_sha256": "a" * 64,
                    "content": (
                        "identifier: sales\n"
                        "columns:\n"
                        "- column_name: current_nrx\n"
                        "- column_name: prior_nrx\n"
                        "  description: Prior-period NRX value\n"
                    ),
                },
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "compatible_paths=`tables/*.yml`" in prompt
    assert "existing_edit_files=`tables/sales.yml`, `tables/market.yml`" in prompt
    assert "new_file_allowed=`false`" in prompt
    assert "target_question_terms=`current`, `farxiga`, `nrx`, `prior`" in prompt
    assert "If existing_edit_files are listed, prefer one of those exact paths" in prompt
    assert "if new_file_allowed=false, do not invent a new path" in prompt
    assert "Patch archetype lines are guidance only" in prompt
    assert "Visible target evidence:" in prompt
    assert "question=\"Show current and prior NRX for Farxiga\"" in prompt
    assert "expected_sql=`SELECT current_nrx, prior_nrx FROM sales`" in prompt
    assert "generated_sql=`SELECT current_nrx, time_bucket FROM sales`" in prompt
    assert (
        "Patch archetype: compactly edit one listed existing table or metric metadata file"
        in prompt
    )
    assert (
        "Patch template hint: existing table or metric YAML only: use exact `find` plus `replace`"
        in prompt
    )
    assert "Current edit-file excerpts for compact find/replace" in prompt
    assert f"path=`tables/sales.yml`; expected_sha256=`{'a' * 64}`" in prompt
    assert "Prior-period NRX value" in prompt


def test_artifact_patch_v2_prompt_lists_visible_family_target_plans():
    first_id = "dddddddddddddddddddddddddddddddd"
    second_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    third_id = "11111111111111111111111111111111"
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context={
            "schema_version": 1,
            "failure_families": {
                "literal_or_parameter_mismatch": [first_id, second_id, third_id],
            },
            "failures": [
                {
                    "question_id": first_id,
                    "question": "Show NRX for Farxiga in the current quarter",
                    "comparison_method": "result_unbound_parameters_error",
                    "expected_sql": "SELECT nrx FROM sales WHERE brand = 'farxiga'",
                    "generated_sql": "SELECT nrx FROM sales WHERE brand = :brand_name",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "guard_regression_risk": "high",
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                },
                {
                    "question_id": second_id,
                    "question": "Show TRX for Lokelma in the current quarter",
                    "comparison_method": "result_unbound_parameters_error",
                    "expected_sql": "SELECT trx FROM sales WHERE brand = 'lokelma'",
                    "generated_sql": "SELECT trx FROM sales WHERE brand = :brand_name",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["'lokelma'"],
                    "extra_generated_tokens": [":brand_name"],
                },
                {
                    "question_id": third_id,
                    "question": "Show market sales by region",
                    "comparison_method": "result_column_mismatch",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["market_sales"],
                },
            ],
            "pass_guards": [],
        },
        visible_benchmark_payload={"question_count": 3, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "Family-level plans below are only for repeated visible failures" in prompt
    assert (
        "Family Plan 1: marker=`visible_family:literal_or_parameter_mismatch`"
        in prompt
    )
    assert f"member_ids=`{first_id}`, `{second_id}`" in prompt
    assert f"member_ids=`{first_id}`, `{second_id}`, `{third_id}`" not in prompt
    assert f"guard_risk=`high`; high_guard_risk_member_ids=`{first_id}`" in prompt
    assert "compatible_paths=`instructions/sql_snippets/filters/*.yml`" in prompt
    assert "new_file_allowed=`true`" in prompt
    assert "High family guard risk: include a visible guard_preservation:<question_id>" in prompt
    assert (
        "member_question_terms=`current`, `farxiga`, `lokelma`, `nrx`, `quarter`, `trx`"
        in prompt
    )
    assert (
        "sql_delta_intents=`literalize_expected_value`, `produce_executable_sql`, `remove_bind_placeholder`"
        in prompt
    )
    assert "expected_token_hints=`'farxiga'`, `'lokelma'`" in prompt
    assert "generated_token_hints=`:brand_name`" in prompt
    assert "token_hints=`'farxiga'`, `'lokelma'`, `:brand_name`" in prompt
    assert "shared_expected_token_hints=(none)" in prompt
    assert "shared_generated_token_hints=`:brand_name`" in prompt
    assert "shared_token_hints=`:brand_name`" in prompt
    assert "prefer shared_expected_token_hints over shared_generated_token_hints" in prompt
    assert (
        "Patch archetype: create one SQL-snippet filter asset with display_name and one literal-safe predicate or fragment"
        in prompt
    )
    assert (
        "Patch template hint: new `instructions/sql_snippets/<filters|expressions|measures>/<name>.yml`"
        in prompt
    )
    assert "`display_name: <short name>` and `sql: |`" in prompt
    assert "do not anchor only to generated token(s) `:brand_name`" in prompt
    assert "Visible family evidence: failure_count=`2`" in prompt
    assert "sample_questions=\"Show NRX for Farxiga in the current quarter" in prompt
    assert "methods=`result_unbound_parameters_error`" in prompt
    assert "Visible shared-member evidence:" in prompt
    assert f"id=`{first_id}`; question=\"Show NRX for Farxiga" in prompt
    assert "expected_sql=`SELECT nrx FROM sales WHERE brand = 'farxiga'`" in prompt
    assert "generated_sql=`SELECT nrx FROM sales WHERE brand = :brand_name`" in prompt
    assert (
        "Shared family token evidence: choose at least one shared_token_hints value for structural_tokens and expected_visible_effect"
        in prompt
    )
    assert f"id=`{third_id}`" not in prompt
    assert "Causal repair recipe" in prompt


def test_artifact_patch_v2_prompt_lists_shared_expected_family_tokens():
    first_id = "dddddddddddddddddddddddddddddddd"
    second_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context={
            "schema_version": 1,
            "failure_families": {
                "selected_metric_or_column_mismatch": [first_id, second_id],
            },
            "failures": [
                {
                    "question_id": first_id,
                    "question": "Show prior NRX for Farxiga",
                    "comparison_method": "result_column_mismatch",
                    "expected_sql": "SELECT prior_nrx FROM sales",
                    "generated_sql": "SELECT current_nrx FROM sales",
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                },
                {
                    "question_id": second_id,
                    "question": "Show prior NRX for Lokelma",
                    "comparison_method": "result_column_mismatch",
                    "expected_sql": "SELECT prior_nrx FROM sales",
                    "generated_sql": "SELECT current_nrx FROM sales",
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                },
            ],
            "pass_guards": [],
        },
        visible_benchmark_payload={"question_count": 2, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert (
        "Family Plan 1: marker=`visible_family:selected_metric_or_column_mismatch`"
        in prompt
    )
    assert "expected_token_hints=`prior_nrx`" in prompt
    assert "generated_token_hints=`current_nrx`" in prompt
    assert (
        "sql_delta_intents=`remove_unrequested_result_columns`, `restore_expected_result_columns`"
        in prompt
    )
    assert "shared_expected_token_hints=`prior_nrx`" in prompt
    assert "shared_generated_token_hints=`current_nrx`" in prompt
    assert (
        "Patch archetype: create one SQL-snippet expression or measure asset with display_name"
        in prompt
    )
    assert (
        "Patch template hint: new `instructions/sql_snippets/<filters|expressions|measures>/<name>.yml`"
        in prompt
    )
    assert "using expected token(s) `prior_nrx`" in prompt
    assert "do not anchor only to generated token(s) `current_nrx`" in prompt
    assert (
        "Shared expected-token evidence: choose at least one shared_expected_token_hints value for executable structural_tokens and changed content."
        in prompt
    )


def test_artifact_patch_v2_prompt_skips_family_plan_without_shared_edit_surface():
    first_id = "dddddddddddddddddddddddddddddddd"
    second_id = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context={
            "schema_version": 1,
            "failure_families": {
                "selected_metric_or_column_mismatch": [first_id, second_id],
            },
            "failures": [
                {
                    "question_id": first_id,
                    "question": "Show NRX for Farxiga in the current quarter",
                    "comparison_method": "result_column_mismatch",
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["prior_nrx"],
                },
                {
                    "question_id": second_id,
                    "question": "Show TRX for Lokelma in the current quarter",
                    "comparison_method": "result_column_mismatch",
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["prior_trx"],
                },
            ],
            "pass_guards": [],
        },
        visible_benchmark_payload={"question_count": 2, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "Family-level plans below are only for repeated visible failures" not in prompt
    assert "Family Plan 1: marker=`visible_family:selected_metric_or_column_mismatch`" not in prompt
    assert "marker=`target_train_failure:" in prompt


def test_composite_artifact_patch_v2_mode_is_bounded_multi_file_patch_mode():
    assert serving_candidate_allowed_operation_groups("composite_artifact_patch_v2") == frozenset(
        {"artifact_patch"}
    )
    assert serving_candidate_max_mutation_operations("composite_artifact_patch_v2") is None
    assert serving_candidate_is_artifact_patch_mode("composite_artifact_patch_v2") is True
    assert serving_candidate_is_strict_artifact_patch_mode("composite_artifact_patch_v2") is True
    assert serving_candidate_artifact_patch_file_edit_limits(
        "composite_artifact_patch_v2"
    ) == (2, 4)

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/text_instruction.md",
                    "sha256": "a" * 64,
                    "expected_sha256": "a" * 64,
                    "content": "Existing instruction",
                },
                {
                    "path": "tables/sales.yml",
                    "sha256": "b" * 64,
                    "expected_sha256": "b" * 64,
                    "content": "identifier: sales\n",
                },
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="composite_artifact_patch_v2",
    )

    assert "Mode: composite_artifact_patch_v2" in prompt
    assert "two to four decomposed-space file edits" in prompt
    assert "one-file patches are not valid" in prompt
    assert "one executable visible example, one narrow instruction/guidance edit" in prompt
    assert "April/p02b pattern lessons" in prompt
    assert "Do not use hidden holdout text, SQL, query text, or failed-ID lists" in prompt
    assert "do not return a one-file generic instruction rewrite" in prompt


def test_executable_artifact_patch_v2_mode_requires_executable_assets():
    assert serving_candidate_allowed_operation_groups("executable_artifact_patch_v2") == frozenset(
        {"artifact_patch"}
    )
    assert serving_candidate_max_mutation_operations("executable_artifact_patch_v2") is None
    assert serving_candidate_is_artifact_patch_mode("executable_artifact_patch_v2") is True
    assert serving_candidate_is_strict_artifact_patch_mode("executable_artifact_patch_v2") is True
    assert serving_candidate_artifact_patch_requires_executable_asset(
        "executable_artifact_patch_v2"
    ) is True
    assert serving_candidate_artifact_patch_file_edit_limits(
        "executable_artifact_patch_v2"
    ) == (1, 3)
    assert serving_candidate_is_executable_artifact_path(
        "instructions/examples/example_visible_literal.yml"
    ) is True
    assert serving_candidate_is_executable_artifact_path(
        "instructions/sql_snippets/filters/time_window.yml"
    ) is True
    assert serving_candidate_is_executable_artifact_path(
        "instructions/text_instruction.md"
    ) is False
    assert serving_candidate_is_executable_artifact_path("tables/sales.yml") is False

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/text_instruction.md",
                    "sha256": "a" * 64,
                    "expected_sha256": "a" * 64,
                    "content": "Existing instruction",
                },
                {
                    "path": "instructions/examples/example_visible_literal.yml",
                    "sha256": "b" * 64,
                    "expected_sha256": "b" * 64,
                    "content": "id: visible_literal\nquestion: Example\nsql: SELECT 1\n",
                },
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="executable_artifact_patch_v2",
    )

    assert "Mode: executable_artifact_patch_v2" in prompt
    assert "At least one changed file must be an executable asset" in prompt
    assert "instruction-only and instruction-plus-metadata patches are invalid" in prompt
    assert "directly executable Databricks SQL with literal values" in prompt
    assert "must not be literal-swap clones of passing guards" in prompt
    assert "structurally near-identical to visible pass-guard SQL" in prompt
    assert "return no_change rather than another prose-only or metadata-only patch" in prompt


def test_benchmark_example_set_mode_is_explicit_structured_bundle_mode():
    assert serving_candidate_allowed_operation_groups("benchmark_example_set") == frozenset(
        {"example_sql"}
    )
    assert serving_candidate_max_mutation_operations("benchmark_example_set") == 9

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="benchmark_example_set",
    )

    assert "Mode: benchmark_example_set" in prompt
    assert "structured alternative to global text-instruction patches" in prompt
    assert "set id to a visible benchmark question_id and sql to null" in prompt
    assert "Do not copy long benchmark SQL into the response" in prompt
    assert "example-set modes" in prompt


def test_typed_template_mode_materializes_visible_templates_and_guards():
    guard_id = "a" * 32
    target_id = "b" * 32
    payload = {
        "candidate_name": "typed_template_period_fix",
        "rationale": "Target visible period-comparison family.",
        "expected_impact": "Adds executable visible template.",
        "risk": "Preserves visible pass guards.",
        "owner_followups": [],
        "operations": [
            {
                "op": "add_example_sql",
                "id": target_id,
                "question": None,
                "sql": None,
                "usage_guidance": None,
                "reason": f"target_train_failure:{target_id}",
            }
        ],
    }
    materialized, summary = materialize_typed_template_examples(
        payload,
        benchmark_by_question_id={
            guard_id: {
                "question": "Visible passing guard",
                "expected_sql": "SELECT 'guard' AS answer",
            },
            target_id: {
                "question": "Visible percentage difference failure",
                "expected_sql": "SELECT current_nrx, prior_nrx, percentage_difference FROM t",
            },
        },
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "guard_regression_risk": "medium",
                }
            ]
        },
        required_pass_guard_ids=(guard_id,),
        max_mutation_operations=9,
    )

    operations = materialized["operations"]
    assert [operation["id"] for operation in operations] == [guard_id, target_id]
    assert operations[0]["reason"] == f"guard_preservation:{guard_id}"
    assert operations[0]["sql"] == "SELECT 'guard' AS answer"
    assert operations[1]["reason"] == f"target_train_failure:{target_id}"
    assert operations[1]["sql"] == "SELECT current_nrx, prior_nrx, percentage_difference FROM t"
    assert "current-versus-previous period CTEs" in operations[1]["usage_guidance"]
    assert materialized["file_edits"] is None
    assert summary["typed_template_materialized_guard_ids"] == [guard_id]
    assert summary["typed_template_materialized_target_ids"] == [target_id]
    assert summary["typed_template_family_by_target_id"] == {
        target_id: "percentage_or_comparison_metric"
    }


def test_typed_template_mode_chooses_dominant_visible_family_when_sparse():
    literal_id = "c" * 32
    schema_id = "d" * 32
    materialized, summary = materialize_typed_template_examples(
        {
            "candidate_name": "sparse_typed_template",
            "rationale": "Endpoint picked the mode but omitted a target.",
            "expected_impact": "Materialize the top visible family.",
            "risk": "Guard preserving.",
            "owner_followups": [],
            "operations": [],
        },
        benchmark_by_question_id={
            literal_id: {
                "question": "Visible unbound parameter failure",
                "expected_sql": "SELECT * FROM t WHERE brand_name = 'FARXIGA'",
            },
            schema_id: {
                "question": "Visible output schema failure",
                "expected_sql": "SELECT time_grp_desc, region, nrx FROM t",
            },
        },
        structured_failure_context={
            "failures": [
                {
                    "question_id": schema_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "guard_regression_risk": "low",
                },
                {
                    "question_id": literal_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "guard_regression_risk": "medium",
                },
            ]
        },
        required_pass_guard_ids=(),
        max_mutation_operations=9,
    )

    operations = materialized["operations"]
    assert len(operations) == 1
    assert operations[0]["id"] == literal_id
    assert operations[0]["reason"] == f"target_train_failure:{literal_id}"
    assert "Never emit unbound :param placeholders" in operations[0]["usage_guidance"]
    assert summary["typed_template_materialized_target_ids"] == [literal_id]


def test_typed_template_mode_is_explicit_materialized_example_mode():
    assert serving_candidate_allowed_operation_groups("typed_template") == frozenset({"example_sql"})
    assert serving_candidate_max_mutation_operations("typed_template") == 9

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="typed_template",
    )

    assert "Mode: typed_template" in prompt
    assert "deterministically materialize executable visible SQL" in prompt
    assert "recurring visible result-error families" in prompt
    assert "Set id to a visible benchmark question_id and sql to null" in prompt
    assert "hidden holdout" in prompt.lower()


def test_schema_time_window_mode_is_reusable_snippet_successor_mode():
    assert serving_candidate_allowed_operation_groups("schema_time_window") == frozenset(
        {"sql_snippet"}
    )
    assert serving_candidate_max_mutation_operations("schema_time_window") == 1

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="schema_time_window",
    )

    assert "Mode: schema_time_window" in prompt
    assert "less benchmark-shaped successor to typed_template" in prompt
    assert "Allowed operation: add_sql_snippet" in prompt
    assert "Always set display_name" in prompt
    assert "do not add exact benchmark examples" in prompt
    assert "output-schema, metric-column, period-comparison, or time-window failures" in prompt
    assert "no bind placeholders" in prompt
    assert "not a full SELECT/WITH query" in prompt
    assert "not a menu or list of alternative predicates" in prompt
    assert "hidden holdout" in prompt.lower()


def test_deterministic_guidance_mode_is_single_visible_family_instruction_mode():
    assert serving_candidate_allowed_operation_groups("deterministic_guidance") == frozenset(
        {"text_instruction"}
    )
    assert serving_candidate_max_mutation_operations("deterministic_guidance") == 1

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="deterministic_guidance",
    )

    assert "Mode: deterministic_guidance" in prompt
    assert "Allowed operation: append_text_instruction" in prompt
    assert "exactly one concise instruction under 900 characters" in prompt
    assert "literal values instead of bind placeholders" in prompt
    assert "required output aliases" in prompt
    assert "period-comparison semantics" in prompt
    assert "time-window labels" in prompt
    assert "Do not add examples, SQL snippets, metadata, joins, file_edits" in prompt
    assert "copied benchmark SQL" in prompt
    assert "hidden holdout" in prompt.lower()


def test_schema_metadata_aliases_mode_is_guarded_structural_metadata_mode():
    assert serving_candidate_allowed_operation_groups("schema_metadata_aliases") == frozenset(
        {"metadata"}
    )
    assert serving_candidate_max_mutation_operations("schema_metadata_aliases") == 2

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 2, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="schema_metadata_aliases",
    )

    assert "Mode: schema_metadata_aliases" in prompt
    assert "Allowed operations: set_column_config or set_table_description" in prompt
    assert "Touch at most two existing tables or columns" in prompt
    assert "structural follow-up after examples, snippets, and text guidance plateaued" in prompt
    assert "ambiguous business columns" in prompt
    assert "required output column aliases" in prompt
    assert "Prefer column descriptions and synonyms over enabling entity matching" in prompt
    assert "Do not add examples, SQL snippets, joins, text instructions, file_edits" in prompt
    assert "hidden holdout" in prompt.lower()

    accepted = assess_serving_candidate_risk(
        {
            "candidate_name": "brand_metadata_aliases",
            "operations": [
                {
                    "op": "set_column_config",
                    "table_identifier": "catalog.schema.sales",
                    "column_name": "brand_name",
                    "description": "Commercial brand or product name used for exact literal filters.",
                    "synonyms": ["brand", "product"],
                },
                {
                    "op": "set_column_config",
                    "table_identifier": "catalog.schema.sales",
                    "column_name": "trx",
                    "description": "Total prescription transaction count.",
                    "synonyms": ["total prescriptions"],
                },
            ],
        },
        allowed_operation_groups=serving_candidate_allowed_operation_groups(
            "schema_metadata_aliases"
        ),
        max_mutation_operations=serving_candidate_max_mutation_operations(
            "schema_metadata_aliases"
        ),
    )
    assert accepted["accepted"] is True

    too_broad = assess_serving_candidate_risk(
        {
            "candidate_name": "too_many_metadata_edits",
            "operations": [
                {
                    "op": "set_column_config",
                    "table_identifier": "catalog.schema.sales",
                    "column_name": f"col_{index}",
                    "description": f"Description {index}",
                }
                for index in range(3)
            ],
        },
        allowed_operation_groups=serving_candidate_allowed_operation_groups(
            "schema_metadata_aliases"
        ),
        max_mutation_operations=serving_candidate_max_mutation_operations(
            "schema_metadata_aliases"
        ),
    )
    assert too_broad["accepted"] is False
    assert "too_many_operations:3" in too_broad["reasons"]

    wrong_group = assess_serving_candidate_risk(
        {
            "candidate_name": "wrong_structural_artifact",
            "operations": [
                {
                    "op": "append_text_instruction",
                    "content": "Use exact aliases.",
                }
            ],
        },
        allowed_operation_groups=serving_candidate_allowed_operation_groups(
            "schema_metadata_aliases"
        ),
        max_mutation_operations=serving_candidate_max_mutation_operations(
            "schema_metadata_aliases"
        ),
    )
    assert wrong_group["accepted"] is False
    assert "disallowed_operation_groups:text_instruction" in wrong_group["reasons"]

    hidden_column = assess_serving_candidate_risk(
        {
            "candidate_name": "hide_misleading_column",
            "operations": [
                {
                    "op": "set_column_config",
                    "table_identifier": "catalog.schema.sales",
                    "column_name": "brand_name",
                    "exclude": True,
                }
            ],
        },
        candidate_mode="schema_metadata_aliases",
        allowed_operation_groups=serving_candidate_allowed_operation_groups(
            "schema_metadata_aliases"
        ),
        max_mutation_operations=serving_candidate_max_mutation_operations(
            "schema_metadata_aliases"
        ),
    )
    assert hidden_column["accepted"] is False
    assert "metadata_aliases_cannot_exclude_columns" in hidden_column["reasons"]

    copied_sql = assess_serving_candidate_risk(
        {
            "candidate_name": "copied_sql_column_description",
            "operations": [
                {
                    "op": "set_column_config",
                    "table_identifier": "catalog.schema.sales",
                    "column_name": "brand_name",
                    "description": "Use SELECT brand_name FROM sales WHERE brand_name = :brand_name.",
                }
            ],
        },
        candidate_mode="schema_metadata_aliases",
        allowed_operation_groups=serving_candidate_allowed_operation_groups(
            "schema_metadata_aliases"
        ),
        max_mutation_operations=serving_candidate_max_mutation_operations(
            "schema_metadata_aliases"
        ),
    )
    assert copied_sql["accepted"] is False
    assert "metadata_aliases_contains_placeholder" in copied_sql["reasons"]
    assert "metadata_aliases_contains_sql_payload" in copied_sql["reasons"]


def test_assess_serving_candidate_risk_rejects_sql_snippet_bind_placeholder():
    payload = {
        "candidate_name": "parameterized_snippet",
        "rationale": "Use runtime parameter in a reusable filter.",
        "expected_impact": "Fixes brand filters.",
        "risk": "Low.",
        "operations": [
            {
                "op": "add_sql_snippet",
                "group": "filters",
                "display_name": "Brand parameter filter",
                "sql": "brand_name = :brand_name",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("schema_time_window"),
        max_mutation_operations=serving_candidate_max_mutation_operations("schema_time_window"),
    )

    assert assessment["accepted"] is False
    assert "sql_snippet_contains_placeholder" in assessment["reasons"]


def test_assess_serving_candidate_risk_rejects_sql_snippet_full_query():
    payload = {
        "candidate_name": "full_query_snippet",
        "rationale": "Use a complete query as a snippet.",
        "expected_impact": "Fixes time window failures.",
        "risk": "Low.",
        "operations": [
            {
                "op": "add_sql_snippet",
                "group": "measures",
                "display_name": "Current 13 week NRx",
                "sql": "WITH x AS (SELECT 1) SELECT * FROM x",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("schema_time_window"),
        max_mutation_operations=serving_candidate_max_mutation_operations("schema_time_window"),
    )

    assert assessment["accepted"] is False
    assert "sql_snippet_full_statement_not_allowed:with" in assessment["reasons"]


def test_assess_serving_candidate_risk_rejects_filter_snippet_option_menu():
    payload = {
        "candidate_name": "filter_option_menu",
        "rationale": "Offer all observed week buckets.",
        "expected_impact": "Fixes time-window failures.",
        "risk": "Low.",
        "operations": [
            {
                "op": "add_sql_snippet",
                "group": "filters",
                "display_name": "Literal-safe week bucket filters",
                "usage_guidance": "Choose exactly one predicate from this snippet.",
                "sql": ["wk_grp = 'C5W'", "wk_grp = 'C13W'", "wk_grp = 'P13W'"],
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("schema_time_window"),
        max_mutation_operations=serving_candidate_max_mutation_operations("schema_time_window"),
    )

    assert assessment["accepted"] is False
    assert "sql_snippet_filter_has_multiple_alternatives" in assessment["reasons"]


def test_sql_snippet_option_menu_guard_allows_single_multiline_filter_expression():
    payload = {
        "candidate_name": "single_multiline_filter",
        "rationale": "Add a reusable filter with one expression split across lines.",
        "expected_impact": "Preserves literal filtering without adding examples.",
        "risk": "Low.",
        "operations": [
            {
                "op": "add_sql_snippet",
                "group": "filters",
                "display_name": "Farxiga diabetes market filter",
                "usage_guidance": "Use this complete predicate together.",
                "sql": [
                    "brand_name = 'FARXIGA' AND",
                    "market_name = 'DIABETES'",
                ],
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        allowed_operation_groups=serving_candidate_allowed_operation_groups("schema_time_window"),
        max_mutation_operations=serving_candidate_max_mutation_operations("schema_time_window"),
    )
    candidate, summary = apply_serving_candidate(_config(), payload)

    assert assessment["accepted"] is True
    assert "sql_snippet_filter_has_multiple_alternatives" not in assessment["reasons"]
    assert summary["changed"] is True
    snippets = candidate.instructions.sql_snippets
    assert snippets is not None
    assert snippets.filters[0].sql == [
        "brand_name = 'FARXIGA' AND",
        "market_name = 'DIABETES'",
    ]


def test_apply_serving_candidate_skips_invalid_sql_snippet_shape():
    _, summary = apply_serving_candidate(
        _config(),
        {
            "candidate_name": "bad_snippet",
            "rationale": "",
            "expected_impact": "",
            "risk": "",
            "owner_followups": [],
            "operations": [
                {
                    "op": "add_sql_snippet",
                    "group": "filters",
                    "display_name": "Parameterized brand",
                    "sql": "brand_name = :brand_name",
                },
                {
                    "op": "add_sql_snippet",
                    "group": "measures",
                    "display_name": "Full query",
                    "sql": "SELECT count(*) FROM table",
                },
                {
                    "op": "add_sql_snippet",
                    "group": "filters",
                    "display_name": "Week bucket filter menu",
                    "usage_guidance": "Choose exactly one predicate.",
                    "sql": ["wk_grp = 'C5W'", "wk_grp = 'C13W'"],
                },
            ],
        },
    )

    assert summary["changed"] is False
    assert [item["reason"] for item in summary["skipped_operations"]] == [
        "sql_snippet_contains_placeholder",
        "sql_snippet_full_statement_not_allowed:select",
        "sql_snippet_filter_has_multiple_alternatives",
    ]


def test_apply_serving_candidate_derives_sql_snippet_display_name_when_missing():
    candidate, summary = apply_serving_candidate(
        _config(),
        {
            "candidate_name": "derived_snippet_name",
            "rationale": "",
            "expected_impact": "",
            "risk": "",
            "owner_followups": [],
            "operations": [
                {
                    "op": "add_sql_snippet",
                    "group": "filters",
                    "display_name": None,
                    "description": "Literal-safe week-grain filter for explicit current-13-week time windows.",
                    "question": "How should explicit 13-week time windows be filtered?",
                    "id": "literal_13_week_time_window_filter",
                    "sql": "wk_grp = 'C13W'",
                    "synonyms": ["current 13 weeks", "C13W"],
                },
            ],
        },
    )

    assert summary["changed"] is True
    assert summary["applied_operations"] == [
        {
            "index": 0,
            "op": "add_sql_snippet",
            "reason": "added_sql_snippet",
            "display_name": "Literal-safe week-grain filter for explicit current-13-week time windows.",
        }
    ]
    snippets = candidate.instructions.sql_snippets
    assert snippets is not None
    assert snippets.filters[0].display_name == (
        "Literal-safe week-grain filter for explicit current-13-week time windows."
    )
    assert snippets.filters[0].sql == ["wk_grp = 'C13W'"]


def test_build_serving_candidate_prompt_includes_visible_proxy_guards_without_hidden_payload():
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_proxy_guard_payload={
            "schema_version": "maxgenie.visible_proxy_guard_context.v1",
            "required_guard_ids": ["visible-guard-one"],
            "required_families": ["period_comparison"],
            "guards": [
                {
                    "question_id": "visible-guard-one",
                    "source": "validation",
                    "primary_family": "period_comparison",
                    "families": ["period_comparison", "time_window"],
                    "reason": "visible_validation_pass_proxy_guard",
                    "question": "Visible validation comparison question",
                }
            ],
            "policy": {
                "hidden_holdout_content": "omitted",
            },
        },
        visible_benchmark_payload={
            "question_count": 2,
            "questions": [
                {"id": "visible-guard-one", "question": "Visible validation comparison question"},
            ],
        },
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "## Visible Family Proxy Guards" in prompt
    assert "derived only from train and visible-validation pass records" in prompt
    assert "visible-guard-one" in prompt
    assert "period_comparison" in prompt
    assert "hidden_holdout_content" in prompt
    assert "secret holdout" not in prompt.lower()


def test_build_serving_candidate_prompt_includes_structured_candidate_context():
    failure_context = "\n".join(
        [
            "# Failure Context",
            "## Prior Candidate Lessons",
            "These lessons come from visible train/validation gates only.",
            "### Accepted Patterns To Preserve",
            "- `accepted_visible_example`: accepted at train=77.8%, visible_validation=75.0%.",
            "  - Operations to preserve as precedent: file_edit path=instructions/examples/accepted_visible_example.yml mode=create_file reason=target_train_failure:q-example",
            "### Rejected Patterns To Avoid Or Revise",
            "- Rejected candidate modes: `metadata`=1, `sql_snippet`=1",
            "- `broad_metadata_alias_patch`: rejected because train_health_failed: 77.8% -> 0.0%.",
            "  - Operations that failed the gates: file_edit path=tables/accounts.yml mode=find_replace reason=target_train_failure:q-metadata",
            "### Skipped Or Risk-Rejected Patterns To Avoid",
            "- `wide_snippet_bundle`: skipped because no_valid_serving_operations.",
            "  - Candidate shape refused before benchmarking: file_edit path=instructions/text_instruction.md mode=append_content reason=target_train_failure:q-text",
            "- `another_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `third_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `fourth_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `fifth_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `sixth_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `seventh_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `eighth_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "- `ninth_rejected_patch`: skipped because serving_candidate_risk_rejected.",
            "  - Candidate shape refused before benchmarking: file_edit path=instructions/sql_snippets/measures/ratio.yml mode=replace_file reason=target_train_failure:q-ratio",
            "## Prior Optimization Attempts",
            "- **serving_iteration_03** \u2192 rejected (train=66.7%, val=skipped)",
            "  - Reason: pass guard regressed: guard-one",
            "  - Candidate mode: `metadata`",
            "- **serving_iteration_04** \u2192 rejected (train=77.8%, val=skipped)",
            "  - Reason: known failures did not improve; train remained at 77.8%",
            "  - Candidate mode: `sql_snippet`",
        ]
    )
    structured_context = {
        "schema_version": 1,
        "iteration_label": "serving_iteration_05",
        "train_pass_rate": 77.78,
        "best_train_rate": 77.78,
        "best_validation_rate": 75.0,
        "train_question_count": 9,
        "failed_question_count": 2,
        "failure_families": {
            "percentage_or_comparison_metric": [
                "dddddddddddddddddddddddddddddddd",
                "cccccccccccccccccccccccccccccccc",
            ]
        },
        "failures": [
            {
                "question_id": "dddddddddddddddddddddddddddddddd",
                "likely_root_cause_family": "percentage_or_comparison_metric",
                "suggested_operation_families": ["add_example_sql", "add_sql_snippet"],
                "guard_regression_risk": "medium",
                "confidence": 0.75,
                "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                "extra_generated_tokens": ["current_nrx"],
            },
            {
                "question_id": "cccccccccccccccccccccccccccccccc",
                "likely_root_cause_family": "percentage_or_comparison_metric",
                "suggested_operation_families": ["add_example_sql", "add_sql_snippet"],
                "guard_regression_risk": "medium",
                "confidence": 0.75,
                "missing_expected_tokens": ["prior_trx"],
                "extra_generated_tokens": [],
            },
        ],
        "pass_guards": [
            {
                "question_id": "guard-one",
                "source": "accepted_candidate",
                "latest_status": "passed",
                "guard_reason": "accepted_candidate_fix",
                "accepted_candidate": "serving_iteration_04",
            },
            {
                "question_id": "visible-validation-guard",
                "question": "Compare current and prior period at national level",
                "source": "baseline_validation",
                "latest_status": "passed",
                "guard_reason": "baseline_validation_pass",
            }
        ],
    }

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context=failure_context,
        structured_failure_context=structured_context,
        trace_context_pack={
            "schema_version": "maxgenie.failure_context_pack.v1",
            "scorecard": {
                "baseline": {
                    "full": {
                        "pass_rate": 20.0,
                        "failed_ids": ["visible-train-failure", "secret-holdout-id"],
                        "question": "Secret holdout full question",
                        "expected_sql": "SELECT secret FROM hidden_table",
                    },
                    "train": {
                        "pass_rate": 22.22,
                        "failed_ids": ["visible-train-failure"],
                    },
                },
                "current": {
                    "holdout": {
                        "pass_rate": 33.33,
                        "failed_ids": ["secret-holdout-id"],
                    },
                },
            },
            "hidden_holdout_summary": {
                "ids": ["secret-generic-id"],
                "failed_question_ids": ["secret-failed-question-id"],
                "question_ids": ["secret-nested-id"],
                "query": "SELECT secret_query FROM hidden_query_table",
                "query_text": "Secret holdout query text",
                "sql_text": "SELECT secret_sql_text FROM hidden_sql_text_table",
                "statement": "SELECT secret_statement FROM hidden_statement_table",
                "text": "Secret generic hidden holdout text",
                "failures": [
                    {
                        "question": "Nested secret holdout question",
                        "expected_query": "SELECT nested_query FROM hidden_table",
                        "expected_sql": "SELECT nested_secret FROM hidden_table",
                    }
                ],
            },
            "target_failures": [{"question_id": "dddddddddddddddddddddddddddddddd"}],
            "guard_regressions": [{"question_id": "guard-one"}],
        },
        visible_benchmark_payload={"question_count": 9, "questions": []},
        diagnostics_payload={},
        optimization_summary={"accepted": 2, "rejected": 2, "skipped": 0},
        candidate_mode="example_set",
    )

    assert "## Candidate Context Summary" in prompt
    assert "## Compact Trace Memory" in prompt
    assert "maxgenie.failure_context_pack.v1" in prompt
    assert "secret-holdout-id" not in prompt
    assert "secret-generic-id" not in prompt
    assert "secret-failed-question-id" not in prompt
    assert "secret-nested-id" not in prompt
    assert "secret_query" not in prompt
    assert "Secret holdout query text" not in prompt
    assert "secret_sql_text" not in prompt
    assert "secret_statement" not in prompt
    assert "Secret generic hidden holdout text" not in prompt
    assert "Nested secret holdout question" not in prompt
    assert "hidden_table" not in prompt
    assert "hidden_query_table" not in prompt
    assert "hidden_sql_text_table" not in prompt
    assert "hidden_statement_table" not in prompt
    assert "visible-train-failure" in prompt
    assert "this section can include hidden holdout question ids" in prompt
    assert "this section can include hidden holdout payloads" in prompt
    assert "train=77.78%" in prompt
    assert "best_validation=75.00%" in prompt
    assert "remaining_train_failures=2" in prompt
    assert "`dddddddddddddddddddddddddddddddd`" in prompt
    assert "`cccccccccccccccccccccccccccccccc`" in prompt
    assert "Structural hints: missing expected tokens" in prompt
    assert "`percentage_difference`" in prompt
    assert "`guard-one` source=`accepted_candidate` latest=`passed` reason=`accepted_candidate_fix`" in prompt
    assert "## Visible Generalization Proxy Guards" in prompt
    assert "`visible-validation-guard` latest=`passed`" in prompt
    assert "### Prior Candidate Lessons To Preserve Or Avoid" in prompt
    assert "visible train/validation lessons are hard synthesis constraints" in prompt
    assert "Prior train-health collapse detected" in prompt
    assert "choose a narrower visible target plan or return no_change" in prompt
    assert "accepted_visible_example" in prompt
    assert "broad_metadata_alias_patch" in prompt
    assert "wide_snippet_bundle" in prompt
    assert "File-surface lessons from prior patches" in prompt
    assert (
        "preserve examples: file_edit path=instructions/examples/accepted_visible_example.yml "
        "mode=create_file reason=target_train_failure:q-example"
    ) in prompt
    assert (
        "avoid_or_revise text_instruction: file_edit path=instructions/text_instruction.md "
        "mode=append_content reason=target_train_failure:q-text"
    ) in prompt
    assert (
        "avoid_or_revise metadata: file_edit path=tables/accounts.yml "
        "mode=find_replace reason=target_train_failure:q-metadata"
    ) in prompt
    assert (
        "file_edit path=instructions/sql_snippets/measures/ratio.yml "
        "mode=replace_file reason=target_train_failure:q-ratio"
    ) in prompt
    assert "File-edit shapes to avoid or materially revise" in prompt
    assert "Rejected counts by candidate mode: `metadata`=1, `sql_snippet`=1." in prompt
    assert "Prior pass-guard regressions: `1`" in prompt
    assert "accepted=`2`; rejected=`2`; skipped=`0`" in prompt


def test_artifact_patch_v2_prompt_replays_prior_file_surface_distinction():
    failure_context = "\n".join(
        [
            "# Failure Context",
            "## Prior Candidate Lessons",
            "These lessons come from visible train/validation gates only.",
            "### Accepted Patterns To Preserve",
            "- `accepted_literal_snippet`: accepted at train=66.7%, visible_validation=58.3%.",
            "  - Operations to preserve as precedent: file_edit path=instructions/sql_snippets/filters/literal_brand.yml mode=create_file reason=target_train_failure:q-brand",
            "### Rejected Patterns To Avoid Or Revise",
            "- `rejected_text_only_guidance`: rejected because train_health_failed: 66.7% -> 0.0%.",
            "  - Operations that failed the gates: file_edit path=instructions/text_instruction.md mode=append_content reason=target_train_failure:q-brand",
            "- `collapsed_literal_example`: rejected because train_health_failed: 66.7% -> 0.0%.",
            "  - Operations that failed the gates: file_edit path=instructions/examples/collapsed_literal.yml mode=replace_file reason=target_train_failure:q-brand",
            "- `collapsed_literal_snippet`: rejected because train_health_failed: 66.7% -> 0.0%.",
            "  - Operations that failed the gates: file_edit path=instructions/sql_snippets/filters/collapsed_literal.yml mode=replace_file reason=target_train_failure:q-brand",
        ]
    )
    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context=failure_context,
        structured_failure_context={
            "schema_version": 1,
            "train_pass_rate": 66.67,
            "best_train_rate": 66.67,
            "best_validation_rate": 58.33,
            "failed_question_count": 1,
            "failure_families": {
                "literal_or_parameter_mismatch": ["q-brand"],
            },
            "failures": [
                {
                    "question_id": "q-brand",
                    "question": "Show NRX for Farxiga",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "guard_regression_risk": "medium",
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
        visible_benchmark_payload={"question_count": 1, "questions": []},
        decomposed_space_payload={
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "files": [
                {
                    "path": "instructions/sql_snippets/filters/literal_brand.yml",
                    "sha256": "c" * 64,
                    "expected_sha256": "c" * 64,
                    "content": (
                        "display_name: Farxiga literal filter\n"
                        "sql: |\n"
                        "  brand_name = 'farxiga'\n"
                    ),
                }
            ],
        },
        diagnostics_payload={},
        optimization_summary={"accepted": 1, "rejected": 1, "skipped": 0},
        candidate_mode="artifact_patch_v2",
    )

    assert "accepted_literal_snippet" in prompt
    assert "rejected_text_only_guidance" in prompt
    assert "Prior train-health collapse detected" in prompt
    assert "Repeated new executable artifact collapse detected" in prompt
    assert "do not create another new instructions/examples or instructions/sql_snippets file" in prompt
    assert "new_file_allowed=`false`" in prompt
    assert "existing_edit_files=`instructions/sql_snippets/filters/literal_brand.yml`" in prompt
    assert (
        "When prior lessons report repeated new executable artifact collapse, "
        "executable example/snippet plans set new_file_allowed=false"
    ) in prompt
    assert (
        "Patch archetype: compactly edit one listed existing SQL-snippet asset"
        in prompt
    )
    assert "do not create a new SQL-snippet file" in prompt
    assert (
        "Patch template hint: existing instructions/sql_snippets YAML only: use exact `find` plus `replace`"
        in prompt
    )
    assert "Current edit-file excerpts for compact find/replace" in prompt
    assert f"path=`instructions/sql_snippets/filters/literal_brand.yml`; expected_sha256=`{'c' * 64}`" in prompt
    assert "File-surface lessons from prior patches" in prompt
    assert (
        "preserve sql_snippets: file_edit path=instructions/sql_snippets/filters/literal_brand.yml "
        "mode=create_file reason=target_train_failure:q-brand"
    ) in prompt
    assert (
        "avoid_or_revise text_instruction: file_edit path=instructions/text_instruction.md "
        "mode=append_content reason=target_train_failure:q-brand"
    ) in prompt
    assert "File-edit shapes to avoid or materially revise" in prompt
    assert (
        "file_edit path=instructions/text_instruction.md "
        "mode=append_content reason=target_train_failure:q-brand"
    ) in prompt
    assert "secret" not in prompt.lower()


def test_build_serving_candidate_prompt_includes_generalized_visible_family_guidance():
    structured_context = {
        "schema_version": 1,
        "iteration_label": "serving_iteration_06",
        "train_pass_rate": 22.22,
        "best_train_rate": 33.33,
        "best_validation_rate": 25.0,
        "train_question_count": 9,
        "failed_question_count": 3,
        "failure_families": {
            "literal_or_parameter_mismatch": ["q-unbound"],
            "selected_metric_or_column_mismatch": ["q-schema"],
            "percentage_or_comparison_metric": ["q-pct"],
        },
        "failures": [
            {
                "question_id": "q-unbound",
                "likely_root_cause_family": "literal_or_parameter_mismatch",
                "suggested_operation_families": ["add_example_sql"],
                "guard_regression_risk": "medium",
                "confidence": 0.8,
                "missing_expected_tokens": ["'farxiga'"],
                "extra_generated_tokens": [":brand_name", ":time_grp_type"],
                "comparison_method": "result_unbound_parameters_error",
            },
            {
                "question_id": "q-schema",
                "likely_root_cause_family": "selected_metric_or_column_mismatch",
                "suggested_operation_families": ["set_column_config"],
                "guard_regression_risk": "high",
                "confidence": 0.65,
                "missing_expected_tokens": ["time_grp_desc", "region"],
                "extra_generated_tokens": ["wk_grp_desc", "exists_flag"],
                "comparison_method": "result_column_mismatch",
            },
            {
                "question_id": "q-pct",
                "likely_root_cause_family": "percentage_or_comparison_metric",
                "suggested_operation_families": ["add_sql_snippet"],
                "guard_regression_risk": "medium",
                "confidence": 0.75,
                "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                "extra_generated_tokens": ["current_nrx"],
                "comparison_method": "result_column_mismatch",
            },
        ],
        "pass_guards": [],
    }

    prompt = build_serving_candidate_prompt(
        current_config=_config(),
        failure_context="# Failure Context\n",
        structured_failure_context=structured_context,
        visible_benchmark_payload={"question_count": 9, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert "## Generalized Visible-Family Guidance" in prompt
    assert "computed only from train-visible failures" in prompt
    assert "Literalization family" in prompt
    assert ":brand_name" in prompt
    assert "Period-comparison family" in prompt
    assert "prior_nrx" in prompt
    assert "Output-schema family" in prompt
    assert "time_grp_desc" in prompt
    assert "Causal repair recipe" in prompt
    assert "Add one narrow visible example asset" in prompt
    assert "Prefer metadata or a reusable measure/expression snippet" in prompt
    assert "Patch the denominator/current-vs-prior rule" in prompt
    assert "Avoid: current-period-only metrics" in prompt
    assert "secret holdout" not in prompt.lower()


def test_assess_serving_candidate_risk_requires_guardrail_acknowledgement_when_requested():
    payload = {
        "candidate_name": "nrx_measure",
        "rationale": "Fixes the NRx metric mapping.",
        "expected_impact": "Improves NRx questions.",
        "risk": "Low blast radius.",
        "operations": [
            {
                "op": "add_sql_snippet",
                "group": "measures",
                "display_name": "NRx measure",
                "sql": "SUM(new_prescriptions)",
            }
        ],
    }

    assessment = assess_serving_candidate_risk(
        payload,
        require_guardrail_acknowledgement=True,
    )

    assert assessment["accepted"] is False
    assert "missing_guardrail_acknowledgement" in assessment["reasons"]

    payload["risk"] = "Preserves the pass guard SQL patterns and avoids prior regressions."
    accepted = assess_serving_candidate_risk(
        payload,
        require_guardrail_acknowledgement=True,
    )

    assert accepted["accepted"] is True


def test_serving_candidate_client_posts_chat_payload_and_parses_json():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClient(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(candidate_payload),
                    }
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="candidate-endpoint",
            model="candidate-model",
            reasoning_effort="xhigh",
            api_format="chat",
        ),
    )

    proposal = client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="example_sql",
    )

    call = wc.api_client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/serving-endpoints/candidate-endpoint/invocations"
    assert call["body"]["model"] == "candidate-model"
    assert "temperature" not in call["body"]
    assert call["body"]["reasoning_effort"] == "xhigh"
    assert call["body"]["response_format"]["type"] == "json_schema"
    response_schema = call["body"]["response_format"]["json_schema"]["schema"]
    file_edit_schema = response_schema["properties"]["file_edits"]["items"]
    assert "description" in file_edit_schema["properties"]
    assert "comment" in file_edit_schema["properties"]
    assert "description" in file_edit_schema["required"]
    assert "comment" in file_edit_schema["required"]
    assert "Mode: example_sql" in call["body"]["messages"][1]["content"]
    assert "Prior Optimization Attempts as hard constraints" in call["body"]["messages"][1]["content"]
    assert proposal.payload == {**candidate_payload, "causal_patch_plan": None}


def test_serving_candidate_client_translates_adaptive_thinking_controls():
    """Adaptive endpoints receive thinking.type=adaptive + output_config.effort."""
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClient(
        {"choices": [{"message": {"content": json.dumps(candidate_payload)}}]}
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="databricks-claude-opus-4-8",
            reasoning_effort="xhigh",
            api_format="chat",
        ),
    )

    client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="example_sql",
    )

    body = wc.api_client.calls[0]["body"]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}  # xhigh -> high
    assert "reasoning_effort" not in body
    assert "response_format" not in body


def test_serving_candidate_client_adaptive_without_thinking_allows_response_format():
    """When thinking is disabled, json_schema response_format remains available."""
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "x",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "x"}],
    }
    wc = _FakeWorkspaceClient(
        {"choices": [{"message": {"content": json.dumps(candidate_payload)}}]}
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="databricks-claude-opus-4-8",
            reasoning_effort="none",
            api_format="chat",
        ),
    )

    client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="example_sql",
    )

    body = wc.api_client.calls[0]["body"]
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert body["response_format"]["type"] == "json_schema"


def test_serving_candidate_client_repairs_truncated_object_delimiter():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "file_edits": None,
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClient(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(candidate_payload)[:-1],
                    }
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    proposal = client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
    )

    assert proposal.payload == {**candidate_payload, "causal_patch_plan": None}
    assert proposal.parse_repairs == ("balanced_json_delimiters",)


def test_serving_candidate_client_repairs_trailing_json_commas_and_prose():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "file_edits": None,
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    raw_json = json.dumps(candidate_payload).replace("}]", "},]")
    wc = _FakeWorkspaceClient(
        {
            "choices": [
                {
                    "message": {
                        "content": f"Here is the candidate:\n{raw_json}\nDone.",
                    }
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    proposal = client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
    )

    assert proposal.payload == {**candidate_payload, "causal_patch_plan": None}
    assert proposal.parse_repairs == (
        "extracted_balanced_json_object",
        "removed_trailing_json_commas",
    )


def test_serving_candidate_client_normalizes_file_edit_nullable_defaults():
    candidate_payload = {
        "candidate_name": "patch",
        "rationale": "Patch one file.",
        "expected_impact": "Improves one visible failure.",
        "risk": "Hash-checked edit.",
        "owner_followups": [],
        "operations": [],
        "file_edits": [
            {
                "path": "instructions/text_instruction.md",
                "append_content": "\nUse literal time filters.\n",
                "expected_sha256": "a" * 64,
                "reason": "target_train_failure:" + "b" * 32,
            }
        ],
    }
    wc = _FakeWorkspaceClient(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(candidate_payload),
                    }
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    proposal = client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
        candidate_mode="artifact_patch_v2",
    )

    assert proposal.payload["file_edits"] == [
        {
            "path": "instructions/text_instruction.md",
            "relative_path": None,
            "content": None,
            "append_content": "\nUse literal time filters.\n",
            "find": None,
            "replace": None,
            "expected_sha256": "a" * 64,
            "reason": "target_train_failure:" + "b" * 32,
            "description": None,
            "comment": None,
        }
    ]
    assert proposal.payload["causal_patch_plan"] is None


def test_serving_candidate_client_includes_explicit_chat_temperature():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClient(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(candidate_payload),
                    }
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="candidate-endpoint",
            temperature=0.0,
            api_format="chat",
        ),
    )

    client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
    )

    call = wc.api_client.calls[0]
    assert call["body"]["temperature"] == 0.0


def test_serving_candidate_client_posts_responses_payload_for_gpt_endpoint():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClient(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(candidate_payload),
                        }
                    ],
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="databricks-gpt-5-5-pro",
            reasoning_effort="xhigh",
        ),
    )

    proposal = client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
    )

    call = wc.api_client.calls[0]
    assert call["method"] == "POST"
    assert call["path"] == "/serving-endpoints/responses"
    assert call["body"]["model"] == "databricks-gpt-5-5-pro"
    assert call["body"]["reasoning"]["effort"] == "high"
    assert call["body"]["text"]["format"]["type"] == "json_schema"
    assert call["body"]["text"]["format"]["name"] == "maxgenie_candidate"
    assert "json_schema" not in call["body"]["text"]["format"]
    operation_schema = call["body"]["text"]["format"]["schema"]["properties"]["operations"]["items"]
    top_level_schema = call["body"]["text"]["format"]["schema"]
    causal_plan_schema = top_level_schema["properties"]["causal_patch_plan"]
    assert set(operation_schema["required"]) == set(operation_schema["properties"])
    assert "causal_patch_plan" in top_level_schema["required"]
    assert set(causal_plan_schema["required"]) == set(causal_plan_schema["properties"])
    assert "max_tokens" not in call["body"]
    assert "temperature" not in call["body"]
    assert call["body"]["max_output_tokens"] == 16000
    assert "Use exactly one operation group per candidate" in call["body"]["input"][0]["content"]
    assert proposal.payload == {**candidate_payload, "causal_patch_plan": None}


def test_serving_candidate_client_preserves_xhigh_for_non_pro_gpt_responses_endpoint():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClient(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(candidate_payload),
                        }
                    ],
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="databricks-gpt-5-5",
            reasoning_effort="xhigh",
        ),
    )

    client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
    )

    call = wc.api_client.calls[0]
    assert call["path"] == "/serving-endpoints/responses"
    assert call["body"]["model"] == "databricks-gpt-5-5"
    assert call["body"]["reasoning"]["effort"] == "xhigh"


def test_serving_candidate_client_rejects_incomplete_responses_without_text():
    response = {
        "object": "response",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning", "summary": []}],
    }
    wc = _FakeWorkspaceClient(response)
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="databricks-gpt-5-5-pro"),
    )

    with pytest.raises(ServingCandidateResponseError) as exc_info:
        client.propose(
            current_config=_config(),
            failure_context="# Failure Context\n",
            visible_benchmark_payload={"question_count": 1, "questions": []},
            diagnostics_payload={},
            optimization_summary={},
        )
    exc = exc_info.value
    assert "did not contain usable candidate content" in str(exc)
    assert "reason=max_output_tokens" in str(exc)
    assert exc.response is response
    assert exc.content == ""


def test_serving_candidate_client_preserves_malformed_json_context():
    wc = _FakeWorkspaceClient(
        {
            "choices": [
                {
                    "message": {
                        "content": '{"candidate_name":"bad","operations":[{"op":"add_example_sql","sql":"SELECT',
                    }
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    with pytest.raises(ServingCandidateResponseError) as exc_info:
        client.propose(
            current_config=_config(),
            failure_context="# Failure Context\n",
            visible_benchmark_payload={"question_count": 1, "questions": []},
            diagnostics_payload={},
            optimization_summary={},
        )

    exc = exc_info.value
    assert "valid candidate JSON" in str(exc)
    assert exc.request["messages"][0]["role"] == "system"
    assert exc.response is wc.api_client.response
    assert exc.content.startswith('{"candidate_name":"bad"')


def test_serving_candidate_client_propagates_endpoint_errors_without_fallback():
    class _RejectingApiClient:
        def __init__(self):
            self.calls = []

        def do(self, method, path, body=None, query=None):
            self.calls.append({"method": method, "path": path, "body": body, "query": query})
            raise RuntimeError("endpoint rejected unsupported parameter")

    class _RejectingWorkspaceClient:
        def __init__(self):
            self.api_client = _RejectingApiClient()

    wc = _RejectingWorkspaceClient()
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    with pytest.raises(RuntimeError, match="endpoint rejected unsupported parameter"):
        client.propose(
            current_config=_config(),
            failure_context="# Failure Context\n",
            visible_benchmark_payload={"question_count": 1, "questions": []},
            diagnostics_payload={},
            optimization_summary={},
        )

    assert len(wc.api_client.calls) == 1


def test_serving_candidate_client_records_transport_errors_as_invalid_response():
    class _ChunkedApiClient:
        def __init__(self):
            self.calls = []

        def do(self, method, path, body=None, query=None):
            self.calls.append({"method": method, "path": path, "body": body, "query": query})
            raise requests.exceptions.ChunkedEncodingError("broken chunk")

    class _ChunkedWorkspaceClient:
        def __init__(self):
            self.api_client = _ChunkedApiClient()

    wc = _ChunkedWorkspaceClient()
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(endpoint="databricks-gpt-5-5"),
    )

    with pytest.raises(ServingCandidateResponseError) as exc_info:
        client.propose(
            current_config=_config(),
            failure_context="# Failure Context\n",
            visible_benchmark_payload={"question_count": 1, "questions": []},
            diagnostics_payload={},
            optimization_summary={},
        )

    exc = exc_info.value
    assert "Serving endpoint request failed" in str(exc)
    assert "broken chunk" in str(exc)
    assert exc.request["model"] == "databricks-gpt-5-5"
    assert exc.response == {}
    assert exc.content == ""
    assert len(wc.api_client.calls) == 1


def test_serving_candidate_client_extends_and_restores_sdk_timeouts():
    candidate_payload = {
        "candidate_name": "no_change",
        "rationale": "No safe change.",
        "expected_impact": "None",
        "risk": "None",
        "owner_followups": [],
        "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
    }
    wc = _FakeWorkspaceClientWithTimeouts(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(candidate_payload)}],
                }
            ]
        }
    )
    client = ServingCandidateClient(
        wc=wc,
        config=ServingCandidateConfig(
            endpoint="databricks-gpt-5-5-pro",
            request_timeout_seconds=900,
            retry_timeout_seconds=1800,
        ),
    )

    client.propose(
        current_config=_config(),
        failure_context="# Failure Context\n",
        visible_benchmark_payload={"question_count": 1, "questions": []},
        diagnostics_payload={},
        optimization_summary={},
    )

    call = wc.api_client.calls[0]
    assert call["http_timeout"] == 900.0
    assert call["retry_timeout"] == 1800
    assert wc.api_client._api_client._http_timeout_seconds == 60.0
    assert wc.api_client._api_client._retry_timeout_seconds == 300


def test_serving_config_rejects_missing_endpoint():
    with pytest.raises(ValueError, match="Serving endpoint is required"):
        ServingCandidateConfig(endpoint="")


def test_serving_config_rejects_unknown_api_format():
    with pytest.raises(ValueError, match="Unsupported serving API format"):
        ServingCandidateConfig(endpoint="candidate-endpoint", api_format="bad")


def test_serving_config_rejects_out_of_range_temperature():
    with pytest.raises(ValueError, match="temperature must be between"):
        ServingCandidateConfig(endpoint="candidate-endpoint", temperature=-0.1)


def test_serving_config_rejects_too_short_timeout():
    with pytest.raises(ValueError, match="at least 60 seconds"):
        ServingCandidateConfig(endpoint="candidate-endpoint", request_timeout_seconds=30)


def test_serving_config_rejects_retry_timeout_below_request_timeout():
    with pytest.raises(ValueError, match="greater than or equal"):
        ServingCandidateConfig(
            endpoint="candidate-endpoint",
            request_timeout_seconds=900,
            retry_timeout_seconds=300,
        )
