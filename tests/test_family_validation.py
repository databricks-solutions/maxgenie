"""Tests for visible family-aware validation helpers."""

import json

from maxgenie.family_validation import (
    build_proxy_guard_requirements,
    classify_benchmark_family,
    compute_visible_proxy_pressure,
)


def test_classify_benchmark_family_uses_question_issue_and_token_hints():
    result = classify_benchmark_family({
        "question_id": "q-current-prior",
        "question": "Percentage difference for Farxiga current 13 weeks vs previous 13 weeks for NRx",
        "issue_class": "period comparison mismatch",
        "missing_expected_tokens": ["percentage_difference", "prior_nrx"],
    })

    assert result.question_id == "q-current-prior"
    assert result.primary_family == "period_comparison"
    assert "time_window" in result.families
    assert "metric_selection" in result.families
    assert "market_product_filter" in result.families
    assert "issue_class:comparison" in result.signals


def test_build_proxy_guard_requirements_uses_train_and_validation_only():
    records = [
        {
            "question_id": "train-guard",
            "split": "train",
            "question": "NRx for Farxiga in current 13 weeks",
            "status": "passed",
        },
        {
            "question_id": "validation-guard",
            "split": "validation",
            "question": "Show sales by region",
            "status": "passed",
        },
        {
            "question_id": "train-failure",
            "split": "train",
            "question": "Compare Lokelma current quarter vs previous quarter",
            "status": "failed",
        },
        {
            "question_id": "hidden-holdout",
            "split": "holdout",
            "question": "Secret holdout question must never be surfaced",
            "status": "passed",
        },
    ]

    result = build_proxy_guard_requirements(records)
    serialized = json.dumps(result.to_dict())

    assert result.required_guard_ids == ("train-guard", "validation-guard")
    assert result.ignored_hidden_count == 1
    assert "time_window" in result.required_families
    assert "geography_breakdown" in result.required_families
    assert "Secret holdout" not in serialized
    assert "hidden-holdout" not in serialized
    assert "train-failure" not in result.required_guard_ids


def test_compute_visible_proxy_pressure_warns_on_score_lift_with_guard_and_family_loss():
    before = [
        {
            "question_id": "guard-period",
            "split": "train",
            "question": "Percentage difference for Farxiga current vs previous 13 weeks for NRx",
            "status": "passed",
        },
        {
            "question_id": "guard-region",
            "split": "validation",
            "question": "Show Farxiga sales by region",
            "status": "passed",
        },
        {
            "question_id": "hidden-before",
            "split": "test",
            "question": "Hidden benchmark text",
            "status": "passed",
        },
    ]
    after = [
        {
            "question_id": "guard-period",
            "split": "train",
            "question": "Percentage difference for Farxiga current vs previous 13 weeks for NRx",
            "status": "passed",
        },
        {
            "question_id": "guard-region",
            "split": "validation",
            "question": "Show Farxiga sales by region",
            "status": "failed",
        },
        {
            "question_id": "new-train-pass",
            "split": "train",
            "question": "NBRx for Lokelma market specialty view current 6 months",
            "status": "passed",
        },
        {
            "question_id": "hidden-after",
            "split": "holdout",
            "question": "Secret holdout question must never be surfaced",
            "status": "failed",
        },
    ]

    report = compute_visible_proxy_pressure(
        before,
        after,
        before_train_rate=40.0,
        after_train_rate=60.0,
        before_validation_rate=50.0,
        after_validation_rate=50.0,
    )
    serialized = json.dumps(report.to_dict())

    assert report.visible_scores_improved is True
    assert report.train_delta == 20.0
    assert report.validation_delta == 0.0
    assert report.regressed_guard_ids == ("guard-region",)
    assert "geography_breakdown" in report.degraded_families
    assert report.pressure_score > 0
    assert "visible_score_lift_with_proxy_degradation" in report.warnings
    assert "Secret holdout" not in serialized
    assert "hidden-after" not in serialized
