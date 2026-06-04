"""Pure benchmark-summary helper functions, carved out of workspace_flow."""

from __future__ import annotations

from typing import Any

from maxgenie.benchmark_service import BenchmarkRun

__all__ = [
    "_aggregate_comparison_metrics",
    "_comparison_metrics_from_run",
    "_comparison_mode",
    "_derived_full_summary",
    "_summarize_run",
]


def _derived_full_summary(
    *,
    latest_train: BenchmarkRun | None,
    latest_dev_summary: dict[str, Any] | None,
    latest_test_pass_rate: float | None,
    latest_test_question_count: int,
) -> dict[str, Any] | None:
    """Build an overall summary from the latest train/validation/test artifacts when full is absent."""
    dev_summary = latest_dev_summary or {}
    validation_strategy = str(dev_summary.get("validation_strategy", "blind_split"))
    dev_total = int(dev_summary.get("question_count", 0) or 0)
    dev_pass_rate = (
        float(dev_summary["pass_rate"])
        if "pass_rate" in dev_summary and dev_summary["pass_rate"] is not None
        else None
    )

    if validation_strategy == "small_sample_kfold":
        if dev_total > 0 and dev_pass_rate is None:
            return None
        if latest_test_question_count > 0 and latest_test_pass_rate is None:
            return None

        dev_passed = int(dev_summary.get("passed", round(dev_total * (dev_pass_rate or 0.0) / 100)))
        dev_failed = int(dev_summary.get("failed", max(dev_total - dev_passed, 0)))
        dev_errors = int(dev_summary.get("errors", 0))
        test_total = max(latest_test_question_count, 0) if latest_test_pass_rate is not None else 0
        test_passed = round(test_total * latest_test_pass_rate / 100) if latest_test_pass_rate is not None else 0
        test_failed = max(test_total - test_passed, 0)
        overall_total = dev_total + test_total
        overall_passed = dev_passed + test_passed
        overall_failed = dev_failed + test_failed
        overall_errors = dev_errors
        pass_rate = (overall_passed / overall_total * 100) if overall_total else 0.0

        validation_summary = {
            "question_count": dev_total,
            "passed": dev_passed,
            "failed": dev_failed,
            "errors": dev_errors,
            "pass_rate": round(dev_pass_rate or 0.0, 2),
            "validation_strategy": validation_strategy,
            "fold_count": int(dev_summary.get("fold_count", 0)),
            "mean_fold_pass_rate": dev_summary.get("mean_fold_pass_rate"),
            "min_fold_pass_rate": dev_summary.get("min_fold_pass_rate"),
        }

        return {
            "overall": {
                "total": overall_total,
                "passed": overall_passed,
                "failed": overall_failed,
                "errors": overall_errors,
                "pass_rate": round(pass_rate, 2),
            },
            "overall_basis": "visible_kfold_plus_holdout",
            "train": _summarize_run(latest_train) if latest_train else None,
            "validation": validation_summary,
            "test": {
                "question_count": test_total,
                "passed": test_passed,
                "failed": test_failed,
                "errors": 0,
                "pass_rate": round(latest_test_pass_rate or 0.0, 2),
            },
        }

    if latest_train is None:
        return None
    if dev_total > 0 and dev_pass_rate is None:
        return None
    if latest_test_question_count > 0 and latest_test_pass_rate is None:
        return None

    dev_total = max(dev_total, 0) if dev_pass_rate is not None else 0
    dev_passed = round(dev_total * dev_pass_rate / 100) if dev_pass_rate is not None else 0
    dev_failed = max(dev_total - dev_passed, 0)
    test_total = max(latest_test_question_count, 0) if latest_test_pass_rate is not None else 0
    test_passed = round(test_total * latest_test_pass_rate / 100) if latest_test_pass_rate is not None else 0
    test_failed = max(test_total - test_passed, 0)
    overall_total = latest_train.total + dev_total + test_total
    overall_passed = latest_train.passed + dev_passed + test_passed
    overall_failed = latest_train.failed + dev_failed + test_failed
    overall_errors = latest_train.errors
    pass_rate = (overall_passed / overall_total * 100) if overall_total else 0.0

    validation_summary = {
        "question_count": dev_total,
        "passed": dev_passed,
        "failed": dev_failed,
        "errors": 0,
        "pass_rate": round(dev_pass_rate or 0.0, 2),
        "validation_strategy": validation_strategy,
    }

    return {
        "overall": {
            "total": overall_total,
            "passed": overall_passed,
            "failed": overall_failed,
            "errors": overall_errors,
            "pass_rate": round(pass_rate, 2),
        },
        "train": _summarize_run(latest_train),
        "validation": validation_summary,
        "test": {
            "question_count": test_total,
            "passed": test_passed,
            "failed": test_failed,
            "errors": 0,
            "pass_rate": round(latest_test_pass_rate or 0.0, 2),
        },
    }


def _summarize_run(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "total": run.total,
        "passed": run.passed,
        "failed": run.failed,
        "errors": run.errors,
        "pass_rate": round(run.pass_rate, 2),
        **_comparison_metrics_from_run(run),
    }


def _comparison_mode(*, result_based_question_count: int, string_based_question_count: int) -> str:
    if result_based_question_count > 0 and string_based_question_count > 0:
        return "mixed"
    if result_based_question_count > 0:
        return "result_based"
    if string_based_question_count > 0:
        return "string_only"
    return "unscored"


def _comparison_metrics_from_run(run: BenchmarkRun) -> dict[str, Any]:
    scored_question_count = int(getattr(run, "scored_question_count", 0) or 0)
    result_based_question_count = int(getattr(run, "result_based_question_count", 0) or 0)
    string_based_question_count = int(getattr(run, "string_based_question_count", 0) or 0)
    metrics = {
        "scored_question_count": scored_question_count,
        "average_comparison_score": getattr(run, "average_comparison_score", None),
        "result_based_question_count": result_based_question_count,
        "string_based_question_count": string_based_question_count,
        "comparison_method_counts": dict(getattr(run, "comparison_method_counts", {}) or {}),
        "comparison_mode": _comparison_mode(
            result_based_question_count=result_based_question_count,
            string_based_question_count=string_based_question_count,
        ),
    }
    if scored_question_count == 0 and result_based_question_count == 0 and string_based_question_count == 0:
        return {}
    return metrics


def _aggregate_comparison_metrics(metric_sources: list[dict[str, Any]]) -> dict[str, Any]:
    scored_question_count = 0
    weighted_score_total = 0.0
    result_based_question_count = 0
    string_based_question_count = 0
    comparison_method_counts: dict[str, int] = {}

    for source in metric_sources:
        source_scored_count = int(source.get("scored_question_count", 0) or 0)
        scored_question_count += source_scored_count
        average_score = source.get("average_comparison_score")
        if source_scored_count > 0 and average_score is not None:
            weighted_score_total += float(average_score) * source_scored_count
        result_based_question_count += int(source.get("result_based_question_count", 0) or 0)
        string_based_question_count += int(source.get("string_based_question_count", 0) or 0)
        method_counts = source.get("comparison_method_counts")
        if isinstance(method_counts, dict):
            for method, count in method_counts.items():
                if not method:
                    continue
                comparison_method_counts[str(method)] = comparison_method_counts.get(str(method), 0) + int(count)

    return {
        "scored_question_count": scored_question_count,
        "average_comparison_score": (
            round(weighted_score_total / scored_question_count, 4)
            if scored_question_count > 0
            else None
        ),
        "result_based_question_count": result_based_question_count,
        "string_based_question_count": string_based_question_count,
        "comparison_method_counts": dict(sorted(comparison_method_counts.items())),
        "comparison_mode": _comparison_mode(
            result_based_question_count=result_based_question_count,
            string_based_question_count=string_based_question_count,
        ),
    }
