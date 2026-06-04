"""Export Genie space assets into an editable workspace."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxgenie.assembler import assemble_config
from maxgenie.artifacts import ArtifactManager
from maxgenie.benchmark_service import (
    BenchmarkQuestion,
    BenchmarkRun,
    BenchmarkService,
    BenchmarkStatus,
)
from maxgenie.decomposer import decompose_config
from maxgenie.genie_service import GenieService
from maxgenie.schemas import GenieBenchmarks, GenieBenchmarkAnswer, GenieBenchmarkQuestion, GenieSpaceConfig

_PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "optimization_prompt.md"
_SMALL_SAMPLE_BENCHMARK_THRESHOLD = 20


@dataclass(frozen=True)
class ValidationFold:
    """One grouped validation fold over the visible benchmark pool."""

    fold_index: int
    train_ids: list[str]
    validation_ids: list[str]


@dataclass(frozen=True)
class BenchmarkSplit:
    """Train/validation/test split for benchmark questions."""

    train_questions: list[BenchmarkQuestion]
    dev_questions: list[BenchmarkQuestion]
    test_questions: list[BenchmarkQuestion]
    split_seed: int
    validation_fraction: float
    test_fraction: float
    split_strategy: str = "benchmark_family_stratified"
    validation_strategy: str = "blind_split"
    validation_folds: list[ValidationFold] = field(default_factory=list)

    @property
    def validation_questions(self) -> list[BenchmarkQuestion]:
        """Alias for the validation split."""
        return self.dev_questions


@dataclass(frozen=True)
class ExportBundle:
    """Output of one space export."""

    space_id: str
    space_title: str | None
    config: GenieSpaceConfig
    questions: list[BenchmarkQuestion]
    split: BenchmarkSplit
    prompt_path: Path
    benchmark_source: str
    warehouse_id: str | None


@dataclass(frozen=True)
class _QuestionOutcomeSummary:
    passed: int
    failed: int
    errors: int
    score: float


@dataclass(frozen=True)
class _QuestionGroupSummary:
    indices: tuple[int, ...]
    size: int
    passed: int
    failed: int
    errors: int
    score_sum: float


def benchmark_questions_to_payload(questions: list[BenchmarkQuestion]) -> dict[str, Any]:
    """Serialize benchmark questions for private workspace-only storage."""
    return {
        "question_count": len(questions),
        "questions": serialize_benchmark_questions(questions),
    }


def benchmark_questions_to_model(questions: list[BenchmarkQuestion]) -> GenieBenchmarks:
    """Convert benchmark questions into Genie benchmark config objects."""
    rows = [
        GenieBenchmarkQuestion(
            id=question.id,
            question=[question.question],
            answer=(
                [GenieBenchmarkAnswer(format="SQL", content=[question.expected_sql])]
                if question.expected_sql
                else None
            ),
        )
        for question in questions
    ]
    return GenieBenchmarks(questions=rows)


def extract_benchmark_questions_from_config(config: GenieSpaceConfig) -> list[BenchmarkQuestion]:
    """Extract benchmark questions from config."""
    questions: list[BenchmarkQuestion] = []
    if not config.benchmarks or not config.benchmarks.questions:
        return questions

    for benchmark in config.benchmarks.questions:
        question_text = benchmark.question[0] if benchmark.question else ""
        expected_sql = None
        if benchmark.answer:
            for answer in benchmark.answer:
                if answer.format.upper() == "SQL":
                    expected_sql = "".join(answer.content) if answer.content else None
                    break

        questions.append(
            BenchmarkQuestion(
                id=benchmark.id,
                question=question_text,
                expected_sql=expected_sql,
            )
        )
    return questions


def serialize_benchmark_questions(questions: list[BenchmarkQuestion]) -> list[dict[str, Any]]:
    """Serialize benchmark questions with full details."""
    return [
        {
            "id": question.id,
            "question": question.question,
            "expected_sql": question.expected_sql,
        }
        for question in questions
    ]


def _question_intent_signature(question: str) -> str:
    """Build a coarse intent signature so related phrasings stay in the same split."""
    normalized = question.lower()
    normalized = re.sub(r'"[^"]*"', "<value>", normalized)
    normalized = re.sub(r"'[^']*'", "<value>", normalized)
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    tokens = re.findall(r"[a-z<>]+", normalized)
    structural_tokens = {
        "show",
        "list",
        "what",
        "compare",
        "difference",
        "diff",
        "percentage",
        "percent",
        "sales",
        "market",
        "speciality",
        "specialty",
        "view",
        "current",
        "previous",
        "past",
        "week",
        "weeks",
        "month",
        "months",
        "quarter",
        "quarters",
        "year",
        "years",
        "national",
        "regional",
        "region",
        "regions",
        "level",
        "team",
        "teams",
        "brand",
        "brands",
        "trx",
        "nrx",
        "nbrx",
        "share",
        "performance",
        "vs",
        "by",
        "for",
        "in",
        "at",
        "from",
        "to",
        "on",
        "with",
        "and",
    }
    signature_tokens: list[str] = []
    for token in tokens:
        signature_tokens.append(token if token in structural_tokens else "<entity>")
    collapsed: list[str] = []
    for token in signature_tokens:
        if not collapsed or collapsed[-1] != token:
            collapsed.append(token)
    return " ".join(collapsed[:12]) or "<empty>"


def _normalize_sql_family(sql: str) -> str:
    """Normalize SQL into a family signature for split grouping."""
    normalized = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    normalized = re.sub(r"--[^\r\n]*", " ", normalized)
    normalized = normalized.lower()
    normalized = normalized.replace("`", "")
    normalized = re.sub(r"\s+", " ", normalized).strip(" ;\n\r\t")
    return normalized


def _benchmark_family_signature(question: BenchmarkQuestion) -> str:
    """Group benchmark variants by expected SQL when available."""
    if question.expected_sql:
        normalized_sql = _normalize_sql_family(question.expected_sql)
        if normalized_sql:
            return f"sql::{normalized_sql}"
    return f"intent::{_question_intent_signature(question.question)}"


def _recommended_validation_fold_count(
    *,
    total_question_count: int,
    visible_question_count: int,
    validation_enabled: bool,
) -> int:
    """Choose a stable grouped-fold count for small benchmark sets."""
    if not validation_enabled:
        return 0
    if total_question_count >= _SMALL_SAMPLE_BENCHMARK_THRESHOLD or visible_question_count < 4:
        return 0
    if visible_question_count < 9:
        return 2
    return 3


def _compute_split_counts(
    total_question_count: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    if validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("validation_fraction and test_fraction must be non-negative.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.0.")
    if total_question_count < 2 or (validation_fraction <= 0.0 and test_fraction <= 0.0):
        return 0, 0
    if validation_fraction > 0.0 and total_question_count < 3:
        return 0, 0

    dev_count = (
        max(1, round(total_question_count * validation_fraction))
        if validation_fraction > 0.0
        else 0
    )
    test_count = max(1, round(total_question_count * test_fraction))
    max_holdout = total_question_count - 1
    if dev_count + test_count > max_holdout:
        overflow = dev_count + test_count - max_holdout
        if test_count >= dev_count and test_count > 0:
            test_count = max(1, test_count - overflow)
        else:
            dev_count = max(0, dev_count - overflow)
    if dev_count + test_count > max_holdout:
        test_count = max(0, max_holdout - dev_count)
    return dev_count, test_count


def create_grouped_validation_folds(
    questions: list[BenchmarkQuestion],
    k: int,
    seed: int,
) -> list[ValidationFold]:
    """Create grouped cross-validation folds using coarse question intent."""
    if k < 2:
        raise ValueError("k must be at least 2.")
    if len(questions) < k:
        raise ValueError("Need at least one visible question per fold.")

    groups: dict[str, list[int]] = {}
    for index, question in enumerate(questions):
        groups.setdefault(_benchmark_family_signature(question), []).append(index)

    rng = random.Random(seed)
    grouped_indices = [list(indices) for indices in groups.values()]
    for group in grouped_indices:
        rng.shuffle(group)
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)

    fold_buckets: list[list[int]] = [[] for _ in range(k)]
    for group in grouped_indices:
        target_index = min(range(k), key=lambda idx: (len(fold_buckets[idx]), idx))
        fold_buckets[target_index].extend(group)

    for fold_index, bucket in enumerate(fold_buckets):
        if bucket:
            continue
        source_index = max(range(k), key=lambda idx: len(fold_buckets[idx]))
        moved = fold_buckets[source_index].pop()
        fold_buckets[fold_index].append(moved)

    question_order = [question.id for question in questions]
    folds: list[ValidationFold] = []
    for fold_index, bucket in enumerate(fold_buckets, start=1):
        validation_set = set(bucket)
        folds.append(
            ValidationFold(
                fold_index=fold_index,
                train_ids=[
                    question_id
                    for index, question_id in enumerate(question_order)
                    if index not in validation_set
                ],
                validation_ids=[
                    question_id
                    for index, question_id in enumerate(question_order)
                    if index in validation_set
                ],
            )
        )

    return folds


def _finalize_split(
    *,
    questions: list[BenchmarkQuestion],
    train_indices: set[int],
    dev_indices: set[int],
    test_indices: set[int],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    split_strategy: str,
) -> BenchmarkSplit:
    train_questions = [question for index, question in enumerate(questions) if index in train_indices]
    dev_questions = [question for index, question in enumerate(questions) if index in dev_indices]
    test_questions = [question for index, question in enumerate(questions) if index in test_indices]
    visible_questions = train_questions + dev_questions
    fold_count = _recommended_validation_fold_count(
        total_question_count=len(questions),
        visible_question_count=len(visible_questions),
        validation_enabled=bool(dev_questions),
    )
    validation_folds = (
        create_grouped_validation_folds(visible_questions, k=fold_count, seed=seed)
        if fold_count
        else []
    )

    return BenchmarkSplit(
        train_questions=train_questions,
        dev_questions=dev_questions,
        test_questions=test_questions,
        split_seed=seed,
        validation_fraction=validation_fraction if dev_questions else 0.0,
        test_fraction=test_fraction if test_questions else 0.0,
        split_strategy=split_strategy,
        validation_strategy="small_sample_kfold" if validation_folds else "blind_split",
        validation_folds=validation_folds,
    )


def split_train_dev_test(
    questions: list[BenchmarkQuestion],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> BenchmarkSplit:
    """Split benchmark questions into train/dev/test sets."""
    dev_count, test_count = _compute_split_counts(
        len(questions),
        validation_fraction,
        test_fraction,
    )
    if len(questions) < 2 or (dev_count == 0 and test_count == 0):
        return BenchmarkSplit(
            train_questions=list(questions),
            dev_questions=[],
            test_questions=[],
            split_seed=seed,
            validation_fraction=0.0,
            test_fraction=0.0,
            split_strategy="benchmark_family_stratified",
        )

    groups: dict[str, list[int]] = {}
    for index, question in enumerate(questions):
        groups.setdefault(_benchmark_family_signature(question), []).append(index)

    rng = random.Random(seed)
    grouped_indices = list(groups.values())
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)

    test_indices: set[int] = set()
    dev_indices: set[int] = set()
    train_indices: set[int] = set()
    for group in grouped_indices:
        group_set = set(group)
        remaining_after_group = len(questions) - (
            len(test_indices) + len(dev_indices) + len(train_indices) + len(group_set)
        )
        bucket_scores: list[tuple[float, str]] = []
        current_test = len(test_indices)
        current_dev = len(dev_indices)
        current_train = len(train_indices)

        if current_train + len(group_set) + remaining_after_group >= 1:
            bucket_scores.append(
                (
                    abs(test_count - current_test) + abs(dev_count - current_dev),
                    "train",
                )
            )
        if current_train + remaining_after_group >= 1:
            bucket_scores.append(
                (
                    abs(test_count - (current_test + len(group_set))) + abs(dev_count - current_dev),
                    "test",
                )
            )
        if current_train + remaining_after_group >= 1:
            bucket_scores.append(
                (
                    abs(test_count - current_test) + abs(dev_count - (current_dev + len(group_set))),
                    "dev",
                )
            )

        _, bucket = min(
            bucket_scores,
            key=lambda item: (
                item[0],
                {"test": 0, "dev": 1, "train": 2}[item[1]],
            ),
        )
        if bucket == "test":
            test_indices.update(group_set)
        elif bucket == "dev":
            dev_indices.update(group_set)
        else:
            train_indices.update(group_set)

    all_indices = set(range(len(questions)))
    assigned_holdout = test_indices | dev_indices
    train_indices |= all_indices - assigned_holdout

    while len(train_indices) == 0 and dev_indices:
        moved = min(dev_indices)
        dev_indices.remove(moved)
        train_indices.add(moved)
    while len(train_indices) == 0 and test_indices:
        moved = min(test_indices)
        test_indices.remove(moved)
        train_indices.add(moved)

    return _finalize_split(
        questions=questions,
        train_indices=train_indices,
        dev_indices=dev_indices,
        test_indices=test_indices,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        split_strategy="benchmark_family_stratified",
    )


def _result_outcome_summary(result: Any) -> _QuestionOutcomeSummary:
    raw_status = getattr(result.status, "value", result.status)
    status = BenchmarkStatus(str(raw_status))
    if result.comparison_score is not None:
        score = float(result.comparison_score)
    elif status == BenchmarkStatus.PASSED:
        score = 1.0
    else:
        score = 0.0
    return _QuestionOutcomeSummary(
        passed=1 if status == BenchmarkStatus.PASSED else 0,
        failed=1 if status == BenchmarkStatus.FAILED else 0,
        errors=1 if status == BenchmarkStatus.ERROR else 0,
        score=score,
    )


def _build_group_summaries(
    questions: list[BenchmarkQuestion],
    baseline_run: BenchmarkRun,
) -> list[_QuestionGroupSummary]:
    baseline_by_id = {result.question_id: result for result in baseline_run.results}
    missing_ids = [question.id for question in questions if question.id not in baseline_by_id]
    if missing_ids:
        raise ValueError(
            "Baseline run is missing benchmark questions required for balanced splitting: "
            + ", ".join(sorted(missing_ids))
        )

    grouped_indices: dict[str, list[int]] = {}
    for index, question in enumerate(questions):
        grouped_indices.setdefault(_benchmark_family_signature(question), []).append(index)

    group_summaries: list[_QuestionGroupSummary] = []
    for indices in grouped_indices.values():
        passed = 0
        failed = 0
        errors = 0
        score_sum = 0.0
        for index in indices:
            summary = _result_outcome_summary(baseline_by_id[questions[index].id])
            passed += summary.passed
            failed += summary.failed
            errors += summary.errors
            score_sum += summary.score
        group_summaries.append(
            _QuestionGroupSummary(
                indices=tuple(indices),
                size=len(indices),
                passed=passed,
                failed=failed,
                errors=errors,
                score_sum=score_sum,
            )
        )
    return group_summaries


def _empty_bucket_summary() -> dict[str, float]:
    return {
        "size": 0.0,
        "passed": 0.0,
        "failed": 0.0,
        "errors": 0.0,
        "score_sum": 0.0,
    }


def _accumulate_group(bucket_summary: dict[str, float], group: _QuestionGroupSummary) -> dict[str, float]:
    return {
        "size": bucket_summary["size"] + group.size,
        "passed": bucket_summary["passed"] + group.passed,
        "failed": bucket_summary["failed"] + group.failed,
        "errors": bucket_summary["errors"] + group.errors,
        "score_sum": bucket_summary["score_sum"] + group.score_sum,
    }


def _bucket_objective(
    bucket_summary: dict[str, float],
    *,
    target_size: int,
    global_pass_rate: float,
    global_fail_rate: float,
    global_error_rate: float,
    global_mean_score: float,
) -> float:
    size = bucket_summary["size"]
    if target_size == 0:
        return 0.0 if size == 0 else 1000.0 + size * 50.0

    size_penalty = abs(size - target_size) * 8.0
    if size == 0:
        return size_penalty + 25.0

    pass_rate = bucket_summary["passed"] / size
    fail_rate = bucket_summary["failed"] / size
    error_rate = bucket_summary["errors"] / size
    mean_score = bucket_summary["score_sum"] / size
    return (
        size_penalty
        + abs(pass_rate - global_pass_rate) * 5.0
        + abs(fail_rate - global_fail_rate) * 4.0
        + abs(error_rate - global_error_rate) * 6.0
        + abs(mean_score - global_mean_score) * 3.0
    )


def _assignment_objective(
    assignment: list[str],
    groups: list[_QuestionGroupSummary],
    *,
    target_sizes: dict[str, int],
    global_pass_rate: float,
    global_fail_rate: float,
    global_error_rate: float,
    global_mean_score: float,
) -> float:
    bucket_summaries = {
        "train": _empty_bucket_summary(),
        "dev": _empty_bucket_summary(),
        "test": _empty_bucket_summary(),
    }
    for bucket_name, group in zip(assignment, groups):
        bucket_summaries[bucket_name] = _accumulate_group(bucket_summaries[bucket_name], group)

    if bucket_summaries["train"]["size"] <= 0:
        return float("inf")
    if target_sizes["dev"] > 0 and bucket_summaries["dev"]["size"] <= 0:
        return float("inf")
    if target_sizes["test"] > 0 and bucket_summaries["test"]["size"] <= 0:
        return float("inf")

    total = 0.0
    for bucket_name, summary in bucket_summaries.items():
        total += _bucket_objective(
            summary,
            target_size=target_sizes[bucket_name],
            global_pass_rate=global_pass_rate,
            global_fail_rate=global_fail_rate,
            global_error_rate=global_error_rate,
            global_mean_score=global_mean_score,
        )
    non_empty_bucket_names = [
        bucket_name
        for bucket_name, summary in bucket_summaries.items()
        if summary["size"] > 0
    ]
    for index, bucket_name in enumerate(non_empty_bucket_names):
        first_pass_rate = bucket_summaries[bucket_name]["passed"] / bucket_summaries[bucket_name]["size"]
        for other_bucket_name in non_empty_bucket_names[index + 1:]:
            second_pass_rate = (
                bucket_summaries[other_bucket_name]["passed"]
                / bucket_summaries[other_bucket_name]["size"]
            )
            total += abs(first_pass_rate - second_pass_rate) * 2.0
    return round(total, 6)


def _initial_balanced_assignment(
    groups: list[_QuestionGroupSummary],
    *,
    target_sizes: dict[str, int],
    global_pass_rate: float,
    global_fail_rate: float,
    global_error_rate: float,
    global_mean_score: float,
    seed: int,
    trial_index: int,
) -> list[str]:
    rng = random.Random((seed + 1) * 1009 + trial_index)
    order = list(range(len(groups)))
    rng.shuffle(order)
    order.sort(
        key=lambda index: (
            groups[index].size,
            abs((groups[index].passed / groups[index].size) - global_pass_rate),
            abs((groups[index].score_sum / groups[index].size) - global_mean_score),
        ),
        reverse=True,
    )

    assignment = ["train"] * len(groups)
    bucket_summaries = {
        "train": _empty_bucket_summary(),
        "dev": _empty_bucket_summary(),
        "test": _empty_bucket_summary(),
    }
    active_buckets = [
        bucket_name
        for bucket_name in ("train", "dev", "test")
        if bucket_name == "train" or target_sizes[bucket_name] > 0
    ]

    for index in order:
        group = groups[index]
        bucket_order = list(active_buckets)
        rng.shuffle(bucket_order)
        bucket_order.sort(key=lambda bucket_name: {"train": 0, "dev": 1, "test": 2}[bucket_name])

        best_bucket = "train"
        best_score = float("inf")
        for bucket_name in bucket_order:
            candidate_summary = _accumulate_group(bucket_summaries[bucket_name], group)
            score = _bucket_objective(
                candidate_summary,
                target_size=target_sizes[bucket_name],
                global_pass_rate=global_pass_rate,
                global_fail_rate=global_fail_rate,
                global_error_rate=global_error_rate,
                global_mean_score=global_mean_score,
            )
            overflow = max(candidate_summary["size"] - target_sizes[bucket_name], 0.0)
            score += overflow * 6.0
            if score < best_score:
                best_score = score
                best_bucket = bucket_name
        assignment[index] = best_bucket
        bucket_summaries[best_bucket] = _accumulate_group(bucket_summaries[best_bucket], group)
    return assignment


def _refine_balanced_assignment(
    assignment: list[str],
    groups: list[_QuestionGroupSummary],
    *,
    target_sizes: dict[str, int],
    global_pass_rate: float,
    global_fail_rate: float,
    global_error_rate: float,
    global_mean_score: float,
) -> list[str]:
    active_buckets = [
        bucket_name
        for bucket_name in ("train", "dev", "test")
        if bucket_name == "train" or target_sizes[bucket_name] > 0
    ]
    best_assignment = list(assignment)
    best_score = _assignment_objective(
        best_assignment,
        groups,
        target_sizes=target_sizes,
        global_pass_rate=global_pass_rate,
        global_fail_rate=global_fail_rate,
        global_error_rate=global_error_rate,
        global_mean_score=global_mean_score,
    )
    improved = True
    while improved:
        improved = False
        for index, current_bucket in enumerate(list(best_assignment)):
            for bucket_name in active_buckets:
                if bucket_name == current_bucket:
                    continue
                candidate = list(best_assignment)
                candidate[index] = bucket_name
                score = _assignment_objective(
                    candidate,
                    groups,
                    target_sizes=target_sizes,
                    global_pass_rate=global_pass_rate,
                    global_fail_rate=global_fail_rate,
                    global_error_rate=global_error_rate,
                    global_mean_score=global_mean_score,
                )
                if score < best_score:
                    best_assignment = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        for left in range(len(best_assignment)):
            for right in range(left + 1, len(best_assignment)):
                if best_assignment[left] == best_assignment[right]:
                    continue
                candidate = list(best_assignment)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                score = _assignment_objective(
                    candidate,
                    groups,
                    target_sizes=target_sizes,
                    global_pass_rate=global_pass_rate,
                    global_fail_rate=global_fail_rate,
                    global_error_rate=global_error_rate,
                    global_mean_score=global_mean_score,
                )
                if score < best_score:
                    best_assignment = candidate
                    best_score = score
                    improved = True
                    break
            if improved:
                break
    return best_assignment


def split_train_dev_test_balanced(
    questions: list[BenchmarkQuestion],
    baseline_run: BenchmarkRun,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> BenchmarkSplit:
    """Split questions after a full baseline to balance difficulty across buckets."""
    dev_count, test_count = _compute_split_counts(
        len(questions),
        validation_fraction,
        test_fraction,
    )
    if len(questions) < 2 or (dev_count == 0 and test_count == 0):
        return BenchmarkSplit(
            train_questions=list(questions),
            dev_questions=[],
            test_questions=[],
            split_seed=seed,
            validation_fraction=0.0,
            test_fraction=0.0,
            split_strategy="baseline_outcome_balanced",
        )

    groups = _build_group_summaries(questions, baseline_run)
    total_questions = len(questions)
    total_passed = sum(group.passed for group in groups)
    total_failed = sum(group.failed for group in groups)
    total_errors = sum(group.errors for group in groups)
    total_score = sum(group.score_sum for group in groups)

    target_sizes = {
        "train": max(total_questions - dev_count - test_count, 1),
        "dev": dev_count,
        "test": test_count,
    }
    global_pass_rate = total_passed / total_questions if total_questions else 0.0
    global_fail_rate = total_failed / total_questions if total_questions else 0.0
    global_error_rate = total_errors / total_questions if total_questions else 0.0
    global_mean_score = total_score / total_questions if total_questions else 0.0

    trial_count = min(max(len(groups) * 12, 48), 512)
    best_assignment: list[str] | None = None
    best_score = float("inf")
    for trial_index in range(trial_count):
        candidate = _initial_balanced_assignment(
            groups,
            target_sizes=target_sizes,
            global_pass_rate=global_pass_rate,
            global_fail_rate=global_fail_rate,
            global_error_rate=global_error_rate,
            global_mean_score=global_mean_score,
            seed=seed,
            trial_index=trial_index,
        )
        candidate = _refine_balanced_assignment(
            candidate,
            groups,
            target_sizes=target_sizes,
            global_pass_rate=global_pass_rate,
            global_fail_rate=global_fail_rate,
            global_error_rate=global_error_rate,
            global_mean_score=global_mean_score,
        )
        score = _assignment_objective(
            candidate,
            groups,
            target_sizes=target_sizes,
            global_pass_rate=global_pass_rate,
            global_fail_rate=global_fail_rate,
            global_error_rate=global_error_rate,
            global_mean_score=global_mean_score,
        )
        if score < best_score:
            best_assignment = candidate
            best_score = score

    if best_assignment is None:
        raise ValueError("Could not produce a balanced benchmark split from the baseline run.")

    train_indices: set[int] = set()
    dev_indices: set[int] = set()
    test_indices: set[int] = set()
    for bucket_name, group in zip(best_assignment, groups):
        target = {
            "train": train_indices,
            "dev": dev_indices,
            "test": test_indices,
        }[bucket_name]
        target.update(group.indices)

    return _finalize_split(
        questions=questions,
        train_indices=train_indices,
        dev_indices=dev_indices,
        test_indices=test_indices,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        split_strategy="baseline_outcome_balanced",
    )


def split_train_test(
    questions: list[BenchmarkQuestion],
    test_fraction: float,
    seed: int,
) -> BenchmarkSplit:
    """Backward-compatible wrapper for train/test splits."""
    return split_train_dev_test(
        questions,
        validation_fraction=0.0,
        test_fraction=test_fraction,
        seed=seed,
    )


def render_optimization_task(
    space_id: str,
    train_count: int,
    validation_count: int,
    test_count: int,
    split_strategy: str,
    validation_strategy: str,
    validation_fold_count: int,
) -> str:
    """Render the workspace task file for the optimizing CLI session."""
    template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    if validation_strategy == "small_sample_kfold":
        validation_notes = (
            f"- `benchmarks/validation_folds.json` defines `{validation_fold_count}` grouped validation folds over the visible pool.\n"
            "- `maxgenie benchmark --validation` runs grouped validation folds and writes an aggregate score plus fold summary.\n"
        )
    else:
        validation_notes = (
            f"- `benchmarks/validation_set.json` contains `{validation_count}` blind validation IDs only.\n"
            "- `maxgenie benchmark --validation` runs the blind validation set and writes only a pass-rate score.\n"
        )
    return template.format(
        space_id=space_id,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        split_strategy=split_strategy,
        validation_notes=validation_notes,
    )


class SpaceExporter:
    """Exports remote Genie space config and benchmark questions."""

    def __init__(
        self,
        genie_service: GenieService,
        benchmark_service: BenchmarkService,
        artifacts: ArtifactManager,
    ):
        self.genie_service = genie_service
        self.benchmark_service = benchmark_service
        self.artifacts = artifacts

    def _workspace_config_for_recovery(self, workspace_dir: Path) -> GenieSpaceConfig | None:
        space_dir = workspace_dir / "space"
        for legacy_path in (space_dir / "current.space.yml", space_dir / "original.space.yml"):
            if legacy_path.exists():
                return GenieSpaceConfig.from_yaml(str(legacy_path))

        meta_path = space_dir / "meta.yml"
        if meta_path.exists():
            try:
                return assemble_config(space_dir)
            except Exception:
                return None
        return None

    def _table_signature(self, config: GenieSpaceConfig) -> tuple[str, ...]:
        identifiers: list[str] = []
        data_sources = config.data_sources
        if data_sources:
            if data_sources.tables:
                identifiers.extend(table.identifier for table in data_sources.tables if table.identifier)
            if data_sources.metric_views:
                identifiers.extend(view.identifier for view in data_sources.metric_views if view.identifier)
        return tuple(sorted(set(identifiers)))

    def _normalize_title(self, value: str | None) -> str:
        if not value:
            return ""
        normalized = re.sub(r"\([^)]*\)", " ", value.lower())
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    def _recover_local_benchmarks(
        self,
        *,
        space_id: str,
        config: GenieSpaceConfig,
    ) -> tuple[list[BenchmarkQuestion], str] | None:
        workspace_root = self.artifacts.workspace_dir.parent
        source_tables = self._table_signature(config)
        if not source_tables or not workspace_root.exists():
            return None

        source_title = self._normalize_title(config.title)
        best_candidate: tuple[int, list[BenchmarkQuestion], str] | None = None

        for candidate_dir in sorted(workspace_root.iterdir()):
            if not candidate_dir.is_dir() or candidate_dir.name == space_id:
                continue

            benchmark_path = candidate_dir / "benchmarks" / "full.benchmarks.json"
            if not benchmark_path.exists():
                continue

            candidate_config = self._workspace_config_for_recovery(candidate_dir)
            if candidate_config is None:
                continue

            candidate_tables = self._table_signature(candidate_config)
            if candidate_tables != source_tables:
                continue

            questions = self.load_questions_from_file(benchmark_path)
            if not questions:
                continue

            candidate_title = self._normalize_title(candidate_config.title)
            score = len(questions)
            if candidate_title == source_title:
                score += 100
            elif source_title and candidate_title and (
                source_title in candidate_title or candidate_title in source_title
            ):
                score += 50

            if (config.description or "").strip() and (config.description or "").strip() == (
                candidate_config.description or ""
            ).strip():
                score += 25

            source_examples = (
                len(config.config.sample_questions)
                if config.config and config.config.sample_questions
                else 0
            )
            candidate_examples = (
                len(candidate_config.config.sample_questions)
                if candidate_config.config and candidate_config.config.sample_questions
                else 0
            )
            if source_examples == candidate_examples and source_examples > 0:
                score += 10

            benchmark_source = f"local_workspace:{candidate_dir.name}"
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, questions, benchmark_source)

        if best_candidate is None:
            return None
        return best_candidate[1], best_candidate[2]

    def _write_workspace_bundle(
        self,
        *,
        space_id: str,
        space_title: str | None,
        config: GenieSpaceConfig,
        questions: list[BenchmarkQuestion],
        benchmark_source: str,
        workspace_host: str | None,
        validation_fraction: float,
        test_fraction: float,
        split_seed: int,
        split: BenchmarkSplit | None = None,
    ) -> ExportBundle:
        if split is None:
            split = split_train_dev_test(
                questions,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=split_seed,
            )
        self._reset_split_dependent_results()

        decompose_config(config, self.artifacts.space_dir, include_benchmarks=False)

        self.artifacts.write_json_path(
            self.artifacts.train_set_path,
            {
                "space_id": space_id,
                "question_count": len(split.train_questions),
                "questions": serialize_benchmark_questions(split.train_questions),
            },
        )
        self.artifacts.write_json_path(
            self.artifacts.validation_set_path,
            {
                "space_id": space_id,
                "question_count": len(split.validation_questions),
                "ids": [question.id for question in split.validation_questions],
            },
        )
        self.artifacts.write_json_path(
            self.artifacts.validation_folds_path,
            {
                "space_id": space_id,
                "validation_strategy": split.validation_strategy,
                "visible_question_count": len(split.train_questions) + len(split.validation_questions),
                "fold_count": len(split.validation_folds),
                "folds": [
                    {
                        "fold_index": fold.fold_index,
                        "train_ids": fold.train_ids,
                        "validation_ids": fold.validation_ids,
                    }
                    for fold in split.validation_folds
                ],
            },
        )
        self.artifacts.write_json_path(
            self.artifacts.test_set_path,
            {
                "space_id": space_id,
                "question_count": len(split.test_questions),
                "ids": [question.id for question in split.test_questions],
            },
        )
        self.artifacts.write_json_path(
            self.artifacts.split_meta_path,
            {
                "space_id": space_id,
                "source_space_id": space_id,
                "workspace_host": workspace_host,
                "warehouse_id": config.warehouse_id,
                "benchmark_source": benchmark_source,
                "split_seed": split.split_seed,
                "validation_fraction": split.validation_fraction,
                "test_fraction": split.test_fraction,
                "train_ids": [question.id for question in split.train_questions],
                "validation_ids": [question.id for question in split.validation_questions],
                "test_ids": [question.id for question in split.test_questions],
                "visible_ids": [
                    question.id for question in [*split.train_questions, *split.validation_questions]
                ],
                "split_strategy": split.split_strategy,
                "validation_strategy": split.validation_strategy,
                "validation_fold_count": len(split.validation_folds),
            },
        )

        if questions:
            self.artifacts.write_json_path(
                self.artifacts.benchmark_payload_path,
                {
                    "benchmark_source": benchmark_source,
                    "payload_scope": "visible_pool",
                    **benchmark_questions_to_payload(
                        [*split.train_questions, *split.validation_questions],
                    ),
                },
            )
            self.artifacts.write_json_path(
                self.artifacts.hidden_test_payload_path,
                {
                    "benchmark_source": benchmark_source,
                    "payload_scope": "hidden_test",
                    **benchmark_questions_to_payload(split.test_questions),
                },
            )

        prompt_path = self.artifacts.write_text_path(
            self.artifacts.prompt_path,
            render_optimization_task(
                space_id=space_id,
                train_count=len(split.train_questions),
                validation_count=len(split.validation_questions),
                test_count=len(split.test_questions),
                split_strategy=split.split_strategy,
                validation_strategy=split.validation_strategy,
                validation_fold_count=len(split.validation_folds),
            ),
        )

        return ExportBundle(
            space_id=space_id,
            space_title=space_title,
            config=config,
            questions=questions,
            split=split,
            prompt_path=prompt_path,
            benchmark_source=benchmark_source,
            warehouse_id=config.warehouse_id,
        )

    def rewrite_split_from_baseline(
        self,
        *,
        space_id: str,
        space_title: str | None,
        config: GenieSpaceConfig,
        questions: list[BenchmarkQuestion],
        benchmark_source: str,
        workspace_host: str | None,
        validation_fraction: float,
        test_fraction: float,
        split_seed: int,
        baseline_run: BenchmarkRun,
    ) -> ExportBundle:
        split = split_train_dev_test_balanced(
            questions,
            baseline_run=baseline_run,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
        )
        return self._write_workspace_bundle(
            space_id=space_id,
            space_title=space_title,
            config=config,
            questions=questions,
            benchmark_source=benchmark_source,
            workspace_host=workspace_host,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            split=split,
        )

    def _reset_split_dependent_results(self) -> None:
        """Clear benchmark summary artifacts that depend on the current split definition."""
        stale_paths = (
            self.artifacts.baseline_train_path,
            self.artifacts.latest_train_path,
            self.artifacts.baseline_dev_score_path,
            self.artifacts.latest_dev_score_path,
            self.artifacts.baseline_dev_summary_path,
            self.artifacts.latest_dev_summary_path,
            self.artifacts.baseline_test_score_path,
            self.artifacts.latest_test_score_path,
            self.artifacts.latest_full_summary_path,
            self.artifacts.space_diagnostics_path,
            self.artifacts.curation_summary_path,
            self.artifacts.query_history_path,
            self.artifacts.query_history_summary_path,
            self.artifacts.query_history_recommendations_path,
            self.artifacts.final_report_path,
            self.artifacts.benchmark_payload_path,
            self.artifacts.hidden_test_payload_path,
        )
        for path in stale_paths:
            if path.exists():
                path.unlink()

    def export(
        self,
        space_id: str,
        *,
        workspace_host: str | None = None,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        split_seed: int = 42,
    ) -> ExportBundle:
        info, config = self.genie_service.get(space_id)
        if not config.warehouse_id and info.warehouse_id:
            config.warehouse_id = info.warehouse_id

        questions = extract_benchmark_questions_from_config(config)
        benchmark_source = "space_config"
        if not questions:
            questions = self.benchmark_service.get_space_benchmarks(space_id)
            benchmark_source = "space_api"
        if not questions:
            recovered = self._recover_local_benchmarks(space_id=space_id, config=config)
            if recovered is not None:
                questions, benchmark_source = recovered

        return self._write_workspace_bundle(
            space_id=space_id,
            space_title=info.title,
            config=config,
            questions=questions,
            benchmark_source=benchmark_source,
            workspace_host=workspace_host,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
        )

    def export_from_local_files(
        self,
        space_id: str,
        *,
        workspace_host: str | None = None,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        split_seed: int = 42,
    ) -> ExportBundle:
        """Migrate a legacy local workspace into the decomposed layout."""
        legacy_candidates = (
            self.artifacts.space_dir / "current.space.yml",
            self.artifacts.space_dir / "original.space.yml",
        )
        config_path = next((path for path in legacy_candidates if path.exists()), None)
        if config_path is None:
            raise FileNotFoundError("No legacy local space YAML found.")

        config = GenieSpaceConfig.from_yaml(str(config_path))
        info_warehouse_id = None
        try:
            info, _ = self.genie_service.get(space_id)
            info_warehouse_id = info.warehouse_id
        except Exception:
            info_warehouse_id = None
        if not config.warehouse_id and info_warehouse_id:
            config.warehouse_id = info_warehouse_id
        questions = extract_benchmark_questions_from_config(config)
        benchmark_source = f"legacy_config:{config_path.name}"
        legacy_benchmarks = self.artifacts.benchmarks_dir / "full.benchmarks.json"
        if not questions and legacy_benchmarks.exists():
            questions = self.load_questions_from_file(legacy_benchmarks)
            benchmark_source = f"legacy_benchmarks:{legacy_benchmarks.name}"

        return self._write_workspace_bundle(
            space_id=space_id,
            space_title=config.title,
            config=config,
            questions=questions,
            benchmark_source=benchmark_source,
            workspace_host=workspace_host,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
        )

    def load_questions_from_file(self, path: str | Path) -> list[BenchmarkQuestion]:
        """Load benchmark questions from an exported JSON file."""
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        rows = data.get("questions", []) if isinstance(data, dict) else data
        return [
            BenchmarkQuestion(
                id=row.get("id", f"q{i + 1}"),
                question=row["question"],
                expected_sql=row.get("expected_sql"),
            )
            for i, row in enumerate(rows)
        ]
