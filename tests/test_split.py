"""Tests for train/test split and k-fold logic."""

from maxgenie.benchmark_service import BenchmarkQuestion, BenchmarkResult, BenchmarkRun, BenchmarkStatus
from maxgenie.exporter import (
    create_grouped_validation_folds,
    split_train_dev_test,
    split_train_dev_test_balanced,
    split_train_test,
)


def _make_questions(n: int) -> list[BenchmarkQuestion]:
    return [
        BenchmarkQuestion(id=f"q{i:03d}", question=f"Question {i}", expected_sql=f"SELECT {i}")
        for i in range(n)
    ]


def _baseline_run(
    questions: list[BenchmarkQuestion],
    *,
    passed_ids: set[str],
    score_by_id: dict[str, float] | None = None,
) -> BenchmarkRun:
    run = BenchmarkRun(space_id="space-123")
    run.results = [
        BenchmarkResult(
            question_id=question.id,
            question=question.question,
            status=BenchmarkStatus.PASSED if question.id in passed_ids else BenchmarkStatus.FAILED,
            expected_sql=question.expected_sql,
            generated_sql=question.expected_sql if question.id in passed_ids else "SELECT 0",
            comparison_score=(
                (score_by_id or {}).get(question.id, 1.0 if question.id in passed_ids else 0.2)
            ),
        )
        for question in questions
    ]
    return run


def _bucket_pass_rate(questions: list[BenchmarkQuestion], passed_ids: set[str]) -> float:
    if not questions:
        return 0.0
    passed = sum(1 for question in questions if question.id in passed_ids)
    return passed / len(questions)


def test_split_produces_disjoint_sets():
    questions = _make_questions(10)
    split = split_train_test(questions, test_fraction=0.2, seed=42)
    train = split.train_questions
    test = split.test_questions

    train_ids = {q.id for q in train}
    test_ids = {q.id for q in test}
    all_ids = {q.id for q in questions}

    assert train_ids & test_ids == set(), "Train and test sets must be disjoint"
    assert train_ids | test_ids == all_ids, "Train + test must cover all questions"


def test_split_same_seed_same_result():
    questions = _make_questions(15)
    split1 = split_train_test(questions, test_fraction=0.2, seed=99)
    split2 = split_train_test(questions, test_fraction=0.2, seed=99)
    train1, test1 = split1.train_questions, split1.test_questions
    train2, test2 = split2.train_questions, split2.test_questions

    assert [q.id for q in train1] == [q.id for q in train2]
    assert [q.id for q in test1] == [q.id for q in test2]


def test_split_different_seed_different_result():
    questions = _make_questions(15)
    test1 = split_train_test(questions, test_fraction=0.2, seed=1).test_questions
    test2 = split_train_test(questions, test_fraction=0.2, seed=2).test_questions

    assert {q.id for q in test1} != {q.id for q in test2}


def test_split_disabled_when_fraction_zero():
    questions = _make_questions(10)
    split = split_train_test(questions, test_fraction=0.0, seed=42)
    train = split.train_questions
    test = split.test_questions

    assert len(train) == 10
    assert len(test) == 0


def test_split_minimum_one_test_question():
    questions = _make_questions(3)
    split = split_train_test(questions, test_fraction=0.2, seed=42)
    train = split.train_questions
    test = split.test_questions

    assert len(test) >= 1
    assert len(train) >= 1


def test_split_single_question_returns_no_split():
    questions = _make_questions(1)
    split = split_train_test(questions, test_fraction=0.2, seed=42)
    train = split.train_questions
    test = split.test_questions

    assert len(train) == 1
    assert len(test) == 0


def test_split_respects_fraction():
    questions = _make_questions(20)
    split = split_train_test(questions, test_fraction=0.3, seed=42)
    train = split.train_questions
    test = split.test_questions

    assert len(test) == 6  # 30% of 20
    assert len(train) == 14


def test_three_way_split_produces_disjoint_sets():
    questions = _make_questions(15)
    split = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    train_ids = {q.id for q in split.train_questions}
    dev_ids = {q.id for q in split.dev_questions}
    test_ids = {q.id for q in split.test_questions}
    all_ids = {q.id for q in questions}

    assert train_ids & dev_ids == set()
    assert train_ids & test_ids == set()
    assert dev_ids & test_ids == set()
    assert train_ids | dev_ids | test_ids == all_ids
    assert len(split.train_questions) == 9
    assert len(split.dev_questions) == 3
    assert len(split.test_questions) == 3


def test_three_way_split_same_seed_same_result():
    questions = _make_questions(15)
    split1 = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    split2 = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )

    assert [q.id for q in split1.train_questions] == [q.id for q in split2.train_questions]
    assert [q.id for q in split1.dev_questions] == [q.id for q in split2.dev_questions]
    assert [q.id for q in split1.test_questions] == [q.id for q in split2.test_questions]
def test_three_way_split_enables_small_sample_grouped_validation():
    questions = _make_questions(15)
    split = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    visible_ids = {
        question.id for question in [*split.train_questions, *split.dev_questions]
    }
    fold_validation_ids = {
        question_id
        for fold in split.validation_folds
        for question_id in fold.validation_ids
    }

    assert split.validation_strategy == "small_sample_kfold"
    assert len(split.validation_folds) == 3
    assert fold_validation_ids == visible_ids


def test_three_way_split_keeps_blind_validation_for_larger_sets():
    questions = _make_questions(24)
    split = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert split.validation_strategy == "blind_split"
    assert split.validation_folds == []


def test_grouped_validation_folds_cover_visible_pool_once():
    questions = _make_questions(12)
    folds = create_grouped_validation_folds(questions, k=3, seed=42)

    all_validation_ids = [
        question_id
        for fold in folds
        for question_id in fold.validation_ids
    ]

    assert len(folds) == 3
    assert len(all_validation_ids) == 12
    assert set(all_validation_ids) == {question.id for question in questions}


def test_three_way_split_keeps_same_sql_family_in_one_bucket():
    questions = [
        BenchmarkQuestion(id="q1", question="show farxiga sales", expected_sql="SELECT brand_name, SUM(trx) FROM sales WHERE brand_name = 'Farxiga'"),
        BenchmarkQuestion(id="q2", question="show jardiance sales", expected_sql="SELECT brand_name, SUM(trx) FROM sales WHERE brand_name = 'Jardiance'"),
        BenchmarkQuestion(id="q3", question="show current farxiga sales", expected_sql="SELECT brand_name, SUM(trx) FROM sales WHERE brand_name = 'Farxiga'"),
        BenchmarkQuestion(id="q4", question="show current jardiance sales", expected_sql="SELECT brand_name, SUM(trx) FROM sales WHERE brand_name = 'Jardiance'"),
        BenchmarkQuestion(id="q5", question="show market share", expected_sql="SELECT market_name, AVG(share) FROM sales"),
        BenchmarkQuestion(id="q6", question="show regional share", expected_sql="SELECT region_name, AVG(share) FROM sales"),
    ]

    split = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )
    buckets = {
        "train": {question.id for question in split.train_questions},
        "dev": {question.id for question in split.dev_questions},
        "test": {question.id for question in split.test_questions},
    }

    farxiga_family = {"q1", "q3"}
    jardiance_family = {"q2", "q4"}

    assert any(farxiga_family <= bucket for bucket in buckets.values())
    assert any(jardiance_family <= bucket for bucket in buckets.values())


def test_balanced_split_reduces_pass_rate_imbalance_against_baseline():
    questions = _make_questions(15)
    passed_ids = {"q000", "q001", "q002", "q012", "q013", "q014"}
    baseline = _baseline_run(
        questions,
        passed_ids=passed_ids,
        score_by_id={
            "q000": 0.95,
            "q001": 0.92,
            "q002": 0.93,
            "q012": 0.97,
            "q013": 0.96,
            "q014": 0.98,
        },
    )

    stratified = split_train_dev_test(
        questions,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )
    balanced = split_train_dev_test_balanced(
        questions,
        baseline_run=baseline,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    overall_pass_rate = len(passed_ids) / len(questions)
    stratified_imbalance = sum(
        abs(_bucket_pass_rate(bucket, passed_ids) - overall_pass_rate)
        for bucket in (stratified.train_questions, stratified.dev_questions, stratified.test_questions)
    )
    balanced_imbalance = sum(
        abs(_bucket_pass_rate(bucket, passed_ids) - overall_pass_rate)
        for bucket in (balanced.train_questions, balanced.dev_questions, balanced.test_questions)
    )

    assert balanced.split_strategy == "baseline_outcome_balanced"
    assert balanced_imbalance < stratified_imbalance


def test_balanced_split_is_stable_for_same_seed():
    questions = _make_questions(15)
    passed_ids = {"q001", "q004", "q006", "q009", "q011", "q014"}
    baseline = _baseline_run(questions, passed_ids=passed_ids)

    split1 = split_train_dev_test_balanced(
        questions,
        baseline_run=baseline,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    split2 = split_train_dev_test_balanced(
        questions,
        baseline_run=baseline,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )

    assert [q.id for q in split1.train_questions] == [q.id for q in split2.train_questions]
    assert [q.id for q in split1.dev_questions] == [q.id for q in split2.dev_questions]
    assert [q.id for q in split1.test_questions] == [q.id for q in split2.test_questions]
