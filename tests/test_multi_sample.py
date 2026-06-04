"""Tests for the multi-sample scoring and minimum-detectable-difference core.

The numbers below are anchored to a measured run-to-run noise study so these
tests double as a regression lock on the documented promotion contract:

    baseline full pass rates : [26.67, 20, 20, 20, 26.67, 20]  -> mean 22.22, s 3.44
    incumbent full pass rates: [60, 53.33, 60, 66.67, 66.67, 53.33] -> mean 60.00, s 5.97
    sigma_hat_full = max(3.44, 5.97) = 5.97  ->  MDD_full = 2 * sigma = ~11.93 pts
"""

from __future__ import annotations

import math

import pytest

from maxgenie.benchmark_service import BenchmarkResult, BenchmarkRun, BenchmarkStatus
from maxgenie.multi_sample import (
    NOISE_MODEL_SCHEMA,
    PromotionDecision,
    SampleAggregate,
    aggregate_samples,
    build_noise_model,
    lift_threshold,
    minimum_detectable_difference,
    promotion_decision,
    sample_std,
)


def _run_from_pass_flags(pass_flags: list[bool], *, qids: list[str] | None = None) -> BenchmarkRun:
    """Build a BenchmarkRun where pass_flags[i] decides question i's status."""
    if qids is None:
        qids = [f"Q{i + 1:02d}" for i in range(len(pass_flags))]
    assert len(qids) == len(pass_flags)
    results = [
        BenchmarkResult(
            question_id=qid,
            question=f"question {qid}",
            status=BenchmarkStatus.PASSED if passed else BenchmarkStatus.FAILED,
            comparison_score=1.0 if passed else 0.0,
            comparison_method="result_match" if passed else "result_mismatch",
        )
        for qid, passed in zip(qids, pass_flags)
    ]
    return BenchmarkRun(space_id="clone-space", results=results)


def _run_with_passed_count(passed: int, total: int = 15) -> BenchmarkRun:
    return _run_from_pass_flags([i < passed for i in range(total)])


# Pass counts out of 15 reproducing the documented baseline / incumbent series.
_BASELINE_PASS_COUNTS = [4, 3, 3, 3, 4, 3]  # -> 26.67, 20, 20, 20, 26.67, 20
_INCUMBENT_PASS_COUNTS = [9, 8, 9, 10, 10, 8]  # -> 60, 53.33, 60, 66.67, 66.67, 53.33


# --------------------------------------------------------------------------- #
# sample_std
# --------------------------------------------------------------------------- #
def test_sample_std_matches_ddof_one():
    # ddof=1 sample std of the baseline full series.
    values = [26.67, 20.0, 20.0, 20.0, 26.67, 20.0]
    assert sample_std(values) == pytest.approx(3.44, abs=0.02)


def test_sample_std_single_value_is_zero():
    assert sample_std([42.0]) == 0.0


def test_sample_std_empty_is_zero():
    assert sample_std([]) == 0.0


# --------------------------------------------------------------------------- #
# aggregate_samples
# --------------------------------------------------------------------------- #
def test_aggregate_baseline_series_mean_and_std():
    runs = [_run_with_passed_count(p) for p in _BASELINE_PASS_COUNTS]
    agg = aggregate_samples(runs)
    assert isinstance(agg, SampleAggregate)
    assert agg.n == 6
    assert agg.mean_pass_rate == pytest.approx(22.22, abs=0.05)
    assert agg.sample_std == pytest.approx(3.44, abs=0.05)


def test_aggregate_incumbent_series_mean_and_std():
    runs = [_run_with_passed_count(p) for p in _INCUMBENT_PASS_COUNTS]
    agg = aggregate_samples(runs)
    assert agg.mean_pass_rate == pytest.approx(60.0, abs=0.05)
    assert agg.sample_std == pytest.approx(5.97, abs=0.05)


def test_aggregate_single_run_has_zero_std():
    agg = aggregate_samples([_run_with_passed_count(9)])
    assert agg.n == 1
    assert agg.mean_pass_rate == pytest.approx(60.0, abs=0.05)
    assert agg.sample_std == 0.0


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_samples([])


def test_per_question_pass_fraction_tracks_flips():
    # Q04 passes in 3 of 6 runs -> a 50% flip; Q01 always passes; Q02 never.
    flags_by_run = [
        [True, False, True, True],
        [True, False, False, False],
        [True, False, True, False],
        [True, False, True, True],
        [True, False, False, True],
        [True, False, False, False],
    ]
    qids = ["Q01", "Q02", "Q03", "Q04"]
    runs = [_run_from_pass_flags(flags, qids=qids) for flags in flags_by_run]
    agg = aggregate_samples(runs)
    assert agg.per_question_pass_fraction["Q01"] == pytest.approx(1.0)
    assert agg.per_question_pass_fraction["Q02"] == pytest.approx(0.0)
    # Q04 passes in runs 1,4,5 -> 3/6, the canonical 50% coin-flip.
    assert agg.per_question_pass_fraction["Q04"] == pytest.approx(0.5, abs=1e-6)
    assert agg.per_question_pass_counts["Q04"] == 3
    assert "Q04" in agg.flipping_question_ids
    assert "Q01" not in agg.flipping_question_ids


# --------------------------------------------------------------------------- #
# minimum_detectable_difference
# --------------------------------------------------------------------------- #
def test_mdd_is_two_times_max_sigma():
    baseline = [_run_with_passed_count(p) for p in _BASELINE_PASS_COUNTS]
    incumbent = [_run_with_passed_count(p) for p in _INCUMBENT_PASS_COUNTS]
    sigma_b = aggregate_samples(baseline).sample_std
    sigma_i = aggregate_samples(incumbent).sample_std
    mdd = minimum_detectable_difference(sigma_b, sigma_i)
    assert mdd == pytest.approx(11.93, abs=0.1)
    # Equals 2 * the larger sigma.
    assert mdd == pytest.approx(2 * max(sigma_b, sigma_i), abs=1e-9)


def test_mdd_ignores_none_and_handles_empty():
    assert minimum_detectable_difference(None, 5.0) == pytest.approx(10.0)
    assert minimum_detectable_difference() == 0.0


# --------------------------------------------------------------------------- #
# lift_threshold
# --------------------------------------------------------------------------- #
def test_lift_threshold_single_run_fixed_baseline_is_two_sigma():
    # n_R=1, baseline treated as a fixed reference (baseline_n=None) -> 2*sigma.
    assert lift_threshold(5.97, candidate_n=1, baseline_n=None) == pytest.approx(11.94, abs=1e-6)


def test_lift_threshold_two_sample_form():
    # n_R=6, n_base=6: 2 * sqrt(sigma^2/6 + sigma^2/6).
    sigma = 5.97
    expected = 2 * math.sqrt(sigma**2 / 6 + sigma**2 / 6)
    assert lift_threshold(sigma, candidate_n=6, baseline_n=6) == pytest.approx(expected, abs=1e-6)


def test_lift_threshold_shrinks_with_more_samples():
    sigma = 5.97
    one = lift_threshold(sigma, candidate_n=1, baseline_n=6)
    many = lift_threshold(sigma, candidate_n=9, baseline_n=6)
    assert many < one


# --------------------------------------------------------------------------- #
# promotion_decision
# --------------------------------------------------------------------------- #
def test_promotion_requires_lift_above_mdd():
    baseline_mean = 22.22
    sigma = 5.97
    mdd = minimum_detectable_difference(sigma)  # 11.94, fixed-baseline single run

    just_above = promotion_decision(
        candidate_full_mean=baseline_mean + mdd + 0.2,
        baseline_full_mean=baseline_mean,
        sigma_full=sigma,
        candidate_n=1,
        baseline_n=None,
    )
    assert just_above.promote is True
    assert just_above.full_significant is True

    just_below = promotion_decision(
        candidate_full_mean=baseline_mean + mdd - 0.2,
        baseline_full_mean=baseline_mean,
        sigma_full=sigma,
        candidate_n=1,
        baseline_n=None,
    )
    assert just_below.promote is False
    assert "full_lift_below_mdd" in just_below.reasons


def test_promotion_blocks_on_holdout_regression():
    baseline_mean = 22.22
    sigma = 5.97
    decision = promotion_decision(
        candidate_full_mean=baseline_mean + 20.0,  # clears full easily
        baseline_full_mean=baseline_mean,
        sigma_full=sigma,
        candidate_n=1,
        baseline_n=None,
        candidate_holdout_mean=66.67,
        baseline_holdout_mean=100.0,  # candidate regressed holdout
        holdout_epsilon=0.1,
    )
    assert decision.promote is False
    assert "holdout_regressed" in decision.reasons


def test_promotion_allows_holdout_within_epsilon():
    decision = promotion_decision(
        candidate_full_mean=50.0,
        baseline_full_mean=22.22,
        sigma_full=5.97,
        candidate_n=1,
        baseline_n=None,
        candidate_holdout_mean=99.95,
        baseline_holdout_mean=100.0,
        holdout_epsilon=0.1,
    )
    assert decision.holdout_non_regressed is True
    assert decision.promote is True


def test_repeated_recipe_predicate_uses_two_sample_threshold():
    # With 6 repeats each, a ~7pt lift clears the (shrunk) two-sample threshold
    # even though it is far below the single-run MDD of ~11.9.
    sigma = 5.97
    decision = promotion_decision(
        candidate_full_mean=29.5,
        baseline_full_mean=22.22,
        sigma_full=sigma,
        candidate_n=6,
        baseline_n=6,
    )
    threshold = lift_threshold(sigma, candidate_n=6, baseline_n=6)
    assert threshold == pytest.approx(6.89, abs=0.05)
    assert decision.full_threshold == pytest.approx(threshold, abs=1e-6)
    assert decision.promote is True


def test_promotion_decision_is_serializable():
    decision = promotion_decision(
        candidate_full_mean=40.0,
        baseline_full_mean=22.22,
        sigma_full=5.97,
    )
    payload = decision.as_dict()
    assert payload["promote"] is True
    assert "full_lift" in payload
    assert "full_threshold" in payload
    assert isinstance(payload["reasons"], list)


# --------------------------------------------------------------------------- #
# build_noise_model
# --------------------------------------------------------------------------- #
def test_build_noise_model_from_aggregates():
    baseline = aggregate_samples([_run_with_passed_count(p) for p in _BASELINE_PASS_COUNTS])
    incumbent = aggregate_samples([_run_with_passed_count(p) for p in _INCUMBENT_PASS_COUNTS])
    model = build_noise_model({"baseline": baseline, "incumbent": incumbent})
    assert model["schema"] == NOISE_MODEL_SCHEMA
    # sigma_hat = max(baseline std ~3.44, incumbent std ~5.97)
    assert model["sigma_hat_full"] == pytest.approx(5.97, abs=0.05)
    assert model["mdd_full"] == pytest.approx(11.93, abs=0.1)
    assert model["subjects"]["incumbent"]["contributes_to_sigma"] is True
    assert model["subjects"]["baseline"]["n"] == 6


def test_build_noise_model_accepts_dicts():
    model = build_noise_model(
        {
            "baseline": {"mean_pass_rate": 22.22, "sample_std": 3.44, "n": 6},
            "incumbent": {"mean_pass_rate": 60.0, "sample_std": 5.97, "n": 6},
        }
    )
    assert model["sigma_hat_full"] == pytest.approx(5.97, abs=1e-6)
    assert model["mdd_full"] == pytest.approx(11.94, abs=1e-6)


def test_build_noise_model_ignores_single_sample_subjects():
    # n=1 subjects have no measurable std and must not drive sigma to 0.
    model = build_noise_model(
        {
            "baseline": {"mean_pass_rate": 22.22, "sample_std": 0.0, "n": 1},
            "incumbent": {"mean_pass_rate": 60.0, "sample_std": 5.97, "n": 6},
        }
    )
    assert model["sigma_hat_full"] == pytest.approx(5.97, abs=1e-6)
    assert model["subjects"]["baseline"]["contributes_to_sigma"] is False


def test_build_noise_model_empty_is_zero_sigma():
    model = build_noise_model({})
    assert model["sigma_hat_full"] == 0.0
    assert model["mdd_full"] == 0.0
