"""Tests for noise-aware terminal result selection (MDD-gated promotion).

These exercise the pure ``_terminal_result_selection`` gate directly. The key
guarantee is that without a noise model the original epsilon behaviour is
unchanged, and with a noise model the full-score gate uses the minimum
detectable difference instead of a fixed epsilon.
"""

from __future__ import annotations

from maxgenie.workspace_flow import _terminal_result_selection


def _optimization_summary(*, baseline_full: float, baseline_holdout: float) -> dict:
    return {
        "baseline": {"full_pass_rate": baseline_full, "test_rate": baseline_holdout},
        "best_checkpoint_path": "/candidate/checkpoint",
        "baseline_checkpoint_path": "/baseline/checkpoint",
    }


def _final_summary(*, full: float, holdout: float) -> dict:
    return {"overall": {"pass_rate": full}, "test": {"pass_rate": holdout}}


_COMPARABLE = {"comparable": True, "status": "comparable", "errors": []}


# --------------------------------------------------------------------------- #
# Epsilon fallback (no noise model) -- unchanged legacy behaviour.
# --------------------------------------------------------------------------- #
def test_epsilon_promotes_clear_improvement():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=40.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
    )
    assert result["status"] == "promotable"
    assert result["noise_aware"] is False
    assert result["promotion_decision"] is None
    assert result["selected_checkpoint_path"] == "/candidate/checkpoint"


def test_epsilon_blocks_within_epsilon_improvement():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=22.25, holdout=33.33),
        protocol_preflight=_COMPARABLE,
    )
    assert result["status"] == "not_promotable"
    assert "final_full_did_not_improve_over_baseline" in result["reasons"]
    assert result["selected_checkpoint_path"] == "/baseline/checkpoint"


def test_epsilon_blocks_holdout_regression():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=100.0),
        final_summary=_final_summary(full=40.0, holdout=66.67),
        protocol_preflight=_COMPARABLE,
    )
    assert result["status"] == "not_promotable"
    assert "final_holdout_regressed_below_baseline" in result["reasons"]


# --------------------------------------------------------------------------- #
# Noise-aware branch.
# --------------------------------------------------------------------------- #
def test_noise_aware_blocks_lift_within_mdd():
    # +9 pts looks like an improvement but is below the ~11.9 single-run MDD.
    noise = {
        "sigma_full": 5.97,
        "baseline_full_mean": 22.22,
        "final_full_mean": 31.0,
        "baseline_holdout_mean": 33.33,
        "final_holdout_mean": 33.33,
        "baseline_n": None,
        "candidate_n": 1,
    }
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=31.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        noise_model=noise,
    )
    assert result["status"] == "not_promotable"
    assert result["noise_aware"] is True
    assert "final_full_within_noise_of_baseline" in result["reasons"]
    assert result["mdd_full"] is not None
    assert result["promotion_decision"]["promote"] is False
    # Reported full rate is the multi-sample mean, not the single run.
    assert result["final_full_rate"] == 31.0


def test_noise_aware_promotes_lift_above_mdd():
    noise = {
        "sigma_full": 5.97,
        "baseline_full_mean": 22.22,
        "final_full_mean": 40.0,  # +17.8 clears the ~11.9 MDD
        "baseline_holdout_mean": 33.33,
        "final_holdout_mean": 33.33,
        "baseline_n": None,
        "candidate_n": 1,
    }
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=40.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        noise_model=noise,
    )
    assert result["status"] == "promotable"
    assert result["noise_aware"] is True
    assert result["promotion_decision"]["promote"] is True


def test_noise_aware_repeated_recipe_shrinks_bar():
    # A +8 pt lift fails the single-run MDD but clears the two-sample bar when
    # both baseline and candidate are sampled 6x.
    noise = {
        "sigma_full": 5.97,
        "baseline_full_mean": 22.22,
        "final_full_mean": 30.2,
        "baseline_holdout_mean": 33.33,
        "final_holdout_mean": 33.33,
        "baseline_n": 6,
        "candidate_n": 6,
    }
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=30.2, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        noise_model=noise,
    )
    assert result["status"] == "promotable"
    assert result["promotion_decision"]["full_threshold"] < result["mdd_full"]


def test_noise_aware_still_blocks_holdout_regression():
    noise = {
        "sigma_full": 5.97,
        "baseline_full_mean": 22.22,
        "final_full_mean": 60.0,  # clears full easily
        "baseline_holdout_mean": 100.0,
        "final_holdout_mean": 66.67,  # but regresses holdout
        "candidate_n": 1,
    }
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=100.0),
        final_summary=_final_summary(full=60.0, holdout=66.67),
        protocol_preflight=_COMPARABLE,
        noise_model=noise,
    )
    assert result["status"] == "not_promotable"
    assert "final_holdout_regressed_below_baseline" in result["reasons"]


def test_zero_sigma_falls_back_to_epsilon():
    # sigma_full == 0 means "no usable noise model" -> epsilon path.
    noise = {"sigma_full": 0.0, "baseline_full_mean": 22.22, "final_full_mean": 23.0}
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=23.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        noise_model=noise,
    )
    assert result["noise_aware"] is False
    # +0.78 over baseline clears the 0.1 epsilon, so the legacy gate promotes.
    assert result["status"] == "promotable"


def test_noise_aware_blocks_when_protocol_non_comparable():
    noise = {"sigma_full": 5.97, "baseline_full_mean": 22.22, "final_full_mean": 60.0}
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=60.0, holdout=33.33),
        protocol_preflight={"comparable": False, "status": "non_comparable", "errors": []},
        noise_model=noise,
    )
    assert result["status"] == "not_promotable"
    assert "protocol_non_comparable" in result["reasons"]


# --------------------------------------------------------------------------- #
# Greedy best-ever branch (Round C): no MDD; promote any strict improvement.
# --------------------------------------------------------------------------- #
def test_greedy_promotes_any_margin_ignoring_mdd():
    # +8.78 pts is below the ~11.9 single-run MDD, so the noise-aware gate would
    # block it; greedy best-ever ignores the noise model and promotes the strict
    # improvement, reporting the single-run rate (not the multi-sample mean).
    noise = {
        "sigma_full": 5.97,
        "baseline_full_mean": 22.22,
        "final_full_mean": 31.0,
        "baseline_holdout_mean": 33.33,
        "final_holdout_mean": 33.33,
        "candidate_n": 1,
    }
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=31.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        noise_model=noise,
        greedy_best_ever=True,
    )
    assert result["status"] == "promotable"
    assert result["promotion_mode"] == "greedy_best_ever"
    assert result["noise_aware"] is False
    assert result["mdd_full"] is None
    assert result["promotion_decision"] is None
    assert result["final_full_rate"] == 31.0
    assert result["selected_checkpoint_path"] == "/candidate/checkpoint"


def test_greedy_blocks_when_no_strict_improvement():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=40.0, baseline_holdout=33.33),
        final_summary=_final_summary(full=40.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        greedy_best_ever=True,
    )
    assert result["status"] == "not_promotable"
    assert result["promotion_mode"] == "greedy_best_ever"
    assert "final_full_did_not_improve_over_baseline" in result["reasons"]
    assert result["selected_checkpoint_path"] == "/baseline/checkpoint"


def test_greedy_still_blocks_holdout_regression():
    # Honest win: even a large visible gain must not regress the hidden holdout.
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=100.0),
        final_summary=_final_summary(full=60.0, holdout=66.67),
        protocol_preflight=_COMPARABLE,
        greedy_best_ever=True,
    )
    assert result["status"] == "not_promotable"
    assert "final_holdout_regressed_below_baseline" in result["reasons"]


def test_greedy_blocks_when_protocol_non_comparable():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=60.0, holdout=33.33),
        protocol_preflight={"comparable": False, "status": "non_comparable", "errors": []},
        greedy_best_ever=True,
    )
    assert result["status"] == "not_promotable"
    assert "protocol_non_comparable" in result["reasons"]
