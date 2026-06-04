"""Tests for the cross-space robustness (transfer) gate.

The gate is inactive by default (no robustness spaces -> primary verdict passes
through unchanged), which keeps the single-space "default space only" posture
byte-for-byte. When robustness spaces are supplied, a primary win is kept only if
it re-clears promotion on every robustness space under that space's own seed.
"""

from __future__ import annotations

from maxgenie.multi_sample import (
    TRANSFER_GATE_SCHEMA,
    TransferGateDecision,
    evaluate_transfer_gate,
    promotion_decision,
)
from maxgenie.workspace_flow import _terminal_result_selection


def _decision(*, candidate: float, baseline: float = 22.22, sigma: float = 5.97) -> "object":
    return promotion_decision(
        candidate_full_mean=candidate,
        baseline_full_mean=baseline,
        sigma_full=sigma,
        candidate_n=1,
        baseline_n=None,
    )


_PROMOTING = _decision(candidate=40.0)  # +17.8 clears single-run MDD (~11.9)
_BLOCKING = _decision(candidate=28.0)  # +5.8 below MDD -> not promotable


# --------------------------------------------------------------------------- #
# evaluate_transfer_gate -- pure
# --------------------------------------------------------------------------- #
def test_inactive_passes_primary_through_when_promoting():
    gate = evaluate_transfer_gate(primary=_PROMOTING, transfers=None)
    assert isinstance(gate, TransferGateDecision)
    assert gate.active is False
    assert gate.promote is True
    assert gate.transfer_space_count == 0
    assert "transfer_gate_inactive_no_robustness_spaces" in gate.reasons


def test_inactive_passes_primary_through_when_blocking():
    gate = evaluate_transfer_gate(primary=_BLOCKING, transfers={})
    assert gate.active is False
    assert gate.promote is False
    assert gate.primary_promote is False


def test_active_all_pass_promotes():
    gate = evaluate_transfer_gate(
        primary=_PROMOTING,
        transfers={"spaceB": _decision(candidate=41.0), "spaceC": _decision(candidate=39.0)},
    )
    assert gate.active is True
    assert gate.promote is True
    assert gate.transfer_space_count == 2
    assert gate.passing_space_count == 2
    assert gate.failing_space_ids == ()
    assert gate.reasons == ()


def test_active_one_transfer_fails_blocks():
    gate = evaluate_transfer_gate(
        primary=_PROMOTING,
        transfers={"spaceB": _decision(candidate=41.0), "spaceC": _BLOCKING},
    )
    assert gate.promote is False
    assert gate.failing_space_ids == ("spaceC",)
    assert "transfer_failed:spaceC" in gate.reasons


def test_active_primary_fails_blocks_even_if_transfers_pass():
    gate = evaluate_transfer_gate(
        primary=_BLOCKING,
        transfers={"spaceB": _decision(candidate=41.0)},
    )
    assert gate.promote is False
    assert "primary_not_promotable" in gate.reasons


def test_accepts_sequence_of_tuples():
    gate = evaluate_transfer_gate(
        primary=_PROMOTING,
        transfers=[("spaceB", _decision(candidate=41.0))],
    )
    assert gate.active is True
    assert gate.promote is True


def test_as_dict_is_serializable_with_schema():
    gate = evaluate_transfer_gate(
        primary=_PROMOTING,
        transfers={"spaceB": _BLOCKING},
    )
    payload = gate.as_dict()
    assert payload["schema"] == TRANSFER_GATE_SCHEMA
    assert payload["promote"] is False
    assert payload["per_space"][0]["space_id"] == "spaceB"
    assert isinstance(payload["reasons"], list)


# --------------------------------------------------------------------------- #
# _terminal_result_selection composition
# --------------------------------------------------------------------------- #
def _optimization_summary(*, baseline_full: float, baseline_holdout: float) -> dict:
    return {
        "baseline": {"full_pass_rate": baseline_full, "test_rate": baseline_holdout},
        "best_checkpoint_path": "/candidate/checkpoint",
        "baseline_checkpoint_path": "/baseline/checkpoint",
    }


def _final_summary(*, full: float, holdout: float) -> dict:
    return {"overall": {"pass_rate": full}, "test": {"pass_rate": holdout}}


_COMPARABLE = {"comparable": True, "status": "comparable", "errors": []}


def test_terminal_selection_default_gate_inactive_and_unchanged():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=40.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
    )
    assert result["status"] == "promotable"
    assert result["transfer_gate"]["active"] is False
    assert result["selected_checkpoint_path"] == "/candidate/checkpoint"


def test_terminal_selection_active_transfer_failure_downgrades():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=40.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        transfer_decisions={"spaceB": _BLOCKING},
    )
    assert result["status"] == "not_promotable"
    assert result["transfer_gate"]["active"] is True
    assert "transfer_failed:spaceB" in result["reasons"]
    # On a blocked transfer the baseline checkpoint is retained.
    assert result["selected_checkpoint_path"] == "/baseline/checkpoint"


def test_terminal_selection_active_transfer_success_keeps_promotion():
    result = _terminal_result_selection(
        optimization_summary=_optimization_summary(baseline_full=22.22, baseline_holdout=33.33),
        final_summary=_final_summary(full=40.0, holdout=33.33),
        protocol_preflight=_COMPARABLE,
        transfer_decisions={"spaceB": _decision(candidate=41.0)},
    )
    assert result["status"] == "promotable"
    assert result["transfer_gate"]["active"] is True
    assert result["transfer_gate"]["passing_space_count"] == 1
    assert result["selected_checkpoint_path"] == "/candidate/checkpoint"
