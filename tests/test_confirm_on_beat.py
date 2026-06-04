"""Tests for the optional confirm-on-beat anti-noise re-run.

``_confirm_candidate_beat`` re-runs the visible (train + validation) benchmark and
confirms an accepted beat only when no re-run regresses the visible reference. It is
exercised here with a duck-typed stub so the pure re-run/decision logic is covered
without a live workspace.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxgenie.workspace_flow import OptimizationWorkspace


class _FakeTrainRun:
    def __init__(self, pass_rate: float) -> None:
        self.pass_rate = pass_rate


def _make_confirm_stub(train_rates, validation_rates, confirm_on_beat):
    queues = {"train": list(train_rates), "validation": list(validation_rates)}
    stub = SimpleNamespace()
    stub._confirm_on_beat = confirm_on_beat

    def _run_train(**_kwargs):
        return _FakeTrainRun(queues["train"].pop(0)), {}

    def _run_validation(**_kwargs):
        return {"pass_rate": queues["validation"].pop(0)}

    stub._run_train_benchmark_internal = _run_train
    stub.run_validation_benchmark = _run_validation
    return stub


def test_confirm_on_beat_confirms_when_no_regression():
    stub = _make_confirm_stub([66.7, 66.7], [33.3, 33.3], confirm_on_beat=2)
    out = OptimizationWorkspace._confirm_candidate_beat(
        stub,
        pass_label="sweep.candidate",
        reference_train_rate=55.6,
        reference_validation_rate=33.3,
    )
    assert out["confirmed"] is True
    assert out["details"]["confirm_on_beat_runs"] == 2
    assert len(out["details"]["confirm_samples"]) == 2


def test_confirm_on_beat_rejects_on_train_regression():
    # Second re-run drops train below the reference -> the beat was noise.
    stub = _make_confirm_stub([66.7, 40.0], [33.3, 33.3], confirm_on_beat=2)
    out = OptimizationWorkspace._confirm_candidate_beat(
        stub,
        pass_label="sweep.candidate",
        reference_train_rate=55.6,
        reference_validation_rate=33.3,
    )
    assert out["confirmed"] is False
    assert "regressed visible reference" in out["reason"]
    # Stops early on the failing re-run, so only two samples were taken.
    assert len(out["details"]["confirm_samples"]) == 2


def test_confirm_on_beat_rejects_on_validation_regression():
    stub = _make_confirm_stub([66.7], [20.0], confirm_on_beat=1)
    out = OptimizationWorkspace._confirm_candidate_beat(
        stub,
        pass_label="sweep.candidate",
        reference_train_rate=55.6,
        reference_validation_rate=33.3,
    )
    assert out["confirmed"] is False
    assert "regressed visible reference" in out["reason"]


def test_confirm_on_beat_zero_runs_is_vacuously_confirmed():
    stub = _make_confirm_stub([], [], confirm_on_beat=0)
    out = OptimizationWorkspace._confirm_candidate_beat(
        stub,
        pass_label="sweep.candidate",
        reference_train_rate=55.6,
        reference_validation_rate=33.3,
    )
    assert out["confirmed"] is True
    assert out["details"]["confirm_on_beat_runs"] == 0
