"""Tests for maxgenie.mlflow_tracking contract."""

from __future__ import annotations

import importlib
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRun:
    def __init__(self, run_name: str | None = None, nested: bool = False) -> None:
        self.run_name = run_name
        self.nested = nested


class _FakeMlflow:
    def __init__(self, *, raise_on: set[str] | None = None) -> None:
        self._raise_on = raise_on or set()
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.start_run_calls: list[dict[str, Any]] = []
        self.params: list[tuple[str, object]] = []
        self.metrics: list[tuple[str, float]] = []
        self.log_params_calls: list[dict[str, Any]] = []
        self.end_run_calls: list[dict[str, Any]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def set_experiment(self, name: str) -> None:
        self.experiments.append(name)

    def start_run(self, *, run_name: str | None = None, nested: bool = False) -> _FakeRun:
        call: dict[str, Any] = {"run_name": run_name, "nested": nested}
        self.start_run_calls.append(call)
        return _FakeRun(run_name=run_name, nested=nested)

    def log_params(self, params: dict[str, Any]) -> None:
        self.log_params_calls.append(dict(params))

    def log_param(self, key: str, value: object) -> None:
        if "log_param" in self._raise_on:
            raise RuntimeError("boom")
        self.params.append((key, value))

    def log_metric(self, key: str, value: float) -> None:
        if "log_metric" in self._raise_on:
            raise RuntimeError("boom")
        self.metrics.append((key, float(value)))

    def end_run(self, status: str = "FINISHED") -> None:
        self.end_run_calls.append({"status": status})


# ---------------------------------------------------------------------------
# Test 1: get_tracker(enabled=False) returns NoOpMlflowTracker
# ---------------------------------------------------------------------------


def test_get_tracker_disabled_returns_noop() -> None:
    from maxgenie.mlflow_tracking import NoOpMlflowTracker, get_tracker

    tracker = get_tracker(enabled=False)
    assert isinstance(tracker, NoOpMlflowTracker)

    # All methods accept full keyword signatures and return None without raising
    assert tracker.start_optimization_run(
        strategy="s", label="l", baseline_metrics={"x": 1.0}, params={"p": 2}
    ) is None
    assert tracker.start_candidate_run(spec_name="n", family="f", sweep_index=0) is None
    assert tracker.log_candidate_result({"outcome": "accepted", "train_rate": 90.0}) is None
    assert tracker.end_candidate_run(status="FINISHED") is None
    assert tracker.log_benchmark_summary(label="lbl", summary={"overall": {"pass_rate": 80.0}}) is None
    assert tracker.set_optimization_status({"accepted": 1, "rejected": 0}) is None
    assert tracker.end_optimization_run(status="FINISHED") is None


# ---------------------------------------------------------------------------
# Test 2: NoOp never imports mlflow
# ---------------------------------------------------------------------------


class _BoomModule:
    """Sentinel that raises RuntimeError on any attribute access."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"mlflow touched! (attribute: {name})")


def test_noop_never_imports_mlflow(monkeypatch: Any) -> None:
    boom = _BoomModule()
    monkeypatch.setitem(sys.modules, "mlflow", boom)  # type: ignore[arg-type]

    # Reload the module under test to prove no side-effect import
    import maxgenie.mlflow_tracking as tracking_mod

    importlib.reload(tracking_mod)

    tracker = tracking_mod.get_tracker(enabled=False)
    assert isinstance(tracker, tracking_mod.NoOpMlflowTracker)

    # Exercise every method — none should touch the boom module
    tracker.start_optimization_run(strategy="s", label="l")
    tracker.start_candidate_run(spec_name="n", family="f", sweep_index=0)
    tracker.log_candidate_result({"outcome": "accepted"})
    tracker.end_candidate_run()
    tracker.log_benchmark_summary(label="lbl", summary={})
    tracker.set_optimization_status({})
    tracker.end_optimization_run()


# ---------------------------------------------------------------------------
# Test 3: Missing mlflow falls back to NoOp
# ---------------------------------------------------------------------------


def test_get_tracker_missing_mlflow_falls_back_to_noop(monkeypatch: Any) -> None:
    # Setting sys.modules["mlflow"] = None makes `import mlflow` raise ImportError
    monkeypatch.setitem(sys.modules, "mlflow", None)  # type: ignore[arg-type]

    from maxgenie.mlflow_tracking import NoOpMlflowTracker, get_tracker

    tracker = get_tracker(enabled=True, mlflow_module=None)
    assert isinstance(tracker, NoOpMlflowTracker)


# ---------------------------------------------------------------------------
# Test 4: start_optimization_run run name and logging
# ---------------------------------------------------------------------------


def test_start_optimization_run_uses_expected_run_name_and_logs() -> None:
    from maxgenie.mlflow_tracking import get_tracker

    fake = _FakeMlflow()
    tracker = get_tracker(
        enabled=True,
        experiment_name="exp-A",
        tracking_uri="sqlite:///mlruns.db",
        mlflow_module=fake,
    )
    tracker.start_optimization_run(
        strategy="centaur",
        label="optimize",
        baseline_metrics={"pass_rate": 50.0, "not_numeric": "x"},
        params={"max_iterations": 10},
    )

    assert fake.tracking_uris == ["sqlite:///mlruns.db"]
    assert fake.experiments == ["exp-A"]

    assert len(fake.start_run_calls) == 1
    call = fake.start_run_calls[0]
    assert call["run_name"] == "centaur/optimize"
    assert call.get("nested") is False or call.get("nested") is None or call.get("nested") == False  # noqa: E712

    assert fake.log_params_calls == [{"max_iterations": 10}]

    metric_keys = [k for k, _ in fake.metrics]
    assert "baseline.pass_rate" in metric_keys
    assert any(v == 50.0 for k, v in fake.metrics if k == "baseline.pass_rate")
    # non-numeric baseline entries must not be logged
    assert "baseline.not_numeric" not in metric_keys


# ---------------------------------------------------------------------------
# Test 5: start_candidate_run nested=True and param logging
# ---------------------------------------------------------------------------


def test_start_candidate_run_uses_nested_and_logs_params() -> None:
    from maxgenie.mlflow_tracking import get_tracker

    fake = _FakeMlflow()
    tracker = get_tracker(enabled=True, mlflow_module=fake)

    tracker.start_optimization_run(strategy="centaur", label="opt")
    tracker.start_candidate_run(spec_name="entity_matching", family="default", sweep_index=2)

    # The last start_run call must be the child
    last_call = fake.start_run_calls[-1]
    assert last_call["run_name"] == "centaur/entity_matching/sweep-2"
    assert last_call["nested"] is True

    assert ("candidate_name", "entity_matching") in fake.params
    assert ("family", "default") in fake.params
    assert ("sweep_index", 2) in fake.params


# ---------------------------------------------------------------------------
# Test 6: log_candidate_result forwards metrics and params correctly
# ---------------------------------------------------------------------------


def test_log_candidate_result_forwards_metrics_and_params() -> None:
    from maxgenie.mlflow_tracking import get_tracker

    fake = _FakeMlflow()
    tracker = get_tracker(enabled=True, mlflow_module=fake)

    # First call: accepted, no reason, not suppressed
    tracker.log_candidate_result(
        {
            "outcome": "accepted",
            "reason": None,
            "train_rate": 90.0,
            "validation_rate": 85.0,
            "suppressed": False,
            "candidate_name": "x",
        }
    )

    param_keys = [k for k, _ in fake.params]
    assert "outcome" in param_keys
    assert ("outcome", "accepted") in fake.params
    # reason is None — must NOT be logged
    assert "reason" not in param_keys
    # suppressed is False — must NOT be logged
    assert "suppressed" not in param_keys

    metric_keys_1 = [k for k, _ in fake.metrics]
    assert "train_rate" in metric_keys_1
    assert "validation_rate" in metric_keys_1
    assert any(v == 90.0 for k, v in fake.metrics if k == "train_rate")
    assert any(v == 85.0 for k, v in fake.metrics if k == "validation_rate")

    metrics_before = len(fake.metrics)
    params_before = len(fake.params)

    # Second call: rejected, with reason, suppressed=True, train_rate=None
    tracker.log_candidate_result(
        {
            "outcome": "rejected",
            "reason": "train regressed",
            "suppressed": True,
            "train_rate": None,
        }
    )

    assert ("reason", "train regressed") in fake.params
    assert ("suppressed", "true") in fake.params
    # train_rate=None must not log an additional metric
    train_rate_calls = [v for k, v in fake.metrics if k == "train_rate"]
    assert len(train_rate_calls) == 1  # only the one from the first call


# ---------------------------------------------------------------------------
# Test 7: set_optimization_status and end_optimization_run
# ---------------------------------------------------------------------------


def test_set_optimization_status_and_end_run() -> None:
    from maxgenie.mlflow_tracking import get_tracker

    fake = _FakeMlflow()
    tracker = get_tracker(enabled=True, mlflow_module=fake)

    tracker.start_optimization_run(strategy="centaur", label="run1")
    tracker.set_optimization_status(
        {
            "accepted": 3,
            "rejected": 1,
            "skipped": 0,
            "best_train_rate": 100.0,
            "best_validation_rate": 83.3,
            "stop_reason": "validation_plateau",
            "final": {"overall": {"pass_rate": 93.33}},
        }
    )

    metric_dict = {k: v for k, v in fake.metrics}
    assert metric_dict.get("final.accepted") == 3.0
    assert metric_dict.get("final.rejected") == 1.0
    assert metric_dict.get("final.skipped") == 0.0
    assert metric_dict.get("final.best_train_rate") == 100.0
    assert abs(metric_dict.get("final.best_validation_rate", -1) - 83.3) < 0.001
    assert abs(metric_dict.get("final.pass_rate", -1) - 93.33) < 0.001

    assert ("stop_reason", "validation_plateau") in fake.params

    tracker.end_optimization_run()
    assert len(fake.end_run_calls) == 1
    assert fake.end_run_calls[0] == {"status": "FINISHED"}


# ---------------------------------------------------------------------------
# Test 8: Tracker errors are swallowed
# ---------------------------------------------------------------------------


def test_tracker_errors_are_swallowed() -> None:
    from maxgenie.mlflow_tracking import get_tracker

    fake = _FakeMlflow(raise_on={"log_metric"})
    tracker = get_tracker(enabled=True, mlflow_module=fake)

    # Must not raise even though fake raises RuntimeError on log_metric
    tracker.log_candidate_result({"outcome": "accepted", "train_rate": 90.0})


# ---------------------------------------------------------------------------
# Test 9: tracker_from_env honors env vars
# ---------------------------------------------------------------------------


def test_tracker_from_env_honors_env(monkeypatch: Any) -> None:
    from maxgenie import mlflow_tracking
    from maxgenie.mlflow_tracking import MlflowTracker, NoOpMlflowTracker

    monkeypatch.setitem(sys.modules, "mlflow", _FakeMlflow())

    # Truthy variants
    for truthy in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("MAXGENIE_MLFLOW_ENABLED", truthy)
        tracker = mlflow_tracking.tracker_from_env()
        assert isinstance(tracker, MlflowTracker), f"Expected MlflowTracker for ENABLED={truthy!r}"

    # Falsy variants — need a real mlflow or the get_tracker code path to not matter
    # tracker_from_env with enabled=False always returns NoOp regardless of mlflow presence
    for falsy in ("", "0", "false", "no"):
        monkeypatch.setenv("MAXGENIE_MLFLOW_ENABLED", falsy)
        tracker = mlflow_tracking.tracker_from_env()
        assert isinstance(tracker, NoOpMlflowTracker), f"Expected NoOpMlflowTracker for ENABLED={falsy!r}"

    # Without the env var at all
    monkeypatch.delenv("MAXGENIE_MLFLOW_ENABLED", raising=False)
    tracker = mlflow_tracking.tracker_from_env()
    assert isinstance(tracker, NoOpMlflowTracker)
