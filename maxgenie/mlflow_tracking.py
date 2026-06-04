"""Opt-in MLflow experiment tracking for MaxGenie's autonomous optimization flow.

Import this module freely — it has no side effects at import time. Call
`get_tracker(enabled=True)` (or `tracker_from_env()`) to obtain a real tracker;
otherwise a no-op stub is returned so the optimization loop never fails due to
tracking errors.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MlflowTrackerProtocol(Protocol):
    def start_optimization_run(
        self,
        *,
        strategy: str,
        label: str,
        baseline_metrics: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None: ...

    def start_candidate_run(
        self,
        *,
        spec_name: str,
        family: str,
        sweep_index: int,
    ) -> None: ...

    def log_candidate_result(self, result: dict[str, Any]) -> None: ...

    def end_candidate_run(self, status: str = "FINISHED") -> None: ...

    def log_benchmark_summary(self, *, label: str, summary: dict[str, Any]) -> None: ...

    def set_optimization_status(self, summary: dict[str, Any]) -> None: ...

    def end_optimization_run(self, status: str = "FINISHED") -> None: ...


class NoOpMlflowTracker:
    """No-op tracker. All methods accept the real signatures and return None."""

    def start_optimization_run(
        self,
        *,
        strategy: str,
        label: str,
        baseline_metrics: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        return None

    def start_candidate_run(
        self,
        *,
        spec_name: str,
        family: str,
        sweep_index: int,
    ) -> None:
        return None

    def log_candidate_result(self, result: dict[str, Any]) -> None:
        return None

    def end_candidate_run(self, status: str = "FINISHED") -> None:
        return None

    def log_benchmark_summary(self, *, label: str, summary: dict[str, Any]) -> None:
        return None

    def set_optimization_status(self, summary: dict[str, Any]) -> None:
        return None

    def end_optimization_run(self, status: str = "FINISHED") -> None:
        return None


class MlflowTracker:
    """Real tracker. Uses mlflow as an injected module (default: the installed package)."""

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        mlflow_module: Any | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._mlflow = mlflow_module
        self._strategy: str = ""
        self._parent_run: Any = None
        self._child_run: Any = None

    def start_optimization_run(
        self,
        *,
        strategy: str,
        label: str,
        baseline_metrics: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        try:
            mlflow = self._mlflow
            if self._tracking_uri is not None:
                mlflow.set_tracking_uri(self._tracking_uri)
            if self._experiment_name is not None:
                mlflow.set_experiment(self._experiment_name)
            self._strategy = strategy
            self._parent_run = mlflow.start_run(run_name=f"{strategy}/{label}")
            if params:
                mlflow.log_params(params)
            if baseline_metrics:
                for k, v in baseline_metrics.items():
                    if isinstance(v, (int, float)) and v is not None:
                        mlflow.log_metric(f"baseline.{k}", float(v))
        except Exception:
            logger.debug("mlflow start_optimization_run failed", exc_info=True)

    def start_candidate_run(
        self,
        *,
        spec_name: str,
        family: str,
        sweep_index: int,
    ) -> None:
        try:
            mlflow = self._mlflow
            run_name = f"{self._strategy}/{spec_name}/sweep-{sweep_index}"
            self._child_run = mlflow.start_run(run_name=run_name, nested=True)
            mlflow.log_param("candidate_name", spec_name)
            mlflow.log_param("family", family)
            mlflow.log_param("sweep_index", sweep_index)
        except Exception:
            logger.debug("mlflow start_candidate_run failed", exc_info=True)

    def log_candidate_result(self, result: dict[str, Any]) -> None:
        try:
            mlflow = self._mlflow
            mlflow.log_param("outcome", result["outcome"])
            reason = result.get("reason")
            if reason:
                mlflow.log_param("reason", str(reason)[:250])
            train_rate = result.get("train_rate")
            if train_rate is not None:
                mlflow.log_metric("train_rate", float(train_rate))
            validation_rate = result.get("validation_rate")
            if validation_rate is not None:
                mlflow.log_metric("validation_rate", float(validation_rate))
            if result.get("suppressed"):
                mlflow.log_param("suppressed", "true")
        except Exception:
            logger.debug("mlflow log_candidate_result failed", exc_info=True)

    def end_candidate_run(self, status: str = "FINISHED") -> None:
        try:
            self._mlflow.end_run(status=status)
            self._child_run = None
        except Exception:
            logger.debug("mlflow end_candidate_run failed", exc_info=True)

    def log_benchmark_summary(self, *, label: str, summary: dict[str, Any]) -> None:
        try:
            mlflow = self._mlflow
            overall = summary.get("overall")
            if isinstance(overall, dict) and "pass_rate" in overall:
                mlflow.log_metric(f"{label}.pass_rate", float(overall["pass_rate"]))
            for split in ("train", "validation", "test"):
                split_data = summary.get(split)
                if isinstance(split_data, dict) and "pass_rate" in split_data:
                    mlflow.log_metric(
                        f"{label}.{split}_pass_rate", float(split_data["pass_rate"])
                    )
        except Exception:
            logger.debug("mlflow log_benchmark_summary failed", exc_info=True)

    def set_optimization_status(self, summary: dict[str, Any]) -> None:
        try:
            if self._parent_run is None:
                logger.debug("mlflow set_optimization_status: no active parent run")
                return
            mlflow = self._mlflow
            for key in ("accepted", "rejected", "skipped"):
                if key in summary:
                    mlflow.log_metric(f"final.{key}", float(summary[key]))
            if "best_train_rate" in summary:
                mlflow.log_metric("final.best_train_rate", float(summary["best_train_rate"]))
            if "best_validation_rate" in summary:
                mlflow.log_metric(
                    "final.best_validation_rate", float(summary["best_validation_rate"])
                )
            try:
                pass_rate = summary["final"]["overall"]["pass_rate"]
                mlflow.log_metric("final.pass_rate", float(pass_rate))
            except (KeyError, TypeError):
                pass
            if "stop_reason" in summary:
                mlflow.log_param("stop_reason", summary["stop_reason"])
        except Exception:
            logger.debug("mlflow set_optimization_status failed", exc_info=True)

    def end_optimization_run(self, status: str = "FINISHED") -> None:
        try:
            if self._parent_run is None:
                return
            self._mlflow.end_run(status=status)
            self._parent_run = None
        except Exception:
            logger.debug("mlflow end_optimization_run failed", exc_info=True)


def get_tracker(
    *,
    enabled: bool,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    mlflow_module: Any | None = None,
) -> MlflowTrackerProtocol:
    """Return MlflowTracker when enabled and mlflow importable, else NoOpMlflowTracker."""
    if not enabled:
        return NoOpMlflowTracker()

    module = mlflow_module
    if module is None:
        try:
            import mlflow  # noqa: PLC0415

            module = mlflow
        except ImportError:
            logger.debug("mlflow not installed; falling back to NoOpMlflowTracker")
            return NoOpMlflowTracker()

    return MlflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        mlflow_module=module,
    )


def tracker_from_env() -> MlflowTrackerProtocol:
    """Build a tracker from environment variables.

    Reads:
    - MAXGENIE_MLFLOW_ENABLED  (truthy: "1", "true", "yes", case-insensitive)
    - MAXGENIE_MLFLOW_EXPERIMENT
    - MAXGENIE_MLFLOW_TRACKING_URI
    """
    raw_enabled = os.getenv("MAXGENIE_MLFLOW_ENABLED", "").strip().lower()
    enabled = raw_enabled in ("1", "true", "yes")
    experiment_name = os.getenv("MAXGENIE_MLFLOW_EXPERIMENT") or None
    tracking_uri = os.getenv("MAXGENIE_MLFLOW_TRACKING_URI") or None
    return get_tracker(
        enabled=enabled,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )
