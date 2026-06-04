"""Databricks job launcher for unattended optimizer runs."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as workspace_service

from maxgenie.advisory import (
    ADVISORY_BEST_PRACTICES_CLASSIFICATION,
    NO_BENCHMARK_WARNING,
    NOT_BENCHMARK_VERIFIED_LABEL,
    NOT_BENCHMARK_VERIFIED_STATUS,
    is_advisory_best_practices,
)


ComputeMode = Literal["serverless", "existing_cluster", "new_cluster"]
SkillScope = Literal["user", "workspace"]

DEFAULT_JOB_DEPENDENCIES: tuple[str, ...] = (
    "databricks-sdk>=0.44.0",
    "httpx>=0.27.0",
    "pydantic>=2.6.0",
    "PyYAML>=6.0",
    "typer>=0.12.0",
    "python-dotenv>=1.0.0",
)
_TERMINAL_LIFECYCLE_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
_ACTIVE_LIFECYCLE_STATES = frozenset(
    {"PENDING", "RUNNING", "TERMINATING", "BLOCKED", "WAITING_FOR_RETRY"}
)


@dataclass(frozen=True)
class WorkspaceJobSettings:
    """Settings for submitting a one-time workspace optimizer run."""

    space_url: str
    runner_path: str
    workspace_root: str
    strategy: str
    run_name: str = "Space Optimizer Long Run"
    use_latest_observed_benchmark: bool = True
    fresh_lineage: bool = False
    bootstrap_only: bool = False
    max_iterations: int | None = None
    plateau_rounds: int = 3
    candidate_budget: int | None = None
    delay_seconds: float = 12.0
    match_threshold: float = 0.85
    warehouse_id: str | None = None
    clone_parent_path: str | None = None
    serving_endpoint: str | None = None
    serving_model: str | None = None
    serving_temperature: float | None = None
    serving_max_tokens: int = 16000
    serving_reasoning_effort: str | None = "low"
    serving_response_format: str = "json_schema"
    serving_api_format: str = "auto"
    serving_candidate_mode: str | None = None
    observed_benchmark_seed: str | None = None
    fixed_benchmark_artifacts_path: str | None = None
    k_samples: int = 1
    noise_sigma_full: float | None = None
    noise_baseline_n: int | None = None
    adaptive_recipes: bool = False
    greedy_best_ever: bool = True
    confirm_on_beat: int = 0
    result_content_match: bool = False
    audit_checkpoint_path: str | None = None
    audit_patch_path: str | None = None
    timeout_seconds: int = 14400
    compute_mode: ComputeMode = "serverless"
    existing_cluster_id: str | None = None
    new_cluster: dict[str, Any] | None = None
    serverless_environment_key: str = "maxgenie"
    serverless_environment_version: str = "2"
    dependencies: tuple[str, ...] = field(default_factory=lambda: DEFAULT_JOB_DEPENDENCIES)
    install_dependencies: bool = True
    idempotency_token: str | None = None

    def __post_init__(self) -> None:
        if self.compute_mode == "existing_cluster" and not self.existing_cluster_id:
            raise ValueError("existing_cluster compute mode requires existing_cluster_id.")
        if self.compute_mode == "new_cluster" and not self.new_cluster:
            raise ValueError("new_cluster compute mode requires new_cluster JSON.")
        if self.compute_mode not in {"serverless", "existing_cluster", "new_cluster"}:
            raise ValueError("compute_mode must be serverless, existing_cluster, or new_cluster.")
        if (
            self.strategy in {"serving", "serving_autonomous"}
            and not self.serving_endpoint
            and not self.audit_checkpoint_path
            and not self.audit_patch_path
        ):
            raise ValueError("serving strategy requires a serving endpoint.")


def _current_user_name(wc: WorkspaceClient) -> str:
    me = wc.current_user.me()
    user_name = getattr(me, "user_name", None)
    if user_name:
        return str(user_name)
    raise RuntimeError("Could not resolve current Databricks user name.")


def default_runner_path(
    wc: WorkspaceClient,
    *,
    skill_scope: SkillScope = "user",
) -> str:
    """Return the default synced skill runner path for a workspace."""
    if skill_scope == "workspace":
        return "/Workspace/.assistant/skills/maxgenie/scripts/runner.py"
    user_name = _current_user_name(wc)
    return f"/Workspace/Users/{user_name}/.assistant/skills/maxgenie/scripts/runner.py"


def _parameter_list(settings: WorkspaceJobSettings) -> list[str]:
    params = [
        "--mode",
        "full",
        "--space-url",
        settings.space_url,
        "--workspace-root",
        settings.workspace_root,
        "--strategy",
        settings.strategy,
        "--plateau-rounds",
        str(settings.plateau_rounds),
        "--delay-seconds",
        str(settings.delay_seconds),
        "--match-threshold",
        str(settings.match_threshold),
        "--serving-max-tokens",
        str(settings.serving_max_tokens),
        "--serving-response-format",
        settings.serving_response_format,
        "--serving-api-format",
        settings.serving_api_format,
    ]
    if settings.max_iterations is not None:
        params.extend(["--max-iterations", str(settings.max_iterations)])
    if settings.candidate_budget is not None:
        params.extend(["--candidate-budget", str(settings.candidate_budget)])
    if settings.use_latest_observed_benchmark:
        params.append("--use-latest-observed-benchmark")
    if settings.fresh_lineage:
        params.append("--fresh-lineage")
    if settings.bootstrap_only:
        params.append("--bootstrap-only")
    if settings.warehouse_id:
        params.extend(["--warehouse-id", settings.warehouse_id])
    if settings.clone_parent_path:
        params.extend(["--clone-parent-path", settings.clone_parent_path])
    if settings.serving_endpoint:
        params.extend(["--serving-endpoint", settings.serving_endpoint])
    if settings.serving_model:
        params.extend(["--serving-model", settings.serving_model])
    if settings.serving_temperature is not None:
        params.extend(["--serving-temperature", str(settings.serving_temperature)])
    if settings.serving_reasoning_effort:
        params.extend(["--serving-reasoning-effort", settings.serving_reasoning_effort])
    if settings.serving_candidate_mode:
        params.extend(["--serving-candidate-mode", settings.serving_candidate_mode])
    if settings.observed_benchmark_seed:
        params.extend(["--observed-benchmark-seed", settings.observed_benchmark_seed])
    if settings.fixed_benchmark_artifacts_path:
        params.extend(["--fixed-benchmark-artifacts-path", settings.fixed_benchmark_artifacts_path])
    # Noise-aware / adaptive flags. Emitted only when non-default so the legacy job
    # parameter list stays byte-for-byte identical when none are set.
    if settings.k_samples and settings.k_samples != 1:
        params.extend(["--k-samples", str(settings.k_samples)])
    if settings.noise_sigma_full is not None:
        params.extend(["--noise-sigma-full", str(settings.noise_sigma_full)])
    if settings.noise_baseline_n is not None:
        params.extend(["--noise-baseline-n", str(settings.noise_baseline_n)])
    if settings.adaptive_recipes:
        params.append("--adaptive-recipes")
    if settings.greedy_best_ever:
        params.append("--greedy-best-ever")
    else:
        params.append("--no-greedy-best-ever")
    if settings.confirm_on_beat and settings.confirm_on_beat > 0:
        params.extend(["--confirm-on-beat", str(settings.confirm_on_beat)])
    if settings.result_content_match:
        params.append("--content-match")
    if settings.audit_checkpoint_path:
        params.extend(["--audit-checkpoint-path", settings.audit_checkpoint_path])
    if settings.audit_patch_path:
        params.extend(["--audit-patch-path", settings.audit_patch_path])
    return params


def build_runs_submit_payload(settings: WorkspaceJobSettings) -> dict[str, Any]:
    """Build a Jobs runs/submit request payload."""
    task: dict[str, Any] = {
        "task_key": "optimize",
        "description": "Run one unattended Genie space optimization loop.",
        "spark_python_task": {
            "python_file": settings.runner_path,
            "source": "WORKSPACE",
            "parameters": _parameter_list(settings),
        },
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": 0,
    }

    payload: dict[str, Any] = {
        "run_name": settings.run_name,
        "timeout_seconds": settings.timeout_seconds,
        "tasks": [task],
    }
    if settings.idempotency_token:
        payload["idempotency_token"] = settings.idempotency_token

    if settings.compute_mode == "serverless":
        task["environment_key"] = settings.serverless_environment_key
        payload["environments"] = [
            {
                "environment_key": settings.serverless_environment_key,
                "spec": {
                    "environment_version": settings.serverless_environment_version,
                    **(
                        {"dependencies": list(settings.dependencies)}
                        if settings.install_dependencies and settings.dependencies
                        else {}
                    ),
                },
            }
        ]
        return payload

    if settings.compute_mode == "existing_cluster":
        task["existing_cluster_id"] = settings.existing_cluster_id
    else:
        task["new_cluster"] = settings.new_cluster

    if settings.install_dependencies and settings.dependencies:
        task["libraries"] = [
            {"pypi": {"package": dependency}}
            for dependency in settings.dependencies
        ]
    return payload


def upload_directory_to_workspace_files(
    wc: WorkspaceClient,
    *,
    source_dir: str | Path,
    destination_dir: str,
    overwrite: bool = True,
) -> list[str]:
    """Upload a local directory tree to workspace objects and return runtime paths."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"source_dir must be an existing directory: {source}")

    destination_root = destination_dir.rstrip("/")
    if not destination_root.startswith("/Workspace/"):
        raise ValueError("destination_dir must be an absolute /Workspace path.")

    uploaded: list[str] = []
    created_dirs: set[str] = set()
    directories = [source, *sorted(path for path in source.rglob("*") if path.is_dir())]
    for directory in directories:
        relative = directory.relative_to(source)
        workspace_directory = (
            destination_root
            if str(relative) == "."
            else f"{destination_root}/{relative.as_posix()}"
        )
        api_directory = _workspace_object_api_path(workspace_directory)
        if api_directory not in created_dirs:
            wc.workspace.mkdirs(api_directory)
            created_dirs.add(api_directory)

    for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = file_path.relative_to(source).as_posix()
        workspace_path = f"{destination_root}/{relative}"
        content = base64.b64encode(file_path.read_bytes()).decode("ascii")
        wc.workspace.import_(
            _workspace_object_api_path(workspace_path),
            content=content,
            format=workspace_service.ImportFormat.RAW,
            overwrite=overwrite,
        )
        uploaded.append(workspace_path)
    return uploaded


def _workspace_object_api_path(path: str) -> str:
    """Convert notebook-style /Workspace paths to Workspace object API paths."""
    if path == "/Workspace":
        return "/"
    if path.startswith("/Workspace/"):
        return path[len("/Workspace") :]
    if path.startswith(("/Users/", "/Shared/")):
        return path
    raise ValueError("workspace object path must be under /Workspace, /Users, or /Shared.")


def _non_empty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@contextmanager
def temporary_workspace_api_timeouts(
    wc: WorkspaceClient,
    *,
    request_timeout_seconds: int,
    retry_timeout_seconds: int,
) -> Any:
    api_client = getattr(wc, "api_client", None)
    base_client = getattr(api_client, "_api_client", None)
    targets: list[tuple[Any, str, Any]] = []
    for target, attr, requested_value in (
        (base_client, "_http_timeout_seconds", float(request_timeout_seconds)),
        (base_client, "_retry_timeout_seconds", int(retry_timeout_seconds)),
    ):
        if target is None or not hasattr(target, attr):
            continue
        targets.append((target, attr, getattr(target, attr)))
        setattr(target, attr, requested_value)
    try:
        yield
    finally:
        for target, attr, original_value in targets:
            setattr(target, attr, original_value)


def _first_task_parameters(payload: Mapping[str, Any]) -> list[str]:
    root_task = payload.get("spark_python_task")
    if isinstance(root_task, Mapping) and isinstance(root_task.get("parameters"), list):
        return [str(item) for item in root_task["parameters"]]
    for task in payload.get("tasks", []) or []:
        if not isinstance(task, Mapping):
            continue
        spark_python_task = task.get("spark_python_task")
        if isinstance(spark_python_task, Mapping) and isinstance(spark_python_task.get("parameters"), list):
            return [str(item) for item in spark_python_task["parameters"]]
    return []


def _parameter_value(parameters: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, item in enumerate(parameters):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item != name:
            continue
        if index + 1 >= len(parameters):
            return "true"
        next_item = parameters[index + 1]
        if next_item.startswith("--"):
            return "true"
        return next_item
    return None


def _space_id_from_url(space_url: str | None) -> str | None:
    if not space_url:
        return None
    parts = [part for part in urlsplit(space_url).path.split("/") if part]
    for index, part in enumerate(parts):
        if part == "rooms" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _workspace_artifact_dir(payload: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    parameters = _first_task_parameters(payload)
    space_url = _parameter_value(parameters, "--space-url")
    workspace_root = _parameter_value(parameters, "--workspace-root")
    space_id = _space_id_from_url(space_url)
    if not workspace_root or not space_id:
        return None, space_id, space_url
    return f"{workspace_root.rstrip('/')}/{space_id}", space_id, space_url


def _read_workspace_text(wc: WorkspaceClient, path: str) -> str | None:
    try:
        with wc.workspace.download(path) as response:
            content = response.read()
    except Exception:
        return None
    if isinstance(content, str):
        return content
    return bytes(content).decode("utf-8", errors="replace")


def _read_workspace_json(wc: WorkspaceClient, path: str) -> dict[str, Any]:
    text = _read_workspace_text(wc, path)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _workspace_object_api_path_for_status(path: str) -> str:
    if path.startswith("/Workspace/"):
        return path[len("/Workspace") :]
    return path


def _workspace_file_editor_url(
    wc: WorkspaceClient,
    *,
    workspace_path: str,
    run_page_url: str | None,
) -> str | None:
    try:
        status = wc.api_client.do(
            "GET",
            "/api/2.0/workspace/get-status",
            query={"path": _workspace_object_api_path_for_status(workspace_path)},
        )
    except Exception:
        return None
    object_id = _non_empty_text(status.get("object_id") if isinstance(status, dict) else None)
    if not object_id:
        object_id = _non_empty_text(status.get("resource_id") if isinstance(status, dict) else None)
    if not object_id:
        return None

    parsed = urlsplit(run_page_url or "")
    if not parsed.scheme or not parsed.netloc:
        host = _non_empty_text(getattr(getattr(wc, "config", None), "host", None))
        parsed = urlsplit(host or "")
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, f"/editor/files/{object_id}", parsed.query, ""))


def _score_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    overall = summary.get("overall") if isinstance(summary.get("overall"), Mapping) else {}
    total = overall.get("total") or overall.get("question_count")
    passed = overall.get("passed")
    pass_rate = overall.get("pass_rate")
    return {
        "total": int(total) if isinstance(total, (int, float)) else None,
        "passed": int(passed) if isinstance(passed, (int, float)) else None,
        "pass_rate": float(pass_rate) if isinstance(pass_rate, (int, float)) else None,
    }


def _rate_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    value = float(value)
    return f"{value:.0f}%" if abs(value - round(value)) < 0.005 else f"{value:.2f}%"


def _score_text(score: Mapping[str, Any]) -> str:
    rate = score.get("pass_rate")
    passed = score.get("passed")
    total = score.get("total")
    if isinstance(rate, (int, float)) and isinstance(passed, int) and isinstance(total, int):
        return f"{_rate_text(rate)} ({passed}/{total})"
    if isinstance(rate, (int, float)):
        return _rate_text(rate)
    return "n/a"


def _clone_space_url(
    *,
    source_space_url: str | None,
    run_page_url: str | None,
    clone_space_id: str | None,
) -> str | None:
    if not clone_space_id:
        return None
    source_parts = urlsplit(source_space_url or "")
    run_parts = urlsplit(run_page_url or "")
    scheme = source_parts.scheme or run_parts.scheme or "https"
    netloc = source_parts.netloc or run_parts.netloc
    if not netloc:
        return None
    query = source_parts.query or run_parts.query
    return urlunsplit((scheme, netloc, f"/genie/rooms/{clone_space_id}", query, ""))


def _fetch_genie_title(wc: WorkspaceClient, space_id: str | None) -> str | None:
    if not space_id:
        return None
    try:
        response = wc.api_client.do("GET", f"/api/2.0/genie/spaces/{space_id}")
    except Exception:
        return None
    return _non_empty_text(response.get("title")) if isinstance(response, dict) else None


def _observed_ui_summary(
    wc: WorkspaceClient,
    *,
    source_space_id: str | None,
    expected_question_count: int | None,
    timeout_seconds: int = 15,
) -> dict[str, Any] | None:
    if not source_space_id:
        return None
    started_at = time.monotonic()
    try:
        from maxgenie.observed_benchmarks import latest_eval_run_observed_summary

        with temporary_workspace_api_timeouts(
            wc,
            request_timeout_seconds=max(1, min(5, timeout_seconds)),
            retry_timeout_seconds=max(1, timeout_seconds),
        ):
            payload = latest_eval_run_observed_summary(
                wc=wc,
                space_id=source_space_id,
                expected_question_count=expected_question_count,
            )
    except Exception as exc:
        return {"error": str(exc)}
    elapsed_seconds = time.monotonic() - started_at
    if elapsed_seconds > timeout_seconds:
        return {"skipped": "budget_exceeded", "elapsed_seconds": elapsed_seconds}

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return None
    total = int(payload.get("result_count") or payload.get("question_count") or len(results) or 0)
    if total <= 0:
        return None
    passed = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        score = result.get("score")
        status = str(result.get("status") or result.get("assessment") or "").lower()
        if isinstance(score, (int, float)) and float(score) >= 1.0:
            passed += 1
        elif status in {"good", "passed", "pass"}:
            passed += 1
    return {
        "source": "latest_genie_eval_run",
        "eval_run_id": payload.get("eval_run_id"),
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) * 100.0,
        "assessment_counts": payload.get("assessment_counts"),
    }


def run_artifact_summary(
    wc: WorkspaceClient,
    payload: Mapping[str, Any],
    *,
    include_observed_ui: bool = False,
) -> dict[str, Any] | None:
    """Read terminal optimizer artifacts referenced by a submitted Jobs run."""
    workspace_dir, source_space_id, source_space_url = _workspace_artifact_dir(payload)
    if not workspace_dir:
        return None

    clone_meta = _read_workspace_json(wc, f"{workspace_dir}/clone_meta.json")
    latest_full = _read_workspace_json(wc, f"{workspace_dir}/results/latest_full_summary.json")
    baseline_full = _read_workspace_json(wc, f"{workspace_dir}/results/baseline_full_summary.json")
    optimization_summary = _read_workspace_json(wc, f"{workspace_dir}/results/optimization_summary.json")
    advisory_summary = _read_workspace_json(wc, f"{workspace_dir}/results/advisory_best_practices.json")
    workspace_meta = _read_workspace_json(wc, f"{workspace_dir}/workspace_meta.json")
    is_advisory = is_advisory_best_practices(advisory_summary) or is_advisory_best_practices(workspace_meta)
    final_report_path = f"{workspace_dir}/final_report.md"
    final_report_notebook_path = f"{workspace_dir}/final_report.ipynb"
    final_report = _read_workspace_text(wc, final_report_path)

    clone_space_id = (
        _non_empty_text(clone_meta.get("clone_space_id"))
        or _non_empty_text(workspace_meta.get("active_space_id"))
        or _non_empty_text(latest_full.get("clone_space_id"))
    )
    run_page_url = canonical_run_page_url(payload)
    final_report_url = (
        _workspace_file_editor_url(
            wc,
            workspace_path=final_report_path,
            run_page_url=run_page_url,
        )
        if final_report is not None
        else None
    )
    final_report_notebook_url = _workspace_file_editor_url(
        wc,
        workspace_path=final_report_notebook_path,
        run_page_url=run_page_url,
    )
    clone_url = _clone_space_url(
        source_space_url=source_space_url,
        run_page_url=run_page_url,
        clone_space_id=clone_space_id,
    )
    clone_title = (
        _non_empty_text(workspace_meta.get("clone_title"))
        or _fetch_genie_title(wc, clone_space_id)
        or clone_space_id
    )
    baseline_score = (
        {"total": None, "passed": None, "pass_rate": None}
        if is_advisory
        else _score_payload(baseline_full)
    )
    final_score = (
        {"total": None, "passed": None, "pass_rate": None}
        if is_advisory
        else _score_payload(latest_full)
    )
    baseline_rate = baseline_score.get("pass_rate")
    final_rate = final_score.get("pass_rate")
    delta = (
        float(final_rate) - float(baseline_rate)
        if isinstance(baseline_rate, (int, float)) and isinstance(final_rate, (int, float))
        else None
    )
    baseline_source = (
        "no_benchmark"
        if is_advisory
        else (
            _non_empty_text(baseline_full.get("baseline_source"))
            or _non_empty_text(workspace_meta.get("observed_benchmark_reuse_policy"))
            or "benchmark_run"
        )
    )
    state = payload.get("state") if isinstance(payload.get("state"), Mapping) else {}
    is_terminal = str(state.get("life_cycle_state") or "") in _TERMINAL_LIFECYCLE_STATES
    observed_ui = (
        _observed_ui_summary(
            wc,
            source_space_id=source_space_id,
            expected_question_count=final_score.get("total") if isinstance(final_score.get("total"), int) else None,
        )
        if include_observed_ui and is_terminal and final_score.get("pass_rate") is not None
        else None
    )
    if is_advisory:
        terminal_selection = {
            "status": NOT_BENCHMARK_VERIFIED_STATUS,
            "reasons": ["no_benchmark"],
        }
        split_scores = {}
    else:
        terminal_selection = (
            optimization_summary.get("terminal_selection")
            if isinstance(optimization_summary.get("terminal_selection"), Mapping)
            else {}
        )
        split_scores = {
            "train": (latest_full.get("train") or {}).get("pass_rate") if isinstance(latest_full.get("train"), Mapping) else None,
            "validation": (
                (latest_full.get("validation") or {}).get("pass_rate")
                if isinstance(latest_full.get("validation"), Mapping)
                else None
            ),
            "holdout": (latest_full.get("test") or {}).get("pass_rate") if isinstance(latest_full.get("test"), Mapping) else None,
        }
    return {
        "workspace_dir": workspace_dir,
        "final_report_path": final_report_path if final_report is not None else None,
        "final_report_url": final_report_url,
        "final_report_notebook_path": final_report_notebook_path,
        "final_report_notebook_url": final_report_notebook_url,
        "clone_space_id": clone_space_id,
        "clone_title": clone_title,
        "clone_url": clone_url,
        "baseline_source": baseline_source,
        "baseline_score": baseline_score,
        "final_score": final_score,
        "delta_pass_rate": delta,
        "split_scores": split_scores,
        "accepted_candidates": optimization_summary.get("accepted"),
        "rejected_candidates": optimization_summary.get("rejected"),
        "skipped_candidates": optimization_summary.get("skipped"),
        "terminal_selection": terminal_selection,
        "observed_ui_benchmark": observed_ui,
        "advisory_summary": advisory_summary if is_advisory else None,
        "evidence_classification": (
            ADVISORY_BEST_PRACTICES_CLASSIFICATION if is_advisory else None
        ),
        "benchmark_verification_status": (
            NOT_BENCHMARK_VERIFIED_STATUS if is_advisory else None
        ),
        "benchmark_verification_label": (
            NOT_BENCHMARK_VERIFIED_LABEL if is_advisory else None
        ),
        "no_benchmark_warning": (
            _non_empty_text(advisory_summary.get("warning"))
            or _non_empty_text(workspace_meta.get("no_benchmark_warning"))
            or NO_BENCHMARK_WARNING
            if is_advisory
            else None
        ),
    }


def _recorded_at_ms(record: Mapping[str, Any]) -> int | None:
    recorded_at = _non_empty_text(record.get("recorded_at") or record.get("timestamp"))
    if recorded_at is None:
        return None
    try:
        parsed = datetime.fromisoformat(recorded_at)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _iteration_records(
    text: str | None,
    *,
    max_events: int = 80,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> list[dict[str, Any]]:
    if not text:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if since_ms is not None or until_ms is not None:
                recorded_ms = _recorded_at_ms(payload)
                if recorded_ms is not None and since_ms is not None and recorded_ms < since_ms:
                    continue
                if recorded_ms is not None and until_ms is not None and recorded_ms > until_ms:
                    continue
            records.append(payload)
    return records[-max_events:]


def _inline_text(value: Any, *, limit: int = 140) -> str | None:
    text = _non_empty_text(value)
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _record_summary_score(record: Mapping[str, Any]) -> str | None:
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        return None
    score = {
        "pass_rate": summary.get("pass_rate"),
        "passed": summary.get("passed"),
        "total": summary.get("question_count") or summary.get("total"),
    }
    rendered = _score_text(score)
    return rendered if rendered != "n/a" else None


def _benchmark_label(record: Mapping[str, Any]) -> str:
    label = _non_empty_text(record.get("label"))
    benchmark_phase = _non_empty_text(record.get("benchmark_phase"))
    phase = _non_empty_text(record.get("phase")) or "benchmark"
    return label or benchmark_phase or phase


def _candidate_progress_text(record: Mapping[str, Any]) -> str:
    outcome = _non_empty_text(record.get("outcome")) or "evaluated"
    name = _non_empty_text(record.get("name")) or _non_empty_text(record.get("label")) or "candidate"
    score_parts = []
    if isinstance(record.get("train_rate"), (int, float)):
        score_parts.append(f"train {_rate_text(record.get('train_rate'))}")
    if isinstance(record.get("validation_rate"), (int, float)):
        score_parts.append(f"validation {_rate_text(record.get('validation_rate'))}")
    scores = f" ({', '.join(score_parts)})" if score_parts else ""
    reason = _inline_text(record.get("reason"), limit=120)
    suffix = f" - {reason}" if reason else ""
    return f"Candidate `{name}` {outcome}{scores}{suffix}"


def _progress_message(record: Mapping[str, Any]) -> str | None:
    phase = _non_empty_text(record.get("phase"))
    if phase is None or phase.startswith("serving_candidate_replay_"):
        return None
    if phase == "export":
        return "Exported source space assets."
    if phase == "fixed_benchmark_artifacts_applied":
        return "Applied fixed benchmark seed."
    if phase == "split_rebalanced":
        return "Prepared train, validation, and hidden holdout splits."
    if phase == "clone_created":
        return "Created optimization clone."
    if phase == "clone_reused":
        return "Reused optimization clone."
    if phase == "clone_recovered":
        return "Recovered optimization clone from checkpoint."
    if phase == "push":
        return "Pushed candidate state to the optimization clone."
    if phase == "benchmark_baseline_observed":
        score = _record_summary_score(record)
        return f"Loaded observed baseline benchmark: {score}." if score else "Loaded observed baseline benchmark."
    if phase == "benchmark_questions_start":
        count = record.get("question_count")
        label = _benchmark_label(record)
        count_text = f" ({int(count)} questions)" if isinstance(count, (int, float)) else ""
        return f"Benchmarking `{label}`{count_text}."
    if phase == "benchmark_questions_error":
        return f"Benchmark `{_benchmark_label(record)}` failed: {_inline_text(record.get('error_message')) or 'error'}"
    if phase == "benchmark_questions_complete":
        score = _record_summary_score(record)
        label = _benchmark_label(record)
        return f"Completed benchmark `{label}`: {score}." if score else f"Completed benchmark `{label}`."
    if phase in {"benchmark_train", "benchmark_validation", "benchmark_test", "benchmark_full"}:
        score = _record_summary_score(record)
        label = _benchmark_label(record)
        return f"Completed `{label}`: {score}." if score else f"Completed `{label}`."
    if phase.endswith("_start") and phase.startswith("benchmark_"):
        count = record.get("question_count")
        count_text = f" ({int(count)} questions)" if isinstance(count, (int, float)) else ""
        return f"Benchmarking `{phase[:-6]}`{count_text}."
    if phase == "serving_candidate_proposal_start":
        return "Requesting the next candidate proposal."
    if phase == "serving_candidate_proposal_invalid":
        return f"Candidate proposal was invalid: {_inline_text(record.get('reason')) or 'invalid response'}"
    if phase == "serving_candidate_proposal_received":
        return "Received candidate proposal."
    if phase == "candidate_evaluated":
        return _candidate_progress_text(record)
    if phase == "terminal_result_selection":
        return "Selected terminal optimization result."
    if phase == "generalization_audit":
        return "Completed generalization audit."
    if phase == "optimization_completed":
        return "Optimization completed."
    if phase == "protocol_preflight":
        return "Validated production protocol preflight."
    return None


def _record_fingerprint(record: Mapping[str, Any]) -> str:
    values = [
        record.get("timestamp"),
        record.get("phase"),
        record.get("label"),
        record.get("name"),
        record.get("outcome"),
        record.get("train_rate"),
        record.get("validation_rate"),
        record.get("reason"),
    ]
    return "|".join("" if value is None else str(value) for value in values)


def _progress_time_window(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Return the artifact event window for the latest task attempt in a Jobs run."""
    latest_task_start: int | None = None
    latest_task_end: int | None = None
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            task_start = _timestamp_ms(task.get("start_time"))
            if task_start is None:
                continue
            if latest_task_start is None or task_start > latest_task_start:
                latest_task_start = task_start
                latest_task_end = _timestamp_ms(task.get("end_time"))
    if latest_task_start is not None:
        return latest_task_start, latest_task_end
    return _timestamp_ms(payload.get("start_time")), _timestamp_ms(payload.get("end_time"))


def run_progress_summary(
    wc: WorkspaceClient,
    payload: Mapping[str, Any],
    *,
    max_events: int = 80,
) -> dict[str, Any] | None:
    """Read live optimizer progress events for a submitted Jobs run."""
    workspace_dir, _, _ = _workspace_artifact_dir(payload)
    if not workspace_dir:
        return None
    run_start_ms, run_end_ms = _progress_time_window(payload)
    records = _iteration_records(
        _read_workspace_text(wc, f"{workspace_dir}/history/iteration_log.jsonl"),
        max_events=max_events,
        since_ms=(run_start_ms - 300_000 if run_start_ms is not None else None),
        until_ms=(run_end_ms + 300_000 if run_end_ms is not None else None),
    )
    baseline_full = _read_workspace_json(wc, f"{workspace_dir}/results/baseline_full_summary.json")
    latest_full = _read_workspace_json(wc, f"{workspace_dir}/results/latest_full_summary.json")
    baseline_score = _score_payload(baseline_full)
    latest_score = _score_payload(latest_full)
    candidate_records = [
        record
        for record in records
        if _non_empty_text(record.get("phase")) == "candidate_evaluated"
    ]
    latest_candidate = candidate_records[-1] if candidate_records else None
    best_train = max(
        (float(record["train_rate"]) for record in candidate_records if isinstance(record.get("train_rate"), (int, float))),
        default=None,
    )
    best_validation = max(
        (
            float(record["validation_rate"])
            for record in candidate_records
            if isinstance(record.get("validation_rate"), (int, float))
        ),
        default=None,
    )
    messages: list[str] = []
    for record in records:
        message = _progress_message(record)
        if message is None:
            continue
        if messages and messages[-1] == message:
            continue
        messages.append(message)
    latest_record = records[-1] if records else {}
    baseline_status = None
    if baseline_score.get("pass_rate") is None:
        benchmark_started = any(
            _non_empty_text(record.get("phase")) in {"benchmark_questions_start", "benchmark_train_subset_start"}
            for record in records
        )
        baseline_status = "baseline benchmarking in progress" if benchmark_started else "waiting for baseline artifacts"
    return {
        "workspace_dir": workspace_dir,
        "event_count": len(records),
        "latest_message": messages[-1] if messages else baseline_status,
        "recent_messages": messages[-3:],
        "last_candidate": _candidate_progress_text(latest_candidate) if latest_candidate else None,
        "running_best_train": best_train,
        "running_best_validation": best_validation,
        "baseline_score": baseline_score,
        "latest_full_score": latest_score,
        "baseline_status": baseline_status,
        "fingerprint": _record_fingerprint(latest_record) if latest_record else baseline_status,
    }


def _job_id_from_fragment(fragment: str) -> str | None:
    parts = [part for part in fragment.strip("#/").split("/") if part]
    if len(parts) >= 2 and parts[0] == "job":
        return parts[1]
    if parts and parts[0].startswith("job-run-"):
        return parts[0][len("job-run-") :] or None
    return None


def _canonical_run_page_url(
    page_url: str | None,
    *,
    job_id: str | None,
    run_id: str | None,
) -> str | None:
    page_url = _non_empty_text(page_url)
    if not page_url:
        return None
    if not job_id or not run_id:
        return page_url
    parsed = urlsplit(page_url)
    if not parsed.scheme or not parsed.netloc:
        return page_url
    path = parsed.path or "/"
    fragment = f"job/{job_id}/run/{run_id}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, fragment))


def canonical_run_page_url(
    payload: Mapping[str, Any],
    *,
    fallback_run_page_url: str | None = None,
    fallback_run_id: str | int | None = None,
    fallback_job_id: str | int | None = None,
) -> str | None:
    """Return the current Databricks Jobs UI URL for a run payload."""
    page_url = _non_empty_text(payload.get("run_page_url")) or fallback_run_page_url
    run_id = _non_empty_text(payload.get("run_id")) or _non_empty_text(fallback_run_id)
    job_id = (
        _non_empty_text(payload.get("job_id"))
        or _non_empty_text(fallback_job_id)
        or _job_id_from_fragment(urlsplit(page_url or "").fragment)
    )
    return _canonical_run_page_url(page_url, job_id=job_id, run_id=run_id)


def submit_workspace_run(
    wc: WorkspaceClient,
    *,
    settings: WorkspaceJobSettings,
) -> dict[str, Any]:
    """Submit a one-time Databricks job run."""
    payload = build_runs_submit_payload(settings)
    response = wc.api_client.do("POST", "/api/2.1/jobs/runs/submit", body=payload)
    run_id = response.get("run_id")
    status_payload: dict[str, Any] | None = None
    status_error: str | None = None
    if run_id:
        try:
            status_payload = get_run_status(wc, run_id=run_id)
        except Exception as exc:
            status_error = str(exc)
    run_page_url = canonical_run_page_url(
        status_payload or response,
        fallback_run_page_url=_non_empty_text(response.get("run_page_url")),
        fallback_run_id=run_id,
    )
    return {
        "request": payload,
        "response": response,
        "status": status_payload,
        "status_error": status_error,
        "status_summary": summarize_run_status(status_payload) if status_payload else None,
        "run_id": run_id,
        "run_page_url": run_page_url,
    }


def get_run_status(wc: WorkspaceClient, *, run_id: int | str) -> dict[str, Any]:
    """Fetch a Databricks job run status."""
    return wc.api_client.do("GET", "/api/2.1/jobs/runs/get", query={"run_id": str(run_id)})


def cancel_run(wc: WorkspaceClient, *, run_id: int | str) -> dict[str, Any]:
    """Cancel a Databricks job run."""
    return wc.api_client.do("POST", "/api/2.1/jobs/runs/cancel", body={"run_id": int(run_id)})


def _timestamp_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return None


def _elapsed_seconds(
    *,
    start_time_ms: int | None,
    end_time_ms: int | None,
    now_ms: int,
) -> float | None:
    if start_time_ms is None:
        return None
    effective_end_ms = end_time_ms if end_time_ms is not None else now_ms
    if effective_end_ms < start_time_ms:
        return None
    return round((effective_end_ms - start_time_ms) / 1000.0, 1)


def summarize_run_status(
    payload: dict[str, Any],
    *,
    now_ms: int | None = None,
    stale_after_seconds: int = 3600,
) -> dict[str, Any]:
    """Return a compact monitor-friendly summary for a Jobs run payload."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    state = payload.get("state", {}) if isinstance(payload.get("state"), dict) else {}
    start_time_ms = _timestamp_ms(payload.get("start_time"))
    end_time_ms = _timestamp_ms(payload.get("end_time"))
    elapsed_seconds = _elapsed_seconds(
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        now_ms=now_ms,
    )
    lifecycle_state = str(state.get("life_cycle_state") or "UNKNOWN")
    job_id = _non_empty_text(payload.get("job_id"))
    run_id = _non_empty_text(payload.get("run_id"))
    run_page_url = canonical_run_page_url(payload, fallback_run_id=run_id, fallback_job_id=job_id)
    task_summaries: list[dict[str, Any]] = []
    latest_task_update_ms: int | None = None
    for task in payload.get("tasks", []) or []:
        if not isinstance(task, dict):
            continue
        task_state = task.get("state", {}) if isinstance(task.get("state"), dict) else {}
        task_start_ms = _timestamp_ms(task.get("start_time"))
        task_end_ms = _timestamp_ms(task.get("end_time"))
        task_update_candidates = [
            value
            for value in (latest_task_update_ms, task_start_ms, task_end_ms)
            if value is not None
        ]
        latest_task_update_ms = max(task_update_candidates) if task_update_candidates else None
        task_summaries.append(
            {
                "task_key": task.get("task_key"),
                "run_id": task.get("run_id"),
                "run_page_url": canonical_run_page_url(
                    task,
                    fallback_job_id=job_id,
                ),
                "life_cycle_state": task_state.get("life_cycle_state"),
                "result_state": task_state.get("result_state"),
                "state_message": task_state.get("state_message"),
                "elapsed_seconds": _elapsed_seconds(
                    start_time_ms=task_start_ms,
                    end_time_ms=task_end_ms,
                    now_ms=now_ms,
                ),
            }
        )
    stale = False
    stale_reason: str | None = None
    if (
        lifecycle_state in _ACTIVE_LIFECYCLE_STATES
        and elapsed_seconds is not None
        and elapsed_seconds >= stale_after_seconds
    ):
        stale = True
        if latest_task_update_ms is None:
            stale_reason = f"active for {elapsed_seconds:.1f}s with no task timing updates"
        else:
            quiet_seconds = round((now_ms - latest_task_update_ms) / 1000.0, 1)
            stale_reason = (
                f"active for {elapsed_seconds:.1f}s; "
                f"latest task timing update {quiet_seconds:.1f}s ago"
            )
    return {
        "run_id": payload.get("run_id"),
        "job_id": payload.get("job_id"),
        "run_name": payload.get("run_name"),
        "run_page_url": run_page_url,
        "life_cycle_state": lifecycle_state,
        "result_state": state.get("result_state"),
        "state_message": state.get("state_message"),
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "elapsed_seconds": elapsed_seconds,
        "stale": stale,
        "stale_reason": stale_reason,
        "tasks": task_summaries,
    }


def _format_elapsed(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return "unknown elapsed"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def format_run_status_for_chat(summary: Mapping[str, Any]) -> str:
    """Render a compact status update suitable for a Genie Code chat response."""
    run_id = summary.get("run_id") or "unknown"
    lifecycle = summary.get("life_cycle_state") or "UNKNOWN"
    result = summary.get("result_state")
    state_text = f"{lifecycle}" + (f" / {result}" if result else "")
    elapsed = _format_elapsed(summary.get("elapsed_seconds"))
    lines = [f"Run `{run_id}` is `{state_text}` ({elapsed})."]
    run_page_url = _non_empty_text(summary.get("run_page_url"))
    if run_page_url:
        lines.append(f"Track progress: [Job Run {run_id}]({run_page_url})")
    message = _non_empty_text(summary.get("state_message"))
    if message:
        lines.append(f"Message: {message}")
    stale_reason = _non_empty_text(summary.get("stale_reason"))
    if stale_reason:
        lines.append(f"Watch note: {stale_reason}")
    tasks = summary.get("tasks")
    if isinstance(tasks, list) and tasks:
        lines.append("")
        lines.append("Task states:")
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            task_key = task.get("task_key") or "task"
            task_run_id = task.get("run_id") or "unknown"
            task_lifecycle = task.get("life_cycle_state") or "UNKNOWN"
            task_result = task.get("result_state")
            task_state = f"{task_lifecycle}" + (f" / {task_result}" if task_result else "")
            task_elapsed = _format_elapsed(task.get("elapsed_seconds"))
            task_url = _non_empty_text(task.get("run_page_url"))
            if task_url:
                task_label = f"[`{task_run_id}`]({task_url})"
            else:
                task_label = f"`{task_run_id}`"
            task_line = f"- `{task_key}` {task_label}: `{task_state}` ({task_elapsed})"
            task_message = _non_empty_text(task.get("state_message"))
            if task_message:
                task_line += f" - {task_message}"
            lines.append(task_line)
    artifacts = summary.get("artifact_summary")
    if isinstance(artifacts, Mapping):
        lines.append("")
        lines.append("Optimization result:")
        clone_url = _non_empty_text(artifacts.get("clone_url"))
        clone_title = _non_empty_text(artifacts.get("clone_title")) or _non_empty_text(
            artifacts.get("clone_space_id")
        )
        if clone_url:
            lines.append(f"- Optimized space: [open optimized space]({clone_url})")
        if clone_title:
            lines.append(f"- Optimized space title: `{clone_title}`")

        advisory_summary = artifacts.get("advisory_summary")
        is_advisory = (
            artifacts.get("evidence_classification") == ADVISORY_BEST_PRACTICES_CLASSIFICATION
            or is_advisory_best_practices(advisory_summary)
        )
        if is_advisory:
            warning = (
                _non_empty_text(artifacts.get("no_benchmark_warning"))
                or _non_empty_text(advisory_summary.get("warning") if isinstance(advisory_summary, Mapping) else None)
                or NO_BENCHMARK_WARNING
            )
            change_count = (
                advisory_summary.get("change_count")
                if isinstance(advisory_summary, Mapping)
                else None
            )
            curation_mode = (
                _non_empty_text(advisory_summary.get("curation_mode"))
                if isinstance(advisory_summary, Mapping)
                else None
            )
            lines.append(f"- {warning}")
            lines.append(f"- Evidence classification: `{ADVISORY_BEST_PRACTICES_CLASSIFICATION}`")
            lines.append(f"- Benchmark verification: `{NOT_BENCHMARK_VERIFIED_LABEL}`")
            if curation_mode:
                lines.append(f"- Advisory mode: `{curation_mode}`")
            if isinstance(change_count, (int, float)):
                lines.append(f"- Advisory changes: `{int(change_count)}`")
            lines.append("- Final full score: `n/a (no benchmark)`")

        baseline_score = artifacts.get("baseline_score")
        final_score = artifacts.get("final_score")
        if (
            not is_advisory
            and isinstance(final_score, Mapping)
            and final_score.get("pass_rate") is not None
        ):
            lines.append(f"- Final full score: `{_score_text(final_score)}`")
        if (
            not is_advisory
            and isinstance(baseline_score, Mapping)
            and baseline_score.get("pass_rate") is not None
        ):
            lines.append(f"- Run baseline: `{_score_text(baseline_score)}`")
        delta = artifacts.get("delta_pass_rate")
        if not is_advisory and isinstance(delta, (int, float)):
            lines.append(f"- Run delta: `{delta:+.2f}%`")
        split_scores = artifacts.get("split_scores")
        if not is_advisory and isinstance(split_scores, Mapping):
            train = _rate_text(split_scores.get("train"))
            validation = _rate_text(split_scores.get("validation"))
            holdout = _rate_text(split_scores.get("holdout"))
            lines.append(f"- Split scores: train `{train}`, validation `{validation}`, holdout `{holdout}`")

        observed_ui = artifacts.get("observed_ui_benchmark")
        if isinstance(observed_ui, Mapping) and observed_ui.get("pass_rate") is not None:
            observed_score = _score_text(observed_ui)
            eval_run_id = _non_empty_text(observed_ui.get("eval_run_id"))
            eval_suffix = f" from eval run `{eval_run_id}`" if eval_run_id else ""
            lines.append(f"- Existing Genie UI eval: `{observed_score}`{eval_suffix}")
            baseline_source = str(artifacts.get("baseline_source") or "")
            if baseline_source in {"benchmark_run", "fresh_baseline_benchmark_run"}:
                lines.append(
                    "- Note: `no net improvement` is measured against the fresh MaxGenie "
                    "result-based baseline from this run, not against the existing Genie UI eval."
                )
        elif isinstance(observed_ui, Mapping) and observed_ui.get("error"):
            lines.append(f"- Existing Genie UI eval: unavailable ({observed_ui.get('error')})")
        elif isinstance(observed_ui, Mapping) and observed_ui.get("skipped"):
            lines.append(f"- Existing Genie UI eval: skipped ({observed_ui.get('skipped')})")

        terminal_selection = artifacts.get("terminal_selection")
        if isinstance(terminal_selection, Mapping) and terminal_selection.get("status"):
            reasons = terminal_selection.get("reasons")
            reason_text = (
                ", ".join(str(reason) for reason in reasons)
                if isinstance(reasons, list) and reasons
                else "none"
            )
            lines.append(
                f"- Terminal selection: `{terminal_selection.get('status')}`; reasons `{reason_text}`"
            )
        report_url = _non_empty_text(
            artifacts.get("final_report_notebook_url") or artifacts.get("final_report_url")
        )
        raw_report_url = _non_empty_text(artifacts.get("final_report_url"))
        report_path = _non_empty_text(artifacts.get("final_report_path"))
        notebook_path = _non_empty_text(artifacts.get("final_report_notebook_path"))
        if report_url:
            lines.append(f"- Final report: [open rendered report]({report_url})")
            if raw_report_url and raw_report_url != report_url:
                lines.append(f"- Raw report file: [open markdown report]({raw_report_url})")
        elif notebook_path:
            lines.append(f"- Final report: `{notebook_path}`")
        elif report_path:
            lines.append(f"- Final report: `{report_path}`")
    progress = summary.get("progress_summary")
    if isinstance(progress, Mapping):
        lines.append("")
        lines.append("Progress:")
        latest_message = _non_empty_text(progress.get("latest_message"))
        if latest_message:
            lines.append(f"- Latest: {latest_message}")
        baseline_score = progress.get("baseline_score")
        if isinstance(baseline_score, Mapping) and baseline_score.get("pass_rate") is not None:
            lines.append(f"- Baseline: `{_score_text(baseline_score)}`")
        else:
            baseline_status = _non_empty_text(progress.get("baseline_status"))
            if baseline_status:
                lines.append(f"- Baseline: {baseline_status}.")
        latest_full_score = progress.get("latest_full_score")
        if isinstance(latest_full_score, Mapping) and latest_full_score.get("pass_rate") is not None:
            lines.append(f"- Latest full score: `{_score_text(latest_full_score)}`")
        last_candidate = _non_empty_text(progress.get("last_candidate"))
        if last_candidate:
            lines.append(f"- Last candidate: {last_candidate}")
        best_parts = []
        if isinstance(progress.get("running_best_train"), (int, float)):
            best_parts.append(f"train `{_rate_text(progress.get('running_best_train'))}`")
        if isinstance(progress.get("running_best_validation"), (int, float)):
            best_parts.append(f"validation `{_rate_text(progress.get('running_best_validation'))}`")
        if best_parts:
            lines.append(f"- Running best: {', '.join(best_parts)}")
        recent_messages = progress.get("recent_messages")
        if isinstance(recent_messages, list):
            for message in recent_messages[-2:]:
                text = _non_empty_text(message)
                if text and text != latest_message:
                    lines.append(f"- Recent: {text}")
    return "\n".join(lines)


def wait_for_run(
    wc: WorkspaceClient,
    *,
    run_id: int | str,
    poll_seconds: float = 30.0,
    timeout_seconds: int = 14400,
    on_tick: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll a job run until it reaches a terminal lifecycle state."""
    deadline = time.time() + timeout_seconds
    latest: dict[str, Any] = {}
    while time.time() < deadline:
        latest = get_run_status(wc, run_id=run_id)
        if on_tick is not None:
            on_tick(latest)
        state = latest.get("state", {}) if isinstance(latest, dict) else {}
        lifecycle_state = str(state.get("life_cycle_state") or "")
        if lifecycle_state in _TERMINAL_LIFECYCLE_STATES:
            return latest
        time.sleep(poll_seconds)
    raise TimeoutError(f"Run {run_id} did not finish within {timeout_seconds} seconds.")


def parse_new_cluster_json(raw: str | None) -> dict[str, Any] | None:
    """Parse a new-cluster JSON string or file path."""
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    candidate_path = Path(text).expanduser() if not text.startswith("{") else None
    if candidate_path is not None and candidate_path.exists():
        text = candidate_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("New cluster configuration must be a JSON object.")
    return payload
