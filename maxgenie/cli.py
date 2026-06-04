"""CLI entrypoint for MaxGenie."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import typer
from databricks.sdk.errors import DatabricksError

from maxgenie.advisory import (
    ADVISORY_BEST_PRACTICES_CLASSIFICATION,
    NO_BENCHMARK_WARNING,
    NOT_BENCHMARK_VERIFIED_LABEL,
)
from maxgenie.assembler import assemble_config
from maxgenie.artifacts import ArtifactManager
from maxgenie.auth import AuthConfigurationError, create_workspace_client, diagnose_auth
from maxgenie.benchmark_service import BenchmarkService
from maxgenie.config import (
    AUTONOMOUS_STRATEGY_CENTAUR,
    AUTONOMOUS_STRATEGY_SERVING,
    DEFAULT_AUTONOMOUS_STRATEGY,
    STRATEGY_WORKSPACE_ROOT_NAMES,
    default_serving_endpoint,
    default_serving_model,
    default_serving_reasoning_effort,
    default_serving_temperature,
    default_space_url,
    default_workspace_dir,
    default_workspace_root_for_strategy,
    normalize_autonomous_strategy,
    parse_space_url,
    resolve_space_url,
)
from maxgenie.exporter import SpaceExporter
from maxgenie.genie_service import GenieService
from maxgenie.observed_benchmarks import (
    latest_eval_run_observed_payload,
    load_observed_benchmark_run,
    write_latest_observed_benchmark_seed,
)
from maxgenie.positive_control import (
    materialize_positive_control_patch_sequence,
    resolve_positive_control,
)
from maxgenie.protocol_preflight import validate_protocol_preflight
from maxgenie.schemas import GenieSpaceConfig
from maxgenie.space_curation import (
    CURATION_MODE_SAFE_DEFAULTS,
    SUPPORTED_CURATION_MODES,
    NARROW_CURATION_MODES,
)
from maxgenie.query_history import build_expansion_candidates, discover_best_warehouse_id
from maxgenie.multi_sample import build_noise_model
from maxgenie.workspace_flow import (
    OptimizationWorkspace,
    WorkspaceFlowError,
    infer_workspace_dir,
)
from maxgenie.workspace_jobs import (
    WorkspaceJobSettings,
    build_runs_submit_payload,
    cancel_run,
    default_runner_path,
    format_run_status_for_chat,
    get_run_status,
    parse_new_cluster_json,
    run_artifact_summary,
    run_progress_summary,
    submit_workspace_run,
    summarize_run_status,
    temporary_workspace_api_timeouts,
    upload_directory_to_workspace_files,
    wait_for_run,
)

app = typer.Typer(name="maxgenie", help="CLI toolkit for Genie Space optimization workspaces")


def _databricks_api_error_message(action: str, exc: DatabricksError) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"Databricks API error while {action}: {message}"


def _workspace_from_inputs(
    workspace_root: str,
    space_id: str | None = None,
) -> ArtifactManager:
    workspace_dir = infer_workspace_dir(workspace_root, space_id)
    return ArtifactManager(workspace_dir=workspace_dir, space_id=workspace_dir.name)


def _load_split_meta(artifacts: ArtifactManager) -> dict[str, Any]:
    if not artifacts.split_meta_path.exists():
        raise WorkspaceFlowError(
            "Workspace metadata is missing. Run `maxgenie export --space-url ...` or `maxgenie optimize --space-url ...` first."
        )
    payload = artifacts.read_json_path(artifacts.split_meta_path)
    if not isinstance(payload, dict):
        raise WorkspaceFlowError("Invalid split_meta.json payload.")
    return payload


def _looks_like_missing_space_error(message: str) -> bool:
    lowered = message.lower()
    return "unable to get space" in lowered or "does not exist" in lowered


_PREFERRED_DISCOVERED_SERVING_ENDPOINTS: tuple[str, ...] = (
    "databricks-gpt-5-5",
    "databricks-gpt-5-5-pro",
    "databricks-claude-opus-4-8",
    "databricks-claude-opus-4-7",
    "databricks-claude-opus-4-6",
)


def _resource_value(resource: Any, key: str) -> Any:
    if isinstance(resource, dict):
        return resource.get(key)
    return getattr(resource, key, None)


def _normalized_resource_state(value: Any) -> str:
    return str(value or "").split(".")[-1].upper()


def _serving_endpoint_is_ready(endpoint: Any) -> bool:
    state = _resource_value(endpoint, "state")
    ready = _resource_value(state, "ready")
    if ready is None:
        return True
    return _normalized_resource_state(ready) == "READY"


def _discover_best_serving_endpoint(wc: Any) -> str | None:
    try:
        endpoints = list(wc.serving_endpoints.list())
    except Exception:
        try:
            response = wc.api_client.do("GET", "/api/2.0/serving-endpoints")
        except Exception:
            return None
        endpoints = response.get("endpoints", []) if isinstance(response, dict) else []

    ready_names: list[str] = []
    for endpoint in endpoints:
        name = _resource_value(endpoint, "name")
        if not name or not _serving_endpoint_is_ready(endpoint):
            continue
        ready_names.append(str(name))

    if not ready_names:
        return None
    ready_name_set = set(ready_names)
    for preferred_name in _PREFERRED_DISCOVERED_SERVING_ENDPOINTS:
        if preferred_name in ready_name_set:
            return preferred_name
    for prefix in ("databricks-claude-", "databricks-gpt-5-"):
        for name in ready_names:
            if name.startswith(prefix):
                return name
    return ready_names[0]


def _resolve_serving_inputs_for_submit(
    wc: Any,
    *,
    strategy: str,
    serving_endpoint: str | None,
    serving_model: str | None,
) -> tuple[str | None, str | None]:
    if strategy != AUTONOMOUS_STRATEGY_SERVING:
        return serving_endpoint or default_serving_endpoint(), serving_model or default_serving_model()

    resolved_endpoint = serving_endpoint or default_serving_endpoint()
    if not resolved_endpoint:
        resolved_endpoint = _discover_best_serving_endpoint(wc)

    resolved_model = serving_model or default_serving_model()
    if resolved_endpoint and not resolved_model:
        resolved_model = resolved_endpoint
    return resolved_endpoint, resolved_model


def _build_flow(
    *,
    artifacts: ArtifactManager,
    profile: str | None,
    host: str | None,
    delay: float,
    match_threshold: float,
    warehouse_id: str | None,
    clone_parent_path: str | None = None,
    result_content_match: bool | None = None,
) -> OptimizationWorkspace:
    try:
        wc = create_workspace_client(profile=profile, host=host)
    except AuthConfigurationError as exc:
        raise WorkspaceFlowError(str(exc)) from exc

    genie_service = GenieService(wc)
    benchmark_service = BenchmarkService(wc)
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    return OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=delay,
        match_threshold=match_threshold,
        warehouse_id=warehouse_id,
        clone_parent_path=clone_parent_path,
        result_content_match=result_content_match,
    )


def _resolved_space_ref(space_url: str | None, *, required: bool) -> Any:
    try:
        resolved_space_url = resolve_space_url(space_url)
    except ValueError as exc:
        if required:
            raise WorkspaceFlowError(str(exc)) from exc
        return None

    try:
        return parse_space_url(resolved_space_url)
    except ValueError as exc:
        raise WorkspaceFlowError(str(exc)) from exc


def _benchmark_mode(train: bool, validation: bool, test: bool, full: bool) -> str:
    modes = [
        name
        for name, enabled in (("train", train), ("validation", validation), ("test", test), ("full", full))
        if enabled
    ]
    if len(modes) != 1:
        raise WorkspaceFlowError("Choose exactly one of --train, --validation, --test, or --full.")
    return modes[0]


def _format_optional_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _resolved_workspace_root(
    workspace_root: str | None,
    *,
    strategy: str,
) -> Path:
    if workspace_root and workspace_root.strip():
        return Path(workspace_root).expanduser().resolve()
    return default_workspace_root_for_strategy(strategy)


@app.command()
def doctor(
    space_url: str = typer.Option(None, "--space-url", help="Optional Genie Space URL to infer workspace host"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile to test"),
):
    """Validate Databricks authentication and print setup guidance."""

    host = None
    try:
        ref = _resolved_space_ref(space_url, required=False)
    except WorkspaceFlowError as exc:
        typer.secho(f"Space URL parse warning: {exc}", fg=typer.colors.YELLOW)
        ref = None
    if ref is not None:
        host = ref.host

    diag = diagnose_auth(profile=profile, host=host)
    color = typer.colors.GREEN if diag.ok else typer.colors.RED
    typer.secho(f"Auth status: {'OK' if diag.ok else 'NOT READY'}", fg=color)
    typer.echo(f"Mode: {diag.mode}")
    typer.echo(f"Message: {diag.message}")

    if not diag.ok:
        typer.echo("\nSetup options:")
        typer.echo("1) Use profile: maxgenie optimize --profile <profile> --space-url <url>")
        typer.echo("2) Set env vars: DATABRICKS_HOST + DATABRICKS_TOKEN")
        typer.echo("3) Or set DATABRICKS_CONFIG_PROFILE")
        typer.echo("4) Optionally set MAXGENIE_DEFAULT_SPACE_URL for your default test space")
        raise typer.Exit(2)


@app.command()
def export(
    space_url: str = typer.Option(
        None,
        "--space-url",
        help="Genie Space URL. Falls back to MAXGENIE_DEFAULT_SPACE_URL if unset.",
    ),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
    validation_fraction: float = typer.Option(0.2, "--validation-fraction", min=0.0, max=0.5),
    test_fraction: float = typer.Option(0.2, "--test-fraction", min=0.0, max=0.5),
    split_seed: int = typer.Option(42, "--split-seed"),
    fresh_lineage: bool = typer.Option(
        False,
        "--fresh-lineage",
        help="Allow export to overwrite an existing workspace lineage.",
    ),
):
    """Download space assets into a decomposed optimization workspace."""

    try:
        ref = _resolved_space_ref(space_url, required=True)
        artifacts = ArtifactManager(
            workspace_dir=default_workspace_dir(workspace_root, ref.space_id),
            space_id=ref.space_id,
        )
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=ref.host,
            delay=12.0,
            match_threshold=0.85,
            warehouse_id=None,
        )
        bundle = flow.export_workspace(
            ref.space_id,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
        )
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Workspace: {artifacts.workspace_dir}")
    typer.echo(f"Train benchmarks: {len(bundle.split.train_questions)}")
    typer.echo(f"Validation benchmarks: {len(bundle.split.validation_questions)}")
    typer.echo(f"Test benchmarks: {len(bundle.split.test_questions)}")
    if bundle.split.validation_strategy == "small_sample_kfold":
        typer.echo(f"Validation strategy: grouped {len(bundle.split.validation_folds)}-fold over train+validation")
    else:
        typer.echo("Validation strategy: blind split")
    typer.echo(f"Benchmark source: {bundle.benchmark_source}")
    typer.echo(f"Task file: {bundle.prompt_path}")


@app.command()
def clone(
    space_url: str = typer.Option(
        None,
        "--space-url",
        help="Optional Genie Space URL. Falls back to MAXGENIE_DEFAULT_SPACE_URL when --space-id is omitted.",
    ),
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
    warehouse_id: str = typer.Option(None, "--warehouse-id", help="Warehouse id override when the local source snapshot does not include it"),
):
    """Create or reuse the single optimization clone for this workspace."""

    host = None
    source_space_id = space_id

    try:
        ref = _resolved_space_ref(space_url, required=source_space_id is None)
        if ref is not None:
            host = ref.host
            source_space_id = ref.space_id
        artifacts = _workspace_from_inputs(workspace_root, source_space_id)
        split_meta = _load_split_meta(artifacts)
        source_space_id = source_space_id or str(split_meta.get("source_space_id") or split_meta.get("space_id"))
        host = host or split_meta.get("workspace_host")
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=host,
            delay=12.0,
            match_threshold=0.85,
            warehouse_id=None,
        )
        try:
            source_config = flow.genie_service.get(source_space_id)[1]
            source_title = source_config.title
        except Exception:
            legacy_candidates = (
                artifacts.space_dir / "current.space.yml",
                artifacts.space_dir / "original.space.yml",
            )
            legacy_path = next((path for path in legacy_candidates if path.exists()), None)
            if legacy_path is None:
                raise WorkspaceFlowError(
                    f"Could not fetch source space `{source_space_id}` and no legacy local source YAML is available."
                )
            source_config = GenieSpaceConfig.from_yaml(str(legacy_path))
            source_title = source_config.title
        clone_info = flow.ensure_clone(
            source_space_id=source_space_id,
            source_config=source_config,
            source_title=source_title,
            input_space_id=source_space_id,
            fallback_warehouse_id=warehouse_id,
        )
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Source space: {source_space_id}")
    typer.echo(f"Clone space: {clone_info.space_id}")
    typer.echo(f"Clone title: {clone_info.title}")


@app.command()
def push(
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
):
    """Reassemble local assets and push them to the clone."""

    try:
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        split_meta = _load_split_meta(artifacts)
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=split_meta.get("workspace_host"),
            delay=12.0,
            match_threshold=0.85,
            warehouse_id=None,
        )
        info = flow.push()
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Pushed to clone: {info.space_id}")
    typer.echo(f"Title: {info.title}")


@app.command()
def curate(
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
    mode: str = typer.Option(
        CURATION_MODE_SAFE_DEFAULTS,
        "--mode",
        help="Curation pass: safe_defaults, descriptions, synonyms, entity_matching, format_assistance, matching, hide_noise, or all.",
    ),
):
    """Apply deterministic local curation to the decomposed Genie assets."""

    try:
        if mode not in SUPPORTED_CURATION_MODES:
            raise WorkspaceFlowError(
                "Unsupported curation mode. Choose one of: " + ", ".join(sorted(SUPPORTED_CURATION_MODES))
            )
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        flow = OptimizationWorkspace(
            genie_service=None,  # type: ignore[arg-type]
            benchmark_service=None,  # type: ignore[arg-type]
            artifacts=artifacts,
            exporter=None,  # type: ignore[arg-type]
            delay_seconds=0.0,
            match_threshold=0.85,
        )
        summary = flow.curate_workspace_mode(mode=mode)
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Curation mode: {summary.get('mode')}")
    typer.echo(f"Curation changed workspace: {'yes' if summary.get('changed') else 'no'}")
    typer.echo(f"Table descriptions added: {summary.get('table_descriptions_added', 0)}")
    typer.echo(f"Column descriptions added: {summary.get('column_descriptions_added', 0)}")
    typer.echo(f"Columns hidden: {summary.get('columns_hidden', 0)}")
    typer.echo(f"Entity-matching enabled: {summary.get('entity_matching_enabled', 0)}")
    typer.echo(f"Format assistance enabled: {summary.get('format_assistance_enabled', 0)}")
    typer.echo(f"Synonyms added: {summary.get('synonyms_added', 0)}")


@app.command("curate-loop")
def curate_loop(
    space_id: str = typer.Option(None, "--space-id", help="Space id if running outside workspace/<space_id>"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    modes: str = typer.Option(
        None,
        "--modes",
        help=(
            "Comma-separated ordered list of narrow curation modes to gate. "
            "Defaults to all narrow modes: descriptions,hide_noise,synonyms,"
            "entity_matching,format_assistance,matching"
        ),
    ),
    delay: float = typer.Option(12.0, "--delay", help="Delay between benchmark questions"),
    match_threshold: float = typer.Option(0.85, "--match-threshold", min=0.0, max=1.0),
    warehouse_id: str = typer.Option(None, "--warehouse-id", help="SQL warehouse id"),
    label: str = typer.Option(None, "--label", help="Run label prefix"),
):
    """Apply one manual benchmark-gated curation sweep over narrow modes.

    Each mode is applied independently. After each pass, train then validation benchmarks
    are run. A pass is accepted only when validation improves and train does not regress
    materially from the accepted reference state. Rejected passes are automatically reverted.
    """

    mode_list: list[str]
    if modes:
        mode_list = [m.strip() for m in modes.split(",") if m.strip()]
        bad = [m for m in mode_list if m not in SUPPORTED_CURATION_MODES]
        if bad:
            typer.secho(
                f"Unknown curation mode(s): {', '.join(bad)}. "
                "Choose from: " + ", ".join(sorted(SUPPORTED_CURATION_MODES)),
                fg=typer.colors.RED,
            )
            raise typer.Exit(2)
    else:
        mode_list = list(NARROW_CURATION_MODES)

    try:
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        split_meta = _load_split_meta(artifacts)
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=split_meta.get("workspace_host"),
            delay=delay,
            match_threshold=match_threshold,
            warehouse_id=warehouse_id,
        )
        resolved_label = label or datetime.now().strftime("curate_loop.%Y%m%dT%H%M%S")
        result = flow.curate_and_gate(modes=mode_list, label=resolved_label)
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    if not result.get("gating_active", True):
        typer.secho(
            "Warning: no baseline train or validation scores were found. "
            "All modes with changes were accepted without regression gating. "
            "Run `maxgenie benchmark --full` first to enable score-based gating.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"Modes tried: {', '.join(result['modes_tried'])}")
    typer.echo(f"Accepted: {result['accepted']}  Rejected: {result['rejected']}  Skipped: {result['skipped']}")
    for pass_outcome in result.get("passes", []):
        outcome = pass_outcome["outcome"]
        mode = pass_outcome["mode"]
        if outcome == "accepted":
            color = typer.colors.GREEN
            train_rate = pass_outcome.get("train_rate")
            validation_rate = pass_outcome.get("validation_rate", pass_outcome.get("dev_rate"))
            detail = f"train={train_rate:.1f}%  validation={validation_rate:.1f}%"
        elif outcome == "rejected":
            color = typer.colors.RED
            detail = pass_outcome.get("reason", "")
        else:
            color = typer.colors.YELLOW
            detail = pass_outcome.get("reason", "")
        typer.secho(f"  [{outcome.upper():8s}] {mode}: {detail}", fg=color)


@app.command()
def benchmark(
    train: bool = typer.Option(False, "--train", help="Run the train split with failures first"),
    validation: bool = typer.Option(
        False,
        "--validation",
        help="Run the blind validation split and store only score",
    ),
    dev: bool = typer.Option(False, "--dev", hidden=True),
    test: bool = typer.Option(False, "--test", help="Run the final blind test split and store only score"),
    full: bool = typer.Option(False, "--full", help="Run train, validation, and final test and store summary"),
    label: str = typer.Option(None, "--label", help="Optional run label"),
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
    delay: float = typer.Option(12.0, "--delay", help="Delay between benchmark questions"),
    match_threshold: float = typer.Option(0.85, "--match-threshold", min=0.0, max=1.0),
    warehouse_id: str = typer.Option(None, "--warehouse-id", help="SQL warehouse id for result-based comparison"),
):
    """Run benchmark evaluations against the optimization clone."""

    try:
        mode = _benchmark_mode(train, validation or dev, test, full)
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        split_meta = _load_split_meta(artifacts)
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=split_meta.get("workspace_host"),
            delay=delay,
            match_threshold=match_threshold,
            warehouse_id=warehouse_id,
        )
        resolved_label = label or datetime.now().strftime(f"{mode}.%Y%m%dT%H%M%S")
        if mode == "train":
            run = flow.run_train_benchmark(label=resolved_label)
            typer.echo(json.dumps(run.to_dict()["summary"], indent=2))
        elif mode == "validation":
            summary = flow.run_validation_benchmark(label=resolved_label)
            typer.echo(f"{summary['pass_rate']:.1f}")
        elif mode == "test":
            summary = flow.run_test_benchmark(label=resolved_label)
            typer.echo(f"{summary['pass_rate']:.1f}")
        else:
            summary = flow.run_full_benchmark(label=resolved_label)
            typer.echo(json.dumps(summary, indent=2))
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)


@app.command()
def status(
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
):
    """Show workspace status, clone health, and latest scores."""

    try:
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        split_meta = _load_split_meta(artifacts)
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=split_meta.get("workspace_host"),
            delay=12.0,
            match_threshold=0.85,
            warehouse_id=None,
        )
        workspace_status = flow.status()
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Input space: {workspace_status.input_space_id or 'unknown'}")
    typer.echo(f"Managed source space: {workspace_status.managed_source_space_id or 'unknown'}")
    typer.echo(f"Source space: {workspace_status.source_space_id or 'unknown'}")
    typer.echo(f"Clone space: {workspace_status.clone_space_id or 'missing'}")
    typer.echo(f"Clone accessible: {'yes' if workspace_status.clone_accessible else 'no'}")
    if workspace_status.git_commit_sha is not None:
        typer.echo(f"Git commit: {workspace_status.git_commit_sha}")
    if workspace_status.git_dirty is not None:
        typer.echo(f"Git dirty: {'yes' if workspace_status.git_dirty else 'no'}")
    if workspace_status.autonomous_strategy is not None:
        typer.echo(f"Autonomous strategy: {workspace_status.autonomous_strategy}")
    typer.echo(f"Baseline train pass rate: {_format_optional_rate(workspace_status.baseline_train_pass_rate)}")
    typer.echo(f"Latest train pass rate: {_format_optional_rate(workspace_status.latest_train_pass_rate)}")
    typer.echo(
        "Latest validation pass rate: "
        f"{_format_optional_rate(workspace_status.latest_validation_pass_rate)}"
    )
    typer.echo(f"Latest test pass rate: {_format_optional_rate(workspace_status.latest_test_pass_rate)}")
    typer.echo(f"Latest full pass rate: {_format_optional_rate(workspace_status.latest_full_pass_rate)}")
    if workspace_status.external_table_reference_count is not None:
        typer.echo(f"External table reference warnings: {workspace_status.external_table_reference_count}")
    if workspace_status.entity_matching_candidate_count is not None:
        typer.echo(f"Entity-matching candidates still disabled: {workspace_status.entity_matching_candidate_count}")
    if workspace_status.format_assistance_candidate_count is not None:
        typer.echo(f"Format-assistance candidates still disabled: {workspace_status.format_assistance_candidate_count}")
    if workspace_status.described_column_coverage_percent is not None:
        typer.echo(f"Column description coverage: {workspace_status.described_column_coverage_percent:.2f}%")
    if workspace_status.curation_candidate_status is not None:
        typer.echo(f"Curation candidate status: {workspace_status.curation_candidate_status}")
    if workspace_status.curation_change_count is not None:
        typer.echo(f"Curation changes prepared locally: {workspace_status.curation_change_count}")
    if workspace_status.hidden_column_count is not None:
        typer.echo(f"Columns hidden as noise: {workspace_status.hidden_column_count}")
    if workspace_status.query_history_status is not None:
        typer.echo(f"Query-history status: {workspace_status.query_history_status}")
    if workspace_status.query_history_query_count is not None:
        typer.echo(f"Query-history rows collected: {workspace_status.query_history_query_count}")
    if workspace_status.query_history_feedback_thumbs_down is not None:
        typer.echo(f"Query-history thumbs-down signals: {workspace_status.query_history_feedback_thumbs_down}")
    if workspace_status.query_history_trusted_asset_candidates is not None:
        typer.echo(
            f"Query-history trusted-asset candidates: {workspace_status.query_history_trusted_asset_candidates}"
        )
    if workspace_status.query_history_benchmark_candidates is not None:
        typer.echo(
            f"Query-history benchmark candidates: {workspace_status.query_history_benchmark_candidates}"
        )
    if workspace_status.last_stop_reason is not None:
        typer.echo(f"Last stop reason: {workspace_status.last_stop_reason}")
    if workspace_status.blocked_candidate_ids:
        typer.echo(f"Repeated unresolved failures: {', '.join(workspace_status.blocked_candidate_ids)}")


@app.command("query-history")
def query_history_command(
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
    warehouse_id: str = typer.Option(None, "--warehouse-id", help="SQL warehouse id override"),
    days: int = typer.Option(30, "--days", min=1, max=365),
    limit: int = typer.Option(200, "--limit", min=1, max=1000),
):
    """Collect Genie-space query history and feedback signals into workspace artifacts."""

    try:
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        split_meta = _load_split_meta(artifacts)
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=split_meta.get("workspace_host"),
            delay=12.0,
            match_threshold=0.85,
            warehouse_id=None,
        )
        summary = flow.collect_query_history(
            label=datetime.now().strftime("query-history.%Y%m%dT%H%M%S"),
            warehouse_id=warehouse_id,
            days=days,
            limit=limit,
        )
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Status: {summary.get('status')}")
    typer.echo(f"Warehouse: {summary.get('warehouse_id') or 'n/a'}")
    typer.echo(f"Queries collected: {summary.get('query_count', 0)}")
    typer.echo(f"Successful queries: {summary.get('successful_query_count', 0)}")
    typer.echo(f"Failed queries: {summary.get('failed_query_count', 0)}")
    typer.echo(f"Thumbs up: {summary.get('feedback_thumbs_up', 0)}")
    typer.echo(f"Thumbs down: {summary.get('feedback_thumbs_down', 0)}")
    if summary.get("message"):
        typer.echo(f"Note: {summary.get('message')}")
    typer.echo(f"Summary artifact: {artifacts.query_history_summary_path}")
    typer.echo(f"Recommendations artifact: {artifacts.query_history_recommendations_path}")


@app.command("ingest-history")
def ingest_history_command(
    space_url: str = typer.Option(None, "--space-url", help="Genie Space URL (used for workspace resolution only)."),
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>."),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root."),
    out: str = typer.Option(None, "--out", help="Override output path for expansion_candidates.json."),
):
    """Convert query-history recommendations into a human-reviewable benchmark expansion-candidates file."""

    try:
        resolved_space_ref = _resolved_space_ref(space_url, required=False)
        resolved_space_id = space_id or (resolved_space_ref.space_id if resolved_space_ref else None)
        artifacts = _workspace_from_inputs(workspace_root, resolved_space_id)
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    rec_path = artifacts.query_history_recommendations_path
    if not rec_path.exists():
        typer.secho(
            f"No query_history_recommendations.json in {artifacts.results_dir}. "
            "Run `maxgenie optimize` or `maxgenie benchmark` against a Genie space with conversation history first.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)

    try:
        recommendations: dict[str, Any] = artifacts.read_json_path(rec_path)
    except Exception as exc:
        typer.secho(f"Failed to read recommendations file: {exc}", fg=typer.colors.RED)
        raise typer.Exit(2)

    if not isinstance(recommendations, dict):
        typer.secho("recommendations file has unexpected format (expected a JSON object).", fg=typer.colors.RED)
        raise typer.Exit(2)

    recommendations = {**recommendations, "_source_path": str(rec_path)}
    payload = build_expansion_candidates(recommendations, now_iso=datetime.now(timezone.utc).isoformat())

    if out:
        output_path = Path(out).expanduser().resolve()
    else:
        output_path = artifacts.benchmarks_dir / "expansion_candidates.json"

    artifacts.write_json_path(output_path, payload)

    count = payload["candidate_count"]
    typer.echo(f"{count} candidates written to {output_path}. Review and append to train_set.json as needed.")


def run_status_summary_payload(
    *,
    run_id: str,
    space_url: str | None = None,
    profile: str | None = None,
    stale_after_minutes: int = 60,
    include_observed_ui: bool = False,
) -> dict[str, Any]:
    """Fetch and summarize a submitted optimizer job run."""
    ref = _resolved_space_ref(space_url, required=False)
    wc = create_workspace_client(profile=profile, host=ref.host if ref else None)
    with temporary_workspace_api_timeouts(
        wc,
        request_timeout_seconds=5,
        retry_timeout_seconds=15,
    ):
        status_payload = get_run_status(wc, run_id=run_id)
        summary = summarize_run_status(
            status_payload,
            stale_after_seconds=stale_after_minutes * 60,
        )
        progress_summary = run_progress_summary(wc, status_payload)
        if progress_summary is not None:
            summary["progress_summary"] = progress_summary
        if str(summary.get("life_cycle_state") or "") in {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}:
            artifact_summary = run_artifact_summary(
                wc,
                status_payload,
                include_observed_ui=include_observed_ui,
            )
            if artifact_summary is not None:
                summary["artifact_summary"] = artifact_summary
    return summary


def _status_summary_with_progress(
    wc: Any,
    payload: dict[str, Any],
    *,
    stale_after_seconds: int,
    include_artifacts: bool = True,
    include_observed_ui: bool = False,
) -> dict[str, Any]:
    summary = summarize_run_status(payload, stale_after_seconds=stale_after_seconds)
    progress_summary = run_progress_summary(wc, payload)
    if progress_summary is not None:
        summary["progress_summary"] = progress_summary
    if include_artifacts:
        artifact_summary = run_artifact_summary(
            wc,
            payload,
            include_observed_ui=include_observed_ui,
        )
        if artifact_summary is not None:
            summary["artifact_summary"] = artifact_summary
    return summary


def _monitor_fingerprint(summary: dict[str, Any]) -> str:
    progress = summary.get("progress_summary")
    progress_fingerprint = progress.get("fingerprint") if isinstance(progress, dict) else None
    return "|".join(
        str(value or "")
        for value in (
            summary.get("life_cycle_state"),
            summary.get("result_state"),
            summary.get("state_message"),
            progress_fingerprint,
        )
    )


@app.command()
def report(
    space_id: str = typer.Option(None, "--space-id", help="Workspace space id if running outside workspace/<space_id>"),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str = typer.Option("./workspace", "--workspace-root", help="Local workspace root"),
):
    """Generate the final optimization report for this workspace."""

    try:
        artifacts = _workspace_from_inputs(workspace_root, space_id)
        split_meta = _load_split_meta(artifacts)
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=split_meta.get("workspace_host"),
            delay=12.0,
            match_threshold=0.85,
            warehouse_id=None,
        )
        report_path = flow.write_report()
    except WorkspaceFlowError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Report written: {report_path}")


@app.command("submit-run")
def submit_run_command(
    space_url: str = typer.Option(
        None,
        "--space-url",
        help="Genie Space URL. Falls back to MAXGENIE_DEFAULT_SPACE_URL if unset.",
    ),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile for submitting the job"),
    runner_path: str | None = typer.Option(
        None,
        "--runner-path",
        help="Workspace path to the synced skill runner. Defaults to the current user's skill path.",
    ),
    skill_scope: str = typer.Option(
        "user",
        "--skill-scope",
        help="Default runner path scope when --runner-path is omitted: user or workspace.",
    ),
    workspace_root: str | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace-files run root. Defaults to the current user's conversations folder.",
    ),
    strategy: str = typer.Option(
        "serving",
        "--strategy",
        help="Optimizer strategy for the job run. Use serving for workspace serving-backed candidates.",
    ),
    run_name: str = typer.Option("Space Optimizer Long Run", "--run-name"),
    use_latest_observed_benchmark: bool = typer.Option(
        True,
        "--use-latest-observed-benchmark/--fresh-baseline",
        help="Reuse the latest completed Genie benchmark eval run as the initial baseline.",
    ),
    fresh_lineage: bool = typer.Option(False, "--fresh-lineage"),
    bootstrap_only: bool = typer.Option(
        False,
        "--bootstrap-only",
        help="Submit a baseline-only canary job that stops before candidate generation.",
    ),
    max_iterations: int | None = typer.Option(
        None,
        "--max-iterations",
        min=1,
        help="Maximum autonomous sweeps. Omit for no sweep cap; plateau/timeout still stop the run.",
    ),
    plateau_rounds: int = typer.Option(3, "--plateau-rounds", min=1),
    candidate_budget: int | None = typer.Option(None, "--candidate-budget", min=1),
    delay_seconds: float = typer.Option(12.0, "--delay-seconds"),
    match_threshold: float = typer.Option(0.85, "--match-threshold", min=0.0, max=1.0),
    warehouse_id: str | None = typer.Option(None, "--warehouse-id"),
    clone_parent_path: str | None = typer.Option(
        None,
        "--clone-parent-path",
        help=(
            "Workspace folder for disposable Genie clones, such as /Users/<user> or /Shared/<team>. "
            "Defaults to the Databricks run-as user's home folder."
        ),
    ),
    serving_endpoint: str | None = typer.Option(
        None,
        "--serving-endpoint",
        help="Workspace serving endpoint for --strategy serving. Falls back to MAXGENIE_SERVING_ENDPOINT.",
    ),
    serving_model: str | None = typer.Option(
        None,
        "--serving-model",
        help="Optional model name for the serving request. Falls back to MAXGENIE_SERVING_MODEL.",
    ),
    serving_temperature: float | None = typer.Option(None, "--serving-temperature", min=0.0, max=2.0),
    serving_max_tokens: int = typer.Option(16000, "--serving-max-tokens", min=512),
    serving_reasoning_effort: str | None = typer.Option(None, "--serving-reasoning-effort"),
    serving_response_format: str = typer.Option("json_schema", "--serving-response-format"),
    serving_api_format: str = typer.Option("auto", "--serving-api-format"),
    serving_candidate_mode: str | None = typer.Option(
        None,
        "--serving-candidate-mode",
        help=(
            "Force one serving candidate mode for every serving iteration, such as "
            "artifact_patch_v2 or skeleton_artifact_patch_v2. "
            "Intended for controlled validation lanes."
        ),
    ),
    observed_benchmark_seed: str | None = typer.Option(
        None,
        "--observed-benchmark-seed",
        help=(
            "Workspace or local JSON seed with already-observed benchmark results to use "
            "as the initial baseline without rerunning all source benchmarks."
        ),
    ),
    fixed_benchmark_artifacts_path: str | None = typer.Option(
        None,
        "--fixed-benchmark-artifacts-path",
        help=(
            "Workspace or local run-root path containing frozen benchmark payload and split artifacts "
            "to reuse exactly for comparable tournament lanes."
        ),
    ),
    k_samples: int = typer.Option(
        1,
        "--k-samples",
        min=1,
        max=10,
        help=(
            "Repeat the population baseline benchmark this many times inside the job and aggregate. "
            "K=1 (default) preserves single-sample behaviour; K>1 shrinks promotion decision noise by ~sqrt(K)."
        ),
    ),
    noise_sigma_full: float | None = typer.Option(
        None,
        "--noise-sigma-full",
        min=0.0,
        help=(
            "Measured full-score run-to-run noise (sigma) from measure-noise. When set, terminal "
            "selection in the job promotes only when the final full mean lift exceeds the MDD (2*sigma)."
        ),
    ),
    noise_baseline_n: int | None = typer.Option(
        None,
        "--noise-baseline-n",
        min=1,
        help=(
            "Number of repeats behind the measured baseline mean; used with --noise-sigma-full for the "
            "two-sample promotion threshold. Omit to treat the baseline as a fixed reference (single-run MDD)."
        ),
    ),
    adaptive_recipes: bool = typer.Option(
        False,
        "--adaptive-recipes/--no-adaptive-recipes",
        help=(
            "Let fresh (non-recovery) serving iterations pick the recipe mode that best fits the dominant "
            "visible failure family instead of always using artifact_patch_v2. Off by default."
        ),
    ),
    greedy_best_ever: bool = typer.Option(
        True,
        "--greedy-best-ever/--no-greedy-best-ever",
        help=(
            "Greedy best-ever promotion in the submitted run: promote any strict improvement over the "
            "run baseline (no minimum detectable difference / multi-sample gate). Enabled by default "
            "from the Stage 1 tournament winner; use --no-greedy-best-ever for MDD-only selection. "
            "The holdout non-regression and protocol-comparable gates still apply."
        ),
    ),
    confirm_on_beat: int = typer.Option(
        0,
        "--confirm-on-beat",
        min=0,
        help=(
            "Optional anti-noise insurance for the submitted run: re-run an accepted candidate's visible "
            "benchmark this many times and reject if any re-run regresses the visible reference."
        ),
    ),
    content_match: bool = typer.Option(
        False,
        "--content-match/--no-content-match",
        help=(
            "Score the submitted run with the value-first content comparator instead of the "
            "legacy byte-exact column-set comparator. The new comparability epoch: re-baseline "
            "under it. Requires a SQL warehouse."
        ),
    ),
    audit_checkpoint_path: str | None = typer.Option(
        None,
        "--audit-checkpoint-path",
        help=(
            "Checkpoint path to restore and audit in the submitted run. "
            "When set, the run skips candidate generation."
        ),
    ),
    audit_patch_path: str | None = typer.Option(
        None,
        "--audit-patch-path",
        help=(
            "Artifact patch sequence path to replay and audit in the submitted run. "
            "When set, the run skips candidate generation."
        ),
    ),
    positive_control: str | None = typer.Option(
        None,
        "--positive-control",
        help=(
            "Resolve a named or JSON positive control locally, upload its checkpoint or patch "
            "sequence to the workspace run root, and submit an audit job."
        ),
    ),
    timeout_seconds: int = typer.Option(14400, "--timeout-seconds", min=300),
    compute_mode: str = typer.Option(
        "serverless",
        "--compute-mode",
        help="serverless, existing_cluster, or new_cluster.",
    ),
    existing_cluster_id: str | None = typer.Option(None, "--existing-cluster-id"),
    new_cluster_json: str | None = typer.Option(
        None,
        "--new-cluster-json",
        help="New-cluster JSON object or path when --compute-mode new_cluster is used.",
    ),
    serverless_environment_version: str = typer.Option("2", "--serverless-environment-version"),
    install_dependencies: bool = typer.Option(
        True,
        "--install-dependencies/--no-install-dependencies",
        help="Install the optimizer runtime dependencies on the job task.",
    ),
    idempotency_token: str | None = typer.Option(None, "--idempotency-token"),
    require_production_protocol: bool = typer.Option(
        True,
        "--require-production-protocol/--allow-diagnostic",
        help=(
            "Require MaxGenie's production-comparable protocol preflight before submission. "
            "Use --allow-diagnostic for exploratory runs that are not promotion evidence."
        ),
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    wait: bool = typer.Option(False, "--wait", help="Poll until the submitted run finishes."),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds", min=5.0),
):
    """Submit an unattended Databricks job run for a synced MaxGenie skill."""
    try:
        ref = _resolved_space_ref(space_url, required=True)
        try:
            resolved_strategy = normalize_autonomous_strategy(strategy)
        except ValueError as exc:
            raise WorkspaceFlowError(str(exc)) from exc
        wc = create_workspace_client(profile=profile, host=ref.host)
        if skill_scope not in {"user", "workspace"}:
            raise WorkspaceFlowError("--skill-scope must be user or workspace.")
        resolved_runner_path = runner_path or default_runner_path(wc, skill_scope=skill_scope)  # type: ignore[arg-type]
        user_name = getattr(wc.current_user.me(), "user_name", None)
        resolved_workspace_root = workspace_root or (
            f"/Workspace/Users/{user_name}/.maxgenie/runs/job"
            if user_name
            else "/Workspace/.maxgenie/runs/job"
        )
        resolved_audit_checkpoint_path = audit_checkpoint_path
        resolved_audit_patch_path = audit_patch_path
        audit_modes = sum(
            1
            for value in (positive_control, audit_checkpoint_path, audit_patch_path)
            if value
        )
        if audit_modes > 1:
            raise WorkspaceFlowError(
                "Use only one of --positive-control, --audit-checkpoint-path, or --audit-patch-path."
            )
        if positive_control:
            resolved_control = resolve_positive_control(positive_control, base_dir=Path.cwd())
            control_summary = resolved_control.to_summary()
            if not resolved_control.available:
                raise WorkspaceFlowError(
                    f"Positive control `{positive_control}` is unavailable: "
                    + ", ".join(control_summary.get("missing_paths", []))
                )
            if resolved_control.audit_target_kind == "checkpoint" and resolved_control.checkpoint_path is not None:
                resolved_audit_checkpoint_path = (
                    f"{resolved_workspace_root.rstrip('/')}/positive_controls/"
                    f"{resolved_control.spec.name}/{resolved_control.checkpoint_path.name}"
                )
                if not dry_run:
                    upload_directory_to_workspace_files(
                        wc,
                        source_dir=resolved_control.checkpoint_path,
                        destination_dir=resolved_audit_checkpoint_path,
                        overwrite=True,
                    )
            elif resolved_control.audit_target_kind == "patch" and resolved_control.patch_path is not None:
                resolved_audit_patch_path = (
                    f"{resolved_workspace_root.rstrip('/')}/positive_controls/"
                    f"{resolved_control.spec.name}/patch"
                )
                with tempfile.TemporaryDirectory(prefix="maxgenie-positive-control-") as tmpdir:
                    local_patch_dir = materialize_positive_control_patch_sequence(
                        resolved_control,
                        destination_dir=Path(tmpdir) / "patch",
                    )
                    if not dry_run:
                        upload_directory_to_workspace_files(
                            wc,
                            source_dir=local_patch_dir,
                            destination_dir=resolved_audit_patch_path,
                            overwrite=True,
                        )
            else:
                raise WorkspaceFlowError(
                    f"Positive control `{positive_control}` does not resolve to an auditable target."
                )
        resolved_serving_temperature = (
            serving_temperature
            if serving_temperature is not None
            else default_serving_temperature()
        )
        resolved_warehouse_id = warehouse_id or discover_best_warehouse_id(wc)
        resolved_serving_endpoint, resolved_serving_model = _resolve_serving_inputs_for_submit(
            wc,
            strategy=resolved_strategy,
            serving_endpoint=serving_endpoint,
            serving_model=serving_model,
        )
        settings = WorkspaceJobSettings(
            space_url=ref.url,
            runner_path=resolved_runner_path,
            workspace_root=resolved_workspace_root,
            strategy=resolved_strategy,
            run_name=run_name,
            use_latest_observed_benchmark=use_latest_observed_benchmark,
            fresh_lineage=fresh_lineage,
            bootstrap_only=bootstrap_only,
            max_iterations=max_iterations,
            plateau_rounds=plateau_rounds,
            candidate_budget=candidate_budget,
            delay_seconds=delay_seconds,
            match_threshold=match_threshold,
            warehouse_id=resolved_warehouse_id,
            clone_parent_path=clone_parent_path,
            serving_endpoint=resolved_serving_endpoint,
            serving_model=resolved_serving_model,
            serving_temperature=resolved_serving_temperature,
            serving_max_tokens=serving_max_tokens,
            serving_reasoning_effort=(
                serving_reasoning_effort
                if serving_reasoning_effort is not None
                else default_serving_reasoning_effort()
            ),
            serving_response_format=serving_response_format,
            serving_api_format=serving_api_format,
            serving_candidate_mode=serving_candidate_mode,
            observed_benchmark_seed=observed_benchmark_seed,
            fixed_benchmark_artifacts_path=fixed_benchmark_artifacts_path,
            k_samples=k_samples,
            noise_sigma_full=noise_sigma_full,
            noise_baseline_n=noise_baseline_n,
            adaptive_recipes=adaptive_recipes,
            greedy_best_ever=greedy_best_ever,
            confirm_on_beat=confirm_on_beat,
            result_content_match=bool(content_match),
            audit_checkpoint_path=resolved_audit_checkpoint_path,
            audit_patch_path=resolved_audit_patch_path,
            timeout_seconds=timeout_seconds,
            compute_mode=compute_mode,  # type: ignore[arg-type]
            existing_cluster_id=existing_cluster_id,
            new_cluster=parse_new_cluster_json(new_cluster_json),
            serverless_environment_version=serverless_environment_version,
            install_dependencies=install_dependencies,
            idempotency_token=idempotency_token,
        )
        if require_production_protocol:
            preflight = validate_protocol_preflight(
                {
                    "mode": (
                        "checkpoint_audit"
                        if resolved_audit_checkpoint_path
                        else (
                            "patch_replay_audit"
                            if resolved_audit_patch_path
                            else "autonomous"
                        )
                    ),
                    "strategy": resolved_strategy,
                    "comparison_mode": "result_based" if resolved_warehouse_id else "string_only",
                    "warehouse_id": resolved_warehouse_id,
                    "benchmark_coverage": "15/15",
                    "split_strategy": "canonical",
                    "fresh_baseline": not (
                        observed_benchmark_seed is not None or use_latest_observed_benchmark
                    ),
                    "serving_endpoint": settings.serving_endpoint,
                    "serving_model": settings.serving_model,
                    "serving_reasoning_effort": settings.serving_reasoning_effort,
                    "observed_benchmark_reuse_policy": (
                        "explicit_observed_benchmark_seed"
                        if observed_benchmark_seed is not None
                        else (
                            "latest_observed_initial_baseline"
                            if use_latest_observed_benchmark
                            else "fresh_baseline_benchmark_run"
                        )
                    ),
                }
            )
            if not preflight.comparable:
                reasons = ", ".join(issue.code for issue in preflight.errors)
                raise WorkspaceFlowError(
                    "Production protocol preflight failed: "
                    f"{reasons}. Re-run with --allow-diagnostic to submit an exploratory run "
                    "that is not promotion evidence."
                )
        if dry_run:
            typer.echo(json.dumps(build_runs_submit_payload(settings), indent=2, ensure_ascii=False))
            return
        submitted = submit_workspace_run(wc, settings=settings)
    except (AuthConfigurationError, WorkspaceFlowError, ValueError, json.JSONDecodeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    if submitted.get("run_page_url"):
        print(str(submitted["run_page_url"]), flush=True)
        print(f"Run page: {submitted['run_page_url']}", flush=True)
    print(
        "Final report (when complete): "
        f"{settings.workspace_root.rstrip('/')}/{ref.space_id}/final_report.ipynb",
        flush=True,
    )
    print(f"Submitted run: {submitted.get('run_id')}", flush=True)
    if submitted.get("status_error"):
        typer.secho(
            f"Initial status check failed: {submitted['status_error']}",
            fg=typer.colors.YELLOW,
        )
    status_summary = submitted.get("status_summary")
    if isinstance(status_summary, dict):
        lifecycle = status_summary.get("life_cycle_state") or "UNKNOWN"
        result_state = status_summary.get("result_state")
        state_text = str(lifecycle) + (f" / {result_state}" if result_state else "")
        print(f"Run state: {state_text}", flush=True)
    if wait and submitted.get("run_id"):
        last_monitor_fingerprint: str | None = None
        last_monitor_printed_at = 0.0
        last_monitor_summary: dict[str, Any] | None = None
        heartbeat_seconds = max(60.0, poll_seconds * 4)

        def print_status_tick(status_payload: dict[str, Any]) -> None:
            nonlocal last_monitor_fingerprint, last_monitor_printed_at, last_monitor_summary
            summary = _status_summary_with_progress(
                wc,
                status_payload,
                stale_after_seconds=timeout_seconds,
                include_artifacts=False,
            )
            last_monitor_summary = summary
            fingerprint = _monitor_fingerprint(summary)
            now = time.monotonic()
            if (
                fingerprint == last_monitor_fingerprint
                and now - last_monitor_printed_at < heartbeat_seconds
            ):
                return
            last_monitor_fingerprint = fingerprint
            last_monitor_printed_at = now
            print("", flush=True)
            print(format_run_status_for_chat(summary), flush=True)

        try:
            final_status = wait_for_run(
                wc,
                run_id=submitted["run_id"],
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
                on_tick=print_status_tick,
            )
        except TimeoutError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            if last_monitor_summary is not None:
                print("", flush=True)
                print("Last known run status before timeout:", flush=True)
                print(format_run_status_for_chat(last_monitor_summary), flush=True)
            raise typer.Exit(2)
        except DatabricksError as exc:
            typer.secho(
                _databricks_api_error_message(
                    f"polling submitted run {submitted['run_id']}", exc
                ),
                fg=typer.colors.RED,
            )
            if last_monitor_summary is not None:
                print("", flush=True)
                print("Last known run status before polling failed:", flush=True)
                print(format_run_status_for_chat(last_monitor_summary), flush=True)
            raise typer.Exit(2)
        final_summary = _status_summary_with_progress(
            wc,
            final_status,
            stale_after_seconds=timeout_seconds,
        )
        print("", flush=True)
        print(format_run_status_for_chat(final_summary), flush=True)


@app.command("run-status")
def run_status_command(
    run_id: str = typer.Argument(..., help="Databricks job run id"),
    space_url: str = typer.Option(
        None,
        "--space-url",
        help="Optional Genie Space URL used only to select the matching workspace host/profile.",
    ),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print a compact monitor summary instead of raw JSON.",
    ),
    chat: bool = typer.Option(
        False,
        "--chat",
        help="Print a compact Markdown status update for Genie Code chat.",
    ),
    stale_after_minutes: int = typer.Option(
        60,
        "--stale-after-minutes",
        min=1,
        help="Mark active runs as stale after this many minutes in summary output.",
    ),
    include_observed_ui: bool = typer.Option(
        False,
        "--include-observed-ui",
        help="Include the latest Genie UI eval summary; off by default so status stays fast.",
    ),
):
    """Show the status for a submitted optimizer job run."""
    try:
        if summary or chat:
            summary_payload = run_status_summary_payload(
                run_id=run_id,
                space_url=space_url,
                profile=profile,
                stale_after_minutes=stale_after_minutes,
                include_observed_ui=include_observed_ui,
            )
            if chat:
                typer.echo(format_run_status_for_chat(summary_payload))
            else:
                typer.echo(json.dumps(summary_payload, indent=2, ensure_ascii=False))
            return
        ref = _resolved_space_ref(space_url, required=False)
        wc = create_workspace_client(profile=profile, host=ref.host if ref else None)
        status_payload = get_run_status(wc, run_id=run_id)
    except (AuthConfigurationError, WorkspaceFlowError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)
    except DatabricksError as exc:
        typer.secho(
            _databricks_api_error_message(f"fetching run {run_id} status", exc),
            fg=typer.colors.RED,
        )
        raise typer.Exit(2)
    typer.echo(json.dumps(status_payload, indent=2, ensure_ascii=False))


@app.command("cancel-run")
def cancel_run_command(
    run_id: str = typer.Argument(..., help="Databricks job run id"),
    space_url: str = typer.Option(
        None,
        "--space-url",
        help="Optional Genie Space URL used only to select the matching workspace host/profile.",
    ),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
):
    """Cancel a submitted optimizer job run."""
    try:
        ref = _resolved_space_ref(space_url, required=False)
        wc = create_workspace_client(profile=profile, host=ref.host if ref else None)
        response = cancel_run(wc, run_id=run_id)
    except (AuthConfigurationError, WorkspaceFlowError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)
    typer.echo(json.dumps(response, indent=2, ensure_ascii=False))


def _extract_sample_aggregate(payload: object, *, source: str) -> dict:
    """Pull a SampleAggregate-shaped dict (mean_pass_rate/sample_std/n) out of JSON.

    Accepts three shapes so the same file an ``optimize`` run already writes can be
    fed back in directly:
      * a bare ``SampleAggregate.as_dict()`` (has ``mean_pass_rate``),
      * a full-benchmark history snapshot carrying a ``multi_sample`` block,
      * an iteration record wrapping the aggregate under ``aggregate``.
    """
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{source}: expected a JSON object, got {type(payload).__name__}")
    for key in ("multi_sample", "aggregate"):
        nested = payload.get(key)
        if isinstance(nested, dict) and "mean_pass_rate" in nested:
            return nested
    if "mean_pass_rate" in payload:
        return payload
    raise typer.BadParameter(
        f"{source}: no multi-sample aggregate found "
        "(expected 'mean_pass_rate' at top level, or a 'multi_sample'/'aggregate' block)"
    )


@app.command("measure-noise")
def measure_noise_command(
    aggregate: list[str] = typer.Option(
        None,
        "--aggregate",
        "-a",
        help=(
            "Measured subject as name=PATH, where PATH is a JSON file holding a "
            "multi-sample aggregate (the 'multi_sample' block an `optimize "
            "--k-samples K` run writes into its full-benchmark snapshot, or a bare "
            "SampleAggregate dict). Repeat per subject, e.g. "
            "-a baseline=baseline.json -a incumbent=incumbent.json."
        ),
    ),
    output: Path = typer.Option(
        Path("noise_model.json"),
        "--output",
        "-o",
        help="Where to write the consolidated noise model JSON.",
    ),
    z: float = typer.Option(
        2.0,
        "--z",
        min=1.0,
        max=4.0,
        help="Sigma multiplier for the minimum detectable difference (default 2 ~= 95%).",
    ),
):
    """Characterise run-to-run benchmark noise into a reusable noise model.

    Consumes the multi-sample aggregates produced by `optimize --k-samples K`
    (one or more subjects -- typically a no-change baseline and an incumbent
    checkpoint audit) and writes a noise_model.json carrying sigma_hat_full, the
    minimum detectable difference, and the promotion predicate. Feed the result
    back via `optimize --noise-sigma-full <sigma>` to gate promotion on real
    signal instead of a fixed epsilon.
    """
    if not aggregate:
        typer.secho("Provide at least one --aggregate name=PATH.", fg=typer.colors.RED)
        raise typer.Exit(2)

    subjects: dict[str, dict] = {}
    for spec in aggregate:
        name, sep, path_str = spec.partition("=")
        name = name.strip()
        if not sep or not name or not path_str.strip():
            typer.secho(f"Invalid --aggregate '{spec}'; expected name=PATH.", fg=typer.colors.RED)
            raise typer.Exit(2)
        path = Path(path_str.strip())
        if not path.exists():
            typer.secho(f"Aggregate file not found for '{name}': {path}", fg=typer.colors.RED)
            raise typer.Exit(2)
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            typer.secho(f"Could not parse JSON for '{name}' ({path}): {exc}", fg=typer.colors.RED)
            raise typer.Exit(2)
        subjects[name] = _extract_sample_aggregate(payload, source=f"{name} ({path})")

    model = build_noise_model(subjects, z=z)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False))
    typer.echo(json.dumps(model, indent=2, ensure_ascii=False))
    typer.secho(
        f"Noise model written to {output} "
        f"(sigma_hat_full={model['sigma_hat_full']}, mdd_full={model['mdd_full']}).",
        fg=typer.colors.GREEN,
    )


@app.command()
def optimize(
    space_url: str = typer.Option(
        None,
        "--space-url",
        help="Genie Space URL. Falls back to MAXGENIE_DEFAULT_SPACE_URL if unset.",
    ),
    profile: str = typer.Option(None, "--profile", "-p", help="Databricks profile"),
    workspace_root: str | None = typer.Option(
        None,
        "--workspace-root",
        help="Local workspace root. Defaults to a strategy-specific root when omitted.",
    ),
    validation_fraction: float = typer.Option(0.2, "--validation-fraction", min=0.0, max=0.5),
    test_fraction: float = typer.Option(0.2, "--test-fraction", min=0.0, max=0.5),
    split_seed: int = typer.Option(42, "--split-seed"),
    delay: float = typer.Option(12.0, "--delay", help="Delay between benchmark questions"),
    match_threshold: float = typer.Option(0.85, "--match-threshold", min=0.0, max=1.0),
    warehouse_id: str = typer.Option(None, "--warehouse-id", help="SQL warehouse id for result-based comparison"),
    clone_parent_path: str | None = typer.Option(
        None,
        "--clone-parent-path",
        help=(
            "Workspace folder for disposable Genie clones, such as /Users/<user> or /Shared/<team>. "
            "Defaults to the Databricks run-as user's home folder."
        ),
    ),
    bootstrap_only: bool = typer.Option(
        False,
        "--bootstrap-only",
        help="Stop after export, clone adoption, split balancing, and baseline benchmarking.",
    ),
    plateau_rounds: int = typer.Option(3, "--plateau-rounds", min=1, help="Stop after this many no-gain sweeps"),
    max_iterations: int | None = typer.Option(
        None,
        "--max-iterations",
        min=1,
        help="Maximum autonomous sweeps. Omit for no sweep cap; plateau/timeout still stop the run.",
    ),
    candidate_budget: int | None = typer.Option(None, "--candidate-budget", min=1, help="Maximum candidate attempts"),
    strategy: str = typer.Option(
        DEFAULT_AUTONOMOUS_STRATEGY,
        "--strategy",
        help=(
            "Autonomous proposal strategy: centaur (default, recommended)"
            " | serving."
        ),
    ),
    serving_endpoint: str | None = typer.Option(
        None,
        "--serving-endpoint",
        help="Workspace serving endpoint used by --strategy serving. Falls back to MAXGENIE_SERVING_ENDPOINT.",
    ),
    serving_model: str | None = typer.Option(
        None,
        "--serving-model",
        help="Optional model name for the serving request. Falls back to MAXGENIE_SERVING_MODEL.",
    ),
    serving_temperature: float | None = typer.Option(
        None,
        "--serving-temperature",
        min=0.0,
        max=2.0,
        help="Temperature for serving-backed candidate proposals. Omitted unless explicitly set.",
    ),
    serving_max_tokens: int = typer.Option(
        16000,
        "--serving-max-tokens",
        min=512,
        help="Maximum response tokens for serving-backed candidate proposals.",
    ),
    serving_reasoning_effort: str | None = typer.Option(
        None,
        "--serving-reasoning-effort",
        help="Reasoning effort for the serving request. Defaults to MAXGENIE_SERVING_REASONING_EFFORT or low.",
    ),
    serving_response_format: str = typer.Option(
        "json_schema",
        "--serving-response-format",
        help="Response format for serving-backed proposals: json_schema, json_object, or none.",
    ),
    serving_api_format: str = typer.Option(
        "auto",
        "--serving-api-format",
        help="Serving API format for proposals: auto, chat, or responses.",
    ),
    serving_candidate_mode: str | None = typer.Option(
        None,
        "--serving-candidate-mode",
        help=(
            "Force one serving candidate mode for every serving iteration, such as "
            "artifact_patch_v2 or skeleton_artifact_patch_v2. "
            "Intended for controlled validation lanes."
        ),
    ),
    greedy_best_ever: bool = typer.Option(
        True,
        "--greedy-best-ever/--no-greedy-best-ever",
        help=(
            "Greedy best-ever promotion: promote any strict improvement over the run "
            "baseline (no minimum detectable difference / multi-sample gate). Enabled by default "
            "from the Stage 1 tournament winner; use --no-greedy-best-ever for MDD-only selection. "
            "The holdout non-regression and protocol-comparable gates still apply."
        ),
    ),
    confirm_on_beat: int = typer.Option(
        0,
        "--confirm-on-beat",
        min=0,
        help=(
            "Optional anti-noise insurance: re-run an accepted candidate's visible "
            "benchmark this many times and reject if any re-run regresses the visible "
            "reference. 0 disables it."
        ),
    ),
    content_match: bool | None = typer.Option(
        None,
        "--content-match/--no-content-match",
        help=(
            "Score answers with the value-first content comparator instead of the legacy "
            "byte-exact column-set comparator. Omit to use the MAXGENIE_RESULT_CONTENT_MATCH "
            "env switch (default off). Requires a SQL warehouse."
        ),
    ),
    fixed_benchmark_artifacts_path: Path | None = typer.Option(
        None,
        "--fixed-benchmark-artifacts-path",
        help=(
            "Run-root path containing frozen benchmark payload and split artifacts "
            "to reuse exactly for comparable tournament lanes."
        ),
    ),
    k_samples: int = typer.Option(
        1,
        "--k-samples",
        min=1,
        max=10,
        help=(
            "Repeat the population baseline benchmark this many times and aggregate "
            "(mean, sample std, per-question pass fraction). K=1 (default) preserves "
            "single-sample behaviour; K>1 shrinks the promotion decision noise by ~sqrt(K)."
        ),
    ),
    noise_sigma_full: float | None = typer.Option(
        None,
        "--noise-sigma-full",
        min=0.0,
        help=(
            "Measured full-score run-to-run noise (sigma, in pass-rate points), e.g. from "
            "the measure-noise command. When set, terminal selection promotes only when the "
            "final full mean lift over baseline exceeds the minimum detectable difference "
            "(2*sigma), instead of a fixed epsilon."
        ),
    ),
    noise_baseline_n: int | None = typer.Option(
        None,
        "--noise-baseline-n",
        min=1,
        help=(
            "Number of repeats behind the measured baseline mean. Used with --noise-sigma-full "
            "to compute the two-sample promotion threshold; omit to treat the baseline as a "
            "fixed reference (single-run MDD)."
        ),
    ),
    adaptive_recipes: bool = typer.Option(
        False,
        "--adaptive-recipes/--no-adaptive-recipes",
        help=(
            "Let fresh (non-recovery) serving iterations pick the candidate recipe mode that "
            "best fits the dominant visible failure family (e.g. literal examples for "
            "unbound-parameter failures, time-window snippets for time-filter mismatches) "
            "instead of always using the strict artifact_patch_v2 default. Recovery iterations "
            "and ambiguous families still fall back to artifact_patch_v2, so the comparable "
            "measurement backbone is preserved. Off by default."
        ),
    ),
    fresh_lineage: bool = typer.Option(
        False,
        "--fresh-lineage",
        help="Ignore any existing MaxGenie-managed lineage and bootstrap from the input space again.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration, clone access, and baseline readiness without running optimization.",
    ),
    observed_benchmark_seed: Path | None = typer.Option(
        None,
        "--observed-benchmark-seed",
        help=(
            "JSON seed with already-observed benchmark questions/results from the workspace "
            "authoring runtime. When provided, MaxGenie uses it to balance splits and "
            "initialize baseline artifacts without rerunning the initial full benchmark."
        ),
    ),
    use_latest_observed_benchmark: bool = typer.Option(
        False,
        "--use-latest-observed-benchmark",
        help=(
            "Fetch the latest completed Genie benchmark eval run for the source space and "
            "use it as the initial baseline without rerunning all source benchmarks."
        ),
    ),
    audit_checkpoint_path: Path | None = typer.Option(
        None,
        "--audit-checkpoint-path",
        help=(
            "Restore this checkpoint into the optimization clone and run only the "
            "final benchmark/report audit. Skips candidate generation."
        ),
    ),
    audit_patch_path: Path | None = typer.Option(
        None,
        "--audit-patch-path",
        help=(
            "Replay this artifact patch sequence into the optimization clone and run only "
            "the final benchmark/report audit. Skips candidate generation."
        ),
    ),
    positive_control: str | None = typer.Option(
        None,
        "--positive-control",
        help=(
            "Resolve a named or JSON positive control and audit it instead of generating candidates."
        ),
    ),
):
    """Run the end-to-end MaxGenie optimization flow for one space lineage."""

    try:
        ref = _resolved_space_ref(space_url, required=True)
        try:
            resolved_strategy = normalize_autonomous_strategy(strategy)
        except ValueError as exc:
            raise WorkspaceFlowError(str(exc)) from exc
        resolved_workspace_root = _resolved_workspace_root(
            workspace_root,
            strategy=resolved_strategy,
        )
        resolved_serving_endpoint = serving_endpoint or default_serving_endpoint()
        resolved_serving_model = serving_model or default_serving_model()
        resolved_serving_temperature = (
            serving_temperature
            if serving_temperature is not None
            else default_serving_temperature()
        )
        resolved_serving_reasoning_effort = (
            serving_reasoning_effort
            if serving_reasoning_effort is not None
            else default_serving_reasoning_effort()
        )
        artifacts = ArtifactManager(
            workspace_dir=default_workspace_dir(resolved_workspace_root, ref.space_id),
            space_id=ref.space_id,
        )
        audit_modes = sum(
            1
            for value in (positive_control, audit_checkpoint_path, audit_patch_path)
            if value is not None
        )
        if audit_modes > 1:
            raise WorkspaceFlowError(
                "Use only one of --positive-control, --audit-checkpoint-path, or --audit-patch-path."
            )
        if positive_control:
            resolved_control = resolve_positive_control(positive_control, base_dir=Path.cwd())
            control_summary = resolved_control.to_summary()
            artifacts.write_json_path(
                artifacts.results_dir / "positive_control.json",
                control_summary,
            )
            artifacts.update_workspace_meta({"positive_control": control_summary})
            if not resolved_control.available:
                raise WorkspaceFlowError(
                    f"Positive control `{positive_control}` is unavailable: "
                    + ", ".join(control_summary.get("missing_paths", []))
                )
            if resolved_control.audit_target_kind == "checkpoint" and resolved_control.checkpoint_path is not None:
                audit_checkpoint_path = resolved_control.checkpoint_path
            elif resolved_control.audit_target_kind == "patch" and resolved_control.patch_path is not None:
                audit_patch_path = materialize_positive_control_patch_sequence(
                    resolved_control,
                    destination_dir=artifacts.results_dir / "positive_control_patch",
                )
                control_summary = {
                    **control_summary,
                    "materialized_patch_path": str(audit_patch_path),
                }
                artifacts.write_json_path(
                    artifacts.results_dir / "positive_control.json",
                    control_summary,
                )
                artifacts.update_workspace_meta({"positive_control": control_summary})
            else:
                raise WorkspaceFlowError(
                    f"Positive control `{positive_control}` does not resolve to an auditable target."
                )
        flow = _build_flow(
            artifacts=artifacts,
            profile=profile,
            host=ref.host,
            delay=delay,
            match_threshold=match_threshold,
            warehouse_id=warehouse_id,
            clone_parent_path=clone_parent_path,
            result_content_match=content_match,
        )
        # Wire the measured noise model into the terminal-selection gate. When
        # --noise-sigma-full is set, promotion uses the minimum detectable
        # difference rather than a fixed epsilon (apple-to-apple promotion).
        if noise_sigma_full is not None and noise_sigma_full > 0:
            flow._terminal_noise_sigma_override = float(noise_sigma_full)
        if noise_baseline_n is not None:
            flow._terminal_noise_baseline_n = int(noise_baseline_n)
        # Opt-in adaptive recipe selection: fresh iterations may match the recipe
        # mode to the dominant visible failure family. Default off keeps every
        # iteration on the strict comparable artifact_patch_v2 path.
        flow._adaptive_recipe_modes = bool(adaptive_recipes)
        git_start_state = flow.capture_git_state(stage="start")
        if git_start_state.get("available") and git_start_state.get("dirty"):
            typer.secho(
                "Warning: tracked MaxGenie code changes are present. The run will continue and record the dirty state in workspace metadata.",
                fg=typer.colors.YELLOW,
            )
        uses_serving = resolved_strategy == AUTONOMOUS_STRATEGY_SERVING
        artifacts.update_workspace_meta(
            {
                "autonomous_strategy": resolved_strategy,
                "serving_endpoint": resolved_serving_endpoint if uses_serving else None,
                "serving_model": resolved_serving_model if uses_serving else None,
                "serving_temperature": resolved_serving_temperature if uses_serving else None,
                "serving_reasoning_effort": (
                    resolved_serving_reasoning_effort if uses_serving else None
                ),
                "serving_candidate_mode": serving_candidate_mode,
                "fixed_benchmark_artifacts_path": (
                    str(fixed_benchmark_artifacts_path)
                    if fixed_benchmark_artifacts_path is not None
                    else None
                ),
                "k_samples": k_samples,
                "noise_sigma_full": noise_sigma_full,
                "noise_baseline_n": noise_baseline_n,
                "adaptive_recipes": bool(adaptive_recipes),
                "bootstrap_only": bootstrap_only,
                "product_commit_sha": os.environ.get("MAXGENIE_PRODUCT_COMMIT_SHA"),
                "product_branch": os.environ.get("MAXGENIE_PRODUCT_BRANCH"),
                "product_build_dirty": os.environ.get("MAXGENIE_PRODUCT_BUILD_DIRTY"),
                "product_build_at": os.environ.get("MAXGENIE_PRODUCT_BUILD_AT"),
                "fresh_baseline": not (observed_benchmark_seed is not None or use_latest_observed_benchmark),
                "observed_benchmark_reuse_policy": (
                    "observed_seed_initial_baseline"
                    if observed_benchmark_seed is not None
                    else (
                        "latest_observed_initial_baseline"
                        if use_latest_observed_benchmark
                        else "fresh_baseline_benchmark_run"
                    )
                ),
            }
        )
        lineage = None if fresh_lineage else artifacts.load_lineage(ref.space_id)
        managed_source_space_id = (
            str(lineage["managed_source_space_id"])
            if isinstance(lineage, dict) and lineage.get("managed_source_space_id")
            else None
        )
        bootstrap_source_space_id = managed_source_space_id or ref.space_id
        try:
            bundle = flow.export_workspace(
                bootstrap_source_space_id,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                split_seed=split_seed,
                allow_empty_benchmarks=True,
            )
        except WorkspaceFlowError as exc:
            if (
                managed_source_space_id
                and not fresh_lineage
                and _looks_like_missing_space_error(str(exc))
                and artifacts.space_dir.exists()
                and any(artifacts.space_dir.iterdir())
            ):
                recovered_clone = flow.recover_managed_clone_from_workspace(
                    reason=f"resume_export_failed: {exc}",
                    restore_best_checkpoint=True,
                    fallback_warehouse_id=warehouse_id,
                )
                managed_source_space_id = recovered_clone.space_id
                bootstrap_source_space_id = managed_source_space_id
                lineage_mode = "resume_recovered"
                artifacts.update_workspace_meta(
                    {
                        "input_space_id": ref.space_id,
                        "managed_source_space_id": managed_source_space_id,
                        "active_space_id": recovered_clone.space_id,
                        "bootstrap_source_space_id": bootstrap_source_space_id,
                        "lineage_mode": lineage_mode,
                        "workspace_host": ref.host,
                    }
                )
                artifacts.save_lineage(
                    ref.space_id,
                    {
                        "input_space_id": ref.space_id,
                        "managed_source_space_id": managed_source_space_id,
                        "active_space_id": recovered_clone.space_id,
                        "workspace_dir": str(artifacts.workspace_dir),
                        "workspace_host": ref.host,
                    },
                )
                bundle = flow.export_workspace(
                    bootstrap_source_space_id,
                    validation_fraction=validation_fraction,
                    test_fraction=test_fraction,
                    split_seed=split_seed,
                    allow_empty_benchmarks=True,
                )
            else:
                raise
        if managed_source_space_id and not fresh_lineage:
            clone_info = flow.attach_existing_clone(
                clone_space_id=managed_source_space_id,
                source_space_id=ref.space_id,
                input_space_id=ref.space_id,
            )
            lineage_mode = "resume"
        else:
            clone_info = flow.ensure_clone(
                source_space_id=ref.space_id,
                source_config=bundle.config,
                source_title=bundle.space_title,
                input_space_id=ref.space_id,
                fallback_warehouse_id=warehouse_id,
                strategy_label=STRATEGY_WORKSPACE_ROOT_NAMES.get(resolved_strategy),
            )
            managed_source_space_id = clone_info.space_id
            lineage_mode = "fresh"

        artifacts.update_workspace_meta(
            {
                "input_space_id": ref.space_id,
                "managed_source_space_id": managed_source_space_id,
                "active_space_id": clone_info.space_id,
                "bootstrap_source_space_id": bootstrap_source_space_id,
                "lineage_mode": lineage_mode,
                "workspace_host": ref.host,
            }
        )
        artifacts.save_lineage(
            ref.space_id,
            {
                "input_space_id": ref.space_id,
                "managed_source_space_id": managed_source_space_id,
                "active_space_id": clone_info.space_id,
                "workspace_dir": str(artifacts.workspace_dir),
                "workspace_host": ref.host,
            },
        )
        if observed_benchmark_seed is not None and use_latest_observed_benchmark:
            raise WorkspaceFlowError(
                "Use either --observed-benchmark-seed or --use-latest-observed-benchmark, not both."
            )
        if fixed_benchmark_artifacts_path is not None and use_latest_observed_benchmark:
            raise WorkspaceFlowError(
                "Use --fixed-benchmark-artifacts-path with --fresh-baseline or an explicit "
                "--observed-benchmark-seed, not with --use-latest-observed-benchmark."
            )
        if fixed_benchmark_artifacts_path is not None:
            bundle = flow.apply_benchmark_artifact_seed(
                bundle=bundle,
                seed_path=fixed_benchmark_artifacts_path,
            )
        if not bundle.questions:
            advisory_summary = flow.run_no_benchmark_advisory_best_practices(
                label=ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                curation_mode=CURATION_MODE_SAFE_DEFAULTS,
            )
            advisory_clone_space_id = str(advisory_summary.get("clone_space_id") or clone_info.space_id)
            managed_source_space_id = advisory_clone_space_id
            report_path = flow.write_report()
            git_end_state = flow.capture_git_state(stage="end")
            artifacts.save_lineage(
                ref.space_id,
                {
                    "input_space_id": ref.space_id,
                    "managed_source_space_id": managed_source_space_id,
                    "active_space_id": advisory_clone_space_id,
                    "workspace_dir": str(artifacts.workspace_dir),
                    "workspace_host": ref.host,
                    "stop_reason": "no_benchmark_advisory_completed",
                    "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                    "benchmark_verification": NOT_BENCHMARK_VERIFIED_LABEL,
                },
            )
            typer.secho(NO_BENCHMARK_WARNING, fg=typer.colors.YELLOW, bold=True)
            typer.echo(f"Workspace: {artifacts.workspace_dir}")
            typer.echo(f"Task file: {bundle.prompt_path}")
            typer.echo(f"Lineage mode: {lineage_mode}")
            typer.echo(f"Input space: {ref.space_id}")
            typer.echo(f"Managed source space: {managed_source_space_id}")
            typer.echo(f"Clone space: {advisory_clone_space_id}")
            typer.echo(f"Benchmark source: {bundle.benchmark_source}")
            typer.echo("Benchmark verification: not benchmark-verified")
            typer.echo(f"Evidence classification: {ADVISORY_BEST_PRACTICES_CLASSIFICATION}")
            typer.echo(f"Advisory mode: {advisory_summary.get('curation_mode')}")
            typer.echo(f"Advisory changes: {int(advisory_summary.get('change_count', 0) or 0)}")
            typer.echo("Final full pass rate: n/a (no benchmark)")
            if git_start_state.get("available"):
                typer.echo(
                    "Git start state: "
                    f"{git_start_state.get('commit_sha')} ({'dirty' if git_start_state.get('dirty') else 'clean'})"
                )
            if git_end_state.get("available"):
                typer.echo(
                    "Git end state: "
                    f"{git_end_state.get('commit_sha')} ({'dirty' if git_end_state.get('dirty') else 'clean'})"
                )
            typer.echo(f"Report: {report_path}")
            typer.echo("")
            typer.echo("Next steps:")
            typer.echo(f"1. cd {artifacts.workspace_dir}")
            typer.echo("2. Review final_report.md and results/advisory_best_practices.json")
            typer.echo("3. Add owner-approved benchmark questions before treating these changes as verified")
            raise typer.Exit(0)
        if observed_benchmark_seed is None and not use_latest_observed_benchmark:
            try:
                observed_context = latest_eval_run_observed_payload(
                    wc=flow.genie_service.wc,
                    space_id=ref.space_id,
                    expected_question_count=len(bundle.questions),
                )
                artifacts.write_json_path(
                    artifacts.results_dir / "latest_observed_benchmark_context.json",
                    observed_context,
                )
                artifacts.record_iteration(
                    {
                        "phase": "benchmark_observation_context_fetched",
                        "source": "latest_genie_eval_run",
                        "eval_run_id": observed_context.get("eval_run_id"),
                        "question_count": observed_context.get("question_count"),
                        "result_count": observed_context.get("result_count"),
                    }
                )
            except Exception as exc:
                artifacts.record_iteration(
                    {
                        "phase": "benchmark_observation_context_unavailable",
                        "source": "latest_genie_eval_run",
                        "error": str(exc),
                    }
                )
        if observed_benchmark_seed is None and use_latest_observed_benchmark:
            try:
                observed_benchmark_seed = write_latest_observed_benchmark_seed(
                    wc=flow.genie_service.wc,
                    space_id=ref.space_id,
                    path=artifacts.results_dir / "latest_observed_benchmark_seed.json",
                    expected_question_count=len(bundle.questions),
                )
            except Exception as exc:
                raise WorkspaceFlowError(
                    "Failed to fetch the latest completed Genie benchmark eval run: "
                    f"{exc}"
                ) from exc
            artifacts.record_iteration(
                {
                    "phase": "benchmark_observation_fetched",
                    "source": "latest_genie_eval_run",
                    "seed_path": str(observed_benchmark_seed),
                    "question_count": len(bundle.questions),
                }
            )
        if observed_benchmark_seed is not None:
            population_baseline, observed_payload = load_observed_benchmark_run(
                observed_benchmark_seed,
                questions=bundle.questions,
                space_id=clone_info.space_id,
                match_threshold=match_threshold,
            )
            artifacts.write_json_path(
                artifacts.results_dir / "observed_benchmark_seed.json",
                observed_payload,
            )
            artifacts.record_iteration(
                {
                    "phase": "benchmark_observation_ingested",
                    "label": "population_baseline",
                    "seed_path": str(observed_benchmark_seed),
                    "question_count": len(bundle.questions),
                    "summary": {
                        "total": population_baseline.total,
                        "passed": population_baseline.passed,
                        "failed": population_baseline.failed,
                        "errors": population_baseline.errors,
                        "pass_rate": round(population_baseline.pass_rate, 2),
                    },
                }
            )
        else:
            population_baseline = flow.run_population_benchmark(
                label="population_baseline",
                questions=bundle.questions,
                k_samples=k_samples,
            )
        if fixed_benchmark_artifacts_path is None:
            bundle = flow.rewrite_split_from_population_baseline(
                bundle=bundle,
                baseline_run=population_baseline,
            )
        else:
            artifacts.record_iteration(
                {
                    "phase": "split_rebalance_skipped",
                    "reason": "fixed_benchmark_artifacts_path",
                    "seed_path": str(fixed_benchmark_artifacts_path),
                }
            )
        # Precompute the full answer key (gold rows for ALL 15 expected SQLs) BEFORE any
        # baseline/candidate comparison. compare() alone caches only the questions whose
        # *generated* SQL succeeds, so the dominant unbound-parameter failures would
        # otherwise have no gold rows and no visible-failure expected-row preview.
        flow.precompute_answer_key(bundle.questions)
        if observed_benchmark_seed is not None:
            baseline = flow.initialize_baseline_from_observed_population(
                label="baseline",
                observed_run=population_baseline,
            )
        elif fixed_benchmark_artifacts_path is not None and k_samples > 1:
            baseline = flow.initialize_baseline_from_population_samples(
                label="baseline",
                questions=bundle.questions,
                population_label="population_baseline",
            )
        else:
            baseline = flow.run_full_benchmark(label="baseline", baseline=True)
        curation_summary = flow.curate_workspace_mode(
            label="baseline_auto_curate",
            mode=CURATION_MODE_SAFE_DEFAULTS,
        )
        query_history_summary = flow.collect_query_history(
            label="baseline_query_history",
            warehouse_id=warehouse_id or bundle.warehouse_id,
        )
        optimization_summary = None
        if dry_run:
            checks_passed = True
            typer.secho("Dry-run validation:", bold=True)
            typer.secho(f"  Strategy: {resolved_strategy}", fg=typer.colors.CYAN)
            typer.secho(f"  Clone: {clone_info.space_id}", fg=typer.colors.CYAN)
            try:
                flow.preflight_space(clone_info.space_id)
                typer.secho("  Clone accessible: yes", fg=typer.colors.GREEN)
            except Exception as exc:
                typer.secho(f"  Clone accessible: NO — {exc}", fg=typer.colors.RED)
                checks_passed = False
            baseline_exists = artifacts.baseline_train_path.exists()
            typer.secho(f"  Baseline results: {'yes' if baseline_exists else 'NO'}", fg=typer.colors.GREEN if baseline_exists else typer.colors.RED)
            if not baseline_exists:
                checks_passed = False
            if uses_serving:
                if resolved_serving_endpoint:
                    typer.secho(
                        f"  Serving endpoint: {resolved_serving_endpoint}",
                        fg=typer.colors.GREEN,
                    )
                else:
                    typer.secho("  Serving endpoint: NOT CONFIGURED", fg=typer.colors.RED)
                    checks_passed = False
            if checks_passed:
                typer.secho("All pre-flight checks passed.", fg=typer.colors.GREEN, bold=True)
            else:
                typer.secho("Some pre-flight checks failed.", fg=typer.colors.RED, bold=True)
                raise typer.Exit(1)
            raise typer.Exit(0)
        if audit_checkpoint_path is not None:
            optimization_summary = flow.run_checkpoint_audit(
                label="checkpoint_audit",
                checkpoint_path=audit_checkpoint_path,
                k_samples=k_samples,
            )
        elif audit_patch_path is not None:
            optimization_summary = flow.run_patch_replay_audit(
                label="patch_replay_audit",
                patch_path=audit_patch_path,
            )
        elif not bootstrap_only:
            optimization_summary = flow.run_autonomous_optimization(
                label="optimize",
                plateau_rounds=plateau_rounds,
                max_iterations=max_iterations,
                candidate_budget=candidate_budget,
                strategy=resolved_strategy,
                serving_endpoint=resolved_serving_endpoint,
                serving_model=resolved_serving_model,
                serving_temperature=resolved_serving_temperature,
                serving_max_tokens=serving_max_tokens,
                serving_reasoning_effort=resolved_serving_reasoning_effort,
                serving_response_format=serving_response_format,
                serving_api_format=serving_api_format,
                serving_candidate_mode=serving_candidate_mode,
                greedy_best_ever=greedy_best_ever,
                confirm_on_beat=confirm_on_beat,
            )
        if optimization_summary is not None:
            artifacts.save_lineage(
                ref.space_id,
                {
                    "input_space_id": ref.space_id,
                    "managed_source_space_id": managed_source_space_id,
                    "active_space_id": clone_info.space_id,
                    "workspace_dir": str(artifacts.workspace_dir),
                    "workspace_host": ref.host,
                    "best_checkpoint_path": optimization_summary.get("best_checkpoint_path"),
                    "stop_reason": optimization_summary.get("stop_reason"),
                    "latest_full_pass_rate": (
                        optimization_summary.get("final", {}).get("overall", {}).get("pass_rate")
                        if isinstance(optimization_summary.get("final"), dict)
                        else baseline.get("overall", {}).get("pass_rate")
                    ),
                },
            )
        report_path = flow.write_report()
        git_end_state = flow.capture_git_state(stage="end")
    except (WorkspaceFlowError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(2)

    typer.echo(f"Workspace: {artifacts.workspace_dir}")
    typer.echo(f"Task file: {bundle.prompt_path}")
    typer.echo(f"Lineage mode: {lineage_mode}")
    typer.echo(f"Input space: {ref.space_id}")
    typer.echo(f"Managed source space: {managed_source_space_id}")
    typer.echo(f"Clone space: {clone_info.space_id}")
    typer.echo(f"Benchmark source: {bundle.benchmark_source}")
    typer.echo(f"Split strategy: {bundle.split.split_strategy}")
    typer.echo(f"Strategy: {resolved_strategy}")
    if uses_serving and resolved_serving_endpoint:
        typer.echo(f"Serving endpoint: {resolved_serving_endpoint}")
        if resolved_serving_model:
            typer.echo(f"Serving model: {resolved_serving_model}")
    if bundle.split.validation_strategy == "small_sample_kfold":
        typer.echo(f"Validation strategy: grouped {len(bundle.split.validation_folds)}-fold over train+validation")
    else:
        typer.echo("Validation strategy: blind split")
    typer.echo(f"Baseline train pass rate: {baseline['train']['pass_rate']:.2f}%")
    typer.echo(f"Baseline validation pass rate: {baseline['validation']['pass_rate']:.2f}%")
    typer.echo(f"Baseline test pass rate: {baseline['test']['pass_rate']:.2f}%")
    typer.echo(f"Baseline full pass rate: {baseline['overall']['pass_rate']:.2f}%")
    typer.echo(f"Auto-curation mode: {curation_summary.get('mode')}")
    typer.echo(f"Auto-curation changes: {len(curation_summary.get('changes', []))}")
    typer.echo(f"Query-history status: {query_history_summary.get('status')}")
    if git_start_state.get("available"):
        typer.echo(
            "Git start state: "
            f"{git_start_state.get('commit_sha')} ({'dirty' if git_start_state.get('dirty') else 'clean'})"
        )
    if git_end_state.get("available"):
        typer.echo(
            "Git end state: "
            f"{git_end_state.get('commit_sha')} ({'dirty' if git_end_state.get('dirty') else 'clean'})"
        )
    if optimization_summary is not None:
        typer.echo(f"Accepted candidates: {optimization_summary.get('accepted', 0)}")
        typer.echo(f"Rejected candidates: {optimization_summary.get('rejected', 0)}")
        typer.echo(f"Skipped candidates: {optimization_summary.get('skipped', 0)}")
        typer.echo(f"Stop reason: {optimization_summary.get('stop_reason')}")
        final_summary = optimization_summary.get("final", {})
        if isinstance(final_summary, dict) and final_summary.get("overall"):
            typer.echo(f"Final full pass rate: {final_summary['overall']['pass_rate']:.2f}%")
    typer.echo(f"Report: {report_path}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"1. cd {artifacts.workspace_dir}")
    typer.echo("2. Review OPTIMIZATION_TASK.md, final_report.md, and results/optimization_summary.json")
    typer.echo("3. Use maxgenie curate-loop for a manual one-sweep pass if you want to probe specific curation modes")
    typer.echo("4. Review results/query_history_summary.json and results/query_history_recommendations.json if query history was available")
    typer.echo("5. Inspect checkpoints/ if you want to compare accepted states or restore a previous winner")



def main() -> None:
    app()


if __name__ == "__main__":
    main()
