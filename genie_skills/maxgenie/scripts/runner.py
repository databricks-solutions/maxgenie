"""Unified runner for the MaxGenie workspace skill."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


DEFAULT_QUESTION = "Give me sample 5 questions"


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "omit", "unset"}:
        return None
    return float(value)


def _optional_positive_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "omit", "unset", "unbounded"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer or unbounded")
    return parsed


def _script_path() -> Path:
    raw_path = globals().get("__file__") or sys.argv[0]
    if not raw_path:
        raise RuntimeError("Could not resolve the skill runner path.")
    return Path(str(raw_path)).resolve()


def _skill_root() -> Path:
    return _script_path().parents[1]


def _repo_root() -> Path:
    return _script_path().parents[3]


def _existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _import_paths() -> list[str]:
    skill_root = _skill_root()
    repo_root = _repo_root()
    paths = [
        str(skill_root / "lib"),
        str(skill_root),
    ]
    if (repo_root / "maxgenie").exists():
        paths.insert(0, str(repo_root))
    return paths


def _ensure_import_paths() -> None:
    for path in reversed(_import_paths()):
        if path not in sys.path:
            sys.path.insert(0, path)


def _maxgenie_pythonpath() -> str:
    paths = _import_paths()
    existing = os.environ.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def _load_build_info() -> dict[str, object]:
    path = _skill_root() / "BUILD_INFO.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _apply_build_info_env() -> None:
    build_info = _load_build_info()
    for env_name, key in (
        ("MAXGENIE_PRODUCT_COMMIT_SHA", "commit_sha"),
        ("MAXGENIE_PRODUCT_BRANCH", "branch"),
        ("MAXGENIE_PRODUCT_BUILD_DIRTY", "dirty"),
        ("MAXGENIE_PRODUCT_BUILD_AT", "built_at"),
    ):
        value = build_info.get(key)
        if value is not None:
            os.environ.setdefault(env_name, str(value))


def _load_packaged_maxgenie() -> bool:
    package_dir = _existing_path(
        _skill_root() / "lib" / "engine",
        _skill_root() / "lib" / "maxgenie",
    )
    if package_dir is None:
        return False

    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "maxgenie",
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return False

    module = importlib.util.module_from_spec(spec)
    sys.modules["maxgenie"] = module
    spec.loader.exec_module(module)
    return True


def _ensure_maxgenie_package() -> None:
    if "maxgenie" in sys.modules:
        return
    if _load_packaged_maxgenie():
        return
    if importlib.util.find_spec("maxgenie") is not None:
        return
    raise ModuleNotFoundError("Could not load bundled optimizer package.")


def run_full(args: argparse.Namespace) -> int:
    """Run the full MaxGenie path in-process."""
    _ensure_import_paths()
    os.environ["PYTHONPATH"] = _maxgenie_pythonpath()
    _apply_build_info_env()
    _ensure_maxgenie_package()

    from maxgenie.cli import optimize

    try:
        optimize(
            space_url=args.space_url,
            profile=args.profile,
            workspace_root=args.workspace_root,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            delay=args.delay_seconds,
            match_threshold=args.match_threshold,
            warehouse_id=args.warehouse_id,
            clone_parent_path=args.clone_parent_path,
            bootstrap_only=args.bootstrap_only,
            plateau_rounds=args.plateau_rounds,
            max_iterations=args.max_iterations,
            candidate_budget=args.candidate_budget,
            strategy=args.strategy,
            fresh_lineage=args.fresh_lineage,
            dry_run=args.dry_run,
            observed_benchmark_seed=args.observed_benchmark_seed,
            use_latest_observed_benchmark=args.use_latest_observed_benchmark,
            audit_checkpoint_path=args.audit_checkpoint_path,
            audit_patch_path=args.audit_patch_path,
            positive_control=args.positive_control,
            serving_endpoint=args.serving_endpoint,
            serving_model=args.serving_model,
            serving_temperature=args.serving_temperature,
            serving_max_tokens=args.serving_max_tokens,
            serving_reasoning_effort=args.serving_reasoning_effort,
            serving_response_format=args.serving_response_format,
            serving_api_format=args.serving_api_format,
            serving_candidate_mode=args.serving_candidate_mode,
            fixed_benchmark_artifacts_path=args.fixed_benchmark_artifacts_path,
            k_samples=getattr(args, "k_samples", 1),
            noise_sigma_full=getattr(args, "noise_sigma_full", None),
            noise_baseline_n=getattr(args, "noise_baseline_n", None),
            adaptive_recipes=getattr(args, "adaptive_recipes", False),
            greedy_best_ever=getattr(args, "greedy_best_ever", False),
            confirm_on_beat=getattr(args, "confirm_on_beat", 0),
            content_match=getattr(args, "content_match", None),
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def run_submit(args: argparse.Namespace) -> int:
    """Submit the full path as a Databricks job run."""
    _ensure_import_paths()
    os.environ["PYTHONPATH"] = _maxgenie_pythonpath()
    _apply_build_info_env()
    _ensure_maxgenie_package()

    from maxgenie.cli import submit_run_command

    runner_path = args.runner_path
    local_runner_path = str(Path(__file__).resolve())
    if runner_path is None and local_runner_path.startswith("/Workspace/"):
        runner_path = local_runner_path

    try:
        submit_run_command(
            space_url=args.space_url,
            profile=args.profile,
            runner_path=runner_path,
            skill_scope=args.skill_scope,
            workspace_root=args.workspace_root,
            strategy=args.strategy,
            run_name=args.run_name,
            use_latest_observed_benchmark=args.use_latest_observed_benchmark,
            fresh_lineage=args.fresh_lineage,
            bootstrap_only=args.bootstrap_only,
            max_iterations=args.max_iterations,
            plateau_rounds=args.plateau_rounds,
            candidate_budget=args.candidate_budget,
            delay_seconds=args.delay_seconds,
            match_threshold=args.match_threshold,
            warehouse_id=args.warehouse_id,
            clone_parent_path=args.clone_parent_path,
            serving_endpoint=args.serving_endpoint,
            serving_model=args.serving_model,
            serving_temperature=args.serving_temperature,
            serving_max_tokens=args.serving_max_tokens,
            serving_reasoning_effort=args.serving_reasoning_effort,
            serving_response_format=args.serving_response_format,
            serving_api_format=args.serving_api_format,
            serving_candidate_mode=args.serving_candidate_mode,
            observed_benchmark_seed=args.observed_benchmark_seed,
            fixed_benchmark_artifacts_path=args.fixed_benchmark_artifacts_path,
            k_samples=getattr(args, "k_samples", 1),
            noise_sigma_full=getattr(args, "noise_sigma_full", None),
            noise_baseline_n=getattr(args, "noise_baseline_n", None),
            adaptive_recipes=getattr(args, "adaptive_recipes", False),
            greedy_best_ever=getattr(args, "greedy_best_ever", False),
            confirm_on_beat=getattr(args, "confirm_on_beat", 0),
            content_match=getattr(args, "content_match", None),
            audit_checkpoint_path=args.audit_checkpoint_path,
            audit_patch_path=args.audit_patch_path,
            positive_control=args.positive_control,
            timeout_seconds=args.timeout_seconds,
            compute_mode=args.compute_mode,
            existing_cluster_id=args.existing_cluster_id,
            new_cluster_json=args.new_cluster_json,
            serverless_environment_version=args.serverless_environment_version,
            install_dependencies=args.install_dependencies,
            idempotency_token=args.idempotency_token,
            require_production_protocol=args.require_production_protocol,
            dry_run=args.dry_run,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def run_status(args: argparse.Namespace) -> int:
    """Fetch and print a submitted Databricks job run status."""
    _ensure_import_paths()
    os.environ["PYTHONPATH"] = _maxgenie_pythonpath()
    _apply_build_info_env()
    _ensure_maxgenie_package()

    if not args.run_id:
        print("No run id provided. Pass --run-id.", file=sys.stderr)
        return 2

    from maxgenie.cli import run_status_summary_payload
    from maxgenie.workspace_jobs import format_run_status_for_chat

    try:
        summary = run_status_summary_payload(
            run_id=str(args.run_id),
            space_url=args.space_url,
            profile=args.profile,
            stale_after_minutes=args.stale_after_minutes,
            include_observed_ui=args.include_observed_ui,
        )
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.status_format == "json":
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(format_run_status_for_chat(summary))
    return 0


def run_lite(args: argparse.Namespace) -> int:
    """Run the self-contained fallback path in-process."""
    if not args.space_url:
        print(
            "No Genie space URL provided. Pass --space-url or set MAXGENIE_DEFAULT_SPACE_URL.",
            file=sys.stderr,
        )
        return 2

    script_path = _existing_path(
        _skill_root() / "scripts" / "smoke.py",
        _repo_root() / "genie_skills" / "maxgenie" / "scripts" / "smoke.py",
    )
    if script_path is None:
        print("Could not find smoke.py.", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    spec = importlib.util.spec_from_file_location("skill_smoke_runtime", script_path)
    if spec is None or spec.loader is None:
        print(f"Could not load {script_path}.", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lite_args = argparse.Namespace(
        space_url=args.space_url,
        profile=args.profile,
        out_dir=out_dir,
        title_prefix=args.title_prefix,
        parent_path=args.parent_path,
        benchmark_limit=args.benchmark_limit,
        max_candidates=args.max_candidates,
        delay_seconds=args.delay_seconds,
        conversation_question=args.conversation_question,
        preserve_parent_path=args.preserve_parent_path,
        trash_clone=args.trash_clone,
        verbose=args.verbose,
    )
    result = module.run_poc(lite_args)
    print(
        json.dumps(
            {
                "report": str(result.run_dir / "report.md"),
                "clone_trashed": result.clone_trashed,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MaxGenie from a synced skill or local shell.")
    parser.add_argument("--mode", choices=("full", "lite", "submit", "status"), default="full")
    parser.add_argument("--space-url", default=os.environ.get("MAXGENIE_DEFAULT_SPACE_URL"))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--workspace-root", default="runs/full")
    parser.add_argument("--strategy", default="centaur")
    parser.add_argument("--max-iterations", type=_optional_positive_int, default=None)
    parser.add_argument("--candidate-budget", type=_optional_positive_int, default=None)
    parser.add_argument("--plateau-rounds", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--delay-seconds", type=float, default=12.0)
    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--warehouse-id", default=None)
    parser.add_argument("--clone-parent-path", default=None)
    parser.add_argument("--observed-benchmark-seed", default=None)
    parser.add_argument("--use-latest-observed-benchmark", action="store_true")
    parser.add_argument("--audit-checkpoint-path", default=None)
    parser.add_argument("--audit-patch-path", default=None)
    parser.add_argument("--positive-control", default=None)
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh-lineage", action="store_true")
    parser.add_argument("--serving-endpoint", default=None)
    parser.add_argument("--serving-model", default=None)
    parser.add_argument("--serving-temperature", type=_optional_float, default=None)
    parser.add_argument("--serving-max-tokens", type=int, default=16000)
    parser.add_argument("--serving-reasoning-effort", default=None)
    parser.add_argument("--serving-response-format", default="json_schema")
    parser.add_argument("--serving-api-format", choices=("auto", "chat", "responses"), default="auto")
    parser.add_argument("--serving-candidate-mode", default=None)
    parser.add_argument("--fixed-benchmark-artifacts-path", default=None)
    parser.add_argument("--k-samples", type=int, default=1)
    parser.add_argument("--noise-sigma-full", type=_optional_float, default=None)
    parser.add_argument("--noise-baseline-n", type=_optional_positive_int, default=None)
    parser.add_argument("--adaptive-recipes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--greedy-best-ever", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-on-beat", type=int, default=0)
    parser.add_argument("--content-match", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--runner-path", default=None)
    parser.add_argument("--skill-scope", choices=("user", "workspace"), default="user")
    parser.add_argument("--run-name", default="Space Optimizer Long Run")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--compute-mode", choices=("serverless", "existing_cluster", "new_cluster"), default="serverless")
    parser.add_argument("--existing-cluster-id", default=None)
    parser.add_argument("--new-cluster-json", default=None)
    parser.add_argument("--serverless-environment-version", default="2")
    parser.add_argument("--install-dependencies", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--idempotency-token", default=None)
    parser.add_argument(
        "--require-production-protocol",
        dest="require_production_protocol",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--allow-diagnostic",
        dest="require_production_protocol",
        action="store_false",
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--status-format", choices=("chat", "json"), default="chat")
    parser.add_argument("--stale-after-minutes", type=int, default=60)
    parser.add_argument("--include-observed-ui", action="store_true")

    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--title-prefix", default="Space Optimizer Lite")
    parser.add_argument("--parent-path", default=None)
    parser.add_argument("--benchmark-limit", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--conversation-question", default=DEFAULT_QUESTION)
    parser.add_argument("--preserve-parent-path", action="store_true")
    parser.add_argument("--trash-clone", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _namespace_from_kwargs(**overrides: object) -> argparse.Namespace:
    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_skill(**overrides: object) -> int:
    """Programmatic in-process entrypoint for the workspace runtime."""
    args = _namespace_from_kwargs(**overrides)
    if args.mode == "lite":
        return run_lite(args)
    if args.mode == "submit":
        return run_submit(args)
    if args.mode == "status":
        return run_status(args)
    return run_full(args)


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "lite":
        return run_lite(args)
    if args.mode == "submit":
        return run_submit(args)
    if args.mode == "status":
        return run_status(args)
    return run_full(args)


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
