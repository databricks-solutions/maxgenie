import importlib.util
from pathlib import Path


def _load_runner_module():
    path = Path(__file__).resolve().parents[1] / "genie_skills" / "maxgenie" / "scripts" / "runner.py"
    spec = importlib.util.spec_from_file_location("maxgenie_skill_runner", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_candidate_budget_defaults_to_unbounded():
    runner = _load_runner_module()

    args = runner.build_parser().parse_args([])

    assert args.candidate_budget is None


def test_runner_max_iterations_defaults_to_unbounded():
    runner = _load_runner_module()

    args = runner.build_parser().parse_args([])

    assert args.max_iterations is None


def test_runner_candidate_budget_accepts_explicit_limit():
    runner = _load_runner_module()

    args = runner.build_parser().parse_args(["--candidate-budget", "2"])

    assert args.candidate_budget == 2


def test_runner_max_iterations_accepts_explicit_limit_or_unbounded_alias():
    runner = _load_runner_module()

    limited = runner.build_parser().parse_args(["--max-iterations", "6"])
    unbounded = runner.build_parser().parse_args(["--max-iterations", "unbounded"])

    assert limited.max_iterations == 6
    assert unbounded.max_iterations is None


def test_runner_noise_and_adaptive_flags_default_off():
    # Defaults must match the optimize defaults so the job path is byte-for-byte
    # identical for the non-greedy tuning knobs when the run does not opt in.
    runner = _load_runner_module()

    args = runner.build_parser().parse_args([])

    assert args.k_samples == 1
    assert args.noise_sigma_full is None
    assert args.noise_baseline_n is None
    assert args.adaptive_recipes is False
    assert args.greedy_best_ever is True


def test_runner_noise_and_adaptive_flags_parse_explicit_values():
    runner = _load_runner_module()

    args = runner.build_parser().parse_args(
        [
            "--k-samples",
            "6",
            "--noise-sigma-full",
            "5.97",
            "--noise-baseline-n",
            "6",
            "--adaptive-recipes",
        ]
    )

    assert args.k_samples == 6
    assert args.noise_sigma_full == 5.97
    assert args.noise_baseline_n == 6
    assert args.adaptive_recipes is True


def test_runner_namespace_from_kwargs_exposes_noise_defaults():
    # The in-process run_skill(**overrides) path builds its namespace from the
    # parser defaults, so the new fields must be present even when not overridden.
    runner = _load_runner_module()

    args = runner._namespace_from_kwargs(space_url="https://example/genie/rooms/abc")

    assert args.k_samples == 1
    assert args.noise_sigma_full is None
    assert args.noise_baseline_n is None
    assert args.adaptive_recipes is False
    assert args.greedy_best_ever is True


def test_runner_can_disable_greedy_best_ever():
    runner = _load_runner_module()

    args = runner.build_parser().parse_args(["--no-greedy-best-ever"])

    assert args.greedy_best_ever is False


def test_runner_accepts_status_mode_defaults_to_chat_format():
    runner = _load_runner_module()

    args = runner.build_parser().parse_args(["--mode", "status", "--run-id", "123"])

    assert args.mode == "status"
    assert args.run_id == "123"
    assert args.status_format == "chat"
    assert args.stale_after_minutes == 60
