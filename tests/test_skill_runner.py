from types import SimpleNamespace

import genie_skills.maxgenie.scripts.runner as skill_runner
import maxgenie.cli as cli_module


def test_runner_parser_defaults_to_production_protocol_and_supports_diagnostic_escape():
    default_args = skill_runner.build_parser().parse_args([])
    diagnostic_args = skill_runner.build_parser().parse_args(
        ["--allow-diagnostic", "--serving-candidate-mode", "artifact_patch"]
    )

    assert default_args.require_production_protocol is True
    assert diagnostic_args.require_production_protocol is False
    assert diagnostic_args.serving_candidate_mode == "artifact_patch"


def test_runner_full_forwards_positive_control(monkeypatch):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "optimize", fake_optimize)
    args = skill_runner._namespace_from_kwargs(
        mode="full",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        positive_control="example_checkpoint_control",
    )

    result = skill_runner.run_full(args)

    assert result == 0
    assert captured["positive_control"] == "example_checkpoint_control"


def test_runner_full_forwards_audit_patch_path(monkeypatch):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "optimize", fake_optimize)
    args = skill_runner._namespace_from_kwargs(
        mode="full",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        audit_patch_path="/Workspace/runs/positive_controls/example/patch",
    )

    result = skill_runner.run_full(args)

    assert result == 0
    assert captured["audit_patch_path"] == "/Workspace/runs/positive_controls/example/patch"


def test_runner_full_forwards_fixed_benchmark_artifacts_path(monkeypatch):
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "optimize", fake_optimize)
    args = skill_runner._namespace_from_kwargs(
        mode="full",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        fixed_benchmark_artifacts_path="/Workspace/runs/canary/space",
    )

    result = skill_runner.run_full(args)

    assert result == 0
    assert captured["fixed_benchmark_artifacts_path"] == "/Workspace/runs/canary/space"


def test_runner_submit_forwards_production_protocol_flag(monkeypatch):
    captured = {}

    def fake_submit_run_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "submit_run_command", fake_submit_run_command)
    args = skill_runner._namespace_from_kwargs(
        mode="submit",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        runner_path="/Workspace/tmp/maxgenie/scripts/runner.py",
        serving_endpoint="candidate-endpoint",
        positive_control="example_checkpoint_control",
        serving_candidate_mode="artifact_patch",
        require_production_protocol=False,
    )

    result = skill_runner.run_submit(args)

    assert result == 0
    assert captured["require_production_protocol"] is False
    assert captured["positive_control"] == "example_checkpoint_control"
    assert captured["serving_candidate_mode"] == "artifact_patch"


def test_runner_submit_forwards_fixed_benchmark_artifacts_path(monkeypatch):
    captured = {}

    def fake_submit_run_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "submit_run_command", fake_submit_run_command)
    args = skill_runner._namespace_from_kwargs(
        mode="submit",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        runner_path="/Workspace/tmp/maxgenie/scripts/runner.py",
        serving_endpoint="candidate-endpoint",
        fixed_benchmark_artifacts_path="/Workspace/runs/canary/space",
    )

    result = skill_runner.run_submit(args)

    assert result == 0
    assert captured["fixed_benchmark_artifacts_path"] == "/Workspace/runs/canary/space"
    assert captured["observed_benchmark_seed"] is None


def test_runner_submit_forwards_observed_benchmark_seed(monkeypatch):
    captured = {}

    def fake_submit_run_command(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_module, "submit_run_command", fake_submit_run_command)
    args = skill_runner._namespace_from_kwargs(
        mode="submit",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        runner_path="/Workspace/tmp/maxgenie/scripts/runner.py",
        serving_endpoint="candidate-endpoint",
        observed_benchmark_seed="/Workspace/runs/canary/observed_seed.json",
    )

    result = skill_runner.run_submit(args)

    assert result == 0
    assert captured["observed_benchmark_seed"] == "/Workspace/runs/canary/observed_seed.json"


def test_runner_status_prints_chat_summary(monkeypatch, capsys):
    def fake_run_status_summary_payload(**kwargs):
        assert kwargs["run_id"] == "123"
        return {
            "run_id": 123,
            "run_page_url": "https://workspace/?o=1#job/999/run/123",
            "life_cycle_state": "RUNNING",
            "elapsed_seconds": 65.0,
            "tasks": [],
        }

    monkeypatch.setattr(cli_module, "run_status_summary_payload", fake_run_status_summary_payload)
    args = skill_runner._namespace_from_kwargs(
        mode="status",
        run_id="123",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    result = skill_runner.run_status(args)

    captured = capsys.readouterr()
    assert result == 0
    assert "Run `123` is `RUNNING`" in captured.out
    assert "https://workspace/?o=1#job/999/run/123" in captured.out


def test_run_skill_submit_is_non_blocking_by_default(monkeypatch):
    captured = {}

    def fake_run_submit(args):
        captured["wait"] = args.wait
        return 0

    monkeypatch.setattr(skill_runner, "run_submit", fake_run_submit)

    result = skill_runner.run_skill(
        mode="submit",
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    assert result == 0
    assert captured["wait"] is False


def test_run_skill_submit_respects_explicit_wait_false(monkeypatch):
    captured = {}

    def fake_run_submit(args):
        captured["wait"] = args.wait
        return 0

    monkeypatch.setattr(skill_runner, "run_submit", fake_run_submit)

    result = skill_runner.run_skill(
        mode="submit",
        wait=False,
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    assert result == 0
    assert captured["wait"] is False


def test_run_skill_submit_respects_explicit_wait_true(monkeypatch):
    captured = {}

    def fake_run_submit(args):
        captured["wait"] = args.wait
        return 0

    monkeypatch.setattr(skill_runner, "run_submit", fake_run_submit)

    result = skill_runner.run_skill(
        mode="submit",
        wait=True,
        space_url="https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )

    assert result == 0
    assert captured["wait"] is True
