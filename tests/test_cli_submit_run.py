import json
from pathlib import Path
from types import SimpleNamespace

from databricks.sdk.errors import PermissionDenied
from typer.testing import CliRunner

import maxgenie.cli as cli_module


SPACE_URL = "https://dbc.example.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


class _FakeCurrentUser:
    def me(self):
        return type("User", (), {"user_name": "user@example.com"})()


class _FakeWorkspaceClient:
    def __init__(self):
        self.current_user = _FakeCurrentUser()


class _FakeApiClient:
    def do(self, method: str, path: str, body: dict | None = None):
        if method == "GET" and path == "/api/2.0/sql/warehouses":
            return {
                "warehouses": [
                    {"id": "wh-stopped", "state": "STOPPED"},
                    {"id": "wh-running", "state": "RUNNING"},
                ]
            }
        raise AssertionError(f"Unexpected request: {method} {path}")


class _FakeServingEndpoints:
    def list(self):
        return [
            SimpleNamespace(
                name="databricks-gpt-5-5-pro",
                state={"ready": "READY"},
            ),
            SimpleNamespace(
                name="databricks-gpt-5-5",
                state={"ready": "READY"},
            ),
            SimpleNamespace(
                name="databricks-claude-opus-4-8",
                state={"ready": "READY"},
            ),
        ]


class _FakeWorkspaceClientWithResources(_FakeWorkspaceClient):
    def __init__(self):
        super().__init__()
        self.api_client = _FakeApiClient()
        self.serving_endpoints = _FakeServingEndpoints()


def _patch_workspace_client(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "create_workspace_client",
        lambda profile=None, host=None: _FakeWorkspaceClient(),
    )


def test_submit_run_requires_production_protocol_by_default(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "Production protocol preflight failed" in result.output
    assert "--allow-diagnostic" in result.output


def test_submit_run_allow_diagnostic_bypasses_production_preflight(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--allow-diagnostic",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert "--use-latest-observed-benchmark" in params
    assert "--warehouse-id" not in params


def test_submit_run_accepts_comparable_protocol_by_default(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "xhigh",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--warehouse-id") + 1] == "warehouse-123"
    assert "--use-latest-observed-benchmark" not in params


def test_submit_run_passes_fixed_artifacts_with_explicit_observed_seed(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "medium",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--fixed-benchmark-artifacts-path",
            "/Workspace/runs/seed",
            "--observed-benchmark-seed",
            "/Workspace/runs/shared_baseline_seed.json",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--fixed-benchmark-artifacts-path") + 1] == "/Workspace/runs/seed"
    assert (
        params[params.index("--observed-benchmark-seed") + 1]
        == "/Workspace/runs/shared_baseline_seed.json"
    )
    assert "--use-latest-observed-benchmark" not in params


def test_submit_run_discovers_production_endpoint_model_and_warehouse(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "create_workspace_client",
        lambda profile=None, host=None: _FakeWorkspaceClientWithResources(),
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--strategy",
            "serving",
            "--fresh-baseline",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert params[params.index("--warehouse-id") + 1] == "wh-running"
    assert params[params.index("--serving-endpoint") + 1] == "databricks-gpt-5-5"
    assert params[params.index("--serving-model") + 1] == "databricks-gpt-5-5"
    assert params[params.index("--serving-reasoning-effort") + 1] == "low"


def test_submit_run_prints_canonical_run_page_url_first(monkeypatch):
    _patch_workspace_client(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "submit_workspace_run",
        lambda *args, **kwargs: {
            "run_id": 123,
            "run_page_url": "https://dbc.example.com/?o=1#job/9/run/123",
            "state": {"life_cycle_state": "PENDING"},
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "xhigh",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
        ],
    )

    assert result.exit_code == 0
    output_lines = result.output.splitlines()
    assert output_lines[0] == "https://dbc.example.com/?o=1#job/9/run/123"
    space_id = cli_module.parse_space_url(SPACE_URL).space_id
    assert (
        "Final report (when complete): "
        "/Workspace/Users/user@example.com/.maxgenie/runs/job/"
        f"{space_id}/final_report.ipynb"
    ) in output_lines
    assert "#job-run" not in result.output


def test_submit_run_passes_bootstrap_only_for_job_canary(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "xhigh",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--bootstrap-only",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert "--bootstrap-only" in params


def test_submit_run_passes_greedy_best_ever_and_confirm_on_beat(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-claude-opus-4-8",
            "--serving-reasoning-effort",
            "medium",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--greedy-best-ever",
            "--confirm-on-beat",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert "--greedy-best-ever" in params
    assert params[params.index("--confirm-on-beat") + 1] == "2"


def test_submit_run_uses_greedy_best_ever_by_default(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "xhigh",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert "--greedy-best-ever" in params
    assert "--confirm-on-beat" not in params


def test_submit_run_can_disable_greedy_best_ever(monkeypatch):
    _patch_workspace_client(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "xhigh",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--no-greedy-best-ever",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    assert "--greedy-best-ever" not in params
    assert "--no-greedy-best-ever" in params


def test_submit_run_positive_control_sets_workspace_checkpoint_path(monkeypatch, tmp_path: Path):
    _patch_workspace_client(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
    spec_path = tmp_path / "positive_control.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "local_checkpoint",
                "checkpoint_path": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--workspace-root",
            "/Workspace/Users/user@example.com/.conversations/space-optimizer/runs/p-control",
            "--positive-control",
            str(spec_path),
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    checkpoint_path = params[params.index("--audit-checkpoint-path") + 1]
    assert checkpoint_path == (
        "/Workspace/Users/user@example.com/.conversations/space-optimizer/runs/p-control/"
        "positive_controls/local_checkpoint/checkpoint"
    )
    assert "--serving-endpoint" not in params


def test_submit_run_positive_control_sets_workspace_patch_path(monkeypatch, tmp_path: Path):
    _patch_workspace_client(monkeypatch)
    patch_dir = tmp_path / "patch"
    patch_dir.mkdir()
    (patch_dir / "patch_sequence.json").write_text(
        json.dumps(
            {
                "schema_version": "maxgenie.artifact_patch_sequence.v1",
                "payloads": [
                    {
                        "candidate_name": "one",
                        "candidate_mode": "artifact_patch_v2",
                        "file_edits": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "positive_control.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "local_patch",
                "patch_path": str(patch_dir),
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--workspace-root",
            "/Workspace/Users/user@example.com/.conversations/space-optimizer/runs/p-control",
            "--positive-control",
            str(spec_path),
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    params = payload["tasks"][0]["spark_python_task"]["parameters"]
    patch_path = params[params.index("--audit-patch-path") + 1]
    assert patch_path == (
        "/Workspace/Users/user@example.com/.conversations/space-optimizer/runs/p-control/"
        "positive_controls/local_patch/patch"
    )
    assert "--audit-checkpoint-path" not in params


def test_submit_run_wait_reports_databricks_polling_error_with_last_status(monkeypatch):
    _patch_workspace_client(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "submit_workspace_run",
        lambda *args, **kwargs: {
            "run_id": 123,
            "run_page_url": "https://dbc.example.com/#job/1/run/123",
            "state": {"life_cycle_state": "RUNNING"},
        },
    )

    def fail_wait(*args, **kwargs):
        on_tick = kwargs["on_tick"]
        on_tick(
            {
                "run_id": 123,
                "run_page_url": "https://dbc.example.com/#job/1/run/123",
                "state": {"life_cycle_state": "RUNNING"},
                "tasks": [],
            }
        )
        raise PermissionDenied("Source IP address is blocked by workspace IP ACL")

    monkeypatch.setattr(cli_module, "wait_for_run", fail_wait)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "submit-run",
            "--space-url",
            SPACE_URL,
            "--serving-endpoint",
            "candidate-endpoint",
            "--serving-model",
            "databricks-gpt-5-5",
            "--serving-reasoning-effort",
            "xhigh",
            "--warehouse-id",
            "warehouse-123",
            "--fresh-baseline",
            "--wait",
        ],
    )

    assert result.exit_code == 2
    assert "Databricks API error while polling submitted run 123" in result.output
    assert "Source IP address is blocked by workspace IP ACL" in result.output
    assert "Last known run status before polling failed" in result.output
    assert "Run `123` is `RUNNING`" in result.output


def test_run_status_reports_databricks_api_error_without_traceback(monkeypatch):
    _patch_workspace_client(monkeypatch)

    def fail_status(*args, **kwargs):
        raise PermissionDenied("Source IP address is blocked by workspace IP ACL")

    monkeypatch.setattr(cli_module, "get_run_status", fail_status)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "run-status",
            "123",
            "--space-url",
            SPACE_URL,
            "--chat",
        ],
    )

    assert result.exit_code == 2
    assert "Databricks API error while fetching run 123 status" in result.output
    assert "Source IP address is blocked by workspace IP ACL" in result.output
    assert "Traceback" not in result.output
