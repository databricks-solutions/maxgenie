from maxgenie.auth import create_workspace_client, diagnose_auth


def test_create_workspace_client_passes_host_with_profile(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_workspace_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("maxgenie.auth.WorkspaceClient", _fake_workspace_client)

    create_workspace_client(
        profile="dev",
        host="https://dbc-test.cloud.databricks.com",
    )

    assert captured["profile"] == "dev"
    assert captured["host"] == "https://dbc-test.cloud.databricks.com"


def test_create_workspace_client_uses_profile_only_when_host_absent(monkeypatch):
    captured: dict[str, str] = {}

    def _fake_workspace_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("maxgenie.auth.WorkspaceClient", _fake_workspace_client)

    create_workspace_client(profile="dev")

    assert captured == {"profile": "dev"}


def test_create_workspace_client_uses_matching_profile_for_host(monkeypatch, tmp_path):
    captured: dict[str, str] = {}
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[workspace-dev]\n"
        "host = https://dbc-test.cloud.databricks.com\n"
        "\n"
        "[dev]\n"
        "host = https://other-workspace.cloud.databricks.com\n"
    )

    def _fake_workspace_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setattr("maxgenie.auth._databricks_config_path", lambda: config_path)
    monkeypatch.setattr("maxgenie.auth.WorkspaceClient", _fake_workspace_client)

    create_workspace_client(host="https://dbc-test.cloud.databricks.com")

    assert captured == {
        "profile": "workspace-dev",
        "host": "https://dbc-test.cloud.databricks.com",
    }


def test_create_workspace_client_prefers_host_token_env_over_matched_profile(monkeypatch, tmp_path):
    captured: dict[str, str] = {}
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[workspace-dev]\n"
        "host = https://dbc-test.cloud.databricks.com\n"
    )

    def _fake_workspace_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("DATABRICKS_HOST", "https://already-set.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")
    monkeypatch.setattr("maxgenie.auth._databricks_config_path", lambda: config_path)
    monkeypatch.setattr("maxgenie.auth.WorkspaceClient", _fake_workspace_client)

    create_workspace_client(host="https://dbc-test.cloud.databricks.com")

    assert captured == {}


def test_diagnose_auth_reports_host_matched_profile(monkeypatch, tmp_path):
    config_path = tmp_path / ".databrickscfg"
    config_path.write_text(
        "[workspace-dev]\n"
        "host = https://dbc-test.cloud.databricks.com\n"
    )

    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.setattr("maxgenie.auth._databricks_config_path", lambda: config_path)
    monkeypatch.setattr("maxgenie.auth.create_workspace_client", lambda profile=None, host=None: object())

    diagnostics = diagnose_auth(host="https://dbc-test.cloud.databricks.com")

    assert diagnostics.ok is True
    assert diagnostics.mode == "host_profile:workspace-dev"
