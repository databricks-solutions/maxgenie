import importlib.util
import sys
import types
from pathlib import Path


def _load_sync_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "genie_skills"
        / "maxgenie"
        / "scripts"
        / "sync_to_workspace.py"
    )
    spec = importlib.util.spec_from_file_location("sync_to_workspace", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_sync_uses_stable_bundle_name_and_workspace_bundle_root():
    sync_module = _load_sync_module()

    bundle_name = sync_module.derived_bundle_name(sync_module.DEFAULT_DESTINATION)
    bundle_root = sync_module.bundle_root_path(sync_module.DEFAULT_DESTINATION, bundle_name)

    assert bundle_name == "workspace-optimizer-skill"
    assert bundle_root == "/Workspace/.bundle/workspace-optimizer-skill"


def test_custom_user_sync_destination_gets_isolated_bundle_state():
    sync_module = _load_sync_module()
    destination = (
        "/Workspace/Users/user@example.com/.assistant/skills/"
        "maxgenie-preview"
    )

    bundle_name = sync_module.derived_bundle_name(destination)
    bundle_root = sync_module.bundle_root_path(destination, bundle_name)

    assert bundle_name.startswith("workspace-optimizer-skill-")
    assert bundle_name != sync_module.BUNDLE_NAME
    assert bundle_root == f"/Workspace/Users/user@example.com/.bundle/{bundle_name}"


def test_workspace_files_upload_uses_workspace_api_paths(tmp_path, monkeypatch):
    sync_module = _load_sync_module()
    payload = tmp_path / "payload"
    (payload / "scripts").mkdir(parents=True)
    (payload / "SKILL.md").write_text("skill", encoding="utf-8")
    (payload / "databricks.yml").write_text("bundle config", encoding="utf-8")
    (payload / "scripts" / "runner.py").write_text("print('ok')", encoding="utf-8")
    mkdirs: list[str] = []
    imports: list[tuple[str, str, bool]] = []

    class _Workspace:
        def mkdirs(self, path):
            mkdirs.append(path)

        def import_(self, path, *, content, format, overwrite):
            imports.append((path, format, overwrite))

    class _Client:
        workspace = _Workspace()

    fake_workspace_service = types.SimpleNamespace(
        ImportFormat=types.SimpleNamespace(RAW="RAW")
    )
    monkeypatch.setitem(
        sys.modules,
        "databricks.sdk.service.workspace",
        fake_workspace_service,
    )
    monkeypatch.setitem(
        sys.modules,
        "databricks.sdk.service",
        types.SimpleNamespace(workspace=fake_workspace_service),
    )

    uploaded = sync_module.upload_payload_with_workspace_client(
        _Client(),
        payload=payload,
        destination="/Workspace/Users/user@example.com/.conversations/runner",
    )

    assert "/Users/user@example.com/.conversations/runner" in mkdirs
    assert "/Users/user@example.com/.conversations/runner/scripts" in mkdirs
    assert {
        path
        for path, _format, _overwrite in imports
    } == {
        "/Users/user@example.com/.conversations/runner/SKILL.md",
        "/Users/user@example.com/.conversations/runner/scripts/runner.py",
    }
    assert all("databricks.yml" not in path for path, _format, _overwrite in imports)
    assert all(item[1:] == ("RAW", True) for item in imports)
    assert uploaded == [
        "/Workspace/Users/user@example.com/.conversations/runner/SKILL.md",
        "/Workspace/Users/user@example.com/.conversations/runner/scripts/runner.py",
    ]


def test_auto_sync_uses_workspace_files_when_cli_is_missing(tmp_path, monkeypatch):
    sync_module = _load_sync_module()
    payload = tmp_path / "payload"
    payload.mkdir()
    calls: list[dict] = []
    monkeypatch.setattr(sync_module.shutil, "which", lambda name: None)

    def fake_workspace_files_sync(payload_arg, **kwargs):
        calls.append({"payload": payload_arg, **kwargs})
        return 0

    monkeypatch.setattr(
        sync_module,
        "sync_payload_with_workspace_files",
        fake_workspace_files_sync,
    )

    exit_code = sync_module.sync_payload(
        payload,
        destination="/Workspace/Users/user@example.com/.conversations/runner",
        profile="my-profile",
        host="https://example.cloud.databricks.com",
        dry_run=False,
    )

    assert exit_code == 0
    assert calls == [
        {
            "payload": payload,
            "destination": "/Workspace/Users/user@example.com/.conversations/runner",
            "profile": "my-profile",
            "host": "https://example.cloud.databricks.com",
            "dry_run": False,
        }
    ]
