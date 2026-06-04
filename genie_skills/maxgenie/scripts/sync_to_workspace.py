"""Build and sync the workspace skill payload to Databricks workspace files."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Literal


DEFAULT_DESTINATION = "/Workspace/.assistant/skills/maxgenie"
PAYLOAD_NAME = "skill"
BUNDLE_NAME = "workspace-optimizer-skill"
SyncMethod = Literal["auto", "bundle", "workspace-files"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def skill_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def build_payload(build_dir: Path) -> Path:
    root = repo_root()
    source = skill_source_root()
    payload = build_dir / PAYLOAD_NAME
    if payload.exists():
        shutil.rmtree(payload)
    payload.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source / "SKILL.md", payload / "SKILL.md")
    (payload / "BUILD_INFO.json").write_text(
        json.dumps(build_info(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copytree(
        source / "scripts",
        payload / "scripts",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "sync_to_workspace.py"),
    )
    if (source / "agents").exists():
        copy_tree(source / "agents", payload / "agents")

    lib_dir = payload / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    copy_tree(root / "maxgenie", lib_dir / "engine")
    copy_tree(root / "templates", lib_dir / "templates")
    return payload


def _git_text(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def build_info(root: Path) -> dict[str, Any]:
    status = _git_text(root, "status", "--short")
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": _git_text(root, "rev-parse", "HEAD"),
        "branch": _git_text(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "changed_files": status.splitlines() if status else [],
    }


def derived_bundle_name(destination: str, explicit_bundle_name: str | None = None) -> str:
    if explicit_bundle_name:
        return explicit_bundle_name
    if destination == DEFAULT_DESTINATION:
        return BUNDLE_NAME
    destination_hash = hashlib.sha1(destination.encode("utf-8")).hexdigest()[:12]
    return f"{BUNDLE_NAME}-{destination_hash}"


def bundle_root_path(destination: str, bundle_name: str) -> str:
    parts = [part for part in destination.split("/") if part]
    if len(parts) >= 3 and parts[0] == "Workspace" and parts[1] == "Users":
        return f"/Workspace/Users/{parts[2]}/.bundle/{bundle_name}"
    return f"/Workspace/.bundle/{bundle_name}"


def write_bundle_config(
    payload: Path,
    *,
    destination: str,
    host: str,
    bundle_name: str | None = None,
) -> None:
    resolved_bundle_name = derived_bundle_name(destination, bundle_name)
    bundle_root = bundle_root_path(destination, resolved_bundle_name)
    bundle_config = f"""bundle:
  name: {resolved_bundle_name}

workspace:
  host: {host}
  root_path: {bundle_root}
  file_path: {destination}
  state_path: {bundle_root}/state

sync:
  include:
    - BUILD_INFO.json
    - SKILL.md
    - agents/**
    - scripts/**
    - lib/**
  exclude:
    - databricks.yml
"""
    (payload / "databricks.yml").write_text(bundle_config, encoding="utf-8")


def sync_payload(
    payload: Path,
    *,
    destination: str,
    profile: str | None,
    host: str,
    dry_run: bool,
    bundle_name: str | None = None,
    sync_method: SyncMethod = "auto",
) -> int:
    write_bundle_config(payload, destination=destination, host=host, bundle_name=bundle_name)
    if sync_method not in {"auto", "bundle", "workspace-files"}:
        raise ValueError("sync_method must be one of: auto, bundle, workspace-files")
    if sync_method == "workspace-files" or (
        sync_method == "auto" and shutil.which("databricks") is None
    ):
        return sync_payload_with_workspace_files(
            payload,
            destination=destination,
            profile=profile,
            host=host,
            dry_run=dry_run,
        )

    command = ["databricks", "bundle", "validate" if dry_run else "sync"]
    if profile:
        command.extend(["--profile", profile])
    if dry_run:
        command.extend(["--output", "json"])
    else:
        command.append("--full")
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=payload, check=False)
    return int(completed.returncode)


def sync_payload_with_workspace_files(
    payload: Path,
    *,
    destination: str,
    profile: str | None,
    host: str,
    dry_run: bool,
) -> int:
    """Upload the built payload through the Workspace API.

    This path is intentionally independent of the Databricks CLI so workspace
    job runners can be synced from minimal local executor environments.
    """
    if dry_run:
        preview = {
            "method": "workspace-files",
            "destination": destination,
            "file_count": len([path for path in payload.rglob("*") if path.is_file()]),
        }
        print(json.dumps(preview, indent=2), flush=True)
        return 0

    from databricks.sdk import WorkspaceClient

    wc = WorkspaceClient(profile=profile, host=host)
    uploaded = upload_payload_with_workspace_client(
        wc,
        payload=payload,
        destination=destination,
    )
    print(
        f"Uploaded {len(uploaded)} files to {destination} using Workspace API.",
        flush=True,
    )
    return 0


def upload_payload_with_workspace_client(
    wc: Any,
    *,
    payload: Path,
    destination: str,
) -> list[str]:
    """Upload a payload directory to workspace files and return uploaded paths."""
    source = payload.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"payload must be an existing directory: {source}")
    destination_root = destination.rstrip("/")
    if not destination_root.startswith("/Workspace/"):
        raise ValueError("destination must be an absolute /Workspace path.")

    from databricks.sdk.service import workspace as workspace_service

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
        if api_directory in created_dirs:
            continue
        wc.workspace.mkdirs(api_directory)
        created_dirs.add(api_directory)

    for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = file_path.relative_to(source).as_posix()
        if relative == "databricks.yml":
            continue
        workspace_path = f"{destination_root}/{relative}"
        content = base64.b64encode(file_path.read_bytes()).decode("ascii")
        wc.workspace.import_(
            _workspace_object_api_path(workspace_path),
            content=content,
            format=workspace_service.ImportFormat.RAW,
            overwrite=True,
        )
        uploaded.append(workspace_path)
    return uploaded


def _workspace_object_api_path(path: str) -> str:
    if path == "/Workspace":
        return "/"
    if path.startswith("/Workspace/"):
        return path[len("/Workspace") :]
    if path.startswith(("/Users/", "/Shared/")):
        return path
    raise ValueError("workspace object path must be under /Workspace, /Users, or /Shared.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync the workspace skill to Databricks.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument("--build-dir", default=str(repo_root() / ".maxgenie-build" / "workspace-skill-build"))
    parser.add_argument("--bundle-name", default=None)
    parser.add_argument(
        "--sync-method",
        choices=("auto", "bundle", "workspace-files"),
        default="auto",
        help="Use Databricks bundle sync, Workspace API upload, or auto fallback.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.host:
        print("Provide --host or set DATABRICKS_HOST.", file=sys.stderr)
        return 2
    payload = build_payload(Path(args.build_dir).expanduser().resolve())
    return sync_payload(
        payload,
        destination=args.destination,
        profile=args.profile,
        host=args.host,
        dry_run=args.dry_run,
        bundle_name=args.bundle_name,
        sync_method=args.sync_method,
    )


if __name__ == "__main__":
    raise SystemExit(main())
