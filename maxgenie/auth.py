"""Authentication helpers for Databricks workspace connectivity."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

from databricks.sdk import WorkspaceClient


class AuthConfigurationError(RuntimeError):
    """Raised when Databricks authentication cannot be resolved."""


@dataclass(frozen=True)
class AuthDiagnostics:
    """Diagnostic outcome for auth readiness checks."""

    ok: bool
    mode: str
    message: str


def _normalize_host(host: str | None) -> str | None:
    if not host:
        return None
    host = host.strip()
    if host.startswith("https://") or host.startswith("http://"):
        return host
    return f"https://{host}"


def _databricks_config_path() -> Path:
    return Path.home() / ".databrickscfg"


def _resolve_profile_for_host(host: str | None) -> str | None:
    normalized_host = _normalize_host(host)
    if not normalized_host:
        return None

    config_path = _databricks_config_path()
    if not config_path.exists():
        return None

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(config_path)
    except configparser.Error:
        return None

    for section in parser.sections():
        configured_host = _normalize_host(parser.get(section, "host", fallback=None))
        if configured_host == normalized_host:
            return section
    return None


def create_workspace_client(profile: str | None = None, host: str | None = None) -> WorkspaceClient:
    """Create authenticated WorkspaceClient with profile-first fallback."""
    normalized_host = _normalize_host(host)
    if profile:
        try:
            if normalized_host:
                return WorkspaceClient(profile=profile, host=normalized_host)
            return WorkspaceClient(profile=profile)
        except Exception as exc:  # pragma: no cover - API client error details are runtime-only
            raise AuthConfigurationError(
                f"Failed to authenticate with profile '{profile}': {exc}"
            ) from exc

    has_env_host_token = bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))
    matched_profile = _resolve_profile_for_host(normalized_host) if not has_env_host_token else None
    if matched_profile:
        try:
            return WorkspaceClient(profile=matched_profile, host=normalized_host)
        except Exception as exc:  # pragma: no cover - API client error details are runtime-only
            raise AuthConfigurationError(
                f"Failed to authenticate with profile '{matched_profile}' matched from host '{normalized_host}': {exc}"
            ) from exc

    if normalized_host and not os.getenv("DATABRICKS_HOST"):
        os.environ["DATABRICKS_HOST"] = normalized_host

    try:
        return WorkspaceClient()
    except Exception as exc:  # pragma: no cover - API client error details are runtime-only
        setup_msg = (
            "Databricks authentication is not configured. "
            "Provide --profile <name> or set DATABRICKS_HOST and DATABRICKS_TOKEN. "
            "You can also set DATABRICKS_CONFIG_PROFILE in your environment."
        )
        raise AuthConfigurationError(f"{setup_msg} Original error: {exc}") from exc


def diagnose_auth(profile: str | None = None, host: str | None = None) -> AuthDiagnostics:
    """Check whether auth is likely configured and report the selected mode."""
    if profile:
        try:
            create_workspace_client(profile=profile, host=host)
            return AuthDiagnostics(ok=True, mode=f"profile:{profile}", message="Profile authentication works.")
        except Exception as exc:
            return AuthDiagnostics(ok=False, mode=f"profile:{profile}", message=str(exc))

    normalized_host = _normalize_host(host)
    env_host = os.getenv("DATABRICKS_HOST") or normalized_host
    env_token = os.getenv("DATABRICKS_TOKEN")
    if env_host and env_token:
        try:
            create_workspace_client(profile=None, host=host)
            return AuthDiagnostics(ok=True, mode="env:host_token", message="Host/token authentication works.")
        except Exception as exc:
            return AuthDiagnostics(ok=False, mode="env:host_token", message=str(exc))

    matched_profile = _resolve_profile_for_host(normalized_host)
    if matched_profile:
        try:
            create_workspace_client(profile=matched_profile, host=host)
            return AuthDiagnostics(
                ok=True,
                mode=f"host_profile:{matched_profile}",
                message="Matched workspace host to profile authentication.",
            )
        except Exception as exc:
            return AuthDiagnostics(ok=False, mode=f"host_profile:{matched_profile}", message=str(exc))

    env_profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    if env_profile:
        try:
            create_workspace_client(profile=env_profile, host=host)
            return AuthDiagnostics(ok=True, mode=f"env_profile:{env_profile}", message="Env profile authentication works.")
        except Exception as exc:
            return AuthDiagnostics(ok=False, mode=f"env_profile:{env_profile}", message=str(exc))

    return AuthDiagnostics(
        ok=False,
        mode="none",
        message=(
            "No Databricks auth detected. Set --profile, DATABRICKS_CONFIG_PROFILE, or "
            "both DATABRICKS_HOST and DATABRICKS_TOKEN."
        ),
    )
