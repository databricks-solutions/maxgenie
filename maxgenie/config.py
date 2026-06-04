"""Configuration and URL parsing for MaxGenie."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


SPACE_ID_RE = re.compile(r"[0-9a-f]{32}")
DEFAULT_SPACE_URL_ENV_VARS: tuple[str, ...] = (
    "MAXGENIE_DEFAULT_SPACE_URL",
    "MAXGENIE_TEST_SPACE_URL",
)
AUTONOMOUS_STRATEGY_CENTAUR = "centaur"
AUTONOMOUS_STRATEGY_SERVING = "serving_autonomous"
DEFAULT_AUTONOMOUS_STRATEGY = AUTONOMOUS_STRATEGY_CENTAUR
SUPPORTED_AUTONOMOUS_STRATEGIES: frozenset[str] = frozenset(
    {
        AUTONOMOUS_STRATEGY_CENTAUR,
        AUTONOMOUS_STRATEGY_SERVING,
        "centaur",
        "serving",
    }
)

STRATEGY_WORKSPACE_ROOT_NAMES: dict[str, str] = {
    AUTONOMOUS_STRATEGY_CENTAUR: "centaur",
    AUTONOMOUS_STRATEGY_SERVING: "serving",
}
DEFAULT_SERVING_ENDPOINT_ENV_VARS: tuple[str, ...] = (
    "MAXGENIE_SERVING_ENDPOINT",
    "MAXGENIE_CANDIDATE_ENDPOINT",
)
DEFAULT_SERVING_MODEL_ENV_VARS: tuple[str, ...] = (
    "MAXGENIE_SERVING_MODEL",
    "MAXGENIE_CANDIDATE_MODEL",
)
DEFAULT_SERVING_TEMPERATURE_ENV_VARS: tuple[str, ...] = (
    "MAXGENIE_SERVING_TEMPERATURE",
    "MAXGENIE_CANDIDATE_TEMPERATURE",
)
DEFAULT_SERVING_REASONING_EFFORT_ENV_VARS: tuple[str, ...] = (
    "MAXGENIE_SERVING_REASONING_EFFORT",
    "MAXGENIE_CANDIDATE_REASONING_EFFORT",
)

@dataclass(frozen=True)
class ParsedSpaceRef:
    """Normalized Genie space reference parsed from URL."""

    url: str
    host: str
    space_id: str


@dataclass(frozen=True)
class OptimizeSettings:
    """Runtime settings for autonomous optimization."""

    delay_seconds: float = 0.5
    match_threshold: float = 0.85
    max_iterations: int = 30
    plateau_loops: int = 3
    full_run_every_accepted: int = 3
    test_fraction: float = 0.2
    split_seed: int = 42
    serving_model: str | None = None
    use_serving_policy: bool = True
    warehouse_id: str | None = None
    kfold: int = 0


def parse_space_url(space_url: str) -> ParsedSpaceRef:
    """Parse a Genie Space URL and extract host + space_id.

    Accepted examples:
    - https://<host>/genie/rooms/<space_id>
    - https://<host>/genie/spaces/<space_id>
    - any URL containing a 32-char lowercase hex space id in path/query/fragment
    """
    parsed = urlparse(space_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            "Invalid space URL. Expected full URL like "
            "https://<workspace-host>/genie/rooms/<space_id>."
        )

    text = " ".join([parsed.path or "", parsed.query or "", parsed.fragment or ""])
    match = SPACE_ID_RE.search(text)
    if not match:
        raise ValueError(
            "Could not find Genie space_id in URL. "
            "Expected a 32-char lowercase hex id in path/query."
        )

    return ParsedSpaceRef(url=space_url, host=parsed.netloc, space_id=match.group(0))


def default_space_url() -> str | None:
    """Return the configured default Genie Space URL from the environment, if any."""
    for env_var in DEFAULT_SPACE_URL_ENV_VARS:
        value = os.getenv(env_var)
        if value and value.strip():
            return value.strip()

    for directory in (Path.cwd(), *Path.cwd().parents):
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() not in DEFAULT_SPACE_URL_ENV_VARS:
                continue
            value = raw_value.strip().strip("'").strip('"')
            if value:
                return value
    return None


def _default_env_value(env_vars: tuple[str, ...]) -> str | None:
    for env_var in env_vars:
        value = os.getenv(env_var)
        if value and value.strip():
            return value.strip()

    for directory in (Path.cwd(), *Path.cwd().parents):
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() not in env_vars:
                continue
            value = raw_value.strip().strip("'").strip('"')
            if value:
                return value
    return None


def default_serving_endpoint() -> str | None:
    """Return the configured workspace serving endpoint for candidate proposals.

    The workspace resolver discovers the preferred ready endpoint when no environment
    override is configured, so non-serving runs never get a serving endpoint injected.
    """
    return _default_env_value(DEFAULT_SERVING_ENDPOINT_ENV_VARS)


def default_serving_model() -> str | None:
    """Return the configured serving model override, if any."""
    return _default_env_value(DEFAULT_SERVING_MODEL_ENV_VARS)


def default_serving_temperature() -> float | None:
    """Return the configured serving temperature, if any."""
    value = _default_env_value(DEFAULT_SERVING_TEMPERATURE_ENV_VARS)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"none", "null", "omit", "unset"}:
        return None
    try:
        temperature = float(normalized)
    except ValueError as exc:
        raise ValueError(
            "MAXGENIE_SERVING_TEMPERATURE must be a number between 0.0 and 2.0, or one of: none, null, omit, unset."
        ) from exc
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("MAXGENIE_SERVING_TEMPERATURE must be between 0.0 and 2.0.")
    return temperature


def default_serving_reasoning_effort() -> str:
    """Return the configured serving reasoning/thinking effort.

    Some endpoints map this to ``output_config.effort`` while others use
    ``reasoning_effort``. The default is intentionally low because the canonical
    reasoning-effort sweep showed no demo-score gain from xhigh over low, while
    xhigh materially increased runtime.
    """
    return _default_env_value(DEFAULT_SERVING_REASONING_EFFORT_ENV_VARS) or "low"


def normalize_autonomous_strategy(strategy: str) -> str:
    """Normalize strategy aliases to a canonical autonomous strategy name."""
    lowered = strategy.strip().lower()
    alias_map = {
        "centaur": AUTONOMOUS_STRATEGY_CENTAUR,
        "serving": AUTONOMOUS_STRATEGY_SERVING,
        AUTONOMOUS_STRATEGY_CENTAUR: AUTONOMOUS_STRATEGY_CENTAUR,
        AUTONOMOUS_STRATEGY_SERVING: AUTONOMOUS_STRATEGY_SERVING,
    }
    try:
        return alias_map[lowered]
    except KeyError as exc:
        raise ValueError(
            "Unsupported strategy. Choose one of "
            f"{', '.join(sorted(SUPPORTED_AUTONOMOUS_STRATEGIES))}."
        ) from exc


def default_workspace_root_for_strategy(
    strategy: str,
    *,
    base_dir: str | Path = ".",
) -> Path:
    """Return the strategy-specific default workspace root."""
    normalized_strategy = normalize_autonomous_strategy(strategy)
    try:
        root_name = STRATEGY_WORKSPACE_ROOT_NAMES[normalized_strategy]
    except KeyError as exc:
        raise ValueError(f"Unsupported strategy for workspace root: {strategy}") from exc
    return Path(base_dir).expanduser().resolve() / root_name


def resolve_space_url(space_url: str | None) -> str:
    """Resolve a Genie Space URL from CLI input or repo-local environment."""
    if space_url and space_url.strip():
        return space_url.strip()

    resolved_default = default_space_url()
    if resolved_default:
        return resolved_default

    raise ValueError(
        "No space URL provided. Use --space-url or set MAXGENIE_DEFAULT_SPACE_URL "
        "(MAXGENIE_TEST_SPACE_URL is also supported)."
    )


def default_workspace_dir(root: str | Path, space_id: str) -> Path:
    """Compute workspace directory for a given space id."""
    return Path(root).expanduser().resolve() / space_id
