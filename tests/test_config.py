from pathlib import Path

import pytest

from maxgenie.config import (
    AUTONOMOUS_STRATEGY_CENTAUR,
    AUTONOMOUS_STRATEGY_SERVING,
    default_serving_endpoint,
    default_serving_model,
    default_serving_reasoning_effort,
    default_serving_temperature,
    default_space_url,
    default_workspace_root_for_strategy,
    normalize_autonomous_strategy,
    parse_space_url,
    resolve_space_url,
)


def test_parse_space_url_extracts_space_id_and_host():
    ref = parse_space_url("https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    assert ref.host == "dbc-12345.cloud.databricks.com"
    assert ref.space_id == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_parse_space_url_raises_on_missing_id():
    with pytest.raises(ValueError, match="Could not find Genie space_id"):
        parse_space_url("https://dbc-12345.cloud.databricks.com/genie/rooms")


def test_resolve_space_url_prefers_cli_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAXGENIE_DEFAULT_SPACE_URL", "https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

    resolved = resolve_space_url("https://dbc-67890.cloud.databricks.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert resolved == "https://dbc-67890.cloud.databricks.com/genie/rooms/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_resolve_space_url_uses_default_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAXGENIE_DEFAULT_SPACE_URL", "https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

    assert default_space_url() == "https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert resolve_space_url(None) == "https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_resolve_space_url_raises_without_cli_or_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MAXGENIE_DEFAULT_SPACE_URL", raising=False)
    monkeypatch.delenv("MAXGENIE_TEST_SPACE_URL", raising=False)
    monkeypatch.chdir(Path("/tmp"))

    with pytest.raises(ValueError, match="No space URL provided"):
        resolve_space_url(None)


def test_default_space_url_reads_repo_local_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MAXGENIE_DEFAULT_SPACE_URL", raising=False)
    monkeypatch.delenv("MAXGENIE_TEST_SPACE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "MAXGENIE_DEFAULT_SPACE_URL=https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert default_space_url() == "https://dbc-12345.cloud.databricks.com/genie/rooms/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_normalize_autonomous_strategy_accepts_aliases():
    assert normalize_autonomous_strategy("centaur") == AUTONOMOUS_STRATEGY_CENTAUR
    assert normalize_autonomous_strategy("serving") == AUTONOMOUS_STRATEGY_SERVING
    assert normalize_autonomous_strategy(AUTONOMOUS_STRATEGY_CENTAUR) == AUTONOMOUS_STRATEGY_CENTAUR
    assert normalize_autonomous_strategy(AUTONOMOUS_STRATEGY_SERVING) == AUTONOMOUS_STRATEGY_SERVING


def test_normalize_autonomous_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        normalize_autonomous_strategy("random")


def test_default_workspace_root_for_strategy_uses_strategy_specific_directories(tmp_path):
    assert default_workspace_root_for_strategy(
        AUTONOMOUS_STRATEGY_CENTAUR,
        base_dir=tmp_path,
    ) == tmp_path / "centaur"
    assert default_workspace_root_for_strategy(
        AUTONOMOUS_STRATEGY_SERVING,
        base_dir=tmp_path,
    ) == tmp_path / "serving"


def test_default_serving_settings_read_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAXGENIE_SERVING_ENDPOINT", "candidate-endpoint")
    monkeypatch.setenv("MAXGENIE_SERVING_MODEL", "candidate-model")
    monkeypatch.setenv("MAXGENIE_SERVING_TEMPERATURE", "0")
    monkeypatch.setenv("MAXGENIE_SERVING_REASONING_EFFORT", "xhigh")

    assert default_serving_endpoint() == "candidate-endpoint"
    assert default_serving_model() == "candidate-model"
    assert default_serving_temperature() == 0.0
    assert default_serving_reasoning_effort() == "xhigh"


def test_default_serving_reasoning_effort_is_low(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MAXGENIE_SERVING_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("MAXGENIE_CANDIDATE_REASONING_EFFORT", raising=False)
    monkeypatch.chdir(Path("/tmp"))

    assert default_serving_reasoning_effort() == "low"


def test_default_serving_temperature_can_be_omitted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAXGENIE_SERVING_TEMPERATURE", "omit")

    assert default_serving_temperature() is None
