import json
from pathlib import Path

import pytest

from maxgenie.positive_control import (
    EXAMPLE_CHECKPOINT_CONTROL_NAME,
    EXAMPLE_PATCH_CONTROL_NAME,
    POSITIVE_CONTROL_SCHEMA_VERSION,
    PositiveControlSpec,
    load_positive_control_spec,
    materialize_positive_control_patch_sequence,
    resolve_positive_control,
    summarize_positive_control,
)


def test_json_spec_resolves_relative_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "winner"
    checkpoint.mkdir(parents=True)
    spec_path = tmp_path / "control.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "local_winner",
                "description": "local checkpoint",
                "checkpoint_path": "checkpoints/winner",
                "required_protocol": {"comparison_mode": "result_based"},
                "expected_scores": {"full_pass_rate": 60.0},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_positive_control(spec_path)

    assert resolved.available is True
    assert resolved.audit_target_kind == "checkpoint"
    assert resolved.audit_target_path == checkpoint
    assert resolved.spec.required_protocol == {"comparison_mode": "result_based"}
    assert resolved.spec.expected_scores == {"full_pass_rate": 60.0}


def test_load_positive_control_spec_validates_name(tmp_path: Path):
    spec_path = tmp_path / "invalid.json"
    spec_path.write_text(json.dumps({"checkpoint_path": "checkpoint"}), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a non-empty 'name'"):
        load_positive_control_spec(spec_path)


def test_example_control_is_available_when_local_artifact_exists(tmp_path: Path):
    checkpoint = (
        tmp_path
        / "examples"
        / "positive_controls"
        / "00000000000000000000000000000000"
        / "checkpoints"
        / "winner"
    )
    checkpoint.mkdir(parents=True)

    summary = summarize_positive_control(EXAMPLE_CHECKPOINT_CONTROL_NAME, base_dir=tmp_path)

    assert summary["schema_version"] == POSITIVE_CONTROL_SCHEMA_VERSION
    assert summary["available"] is True
    assert summary["audit_target_kind"] == "checkpoint"
    assert summary["audit_target_path"] == str(checkpoint)
    assert summary["required_protocol"]["comparison_mode"] == "result_based"
    assert summary["required_protocol"]["required_result_coverage"] == "15/15"
    assert summary["expected_scores"]["holdout_pass_rate"] == 0.0


def test_example_control_gracefully_reports_missing_artifact(tmp_path: Path):
    summary = summarize_positive_control(EXAMPLE_CHECKPOINT_CONTROL_NAME, base_dir=tmp_path)

    assert summary["available"] is False
    assert summary["status"] == "unavailable"
    assert summary["audit_target_kind"] == "checkpoint"
    assert summary["audit_target_path"].endswith("checkpoints/winner")
    assert summary["missing_paths"]
    assert summary["required_protocol"]["hidden_holdout_policy"].startswith("audit only")


def test_example_patch_control_materializes_strict_sequence(tmp_path: Path):
    root = (
        tmp_path
        / "examples"
        / "positive_controls"
        / "00000000000000000000000000000000"
    )
    baseline_space = root / "checkpoints" / "baseline" / "space"
    winner_space = root / "checkpoints" / "winner" / "space"
    for space in (baseline_space, winner_space):
        (space / "instructions" / "examples").mkdir(parents=True)
        (space / "tables").mkdir()
    (baseline_space / "instructions" / "text_instruction.md").write_text("old\n", encoding="utf-8")
    (winner_space / "instructions" / "text_instruction.md").write_text("new\n", encoding="utf-8")
    (winner_space / "instructions" / "examples" / "a.yml").write_text("id: a\n", encoding="utf-8")
    (winner_space / "tables" / "sales.yml").write_text("identifier: sales\n", encoding="utf-8")

    resolved = resolve_positive_control(EXAMPLE_PATCH_CONTROL_NAME, base_dir=tmp_path)
    materialized = materialize_positive_control_patch_sequence(
        resolved,
        destination_dir=tmp_path / "materialized_patch",
    )
    payload = json.loads((materialized / "patch_sequence.json").read_text(encoding="utf-8"))

    assert resolved.audit_target_kind == "patch"
    assert payload["candidate_mode"] == "artifact_patch_v2"
    assert payload["file_edit_count"] == 3
    assert all(len(item["file_edits"]) <= 2 for item in payload["payloads"])
    assert payload["payloads"][0]["candidate_name"].startswith(EXAMPLE_PATCH_CONTROL_NAME)


def test_spec_object_can_resolve_patch_target(tmp_path: Path):
    patch_dir = tmp_path / "patches" / "candidate"
    patch_dir.mkdir(parents=True)
    spec = PositiveControlSpec(
        name="patch_control",
        patch_path="patches/candidate",
        required_protocol={"split": "fixed"},
    )

    resolved = resolve_positive_control(spec, base_dir=tmp_path)

    assert resolved.available is True
    assert resolved.audit_target_kind == "patch"
    assert summarize_positive_control(resolved)["patch_path"] == str(patch_dir)
