"""Positive-control helpers for comparable MaxGenie audits.

This module is intentionally local and side-effect free. It does not submit,
poll, or inspect Databricks jobs; it only resolves named controls to local
checkpoint or patch artifacts that another workflow can audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from maxgenie.artifact_patch import (
    build_artifact_patch_sequence_from_space_dirs,
    write_artifact_patch_sequence,
)


POSITIVE_CONTROL_SCHEMA_VERSION = "maxgenie.positive_control.v1"
EXAMPLE_CHECKPOINT_CONTROL_NAME = "example_checkpoint_control"
EXAMPLE_PATCH_CONTROL_NAME = "example_patch_control"
_EXAMPLE_SPACE_ID = "00000000000000000000000000000000"
_EXAMPLE_CONTROL_ROOT = f"examples/positive_controls/{_EXAMPLE_SPACE_ID}"
_EXAMPLE_BASELINE_CHECKPOINT = f"{_EXAMPLE_CONTROL_ROOT}/checkpoints/baseline"
_EXAMPLE_WINNER_CHECKPOINT = f"{_EXAMPLE_CONTROL_ROOT}/checkpoints/winner"


def canonical_required_protocol() -> dict[str, Any]:
    """Return the protocol required for a production-comparable audit."""

    return {
        "baseline": "fresh source-space baseline",
        "split": "fixed canonical train/validation/hidden-test split",
        "comparison_mode": "result_based",
        "required_scored_questions": 15,
        "required_result_coverage": "15/15",
        "serving_endpoint": "databricks-gpt-5-5",
        "reasoning_effort": "low",
        "observed_benchmark_reuse": "disabled for baseline scoring",
        "hidden_holdout_policy": "audit only; never use hidden question text or SQL for proposal generation",
    }


@dataclass(frozen=True)
class PositiveControlSpec:
    """Named local checkpoint or patch that should reproduce known behavior."""

    name: str
    description: str | None = None
    checkpoint_path: str | None = None
    patch_path: str | None = None
    artifact_root: str | None = None
    required_protocol: Mapping[str, Any] = field(default_factory=canonical_required_protocol)
    expected_scores: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    schema_version: str = POSITIVE_CONTROL_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositiveControlSpec":
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Positive control spec requires a non-empty 'name'")
        return cls(
            name=name,
            description=_optional_str(payload.get("description")),
            checkpoint_path=_optional_str(payload.get("checkpoint_path")),
            patch_path=_optional_str(payload.get("patch_path")),
            artifact_root=_optional_str(payload.get("artifact_root")),
            required_protocol=dict(payload.get("required_protocol") or canonical_required_protocol()),
            expected_scores=dict(payload.get("expected_scores") or {}),
            source=_optional_str(payload.get("source")),
            schema_version=_optional_str(payload.get("schema_version")) or POSITIVE_CONTROL_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "checkpoint_path": self.checkpoint_path,
            "patch_path": self.patch_path,
            "artifact_root": self.artifact_root,
            "required_protocol": dict(self.required_protocol),
            "expected_scores": dict(self.expected_scores),
            "source": self.source,
        }


@dataclass(frozen=True)
class ResolvedPositiveControl:
    """A positive-control spec resolved against the local filesystem."""

    spec: PositiveControlSpec
    base_dir: Path
    checkpoint_path: Path | None = None
    patch_path: Path | None = None
    artifact_root: Path | None = None
    missing_paths: tuple[Path, ...] = ()

    @property
    def available(self) -> bool:
        return not self.missing_paths and (self.checkpoint_path is not None or self.patch_path is not None)

    @property
    def audit_target_path(self) -> Path | None:
        return self.checkpoint_path or self.patch_path

    @property
    def audit_target_kind(self) -> str | None:
        if self.checkpoint_path is not None:
            return "checkpoint"
        if self.patch_path is not None:
            return "patch"
        return None

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.spec.schema_version,
            "name": self.spec.name,
            "available": self.available,
            "status": "available" if self.available else "unavailable",
            "description": self.spec.description,
            "audit_target_kind": self.audit_target_kind,
            "audit_target_path": _path_to_str(self.audit_target_path),
            "checkpoint_path": _path_to_str(self.checkpoint_path),
            "patch_path": _path_to_str(self.patch_path),
            "artifact_root": _path_to_str(self.artifact_root),
            "missing_paths": [_path_to_str(path) for path in self.missing_paths],
            "required_protocol": dict(self.spec.required_protocol),
            "expected_scores": dict(self.spec.expected_scores),
            "source": self.spec.source,
        }


def load_positive_control_spec(path: str | Path) -> PositiveControlSpec:
    """Load a positive-control spec from a JSON file."""

    spec_path = Path(path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Positive control spec must be a JSON object: {spec_path}")
    return PositiveControlSpec.from_dict(payload)


def resolve_positive_control(
    control: str | Path | PositiveControlSpec,
    *,
    base_dir: str | Path | None = None,
) -> ResolvedPositiveControl:
    """Resolve a JSON spec, known control name, or spec object locally."""

    if isinstance(control, PositiveControlSpec):
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        return _resolve_spec(control, root)

    control_path = Path(control)
    if control_path.exists() and control_path.is_file():
        spec = load_positive_control_spec(control_path)
        root = Path(base_dir) if base_dir is not None else control_path.parent
        return _resolve_spec(spec, root)

    control_name = str(control)
    if control_name in {EXAMPLE_CHECKPOINT_CONTROL_NAME, EXAMPLE_PATCH_CONTROL_NAME}:
        spec = known_positive_control_spec(control_name)
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        return _resolve_spec(spec, root)

    raise ValueError(f"Unknown positive control: {control_name}")


def summarize_positive_control(
    control: str | Path | PositiveControlSpec | ResolvedPositiveControl,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a serializable availability and audit-protocol summary."""

    if isinstance(control, ResolvedPositiveControl):
        resolved = control
    else:
        resolved = resolve_positive_control(control, base_dir=base_dir)
    return resolved.to_summary()


def known_positive_control_spec(name: str) -> PositiveControlSpec:
    """Return a built-in positive-control spec by name."""

    if name == EXAMPLE_CHECKPOINT_CONTROL_NAME:
        return PositiveControlSpec(
            name=EXAMPLE_CHECKPOINT_CONTROL_NAME,
            description=(
                "Example positive control. The audit target is a local checkpoint "
                "under examples/positive_controls."
            ),
            checkpoint_path=_EXAMPLE_WINNER_CHECKPOINT,
            artifact_root=_EXAMPLE_CONTROL_ROOT,
            required_protocol=canonical_required_protocol(),
            expected_scores={
                "full_pass_rate": 0.0,
                "train_pass_rate": 0.0,
                "validation_pass_rate": 0.0,
                "holdout_pass_rate": 0.0,
            },
            source="example_local_artifact",
        )
    if name == EXAMPLE_PATCH_CONTROL_NAME:
        return PositiveControlSpec(
            name=EXAMPLE_PATCH_CONTROL_NAME,
            description=(
                "Example positive-control patch replay. The audit target is a strict "
                "artifact_patch_v2 sequence built from a local baseline-to-winner checkpoint delta."
            ),
            patch_path=_EXAMPLE_CONTROL_ROOT,
            artifact_root=_EXAMPLE_CONTROL_ROOT,
            required_protocol=canonical_required_protocol(),
            expected_scores={
                "full_pass_rate": 0.0,
                "train_pass_rate": 0.0,
                "validation_pass_rate": 0.0,
                "holdout_pass_rate": 0.0,
            },
            source="example_local_artifact",
        )
    raise ValueError(f"Unknown positive control: {name}")


def materialize_positive_control_patch_sequence(
    control: str | Path | PositiveControlSpec | ResolvedPositiveControl,
    *,
    destination_dir: str | Path,
    base_dir: str | Path | None = None,
) -> Path:
    """Write a positive-control patch sequence and return its directory."""

    resolved = (
        control
        if isinstance(control, ResolvedPositiveControl)
        else resolve_positive_control(control, base_dir=base_dir)
    )
    if not resolved.available or resolved.patch_path is None:
        raise ValueError(f"Positive control `{resolved.spec.name}` does not resolve to a patch target.")

    if resolved.spec.name == EXAMPLE_PATCH_CONTROL_NAME:
        baseline_space_dir = resolved.base_dir / _EXAMPLE_BASELINE_CHECKPOINT / "space"
        winner_space_dir = resolved.base_dir / _EXAMPLE_WINNER_CHECKPOINT / "space"
        if not baseline_space_dir.exists():
            raise ValueError(f"Positive-control baseline space is missing: {baseline_space_dir}")
        if not winner_space_dir.exists():
            raise ValueError(f"Positive-control winner space is missing: {winner_space_dir}")
        sequence = build_artifact_patch_sequence_from_space_dirs(
            baseline_space_dir,
            winner_space_dir,
            candidate_name_prefix=resolved.spec.name,
            candidate_mode="artifact_patch_v2",
            max_file_edits=2,
        )
        write_artifact_patch_sequence(destination_dir, sequence)
        return Path(destination_dir).expanduser()

    source_path = resolved.patch_path
    if source_path.is_dir() and (source_path / "patch_sequence.json").exists():
        destination = Path(destination_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "patch_sequence.json").write_text(
            (source_path / "patch_sequence.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return destination
    raise ValueError(
        f"Positive control `{resolved.spec.name}` does not provide a patch_sequence.json."
    )


def _resolve_spec(spec: PositiveControlSpec, base_dir: Path) -> ResolvedPositiveControl:
    root = base_dir.expanduser()
    artifact_root = _resolve_optional_path(spec.artifact_root, root)
    checkpoint_path = _resolve_optional_path(spec.checkpoint_path, root)
    patch_path = _resolve_optional_path(spec.patch_path, root)

    missing_paths = []
    for candidate in (artifact_root, checkpoint_path, patch_path):
        if candidate is not None and not candidate.exists():
            missing_paths.append(candidate)

    if checkpoint_path is None and patch_path is None:
        missing_paths.append(root)

    return ResolvedPositiveControl(
        spec=spec,
        base_dir=root,
        checkpoint_path=checkpoint_path,
        patch_path=patch_path,
        artifact_root=artifact_root,
        missing_paths=tuple(missing_paths),
    )


def _resolve_optional_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string or null, got {type(value).__name__}")
    return value


def _path_to_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None
