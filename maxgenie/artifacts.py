"""Artifact management for one optimization workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


class ArtifactManager:
    """Handles filesystem layout and artifact writes for one optimization workspace."""

    def __init__(self, workspace_dir: str | Path, space_id: str):
        self.space_id = space_id
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()

        self.space_dir = self.workspace_dir / "space"
        self.benchmarks_dir = self.workspace_dir / "benchmarks"
        self.results_dir = self.workspace_dir / "results"
        self.history_dir = self.workspace_dir / "history"
        self.checkpoints_dir = self.workspace_dir / "checkpoints"

        self.subsets_dir = self.workspace_dir / "subsets"

        self.train_run_history_dir = self.history_dir / "train_runs"
        self.validation_run_history_dir = self.history_dir / "validation_runs"
        self.dev_run_history_dir = self.validation_run_history_dir
        self.test_run_history_dir = self.history_dir / "test_runs"
        self.full_run_history_dir = self.history_dir / "full_runs"
        self.recovery_checkpoints_dir = self.history_dir / "recovery_checkpoints"

        self.prompt_path = self.workspace_dir / "OPTIMIZATION_TASK.md"
        self.clone_meta_path = self.workspace_dir / "clone_meta.json"
        self.workspace_meta_path = self.workspace_dir / "workspace_meta.json"
        self.final_report_path = self.workspace_dir / "final_report.md"
        self.final_report_notebook_path = self.workspace_dir / "final_report.ipynb"
        self.benchmark_cache_path = self.workspace_dir / ".benchmark_cache.json"
        self.benchmark_payload_path = self.workspace_dir / ".benchmark_payload.json"
        self.hidden_test_payload_path = self.workspace_dir / ".hidden_test_payload.json"
        # Precomputed result-based answer key: expected (answer) rows for every
        # benchmark question, executed once and reused for all candidate diffs.
        self.answer_key_path = self.workspace_dir / ".answer_key.json"
        self.lineage_registry_path = self.workspace_dir / ".maxgenie_lineages.json"
        self.legacy_lineage_registry_path = self.workspace_dir.parent / ".maxgenie_lineages.json"

        self.train_set_path = self.benchmarks_dir / "train_set.json"
        self.validation_set_path = self.benchmarks_dir / "validation_set.json"
        self.dev_set_path = self.validation_set_path
        self.validation_folds_path = self.benchmarks_dir / "validation_folds.json"
        self.dev_folds_path = self.validation_folds_path
        self.test_set_path = self.benchmarks_dir / "test_set.json"
        self.split_meta_path = self.benchmarks_dir / "split_meta.json"

        self.baseline_train_path = self.results_dir / "baseline_train.json"
        self.latest_train_path = self.results_dir / "latest_train.json"
        self.baseline_validation_score_path = self.results_dir / "baseline_validation_score.txt"
        self.latest_validation_score_path = self.results_dir / "latest_validation_score.txt"
        self.baseline_validation_summary_path = self.results_dir / "baseline_validation_summary.json"
        self.latest_validation_summary_path = self.results_dir / "latest_validation_summary.json"
        self.baseline_dev_score_path = self.baseline_validation_score_path
        self.latest_dev_score_path = self.latest_validation_score_path
        self.baseline_dev_summary_path = self.baseline_validation_summary_path
        self.latest_dev_summary_path = self.latest_validation_summary_path
        self.baseline_test_score_path = self.results_dir / "baseline_test_score.txt"
        self.latest_test_score_path = self.results_dir / "latest_test_score.txt"
        self.baseline_full_summary_path = self.results_dir / "baseline_full_summary.json"
        self.latest_full_summary_path = self.results_dir / "latest_full_summary.json"
        self.space_diagnostics_path = self.results_dir / "space_diagnostics.json"
        self.curation_summary_path = self.results_dir / "curation_summary.json"
        self.advisory_best_practices_path = self.results_dir / "advisory_best_practices.json"
        self.optimization_summary_path = self.results_dir / "optimization_summary.json"
        self.protocol_preflight_path = self.results_dir / "protocol_preflight.json"
        self.promotion_eligibility_path = self.results_dir / "promotion_eligibility.json"
        self.comparable_run_manifest_path = self.results_dir / "comparable_run_manifest.json"
        self.terminal_selection_path = self.results_dir / "terminal_selection.json"
        self.generalization_audit_path = self.results_dir / "generalization_audit.json"
        self.visible_proxy_pressure_path = self.results_dir / "visible_proxy_pressure.json"
        self.trace_memory_path = self.results_dir / "trace_memory.json"
        self.failure_context_pack_path = self.results_dir / "failure_context_pack.json"
        self.cross_run_memory_path = self.results_dir / "cross_run_memory.json"
        self.query_history_path = self.results_dir / "query_history.json"
        self.query_history_summary_path = self.results_dir / "query_history_summary.json"
        self.query_history_recommendations_path = self.results_dir / "query_history_recommendations.json"
        self.blocked_failures_path = self.results_dir / "blocked_failures.md"

        self._legacy_dev_run_history_dir = self.history_dir / "dev_runs"
        self._legacy_dev_set_path = self.benchmarks_dir / "dev_set.json"
        self._legacy_dev_folds_path = self.benchmarks_dir / "dev_folds.json"
        self._legacy_baseline_dev_score_path = self.results_dir / "baseline_dev_score.txt"
        self._legacy_latest_dev_score_path = self.results_dir / "latest_dev_score.txt"
        self._legacy_baseline_dev_summary_path = self.results_dir / "baseline_dev_summary.json"
        self._legacy_latest_dev_summary_path = self.results_dir / "latest_dev_summary.json"

        for directory in (
            self.workspace_dir,
            self.space_dir,
            self.benchmarks_dir,
            self.results_dir,
            self.history_dir,
            self.checkpoints_dir,
            self.subsets_dir,
            self.train_run_history_dir,
            self.validation_run_history_dir,
            self.test_run_history_dir,
            self.full_run_history_dir,
            self.recovery_checkpoints_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._migrate_validation_artifacts()

    def _migrate_legacy_file(self, legacy_path: Path, canonical_path: Path) -> None:
        if legacy_path == canonical_path or not legacy_path.exists() or canonical_path.exists():
            return
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.replace(canonical_path)

    def _migrate_legacy_directory(self, legacy_dir: Path, canonical_dir: Path) -> None:
        if legacy_dir == canonical_dir or not legacy_dir.exists():
            return
        canonical_dir.mkdir(parents=True, exist_ok=True)
        for child in legacy_dir.iterdir():
            target = canonical_dir / child.name
            if target.exists():
                continue
            child.replace(target)
        try:
            legacy_dir.rmdir()
        except OSError:
            pass

    def _migrate_validation_artifacts(self) -> None:
        """Promote legacy dev-named artifacts to validation-named paths."""
        self._migrate_legacy_directory(self._legacy_dev_run_history_dir, self.validation_run_history_dir)
        file_pairs = (
            (self._legacy_dev_set_path, self.validation_set_path),
            (self._legacy_dev_folds_path, self.validation_folds_path),
            (self._legacy_baseline_dev_score_path, self.baseline_validation_score_path),
            (self._legacy_latest_dev_score_path, self.latest_validation_score_path),
            (self._legacy_baseline_dev_summary_path, self.baseline_validation_summary_path),
            (self._legacy_latest_dev_summary_path, self.latest_validation_summary_path),
        )
        for legacy_path, canonical_path in file_pairs:
            self._migrate_legacy_file(legacy_path, canonical_path)

    def config_hash(self, serialized_space: dict[str, Any]) -> str:
        raw = json.dumps(serialized_space, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    def write_json(self, relative_path: str, payload: Any) -> Path:
        return self.write_json_path(self.workspace_dir / relative_path, payload)

    def write_json_path(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return path

    def read_json(self, relative_path: str) -> Any:
        return self.read_json_path(self.workspace_dir / relative_path)

    def read_json_path(self, path: Path) -> Any:
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def write_text(self, relative_path: str, content: str) -> Path:
        return self.write_text_path(self.workspace_dir / relative_path, content)

    def write_text_path(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read_text_path(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write_score_path(self, path: Path, score: float) -> Path:
        return self.write_text_path(path, f"{score:.1f}\n")

    def read_score_path(self, path: Path) -> float | None:
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return float(raw)

    def record_iteration(self, payload: dict[str, Any]) -> Path:
        path = self.history_dir / "iteration_log.jsonl"
        payload = {
            "recorded_at": datetime.now().isoformat(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def subset_path(self, subset_name: str) -> Path:
        return self.subsets_dir / f"{subset_name}.json"

    def history_snapshot_path(self, kind: str, label: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        directory_map = {
            "train": self.train_run_history_dir,
            "validation": self.validation_run_history_dir,
            "dev": self.validation_run_history_dir,
            "test": self.test_run_history_dir,
            "full": self.full_run_history_dir,
        }
        try:
            directory = directory_map[kind]
        except KeyError as exc:
            raise ValueError(f"Unsupported snapshot kind: {kind}") from exc
        return directory / f"{stamp}.{label}.json"

    def write_history_snapshot(self, kind: str, label: str, payload: Any) -> Path:
        return self.write_json_path(self.history_snapshot_path(kind, label), payload)

    def load_clone_meta(self) -> dict[str, Any] | None:
        if not self.clone_meta_path.exists():
            return None
        payload = self.read_json_path(self.clone_meta_path)
        return payload if isinstance(payload, dict) else None

    def save_clone_meta(
        self,
        clone_space_id: str,
        source_space_id: str,
        *,
        input_space_id: str | None = None,
        managed_source_space_id: str | None = None,
        active_space_id: str | None = None,
        clone_parent_path: str | None = None,
        clone_parent_path_source: str | None = None,
    ) -> Path:
        payload = {
            "clone_space_id": clone_space_id,
            "source_space_id": source_space_id,
            "input_space_id": input_space_id or source_space_id,
            "managed_source_space_id": managed_source_space_id or source_space_id,
            "active_space_id": active_space_id or clone_space_id,
            "created_at": datetime.now().isoformat(),
        }
        if clone_parent_path:
            payload["clone_parent_path"] = clone_parent_path
        if clone_parent_path_source:
            payload["clone_parent_path_source"] = clone_parent_path_source
        return self.write_json_path(self.clone_meta_path, payload)

    def load_benchmark_payload(self) -> dict[str, Any] | None:
        if not self.benchmark_payload_path.exists():
            return None
        payload = self.read_json_path(self.benchmark_payload_path)
        return payload if isinstance(payload, dict) else None

    def load_hidden_test_payload(self) -> dict[str, Any] | None:
        if not self.hidden_test_payload_path.exists():
            return None
        payload = self.read_json_path(self.hidden_test_payload_path)
        return payload if isinstance(payload, dict) else None

    def load_workspace_meta(self) -> dict[str, Any] | None:
        if not self.workspace_meta_path.exists():
            return None
        try:
            payload = self.read_json_path(self.workspace_meta_path)
        except FileNotFoundError:
            return None
        return payload if isinstance(payload, dict) else None

    def update_workspace_meta(self, payload: dict[str, Any]) -> Path:
        existing = self.load_workspace_meta() or {}
        existing.update(payload)
        existing["updated_at"] = datetime.now().isoformat()
        return self.write_json_path(self.workspace_meta_path, existing)

    def _load_lineage_registry_path(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        payload = self.read_json_path(path)
        return payload if isinstance(payload, dict) else {}

    def load_lineage_registry(self) -> dict[str, Any]:
        return self._load_lineage_registry_path(self.lineage_registry_path)

    def load_legacy_lineage_registry(self) -> dict[str, Any]:
        return self._load_lineage_registry_path(self.legacy_lineage_registry_path)

    def load_lineage(self, input_space_id: str) -> dict[str, Any] | None:
        payload = self.load_lineage_registry().get(input_space_id)
        if isinstance(payload, dict):
            return payload
        legacy_payload = self.load_legacy_lineage_registry().get(input_space_id)
        return legacy_payload if isinstance(legacy_payload, dict) else None

    def save_lineage(self, input_space_id: str, payload: dict[str, Any]) -> Path:
        registry = self.load_lineage_registry()
        registry[input_space_id] = {
            **payload,
            "updated_at": datetime.now().isoformat(),
        }
        return self.write_json_path(self.lineage_registry_path, registry)

    def _safe_snapshot_label(self, label: str) -> str:
        sanitized = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in label.strip()
        ).strip("._-")
        return sanitized or "snapshot"

    def _create_space_snapshot(
        self,
        *,
        parent_dir: Path,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_label = self._safe_snapshot_label(label)
        checkpoint_dir = parent_dir / f"{stamp}.{safe_label}"
        suffix = 1
        while checkpoint_dir.exists():
            checkpoint_dir = parent_dir / f"{stamp}.{safe_label}.{suffix}"
            suffix += 1
        target_space_dir = checkpoint_dir / "space"
        shutil.copytree(self.space_dir, target_space_dir)
        if metadata is not None:
            self.write_json_path(checkpoint_dir / "metadata.json", metadata)
        return checkpoint_dir

    def create_checkpoint(self, *, label: str, metadata: dict[str, Any] | None = None) -> Path:
        return self._create_space_snapshot(
            parent_dir=self.checkpoints_dir,
            label=label,
            metadata=metadata,
        )

    def create_recovery_checkpoint(self, *, label: str, metadata: dict[str, Any] | None = None) -> Path:
        return self._create_space_snapshot(
            parent_dir=self.recovery_checkpoints_dir,
            label=label,
            metadata=metadata,
        )
