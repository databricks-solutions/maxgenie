"""Tests for the workspace-oriented export flow."""

import base64
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from databricks.sdk.service import workspace as workspace_service
from typer.testing import CliRunner

import maxgenie.cli as cli_module
import maxgenie.workspace_flow as workspace_flow_module
from maxgenie.advisory import NO_BENCHMARK_WARNING
from maxgenie.artifacts import ArtifactManager
from maxgenie.artifact_patch import (
    build_artifact_patch_sequence_from_space_dirs,
    write_artifact_patch_sequence,
)
from maxgenie.assembler import assemble_config
from maxgenie.benchmark_service import BenchmarkQuestion, BenchmarkResult, BenchmarkRun, BenchmarkStatus
from maxgenie.config import (
    AUTONOMOUS_STRATEGY_CENTAUR,
)
from maxgenie.decomposer import decompose_config
from maxgenie.exporter import BenchmarkSplit, ExportBundle, SpaceExporter, ValidationFold
from maxgenie.genie_service import GenieSpaceInfo
from maxgenie.schemas import (
    GenieBenchmarkAnswer,
    GenieBenchmarkQuestion,
    GenieBenchmarks,
    GenieColumnConfig,
    GenieConfig,
    GenieDataSources,
    GenieExampleSQL,
    GenieExampleSQLParameter,
    GenieInstruction,
    GenieInstructions,
    GenieSQLSnippet,
    GenieSQLSnippets,
    GenieSampleQuestion,
    GenieSpaceConfig,
    GenieTableConfig,
)
from maxgenie.benchmark_summaries import _derived_full_summary
from maxgenie.serving_candidates import (
    ServingCandidateConfig,
    ServingCandidateProposal,
    ServingCandidateResponseError,
)
from maxgenie.workspace_flow import (
    CandidateSpec,
    OptimizationWorkspace,
    WorkspaceFlowError,
    _RESULT_CONTENT_MATCH_ENV,
    _benchmark_cache_hash,
    _benchmark_issue_details,
    _copytree_for_patch_validation,
    _formatted_clone_title,
    _generalization_audit,
    _merge_benchmark_runs,
    _normalize_clone_parent_path,
    _resolve_result_content_match,
    _terminal_result_selection,
)


def test_write_report_notebook_imports_workspace_notebook(tmp_path: Path):
    class FakeWorkspace:
        def __init__(self):
            self.imports = []

        def import_(self, path, content, format, overwrite):
            self.imports.append(
                {
                    "path": path,
                    "content": content,
                    "format": format,
                    "overwrite": overwrite,
                }
            )

    workspace = FakeWorkspace()
    flow = object.__new__(OptimizationWorkspace)
    flow.artifacts = SimpleNamespace(
        final_report_notebook_path=Path(
            "/Workspace/Users/user@example.com/.maxgenie/runs/job/"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/final_report.ipynb"
        ),
        results_dir=tmp_path,
        write_json_path=lambda path, payload: path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        ),
    )
    flow.genie_service = SimpleNamespace(wc=SimpleNamespace(workspace=workspace))

    flow._write_report_notebook("# Report\n\nBody\n")

    assert len(workspace.imports) == 1
    upload = workspace.imports[0]
    assert (
        upload["path"]
        == "/Users/user@example.com/.maxgenie/runs/job/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/final_report.ipynb"
    )
    assert upload["format"] == workspace_service.ImportFormat.JUPYTER
    assert upload["overwrite"] is True
    notebook = json.loads(base64.b64decode(upload["content"]).decode("utf-8"))
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert notebook["cells"][0]["source"] == ["# Report\n", "\n", "Body\n"]


def test_data_probe_sql_is_bounded_read_only_aggregate() -> None:
    sql = OptimizationWorkspace._data_probe_sql(
        table_identifier="catalog.schema.sales table",
        column_name="brand name",
    )

    assert sql.startswith("WITH sample AS")
    assert "FROM `catalog`.`schema`.`sales table`" in sql
    assert "CAST(`brand name` AS STRING)" in sql
    assert "LIMIT 50000" in sql
    assert "trim_mismatch_count" in sql
    assert "APPROX_COUNT_DISTINCT(TRIM(value))" in sql


class _DummyCurrentUser:
    def __init__(self, user_name: str | None = "optimizer@example.com"):
        self._user_name = user_name

    def me(self):
        return type("User", (), {"user_name": self._user_name})()


class _DummyWorkspaceClient:
    def __init__(self, user_name: str | None = "optimizer@example.com"):
        self.current_user = _DummyCurrentUser(user_name)


def test_merge_benchmark_runs_carries_forward_unrerun_train_questions():
    questions = [
        BenchmarkQuestion(id="q1", question="Question 1"),
        BenchmarkQuestion(id="q2", question="Question 2"),
        BenchmarkQuestion(id="q3", question="Question 3"),
    ]
    previous = BenchmarkRun(space_id="space")
    previous.results = [
        BenchmarkResult(question_id="q1", question="Question 1", status=BenchmarkStatus.PASSED),
        BenchmarkResult(question_id="q2", question="Question 2", status=BenchmarkStatus.FAILED),
        BenchmarkResult(question_id="q3", question="Question 3", status=BenchmarkStatus.PASSED),
    ]
    subset = BenchmarkRun(space_id="space")
    subset.results = [
        BenchmarkResult(question_id="q2", question="Question 2", status=BenchmarkStatus.PASSED),
    ]

    merged = _merge_benchmark_runs(
        ordered_questions=questions,
        runs=[subset],
        carry_forward_run=previous,
    )

    assert [result.question_id for result in merged.results] == ["q1", "q2", "q3"]
    assert [result.status for result in merged.results] == [
        BenchmarkStatus.PASSED,
        BenchmarkStatus.PASSED,
        BenchmarkStatus.PASSED,
    ]


def test_benchmark_issue_details_tolerates_structured_error_message():
    result = BenchmarkResult(
        question_id="q1",
        question="Question 1",
        status=BenchmarkStatus.ERROR,
        error_message={"error": "Polling timeout exceeded"},  # type: ignore[arg-type]
    )

    assert _benchmark_issue_details([result]) == [
        {
            "question_id": "q1",
            "status": "error",
            "issue_class": "timeout",
            "error_message": '{"error": "Polling timeout exceeded"}',
        }
    ]


def test_terminal_selection_blocks_holdout_regression():
    selection = _terminal_result_selection(
        optimization_summary={
            "baseline": {
                "full_pass_rate": 40.0,
                "test_rate": 33.33,
            },
            "baseline_checkpoint_path": "/tmp/baseline",
            "best_checkpoint_path": "/tmp/best",
        },
        final_summary={
            "overall": {"pass_rate": 53.33},
            "test": {"pass_rate": 0.0},
        },
    )

    assert selection["status"] == "not_promotable"
    assert selection["selected_checkpoint_path"] == "/tmp/baseline"
    assert selection["candidate_checkpoint_path"] == "/tmp/best"
    assert "final_holdout_regressed_below_baseline" in selection["reasons"]


def test_terminal_selection_requires_full_improvement():
    selection = _terminal_result_selection(
        optimization_summary={
            "baseline": {
                "full_pass_rate": 40.0,
                "test_rate": 33.33,
            },
            "baseline_checkpoint_path": "/tmp/baseline",
            "best_checkpoint_path": "/tmp/best",
        },
        final_summary={
            "overall": {"pass_rate": 40.0},
            "test": {"pass_rate": 33.33},
        },
    )

    assert selection["status"] == "not_promotable"
    assert selection["selected_checkpoint_path"] == "/tmp/baseline"
    assert selection["candidate_checkpoint_path"] == "/tmp/best"
    assert selection["reasons"] == ["final_full_did_not_improve_over_baseline"]
    assert "final full improves" in selection["policy"]


def test_terminal_selection_blocks_non_comparable_protocol():
    selection = _terminal_result_selection(
        optimization_summary={
            "baseline": {
                "full_pass_rate": 40.0,
                "test_rate": 33.33,
            },
            "baseline_checkpoint_path": "/tmp/baseline",
            "best_checkpoint_path": "/tmp/best",
        },
        final_summary={
            "overall": {"pass_rate": 53.33},
            "test": {"pass_rate": 100.0},
        },
        protocol_preflight={
            "status": "non_comparable",
            "comparable": False,
            "errors": [{"code": "incomplete_result_based_benchmark_coverage"}],
        },
    )

    assert selection["status"] == "not_promotable"
    assert selection["selected_checkpoint_path"] == "/tmp/baseline"
    assert selection["protocol_status"] == "non_comparable"
    assert selection["protocol_blocking_reasons"] == [
        "incomplete_result_based_benchmark_coverage"
    ]
    assert "protocol_non_comparable" in selection["reasons"]
    assert "protocol preflight is comparable" in selection["policy"]


def test_generalization_audit_freezes_may_incumbent_but_blocks_production_golden():
    audit = _generalization_audit(
        optimization_summary={
            "baseline": {
                "full_pass_rate": 40.0,
                "test_rate": 33.33,
            },
            "terminal_selection": {
                "status": "promotable",
                "selected_checkpoint_path": "/tmp/best",
            },
        },
        final_summary={
            "overall": {"pass_rate": 73.33},
            "train": {"pass_rate": 100.0},
            "validation": {"pass_rate": 83.33, "question_count": 12},
            "test": {
                "pass_rate": 33.33,
                "question_count": 3,
                "passed": 1,
                "failed": 2,
                "errors": 0,
                "skipped": 0,
                "failed_ids": ["hidden-q1", "hidden-q2"],
                "issue_class_counts": {"sql_mismatch": 2},
            },
        },
    )

    assert audit["status"] == "may_workspace_incumbent_not_production_golden"
    assert audit["decisions"]["may_workspace_incumbent"] is True
    assert audit["decisions"]["production_golden_candidate"] is False
    assert audit["metrics"]["validation_holdout_gap"] == 50.0
    assert audit["hidden_holdout_summary"]["issue_class_counts"] == {"sql_mismatch": 2}
    assert "failed_ids" not in audit["hidden_holdout_summary"]
    assert "visible_holdout_gap" in audit["flags"]
    assert "small_hidden_holdout" in audit["flags"]


def test_load_workspace_meta_tolerates_stale_workspace_file_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    original_exists = Path.exists

    def stale_exists(path: Path) -> bool:
        if path == artifacts.workspace_meta_path:
            return True
        return original_exists(path)

    def stale_read_json_path(path: Path):
        if path == artifacts.workspace_meta_path:
            raise FileNotFoundError(str(path))
        return {}

    monkeypatch.setattr(Path, "exists", stale_exists)
    monkeypatch.setattr(artifacts, "read_json_path", stale_read_json_path)

    assert artifacts.load_workspace_meta() is None


def test_checkpoint_audit_restores_checkpoint_and_writes_generalization_artifacts(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    baseline_config = _space_config()
    decompose_config(baseline_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.write_json_path(artifacts.split_meta_path, {"split_strategy": "canonical"})
    artifacts.update_workspace_meta(
        {
            "fresh_baseline": True,
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )

    train_question = BenchmarkQuestion(
        id="a" * 32,
        question="Train question",
        expected_sql="SELECT 1",
    )
    baseline_train = BenchmarkRun(space_id="clone-123")
    baseline_train.results = [
        BenchmarkResult(
            question_id=train_question.id,
            question=train_question.question,
            status=BenchmarkStatus.FAILED,
            expected_sql=train_question.expected_sql,
            generated_sql="SELECT 0",
            comparison_score=0.2,
        )
    ]
    baseline_validation = {
        "question_count": 12,
        "passed": 5,
        "failed": 7,
        "errors": 0,
        "pass_rate": 41.67,
        "validation_strategy": "grouped_kfold",
    }
    baseline_full = {
        "train": {"question_count": 9, "passed": 4, "failed": 5, "errors": 0, "pass_rate": 44.44},
        "validation": baseline_validation,
        "test": {"question_count": 3, "passed": 1, "failed": 2, "errors": 0, "pass_rate": 33.33},
        "overall": {"question_count": 15, "passed": 6, "failed": 9, "errors": 0, "pass_rate": 40.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_train.to_dict())
    artifacts.write_json_path(artifacts.latest_train_path, baseline_train.to_dict())
    artifacts.write_json_path(artifacts.baseline_validation_summary_path, baseline_validation)
    artifacts.write_json_path(artifacts.latest_validation_summary_path, baseline_validation)
    artifacts.write_score_path(artifacts.baseline_validation_score_path, 41.67)
    artifacts.write_score_path(artifacts.latest_validation_score_path, 41.67)
    artifacts.write_score_path(artifacts.baseline_test_score_path, 33.33)
    artifacts.write_score_path(artifacts.latest_test_score_path, 33.33)
    artifacts.write_json_path(artifacts.baseline_full_summary_path, baseline_full)
    artifacts.write_json_path(artifacts.latest_full_summary_path, baseline_full)

    checkpoint_config = _space_config()
    checkpoint_config.title = "Checkpoint Space"
    checkpoint_dir = tmp_path / "s20_checkpoint"
    decompose_config(checkpoint_config, checkpoint_dir / "space", include_benchmarks=False)

    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(baseline_config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(baseline_config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0,
    )
    final_summary = {
        "train": {"question_count": 9, "passed": 9, "failed": 0, "errors": 0, "pass_rate": 100.0},
        "validation": {"question_count": 12, "passed": 10, "failed": 2, "errors": 0, "pass_rate": 83.33},
        "test": {
            "question_count": 3,
            "passed": 1,
            "failed": 2,
            "errors": 0,
            "pass_rate": 33.33,
            "issue_class_counts": {"sql_mismatch": 2},
        },
        "overall": {
            "question_count": 15,
            "passed": 11,
            "failed": 4,
            "errors": 0,
            "pass_rate": 73.33,
            "comparison_metrics": {
                "scored_question_count": 15,
                "result_based_question_count": 15,
                "comparison_method_counts": {"result_match": 15},
            },
        },
        "evaluation_context": {
            "preferred_comparison_mode": "result_based",
            "warehouse_id": "warehouse-123",
        },
    }
    full_calls: list[dict[str, object]] = []

    def fake_full_benchmark(*, label: str, baseline: bool = False):
        full_calls.append({"label": label, "baseline": baseline})
        return final_summary

    flow.run_full_benchmark = fake_full_benchmark  # type: ignore[method-assign]

    summary = flow.run_checkpoint_audit(
        label="checkpoint_audit",
        checkpoint_path=checkpoint_dir,
    )

    assert full_calls == [{"label": "checkpoint_audit.final", "baseline": False}]
    assert assemble_config(artifacts.space_dir).title == "Checkpoint Space"
    assert summary["mode"] == "checkpoint_audit"
    assert summary["best_checkpoint_path"] == str(checkpoint_dir)
    assert summary["terminal_selection"]["status"] == "promotable"
    assert summary["generalization_audit"]["decisions"]["may_workspace_incumbent"] is True
    assert artifacts.terminal_selection_path.exists()
    assert artifacts.generalization_audit_path.exists()
    log_text = artifacts.read_text_path(artifacts.history_dir / "iteration_log.jsonl")
    assert '"phase": "checkpoint_audit_restored"' in log_text
    assert '"phase": "checkpoint_audit_completed"' in log_text


def _checkpoint_audit_flow(tmp_path: Path):
    """Build an OptimizationWorkspace with baseline artifacts + a checkpoint for audit tests.

    Mirrors the setup of test_checkpoint_audit_restores_checkpoint_and_writes_generalization_artifacts;
    returns (flow, artifacts, checkpoint_dir, final_summary, all_questions).
    """
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    baseline_config = _space_config()
    decompose_config(baseline_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.write_json_path(artifacts.split_meta_path, {"split_strategy": "canonical"})
    artifacts.update_workspace_meta(
        {
            "fresh_baseline": True,
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )
    train_question = BenchmarkQuestion(id="a" * 32, question="Train question", expected_sql="SELECT 1")
    baseline_train = BenchmarkRun(space_id="clone-123")
    baseline_train.results = [
        BenchmarkResult(
            question_id=train_question.id,
            question=train_question.question,
            status=BenchmarkStatus.FAILED,
            expected_sql=train_question.expected_sql,
            generated_sql="SELECT 0",
            comparison_score=0.2,
        )
    ]
    baseline_validation = {
        "question_count": 12,
        "passed": 5,
        "failed": 7,
        "errors": 0,
        "pass_rate": 41.67,
        "validation_strategy": "grouped_kfold",
    }
    baseline_full = {
        "train": {"question_count": 9, "passed": 4, "failed": 5, "errors": 0, "pass_rate": 44.44},
        "validation": baseline_validation,
        "test": {"question_count": 3, "passed": 1, "failed": 2, "errors": 0, "pass_rate": 33.33},
        "overall": {"question_count": 15, "passed": 6, "failed": 9, "errors": 0, "pass_rate": 40.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_train.to_dict())
    artifacts.write_json_path(artifacts.latest_train_path, baseline_train.to_dict())
    artifacts.write_json_path(artifacts.baseline_validation_summary_path, baseline_validation)
    artifacts.write_json_path(artifacts.latest_validation_summary_path, baseline_validation)
    artifacts.write_score_path(artifacts.baseline_validation_score_path, 41.67)
    artifacts.write_score_path(artifacts.latest_validation_score_path, 41.67)
    artifacts.write_score_path(artifacts.baseline_test_score_path, 33.33)
    artifacts.write_score_path(artifacts.latest_test_score_path, 33.33)
    artifacts.write_json_path(artifacts.baseline_full_summary_path, baseline_full)
    artifacts.write_json_path(artifacts.latest_full_summary_path, baseline_full)
    checkpoint_config = _space_config()
    checkpoint_config.title = "Checkpoint Space"
    checkpoint_dir = tmp_path / "r2_checkpoint"
    decompose_config(checkpoint_config, checkpoint_dir / "space", include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(baseline_config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(baseline_config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0,
    )
    final_summary = {
        "train": {"question_count": 9, "passed": 9, "failed": 0, "errors": 0, "pass_rate": 100.0},
        "validation": {"question_count": 12, "passed": 10, "failed": 2, "errors": 0, "pass_rate": 83.33},
        "test": {
            "question_count": 3,
            "passed": 1,
            "failed": 2,
            "errors": 0,
            "pass_rate": 33.33,
            "issue_class_counts": {"sql_mismatch": 2},
        },
        "overall": {
            "question_count": 15,
            "passed": 11,
            "failed": 4,
            "errors": 0,
            "pass_rate": 73.33,
            "comparison_metrics": {
                "scored_question_count": 15,
                "result_based_question_count": 15,
                "comparison_method_counts": {"result_match": 15},
            },
        },
        "evaluation_context": {
            "preferred_comparison_mode": "result_based",
            "warehouse_id": "warehouse-123",
        },
    }
    extra_question = BenchmarkQuestion(id="b" * 32, question="Holdout question", expected_sql="SELECT 2")
    return flow, artifacts, checkpoint_dir, final_summary, [train_question, extra_question]


def test_checkpoint_audit_k_samples_runs_population_benchmark(tmp_path: Path):
    """k_samples>1 K-samples the restored checkpoint state via a population benchmark, so the
    audit emits a multi_sample aggregate (sigma + per-question pass fractions incl. the hidden
    holdout) that measure-noise consumes — in ADDITION to the structured final benchmark."""
    flow, artifacts, checkpoint_dir, final_summary, all_questions = _checkpoint_audit_flow(tmp_path)

    call_order: list[str] = []
    full_calls: list[str] = []

    def fake_full_benchmark(*, label: str, baseline: bool = False):
        call_order.append(f"full:{label}")
        full_calls.append(label)
        return final_summary

    population_calls: list[dict[str, object]] = []

    def fake_population_benchmark(*, label: str, questions, k_samples: int = 1):
        call_order.append(f"population:{label}")
        population_calls.append(
            {"label": label, "k_samples": k_samples, "question_count": len(questions)}
        )
        return BenchmarkRun(space_id="clone-123")

    flow.run_full_benchmark = fake_full_benchmark  # type: ignore[method-assign]
    flow.run_population_benchmark = fake_population_benchmark  # type: ignore[method-assign]
    flow._load_all_benchmark_questions = lambda: list(all_questions)  # type: ignore[method-assign]

    summary = flow.run_checkpoint_audit(
        label="checkpoint_audit",
        checkpoint_path=checkpoint_dir,
        k_samples=5,
    )

    assert population_calls == [
        {"label": "checkpoint_audit_population", "k_samples": 5, "question_count": len(all_questions)}
    ]
    assert full_calls == ["checkpoint_audit.final"]
    # The K-sampled population benchmark must run BEFORE the structured final benchmark.
    assert call_order == ["population:checkpoint_audit_population", "full:checkpoint_audit.final"]
    assert summary["mode"] == "checkpoint_audit"
    assert summary["audit_k_samples"] == 5
    log_text = artifacts.read_text_path(artifacts.history_dir / "iteration_log.jsonl")
    assert '"phase": "checkpoint_audit_population_sampled"' in log_text


def test_checkpoint_audit_single_sample_skips_population_benchmark(tmp_path: Path):
    """k_samples=1 (default) preserves the original single-benchmark audit byte-for-byte:
    no population benchmark, exactly one final benchmark."""
    flow, artifacts, checkpoint_dir, final_summary, _all = _checkpoint_audit_flow(tmp_path)

    full_calls: list[str] = []

    def fake_full_benchmark(*, label: str, baseline: bool = False):
        full_calls.append(label)
        return final_summary

    population_calls: list[str] = []

    def fake_population_benchmark(*, label: str, questions, k_samples: int = 1):
        population_calls.append(label)
        return BenchmarkRun(space_id="clone-123")

    flow.run_full_benchmark = fake_full_benchmark  # type: ignore[method-assign]
    flow.run_population_benchmark = fake_population_benchmark  # type: ignore[method-assign]

    summary = flow.run_checkpoint_audit(label="checkpoint_audit", checkpoint_path=checkpoint_dir)

    assert population_calls == []
    assert full_calls == ["checkpoint_audit.final"]
    assert summary["audit_k_samples"] == 1


def test_patch_replay_audit_applies_sequence_and_writes_generalization_artifacts(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    baseline_config = _space_config()
    decompose_config(baseline_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.write_json_path(artifacts.split_meta_path, {"split_strategy": "canonical"})
    artifacts.update_workspace_meta(
        {
            "fresh_baseline": True,
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )

    train_question = BenchmarkQuestion(
        id="a" * 32,
        question="Train question",
        expected_sql="SELECT 1",
    )
    baseline_train = BenchmarkRun(space_id="clone-123")
    baseline_train.results = [
        BenchmarkResult(
            question_id=train_question.id,
            question=train_question.question,
            status=BenchmarkStatus.FAILED,
            expected_sql=train_question.expected_sql,
            generated_sql="SELECT 0",
            comparison_score=0.2,
        )
    ]
    baseline_validation = {
        "question_count": 12,
        "passed": 5,
        "failed": 7,
        "errors": 0,
        "pass_rate": 41.67,
        "validation_strategy": "grouped_kfold",
    }
    baseline_full = {
        "train": {"question_count": 9, "passed": 4, "failed": 5, "errors": 0, "pass_rate": 44.44},
        "validation": baseline_validation,
        "test": {"question_count": 3, "passed": 1, "failed": 2, "errors": 0, "pass_rate": 33.33},
        "overall": {"question_count": 15, "passed": 6, "failed": 9, "errors": 0, "pass_rate": 40.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_train.to_dict())
    artifacts.write_json_path(artifacts.latest_train_path, baseline_train.to_dict())
    artifacts.write_json_path(artifacts.baseline_validation_summary_path, baseline_validation)
    artifacts.write_json_path(artifacts.latest_validation_summary_path, baseline_validation)
    artifacts.write_score_path(artifacts.baseline_validation_score_path, 41.67)
    artifacts.write_score_path(artifacts.latest_validation_score_path, 41.67)
    artifacts.write_score_path(artifacts.baseline_test_score_path, 33.33)
    artifacts.write_score_path(artifacts.latest_test_score_path, 33.33)
    artifacts.write_json_path(artifacts.baseline_full_summary_path, baseline_full)
    artifacts.write_json_path(artifacts.latest_full_summary_path, baseline_full)

    winner_config = _space_config()
    winner_config.instructions = GenieInstructions(
        text_instructions=[GenieInstruction(content=["Use exact table names", "Prefer tested examples"])]
    )
    winner_space = tmp_path / "winner_space"
    decompose_config(winner_config, winner_space, include_benchmarks=False)
    sequence = build_artifact_patch_sequence_from_space_dirs(
        artifacts.space_dir,
        winner_space,
        candidate_name_prefix="positive_control",
        max_file_edits=2,
    )
    patch_dir = tmp_path / "patch_sequence"
    write_artifact_patch_sequence(patch_dir, sequence)

    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(baseline_config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(baseline_config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0,
    )
    final_summary = {
        "train": {"question_count": 9, "passed": 9, "failed": 0, "errors": 0, "pass_rate": 100.0},
        "validation": {"question_count": 12, "passed": 10, "failed": 2, "errors": 0, "pass_rate": 83.33},
        "test": {
            "question_count": 3,
            "passed": 1,
            "failed": 2,
            "errors": 0,
            "pass_rate": 33.33,
            "issue_class_counts": {"sql_mismatch": 2},
        },
        "overall": {
            "question_count": 15,
            "passed": 11,
            "failed": 4,
            "errors": 0,
            "pass_rate": 73.33,
            "comparison_metrics": {
                "scored_question_count": 15,
                "result_based_question_count": 15,
                "comparison_method_counts": {"result_match": 15},
            },
        },
        "evaluation_context": {
            "preferred_comparison_mode": "result_based",
            "warehouse_id": "warehouse-123",
        },
    }
    full_calls: list[dict[str, object]] = []

    def fake_full_benchmark(*, label: str, baseline: bool = False):
        full_calls.append({"label": label, "baseline": baseline})
        return final_summary

    flow.run_full_benchmark = fake_full_benchmark  # type: ignore[method-assign]

    summary = flow.run_patch_replay_audit(
        label="patch_replay_audit",
        patch_path=patch_dir,
    )

    assert full_calls == [{"label": "patch_replay_audit.final", "baseline": False}]
    assert assemble_config(artifacts.space_dir).instructions.text_instructions[0].content == [
        "Use exact table names",
        "Prefer tested examples",
    ]
    assert summary["mode"] == "patch_replay_audit"
    assert summary["accepted"] == sequence["payload_count"]
    assert summary["best_checkpoint_path"]
    assert summary["terminal_selection"]["status"] == "promotable"
    assert summary["generalization_audit"]["decisions"]["may_workspace_incumbent"] is True
    log_text = artifacts.read_text_path(artifacts.history_dir / "iteration_log.jsonl")
    assert '"phase": "patch_replay_payload_applied"' in log_text
    assert '"phase": "patch_replay_audit_completed"' in log_text


def test_merge_benchmark_runs_still_rejects_missing_without_carry_forward():
    questions = [
        BenchmarkQuestion(id="q1", question="Question 1"),
        BenchmarkQuestion(id="q2", question="Question 2"),
    ]
    subset = BenchmarkRun(space_id="space")
    subset.results = [
        BenchmarkResult(question_id="q2", question="Question 2", status=BenchmarkStatus.PASSED),
    ]

    with pytest.raises(WorkspaceFlowError, match="q1"):
        _merge_benchmark_runs(ordered_questions=questions, runs=[subset])


class _DummyGenieService:
    def __init__(self, config: GenieSpaceConfig):
        self._config = config
        self.wc = _DummyWorkspaceClient()
        self.updated_title: str | None = None
        self.created_title: str | None = None
        self.created_parent_path: str | None = None
        self.updated_parent_path: str | None = None

    def get(self, space_id: str):
        return (
            GenieSpaceInfo(
                space_id=space_id,
                title=self._config.title or "Demo Space",
                warehouse_id="wh-123",
            ),
            self._config,
        )

    def update(
        self,
        space_id: str,
        config: GenieSpaceConfig,
        title: str | None = None,
        description: str | None = None,
    ) -> GenieSpaceInfo:
        self._config = config
        self.updated_title = title
        self.updated_parent_path = config.parent_path
        return GenieSpaceInfo(
            space_id=space_id,
            title=title or "Demo Space",
            description=description,
            warehouse_id=config.warehouse_id,
        )

    def create(
        self,
        config: GenieSpaceConfig,
        warehouse_id: str | None = None,
        title: str | None = None,
        parent_path: str | None = None,
    ) -> GenieSpaceInfo:
        self._config = config
        self.created_title = title
        self.created_parent_path = parent_path
        return GenieSpaceInfo(
            space_id="clone-123",
            title=title or "Demo Space",
            warehouse_id=warehouse_id or config.warehouse_id,
        )


class _DummyBenchmarkService:
    def __init__(self):
        self.start_attempts = 0
        self.message_attempts = 0

    def get_space_benchmarks(self, space_id: str) -> list[BenchmarkQuestion]:
        return []

    def start_conversation(self, space_id: str, question: str) -> tuple[str, str]:
        self.start_attempts += 1
        return "conv-1", "msg-1"

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> dict[str, str]:
        self.message_attempts += 1
        return {"status": "PENDING"}


class _PassingRunBenchmarkService(_DummyBenchmarkService):
    def run_benchmarks(
        self,
        *,
        space_id: str,
        questions: list[BenchmarkQuestion],
        delay_between_questions: float,
        fail_fast: bool,
        match_threshold: float,
        cache,
        config_hash: str,
        cache_save_path=None,
        result_comparator=None,
        **kwargs,
    ) -> BenchmarkRun:
        run = BenchmarkRun(space_id=space_id)
        run.results = [
            BenchmarkResult(
                question_id=question.id,
                question=question.question,
                status=BenchmarkStatus.PASSED,
                expected_sql=question.expected_sql,
                generated_sql=question.expected_sql,
                comparison_score=1.0,
            )
            for question in questions
        ]
        return run


class _RecoveringBenchmarkService(_DummyBenchmarkService):
    def __init__(self, genie_service):
        super().__init__()
        self.calls = 0
        self.genie_service = genie_service

    def run_benchmarks(
        self,
        *,
        space_id: str,
        questions: list[BenchmarkQuestion],
        delay_between_questions: float,
        fail_fast: bool,
        match_threshold: float,
        cache,
        config_hash: str,
        cache_save_path=None,
        result_comparator=None,
        **kwargs,
    ) -> BenchmarkRun:
        self.calls += 1
        run = BenchmarkRun(space_id=space_id)
        if self.calls == 1:
            self.genie_service.missing_ids.add(space_id)
            run.results = [
                BenchmarkResult(
                    question_id=question.id,
                    question=question.question,
                    status=BenchmarkStatus.ERROR,
                    expected_sql=question.expected_sql,
                    error_message=(
                        f"Unable to get space [{space_id}]. "
                        f"Caused by Node with resource name Some(datarooms/{space_id}) does not exist."
                    ),
                )
                for question in questions
            ]
            return run

        run.results = [
            BenchmarkResult(
                question_id=question.id,
                question=question.question,
                status=BenchmarkStatus.PASSED,
                expected_sql=question.expected_sql,
                generated_sql=question.expected_sql,
                comparison_score=1.0,
            )
            for question in questions
        ]
        return run


class _EventuallyReadyBenchmarkService(_DummyBenchmarkService):
    def __init__(self, fail_count: int):
        super().__init__()
        self.fail_count = fail_count

    def start_conversation(self, space_id: str, question: str) -> tuple[str, str]:
        self.start_attempts += 1
        if self.start_attempts <= self.fail_count:
            raise RuntimeError("conversation backend warming up")
        return "conv-1", "msg-1"


class _EventuallyReadableMessageBenchmarkService(_DummyBenchmarkService):
    def __init__(self, fail_count: int):
        super().__init__()
        self.fail_count = fail_count

    def get_message(self, space_id: str, conversation_id: str, message_id: str) -> dict[str, str]:
        self.message_attempts += 1
        if self.message_attempts <= self.fail_count:
            raise RuntimeError("message endpoint not ready")
        return {"status": "PENDING"}


class _AllPassingBenchmarkService(_DummyBenchmarkService):
    def run_benchmarks(
        self,
        *,
        space_id: str,
        questions: list[BenchmarkQuestion],
        delay_between_questions: float,
        fail_fast: bool,
        match_threshold: float,
        cache,
        config_hash: str,
        cache_save_path=None,
        result_comparator=None,
        **kwargs,
    ) -> BenchmarkRun:
        run = BenchmarkRun(space_id=space_id)
        run.results = [
            BenchmarkResult(
                question_id=question.id,
                question=question.question,
                status=BenchmarkStatus.PASSED,
                expected_sql=question.expected_sql,
                generated_sql=question.expected_sql,
                comparison_score=1.0,
            )
            for question in questions
        ]
        return run


class _FailingGenieService(_DummyGenieService):
    def get(self, space_id: str):
        raise RuntimeError("remote access denied")


class _MissingCloneGenieService(_DummyGenieService):
    def __init__(self, config: GenieSpaceConfig, *, missing_ids: set[str]):
        super().__init__(config)
        self.missing_ids = set(missing_ids)
        self.created_ids: list[str] = []

    def get(self, space_id: str):
        if space_id in self.missing_ids:
            raise RuntimeError(
                f"Unable to get space [{space_id}]. "
                f"Caused by Node with resource name Some(datarooms/{space_id}) does not exist."
            )
        return super().get(space_id)

    def create(
        self,
        config: GenieSpaceConfig,
        warehouse_id: str | None = None,
        title: str | None = None,
        parent_path: str | None = None,
    ) -> GenieSpaceInfo:
        info = super().create(
            config=config,
            warehouse_id=warehouse_id,
            title=title,
            parent_path=parent_path,
        )
        new_id = f"recovered-{len(self.created_ids) + 1}"
        self.created_ids.append(new_id)
        return GenieSpaceInfo(
            space_id=new_id,
            title=info.title,
            warehouse_id=info.warehouse_id,
            description=info.description,
        )


class _MissingAfterUpdateGenieService(_MissingCloneGenieService):
    def update(
        self,
        space_id: str,
        config: GenieSpaceConfig,
        title: str | None = None,
        description: str | None = None,
    ) -> GenieSpaceInfo:
        info = super().update(
            space_id=space_id,
            config=config,
            title=title,
            description=description,
        )
        self.missing_ids.add(space_id)
        return info


class _ComparatorWithFingerprint:
    def cache_fingerprint(self) -> str:
        return "result-comparator|v2"


def _space_config() -> GenieSpaceConfig:
    return GenieSpaceConfig(
        title="Demo Space",
        warehouse_id="wh-123",
        config=GenieConfig(
            sample_questions=[GenieSampleQuestion(question=["Sample question"])]
        ),
        data_sources=GenieDataSources(
            tables=[GenieTableConfig(identifier="catalog.schema.sales")]
        ),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact table names"])],
        ),
        benchmarks=GenieBenchmarks(
            questions=[
                GenieBenchmarkQuestion(
                    id="a" * 32,
                    question=["Train question"],
                    answer=[GenieBenchmarkAnswer(format="SQL", content=["SELECT 1"])],
                ),
                GenieBenchmarkQuestion(
                    id="b" * 32,
                    question=["Held-out question"],
                    answer=[GenieBenchmarkAnswer(format="SQL", content=["SELECT 2"])],
                ),
            ]
        ),
    )


def _space_config_with_question_count(count: int) -> GenieSpaceConfig:
    benchmarks = []
    for index in range(count):
        benchmarks.append(
            GenieBenchmarkQuestion(
                id=f"{index:032x}"[-32:],
                question=[f"Question {index}"],
                answer=[GenieBenchmarkAnswer(format="SQL", content=[f"SELECT {index}"])],
            )
        )
    config = _space_config()
    config.benchmarks = GenieBenchmarks(questions=benchmarks)
    return config


def test_baseline_train_benchmark_does_not_early_stop_core_pass_guards(tmp_path: Path):
    space_id = "f" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 3,
            "questions": [
                {"id": f"{index:032x}"[-32:], "question": f"Question {index}", "expected_sql": f"SELECT {index}"}
                for index in range(3)
            ],
        },
    )
    decompose_config(_space_config_with_question_count(3), artifacts.space_dir, include_benchmarks=False)

    class _EarlyStopAwareBenchmarkService(_DummyBenchmarkService):
        def __init__(self):
            super().__init__()
            self.min_pass_rates: list[float | None] = []

        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            min_pass_rate = kwargs.get("min_pass_rate")
            self.min_pass_rates.append(min_pass_rate)
            run = BenchmarkRun(space_id=space_id)
            passed_so_far = 0
            total = len(questions)
            for index, question in enumerate(questions):
                status = BenchmarkStatus.FAILED if index == 0 else BenchmarkStatus.PASSED
                if status == BenchmarkStatus.PASSED:
                    passed_so_far += 1
                run.results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        status=status,
                        expected_sql=question.expected_sql,
                        generated_sql=question.expected_sql if status == BenchmarkStatus.PASSED else "SELECT wrong",
                        comparison_score=1.0 if status == BenchmarkStatus.PASSED else 0.0,
                    )
                )
                if min_pass_rate is not None:
                    remaining = total - (index + 1)
                    if (passed_so_far + remaining) / total * 100 < min_pass_rate:
                        break
            return run

    benchmark_service = _EarlyStopAwareBenchmarkService()
    genie_service = _DummyGenieService(_space_config_with_question_count(3))
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    run = flow.run_train_benchmark(label="baseline.train", baseline=True)

    assert [result.question_id for result in run.results] == [f"{index:032x}"[-32:] for index in range(3)]
    assert run.pass_rate == pytest.approx(66.67, abs=0.01)
    assert benchmark_service.min_pass_rates == [None]


def _artifact_patch_validation_flow(tmp_path: Path) -> OptimizationWorkspace:
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    return OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0,
    )


def _content_match_workspace(tmp_path: Path, **kwargs: object) -> OptimizationWorkspace:
    space_id = "2" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    return OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0,
        warehouse_id="wh-123",
        **kwargs,
    )


def test_resolve_result_content_match_explicit_argument_wins(monkeypatch) -> None:
    # An explicit argument overrides the env switch in both directions.
    monkeypatch.setenv(_RESULT_CONTENT_MATCH_ENV, "1")
    assert _resolve_result_content_match(False) is False
    monkeypatch.delenv(_RESULT_CONTENT_MATCH_ENV, raising=False)
    assert _resolve_result_content_match(True) is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True), ("true", True), ("YES", True), ("y", True), ("on", True),
        ("0", False), ("false", False), ("", False), ("nope", False),
    ],
)
def test_resolve_result_content_match_env_truthiness(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv(_RESULT_CONTENT_MATCH_ENV, raw)
    assert _resolve_result_content_match(None) is expected


def test_resolve_result_content_match_defaults_off_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(_RESULT_CONTENT_MATCH_ENV, raising=False)
    assert _resolve_result_content_match(None) is False


def test_workspace_comparator_defaults_to_byte_exact(tmp_path: Path, monkeypatch) -> None:
    # Customer-safe default: no env, no argument -> legacy byte-exact comparator.
    monkeypatch.delenv(_RESULT_CONTENT_MATCH_ENV, raising=False)
    flow = _content_match_workspace(tmp_path)
    assert flow.comparator is not None
    assert flow.comparator._content_match is False


def test_workspace_comparator_content_match_via_argument(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(_RESULT_CONTENT_MATCH_ENV, raising=False)
    flow = _content_match_workspace(tmp_path, result_content_match=True)
    assert flow.comparator is not None
    assert flow.comparator._content_match is True


def test_workspace_comparator_content_match_via_env(tmp_path: Path, monkeypatch) -> None:
    # The campaign switch turns the fair ruler on without a code change.
    monkeypatch.setenv(_RESULT_CONTENT_MATCH_ENV, "1")
    flow = _content_match_workspace(tmp_path)
    assert flow.comparator is not None
    assert flow.comparator._content_match is True


def test_broad_artifact_patch_allows_hidden_holdout_policy_boilerplate(
    tmp_path: Path,
) -> None:
    flow = _artifact_patch_validation_flow(tmp_path)
    current_config = assemble_config(flow.artifacts.space_dir)
    target_id = "a" * 32
    expected_sha256 = hashlib.sha256(
        (flow.artifacts.space_dir / "instructions/text_instruction.md").read_bytes()
    ).hexdigest()
    payload = {
        "candidate_name": "policy_clean_broad_patch",
        "candidate_mode": "freedom_artifact_patch_v1",
        "rationale": (
            f"target_train_failure:{target_id}: add a narrow visible instruction."
        ),
        "expected_impact": (
            f"target_train_failure:{target_id}: improve visible literal handling."
        ),
        "risk": "Low. The payload contains no hidden holdout content.",
        "operations": [],
        "file_edits": [
            {
                "path": "instructions/text_instruction.md",
                "append_content": (
                    "\nKeep visible literal filters executable."
                ),
                "reason": f"target_train_failure:{target_id}: visible-only guidance.",
                "expected_sha256": expected_sha256,
            }
        ],
    }

    _, summary = flow._apply_artifact_patch_payload(
        payload=payload,
        current_config=current_config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                }
            ]
        },
    )

    assert summary["changed"] is True
    assert "reason" not in summary


def test_broad_artifact_patch_rejects_hidden_holdout_question_reference(
    tmp_path: Path,
) -> None:
    flow = _artifact_patch_validation_flow(tmp_path)
    current_config = assemble_config(flow.artifacts.space_dir)
    target_id = "a" * 32
    expected_sha256 = hashlib.sha256(
        (flow.artifacts.space_dir / "instructions/text_instruction.md").read_bytes()
    ).hexdigest()
    payload = {
        "candidate_name": "hidden_reference_broad_patch",
        "candidate_mode": "freedom_artifact_patch_v1",
        "rationale": (
            f"target_train_failure:{target_id}: add a narrow visible instruction."
        ),
        "expected_impact": "This should also fix a hidden holdout question.",
        "risk": "Medium.",
        "operations": [],
        "file_edits": [
            {
                "path": "instructions/text_instruction.md",
                "append_content": (
                    "\nKeep visible literal filters executable."
                ),
                "reason": f"target_train_failure:{target_id}: visible-only guidance.",
                "expected_sha256": expected_sha256,
            }
        ],
    }

    _, summary = flow._apply_artifact_patch_payload(
        payload=payload,
        current_config=current_config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                }
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert (
        "broad_artifact_patch_hidden_holdout_reference"
        in summary["risk_assessment"]["reasons"]
    )


def test_patch_validation_copy_tolerates_workspace_file_listing_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    nested = source / "instructions"
    nested.mkdir(parents=True)
    keep_path = nested / "text_instruction.md"
    vanish_path = nested / "sql_functions.yml"
    keep_path.write_text("keep", encoding="utf-8")
    vanish_path.write_text("stale", encoding="utf-8")
    real_copy2 = workspace_flow_module.shutil.copy2

    def flaky_copy2(src, dst, *args, **kwargs):
        if Path(src).name == "sql_functions.yml":
            Path(src).unlink()
            raise FileNotFoundError(str(src))
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(workspace_flow_module.shutil, "copy2", flaky_copy2)

    missing_paths = _copytree_for_patch_validation(source, target)

    assert missing_paths == ["instructions/sql_functions.yml"]
    assert (target / "instructions" / "text_instruction.md").read_text(encoding="utf-8") == "keep"
    assert not (target / "instructions" / "sql_functions.yml").exists()


def _baseline_run_for_questions(
    questions: list[BenchmarkQuestion],
    *,
    passed_ids: set[str],
) -> BenchmarkRun:
    run = BenchmarkRun(space_id="space-123")
    run.results = [
        BenchmarkResult(
            question_id=question.id,
            question=question.question,
            status=BenchmarkStatus.PASSED if question.id in passed_ids else BenchmarkStatus.FAILED,
            expected_sql=question.expected_sql,
            generated_sql=question.expected_sql if question.id in passed_ids else "SELECT 0",
            comparison_score=1.0 if question.id in passed_ids else 0.15,
        )
        for question in questions
    ]
    return run


def _sample_run_for_questions(
    questions: list[BenchmarkQuestion],
    *,
    passed_ids: set[str],
) -> BenchmarkRun:
    run = BenchmarkRun(space_id="clone-123")
    run.results = [
        BenchmarkResult(
            question_id=question.id,
            question=question.question,
            status=BenchmarkStatus.PASSED if question.id in passed_ids else BenchmarkStatus.FAILED,
            expected_sql=question.expected_sql,
            generated_sql=question.expected_sql if question.id in passed_ids else "SELECT 0",
            comparison_score=1.0 if question.id in passed_ids else 0.0,
            comparison_method="result_match" if question.id in passed_ids else "result_mismatch",
            comparison_details=(
                "sample passed" if question.id in passed_ids else "sample failed"
            ),
        )
        for question in questions
    ]
    return run


def _successful_run(space_id: str, questions: list[BenchmarkQuestion]) -> BenchmarkRun:
    run = BenchmarkRun(space_id=space_id)
    run.results = [
        BenchmarkResult(
            question_id=question.id,
            question=question.question,
            status=BenchmarkStatus.PASSED,
            expected_sql=question.expected_sql,
            generated_sql=question.expected_sql,
            comparison_score=1.0,
        )
        for question in questions
    ]
    return run


class _FakeCliFlow:
    def __init__(self, artifacts: ArtifactManager):
        self.artifacts = artifacts
        self.capture_stages: list[str] = []
        self.export_calls = 0
        self.population_calls = 0
        self.rewrite_calls = 0
        self.attach_calls = 0
        self.ensure_calls = 0
        self.curate_calls = 0
        self.query_history_calls = 0
        self.optimize_calls = 0
        self.report_calls = 0
        self.full_benchmark_calls: list[dict[str, object]] = []

    def capture_git_state(self, *, stage: str) -> dict[str, object]:
        self.capture_stages.append(stage)
        return {"available": False}

    def export_workspace(self, *args, **kwargs):
        self.export_calls += 1
        raise AssertionError("export_workspace should not run in this test")

    def run_population_benchmark(self, *args, **kwargs):
        self.population_calls += 1
        raise AssertionError("population baseline should not run in this test")

    def rewrite_split_from_population_baseline(self, *args, **kwargs):
        self.rewrite_calls += 1
        raise AssertionError("split rewrite should not run in this test")

    def attach_existing_clone(self, *args, **kwargs):
        self.attach_calls += 1
        raise AssertionError("attach_existing_clone should not run in this test")

    def ensure_clone(self, *args, **kwargs):
        self.ensure_calls += 1
        raise AssertionError("ensure_clone should not run in this test")

    def curate_workspace_mode(self, *, label: str, mode: str):
        self.curate_calls += 1
        return {"mode": mode, "changes": []}

    def collect_query_history(self, *, label: str, warehouse_id: str | None):
        self.query_history_calls += 1
        return {"status": "unavailable", "warehouse_id": warehouse_id}

    def run_full_benchmark(self, *, label: str, baseline: bool = False):
        self.full_benchmark_calls.append({"label": label, "baseline": baseline})
        return {
            "train": {"pass_rate": 70.0},
            "validation": {"pass_rate": 75.0},
            "test": {"pass_rate": 80.0},
            "overall": {"pass_rate": 75.0},
        }

    def run_autonomous_optimization(self, **kwargs):
        self.optimize_calls += 1
        return {
            "accepted": 1,
            "rejected": 0,
            "skipped": 0,
            "stop_reason": "plateau",
            "best_checkpoint_path": None,
            "final": {"overall": {"pass_rate": 95.0}},
        }

    def write_report(self) -> Path:
        self.report_calls += 1
        self.artifacts.final_report_path.write_text("report", encoding="utf-8")
        return self.artifacts.final_report_path


def test_optimize_fixed_k_samples_initializes_baseline_from_population_samples(
    tmp_path: Path,
    monkeypatch,
):
    space_id = "b" * 32
    questions = [
        BenchmarkQuestion(id=f"{index:032x}"[-32:], question=f"Question {index}", expected_sql=f"SELECT {index}")
        for index in range(3)
    ]
    bundle = ExportBundle(
        space_id=space_id,
        space_title="Demo",
        config=_space_config_with_question_count(3),
        questions=questions,
        split=BenchmarkSplit(
            train_questions=[questions[0]],
            dev_questions=[questions[1]],
            test_questions=[questions[2]],
            split_seed=42,
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
        prompt_path=tmp_path / "prompt.md",
        benchmark_source="fixed_seed",
        warehouse_id="wh-123",
    )
    captured: dict[str, object] = {}

    class _FixedSeedFlow:
        def __init__(self, artifacts: ArtifactManager):
            self.artifacts = artifacts
            self.genie_service = SimpleNamespace(wc=SimpleNamespace())
            self.population_calls: list[dict[str, object]] = []
            self.sample_baseline_calls: list[dict[str, object]] = []
            self.full_benchmark_calls: list[dict[str, object]] = []
            self.rewrite_calls = 0

        def capture_git_state(self, *, stage: str) -> dict[str, object]:
            return {"available": False, "stage": stage}

        def export_workspace(self, *args, **kwargs):
            return bundle

        def ensure_clone(self, *args, **kwargs):
            return GenieSpaceInfo(space_id="clone-123", title="Clone", warehouse_id="wh-123")

        def apply_benchmark_artifact_seed(self, *args, **kwargs):
            return bundle

        def run_population_benchmark(self, *, label: str, questions, k_samples: int = 1):
            self.population_calls.append(
                {"label": label, "question_count": len(questions), "k_samples": k_samples}
            )
            return _sample_run_for_questions(list(questions), passed_ids=set())

        def rewrite_split_from_population_baseline(self, *args, **kwargs):
            self.rewrite_calls += 1
            raise AssertionError("fixed benchmark artifacts must not rewrite the split")

        def precompute_answer_key(self, questions):
            return 0

        def initialize_baseline_from_population_samples(self, *, label: str, questions, population_label: str):
            self.sample_baseline_calls.append(
                {"label": label, "question_count": len(questions), "population_label": population_label}
            )
            return {
                "overall": {"pass_rate": 0.0},
                "train": {"pass_rate": 0.0},
                "validation": {"pass_rate": 0.0},
                "test": {"pass_rate": 0.0},
            }

        def run_full_benchmark(self, *, label: str, baseline: bool = False):
            self.full_benchmark_calls.append({"label": label, "baseline": baseline})
            raise AssertionError("fixed K-sample runs should initialize from population samples")

        def curate_workspace_mode(self, *, label: str, mode: str):
            return {"mode": mode, "changes": []}

        def collect_query_history(self, *, label: str, warehouse_id: str | None):
            return {"status": "unavailable", "warehouse_id": warehouse_id}

        def write_report(self) -> Path:
            self.artifacts.final_report_path.parent.mkdir(parents=True, exist_ok=True)
            self.artifacts.final_report_path.write_text("report", encoding="utf-8")
            return self.artifacts.final_report_path

    def fake_build_flow(*, artifacts: ArtifactManager, **kwargs):
        flow = _FixedSeedFlow(artifacts)
        captured["flow"] = flow
        return flow

    monkeypatch.setattr(cli_module, "_build_flow", fake_build_flow)
    monkeypatch.setattr(
        cli_module,
        "latest_eval_run_observed_payload",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no observed run")),
    )
    monkeypatch.setattr(
        cli_module,
        "default_workspace_dir",
        lambda root, space_id: tmp_path / "workspace" / space_id,
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "optimize",
            "--space-url",
            f"https://workspace.example.com/genie/rooms/{space_id}",
            "--workspace-root",
            str(tmp_path / "workspace"),
            "--fixed-benchmark-artifacts-path",
            str(tmp_path / "seed"),
            "--k-samples",
            "5",
            "--bootstrap-only",
        ],
    )

    assert result.exit_code == 0, result.output
    flow = captured["flow"]
    assert flow.population_calls == [
        {"label": "population_baseline", "question_count": 3, "k_samples": 5}
    ]
    assert flow.sample_baseline_calls == [
        {"label": "baseline", "question_count": 3, "population_label": "population_baseline"}
    ]
    assert flow.full_benchmark_calls == []
    assert flow.rewrite_calls == 0


def test_optimize_no_benchmark_runs_advisory_instead_of_hard_stop(
    tmp_path: Path,
    monkeypatch,
):
    space_id = "c" * 32
    empty_bundle = ExportBundle(
        space_id=space_id,
        space_title="No Benchmark Space",
        config=_space_config(),
        questions=[],
        split=BenchmarkSplit(
            train_questions=[],
            dev_questions=[],
            test_questions=[],
            split_seed=42,
            validation_fraction=0.2,
            test_fraction=0.2,
            split_strategy="no_benchmark",
        ),
        prompt_path=tmp_path / "prompt.md",
        benchmark_source="space_config",
        warehouse_id="wh-123",
    )
    captured: dict[str, object] = {}

    class _NoBenchmarkFlow:
        def __init__(self, artifacts: ArtifactManager):
            self.artifacts = artifacts
            self.genie_service = SimpleNamespace(wc=SimpleNamespace())
            self.export_kwargs: dict[str, object] = {}
            self.ensure_calls = 0
            self.advisory_calls = 0
            self.population_calls = 0
            self.optimize_calls = 0

        def capture_git_state(self, *, stage: str) -> dict[str, object]:
            return {"available": False, "stage": stage}

        def export_workspace(self, *args, **kwargs):
            self.export_kwargs = dict(kwargs)
            return empty_bundle

        def ensure_clone(self, *args, **kwargs):
            self.ensure_calls += 1
            return GenieSpaceInfo(space_id="clone-123", title="Clone", warehouse_id="wh-123")

        def run_no_benchmark_advisory_best_practices(self, *, label: str, curation_mode: str):
            self.advisory_calls += 1
            return {
                "mode": label,
                "curation_mode": curation_mode,
                "clone_space_id": "clone-123",
                "change_count": 1,
            }

        def run_population_benchmark(self, *args, **kwargs):
            self.population_calls += 1
            raise AssertionError("no-benchmark advisory must not run a population baseline")

        def run_autonomous_optimization(self, **kwargs):
            self.optimize_calls += 1
            raise AssertionError("no-benchmark advisory must not run optimization candidates")

        def write_report(self) -> Path:
            self.artifacts.final_report_path.parent.mkdir(parents=True, exist_ok=True)
            self.artifacts.final_report_path.write_text("report", encoding="utf-8")
            return self.artifacts.final_report_path

    def fake_build_flow(*, artifacts: ArtifactManager, **kwargs):
        flow = _NoBenchmarkFlow(artifacts)
        captured["flow"] = flow
        return flow

    monkeypatch.setattr(cli_module, "_build_flow", fake_build_flow)
    monkeypatch.setattr(
        cli_module,
        "default_workspace_dir",
        lambda root, space_id: tmp_path / "workspace" / space_id,
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "optimize",
            "--space-url",
            f"https://workspace.example.com/genie/rooms/{space_id}",
            "--workspace-root",
            str(tmp_path / "workspace"),
            "--bootstrap-only",
        ],
    )

    assert result.exit_code == 0, result.output
    flow = captured["flow"]
    assert flow.export_kwargs["allow_empty_benchmarks"] is True
    assert flow.ensure_calls == 1
    assert flow.advisory_calls == 1
    assert flow.population_calls == 0
    assert flow.optimize_calls == 0
    assert NO_BENCHMARK_WARNING in result.output
    assert "Evidence classification: advisory_best_practices" in result.output
    assert "Final full pass rate: n/a (no benchmark)" in result.output


def test_lineage_registry_per_workspace(tmp_path: Path):
    space_id = "1" * 32
    left = ArtifactManager(tmp_path / "run_a" / space_id, space_id)
    right = ArtifactManager(tmp_path / "run_b" / space_id, space_id)

    left.save_lineage(space_id, {"managed_source_space_id": "clone-a"})
    right.save_lineage(space_id, {"managed_source_space_id": "clone-b"})

    assert left.lineage_registry_path != right.lineage_registry_path
    assert left.load_lineage(space_id) == {
        "managed_source_space_id": "clone-a",
        "updated_at": left.load_lineage(space_id)["updated_at"],
    }
    assert right.load_lineage(space_id) == {
        "managed_source_space_id": "clone-b",
        "updated_at": right.load_lineage(space_id)["updated_at"],
    }


def test_lineage_migration_fallback(tmp_path: Path):
    space_id = "2" * 32
    artifacts = ArtifactManager(tmp_path / "run" / space_id, space_id)
    legacy_payload = {
        space_id: {
            "managed_source_space_id": "legacy-clone",
            "workspace_dir": str(artifacts.workspace_dir),
        }
    }
    artifacts.write_json_path(artifacts.legacy_lineage_registry_path, legacy_payload)

    loaded = artifacts.load_lineage(space_id)
    artifacts.save_lineage(space_id, {"managed_source_space_id": "new-clone"})

    assert loaded == legacy_payload[space_id]
    assert artifacts.read_json_path(artifacts.legacy_lineage_registry_path) == legacy_payload
    assert artifacts.lineage_registry_path.exists()
    assert artifacts.load_lineage(space_id) == {
        "managed_source_space_id": "new-clone",
        "updated_at": artifacts.load_lineage(space_id)["updated_at"],
    }


def test_apply_benchmark_artifact_seed_copies_exact_split_artifacts(tmp_path: Path):
    seed_space_id = "1" * 32
    target_space_id = "2" * 32
    q_train = BenchmarkQuestion(
        id="a" * 32,
        question="Train question",
        expected_sql="SELECT 1",
    )
    q_validation = BenchmarkQuestion(
        id="b" * 32,
        question="Validation question",
        expected_sql="SELECT 2",
    )
    q_test = BenchmarkQuestion(id="c" * 32, question="Test question", expected_sql="SELECT 3")
    seed_artifacts = ArtifactManager(tmp_path / "seed" / seed_space_id, seed_space_id)
    seed_artifacts.write_json_path(
        seed_artifacts.benchmark_payload_path,
        {
            "benchmark_source": "space_config",
            "payload_scope": "visible_pool",
            "question_count": 2,
            "questions": [
                {
                    "id": q_train.id,
                    "question": q_train.question,
                    "expected_sql": q_train.expected_sql,
                },
                {
                    "id": q_validation.id,
                    "question": q_validation.question,
                    "expected_sql": q_validation.expected_sql,
                },
            ],
        },
    )
    seed_artifacts.write_json_path(
        seed_artifacts.hidden_test_payload_path,
        {
            "benchmark_source": "space_config",
            "payload_scope": "hidden_test",
            "question_count": 1,
            "questions": [
                {
                    "id": q_test.id,
                    "question": q_test.question,
                    "expected_sql": q_test.expected_sql,
                }
            ],
        },
    )
    seed_artifacts.write_json_path(
        seed_artifacts.train_set_path,
        {
            "space_id": seed_space_id,
            "question_count": 1,
            "questions": [
                {
                    "id": q_train.id,
                    "question": q_train.question,
                    "expected_sql": q_train.expected_sql,
                }
            ],
        },
    )
    seed_artifacts.write_json_path(
        seed_artifacts.validation_set_path,
        {"space_id": seed_space_id, "question_count": 1, "ids": [q_validation.id]},
    )
    seed_artifacts.write_json_path(
        seed_artifacts.validation_folds_path,
        {
            "space_id": seed_space_id,
            "validation_strategy": "small_sample_kfold",
            "visible_question_count": 2,
            "fold_count": 1,
            "folds": [
                {
                    "fold_index": 0,
                    "train_ids": [q_train.id],
                    "validation_ids": [q_validation.id],
                }
            ],
        },
    )
    seed_artifacts.write_json_path(
        seed_artifacts.test_set_path,
        {"space_id": seed_space_id, "question_count": 1, "ids": [q_test.id]},
    )
    seed_artifacts.write_json_path(
        seed_artifacts.split_meta_path,
        {
            "space_id": seed_space_id,
            "source_space_id": seed_space_id,
            "warehouse_id": "wh-123",
            "benchmark_source": "space_config",
            "split_seed": 42,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "train_ids": [q_train.id],
            "validation_ids": [q_validation.id],
            "test_ids": [q_test.id],
            "visible_ids": [q_train.id, q_validation.id],
            "split_strategy": "baseline_outcome_balanced",
            "validation_strategy": "small_sample_kfold",
            "validation_fold_count": 1,
        },
    )
    seed_payload_hash = hashlib.sha256(
        seed_artifacts.benchmark_payload_path.read_bytes()
    ).hexdigest()
    seed_split_hash = hashlib.sha256(seed_artifacts.split_meta_path.read_bytes()).hexdigest()

    target_artifacts = ArtifactManager(tmp_path / "target" / target_space_id, target_space_id)
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=target_artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=target_artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    bundle = ExportBundle(
        space_id=seed_space_id,
        space_title="Demo Space",
        config=_space_config(),
        questions=[q_train, q_validation, q_test],
        split=BenchmarkSplit(
            train_questions=[q_train, q_validation],
            dev_questions=[],
            test_questions=[q_test],
            split_seed=7,
            validation_fraction=0.0,
            test_fraction=0.2,
        ),
        prompt_path=target_artifacts.prompt_path,
        benchmark_source="space_config",
        warehouse_id="wh-123",
    )

    seeded = flow.apply_benchmark_artifact_seed(
        bundle=bundle,
        seed_path=seed_artifacts.workspace_dir,
    )

    assert [question.id for question in seeded.split.train_questions] == [q_train.id]
    assert [question.id for question in seeded.split.validation_questions] == [
        q_validation.id
    ]
    assert [question.id for question in seeded.split.test_questions] == [q_test.id]
    assert seeded.split.validation_strategy == "small_sample_kfold"
    assert seeded.split.validation_folds == [
        ValidationFold(
            fold_index=0,
            train_ids=[q_train.id],
            validation_ids=[q_validation.id],
        )
    ]
    assert (
        hashlib.sha256(target_artifacts.benchmark_payload_path.read_bytes()).hexdigest()
        == seed_payload_hash
    )
    assert (
        hashlib.sha256(target_artifacts.split_meta_path.read_bytes()).hexdigest()
        == seed_split_hash
    )


def test_preflight_with_recovery_recovers_deleted_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    space_id = "8" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.update_workspace_meta(
        {
            "input_space_id": space_id,
            "managed_source_space_id": "clone-123",
            "active_space_id": "clone-123",
        }
    )
    local_config = _space_config()
    local_config.title = "Recovered Demo Space"
    decompose_config(local_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": "b" * 32, "question": "Validation question", "expected_sql": "SELECT 2"},
            ],
        },
    )
    genie_service = _MissingCloneGenieService(_space_config(), missing_ids={"clone-123"})
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    clone_space_id, _ = flow._preflight_with_recovery()

    clone_meta = artifacts.load_clone_meta()
    workspace_meta = artifacts.load_workspace_meta()
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert clone_space_id == "recovered-1"
    assert clone_meta is not None
    assert clone_meta["clone_space_id"] == "recovered-1"
    assert workspace_meta is not None
    assert workspace_meta["managed_source_space_id"] == "recovered-1"
    assert '"phase": "clone_recovered"' in iteration_log


def test_normalize_clone_parent_path_accepts_workspace_file_paths():
    assert _normalize_clone_parent_path("/Workspace/Users/user@example.com") == "/Users/user@example.com"
    assert _normalize_clone_parent_path("/Workspace/Shared/team/maxgenie/") == "/Shared/team/maxgenie"
    assert _normalize_clone_parent_path("  /Users/user@example.com/  ") == "/Users/user@example.com"


def test_failed_ids_from_validation_snapshot_includes_failed_and_error_ids(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.validation_run_history_dir.mkdir(parents=True, exist_ok=True)
    artifacts.write_json_path(
        artifacts.validation_run_history_dir / "20260523T120000.gate.validation.json",
        {
            "failed_ids": ["failed-1"],
            "error_ids": ["error-1"],
        },
    )
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    assert flow._failed_ids_from_snapshot(kind="validation", label="gate.validation") == [
        "failed-1",
        "error-1",
    ]


def test_serving_candidate_parse_error_is_recorded_as_skipped_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _MalformedCandidateClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **_kwargs):
            raise ServingCandidateResponseError(
                "bad json",
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"choices": [{"message": {"content": "{bad"}}]},
                content="{bad",
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _MalformedCandidateClient)

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    assert candidate.to_serialized_space() == config.to_serialized_space()
    assert summary["changed"] is False
    assert summary["reason"] == "invalid_serving_response_json"
    assert Path(summary["raw_content_path"]).read_text(encoding="utf-8") == "{bad"
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert '"phase": "serving_candidate_proposal_invalid"' in iteration_log


def test_artifact_patch_serving_candidate_updates_decomposed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _PatchCandidateClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            assert kwargs["candidate_mode"] == "artifact_patch"
            assert kwargs["decomposed_space_payload"]["files"]
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "patch_instruction",
                    "rationale": "Coherent instruction patch.",
                    "expected_impact": "Improve visible period guidance.",
                    "risk": "Allowlisted file edit only.",
                    "owner_followups": [],
                    "file_edits": [
                        {
                            "path": "instructions/text_instruction.md",
                            "content": original_instruction + "\nUse exact current/prior time-bucket joins.\n",
                            "expected_sha256": hashlib.sha256(
                                original_instruction.encode("utf-8")
                            ).hexdigest(),
                        }
                    ],
                    "operations": [],
                },
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _PatchCandidateClient)

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="artifact_patch",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    assert summary["changed"] is True
    assert summary["candidate_mode"] == "artifact_patch"
    assert summary["file_edit_count"] == 1
    assert summary["risk_assessment"]["accepted"] is True
    assert "Use exact current/prior time-bucket joins." in instruction_path.read_text(encoding="utf-8")
    assert "Use exact current/prior time-bucket joins." in "\n".join(
        candidate.instructions.text_instructions[0].content
    )
    assert Path(summary["payload_path"]).exists()
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert '"forced_candidate_mode": "artifact_patch"' in iteration_log


def test_artifact_patch_v2_serving_candidate_repairs_missing_hash_and_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _PatchV2CandidateClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            assert kwargs["candidate_mode"] == "artifact_patch_v2"
            assert kwargs["decomposed_space_payload"]["files"]
            first_file = kwargs["decomposed_space_payload"]["files"][0]
            assert first_file["expected_sha256"] == first_file["sha256"]
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "patch_instruction_v2",
                    "rationale": "Precise instruction patch.",
                    "expected_impact": "Improve visible period guidance.",
                    "risk": "Allowlisted file edit only.",
                    "owner_followups": [],
                    "file_edits": [
                        {
                            "relative_path": "instructions/text_instruction.md",
                            "content": original_instruction + "\nUse exact current/prior time-bucket joins.\n",
                        }
                    ],
                    "operations": [{"op": "no_change"}],
                },
                parse_repairs=("balanced_json_delimiters",),
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _PatchV2CandidateClient)

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="artifact_patch_v2",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    assert summary["changed"] is True, summary["risk_assessment"]
    assert summary["candidate_mode"] == "artifact_patch_v2"
    assert summary["parse_repairs"] == ["balanced_json_delimiters"]
    assert summary["artifact_patch_repair"]["normalized_operations"] is True
    assert summary["artifact_patch_repair"]["filled_expected_sha256_paths"] == [
        "instructions/text_instruction.md"
    ]
    assert summary["artifact_patch_repair"]["filled_reason_paths"] == [
        "instructions/text_instruction.md"
    ]
    assert summary["risk_assessment"]["accepted"] is True
    assert "Use exact current/prior time-bucket joins." in instruction_path.read_text(encoding="utf-8")
    assert "Use exact current/prior time-bucket joins." in "\n".join(
        candidate.instructions.text_instructions[0].content
    )
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert '"phase": "serving_candidate_payload_parse_repaired"' in iteration_log
    assert '"phase": "serving_candidate_artifact_patch_repaired"' in iteration_log


def test_artifact_patch_v2_rejects_generic_patch_when_visible_context_exists(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "generic_instruction_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": "Add generic period guidance.",
            "expected_impact": "Improve SQL quality.",
            "risk": "Allowlisted instruction edit.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": "general period guidance",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "time_filter_mismatch",
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert "artifact_patch_v2_missing_visible_effect_mapping" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["applied"] is True
    assert verification["missing_visible_effect_paths"] == [
        "instructions/text_instruction.md"
    ]
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_records_explicit_no_change_without_risk_rejection(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "no_change_after_no_safe_patch_surface",
            "candidate_mode": "artifact_patch_v2",
            "rationale": "No safe change.",
            "expected_impact": "No mutation.",
            "risk": "No mutation.",
            "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
            "file_edits": None,
            "causal_patch_plan": None,
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "time_filter_mismatch",
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "no_change"
    assert summary["artifact_patch"]["rejected_count"] == 0
    assert summary["artifact_patch"]["edit_count"] == 0
    assert summary["risk_assessment"]["accepted"] is True
    assert summary["risk_assessment"]["reason"] is None
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is True
    assert verification["policy"]["reason"] == "explicit_no_change"


def test_artifact_patch_v2_accepts_patch_with_visible_effect_marker(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    second_id = "b" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "targeted_instruction_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the current/prior period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "global default-window rewrites",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["current", "prior"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
                {
                    "question_id": second_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True
    assert summary["risk_assessment"]["accepted"] is True
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is True
    assert verification["causal_patch_plan"] == {
        "present": True,
        "selected_marker": f"target_train_failure:{target_id}",
        "target_train_failure_id": target_id,
        "target_family": "time_filter_mismatch",
        "selected_family_marker": "",
        "family": "time_filter_mismatch",
        "root_cause": "time window mismatch",
        "edit_surface_evidence": "instruction guidance edit surface",
        "repair_shape": "append one narrow time-window instruction",
        "avoid": "global default-window rewrites",
        "expected_visible_effect": "generated SQL uses current/prior labels",
        "compatible_paths": ["instructions/text_instruction.md"],
        "structural_tokens": ["current", "prior"],
        "available_structural_tokens": [],
        "available_shared_structural_tokens": [],
        "available_shared_missing_expected_tokens": [],
        "available_operation_families": ["append_text_instruction"],
        "available_sql_delta_intents": ["align_time_window_filter"],
        "selected_sql_delta_intents": [],
        "recipe_anchors": {
            "repair_shape": [
                "time",
                "window",
                "grain",
                "bucket",
                "period",
                "label",
                "instruction",
                "text",
                "guidance",
            ],
            "avoid": ["default", "window", "global", "full-query", "snippet"],
            "root_cause": [
                "time",
                "window",
                "period",
                "date",
                "grain",
                "bucket",
            ],
            "edit_surface": [
                "instruction",
                "guidance",
            ],
        },
        "root_cause_matches_visible_evidence": True,
        "edit_surface_matches_visible_evidence": True,
        "repair_shape_matches_recipe": True,
        "avoid_matches_recipe": True,
        "structural_tokens_match_visible_evidence": True,
        "expected_visible_effect_matches_visible_evidence": True,
        "structural_tokens_include_shared_family_token": True,
        "expected_visible_effect_matches_shared_family_token": True,
        "structural_tokens_include_shared_missing_expected_token": True,
        "selected_family_marker_matches_plan": True,
        "matches_visible_target": True,
        "matches_visible_family": True,
        "policy": "present_plan_consistency_checked",
    }
    assert verification["edit_mappings"] == [
        {
            "path": "instructions/text_instruction.md",
            "effect_markers": [f"target_train_failure:{target_id}"],
            "local_effect_markers": [f"target_train_failure:{target_id}"],
            "path_operation_families": ["append_text_instruction"],
        }
    ]
    assert "Use precise current and prior period labels." in instruction_path.read_text(
        encoding="utf-8"
    )
    assert "Use precise current and prior period labels." in "\n".join(
        candidate.instructions.text_instructions[0].content
    )


def test_artifact_patch_v2_rejects_plan_marker_only_in_global_rationale(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "global_only_marker_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the current/prior period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "global default-window rewrites",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["current", "prior"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        "visible_family:time_filter_mismatch"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "selected_marker_not_used_by_each_edit",
        "selected_marker": f"target_train_failure:{target_id}",
        "candidate_markers": [f"target_train_failure:{target_id}"],
        "edit_paths_without_selected_marker": ["instructions/text_instruction.md"],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_visible_patch_without_causal_plan(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "targeted_instruction_without_plan",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the current/prior period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_missing_causal_patch_plan" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["missing_causal_patch_plan"] is True
    assert verification["causal_patch_plan"] == {
        "present": False,
        "matches_visible_target": False,
        "matches_visible_family": False,
        "policy": "missing_plan_non_gating",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_target_mismatch(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    first_id = "a" * 32
    second_id = "b" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "contradictory_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{first_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{second_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "unrelated target markers",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{first_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": first_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
                {
                    "question_id": second_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "selected_marker_not_used_by_patch",
        "selected_marker": f"target_train_failure:{second_id}",
        "candidate_markers": [f"target_train_failure:{second_id}"],
        "patch_markers": [f"target_train_failure:{first_id}"],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_raw_id_causal_plan_marker(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "raw_marker_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": target_id,
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "raw marker ambiguity",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "selected_marker_not_exact_visible_marker",
        "selected_marker": target_id,
        "required_marker_shapes": [
            "target_train_failure:<visible_failure_id>",
            "visible_family:<visible_failure_family>",
            "failure_family:<visible_failure_family>",
        ],
        "patch_markers": [f"target_train_failure:{target_id}"],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_accepts_matching_family_causal_plan_marker(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    second_id = "b" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "family_marker_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": "Target visible family visible_family:time_filter_mismatch.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": "visible_family:time_filter_mismatch",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "global default-window guidance",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        "visible_family:time_filter_mismatch"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
                {
                    "question_id": second_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True
    assert summary["risk_assessment"]["accepted"] is True
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_patch_plan"]["selected_family_marker"] == (
        "time_filter_mismatch"
    )
    assert verification["causal_patch_plan"]["selected_family_marker_matches_plan"] is True
    assert verification["causal_plan_inconsistency"] is None
    assert "Use precise current and prior period labels." in instruction_path.read_text(
        encoding="utf-8"
    )
    assert "Use precise current and prior period labels." in "\n".join(
        candidate.instructions.text_instructions[0].content
    )


def test_artifact_patch_v2_rejects_family_plan_without_shared_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    family = "literal_or_parameter_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "family_member_literal_token_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Generated SQL literalizes 'farxiga'.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "literal placeholder mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow literal instruction",
                "avoid": "bind placeholder guard clone",
                "expected_visible_effect": "generated SQL literalizes 'farxiga'",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["'farxiga'"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nLiteralize the 'farxiga' visible entity value.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": f"literal guidance for visible_family:{family}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                },
                {
                    "question_id": "b" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["'lokelma'"],
                    "extra_generated_tokens": [":brand_name"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "family_structural_tokens_missing_shared_evidence",
        "selected_marker": f"visible_family:{family}",
        "plan_structural_tokens": ["'farxiga'"],
        "available_shared_structural_tokens": [":brand_name"],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_accepts_family_plan_with_normalized_shared_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    family = "literal_or_parameter_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "family_shared_token_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Generated SQL removes brand_name placeholders.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "literal placeholder mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow literal instruction",
                "avoid": "bind placeholder guard clone",
                "expected_visible_effect": "generated SQL removes brand_name placeholders",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["brand_name"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nWhen a visible entity value is known, do not emit "
                        "brand_name placeholders.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": f"literal guidance for visible_family:{family}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                },
                {
                    "question_id": "b" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["'lokelma'"],
                    "extra_generated_tokens": [":brand_name"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True
    assert summary["risk_assessment"]["accepted"] is True
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] is None
    assert verification["causal_patch_plan"][
        "available_shared_structural_tokens"
    ] == [":brand_name"]
    assert verification["causal_patch_plan"][
        "structural_tokens_include_shared_family_token"
    ] is True
    assert "brand_name placeholders" in instruction_path.read_text(encoding="utf-8")
    assert "brand_name placeholders" in "\n".join(
        candidate.instructions.text_instructions[0].content
    )


def test_artifact_patch_v2_rejects_family_executable_without_shared_expected_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "expressions"
        / "prior_nrx_family.yml"
    )
    family = "selected_metric_or_column_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "family_wrong_metric_expression_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Generated SQL stops using current_nrx as the metric alias.",
            "risk": "New allowlisted expression snippet.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "metric column alias mismatch for current_nrx",
                "edit_surface_evidence": "snippet expression edit surface",
                "repair_shape": "add one expression alias snippet",
                "avoid": "alternate descriptor lookup",
                "expected_visible_effect": "generated SQL keeps current_nrx as the visible family token",
                "compatible_paths": ["instructions/sql_snippets/expressions/*.yml"],
                "structural_tokens": ["current_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/expressions/prior_nrx_family.yml",
                    "content": "\n".join(
                        [
                            "display_name: Current NRX family alias",
                            "sql: |",
                            "  current_nrx",
                            "",
                        ]
                    ),
                    "reason": f"metric alias snippet for visible_family:{family}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                },
                {
                    "question_id": "b" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_executable_family_missing_expected_token" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["executable_family_missing_expected_token_gaps"] == [
        {
            "path": "instructions/sql_snippets/expressions/prior_nrx_family.yml",
            "selected_marker": f"visible_family:{family}",
            "selected_family": family,
            "required_shared_missing_expected_tokens": ["prior_nrx"],
            "plan_structural_tokens": ["current_nrx"],
            "plan_has_shared_missing_expected_token": False,
            "content_has_shared_missing_expected_token": False,
        }
    ]
    assert not snippet_path.exists()


def test_artifact_patch_v2_accepts_family_executable_with_shared_expected_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "expressions"
        / "prior_nrx_family.yml"
    )
    family = "selected_metric_or_column_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "family_metric_expression_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Generated SQL returns prior_nrx as the metric alias.",
            "risk": "New allowlisted expression snippet.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "metric column alias mismatch for prior_nrx",
                "edit_surface_evidence": "snippet expression edit surface",
                "repair_shape": "add one expression alias snippet",
                "avoid": "alternate descriptor lookup",
                "expected_visible_effect": "generated SQL returns prior_nrx as the visible family token",
                "compatible_paths": ["instructions/sql_snippets/expressions/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/expressions/prior_nrx_family.yml",
                    "content": "\n".join(
                        [
                            "display_name: Prior NRX family alias",
                            "sql: |",
                            "  prior_nrx",
                            "",
                        ]
                    ),
                    "reason": f"metric alias snippet for visible_family:{family}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                },
                {
                    "question_id": "b" * 32,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True, summary["risk_assessment"]
    assert summary["risk_assessment"]["reasons"] == []
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["executable_family_missing_expected_token_gaps"] == []
    assert "prior_nrx" in snippet_path.read_text(encoding="utf-8")


def test_artifact_patch_v2_rejects_family_causal_plan_marker_mismatch(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "mismatched_family_marker_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": "Target visible family visible_family:time_filter_mismatch.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": "visible_family:time_filter_mismatch",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal placeholder mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "family marker mismatch",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
                {
                    "question_id": "b" * 32,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "selected_family_marker_mismatch",
        "selected_marker": "visible_family:time_filter_mismatch",
        "selected_family_marker": "time_filter_mismatch",
        "plan_family": "literal_or_parameter_mismatch",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_without_family(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "missing_family_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "ambiguous failure-family targeting",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "missing_family",
        "selected_marker": f"target_train_failure:{target_id}",
        "target_family": "time_filter_mismatch",
        "selected_family_marker": "",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_family_mismatch(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    other_id = "b" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "wrong_family_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal placeholder mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "wrong failure family",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
                {
                    "question_id": other_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "selected_target_family_mismatch",
        "selected_marker": f"target_train_failure:{target_id}",
        "target_train_failure_id": target_id,
        "target_family": "time_filter_mismatch",
        "plan_family": "literal_or_parameter_mismatch",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


@pytest.mark.parametrize(
    ("empty_field", "expected_reason"),
    [
        ("repair_shape", "missing_repair_shape"),
        ("avoid", "missing_avoid"),
    ],
)
def test_artifact_patch_v2_rejects_causal_plan_without_repair_or_avoid(
    tmp_path: Path,
    empty_field: str,
    expected_reason: str,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    causal_patch_plan = {
        "selected_marker": f"target_train_failure:{target_id}",
        "family": "time_filter_mismatch",
        "root_cause": "time window mismatch",
        "edit_surface_evidence": "instruction guidance edit surface",
        "repair_shape": "append one narrow time-window instruction",
        "avoid": "global default-window changes",
        "expected_visible_effect": "generated SQL uses current and prior period labels",
        "compatible_paths": ["instructions/text_instruction.md"],
        "structural_tokens": [],
    }
    causal_patch_plan[empty_field] = ""

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": f"missing_{empty_field}_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": causal_patch_plan,
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": expected_reason,
        "selected_marker": f"target_train_failure:{target_id}",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


@pytest.mark.parametrize(
    ("root_cause", "expected_inconsistency"),
    [
        (
            "",
            {
                "reason": "missing_root_cause",
                "selected_marker": "target_train_failure:{target_id}",
            },
        ),
        (
            "metadata alias issue",
            {
                "reason": "root_cause_not_from_visible_evidence",
                "selected_marker": "target_train_failure:{target_id}",
                "root_cause": "metadata alias issue",
                "recipe_anchors": [
                    "time",
                    "window",
                    "period",
                    "date",
                    "grain",
                    "bucket",
                ],
            },
        ),
    ],
)
def test_artifact_patch_v2_rejects_causal_plan_root_cause_not_visible(
    tmp_path: Path,
    root_cause: str,
    expected_inconsistency: dict[str, object],
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "bad_root_cause_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": root_cause,
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "global default-window changes",
                "expected_visible_effect": "generated SQL uses current and prior period labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        key: (
            str(value).format(target_id=target_id)
            if isinstance(value, str)
            else value
        )
        for key, value in expected_inconsistency.items()
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


@pytest.mark.parametrize(
    ("edit_surface_evidence", "expected_inconsistency"),
    [
        (
            "",
            {
                "reason": "missing_edit_surface_evidence",
                "selected_marker": "target_train_failure:{target_id}",
            },
        ),
        (
            "metadata alias edit surface",
            {
                "reason": "edit_surface_evidence_not_from_visible_evidence",
                "selected_marker": "target_train_failure:{target_id}",
                "edit_surface_evidence": "metadata alias edit surface",
                "recipe_anchors": [
                    "instruction",
                    "guidance",
                ],
            },
        ),
    ],
)
def test_artifact_patch_v2_rejects_causal_plan_edit_surface_not_visible(
    tmp_path: Path,
    edit_surface_evidence: str,
    expected_inconsistency: dict[str, object],
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "bad_edit_surface_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": edit_surface_evidence,
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "global default-window changes",
                "expected_visible_effect": "generated SQL uses current and prior period labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    expected = {
        key: (
            value.format(target_id=target_id)
            if isinstance(value, str)
            else value
        )
        for key, value in expected_inconsistency.items()
    }
    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == expected
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_repair_not_from_recipe(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "wrong_repair_recipe_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "add metadata aliases for business entities",
                "avoid": "global default-window changes",
                "expected_visible_effect": "generated SQL uses current and prior period labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "repair_shape_not_from_recipe",
        "selected_marker": f"target_train_failure:{target_id}",
        "repair_shape": "add metadata aliases for business entities",
        "recipe_anchors": [
            "time",
            "window",
            "grain",
            "bucket",
            "period",
            "label",
            "instruction",
            "text",
            "guidance",
        ],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_avoid_not_from_recipe(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "wrong_avoid_recipe_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "unrelated target marker drift",
                "expected_visible_effect": "generated SQL uses current and prior period labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "avoid_not_from_recipe",
        "selected_marker": f"target_train_failure:{target_id}",
        "avoid": "unrelated target marker drift",
        "recipe_anchors": ["default", "window", "global", "full-query", "snippet"],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_without_expected_effect(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "missing_effect_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "empty expected effect",
                "expected_visible_effect": "",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "missing_expected_visible_effect",
        "selected_marker": f"target_train_failure:{target_id}",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_effect_without_visible_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "generic_effect_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve prior/current percentage output.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "percentage_or_comparison_metric",
                "root_cause": "percentage comparison misses prior_nrx",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow comparison instruction",
                "avoid": "generic visible-effect claim",
                "expected_visible_effect": "generated SQL matches the visible result shape",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nFor percentage comparisons, preserve prior_nrx and "
                        "percentage_difference aliases.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "targeted period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "expected_visible_effect_not_from_visible_evidence",
        "selected_marker": f"target_train_failure:{target_id}",
        "expected_visible_effect": "generated SQL matches the visible result shape",
        "available_structural_tokens": [
            "current_nrx",
            "percentage_difference",
            "prior_nrx",
        ],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_unrelated_structural_tokens(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "unrelated_token_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve prior/current percentage output.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "percentage_or_comparison_metric",
                "root_cause": "percentage comparison misses prior_nrx",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow comparison instruction",
                "avoid": "invented structural token",
                "expected_visible_effect": "generated SQL keeps prior_nrx alias",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["unrelated_alias"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nFor percentage comparisons, preserve prior_nrx and "
                        "percentage_difference aliases.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": "tighten comparison guidance",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "structural_tokens_not_from_visible_evidence",
        "selected_marker": f"target_train_failure:{target_id}",
        "plan_structural_tokens": ["unrelated_alias"],
        "available_structural_tokens": [
            "current_nrx",
            "percentage_difference",
            "prior_nrx",
        ],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_mixed_structural_tokens(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "mixed_token_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve prior/current percentage output.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "percentage_or_comparison_metric",
                "root_cause": "percentage comparison misses prior_nrx",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow comparison instruction",
                "avoid": "mixing visible and invented structural tokens",
                "expected_visible_effect": "generated SQL keeps prior_nrx alias",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["prior_nrx", "hallucinated_alias"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nFor percentage comparisons, preserve prior_nrx and "
                        "percentage_difference aliases.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": "tighten comparison guidance",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "structural_tokens_not_from_visible_evidence",
        "selected_marker": f"target_train_failure:{target_id}",
        "plan_structural_tokens": ["prior_nrx", "hallucinated_alias"],
        "available_structural_tokens": [
            "current_nrx",
            "percentage_difference",
            "prior_nrx",
        ],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_without_compatible_paths(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "missing_compatible_paths_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "unbounded artifact path changes",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": [],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": "tighten period guidance",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "missing_compatible_paths",
        "selected_marker": f"target_train_failure:{target_id}",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_causal_plan_path_mismatch(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "wrong_path_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "add one executable example",
                "avoid": "text-only guidance",
                "expected_visible_effect": "generated SQL uses current/prior labels",
                "compatible_paths": ["instructions/examples/*.yml"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "changed_path_not_in_plan_compatible_paths",
        "selected_marker": f"target_train_failure:{target_id}",
        "compatible_paths": ["instructions/examples/*.yml"],
        "changed_paths": ["instructions/text_instruction.md"],
        "incompatible_paths": ["instructions/text_instruction.md"],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_high_risk_target_without_guard_marker(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    guard_id = "g" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "high_risk_without_guard_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": "High guard risk target, but no explicit guard marker.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": "tighten period guidance",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                    "guard_regression_risk": "high",
                }
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                }
            ],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_high_risk_target_without_guard_preservation" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["high_risk_guard_gap"] == {
        "target_train_failure_id": target_id,
        "guard_regression_risk": "high",
        "visible_pass_guard_ids": [guard_id],
        "required_marker_shape": "guard_preservation:<visible_pass_guard_id>",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_high_risk_family_without_guard_marker(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    first_id = "a" * 32
    second_id = "b" * 32
    third_id = "c" * 32
    guard_id = "g" * 32
    family = "time_filter_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "high_risk_family_without_guard_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Generated SQL uses current and prior period labels.",
            "risk": "High guard risk family plan, but no explicit guard marker.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow period instruction",
                "avoid": "global default-window rewrites",
                "expected_visible_effect": (
                    "generated SQL uses current and prior period labels"
                ),
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten shared period guidance for "
                        f"visible_family:{family}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failure_families": {
                family: [first_id, second_id, third_id],
            },
            "failures": [
                {
                    "question_id": first_id,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["append_text_instruction"],
                    "guard_regression_risk": "high",
                },
                {
                    "question_id": second_id,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["append_text_instruction"],
                    "guard_regression_risk": "medium",
                },
                {
                    "question_id": third_id,
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["set_column_config"],
                    "guard_regression_risk": "high",
                },
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                }
            ],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_high_risk_target_without_guard_preservation" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["high_risk_guard_gap"] == {
        "visible_family": family,
        "guard_regression_risk": "high",
        "high_guard_risk_member_ids": [first_id],
        "visible_pass_guard_ids": [guard_id],
        "required_marker_shape": "guard_preservation:<visible_pass_guard_id>",
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_accepts_high_risk_target_with_guard_marker(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    guard_id = "g" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "high_risk_with_guard_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": (
                f"Target visible failure target_train_failure:{target_id} while "
                f"preserving visible guard guard_preservation:{guard_id}."
            ),
            "expected_impact": "Improve the visible period SQL pattern.",
            "risk": f"Explicitly preserves guard_preservation:{guard_id}.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow time-window instruction",
                "avoid": "global default-window rewrites",
                "expected_visible_effect": "generated SQL uses current and prior period labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": [],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                    "guard_regression_risk": "high",
                }
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                }
            ],
        },
    )

    assert summary["changed"] is True
    assert summary["risk_assessment"]["accepted"] is True
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["high_risk_guard_gap"] is None
    assert f"guard_preservation:{guard_id}" in verification["effect_markers"]
    assert "Use precise current and prior period labels." in instruction_path.read_text(
        encoding="utf-8"
    )
    assert "Use precise current and prior period labels." in "\n".join(
        candidate.instructions.text_instructions[0].content
    )


def test_artifact_patch_v2_rejects_marker_on_incompatible_artifact_path(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    table_path = artifacts.space_dir / "tables" / "sales.yml"
    original_table = table_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "incompatible_table_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve a reusable snippet-shaped failure.",
            "risk": "Allowlisted table edit.",
            "operations": [],
            "file_edits": [
                {
                    "path": "tables/sales.yml",
                    "append_content": "\n# unsupported local table note\n",
                    "expected_sha256": hashlib.sha256(
                        original_table.encode("utf-8")
                    ).hexdigest(),
                    "reason": "table note for visible target",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert "artifact_patch_v2_visible_effect_path_incompatible" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is False
    assert verification["incompatible_visible_effect_paths"] == [
        {
            "path": "tables/sales.yml",
            "path_operation_families": [
                "set_column_config",
                "set_table_description",
            ],
            "markers": [
                {
                    "marker": f"target_train_failure:{target_id}",
                    "suggested_operation_families": ["add_sql_snippet"],
                }
            ],
        }
    ]
    assert table_path.read_text(encoding="utf-8") == original_table


def test_artifact_patch_v2_rejects_edit_content_without_causal_plan_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    table_path = artifacts.space_dir / "tables" / "sales.yml"
    original_table = table_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "metadata_without_plan_token_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Make generated SQL include prior_nrx.",
            "risk": "Allowlisted table metadata edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "selected_metric_or_column_mismatch",
                "root_cause": "metric column prior_nrx mismatch",
                "edit_surface_evidence": "metadata column schema edit surface",
                "repair_shape": "metadata column prior_nrx alias clarification",
                "avoid": "alternate descriptor lookup",
                "expected_visible_effect": "generated SQL includes prior_nrx",
                "compatible_paths": ["tables/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "tables/sales.yml",
                    "append_content": "\n# Metric metadata guidance for the selected target.\n",
                    "expected_sha256": hashlib.sha256(
                        original_table.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "metadata update for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "missing_expected_tokens": ["prior_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_edit_content_missing_causal_plan_token" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["off_target_causal_plan_content_edits"] == [
        {
            "path": "tables/sales.yml",
            "selected_marker": f"target_train_failure:{target_id}",
            "plan_structural_tokens": ["prior_nrx"],
        }
    ]
    assert table_path.read_text(encoding="utf-8") == original_table


def test_artifact_patch_v2_rejects_example_without_target_question_overlap(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    example_path = artifacts.space_dir / "instructions" / "examples" / "unrelated.yml"
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "unrelated_example_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Make generated SQL include prior_nrx.",
            "risk": "One visible executable example.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "selected_metric_or_column_mismatch",
                "root_cause": "metric column prior_nrx mismatch",
                "edit_surface_evidence": "example executable edit surface",
                "repair_shape": "example column alias prior_nrx clarification",
                "avoid": "lookup guard literal copy",
                "expected_visible_effect": "generated SQL includes prior_nrx",
                "compatible_paths": ["instructions/examples/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/unrelated.yml",
                    "content": "\n".join(
                        [
                            "id: unrelated_example",
                            "question: Show sales by market",
                            "sql: SELECT prior_nrx FROM sales",
                            "",
                        ]
                    ),
                    "reason": (
                        "example update for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Show current and prior NRX for Farxiga",
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                    "missing_expected_tokens": ["prior_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_example_not_target_aligned" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["off_target_example_edits"] == [
        {
            "path": "instructions/examples/unrelated.yml",
            "selected_marker": f"target_train_failure:{target_id}",
            "target_train_failure_id": target_id,
            "target_question_terms": ["current", "farxiga", "nrx", "prior"],
            "candidate_question_terms": ["market", "sales"],
        }
    ]
    assert not example_path.exists()


def test_artifact_patch_v2_rejects_example_without_family_question_overlap(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    example_path = artifacts.space_dir / "instructions" / "examples" / "unrelated.yml"
    first_id = "a" * 32
    second_id = "b" * 32
    third_id = "c" * 32
    family = "selected_metric_or_column_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "unrelated_family_example_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Make generated SQL include prior_nrx.",
            "risk": "One visible executable example.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "metric column prior_nrx mismatch",
                "edit_surface_evidence": "example executable edit surface",
                "repair_shape": "example column alias prior_nrx clarification",
                "avoid": "lookup guard literal copy",
                "expected_visible_effect": "generated SQL includes prior_nrx",
                "compatible_paths": ["instructions/examples/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/unrelated.yml",
                    "content": "\n".join(
                        [
                            "id: unrelated_family_example",
                            "question: Show sales by market",
                            "sql: SELECT prior_nrx FROM sales",
                            "",
                        ]
                    ),
                    "reason": f"example update for visible_family:{family}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": first_id,
                    "question": "Show current and prior NRX for Farxiga",
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_example_sql"],
                    "missing_expected_tokens": ["prior_nrx"],
                },
                {
                    "question_id": second_id,
                    "question": "Show current and prior TRX for Lokelma",
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_example_sql"],
                    "missing_expected_tokens": ["prior_nrx"],
                },
                {
                    "question_id": third_id,
                    "question": "Show sales by market",
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["set_column_config"],
                    "missing_expected_tokens": ["sales_market"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_example_not_target_aligned" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["off_target_example_edits"] == [
        {
            "path": "instructions/examples/unrelated.yml",
            "selected_marker": f"visible_family:{family}",
            "selected_family": family,
            "family_question_terms": {
                first_id: ["current", "farxiga", "nrx", "prior"],
                second_id: ["current", "lokelma", "prior", "trx"],
            },
            "candidate_question_terms": ["market", "sales"],
        }
    ]
    assert not example_path.exists()


def test_artifact_patch_v2_rejects_family_marker_without_shared_edit_surface(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    example_path = artifacts.space_dir / "instructions" / "examples" / "family.yml"
    first_id = "a" * 32
    second_id = "b" * 32
    family = "selected_metric_or_column_mismatch"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "mixed_surface_family_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible family visible_family:{family}.",
            "expected_impact": "Make generated SQL include prior_nrx.",
            "risk": "One visible executable example.",
            "causal_patch_plan": {
                "selected_marker": f"visible_family:{family}",
                "family": family,
                "root_cause": "metric column prior_nrx mismatch",
                "edit_surface_evidence": "example executable edit surface",
                "repair_shape": "example column alias prior_nrx clarification",
                "avoid": "lookup guard literal copy",
                "expected_visible_effect": "generated SQL includes prior_nrx",
                "compatible_paths": ["instructions/examples/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/family.yml",
                    "content": "\n".join(
                        [
                            "id: mixed_surface_family_example",
                            "question: Show current and prior NRX for Farxiga",
                            "sql: SELECT prior_nrx FROM sales",
                            "",
                        ]
                    ),
                    "reason": f"example update for visible_family:{family}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": first_id,
                    "question": "Show current and prior NRX for Farxiga",
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["add_example_sql"],
                    "missing_expected_tokens": ["prior_nrx"],
                },
                {
                    "question_id": second_id,
                    "question": "Show current and prior TRX for Lokelma",
                    "likely_root_cause_family": family,
                    "suggested_operation_families": ["set_column_config"],
                    "missing_expected_tokens": ["prior_trx"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_inconsistent" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["causal_plan_inconsistency"] == {
        "reason": "family_marker_without_shared_edit_surface",
        "selected_marker": f"visible_family:{family}",
        "selected_family_marker": family,
    }
    assert not example_path.exists()


def test_artifact_patch_v2_rejects_new_file_for_existing_surface_plan(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    new_table_path = artifacts.space_dir / "tables" / "invented_sales.yml"
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "invented_metadata_file_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Make generated SQL include prior_nrx.",
            "risk": "Metadata edit only.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "selected_metric_or_column_mismatch",
                "root_cause": "metric column prior_nrx mismatch",
                "edit_surface_evidence": "metadata table column schema edit surface",
                "repair_shape": "metadata column prior_nrx alias clarification",
                "avoid": "alternate descriptor lookup",
                "expected_visible_effect": "generated SQL includes prior_nrx",
                "compatible_paths": ["tables/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "tables/invented_sales.yml",
                    "content": "\n".join(
                        [
                            "identifier: invented_sales",
                            "column_configs:",
                            "  - column_name: prior_nrx",
                            "    description: Prior NRx metric column for the visible target.",
                            "",
                        ]
                    ),
                    "reason": (
                        "metadata update for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "missing_expected_tokens": ["prior_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_causal_plan_unexpected_new_file" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["unexpected_new_file_edits"] == [
        {
            "path": "tables/invented_sales.yml",
            "selected_marker": f"target_train_failure:{target_id}",
            "operation_families": ["set_column_config"],
            "allowed_new_file_operation_families": [
                "add_example_sql",
                "add_join_spec",
                "add_sql_snippet",
            ],
        }
    ]
    assert not new_table_path.exists()


def test_artifact_patch_v2_rejects_full_file_rewrite_for_existing_file(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "full_file_instruction_rewrite_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL uses current/prior labels.",
            "risk": "One existing text-guidance asset.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window current prior mismatch",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append narrow current prior period guidance",
                "avoid": "global default-window rewrites",
                "expected_visible_effect": "generated SQL uses current prior labels",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["current", "prior"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": (
                        original_instruction
                        + "\nUse precise current and prior period labels.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "tighten period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["current", "prior"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_full_file_rewrite_not_allowed" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["full_file_rewrite_edits"] == [
        {
            "path": "instructions/text_instruction.md",
            "selected_marker": f"target_train_failure:{target_id}",
            "edit_mode": "replace_file",
            "action": "updated",
        }
    ]
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_text_patch_without_target_failure_tokens(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "generic_marker_instruction_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the period-comparison SQL pattern.",
            "risk": "Allowlisted instruction edit.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "targeted period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_text_edit_missing_failure_token" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["off_target_text_edits"] == [
        {
            "path": "instructions/text_instruction.md",
            "markers": [f"target_train_failure:{target_id}"],
            "required_tokens": [
                "current_nrx",
                "percentage_difference",
                "prior_nrx",
            ],
        }
    ]
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_accepts_text_patch_with_target_failure_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "token_targeted_instruction_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve prior/current percentage output.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "percentage_or_comparison_metric",
                "root_cause": "percentage comparison misses prior_nrx",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow percentage-comparison instruction",
                "avoid": "current-period-only metric guidance",
                "expected_visible_effect": "generated SQL preserves prior_nrx and percentage_difference aliases",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["prior_nrx", "percentage_difference"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nFor percentage comparisons, preserve prior_nrx and "
                        "percentage_difference aliases.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "targeted period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True
    assert summary["risk_assessment"]["accepted"] is True
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["off_target_text_edits"] == []
    assert "percentage_difference aliases" in instruction_path.read_text(
        encoding="utf-8"
    )
    assert "percentage_difference aliases" in "\n".join(
        candidate.instructions.text_instructions[0].content
    )


def test_artifact_patch_v2_rejects_text_only_patch_after_prior_train_collapse(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "repeat_text_patch_after_collapse",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve prior/current percentage output.",
            "risk": "Allowlisted instruction edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "percentage_or_comparison_metric",
                "root_cause": "percentage comparison misses prior_nrx",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "append one narrow percentage-comparison instruction",
                "avoid": "current-period-only metric guidance",
                "expected_visible_effect": "generated SQL preserves prior_nrx and percentage_difference aliases",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["prior_nrx", "percentage_difference"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nFor percentage comparisons, preserve prior_nrx and "
                        "percentage_difference aliases.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "targeted period guidance for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "percentage_or_comparison_metric",
                    "suggested_operation_families": ["append_text_instruction"],
                    "missing_expected_tokens": ["prior_nrx", "percentage_difference"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="\n".join(
            [
                "## Prior Candidate Lessons",
                "### Rejected Patterns To Avoid Or Revise",
                "- `broad_text_patch`: rejected because train_health_failed: 33.3% -> 0.0%.",
            ]
        ),
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_text_only_after_prior_train_collapse" in summary[
        "risk_assessment"
    ]["reasons"]
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_repeated_single_file_shape_after_train_collapse(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "filters"
        / "brand_filter.yml"
    )
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "repeat_filter_file_after_collapse",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL uses brand_name literal filtering.",
            "risk": "New allowlisted filter snippet.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder mismatch for brand_name",
                "edit_surface_evidence": "snippet fragment filter edit surface",
                "repair_shape": "add one literal SQL fragment filter",
                "avoid": "bind placeholders",
                "expected_visible_effect": "generated SQL replaces brand_name placeholder with literal filter",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["brand_name"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/brand_filter.yml",
                    "content": "\n".join(
                        [
                            "display_name: Farxiga brand filter",
                            "content: LOWER(brand_name) = 'farxiga'",
                            "",
                        ]
                    ),
                    "reason": (
                        "literal brand filter for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="\n".join(
            [
                "## Prior Candidate Lessons",
                "### Rejected Patterns To Avoid Or Revise",
                "- `bad_filter_patch`: rejected because train_health_failed: 33.3% -> 0.0%.",
                "  - Operations that failed the gates: file_edit path=instructions/sql_snippets/filters/brand_filter.yml mode=replace_file reason=target_train_failure:q-period",
            ]
        ),
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_repeated_single_file_shape_after_prior_train_collapse" in summary[
        "risk_assessment"
    ]["reasons"]
    assert summary["risk_assessment"]["prior_train_collapse_shapes"] == [
        {
            "path": "instructions/sql_snippets/filters/brand_filter.yml",
            "edit_mode": "replace_file",
        }
    ]
    assert not snippet_path.exists()


def test_artifact_patch_v2_rejects_exact_repeated_prior_refused_shape(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "filters"
        / "brand_filter.yml"
    )
    target_id = "a" * 32
    reason = f"literal brand filter for target_train_failure:{target_id}"
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "repeat_refused_filter_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL replaces :brand_name with brand_name literal filtering.",
            "risk": "New allowlisted filter snippet.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder mismatch for brand_name",
                "edit_surface_evidence": "snippet fragment filter edit surface",
                "repair_shape": "add one literal SQL fragment filter",
                "avoid": "bind placeholders",
                "expected_visible_effect": "generated SQL replaces :brand_name with brand_name literal filter",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["brand_name"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/brand_filter.yml",
                    "content": "\n".join(
                        [
                            "display_name: Farxiga brand filter",
                            "content: LOWER(brand_name) = 'farxiga'",
                            "",
                        ]
                    ),
                    "reason": reason,
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="\n".join(
            [
                "## Prior Candidate Lessons",
                "### Skipped Or Risk-Rejected Patterns To Avoid",
                "- `prior_filter_patch`: skipped because serving_candidate_risk_rejected.",
                (
                    "  - Candidate shape refused before benchmarking: "
                    "file_edit path=instructions/sql_snippets/filters/brand_filter.yml "
                    f"mode=replace_file reason={reason}"
                ),
            ]
        ),
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_repeated_single_file_shape_after_prior_refusal" in summary[
        "risk_assessment"
    ]["reasons"]
    assert summary["risk_assessment"]["prior_refused_shapes"] == [
        {
            "path": "instructions/sql_snippets/filters/brand_filter.yml",
            "edit_mode": "replace_file",
            "reason": reason,
        }
    ]
    assert not snippet_path.exists()


def test_artifact_patch_v2_rejects_query_plan_guidance_in_metadata(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    table_path = artifacts.space_dir / "tables" / "sales.yml"
    original_table = table_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "metadata_query_plan_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL restores expected result columns.",
            "risk": "Allowlisted metadata edit.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "selected_metric_or_column_mismatch",
                "root_cause": "column schema mismatch for selected sales result columns",
                "edit_surface_evidence": "metadata column schema edit surface",
                "repair_shape": "metadata column result schema clarification",
                "avoid": "new executable examples",
                "expected_visible_effect": (
                    "restore_expected_result_columns using all_sales with market_name, "
                    "region, time_bucket_id, and wk_grp_desc"
                ),
                "compatible_paths": ["tables/*.yml"],
                "structural_tokens": [
                    "all_sales",
                    "market_name",
                    "region",
                    "time_bucket_id",
                    "wk_grp_desc",
                ],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "tables/sales.yml",
                    "append_content": (
                        "\n# For brand sales prompts, build an all_sales aggregation "
                        "that returns market_name, region, time_bucket_id, and "
                        "wk_grp_desc rather than a lookup.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_table.encode("utf-8")
                    ).hexdigest(),
                    "reason": (
                        "metadata result schema repair for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["set_column_config"],
                    "missing_expected_tokens": [
                        "all_sales",
                        "market_name",
                        "region",
                        "time_bucket_id",
                        "wk_grp_desc",
                    ],
                    "extra_generated_tokens": ["metric_type"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_metadata_query_plan_guidance" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["metadata_query_plan_edits"] == [
        {
            "path": "tables/sales.yml",
            "matched_query_plan_terms": ["all_sales", "returns "],
            "matched_result_schema_tokens": [
                "market_name",
                "region",
                "time_bucket_id",
                "wk_grp_desc",
            ],
        }
    ]
    assert table_path.read_text(encoding="utf-8") == original_table


def test_artifact_patch_v2_rejects_executable_literal_patch_without_expected_literal(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "filters"
        / "brand_filter.yml"
    )
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "generic_literal_filter_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL replaces :brand_name with brand_name literal filtering.",
            "risk": "New allowlisted filter snippet.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder mismatch for brand_name",
                "edit_surface_evidence": "snippet fragment filter edit surface",
                "repair_shape": "add one literal SQL fragment filter",
                "avoid": "bind placeholders",
                "expected_visible_effect": "generated SQL replaces :brand_name with brand_name literal filter",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["brand_name"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/brand_filter.yml",
                    "content": "\n".join(
                        [
                            "display_name: Generic brand filter",
                            "sql: |",
                            "  brand_name IS NOT NULL",
                            "",
                        ]
                    ),
                    "reason": (
                        "literal brand filter for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="",
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_executable_literal_missing_expected_token" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["executable_literal_expected_token_gaps"] == [
        {
            "path": "instructions/sql_snippets/filters/brand_filter.yml",
            "selected_marker": f"target_train_failure:{target_id}",
            "target_train_failure_id": target_id,
            "required_expected_literals": ["'farxiga'"],
            "plan_structural_tokens": ["brand_name"],
            "plan_has_expected_literal": False,
            "content_has_expected_literal": False,
        }
    ]
    assert not snippet_path.exists()


def test_artifact_patch_v2_accepts_executable_literal_patch_with_expected_literal(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "filters"
        / "brand_filter.yml"
    )
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "literal_filter_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL replaces :brand_name with 'farxiga' literal filtering.",
            "risk": "New allowlisted filter snippet.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder mismatch for 'farxiga'",
                "edit_surface_evidence": "snippet fragment filter edit surface",
                "repair_shape": "add one literal SQL fragment filter",
                "avoid": "bind placeholders",
                "expected_visible_effect": "literalize_expected_value: generated SQL replaces :brand_name with 'farxiga' literal filter",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["'farxiga'"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/brand_filter.yml",
                    "content": "\n".join(
                        [
                            "display_name: Farxiga brand filter",
                            "sql: |",
                            "  LOWER(brand_name) = 'farxiga'",
                            "",
                        ]
                    ),
                    "reason": (
                        "literal 'farxiga' filter for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="",
    )

    assert summary["changed"] is True, summary["risk_assessment"]
    assert summary["risk_assessment"]["reasons"] == []
    causal_plan = summary["risk_assessment"]["visible_effect_verification"][
        "causal_patch_plan"
    ]
    assert causal_plan["available_sql_delta_intents"] == [
        "literalize_expected_value",
        "remove_bind_placeholder",
    ]
    assert causal_plan["selected_sql_delta_intents"] == [
        "literalize_expected_value"
    ]
    assert (
        summary["risk_assessment"]["visible_effect_verification"][
            "executable_literal_expected_token_gaps"
        ]
        == []
    )
    assert "LOWER(brand_name) = 'farxiga'" in snippet_path.read_text(
        encoding="utf-8"
    )


def test_artifact_patch_v2_rejects_executable_patch_without_missing_expected_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "expressions"
        / "prior_nrx_alias.yml"
    )
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "wrong_metric_expression_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL stops using current_nrx as the metric alias.",
            "risk": "New allowlisted expression snippet.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "selected_metric_or_column_mismatch",
                "root_cause": "metric column alias mismatch for current_nrx",
                "edit_surface_evidence": "snippet expression edit surface",
                "repair_shape": "add one expression alias snippet",
                "avoid": "alternate descriptor lookup",
                "expected_visible_effect": "generated SQL stops using current_nrx as the alias",
                "compatible_paths": ["instructions/sql_snippets/expressions/*.yml"],
                "structural_tokens": ["current_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/expressions/prior_nrx_alias.yml",
                    "content": "\n".join(
                        [
                            "display_name: Current NRX alias",
                            "sql: |",
                            "  current_nrx",
                            "",
                        ]
                    ),
                    "reason": (
                        "metric alias snippet for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="",
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_executable_missing_expected_token" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["executable_missing_expected_token_gaps"] == [
        {
            "path": "instructions/sql_snippets/expressions/prior_nrx_alias.yml",
            "selected_marker": f"target_train_failure:{target_id}",
            "target_train_failure_id": target_id,
            "required_missing_expected_tokens": ["prior_nrx"],
            "plan_structural_tokens": ["current_nrx"],
            "plan_has_missing_expected_token": False,
            "content_has_missing_expected_token": False,
        }
    ]
    assert not snippet_path.exists()


def test_artifact_patch_v2_accepts_executable_patch_with_missing_expected_token(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    snippet_path = (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "expressions"
        / "prior_nrx_alias.yml"
    )
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "metric_expression_file",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Generated SQL returns prior_nrx as the metric alias.",
            "risk": "New allowlisted expression snippet.",
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "selected_metric_or_column_mismatch",
                "root_cause": "metric column alias mismatch for prior_nrx",
                "edit_surface_evidence": "snippet expression edit surface",
                "repair_shape": "add one expression alias snippet",
                "avoid": "alternate descriptor lookup",
                "expected_visible_effect": "generated SQL returns prior_nrx as the alias",
                "compatible_paths": ["instructions/sql_snippets/expressions/*.yml"],
                "structural_tokens": ["prior_nrx"],
            },
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/expressions/prior_nrx_alias.yml",
                    "content": "\n".join(
                        [
                            "display_name: Prior NRX alias",
                            "sql: |",
                            "  prior_nrx",
                            "",
                        ]
                    ),
                    "reason": (
                        "metric alias snippet for "
                        f"target_train_failure:{target_id}"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["prior_nrx"],
                    "extra_generated_tokens": ["current_nrx"],
                }
            ],
            "pass_guards": [],
        },
        failure_context="",
    )

    assert summary["changed"] is True, summary["risk_assessment"]
    assert summary["risk_assessment"]["reasons"] == []
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["executable_missing_expected_token_gaps"] == []
    assert "prior_nrx" in snippet_path.read_text(encoding="utf-8")


def test_artifact_patch_v2_rejects_multiple_target_plans(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    first_id = "a" * 32
    second_id = "b" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "two_unrelated_target_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": "Patch two visible failures in one response.",
            "expected_impact": "Improve two separate visible failures.",
            "risk": "Two compact file edits.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nUse precise current and prior period labels.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": f"target_train_failure:{first_id}",
                },
                {
                    "path": "instructions/examples/brand_literal_example.yml",
                    "content": "\n".join(
                        [
                            "id: brand_literal_example",
                            "question: what is sales of Farxiga?",
                            "sql: SELECT 1",
                            "",
                        ]
                    ),
                    "expected_sha256": None,
                    "reason": f"target_train_failure:{second_id}",
                },
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": first_id,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                },
                {
                    "question_id": second_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                },
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_inconsistent_target_plan" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["inconsistent_target_plan"] == {
        "reason": "multiple_target_train_failure_markers",
        "target_train_failure_ids": [first_id, second_id],
        "family_markers": [],
        "target_families": [
            "literal_or_parameter_mismatch",
            "time_filter_mismatch",
        ],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction
    assert not (
        artifacts.space_dir / "instructions" / "examples" / "brand_literal_example.yml"
    ).exists()


def test_artifact_patch_v2_rejects_guard_only_marker_without_target_plan(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    guard_id = "g" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "guard_only_instruction_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Preserve visible guard guard_preservation:{guard_id}.",
            "expected_impact": "Preserve a passing guard.",
            "risk": "No train target named.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "\nKeep current working guard behavior stable.\n",
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": f"guard_preservation:{guard_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                }
            ],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_inconsistent_target_plan" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["inconsistent_target_plan"] == {
        "reason": "missing_improvement_target_marker",
        "target_train_failure_ids": [],
        "family_markers": [],
        "target_families": [],
    }
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_trusted_query_routing_rewrite(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "broad_trusted_query_routing_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve routing by changing trusted-query behavior.",
            "risk": "Instruction-only routing rewrite.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": (
                        "\nRewrite trusted-query routing to prefer all generated "
                        "queries before existing examples.\n"
                    ),
                    "expected_sha256": hashlib.sha256(
                        original_instruction.encode("utf-8")
                    ).hexdigest(),
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert "artifact_patch_v2_trusted_query_routing_rewrite" in summary[
        "risk_assessment"
    ]["reasons"]
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["overbroad_instruction_edits"] == [
        {
            "path": "instructions/text_instruction.md",
            "reason": "trusted_query_routing_rewrite",
        }
    ]
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_artifact_patch_v2_rejects_full_query_sql_snippet_file(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    target_id = "a" * 32
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "default_full_query_snippet_patch",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve literal filter reuse with a snippet.",
            "risk": "One SQL snippet file only.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/full_query.yml",
                    "content": "\n".join(
                        [
                            "display_name: Full query filter",
                            "sql: |",
                            "  SELECT * FROM sales WHERE brand_name = 'Farxiga'",
                            "",
                        ]
                    ),
                    "expected_sha256": None,
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    assert (
        "executable_artifact_patch_sql_snippet_full_query"
        in summary["risk_assessment"]["reasons"]
    )
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["invalid_executable_edits"] == [
        {
            "path": "instructions/sql_snippets/filters/full_query.yml",
            "reasons": ["executable_artifact_patch_sql_snippet_full_query"],
        }
    ]
    assert not (
        artifacts.space_dir
        / "instructions"
        / "sql_snippets"
        / "filters"
        / "full_query.yml"
    ).exists()


def test_artifact_patch_v2_rejects_structural_guard_sql_clone(
    tmp_path: Path,
):
    space_id = "1" * 32
    target_id = "a" * 32
    guard_id = "b" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    guard_sql = "\n".join(
        [
            "WITH brand_sales AS (",
            "  SELECT brand_name, market_name, SUM(total_prescriptions) AS trx",
            "  FROM sales",
            "  WHERE brand_name = 'Lokelma' AND hierarchy_level = 'NATN'",
            "  GROUP BY brand_name, market_name",
            ")",
            "SELECT brand_name, market_name, trx FROM brand_sales",
        ]
    )
    target_sql = guard_sql.replace("'Lokelma'", "'Farxiga'")

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "default_farxiga_sales_structural_clone",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Improve the selected metric output.",
            "risk": "Single visible executable example.",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/farxiga_sales_clone.yml",
                    "content": "\n".join(
                        [
                            "id: farxiga_sales_clone",
                            "question: what is sales of Farxiga?",
                            "sql: |",
                            *[f"  {line}" for line in target_sql.splitlines()],
                            "",
                        ]
                    ),
                    "expected_sha256": None,
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                }
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "question": "what is sales of Lokelma?",
                    "expected_sql": guard_sql,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                }
            ],
        },
    )

    assert summary["changed"] is False
    assert (
        f"executable_artifact_patch_example_guard_sql_clone:{guard_id}"
        in summary["risk_assessment"]["reasons"]
    )
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["invalid_executable_edits"] == [
        {
            "path": "instructions/examples/farxiga_sales_clone.yml",
            "reasons": [
                f"executable_artifact_patch_example_guard_sql_clone:{guard_id}"
            ],
        }
    ]
    assert not (
        artifacts.space_dir
        / "instructions"
        / "examples"
        / "farxiga_sales_clone.yml"
    ).exists()


def test_composite_artifact_patch_v2_rejects_single_instruction_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _CompositePatchCandidateClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            assert kwargs["candidate_mode"] == "composite_artifact_patch_v2"
            assert kwargs["decomposed_space_payload"]["files"]
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "single_instruction_patch",
                    "rationale": "Too narrow for composite mode.",
                    "expected_impact": "Would only add a text instruction.",
                    "risk": "No coordinated artifact change.",
                    "owner_followups": [],
                    "file_edits": [
                        {
                            "path": "instructions/text_instruction.md",
                            "content": original_instruction + "\nUse C13W literal labels.\n",
                            "expected_sha256": hashlib.sha256(
                                original_instruction.encode("utf-8")
                            ).hexdigest(),
                            "reason": "visible_family:literal_time_window_guidance",
                        }
                    ],
                    "operations": [],
                },
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _CompositePatchCandidateClient)

    _candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="composite_artifact_patch_v2",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    assert summary["changed"] is False
    assert summary["candidate_mode"] == "composite_artifact_patch_v2"
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert (
        "too_few_composite_artifact_patch_file_edits:1"
        in summary["risk_assessment"]["reasons"]
    )
    assert (
        "composite_artifact_patch_rejects_instruction_only_patch"
        in summary["risk_assessment"]["reasons"]
    )
    assert instruction_path.read_text(encoding="utf-8") == original_instruction


def test_executable_artifact_patch_v2_rejects_instruction_metadata_only_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    table_path = artifacts.space_dir / "tables" / "sales.yml"
    original_instruction = instruction_path.read_text(encoding="utf-8")
    original_table = table_path.read_text(encoding="utf-8")
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _InstructionMetadataPatchClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            assert kwargs["candidate_mode"] == "executable_artifact_patch_v2"
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "instruction_metadata_only_patch",
                    "rationale": "This repeats the negative composite shape.",
                    "expected_impact": "Would add prose and table comments only.",
                    "risk": "No executable asset is changed.",
                    "owner_followups": [],
                    "file_edits": [
                        {
                            "path": "instructions/text_instruction.md",
                            "append_content": "\nUse C13W literal labels.\n",
                            "expected_sha256": hashlib.sha256(
                                original_instruction.encode("utf-8")
                            ).hexdigest(),
                            "reason": "visible_family:literal_time_window_guidance",
                        },
                        {
                            "path": "tables/sales.yml",
                            "append_content": "\n# Visible-family metadata support only.\n",
                            "expected_sha256": hashlib.sha256(
                                original_table.encode("utf-8")
                            ).hexdigest(),
                            "reason": "visible_family:metadata_support",
                        },
                    ],
                    "operations": [],
                },
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _InstructionMetadataPatchClient)

    _candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="executable_artifact_patch_v2",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    assert summary["changed"] is False
    assert summary["candidate_mode"] == "executable_artifact_patch_v2"
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert (
        "executable_artifact_patch_requires_example_or_snippet_file"
        in summary["risk_assessment"]["reasons"]
    )
    assert instruction_path.read_text(encoding="utf-8") == original_instruction
    assert table_path.read_text(encoding="utf-8") == original_table


def test_executable_artifact_patch_v2_accepts_visible_example_asset(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "visible_literal_example_asset",
            "candidate_mode": "executable_artifact_patch_v2",
            "rationale": "Use a concrete visible executable example instead of prose.",
            "expected_impact": "Should exercise the executable artifact patch path.",
            "risk": "Low because this creates one visible example asset.",
            "owner_followups": [],
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/example_visible_literal.yml",
                    "content": "\n".join(
                        [
                            "id: visible_literal_example",
                            "question: Train question",
                            "sql: SELECT 1",
                            "usage_guidance: Preserve concrete literal SQL for this visible family.",
                            "",
                        ]
                    ),
                    "reason": "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                }
            ],
        },
        current_config=config,
    )

    assert summary["changed"] is True
    assert summary["risk_assessment"]["accepted"] is True
    assert summary["candidate_mode"] == "executable_artifact_patch_v2"
    assert summary["risk_assessment"]["visible_effect_verification"]["accepted"] is True
    assert "target_train_failure:" in summary["risk_assessment"]["visible_effect_verification"][
        "effect_markers"
    ]
    assert (
        artifacts.space_dir / "instructions" / "examples" / "example_visible_literal.yml"
    ).exists()
    examples = candidate.instructions.example_question_sqls if candidate.instructions else []
    assert examples is not None
    assert examples[-1].question == ["Train question"]
    assert examples[-1].sql == ["SELECT 1"]


def test_broad_artifact_patch_validates_payload_content_for_new_example(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    target_id = "a" * 32

    candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "data_probe_literal_example",
            "candidate_mode": "data_probe_artifact_patch_v1",
            "rationale": (
                "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa failed with "
                "literal_or_parameter_mismatch and needs one executable example."
            ),
            "expected_impact": "Should teach a literalized SQL pattern for the visible failure.",
            "risk": "Low because this creates one narrow executable example asset.",
            "owner_followups": [],
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/example_data_probe_literal.yml",
                    "content": "\n".join(
                        [
                            "question: Train question",
                            "usage_guidance: Use concrete literals from the visible prompt.",
                            "sql: SELECT 1",
                            "",
                        ]
                    ),
                    "reason": (
                        "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
                        "shows literal_or_parameter_mismatch."
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Train question",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True, summary["risk_assessment"]
    assert summary["candidate_mode"] == "data_probe_artifact_patch_v1"
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is True
    assert "executable_artifact_patch_example_missing_question_or_sql" not in verification[
        "reasons"
    ]
    examples = candidate.instructions.example_question_sqls if candidate.instructions else []
    assert examples is not None
    assert examples[-1].question == ["Train question"]
    assert examples[-1].sql == ["SELECT 1"]


def test_executable_artifact_patch_v2_rejects_literal_swap_guard_clone(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "example_farxiga_default_sales_schema_guarded",
            "candidate_mode": "executable_artifact_patch_v2",
            "rationale": "Target a selected_metric_or_column_mismatch visible failure.",
            "expected_impact": "Should improve target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
            "risk": (
                "Preserves Lokelma sales pass guards by reusing the same schema visible "
                "in guard SQL, changing only the requested brand literal to Farxiga."
            ),
            "owner_followups": [],
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/farxiga_default_sales_c5w_natn.yml",
                    "content": "\n".join(
                        [
                            "id: farxiga_default_sales_c5w_natn",
                            "question: what is sales of Farxiga?",
                            "sql: SELECT 'Farxiga' AS brand_name, 1 AS metric_value",
                            "",
                        ]
                    ),
                    "reason": (
                        "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
                        "selected_metric_or_column_mismatch; this executable example "
                        "mirrors the passing Lokelma sales guard pattern with only the "
                        "brand literal changed to Farxiga."
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                }
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert (
        "executable_artifact_patch_example_literal_swap_guard_clone"
        in summary["risk_assessment"]["reasons"]
    )
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is False
    assert "target_train_failure:" in verification["effect_markers"]
    assert not (
        artifacts.space_dir
        / "instructions"
        / "examples"
        / "farxiga_default_sales_c5w_natn.yml"
    ).exists()


def test_executable_artifact_patch_v2_rejects_structural_guard_sql_clone(
    tmp_path: Path,
):
    space_id = "1" * 32
    guard_id = "b" * 32
    target_id = "a" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    guard_sql = "\n".join(
        [
            "WITH brand_sales AS (",
            "  SELECT brand_name, market_name, SUM(total_prescriptions) AS trx",
            "  FROM sales",
            "  WHERE brand_name = 'Lokelma' AND hierarchy_level = 'NATN'",
            "  GROUP BY brand_name, market_name",
            ")",
            "SELECT brand_name, market_name, trx FROM brand_sales",
        ]
    )
    target_sql = guard_sql.replace("'Lokelma'", "'Farxiga'")

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "example_farxiga_sales_structural_clone",
            "candidate_mode": "executable_artifact_patch_v2",
            "rationale": "Target visible selected_metric_or_column_mismatch.",
            "expected_impact": f"Improve target_train_failure:{target_id}.",
            "risk": "Single visible executable example; no global edits.",
            "owner_followups": [],
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/farxiga_sales_structural_clone.yml",
                    "content": "\n".join(
                        [
                            f"id: {target_id}",
                            "question: what is sales of Farxiga?",
                            "sql: |",
                            *[f"  {line}" for line in target_sql.splitlines()],
                            "",
                        ]
                    ),
                    "reason": (
                        f"target_train_failure:{target_id} "
                        "selected_metric_or_column_mismatch"
                    ),
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "selected_metric_or_column_mismatch",
                }
            ],
            "pass_guards": [
                {
                    "question_id": guard_id,
                    "question": "what is sales of Lokelma?",
                    "expected_sql": guard_sql,
                    "source": "baseline_validation",
                    "latest_status": "passed",
                    "guard_reason": "visible_pass_guard",
                }
            ],
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert (
        f"executable_artifact_patch_example_guard_sql_clone:{guard_id}"
        in summary["risk_assessment"]["reasons"]
    )
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is False
    assert verification["visible_pass_guard_ids_checked"] == [guard_id]
    assert not (
        artifacts.space_dir
        / "instructions"
        / "examples"
        / "farxiga_sales_structural_clone.yml"
    ).exists()


def test_executable_artifact_patch_v2_rejects_missing_visible_effect_mapping(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "unmapped_visible_example_asset",
            "candidate_mode": "executable_artifact_patch_v2",
            "rationale": "Add a generic executable example.",
            "expected_impact": "Improve behavior.",
            "risk": "Low.",
            "owner_followups": [],
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/examples/example_generic.yml",
                    "content": "\n".join(
                        [
                            "id: generic_example",
                            "question: Generic train question",
                            "sql: SELECT 1",
                            "",
                        ]
                    ),
                    "reason": "generic executable example",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                }
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["reason"] == "artifact_patch_risk_rejected"
    assert (
        "executable_artifact_patch_missing_visible_effect_mapping"
        in summary["risk_assessment"]["reasons"]
    )
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["accepted"] is False
    assert verification["visible_failure_families"] == ["literal_or_parameter_mismatch"]
    assert not (
        artifacts.space_dir / "instructions" / "examples" / "example_generic.yml"
    ).exists()


def test_executable_artifact_patch_v2_rejects_bad_snippet_file_shape(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "bad_type_snippet",
            "candidate_mode": "executable_artifact_patch_v2",
            "rationale": "Target visible literal placeholder failures.",
            "expected_impact": "Should address literal_or_parameter_mismatch failures.",
            "risk": "One SQL snippet file only.",
            "owner_followups": [],
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/bad_type.yml",
                    "content": "\n".join(
                        [
                            "display_name:",
                            "  - Bad type",
                            "sql:",
                            "  fragment: brand = 'Lokelma'",
                            "",
                        ]
                    ),
                    "reason": "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                }
            ]
        },
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]
    assert "executable_artifact_patch_sql_snippet_sql_must_be_string_or_list" in reasons
    assert "executable_artifact_patch_sql_snippet_display_name_must_be_string" in reasons
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["changed_executable_paths"] == [
        "instructions/sql_snippets/filters/bad_type.yml"
    ]
    assert verification["reasons"] == [
        "executable_artifact_patch_sql_snippet_sql_must_be_string_or_list",
        "executable_artifact_patch_sql_snippet_display_name_must_be_string",
    ]
    assert not (
        artifacts.space_dir / "instructions" / "sql_snippets" / "filters" / "bad_type.yml"
    ).exists()
    assert verification["accepted"] is False
    assert "target_train_failure:" in verification["effect_markers"]


def test_artifact_patch_v2_rejects_sql_snippet_without_sql_field(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "snippet_missing_sql",
            "candidate_mode": "artifact_patch_v2",
            "rationale": "Target visible literal placeholder failures.",
            "expected_impact": "Should address literalize_expected_value for target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.",
            "risk": "One SQL snippet file only.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder",
                "edit_surface_evidence": "snippet filter edit surface",
                "repair_shape": "add one filter snippet",
                "avoid": "bind placeholders",
                "expected_visible_effect": "literalize_expected_value for 'farxiga'",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["'farxiga'"],
            },
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/missing_sql.yml",
                    "content": "\n".join(
                        [
                            "display_name: Missing SQL",
                            "description: Literalize Farxiga filter",
                            "",
                        ]
                    ),
                    "reason": "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": "a" * 32,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
        },
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]
    assert "executable_artifact_patch_sql_snippet_missing_sql" in reasons
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["invalid_executable_edits"] == [
        {
            "path": "instructions/sql_snippets/filters/missing_sql.yml",
            "reasons": ["executable_artifact_patch_sql_snippet_missing_sql"],
        }
    ]
    assert not (
        artifacts.space_dir / "instructions" / "sql_snippets" / "filters" / "missing_sql.yml"
    ).exists()


def test_artifact_patch_v2_rejects_invalid_example_yaml_shape(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    target_id = "a" * 32
    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "invalid_yaml_example",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Should literalize_expected_value for 'farxiga'.",
            "risk": "One executable example only.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder",
                "edit_surface_evidence": "example executable edit surface",
                "repair_shape": "add one literal example SQL",
                "avoid": "bind placeholders",
                "expected_visible_effect": "literalize_expected_value for 'farxiga'",
                "compatible_paths": ["instructions/examples/*.yml"],
                "structural_tokens": ["'farxiga'"],
            },
            "file_edits": [
                {
                    "path": "instructions/examples/invalid_yaml.yml",
                    "content": "\n".join(
                        [
                            "question: [Show Farxiga sales",
                            "sql: SELECT 'farxiga'",
                            "",
                        ]
                    ),
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Show Farxiga sales",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]
    assert "executable_artifact_patch_example_invalid_yaml" in reasons
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["invalid_executable_edits"] == [
        {
            "path": "instructions/examples/invalid_yaml.yml",
            "reasons": ["executable_artifact_patch_example_invalid_yaml"],
        }
    ]
    assert not (
        artifacts.space_dir / "instructions" / "examples" / "invalid_yaml.yml"
    ).exists()


def test_artifact_patch_v2_rejects_new_executable_after_repeated_train_collapse(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    target_id = "a" * 32
    failure_context = "\n".join(
        [
            "# Failure Context",
            "## Prior Candidate Lessons",
            "### Rejected Patterns To Avoid Or Revise",
            "- `collapsed_example`: rejected because train_health_failed: 11.1% -> 0.0%.",
            "  - Operations that failed the gates: file_edit path=instructions/examples/collapsed_example.yml mode=replace_file reason=target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "- `collapsed_snippet`: rejected because train_health_failed: 11.1% -> 0.0%.",
            "  - Operations that failed the gates: file_edit path=instructions/sql_snippets/filters/collapsed_snippet.yml mode=replace_file reason=target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ]
    )

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "another_new_literal_example",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Should literalize_expected_value for 'farxiga'.",
            "risk": "One executable example only.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder",
                "edit_surface_evidence": "example executable edit surface",
                "repair_shape": "add one literal example SQL",
                "avoid": "bind placeholders",
                "expected_visible_effect": "literalize_expected_value for 'farxiga'",
                "compatible_paths": ["instructions/examples/*.yml"],
                "structural_tokens": ["'farxiga'"],
            },
            "file_edits": [
                {
                    "path": "instructions/examples/another_literal.yml",
                    "content": "\n".join(
                        [
                            "question: Show Farxiga sales",
                            "sql: |",
                            "  SELECT 'farxiga' AS brand_name",
                            "",
                        ]
                    ),
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Show Farxiga sales",
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_example_sql"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
        failure_context=failure_context,
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]
    assert (
        "artifact_patch_v2_new_executable_asset_after_repeated_train_collapse"
        in reasons
    )
    collapse_evidence = summary["risk_assessment"][
        "prior_train_collapse_new_executable_assets"
    ]
    assert collapse_evidence == [
        {
            "path": "instructions/examples/another_literal.yml",
            "prior_new_executable_train_collapse_count": 2,
            "prior_new_executable_train_collapse_paths": [
                "instructions/examples/collapsed_example.yml",
                "instructions/sql_snippets/filters/collapsed_snippet.yml",
            ],
        }
    ]
    assert not (
        artifacts.space_dir / "instructions" / "examples" / "another_literal.yml"
    ).exists()


def test_artifact_patch_v2_rejects_executable_yaml_content_wrapper(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    target_id = "a" * 32
    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "wrapped_snippet_yaml",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Should literalize_expected_value for 'farxiga'.",
            "risk": "One SQL snippet file only.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder",
                "edit_surface_evidence": "snippet filter edit surface",
                "repair_shape": "add one filter snippet",
                "avoid": "bind placeholders",
                "expected_visible_effect": "literalize_expected_value for 'farxiga'",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["'farxiga'"],
            },
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/wrapped.yml",
                    "content": "\n".join(
                        [
                            "content: |",
                            "  display_name: Wrapped Farxiga filter",
                            "  sql: |",
                            "    LOWER(brand_name) = 'farxiga'",
                            "",
                        ]
                    ),
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]
    assert "executable_artifact_patch_sql_snippet_unexpected_content_wrapper" in reasons
    verification = summary["risk_assessment"]["visible_effect_verification"]
    assert verification["invalid_executable_edits"] == [
        {
            "path": "instructions/sql_snippets/filters/wrapped.yml",
            "reasons": [
                "executable_artifact_patch_sql_snippet_unexpected_content_wrapper",
                "executable_artifact_patch_sql_snippet_missing_sql",
                "executable_artifact_patch_sql_snippet_display_name_not_repairable",
            ],
        }
    ]
    assert not (
        artifacts.space_dir / "instructions" / "sql_snippets" / "filters" / "wrapped.yml"
    ).exists()


def test_artifact_patch_v2_rejects_sql_snippet_bad_field_types(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    target_id = "a" * 32
    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "bad_type_snippet",
            "candidate_mode": "artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Should literalize_expected_value for 'farxiga'.",
            "risk": "One SQL snippet file only.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "literal_or_parameter_mismatch",
                "root_cause": "literal parameter placeholder",
                "edit_surface_evidence": "snippet filter edit surface",
                "repair_shape": "add one filter snippet",
                "avoid": "bind placeholders",
                "expected_visible_effect": "literalize_expected_value for 'farxiga'",
                "compatible_paths": ["instructions/sql_snippets/filters/*.yml"],
                "structural_tokens": ["'farxiga'"],
            },
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/filters/bad_type.yml",
                    "content": "\n".join(
                        [
                            "display_name:",
                            "  nested: Bad",
                            "description: Literalize Farxiga filter",
                            "sql:",
                            "  nested: SELECT 'farxiga'",
                            "",
                        ]
                    ),
                    "reason": f"target_train_failure:{target_id}",
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "likely_root_cause_family": "literal_or_parameter_mismatch",
                    "suggested_operation_families": ["add_sql_snippet"],
                    "missing_expected_tokens": ["'farxiga'"],
                    "extra_generated_tokens": [":brand_name"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]


def test_skeleton_artifact_patch_v2_accepts_bound_existing_file_edit(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    instruction_text = instruction_path.read_text(encoding="utf-8")
    assert "Use exact table names" in instruction_text
    expected_sha = hashlib.sha256(instruction_path.read_bytes()).hexdigest()
    target_id = "a" * 32

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "skeleton_weekly_instruction",
            "candidate_mode": "skeleton_artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Should align_time_window_filter for weekly buckets.",
            "risk": "One skeleton-bound instruction edit.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch needs weekly bucket literal",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "add a bounded weekly instruction",
                "avoid": "global default window changes",
                "expected_visible_effect": "align_time_window_filter weekly",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["weekly"],
            },
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "relative_path": None,
                    "content": None,
                    "append_content": None,
                    "find": "Use exact table names",
                    "replace": (
                        "Use exact table names\n"
                        "When visible questions ask for weekly buckets, preserve weekly bucket labels."
                    ),
                    "expected_sha256": expected_sha,
                    "reason": f"target_train_failure:{target_id}",
                    "description": None,
                    "comment": None,
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Show weekly sales",
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                    "guard_regression_risk": "low",
                    "missing_expected_tokens": ["weekly"],
                    "extra_generated_tokens": ["monthly"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is True
    risk = summary["risk_assessment"]
    assert risk["accepted"] is True
    skeleton_binding = risk["visible_effect_verification"]["skeleton_binding"]
    assert skeleton_binding["accepted"] is True
    assert skeleton_binding["selected_skeleton_id"] == "skel_01_aaaaaaaa"
    assert "weekly bucket labels" in instruction_path.read_text(encoding="utf-8")


def test_skeleton_artifact_patch_v2_rejects_unbound_find_anchor(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    flow = OptimizationWorkspace(
        genie_service=_DummyGenieService(config),
        benchmark_service=_DummyBenchmarkService(),
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=_DummyGenieService(config),
            benchmark_service=_DummyBenchmarkService(),
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    instruction_path = artifacts.space_dir / "instructions" / "text_instruction.md"
    expected_sha = hashlib.sha256(instruction_path.read_bytes()).hexdigest()
    target_id = "a" * 32

    _candidate, summary = flow._apply_artifact_patch_payload(
        payload={
            "candidate_name": "skeleton_drifted_instruction",
            "candidate_mode": "skeleton_artifact_patch_v2",
            "rationale": f"Target visible failure target_train_failure:{target_id}.",
            "expected_impact": "Should align_time_window_filter for weekly buckets.",
            "risk": "One skeleton-bound instruction edit.",
            "owner_followups": [],
            "operations": [],
            "causal_patch_plan": {
                "selected_marker": f"target_train_failure:{target_id}",
                "family": "time_filter_mismatch",
                "root_cause": "time window mismatch needs weekly bucket literal",
                "edit_surface_evidence": "instruction guidance edit surface",
                "repair_shape": "add a bounded weekly instruction",
                "avoid": "global default window changes",
                "expected_visible_effect": "align_time_window_filter weekly",
                "compatible_paths": ["instructions/text_instruction.md"],
                "structural_tokens": ["weekly"],
            },
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "relative_path": None,
                    "content": None,
                    "append_content": None,
                    "find": "exact table names",
                    "replace": "exact table names and weekly bucket labels",
                    "expected_sha256": expected_sha,
                    "reason": f"target_train_failure:{target_id}",
                    "description": None,
                    "comment": None,
                }
            ],
        },
        current_config=config,
        structured_failure_context={
            "failures": [
                {
                    "question_id": target_id,
                    "question": "Show weekly sales",
                    "likely_root_cause_family": "time_filter_mismatch",
                    "suggested_operation_families": ["append_text_instruction"],
                    "guard_regression_risk": "low",
                    "missing_expected_tokens": ["weekly"],
                    "extra_generated_tokens": ["monthly"],
                }
            ],
            "pass_guards": [],
        },
    )

    assert summary["changed"] is False
    reasons = summary["risk_assessment"]["reasons"]
    assert "skeleton_artifact_patch_binding_failed" in reasons
    skeleton_binding = summary["risk_assessment"]["visible_effect_verification"][
        "skeleton_binding"
    ]
    assert "skeleton_artifact_patch_no_matching_skeleton" in skeleton_binding["reasons"]
    assert instruction_path.read_text(encoding="utf-8").strip() == "Use exact table names"


def test_executable_artifact_patch_v2_repairs_missing_snippet_display_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=genie_service,
            benchmark_service=benchmark_service,
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _SnippetPatchClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            assert kwargs["candidate_mode"] == "executable_artifact_patch_v2"
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "missing_display_name_snippet_patch",
                    "rationale": "Add a concrete snippet asset.",
                    "expected_impact": "Should repair the snippet display name before assembly.",
                    "risk": "One new executable snippet only.",
                    "owner_followups": [],
                    "file_edits": [
                        {
                            "path": "instructions/sql_snippets/expressions/literal_time_bucket_selector.yml",
                            "content": "\n".join(
                                [
                                    "description: Use literal SQL constants, not bind parameters, for week/month time buckets.",
                                    "sql: |",
                                    "  CASE",
                                    "    WHEN 'wk_grp' = 'wk_grp' THEN tb_wk_id",
                                    "  END",
                                    "",
                                ]
                            ),
                            "reason": "target_train_failure:literal_or_parameter_mismatch",
                        }
                    ],
                    "operations": [],
                },
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _SnippetPatchClient)

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="executable_artifact_patch_v2",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    assert summary["changed"] is True
    assert summary["candidate_mode"] == "executable_artifact_patch_v2"
    assert summary["artifact_patch_repair"]["filled_sql_snippet_display_name_paths"] == [
        "instructions/sql_snippets/expressions/literal_time_bucket_selector.yml"
    ]
    snippets = candidate.instructions.sql_snippets
    assert snippets is not None
    assert snippets.expressions is not None
    assert snippets.expressions[0].display_name == (
        "Use literal SQL constants, not bind parameters, for week/month time buckets."
    )
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert '"phase": "serving_candidate_artifact_patch_repaired"' in iteration_log


def test_serving_candidate_receives_visible_proxy_guard_context_without_hidden_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    train_guard_id = "a" * 32
    validation_guard_id = "b" * 32
    hidden_holdout_id = "c" * 32
    target_id = "d" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 4,
            "questions": [
                {
                    "id": train_guard_id,
                    "question": "Train pass guard for current versus prior period",
                    "expected_sql": "SELECT 1",
                },
                {
                    "id": validation_guard_id,
                    "question": "Validation pass guard by region",
                    "expected_sql": "SELECT 2",
                },
                {
                    "id": target_id,
                    "question": "Failed train target",
                    "expected_sql": "SELECT 3",
                },
                {
                    "id": hidden_holdout_id,
                    "question": "Secret holdout question must not leak",
                    "expected_sql": "SELECT 4",
                },
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "questions": [
                {
                    "id": train_guard_id,
                    "question": "Train pass guard for current versus prior period",
                    "expected_sql": "SELECT 1",
                },
                {
                    "id": target_id,
                    "question": "Failed train target",
                    "expected_sql": "SELECT 3",
                },
            ],
        },
    )
    train_run = BenchmarkRun(space_id=space_id)
    train_run.results = [
        BenchmarkResult(
            question_id=train_guard_id,
            question="Train pass guard for current versus prior period",
            status=BenchmarkStatus.PASSED,
        ),
        BenchmarkResult(
            question_id=target_id,
            question="Failed train target",
            status=BenchmarkStatus.FAILED,
        ),
    ]
    artifacts.write_json_path(artifacts.latest_train_path, train_run.to_dict())
    artifacts.write_json_path(
        artifacts.validation_set_path,
        {"ids": [validation_guard_id]},
    )
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "failed_ids": [],
            "error_ids": [],
            "skipped_ids": [],
            "pass_rate": 100.0,
        },
    )
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=genie_service,
            benchmark_service=benchmark_service,
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _ProxyAwareCandidateClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            guard_payload = kwargs["visible_proxy_guard_payload"]
            serialized = json.dumps(guard_payload)
            visible_payload_json = json.dumps(kwargs["visible_benchmark_payload"])
            assert train_guard_id in guard_payload["required_guard_ids"]
            assert validation_guard_id in guard_payload["required_guard_ids"]
            assert hidden_holdout_id not in serialized
            assert hidden_holdout_id not in visible_payload_json
            assert "Secret holdout" not in serialized
            assert "Secret holdout" not in visible_payload_json
            assert guard_payload["policy"]["hidden_holdout_content"] == "omitted"
            assert kwargs["visible_benchmark_payload"]["policy"]["hidden_holdout_content"] == "omitted"
            assert guard_payload["source"]["visible_sources_only"] is True
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "no_change",
                    "rationale": "No safe change.",
                    "expected_impact": "None.",
                    "risk": "No mutation.",
                    "owner_followups": [],
                    "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
                },
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _ProxyAwareCandidateClient)

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="example_sql",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    iteration_records = [
        json.loads(line)
        for line in artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start_record = next(record for record in iteration_records if record["phase"] == "serving_candidate_proposal_start")
    assert candidate.to_serialized_space() == config.to_serialized_space()
    assert summary["changed"] is False
    assert summary["reason"] == "no_valid_serving_operations"
    assert start_record["visible_proxy_guard_count"] == 2
    assert train_guard_id in start_record["visible_proxy_guard_ids"]
    assert validation_guard_id in start_record["visible_proxy_guard_ids"]


def test_typed_template_serving_candidate_materializes_visible_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    guard_id = "a" * 32
    target_id = "b" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {
                    "id": guard_id,
                    "question": "Visible passing guard",
                    "expected_sql": "SELECT 'guard' AS answer",
                },
                {
                    "id": target_id,
                    "question": "Visible percentage difference failure",
                    "expected_sql": (
                        "SELECT current_nrx, prior_nrx, percentage_difference FROM sales"
                    ),
                },
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "questions": [
                {
                    "id": guard_id,
                    "question": "Visible passing guard",
                    "expected_sql": "SELECT 'guard' AS answer",
                },
                {
                    "id": target_id,
                    "question": "Visible percentage difference failure",
                    "expected_sql": (
                        "SELECT current_nrx, prior_nrx, percentage_difference FROM sales"
                    ),
                },
            ],
        },
    )
    train_run = BenchmarkRun(space_id=space_id)
    train_run.results = [
        BenchmarkResult(
            question_id=guard_id,
            question="Visible passing guard",
            status=BenchmarkStatus.PASSED,
            expected_sql="SELECT 'guard' AS answer",
            generated_sql="SELECT 'guard' AS answer",
            comparison_method="result_match",
            comparison_score=1.0,
        ),
        BenchmarkResult(
            question_id=target_id,
            question="Visible percentage difference failure",
            status=BenchmarkStatus.FAILED,
            expected_sql="SELECT current_nrx, prior_nrx, percentage_difference FROM sales",
            generated_sql="SELECT current_nrx FROM sales",
            comparison_method="result_column_mismatch",
            comparison_score=0.25,
        ),
    ]
    artifacts.write_json_path(artifacts.baseline_train_path, train_run.to_dict())
    artifacts.write_json_path(artifacts.latest_train_path, train_run.to_dict())
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(
        artifacts.baseline_validation_summary_path,
        {"question_count": 0, "pass_rate": 0.0, "failed_ids": [], "error_ids": []},
    )
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {"question_count": 0, "pass_rate": 0.0, "failed_ids": [], "error_ids": []},
    )
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=genie_service,
            benchmark_service=benchmark_service,
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _TypedTemplateCandidateClient:
        def __init__(self, *, wc, config):
            self.wc = wc
            self.config = config

        def propose(self, **kwargs):
            assert kwargs["candidate_mode"] == "typed_template"
            return ServingCandidateProposal(
                request={"messages": [{"role": "user", "content": "prompt"}]},
                response={"candidate": "ok"},
                content="{}",
                payload={
                    "candidate_name": "typed_template_period_fix",
                    "rationale": "Target visible period comparison.",
                    "expected_impact": "Adds typed executable example.",
                    "risk": "Preserves visible pass guard.",
                    "owner_followups": [],
                    "operations": [
                        {
                            "op": "add_example_sql",
                            "id": target_id,
                            "question": None,
                            "sql": None,
                            "usage_guidance": None,
                            "reason": f"target_train_failure:{target_id}",
                        }
                    ],
                },
            )

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _TypedTemplateCandidateClient)

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
            candidate_mode="typed_template",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    examples = candidate.instructions.example_question_sqls or []
    assert summary["changed"] is True
    assert summary["candidate_mode"] == "typed_template"
    assert summary["risk_assessment"]["accepted"] is True
    assert summary["risk_assessment"]["canonicalization"]["typed_template_materialized_guard_ids"] == [
        guard_id
    ]
    assert summary["risk_assessment"]["canonicalization"]["typed_template_materialized_target_ids"] == [
        target_id
    ]
    assert [example.id for example in examples] == [guard_id, target_id]
    assert examples[0].sql == ["SELECT 'guard' AS answer"]
    assert examples[1].sql == ["SELECT current_nrx, prior_nrx, percentage_difference FROM sales"]
    target_guidance = "\n".join(examples[1].usage_guidance or [])
    assert "Typed template:" in target_guidance
    assert "visible SQL" in target_guidance
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert '"phase": "serving_candidate_typed_template_materialized"' in iteration_log


def test_protocol_preflight_uses_full_summary_coverage_when_visible_payload_is_smaller(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 12,
            "questions": [{"id": str(index), "question": "q"} for index in range(12)],
        },
    )
    artifacts.write_json_path(artifacts.split_meta_path, {"split_strategy": "canonical"})
    artifacts.update_workspace_meta(
        {
            "fresh_baseline": True,
            "serving_endpoint": "databricks-gpt-5-5-pro",
            "serving_model": "databricks-gpt-5-5",
            "serving_reasoning_effort": "xhigh",
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=genie_service,
            benchmark_service=benchmark_service,
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    result = flow._write_protocol_preflight(
        mode="autonomous",
        strategy="serving",
        summary={
            "evaluation_context": {
                "preferred_comparison_mode": "result_based",
                "warehouse_id": "warehouse-123",
            },
            "overall": {
                "question_count": 15,
                "comparison_metrics": {
                    "scored_question_count": 15,
                    "result_based_question_count": 15,
                },
            },
        },
    )

    assert result["status"] == "comparable"
    assert result["payload"]["benchmark_coverage"] == {
        "scored": 15,
        "expected": 15,
        "result_based": 15,
        "comparison_method_counts": None,
    }
    assert result["payload"]["result_based_question_count"] == 15
    assert result["payload"]["final_evidence"] is True


def test_protocol_preflight_receives_method_counts_and_blocks_string_fallback(
    tmp_path: Path,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 15,
            "questions": [{"id": str(index), "question": "q"} for index in range(15)],
        },
    )
    artifacts.write_json_path(artifacts.split_meta_path, {"split_strategy": "canonical"})
    artifacts.update_workspace_meta(
        {
            "fresh_baseline": True,
            "serving_endpoint": "databricks-gpt-5-5-pro",
            "serving_model": "databricks-gpt-5-5",
            "serving_reasoning_effort": "xhigh",
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=genie_service,
            benchmark_service=benchmark_service,
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    result = flow._write_protocol_preflight(
        mode="autonomous",
        strategy="serving",
        summary={
            "evaluation_context": {
                "preferred_comparison_mode": "result_based",
                "warehouse_id": "warehouse-123",
            },
            "overall": {
                "question_count": 15,
                "comparison_metrics": {
                    "scored_question_count": 15,
                    "result_based_question_count": 15,
                    "comparison_method_counts": {
                        "result_match": 14,
                        "string_fallback_unbound_parameters": 1,
                    },
                },
            },
        },
    )

    assert result["status"] == "non_comparable"
    assert result["payload"]["comparison_method_counts"] == {
        "result_match": 14,
        "string_fallback_unbound_parameters": 1,
    }
    assert result["errors"][0]["code"] == "string_fallback_in_result_based_evidence"


def test_protocol_preflight_separates_scored_and_result_based_coverage(tmp_path: Path):
    space_id = "2" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 15,
            "questions": [{"id": str(index), "question": "q"} for index in range(15)],
        },
    )
    artifacts.write_json_path(artifacts.split_meta_path, {"split_strategy": "canonical"})
    artifacts.update_workspace_meta(
        {
            "fresh_baseline": True,
            "serving_endpoint": "databricks-gpt-5-5-pro",
            "serving_model": "databricks-gpt-5-5",
            "serving_reasoning_effort": "xhigh",
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=SpaceExporter(
            genie_service=genie_service,
            benchmark_service=benchmark_service,
            artifacts=artifacts,
        ),
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    result = flow._write_protocol_preflight(
        mode="autonomous",
        strategy="serving",
        summary={
            "evaluation_context": {
                "preferred_comparison_mode": "result_based",
                "warehouse_id": "warehouse-123",
            },
            "overall": {
                "question_count": 15,
                "comparison_metrics": {
                    "scored_question_count": 15,
                    "result_based_question_count": 14,
                },
            },
        },
    )

    assert result["payload"]["benchmark_coverage"] == {
        "scored": 15,
        "expected": 15,
        "result_based": 14,
        "comparison_method_counts": None,
    }
    assert [error["code"] for error in result["errors"]] == [
        "incomplete_result_based_benchmark_coverage"
    ]


def test_first_serving_candidate_replays_best_cross_run_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / "current" / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"questions": [{"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"}]},
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})

    previous_artifacts = ArtifactManager(tmp_path / "previous" / space_id, space_id)
    payload_path = previous_artifacts.history_dir / "serving_candidates" / "candidate_payload.json"
    previous_artifacts.write_json_path(
        payload_path,
        {
            "candidate_name": "prior_single_example",
            "rationale": "Prior accepted candidate.",
            "expected_impact": "Recover one visible failure.",
            "risk": "Uses one benchmark-tagged example.",
            "owner_followups": [],
            "operations": [
                {
                    "op": "add_example_sql",
                    "id": "a" * 32,
                    "question": None,
                    "sql": None,
                    "usage_guidance": None,
                    "reason": "target_train_failure:" + "a" * 32,
                }
            ],
        },
    )
    previous_artifacts.write_json_path(
        previous_artifacts.optimization_summary_path,
        {
            "baseline": {"full_pass_rate": 40.0},
            "final": {
                "overall": {"pass_rate": 60.0},
                "test": {"pass_rate": 33.33},
            },
            "passes": [
                {
                    "name": "serving_iteration_01",
                    "outcome": "accepted",
                    "train_rate": 77.78,
                    "validation_rate": 66.67,
                    "checkpoint_path": str(previous_artifacts.checkpoints_dir / "best"),
                    "candidate_summary": {
                        "candidate_name": "prior_single_example",
                        "candidate_mode": "example_sql",
                        "operation_count": 1,
                        "payload_path": str(payload_path),
                        "risk_assessment": {
                            "canonicalization": {
                                "canonicalized_example_sql_ids": ["a" * 32],
                            },
                            "reasons": [],
                        },
                    },
                }
            ],
        },
    )

    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _UnexpectedCandidateClient:
        def __init__(self, **_kwargs):
            raise AssertionError("fresh serving proposal should not be requested")

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _UnexpectedCandidateClient)
    monkeypatch.setattr(
        "maxgenie.workspace_flow.serving_candidate_mode_for_iteration",
        lambda *_args, **_kwargs: "example_sql",
    )

    candidate, summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    examples = candidate.instructions.example_question_sqls or []
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert summary["changed"] is True
    assert summary["candidate_mode"] == "example_sql"
    assert summary["cross_run_replay"]["candidate_name"] == "prior_single_example"
    assert examples[0].question == ["Train question"]
    assert examples[0].sql == ["SELECT 1"]
    assert '"phase": "serving_candidate_replay_loaded"' in iteration_log


def test_default_cross_run_replay_skips_incompatible_candidate_mode(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / "current" / space_id, space_id)
    config = _space_config()
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    lesson = SimpleNamespace(
        attempt=1,
        source_run="deterministic-guidance-20260525T213537Z",
        candidate_name="deterministic_guidance_literalize_bind_parameters",
        candidate_mode="deterministic_guidance",
        payload_path="/runs/deterministic-guidance/candidate_payload.json",
    )
    cross_run_memory = SimpleNamespace(
        best_prior_sequence=(),
        best_prior_lesson=lesson,
    )

    payload, selected_lesson = flow._load_cross_run_replay_payload(
        cross_run_memory=cross_run_memory,
        serving_iteration_index=1,
        optimization_summary=None,
        expected_candidate_mode="artifact_patch_v2",
    )

    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(
        encoding="utf-8"
    )
    assert payload is None
    assert selected_lesson is None
    assert "candidate_mode_incompatible_with_scheduled_mode" in iteration_log
    assert "deterministic_guidance" in iteration_log
    assert "artifact_patch_v2" in iteration_log


def test_serving_candidate_replays_prior_accepted_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    space_id = "1" * 32
    first_id = "a" * 32
    second_id = "b" * 32
    artifacts = ArtifactManager(tmp_path / "current" / space_id, space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {"id": first_id, "question": "First train question", "expected_sql": "SELECT 1"},
                {"id": second_id, "question": "Second train question", "expected_sql": "SELECT 2"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        [
            {"id": first_id, "question": "First train question", "expected_sql": "SELECT 1"},
            {"id": second_id, "question": "Second train question", "expected_sql": "SELECT 2"},
        ],
    )
    artifacts.write_json_path(artifacts.validation_set_path, {"ids": []})
    artifacts.write_json_path(artifacts.space_diagnostics_path, {"summary": "ok"})

    previous_artifacts = ArtifactManager(tmp_path / "previous" / space_id, space_id)
    first_payload_path = previous_artifacts.history_dir / "serving_candidates" / "first_payload.json"
    second_payload_path = previous_artifacts.history_dir / "serving_candidates" / "second_payload.json"
    previous_artifacts.write_json_path(
        first_payload_path,
        {
            "candidate_name": "prior_first_example",
            "rationale": "Prior first accepted candidate.",
            "expected_impact": "Recover first visible failure.",
            "risk": "Uses one benchmark-tagged example.",
            "owner_followups": [],
            "operations": [
                {
                    "op": "add_example_sql",
                    "id": first_id,
                    "question": None,
                    "sql": None,
                    "usage_guidance": None,
                    "reason": "target_train_failure:" + first_id,
                }
            ],
        },
    )
    previous_artifacts.write_json_path(
        second_payload_path,
        {
            "candidate_name": "prior_second_example",
            "rationale": "Prior second accepted candidate.",
            "expected_impact": "Recover second visible failure.",
            "risk": "Uses one benchmark-tagged example.",
            "owner_followups": [],
            "operations": [
                {
                    "op": "add_example_sql",
                    "id": second_id,
                    "question": None,
                    "sql": None,
                    "usage_guidance": None,
                    "reason": "target_train_failure:" + second_id,
                }
            ],
        },
    )
    previous_artifacts.write_json_path(
        previous_artifacts.optimization_summary_path,
        {
            "baseline": {"full_pass_rate": 40.0},
            "final": {
                "overall": {"pass_rate": 60.0},
                "test": {"pass_rate": 33.33},
            },
            "passes": [
                {
                    "name": "serving_iteration_01",
                    "outcome": "accepted",
                    "train_rate": 66.67,
                    "validation_rate": 58.33,
                    "checkpoint_path": str(previous_artifacts.checkpoints_dir / "first"),
                    "candidate_summary": {
                        "candidate_name": "prior_first_example",
                        "candidate_mode": "example_sql",
                        "operation_count": 1,
                        "payload_path": str(first_payload_path),
                        "risk_assessment": {
                            "canonicalization": {
                                "canonicalized_example_sql_ids": [first_id],
                            },
                            "reasons": [],
                        },
                    },
                },
                {
                    "name": "serving_iteration_02",
                    "outcome": "accepted",
                    "train_rate": 77.78,
                    "validation_rate": 66.67,
                    "checkpoint_path": str(previous_artifacts.checkpoints_dir / "second"),
                    "candidate_summary": {
                        "candidate_name": "prior_second_example",
                        "candidate_mode": "example_sql",
                        "operation_count": 1,
                        "payload_path": str(second_payload_path),
                        "risk_assessment": {
                            "canonicalization": {
                                "canonicalized_example_sql_ids": [second_id],
                            },
                            "reasons": [],
                        },
                    },
                },
            ],
        },
    )

    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    class _UnexpectedCandidateClient:
        def __init__(self, **_kwargs):
            raise AssertionError("fresh serving proposal should not be requested")

    monkeypatch.setattr("maxgenie.workspace_flow.ServingCandidateClient", _UnexpectedCandidateClient)
    monkeypatch.setattr(
        "maxgenie.workspace_flow.serving_candidate_mode_for_iteration",
        lambda *_args, **_kwargs: "example_sql",
    )

    first_candidate, first_summary = flow._run_serving_candidate(
        current_config=config,
        spec=CandidateSpec(
            name="serving_iteration_01",
            family="serving",
            description="Serving candidate",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )
    artifacts.write_json_path(
        artifacts.optimization_summary_path,
        {
            "passes": [
                {
                    "name": "serving_iteration_01",
                    "outcome": "accepted",
                    "candidate_summary": first_summary,
                }
            ],
        },
    )

    second_candidate, second_summary = flow._run_serving_candidate(
        current_config=first_candidate,
        spec=CandidateSpec(
            name="serving_iteration_02",
            family="serving",
            description="Serving candidate",
        ),
        serving_config=ServingCandidateConfig(endpoint="candidate-endpoint", api_format="chat"),
    )

    examples = second_candidate.instructions.example_question_sqls or []
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert first_summary["cross_run_replay"]["candidate_name"] == "prior_first_example"
    assert second_summary["cross_run_replay"]["candidate_name"] == "prior_second_example"
    assert [example.question[0] for example in examples] == [
        "First train question",
        "Second train question",
    ]
    assert '"candidate_name": "prior_first_example"' in iteration_log
    assert '"candidate_name": "prior_second_example"' in iteration_log


def test_cross_run_replay_falls_back_to_compatible_sequence_suffix(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    genie_service = _DummyGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    s18_first = SimpleNamespace(
        attempt=1,
        source_run="s18",
        candidate_name="shared_first_payload",
        payload_path="/runs/s18/first.json",
        validation_rate=75.0,
        train_rate=88.89,
        full_delta=26.67,
    )
    s15_first = SimpleNamespace(
        attempt=1,
        source_run="s15",
        candidate_name="shared_first_payload",
        payload_path="/runs/s15/first.json",
        validation_rate=66.67,
        train_rate=77.78,
        full_delta=20.0,
    )
    s15_second = SimpleNamespace(
        attempt=2,
        source_run="s15",
        candidate_name="compatible_second_payload",
        payload_path="/runs/s15/second.json",
        validation_rate=66.67,
        train_rate=77.78,
        full_delta=20.0,
    )
    unrelated_first = SimpleNamespace(
        attempt=1,
        source_run="other",
        candidate_name="unrelated_first_payload",
        payload_path="/runs/other/first.json",
        validation_rate=83.33,
        train_rate=88.89,
        full_delta=33.33,
    )
    cross_run_memory = SimpleNamespace(
        best_prior_sequence=(s18_first,),
        best_prior_lesson=s18_first,
        accepted_lessons=(s18_first, unrelated_first, s15_first, s15_second),
    )
    optimization_summary = {
        "passes": [
            {
                "name": "serving_iteration_01",
                "outcome": "accepted",
                "candidate_summary": {
                    "candidate_name": "shared_first_payload",
                    "cross_run_replay": {
                        "candidate_name": "shared_first_payload",
                        "payload_path": "/runs/s18/first.json",
                    },
                },
            }
        ]
    }

    lesson = flow._cross_run_replay_lesson_for_iteration(
        cross_run_memory=cross_run_memory,
        serving_iteration_index=2,
        optimization_summary=optimization_summary,
    )

    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert lesson is s15_second
    assert "serving_candidate_replay_sequence_fallback" in iteration_log
    assert "compatible_second_payload" in iteration_log


def test_ensure_clone_defaults_to_current_user_home_for_portable_permissions(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    source_config = _space_config()
    source_config.parent_path = "/Users/source-owner@example.com"
    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    flow.ensure_clone(
        source_space_id=space_id,
        source_config=source_config,
        source_title="Source Space",
        input_space_id=space_id,
    )

    clone_meta = artifacts.load_clone_meta()
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert genie_service.created_parent_path == "/Users/optimizer@example.com"
    assert genie_service._config.parent_path == "/Users/optimizer@example.com"
    assert clone_meta is not None
    assert clone_meta["clone_parent_path"] == "/Users/optimizer@example.com"
    assert clone_meta["clone_parent_path_source"] == "current_user"
    assert '"clone_parent_path_source": "current_user"' in iteration_log


def test_ensure_clone_honors_configured_clone_parent_path(tmp_path: Path):
    space_id = "2" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    source_config = _space_config()
    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
        clone_parent_path="/Workspace/Shared/team-maxgenie",
    )

    flow.ensure_clone(
        source_space_id=space_id,
        source_config=source_config,
        source_title="Source Space",
        input_space_id=space_id,
    )

    clone_meta = artifacts.load_clone_meta()

    assert genie_service.created_parent_path == "/Shared/team-maxgenie"
    assert clone_meta is not None
    assert clone_meta["clone_parent_path"] == "/Shared/team-maxgenie"
    assert clone_meta["clone_parent_path_source"] == "argument"


def test_ensure_clone_honors_env_clone_parent_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MAXGENIE_CLONE_PARENT_PATH", "/Workspace/Shared/env-team")
    space_id = "3" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    source_config = _space_config()
    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    flow.ensure_clone(
        source_space_id=space_id,
        source_config=source_config,
        source_title="Source Space",
        input_space_id=space_id,
    )

    clone_meta = artifacts.load_clone_meta()

    assert genie_service.created_parent_path == "/Shared/env-team"
    assert clone_meta is not None
    assert clone_meta["clone_parent_path"] == "/Shared/env-team"
    assert clone_meta["clone_parent_path_source"] == "env:MAXGENIE_CLONE_PARENT_PATH"


def test_push_preserves_clone_parent_path_instead_of_source_parent(tmp_path: Path):
    space_id = "4" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
        clone_parent_path="/Users/optimizer@example.com",
        clone_parent_path_source="current_user",
    )
    remote_config = _space_config()
    remote_config.parent_path = "/Users/optimizer@example.com"
    editable_config = _space_config()
    editable_config.parent_path = "/Users/source-owner@example.com"
    decompose_config(editable_config, artifacts.space_dir, include_benchmarks=False)
    genie_service = _DummyGenieService(remote_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    flow.push()

    assert genie_service.updated_parent_path == "/Users/optimizer@example.com"


def test_export_workspace_writes_decomposed_assets_and_redacted_test_set(tmp_path: Path):
    space_id = "0" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(space_id, validation_fraction=0.0, test_fraction=0.5, split_seed=7)

    assert bundle.prompt_path.exists()
    assert (artifacts.space_dir / "meta.yml").exists()
    assert not (artifacts.space_dir / "benchmarks.yml").exists()

    train_payload = artifacts.read_json_path(artifacts.train_set_path)
    dev_payload = artifacts.read_json_path(artifacts.dev_set_path)
    dev_folds_payload = artifacts.read_json_path(artifacts.dev_folds_path)
    test_payload = artifacts.read_json_path(artifacts.test_set_path)
    benchmark_payload = artifacts.read_json_path(artifacts.benchmark_payload_path)
    hidden_test_payload = artifacts.read_json_path(artifacts.hidden_test_payload_path)
    blocked_notes = artifacts.read_text_path(artifacts.blocked_failures_path)

    assert train_payload["question_count"] == 1
    assert dev_payload["question_count"] == 0
    assert dev_folds_payload["validation_strategy"] == "blind_split"
    assert "expected_sql" in train_payload["questions"][0]
    assert test_payload["question_count"] == 1
    assert "ids" in test_payload
    assert "questions" not in test_payload
    assert benchmark_payload["payload_scope"] == "visible_pool"
    assert benchmark_payload["question_count"] == 1
    assert hidden_test_payload["payload_scope"] == "hidden_test"
    assert hidden_test_payload["question_count"] == 1
    assert blocked_notes.startswith("# Blocked Failures")


def test_export_workspace_clears_stale_split_results(tmp_path: Path):
    space_id = "9" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    artifacts.write_text_path(artifacts.latest_test_score_path, "33.3\n")
    artifacts.write_text_path(artifacts.latest_dev_score_path, "50.0\n")
    artifacts.write_json_path(artifacts.latest_dev_summary_path, {"pass_rate": 50.0})
    artifacts.write_json_path(artifacts.latest_train_path, {"space_id": space_id, "started_at": None, "completed_at": None, "results": []})
    artifacts.write_json_path(artifacts.latest_full_summary_path, {"overall": {"pass_rate": 12.0}})

    flow.export_workspace(space_id, validation_fraction=0.2, test_fraction=0.2, split_seed=7)

    assert not artifacts.latest_train_path.exists()
    assert not artifacts.latest_dev_score_path.exists()
    assert not artifacts.latest_dev_summary_path.exists()
    assert not artifacts.latest_test_score_path.exists()
    assert not artifacts.latest_full_summary_path.exists()


def test_export_workspace_overwrites_stale_split_payloads(tmp_path: Path):
    space_id = "98" * 16
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "benchmark_source": "stale",
            "payload_scope": "visible_pool",
            "question_count": 1,
            "questions": [
                {"id": "z" * 32, "question": "Stale visible question", "expected_sql": "SELECT 99"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.hidden_test_payload_path,
        {
            "benchmark_source": "stale",
            "payload_scope": "hidden_test",
            "question_count": 1,
            "questions": [
                {"id": "y" * 32, "question": "Stale hidden question", "expected_sql": "SELECT 98"},
            ],
        },
    )

    flow.export_workspace(space_id, validation_fraction=0.0, test_fraction=0.5, split_seed=7)

    benchmark_payload = artifacts.read_json_path(artifacts.benchmark_payload_path)
    hidden_test_payload = artifacts.read_json_path(artifacts.hidden_test_payload_path)
    payload_ids = {question["id"] for question in benchmark_payload["questions"]}
    hidden_ids = {question["id"] for question in hidden_test_payload["questions"]}

    assert benchmark_payload["benchmark_source"] == "space_config"
    assert hidden_test_payload["benchmark_source"] == "space_config"
    assert payload_ids | hidden_ids == {"a" * 32, "b" * 32}
    assert payload_ids.isdisjoint(hidden_ids)


def test_rewrite_split_from_population_baseline_updates_workspace_strategy(tmp_path: Path):
    space_id = "8" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    genie_service = _DummyGenieService(_space_config_with_question_count(15))
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(space_id, validation_fraction=0.2, test_fraction=0.2, split_seed=42)
    baseline_run = _baseline_run_for_questions(
        bundle.questions,
        passed_ids={bundle.questions[index].id for index in (0, 1, 2, 12, 13, 14)},
    )

    rewritten = flow.rewrite_split_from_population_baseline(
        bundle=bundle,
        baseline_run=baseline_run,
    )

    split_meta = artifacts.read_json_path(artifacts.split_meta_path)
    prompt = artifacts.prompt_path.read_text(encoding="utf-8")
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert rewritten.split.split_strategy == "baseline_outcome_balanced"
    assert split_meta["split_strategy"] == "baseline_outcome_balanced"
    assert "Split strategy: `baseline_outcome_balanced`." in prompt
    assert '"phase": "split_rebalanced"' in iteration_log


def test_initialize_baseline_from_observed_population_writes_baseline_artifacts(tmp_path: Path):
    space_id = "9" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    genie_service = _DummyGenieService(_space_config_with_question_count(15))
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(space_id, validation_fraction=0.2, test_fraction=0.2, split_seed=42)
    observed_run = _baseline_run_for_questions(
        bundle.questions,
        passed_ids={bundle.questions[index].id for index in (0, 1, 2, 12, 13, 14)},
    )
    flow.rewrite_split_from_population_baseline(
        bundle=bundle,
        baseline_run=observed_run,
    )

    summary = flow.initialize_baseline_from_observed_population(
        label="baseline",
        observed_run=observed_run,
    )

    baseline_train = artifacts.read_json_path(artifacts.baseline_train_path)
    validation_summary = artifacts.read_json_path(artifacts.baseline_validation_summary_path)
    baseline_full = artifacts.read_json_path(artifacts.baseline_full_summary_path)
    latest_full = artifacts.read_json_path(artifacts.latest_full_summary_path)

    assert baseline_train["baseline_source"] == "observed_benchmark_seed"
    assert validation_summary["baseline_source"] == "observed_benchmark_seed"
    assert baseline_full["baseline_source"] == "observed_benchmark_seed"
    assert latest_full["baseline_source"] == "observed_benchmark_seed"
    assert baseline_full["overall"]["pass_rate"] == latest_full["overall"]["pass_rate"]
    assert summary["overall"]["pass_rate"] == latest_full["overall"]["pass_rate"]
    assert "benchmark_baseline_observed" in artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")


def test_initialize_baseline_from_population_samples_uses_stable_pass_guards(
    tmp_path: Path,
):
    space_id = "a" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    genie_service = _DummyGenieService(_space_config_with_question_count(15))
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(space_id, validation_fraction=0.2, test_fraction=0.2, split_seed=42)
    flaky_train_id = bundle.split.train_questions[0].id
    stable_train_id = bundle.split.train_questions[1].id
    holdout_ids = {question.id for question in bundle.split.test_questions}
    sample_pass_sets = [
        {flaky_train_id, stable_train_id},
        {flaky_train_id, stable_train_id},
        {stable_train_id},
        {stable_train_id},
        set(),
    ]
    flow._full_sample_runs = {
        "population_baseline": [
            _sample_run_for_questions(bundle.questions, passed_ids=passed_ids)
            for passed_ids in sample_pass_sets
        ]
    }

    summary = flow.initialize_baseline_from_population_samples(
        label="baseline",
        questions=bundle.questions,
    )

    baseline_train = artifacts.read_json_path(artifacts.baseline_train_path)
    validation_summary = artifacts.read_json_path(artifacts.baseline_validation_summary_path)
    baseline_full = artifacts.read_json_path(artifacts.baseline_full_summary_path)
    train_results = {
        result["question_id"]: result
        for result in baseline_train["results"]
    }
    train_meta = baseline_train["multi_sample_baseline"]

    assert train_results[flaky_train_id]["status"] == "failed"
    assert train_results[stable_train_id]["status"] == "passed"
    assert flaky_train_id in baseline_train["diff"]["known_failures_after"]
    assert stable_train_id not in baseline_train["diff"]["known_failures_after"]
    assert train_meta["per_question_pass_fraction"][flaky_train_id] == 0.4
    assert train_meta["per_question_pass_fraction"][stable_train_id] == 0.8
    assert not (set(train_meta["per_question_pass_fraction"]) & holdout_ids)
    assert validation_summary["baseline_source"] == "population_multi_sample_baseline"
    assert baseline_full["baseline_source"] == "population_multi_sample_baseline"
    assert "per_question_pass_fraction" not in baseline_full["multi_sample_baseline"]
    assert summary["overall"]["pass_rate"] == baseline_full["overall"]["pass_rate"]
    assert "benchmark_baseline_population_multi_sample" in artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")


def test_benchmark_cache_hash_includes_strategy_and_threshold():
    fingerprint = _benchmark_cache_hash(
        config_hash="hash123",
        match_threshold=0.85,
        comparator=None,
    )
    comparator_fingerprint = _benchmark_cache_hash(
        config_hash="hash123",
        match_threshold=0.85,
        comparator=_ComparatorWithFingerprint(),
    )

    assert fingerprint == "hash123|threshold=0.850|strategy=string-compare"
    assert comparator_fingerprint == "hash123|threshold=0.850|strategy=result-comparator|v2"


def test_derived_full_summary_uses_latest_train_and_test_score():
    train_run = type("TrainRun", (), {})()
    train_run.total = 12
    train_run.passed = 12
    train_run.failed = 0
    train_run.errors = 0
    train_run.pass_rate = 100.0

    summary = _derived_full_summary(
        latest_train=train_run,
        latest_dev_summary={
            "question_count": 2,
            "pass_rate": 50.0,
            "validation_strategy": "blind_split",
        },
        latest_test_pass_rate=33.3,
        latest_test_question_count=3,
    )

    assert summary == {
        "overall": {
            "total": 17,
            "passed": 14,
            "failed": 3,
            "errors": 0,
            "pass_rate": 82.35,
        },
        "train": {
            "total": 12,
            "passed": 12,
            "failed": 0,
            "errors": 0,
            "pass_rate": 100.0,
        },
        "validation": {
            "question_count": 2,
            "passed": 1,
            "failed": 1,
            "errors": 0,
            "pass_rate": 50.0,
            "validation_strategy": "blind_split",
        },
        "test": {
            "question_count": 3,
            "passed": 1,
            "failed": 2,
            "errors": 0,
            "pass_rate": 33.3,
        },
    }


def test_derived_full_summary_uses_kfold_validation_as_visible_pool():
    train_run = type("TrainRun", (), {})()
    train_run.total = 9
    train_run.passed = 9
    train_run.failed = 0
    train_run.errors = 0
    train_run.pass_rate = 100.0

    summary = _derived_full_summary(
        latest_train=train_run,
        latest_dev_summary={
            "question_count": 12,
            "passed": 10,
            "failed": 2,
            "errors": 0,
            "pass_rate": 83.33,
            "validation_strategy": "small_sample_kfold",
            "fold_count": 3,
            "mean_fold_pass_rate": 83.33,
            "min_fold_pass_rate": 75.0,
        },
        latest_test_pass_rate=33.3,
        latest_test_question_count=3,
    )

    assert summary == {
        "overall": {
            "total": 15,
            "passed": 11,
            "failed": 4,
            "errors": 0,
            "pass_rate": 73.33,
        },
        "overall_basis": "visible_kfold_plus_holdout",
        "train": {
            "total": 9,
            "passed": 9,
            "failed": 0,
            "errors": 0,
            "pass_rate": 100.0,
        },
        "validation": {
            "question_count": 12,
            "passed": 10,
            "failed": 2,
            "errors": 0,
            "pass_rate": 83.33,
            "validation_strategy": "small_sample_kfold",
            "fold_count": 3,
            "mean_fold_pass_rate": 83.33,
            "min_fold_pass_rate": 75.0,
        },
        "test": {
            "question_count": 3,
            "passed": 1,
            "failed": 2,
            "errors": 0,
            "pass_rate": 33.3,
        },
    }


def test_export_workspace_falls_back_to_legacy_local_yaml(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    config.to_yaml(str(artifacts.space_dir / "current.space.yml"))

    genie_service = _FailingGenieService(config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(space_id, validation_fraction=0.0, test_fraction=0.5, split_seed=11)

    assert bundle.space_id == space_id
    assert artifacts.split_meta_path.exists()
    split_meta = artifacts.read_json_path(artifacts.split_meta_path)
    assert split_meta["source_space_id"] == space_id


def test_export_workspace_recovers_benchmarks_from_matching_local_workspace(tmp_path: Path):
    source_space_id = "2" * 32
    recovery_space_id = "3" * 32

    current_artifacts = ArtifactManager(tmp_path / source_space_id, source_space_id)
    recovery_artifacts = ArtifactManager(tmp_path / recovery_space_id, recovery_space_id)

    source_config = _space_config()
    source_config.title = "(Clone) Demo Space"
    source_config.benchmarks = None

    recovery_config = _space_config()
    recovery_config.to_yaml(str(recovery_artifacts.space_dir / "current.space.yml"))
    recovery_artifacts.write_json_path(
        recovery_artifacts.benchmarks_dir / "full.benchmarks.json",
        [
            {
                "id": "a" * 32,
                "question": "Train question",
                "expected_sql": "SELECT 1",
            },
            {
                "id": "b" * 32,
                "question": "Held-out question",
                "expected_sql": "SELECT 2",
            },
        ],
    )

    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=current_artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=current_artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(source_space_id, validation_fraction=0.0, test_fraction=0.5, split_seed=13)

    assert bundle.benchmark_source == f"local_workspace:{recovery_space_id}"
    payload = current_artifacts.load_benchmark_payload()
    assert payload is not None
    assert payload["question_count"] == 1
    hidden_payload = current_artifacts.load_hidden_test_payload()
    assert hidden_payload is not None
    assert hidden_payload["question_count"] == 1
    split_meta = current_artifacts.read_json_path(current_artifacts.split_meta_path)
    assert split_meta["benchmark_source"] == f"local_workspace:{recovery_space_id}"


def test_export_workspace_errors_when_no_benchmarks_exist_anywhere(tmp_path: Path):
    space_id = "4" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)

    source_config = _space_config()
    source_config.benchmarks = None

    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    with pytest.raises(WorkspaceFlowError, match="no benchmark set"):
        flow.export_workspace(space_id, validation_fraction=0.0, test_fraction=0.5, split_seed=17)


def test_export_workspace_allows_empty_benchmarks_for_fixed_seed(tmp_path: Path):
    space_id = "4" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)

    source_config = _space_config()
    source_config.benchmarks = None

    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    bundle = flow.export_workspace(
        space_id,
        validation_fraction=0.0,
        test_fraction=0.5,
        split_seed=17,
        allow_empty_benchmarks=True,
    )

    assert bundle.questions == []
    assert bundle.split.train_questions == []
    assert bundle.split.validation_questions == []
    assert bundle.split.test_questions == []
    assert artifacts.space_dir.exists()
    assert not artifacts.benchmark_payload_path.exists()


def test_no_benchmark_advisory_best_practices_updates_clone_only(tmp_path: Path):
    space_id = "4" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)

    source_config = _space_config()
    source_config.benchmarks = None
    genie_service = _DummyGenieService(source_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    decompose_config(source_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )

    summary = flow.run_no_benchmark_advisory_best_practices()

    assert summary["mode"] == "advisory_best_practices"
    assert summary["benchmark_verification_label"] == "not benchmark-verified"
    assert summary["source_space_modified"] is False
    assert summary["promotion_eligible"] is False
    assert summary["clone_space_id"] == "clone-123"
    assert artifacts.advisory_best_practices_path.exists()
    curation_summary = artifacts.read_json_path(artifacts.curation_summary_path)
    assert curation_summary["candidate_status"] == "not benchmark-verified"
    assert curation_summary["evidence_classification"] == "advisory_best_practices"
    assert genie_service.updated_title == "Demo Space"
    assert benchmark_service.start_attempts >= 1
    log_text = artifacts.read_text_path(artifacts.history_dir / "iteration_log.jsonl")
    assert '"phase": "no_benchmark_advisory_started"' in log_text
    assert '"phase": "no_benchmark_advisory_completed"' in log_text
    assert '"phase": "push"' in log_text


def test_push_preserves_existing_clone_title(tmp_path: Path):
    space_id = "5" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)

    local_config = _space_config()
    local_config.title = "(Clone) Demo Space"
    decompose_config(local_config, artifacts.space_dir, include_benchmarks=False)

    remote_config = _space_config()
    remote_config.title = "[maxgenie 88.80%] Demo Space"
    genie_service = _DummyGenieService(remote_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    flow.push()

    assert genie_service.updated_title == "[maxgenie 88.80%] Demo Space"


def test_clone_title_rewrites_legacy_maxgenie_prefix():
    assert (
        _formatted_clone_title(
            "[MaxGenie 50.00% full] Demo Space",
            full_pass_rate=75.0,
            fallback="space-123",
        )
        == "[maxgenie 75%] Demo Space"
    )


def test_run_full_benchmark_refreshes_clone_title_with_latest_score(tmp_path: Path):
    space_id = "4" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.dev_set_path,
        {
            "question_count": 1,
            "ids": ["b" * 32],
        },
    )
    artifacts.write_json_path(
        artifacts.dev_folds_path,
        {
            "validation_strategy": "blind_split",
            "question_count": 1,
            "ids": ["b" * 32],
        },
    )
    artifacts.write_json_path(
        artifacts.test_set_path,
        {
            "question_count": 1,
            "ids": ["c" * 32],
        },
    )
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 3,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": "b" * 32, "question": "Validation question", "expected_sql": "SELECT 2"},
                {"id": "c" * 32, "question": "Held-out question", "expected_sql": "SELECT 3"},
            ],
        },
    )

    remote_config = _space_config()
    remote_config.title = "[Space Optimizer 50.00% full] Demo Space"
    remote_config.benchmarks = None
    genie_service = _DummyGenieService(remote_config)
    benchmark_service = _PassingRunBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    summary = flow.run_full_benchmark(label="full-score")

    assert summary["overall"]["pass_rate"] == 100.0
    assert genie_service.updated_title == "[maxgenie 100%] Demo Space"


def test_push_rehydrates_remote_benchmarks_from_local_payload(tmp_path: Path):
    space_id = "a" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)

    local_config = _space_config()
    decompose_config(local_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": "b" * 32, "question": "Held-out question", "expected_sql": "SELECT 2"},
            ],
        },
    )

    remote_config = _space_config()
    remote_config.benchmarks = None
    genie_service = _DummyGenieService(remote_config)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    flow.push()

    assert genie_service._config.benchmarks is not None
    assert len(genie_service._config.benchmarks.questions or []) == 2


def test_push_recovers_deleted_clone_from_local_workspace(tmp_path: Path):
    space_id = "c" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.update_workspace_meta(
        {
            "input_space_id": space_id,
            "managed_source_space_id": "clone-123",
            "active_space_id": "clone-123",
        }
    )

    local_config = _space_config()
    local_config.title = "Recovered Demo Space"
    decompose_config(local_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": "b" * 32, "question": "Held-out question", "expected_sql": "SELECT 2"},
            ],
        },
    )

    genie_service = _MissingCloneGenieService(_space_config(), missing_ids={"clone-123"})
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    info = flow.push()

    clone_meta = artifacts.load_clone_meta()
    workspace_meta = artifacts.load_workspace_meta()
    lineage = artifacts.load_lineage(space_id)
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert info.space_id == "recovered-1"
    assert clone_meta is not None
    assert clone_meta["clone_space_id"] == "recovered-1"
    assert workspace_meta is not None
    assert workspace_meta["managed_source_space_id"] == "recovered-1"
    assert lineage is not None
    assert lineage["managed_source_space_id"] == "recovered-1"
    assert '"phase": "clone_recovered"' in iteration_log
    assert '"recovered_clone": true' in iteration_log


def test_push_recovers_when_clone_becomes_unreadable_after_update(tmp_path: Path):
    space_id = "c1" * 16
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
        clone_parent_path="/Users/optimizer@example.com",
        clone_parent_path_source="current_user",
    )
    artifacts.update_workspace_meta(
        {
            "input_space_id": space_id,
            "managed_source_space_id": "clone-123",
            "active_space_id": "clone-123",
        }
    )
    local_config = _space_config()
    local_config.title = "Recovered After Update"
    decompose_config(local_config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
            ],
        },
    )

    genie_service = _MissingAfterUpdateGenieService(_space_config(), missing_ids=set())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    info = flow.push()

    clone_meta = artifacts.load_clone_meta()
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert info.space_id == "recovered-1"
    assert clone_meta is not None
    assert clone_meta["clone_space_id"] == "recovered-1"
    assert '"reason": "push_preflight_failed:' in iteration_log
    assert '"recovered_clone": true' in iteration_log


def test_run_train_benchmark_recovers_when_clone_disappears_mid_run(tmp_path: Path):
    space_id = "d" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.update_workspace_meta(
        {
            "input_space_id": space_id,
            "managed_source_space_id": "clone-123",
            "active_space_id": "clone-123",
        }
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
            ],
        },
    )

    local_config = _space_config()
    decompose_config(local_config, artifacts.space_dir, include_benchmarks=False)

    genie_service = _MissingCloneGenieService(_space_config(), missing_ids=set())
    benchmark_service = _RecoveringBenchmarkService(genie_service)
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    run = flow.run_train_benchmark(label="train-recovery")

    clone_meta = artifacts.load_clone_meta()
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert benchmark_service.calls == 2
    assert run.pass_rate == 100.0
    assert clone_meta is not None
    assert clone_meta["clone_space_id"] == "recovered-1"
    assert '"phase": "benchmark_recovery_checkpoint_created"' in iteration_log
    assert '"phase": "clone_recovered"' in iteration_log
    assert '"restored_checkpoint_source": "explicit"' in iteration_log
    assert '"phase": "benchmark_train_subset"' in iteration_log
    assert '"clone_space_id": "recovered-1"' in iteration_log
    assert '"requested_clone_space_id": "clone-123"' in iteration_log


def test_benchmark_missing_clone_recovery_preserves_inflight_candidate_state(tmp_path: Path):
    space_id = "e" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(
        clone_space_id="clone-123",
        source_space_id=space_id,
        input_space_id=space_id,
        managed_source_space_id="clone-123",
        active_space_id="clone-123",
    )
    artifacts.update_workspace_meta(
        {
            "input_space_id": space_id,
            "managed_source_space_id": "clone-123",
            "active_space_id": "clone-123",
        }
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
            ],
        },
    )

    accepted_config = _space_config()
    accepted_config.title = "Accepted Checkpoint"
    decompose_config(accepted_config, artifacts.space_dir, include_benchmarks=False)
    best_checkpoint = artifacts.create_checkpoint(
        label="accepted",
        metadata={"score": {"train": 66.67}},
    )
    artifacts.write_json_path(
        artifacts.optimization_summary_path,
        {"best_checkpoint_path": str(best_checkpoint)},
    )

    candidate_config = _space_config()
    candidate_config.title = "In Flight Candidate"
    decompose_config(candidate_config, artifacts.space_dir, include_benchmarks=False)

    genie_service = _MissingCloneGenieService(_space_config(), missing_ids=set())
    benchmark_service = _RecoveringBenchmarkService(genie_service)
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    run = flow.run_train_benchmark(label="train-recovery")

    lineage = artifacts.load_lineage(space_id)
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")

    assert run.pass_rate == 100.0
    assert genie_service._config.title == "In Flight Candidate"
    assert lineage is not None
    assert lineage["restored_checkpoint_source"] == "explicit"
    assert lineage["recovery_checkpoint_path"]
    assert '"phase": "benchmark_recovery_checkpoint_created"' in iteration_log
    assert '"restored_checkpoint_source": "explicit"' in iteration_log


def test_preflight_space_retries_until_conversation_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    space_id = "6" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _EventuallyReadyBenchmarkService(fail_count=2)
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    monkeypatch.setattr("maxgenie.workspace_flow._CONVERSATION_READY_POLL_SECONDS", 0.0)
    monkeypatch.setattr("maxgenie.workspace_flow._CONVERSATION_READY_TIMEOUT_SECONDS", 1.0)

    info, _ = flow.preflight_space(space_id)

    assert info.space_id == space_id
    assert benchmark_service.start_attempts == 3
    assert benchmark_service.message_attempts == 1


def test_preflight_space_retries_until_message_endpoint_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    space_id = "7" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _EventuallyReadableMessageBenchmarkService(fail_count=2)
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    monkeypatch.setattr("maxgenie.workspace_flow._CONVERSATION_READY_POLL_SECONDS", 0.0)
    monkeypatch.setattr("maxgenie.workspace_flow._CONVERSATION_READY_TIMEOUT_SECONDS", 1.0)

    info, _ = flow.preflight_space(space_id)

    assert info.space_id == space_id
    assert benchmark_service.start_attempts == 3
    assert benchmark_service.message_attempts == 3


def test_run_dev_benchmark_rehydrates_clone_benchmarks_from_local_payload(tmp_path: Path):
    space_id = "b" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    decompose_config(_space_config(), artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": "b" * 32, "question": "Held-out question", "expected_sql": "SELECT 2"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.dev_folds_path,
        {
            "validation_strategy": "blind_split",
            "question_count": 1,
            "ids": ["b" * 32],
        },
    )
    artifacts.write_json_path(
        artifacts.dev_set_path,
        {
            "question_count": 1,
            "ids": ["b" * 32],
        },
    )
    remote_config = _space_config()
    remote_config.benchmarks = None
    genie_service = _DummyGenieService(remote_config)
    benchmark_service = _PassingRunBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    summary = flow.run_dev_benchmark(label="dev-fallback")

    assert summary["question_count"] == 1
    assert summary["pass_rate"] == 100.0


def test_curate_workspace_writes_summary_and_rewrites_space(tmp_path: Path):
    space_id = "7" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    config.data_sources.tables[0].column_configs = [
        GenieColumnConfig(column_name="brand_name"),
        GenieColumnConfig(column_name="az_brand_id"),
    ]
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)

    flow = OptimizationWorkspace(
        genie_service=None,  # type: ignore[arg-type]
        benchmark_service=None,  # type: ignore[arg-type]
        artifacts=artifacts,
        exporter=None,  # type: ignore[arg-type]
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    summary = flow.curate_workspace(label="test-curate")
    meta = artifacts.read_json_path(artifacts.curation_summary_path)

    assert summary["changed"] is True
    assert meta["column_descriptions_added"] >= 1
    assert artifacts.space_diagnostics_path.exists()


def test_curate_workspace_mode_safe_defaults_records_mode(tmp_path: Path):
    space_id = "1" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    config = _space_config()
    config.data_sources.tables[0].column_configs = [
        GenieColumnConfig(column_name="brand_name"),
        GenieColumnConfig(column_name="az_brand_id"),
    ]
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)

    flow = OptimizationWorkspace(
        genie_service=None,  # type: ignore[arg-type]
        benchmark_service=None,  # type: ignore[arg-type]
        artifacts=artifacts,
        exporter=None,  # type: ignore[arg-type]
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    summary = flow.curate_workspace_mode(label="safe-defaults", mode="safe_defaults")
    meta = artifacts.read_json_path(artifacts.curation_summary_path)

    assert summary["mode"] == "safe_defaults"
    assert meta["mode"] == "safe_defaults"
    assert meta["candidate_status"] == "prepared"


def test_write_report_includes_curation_and_query_history_sections(tmp_path: Path):
    space_id = "8" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    decompose_config(_space_config(), artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.baseline_full_summary_path,
        {
            "baseline_source": "latest_genie_eval_run",
            "overall": {"total": 15, "passed": 6, "failed": 9, "errors": 0, "pass_rate": 40.0},
        },
    )
    artifacts.write_json_path(
        artifacts.results_dir / "observed_benchmark_seed.json",
        {
            "source": "latest_genie_eval_run",
            "eval_run_id": "eval-123",
            "assessment_counts": {"good": 6, "bad": 5, "needs_review": 4},
        },
    )
    artifacts.write_json_path(
        artifacts.optimization_summary_path,
        {
            "mode": "autonomous",
            "strategy": "centaur",
            "accepted": 1,
            "rejected": 1,
            "skipped": 0,
            "stop_reason": "candidate_budget_reached",
            "sweeps_completed": 1,
            "attempted_candidates": 2,
            "passes": [
                {
                    "name": "literal_examples",
                    "family": "examples",
                    "outcome": "accepted",
                    "train_rate": 50.0,
                    "validation_rate": 46.67,
                    "reason": "no benchmark regression",
                },
                {
                    "name": "guardrails",
                    "family": "instructions",
                    "outcome": "rejected",
                    "train_rate": 40.0,
                    "validation_rate": 40.0,
                    "reason": "validation regressed",
                },
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.latest_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [],
            "summary": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "pass_rate": 0.0},
        },
    )
    artifacts.write_json_path(
        artifacts.latest_full_summary_path,
        {"overall": {"total": 15, "passed": 7, "failed": 8, "errors": 0, "pass_rate": 46.67}},
    )
    artifacts.write_json_path(
        artifacts.curation_summary_path,
        {
            "changed": True,
            "table_descriptions_added": 1,
            "column_descriptions_added": 2,
            "columns_hidden": 1,
            "format_assistance_enabled": 1,
            "entity_matching_enabled": 1,
            "synonyms_added": 1,
            "changes": [],
        },
    )
    artifacts.write_json_path(
        artifacts.query_history_summary_path,
        {
            "status": "available",
            "warehouse_id": "wh-123",
            "query_count": 5,
            "successful_query_count": 4,
            "failed_query_count": 1,
            "feedback_thumbs_up": 2,
            "feedback_thumbs_down": 1,
            "recurring_success_patterns": [],
            "recurring_failure_patterns": [],
        },
    )
    artifacts.write_json_path(
        artifacts.query_history_recommendations_path,
        {
            "status": "available",
            "message": "Derived conservative recommendations from repeated query-history families.",
            "recommended_trusted_asset_candidates": [
                {
                    "recommendation_type": "trusted_asset_candidate",
                    "rationale": "Promote repeated successful patterns carefully.",
                    "query_count": 3,
                    "sample_sql": "SELECT 1",
                    "last_seen_at": "2026-03-28T12:00:00Z",
                }
            ],
            "recommended_benchmark_candidates": [
                {
                    "recommendation_type": "benchmark_candidate",
                    "rationale": "Capture repeated failures as benchmarks.",
                    "query_count": 2,
                    "sample_sql": "SELECT 2",
                    "last_seen_at": "2026-03-28T12:05:00Z",
                }
            ],
            "noisy_one_off_queries": 4,
            "repeated_query_families": 2,
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    report_path = flow.write_report()
    report = report_path.read_text(encoding="utf-8")

    assert "## Automatic Curation" in report
    assert "## Initial To Final" in report
    assert "Observed benchmark eval run: `eval-123`" in report
    assert "Full-score delta: `+6.67%`" in report
    assert "Split-rate deltas:" in report
    assert "## What Changed" in report
    assert "`literal_examples`" in report
    assert "## Query History Signals" in report
    assert "### Trusted-Asset Candidates" in report
    assert "### Benchmark Candidates From Query History" in report


def test_status_refreshes_stale_full_summary_from_newer_component_artifacts(tmp_path: Path):
    space_id = "c" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    decompose_config(_space_config(), artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.latest_full_summary_path,
        {"overall": {"pass_rate": 12.0}},
    )
    artifacts.write_json_path(
        artifacts.latest_train_path,
        {
            "space_id": space_id,
            "started_at": "2026-03-28T10:00:00",
            "completed_at": "2026-03-28T10:01:00",
            "results": [
                {
                    "question_id": "a" * 32,
                    "question": "Train question",
                    "status": "passed",
                    "generated_sql": "SELECT 1",
                    "expected_sql": "SELECT 1",
                    "comparison_score": 1.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 1.0,
                }
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.latest_dev_summary_path,
        {
            "question_count": 2,
            "passed": 2,
            "failed": 0,
            "errors": 0,
            "pass_rate": 100.0,
            "validation_strategy": "small_sample_kfold",
            "fold_count": 2,
            "mean_fold_pass_rate": 100.0,
            "min_fold_pass_rate": 100.0,
        },
    )
    artifacts.write_json_path(
        artifacts.test_set_path,
        {"question_count": 1, "ids": ["b" * 32]},
    )
    artifacts.write_text_path(artifacts.latest_test_score_path, "0.0\n")

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    status = flow.status()

    assert status.latest_full_pass_rate == 66.67


# ---------------------------------------------------------------------------
# curate_and_gate tests
# ---------------------------------------------------------------------------


def _gate_flow(
    tmp_path: Path,
    *,
    space_id: str,
    benchmark_service,
    has_dev_set: bool = True,
) -> tuple["OptimizationWorkspace", ArtifactManager]:
    """Build a minimal workspace suitable for curate_and_gate tests."""
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)

    # Space config with columns so curation passes have something to change.
    config = _space_config()
    config.data_sources.tables[0].column_configs = [
        GenieColumnConfig(column_name="brand_name"),
        GenieColumnConfig(column_name="az_brand_id"),
    ]
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)

    # Benchmark payload (both questions visible to clone).
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 2,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": "b" * 32, "question": "Dev question", "expected_sql": "SELECT 2"},
            ],
        },
    )
    # Train set contains one question.
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "a" * 32, "question": "Train question", "expected_sql": "SELECT 1"},
            ],
        },
    )
    # Split meta (required by _load_split_meta but not used in gate logic).
    artifacts.write_json_path(
        artifacts.split_meta_path,
        {"source_space_id": space_id, "workspace_host": "https://host.example.com"},
    )

    if has_dev_set:
        artifacts.write_json_path(
            artifacts.dev_set_path,
            {"question_count": 1, "ids": ["b" * 32]},
        )
        # Seed a baseline dev summary so reference_dev_rate is available.
        artifacts.write_json_path(
            artifacts.latest_dev_summary_path,
            {
                "question_count": 1,
                "passed": 1,
                "failed": 0,
                "errors": 0,
                "pass_rate": 100.0,
                "validation_strategy": "blind_split",
            },
        )

    remote_config = _space_config()
    remote_config.benchmarks = None
    genie_service = _DummyGenieService(remote_config)
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )
    return flow, artifacts


def test_curate_and_gate_rejects_when_validation_does_not_improve(tmp_path: Path):
    space_id = "c" * 32
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_PassingRunBenchmarkService(),
    )

    # descriptions mode should produce changes (column_configs present).
    result = flow.curate_and_gate(modes=["descriptions"], label="gate_test")

    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert result["passes"][0]["outcome"] == "rejected"
    assert "validation did not improve" in result["passes"][0]["reason"]
    # curation_summary.json should record the loop result.
    meta = artifacts.read_json_path(artifacts.curation_summary_path)
    assert meta["rejected"] == 1


def test_candidate_accepts_train_lift_when_validation_ties(tmp_path: Path):
    space_id = "c1" * 16
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_PassingRunBenchmarkService(),
    )
    baseline_payload = {
        "space_id": space_id,
        "started_at": None,
        "completed_at": None,
        "results": [
            {
                "question_id": "a" * 32,
                "question": "Train question",
                "status": "failed",
                "expected_sql": "SELECT 1",
                "generated_sql": "SELECT 0",
                "comparison_score": 0.0,
                "comparison_method": "sql_similarity",
                "comparison_details": None,
                "error_message": None,
                "duration_seconds": 0.1,
            }
        ],
        "summary": {"total": 1, "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_payload)
    artifacts.write_json_path(artifacts.latest_train_path, baseline_payload)
    artifacts.write_text_path(artifacts.baseline_validation_score_path, "100.0\n")
    artifacts.write_text_path(artifacts.latest_validation_score_path, "100.0\n")
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "pass_rate": 100.0,
            "validation_strategy": "blind_split",
        },
    )

    result, train_rate, validation_rate = flow._run_candidate_spec(
        spec=CandidateSpec(
            name="descriptions",
            family="curation",
            description="Description curation",
        ),
        label="validation_tie",
        reference_train_rate=0.0,
        reference_validation_rate=100.0,
        gating_active=True,
    )

    assert result["outcome"] == "accepted"
    assert result["acceptance_class"] == "train_lift_validation_tie"
    assert train_rate == 100.0
    assert validation_rate == 100.0


def test_candidate_rejects_train_lift_validation_tie_when_proxy_family_degrades(tmp_path: Path):
    space_id = "c2" * 16
    train_id = "a" * 32
    guard_id = "b" * 32
    alternate_id = "c" * 32

    class _ProxyRegressionBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            for question in questions:
                passed = question.id in {train_id, alternate_id}
                run.results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        status=BenchmarkStatus.PASSED if passed else BenchmarkStatus.FAILED,
                        expected_sql=question.expected_sql,
                        generated_sql=question.expected_sql if passed else "SELECT 99",
                        comparison_score=1.0 if passed else 0.0,
                    )
                )
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_ProxyRegressionBenchmarkService(),
    )
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 3,
            "questions": [
                {
                    "id": train_id,
                    "question": "NRx for Farxiga current 13 weeks",
                    "expected_sql": "SELECT 1",
                },
                {
                    "id": guard_id,
                    "question": "Percentage difference for Farxiga current vs previous 13 weeks for NRx",
                    "expected_sql": "SELECT 2",
                },
                {
                    "id": alternate_id,
                    "question": "Show Farxiga sales by region",
                    "expected_sql": "SELECT 3",
                },
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 1,
            "questions": [
                {
                    "id": train_id,
                    "question": "NRx for Farxiga current 13 weeks",
                    "expected_sql": "SELECT 1",
                }
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.validation_set_path,
        {"question_count": 2, "ids": [guard_id, alternate_id]},
    )
    baseline_payload = {
        "space_id": space_id,
        "started_at": None,
        "completed_at": None,
        "results": [
            {
                "question_id": train_id,
                "question": "NRx for Farxiga current 13 weeks",
                "status": "failed",
                "expected_sql": "SELECT 1",
                "generated_sql": "SELECT 0",
                "comparison_score": 0.0,
                "comparison_method": "sql_similarity",
                "comparison_details": None,
                "error_message": None,
                "duration_seconds": 0.1,
            }
        ],
        "summary": {"total": 1, "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_payload)
    artifacts.write_json_path(artifacts.latest_train_path, baseline_payload)
    artifacts.write_text_path(artifacts.baseline_validation_score_path, "50.0\n")
    artifacts.write_text_path(artifacts.latest_validation_score_path, "50.0\n")
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 2,
            "passed": 1,
            "failed": 1,
            "errors": 0,
            "pass_rate": 50.0,
            "failed_ids": [alternate_id],
            "error_ids": [],
            "validation_strategy": "blind_split",
        },
    )

    result, train_rate, validation_rate = flow._run_candidate_spec(
        spec=CandidateSpec(
            name="descriptions",
            family="curation",
            description="Description curation",
        ),
        label="proxy_regression",
        reference_train_rate=0.0,
        reference_validation_rate=50.0,
        gating_active=True,
    )

    assert result["outcome"] == "rejected"
    assert result["rejection_class"] == "visible_proxy_regression"
    assert result["reason"] == "visible proxy guards degraded despite visible score lift"
    assert result["visible_proxy_pressure"]["regressed_guard_ids"] == (guard_id,)
    assert "period_comparison" in result["visible_proxy_pressure"]["degraded_families"]
    assert train_rate == 0.0
    assert validation_rate == 50.0


def test_curate_and_gate_skips_mode_with_no_changes(tmp_path: Path):
    space_id = "d" * 32
    # Use a space config with no columns -> synonyms mode won't change anything.
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    config = _space_config()
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.split_meta_path,
        {"source_space_id": space_id, "workspace_host": "https://host.example.com"},
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {"question_count": 0, "questions": []},
    )

    remote_config = _space_config()
    remote_config.benchmarks = None
    genie_service = _DummyGenieService(remote_config)
    benchmark_service = _PassingRunBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    # synonyms mode won't add synonyms to a space with no column_configs.
    result = flow.curate_and_gate(modes=["synonyms"], label="gate_skip_test")

    assert result["accepted"] == 0
    assert result["rejected"] == 0
    assert result["skipped"] == 1
    assert result["passes"][0]["outcome"] == "skipped"
    assert result["passes"][0]["reason"] == "no_changes"


def test_curate_and_gate_rejects_on_train_regression(tmp_path: Path):
    space_id = "e" * 32

    class _FailingBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(self, *, space_id, questions, delay_between_questions, fail_fast,
                           match_threshold, cache, config_hash, cache_save_path=None, result_comparator=None, **kwargs) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            run.results = [
                BenchmarkResult(
                    question_id=q.id,
                    question=q.question,
                    status=BenchmarkStatus.FAILED,
                    expected_sql=q.expected_sql,
                    generated_sql="SELECT 99",
                    comparison_score=0.0,
                )
                for q in questions
            ]
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_FailingBenchmarkService(),
    )
    # Write a baseline train run with 100% pass rate so regression is detectable.
    artifacts.write_json_path(
        artifacts.baseline_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [
                {
                    "question_id": "a" * 32,
                    "question": "Train question",
                    "status": "passed",
                    "expected_sql": "SELECT 1",
                    "generated_sql": "SELECT 1",
                    "comparison_score": 1.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 0.1,
                }
            ],
            "summary": {"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 100.0},
        },
    )

    result = flow.curate_and_gate(modes=["descriptions"], label="gate_regression_test")

    assert result["rejected"] == 1
    assert result["accepted"] == 0
    assert result["passes"][0]["outcome"] == "rejected"
    reason = result["passes"][0]["reason"]
    assert "train regressed" in reason or reason.startswith("train_health_failed")


def test_curate_and_gate_rejects_when_pass_guard_regresses_even_if_train_rate_ties(tmp_path: Path):
    space_id = "e2" * 16
    guard_id = "a" * 32
    failure_id = "c" * 32

    class _SwappingBenchmarkService(_DummyBenchmarkService):
        def __init__(self):
            super().__init__()
            self.validation_calls = 0

        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            for question in questions:
                if question.id == guard_id:
                    status = BenchmarkStatus.FAILED
                    generated_sql = "SELECT broken_guard"
                    score = 0.0
                else:
                    status = BenchmarkStatus.PASSED
                    generated_sql = question.expected_sql
                    score = 1.0
                run.results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        status=status,
                        expected_sql=question.expected_sql,
                        generated_sql=generated_sql,
                        comparison_score=score,
                    )
                )
            return run

    benchmark_service = _SwappingBenchmarkService()
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=benchmark_service,
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 2,
            "questions": [
                {"id": guard_id, "question": "Guard question", "expected_sql": "SELECT 1"},
                {"id": failure_id, "question": "Failure question", "expected_sql": "SELECT 2"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.baseline_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [
                {
                    "question_id": guard_id,
                    "question": "Guard question",
                    "status": "passed",
                    "expected_sql": "SELECT 1",
                    "generated_sql": "SELECT 1",
                    "comparison_score": 1.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 0.1,
                },
                {
                    "question_id": failure_id,
                    "question": "Failure question",
                    "status": "failed",
                    "expected_sql": "SELECT 2",
                    "generated_sql": "SELECT wrong",
                    "comparison_score": 0.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 0.1,
                },
            ],
            "summary": {"total": 2, "passed": 1, "failed": 1, "errors": 0, "pass_rate": 50.0},
        },
    )

    result = flow.curate_and_gate(modes=["descriptions"], label="gate_guard_regression_tie")

    assert result["rejected"] == 1
    assert result["accepted"] == 0
    assert result["passes"][0]["outcome"] == "rejected"
    assert result["passes"][0]["reason"] == f"pass guard regressed: {guard_id}"


def test_curate_and_gate_classifies_accepted_fix_regression(tmp_path: Path):
    space_id = "e3" * 16
    accepted_fix_id = "a" * 32
    remaining_failure_id = "c" * 32

    class _AcceptedFixRegressionBenchmarkService(_DummyBenchmarkService):
        def __init__(self):
            super().__init__()
            self.question_batches: list[list[str]] = []
            self.min_pass_rates: list[float | None] = []

        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            self.question_batches.append([question.id for question in questions])
            self.min_pass_rates.append(kwargs.get("min_pass_rate"))
            run = BenchmarkRun(space_id=space_id)
            for question in questions:
                passed = question.id == remaining_failure_id
                run.results.append(
                    BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        status=BenchmarkStatus.PASSED if passed else BenchmarkStatus.FAILED,
                        expected_sql=question.expected_sql,
                        generated_sql=question.expected_sql if passed else "SELECT regressed_fix",
                        comparison_score=1.0 if passed else 0.0,
                        error_message=None if passed else "Generated SQL did not match expected SQL",
                    )
                )
            return run

    benchmark_service = _AcceptedFixRegressionBenchmarkService()
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=benchmark_service,
    )
    artifacts.write_json_path(
        artifacts.train_set_path,
        {
            "question_count": 2,
            "questions": [
                {"id": accepted_fix_id, "question": "Accepted fix guard", "expected_sql": "SELECT 1"},
                {"id": remaining_failure_id, "question": "Remaining failure", "expected_sql": "SELECT 2"},
            ],
        },
    )
    baseline_failed_result = {
        "question": "Accepted fix guard",
        "status": "failed",
        "expected_sql": "SELECT 1",
        "generated_sql": "SELECT wrong",
        "comparison_score": 0.0,
        "comparison_method": "sql_similarity",
        "comparison_details": None,
        "error_message": "Generated SQL did not match expected SQL",
        "duration_seconds": 0.1,
    }
    artifacts.write_json_path(
        artifacts.baseline_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [
                {"question_id": accepted_fix_id, **baseline_failed_result},
                {
                    "question_id": remaining_failure_id,
                    "question": "Remaining failure",
                    "status": "failed",
                    "expected_sql": "SELECT 2",
                    "generated_sql": "SELECT wrong",
                    "comparison_score": 0.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": "Generated SQL did not match expected SQL",
                    "duration_seconds": 0.1,
                },
            ],
            "summary": {"total": 2, "passed": 0, "failed": 2, "errors": 0, "pass_rate": 0.0},
        },
    )
    artifacts.write_json_path(
        artifacts.latest_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [
                {
                    "question_id": accepted_fix_id,
                    "question": "Accepted fix guard",
                    "status": "passed",
                    "expected_sql": "SELECT 1",
                    "generated_sql": "SELECT 1",
                    "comparison_score": 1.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 0.1,
                },
                {
                    "question_id": remaining_failure_id,
                    "question": "Remaining failure",
                    "status": "failed",
                    "expected_sql": "SELECT 2",
                    "generated_sql": "SELECT wrong",
                    "comparison_score": 0.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": "Generated SQL did not match expected SQL",
                    "duration_seconds": 0.1,
                },
            ],
            "summary": {"total": 2, "passed": 1, "failed": 1, "errors": 0, "pass_rate": 50.0},
        },
    )

    result = flow.curate_and_gate(modes=["descriptions"], label="gate_accepted_fix_regression")

    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert result["rejected"] == 1
    assert result["passes"][0]["reason"] == f"accepted fix regressed: {accepted_fix_id}"
    assert result["passes"][0]["rejection_class"] == "accepted_fix_regression"
    assert result["passes"][0]["gate_details"]["accepted_fix_regressed_ids"] == [accepted_fix_id]
    assert benchmark_service.question_batches == [[remaining_failure_id], [accepted_fix_id]]
    assert benchmark_service.min_pass_rates == [None, 100.0]
    assert '"rejection_class": "accepted_fix_regression"' in iteration_log


def test_curate_and_gate_rejects_when_known_failures_do_not_improve(tmp_path: Path):
    space_id = "e1" * 16

    class _FailingBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            run.results = [
                BenchmarkResult(
                    question_id=q.id,
                    question=q.question,
                    status=BenchmarkStatus.FAILED,
                    expected_sql=q.expected_sql,
                    generated_sql="SELECT 99",
                    comparison_score=0.0,
                )
                for q in questions
            ]
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_FailingBenchmarkService(),
    )
    artifacts.write_json_path(
        artifacts.baseline_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [
                {
                    "question_id": "a" * 32,
                    "question": "Train question",
                    "status": "failed",
                    "expected_sql": "SELECT 1",
                    "generated_sql": "SELECT 99",
                    "comparison_score": 0.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": "Generated SQL did not match expected SQL",
                    "duration_seconds": 0.1,
                }
            ],
            "summary": {"total": 1, "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0},
        },
    )

    result = flow.curate_and_gate(modes=["descriptions"], label="gate_known_failures_test")

    assert result["rejected"] == 1
    assert result["accepted"] == 0
    assert result["passes"][0]["outcome"] == "rejected"
    assert "known failures did not improve" in result["passes"][0]["reason"]


def test_curate_and_gate_keeps_latest_train_artifact_on_rejection(tmp_path: Path):
    space_id = "ea" * 16

    class _FailingBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            run.results = [
                BenchmarkResult(
                    question_id=q.id,
                    question=q.question,
                    status=BenchmarkStatus.FAILED,
                    expected_sql=q.expected_sql,
                    generated_sql="SELECT 99",
                    comparison_score=0.0,
                )
                for q in questions
            ]
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_FailingBenchmarkService(),
    )
    baseline_payload = {
        "space_id": space_id,
        "started_at": None,
        "completed_at": None,
        "results": [
            {
                "question_id": "a" * 32,
                "question": "Train question",
                "status": "passed",
                "expected_sql": "SELECT 1",
                "generated_sql": "SELECT 1",
                "comparison_score": 1.0,
                "comparison_method": "sql_similarity",
                "comparison_details": None,
                "error_message": None,
                "duration_seconds": 0.1,
            }
        ],
        "summary": {"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 100.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_payload)
    artifacts.write_json_path(artifacts.latest_train_path, baseline_payload)

    result = flow.curate_and_gate(modes=["descriptions"], label="gate_latest_train_preserved")

    latest_train = artifacts.read_json_path(artifacts.latest_train_path)
    assert result["rejected"] == 1
    assert latest_train["summary"]["pass_rate"] == 100.0


def test_curate_and_gate_keeps_latest_validation_artifacts_on_rejection(tmp_path: Path):
    space_id = "eb" * 16

    class _ValidationRegressionBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            run.results = [
                BenchmarkResult(
                    question_id=q.id,
                    question=q.question,
                    status=BenchmarkStatus.PASSED if q.id == "a" * 32 else BenchmarkStatus.FAILED,
                    expected_sql=q.expected_sql,
                    generated_sql=q.expected_sql if q.id == "a" * 32 else "SELECT 99",
                    comparison_score=1.0 if q.id == "a" * 32 else 0.0,
                )
                for q in questions
            ]
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_ValidationRegressionBenchmarkService(),
    )
    baseline_payload = {
        "space_id": space_id,
        "started_at": None,
        "completed_at": None,
        "results": [
            {
                "question_id": "a" * 32,
                "question": "Train question",
                "status": "passed",
                "expected_sql": "SELECT 1",
                "generated_sql": "SELECT 1",
                "comparison_score": 1.0,
                "comparison_method": "sql_similarity",
                "comparison_details": None,
                "error_message": None,
                "duration_seconds": 0.1,
            }
        ],
        "summary": {"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 100.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_payload)
    artifacts.write_json_path(artifacts.latest_train_path, baseline_payload)
    artifacts.write_text_path(artifacts.latest_validation_score_path, "100.0\n")
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "pass_rate": 100.0,
            "validation_strategy": "blind_split",
        },
    )

    result = flow.curate_and_gate(
        modes=["descriptions"],
        label="gate_latest_validation_preserved",
    )

    latest_validation = artifacts.read_json_path(artifacts.latest_validation_summary_path)
    latest_validation_score = artifacts.read_score_path(artifacts.latest_validation_score_path)
    assert result["rejected"] == 1
    assert "validation did not improve" in result["passes"][0]["reason"]
    assert latest_validation["pass_rate"] == 100.0
    assert latest_validation_score == 100.0


def test_curate_and_gate_early_stops_kfold_validation_when_improvement_impossible(tmp_path: Path):
    space_id = "ec" * 16
    train_id = "a" * 32
    fold_ids = ["b" * 32, "c" * 32, "d" * 32]

    class _KfoldTimeoutBenchmarkService(_DummyBenchmarkService):
        def __init__(self):
            super().__init__()
            self.question_batches: list[list[str]] = []
            self.cache_save_every_values: list[int | None] = []

        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            cache_save_every=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            self.question_batches.append([question.id for question in questions])
            self.cache_save_every_values.append(cache_save_every)
            run = BenchmarkRun(space_id=space_id)
            for question in questions:
                if question.id == train_id:
                    run.results.append(
                        BenchmarkResult(
                            question_id=question.id,
                            question=question.question,
                            status=BenchmarkStatus.PASSED,
                            expected_sql=question.expected_sql,
                            generated_sql=question.expected_sql,
                            comparison_score=1.0,
                        )
                    )
                else:
                    run.results.append(
                        BenchmarkResult(
                            question_id=question.id,
                            question=question.question,
                            status=BenchmarkStatus.ERROR,
                            expected_sql=question.expected_sql,
                            error_message="Polling timeout exceeded (180 attempts)",
                        )
                    )
            return run

    benchmark_service = _KfoldTimeoutBenchmarkService()
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=benchmark_service,
    )
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {
            "question_count": 4,
            "questions": [
                {"id": train_id, "question": "Train question", "expected_sql": "SELECT 1"},
                {"id": fold_ids[0], "question": "Fold 1 question", "expected_sql": "SELECT 2"},
                {"id": fold_ids[1], "question": "Fold 2 question", "expected_sql": "SELECT 3"},
                {"id": fold_ids[2], "question": "Fold 3 question", "expected_sql": "SELECT 4"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.validation_set_path,
        {"question_count": 3, "ids": fold_ids},
    )
    artifacts.write_json_path(
        artifacts.validation_folds_path,
        {
            "validation_strategy": "small_sample_kfold",
            "visible_question_count": 3,
            "fold_count": 3,
            "folds": [
                {"fold_index": 1, "validation_ids": [fold_ids[0]]},
                {"fold_index": 2, "validation_ids": [fold_ids[1]]},
                {"fold_index": 3, "validation_ids": [fold_ids[2]]},
            ],
        },
    )
    baseline_payload = {
        "space_id": space_id,
        "started_at": None,
        "completed_at": None,
        "results": [
            {
                "question_id": train_id,
                "question": "Train question",
                "status": "passed",
                "expected_sql": "SELECT 1",
                "generated_sql": "SELECT 1",
                "comparison_score": 1.0,
                "comparison_method": "sql_similarity",
                "comparison_details": None,
                "error_message": None,
                "duration_seconds": 0.1,
            }
        ],
        "summary": {"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 100.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_payload)
    artifacts.write_json_path(artifacts.latest_train_path, baseline_payload)
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 3,
            "passed": 2,
            "failed": 1,
            "errors": 0,
            "pass_rate": 66.67,
            "validation_strategy": "small_sample_kfold",
        },
    )

    result = flow.curate_and_gate(modes=["descriptions"], label="gate_kfold_abort")

    validation_snapshot = next(
        artifacts.validation_run_history_dir.glob("*.gate_kfold_abort.descriptions.validation.json")
    )
    validation_summary = artifacts.read_json_path(validation_snapshot)
    iteration_log = artifacts.history_dir.joinpath("iteration_log.jsonl").read_text(encoding="utf-8")
    assert result["rejected"] == 1
    assert "validation improvement impossible" in result["passes"][0]["reason"]
    assert result["passes"][0]["rejection_class"] == "validation_improvement_impossible"
    assert benchmark_service.question_batches == [[train_id], [fold_ids[0]]]
    assert benchmark_service.cache_save_every_values == [1, 1]
    assert validation_summary["early_stop_reason"] == "validation_improvement_impossible"
    assert validation_summary["evaluated_question_count"] == 1
    assert validation_summary["skipped"] == 2
    assert validation_summary["skipped_ids"] == fold_ids[1:]
    assert validation_summary["issue_class_counts"] == {"timeout": 1}
    assert '"phase": "benchmark_validation_fold_early_stop"' in iteration_log
    assert '"phase": "benchmark_questions_start"' in iteration_log
    assert '"cache_save_every": 1' in iteration_log


def test_curate_and_gate_gating_inactive_when_no_baseline(tmp_path: Path):
    space_id = "f0" * 16
    # _gate_flow without has_dev_set=True and without writing a baseline train.
    flow, _ = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_PassingRunBenchmarkService(),
        has_dev_set=False,
    )
    # No baseline_train.json and no dev_summary → gating should be inactive.
    result = flow.curate_and_gate(modes=["descriptions"], label="gate_no_baseline_test")

    assert result["gating_active"] is False
    # Since gating is inactive, changed modes are auto-accepted.
    assert result["accepted"] + result["skipped"] == 1
    assert result["rejected"] == 0


# ---------------------------------------------------------------------------
# Report and status with gated-loop curation summary format
# ---------------------------------------------------------------------------


def _minimal_report_artifacts(artifacts: ArtifactManager, space_id: str) -> None:
    """Write the minimum set of artifacts required to produce a report."""
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    decompose_config(_space_config(), artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.latest_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [],
            "summary": {"total": 0, "passed": 0, "failed": 0, "errors": 0, "pass_rate": 0.0},
        },
    )


def _seed_optimizer_baseline(artifacts: ArtifactManager, space_id: str) -> None:
    artifacts.write_json_path(
        artifacts.baseline_train_path,
        {
            "space_id": space_id,
            "started_at": None,
            "completed_at": None,
            "results": [
                {
                    "question_id": "a" * 32,
                    "question": "Train question",
                    "status": "failed",
                    "expected_sql": "SELECT 1",
                    "generated_sql": "SELECT 0",
                    "comparison_score": 0.0,
                    "comparison_method": "sql_similarity",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 0.1,
                }
            ],
            "summary": {"total": 1, "passed": 0, "failed": 1, "errors": 0, "pass_rate": 0.0},
        },
    )
    artifacts.write_json_path(artifacts.latest_train_path, artifacts.read_json_path(artifacts.baseline_train_path))
    artifacts.write_text_path(artifacts.baseline_validation_score_path, "0.0\n")
    artifacts.write_text_path(artifacts.latest_validation_score_path, "0.0\n")
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 1,
            "passed": 0,
            "failed": 1,
            "errors": 0,
            "pass_rate": 0.0,
            "validation_strategy": "blind_split",
        },
    )
    artifacts.write_text_path(artifacts.baseline_test_score_path, "0.0\n")
    artifacts.write_json_path(
        artifacts.latest_full_summary_path,
        {
            "overall": {
                "question_count": 3,
                "passed": 0,
                "failed": 3,
                "errors": 0,
                "pass_rate": 0.0,
            }
        },
    )
    artifacts.write_json_path(
        artifacts.hidden_test_payload_path,
        {
            "question_count": 1,
            "questions": [
                {"id": "c" * 32, "question": "Held-out test question", "expected_sql": "SELECT 3"},
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.test_set_path,
        {"question_count": 1, "ids": ["c" * 32]},
    )


def test_run_autonomous_optimization_supports_centaur_strategy(tmp_path: Path):
    space_id = "b0" * 16
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_AllPassingBenchmarkService(),
    )
    config = assemble_config(artifacts.space_dir)
    config.data_sources.tables[0].column_configs = [
        GenieColumnConfig(column_name="brand_name"),
        GenieColumnConfig(column_name="territory_group"),
        GenieColumnConfig(column_name="wk_grp"),
        GenieColumnConfig(column_name="mth_grp"),
    ]
    config.instructions = GenieInstructions(
        text_instructions=[
            GenieInstruction(
                content=[
                    "Use LIKE operator for brand matching.",
                    "Prefer case-insensitive exact matching for brand filters.",
                    "Map tb_wk_id and tb_mth_id carefully.",
                    "current_6_months_c6m should rely on mth_grp_desc.",
                ]
            )
        ],
        sql_snippets=GenieSQLSnippets(
            filters=[
                GenieSQLSnippet(
                    id="deadbeefdeadbeefdeadbeefdeadbeef",
                    display_name="Legacy Target",
                    sql=["tb_mth_id = 'C6M' AND target_indicator = 'Y'"],
                )
            ]
        ),
    )
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    _seed_optimizer_baseline(artifacts, space_id)

    summary = flow.run_autonomous_optimization(
        label="centaur",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=3,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )

    persisted = artifacts.read_json_path(artifacts.optimization_summary_path)
    assert summary["strategy"] == AUTONOMOUS_STRATEGY_CENTAUR
    assert persisted["strategy"] == AUTONOMOUS_STRATEGY_CENTAUR
    assert [entry["family"] for entry in persisted["passes"]] == [
        "centaur_prepass",
        "centaur_prepass",
        "centaur_prepass",
    ]
    assert [entry["name"] for entry in persisted["passes"]] == [
        "centaur_filter_snippet_audit",
        "centaur_instruction_audit",
        "centaur_expanded_entity_matching",
    ]


def test_run_autonomous_optimization_skips_suppressed_candidates(tmp_path: Path):
    space_id = "b2" * 16

    class _FailingBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            run.results = [
                BenchmarkResult(
                    question_id=q.id,
                    question=q.question,
                    status=BenchmarkStatus.FAILED,
                    expected_sql=q.expected_sql,
                    generated_sql="SELECT 99",
                    comparison_score=0.0,
                )
                for q in questions
            ]
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_FailingBenchmarkService(),
    )
    _seed_optimizer_baseline(artifacts, space_id)

    first = flow.run_autonomous_optimization(
        label="centaur-first",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=1,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )
    second = flow.run_autonomous_optimization(
        label="centaur-second",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=1,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )

    suppressed = artifacts.read_json_path(artifacts.optimization_summary_path)["suppressed_candidates"]
    assert first["rejected"] == 1
    assert any(
        entry["reason_code"] == "known_failures_not_improved"
        for entry in suppressed.values()
    )
    assert second["skipped"] == 1
    assert second["passes"][0]["outcome"] == "skipped"
    assert second["passes"][0]["suppressed"] is True
    assert second["passes"][0]["suppression_reason_code"] == "known_failures_not_improved"


def test_run_candidate_spec_rejects_train_zero(tmp_path: Path):
    space_id = "b3" * 16

    class _ZeroTrainBenchmarkService(_DummyBenchmarkService):
        def run_benchmarks(
            self,
            *,
            space_id,
            questions,
            delay_between_questions,
            fail_fast,
            match_threshold,
            cache,
            config_hash,
            cache_save_path=None,
            result_comparator=None,
            **kwargs,
        ) -> BenchmarkRun:
            run = BenchmarkRun(space_id=space_id)
            run.results = [
                BenchmarkResult(
                    question_id=q.id,
                    question=q.question,
                    status=BenchmarkStatus.FAILED,
                    expected_sql=q.expected_sql,
                    generated_sql="SELECT 0",
                    comparison_score=0.0,
                )
                for q in questions
            ]
            return run

    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_ZeroTrainBenchmarkService(),
    )

    passing_result = {
        "question_id": "a" * 32,
        "question": "Train question",
        "status": "passed",
        "expected_sql": "SELECT 1",
        "generated_sql": "SELECT 1",
        "comparison_score": 1.0,
        "comparison_method": "sql_similarity",
        "comparison_details": None,
        "error_message": None,
        "duration_seconds": 0.1,
    }
    baseline_payload = {
        "space_id": space_id,
        "started_at": None,
        "completed_at": None,
        "results": [passing_result],
        "summary": {"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 100.0},
    }
    artifacts.write_json_path(artifacts.baseline_train_path, baseline_payload)
    artifacts.write_json_path(artifacts.latest_train_path, baseline_payload)
    artifacts.write_text_path(artifacts.baseline_validation_score_path, "100.0\n")
    artifacts.write_text_path(artifacts.latest_validation_score_path, "100.0\n")
    artifacts.write_json_path(
        artifacts.latest_validation_summary_path,
        {
            "question_count": 1,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "pass_rate": 100.0,
            "validation_strategy": "blind_split",
        },
    )
    artifacts.write_text_path(artifacts.baseline_test_score_path, "100.0\n")

    summary = flow.run_autonomous_optimization(
        label="centaur-health",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=1,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )

    persisted = artifacts.read_json_path(artifacts.optimization_summary_path)
    assert summary["rejected"] >= 1
    health_pass = next(
        (p for p in persisted["passes"] if str(p.get("reason") or "").startswith("train_health_failed")),
        None,
    )
    assert health_pass is not None, f"No train_health_failed pass found in {persisted['passes']}"
    assert health_pass["outcome"] == "rejected"
    suppressed = persisted.get("suppressed_candidates", {})
    health_entry = next(
        (v for v in suppressed.values() if v.get("reason_code") == "train_health_failed"),
        None,
    )
    assert health_entry is not None, f"No train_health_failed suppression entry found in {suppressed}"


def test_run_autonomous_optimization_records_evidence_protocol_and_rerun_plan(tmp_path: Path):
    space_id = "c1" * 16
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_AllPassingBenchmarkService(),
    )
    _seed_optimizer_baseline(artifacts, space_id)

    summary = flow.run_autonomous_optimization(
        label="centaur-evidence",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=1,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )

    persisted = artifacts.read_json_path(artifacts.optimization_summary_path)
    assert summary["baseline_evaluation_context"]["preferred_comparison_mode"] == "string_only"
    assert summary["final_evaluation_context"]["preferred_comparison_mode"] == "string_only"
    assert persisted["evidence_protocol"]["strategy_under_test"] == AUTONOMOUS_STRATEGY_CENTAUR
    assert persisted["evidence_protocol"]["preferred_comparison_mode"] == "string_only"
    assert persisted["rerun_plan"]["recommended_first_rerun"] == AUTONOMOUS_STRATEGY_CENTAUR
    assert persisted["rerun_plan"]["follow_up_rerun"] == "full_strategy_evaluation"


def test_write_report_renders_benchmark_evidence_protocol_and_rerun_plan(tmp_path: Path):
    space_id = "e1" * 16
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    _minimal_report_artifacts(artifacts, space_id)
    artifacts.update_workspace_meta(
        {
            "product_commit_sha": "commit-123",
            "product_branch": "main",
            "bootstrap_only": False,
            "autonomous_strategy": AUTONOMOUS_STRATEGY_CENTAUR,
            "serving_candidate_mode": "artifact_patch_v2",
            "fresh_baseline": True,
            "observed_benchmark_reuse_policy": "fresh_baseline_benchmark_run",
        }
    )
    artifacts.write_json_path(
        artifacts.benchmark_payload_path,
        {"question_count": 15, "questions": [{"id": str(index)} for index in range(15)]},
    )
    artifacts.write_json_path(
        artifacts.hidden_test_payload_path,
        {"question_count": 3},
    )
    artifacts.write_json_path(
        artifacts.split_meta_path,
        {
            "source_space_id": space_id,
            "warehouse_id": "wh-123",
            "split_strategy": "baseline_outcome_balanced",
            "validation_strategy": "small_sample_kfold",
            "split_seed": 42,
        },
    )
    artifacts.write_json_path(artifacts.train_set_path, {"question_count": 9})
    artifacts.write_json_path(artifacts.validation_set_path, {"question_count": 3})
    artifacts.write_json_path(artifacts.test_set_path, {"question_count": 3})
    artifacts.write_json_path(
        artifacts.latest_full_summary_path,
        {
            "overall": {
                "total": 3,
                "passed": 2,
                "failed": 1,
                "errors": 0,
                "pass_rate": 66.67,
                "comparison_metrics": {
                    "scored_question_count": 3,
                    "average_comparison_score": 0.8333,
                    "result_based_question_count": 2,
                    "string_based_question_count": 1,
                    "comparison_method_counts": {
                        "result_exact": 2,
                        "string_fallback_unbound_parameters": 1,
                    },
                    "comparison_mode": "mixed",
                },
            },
            "evaluation_context": {
                "preferred_comparison_mode": "result_based",
                "comparison_fingerprint": "warehouse=wh-123|result-v1",
                "warehouse_id": "wh-123",
                "match_threshold": 0.85,
            },
        },
    )
    artifacts.write_json_path(
        artifacts.protocol_preflight_path,
        {
            "status": "non_comparable",
            "promotion_eligibility": {
                "promotion_eligible": False,
            },
            "errors": [
                {
                    "code": "incomplete_result_based_benchmark_coverage",
                }
            ],
            "warnings": [],
        },
    )
    artifacts.write_json_path(
        artifacts.optimization_summary_path,
        {
            "mode": "autonomous",
            "strategy": AUTONOMOUS_STRATEGY_CENTAUR,
            "accepted": 1,
            "rejected": 0,
            "skipped": 0,
            "stop_reason": "validation_plateau",
            "sweeps_completed": 1,
            "attempted_candidates": 1,
            "evidence_protocol": {
                "strategy_under_test": AUTONOMOUS_STRATEGY_CENTAUR,
                "decision_rule": "Treat optimizer changes as improvements only after rerun evidence shows no regression against the previous benchmark baseline.",
                "preferred_comparison_mode": "result_based",
                "required_local_validation": ["targeted_unit_tests", "compile_checks"],
                "required_reruns": [
                    "rerun_centaur_after_material_changes",
                    "rerun_full_strategy_evaluation_after_promising_batches",
                ],
                "required_comparison_fields": [
                    "train_pass_rate",
                    "validation_pass_rate",
                    "test_pass_rate",
                    "full_pass_rate",
                    "accepted_candidates",
                    "attempted_candidates",
                    "comparison_mode",
                ],
            },
            "rerun_plan": {
                "recommended_first_rerun": AUTONOMOUS_STRATEGY_CENTAUR,
                "first_rerun_reason": "Centaur is the default evidence checkpoint after material optimizer changes because it is faster than a full multi-strategy evaluation while still measuring the default product direction.",
                "follow_up_rerun": "full_strategy_evaluation",
                "follow_up_condition": "Run the full strategy evaluation after a promising Centaur improvement or when a change affects comparative strategy claims.",
                "compare_against": [
                    "baseline_artifacts",
                    "previous_centaur_run",
                    "latest_strategy_evaluation_when_available",
                ],
                "required_artifacts": [
                    "results/latest_full_summary.json",
                    "results/optimization_summary.json",
                    "history/iteration_log.jsonl",
                ],
            },
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    report = flow.write_report().read_text(encoding="utf-8")
    manifest = artifacts.read_json_path(artifacts.comparable_run_manifest_path)

    assert "## Benchmark Evaluation" in report
    assert "Actual full comparison mode: `mixed`" in report
    assert "Preferred comparison mode: `result_based`" in report
    assert "Expected benchmark questions: `15`" in report
    assert "Benchmark coverage: `3/15` scored; `2/15` result-based" in report
    assert "Methods observed: `result_exact=2, string_fallback_unbound_parameters=1`" in report
    assert "String fallback methods: `string_fallback_unbound_parameters=1`" in report
    assert "do not count as result-based leaderboard coverage" in report
    assert "## Protocol Eligibility" in report
    assert "Evidence classification: `diagnostic_non_comparable`" in report
    assert "Blocking reasons: `incomplete_result_based_benchmark_coverage`" in report
    assert "## Evidence Protocol" in report
    assert "Recommended first rerun: `centaur`" in report
    assert "Follow-up rerun: `full_strategy_evaluation`" in report
    assert "Required local validation: `targeted_unit_tests, compile_checks`" in report
    assert manifest["schema_version"] == "maxgenie.comparable_run_manifest.v1"
    assert manifest["source_space_id"] == space_id
    assert manifest["clone_space_id"] == "clone-123"
    assert manifest["product_commit_sha"] == "commit-123"
    assert manifest["candidate_mode"] == "artifact_patch_v2"
    assert manifest["warehouse_id"] == "wh-123"
    assert manifest["evaluator_mode"] == "mixed"
    assert manifest["split_counts"] == {"train": 9, "validation": 3, "holdout": 3}
    assert manifest["evaluation_counts"]["expected"] == 15
    assert manifest["evaluation_counts"]["scored"] == 3
    assert manifest["evaluation_counts"]["result_based"] == 2
    assert manifest["benchmark_payload_hash"]
    assert manifest["split_metadata_hash"]


def test_write_report_renders_gated_loop_curation_summary(tmp_path: Path):
    space_id = "f" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    _minimal_report_artifacts(artifacts, space_id)
    artifacts.write_json_path(
        artifacts.curation_summary_path,
        {
            "label": "curate_loop.20260329T120000",
            "modes_tried": ["descriptions", "hide_noise", "synonyms"],
            "accepted": 1,
            "rejected": 1,
            "skipped": 1,
            "passes": [
                {
                    "mode": "descriptions",
                    "outcome": "accepted",
                    "train_rate": 100.0,
                    "dev_rate": 100.0,
                    "candidate_summary": {
                        "candidate_name": "literal_filter_file",
                        "candidate_mode": "artifact_patch_v2",
                        "risk_assessment": {
                            "visible_effect_verification": {
                                "causal_patch_plan": {
                                    "selected_marker": "target_train_failure:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                    "expected_visible_effect": "literalize_expected_value: generated SQL uses 'farxiga'",
                                    "available_sql_delta_intents": [
                                        "literalize_expected_value",
                                        "remove_bind_placeholder",
                                    ],
                                    "selected_sql_delta_intents": [
                                        "literalize_expected_value"
                                    ],
                                }
                            }
                        },
                    },
                },
                {
                    "mode": "hide_noise",
                    "outcome": "rejected",
                    "reason": "train regressed 100.0% -> 88.9%",
                    "train_rate": 88.9,
                    "dev_rate": None,
                },
                {
                    "mode": "synonyms",
                    "outcome": "skipped",
                    "reason": "no_changes",
                },
            ],
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    report_path = flow.write_report()
    report = report_path.read_text(encoding="utf-8")

    assert "## Benchmark-Gated Curation Loop" in report
    assert "descriptions" in report
    assert "hide_noise" in report
    assert "ACCEPTED" in report
    assert "REJECTED" in report
    assert "SKIPPED" in report
    assert "### Artifact Patch Intent Audit" in report
    assert "literal_filter_file" in report
    assert "selected intents `literalize_expected_value`" in report
    assert "available intents `literalize_expected_value, remove_bind_placeholder`" in report
    assert "Expected visible effect: literalize_expected_value" in report
    assert "## Automatic Curation" not in report  # old format must not appear


def test_write_report_single_mode_curation_summary_still_renders(tmp_path: Path):
    space_id = "9" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    _minimal_report_artifacts(artifacts, space_id)
    artifacts.write_json_path(
        artifacts.curation_summary_path,
        {
            "mode": "safe_defaults",
            "candidate_status": "prepared",
            "changed": True,
            "table_descriptions_added": 1,
            "column_descriptions_added": 2,
            "columns_hidden": 0,
            "format_assistance_enabled": 0,
            "entity_matching_enabled": 0,
            "synonyms_added": 0,
            "changes": [],
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    report_path = flow.write_report()
    report = report_path.read_text(encoding="utf-8")

    assert "## Automatic Curation" in report
    assert "safe_defaults" in report
    assert "## Benchmark-Gated Curation Loop" not in report


def test_write_report_advisory_no_benchmark_suppresses_stale_scores(tmp_path: Path):
    space_id = "9" * 32
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    _minimal_report_artifacts(artifacts, space_id)
    artifacts.update_workspace_meta(
        {
            "no_benchmark_advisory": True,
            "evidence_classification": "advisory_best_practices",
            "benchmark_verification_status": "not_benchmark_verified",
            "no_benchmark_warning": NO_BENCHMARK_WARNING,
        }
    )
    artifacts.write_json_path(
        artifacts.latest_full_summary_path,
        {"overall": {"total": 10, "passed": 9, "pass_rate": 90.0}},
    )
    artifacts.write_json_path(
        artifacts.baseline_full_summary_path,
        {"overall": {"total": 10, "passed": 5, "pass_rate": 50.0}},
    )
    artifacts.write_json_path(
        artifacts.advisory_best_practices_path,
        {
            "mode": "advisory_best_practices",
            "evidence_classification": "advisory_best_practices",
            "benchmark_verification_status": "not_benchmark_verified",
            "benchmark_verification_label": "not benchmark-verified",
            "warning": NO_BENCHMARK_WARNING,
            "change_count": 1,
            "curation_mode": "safe_defaults",
            "suggested_next_step": "Add owner-approved benchmark questions.",
        },
    )
    artifacts.write_json_path(
        artifacts.curation_summary_path,
        {
            "mode": "safe_defaults",
            "candidate_status": "not benchmark-verified",
            "evidence_classification": "advisory_best_practices",
            "changed": True,
            "table_descriptions_added": 1,
            "column_descriptions_added": 0,
            "columns_hidden": 0,
            "format_assistance_enabled": 0,
            "entity_matching_enabled": 0,
            "synonyms_added": 0,
            "changes": [],
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    report = flow.write_report().read_text(encoding="utf-8")

    assert NO_BENCHMARK_WARNING in report
    assert "## No Benchmark Advisory" in report
    assert "Evidence classification: `advisory_best_practices`" in report
    assert "Benchmark verification: `not benchmark-verified`" in report
    assert "Full pass rate: `n/a (no benchmark)`" in report
    assert "Final full score: `n/a`" not in report
    assert "90.00%" not in report
    assert "50.00%" not in report
    assert "## Initial To Final" not in report
    assert "Evidence classification: `leaderboard`" not in report
    assert not artifacts.comparable_run_manifest_path.exists()


def test_status_reflects_latest_full_comparison_mode(tmp_path: Path):
    space_id = "a2" * 16
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    decompose_config(_space_config(), artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.latest_train_path,
        {
            "space_id": space_id,
            "started_at": "2026-03-28T10:00:00",
            "completed_at": "2026-03-28T10:01:00",
            "results": [
                {
                    "question_id": "a" * 32,
                    "question": "Train question",
                    "status": "passed",
                    "generated_sql": "SELECT 1",
                    "expected_sql": "SELECT 1",
                    "comparison_score": 1.0,
                    "comparison_method": "result_exact",
                    "comparison_details": None,
                    "error_message": None,
                    "duration_seconds": 1.0,
                }
            ],
        },
    )
    artifacts.write_json_path(
        artifacts.latest_full_summary_path,
        {
            "overall": {
                "pass_rate": 100.0,
                "comparison_metrics": {
                    "comparison_mode": "result_based",
                },
            }
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    status = flow.status()

    assert status.latest_full_pass_rate == 100.0
    assert status.latest_full_comparison_mode == "result_based"


def test_status_reflects_gated_loop_curation_summary(tmp_path: Path):
    space_id = "a1" * 16
    artifacts = ArtifactManager(tmp_path / space_id, space_id)
    artifacts.save_clone_meta(clone_space_id="clone-123", source_space_id=space_id)
    decompose_config(_space_config(), artifacts.space_dir, include_benchmarks=False)
    artifacts.write_json_path(
        artifacts.curation_summary_path,
        {
            "label": "curate_loop.test",
            "modes_tried": ["descriptions", "hide_noise"],
            "accepted": 2,
            "rejected": 0,
            "skipped": 0,
            "passes": [],
        },
    )

    genie_service = _DummyGenieService(_space_config())
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
    )

    ws = flow.status()

    assert ws.curation_candidate_status is not None
    assert "loop" in ws.curation_candidate_status
    assert "2 accepted" in ws.curation_candidate_status
    assert ws.curation_change_count == 2
    assert ws.hidden_column_count is None  # not tracked in gated format


class _RecordingMlflowTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def start_optimization_run(
        self,
        *,
        strategy: str,
        label: str,
        baseline_metrics: dict | None = None,
        params: dict | None = None,
    ) -> None:
        self.calls.append(
            (
                "start_optimization_run",
                {
                    "strategy": strategy,
                    "label": label,
                    "baseline_metrics": dict(baseline_metrics or {}),
                    "params": dict(params or {}),
                },
            )
        )

    def start_candidate_run(
        self,
        *,
        spec_name: str,
        family: str,
        sweep_index: int,
    ) -> None:
        self.calls.append(
            (
                "start_candidate_run",
                {"spec_name": spec_name, "family": family, "sweep_index": sweep_index},
            )
        )

    def log_candidate_result(self, result: dict) -> None:
        self.calls.append(("log_candidate_result", dict(result)))

    def end_candidate_run(self, status: str = "FINISHED") -> None:
        self.calls.append(("end_candidate_run", {"status": status}))

    def log_benchmark_summary(self, *, label: str, summary: dict) -> None:
        self.calls.append(("log_benchmark_summary", {"label": label}))

    def set_optimization_status(self, summary: dict) -> None:
        self.calls.append(("set_optimization_status", {}))

    def end_optimization_run(self, status: str = "FINISHED") -> None:
        self.calls.append(("end_optimization_run", {"status": status}))


def test_run_autonomous_optimization_mlflow_hooks_fire(tmp_path: Path):
    space_id = "cc" * 16
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_AllPassingBenchmarkService(),
    )
    config = assemble_config(artifacts.space_dir)
    config.data_sources.tables[0].column_configs = [
        GenieColumnConfig(column_name="brand_name"),
        GenieColumnConfig(column_name="territory_group"),
        GenieColumnConfig(column_name="wk_grp"),
        GenieColumnConfig(column_name="mth_grp"),
    ]
    config.instructions = GenieInstructions(
        text_instructions=[
            GenieInstruction(
                content=[
                    "Use LIKE operator for brand matching.",
                    "Prefer case-insensitive exact matching for brand filters.",
                ]
            )
        ],
    )
    decompose_config(config, artifacts.space_dir, include_benchmarks=False)
    _seed_optimizer_baseline(artifacts, space_id)

    tracker = _RecordingMlflowTracker()
    flow._injected_mlflow_tracker = tracker

    summary = flow.run_autonomous_optimization(
        label="centaur",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=3,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )

    assert summary["strategy"] == AUTONOMOUS_STRATEGY_CENTAUR
    method_names = [name for name, _ in tracker.calls]

    assert method_names.count("start_optimization_run") == 1
    assert method_names.count("end_optimization_run") == 1
    assert method_names.count("set_optimization_status") == 1

    start_calls = [args for name, args in tracker.calls if name == "start_candidate_run"]
    end_calls = [args for name, args in tracker.calls if name == "end_candidate_run"]
    result_calls = [args for name, args in tracker.calls if name == "log_candidate_result"]
    assert len(start_calls) == 3
    assert len(end_calls) == 3
    assert len(result_calls) == 3

    start_args = next(args for name, args in tracker.calls if name == "start_optimization_run")
    assert start_args["strategy"] == AUTONOMOUS_STRATEGY_CENTAUR
    assert start_args["label"] == "centaur"
    assert start_args["params"]["max_iterations"] == 1
    assert start_args["params"]["plateau_rounds"] == 1

    assert all(call["sweep_index"] == 1 for call in start_calls)
    spec_names = [call["spec_name"] for call in start_calls]
    assert len(spec_names) == 3
    assert all(isinstance(name, str) and name for name in spec_names)

    for result_args in result_calls:
        assert "outcome" in result_args

    assert method_names[-1] == "end_optimization_run"


def test_run_autonomous_optimization_defaults_to_noop_tracker(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MAXGENIE_MLFLOW_ENABLED", raising=False)
    space_id = "cd" * 16
    flow, artifacts = _gate_flow(
        tmp_path,
        space_id=space_id,
        benchmark_service=_AllPassingBenchmarkService(),
    )
    _seed_optimizer_baseline(artifacts, space_id)

    summary = flow.run_autonomous_optimization(
        label="centaur",
        plateau_rounds=1,
        max_iterations=1,
        candidate_budget=1,
        strategy=AUTONOMOUS_STRATEGY_CENTAUR,
    )
    assert summary["strategy"] == AUTONOMOUS_STRATEGY_CENTAUR


# ---------------------------------------------------------------------------
# parameter_values wiring in OptimizationWorkspace
# ---------------------------------------------------------------------------

def test_workspace_flow_passes_parameter_values_to_comparator(tmp_path: Path):
    """OptimizationWorkspace extracts default_value from example_sqls into comparator."""
    space_id = "ef" * 16
    artifacts = ArtifactManager(tmp_path / space_id, space_id)

    space_cfg = GenieSpaceConfig(
        title="Test Space",
        warehouse_id="wh-test",
        data_sources=GenieDataSources(
            tables=[GenieTableConfig(identifier="catalog.schema.t")]
        ),
        instructions=GenieInstructions(
            example_question_sqls=[
                GenieExampleSQL(
                    question=["How many brands?"],
                    sql=["SELECT * FROM t WHERE brand = :brand_name AND region = :region"],
                    parameters=[
                        GenieExampleSQLParameter(name="brand_name", default_value="Farxiga"),
                        GenieExampleSQLParameter(name="zip_code", default_value={"values": ["10001"]}),
                        GenieExampleSQLParameter(name="region", default_value=None),
                    ],
                ),
                GenieExampleSQL(
                    question=["All rows?"],
                    sql=["SELECT * FROM t"],
                    parameters=None,
                ),
            ],
        ),
    )

    genie_service = _DummyGenieService(space_cfg)
    benchmark_service = _DummyBenchmarkService()
    exporter = SpaceExporter(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
    )
    flow = OptimizationWorkspace(
        genie_service=genie_service,
        benchmark_service=benchmark_service,
        artifacts=artifacts,
        exporter=exporter,
        delay_seconds=0.0,
        match_threshold=0.85,
        warehouse_id="wh-test",
        space_config=space_cfg,
    )

    assert flow.comparator is not None
    assert flow.comparator._parameter_values == {
        "brand_name": "Farxiga",
        "zip_code": "10001",
    }
