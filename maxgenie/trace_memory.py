"""Typed trace memory artifacts for MaxGenie workspaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Iterable

from maxgenie.artifacts import ArtifactManager
from maxgenie.failure_context import build_structured_failure_context

TRACE_MEMORY_SCHEMA_VERSION = "maxgenie.trace_memory.v1"
FAILURE_CONTEXT_PACK_SCHEMA_VERSION = "maxgenie.failure_context_pack.v1"
CROSS_RUN_MEMORY_SCHEMA_VERSION = "maxgenie.cross_run_memory.v1"
HIDDEN_HOLDOUT_POLICY = "hidden final holdout content is intentionally omitted"
_PASS_STATUS = "passed"


@dataclass(frozen=True)
class ScoreSnapshot:
    split: str
    pass_rate: float | None
    passed: int | None = None
    total: int | None = None
    failed: int | None = None
    errors: int | None = None
    failed_ids: tuple[str, ...] = ()
    source_path: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "pass_rate": self.pass_rate,
            "passed": self.passed,
            "total": self.total,
            "failed": self.failed,
            "errors": self.errors,
            "failed_ids": list(self.failed_ids),
            "source_path": self.source_path,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class TraceQuestion:
    question_id: str
    split_membership: tuple[str, ...]
    question: str | None
    status: str | None
    score: float | None
    comparison_method: str | None
    issue_class: str | None
    root_cause_family: str | None
    expected_sql: str | None
    generated_sql: str | None
    missing_expected_tokens: tuple[str, ...]
    guard_regression_risk: str | None
    source_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "split_membership": list(self.split_membership),
            "question": self.question,
            "status": self.status,
            "score": self.score,
            "comparison_method": self.comparison_method,
            "issue_class": self.issue_class,
            "root_cause_family": self.root_cause_family,
            "expected_sql": self.expected_sql,
            "generated_sql": self.generated_sql,
            "missing_expected_tokens": list(self.missing_expected_tokens),
            "guard_regression_risk": self.guard_regression_risk,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class CandidateAttempt:
    attempt: int
    name: str | None
    label: str | None
    family: str | None
    outcome: str | None
    train_rate: float | None
    validation_rate: float | None
    reason: str | None
    candidate_mode: str | None
    skipped_validation: bool
    rejection_class: str | None
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "name": self.name,
            "label": self.label,
            "family": self.family,
            "outcome": self.outcome,
            "train_rate": self.train_rate,
            "validation_rate": self.validation_rate,
            "reason": self.reason,
            "candidate_mode": self.candidate_mode,
            "skipped_validation": self.skipped_validation,
            "rejection_class": self.rejection_class,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class GuardRegression:
    question_id: str
    candidate_name: str | None
    reason: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "candidate_name": self.candidate_name,
            "reason": self.reason,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class TraceMemoryIndex:
    schema_version: str
    generated_at: str
    workspace_dir: str
    space_id: str
    clone: dict[str, Any]
    scores: dict[str, dict[str, ScoreSnapshot]]
    questions: tuple[TraceQuestion, ...]
    candidate_attempts: tuple[CandidateAttempt, ...]
    guard_regressions: tuple[GuardRegression, ...]
    failure_families: dict[str, tuple[str, ...]]
    sources: dict[str, str | None]
    hidden_holdout_policy: str = HIDDEN_HOLDOUT_POLICY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "workspace_dir": self.workspace_dir,
            "space_id": self.space_id,
            "clone": self.clone,
            "scores": {
                group: {
                    split: score.to_dict()
                    for split, score in splits.items()
                }
                for group, splits in self.scores.items()
            },
            "questions": [question.to_dict() for question in self.questions],
            "candidate_attempts": [
                attempt.to_dict()
                for attempt in self.candidate_attempts
            ],
            "guard_regressions": [
                regression.to_dict()
                for regression in self.guard_regressions
            ],
            "failure_families": {
                family: list(question_ids)
                for family, question_ids in self.failure_families.items()
            },
            "sources": self.sources,
            "hidden_holdout_policy": self.hidden_holdout_policy,
        }


@dataclass(frozen=True)
class FailureContextPack:
    schema_version: str
    generated_at: str
    space_id: str
    scorecard: dict[str, Any]
    target_failures: tuple[dict[str, Any], ...]
    pass_guards: tuple[dict[str, Any], ...]
    guard_regressions: tuple[dict[str, Any], ...]
    candidate_lessons: tuple[dict[str, Any], ...]
    failure_families: dict[str, tuple[str, ...]]
    recommended_next_focus: str
    sources: dict[str, str | None]
    cross_run_memory: dict[str, Any] | None = None
    hidden_holdout_policy: str = HIDDEN_HOLDOUT_POLICY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "space_id": self.space_id,
            "scorecard": self.scorecard,
            "target_failures": list(self.target_failures),
            "pass_guards": list(self.pass_guards),
            "guard_regressions": list(self.guard_regressions),
            "candidate_lessons": list(self.candidate_lessons),
            "failure_families": {
                family: list(question_ids)
                for family, question_ids in self.failure_families.items()
            },
            "recommended_next_focus": self.recommended_next_focus,
            "sources": self.sources,
            "cross_run_memory": self.cross_run_memory,
            "hidden_holdout_policy": self.hidden_holdout_policy,
        }


@dataclass(frozen=True)
class CrossRunCandidateLesson:
    attempt: int
    source_run: str
    source_workspace_dir: str
    source_summary_path: str
    candidate_name: str | None
    candidate_mode: str | None
    outcome: str
    train_rate: float | None
    validation_rate: float | None
    baseline_full_rate: float | None
    final_full_rate: float | None
    final_holdout_rate: float | None
    full_delta: float | None
    comparison_mode: str | None
    promotion_eligible: bool | None
    checkpoint_path: str | None
    payload_path: str | None
    operation_count: int | None
    canonicalized_example_sql_ids: tuple[str, ...]
    risk_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "source_run": self.source_run,
            "source_workspace_dir": self.source_workspace_dir,
            "source_summary_path": self.source_summary_path,
            "candidate_name": self.candidate_name,
            "candidate_mode": self.candidate_mode,
            "outcome": self.outcome,
            "train_rate": self.train_rate,
            "validation_rate": self.validation_rate,
            "baseline_full_rate": self.baseline_full_rate,
            "final_full_rate": self.final_full_rate,
            "final_holdout_rate": self.final_holdout_rate,
            "full_delta": self.full_delta,
            "comparison_mode": self.comparison_mode,
            "promotion_eligible": self.promotion_eligible,
            "checkpoint_path": self.checkpoint_path,
            "payload_path": self.payload_path,
            "operation_count": self.operation_count,
            "canonicalized_example_sql_ids": list(self.canonicalized_example_sql_ids),
            "risk_reasons": list(self.risk_reasons),
        }


@dataclass(frozen=True)
class CrossRunMemory:
    schema_version: str
    generated_at: str
    space_id: str
    source_workspace_dir: str
    search_root: str
    runs_considered: int
    accepted_lessons: tuple[CrossRunCandidateLesson, ...]
    rejected_or_skipped_lessons: tuple[CrossRunCandidateLesson, ...]
    best_prior_lesson: CrossRunCandidateLesson | None
    best_prior_sequence: tuple[CrossRunCandidateLesson, ...]
    recommended_replay: str
    hidden_holdout_policy: str = HIDDEN_HOLDOUT_POLICY

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "space_id": self.space_id,
            "source_workspace_dir": self.source_workspace_dir,
            "search_root": self.search_root,
            "runs_considered": self.runs_considered,
            "accepted_lessons": [
                lesson.to_dict()
                for lesson in self.accepted_lessons
            ],
            "rejected_or_skipped_lessons": [
                lesson.to_dict()
                for lesson in self.rejected_or_skipped_lessons
            ],
            "best_prior_lesson": (
                self.best_prior_lesson.to_dict()
                if self.best_prior_lesson is not None
                else None
            ),
            "best_prior_sequence": [
                lesson.to_dict()
                for lesson in self.best_prior_sequence
            ],
            "recommended_replay": self.recommended_replay,
            "hidden_holdout_policy": self.hidden_holdout_policy,
        }


def build_trace_memory_index(artifacts: ArtifactManager) -> TraceMemoryIndex:
    """Build a typed run-memory index from workspace artifacts."""
    optimization_summary = _read_json_dict(artifacts.optimization_summary_path)
    structured_context = build_structured_failure_context(
        results_dir=artifacts.results_dir,
        history_dir=artifacts.history_dir,
        benchmarks_dir=artifacts.benchmarks_dir,
        best_train_rate=(
            optimization_summary.get("best_train_rate")
            if isinstance(optimization_summary, dict)
            else None
        ),
        best_validation_rate=(
            optimization_summary.get("best_validation_rate")
            if isinstance(optimization_summary, dict)
            else None
        ),
    )
    candidate_attempts = _candidate_attempts(
        artifacts=artifacts,
        optimization_summary=optimization_summary,
    )
    return TraceMemoryIndex(
        schema_version=TRACE_MEMORY_SCHEMA_VERSION,
        generated_at=_now_iso(),
        workspace_dir=str(artifacts.workspace_dir),
        space_id=artifacts.space_id,
        clone=artifacts.load_clone_meta() or {},
        scores=_score_groups(artifacts, optimization_summary),
        questions=_visible_questions(artifacts, structured_context),
        candidate_attempts=candidate_attempts,
        guard_regressions=_guard_regressions(candidate_attempts),
        failure_families={
            family: tuple(str(question_id) for question_id in question_ids)
            for family, question_ids in _dict_items(structured_context.get("failure_families"))
            if isinstance(question_ids, list)
        },
        sources=_source_paths(artifacts),
    )


def build_failure_context_pack(
    trace_memory: TraceMemoryIndex,
    *,
    max_failures: int = 8,
    max_pass_guards: int = 8,
    max_candidate_lessons: int = 8,
    cross_run_memory: CrossRunMemory | None = None,
) -> FailureContextPack:
    """Build a compact JSON pack for candidate proposal and supervision."""
    failed_questions = [
        question
        for question in trace_memory.questions
        if _normalized_status(question.status) != _PASS_STATUS
    ]
    failed_questions.sort(
        key=lambda question: (
            question.score is None,
            question.score if question.score is not None else 1.0,
            question.question_id,
        )
    )
    pass_guards = [
        question
        for question in trace_memory.questions
        if _normalized_status(question.status) == _PASS_STATUS
    ]
    candidate_lessons = _candidate_lessons(trace_memory.candidate_attempts)
    return FailureContextPack(
        schema_version=FAILURE_CONTEXT_PACK_SCHEMA_VERSION,
        generated_at=_now_iso(),
        space_id=trace_memory.space_id,
        scorecard=_scorecard(trace_memory),
        target_failures=tuple(
            _question_context_item(question)
            for question in failed_questions[:max_failures]
        ),
        pass_guards=tuple(
            _pass_guard_context_item(question)
            for question in pass_guards[:max_pass_guards]
        ),
        guard_regressions=tuple(
            regression.to_dict()
            for regression in trace_memory.guard_regressions[:max_candidate_lessons]
        ),
        candidate_lessons=tuple(candidate_lessons[:max_candidate_lessons]),
        failure_families=trace_memory.failure_families,
        recommended_next_focus=_recommended_next_focus(
            failed_questions=failed_questions,
            guard_regressions=trace_memory.guard_regressions,
            failure_families=trace_memory.failure_families,
        ),
        sources=trace_memory.sources,
        cross_run_memory=(
            _compact_cross_run_memory(cross_run_memory)
            if cross_run_memory is not None
            else None
        ),
    )


def build_cross_run_memory(
    artifacts: ArtifactManager,
    *,
    search_root: str | Path | None = None,
    max_runs: int = 8,
    max_lessons: int = 12,
) -> CrossRunMemory:
    """Build a compact memory pack from sibling experiment workspaces."""
    resolved_search_root = _resolve_cross_run_search_root(artifacts, search_root)
    workspace_dirs = _sibling_workspace_dirs(
        artifacts=artifacts,
        search_root=resolved_search_root,
        max_runs=max_runs,
    )
    accepted: list[CrossRunCandidateLesson] = []
    rejected_or_skipped: list[CrossRunCandidateLesson] = []
    accepted_by_workspace: dict[str, list[CrossRunCandidateLesson]] = {}
    for workspace_dir in workspace_dirs:
        summary_path = workspace_dir / "results" / "optimization_summary.json"
        summary = _read_json_dict(summary_path)
        if not isinstance(summary, dict):
            continue
        for lesson in _cross_run_lessons_from_summary(
            workspace_dir=workspace_dir,
            summary_path=summary_path,
            summary=summary,
        ):
            if lesson.outcome == "accepted":
                accepted.append(lesson)
                accepted_by_workspace.setdefault(str(workspace_dir), []).append(lesson)
            elif lesson.outcome in {"rejected", "skipped"}:
                rejected_or_skipped.append(lesson)

    accepted.sort(key=_cross_run_lesson_rank, reverse=True)
    rejected_or_skipped.sort(key=_cross_run_lesson_rank, reverse=True)
    accepted = accepted[:max_lessons]
    rejected_or_skipped = rejected_or_skipped[:max_lessons]
    best_prior_lesson = accepted[0] if accepted else None
    best_prior_sequence = _best_prior_sequence(accepted_by_workspace)
    return CrossRunMemory(
        schema_version=CROSS_RUN_MEMORY_SCHEMA_VERSION,
        generated_at=_now_iso(),
        space_id=artifacts.space_id,
        source_workspace_dir=str(artifacts.workspace_dir),
        search_root=str(resolved_search_root),
        runs_considered=len(workspace_dirs),
        accepted_lessons=tuple(accepted),
        rejected_or_skipped_lessons=tuple(rejected_or_skipped),
        best_prior_lesson=best_prior_lesson,
        best_prior_sequence=best_prior_sequence,
        recommended_replay=_recommended_cross_run_replay(best_prior_lesson, best_prior_sequence),
    )


def write_trace_memory_artifacts(
    artifacts: ArtifactManager,
    *,
    max_failures: int = 8,
    max_pass_guards: int = 8,
    max_candidate_lessons: int = 8,
    trace_memory_path: str | Path | None = None,
    failure_context_pack_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write trace-memory index and compact context pack artifacts."""
    trace_memory = build_trace_memory_index(artifacts)
    cross_run_memory = build_cross_run_memory(artifacts)
    context_pack = build_failure_context_pack(
        trace_memory,
        max_failures=max_failures,
        max_pass_guards=max_pass_guards,
        max_candidate_lessons=max_candidate_lessons,
        cross_run_memory=cross_run_memory,
    )
    memory_path = Path(trace_memory_path) if trace_memory_path is not None else artifacts.trace_memory_path
    pack_path = (
        Path(failure_context_pack_path)
        if failure_context_pack_path is not None
        else artifacts.failure_context_pack_path
    )
    artifacts.write_json_path(memory_path, trace_memory.to_dict())
    artifacts.write_json_path(artifacts.cross_run_memory_path, cross_run_memory.to_dict())
    artifacts.write_json_path(pack_path, context_pack.to_dict())
    return memory_path, pack_path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _updated_at(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else None


def _read_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, list) else []


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dict_items(value: Any) -> Iterable[tuple[str, Any]]:
    if not isinstance(value, dict):
        return ()
    return ((str(key), item) for key, item in value.items())


def _summary_from_results(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    typed_results = [result for result in results if isinstance(result, dict)]
    total = len(typed_results)
    passed = sum(1 for result in typed_results if _normalized_status(result.get("status")) == _PASS_STATUS)
    failed = total - passed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": sum(1 for result in typed_results if _normalized_status(result.get("status")) == "error"),
        "pass_rate": round((passed / total * 100) if total else 0.0, 2),
        "failed_ids": [
            str(result.get("question_id"))
            for result in typed_results
            if _normalized_status(result.get("status")) != _PASS_STATUS and result.get("question_id") is not None
        ],
    }


def _score_from_summary(
    *,
    split: str,
    summary: dict[str, Any] | None,
    source_path: Path,
) -> ScoreSnapshot:
    summary = summary or {}
    return ScoreSnapshot(
        split=split,
        pass_rate=_coerce_float(summary.get("pass_rate")),
        passed=_coerce_int(summary.get("passed")),
        total=_coerce_int(summary.get("total") or summary.get("question_count")),
        failed=_coerce_int(summary.get("failed")),
        errors=_coerce_int(summary.get("errors")),
        failed_ids=tuple(str(value) for value in summary.get("failed_ids", []) if str(value).strip()),
        source_path=str(source_path),
        updated_at=_updated_at(source_path),
    )


def _score_from_run_path(path: Path, split: str) -> ScoreSnapshot:
    payload = _read_json_dict(path)
    if payload is None:
        return _score_from_summary(split=split, summary=None, source_path=path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    if "pass_rate" not in summary:
        summary = _summary_from_results(payload)
    return _score_from_summary(split=split, summary=summary, source_path=path)


def _score_from_full_path(path: Path, split: str, section: str) -> ScoreSnapshot:
    payload = _read_json_dict(path)
    summary = payload.get(section) if isinstance(payload, dict) and isinstance(payload.get(section), dict) else None
    return _score_from_summary(split=split, summary=summary, source_path=path)


def _score_from_text_path(path: Path, split: str) -> ScoreSnapshot:
    pass_rate = None
    if path.exists():
        pass_rate = _coerce_float(path.read_text(encoding="utf-8").strip())
    return ScoreSnapshot(
        split=split,
        pass_rate=pass_rate,
        source_path=str(path),
        updated_at=_updated_at(path),
    )


def _score_groups(
    artifacts: ArtifactManager,
    optimization_summary: dict[str, Any] | None,
) -> dict[str, dict[str, ScoreSnapshot]]:
    final_summary = (
        optimization_summary.get("final")
        if isinstance(optimization_summary, dict) and isinstance(optimization_summary.get("final"), dict)
        else {}
    )
    baseline = {
        "full": _score_from_full_path(artifacts.baseline_full_summary_path, "full", "overall"),
        "train": _score_from_run_path(artifacts.baseline_train_path, "train"),
        "validation": _score_from_run_path(artifacts.baseline_validation_summary_path, "validation"),
        "holdout": _score_from_full_path(artifacts.baseline_full_summary_path, "holdout", "test"),
    }
    latest_holdout = (
        _score_from_summary(
            split="holdout",
            summary=final_summary.get("test") if isinstance(final_summary.get("test"), dict) else None,
            source_path=artifacts.optimization_summary_path,
        )
        if final_summary
        else _score_from_text_path(artifacts.latest_test_score_path, "holdout")
    )
    current = {
        "full": _score_from_summary(
            split="full",
            summary=final_summary.get("overall") if isinstance(final_summary.get("overall"), dict) else None,
            source_path=artifacts.optimization_summary_path,
        ) if final_summary else _score_from_full_path(artifacts.latest_full_summary_path, "full", "overall"),
        "train": _score_from_summary(
            split="train",
            summary=final_summary.get("train") if isinstance(final_summary.get("train"), dict) else None,
            source_path=artifacts.optimization_summary_path,
        ) if final_summary else _score_from_run_path(artifacts.latest_train_path, "train"),
        "validation": _score_from_summary(
            split="validation",
            summary=final_summary.get("validation") if isinstance(final_summary.get("validation"), dict) else None,
            source_path=artifacts.optimization_summary_path,
        ) if final_summary else _score_from_run_path(artifacts.latest_validation_summary_path, "validation"),
        "holdout": latest_holdout,
    }
    return {"baseline": baseline, "current": current}


def _question_ids_from_split(path: Path) -> tuple[str, ...]:
    ids: list[str] = []
    for entry in _read_json_list(path):
        if not isinstance(entry, dict):
            continue
        question_id = entry.get("id") or entry.get("question_id")
        if question_id is not None:
            ids.append(str(question_id))
    return tuple(ids)


def _split_membership(artifacts: ArtifactManager) -> dict[str, tuple[str, ...]]:
    membership: dict[str, list[str]] = {}
    split_paths = {
        "train": artifacts.train_set_path,
        "validation": artifacts.validation_set_path,
        "holdout": artifacts.test_set_path,
    }
    for split, path in split_paths.items():
        for question_id in _question_ids_from_split(path):
            membership.setdefault(question_id, []).append(split)
    return {
        question_id: tuple(splits)
        for question_id, splits in membership.items()
    }


def _structured_failures_by_id(structured_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    failures = structured_context.get("failures")
    if not isinstance(failures, list):
        return {}
    return {
        str(failure["question_id"]): failure
        for failure in failures
        if isinstance(failure, dict) and failure.get("question_id") is not None
    }


def _visible_questions(
    artifacts: ArtifactManager,
    structured_context: dict[str, Any],
) -> tuple[TraceQuestion, ...]:
    split_membership = _split_membership(artifacts)
    structured_failures = _structured_failures_by_id(structured_context)
    latest_train = _read_json_dict(artifacts.latest_train_path) or {}
    results = latest_train.get("results") if isinstance(latest_train.get("results"), list) else []
    questions: list[TraceQuestion] = []
    seen: set[str] = set()
    for result in results:
        if not isinstance(result, dict) or result.get("question_id") is None:
            continue
        question_id = str(result["question_id"])
        structured = structured_failures.get(question_id, {})
        questions.append(
            TraceQuestion(
                question_id=question_id,
                split_membership=split_membership.get(question_id, ("train",)),
                question=_optional_str(result.get("question")),
                status=_optional_str(result.get("status")),
                score=_coerce_float(result.get("comparison_score")),
                comparison_method=_optional_str(result.get("comparison_method")),
                issue_class=_optional_str(structured.get("issue_class")),
                root_cause_family=_optional_str(structured.get("likely_root_cause_family")),
                expected_sql=_optional_str(result.get("expected_sql")),
                generated_sql=_optional_str(result.get("generated_sql")),
                missing_expected_tokens=tuple(
                    str(token)
                    for token in structured.get("missing_expected_tokens", [])
                    if str(token).strip()
                ),
                guard_regression_risk=_optional_str(structured.get("guard_regression_risk")),
                source_path=str(artifacts.latest_train_path),
            )
        )
        seen.add(question_id)
    pass_guards = (
        structured_context.get("pass_guards")
        if isinstance(structured_context.get("pass_guards"), list)
        else []
    )
    for guard in pass_guards:
        if not isinstance(guard, dict) or guard.get("question_id") is None:
            continue
        question_id = str(guard["question_id"])
        if question_id in seen:
            continue
        questions.append(
            TraceQuestion(
                question_id=question_id,
                split_membership=split_membership.get(question_id, ("train",)),
                question=_optional_str(guard.get("question")),
                status=_optional_str(guard.get("latest_status")),
                score=None,
                comparison_method=None,
                issue_class=None,
                root_cause_family=None,
                expected_sql=_optional_str(guard.get("expected_sql")),
                generated_sql=None,
                missing_expected_tokens=(),
                guard_regression_risk=None,
                source_path=str(artifacts.latest_train_path),
            )
        )
    return tuple(questions)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalized_status(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _iteration_log_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid trace source artifact: {path}:{line_number}") from exc
        if isinstance(payload, dict) and payload.get("phase") == "candidate_evaluated":
            events.append(payload)
    return events


def _candidate_attempt_fields(entry: dict[str, Any], *, attempt: int, source_path: Path) -> CandidateAttempt:
    candidate_summary = entry.get("candidate_summary") if isinstance(entry.get("candidate_summary"), dict) else {}
    return CandidateAttempt(
        attempt=attempt,
        name=_optional_str(entry.get("name") or candidate_summary.get("candidate_name")),
        label=_optional_str(entry.get("label")),
        family=_optional_str(entry.get("family")),
        outcome=_optional_str(entry.get("outcome")),
        train_rate=_coerce_float(entry.get("train_rate")),
        validation_rate=_coerce_float(entry.get("validation_rate")),
        reason=_optional_str(entry.get("reason")),
        candidate_mode=_optional_str(candidate_summary.get("candidate_mode")),
        skipped_validation=bool(entry.get("skipped_validation", False)),
        rejection_class=_optional_str(entry.get("rejection_class")),
        source_path=str(source_path),
    )


def _candidate_attempts(
    *,
    artifacts: ArtifactManager,
    optimization_summary: dict[str, Any] | None,
) -> tuple[CandidateAttempt, ...]:
    passes = (
        optimization_summary.get("passes")
        if isinstance(optimization_summary, dict) and isinstance(optimization_summary.get("passes"), list)
        else []
    )
    if passes:
        attempts = [
            _candidate_attempt_fields(entry, attempt=index, source_path=artifacts.optimization_summary_path)
            for index, entry in enumerate(passes, start=1)
            if isinstance(entry, dict)
        ]
        return tuple(attempts)
    return tuple(
        _candidate_attempt_fields(entry, attempt=index, source_path=artifacts.history_dir / "iteration_log.jsonl")
        for index, entry in enumerate(_iteration_log_events(artifacts.history_dir / "iteration_log.jsonl"), start=1)
    )


def _guard_regressions(candidate_attempts: tuple[CandidateAttempt, ...]) -> tuple[GuardRegression, ...]:
    regressions: list[GuardRegression] = []
    for attempt in candidate_attempts:
        if not attempt.reason:
            continue
        if "pass guard regressed:" not in attempt.reason and "accepted fix regressed:" not in attempt.reason:
            continue
        _, _, raw_ids = attempt.reason.partition(":")
        for question_id in re.split(r"[, ]+", raw_ids.strip()):
            cleaned = question_id.strip().strip("`")
            if cleaned:
                regressions.append(
                    GuardRegression(
                        question_id=cleaned,
                        candidate_name=attempt.name,
                        reason=attempt.reason,
                        source_path=attempt.source_path,
                    )
                )
    return tuple(regressions)


def _source_paths(artifacts: ArtifactManager) -> dict[str, str | None]:
    paths = {
        "latest_train": artifacts.latest_train_path,
        "baseline_train": artifacts.baseline_train_path,
        "latest_validation_summary": artifacts.latest_validation_summary_path,
        "baseline_validation_summary": artifacts.baseline_validation_summary_path,
        "latest_full_summary": artifacts.latest_full_summary_path,
        "baseline_full_summary": artifacts.baseline_full_summary_path,
        "optimization_summary": artifacts.optimization_summary_path,
        "iteration_log": artifacts.history_dir / "iteration_log.jsonl",
    }
    return {
        name: str(path) if path.exists() else None
        for name, path in paths.items()
    }


def _scorecard(trace_memory: TraceMemoryIndex) -> dict[str, Any]:
    def _score(group: str, split: str) -> dict[str, Any] | None:
        score = trace_memory.scores.get(group, {}).get(split)
        if score is None:
            return None
        payload = score.to_dict()
        if split in {"full", "holdout"}:
            payload["failed_ids"] = []
            payload["failed_ids_policy"] = (
                "omitted because full and holdout summaries can include hidden holdout question ids"
            )
        return payload

    return {
        "baseline": {
            split: _score("baseline", split)
            for split in ("full", "train", "validation", "holdout")
        },
        "current": {
            split: _score("current", split)
            for split in ("full", "train", "validation", "holdout")
        },
    }


def _question_context_item(question: TraceQuestion) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "question": question.question,
        "split_membership": list(question.split_membership),
        "status": question.status,
        "score": question.score,
        "comparison_method": question.comparison_method,
        "issue_class": question.issue_class,
        "root_cause_family": question.root_cause_family,
        "missing_expected_tokens": list(question.missing_expected_tokens[:12]),
        "guard_regression_risk": question.guard_regression_risk,
        "expected_sql": question.expected_sql,
        "generated_sql": question.generated_sql,
    }


def _pass_guard_context_item(question: TraceQuestion) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "question": question.question,
        "split_membership": list(question.split_membership),
        "status": question.status,
        "expected_sql": question.expected_sql,
    }


def _candidate_lessons(candidate_attempts: tuple[CandidateAttempt, ...]) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    for attempt in candidate_attempts:
        lessons.append(
            {
                "attempt": attempt.attempt,
                "name": attempt.name,
                "outcome": attempt.outcome,
                "candidate_mode": attempt.candidate_mode,
                "train_rate": attempt.train_rate,
                "validation_rate": attempt.validation_rate,
                "reason": attempt.reason,
                "rejection_class": attempt.rejection_class,
            }
        )
    return lessons


def _compact_cross_run_memory(cross_run_memory: CrossRunMemory) -> dict[str, Any]:
    payload = cross_run_memory.to_dict()
    return {
        "schema_version": payload["schema_version"],
        "runs_considered": payload["runs_considered"],
        "best_prior_lesson": payload["best_prior_lesson"],
        "best_prior_sequence": payload["best_prior_sequence"][:5],
        "accepted_lessons": payload["accepted_lessons"][:5],
        "rejected_or_skipped_lessons": payload["rejected_or_skipped_lessons"][:5],
        "recommended_replay": payload["recommended_replay"],
        "hidden_holdout_policy": payload["hidden_holdout_policy"],
    }


def _resolve_cross_run_search_root(
    artifacts: ArtifactManager,
    search_root: str | Path | None,
) -> Path:
    if search_root is not None:
        return Path(search_root).expanduser().resolve()
    if artifacts.workspace_dir.name == artifacts.space_id:
        return artifacts.workspace_dir.parent.parent
    return artifacts.workspace_dir.parent


def _sibling_workspace_dirs(
    *,
    artifacts: ArtifactManager,
    search_root: Path,
    max_runs: int,
) -> tuple[Path, ...]:
    if not search_root.exists() or not search_root.is_dir():
        return ()
    candidates: list[Path] = []
    for child in search_root.iterdir():
        if not child.is_dir():
            continue
        for candidate in (child / artifacts.space_id, child):
            if candidate.resolve() == artifacts.workspace_dir.resolve():
                continue
            if (candidate / "results" / "optimization_summary.json").exists():
                candidates.append(candidate.resolve())
                break
    candidates.sort(
        key=lambda path: (path / "results" / "optimization_summary.json").stat().st_mtime,
        reverse=True,
    )
    return tuple(dict.fromkeys(candidates[:max_runs]))


def _summary_pass_rate(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _coerce_float(current)


def _full_delta(baseline_full_rate: float | None, final_full_rate: float | None) -> float | None:
    if baseline_full_rate is None or final_full_rate is None:
        return None
    return round(final_full_rate - baseline_full_rate, 2)


def _candidate_summary(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("candidate_summary")
    return summary if isinstance(summary, dict) else {}


def _risk_assessment(candidate_summary: dict[str, Any]) -> dict[str, Any]:
    risk_assessment = candidate_summary.get("risk_assessment")
    return risk_assessment if isinstance(risk_assessment, dict) else {}


def _canonicalized_example_ids(risk_assessment: dict[str, Any]) -> tuple[str, ...]:
    canonicalization = risk_assessment.get("canonicalization")
    if not isinstance(canonicalization, dict):
        return ()
    values = canonicalization.get("canonicalized_example_sql_ids")
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def _risk_reasons(risk_assessment: dict[str, Any]) -> tuple[str, ...]:
    values = risk_assessment.get("reasons")
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def _cross_run_name(workspace_dir: Path) -> str:
    if workspace_dir.name.startswith("01"):
        return workspace_dir.parent.name
    return workspace_dir.name


def _cross_run_lessons_from_summary(
    *,
    workspace_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> tuple[CrossRunCandidateLesson, ...]:
    passes = summary.get("passes")
    if not isinstance(passes, list):
        return ()
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else {}
    final = summary.get("final") if isinstance(summary.get("final"), dict) else {}
    protocol_preflight = (
        summary.get("protocol_preflight")
        if isinstance(summary.get("protocol_preflight"), dict)
        else {}
    )
    eligibility = (
        protocol_preflight.get("promotion_eligibility")
        if isinstance(protocol_preflight.get("promotion_eligibility"), dict)
        else {}
    )
    final_context = (
        summary.get("final_evaluation_context")
        if isinstance(summary.get("final_evaluation_context"), dict)
        else final.get("evaluation_context") if isinstance(final.get("evaluation_context"), dict) else {}
    )
    comparison_mode = _optional_str(final_context.get("preferred_comparison_mode"))
    promotion_eligible = eligibility.get("promotion_eligible")
    promotion_eligible_bool = promotion_eligible if isinstance(promotion_eligible, bool) else None
    baseline_full_rate = _coerce_float(baseline.get("full_pass_rate"))
    final_full_rate = _summary_pass_rate(final, "overall", "pass_rate")
    final_holdout_rate = _summary_pass_rate(final, "test", "pass_rate")
    lessons: list[CrossRunCandidateLesson] = []
    for attempt, entry in enumerate(passes, start=1):
        if not isinstance(entry, dict):
            continue
        outcome = _normalized_status(entry.get("outcome"))
        if outcome not in {"accepted", "rejected", "skipped"}:
            continue
        candidate_summary = _candidate_summary(entry)
        risk_assessment = _risk_assessment(candidate_summary)
        lessons.append(
            CrossRunCandidateLesson(
                attempt=attempt,
                source_run=_cross_run_name(workspace_dir),
                source_workspace_dir=str(workspace_dir),
                source_summary_path=str(summary_path),
                candidate_name=_optional_str(candidate_summary.get("candidate_name") or entry.get("name")),
                candidate_mode=_optional_str(candidate_summary.get("candidate_mode")),
                outcome=outcome,
                train_rate=_coerce_float(entry.get("train_rate")),
                validation_rate=_coerce_float(entry.get("validation_rate")),
                baseline_full_rate=baseline_full_rate,
                final_full_rate=final_full_rate,
                final_holdout_rate=final_holdout_rate,
                full_delta=_full_delta(baseline_full_rate, final_full_rate),
                comparison_mode=comparison_mode,
                promotion_eligible=promotion_eligible_bool,
                checkpoint_path=_optional_str(entry.get("checkpoint_path")),
                payload_path=_optional_str(candidate_summary.get("payload_path")),
                operation_count=_coerce_int(candidate_summary.get("operation_count")),
                canonicalized_example_sql_ids=_canonicalized_example_ids(risk_assessment),
                risk_reasons=_risk_reasons(risk_assessment),
            )
        )
    return tuple(lessons)


def _rank_value(value: float | None) -> float:
    return value if value is not None else -1.0


def _protocol_rank(lesson: CrossRunCandidateLesson) -> float:
    if lesson.promotion_eligible is True:
        return 2.0
    if str(lesson.comparison_mode or "").lower() == "result_based":
        return 1.0
    return 0.0


def _cross_run_lesson_rank(lesson: CrossRunCandidateLesson) -> tuple[float, float, float, float]:
    return (
        _protocol_rank(lesson),
        _rank_value(lesson.validation_rate),
        _rank_value(lesson.train_rate),
        _rank_value(lesson.full_delta),
    )


def _best_prior_sequence(
    accepted_by_workspace: dict[str, list[CrossRunCandidateLesson]],
) -> tuple[CrossRunCandidateLesson, ...]:
    sequences = [
        tuple(sorted(lessons, key=lambda lesson: lesson.attempt))
        for lessons in accepted_by_workspace.values()
        if lessons
    ]
    if not sequences:
        return ()
    sequences.sort(key=_cross_run_sequence_rank, reverse=True)
    return sequences[0]


def _cross_run_sequence_rank(
    sequence: tuple[CrossRunCandidateLesson, ...],
) -> tuple[float, float, float, float, float, float]:
    final_full_rate = sequence[0].final_full_rate
    final_holdout_rate = sequence[0].final_holdout_rate
    best_validation_rate = max(_rank_value(lesson.validation_rate) for lesson in sequence)
    best_train_rate = max(_rank_value(lesson.train_rate) for lesson in sequence)
    return (
        max(_protocol_rank(lesson) for lesson in sequence),
        _rank_value(final_full_rate),
        _rank_value(final_holdout_rate),
        best_validation_rate,
        best_train_rate,
        float(len(sequence)),
    )


def _recommended_cross_run_replay(
    best_prior_lesson: CrossRunCandidateLesson | None,
    best_prior_sequence: tuple[CrossRunCandidateLesson, ...],
) -> str:
    if best_prior_sequence:
        names = ", ".join(
            str(lesson.candidate_name)
            for lesson in best_prior_sequence[:4]
            if lesson.candidate_name
        )
        source_run = best_prior_sequence[0].source_run
        return (
            f"Replay the ordered accepted candidate sequence from `{source_run}` "
            f"under the current gates before proposing new changes: {names}."
        )
    if best_prior_lesson is None:
        return "No prior accepted candidate patterns were found in sibling runs."
    ids = ", ".join(best_prior_lesson.canonicalized_example_sql_ids[:6])
    if ids:
        return (
            f"Replay the `{best_prior_lesson.candidate_name}` pattern from "
            f"`{best_prior_lesson.source_run}` as evidence, preserving its guard examples "
            f"and avoiding duplicate target IDs already fixed there: {ids}."
        )
    return (
        f"Use `{best_prior_lesson.candidate_name}` from `{best_prior_lesson.source_run}` "
        "as the strongest prior accepted pattern, but regenerate operations from current failures."
    )


def _recommended_next_focus(
    *,
    failed_questions: list[TraceQuestion],
    guard_regressions: tuple[GuardRegression, ...],
    failure_families: dict[str, tuple[str, ...]],
) -> str:
    if guard_regressions:
        return "Prior attempts regressed pass guards; next candidate should narrow scope and preserve listed guards."
    if not failed_questions:
        return "No visible train failures remain; run validation and final holdout before claiming completion."
    if failure_families:
        family, question_ids = max(
            failure_families.items(),
            key=lambda item: (len(item[1]), item[0]),
        )
        return f"Target the largest visible failure family `{family}` covering {len(question_ids)} question(s)."
    return "Target the lowest-scoring visible train failure with a localized structured change."
