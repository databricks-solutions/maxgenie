"""Workspace-oriented optimization flow for the CLI-driven loop."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import dataclass, replace
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from databricks.sdk.service import workspace as workspace_service
import yaml

from maxgenie.advisory import (
    ADVISORY_BEST_PRACTICES_CLASSIFICATION,
    NO_BENCHMARK_WARNING,
    NOT_BENCHMARK_VERIFIED_LABEL,
    NOT_BENCHMARK_VERIFIED_STATUS,
    is_advisory_best_practices,
)
from maxgenie.assembler import assemble_config
from maxgenie.artifact_patch import (
    apply_artifact_patch_candidate,
    load_artifact_patch_sequence,
    repair_artifact_patch_v2_payload,
)
from maxgenie.artifact_patch_semantics import (
    question_overlap_terms,
    sql_delta_intents_from_visible_failure,
)
from maxgenie.artifacts import ArtifactManager
from maxgenie.benchmark_cache import BenchmarkCache
from maxgenie.benchmark_service import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkService,
    BenchmarkStatus,
)
from maxgenie.benchmark_summaries import (
    _aggregate_comparison_metrics,
    _comparison_metrics_from_run,
    _comparison_mode,
    _derived_full_summary,
    _summarize_run,
)
from maxgenie.config import (
    AUTONOMOUS_STRATEGY_CENTAUR,
    AUTONOMOUS_STRATEGY_SERVING,
)
from maxgenie.decomposer import decompose_config
from maxgenie.exporter import (
    BenchmarkSplit,
    ExportBundle,
    SpaceExporter,
    ValidationFold,
    benchmark_questions_to_model,
    benchmark_questions_to_payload,
    extract_benchmark_questions_from_config,
)
from maxgenie.family_validation import (
    build_proxy_guard_requirements,
    compute_visible_proxy_pressure,
)
from maxgenie.genie_service import GenieService, GenieSpaceInfo
from maxgenie.query_history import (
    QueryHistoryCollector,
    build_query_history_recommendations,
    discover_best_warehouse_id,
)
from maxgenie.result_comparator import ResultComparator
from maxgenie.schemas import GenieSpaceConfig
from maxgenie.serving_candidates import (
    ServingCandidateClient,
    ServingCandidateConfig,
    ServingCandidateProposal,
    ServingCandidateResponseError,
    assess_serving_candidate_risk,
    apply_serving_candidate,
    artifact_patch_causal_recipe_anchors,
    baseline_pass_guard_ids,
    baseline_pass_guard_questions,
    build_artifact_patch_skeletons,
    canonicalize_tagged_example_sql,
    count_prior_accepted_fix_regressions,
    count_prior_invalid_responses,
    count_prior_pass_guard_regressions,
    count_prior_rejected_candidates,
    count_prior_rejected_candidate_modes,
    count_prior_skipped_candidates,
    materialize_typed_template_examples,
    repair_example_set_pass_guards,
    serving_candidate_artifact_patch_requires_executable_asset,
    serving_candidate_artifact_patch_file_edit_limits,
    serving_candidate_allowed_operation_groups,
    serving_candidate_is_broad_artifact_patch_mode,
    serving_candidate_is_executable_artifact_path,
    serving_candidate_is_artifact_patch_mode,
    serving_candidate_is_example_set_mode,
    serving_candidate_is_strict_artifact_patch_mode,
    serving_candidate_uses_data_probe,
    serving_candidate_uses_historical_analysis,
    serving_candidate_max_mutation_operations,
    serving_candidate_mode_for_iteration,
    dominant_visible_failure_family,
    validate_artifact_patch_skeleton_binding,
)
from maxgenie.policies import CandidatePolicy, PolicyContext, centaur_prepass_policies, default_policies
from maxgenie.protocol_preflight import (
    STRING_FALLBACK_COMPARISON_METHODS,
    validate_protocol_preflight,
)
from maxgenie.space_curation import (
    CURATION_MODE_SAFE_DEFAULTS,
    curate_space_config,
)
from maxgenie.space_diagnostics import (
    SpaceDiagnostics,
    analyze_space_config,
    expanded_entity_matching_candidates,
    inconsistent_filter_snippet_ids,
    instruction_contradiction_findings,
)
from maxgenie.mlflow_tracking import MlflowTrackerProtocol, tracker_from_env
from maxgenie.multi_sample import (
    PromotionDecision,
    SampleAggregate,
    TransferGateDecision,
    aggregate_samples,
    evaluate_transfer_gate,
    lift_threshold,
    minimum_detectable_difference,
    promotion_decision,
)

logger = logging.getLogger(__name__)

_SPACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CONVERSATION_READY_TIMEOUT_SECONDS = 180.0
_CONVERSATION_READY_POLL_SECONDS = 10.0
_OPTIMIZER_TITLE_TAG = "maxgenie"
_OPTIMIZER_TITLE_PREFIX_RE = re.compile(
    r"^\[(?:(?:MaxGenie|maxgenie[-\w]*)|Space Optimizer(?: [-\w]+)?)(?: [0-9]+(?:\.[0-9]+)?%(?: full)?)?\]\s*",
    re.IGNORECASE,
)
_VALIDATION_IMPROVEMENT_EPSILON = 0.1
_VALIDATION_REGRESSION_EPSILON = 0.1
_TERMINAL_SELECTION_EPSILON = 0.1
_MULTI_SAMPLE_BASELINE_SOURCE = "population_multi_sample_baseline"
_MULTI_SAMPLE_BASELINE_PASS_FRACTION_THRESHOLD = 0.5
_GENERALIZATION_AUDIT_SCHEMA_VERSION = "maxgenie.generalization_audit.v1"
_GENERALIZATION_MAJOR_GAP_PCT = 20.0
_GENERALIZATION_MIN_HOLDOUT_QUESTIONS = 5
_TRAIN_REGRESSION_EPSILON = 0.1
_TRAIN_IMPROVEMENT_EPSILON = 0.1
_TRAIN_HEALTH_FLOOR_ABSOLUTE = 10.0
_TRAIN_HEALTH_MAX_DROP_PCT = 50.0
_CLONE_PARENT_PATH_ENV = "MAXGENIE_CLONE_PARENT_PATH"
_RESULT_CONTENT_MATCH_ENV = "MAXGENIE_RESULT_CONTENT_MATCH"
_EXECUTABLE_ARTIFACT_PATCH_VISIBLE_FAMILIES = frozenset(
    {
        "literal_or_parameter_mismatch",
        "selected_metric_or_column_mismatch",
        "percentage_or_comparison_metric",
        "time_filter_mismatch",
        "schema_or_alias_ambiguity",
        "entity_resolution_mismatch",
    }
)
_EXECUTABLE_ARTIFACT_PATCH_EFFECT_MARKERS = frozenset(
    {
        "target_train_failure:",
        "visible_family:",
        "failure_family:",
        "guard_preservation:",
        "literal_or_parameter_mismatch",
        "selected_metric_or_column_mismatch",
        "percentage_or_comparison_metric",
        "time_filter_mismatch",
        "schema_or_alias_ambiguity",
        "entity_resolution_mismatch",
    }
)
_SQL_BIND_PLACEHOLDER_RE = re.compile(
    r"(?<!:):[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*(?:[^}]*)?\}"
)
_METADATA_QUERY_PLAN_TERMS = frozenset(
    {
        "all_sales",
        "time_buckets",
        "result schema",
        "aggregation schema",
        "distinct-value lookup",
        "generated sql",
        "output columns",
        "return ",
        "returns ",
    }
)
_METADATA_RESULT_SCHEMA_TOKENS = frozenset(
    {
        "brand_metric_value",
        "hierarchy_level",
        "market_name",
        "metric_market_share_pct",
        "metric_type",
        "region",
        "team_name",
        "time_bucket_id",
        "total_market_metric_value",
        "wk_grp_desc",
    }
)
_LITERAL_SWAP_GUARD_CLONE_RE = re.compile(
    "|".join(
        (
            r"\bmirror(?:s|ed|ing)?\b.{0,160}\bpassing\b.{0,80}\bguard\b"
            r".{0,120}\bwith\s+only\b.{0,80}\bliteral\b.{0,80}\bchang",
            r"\breus(?:e|es|ed|ing)\b.{0,80}\bthe\s+same\b.{0,160}\bguard\b"
            r".{0,120}\bchang(?:e|ed|ing)\s+only\b.{0,80}\bliteral\b",
            r"\bcopy(?:ies|ied|ing)?\b.{0,160}\bpassing\b.{0,80}\bguard\b"
            r".{0,120}\bonly\b.{0,80}\bliteral\b.{0,80}\b(?:chang|swap|replac)",
        )
    ),
    re.IGNORECASE,
)
_SQL_PREDICATE_OPERATOR_RE = re.compile(
    r"(?:=|<>|!=|<=|>=|<|>|\bIN\s*\(|\bLIKE\b|\bBETWEEN\b|\bIS\s+(?:NOT\s+)?NULL\b)",
    re.IGNORECASE,
)
_SQL_SNIPPET_OPTION_SET_RE = re.compile(
    r"\b(?:choose|select|use)\s+(?:exactly\s+)?one\b|\bone\s+predicate\b|\balternative(?:s)?\b",
    re.IGNORECASE,
)
_HIDDEN_HOLDOUT_REFERENCE_RE = re.compile(
    r"\b(?:hidden[_ -]?holdout|holdout\s+(?:question|sql|query)|final[_ -]?test|test[_ -]?ids)\b",
    re.IGNORECASE,
)
_SAFE_HIDDEN_HOLDOUT_POLICY_PHRASE_RE = re.compile(
    r"\b(?:contains?\s+)?no\s+hidden[_ -]?holdout\s+content\b"
    r"|\bhidden[_ -]?holdout\s+content\s+(?:is\s+)?omitted\b"
    r"|\bhidden[_ -]?holdout\s+exclusion\b"
    r"|\bhidden[_ -]?holdout\s+isolation\b"
    r"|\bhidden[_ -]?holdout\s+policy\b",
    re.IGNORECASE,
)
_DATA_PROBE_TEXT_COLUMN_RE = re.compile(
    r"(?:brand|market|region|team|name|type|category|segment|status|class|group|desc|label|code|id)$",
    re.IGNORECASE,
)
_DATA_PROBE_MAX_TABLES = 3
_DATA_PROBE_MAX_COLUMNS_PER_TABLE = 8
_DEFAULT_STRICT_ARTIFACT_PATCH_MODES = frozenset(
    {"artifact_patch_v2", "skeleton_artifact_patch_v2"}
)


def _has_forbidden_hidden_holdout_reference(text: str) -> bool:
    """Return true for hidden-holdout leaks while allowing policy boilerplate."""

    redacted = _SAFE_HIDDEN_HOLDOUT_POLICY_PHRASE_RE.sub("", text)
    return _HIDDEN_HOLDOUT_REFERENCE_RE.search(redacted) is not None


class WorkspaceFlowError(RuntimeError):
    """Raised when a workspace command cannot proceed safely."""


@dataclass(frozen=True)
class WorkspaceStatus:
    """High-level workspace status summary."""

    input_space_id: str | None
    source_space_id: str | None
    managed_source_space_id: str | None
    clone_space_id: str | None
    clone_accessible: bool
    baseline_train_pass_rate: float | None
    latest_train_pass_rate: float | None
    latest_validation_pass_rate: float | None
    latest_test_pass_rate: float | None
    latest_full_pass_rate: float | None
    blocked_candidate_ids: list[str]
    latest_full_comparison_mode: str | None = None
    external_table_reference_count: int | None = None
    entity_matching_candidate_count: int | None = None
    format_assistance_candidate_count: int | None = None
    described_column_coverage_percent: float | None = None
    curation_candidate_status: str | None = None
    curation_change_count: int | None = None
    hidden_column_count: int | None = None
    query_history_status: str | None = None
    query_history_query_count: int | None = None
    query_history_feedback_thumbs_down: int | None = None
    query_history_trusted_asset_candidates: int | None = None
    query_history_benchmark_candidates: int | None = None
    git_commit_sha: str | None = None
    git_dirty: bool | None = None
    autonomous_strategy: str | None = None
    last_stop_reason: str | None = None

    @property
    def latest_dev_pass_rate(self) -> float | None:
        """Backward-compatible alias for older callers."""
        return self.latest_validation_pass_rate


def _benchmark_cache_hash(
    *,
    config_hash: str,
    match_threshold: float,
    comparator: ResultComparator | None,
) -> str:
    """Build a cache key that includes benchmark comparison semantics."""
    if comparator is None:
        strategy = "string-compare"
    else:
        strategy = comparator.cache_fingerprint()
    return f"{config_hash}|threshold={match_threshold:.3f}|strategy={strategy}"


def infer_workspace_dir(workspace_root: str | Path, space_id: str | None = None) -> Path:
    """Resolve a workspace directory from cwd or the workspace root."""
    cwd = Path.cwd().resolve()
    if (cwd / "space").is_dir() and (cwd / "benchmarks").is_dir() and (cwd / "history").is_dir():
        return cwd

    root = Path(workspace_root).expanduser().resolve()
    if space_id:
        return root / space_id

    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and _SPACE_ID_RE.match(path.name) and (path / "space").is_dir()
    ] if root.exists() else []

    if len(candidates) == 1:
        return candidates[0]

    raise WorkspaceFlowError(
        "Could not infer the workspace directory. Run from workspace/<space_id> or pass --space-id."
    )


def _load_run(path: Path) -> BenchmarkRun | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return BenchmarkRun.from_dict(payload)


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else None


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_question_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        question_count = payload.get("question_count")
        if isinstance(question_count, int):
            return question_count
        for key in ("ids", "questions"):
            values = payload.get(key)
            if isinstance(values, list):
                return len(values)
    return None


def _question_from_payload(payload: dict[str, Any], *, label: str) -> BenchmarkQuestion:
    question_id = str(payload.get("id") or "").strip()
    question_text = str(payload.get("question") or "").strip()
    if not question_id or not question_text:
        raise WorkspaceFlowError(f"Invalid benchmark question in {label}.")
    expected_sql = payload.get("expected_sql")
    return BenchmarkQuestion(
        id=question_id,
        question=question_text,
        expected_sql=str(expected_sql) if expected_sql is not None else None,
    )


def _questions_from_payload(payload: dict[str, Any], *, label: str) -> list[BenchmarkQuestion]:
    questions_payload = payload.get("questions")
    if not isinstance(questions_payload, list):
        raise WorkspaceFlowError(f"Invalid {label}: expected a questions list.")
    questions: list[BenchmarkQuestion] = []
    for question in questions_payload:
        if not isinstance(question, dict):
            raise WorkspaceFlowError(f"Invalid benchmark question in {label}.")
        questions.append(_question_from_payload(question, label=label))
    return questions


def _coverage_percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100, 2)


def _artifact_mtime(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_mtime


def _latest_component_mtime(paths: list[Path]) -> float | None:
    mtimes = [mtime for mtime in (_artifact_mtime(path) for path in paths) if mtime is not None]
    return max(mtimes) if mtimes else None


def _ids_by_status(run: BenchmarkRun, statuses: set[BenchmarkStatus]) -> list[str]:
    return [result.question_id for result in run.results if result.status in statuses]


def _is_gated_curation_summary(summary: dict[str, Any]) -> bool:
    """Return True when the curation_summary.json was produced by curate_and_gate."""
    return isinstance(summary.get("passes"), list) and isinstance(summary.get("modes_tried"), list)


def _benchmark_evaluation_context(
    comparator: ResultComparator | None,
    *,
    match_threshold: float,
) -> dict[str, Any]:
    return {
        "preferred_comparison_mode": "result_based" if comparator is not None else "string_only",
        "comparison_fingerprint": (
            comparator.cache_fingerprint()
            if comparator is not None
            else "string-compare"
        ),
        "warehouse_id": comparator.warehouse_id if comparator is not None else None,
        "match_threshold": round(match_threshold, 3),
    }


def _comparison_method_counts_text(method_counts: dict[str, Any] | None) -> str:
    if not isinstance(method_counts, dict) or not method_counts:
        return "none"
    parts = [f"{method}={int(count)}" for method, count in sorted(method_counts.items()) if method]
    return ", ".join(parts) if parts else "none"


def _fallback_method_counts_text(method_counts: dict[str, Any] | None) -> str:
    if not isinstance(method_counts, dict) or not method_counts:
        return "none"
    parts: list[str] = []
    for method, count in sorted(method_counts.items()):
        method_name = str(method).strip().lower()
        if method_name not in STRING_FALLBACK_COMPARISON_METHODS:
            continue
        try:
            parsed_count = int(count)
        except (TypeError, ValueError):
            continue
        if parsed_count <= 0:
            continue
        parts.append(f"{method_name}={parsed_count}")
    return ", ".join(parts) if parts else "none"


def _build_evidence_protocol(
    *,
    strategy: str,
    evaluation_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy_under_test": strategy,
        "decision_rule": (
            "Treat optimizer changes as improvements only after rerun evidence "
            "shows no regression against the previous benchmark baseline."
        ),
        "preferred_comparison_mode": evaluation_context.get("preferred_comparison_mode"),
        "required_local_validation": [
            "targeted_unit_tests",
            "compile_checks",
        ],
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
    }


def _build_rerun_plan(
    *,
    strategy: str,
) -> dict[str, Any]:
    return {
        "current_strategy": strategy,
        "recommended_first_rerun": AUTONOMOUS_STRATEGY_CENTAUR,
        "first_rerun_reason": (
            "Centaur is the default evidence checkpoint after material optimizer "
            "changes because it is faster than a full multi-strategy evaluation "
            "while still measuring the default product direction."
        ),
        "follow_up_rerun": "full_strategy_evaluation",
        "follow_up_condition": (
            "Run the full strategy evaluation after a promising Centaur improvement "
            "or when a change affects comparative strategy claims."
        ),
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
    }


def _summarize_subset(run: BenchmarkRun, question_ids: set[str]) -> dict[str, Any]:
    subset = [result for result in run.results if result.question_id in question_ids]
    total = len(subset)
    passed = sum(1 for result in subset if result.status == BenchmarkStatus.PASSED)
    failed = sum(1 for result in subset if result.status == BenchmarkStatus.FAILED)
    errors = sum(1 for result in subset if result.status == BenchmarkStatus.ERROR)
    pass_rate = (passed / total * 100) if total else 0.0
    average_score = (
        round(
            sum(
                float(result.comparison_score)
                if result.comparison_score is not None
                else (1.0 if result.status == BenchmarkStatus.PASSED else 0.0)
                for result in subset
            )
            / total,
            4,
        )
        if total
        else 0.0
    )
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(pass_rate, 2),
        "average_score": average_score,
    }


def _benchmark_issue_class(result: BenchmarkResult) -> str | None:
    if result.status not in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}:
        return None

    message = _benchmark_error_message_text(result.error_message).lower()
    if "polling timeout" in message or "timeout" in message:
        return "timeout"
    if "too many consecutive errors" in message or "polling failed repeatedly" in message:
        return "transient_api_errors"
    if "auth error" in message or "401" in message or "403" in message or "permission" in message:
        return "auth_or_permission"
    if "space gone" in message or "does not exist" in message or "not accessible" in message:
        return "missing_space"
    if "no sql generated" in message:
        return "no_sql_generated"
    if "unbound parameter" in message:
        return "unbound_parameters"
    if "cancelled" in message:
        return "cancelled"
    if result.status == BenchmarkStatus.FAILED:
        return "sql_mismatch"
    return "runtime_error"


def _benchmark_error_message_text(error_message: object) -> str:
    if error_message is None:
        return ""
    if isinstance(error_message, str):
        return error_message
    try:
        return json.dumps(error_message, sort_keys=True)
    except TypeError:
        return str(error_message)


def _benchmark_issue_details(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for result in results:
        issue_class = _benchmark_issue_class(result)
        if issue_class is None:
            continue
        details.append(
            {
                "question_id": result.question_id,
                "status": result.status.value,
                "issue_class": issue_class,
                "error_message": _benchmark_error_message_text(result.error_message) or None,
            }
        )
    return details


def _benchmark_issue_class_counts(results: list[BenchmarkResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detail in _benchmark_issue_details(results):
        issue_class = str(detail["issue_class"])
        counts[issue_class] = counts.get(issue_class, 0) + 1
    return dict(sorted(counts.items()))


def _run_summary_with_issue_classes(run: BenchmarkRun) -> dict[str, Any]:
    summary = _summarize_run(run)
    issue_details = _benchmark_issue_details(run.results)
    if issue_details:
        summary["issue_class_counts"] = _benchmark_issue_class_counts(run.results)
        summary["issue_details"] = issue_details
    return summary


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_split_pass_rate(summary: dict[str, Any], split: str) -> float | None:
    if not isinstance(summary, dict):
        return None
    if split == "overall":
        split_summary = summary.get("overall")
    else:
        split_summary = summary.get(split)
    if not isinstance(split_summary, dict):
        return None
    return _optional_float(split_summary.get("pass_rate"))


def _summary_split_int(summary: dict[str, Any], split: str, *keys: str) -> int | None:
    if not isinstance(summary, dict):
        return None
    split_summary = summary.get("overall") if split == "overall" else summary.get(split)
    if not isinstance(split_summary, dict):
        return None
    for key in keys:
        value = split_summary.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_count_mapping(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            key = str(key)
        try:
            counts[key] = int(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(counts.items()))


def _format_expected_result_preview(
    cols: list[str], rows: list[list[Any]], *, max_rows: int = 5
) -> str:
    """Render a compact preview of expected answer-key rows for proposer context.

    Used only for visible questions; the caller never passes holdout expected rows.
    """
    if not cols and not rows:
        return ""
    shown = rows[:max_rows]
    body = "; ".join(
        "(" + ", ".join("NULL" if value is None else str(value) for value in row) + ")"
        for row in shown
    )
    suffix = f" (+{len(rows) - max_rows} more rows)" if len(rows) > max_rows else ""
    cols_label = ", ".join(str(col) for col in cols)
    return f"cols=[{cols_label}] rows=[{body}]{suffix}"


# Columns whose unbound ``:parameter`` markers drive the dominant failure family. Probing
# their distinct literal values lets the proposer replace a parameter marker with a real
# literal (e.g. ``time_grp_type = 'WEEKLY'``) instead of leaving it unbound.
_PARAMETER_DRIVING_COLUMNS: tuple[str, ...] = (
    "time_grp_type",
    "metric_type",
    "market_scope",
    "hierarchy_level",
    "team_scope",
)
_PARAMETER_DOMAIN_VALUE_LIMIT = 50
_PARAMETER_DOMAIN_MAX_TABLES = 6
_SQL_TABLE_IDENTIFIER_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(`?[A-Za-z0-9_]+`?(?:\.`?[A-Za-z0-9_]+`?){1,2})",
    re.IGNORECASE,
)


def _clean_sql_identifier(identifier: str) -> str:
    return identifier.replace("`", "").strip()


def _extract_sql_table_identifiers(sql: str | None) -> list[str]:
    """Return qualified table identifiers referenced by FROM/JOIN clauses in *sql*."""
    if not sql:
        return []
    seen: list[str] = []
    for match in _SQL_TABLE_IDENTIFIER_RE.finditer(str(sql)):
        identifier = _clean_sql_identifier(match.group(1))
        if identifier and identifier not in seen:
            seen.append(identifier)
    return seen


def _ordered_candidate_tables(
    sqls: Sequence[str | None], *, max_tables: int = _PARAMETER_DOMAIN_MAX_TABLES
) -> list[str]:
    """Rank candidate tables by reference frequency across *sqls* (most-referenced first)."""
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for sql in sqls:
        for table in _extract_sql_table_identifiers(sql):
            if table not in counts:
                counts[table] = 0
                first_seen[table] = len(first_seen)
            counts[table] += 1
    ordered = sorted(counts, key=lambda table: (-counts[table], first_seen[table]))
    return ordered[:max_tables]


def _probe_parameter_domain_values(
    execute_readonly: Callable[[str], tuple[list[str], list[list[Any]]]],
    tables: Sequence[str],
    columns: Sequence[str] = _PARAMETER_DRIVING_COLUMNS,
    *,
    value_limit: int = _PARAMETER_DOMAIN_VALUE_LIMIT,
) -> dict[str, list[str]]:
    """Probe distinct literal values for each parameter-driving column.

    For each column, tries candidate tables in order and keeps the first table that
    yields rows (where the column exists). Read-only and best-effort: any probe error is
    swallowed so this enrichment never breaks a run. Returns ``{column: [values]}`` only
    for columns that resolved to a non-empty domain.

    Safe for proposer context: it returns column *domains*, never holdout answer rows.
    """
    resolved: dict[str, list[str]] = {}
    for column in columns:
        for table in tables:
            query = (
                f"SELECT DISTINCT {column} FROM {table} "
                f"WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {value_limit}"
            )
            try:
                _cols, rows = execute_readonly(query)
            except Exception:  # noqa: BLE001 - optional enrichment; a bad probe must not break a run
                continue
            values: list[str] = []
            for row in rows:
                if not row:
                    continue
                value = "" if row[0] is None else str(row[0]).strip()
                if value and value not in values:
                    values.append(value)
            if values:
                resolved[column] = values[:value_limit]
                break
    return resolved


def _copytree_for_patch_validation(source_dir: Path, destination_dir: Path) -> list[str]:
    """Copy a decomposed space tree into scratch storage, tolerating vanished entries."""
    missing_paths: list[str] = []
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(source_dir):
        root_path = Path(root)
        relative_root = root_path.relative_to(source_dir)
        target_root = destination_dir / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            source_path = root_path / filename
            target_path = target_root / filename
            try:
                shutil.copy2(source_path, target_path)
            except FileNotFoundError:
                missing_paths.append((relative_root / filename).as_posix())
    return missing_paths


def _terminal_result_selection(
    *,
    optimization_summary: dict[str, Any],
    final_summary: dict[str, Any],
    protocol_preflight: dict[str, Any] | None = None,
    noise_model: dict[str, Any] | None = None,
    transfer_decisions: (
        Mapping[str, PromotionDecision]
        | Sequence[tuple[str, PromotionDecision]]
        | None
    ) = None,
    greedy_best_ever: bool = False,
) -> dict[str, Any]:
    baseline = optimization_summary.get("baseline")
    baseline = baseline if isinstance(baseline, dict) else {}
    baseline_full_rate = _optional_float(baseline.get("full_pass_rate"))
    baseline_holdout_rate = _optional_float(baseline.get("test_rate"))
    final_full_rate = _summary_split_pass_rate(final_summary, "overall")
    final_holdout_rate = _summary_split_pass_rate(final_summary, "test")

    # Greedy best-ever promotion (Round C): ignore any multi-sample noise model and
    # promote on a strict best-ever improvement over the run baseline (no MDD gate),
    # any margin. The holdout non-regression and protocol-comparable gates still apply,
    # so an honest win must still not regress the hidden holdout. Default is off, which
    # preserves the noise-aware behaviour below byte-for-byte.
    #
    # Noise-aware (apple-to-apple) promotion: when a noise model with a positive
    # measured sigma is supplied, the full-score gate uses the minimum detectable
    # difference instead of a fixed epsilon, and multi-sample means replace the
    # single-run rates. Without a noise model the original epsilon gates apply, so
    # default single-sample behaviour is byte-for-byte unchanged.
    noise = None if greedy_best_ever else (noise_model if isinstance(noise_model, dict) else None)
    sigma_full = _optional_float(noise.get("sigma_full")) if noise else None
    noise_active = bool(noise) and sigma_full is not None and sigma_full > 0.0

    reasons: list[str] = []
    promotion_detail: dict[str, Any] | None = None
    mdd_full: float | None = None

    if noise_active:
        baseline_full_mean = _optional_float(noise.get("baseline_full_mean"))
        if baseline_full_mean is None:
            baseline_full_mean = baseline_full_rate
        final_full_mean = _optional_float(noise.get("final_full_mean"))
        if final_full_mean is None:
            final_full_mean = final_full_rate
        baseline_holdout_mean = _optional_float(noise.get("baseline_holdout_mean"))
        if baseline_holdout_mean is None:
            baseline_holdout_mean = baseline_holdout_rate
        final_holdout_mean = _optional_float(noise.get("final_holdout_mean"))
        if final_holdout_mean is None:
            final_holdout_mean = final_holdout_rate
        baseline_n = noise.get("baseline_n")
        candidate_n = noise.get("candidate_n") or 1
        holdout_epsilon = _optional_float(noise.get("holdout_epsilon"))
        if holdout_epsilon is None:
            holdout_epsilon = _TERMINAL_SELECTION_EPSILON
        z = _optional_float(noise.get("z")) or 2.0
        mdd_full = minimum_detectable_difference(sigma_full, z=z)

        # Surface the means as the reported rates for transparency.
        baseline_full_rate = baseline_full_mean if baseline_full_mean is not None else baseline_full_rate
        final_full_rate = final_full_mean if final_full_mean is not None else final_full_rate
        baseline_holdout_rate = baseline_holdout_mean if baseline_holdout_mean is not None else baseline_holdout_rate
        final_holdout_rate = final_holdout_mean if final_holdout_mean is not None else final_holdout_rate

        if baseline_full_mean is None:
            reasons.append("missing_baseline_full_rate")
        elif final_full_mean is None:
            reasons.append("missing_final_full_rate")
        else:
            decision = promotion_decision(
                candidate_full_mean=final_full_mean,
                baseline_full_mean=baseline_full_mean,
                sigma_full=sigma_full,
                candidate_n=int(candidate_n) if candidate_n else 1,
                baseline_n=int(baseline_n) if baseline_n else None,
                candidate_holdout_mean=final_holdout_mean,
                baseline_holdout_mean=baseline_holdout_mean,
                holdout_epsilon=holdout_epsilon,
                z=z,
            )
            promotion_detail = decision.as_dict()
            if not decision.full_significant:
                if decision.full_lift < 0:
                    reasons.append("final_full_regressed_below_baseline")
                else:
                    reasons.append("final_full_within_noise_of_baseline")
            if not decision.holdout_non_regressed:
                reasons.append("final_holdout_regressed_below_baseline")

        # Complementary (not duplicate) holdout guard: promotion_decision above only
        # flags *regression*, and only when both holdout means are present. This guard
        # flags a *missing* holdout rate (which promotion_decision leaves as
        # non_regressed=True). The two layers never emit the same reason code.
        if baseline_holdout_mean is None:
            reasons.append("missing_baseline_holdout_rate")
        elif final_holdout_mean is None:
            reasons.append("missing_final_holdout_rate")
    else:
        if baseline_full_rate is None:
            reasons.append("missing_baseline_full_rate")
        elif final_full_rate is None:
            reasons.append("missing_final_full_rate")
        elif final_full_rate < baseline_full_rate - _TERMINAL_SELECTION_EPSILON:
            reasons.append("final_full_regressed_below_baseline")
        elif final_full_rate <= baseline_full_rate + _TERMINAL_SELECTION_EPSILON:
            reasons.append("final_full_did_not_improve_over_baseline")

        if baseline_holdout_rate is None:
            reasons.append("missing_baseline_holdout_rate")
        elif final_holdout_rate is None:
            reasons.append("missing_final_holdout_rate")
        elif final_holdout_rate < baseline_holdout_rate - _TERMINAL_SELECTION_EPSILON:
            reasons.append("final_holdout_regressed_below_baseline")

    protocol_blocking_reasons: list[str] = []
    if isinstance(protocol_preflight, dict):
        protocol_comparable = bool(protocol_preflight.get("comparable"))
        protocol_status = str(protocol_preflight.get("status") or "")
        errors = protocol_preflight.get("errors")
        if isinstance(errors, list):
            protocol_blocking_reasons = [
                str(issue.get("code"))
                for issue in errors
                if isinstance(issue, dict) and issue.get("code")
            ]
        if protocol_status == "non_comparable" or protocol_comparable is False:
            reasons.append("protocol_non_comparable")

    promotable = not reasons

    # Cross-space robustness (transfer) gate. Inactive by default: with no
    # robustness-space decisions supplied it passes the primary verdict through
    # unchanged. When robustness spaces are configured, a primary win is only kept
    # if it also re-clears promotion on every robustness space (each under its own
    # frozen seed and noise model), so a winner must generalise rather than overfit
    # one space. Composing it here keeps the single-space default byte-for-byte.
    primary_for_gate = PromotionDecision(
        promote=promotable,
        full_lift=(
            (final_full_rate - baseline_full_rate)
            if (final_full_rate is not None and baseline_full_rate is not None)
            else 0.0
        ),
        full_threshold=(mdd_full if mdd_full is not None else _TERMINAL_SELECTION_EPSILON),
        full_significant=promotable,
        holdout_delta=None,
        holdout_non_regressed=True,
        reasons=tuple(reasons),
        detail="primary terminal verdict",
    )
    transfer_gate = evaluate_transfer_gate(
        primary=primary_for_gate,
        transfers=transfer_decisions,
    )
    if transfer_gate.active:
        for reason in transfer_gate.reasons:
            if reason.startswith("transfer_failed:") and reason not in reasons:
                reasons.append(reason)
        promotable = transfer_gate.promote

    candidate_checkpoint_path = optimization_summary.get("best_checkpoint_path")
    baseline_checkpoint_path = optimization_summary.get("baseline_checkpoint_path")
    selected_checkpoint_path = candidate_checkpoint_path if promotable else baseline_checkpoint_path
    if greedy_best_ever:
        policy = (
            "Greedy best-ever promotion: a terminal result may be used as the selected "
            "workspace state when the final full score strictly improves over the run "
            "baseline by any margin (no minimum detectable difference), the final holdout "
            "does not regress, and protocol preflight is comparable."
        )
    elif noise_active:
        policy = (
            "A terminal result may be used as the selected workspace state only when "
            "the final full mean lift over the run baseline exceeds the minimum "
            "detectable difference, final holdout does not regress, and protocol "
            "preflight is comparable."
        )
    else:
        policy = (
            "A terminal result may be used as the selected workspace state only when "
            "final full improves over the run baseline, final holdout does not regress, "
            "and protocol preflight is comparable."
        )
    return {
        "status": "promotable" if promotable else "not_promotable",
        "reasons": reasons,
        "promotion_mode": (
            "greedy_best_ever"
            if greedy_best_ever
            else ("noise_aware" if noise_active else "epsilon")
        ),
        "noise_aware": noise_active,
        "mdd_full": mdd_full,
        "promotion_decision": promotion_detail,
        "transfer_gate": transfer_gate.as_dict(),
        "baseline_full_rate": baseline_full_rate,
        "baseline_holdout_rate": baseline_holdout_rate,
        "final_full_rate": final_full_rate,
        "final_holdout_rate": final_holdout_rate,
        "protocol_status": (
            protocol_preflight.get("status")
            if isinstance(protocol_preflight, dict)
            else None
        ),
        "protocol_blocking_reasons": protocol_blocking_reasons,
        "selected_checkpoint_path": selected_checkpoint_path,
        "candidate_checkpoint_path": candidate_checkpoint_path,
        "baseline_checkpoint_path": baseline_checkpoint_path,
        "policy": policy,
    }


def _generalization_audit(
    *,
    optimization_summary: dict[str, Any],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline = optimization_summary.get("baseline")
    baseline = baseline if isinstance(baseline, dict) else {}
    terminal_selection = optimization_summary.get("terminal_selection")
    terminal_selection = terminal_selection if isinstance(terminal_selection, dict) else {}

    baseline_full_rate = _optional_float(baseline.get("full_pass_rate"))
    baseline_holdout_rate = _optional_float(baseline.get("test_rate"))
    final_full_rate = _summary_split_pass_rate(final_summary, "overall")
    final_train_rate = _summary_split_pass_rate(final_summary, "train")
    final_validation_rate = _summary_split_pass_rate(final_summary, "validation")
    final_holdout_rate = _summary_split_pass_rate(final_summary, "test")

    full_delta = (
        round(final_full_rate - baseline_full_rate, 2)
        if final_full_rate is not None and baseline_full_rate is not None
        else None
    )
    holdout_delta = (
        round(final_holdout_rate - baseline_holdout_rate, 2)
        if final_holdout_rate is not None and baseline_holdout_rate is not None
        else None
    )
    train_holdout_gap = (
        round(final_train_rate - final_holdout_rate, 2)
        if final_train_rate is not None and final_holdout_rate is not None
        else None
    )
    validation_holdout_gap = (
        round(final_validation_rate - final_holdout_rate, 2)
        if final_validation_rate is not None and final_holdout_rate is not None
        else None
    )
    visible_reference_rate = final_validation_rate if final_validation_rate is not None else final_train_rate
    visible_holdout_gap = (
        round(visible_reference_rate - final_holdout_rate, 2)
        if visible_reference_rate is not None and final_holdout_rate is not None
        else None
    )

    holdout_question_count = _summary_split_int(final_summary, "test", "question_count", "total") or 0
    test_summary = final_summary.get("test") if isinstance(final_summary, dict) else None
    test_summary = test_summary if isinstance(test_summary, dict) else {}
    hidden_issue_counts = test_summary.get("issue_class_counts")
    hidden_issue_counts = hidden_issue_counts if isinstance(hidden_issue_counts, dict) else {}

    flags: list[str] = []
    if holdout_question_count and holdout_question_count < _GENERALIZATION_MIN_HOLDOUT_QUESTIONS:
        flags.append("small_hidden_holdout")
    if visible_holdout_gap is not None and visible_holdout_gap >= _GENERALIZATION_MAJOR_GAP_PCT:
        flags.append("visible_holdout_gap")
    if holdout_delta is not None and holdout_delta < -_TERMINAL_SELECTION_EPSILON:
        flags.append("holdout_regressed")
    if final_holdout_rate is None:
        flags.append("missing_hidden_holdout_rate")

    terminal_promotable = terminal_selection.get("status") == "promotable"
    full_improved = full_delta is not None and full_delta > _TERMINAL_SELECTION_EPSILON
    holdout_not_regressed = holdout_delta is None or holdout_delta >= -_TERMINAL_SELECTION_EPSILON
    may_workspace_incumbent = terminal_promotable and full_improved and holdout_not_regressed
    production_golden_candidate = may_workspace_incumbent and not flags

    if production_golden_candidate:
        status = "production_golden_candidate"
    elif may_workspace_incumbent:
        status = "may_workspace_incumbent_not_production_golden"
    elif terminal_promotable:
        status = "promotable_no_full_gain"
    else:
        status = "not_promotable"

    owner_actions: list[str] = []
    if "visible_holdout_gap" in flags:
        owner_actions.append(
            "Use validation-visible failures as proxy targets and avoid generating candidates from hidden holdout content."
        )
    if "small_hidden_holdout" in flags:
        owner_actions.append(
            "Expand the final holdout before treating the run as production-golden evidence."
        )
    if hidden_issue_counts:
        owner_actions.append(
            "Review hidden holdout issue-family counts in this report and map them to broader visible benchmark families."
        )
    if not owner_actions:
        owner_actions.append("Run the production comparison protocol before replacing a standing incumbent.")

    return {
        "schema_version": _GENERALIZATION_AUDIT_SCHEMA_VERSION,
        "status": status,
        "flags": flags,
        "hidden_holdout_policy": (
            "Hidden final holdout question text and SQL are withheld from candidate generation; "
            "only aggregate holdout rates and issue-family counts are surfaced in final reports."
        ),
        "metrics": {
            "baseline_full_rate": baseline_full_rate,
            "baseline_holdout_rate": baseline_holdout_rate,
            "final_full_rate": final_full_rate,
            "final_train_rate": final_train_rate,
            "final_validation_rate": final_validation_rate,
            "final_holdout_rate": final_holdout_rate,
            "full_delta": full_delta,
            "holdout_delta": holdout_delta,
            "train_holdout_gap": train_holdout_gap,
            "validation_holdout_gap": validation_holdout_gap,
            "visible_holdout_gap": visible_holdout_gap,
            "holdout_question_count": holdout_question_count,
        },
        "decisions": {
            "may_workspace_incumbent": may_workspace_incumbent,
            "production_golden_candidate": production_golden_candidate,
            "terminal_selection_status": terminal_selection.get("status"),
            "selected_checkpoint_path": terminal_selection.get("selected_checkpoint_path"),
        },
        "hidden_holdout_summary": {
            "question_count": holdout_question_count,
            "passed": _summary_split_int(final_summary, "test", "passed"),
            "failed": _summary_split_int(final_summary, "test", "failed"),
            "errors": _summary_split_int(final_summary, "test", "errors"),
            "skipped": _summary_split_int(final_summary, "test", "skipped"),
            "pass_rate": final_holdout_rate,
            "issue_class_counts": _int_count_mapping(hidden_issue_counts),
        },
        "visible_proxy_guidance": {
            "source": "train and validation only",
            "minimum_gap_for_proxy_focus": _GENERALIZATION_MAJOR_GAP_PCT,
            "active": visible_holdout_gap is not None and visible_holdout_gap >= _GENERALIZATION_MAJOR_GAP_PCT,
        },
        "owner_actions": owner_actions,
    }


def _required_pass_count(*, min_pass_rate: float | None, question_count: int) -> int | None:
    if min_pass_rate is None or question_count <= 0:
        return None
    return max(0, min(question_count, math.ceil((min_pass_rate / 100.0) * question_count - 1e-9)))


def _clone_title_base(title: str, *, fallback: str) -> str:
    stripped = _OPTIMIZER_TITLE_PREFIX_RE.sub("", title).strip()
    return stripped or fallback


def _formatted_clone_title(
    title: str,
    *,
    full_pass_rate: float | None = None,
    fallback: str,
    strategy_label: str | None = None,
) -> str:
    base_title = _clone_title_base(title, fallback=fallback)
    tag = _OPTIMIZER_TITLE_TAG
    if strategy_label:
        tag = _OPTIMIZER_TITLE_TAG
    if full_pass_rate is None:
        return f"[{tag}] {base_title}"
    score_text = (
        f"{full_pass_rate:.0f}%"
        if abs(full_pass_rate - round(full_pass_rate)) < 0.005
        else f"{full_pass_rate:.2f}%"
    )
    return f"[{tag} {score_text}] {base_title}"


def _genie_space_url_from_host(host: Any, space_id: str | None) -> str | None:
    if not host or not space_id:
        return None
    host_text = str(host).strip().rstrip("/")
    if not host_text:
        return None
    if not host_text.startswith(("http://", "https://")):
        host_text = f"https://{host_text}"
    return f"{host_text}/genie/rooms/{space_id}"


def _build_train_diff(previous: BenchmarkRun | None, current: BenchmarkRun) -> dict[str, list[str]]:
    previous_by_id = {result.question_id: result.status for result in previous.results} if previous else {}
    current_by_id = {result.question_id: result.status for result in current.results}

    fixed: list[str] = []
    regressed: list[str] = []
    still_failing: list[str] = []
    new_failures: list[str] = []

    for question_id, status in current_by_id.items():
        previous_status = previous_by_id.get(question_id)
        if previous_status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR} and status == BenchmarkStatus.PASSED:
            fixed.append(question_id)
        elif previous_status == BenchmarkStatus.PASSED and status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}:
            regressed.append(question_id)
        elif previous_status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR} and status in {
            BenchmarkStatus.FAILED,
            BenchmarkStatus.ERROR,
        }:
            still_failing.append(question_id)
        elif previous_status is None and status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}:
            new_failures.append(question_id)

    return {
        "fixed": sorted(fixed),
        "regressed": sorted(regressed),
        "still_failing": sorted(still_failing),
        "new_failures": sorted(new_failures),
        "known_failures_after": sorted(
            question_id
            for question_id, status in current_by_id.items()
            if status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}
        ),
    }


def _accepted_fix_regression_ids(
    *,
    baseline_run: BenchmarkRun | None,
    reference_run: BenchmarkRun | None,
    current_run: BenchmarkRun,
) -> list[str]:
    if baseline_run is None or reference_run is None:
        return []

    baseline_by_id = {result.question_id: result.status for result in baseline_run.results}
    reference_by_id = {result.question_id: result.status for result in reference_run.results}
    current_by_id = {result.question_id: result.status for result in current_run.results}

    regressed_ids = []
    for question_id, current_status in current_by_id.items():
        if current_status not in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}:
            continue
        if reference_by_id.get(question_id) != BenchmarkStatus.PASSED:
            continue
        if baseline_by_id.get(question_id) not in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}:
            continue
        regressed_ids.append(question_id)
    return sorted(regressed_ids)


def _question_payload(question: BenchmarkQuestion) -> dict[str, Any]:
    return {
        "id": question.id,
        "question": question.question,
        "expected_sql": question.expected_sql,
    }



def _merge_benchmark_runs(
    *,
    ordered_questions: list[BenchmarkQuestion],
    runs: list[BenchmarkRun],
    carry_forward_run: BenchmarkRun | None = None,
) -> BenchmarkRun:
    if not runs:
        raise WorkspaceFlowError("Cannot merge an empty benchmark run set.")

    results_by_id = {}
    if carry_forward_run is not None:
        for result in carry_forward_run.results:
            results_by_id[result.question_id] = result
    for run in runs:
        for result in run.results:
            results_by_id[result.question_id] = result

    missing_ids = [question.id for question in ordered_questions if question.id not in results_by_id]
    if missing_ids:
        sample = ", ".join(missing_ids[:3])
        raise WorkspaceFlowError(
            f"Merged train benchmark is missing results for: {sample}"
        )

    merged = BenchmarkRun(
        space_id=runs[-1].space_id,
        started_at=min(run.started_at for run in runs),
        completed_at=max(
            (run.completed_at or run.started_at)
            for run in runs
        ),
    )
    merged.results = [results_by_id[question.id] for question in ordered_questions]
    return merged


def _representative_multi_sample_result(
    *,
    aggregate_status: BenchmarkStatus,
    sample_results: list[BenchmarkResult],
) -> BenchmarkResult | None:
    if not sample_results:
        return None

    if aggregate_status == BenchmarkStatus.PASSED:
        preferred_statuses = (BenchmarkStatus.PASSED,)
    elif aggregate_status == BenchmarkStatus.ERROR:
        preferred_statuses = (BenchmarkStatus.ERROR, BenchmarkStatus.FAILED)
    else:
        preferred_statuses = (BenchmarkStatus.FAILED, BenchmarkStatus.ERROR)

    candidates = [
        result for result in sample_results if result.status in set(preferred_statuses)
    ] or sample_results
    return next((result for result in candidates if result.comparison_method), candidates[0])


def _multi_sample_comparison_details(
    *,
    representative: BenchmarkResult | None,
    pass_count: int,
    sample_count: int,
    observed_count: int,
    pass_fraction: float,
    pass_fraction_threshold: float,
) -> str:
    detail = (
        "Multi-sample population baseline: "
        f"{pass_count}/{sample_count} samples passed "
        f"(observed {observed_count}/{sample_count}; "
        f"pass_fraction={pass_fraction:.4f}; "
        f"pass requires > {pass_fraction_threshold:.4f})."
    )
    if representative and representative.comparison_details:
        detail = f"{detail}\nRepresentative sample: {representative.comparison_details}"
    return detail


def _aggregate_sample_runs_for_baseline(
    *,
    sample_runs: Sequence[BenchmarkRun],
    questions: list[BenchmarkQuestion],
    pass_fraction_threshold: float = _MULTI_SAMPLE_BASELINE_PASS_FRACTION_THRESHOLD,
) -> BenchmarkRun:
    runs = list(sample_runs)
    if not runs:
        raise WorkspaceFlowError("Cannot initialize a multi-sample baseline without samples.")

    sample_count = len(runs)
    results_by_question: dict[str, list[BenchmarkResult]] = {}
    for run in runs:
        for result in run.results:
            results_by_question.setdefault(result.question_id, []).append(result)

    aggregate_run = BenchmarkRun(
        space_id=runs[-1].space_id,
        started_at=min(run.started_at for run in runs),
        completed_at=max((run.completed_at or run.started_at) for run in runs),
    )
    aggregate_results: list[BenchmarkResult] = []
    for question in questions:
        sample_results = results_by_question.get(question.id, [])
        pass_count = sum(
            1 for result in sample_results if result.status == BenchmarkStatus.PASSED
        )
        failed_count = sum(
            1 for result in sample_results if result.status == BenchmarkStatus.FAILED
        )
        error_count = sum(
            1 for result in sample_results if result.status == BenchmarkStatus.ERROR
        )
        missing_count = max(sample_count - len(sample_results), 0)
        pass_fraction = pass_count / sample_count if sample_count else 0.0
        if pass_fraction > pass_fraction_threshold:
            aggregate_status = BenchmarkStatus.PASSED
        elif error_count + missing_count > failed_count:
            aggregate_status = BenchmarkStatus.ERROR
        else:
            aggregate_status = BenchmarkStatus.FAILED

        representative = _representative_multi_sample_result(
            aggregate_status=aggregate_status,
            sample_results=sample_results,
        )
        if representative is None:
            error_message = "Missing from all population baseline samples."
            generated_sql = None
            comparison_method = None
        else:
            error_message = representative.error_message
            generated_sql = representative.generated_sql
            comparison_method = representative.comparison_method

        aggregate_results.append(
            BenchmarkResult(
                question_id=question.id,
                question=question.question,
                status=aggregate_status,
                generated_sql=generated_sql,
                expected_sql=(
                    representative.expected_sql
                    if representative and representative.expected_sql is not None
                    else question.expected_sql
                ),
                error_message=error_message,
                duration_seconds=(
                    sum(result.duration_seconds for result in sample_results) / len(sample_results)
                    if sample_results
                    else 0.0
                ),
                conversation_id=representative.conversation_id if representative else None,
                message_id=representative.message_id if representative else None,
                comparison_score=round(pass_fraction, 4),
                comparison_method=comparison_method,
                comparison_details=_multi_sample_comparison_details(
                    representative=representative,
                    pass_count=pass_count,
                    sample_count=sample_count,
                    observed_count=len(sample_results),
                    pass_fraction=pass_fraction,
                    pass_fraction_threshold=pass_fraction_threshold,
                ),
            )
        )
    aggregate_run.results = aggregate_results
    return aggregate_run


def _multi_sample_baseline_metadata(
    *,
    aggregate: SampleAggregate,
    population_label: str,
    pass_fraction_threshold: float,
    question_ids: Sequence[str] | None = None,
    include_per_question: bool = False,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": _MULTI_SAMPLE_BASELINE_SOURCE,
        "population_label": population_label,
        "n": aggregate.n,
        "mean_pass_rate": round(aggregate.mean_pass_rate, 4),
        "sample_std": round(aggregate.sample_std, 4),
        "pass_fraction_threshold": round(pass_fraction_threshold, 4),
        "decision_rule": "question passes the aggregate baseline when pass_fraction > threshold",
    }
    if not include_per_question:
        return metadata

    ordered_ids = list(dict.fromkeys(str(question_id) for question_id in (question_ids or [])))
    metadata["per_question_pass_fraction"] = {
        question_id: round(
            int(aggregate.per_question_pass_counts.get(question_id, 0)) / aggregate.n,
            4,
        )
        for question_id in ordered_ids
    }
    metadata["per_question_pass_counts"] = {
        question_id: int(aggregate.per_question_pass_counts.get(question_id, 0))
        for question_id in ordered_ids
    }
    metadata["per_question_total_counts"] = {
        question_id: aggregate.n
        for question_id in ordered_ids
    }
    return metadata



def _merge_editable_sections(
    base_config: GenieSpaceConfig,
    editable_config: GenieSpaceConfig,
) -> GenieSpaceConfig:
    merged = deepcopy(base_config)
    merged.version = editable_config.version
    merged.title = editable_config.title
    merged.description = editable_config.description
    merged.warehouse_id = editable_config.warehouse_id
    merged.parent_path = base_config.parent_path
    merged.config = editable_config.config
    merged.data_sources = editable_config.data_sources
    merged.instructions = editable_config.instructions
    return merged


def _config_with_local_benchmarks(
    artifacts: ArtifactManager,
    config: GenieSpaceConfig,
) -> GenieSpaceConfig:
    if config.benchmarks and config.benchmarks.questions:
        return config

    payloads = [
        artifacts.load_benchmark_payload(),
        artifacts.load_hidden_test_payload(),
    ]
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        if payload is None:
            continue
        payload_rows = payload.get("questions")
        if isinstance(payload_rows, list):
            rows.extend(row for row in payload_rows if isinstance(row, dict))

    if not rows:
        return config

    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
    for row in rows:
        question_id = str(row["id"])
        if question_id in seen_ids:
            continue
        seen_ids.add(question_id)
        questions.append(
            BenchmarkQuestion(
                id=question_id,
                question=str(row["question"]),
                expected_sql=str(row["expected_sql"]) if row.get("expected_sql") is not None else None,
            )
        )

    merged = deepcopy(config)
    merged.benchmarks = benchmark_questions_to_model(questions)
    return merged


def _editable_serialized_space(config: GenieSpaceConfig) -> dict[str, Any]:
    editable = deepcopy(config)
    editable.benchmarks = None
    return editable.to_serialized_space()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _git_state(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }

    changed_files: list[str] = []
    for line in status:
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        changed_files.append(path)

    return {
        "available": True,
        "repo_root": str(root),
        "branch": branch or None,
        "commit_sha": commit,
        "dirty": bool(changed_files),
        "changed_files": changed_files,
    }


def _is_missing_space_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "unable to get space" in message or "does not exist" in message


def _normalize_clone_parent_path(parent_path: str | None) -> str | None:
    if parent_path is None:
        return None
    cleaned = str(parent_path).strip()
    if not cleaned:
        return None
    if cleaned == "/Workspace":
        return "/"
    if cleaned.startswith("/Workspace/"):
        cleaned = cleaned[len("/Workspace"):]
    if len(cleaned) > 1:
        cleaned = cleaned.rstrip("/")
    return cleaned or None


def _resolve_result_content_match(explicit: bool | None) -> bool:
    """Effective content-based result-comparison flag.

    An explicit argument always wins. Otherwise the campaign switch
    ``MAXGENIE_RESULT_CONTENT_MATCH`` (truthy: ``1/true/yes/y/on``) is read. The
    default is off so external runs keep the byte-exact result comparator until
    they opt in; the optimization campaign turns it on to make the fair,
    value-first content comparator its ruler. Enabling it opens a new
    comparability epoch, so a run must re-baseline before ranking under it.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(_RESULT_CONTENT_MATCH_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _example_parameter_default_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, list) and values:
            return _example_parameter_default_value(values[0])
        for key in ("value", "default_value"):
            if key in value:
                return _example_parameter_default_value(value.get(key))
        return None
    if isinstance(value, list):
        if not value:
            return None
        return _example_parameter_default_value(value[0])
    return str(value)


def _current_user_clone_parent_path(wc: Any) -> str | None:
    current_user = getattr(wc, "current_user", None)
    if current_user is None:
        return None
    try:
        me = current_user.me()
    except Exception:
        return None
    user_name = getattr(me, "user_name", None) or getattr(me, "userName", None)
    if not user_name:
        return None
    return f"/Users/{str(user_name).strip()}"


def _is_missing_space_chain(exc: BaseException | None) -> bool:
    current = exc
    while current is not None:
        if isinstance(current, Exception) and _is_missing_space_error(current):
            return True
        current = current.__cause__
    return False


def _has_missing_space_error_results(run: BenchmarkRun) -> bool:
    return any(
        result.status == BenchmarkStatus.ERROR
        and result.error_message
        and _is_missing_space_error(RuntimeError(result.error_message))
        for result in run.results
    )


def _looks_like_sql_snippet_option_menu(content_text: str) -> bool:
    if _SQL_SNIPPET_OPTION_SET_RE.search(content_text):
        return True
    sql_lines = _yaml_sql_block_lines(content_text)
    nonempty_lines = [line.strip().rstrip(";") for line in sql_lines if line.strip()]
    if len(nonempty_lines) < 2:
        return False
    independent_predicate_lines = []
    for line in nonempty_lines:
        if not _SQL_PREDICATE_OPERATOR_RE.search(line):
            independent_predicate_lines.append(False)
            continue
        if re.search(r"\b(?:and|or)\s*$", line, re.IGNORECASE):
            independent_predicate_lines.append(False)
            continue
        if re.match(r"^(?:and|or)\b|^[,(]", line, re.IGNORECASE):
            independent_predicate_lines.append(False)
            continue
        independent_predicate_lines.append(True)
    return bool(independent_predicate_lines) and all(independent_predicate_lines)


def _looks_like_literal_swap_guard_clone(evidence_text: str) -> bool:
    compact_text = re.sub(r"\s+", " ", evidence_text).strip()
    if not compact_text:
        return False
    return bool(_LITERAL_SWAP_GUARD_CLONE_RE.search(compact_text))


def _literal_swap_question_terms(text: str) -> set[str]:
    return question_overlap_terms(text)


def _structural_sql_terms(sql_text: str) -> set[str]:
    without_strings = re.sub(r"'(?:''|[^'])*'", "'?'", sql_text.lower())
    without_numbers = re.sub(r"\b\d+(?:\.\d+)?\b", "0", without_strings)
    terms = set(re.findall(r"[a-z_][a-z0-9_]*", without_numbers))
    return {
        term
        for term in terms
        if term
        not in {
            "as",
            "and",
            "or",
            "on",
            "by",
            "in",
            "is",
            "not",
            "null",
        }
    }


def _sql_structural_similarity(left_sql: str, right_sql: str) -> float:
    left_terms = _structural_sql_terms(left_sql)
    right_terms = _structural_sql_terms(right_sql)
    if len(left_terms) < 6 or len(right_terms) < 6:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms), len(right_terms))


def _extract_example_question_sql(content_text: str) -> tuple[str, str] | None:
    try:
        parsed = yaml.safe_load(content_text)
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    question = parsed.get("question")
    sql = parsed.get("sql")
    if isinstance(question, list):
        question_text = " ".join(str(item) for item in question)
    else:
        question_text = str(question or "")
    if isinstance(sql, list):
        sql_text = "\n".join(str(item) for item in sql)
    else:
        sql_text = str(sql or "")
    if not question_text.strip() or not sql_text.strip():
        return None
    return question_text, sql_text


def _parse_artifact_patch_yaml_mapping(
    content_text: str,
    *,
    invalid_reason: str,
    non_mapping_reason: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        parsed = yaml.safe_load(content_text)
    except yaml.YAMLError:
        return None, [invalid_reason]
    if not isinstance(parsed, dict):
        return None, [non_mapping_reason]
    return parsed, []


def _artifact_patch_required_stringish_field_reasons(
    data: dict[str, Any],
    *,
    field_name: str,
    missing_reason: str,
    type_reason: str,
) -> list[str]:
    if field_name not in data or data[field_name] is None:
        return [missing_reason]
    value = data[field_name]
    if isinstance(value, str):
        return [] if value.strip() else [missing_reason]
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return [type_reason]
        return [] if any(str(item).strip() for item in value) else [missing_reason]
    return [type_reason]


def _artifact_patch_stringish_field_is_nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return False
        return any(str(item).strip() for item in value)
    return False


def _artifact_patch_unexpected_content_wrapper_reasons(
    data: dict[str, Any],
    *,
    reason: str,
) -> list[str]:
    return [reason] if "content" in data else []


def _is_artifact_patch_full_file_content_edit(edit: dict[str, Any]) -> bool:
    return isinstance(edit.get("content"), str) and bool(edit.get("content"))


def _literal_swap_guard_clone_reasons(
    *,
    question: str,
    sql: str,
    pass_guards: list[dict[str, Any]],
) -> list[str]:
    candidate_terms = _literal_swap_question_terms(question)
    if not candidate_terms:
        return []

    reasons: list[str] = []
    for guard in pass_guards:
        guard_question = str(guard.get("question") or "")
        guard_sql = str(guard.get("expected_sql") or "")
        if not guard_question.strip() or not guard_sql.strip():
            continue
        guard_terms = _literal_swap_question_terms(guard_question)
        shared_terms = candidate_terms & guard_terms
        candidate_only = candidate_terms - guard_terms
        guard_only = guard_terms - candidate_terms
        if not shared_terms or not candidate_only or not guard_only:
            continue
        if len(candidate_only) > 2 or len(guard_only) > 2:
            continue
        if _sql_structural_similarity(sql, guard_sql) < 0.9:
            continue
        guard_id = str(guard.get("question_id") or "unknown")
        reasons.append(f"executable_artifact_patch_example_guard_sql_clone:{guard_id}")
    return reasons


def _yaml_sql_block_lines(content_text: str) -> list[str]:
    lines = content_text.splitlines()
    output: list[str] = []
    in_sql = False
    for line in lines:
        if re.match(r"^\s*sql\s*:\s*(?:[>|].*)?$", line, re.IGNORECASE):
            in_sql = True
            continue
        if in_sql and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*\s*:", line):
            break
        if in_sql:
            output.append(line.strip())
    return output if output else lines


@dataclass(frozen=True)
class CandidateSpec:
    """One benchmark-gated mutation family."""

    name: str
    family: str
    description: str
    suppression_keys: tuple[str, ...] = ()
    candidate_mode: str | None = None


def _candidate_suppression_reason_code(reason: str | None) -> str | None:
    if not reason:
        return None
    lowered = reason.lower()
    if lowered.startswith("train_health_failed:"):
        return "train_health_failed"
    if "train regressed" in lowered:
        return "train_regressed"
    if "known failures did not improve" in lowered:
        return "known_failures_not_improved"
    if lowered.startswith("push_failed:") and any(
        token in lowered for token in ("proto", "incompatible", "invalid", "unsupported")
    ):
        return "proto_incompatible"
    return None


def _candidate_suppression_entry(
    *,
    spec: CandidateSpec,
    reason: str,
    reason_code: str,
    label: str,
    candidate_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "name": spec.name,
        "family": spec.family,
        "suppression_keys": list(spec.suppression_keys),
        "reason": reason,
        "reason_code": reason_code,
        "label": label,
        "candidate_summary": candidate_summary,
    }


def _matching_suppression_entry(
    spec: CandidateSpec,
    suppressed_candidates: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not suppressed_candidates:
        return None
    for key in spec.suppression_keys:
        entry = suppressed_candidates.get(key)
        if isinstance(entry, dict):
            return entry
    return None


class OptimizationWorkspace:
    """High-level workflow for the CLI-driven optimization loop."""

    def __init__(
        self,
        *,
        genie_service: GenieService,
        benchmark_service: BenchmarkService,
        artifacts: ArtifactManager,
        exporter: SpaceExporter,
        delay_seconds: float = 12.0,
        match_threshold: float = 0.85,
        warehouse_id: str | None = None,
        mlflow_tracker: MlflowTrackerProtocol | None = None,
        space_config: GenieSpaceConfig | None = None,
        clone_parent_path: str | None = None,
        result_content_match: bool | None = None,
    ):
        self.genie_service = genie_service
        self.benchmark_service = benchmark_service
        self.artifacts = artifacts
        self.exporter = exporter
        self.delay_seconds = delay_seconds
        self.match_threshold = match_threshold
        self.policy_candidates = {policy.name: policy for policy in default_policies()}
        self.centaur_prepass_candidates = {policy.name: policy for policy in centaur_prepass_policies()}
        # Fair, value-first result comparison (Phase A). Off by default; opt in via the
        # ``MAXGENIE_RESULT_CONTENT_MATCH`` env switch or an explicit argument. This is the
        # campaign default for the tournament; landing it opens a new comparability epoch.
        self._result_content_match = _resolve_result_content_match(result_content_match)
        if warehouse_id:
            _param_values: dict[str, Any] = {}
            for ex in ((space_config.instructions.example_question_sqls if space_config and space_config.instructions else None) or []):
                for p in (ex.parameters or []):
                    default_value = _example_parameter_default_value(p.default_value)
                    if default_value is None:
                        continue
                    _param_values.setdefault(p.name, default_value)
            self.comparator: ResultComparator | None = ResultComparator(
                wc=self.genie_service.wc,
                warehouse_id=warehouse_id,
                parameter_values=_param_values or None,
                answer_key_path=self.artifacts.answer_key_path,
                content_match=self._result_content_match,
            )
        else:
            self.comparator = None
        # Result-based comparison is mandatory whenever a SQL warehouse is configured
        # for the run. benchmark_service then refuses to silently fall back to
        # string/token comparison (a non-comparable score) and raises instead.
        self._require_result_based = bool(warehouse_id)
        # Greedy best-ever promotion (Round C): when enabled, terminal selection ignores
        # the multi-sample noise model and promotes on any strict best-ever improvement
        # over baseline (no MDD), still gated on holdout non-regression and protocol
        # comparability. ``_confirm_on_beat`` optionally re-runs an accepted candidate's
        # visible benchmark N times as lighter anti-noise insurance. Both default off so
        # existing noise-aware behaviour is unchanged; the run entrypoint sets them.
        self._greedy_best_ever_promotion = False
        self._confirm_on_beat = 0
        # Probed once per run (lazily) and reused: distinct literal/domain values of the
        # parameter-driving columns, used to help the proposer literalize unbound
        # ``:parameter`` markers. Visible-safe (column domains, not holdout answer rows).
        self._parameter_domain_values_cache: dict[str, list[str]] | None = None
        self._strategy_label: str | None = None
        self._clone_parent_path = _normalize_clone_parent_path(clone_parent_path)
        # Injected tracker; None means build from env vars per run (opt-in via MAXGENIE_MLFLOW_ENABLED).
        self._injected_mlflow_tracker = mlflow_tracker

    def _resolve_clone_parent_path(self, source_config: GenieSpaceConfig) -> tuple[str | None, str]:
        if self._clone_parent_path:
            return self._clone_parent_path, "argument"

        env_parent_path = _normalize_clone_parent_path(os.environ.get(_CLONE_PARENT_PATH_ENV))
        if env_parent_path:
            return env_parent_path, f"env:{_CLONE_PARENT_PATH_ENV}"

        current_user_parent_path = _current_user_clone_parent_path(getattr(self.genie_service, "wc", None))
        if current_user_parent_path:
            return current_user_parent_path, "current_user"

        source_parent_path = _normalize_clone_parent_path(source_config.parent_path)
        if source_parent_path:
            return source_parent_path, "source_space_config"

        return None, "api_default"

    def _build_mlflow_tracker(self) -> MlflowTrackerProtocol:
        if self._injected_mlflow_tracker is not None:
            return self._injected_mlflow_tracker
        return tracker_from_env()

    def _expected_benchmark_count(self) -> int:
        payload = self.artifacts.load_benchmark_payload() or {}
        question_count = payload.get("question_count")
        if isinstance(question_count, int) and question_count > 0:
            return question_count
        questions = payload.get("questions")
        if isinstance(questions, list) and questions:
            return len(questions)
        return 15

    def _expected_protocol_benchmark_count(self, summary: dict[str, Any]) -> int:
        candidates = [self._expected_benchmark_count()]
        overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
        for key in ("question_count", "total"):
            value = overall.get(key)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        comparison_metrics = (
            overall.get("comparison_metrics")
            if isinstance(overall.get("comparison_metrics"), dict)
            else {}
        )
        for key in ("scored_question_count", "result_based_question_count"):
            value = comparison_metrics.get(key)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        return max(candidates)

    def _write_protocol_preflight(
        self,
        *,
        mode: str,
        strategy: str,
        summary: dict[str, Any] | None,
        serving_endpoint: str | None = None,
        serving_model: str | None = None,
        serving_reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        workspace_meta = self.artifacts.load_workspace_meta() or {}
        split_meta = _load_json_dict(self.artifacts.split_meta_path) or {}
        summary = summary if isinstance(summary, dict) else {}
        evaluation_context = (
            summary.get("evaluation_context")
            if isinstance(summary.get("evaluation_context"), dict)
            else _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            )
        )
        overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
        comparison_metrics = (
            overall.get("comparison_metrics")
            if isinstance(overall.get("comparison_metrics"), dict)
            else {}
        )
        preferred_comparison = str(
            evaluation_context.get("preferred_comparison_mode")
            or ("result_based" if self.comparator is not None else "string_only")
        )
        result_based_count = comparison_metrics.get("result_based_question_count")
        scored_count = (
            comparison_metrics.get("scored_question_count")
            or overall.get("question_count")
            or overall.get("total")
        )
        expected_benchmark_count = self._expected_protocol_benchmark_count(summary)
        payload = {
            "mode": mode,
            "strategy": strategy,
            "comparison_mode": preferred_comparison,
            "warehouse_id": evaluation_context.get("warehouse_id"),
            "benchmark_coverage": {
                "scored": scored_count,
                "expected": expected_benchmark_count,
                "result_based": result_based_count,
                "comparison_method_counts": comparison_metrics.get("comparison_method_counts"),
            },
            "result_based_question_count": result_based_count,
            "comparison_method_counts": comparison_metrics.get("comparison_method_counts"),
            "scored_questions": comparison_metrics.get("scored_question_count"),
            "final_evidence": True,
            "split_strategy": split_meta.get("split_strategy"),
            "fixed_split": bool(split_meta.get("split_strategy")),
            "fresh_baseline": workspace_meta.get("fresh_baseline"),
            "serving_endpoint": serving_endpoint or workspace_meta.get("serving_endpoint"),
            "serving_model": serving_model or workspace_meta.get("serving_model"),
            "serving_reasoning_effort": (
                serving_reasoning_effort
                or workspace_meta.get("serving_reasoning_effort")
            ),
            "observed_benchmark_reuse_policy": workspace_meta.get(
                "observed_benchmark_reuse_policy"
            ),
        }
        result = validate_protocol_preflight(
            payload,
            expected_benchmark_count=expected_benchmark_count,
        ).to_dict()
        result["payload"] = payload
        eligibility = {
            "promotion_eligible": bool(result.get("comparable")),
            "status": result.get("status"),
            "blocking_reasons": [
                issue.get("code")
                for issue in result.get("errors", [])
                if isinstance(issue, dict)
            ],
            "warning_reasons": [
                issue.get("code")
                for issue in result.get("warnings", [])
                if isinstance(issue, dict)
            ],
        }
        result["promotion_eligibility"] = eligibility
        self.artifacts.write_json_path(self.artifacts.protocol_preflight_path, result)
        self.artifacts.write_json_path(self.artifacts.promotion_eligibility_path, eligibility)
        self.artifacts.record_iteration(
            {
                "phase": "protocol_preflight",
                "mode": mode,
                "strategy": strategy,
                "status": result.get("status"),
                "promotion_eligible": eligibility["promotion_eligible"],
                "blocking_reasons": eligibility["blocking_reasons"],
            }
        )
        return result

    def export_workspace(
        self,
        source_space_id: str,
        *,
        validation_fraction: float,
        test_fraction: float,
        split_seed: int,
        allow_empty_benchmarks: bool = False,
    ) -> ExportBundle:
        workspace_host = getattr(getattr(getattr(self.genie_service, "wc", None), "config", None), "host", None)
        try:
            bundle = self.exporter.export(
                source_space_id,
                workspace_host=workspace_host,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                split_seed=split_seed,
            )
        except Exception as exc:
            legacy_candidates = (
                self.artifacts.space_dir / "current.space.yml",
                self.artifacts.space_dir / "original.space.yml",
            )
            if any(path.exists() for path in legacy_candidates):
                try:
                    bundle = self.exporter.export_from_local_files(
                        source_space_id,
                        workspace_host=workspace_host,
                        validation_fraction=validation_fraction,
                        test_fraction=test_fraction,
                        split_seed=split_seed,
                    )
                    self.artifacts.record_iteration(
                        {
                            "phase": "legacy_export_fallback",
                            "source_space_id": source_space_id,
                            "reason": str(exc),
                        }
                    )
                except Exception as fallback_exc:
                    raise WorkspaceFlowError(
                        f"Failed to export source space `{source_space_id}` and failed to migrate the legacy local workspace: {fallback_exc}"
                    ) from fallback_exc
            else:
                raise WorkspaceFlowError(
                    f"Failed to export source space `{source_space_id}`: {exc}"
                ) from exc
        if not self.artifacts.blocked_failures_path.exists():
            self.artifacts.write_text_path(
                self.artifacts.blocked_failures_path,
                "# Blocked Failures\n\n",
            )
        if not bundle.questions and not allow_empty_benchmarks:
            raise WorkspaceFlowError(
                "Space has no benchmark set and no compatible local benchmark workspace was found."
            )
        self.artifacts.record_iteration(
            {
                "phase": "export",
                "source_space_id": source_space_id,
                "train_count": len(bundle.split.train_questions),
                "validation_count": len(bundle.split.validation_questions),
                "test_count": len(bundle.split.test_questions),
                "benchmark_source": bundle.benchmark_source,
                "prompt_path": str(bundle.prompt_path),
            }
        )
        return bundle

    def apply_benchmark_artifact_seed(
        self,
        *,
        bundle: ExportBundle,
        seed_path: str | Path,
    ) -> ExportBundle:
        """Reuse an already-exported benchmark payload and split exactly."""
        seed_root = Path(seed_path).expanduser()
        required_files = {
            "benchmark_payload": seed_root / ".benchmark_payload.json",
            "hidden_test_payload": seed_root / ".hidden_test_payload.json",
            "train_set": seed_root / "benchmarks" / "train_set.json",
            "validation_set": seed_root / "benchmarks" / "validation_set.json",
            "validation_folds": seed_root / "benchmarks" / "validation_folds.json",
            "test_set": seed_root / "benchmarks" / "test_set.json",
            "split_meta": seed_root / "benchmarks" / "split_meta.json",
        }
        missing = [str(path) for path in required_files.values() if not path.exists()]
        if missing:
            raise WorkspaceFlowError(
                "Fixed benchmark artifact seed is missing required files: "
                + ", ".join(missing)
            )

        benchmark_payload = _load_json_dict(required_files["benchmark_payload"])
        hidden_payload = _load_json_dict(required_files["hidden_test_payload"])
        train_payload = _load_json_dict(required_files["train_set"])
        validation_payload = _load_json_dict(required_files["validation_set"])
        validation_folds_payload = _load_json_dict(required_files["validation_folds"])
        test_payload = _load_json_dict(required_files["test_set"])
        split_meta = _load_json_dict(required_files["split_meta"])
        if not all(
            isinstance(payload, dict)
            for payload in (
                benchmark_payload,
                hidden_payload,
                train_payload,
                validation_payload,
                validation_folds_payload,
                test_payload,
                split_meta,
            )
        ):
            raise WorkspaceFlowError(
                "Fixed benchmark artifact seed contains invalid JSON payloads."
            )

        train_questions = _questions_from_payload(train_payload or {}, label="train_set")
        visible_questions = _questions_from_payload(
            benchmark_payload or {},
            label="benchmark_payload",
        )
        hidden_questions = _questions_from_payload(
            hidden_payload or {},
            label="hidden_test_payload",
        )
        visible_by_id = {question.id: question for question in visible_questions}
        hidden_by_id = {question.id: question for question in hidden_questions}
        validation_ids = [
            str(value)
            for value in (validation_payload or {}).get("ids", [])
            if str(value).strip()
        ]
        test_ids = [
            str(value)
            for value in (test_payload or {}).get("ids", [])
            if str(value).strip()
        ]
        train_ids = [question.id for question in train_questions]
        missing_validation_ids = [
            question_id for question_id in validation_ids if question_id not in visible_by_id
        ]
        missing_test_ids = [
            question_id for question_id in test_ids if question_id not in hidden_by_id
        ]
        if missing_validation_ids or missing_test_ids:
            raise WorkspaceFlowError(
                "Fixed benchmark artifact seed split ids do not match payload questions."
            )

        validation_questions = [visible_by_id[question_id] for question_id in validation_ids]
        test_questions = [hidden_by_id[question_id] for question_id in test_ids]
        visible_ids = {question.id for question in visible_questions}
        missing_train_ids = [
            question_id for question_id in train_ids if question_id not in visible_ids
        ]
        if missing_train_ids:
            raise WorkspaceFlowError(
                "Fixed benchmark artifact seed train ids do not match visible payload questions."
            )

        validation_folds: list[ValidationFold] = []
        for fold_payload in (validation_folds_payload or {}).get("folds", []):
            if not isinstance(fold_payload, dict):
                raise WorkspaceFlowError(
                    "Fixed benchmark artifact seed has an invalid validation fold."
                )
            try:
                fold_index = int(fold_payload.get("fold_index"))
            except (TypeError, ValueError) as exc:
                raise WorkspaceFlowError(
                    "Fixed benchmark artifact seed has an invalid validation fold index."
                ) from exc
            validation_folds.append(
                ValidationFold(
                    fold_index=fold_index,
                    train_ids=[str(value) for value in fold_payload.get("train_ids", [])],
                    validation_ids=[
                        str(value) for value in fold_payload.get("validation_ids", [])
                    ],
                )
            )

        seeded_split = BenchmarkSplit(
            train_questions=train_questions,
            dev_questions=validation_questions,
            test_questions=test_questions,
            split_seed=int((split_meta or {}).get("split_seed", bundle.split.split_seed)),
            validation_fraction=float(
                (split_meta or {}).get("validation_fraction", bundle.split.validation_fraction)
            ),
            test_fraction=float(
                (split_meta or {}).get("test_fraction", bundle.split.test_fraction)
            ),
            split_strategy=str(
                (split_meta or {}).get("split_strategy") or bundle.split.split_strategy
            ),
            validation_strategy=str(
                (split_meta or {}).get("validation_strategy") or bundle.split.validation_strategy
            ),
            validation_folds=validation_folds,
        )

        exact_copies = {
            required_files["benchmark_payload"]: self.artifacts.benchmark_payload_path,
            required_files["hidden_test_payload"]: self.artifacts.hidden_test_payload_path,
            required_files["train_set"]: self.artifacts.train_set_path,
            required_files["validation_set"]: self.artifacts.validation_set_path,
            required_files["validation_folds"]: self.artifacts.validation_folds_path,
            required_files["test_set"]: self.artifacts.test_set_path,
            required_files["split_meta"]: self.artifacts.split_meta_path,
        }
        for source_path, target_path in exact_copies.items():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

        seeded_questions = [*train_questions, *validation_questions, *test_questions]
        benchmark_source = str(
            (split_meta or {}).get("benchmark_source") or bundle.benchmark_source
        )
        seeded_bundle = replace(
            bundle,
            questions=seeded_questions,
            split=seeded_split,
            benchmark_source=benchmark_source,
            warehouse_id=str((split_meta or {}).get("warehouse_id") or bundle.warehouse_id or "")
            or None,
        )
        self.artifacts.record_iteration(
            {
                "phase": "fixed_benchmark_artifacts_applied",
                "seed_path": str(seed_root),
                "benchmark_payload_hash": _file_sha256(self.artifacts.benchmark_payload_path),
                "split_metadata_hash": _file_sha256(self.artifacts.split_meta_path),
                "train_count": len(train_questions),
                "validation_count": len(validation_questions),
                "test_count": len(test_questions),
            }
        )
        return seeded_bundle

    def run_population_benchmark(
        self,
        *,
        label: str,
        questions: list[BenchmarkQuestion],
        k_samples: int = 1,
    ) -> BenchmarkRun:
        clone_space_id, clone_config = self._preflight_with_recovery()
        config_hash = self.artifacts.config_hash(clone_config.to_serialized_space())
        run, sample_runs, aggregate = self._run_benchmarks_multi(
            space_id=clone_space_id,
            questions=questions,
            config_hash=config_hash,
            k_samples=k_samples,
            label=label,
            benchmark_phase="benchmark_population",
        )
        # Stash the aggregate so the terminal-selection gate can read a
        # multi-sample mean + sigma for this population label (e.g. "baseline"
        # vs the final candidate) without re-plumbing the summary pipeline.
        if not hasattr(self, "_full_sample_aggregates"):
            self._full_sample_aggregates: dict[str, SampleAggregate] = {}
        self._full_sample_aggregates[label] = aggregate
        if not hasattr(self, "_full_sample_runs"):
            self._full_sample_runs: dict[str, list[BenchmarkRun]] = {}
        self._full_sample_runs[label] = sample_runs
        payload = {
            **run.to_dict(),
            "ordering": [question.id for question in questions],
            "config_hash": config_hash,
            "population_question_count": len(questions),
        }
        if aggregate.n > 1:
            payload["multi_sample"] = aggregate.as_dict()
        self.artifacts.write_history_snapshot("full", label, payload)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_population",
                "label": label,
                "clone_space_id": clone_space_id,
                "summary": _run_summary_with_issue_classes(run),
                "question_count": len(questions),
            }
        )
        return run

    def rewrite_split_from_population_baseline(
        self,
        *,
        bundle: ExportBundle,
        baseline_run: BenchmarkRun,
    ) -> ExportBundle:
        workspace_host = getattr(
            getattr(getattr(self.genie_service, "wc", None), "config", None),
            "host",
            None,
        )
        rewritten = self.exporter.rewrite_split_from_baseline(
            space_id=bundle.space_id,
            space_title=bundle.space_title,
            config=bundle.config,
            questions=bundle.questions,
            benchmark_source=bundle.benchmark_source,
            workspace_host=workspace_host,
            validation_fraction=bundle.split.validation_fraction,
            test_fraction=bundle.split.test_fraction,
            split_seed=bundle.split.split_seed,
            baseline_run=baseline_run,
        )
        train_ids = {question.id for question in rewritten.split.train_questions}
        validation_ids = {question.id for question in rewritten.split.validation_questions}
        test_ids = {question.id for question in rewritten.split.test_questions}
        self.artifacts.record_iteration(
            {
                "phase": "split_rebalanced",
                "source_space_id": bundle.space_id,
                "clone_space_id": self.require_clone_id(),
                "split_strategy": rewritten.split.split_strategy,
                "train_count": len(train_ids),
                "validation_count": len(validation_ids),
                "test_count": len(test_ids),
                "train_summary": _summarize_subset(baseline_run, train_ids),
                "validation_summary": _summarize_subset(baseline_run, validation_ids),
                "test_summary": _summarize_subset(baseline_run, test_ids),
                "prompt_path": str(rewritten.prompt_path),
            }
        )
        return rewritten

    def _project_observed_run(
        self,
        *,
        observed_run: BenchmarkRun,
        questions: list[BenchmarkQuestion],
        space_id: str,
    ) -> BenchmarkRun:
        results_by_id: dict[str, BenchmarkResult] = {
            result.question_id: result for result in observed_run.results
        }
        missing_ids = [question.id for question in questions if question.id not in results_by_id]
        if missing_ids:
            sample = ", ".join(missing_ids[:5])
            raise WorkspaceFlowError(
                f"Observed benchmark baseline is missing results for: {sample}"
            )
        projected = BenchmarkRun(
            space_id=space_id,
            started_at=observed_run.started_at,
            completed_at=observed_run.completed_at,
        )
        projected.results = [results_by_id[question.id] for question in questions]
        return projected

    def _blind_summary_from_observed_run(
        self,
        *,
        run: BenchmarkRun,
        validation_strategy: str,
    ) -> dict[str, Any]:
        return {
            "question_count": run.total,
            "passed": run.passed,
            "failed": run.failed,
            "errors": run.errors,
            "failed_ids": [
                result.question_id
                for result in run.results
                if result.status == BenchmarkStatus.FAILED
            ],
            "error_ids": [
                result.question_id
                for result in run.results
                if result.status == BenchmarkStatus.ERROR
            ],
            "pass_rate": round(run.pass_rate, 2),
            "validation_strategy": validation_strategy,
            "comparison_metrics": _comparison_metrics_from_run(run),
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "baseline_source": "observed_benchmark_seed",
        }

    def _validation_summary_from_observed_run(
        self,
        *,
        observed_run: BenchmarkRun,
        clone_config: GenieSpaceConfig,
        space_id: str,
    ) -> dict[str, Any]:
        folds_payload = self._load_validation_folds()
        if not folds_payload or folds_payload.get("validation_strategy") != "small_sample_kfold":
            questions = self._resolve_validation_questions(clone_config)
            run = self._project_observed_run(
                observed_run=observed_run,
                questions=questions,
                space_id=space_id,
            )
            return self._blind_summary_from_observed_run(
                run=run,
                validation_strategy="blind_split",
            )

        folds = folds_payload.get("folds", [])
        if not isinstance(folds, list):
            raise WorkspaceFlowError("Invalid validation_folds.json payload.")

        fold_summaries: list[dict[str, Any]] = []
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_questions = 0
        failed_ids: list[str] = []
        error_ids: list[str] = []
        for fold in folds:
            if not isinstance(fold, dict):
                raise WorkspaceFlowError("Invalid validation fold entry.")
            validation_ids = [str(value) for value in fold.get("validation_ids", [])]
            questions = self._resolve_blind_questions(
                clone_config,
                ids=validation_ids,
                label="observed validation fold ids",
            )
            run = self._project_observed_run(
                observed_run=observed_run,
                questions=questions,
                space_id=space_id,
            )
            fold_summary = {
                "fold_index": int(fold.get("fold_index", len(fold_summaries) + 1)),
                "question_count": len(questions),
                "passed": run.passed,
                "failed": run.failed,
                "errors": run.errors,
                "pass_rate": round(run.pass_rate, 2),
                "validation_ids": validation_ids,
                "failed_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.FAILED
                ],
                "error_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.ERROR
                ],
                "comparison_metrics": _comparison_metrics_from_run(run),
                "baseline_source": "observed_benchmark_seed",
            }
            fold_summaries.append(fold_summary)
            total_questions += len(questions)
            total_passed += run.passed
            total_failed += run.failed
            total_errors += run.errors
            failed_ids.extend(fold_summary["failed_ids"])
            error_ids.extend(fold_summary["error_ids"])

        pass_rate = (total_passed / total_questions * 100) if total_questions else 0.0
        fold_pass_rates = [float(fold["pass_rate"]) for fold in fold_summaries]
        return {
            "validation_strategy": "small_sample_kfold",
            "fold_count": len(fold_summaries),
            "visible_question_count": int(folds_payload.get("visible_question_count", total_questions)),
            "question_count": total_questions,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "failed_ids": failed_ids,
            "error_ids": error_ids,
            "pass_rate": round(pass_rate, 2),
            "mean_fold_pass_rate": round(sum(fold_pass_rates) / len(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "min_fold_pass_rate": round(min(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "max_fold_pass_rate": round(max(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "folds": fold_summaries,
            "comparison_metrics": _aggregate_comparison_metrics(
                [
                    fold_summary.get("comparison_metrics", {})
                    for fold_summary in fold_summaries
                    if isinstance(fold_summary, dict)
                ]
            ),
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "baseline_source": "observed_benchmark_seed",
        }

    def _blind_summary_from_projected_baseline(
        self,
        *,
        run: BenchmarkRun,
        validation_strategy: str,
        baseline_source: str,
        aggregate: SampleAggregate | None = None,
        population_label: str | None = None,
        pass_fraction_threshold: float = _MULTI_SAMPLE_BASELINE_PASS_FRACTION_THRESHOLD,
        include_per_question_metadata: bool = False,
    ) -> dict[str, Any]:
        issue_details = _benchmark_issue_details(run.results)
        summary: dict[str, Any] = {
            "question_count": run.total,
            "evaluated_question_count": run.total,
            "passed": run.passed,
            "failed": run.failed,
            "errors": run.errors,
            "skipped": 0,
            "failed_ids": [
                result.question_id
                for result in run.results
                if result.status == BenchmarkStatus.FAILED
            ],
            "error_ids": [
                result.question_id
                for result in run.results
                if result.status == BenchmarkStatus.ERROR
            ],
            "skipped_ids": [],
            "pass_rate": round(run.pass_rate, 2),
            "validation_strategy": validation_strategy,
            "comparison_metrics": _comparison_metrics_from_run(run),
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "baseline_source": baseline_source,
        }
        if aggregate is not None and population_label is not None:
            summary["multi_sample_baseline"] = _multi_sample_baseline_metadata(
                aggregate=aggregate,
                population_label=population_label,
                pass_fraction_threshold=pass_fraction_threshold,
                question_ids=[result.question_id for result in run.results],
                include_per_question=include_per_question_metadata,
            )
        if issue_details:
            summary["issue_class_counts"] = _benchmark_issue_class_counts(run.results)
            summary["issue_details"] = issue_details
        return summary

    def _validation_summary_from_population_sample_run(
        self,
        *,
        aggregate_run: BenchmarkRun,
        clone_config: GenieSpaceConfig,
        space_id: str,
        aggregate: SampleAggregate,
        population_label: str,
        pass_fraction_threshold: float,
    ) -> dict[str, Any]:
        folds_payload = self._load_validation_folds()
        if not folds_payload or folds_payload.get("validation_strategy") != "small_sample_kfold":
            questions = self._resolve_validation_questions(clone_config)
            run = self._project_observed_run(
                observed_run=aggregate_run,
                questions=questions,
                space_id=space_id,
            )
            return self._blind_summary_from_projected_baseline(
                run=run,
                validation_strategy="blind_split",
                baseline_source=_MULTI_SAMPLE_BASELINE_SOURCE,
                aggregate=aggregate,
                population_label=population_label,
                pass_fraction_threshold=pass_fraction_threshold,
                include_per_question_metadata=True,
            )

        folds = folds_payload.get("folds", [])
        if not isinstance(folds, list):
            raise WorkspaceFlowError("Invalid validation_folds.json payload.")

        fold_summaries: list[dict[str, Any]] = []
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_questions = 0
        failed_ids: list[str] = []
        error_ids: list[str] = []
        issue_details: list[dict[str, Any]] = []
        for fold in folds:
            if not isinstance(fold, dict):
                raise WorkspaceFlowError("Invalid validation fold entry.")
            validation_ids = [str(value) for value in fold.get("validation_ids", [])]
            questions = self._resolve_blind_questions(
                clone_config,
                ids=validation_ids,
                label="multi-sample validation fold ids",
            )
            run = self._project_observed_run(
                observed_run=aggregate_run,
                questions=questions,
                space_id=space_id,
            )
            fold_issue_details = _benchmark_issue_details(run.results)
            fold_summary = {
                "fold_index": int(fold.get("fold_index", len(fold_summaries) + 1)),
                "question_count": len(questions),
                "evaluated_question_count": run.total,
                "passed": run.passed,
                "failed": run.failed,
                "errors": run.errors,
                "skipped": 0,
                "pass_rate": round(run.pass_rate, 2),
                "validation_ids": validation_ids,
                "failed_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.FAILED
                ],
                "error_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.ERROR
                ],
                "skipped_ids": [],
                "comparison_metrics": _comparison_metrics_from_run(run),
                "baseline_source": _MULTI_SAMPLE_BASELINE_SOURCE,
                "multi_sample_baseline": _multi_sample_baseline_metadata(
                    aggregate=aggregate,
                    population_label=population_label,
                    pass_fraction_threshold=pass_fraction_threshold,
                    question_ids=validation_ids,
                    include_per_question=True,
                ),
            }
            if fold_issue_details:
                fold_summary["issue_class_counts"] = _benchmark_issue_class_counts(run.results)
                fold_summary["issue_details"] = fold_issue_details
            fold_summaries.append(fold_summary)
            total_questions += len(questions)
            total_passed += run.passed
            total_failed += run.failed
            total_errors += run.errors
            failed_ids.extend(fold_summary["failed_ids"])
            error_ids.extend(fold_summary["error_ids"])
            issue_details.extend(fold_issue_details)

        pass_rate = (total_passed / total_questions * 100) if total_questions else 0.0
        fold_pass_rates = [float(fold["pass_rate"]) for fold in fold_summaries]
        summary: dict[str, Any] = {
            "validation_strategy": "small_sample_kfold",
            "fold_count": len(fold_summaries),
            "visible_question_count": int(folds_payload.get("visible_question_count", total_questions)),
            "question_count": total_questions,
            "evaluated_question_count": total_questions,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "skipped": 0,
            "failed_ids": failed_ids,
            "error_ids": error_ids,
            "skipped_ids": [],
            "pass_rate": round(pass_rate, 2),
            "mean_fold_pass_rate": round(sum(fold_pass_rates) / len(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "min_fold_pass_rate": round(min(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "max_fold_pass_rate": round(max(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "folds": fold_summaries,
            "comparison_metrics": _aggregate_comparison_metrics(
                [
                    fold_summary.get("comparison_metrics", {})
                    for fold_summary in fold_summaries
                    if isinstance(fold_summary, dict)
                ]
            ),
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "baseline_source": _MULTI_SAMPLE_BASELINE_SOURCE,
            "multi_sample_baseline": _multi_sample_baseline_metadata(
                aggregate=aggregate,
                population_label=population_label,
                pass_fraction_threshold=pass_fraction_threshold,
                question_ids=[
                    str(question_id)
                    for fold in folds
                    if isinstance(fold, dict)
                    for question_id in fold.get("validation_ids", [])
                ],
                include_per_question=True,
            ),
        }
        if issue_details:
            issue_counts: dict[str, int] = {}
            for detail in issue_details:
                issue_class = str(detail["issue_class"])
                issue_counts[issue_class] = issue_counts.get(issue_class, 0) + 1
            summary["issue_class_counts"] = dict(sorted(issue_counts.items()))
            summary["issue_details"] = issue_details
        return summary

    def initialize_baseline_from_population_samples(
        self,
        *,
        label: str,
        questions: list[BenchmarkQuestion],
        population_label: str = "population_baseline",
        pass_fraction_threshold: float = _MULTI_SAMPLE_BASELINE_PASS_FRACTION_THRESHOLD,
    ) -> dict[str, Any]:
        sample_runs = (getattr(self, "_full_sample_runs", {}) or {}).get(population_label)
        if not sample_runs:
            raise WorkspaceFlowError(
                f"Population benchmark samples are missing for `{population_label}`."
            )
        aggregate = (getattr(self, "_full_sample_aggregates", {}) or {}).get(population_label)
        if aggregate is None:
            aggregate = aggregate_samples(sample_runs)

        clone_space_id, clone_config = self._preflight_with_recovery()
        clone_config = _config_with_local_benchmarks(self.artifacts, clone_config)
        config_hash = self.artifacts.config_hash(clone_config.to_serialized_space())
        aggregate_run = _aggregate_sample_runs_for_baseline(
            sample_runs=sample_runs,
            questions=questions,
            pass_fraction_threshold=pass_fraction_threshold,
        )

        train_questions = self._load_train_questions()
        train_run = self._project_observed_run(
            observed_run=aggregate_run,
            questions=train_questions,
            space_id=clone_space_id,
        )
        train_payload = {
            **train_run.to_dict(),
            "ordering": [question.id for question in train_questions],
            "diff": _build_train_diff(None, train_run),
            "config_hash": config_hash,
            "subsets": [
                {
                    "name": _MULTI_SAMPLE_BASELINE_SOURCE,
                    "question_count": len(train_questions),
                    "summary": _summarize_run(train_run),
                    "question_ids": [question.id for question in train_questions],
                }
            ],
            "baseline_source": _MULTI_SAMPLE_BASELINE_SOURCE,
            "multi_sample_baseline": _multi_sample_baseline_metadata(
                aggregate=aggregate,
                population_label=population_label,
                pass_fraction_threshold=pass_fraction_threshold,
                question_ids=[question.id for question in train_questions],
                include_per_question=True,
            ),
        }
        self._write_train_artifacts(
            payload=train_payload,
            baseline=True,
            persist_latest=True,
        )
        self.artifacts.write_history_snapshot("train", f"{label}.train", train_payload)

        validation_summary = self._validation_summary_from_population_sample_run(
            aggregate_run=aggregate_run,
            clone_config=clone_config,
            space_id=clone_space_id,
            aggregate=aggregate,
            population_label=population_label,
            pass_fraction_threshold=pass_fraction_threshold,
        )
        self._write_scored_summary_artifacts(
            summary=validation_summary,
            baseline=True,
            persist_latest=True,
            baseline_path=self.artifacts.baseline_validation_score_path,
            latest_path=self.artifacts.latest_validation_score_path,
            baseline_summary_path=self.artifacts.baseline_validation_summary_path,
            latest_summary_path=self.artifacts.latest_validation_summary_path,
        )
        self.artifacts.write_history_snapshot("validation", f"{label}.validation", validation_summary)

        test_questions = self._resolve_test_questions(clone_config)
        test_run = self._project_observed_run(
            observed_run=aggregate_run,
            questions=test_questions,
            space_id=clone_space_id,
        )
        test_summary = self._blind_summary_from_projected_baseline(
            run=test_run,
            validation_strategy="final_holdout",
            baseline_source=_MULTI_SAMPLE_BASELINE_SOURCE,
            aggregate=aggregate,
            population_label=population_label,
            pass_fraction_threshold=pass_fraction_threshold,
            include_per_question_metadata=False,
        )
        self._write_scored_summary_artifacts(
            summary=test_summary,
            baseline=True,
            persist_latest=True,
            baseline_path=self.artifacts.baseline_test_score_path,
            latest_path=self.artifacts.latest_test_score_path,
            baseline_summary_path=None,
            latest_summary_path=None,
        )
        self.artifacts.write_history_snapshot("test", f"{label}.test", test_summary)

        if validation_summary.get("validation_strategy") == "small_sample_kfold":
            overall_total = int(validation_summary["question_count"]) + int(test_summary["question_count"])
            overall_passed = int(validation_summary["passed"]) + int(test_summary["passed"])
            overall_failed = int(validation_summary["failed"]) + int(test_summary["failed"])
            overall_errors = int(validation_summary["errors"]) + int(test_summary["errors"])
            overall_basis = "visible_kfold_plus_holdout"
            overall_comparison_metrics = _aggregate_comparison_metrics(
                [
                    validation_summary.get("comparison_metrics", {}),
                    test_summary.get("comparison_metrics", {}),
                ]
            )
        else:
            overall_total = train_run.total + int(validation_summary["question_count"]) + int(test_summary["question_count"])
            overall_passed = train_run.passed + int(validation_summary["passed"]) + int(test_summary["passed"])
            overall_failed = train_run.failed + int(validation_summary["failed"]) + int(test_summary["failed"])
            overall_errors = train_run.errors + int(validation_summary["errors"]) + int(test_summary["errors"])
            overall_basis = "train_plus_validation_plus_holdout"
            overall_comparison_metrics = _aggregate_comparison_metrics(
                [
                    _comparison_metrics_from_run(train_run),
                    validation_summary.get("comparison_metrics", {}),
                    test_summary.get("comparison_metrics", {}),
                ]
            )
        overall_rate = (overall_passed / overall_total * 100) if overall_total else 0.0
        summary = {
            "label": label,
            "clone_space_id": clone_space_id,
            "overall_basis": overall_basis,
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "baseline_source": _MULTI_SAMPLE_BASELINE_SOURCE,
            "multi_sample_baseline": _multi_sample_baseline_metadata(
                aggregate=aggregate,
                population_label=population_label,
                pass_fraction_threshold=pass_fraction_threshold,
                include_per_question=False,
            ),
            "overall": {
                "total": overall_total,
                "passed": overall_passed,
                "failed": overall_failed,
                "errors": overall_errors,
                "pass_rate": round(overall_rate, 2),
                "comparison_metrics": overall_comparison_metrics,
            },
            "train": _summarize_run(train_run),
            "validation": validation_summary,
            "test": test_summary,
        }
        self.artifacts.write_json_path(self.artifacts.baseline_full_summary_path, summary)
        self.artifacts.write_json_path(self.artifacts.latest_full_summary_path, summary)
        self.artifacts.write_history_snapshot("full", label, summary)
        self.artifacts.update_workspace_meta(
            {
                "baseline_initialization_policy": _MULTI_SAMPLE_BASELINE_SOURCE,
                "baseline_population_label": population_label,
                "baseline_k_samples": aggregate.n,
            }
        )
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_baseline_population_multi_sample",
                "label": label,
                "summary": summary,
            }
        )
        self._refresh_clone_title_with_full_score(summary)
        return summary

    def initialize_baseline_from_observed_population(
        self,
        *,
        label: str,
        observed_run: BenchmarkRun,
    ) -> dict[str, Any]:
        clone_space_id, clone_config = self._preflight_with_recovery()
        clone_config = _config_with_local_benchmarks(self.artifacts, clone_config)
        config_hash = self.artifacts.config_hash(clone_config.to_serialized_space())

        train_questions = self._load_train_questions()
        train_run = self._project_observed_run(
            observed_run=observed_run,
            questions=train_questions,
            space_id=clone_space_id,
        )
        train_payload = {
            **train_run.to_dict(),
            "ordering": [question.id for question in train_questions],
            "diff": _build_train_diff(None, train_run),
            "config_hash": config_hash,
            "subsets": [
                {
                    "name": "observed_seed",
                    "question_count": len(train_questions),
                    "summary": _summarize_run(train_run),
                    "question_ids": [question.id for question in train_questions],
                }
            ],
            "baseline_source": "observed_benchmark_seed",
        }
        self._write_train_artifacts(
            payload=train_payload,
            baseline=True,
            persist_latest=True,
        )
        self.artifacts.write_history_snapshot("train", f"{label}.train", train_payload)

        validation_summary = self._validation_summary_from_observed_run(
            observed_run=observed_run,
            clone_config=clone_config,
            space_id=clone_space_id,
        )
        self._write_scored_summary_artifacts(
            summary=validation_summary,
            baseline=True,
            persist_latest=True,
            baseline_path=self.artifacts.baseline_validation_score_path,
            latest_path=self.artifacts.latest_validation_score_path,
            baseline_summary_path=self.artifacts.baseline_validation_summary_path,
            latest_summary_path=self.artifacts.latest_validation_summary_path,
        )
        self.artifacts.write_history_snapshot("validation", f"{label}.validation", validation_summary)

        test_questions = self._resolve_test_questions(clone_config)
        test_run = self._project_observed_run(
            observed_run=observed_run,
            questions=test_questions,
            space_id=clone_space_id,
        )
        test_summary = self._blind_summary_from_observed_run(
            run=test_run,
            validation_strategy="final_holdout",
        )
        self._write_scored_summary_artifacts(
            summary=test_summary,
            baseline=True,
            persist_latest=True,
            baseline_path=self.artifacts.baseline_test_score_path,
            latest_path=self.artifacts.latest_test_score_path,
            baseline_summary_path=None,
            latest_summary_path=None,
        )
        self.artifacts.write_history_snapshot("test", f"{label}.test", test_summary)

        if validation_summary.get("validation_strategy") == "small_sample_kfold":
            overall_total = int(validation_summary["question_count"]) + int(test_summary["question_count"])
            overall_passed = int(validation_summary["passed"]) + int(test_summary["passed"])
            overall_failed = int(validation_summary["failed"]) + int(test_summary["failed"])
            overall_errors = int(validation_summary["errors"]) + int(test_summary["errors"])
            overall_basis = "visible_kfold_plus_holdout"
            overall_comparison_metrics = _aggregate_comparison_metrics(
                [
                    validation_summary.get("comparison_metrics", {}),
                    test_summary.get("comparison_metrics", {}),
                ]
            )
        else:
            overall_total = train_run.total + int(validation_summary["question_count"]) + int(test_summary["question_count"])
            overall_passed = train_run.passed + int(validation_summary["passed"]) + int(test_summary["passed"])
            overall_failed = train_run.failed + int(validation_summary["failed"]) + int(test_summary["failed"])
            overall_errors = train_run.errors + int(validation_summary["errors"]) + int(test_summary["errors"])
            overall_basis = "train_plus_validation_plus_holdout"
            overall_comparison_metrics = _aggregate_comparison_metrics(
                [
                    _comparison_metrics_from_run(train_run),
                    validation_summary.get("comparison_metrics", {}),
                    test_summary.get("comparison_metrics", {}),
                ]
            )
        overall_rate = (overall_passed / overall_total * 100) if overall_total else 0.0
        summary = {
            "label": label,
            "clone_space_id": clone_space_id,
            "overall_basis": overall_basis,
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "baseline_source": "observed_benchmark_seed",
            "overall": {
                "total": overall_total,
                "passed": overall_passed,
                "failed": overall_failed,
                "errors": overall_errors,
                "pass_rate": round(overall_rate, 2),
                "comparison_metrics": overall_comparison_metrics,
            },
            "train": _summarize_run(train_run),
            "validation": validation_summary,
            "test": test_summary,
        }
        self.artifacts.write_json_path(self.artifacts.baseline_full_summary_path, summary)
        self.artifacts.write_json_path(self.artifacts.latest_full_summary_path, summary)
        self.artifacts.write_history_snapshot("full", label, summary)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_baseline_observed",
                "label": label,
                "summary": summary,
            }
        )
        self._refresh_clone_title_with_full_score(summary)
        return summary

    def attach_existing_clone(
        self,
        *,
        clone_space_id: str,
        source_space_id: str,
        input_space_id: str,
    ) -> GenieSpaceInfo:
        info, _ = self.preflight_space(clone_space_id)
        self.artifacts.save_clone_meta(
            clone_space_id=clone_space_id,
            source_space_id=source_space_id,
            input_space_id=input_space_id,
            managed_source_space_id=clone_space_id,
            active_space_id=clone_space_id,
        )
        self.artifacts.update_workspace_meta({"clone_title": info.title})
        self.artifacts.record_iteration(
            {
                "phase": "clone_adopted",
                "input_space_id": input_space_id,
                "source_space_id": source_space_id,
                "clone_space_id": clone_space_id,
                "managed_source_space_id": clone_space_id,
            }
        )
        return info

    def ensure_clone(
        self,
        *,
        source_space_id: str,
        source_config: GenieSpaceConfig,
        source_title: str | None,
        input_space_id: str | None = None,
        fallback_warehouse_id: str | None = None,
        strategy_label: str | None = None,
    ) -> GenieSpaceInfo:
        if strategy_label:
            self._strategy_label = strategy_label
        meta = self.artifacts.load_clone_meta()
        managed_source_space_id = str(meta.get("managed_source_space_id") or meta.get("source_space_id")) if meta else None
        if meta and managed_source_space_id == source_space_id:
            clone_space_id = str(meta.get("clone_space_id"))
            try:
                info, clone_config = self.genie_service.get(clone_space_id)
                patched_clone_config = _config_with_local_benchmarks(self.artifacts, clone_config)
                if patched_clone_config.benchmarks and not clone_config.benchmarks:
                    info = self.genie_service.update(
                        clone_space_id,
                        config=patched_clone_config,
                        title=info.title,
                        description=patched_clone_config.description,
                    )
                self.artifacts.save_clone_meta(
                    clone_space_id=clone_space_id,
                    source_space_id=source_space_id,
                    input_space_id=input_space_id or source_space_id,
                    managed_source_space_id=clone_space_id,
                    active_space_id=clone_space_id,
                    clone_parent_path=(
                        str(meta.get("clone_parent_path"))
                        if meta.get("clone_parent_path")
                        else None
                    ),
                    clone_parent_path_source=(
                        str(meta.get("clone_parent_path_source"))
                        if meta.get("clone_parent_path_source")
                        else None
                    ),
                )
                self.artifacts.update_workspace_meta({"clone_title": info.title})
                self.preflight_space(clone_space_id)
                self.artifacts.record_iteration(
                    {
                        "phase": "clone_reused",
                        "source_space_id": source_space_id,
                        "clone_space_id": clone_space_id,
                        "clone_parent_path": meta.get("clone_parent_path"),
                        "clone_parent_path_source": meta.get("clone_parent_path_source"),
                    }
                )
                return info
            except Exception:
                pass

        source_config = _config_with_local_benchmarks(self.artifacts, source_config)
        warehouse_id = source_config.warehouse_id or fallback_warehouse_id
        if not warehouse_id:
            try:
                source_info, _ = self.genie_service.get(source_space_id)
            except Exception as exc:
                raise WorkspaceFlowError(
                    f"Failed to fetch source space `{source_space_id}` before cloning: {exc}. "
                    "Provide --warehouse-id if the local config does not include it."
                ) from exc
            warehouse_id = source_info.warehouse_id
        if not warehouse_id:
            raise WorkspaceFlowError("Cannot create clone: missing warehouse_id on the source space.")

        clone_title = _formatted_clone_title(
            source_title or source_space_id,
            fallback=source_space_id,
            strategy_label=strategy_label,
        )
        clone_parent_path, clone_parent_path_source = self._resolve_clone_parent_path(source_config)
        create_config = deepcopy(source_config)
        if clone_parent_path:
            create_config.parent_path = clone_parent_path
        try:
            clone_info = self.genie_service.create(
                config=create_config,
                warehouse_id=warehouse_id,
                title=clone_title,
                parent_path=clone_parent_path,
            )
        except Exception as exc:
            raise WorkspaceFlowError(
                f"Failed to create optimization clone for `{source_space_id}`: {exc}"
            ) from exc
        self.preflight_space(clone_info.space_id)
        self.artifacts.save_clone_meta(
            clone_space_id=clone_info.space_id,
            source_space_id=source_space_id,
            input_space_id=input_space_id or source_space_id,
            managed_source_space_id=clone_info.space_id,
            active_space_id=clone_info.space_id,
            clone_parent_path=clone_parent_path,
            clone_parent_path_source=clone_parent_path_source,
        )
        self.artifacts.update_workspace_meta({"clone_title": clone_info.title})
        self.artifacts.record_iteration(
            {
                "phase": "clone_created",
                "input_space_id": input_space_id or source_space_id,
                "source_space_id": source_space_id,
                "managed_source_space_id": clone_info.space_id,
                "clone_space_id": clone_info.space_id,
                "clone_title": clone_info.title,
                "clone_parent_path": clone_parent_path,
                "clone_parent_path_source": clone_parent_path_source,
            }
        )
        return clone_info

    def _refresh_clone_title_with_full_score(self, full_summary: dict[str, Any]) -> None:
        clone_space_id = self.require_clone_id()
        overall = full_summary.get("overall", {})
        full_pass_rate = overall.get("pass_rate")
        if full_pass_rate is None:
            return

        try:
            remote_info, remote_config = self.genie_service.get(clone_space_id)
        except Exception as exc:
            raise WorkspaceFlowError(
                f"Failed to fetch clone `{clone_space_id}` before updating its benchmark title: {exc}"
            ) from exc

        desired_title = _formatted_clone_title(
            remote_info.title,
            full_pass_rate=float(full_pass_rate),
            fallback=clone_space_id,
            strategy_label=self._strategy_label,
        )
        if desired_title == remote_info.title:
            return

        try:
            updated_info = self.genie_service.update(
                clone_space_id,
                config=remote_config,
                title=desired_title,
                description=remote_config.description,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to update clone `%s` title: %s (non-fatal, continuing)",
                clone_space_id,
                exc,
            )
            self.artifacts.record_iteration(
                {
                    "phase": "clone_title_refresh",
                    "clone_space_id": clone_space_id,
                    "error": str(exc),
                }
            )
            return

        self.artifacts.record_iteration(
            {
                "phase": "clone_title_refresh",
                "clone_space_id": clone_space_id,
                "clone_title": updated_info.title,
                "full_pass_rate": round(float(full_pass_rate), 2),
            }
        )
        workspace_meta = self.artifacts.load_workspace_meta() or {}
        clone_space_url = _genie_space_url_from_host(workspace_meta.get("workspace_host"), clone_space_id)
        self.artifacts.update_workspace_meta(
            {
                "clone_title": updated_info.title,
                "clone_space_url": clone_space_url,
            }
        )

    def require_clone_id(self) -> str:
        meta = self.artifacts.load_clone_meta()
        if not meta or not meta.get("clone_space_id"):
            raise WorkspaceFlowError("No clone metadata found. Run `maxgenie clone` or `maxgenie optimize` first.")
        return str(meta["clone_space_id"])

    def capture_git_state(self, *, stage: str) -> dict[str, Any]:
        state = _git_state()
        meta = self.artifacts.load_workspace_meta() or {}
        git_meta = dict(meta.get("git_state", {}))
        git_meta[stage] = state
        self.artifacts.update_workspace_meta({"git_state": git_meta})
        self.artifacts.record_iteration(
            {
                "phase": "git_state",
                "stage": stage,
                "summary": {
                    "available": bool(state.get("available")),
                    "commit_sha": state.get("commit_sha"),
                    "dirty": state.get("dirty"),
                    "changed_file_count": len(state.get("changed_files", [])),
                },
            }
        )
        return state

    def _lineage_context(self) -> tuple[str, str]:
        workspace_meta = self.artifacts.load_workspace_meta() or {}
        clone_meta = self.artifacts.load_clone_meta() or {}
        input_space_id = str(
            workspace_meta.get("input_space_id")
            or clone_meta.get("input_space_id")
            or self.artifacts.space_id
        )
        managed_source_space_id = str(
            workspace_meta.get("managed_source_space_id")
            or clone_meta.get("managed_source_space_id")
            or clone_meta.get("source_space_id")
            or input_space_id
        )
        return input_space_id, managed_source_space_id

    def _best_checkpoint_path(self) -> Path | None:
        optimization_summary = _load_json_dict(self.artifacts.optimization_summary_path) or {}
        checkpoint_path = optimization_summary.get("best_checkpoint_path")
        if not checkpoint_path:
            return None
        path = Path(str(checkpoint_path))
        return path if path.exists() else None

    def _restore_best_checkpoint_for_final_audit(self, summary: dict[str, Any]) -> str | None:
        checkpoint_path = summary.get("best_checkpoint_path")
        if not checkpoint_path:
            self.artifacts.record_iteration(
                {
                    "phase": "final_audit_checkpoint_restore_skipped",
                    "reason": "missing_best_checkpoint_path",
                }
            )
            return None
        path = Path(str(checkpoint_path))
        if not path.exists():
            self.artifacts.record_iteration(
                {
                    "phase": "final_audit_checkpoint_restore_skipped",
                    "reason": "best_checkpoint_path_missing",
                    "checkpoint_path": str(path),
                }
            )
            return None
        self.restore_space_checkpoint(path)
        self.push()
        self.artifacts.record_iteration(
            {
                "phase": "final_audit_checkpoint_restored",
                "checkpoint_path": str(path),
            }
        )
        return str(path)

    def recover_managed_clone_from_workspace(
        self,
        *,
        reason: str,
        restore_best_checkpoint: bool,
        restore_checkpoint_path: Path | None = None,
        fallback_warehouse_id: str | None = None,
    ) -> GenieSpaceInfo:
        input_space_id, managed_source_space_id = self._lineage_context()
        restored_checkpoint_path: str | None = None
        restored_checkpoint_source: str | None = None
        if restore_checkpoint_path is not None:
            checkpoint_path = restore_checkpoint_path
            restored_checkpoint_source = "explicit"
            if checkpoint_path.exists():
                self.restore_space_checkpoint(checkpoint_path)
                restored_checkpoint_path = str(checkpoint_path)
            else:
                self.artifacts.record_iteration(
                    {
                        "phase": "clone_recovery_checkpoint_restore_skipped",
                        "reason": "explicit_checkpoint_missing",
                        "checkpoint_path": str(checkpoint_path),
                    }
                )
                restored_checkpoint_source = None
        elif restore_best_checkpoint:
            checkpoint_path = self._best_checkpoint_path()
            if checkpoint_path is not None:
                self.restore_space_checkpoint(checkpoint_path)
                restored_checkpoint_path = str(checkpoint_path)
                restored_checkpoint_source = "best"

        source_config = _config_with_local_benchmarks(
            self.artifacts,
            assemble_config(self.artifacts.space_dir),
        )
        source_title = source_config.title or input_space_id
        clone_info = self.ensure_clone(
            source_space_id=input_space_id,
            source_config=source_config,
            source_title=source_title,
            input_space_id=input_space_id,
            fallback_warehouse_id=fallback_warehouse_id or source_config.warehouse_id,
        )
        self.artifacts.update_workspace_meta(
            {
                "input_space_id": input_space_id,
                "managed_source_space_id": clone_info.space_id,
                "active_space_id": clone_info.space_id,
            }
        )
        lineage_payload = {
            "input_space_id": input_space_id,
            "managed_source_space_id": clone_info.space_id,
            "active_space_id": clone_info.space_id,
            "workspace_dir": str(self.artifacts.workspace_dir),
            "workspace_host": getattr(
                getattr(getattr(self.genie_service, "wc", None), "config", None),
                "host",
                None,
            ),
            "restored_checkpoint_path": restored_checkpoint_path,
            "restored_checkpoint_source": restored_checkpoint_source,
        }
        if restored_checkpoint_source == "best":
            lineage_payload["best_checkpoint_path"] = restored_checkpoint_path
        if restored_checkpoint_source == "explicit":
            lineage_payload["recovery_checkpoint_path"] = restored_checkpoint_path
        self.artifacts.save_lineage(input_space_id, lineage_payload)
        self.artifacts.record_iteration(
            {
                "phase": "clone_recovered",
                "reason": reason,
                "input_space_id": input_space_id,
                "previous_managed_source_space_id": managed_source_space_id,
                "clone_space_id": clone_info.space_id,
                "restored_checkpoint_path": restored_checkpoint_path,
                "restored_checkpoint_source": restored_checkpoint_source,
            }
        )
        return clone_info

    def restore_space_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_space_dir = checkpoint_dir / "space"
        if not checkpoint_space_dir.exists():
            raise WorkspaceFlowError(f"Checkpoint is missing space snapshot: {checkpoint_dir}")
        restored_config = assemble_config(checkpoint_space_dir)
        decompose_config(restored_config, self.artifacts.space_dir, include_benchmarks=False)

    def preflight_space(self, space_id: str) -> tuple[GenieSpaceInfo, GenieSpaceConfig]:
        try:
            info, config = self.genie_service.get(space_id)
        except Exception as exc:
            raise WorkspaceFlowError(f"Space `{space_id}` is not accessible: {exc}") from exc

        deadline = time.time() + _CONVERSATION_READY_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while True:
            try:
                conversation_id, message_id = self.benchmark_service.start_conversation(
                    space_id,
                    "maxgenie preflight",
                )
                self.benchmark_service.get_message(space_id, conversation_id, message_id)
                break
            except Exception as exc:
                last_error = exc
                if time.time() >= deadline:
                    raise WorkspaceFlowError(
                        f"Space `{space_id}` does not support stable conversations: {last_error}"
                    ) from last_error
                time.sleep(_CONVERSATION_READY_POLL_SECONDS)

        return info, config

    def push(self) -> GenieSpaceInfo:
        clone_space_id = self.require_clone_id()
        editable_config = assemble_config(self.artifacts.space_dir)
        def recover_after_push_failure(reason: str, desired_config: GenieSpaceConfig) -> GenieSpaceInfo:
            recovered_info = self.recover_managed_clone_from_workspace(
                reason=reason,
                restore_best_checkpoint=False,
                fallback_warehouse_id=desired_config.warehouse_id,
            )
            recovered_config = _config_with_local_benchmarks(self.artifacts, desired_config)
            self.artifacts.record_iteration(
                {
                    "phase": "push",
                    "clone_space_id": recovered_info.space_id,
                    "config_hash": self.artifacts.config_hash(recovered_config.to_serialized_space()),
                    "recovered_clone": True,
                }
            )
            return recovered_info

        try:
            remote_info, remote_config = self.genie_service.get(clone_space_id)
        except Exception as exc:
            if _is_missing_space_error(exc):
                return recover_after_push_failure(
                    reason=f"push_fetch_failed: {exc}",
                    desired_config=editable_config,
                )
            raise WorkspaceFlowError(
                f"Failed to fetch clone `{clone_space_id}` before push: {exc}"
            ) from exc
        remote_config = _config_with_local_benchmarks(self.artifacts, remote_config)
        merged = _merge_editable_sections(remote_config, editable_config)
        try:
            info = self.genie_service.update(
                clone_space_id,
                config=merged,
                title=remote_info.title,
                description=merged.description,
            )
        except Exception as exc:
            if _is_missing_space_chain(exc):
                return recover_after_push_failure(
                    reason=f"push_update_failed: {exc}",
                    desired_config=merged,
                )
            raise WorkspaceFlowError(
                f"Failed to update clone `{clone_space_id}`: {exc}"
            ) from exc
        try:
            self.preflight_space(clone_space_id)
        except Exception as exc:
            if _is_missing_space_chain(exc):
                return recover_after_push_failure(
                    reason=f"push_preflight_failed: {exc}",
                    desired_config=merged,
                )
            raise
        self.artifacts.record_iteration(
            {
                "phase": "push",
                "clone_space_id": clone_space_id,
                "config_hash": self.artifacts.config_hash(merged.to_serialized_space()),
            }
        )
        return info

    def curate_workspace(self, *, label: str = "curate") -> dict[str, Any]:
        return self.curate_workspace_mode(label=label, mode="all")

    def curate_workspace_mode(self, *, label: str = "curate", mode: str) -> dict[str, Any]:
        config = assemble_config(self.artifacts.space_dir)
        curated, summary = curate_space_config(config, mode=mode)
        decompose_config(curated, self.artifacts.space_dir, include_benchmarks=False)
        diagnostics = analyze_space_config(curated)
        summary_payload = summary.to_dict()
        summary_payload["candidate_status"] = "prepared"
        self.artifacts.write_json_path(self.artifacts.curation_summary_path, summary_payload)
        self.artifacts.write_json_path(self.artifacts.space_diagnostics_path, diagnostics.to_dict())
        self.artifacts.record_iteration(
            {
                "phase": "curate",
                "label": label,
                "mode": mode,
                "summary": {
                    "changed": summary.changed,
                    "table_descriptions_added": summary.table_descriptions_added,
                    "column_descriptions_added": summary.column_descriptions_added,
                    "columns_hidden": summary.columns_hidden,
                    "format_assistance_enabled": summary.format_assistance_enabled,
                    "entity_matching_enabled": summary.entity_matching_enabled,
                    "synonyms_added": summary.synonyms_added,
                },
            }
        )
        return summary_payload

    def run_no_benchmark_advisory_best_practices(
        self,
        *,
        label: str = ADVISORY_BEST_PRACTICES_CLASSIFICATION,
        curation_mode: str = CURATION_MODE_SAFE_DEFAULTS,
    ) -> dict[str, Any]:
        self.artifacts.update_workspace_meta(
            {
                "no_benchmark_advisory": True,
                "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                "benchmark_verification_status": NOT_BENCHMARK_VERIFIED_STATUS,
                "benchmark_verification_label": NOT_BENCHMARK_VERIFIED_LABEL,
                "no_benchmark_warning": NO_BENCHMARK_WARNING,
                "auto_promotion_allowed": False,
            }
        )
        self.artifacts.record_iteration(
            {
                "phase": "no_benchmark_advisory_started",
                "label": label,
                "curation_mode": curation_mode,
                "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                "benchmark_verification_status": NOT_BENCHMARK_VERIFIED_STATUS,
            }
        )
        curation_summary = self.curate_workspace_mode(label=label, mode=curation_mode)
        curation_summary = {
            **curation_summary,
            "candidate_status": NOT_BENCHMARK_VERIFIED_LABEL,
            "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
            "benchmark_verification_status": NOT_BENCHMARK_VERIFIED_STATUS,
            "benchmark_verification_label": NOT_BENCHMARK_VERIFIED_LABEL,
            "warning": NO_BENCHMARK_WARNING,
        }
        self.artifacts.write_json_path(self.artifacts.curation_summary_path, curation_summary)
        clone_info = self.push()
        summary = {
            "schema_version": "maxgenie.advisory_best_practices.v1",
            "mode": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
            "label": label,
            "curation_mode": curation_mode,
            "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
            "benchmark_verification_status": NOT_BENCHMARK_VERIFIED_STATUS,
            "benchmark_verification_label": NOT_BENCHMARK_VERIFIED_LABEL,
            "warning": NO_BENCHMARK_WARNING,
            "benchmark_question_count": 0,
            "source_space_modified": False,
            "clone_space_id": clone_info.space_id,
            "candidate_status": NOT_BENCHMARK_VERIFIED_LABEL,
            "changed": bool(curation_summary.get("changed")),
            "changes": curation_summary.get("changes", []),
            "change_count": (
                len(curation_summary.get("changes", []))
                if isinstance(curation_summary.get("changes"), list)
                else 0
            ),
            "curation_summary": curation_summary,
            "promotion_eligible": False,
            "auto_promotion_allowed": False,
            "suggested_next_step": (
                "Add owner-approved benchmark questions, then rerun MaxGenie under the "
                "result-based protocol before treating any changes as verified."
            ),
        }
        self.artifacts.write_json_path(self.artifacts.advisory_best_practices_path, summary)
        self.artifacts.write_json_path(
            self.artifacts.promotion_eligibility_path,
            {
                "status": NOT_BENCHMARK_VERIFIED_STATUS,
                "promotion_eligible": False,
                "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                "benchmark_verification_label": NOT_BENCHMARK_VERIFIED_LABEL,
                "reason": "no_benchmark",
                "warning": NO_BENCHMARK_WARNING,
            },
        )
        self.artifacts.write_json_path(
            self.artifacts.optimization_summary_path,
            {
                "mode": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                "benchmark_verification_status": NOT_BENCHMARK_VERIFIED_STATUS,
                "benchmark_verification_label": NOT_BENCHMARK_VERIFIED_LABEL,
                "accepted": 0,
                "rejected": 0,
                "skipped": 0,
                "stop_reason": "no_benchmark_advisory_completed",
                "terminal_selection": {
                    "status": NOT_BENCHMARK_VERIFIED_STATUS,
                    "reasons": ["no_benchmark"],
                    "selected_checkpoint_path": None,
                },
                "warning": NO_BENCHMARK_WARNING,
            },
        )
        self.artifacts.record_iteration(
            {
                "phase": "no_benchmark_advisory_completed",
                "label": label,
                "curation_mode": curation_mode,
                "clone_space_id": clone_info.space_id,
                "change_count": summary["change_count"],
                "evidence_classification": ADVISORY_BEST_PRACTICES_CLASSIFICATION,
                "benchmark_verification_status": NOT_BENCHMARK_VERIFIED_STATUS,
            }
        )
        return summary

    def _load_visible_benchmark_questions(self) -> list[BenchmarkQuestion]:
        visible_questions: list[BenchmarkQuestion] = []
        seen_ids: set[str] = set()
        for question in self._load_train_questions():
            if question.id in seen_ids:
                continue
            visible_questions.append(question)
            seen_ids.add(question.id)

        payload = self.artifacts.load_benchmark_payload() or {}
        rows = payload.get("questions", [])
        question_map = {
            str(row.get("id")): BenchmarkQuestion(
                id=str(row["id"]),
                question=str(row["question"]),
                expected_sql=str(row["expected_sql"]) if row.get("expected_sql") is not None else None,
            )
            for row in rows
            if isinstance(row, dict) and row.get("id") and row.get("question")
        }
        for question_id in self._load_validation_ids():
            question = question_map.get(question_id)
            if question is None or question.id in seen_ids:
                continue
            visible_questions.append(question)
            seen_ids.add(question.id)
        return visible_questions

    def _config_with_visible_benchmarks(self, config: GenieSpaceConfig) -> GenieSpaceConfig:
        visible_questions = self._load_visible_benchmark_questions()
        merged = deepcopy(config)
        merged.benchmarks = (
            benchmark_questions_to_model(visible_questions) if visible_questions else None
        )
        return merged

    def _attach_visible_expected_rows(
        self, structured_failure_context: dict[str, Any]
    ) -> None:
        """Attach a compact expected answer-key row preview to each visible failure.

        Visible-only by construction: the structured failure context holds only
        train/validation failures, and each lookup keys off that visible question's
        expected SQL. Holdout expected rows are never queried here, so the precomputed
        holdout rows can never reach proposer context.
        """
        comparator = self.comparator
        if comparator is None or not isinstance(structured_failure_context, dict):
            return
        lookup = getattr(comparator, "expected_rows_for_sql", None)
        if not callable(lookup):
            return
        failures = structured_failure_context.get("failures")
        if not isinstance(failures, list):
            return
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            expected_sql = failure.get("expected_sql")
            if not expected_sql:
                continue
            resolved = lookup(str(expected_sql))
            if not resolved:
                continue
            preview = _format_expected_result_preview(resolved[0], resolved[1])
            if preview:
                failure["expected_result_preview"] = preview

    def _visible_sqls_for_domain_probe(
        self, structured_failure_context: dict[str, Any]
    ) -> list[str]:
        """Collect visible expected/generated SQL (failures + pass guards) for table mining.

        Strictly visible: failures and pass guards are train/validation-derived, so no
        holdout SQL is ever mined here.
        """
        sqls: list[str] = []
        for bucket in ("failures", "pass_guards"):
            entries = structured_failure_context.get(bucket)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                for key in ("expected_sql", "generated_sql"):
                    value = entry.get(key)
                    if value:
                        sqls.append(str(value))
        return sqls

    def _attach_parameter_domain_values(
        self, structured_failure_context: dict[str, Any]
    ) -> None:
        """Attach probed literal/domain values for the parameter-driving columns.

        Probes the warehouse once per run (cached) for the distinct values of the columns
        that drive the dominant unbound-parameter failure family, so the proposer can
        literalize a ``:parameter`` marker with a real value. Tables are mined only from
        visible SQL, and the probe returns column domains (never holdout answer rows), so
        nothing here leaks the hidden holdout.
        """
        if not isinstance(structured_failure_context, dict):
            return
        comparator = self.comparator
        if comparator is None:
            return
        execute_readonly = getattr(comparator, "execute_readonly", None)
        if not callable(execute_readonly):
            return
        if self._parameter_domain_values_cache is None:
            tables = _ordered_candidate_tables(
                self._visible_sqls_for_domain_probe(structured_failure_context)
            )
            if not tables:
                self._parameter_domain_values_cache = {}
            else:
                try:
                    self._parameter_domain_values_cache = _probe_parameter_domain_values(
                        execute_readonly, tables
                    )
                except Exception:  # noqa: BLE001 - enrichment only; never break a run on a probe
                    self._parameter_domain_values_cache = {}
        if self._parameter_domain_values_cache:
            structured_failure_context["parameter_domain_values"] = dict(
                self._parameter_domain_values_cache
            )

    def _visible_benchmark_payload(self) -> dict[str, Any]:
        source_payload = self.artifacts.load_benchmark_payload() or {}
        visible_questions = self._load_visible_benchmark_questions()
        payload = benchmark_questions_to_payload(visible_questions)
        return {
            "benchmark_source": source_payload.get("benchmark_source"),
            "payload_scope": "visible_pool",
            **payload,
            "policy": {
                "visible_ids_only": True,
                "source_splits": ["train", "validation"],
                "hidden_holdout_content": "omitted",
            },
        }

    def _serving_candidate_spec(
        self,
        sweep_index: int,
        *,
        candidate_mode: str | None = None,
    ) -> CandidateSpec:
        return CandidateSpec(
            name=f"serving_iteration_{sweep_index:02d}",
            family="serving",
            description="Workspace serving endpoint proposes one structured candidate.",
            candidate_mode=candidate_mode,
        )

    def _load_iteration_records(self) -> list[dict[str, Any]]:
        iteration_log_path = self.artifacts.history_dir / "iteration_log.jsonl"
        if not iteration_log_path.exists():
            return []

        records: list[dict[str, Any]] = []
        for raw_line in iteration_log_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _snapshot_path_for_label(self, *, kind: str, label: str) -> Path | None:
        directory_map = {
            "train": self.artifacts.train_run_history_dir,
            "validation": self.artifacts.validation_run_history_dir,
            "test": self.artifacts.test_run_history_dir,
            "full": self.artifacts.full_run_history_dir,
        }
        directory = directory_map[kind]
        matches = sorted(directory.glob(f"*.{label}.json"))
        return matches[-1] if matches else None

    def _failed_ids_from_snapshot(self, *, kind: str, label: str) -> list[str]:
        snapshot_path = self._snapshot_path_for_label(kind=kind, label=label)
        if snapshot_path is None:
            return []
        payload = _load_json_dict(snapshot_path)
        if payload is None:
            return []

        if kind == "train":
            results = payload.get("results", [])
            return [
                str(result.get("question_id"))
                for result in results
                if isinstance(result, dict)
                and result.get("question_id")
                and str(result.get("status", "")).lower() in {"failed", "error"}
            ]

        failed_ids = payload.get("failed_ids")
        error_ids = payload.get("error_ids")
        output: list[str] = []
        for values in (failed_ids, error_ids):
            if not isinstance(values, list):
                continue
            output.extend(str(value) for value in values if value)
        return output

    def _benchmark_question_lookup(self) -> dict[str, str]:
        payload = self.artifacts.load_benchmark_payload() or {}
        rows = payload.get("questions")
        if not isinstance(rows, list):
            return {}
        lookup: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            lookup[str(row["id"])] = str(row.get("question") or "")
        return lookup

    def _visible_proxy_records(
        self,
        *,
        train_run: BenchmarkRun | None,
        validation_summary: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        question_lookup = self._benchmark_question_lookup()
        if train_run is not None:
            for result in train_run.results:
                records.append(
                    {
                        "question_id": result.question_id,
                        "split": "train",
                        "question": result.question or question_lookup.get(result.question_id),
                        "status": result.status.value,
                        "issue_class": _benchmark_issue_class(result),
                    }
                )

        if isinstance(validation_summary, dict):
            failed_ids = {
                str(value)
                for value in validation_summary.get("failed_ids", [])
                if str(value).strip()
            }
            error_ids = {
                str(value)
                for value in validation_summary.get("error_ids", [])
                if str(value).strip()
            }
            skipped_ids = {
                str(value)
                for value in validation_summary.get("skipped_ids", [])
                if str(value).strip()
            }
            for question_id in self._load_validation_ids():
                if question_id in error_ids:
                    status = "error"
                elif question_id in failed_ids:
                    status = "failed"
                elif question_id in skipped_ids:
                    status = "skipped"
                else:
                    status = "passed"
                records.append(
                    {
                        "question_id": question_id,
                        "split": "validation",
                        "question": question_lookup.get(question_id),
                        "status": status,
                    }
                )
        return records

    def _visible_proxy_guard_payload(self) -> dict[str, Any]:
        train_run = self._load_reference_train_run()
        validation_summary = self._load_validation_summary()
        records = self._visible_proxy_records(
            train_run=train_run,
            validation_summary=validation_summary,
        )
        guard_set = build_proxy_guard_requirements(
            records,
            include_question_text=True,
        )
        payload = guard_set.to_dict()
        payload.update(
            {
                "schema_version": "maxgenie.visible_proxy_guard_context.v1",
                "source_record_count": len(records),
                "source": {
                    "train_run_available": train_run is not None,
                    "validation_summary_available": isinstance(validation_summary, dict),
                    "visible_sources_only": True,
                },
                "policy": {
                    "source_splits": ["train", "validation"],
                    "hidden_holdout_content": "omitted",
                    "use": "proposal_context_and_generalization_pressure",
                },
            }
        )
        return payload

    def _data_probe_payload(self, current_config: GenieSpaceConfig) -> dict[str, Any]:
        """Run bounded read-only aggregate data-shape probes for proposal context."""
        warehouse_id = self._resolve_query_history_warehouse_id(current_config.warehouse_id)
        payload: dict[str, Any] = {
            "schema_version": "maxgenie.data_probe_context.v1",
            "warehouse_id": warehouse_id,
            "probe_scope": "compact_aggregate_diagnostics",
            "policy": {
                "read_only": True,
                "sample_limit_per_probe": 50000,
                "hidden_holdout_content": "omitted",
                "row_level_samples": "omitted",
            },
            "probes": [],
            "blockers": [],
        }
        if not warehouse_id:
            payload["blockers"].append("missing_sql_warehouse")
            return payload
        tables = list(
            (current_config.data_sources.tables or [])
            if current_config.data_sources
            else []
        )
        if not tables:
            payload["blockers"].append("no_source_tables_in_space_config")
            return payload

        comparator = ResultComparator(self.genie_service.wc, warehouse_id)
        for table in tables[:_DATA_PROBE_MAX_TABLES]:
            table_name = str(getattr(table, "identifier", "") or "").strip()
            if not table_name:
                continue
            column_configs = list(table.column_configs or [])
            selected_columns = self._data_probe_columns(column_configs)
            for column_name in selected_columns:
                probe_sql = self._data_probe_sql(
                    table_identifier=table_name,
                    column_name=column_name,
                )
                probe_record: dict[str, Any] = {
                    "table": table_name,
                    "column": column_name,
                    "probe_type": "string_shape_sample",
                }
                try:
                    columns, rows = comparator.execute_readonly(probe_sql)
                except Exception as exc:
                    probe_record.update(
                        {
                            "status": "error",
                            "error": str(exc)[:1000],
                        }
                    )
                    payload["probes"].append(probe_record)
                    continue
                values = rows[0] if rows else []
                summary = {
                    str(column): values[index] if index < len(values) else None
                    for index, column in enumerate(columns)
                }
                probe_record.update(
                    {
                        "status": "ok",
                        "summary": summary,
                    }
                )
                payload["probes"].append(probe_record)
        return payload

    @staticmethod
    def _data_probe_columns(column_configs: list[Any]) -> list[str]:
        candidates: list[str] = []
        fallback: list[str] = []
        for column_config in column_configs:
            if bool(getattr(column_config, "exclude", False)):
                continue
            name = str(getattr(column_config, "column_name", "") or "").strip()
            if not name:
                continue
            if _DATA_PROBE_TEXT_COLUMN_RE.search(name):
                candidates.append(name)
            else:
                fallback.append(name)
        selected = candidates or fallback
        return list(dict.fromkeys(selected))[:_DATA_PROBE_MAX_COLUMNS_PER_TABLE]

    @classmethod
    def _data_probe_sql(cls, *, table_identifier: str, column_name: str) -> str:
        table_sql = cls._quote_sql_identifier(table_identifier)
        column_sql = cls._quote_sql_name(column_name)
        return f"""
WITH sample AS (
  SELECT CAST({column_sql} AS STRING) AS value
  FROM {table_sql}
  LIMIT 50000
)
SELECT
  COUNT(*) AS sampled_rows,
  SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_count,
  SUM(CASE WHEN value IS NOT NULL AND TRIM(value) = '' THEN 1 ELSE 0 END) AS blank_or_trim_blank_count,
  SUM(CASE WHEN value IS NOT NULL AND value <> TRIM(value) THEN 1 ELSE 0 END) AS trim_mismatch_count,
  APPROX_COUNT_DISTINCT(value) AS approx_distinct_raw_count,
  APPROX_COUNT_DISTINCT(TRIM(value)) AS approx_distinct_trimmed_count
FROM sample
""".strip()

    @staticmethod
    def _quote_sql_identifier(identifier: str) -> str:
        parts = [part.strip() for part in identifier.split(".") if part.strip()]
        if not parts:
            raise WorkspaceFlowError("Cannot quote empty SQL identifier.")
        return ".".join(f"`{part.replace('`', '``')}`" for part in parts)

    @staticmethod
    def _quote_sql_name(identifier: str) -> str:
        value = identifier.strip()
        if not value:
            raise WorkspaceFlowError("Cannot quote empty SQL name.")
        return f"`{value.replace('`', '``')}`"

    def _recent_rejected_attempts(self, *, limit: int = 3) -> list[dict[str, Any]]:
        optimization_summary = _load_json_dict(self.artifacts.optimization_summary_path) or {}
        passes = optimization_summary.get("passes", [])
        rejected_passes = [
            item
            for item in passes
            if isinstance(item, dict) and item.get("outcome") == "rejected"
        ]
        if not rejected_passes:
            return []

        rejected_records = [
            record
            for record in self._load_iteration_records()
            if record.get("phase") == "candidate_evaluated" and record.get("outcome") == "rejected"
        ]

        recent: list[dict[str, Any]] = []
        record_index = len(rejected_records) - 1
        for rejected in reversed(rejected_passes):
            matching_record: dict[str, Any] | None = None
            while record_index >= 0:
                candidate_record = rejected_records[record_index]
                record_index -= 1
                if (
                    candidate_record.get("name") == rejected.get("name")
                    and candidate_record.get("family") == rejected.get("family")
                ):
                    matching_record = candidate_record
                    break
            label = str(matching_record.get("label")) if matching_record and matching_record.get("label") else ""
            train_failures = self._failed_ids_from_snapshot(kind="train", label=f"{label}.train") if label else []
            validation_failures = (
                self._failed_ids_from_snapshot(kind="validation", label=f"{label}.validation")
                if label
                else []
            )
            recent.append(
                {
                    "name": rejected.get("name"),
                    "family": rejected.get("family"),
                    "reason": rejected.get("reason"),
                    "candidate_summary": rejected.get("candidate_summary"),
                    "train_failures": train_failures[:5],
                    "validation_failures": validation_failures[:5],
                }
            )
            if len(recent) >= limit:
                break
        recent.reverse()
        return recent

    def _candidate_summary_text(self, candidate_summary: dict[str, Any] | None) -> str | None:
        if not isinstance(candidate_summary, dict):
            return None
        if candidate_summary.get("policy"):
            return f"policy `{candidate_summary['policy']}`"
        if candidate_summary.get("mode"):
            return f"curation mode `{candidate_summary['mode']}`"
        if candidate_summary.get("candidate_name"):
            return f"serving candidate `{candidate_summary['candidate_name']}`"
        if candidate_summary.get("description"):
            return str(candidate_summary["description"])
        if isinstance(candidate_summary.get("changes"), list):
            return f"{len(candidate_summary['changes'])} structured edits"
        return None

    @staticmethod
    def _artifact_patch_intent_audit_lines(
        curation_summary: dict[str, Any],
    ) -> list[str]:
        passes = curation_summary.get("passes")
        if not isinstance(passes, list):
            return []

        lines: list[str] = []
        for pass_outcome in passes:
            if not isinstance(pass_outcome, dict):
                continue
            candidate_summary = pass_outcome.get("candidate_summary")
            if not isinstance(candidate_summary, dict):
                continue
            risk_assessment = candidate_summary.get("risk_assessment")
            if not isinstance(risk_assessment, dict):
                continue
            verification = risk_assessment.get("visible_effect_verification")
            if not isinstance(verification, dict):
                continue
            causal_plan = verification.get("causal_patch_plan")
            if not isinstance(causal_plan, dict):
                continue
            available_intents = [
                str(value)
                for value in causal_plan.get("available_sql_delta_intents", [])
                if str(value).strip()
            ]
            selected_intents = [
                str(value)
                for value in causal_plan.get("selected_sql_delta_intents", [])
                if str(value).strip()
            ]
            if not available_intents and not selected_intents:
                continue
            outcome = str(pass_outcome.get("outcome") or "unknown")
            mode = str(
                candidate_summary.get("candidate_mode")
                or pass_outcome.get("mode")
                or "unknown"
            )
            name = str(
                candidate_summary.get("candidate_name")
                or pass_outcome.get("name")
                or mode
            )
            selected_text = ", ".join(selected_intents) if selected_intents else "none"
            available_text = ", ".join(available_intents) if available_intents else "none"
            marker = str(causal_plan.get("selected_marker") or "n/a")
            lines.append(
                f"- `{name}` ({outcome}, `{mode}`): selected intents `{selected_text}`; "
                f"available intents `{available_text}`; marker `{marker}`."
            )
            expected_effect = str(causal_plan.get("expected_visible_effect") or "").strip()
            if expected_effect:
                lines.append(f"  - Expected visible effect: {expected_effect}")
            if len(lines) >= 10:
                break
        return lines

    def _decomposed_space_payload(self) -> dict[str, Any]:
        """Build a bounded, hash-addressed view of decomposed space files."""
        exact_paths = (
            "instructions/text_instruction.md",
            "instructions/text_instruction.meta.yml",
            "instructions/sql_functions.yml",
            "sample_questions.yml",
        )
        glob_patterns = (
            "tables/*.yml",
            "metric_views/*.yml",
            "instructions/examples/*.yml",
            "instructions/join_specs/*.yml",
            "instructions/sql_snippets/filters/*.yml",
            "instructions/sql_snippets/expressions/*.yml",
            "instructions/sql_snippets/measures/*.yml",
        )
        files: list[dict[str, Any]] = []
        seen: set[str] = set()
        for relative in exact_paths:
            path = self.artifacts.space_dir / relative
            if path.exists() and path.is_file():
                files.append(self._decomposed_file_payload(path, relative))
                seen.add(relative)
        for pattern in glob_patterns:
            for path in sorted(self.artifacts.space_dir.glob(pattern)):
                relative = path.relative_to(self.artifacts.space_dir).as_posix()
                if relative in seen or not path.is_file():
                    continue
                files.append(self._decomposed_file_payload(path, relative))
                seen.add(relative)
        return {
            "schema_version": "maxgenie.decomposed_space_context.v1",
            "space_dir": str(self.artifacts.space_dir),
            "file_count": len(files),
            "files": files,
            "policy": {
                "allowed_paths_only": True,
                "max_file_content_chars": 60000,
                "hidden_holdout_content": "omitted",
            },
        }

    @staticmethod
    def _decomposed_file_payload(path: Path, relative_path: str) -> dict[str, Any]:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > 60000
        digest = hashlib.sha256(raw).hexdigest()
        return {
            "path": relative_path,
            "sha256": digest,
            "expected_sha256": digest,
            "bytes": len(raw),
            "truncated": truncated,
            "content": text[:60000] if truncated else text,
        }

    @staticmethod
    def _artifact_patch_repeats_text_only_after_train_collapse(
        *,
        candidate_mode: str,
        changed_paths: set[str],
        failure_context: str | None,
    ) -> bool:
        if candidate_mode not in _DEFAULT_STRICT_ARTIFACT_PATCH_MODES:
            return False
        if changed_paths != {"instructions/text_instruction.md"}:
            return False
        if not failure_context:
            return False
        lowered = failure_context.lower()
        if (
            "## prior candidate lessons" not in lowered
            and "## prior optimization attempts" not in lowered
        ):
            return False
        return "train_health_failed" in lowered or bool(
            re.search(r"->\s*0(?:\.0+)?%", lowered)
        )

    @staticmethod
    def _artifact_patch_repeated_train_collapse_shapes(
        *,
        candidate_mode: str,
        changed_edits: list[dict[str, Any]],
        failure_context: str | None,
    ) -> list[dict[str, str]]:
        if candidate_mode not in _DEFAULT_STRICT_ARTIFACT_PATCH_MODES or not failure_context:
            return []
        lowered = failure_context.lower()
        if (
            "## prior candidate lessons" not in lowered
            and "## prior optimization attempts" not in lowered
        ):
            return []
        if "train_health_failed" not in lowered and not re.search(
            r"->\s*0(?:\.0+)?%", lowered
        ):
            return []
        shapes = {
            (
                str(edit.get("path") or "").strip(),
                str(edit.get("edit_mode") or "").strip(),
            )
            for edit in changed_edits
            if str(edit.get("path") or "").strip()
            and str(edit.get("edit_mode") or "").strip()
        }
        if len(shapes) != 1:
            return []
        path, edit_mode = next(iter(shapes))
        prior_shape_pattern = (
            r"file_edit\s+path="
            + re.escape(path)
            + r"\s+mode="
            + re.escape(edit_mode)
            + r"(?:\s|$)"
        )
        if not re.search(prior_shape_pattern, failure_context):
            return []
        return [{"path": path, "edit_mode": edit_mode}]

    @staticmethod
    def _artifact_patch_repeated_prior_refused_shapes(
        *,
        candidate_mode: str,
        changed_edits: list[dict[str, Any]],
        failure_context: str | None,
    ) -> list[dict[str, str]]:
        if candidate_mode not in _DEFAULT_STRICT_ARTIFACT_PATCH_MODES or not failure_context:
            return []
        if "Candidate shape refused before benchmarking:" not in failure_context:
            return []
        shapes = {
            (
                str(edit.get("path") or "").strip(),
                str(edit.get("edit_mode") or "").strip(),
                str(edit.get("reason") or "").strip(),
            )
            for edit in changed_edits
            if str(edit.get("path") or "").strip()
            and str(edit.get("edit_mode") or "").strip()
            and str(edit.get("reason") or "").strip()
        }
        if len(shapes) != 1:
            return []
        current_path, current_mode, current_reason = next(iter(shapes))
        for line in failure_context.splitlines():
            if "Candidate shape refused before benchmarking:" not in line:
                continue
            match = re.search(
                r"file_edit\s+path=(?P<path>\S+)\s+mode=(?P<mode>\S+)(?:\s+reason=(?P<reason>.*))?$",
                line,
            )
            if match is None:
                continue
            prior_path = match.group("path").strip()
            prior_mode = match.group("mode").strip()
            prior_reason = (match.group("reason") or "").strip()
            if (
                prior_path == current_path
                and prior_mode == current_mode
                and prior_reason == current_reason
            ):
                return [
                    {
                        "path": current_path,
                        "edit_mode": current_mode,
                        "reason": current_reason,
                    }
                ]
        return []

    @staticmethod
    def _artifact_patch_new_executable_after_repeated_train_collapse(
        *,
        candidate_mode: str,
        changed_edits: list[dict[str, Any]],
        failure_context: str | None,
    ) -> list[dict[str, Any]]:
        if candidate_mode not in _DEFAULT_STRICT_ARTIFACT_PATCH_MODES or not failure_context:
            return []

        current_new_executable_paths = [
            str(edit.get("path") or "").strip()
            for edit in changed_edits
            if str(edit.get("action") or "").strip() == "created"
            and str(edit.get("path") or "").strip()
            and serving_candidate_is_executable_artifact_path(
                str(edit.get("path") or "").strip()
            )
        ]
        if not current_new_executable_paths:
            return []

        prior_new_executable_paths: list[str] = []
        inside_train_collapse = False
        for line in failure_context.splitlines():
            stripped = line.strip()
            if stripped.startswith("- `") and "rejected because" in stripped:
                inside_train_collapse = (
                    "train_health_failed" in stripped
                    or bool(re.search(r"->\s*0(?:\.0+)?%", stripped))
                )
                continue
            if not inside_train_collapse:
                continue
            if "Operations that failed the gates:" not in stripped:
                continue
            for match in re.finditer(
                r"file_edit\s+path=(?P<path>\S+)\s+mode=(?P<mode>\S+)",
                stripped,
            ):
                prior_path = match.group("path").rstrip(".,;")
                prior_mode = match.group("mode").rstrip(".,;")
                if prior_mode not in {"replace_file", "create_file"}:
                    continue
                if serving_candidate_is_executable_artifact_path(prior_path):
                    prior_new_executable_paths.append(prior_path)
            inside_train_collapse = False

        if len(set(prior_new_executable_paths)) < 2:
            return []
        return [
            {
                "path": path,
                "prior_new_executable_train_collapse_count": len(
                    set(prior_new_executable_paths)
                ),
                "prior_new_executable_train_collapse_paths": sorted(
                    set(prior_new_executable_paths)
                )[:6],
            }
            for path in sorted(set(current_new_executable_paths))
        ]

    def _apply_artifact_patch_payload(
        self,
        *,
        payload: dict[str, Any],
        current_config: GenieSpaceConfig,
        structured_failure_context: dict[str, Any] | None = None,
        failure_context: str | None = None,
    ) -> tuple[GenieSpaceConfig, dict[str, Any]]:
        """Validate and apply a decomposed-file patch candidate."""
        candidate_name = str(payload.get("candidate_name") or "artifact_patch_candidate")
        candidate_mode = str(payload.get("candidate_mode") or "artifact_patch")
        strict_v2 = serving_candidate_is_strict_artifact_patch_mode(candidate_mode)
        min_file_edits, max_file_edits = serving_candidate_artifact_patch_file_edit_limits(
            candidate_mode
        )
        explicit_no_change = self._artifact_patch_payload_is_no_change(payload)
        patch_payload = payload
        if explicit_no_change and not isinstance(payload.get("file_edits"), list):
            patch_payload = deepcopy(payload)
            patch_payload["file_edits"] = []
        dry_summary = apply_artifact_patch_candidate(
            self.artifacts.space_dir,
            patch_payload,
            dry_run=True,
            max_file_edits=max_file_edits,
            require_existing_sha256=strict_v2,
            require_edit_reason=strict_v2,
            reject_new_bind_placeholders=strict_v2,
        )
        risk_reasons: list[str] = []
        if dry_summary.get("rejected_count"):
            risk_reasons.extend(
                str(item.get("reason"))
                for item in dry_summary.get("rejected_edits", [])
                if isinstance(item, dict) and item.get("reason")
            )
        changed_edits = [
            item
            for item in dry_summary.get("changed_edits", [])
            if isinstance(item, dict)
        ]
        changed_paths = {
            str(item.get("path") or "")
            for item in changed_edits
            if str(item.get("path") or "").strip()
        }
        if min_file_edits is not None and len(changed_paths) < min_file_edits:
            risk_reasons.append(
                f"too_few_composite_artifact_patch_file_edits:{len(changed_paths)}"
            )
        if candidate_mode == "composite_artifact_patch_v2" and changed_paths == {
            "instructions/text_instruction.md"
        }:
            risk_reasons.append("composite_artifact_patch_rejects_instruction_only_patch")
        if serving_candidate_artifact_patch_requires_executable_asset(candidate_mode):
            executable_changed_paths = sorted(
                path
                for path in changed_paths
                if serving_candidate_is_executable_artifact_path(path)
            )
            if not executable_changed_paths:
                risk_reasons.append(
                    "executable_artifact_patch_requires_example_or_snippet_file"
                )
        if self._artifact_patch_repeats_text_only_after_train_collapse(
            candidate_mode=candidate_mode,
            changed_paths=changed_paths,
            failure_context=failure_context,
        ):
            risk_reasons.append(
                "artifact_patch_v2_text_only_after_prior_train_collapse"
            )
        repeated_train_collapse_shapes = (
            self._artifact_patch_repeated_train_collapse_shapes(
                candidate_mode=candidate_mode,
                changed_edits=changed_edits,
                failure_context=failure_context,
            )
        )
        if repeated_train_collapse_shapes:
            risk_reasons.append(
                "artifact_patch_v2_repeated_single_file_shape_after_prior_train_collapse"
            )
        repeated_prior_refused_shapes = (
            self._artifact_patch_repeated_prior_refused_shapes(
                candidate_mode=candidate_mode,
                changed_edits=changed_edits,
                failure_context=failure_context,
            )
        )
        if repeated_prior_refused_shapes:
            risk_reasons.append(
                "artifact_patch_v2_repeated_single_file_shape_after_prior_refusal"
            )
        repeated_new_executable_train_collapse = (
            self._artifact_patch_new_executable_after_repeated_train_collapse(
                candidate_mode=candidate_mode,
                changed_edits=changed_edits,
                failure_context=failure_context,
            )
        )
        if repeated_new_executable_train_collapse:
            risk_reasons.append(
                "artifact_patch_v2_new_executable_asset_after_repeated_train_collapse"
            )
        if explicit_no_change and not dry_summary.get("rejected_count") and not changed_paths:
            return deepcopy(current_config), self._artifact_patch_summary(
                payload=patch_payload,
                candidate_name=candidate_name,
                patch_summary=dry_summary,
                accepted=True,
                reason="no_change",
                risk_reasons=[],
                candidate_mode=candidate_mode,
                visible_effect_verification={
                    "applied": False,
                    "accepted": True,
                    "reasons": [],
                    "candidate_mode": candidate_mode,
                    "policy": {
                        "requires_visible_effect_mapping": False,
                        "reason": "explicit_no_change",
                    },
                },
                prior_train_collapse_shapes=repeated_train_collapse_shapes,
                prior_refused_shapes=repeated_prior_refused_shapes,
                prior_train_collapse_new_executable_assets=(
                    repeated_new_executable_train_collapse
                ),
            )
        visible_effect_verification = self._verify_artifact_patch_visible_effect(
            payload=patch_payload,
            candidate_mode=candidate_mode,
            changed_paths=changed_paths,
            patch_changed_edits=changed_edits,
            structured_failure_context=structured_failure_context,
            failure_context=failure_context,
        )
        risk_reasons.extend(visible_effect_verification.get("reasons", []))
        if not dry_summary.get("changed_count"):
            risk_reasons.append("artifact_patch_no_effective_file_edits")

        if risk_reasons:
            return deepcopy(current_config), self._artifact_patch_summary(
                payload=patch_payload,
                candidate_name=candidate_name,
                patch_summary=dry_summary,
                accepted=False,
                reason="artifact_patch_risk_rejected",
                risk_reasons=risk_reasons,
                candidate_mode=candidate_mode,
                visible_effect_verification=visible_effect_verification,
                prior_train_collapse_shapes=repeated_train_collapse_shapes,
                prior_refused_shapes=repeated_prior_refused_shapes,
                prior_train_collapse_new_executable_assets=(
                    repeated_new_executable_train_collapse
                ),
            )

        with tempfile.TemporaryDirectory(prefix="maxgenie-artifact-patch-") as tmpdir:
            temp_space_dir = Path(tmpdir) / "space"
            copy_missing_paths = _copytree_for_patch_validation(
                self.artifacts.space_dir,
                temp_space_dir,
            )
            temp_summary = apply_artifact_patch_candidate(
                temp_space_dir,
                patch_payload,
                max_file_edits=max_file_edits,
                require_existing_sha256=strict_v2,
                require_edit_reason=strict_v2,
                reject_new_bind_placeholders=strict_v2,
            )
            if copy_missing_paths:
                temp_summary = {
                    **temp_summary,
                    "source_copy_missing_paths": copy_missing_paths,
                }
            if temp_summary.get("rejected_count"):
                return deepcopy(current_config), self._artifact_patch_summary(
                    payload=patch_payload,
                    candidate_name=candidate_name,
                    patch_summary=temp_summary,
                    accepted=False,
                    reason="artifact_patch_validation_rejected",
                    risk_reasons=[
                        str(item.get("reason"))
                        for item in temp_summary.get("rejected_edits", [])
                        if isinstance(item, dict) and item.get("reason")
                    ],
                    candidate_mode=candidate_mode,
                    visible_effect_verification=visible_effect_verification,
                )
            try:
                assemble_config(temp_space_dir)
            except Exception as exc:
                return deepcopy(current_config), self._artifact_patch_summary(
                    payload=patch_payload,
                    candidate_name=candidate_name,
                    patch_summary=temp_summary,
                    accepted=False,
                    reason="artifact_patch_invalid_decomposed_space",
                    risk_reasons=[str(exc)],
                    candidate_mode=candidate_mode,
                    visible_effect_verification=visible_effect_verification,
                )

        patch_summary = apply_artifact_patch_candidate(
            self.artifacts.space_dir,
            patch_payload,
            max_file_edits=max_file_edits,
            require_existing_sha256=strict_v2,
            require_edit_reason=strict_v2,
            reject_new_bind_placeholders=strict_v2,
        )
        if patch_summary.get("rejected_count"):
            return deepcopy(current_config), self._artifact_patch_summary(
                payload=patch_payload,
                candidate_name=candidate_name,
                patch_summary=patch_summary,
                accepted=False,
                reason="artifact_patch_apply_rejected",
                risk_reasons=[
                    str(item.get("reason"))
                    for item in patch_summary.get("rejected_edits", [])
                    if isinstance(item, dict) and item.get("reason")
                ],
                candidate_mode=candidate_mode,
                visible_effect_verification=visible_effect_verification,
            )
        candidate_config = assemble_config(self.artifacts.space_dir)
        return candidate_config, self._artifact_patch_summary(
            payload=patch_payload,
            candidate_name=candidate_name,
            patch_summary=patch_summary,
            accepted=True,
            reason=None,
            risk_reasons=[],
            candidate_mode=candidate_mode,
            visible_effect_verification=visible_effect_verification,
        )

    @staticmethod
    def _artifact_patch_payload_is_no_change(payload: dict[str, Any]) -> bool:
        file_edits = payload.get("file_edits")
        has_no_file_edits = file_edits is None or (
            isinstance(file_edits, list) and not file_edits
        )
        if not has_no_file_edits:
            return False
        operations = payload.get("operations")
        if isinstance(operations, list) and operations:
            return all(
                isinstance(operation, dict)
                and str(operation.get("op") or "").strip() == "no_change"
                for operation in operations
            )
        if operations in (None, []):
            name = str(payload.get("candidate_name") or "").lower()
            if "no_change" in name or "no change" in name:
                return True
        return False

    def _verify_artifact_patch_visible_effect(
        self,
        *,
        payload: dict[str, Any],
        candidate_mode: str,
        changed_paths: set[str],
        patch_changed_edits: list[dict[str, Any]],
        structured_failure_context: dict[str, Any] | None,
        failure_context: str | None = None,
    ) -> dict[str, Any]:
        """Reject strict patch candidates that are not tied to visible evidence."""

        if candidate_mode in _DEFAULT_STRICT_ARTIFACT_PATCH_MODES:
            return self._verify_strict_artifact_patch_visible_effect(
                payload=payload,
                candidate_mode=candidate_mode,
                changed_paths=changed_paths,
                patch_changed_edits=patch_changed_edits,
                structured_failure_context=structured_failure_context,
                failure_context=failure_context,
            )
        if serving_candidate_is_broad_artifact_patch_mode(candidate_mode):
            return self._verify_broad_artifact_patch_visible_effect(
                payload=payload,
                candidate_mode=candidate_mode,
                changed_paths=changed_paths,
                patch_changed_edits=patch_changed_edits,
                structured_failure_context=structured_failure_context,
            )
        return self._verify_executable_artifact_patch_visible_effect(
            payload=payload,
            candidate_mode=candidate_mode,
            changed_paths=changed_paths,
            patch_changed_edits=patch_changed_edits,
            structured_failure_context=structured_failure_context,
        )

    def _verify_broad_artifact_patch_visible_effect(
        self,
        *,
        payload: dict[str, Any],
        candidate_mode: str,
        changed_paths: set[str],
        patch_changed_edits: list[dict[str, Any]],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reject broad patches that lack visible attribution or leak hidden context."""

        reasons: list[str] = []
        families = self._structured_visible_failure_families(structured_failure_context)
        failure_ids = self._structured_visible_failure_ids(structured_failure_context)
        pass_guards = self._structured_visible_pass_guards(structured_failure_context)
        verification_edits = self._artifact_patch_changed_edits_with_payload_content(
            payload=payload,
            changed_edits=patch_changed_edits,
        )
        evidence_text = self._artifact_patch_visible_effect_text(
            payload,
            verification_edits,
        )
        effect_markers = self._artifact_patch_visible_effect_markers(
            evidence_text=evidence_text,
            families=families,
            failure_ids=failure_ids,
        )
        if not effect_markers:
            reasons.append("broad_artifact_patch_missing_visible_effect_mapping")
        expected_impact = str(payload.get("expected_impact") or "").strip().lower()
        if expected_impact in {"", "none", "n/a", "na", "unknown"}:
            reasons.append("broad_artifact_patch_missing_expected_effect")
        serialized_payload = json.dumps(payload, ensure_ascii=False)
        if _has_forbidden_hidden_holdout_reference(serialized_payload):
            reasons.append("broad_artifact_patch_hidden_holdout_reference")
        for edit in verification_edits:
            path = str(edit.get("path") or edit.get("relative_path") or "")
            local_text = " ".join(
                str(edit.get(key) or "")
                for key in ("reason", "description", "comment")
            ).lower()
            if not self._artifact_patch_visible_effect_markers(
                evidence_text=local_text,
                families=families,
                failure_ids=failure_ids,
            ):
                reasons.append(f"broad_artifact_patch_edit_missing_visible_marker:{path}")
            content_text = self._artifact_patch_edit_content_text(edit)
            if path.startswith("instructions/examples/"):
                reasons.extend(
                    self._example_artifact_patch_content_reasons(
                        path=path,
                        content_text=content_text,
                        evidence_text=evidence_text,
                        pass_guards=pass_guards,
                        full_file_content=_is_artifact_patch_full_file_content_edit(edit),
                    )
                )
            elif path.startswith("instructions/sql_snippets/"):
                reasons.extend(
                    self._sql_snippet_artifact_patch_content_reasons(
                        path=path,
                        content_text=content_text,
                        full_file_content=_is_artifact_patch_full_file_content_edit(edit),
                    )
                )
            elif path.startswith("tables/") or path.startswith("metric_views/"):
                metadata_query_plan = (
                    self._artifact_patch_metadata_query_plan_incompatibility(
                        edit=edit,
                        path=path,
                    )
                )
                if metadata_query_plan is not None:
                    reasons.append("broad_artifact_patch_metadata_query_plan_guidance")

        return {
            "applied": True,
            "accepted": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
            "changed_paths": sorted(changed_paths),
            "visible_failure_families": sorted(families),
            "visible_failure_ids_checked": sorted(failure_ids),
            "visible_pass_guard_ids_checked": sorted(
                str(guard.get("question_id") or "")
                for guard in pass_guards
                if str(guard.get("question_id") or "").strip()
            ),
            "effect_markers": effect_markers,
            "policy": {
                "requires_visible_effect_mapping": True,
                "requires_expected_effect": True,
                "rejects_hidden_holdout_references": True,
                "allows_multi_file_visible_bundle": True,
                "candidate_mode": candidate_mode,
            },
        }

    def _verify_strict_artifact_patch_visible_effect(
        self,
        *,
        payload: dict[str, Any],
        candidate_mode: str,
        changed_paths: set[str],
        patch_changed_edits: list[dict[str, Any]],
        structured_failure_context: dict[str, Any] | None,
        failure_context: str | None = None,
    ) -> dict[str, Any]:
        """Require default v2 file patches to name their visible target."""

        families = self._structured_visible_failure_families(structured_failure_context)
        failure_ids = self._structured_visible_failure_ids(structured_failure_context)
        failure_suggested_ops = self._structured_visible_failure_suggested_ops(
            structured_failure_context
        )
        family_suggested_ops = self._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        failure_tokens = self._structured_visible_failure_tokens(
            structured_failure_context
        )
        family_tokens = self._structured_visible_family_tokens(structured_failure_context)
        pass_guard_ids = self._structured_visible_pass_guard_ids(
            structured_failure_context
        )
        causal_plan_summary = self._artifact_patch_causal_plan_summary(
            payload=payload,
            structured_failure_context=structured_failure_context,
        )
        visible_context_available = bool(families or failure_ids or pass_guard_ids)
        if not visible_context_available:
            return {
                "applied": False,
                "accepted": True,
                "reasons": [],
                "candidate_mode": candidate_mode,
                "causal_patch_plan": causal_plan_summary,
                "policy": {
                    "requires_visible_effect_mapping": False,
                    "reason": "structured_visible_context_unavailable",
                },
            }

        file_edits = payload.get("file_edits")
        edits = [
            edit for edit in file_edits if isinstance(edit, dict)
        ] if isinstance(file_edits, list) else []
        changed_edits = [
            edit
            for edit in edits
            if str(edit.get("path") or edit.get("relative_path") or "") in changed_paths
        ]
        global_text = self._artifact_patch_global_effect_text(payload)
        global_markers = self._artifact_patch_exact_visible_effect_markers(
            evidence_text=global_text,
            families=families,
            failure_ids=failure_ids,
            pass_guard_ids=pass_guard_ids,
        )
        missing_paths: list[str] = []
        incompatible_paths: list[dict[str, Any]] = []
        off_target_text_edits: list[dict[str, Any]] = []
        off_target_plan_content_edits: list[dict[str, Any]] = []
        unexpected_new_file_edits: list[dict[str, Any]] = []
        full_file_rewrite_edits: list[dict[str, Any]] = []
        overbroad_instruction_edits: list[dict[str, Any]] = []
        invalid_executable_edits: list[dict[str, Any]] = []
        off_target_example_edits: list[dict[str, Any]] = []
        executable_literal_token_gaps: list[dict[str, Any]] = []
        executable_missing_expected_token_gaps: list[dict[str, Any]] = []
        executable_family_missing_expected_token_gaps: list[dict[str, Any]] = []
        metadata_query_plan_edits: list[dict[str, Any]] = []
        edit_mappings: list[dict[str, Any]] = []
        pass_guards = self._structured_visible_pass_guards(structured_failure_context)
        executable_evidence_text = self._artifact_patch_visible_effect_text(
            payload,
            [
                edit
                for edit in changed_edits
                if serving_candidate_is_executable_artifact_path(
                    str(edit.get("path") or edit.get("relative_path") or "")
                )
            ],
        )
        for edit in changed_edits:
            path = str(edit.get("path") or edit.get("relative_path") or "")
            local_edit_text = " ".join(
                (
                    path,
                    str(edit.get("reason") or ""),
                    str(edit.get("description") or ""),
                    str(edit.get("comment") or ""),
                )
            ).lower()
            edit_text = " ".join(
                (
                    global_text,
                    local_edit_text,
                )
            ).lower()
            markers = self._artifact_patch_exact_visible_effect_markers(
                evidence_text=edit_text,
                families=families,
                failure_ids=failure_ids,
                pass_guard_ids=pass_guard_ids,
            )
            if not markers:
                missing_paths.append(path)
            incompatibility = self._artifact_patch_marker_path_incompatibility(
                path=path,
                markers=markers,
                failure_suggested_ops=failure_suggested_ops,
                family_suggested_ops=family_suggested_ops,
            )
            if incompatibility is not None:
                incompatible_paths.append(incompatibility)
            text_relevance = self._artifact_patch_text_edit_token_incompatibility(
                edit=edit,
                path=path,
                markers=markers,
                failure_tokens=failure_tokens,
                family_tokens=family_tokens,
            )
            if text_relevance is not None:
                off_target_text_edits.append(text_relevance)
            plan_content_relevance = (
                self._artifact_patch_causal_plan_content_token_incompatibility(
                    edit=edit,
                    path=path,
                    causal_plan_summary=causal_plan_summary,
                )
            )
            if plan_content_relevance is not None:
                off_target_plan_content_edits.append(plan_content_relevance)
            instruction_rewrite = self._artifact_patch_instruction_rewrite_incompatibility(
                edit=edit,
                path=path,
            )
            if instruction_rewrite is not None:
                overbroad_instruction_edits.append(instruction_rewrite)
            metadata_query_plan = (
                self._artifact_patch_metadata_query_plan_incompatibility(
                    edit=edit,
                    path=path,
                )
            )
            if metadata_query_plan is not None:
                metadata_query_plan_edits.append(metadata_query_plan)
            executable_rejection = (
                self._artifact_patch_executable_asset_incompatibility(
                    edit=edit,
                    path=path,
                    evidence_text=executable_evidence_text,
                    pass_guards=pass_guards,
                )
            )
            if executable_rejection is not None:
                invalid_executable_edits.append(executable_rejection)
            literal_token_gap = self._artifact_patch_executable_literal_token_gap(
                edit=edit,
                path=path,
                causal_plan_summary=causal_plan_summary,
                structured_failure_context=structured_failure_context,
            )
            if literal_token_gap is not None:
                executable_literal_token_gaps.append(literal_token_gap)
            missing_expected_token_gap = (
                self._artifact_patch_executable_missing_expected_token_gap(
                    edit=edit,
                    path=path,
                    causal_plan_summary=causal_plan_summary,
                    structured_failure_context=structured_failure_context,
                )
            )
            if missing_expected_token_gap is not None:
                executable_missing_expected_token_gaps.append(
                    missing_expected_token_gap
                )
            family_missing_expected_token_gap = (
                self._artifact_patch_executable_family_missing_expected_token_gap(
                    edit=edit,
                    path=path,
                    causal_plan_summary=causal_plan_summary,
                )
            )
            if family_missing_expected_token_gap is not None:
                executable_family_missing_expected_token_gaps.append(
                    family_missing_expected_token_gap
                )
            target_alignment = (
                self._artifact_patch_example_target_alignment_incompatibility(
                    edit=edit,
                    path=path,
                    causal_plan_summary=causal_plan_summary,
                    structured_failure_context=structured_failure_context,
                )
            )
            if target_alignment is not None:
                off_target_example_edits.append(target_alignment)
            edit_mappings.append(
                {
                    "path": path,
                    "effect_markers": markers,
                    "local_effect_markers": self._artifact_patch_exact_visible_effect_markers(
                        evidence_text=local_edit_text,
                        families=families,
                        failure_ids=failure_ids,
                        pass_guard_ids=pass_guard_ids,
                    ),
                    "path_operation_families": sorted(
                        self._artifact_patch_path_operation_families(path)
                    ),
                }
            )
        target_plan_inconsistency = self._artifact_patch_target_plan_inconsistency(
            edit_mappings=edit_mappings,
            global_markers=global_markers,
            structured_failure_context=structured_failure_context,
        )
        missing_causal_patch_plan = (
            candidate_mode in _DEFAULT_STRICT_ARTIFACT_PATCH_MODES
            and bool(changed_paths)
            and bool(failure_ids or families)
            and not causal_plan_summary.get("present")
        )
        high_risk_guard_gap = self._artifact_patch_high_risk_guard_gap(
            edit_mappings=edit_mappings,
            global_markers=global_markers,
            structured_failure_context=structured_failure_context,
        )
        causal_plan_inconsistency = self._artifact_patch_causal_plan_inconsistency(
            causal_plan_summary=causal_plan_summary,
            edit_mappings=edit_mappings,
            global_markers=global_markers,
            changed_paths=changed_paths,
        )
        new_file_incompatibility = (
            self._artifact_patch_causal_plan_new_file_incompatibility(
                causal_plan_summary=causal_plan_summary,
                changed_edits=patch_changed_edits,
            )
        )
        if new_file_incompatibility:
            unexpected_new_file_edits.extend(new_file_incompatibility)
        full_file_rewrite_incompatibility = (
            self._artifact_patch_causal_plan_full_file_rewrite_incompatibility(
                causal_plan_summary=causal_plan_summary,
                changed_edits=patch_changed_edits,
            )
        )
        if full_file_rewrite_incompatibility:
            full_file_rewrite_edits.extend(full_file_rewrite_incompatibility)
        skeleton_binding = None
        if candidate_mode == "skeleton_artifact_patch_v2":
            skeletons = build_artifact_patch_skeletons(
                structured_failure_context=structured_failure_context,
                decomposed_space_payload=self._decomposed_space_payload(),
                candidate_mode=candidate_mode,
            )
            skeleton_binding = validate_artifact_patch_skeleton_binding(
                payload,
                skeletons=skeletons,
            )

        reasons: list[str] = []
        if missing_paths:
            reasons.append("artifact_patch_v2_missing_visible_effect_mapping")
        if incompatible_paths:
            reasons.append("artifact_patch_v2_visible_effect_path_incompatible")
        if off_target_text_edits:
            reasons.append("artifact_patch_v2_text_edit_missing_failure_token")
        if off_target_plan_content_edits:
            reasons.append(
                "artifact_patch_v2_edit_content_missing_causal_plan_token"
            )
        if target_plan_inconsistency is not None:
            reasons.append("artifact_patch_v2_inconsistent_target_plan")
        if missing_causal_patch_plan:
            reasons.append("artifact_patch_v2_missing_causal_patch_plan")
        if high_risk_guard_gap is not None:
            reasons.append("artifact_patch_v2_high_risk_target_without_guard_preservation")
        if causal_plan_inconsistency is not None:
            reasons.append("artifact_patch_v2_causal_plan_inconsistent")
        if unexpected_new_file_edits:
            reasons.append("artifact_patch_v2_causal_plan_unexpected_new_file")
        if full_file_rewrite_edits:
            reasons.append("artifact_patch_v2_full_file_rewrite_not_allowed")
        if overbroad_instruction_edits:
            reasons.append("artifact_patch_v2_trusted_query_routing_rewrite")
        if invalid_executable_edits:
            reasons.extend(
                reason
                for rejection in invalid_executable_edits
                for reason in rejection.get("reasons", [])
                if isinstance(reason, str) and reason
            )
        if executable_literal_token_gaps:
            reasons.append("artifact_patch_v2_executable_literal_missing_expected_token")
        if executable_missing_expected_token_gaps:
            reasons.append("artifact_patch_v2_executable_missing_expected_token")
        if executable_family_missing_expected_token_gaps:
            reasons.append(
                "artifact_patch_v2_executable_family_missing_expected_token"
            )
        if off_target_example_edits:
            reasons.append("artifact_patch_v2_example_not_target_aligned")
        if metadata_query_plan_edits:
            reasons.append("artifact_patch_v2_metadata_query_plan_guidance")
        if skeleton_binding is not None and not skeleton_binding.get("accepted"):
            reasons.append("skeleton_artifact_patch_binding_failed")

        return {
            "applied": True,
            "accepted": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
            "changed_paths": sorted(changed_paths),
            "missing_visible_effect_paths": missing_paths,
            "incompatible_visible_effect_paths": incompatible_paths,
            "off_target_text_edits": off_target_text_edits,
            "off_target_causal_plan_content_edits": off_target_plan_content_edits,
            "inconsistent_target_plan": target_plan_inconsistency,
            "missing_causal_patch_plan": missing_causal_patch_plan,
            "high_risk_guard_gap": high_risk_guard_gap,
            "causal_plan_inconsistency": causal_plan_inconsistency,
            "unexpected_new_file_edits": unexpected_new_file_edits,
            "full_file_rewrite_edits": full_file_rewrite_edits,
            "overbroad_instruction_edits": overbroad_instruction_edits,
            "invalid_executable_edits": invalid_executable_edits,
            "executable_literal_expected_token_gaps": executable_literal_token_gaps,
            "executable_missing_expected_token_gaps": (
                executable_missing_expected_token_gaps
            ),
            "executable_family_missing_expected_token_gaps": (
                executable_family_missing_expected_token_gaps
            ),
            "off_target_example_edits": off_target_example_edits,
            "metadata_query_plan_edits": metadata_query_plan_edits,
            "skeleton_binding": skeleton_binding,
            "visible_failure_families": sorted(families),
            "visible_failure_ids_checked": sorted(failure_ids),
            "visible_pass_guard_ids_checked": sorted(pass_guard_ids),
            "effect_markers": global_markers,
            "edit_mappings": edit_mappings,
            "causal_patch_plan": causal_plan_summary,
            "policy": {
                "requires_visible_effect_mapping": True,
                "accepted_markers": [
                    "target_train_failure:<visible_failure_id>",
                    "visible_family:<visible_failure_family>",
                    "failure_family:<visible_failure_family>",
                    "guard_preservation:<visible_pass_guard_id>",
                ],
                "hidden_holdout_content": "omitted",
                "rejects_invalid_executable_assets": True,
                "rejects_causal_plan_contradictions": True,
                "requires_causal_plan_tokens_in_changed_content": True,
                "rejects_examples_without_target_question_overlap": True,
                "rejects_new_files_for_existing_surface_plans": True,
                "rejects_full_file_rewrites_for_existing_strict_patches": True,
                "requires_causal_patch_plan_for_default_strict_patches": True,
                "requires_guard_preservation_for_high_risk_targets": True,
                "requires_guard_preservation_for_high_risk_family_plans": True,
                "requires_expected_literal_in_target_executable_patches": True,
                "requires_missing_expected_token_in_target_executable_patches": True,
                "requires_shared_missing_expected_token_in_family_executable_patches": True,
                "rejects_query_plan_guidance_in_metadata": True,
                "enforces_skeleton_binding": candidate_mode
                == "skeleton_artifact_patch_v2",
            },
        }

    def _verify_executable_artifact_patch_visible_effect(
        self,
        *,
        payload: dict[str, Any],
        candidate_mode: str,
        changed_paths: set[str],
        patch_changed_edits: list[dict[str, Any]],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Reject executable file patches that do not predict a visible effect."""

        if not serving_candidate_artifact_patch_requires_executable_asset(candidate_mode):
            return {
                "applied": False,
                "accepted": True,
                "reasons": [],
            }

        file_edits = payload.get("file_edits")
        edits = [edit for edit in file_edits if isinstance(edit, dict)] if isinstance(file_edits, list) else []
        executable_edits = [
            edit
            for edit in edits
            if serving_candidate_is_executable_artifact_path(
                str(edit.get("path") or edit.get("relative_path") or "")
            )
        ]
        reasons: list[str] = []
        families = self._structured_visible_failure_families(structured_failure_context)
        failure_ids = self._structured_visible_failure_ids(structured_failure_context)
        pass_guards = self._structured_visible_pass_guards(structured_failure_context)
        evidence_text = self._artifact_patch_visible_effect_text(payload, executable_edits)
        effect_markers = self._artifact_patch_visible_effect_markers(
            evidence_text=evidence_text,
            families=families,
            failure_ids=failure_ids,
        )
        if not effect_markers:
            reasons.append("executable_artifact_patch_missing_visible_effect_mapping")
        expected_impact = str(payload.get("expected_impact") or "").strip().lower()
        if expected_impact in {"", "none", "n/a", "na", "unknown"}:
            reasons.append("executable_artifact_patch_missing_expected_effect")

        for edit in executable_edits:
            path = str(edit.get("path") or edit.get("relative_path") or "")
            content_text = self._artifact_patch_edit_content_text(edit)
            if not content_text.strip():
                continue
            if path.startswith("instructions/examples/"):
                reasons.extend(
                    self._example_artifact_patch_content_reasons(
                        path=path,
                        content_text=content_text,
                        evidence_text=evidence_text,
                        pass_guards=pass_guards,
                        full_file_content=_is_artifact_patch_full_file_content_edit(edit),
                    )
                )
            elif path.startswith("instructions/sql_snippets/"):
                reasons.extend(
                    self._sql_snippet_artifact_patch_content_reasons(
                        path=path,
                        content_text=content_text,
                        full_file_content=_is_artifact_patch_full_file_content_edit(edit),
                    )
                )

        return {
            "applied": True,
            "accepted": not reasons,
            "reasons": list(dict.fromkeys(reasons)),
            "changed_executable_paths": [
                str(edit.get("path") or edit.get("relative_path") or "")
                for edit in executable_edits
            ],
            "visible_failure_families": sorted(families),
            "visible_failure_ids_checked": sorted(failure_ids),
            "visible_pass_guard_ids_checked": sorted(
                str(guard.get("question_id") or "")
                for guard in pass_guards
                if str(guard.get("question_id") or "").strip()
            ),
            "effect_markers": effect_markers,
            "policy": {
                "requires_visible_effect_mapping": True,
                "requires_expected_effect": True,
                "rejects_example_placeholders": True,
                "rejects_snippet_placeholders_full_queries_and_option_menus": True,
                "rejects_literal_swap_guard_clone_examples": True,
            },
        }

    @staticmethod
    def _structured_visible_failure_families(
        structured_failure_context: dict[str, Any] | None,
    ) -> set[str]:
        families: set[str] = set()
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                family = str(failure.get("likely_root_cause_family") or "").strip()
                if family:
                    families.add(family)
        return families

    @staticmethod
    def _structured_visible_failure_ids(
        structured_failure_context: dict[str, Any] | None,
    ) -> set[str]:
        question_ids: set[str] = set()
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                question_id = str(failure.get("question_id") or "").strip()
                if question_id:
                    question_ids.add(question_id)
        return question_ids

    @staticmethod
    def _structured_visible_failure_suggested_ops(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        suggested_ops: dict[str, set[str]] = {}
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                question_id = str(failure.get("question_id") or "").strip()
                raw_ops = failure.get("suggested_operation_families")
                if not question_id or not isinstance(raw_ops, list):
                    continue
                ops = {str(value).strip() for value in raw_ops if str(value).strip()}
                if ops:
                    suggested_ops[question_id] = ops
        return suggested_ops

    @staticmethod
    def _structured_visible_family_suggested_ops(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        family_op_counts: dict[str, dict[str, int]] = {}
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                family = str(failure.get("likely_root_cause_family") or "").strip()
                raw_ops = failure.get("suggested_operation_families")
                if not family or not isinstance(raw_ops, list):
                    continue
                ops = {str(value).strip() for value in raw_ops if str(value).strip()}
                if ops:
                    counts = family_op_counts.setdefault(family, {})
                    for operation_family in ops:
                        counts[operation_family] = counts.get(operation_family, 0) + 1
        return {
            family: {
                operation_family
                for operation_family, count in counts.items()
                if count >= 2
            }
            for family, counts in family_op_counts.items()
            if any(count >= 2 for count in counts.values())
        }

    @staticmethod
    def _structured_visible_failure_tokens(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        tokens_by_id: dict[str, set[str]] = {}
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                question_id = str(failure.get("question_id") or "").strip()
                if not question_id:
                    continue
                tokens = OptimizationWorkspace._structured_failure_tokens(failure)
                if tokens:
                    tokens_by_id[question_id] = tokens
        return tokens_by_id

    @staticmethod
    def _structured_visible_family_tokens(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        tokens_by_family: dict[str, set[str]] = {}
        shared_ops_by_family = OptimizationWorkspace._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                family = str(failure.get("likely_root_cause_family") or "").strip()
                if not family:
                    continue
                shared_ops = shared_ops_by_family.get(family, set())
                if not shared_ops:
                    continue
                failure_ops = {
                    str(value).strip()
                    for value in failure.get("suggested_operation_families", []) or []
                    if str(value).strip()
                }
                if not failure_ops & shared_ops:
                    continue
                tokens = OptimizationWorkspace._structured_failure_tokens(failure)
                if tokens:
                    tokens_by_family.setdefault(family, set()).update(tokens)
        return tokens_by_family

    @staticmethod
    def _structured_visible_family_shared_tokens(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        token_counts_by_family: dict[str, dict[str, int]] = {}
        shared_ops_by_family = OptimizationWorkspace._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                family = str(failure.get("likely_root_cause_family") or "").strip()
                if not family:
                    continue
                shared_ops = shared_ops_by_family.get(family, set())
                if not shared_ops:
                    continue
                failure_ops = {
                    str(value).strip()
                    for value in failure.get("suggested_operation_families", []) or []
                    if str(value).strip()
                }
                if not failure_ops & shared_ops:
                    continue
                tokens = OptimizationWorkspace._structured_failure_tokens(failure)
                if not tokens:
                    continue
                token_counts = token_counts_by_family.setdefault(family, {})
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
        return {
            family: {token for token, count in counts.items() if count >= 2}
            for family, counts in token_counts_by_family.items()
            if any(count >= 2 for count in counts.values())
        }

    @staticmethod
    def _structured_visible_family_shared_missing_expected_tokens(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        token_counts_by_family: dict[str, dict[str, int]] = {}
        shared_ops_by_family = OptimizationWorkspace._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(failures, list):
            for failure in failures:
                if not isinstance(failure, dict):
                    continue
                family = str(failure.get("likely_root_cause_family") or "").strip()
                if not family:
                    continue
                shared_ops = shared_ops_by_family.get(family, set())
                if not shared_ops:
                    continue
                failure_ops = {
                    str(value).strip()
                    for value in failure.get("suggested_operation_families", []) or []
                    if str(value).strip()
                }
                if not failure_ops & shared_ops:
                    continue
                expected_tokens = {
                    str(value).strip()
                    for value in failure.get("missing_expected_tokens", []) or []
                    if str(value).strip()
                }
                if not expected_tokens:
                    continue
                token_counts = token_counts_by_family.setdefault(family, {})
                for token in expected_tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
        return {
            family: {token for token, count in counts.items() if count >= 2}
            for family, counts in token_counts_by_family.items()
            if any(count >= 2 for count in counts.values())
        }

    @staticmethod
    def _structured_visible_failure_sql_delta_intents(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        intents_by_id: dict[str, set[str]] = {}
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if not isinstance(failures, list):
            return intents_by_id
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            question_id = str(failure.get("question_id") or "").strip()
            if not question_id:
                continue
            intents = set(sql_delta_intents_from_visible_failure(failure))
            if intents:
                intents_by_id[question_id] = intents
        return intents_by_id

    @staticmethod
    def _structured_visible_family_sql_delta_intents(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, set[str]]:
        intents_by_family: dict[str, set[str]] = {}
        shared_ops_by_family = OptimizationWorkspace._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if not isinstance(failures, list):
            return intents_by_family
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            family = str(failure.get("likely_root_cause_family") or "").strip()
            if not family:
                continue
            shared_ops = shared_ops_by_family.get(family, set())
            if not shared_ops:
                continue
            failure_ops = {
                str(value).strip()
                for value in failure.get("suggested_operation_families", []) or []
                if str(value).strip()
            }
            if not failure_ops & shared_ops:
                continue
            intents = set(sql_delta_intents_from_visible_failure(failure))
            if intents:
                intents_by_family.setdefault(family, set()).update(intents)
        return intents_by_family

    @staticmethod
    def _structured_failure_tokens(failure: dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for key in ("missing_expected_tokens", "extra_generated_tokens"):
            values = failure.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                token = str(value).strip()
                if token and len(token) >= 3:
                    tokens.add(token)
        return tokens

    @staticmethod
    def _structured_failure_expected_literal_tokens(
        failure: dict[str, Any],
    ) -> set[str]:
        values = failure.get("missing_expected_tokens")
        if not isinstance(values, list):
            return set()
        literals: set[str] = set()
        for value in values:
            token = str(value).strip()
            if not token or len(token) < 2:
                continue
            if (
                (token.startswith("'") and token.endswith("'"))
                or (token.startswith('"') and token.endswith('"'))
                or re.fullmatch(r"-?\d+(?:\.\d+)?", token)
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}", token)
            ):
                literals.add(token)
        return literals

    @staticmethod
    def _structured_failure_nonliteral_missing_expected_tokens(
        failure: dict[str, Any],
    ) -> set[str]:
        values = failure.get("missing_expected_tokens")
        if not isinstance(values, list):
            return set()
        literal_tokens = OptimizationWorkspace._structured_failure_expected_literal_tokens(
            failure
        )
        tokens: set[str] = set()
        for value in values:
            token = str(value).strip()
            if token and len(token) >= 3 and token not in literal_tokens:
                tokens.add(token)
        return tokens

    @staticmethod
    def _structured_visible_pass_guards(
        structured_failure_context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        pass_guards = (
            structured_failure_context.get("pass_guards", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        return [
            guard
            for guard in pass_guards
            if (
                isinstance(guard, dict)
                and str(guard.get("question") or "").strip()
                and str(guard.get("expected_sql") or "").strip()
            )
        ]

    @staticmethod
    def _structured_visible_pass_guard_ids(
        structured_failure_context: dict[str, Any] | None,
    ) -> set[str]:
        question_ids: set[str] = set()
        pass_guards = (
            structured_failure_context.get("pass_guards", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if isinstance(pass_guards, list):
            for guard in pass_guards:
                if not isinstance(guard, dict):
                    continue
                question_id = str(guard.get("question_id") or "").strip()
                if question_id:
                    question_ids.add(question_id)
        return question_ids

    @staticmethod
    def _artifact_patch_global_effect_text(payload: dict[str, Any]) -> str:
        return " ".join(
            str(payload.get(key) or "")
            for key in ("candidate_name", "rationale", "expected_impact", "risk")
        ).lower()

    @staticmethod
    def _artifact_patch_path_operation_families(path: str) -> set[str]:
        if path == "instructions/text_instruction.md":
            return {"append_text_instruction"}
        if path.startswith("instructions/examples/"):
            return {"add_example_sql"}
        if path.startswith("instructions/sql_snippets/"):
            return {"add_sql_snippet"}
        if path.startswith("instructions/join_specs/"):
            return {"add_join_spec"}
        if path.startswith("tables/") or path.startswith("metric_views/"):
            return {"set_column_config", "set_table_description"}
        return set()

    @classmethod
    def _artifact_patch_marker_path_incompatibility(
        cls,
        *,
        path: str,
        markers: list[str],
        failure_suggested_ops: dict[str, set[str]],
        family_suggested_ops: dict[str, set[str]],
    ) -> dict[str, Any] | None:
        path_ops = cls._artifact_patch_path_operation_families(path)
        if not markers or not path_ops:
            return None
        checked_markers: list[dict[str, Any]] = []
        for marker in markers:
            marker_type, _, marker_value = marker.partition(":")
            allowed_ops: set[str] = set()
            if marker_type == "target_train_failure":
                allowed_ops = failure_suggested_ops.get(marker_value, set())
            elif marker_type in {"visible_family", "failure_family"}:
                allowed_ops = family_suggested_ops.get(marker_value, set())
            else:
                return None
            if not allowed_ops:
                return None
            if path_ops & allowed_ops:
                return None
            checked_markers.append(
                {
                    "marker": marker,
                    "suggested_operation_families": sorted(allowed_ops),
                }
            )
        if not checked_markers:
            return None
        return {
            "path": path,
            "path_operation_families": sorted(path_ops),
            "markers": checked_markers,
        }

    @classmethod
    def _artifact_patch_text_edit_token_incompatibility(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        markers: list[str],
        failure_tokens: dict[str, set[str]],
        family_tokens: dict[str, set[str]],
    ) -> dict[str, Any] | None:
        if path != "instructions/text_instruction.md" or not markers:
            return None
        required_tokens: set[str] = set()
        checked_markers: list[str] = []
        for marker in markers:
            marker_type, _, marker_value = marker.partition(":")
            marker_tokens: set[str] = set()
            if marker_type == "target_train_failure":
                marker_tokens = failure_tokens.get(marker_value, set())
            elif marker_type in {"visible_family", "failure_family"}:
                marker_tokens = family_tokens.get(marker_value, set())
            if marker_tokens:
                required_tokens.update(marker_tokens)
                checked_markers.append(marker)
        if not required_tokens:
            return None
        new_text = cls._artifact_patch_new_text_content(edit)
        if cls._text_contains_any_structural_token(new_text, required_tokens):
            return None
        return {
            "path": path,
            "markers": checked_markers,
            "required_tokens": sorted(required_tokens),
        }

    @classmethod
    def _artifact_patch_causal_plan_content_token_incompatibility(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        causal_plan_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not causal_plan_summary.get("present"):
            return None
        plan_tokens = {
            str(token).strip()
            for token in causal_plan_summary.get("structural_tokens", [])
            if str(token).strip()
        }
        if not plan_tokens:
            return None
        new_text = cls._artifact_patch_new_text_content(edit)
        if cls._text_contains_any_structural_token(new_text, plan_tokens):
            return None
        return {
            "path": path,
            "selected_marker": causal_plan_summary.get("selected_marker"),
            "plan_structural_tokens": sorted(plan_tokens),
        }

    @classmethod
    def _artifact_patch_metadata_query_plan_incompatibility(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
    ) -> dict[str, Any] | None:
        if not (path.startswith("tables/") or path.startswith("metric_views/")):
            return None
        new_text = cls._artifact_patch_new_text_content(edit).lower()
        if not new_text.strip():
            return None
        matched_query_terms = sorted(
            term for term in _METADATA_QUERY_PLAN_TERMS if term in new_text
        )
        matched_result_tokens = sorted(
            token for token in _METADATA_RESULT_SCHEMA_TOKENS if token in new_text
        )
        if not matched_query_terms or len(matched_result_tokens) < 3:
            return None
        return {
            "path": path,
            "matched_query_plan_terms": matched_query_terms,
            "matched_result_schema_tokens": matched_result_tokens,
        }

    @classmethod
    def _artifact_patch_executable_literal_token_gap(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        causal_plan_summary: dict[str, Any],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not serving_candidate_is_executable_artifact_path(path):
            return None
        if not causal_plan_summary.get("present"):
            return None
        if causal_plan_summary.get("selected_family_marker"):
            return None
        if str(causal_plan_summary.get("family") or "") != "literal_or_parameter_mismatch":
            return None
        target_id = str(causal_plan_summary.get("target_train_failure_id") or "").strip()
        if not target_id:
            return None
        target_failure = cls._structured_visible_failures_by_id(
            structured_failure_context
        ).get(target_id)
        if not isinstance(target_failure, dict):
            return None
        required_literals = cls._structured_failure_expected_literal_tokens(
            target_failure
        )
        if not required_literals:
            return None

        plan_tokens = {
            str(token).strip()
            for token in causal_plan_summary.get("structural_tokens", [])
            if str(token).strip()
        }
        plan_has_literal = cls._text_contains_any_structural_token(
            " ".join(plan_tokens),
            required_literals,
        )
        content_has_literal = cls._text_contains_any_structural_token(
            cls._artifact_patch_edit_content_text(edit),
            required_literals,
        )
        if plan_has_literal and content_has_literal:
            return None
        return {
            "path": path,
            "selected_marker": causal_plan_summary.get("selected_marker"),
            "target_train_failure_id": target_id,
            "required_expected_literals": sorted(required_literals),
            "plan_structural_tokens": sorted(plan_tokens),
            "plan_has_expected_literal": plan_has_literal,
            "content_has_expected_literal": content_has_literal,
        }

    @classmethod
    def _artifact_patch_executable_missing_expected_token_gap(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        causal_plan_summary: dict[str, Any],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not serving_candidate_is_executable_artifact_path(path):
            return None
        if not causal_plan_summary.get("present"):
            return None
        if causal_plan_summary.get("selected_family_marker"):
            return None
        target_id = str(causal_plan_summary.get("target_train_failure_id") or "").strip()
        if not target_id:
            return None
        target_failure = cls._structured_visible_failures_by_id(
            structured_failure_context
        ).get(target_id)
        if not isinstance(target_failure, dict):
            return None
        required_tokens = cls._structured_failure_nonliteral_missing_expected_tokens(
            target_failure
        )
        if not required_tokens:
            return None

        plan_tokens = {
            str(token).strip()
            for token in causal_plan_summary.get("structural_tokens", [])
            if str(token).strip()
        }
        plan_has_expected_token = cls._text_contains_any_structural_token(
            " ".join(plan_tokens),
            required_tokens,
        )
        content_has_expected_token = cls._text_contains_any_structural_token(
            cls._artifact_patch_edit_content_text(edit),
            required_tokens,
        )
        if plan_has_expected_token and content_has_expected_token:
            return None
        return {
            "path": path,
            "selected_marker": causal_plan_summary.get("selected_marker"),
            "target_train_failure_id": target_id,
            "required_missing_expected_tokens": sorted(required_tokens),
            "plan_structural_tokens": sorted(plan_tokens),
            "plan_has_missing_expected_token": plan_has_expected_token,
            "content_has_missing_expected_token": content_has_expected_token,
        }

    @classmethod
    def _artifact_patch_executable_family_missing_expected_token_gap(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        causal_plan_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not serving_candidate_is_executable_artifact_path(path):
            return None
        if not causal_plan_summary.get("present"):
            return None
        selected_family = str(
            causal_plan_summary.get("selected_family_marker") or ""
        ).strip()
        if not selected_family:
            return None
        required_tokens = {
            str(token).strip()
            for token in causal_plan_summary.get(
                "available_shared_missing_expected_tokens", []
            )
            if str(token).strip()
        }
        if not required_tokens:
            return None

        plan_tokens = {
            str(token).strip()
            for token in causal_plan_summary.get("structural_tokens", [])
            if str(token).strip()
        }
        plan_has_expected_token = cls._text_contains_any_structural_token(
            " ".join(plan_tokens),
            required_tokens,
        )
        content_has_expected_token = cls._text_contains_any_structural_token(
            cls._artifact_patch_edit_content_text(edit),
            required_tokens,
        )
        if plan_has_expected_token and content_has_expected_token:
            return None
        return {
            "path": path,
            "selected_marker": causal_plan_summary.get("selected_marker"),
            "selected_family": selected_family,
            "required_shared_missing_expected_tokens": sorted(required_tokens),
            "plan_structural_tokens": sorted(plan_tokens),
            "plan_has_shared_missing_expected_token": plan_has_expected_token,
            "content_has_shared_missing_expected_token": content_has_expected_token,
        }

    @classmethod
    def _artifact_patch_causal_plan_new_file_incompatibility(
        cls,
        *,
        causal_plan_summary: dict[str, Any],
        changed_edits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not causal_plan_summary.get("present"):
            return []
        operation_families = {
            str(value).strip()
            for value in causal_plan_summary.get("available_operation_families", [])
            if str(value).strip()
        }
        if not operation_families:
            return []
        if cls._artifact_patch_plan_allows_new_file(operation_families):
            return []
        return [
            {
                "path": str(edit.get("path") or ""),
                "selected_marker": causal_plan_summary.get("selected_marker"),
                "operation_families": sorted(operation_families),
                "allowed_new_file_operation_families": sorted(
                    cls._artifact_patch_new_file_operation_families()
                ),
            }
            for edit in changed_edits
            if str(edit.get("action") or "") == "created"
            and str(edit.get("path") or "").strip()
        ]

    @staticmethod
    def _artifact_patch_new_file_operation_families() -> set[str]:
        return {"add_example_sql", "add_sql_snippet", "add_join_spec"}

    @classmethod
    def _artifact_patch_plan_allows_new_file(
        cls,
        operation_families: set[str],
    ) -> bool:
        return bool(
            operation_families.intersection(
                cls._artifact_patch_new_file_operation_families()
            )
        )

    @classmethod
    def _artifact_patch_causal_plan_full_file_rewrite_incompatibility(
        cls,
        *,
        causal_plan_summary: dict[str, Any],
        changed_edits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not causal_plan_summary.get("present"):
            return []
        return [
            {
                "path": str(edit.get("path") or ""),
                "selected_marker": causal_plan_summary.get("selected_marker"),
                "edit_mode": str(edit.get("edit_mode") or ""),
                "action": str(edit.get("action") or ""),
            }
            for edit in changed_edits
            if str(edit.get("action") or "") == "updated"
            and str(edit.get("edit_mode") or "") == "replace_file"
            and str(edit.get("path") or "").strip()
        ]

    @classmethod
    def _artifact_patch_instruction_rewrite_incompatibility(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
    ) -> dict[str, Any] | None:
        if path != "instructions/text_instruction.md":
            return None
        touched_text = "\n".join(
            str(edit.get(key) or "")
            for key in ("content", "append_content", "find", "replace")
            if edit.get(key) is not None
        )
        if not cls._text_mentions_trusted_query_routing(touched_text):
            return None
        return {
            "path": path,
            "reason": "trusted_query_routing_rewrite",
        }

    @staticmethod
    def _text_mentions_trusted_query_routing(text: str) -> bool:
        lowered = text.lower().replace("_", " ").replace("-", " ")
        if "trusted query" in lowered:
            return True
        if "trusted sql" in lowered:
            return True
        return "trusted" in lowered and "routing" in lowered

    @classmethod
    def _artifact_patch_executable_asset_incompatibility(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        evidence_text: str,
        pass_guards: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not serving_candidate_is_executable_artifact_path(path):
            return None
        content_text = cls._artifact_patch_edit_content_text(edit)
        if not content_text.strip():
            return None
        if path.startswith("instructions/examples/"):
            reasons = cls._example_artifact_patch_content_reasons(
                path=path,
                content_text=content_text,
                evidence_text=evidence_text,
                pass_guards=pass_guards,
                full_file_content=_is_artifact_patch_full_file_content_edit(edit),
            )
        elif path.startswith("instructions/sql_snippets/"):
            reasons = cls._sql_snippet_artifact_patch_content_reasons(
                path=path,
                content_text=content_text,
                full_file_content=_is_artifact_patch_full_file_content_edit(edit),
            )
        else:
            reasons = []
        if not reasons:
            return None
        return {
            "path": path,
            "reasons": list(dict.fromkeys(reasons)),
        }

    @classmethod
    def _artifact_patch_example_target_alignment_incompatibility(
        cls,
        *,
        edit: dict[str, Any],
        path: str,
        causal_plan_summary: dict[str, Any],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not path.startswith("instructions/examples/"):
            return None
        if not causal_plan_summary.get("present"):
            return None
        target_id = str(
            causal_plan_summary.get("target_train_failure_id") or ""
        ).strip()
        example_question_sql = _extract_example_question_sql(
            cls._artifact_patch_edit_content_text(edit)
        )
        if example_question_sql is None:
            return None
        candidate_question, _sql = example_question_sql
        candidate_terms = _literal_swap_question_terms(candidate_question)
        if not candidate_terms:
            return None
        if target_id:
            target_failure = cls._structured_visible_failures_by_id(
                structured_failure_context
            ).get(target_id)
            if not isinstance(target_failure, dict):
                return None
            target_question = str(target_failure.get("question") or "").strip()
            if not target_question:
                return None
            target_terms = _literal_swap_question_terms(target_question)
            if not target_terms:
                return None
            overlap = sorted(target_terms & candidate_terms)
            if overlap:
                return None
            return {
                "path": path,
                "selected_marker": causal_plan_summary.get("selected_marker"),
                "target_train_failure_id": target_id,
                "target_question_terms": sorted(target_terms)[:12],
                "candidate_question_terms": sorted(candidate_terms)[:12],
            }

        selected_family = str(
            causal_plan_summary.get("selected_family_marker")
            or causal_plan_summary.get("family")
            or ""
        ).strip()
        if not selected_family:
            return None
        family_question_terms = cls._structured_visible_family_question_terms(
            structured_failure_context,
            family=selected_family,
        )
        if not family_question_terms:
            return None
        family_term_union = {
            term
            for terms in family_question_terms.values()
            for term in terms
        }
        if candidate_terms & family_term_union:
            return None
        return {
            "path": path,
            "selected_marker": causal_plan_summary.get("selected_marker"),
            "selected_family": selected_family,
            "family_question_terms": {
                question_id: sorted(terms)[:12]
                for question_id, terms in sorted(family_question_terms.items())
            },
            "candidate_question_terms": sorted(candidate_terms)[:12],
        }

    @classmethod
    def _artifact_patch_causal_plan_summary(
        cls,
        *,
        payload: dict[str, Any],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw_plan = payload.get("causal_patch_plan")
        failure_ids = cls._structured_visible_failure_ids(structured_failure_context)
        families = cls._structured_visible_failure_families(structured_failure_context)
        failure_family_by_id = cls._structured_visible_failure_family_by_id(
            structured_failure_context
        )
        failure_suggested_ops_by_id = cls._structured_visible_failure_suggested_ops(
            structured_failure_context
        )
        family_suggested_ops_by_family = cls._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        failure_tokens_by_id = cls._structured_visible_failure_tokens(
            structured_failure_context
        )
        family_tokens_by_family = cls._structured_visible_family_tokens(
            structured_failure_context
        )
        family_shared_tokens_by_family = (
            cls._structured_visible_family_shared_tokens(structured_failure_context)
        )
        family_shared_missing_expected_tokens_by_family = (
            cls._structured_visible_family_shared_missing_expected_tokens(
                structured_failure_context
            )
        )
        failure_sql_delta_intents_by_id = (
            cls._structured_visible_failure_sql_delta_intents(
                structured_failure_context
            )
        )
        family_sql_delta_intents_by_family = (
            cls._structured_visible_family_sql_delta_intents(
                structured_failure_context
            )
        )
        if not isinstance(raw_plan, dict):
            return {
                "present": False,
                "matches_visible_target": False,
                "matches_visible_family": False,
                "policy": "missing_plan_non_gating",
            }

        selected_marker = str(raw_plan.get("selected_marker") or "").strip()
        family = str(raw_plan.get("family") or "").strip()
        target_id = ""
        selected_family_marker = ""
        if selected_marker.startswith("target_train_failure:"):
            target_id = selected_marker.partition(":")[2]
        elif selected_marker.startswith(("visible_family:", "failure_family:")):
            selected_family_marker = selected_marker.partition(":")[2]
        target_family = failure_family_by_id.get(target_id, "")
        selected_family_marker_matches_plan = (
            True if not selected_family_marker else bool(family == selected_family_marker)
        )
        matches_visible_family = (
            bool(family and family == target_family)
            if target_id
            else bool(
                selected_family_marker
                and selected_family_marker in families
                and selected_family_marker_matches_plan
            )
        )
        available_structural_tokens = (
            failure_tokens_by_id.get(target_id, set())
            if target_id
            else family_tokens_by_family.get(selected_family_marker or family, set())
        )
        available_shared_structural_tokens = (
            set()
            if target_id
            else family_shared_tokens_by_family.get(
                selected_family_marker or family,
                set(),
            )
        )
        available_shared_missing_expected_tokens = (
            set()
            if target_id
            else family_shared_missing_expected_tokens_by_family.get(
                selected_family_marker or family,
                set(),
            )
        )
        available_operation_families = (
            failure_suggested_ops_by_id.get(target_id, set())
            if target_id
            else family_suggested_ops_by_family.get(
                selected_family_marker or family,
                set(),
            )
        )
        available_sql_delta_intents = (
            failure_sql_delta_intents_by_id.get(target_id, set())
            if target_id
            else family_sql_delta_intents_by_family.get(
                selected_family_marker or family,
                set(),
            )
        )
        recipe_anchors = artifact_patch_causal_recipe_anchors(
            family=family,
            operation_families=sorted(available_operation_families),
            tokens=sorted(available_structural_tokens),
        )
        plan_structural_tokens = [
            str(value).strip()
            for value in raw_plan.get("structural_tokens", [])
            if str(value).strip()
        ] if isinstance(raw_plan.get("structural_tokens"), list) else []
        expected_visible_effect = str(
            raw_plan.get("expected_visible_effect") or ""
        ).strip()
        selected_sql_delta_intents = sorted(
            intent
            for intent in available_sql_delta_intents
            if intent in expected_visible_effect
        )
        return {
            "present": True,
            "selected_marker": selected_marker,
            "target_train_failure_id": target_id,
            "target_family": target_family,
            "selected_family_marker": selected_family_marker,
            "family": family,
            "root_cause": str(raw_plan.get("root_cause") or "").strip(),
            "edit_surface_evidence": str(
                raw_plan.get("edit_surface_evidence") or ""
            ).strip(),
            "repair_shape": str(raw_plan.get("repair_shape") or "").strip(),
            "avoid": str(raw_plan.get("avoid") or "").strip(),
            "expected_visible_effect": expected_visible_effect,
            "compatible_paths": [
                str(value).strip()
                for value in raw_plan.get("compatible_paths", [])
                if str(value).strip()
            ]
            if isinstance(raw_plan.get("compatible_paths"), list)
            else [],
            "structural_tokens": plan_structural_tokens,
            "available_structural_tokens": sorted(available_structural_tokens),
            "available_shared_structural_tokens": sorted(
                available_shared_structural_tokens
            ),
            "available_shared_missing_expected_tokens": sorted(
                available_shared_missing_expected_tokens
            ),
            "available_operation_families": sorted(available_operation_families),
            "available_sql_delta_intents": sorted(available_sql_delta_intents),
            "selected_sql_delta_intents": selected_sql_delta_intents,
            "recipe_anchors": recipe_anchors,
            "root_cause_matches_visible_evidence": cls._text_contains_any_recipe_anchor(
                str(raw_plan.get("root_cause") or ""),
                recipe_anchors.get("root_cause", []),
            ),
            "edit_surface_matches_visible_evidence": cls._text_contains_any_recipe_anchor(
                str(raw_plan.get("edit_surface_evidence") or ""),
                recipe_anchors.get("edit_surface", []),
            ),
            "repair_shape_matches_recipe": cls._text_contains_any_recipe_anchor(
                str(raw_plan.get("repair_shape") or ""),
                recipe_anchors.get("repair_shape", []),
            ),
            "avoid_matches_recipe": cls._text_contains_any_recipe_anchor(
                str(raw_plan.get("avoid") or ""),
                recipe_anchors.get("avoid", []),
            ),
            "structural_tokens_match_visible_evidence": (
                True
                if not available_structural_tokens
                else cls._structural_tokens_are_from_visible_evidence(
                    plan_structural_tokens,
                    available_structural_tokens,
                )
            ),
            "expected_visible_effect_matches_visible_evidence": (
                True
                if not available_structural_tokens
                else cls._text_contains_any_structural_token(
                    str(raw_plan.get("expected_visible_effect") or ""),
                    available_structural_tokens,
                )
            ),
            "structural_tokens_include_shared_family_token": (
                True
                if not available_shared_structural_tokens
                else any(
                    cls._structural_token_matches_available_token(
                        plan_token,
                        available_shared_structural_tokens,
                    )
                    for plan_token in plan_structural_tokens
                )
            ),
            "expected_visible_effect_matches_shared_family_token": (
                True
                if not available_shared_structural_tokens
                else cls._text_contains_any_structural_token(
                    str(raw_plan.get("expected_visible_effect") or ""),
                    available_shared_structural_tokens,
                )
            ),
            "structural_tokens_include_shared_missing_expected_token": (
                True
                if not available_shared_missing_expected_tokens
                else any(
                    cls._structural_token_matches_available_token(
                        plan_token,
                        available_shared_missing_expected_tokens,
                    )
                    for plan_token in plan_structural_tokens
                )
            ),
            "selected_family_marker_matches_plan": selected_family_marker_matches_plan,
            "matches_visible_target": bool(target_id and target_id in failure_ids),
            "matches_visible_family": matches_visible_family,
            "policy": "present_plan_consistency_checked",
        }

    @classmethod
    def _artifact_patch_causal_plan_inconsistency(
        cls,
        *,
        causal_plan_summary: dict[str, Any],
        edit_mappings: list[dict[str, Any]],
        global_markers: list[str],
        changed_paths: set[str],
    ) -> dict[str, Any] | None:
        if not causal_plan_summary.get("present"):
            return None

        selected_marker = str(causal_plan_summary.get("selected_marker") or "").strip()
        plan_markers = cls._causal_plan_selected_marker_candidates(causal_plan_summary)
        patch_markers: list[str] = []
        patch_markers.extend(global_markers)
        for mapping in edit_mappings:
            markers = mapping.get("effect_markers")
            if isinstance(markers, list):
                patch_markers.extend(str(marker) for marker in markers if str(marker).strip())
        patch_markers = list(dict.fromkeys(patch_markers))

        if not selected_marker:
            return {
                "reason": "missing_selected_marker",
                "selected_marker": selected_marker,
                "patch_markers": patch_markers,
            }
        if not cls._causal_plan_selected_marker_has_exact_shape(
            selected_marker=selected_marker
        ):
            return {
                "reason": "selected_marker_not_exact_visible_marker",
                "selected_marker": selected_marker,
                "required_marker_shapes": [
                    "target_train_failure:<visible_failure_id>",
                    "visible_family:<visible_failure_family>",
                    "failure_family:<visible_failure_family>",
                ],
                "patch_markers": patch_markers,
            }
        if not str(causal_plan_summary.get("family") or "").strip():
            return {
                "reason": "missing_family",
                "selected_marker": selected_marker,
                "target_family": causal_plan_summary.get("target_family"),
                "selected_family_marker": causal_plan_summary.get(
                    "selected_family_marker"
                ),
            }
        if (
            causal_plan_summary.get("selected_family_marker")
            and not causal_plan_summary.get("selected_family_marker_matches_plan")
        ):
            return {
                "reason": "selected_family_marker_mismatch",
                "selected_marker": selected_marker,
                "selected_family_marker": causal_plan_summary.get(
                    "selected_family_marker"
                ),
                "plan_family": causal_plan_summary.get("family"),
            }
        if not (
            causal_plan_summary.get("matches_visible_target")
            or causal_plan_summary.get("matches_visible_family")
        ):
            return {
                "reason": "selected_marker_not_visible",
                "selected_marker": selected_marker,
                "patch_markers": patch_markers,
            }
        if causal_plan_summary.get("selected_family_marker") and not (
            causal_plan_summary.get("available_operation_families") or []
        ):
            return {
                "reason": "family_marker_without_shared_edit_surface",
                "selected_marker": selected_marker,
                "selected_family_marker": causal_plan_summary.get(
                    "selected_family_marker"
                ),
            }
        if (
            causal_plan_summary.get("target_train_failure_id")
            and causal_plan_summary.get("family")
            and not causal_plan_summary.get("matches_visible_family")
        ):
            return {
                "reason": "selected_target_family_mismatch",
                "selected_marker": selected_marker,
                "target_train_failure_id": causal_plan_summary.get(
                    "target_train_failure_id"
                ),
                "target_family": causal_plan_summary.get("target_family"),
                "plan_family": causal_plan_summary.get("family"),
            }
        compatible_paths = [
            str(path).strip()
            for path in causal_plan_summary.get("compatible_paths", [])
            if str(path).strip()
        ]
        if not compatible_paths:
            return {
                "reason": "missing_compatible_paths",
                "selected_marker": selected_marker,
            }
        if not str(causal_plan_summary.get("root_cause") or "").strip():
            return {
                "reason": "missing_root_cause",
                "selected_marker": selected_marker,
            }
        if not causal_plan_summary.get("root_cause_matches_visible_evidence"):
            return {
                "reason": "root_cause_not_from_visible_evidence",
                "selected_marker": selected_marker,
                "root_cause": causal_plan_summary.get("root_cause"),
                "recipe_anchors": causal_plan_summary.get("recipe_anchors", {}).get(
                    "root_cause",
                    [],
                ),
            }
        if not str(causal_plan_summary.get("edit_surface_evidence") or "").strip():
            return {
                "reason": "missing_edit_surface_evidence",
                "selected_marker": selected_marker,
            }
        if not causal_plan_summary.get("edit_surface_matches_visible_evidence"):
            return {
                "reason": "edit_surface_evidence_not_from_visible_evidence",
                "selected_marker": selected_marker,
                "edit_surface_evidence": causal_plan_summary.get(
                    "edit_surface_evidence"
                ),
                "recipe_anchors": causal_plan_summary.get("recipe_anchors", {}).get(
                    "edit_surface",
                    [],
                ),
            }
        if not str(causal_plan_summary.get("repair_shape") or "").strip():
            return {
                "reason": "missing_repair_shape",
                "selected_marker": selected_marker,
            }
        if not str(causal_plan_summary.get("avoid") or "").strip():
            return {
                "reason": "missing_avoid",
                "selected_marker": selected_marker,
            }
        if not str(causal_plan_summary.get("expected_visible_effect") or "").strip():
            return {
                "reason": "missing_expected_visible_effect",
                "selected_marker": selected_marker,
            }
        if (
            causal_plan_summary.get("available_structural_tokens")
            and not causal_plan_summary.get("structural_tokens_match_visible_evidence")
        ):
            return {
                "reason": "structural_tokens_not_from_visible_evidence",
                "selected_marker": selected_marker,
                "plan_structural_tokens": causal_plan_summary.get("structural_tokens"),
                "available_structural_tokens": causal_plan_summary.get(
                    "available_structural_tokens"
                ),
            }
        if (
            causal_plan_summary.get("selected_family_marker")
            and causal_plan_summary.get("available_shared_structural_tokens")
            and not causal_plan_summary.get(
                "structural_tokens_include_shared_family_token"
            )
        ):
            return {
                "reason": "family_structural_tokens_missing_shared_evidence",
                "selected_marker": selected_marker,
                "plan_structural_tokens": causal_plan_summary.get("structural_tokens"),
                "available_shared_structural_tokens": causal_plan_summary.get(
                    "available_shared_structural_tokens"
                ),
            }
        if (
            causal_plan_summary.get("available_structural_tokens")
            and not causal_plan_summary.get(
                "expected_visible_effect_matches_visible_evidence"
            )
        ):
            return {
                "reason": "expected_visible_effect_not_from_visible_evidence",
                "selected_marker": selected_marker,
                "expected_visible_effect": causal_plan_summary.get(
                    "expected_visible_effect"
                ),
                "available_structural_tokens": causal_plan_summary.get(
                    "available_structural_tokens"
                ),
            }
        if (
            causal_plan_summary.get("selected_family_marker")
            and causal_plan_summary.get("available_shared_structural_tokens")
            and not causal_plan_summary.get(
                "expected_visible_effect_matches_shared_family_token"
            )
        ):
            return {
                "reason": "expected_visible_effect_missing_shared_family_token",
                "selected_marker": selected_marker,
                "expected_visible_effect": causal_plan_summary.get(
                    "expected_visible_effect"
                ),
                "available_shared_structural_tokens": causal_plan_summary.get(
                    "available_shared_structural_tokens"
                ),
            }
        if not cls._marker_sets_intersect(plan_markers, patch_markers):
            return {
                "reason": "selected_marker_not_used_by_patch",
                "selected_marker": selected_marker,
                "candidate_markers": sorted(plan_markers),
                "patch_markers": patch_markers,
            }
        edit_paths_without_selected_marker = sorted(
            str(mapping.get("path") or "")
            for mapping in edit_mappings
            if not cls._marker_sets_intersect(
                plan_markers,
                mapping.get("local_effect_markers")
                if isinstance(mapping.get("local_effect_markers"), list)
                else [],
            )
        )
        if edit_paths_without_selected_marker:
            return {
                "reason": "selected_marker_not_used_by_each_edit",
                "selected_marker": selected_marker,
                "candidate_markers": sorted(plan_markers),
                "edit_paths_without_selected_marker": edit_paths_without_selected_marker,
            }

        incompatible_paths = sorted(
            path
            for path in changed_paths
            if not cls._path_matches_any_artifact_hint(path, compatible_paths)
        )
        if incompatible_paths:
            return {
                "reason": "changed_path_not_in_plan_compatible_paths",
                "selected_marker": selected_marker,
                "compatible_paths": compatible_paths,
                "changed_paths": sorted(changed_paths),
                "incompatible_paths": incompatible_paths,
            }
        if not causal_plan_summary.get("repair_shape_matches_recipe"):
            return {
                "reason": "repair_shape_not_from_recipe",
                "selected_marker": selected_marker,
                "repair_shape": causal_plan_summary.get("repair_shape"),
                "recipe_anchors": causal_plan_summary.get("recipe_anchors", {}).get(
                    "repair_shape",
                    [],
                ),
            }
        if not causal_plan_summary.get("avoid_matches_recipe"):
            return {
                "reason": "avoid_not_from_recipe",
                "selected_marker": selected_marker,
                "avoid": causal_plan_summary.get("avoid"),
                "recipe_anchors": causal_plan_summary.get("recipe_anchors", {}).get(
                    "avoid",
                    [],
                ),
            }
        return None

    @staticmethod
    def _text_contains_any_recipe_anchor(text: str, anchors: list[str]) -> bool:
        if not anchors:
            return True
        lowered = text.lower()
        return any(
            str(anchor).strip().lower() in lowered
            for anchor in anchors
            if str(anchor).strip()
        )

    @staticmethod
    def _artifact_patch_high_risk_guard_gap(
        *,
        edit_mappings: list[dict[str, Any]],
        global_markers: list[str],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        all_markers: list[str] = []
        all_markers.extend(global_markers)
        for mapping in edit_mappings:
            markers = mapping.get("effect_markers")
            if isinstance(markers, list):
                all_markers.extend(str(marker) for marker in markers if str(marker).strip())
        all_markers = list(dict.fromkeys(all_markers))
        target_ids = sorted(
            {
                marker.partition(":")[2]
                for marker in all_markers
                if marker.startswith("target_train_failure:")
                and marker.partition(":")[2]
            }
        )
        family_markers = sorted(
            {
                marker.partition(":")[2]
                for marker in all_markers
                if marker.startswith(("visible_family:", "failure_family:"))
                and marker.partition(":")[2]
            }
        )
        has_single_target_plan = len(target_ids) == 1 and not family_markers
        has_single_family_plan = len(family_markers) == 1 and not target_ids
        if not (has_single_target_plan or has_single_family_plan):
            return None

        guard_ids = sorted(
            {
                marker.partition(":")[2]
                for marker in all_markers
                if marker.startswith("guard_preservation:")
                and marker.partition(":")[2]
            }
        )
        if guard_ids:
            return None

        visible_guard_ids = sorted(
            OptimizationWorkspace._structured_visible_pass_guard_ids(
                structured_failure_context
            )
        )
        if not visible_guard_ids:
            return None

        guard_risk_by_id = (
            OptimizationWorkspace._structured_visible_failure_guard_risk_by_id(
                structured_failure_context
            )
        )
        if has_single_target_plan:
            target_id = target_ids[0]
            target_guard_risk = guard_risk_by_id.get(target_id, "").lower()
            if target_guard_risk != "high":
                return None
            return {
                "target_train_failure_id": target_id,
                "guard_regression_risk": target_guard_risk,
                "visible_pass_guard_ids": visible_guard_ids,
                "required_marker_shape": "guard_preservation:<visible_pass_guard_id>",
            }

        family = family_markers[0]
        high_guard_risk_member_ids = (
            OptimizationWorkspace._structured_visible_family_high_guard_risk_member_ids(
                structured_failure_context,
                family=family,
            )
        )
        if not high_guard_risk_member_ids:
            return None
        return {
            "visible_family": family,
            "guard_regression_risk": "high",
            "high_guard_risk_member_ids": high_guard_risk_member_ids,
            "visible_pass_guard_ids": visible_guard_ids,
            "required_marker_shape": "guard_preservation:<visible_pass_guard_id>",
        }

    @staticmethod
    def _causal_plan_selected_marker_candidates(
        causal_plan_summary: dict[str, Any],
    ) -> set[str]:
        selected_marker = str(causal_plan_summary.get("selected_marker") or "").strip()
        candidates = {selected_marker} if selected_marker else set()
        target_id = str(causal_plan_summary.get("target_train_failure_id") or "").strip()
        if target_id:
            candidates.add(f"target_train_failure:{target_id}")
        return candidates

    @staticmethod
    def _causal_plan_selected_marker_has_exact_shape(*, selected_marker: str) -> bool:
        marker = selected_marker.strip()
        if marker.startswith("target_train_failure:"):
            return bool(marker.partition(":")[2])
        if marker.startswith("visible_family:"):
            return bool(marker.partition(":")[2])
        if marker.startswith("failure_family:"):
            return bool(marker.partition(":")[2])
        return False

    @staticmethod
    def _marker_sets_intersect(left: set[str], right: list[str]) -> bool:
        left_lower = {marker.lower() for marker in left if marker}
        right_lower = {marker.lower() for marker in right if marker}
        return bool(left_lower & right_lower)

    @classmethod
    def _path_matches_any_artifact_hint(cls, path: str, hints: list[str]) -> bool:
        normalized_path = path.strip()
        for hint in hints:
            for pattern in cls._expand_artifact_path_hint(hint):
                if fnmatch.fnmatchcase(normalized_path, pattern):
                    return True
        return False

    @staticmethod
    def _expand_artifact_path_hint(hint: str) -> list[str]:
        if "{" not in hint or "}" not in hint:
            return [hint]
        prefix, _, remainder = hint.partition("{")
        choices, _, suffix = remainder.partition("}")
        if not choices:
            return [hint]
        return [f"{prefix}{choice.strip()}{suffix}" for choice in choices.split(",")]

    @classmethod
    def _artifact_patch_target_plan_inconsistency(
        cls,
        *,
        edit_mappings: list[dict[str, Any]],
        global_markers: list[str],
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        all_markers: list[str] = []
        all_markers.extend(global_markers)
        for mapping in edit_mappings:
            markers = mapping.get("effect_markers")
            if not isinstance(markers, list):
                continue
            all_markers.extend(str(marker) for marker in markers if str(marker).strip())
        all_markers = list(dict.fromkeys(all_markers))

        target_ids = sorted(
            {
                marker.partition(":")[2]
                for marker in all_markers
                if marker.startswith("target_train_failure:")
                and marker.partition(":")[2]
            }
        )
        family_markers = sorted(
            {
                marker.partition(":")[2]
                for marker in all_markers
                if (
                    marker.startswith("visible_family:")
                    or marker.startswith("failure_family:")
                )
                and marker.partition(":")[2]
            }
        )
        failure_family_by_id = cls._structured_visible_failure_family_by_id(
            structured_failure_context
        )
        target_families = sorted(
            {
                failure_family_by_id[target_id]
                for target_id in target_ids
                if failure_family_by_id.get(target_id)
            }
        )

        if not target_ids and not family_markers:
            return {
                "reason": "missing_improvement_target_marker",
                "target_train_failure_ids": target_ids,
                "family_markers": family_markers,
                "target_families": target_families,
            }
        if len(target_ids) > 1:
            return {
                "reason": "multiple_target_train_failure_markers",
                "target_train_failure_ids": target_ids,
                "family_markers": family_markers,
                "target_families": target_families,
            }
        if len(family_markers) > 1:
            return {
                "reason": "multiple_visible_family_markers",
                "target_train_failure_ids": target_ids,
                "family_markers": family_markers,
                "target_families": target_families,
            }
        if target_families and family_markers and not set(target_families) & set(
            family_markers
        ):
            return {
                "reason": "target_family_marker_mismatch",
                "target_train_failure_ids": target_ids,
                "family_markers": family_markers,
                "target_families": target_families,
            }
        return None

    @staticmethod
    def _structured_visible_failure_family_by_id(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, str]:
        families_by_id: dict[str, str] = {}
        for failure in OptimizationWorkspace._structured_visible_failures_by_id(
            structured_failure_context
        ).values():
            question_id = str(failure.get("question_id") or "").strip()
            family = str(failure.get("likely_root_cause_family") or "").strip()
            if question_id and family:
                families_by_id[question_id] = family
        return families_by_id

    @staticmethod
    def _structured_visible_failures_by_id(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        failures_by_id: dict[str, dict[str, Any]] = {}
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if not isinstance(failures, list):
            return failures_by_id
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            question_id = str(failure.get("question_id") or "").strip()
            if question_id:
                failures_by_id[question_id] = failure
        return failures_by_id

    @staticmethod
    def _structured_visible_family_question_terms(
        structured_failure_context: dict[str, Any] | None,
        *,
        family: str,
    ) -> dict[str, set[str]]:
        terms_by_id: dict[str, set[str]] = {}
        selected_family = family.strip()
        if not selected_family:
            return terms_by_id
        shared_ops_by_family = OptimizationWorkspace._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        shared_ops = shared_ops_by_family.get(selected_family, set())
        if not shared_ops:
            return terms_by_id
        for failure in OptimizationWorkspace._structured_visible_failures_by_id(
            structured_failure_context
        ).values():
            question_id = str(failure.get("question_id") or "").strip()
            failure_family = str(
                failure.get("likely_root_cause_family") or ""
            ).strip()
            if not question_id or failure_family != selected_family:
                continue
            failure_ops = {
                str(value).strip()
                for value in failure.get("suggested_operation_families", []) or []
                if str(value).strip()
            }
            if not failure_ops & shared_ops:
                continue
            question = str(failure.get("question") or "").strip()
            terms = _literal_swap_question_terms(question)
            if terms:
                terms_by_id[question_id] = terms
        return terms_by_id

    @staticmethod
    def _structured_visible_family_high_guard_risk_member_ids(
        structured_failure_context: dict[str, Any] | None,
        *,
        family: str,
    ) -> list[str]:
        selected_family = family.strip()
        if not selected_family:
            return []
        shared_ops_by_family = OptimizationWorkspace._structured_visible_family_suggested_ops(
            structured_failure_context
        )
        shared_ops = shared_ops_by_family.get(selected_family, set())
        if not shared_ops:
            return []

        high_risk_member_ids: list[str] = []
        for question_id, failure in OptimizationWorkspace._structured_visible_failures_by_id(
            structured_failure_context
        ).items():
            failure_family = str(
                failure.get("likely_root_cause_family") or ""
            ).strip()
            if failure_family != selected_family:
                continue
            failure_ops = {
                str(value).strip()
                for value in failure.get("suggested_operation_families", []) or []
                if str(value).strip()
            }
            if not failure_ops & shared_ops:
                continue
            guard_risk = str(failure.get("guard_regression_risk") or "").strip().lower()
            if guard_risk == "high":
                high_risk_member_ids.append(question_id)
        return list(dict.fromkeys(high_risk_member_ids))

    @staticmethod
    def _structured_visible_failure_guard_risk_by_id(
        structured_failure_context: dict[str, Any] | None,
    ) -> dict[str, str]:
        risks_by_id: dict[str, str] = {}
        failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        if not isinstance(failures, list):
            return risks_by_id
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            question_id = str(failure.get("question_id") or "").strip()
            risk = str(failure.get("guard_regression_risk") or "").strip()
            if question_id and risk:
                risks_by_id[question_id] = risk
        return risks_by_id

    @staticmethod
    def _artifact_patch_new_text_content(edit: dict[str, Any]) -> str:
        return "\n".join(
            str(edit.get(key) or "")
            for key in ("content", "append_content", "replace")
            if edit.get(key) is not None
        )

    @staticmethod
    def _text_contains_any_structural_token(text: str, tokens: set[str]) -> bool:
        lowered = text.lower()
        for token in tokens:
            normalized = str(token).strip().lower()
            if not normalized:
                continue
            if normalized in lowered:
                return True
            bare = re.sub(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$", "", normalized)
            if bare and bare in lowered:
                return True
        return False

    @classmethod
    def _structural_tokens_are_from_visible_evidence(
        cls,
        plan_tokens: list[str],
        available_tokens: set[str],
    ) -> bool:
        if not plan_tokens:
            return False
        return all(
            cls._structural_token_matches_available_token(plan_token, available_tokens)
            for plan_token in plan_tokens
        )

    @staticmethod
    def _structural_token_matches_available_token(
        plan_token: str,
        available_tokens: set[str],
    ) -> bool:
        normalized_plan_token = str(plan_token).strip().lower()
        if not normalized_plan_token:
            return False
        bare_plan_token = re.sub(
            r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$",
            "",
            normalized_plan_token,
        )
        for available_token in available_tokens:
            normalized_available_token = str(available_token).strip().lower()
            if not normalized_available_token:
                continue
            if normalized_plan_token == normalized_available_token:
                return True
            bare_available_token = re.sub(
                r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$",
                "",
                normalized_available_token,
            )
            if bare_plan_token and bare_plan_token == bare_available_token:
                return True
        return False

    @staticmethod
    def _artifact_patch_exact_visible_effect_markers(
        *,
        evidence_text: str,
        families: set[str],
        failure_ids: set[str],
        pass_guard_ids: set[str],
    ) -> list[str]:
        lowered = evidence_text.lower()
        markers: list[str] = []
        for question_id in sorted(failure_ids):
            marker = f"target_train_failure:{question_id.lower()}"
            if marker in lowered:
                markers.append(f"target_train_failure:{question_id}")
        for family in sorted(families):
            visible_marker = f"visible_family:{family.lower()}"
            failure_marker = f"failure_family:{family.lower()}"
            if visible_marker in lowered:
                markers.append(f"visible_family:{family}")
            if failure_marker in lowered:
                markers.append(f"failure_family:{family}")
        for question_id in sorted(pass_guard_ids):
            marker = f"guard_preservation:{question_id.lower()}"
            if marker in lowered:
                markers.append(f"guard_preservation:{question_id}")
        return list(dict.fromkeys(markers))

    @staticmethod
    def _artifact_patch_visible_effect_text(
        payload: dict[str, Any],
        executable_edits: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = [
            str(payload.get("candidate_name") or ""),
            str(payload.get("rationale") or ""),
            str(payload.get("expected_impact") or ""),
            str(payload.get("risk") or ""),
        ]
        for edit in executable_edits:
            parts.extend(
                [
                    str(edit.get("path") or edit.get("relative_path") or ""),
                    str(edit.get("reason") or ""),
                    str(edit.get("description") or ""),
                    str(edit.get("comment") or ""),
                ]
            )
        return " ".join(parts).lower()

    @staticmethod
    def _artifact_patch_visible_effect_markers(
        *,
        evidence_text: str,
        families: set[str],
        failure_ids: set[str],
    ) -> list[str]:
        markers: list[str] = []
        for marker in sorted(_EXECUTABLE_ARTIFACT_PATCH_EFFECT_MARKERS):
            if marker in evidence_text:
                markers.append(marker)
        for family in sorted(families | _EXECUTABLE_ARTIFACT_PATCH_VISIBLE_FAMILIES):
            if family and family.lower() in evidence_text:
                markers.append(f"family:{family}")
        for question_id in sorted(failure_ids):
            if question_id.lower() in evidence_text:
                markers.append(f"question:{question_id}")
        return list(dict.fromkeys(markers))

    @staticmethod
    def _artifact_patch_edit_content_text(edit: dict[str, Any]) -> str:
        return "\n".join(
            str(edit.get(key) or "")
            for key in ("content", "append_content", "find", "replace")
            if edit.get(key) is not None
        )

    @staticmethod
    def _artifact_patch_payload_file_edits_by_path(
        payload: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        file_edits = payload.get("file_edits")
        if not isinstance(file_edits, list):
            return {}
        edits_by_path: dict[str, dict[str, Any]] = {}
        for edit in file_edits:
            if not isinstance(edit, dict):
                continue
            path = str(edit.get("path") or edit.get("relative_path") or "").strip()
            if path:
                edits_by_path[path] = edit
        return edits_by_path

    @classmethod
    def _artifact_patch_changed_edits_with_payload_content(
        cls,
        *,
        payload: dict[str, Any],
        changed_edits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload_edits_by_path = cls._artifact_patch_payload_file_edits_by_path(payload)
        if not payload_edits_by_path:
            return changed_edits
        enriched_edits: list[dict[str, Any]] = []
        for edit in changed_edits:
            path = str(edit.get("path") or edit.get("relative_path") or "").strip()
            payload_edit = payload_edits_by_path.get(path)
            if payload_edit is None:
                enriched_edits.append(edit)
                continue
            enriched = dict(payload_edit)
            enriched.update(edit)
            enriched_edits.append(enriched)
        return enriched_edits

    @staticmethod
    def _example_artifact_patch_content_reasons(
        *,
        path: str,
        content_text: str,
        evidence_text: str = "",
        pass_guards: list[dict[str, Any]] | None = None,
        full_file_content: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        if _SQL_BIND_PLACEHOLDER_RE.search(content_text):
            reasons.append("executable_artifact_patch_example_contains_placeholder")
        if _looks_like_literal_swap_guard_clone(evidence_text):
            reasons.append("executable_artifact_patch_example_literal_swap_guard_clone")
        example_question_sql = _extract_example_question_sql(content_text)
        if example_question_sql is not None and pass_guards:
            question, sql = example_question_sql
            reasons.extend(
                _literal_swap_guard_clone_reasons(
                    question=question,
                    sql=sql,
                    pass_guards=pass_guards,
                )
            )
        if path.endswith(".yml") and full_file_content:
            parsed, parse_reasons = _parse_artifact_patch_yaml_mapping(
                content_text,
                invalid_reason="executable_artifact_patch_example_invalid_yaml",
                non_mapping_reason="executable_artifact_patch_example_yaml_must_be_mapping",
            )
            reasons.extend(parse_reasons)
            if parsed is not None:
                reasons.extend(
                    _artifact_patch_unexpected_content_wrapper_reasons(
                        parsed,
                        reason="executable_artifact_patch_example_unexpected_content_wrapper",
                    )
                )
                reasons.extend(
                    _artifact_patch_required_stringish_field_reasons(
                        parsed,
                        field_name="question",
                        missing_reason=(
                            "executable_artifact_patch_example_missing_question_or_sql"
                        ),
                        type_reason=(
                            "executable_artifact_patch_example_question_must_be_string_or_list"
                        ),
                    )
                )
                reasons.extend(
                    _artifact_patch_required_stringish_field_reasons(
                        parsed,
                        field_name="sql",
                        missing_reason=(
                            "executable_artifact_patch_example_missing_question_or_sql"
                        ),
                        type_reason=(
                            "executable_artifact_patch_example_sql_must_be_string_or_list"
                        ),
                    )
                )
            return reasons
        if path.endswith(".yml"):
            lowered = content_text.lower()
            if "question:" not in lowered or "sql:" not in lowered:
                reasons.append("executable_artifact_patch_example_missing_question_or_sql")
        return reasons

    @staticmethod
    def _sql_snippet_artifact_patch_content_reasons(
        *,
        path: str,
        content_text: str,
        full_file_content: bool = False,
    ) -> list[str]:
        del path
        reasons: list[str] = []
        if _SQL_BIND_PLACEHOLDER_RE.search(content_text):
            reasons.append("executable_artifact_patch_sql_snippet_contains_placeholder")
        if re.search(r"(?im)^\s*(with|select)\b", content_text):
            reasons.append("executable_artifact_patch_sql_snippet_full_query")
        if _looks_like_sql_snippet_option_menu(content_text):
            reasons.append("executable_artifact_patch_sql_snippet_option_menu")
        lowered = content_text.lower()
        has_display_name = "display_name:" in lowered
        has_sql = "sql:" in lowered
        has_repair_source = any(
            f"{key}:" in lowered for key in ("description", "question", "id", "alias")
        )
        if full_file_content:
            parsed, parse_reasons = _parse_artifact_patch_yaml_mapping(
                content_text,
                invalid_reason="executable_artifact_patch_sql_snippet_invalid_yaml",
                non_mapping_reason=(
                    "executable_artifact_patch_sql_snippet_yaml_must_be_mapping"
                ),
            )
            reasons.extend(parse_reasons)
            if parsed is None:
                return reasons
            reasons.extend(
                _artifact_patch_unexpected_content_wrapper_reasons(
                    parsed,
                    reason="executable_artifact_patch_sql_snippet_unexpected_content_wrapper",
                )
            )
            reasons.extend(
                _artifact_patch_required_stringish_field_reasons(
                    parsed,
                    field_name="sql",
                    missing_reason="executable_artifact_patch_sql_snippet_missing_sql",
                    type_reason=(
                        "executable_artifact_patch_sql_snippet_sql_must_be_string_or_list"
                    ),
                )
            )
            display_name = parsed.get("display_name")
            if display_name is not None and not isinstance(display_name, str):
                reasons.append(
                    "executable_artifact_patch_sql_snippet_display_name_must_be_string"
                )
            repair_source = any(
                _artifact_patch_stringish_field_is_nonempty(parsed.get(key))
                for key in ("description", "question", "id", "alias")
            )
            if (
                not _artifact_patch_stringish_field_is_nonempty(display_name)
                and not repair_source
            ):
                reasons.append(
                    "executable_artifact_patch_sql_snippet_display_name_not_repairable"
                )
            return reasons
        if not has_sql:
            reasons.append("executable_artifact_patch_sql_snippet_missing_sql")
        if not has_display_name and not has_repair_source:
            reasons.append("executable_artifact_patch_sql_snippet_display_name_not_repairable")
        return reasons

    @staticmethod
    def _artifact_patch_summary(
        *,
        payload: dict[str, Any],
        candidate_name: str,
        patch_summary: dict[str, Any],
        accepted: bool,
        reason: str | None,
        risk_reasons: list[str],
        candidate_mode: str = "artifact_patch",
        visible_effect_verification: dict[str, Any] | None = None,
        prior_train_collapse_shapes: list[dict[str, str]] | None = None,
        prior_refused_shapes: list[dict[str, str]] | None = None,
        prior_train_collapse_new_executable_assets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        summary = {
            "changed": bool(accepted and patch_summary.get("changed")),
            "candidate_name": candidate_name,
            "rationale": str(payload.get("rationale") or ""),
            "expected_impact": str(payload.get("expected_impact") or ""),
            "risk": str(payload.get("risk") or ""),
            "owner_followups": [
                str(item)
                for item in payload.get("owner_followups", [])
                if str(item).strip()
            ] if isinstance(payload.get("owner_followups"), list) else [],
            "operation_count": 0,
            "file_edit_count": int(patch_summary.get("edit_count", 0) or 0),
            "applied_operations": patch_summary.get("changed_edits", []),
            "skipped_operations": patch_summary.get("skipped_edits", []),
            "artifact_patch": patch_summary,
            "risk_assessment": {
                "accepted": accepted,
                "reason": None if accepted else "artifact_patch_risk_rejected",
                "reasons": risk_reasons,
                "policy": {
                    "allowlisted_decomposed_paths_only": True,
                    "transactional_file_validation": True,
                    "assemble_config_validation": True,
                },
            },
            "candidate_mode": candidate_mode,
        }
        if visible_effect_verification is not None:
            summary["risk_assessment"]["visible_effect_verification"] = (
                visible_effect_verification
            )
        if prior_train_collapse_shapes:
            summary["risk_assessment"]["prior_train_collapse_shapes"] = (
                prior_train_collapse_shapes
            )
        if prior_refused_shapes:
            summary["risk_assessment"]["prior_refused_shapes"] = prior_refused_shapes
        if prior_train_collapse_new_executable_assets:
            summary["risk_assessment"][
                "prior_train_collapse_new_executable_assets"
            ] = prior_train_collapse_new_executable_assets
        if reason is not None:
            summary["reason"] = reason
        return summary

    def _write_serving_candidate_artifacts(
        self,
        *,
        spec_name: str,
        proposal: ServingCandidateProposal,
        proposal_payload: dict[str, Any],
        failure_context: str,
        structured_failure_context: dict[str, Any],
        trace_context_pack: dict[str, Any],
        historical_analyst_artifact: dict[str, Any] | None = None,
        data_probe_payload: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        proposal_dir = self.artifacts.history_dir / "serving_candidates" / f"{stamp}.{spec_name}"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "proposal_dir": proposal_dir,
            "request_path": proposal_dir / "request.json",
            "response_path": proposal_dir / "response.json",
            "payload_path": proposal_dir / "candidate_payload.json",
            "raw_payload_path": proposal_dir / "raw_candidate_payload.json",
            "failure_context_path": proposal_dir / "failure_context.md",
            "structured_failure_context_path": proposal_dir / "structured_failure_context.json",
            "trace_context_pack_path": proposal_dir / "failure_context_pack.json",
        }
        if historical_analyst_artifact is not None:
            paths["historical_analyst_path"] = proposal_dir / "historical_analyst.json"
            paths["historical_analyst_memo_path"] = proposal_dir / "historical_analyst_memo.md"
        if data_probe_payload is not None:
            paths["data_probe_path"] = proposal_dir / "data_probe.json"
        self.artifacts.write_json_path(paths["request_path"], proposal.request)
        self.artifacts.write_json_path(paths["response_path"], proposal.response)
        self.artifacts.write_json_path(paths["raw_payload_path"], proposal.payload)
        self.artifacts.write_json_path(paths["payload_path"], proposal_payload)
        self.artifacts.write_text_path(paths["failure_context_path"], failure_context)
        self.artifacts.write_json_path(paths["structured_failure_context_path"], structured_failure_context)
        self.artifacts.write_json_path(paths["trace_context_pack_path"], trace_context_pack)
        if historical_analyst_artifact is not None:
            self.artifacts.write_json_path(
                paths["historical_analyst_path"],
                historical_analyst_artifact,
            )
            self.artifacts.write_text_path(
                paths["historical_analyst_memo_path"],
                str(historical_analyst_artifact.get("content") or ""),
            )
        if data_probe_payload is not None:
            self.artifacts.write_json_path(paths["data_probe_path"], data_probe_payload)
        return {key: str(value) for key, value in paths.items()}

    def _run_serving_candidate(
        self,
        *,
        current_config: GenieSpaceConfig,
        spec: CandidateSpec,
        serving_config: ServingCandidateConfig,
    ) -> tuple[GenieSpaceConfig, dict[str, Any]]:
        from maxgenie.failure_context import build_failure_context, build_structured_failure_context
        from maxgenie.trace_memory import (
            build_cross_run_memory,
            build_failure_context_pack,
            build_trace_memory_index,
        )

        optimization_summary = _load_json_dict(self.artifacts.optimization_summary_path)
        failure_context = build_failure_context(
            results_dir=self.artifacts.results_dir,
            history_dir=self.artifacts.history_dir,
            benchmarks_dir=self.artifacts.benchmarks_dir,
            iteration_label=spec.name,
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
        structured_failure_context = build_structured_failure_context(
            results_dir=self.artifacts.results_dir,
            history_dir=self.artifacts.history_dir,
            benchmarks_dir=self.artifacts.benchmarks_dir,
            iteration_label=spec.name,
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
        self._attach_visible_expected_rows(structured_failure_context)
        self._attach_parameter_domain_values(structured_failure_context)
        trace_memory = build_trace_memory_index(self.artifacts)
        cross_run_memory = build_cross_run_memory(self.artifacts)
        trace_context_pack = build_failure_context_pack(
            trace_memory,
            cross_run_memory=cross_run_memory,
        ).to_dict()
        self.artifacts.write_json_path(self.artifacts.trace_memory_path, trace_memory.to_dict())
        self.artifacts.write_json_path(self.artifacts.cross_run_memory_path, cross_run_memory.to_dict())
        self.artifacts.write_json_path(self.artifacts.failure_context_pack_path, trace_context_pack)
        visible_payload = self._visible_benchmark_payload()
        diagnostics_payload = _load_json_dict(self.artifacts.space_diagnostics_path)
        match = re.search(r"(\d+)$", spec.name)
        serving_iteration_index = int(match.group(1)) if match else 1
        forced_candidate_mode = (
            spec.candidate_mode.strip()
            if spec.candidate_mode is not None and spec.candidate_mode.strip()
            else None
        )
        if (
            forced_candidate_mode is not None
            and serving_candidate_allowed_operation_groups(forced_candidate_mode) is None
        ):
            raise WorkspaceFlowError(f"Unsupported serving candidate mode: {forced_candidate_mode}")
        prior_guard_regressions = count_prior_pass_guard_regressions(failure_context)
        prior_accepted_fix_regressions = count_prior_accepted_fix_regressions(failure_context)
        prior_rejected_candidates = count_prior_rejected_candidates(failure_context)
        prior_rejected_mode_counts = count_prior_rejected_candidate_modes(failure_context)
        prior_invalid_responses = count_prior_invalid_responses(failure_context)
        prior_skipped_candidates = count_prior_skipped_candidates(failure_context)
        adaptive_recipe_modes = bool(getattr(self, "_adaptive_recipe_modes", False))
        adaptive_dominant_family = (
            dominant_visible_failure_family(structured_failure_context.get("failures"))
            if adaptive_recipe_modes
            else None
        )
        scheduled_candidate_mode = forced_candidate_mode or serving_candidate_mode_for_iteration(
            serving_iteration_index,
            prior_pass_guard_regressions=prior_guard_regressions,
            prior_accepted_fix_regressions=prior_accepted_fix_regressions,
            prior_rejected_candidates=prior_rejected_candidates,
            prior_invalid_responses=prior_invalid_responses,
            prior_skipped_candidates=prior_skipped_candidates,
            prior_rejected_mode_counts=prior_rejected_mode_counts,
            adaptive_modes=adaptive_recipe_modes,
            dominant_failure_family=adaptive_dominant_family,
        )
        if forced_candidate_mode is None:
            replay_payload, replay_lesson = self._load_cross_run_replay_payload(
                cross_run_memory=cross_run_memory,
                serving_iteration_index=serving_iteration_index,
                optimization_summary=optimization_summary,
                expected_candidate_mode=scheduled_candidate_mode,
            )
        else:
            replay_payload = None
            replay_lesson = None
        replay_candidate_mode = (
            replay_lesson.candidate_mode
            if replay_lesson is not None and replay_lesson.candidate_mode
            else None
        )
        candidate_mode = (
            forced_candidate_mode
            or replay_candidate_mode
            or scheduled_candidate_mode
        )
        allowed_operation_groups = serving_candidate_allowed_operation_groups(candidate_mode)
        max_mutation_operations = serving_candidate_max_mutation_operations(candidate_mode)
        required_pass_guard_ids = (
            baseline_pass_guard_ids(failure_context)
            if serving_candidate_is_example_set_mode(candidate_mode)
            else ()
        )
        decomposed_space_payload = (
            self._decomposed_space_payload()
            if serving_candidate_is_artifact_patch_mode(candidate_mode)
            else None
        )
        visible_proxy_guard_payload = self._visible_proxy_guard_payload()
        self.artifacts.record_iteration(
            {
                "phase": "serving_candidate_proposal_start",
                "label": spec.name,
                "candidate_mode": candidate_mode,
                "forced_candidate_mode": forced_candidate_mode,
                "adaptive_recipe_modes": adaptive_recipe_modes,
                "adaptive_dominant_failure_family": adaptive_dominant_family,
                "prior_pass_guard_regressions": prior_guard_regressions,
                "prior_accepted_fix_regressions": prior_accepted_fix_regressions,
                "prior_rejected_candidates": prior_rejected_candidates,
                "prior_rejected_mode_counts": prior_rejected_mode_counts,
                "prior_invalid_responses": prior_invalid_responses,
                "prior_skipped_candidates": prior_skipped_candidates,
                "allowed_operation_groups": (
                    sorted(allowed_operation_groups)
                    if allowed_operation_groups is not None
                    else None
                ),
                "max_mutation_operations": max_mutation_operations,
                "required_pass_guard_ids": list(required_pass_guard_ids),
                "visible_proxy_guard_ids": visible_proxy_guard_payload.get(
                    "required_guard_ids",
                    [],
                ),
                "visible_proxy_families": visible_proxy_guard_payload.get(
                    "required_families",
                    [],
                ),
                "visible_proxy_guard_count": len(
                    visible_proxy_guard_payload.get("required_guard_ids", []) or []
                ),
                "cross_run_memory_runs_considered": cross_run_memory.runs_considered,
                "cross_run_memory_best_prior": (
                    cross_run_memory.best_prior_lesson.candidate_name
                    if cross_run_memory.best_prior_lesson is not None
                    else None
                ),
                "cross_run_replay_source_run": (
                    replay_lesson.source_run
                    if replay_lesson is not None
                    else None
                ),
                "cross_run_replay_candidate_name": (
                    replay_lesson.candidate_name
                    if replay_lesson is not None
                    else None
                ),
            }
        )
        historical_analyst_artifact: dict[str, Any] | None = None
        historical_analyst_memo: str | None = None
        data_probe_payload: dict[str, Any] | None = None
        if replay_payload is not None and replay_lesson is not None:
            proposal = ServingCandidateProposal(
                request={
                    "source": "cross_run_memory_replay",
                    "source_run": replay_lesson.source_run,
                    "source_payload_path": replay_lesson.payload_path,
                    "candidate_mode": candidate_mode,
                },
                response={
                    "source": "cross_run_memory_replay",
                    "candidate_name": replay_lesson.candidate_name,
                },
                content=json.dumps(replay_payload, ensure_ascii=False),
                payload=replay_payload,
            )
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_replay_loaded",
                    "label": spec.name,
                    "candidate_mode": candidate_mode,
                    "source_run": replay_lesson.source_run,
                    "candidate_name": replay_lesson.candidate_name,
                    "payload_path": replay_lesson.payload_path,
                }
            )
        else:
            client = ServingCandidateClient(
                wc=self.genie_service.wc,
                config=serving_config,
            )
            if serving_candidate_uses_historical_analysis(candidate_mode):
                analysis = client.analyze_history(
                    current_config=self._config_with_visible_benchmarks(current_config),
                    failure_context=failure_context,
                    structured_failure_context=structured_failure_context,
                    trace_context_pack=trace_context_pack,
                    visible_proxy_guard_payload=visible_proxy_guard_payload,
                    visible_benchmark_payload=visible_payload,
                    decomposed_space_payload=decomposed_space_payload,
                    optimization_summary=optimization_summary,
                    candidate_mode=candidate_mode,
                )
                historical_analyst_memo = analysis.content
                historical_analyst_artifact = {
                    "request": analysis.request,
                    "response": analysis.response,
                    "content": analysis.content,
                }
                self.artifacts.record_iteration(
                    {
                        "phase": "serving_candidate_historical_analysis_received",
                        "label": spec.name,
                        "candidate_mode": candidate_mode,
                        "content_length": len(analysis.content),
                    }
                )
            data_probe_payload = (
                self._data_probe_payload(current_config)
                if serving_candidate_uses_data_probe(candidate_mode)
                else None
            )
            if data_probe_payload is not None:
                self.artifacts.record_iteration(
                    {
                        "phase": "serving_candidate_data_probe_completed",
                        "label": spec.name,
                        "candidate_mode": candidate_mode,
                        "probe_count": len(data_probe_payload.get("probes", []) or []),
                        "blockers": data_probe_payload.get("blockers", []),
                    }
                )
            try:
                proposal = client.propose(
                    current_config=self._config_with_visible_benchmarks(current_config),
                    failure_context=failure_context,
                    structured_failure_context=structured_failure_context,
                    trace_context_pack=trace_context_pack,
                    visible_proxy_guard_payload=visible_proxy_guard_payload,
                    historical_analyst_memo=historical_analyst_memo,
                    data_probe_payload=data_probe_payload,
                    visible_benchmark_payload=visible_payload,
                    decomposed_space_payload=decomposed_space_payload,
                    diagnostics_payload=diagnostics_payload,
                    optimization_summary=optimization_summary,
                    candidate_mode=candidate_mode,
                )
            except ServingCandidateResponseError as exc:
                stamp = time.strftime("%Y%m%dT%H%M%S")
                proposal_dir = self.artifacts.history_dir / "serving_candidates" / f"{stamp}.{spec.name}"
                proposal_dir.mkdir(parents=True, exist_ok=True)
                request_path = proposal_dir / "request.json"
                response_path = proposal_dir / "response.json"
                content_path = proposal_dir / "raw_content.txt"
                context_path = proposal_dir / "failure_context.md"
                structured_context_path = proposal_dir / "structured_failure_context.json"
                trace_context_pack_path = proposal_dir / "failure_context_pack.json"
                self.artifacts.write_json_path(request_path, exc.request)
                self.artifacts.write_json_path(response_path, exc.response)
                self.artifacts.write_text_path(content_path, exc.content)
                self.artifacts.write_text_path(context_path, failure_context)
                self.artifacts.write_json_path(structured_context_path, structured_failure_context)
                self.artifacts.write_json_path(trace_context_pack_path, trace_context_pack)
                self.artifacts.record_iteration(
                    {
                        "phase": "serving_candidate_proposal_invalid",
                        "label": spec.name,
                        "candidate_mode": candidate_mode,
                        "reason": "invalid_serving_response_json",
                        "error": str(exc),
                        "content_length": len(exc.content),
                        "proposal_dir": str(proposal_dir),
                        "request_path": str(request_path),
                        "response_path": str(response_path),
                        "raw_content_path": str(content_path),
                        "failure_context_path": str(context_path),
                        "structured_failure_context_path": str(structured_context_path),
                        "trace_context_pack_path": str(trace_context_pack_path),
                    }
                )
                return (
                    deepcopy(current_config),
                    {
                        "changed": False,
                        "candidate_name": spec.name,
                        "rationale": "Serving endpoint returned malformed candidate JSON.",
                        "expected_impact": "No candidate applied.",
                        "risk": "None; malformed response was not applied.",
                        "owner_followups": [
                            "Review raw serving response and reduce candidate scope if malformed JSON repeats."
                        ],
                        "operation_count": 0,
                        "applied_operations": [],
                        "skipped_operations": [],
                        "reason": "invalid_serving_response_json",
                        "candidate_mode": candidate_mode,
                        "strategy": "serving_autonomous",
                        "endpoint": serving_config.endpoint,
                        "model": serving_config.model,
                        "reasoning_effort": serving_config.reasoning_effort,
                        "api_format": serving_config.api_format,
                        "proposal_dir": str(proposal_dir),
                        "request_path": str(request_path),
                        "response_path": str(response_path),
                        "raw_content_path": str(content_path),
                        "failure_context_path": str(context_path),
                        "structured_failure_context_path": str(structured_context_path),
                        "trace_context_pack_path": str(trace_context_pack_path),
                    },
                )
        operations = proposal.payload.get("operations")
        operation_count = len(operations) if isinstance(operations, list) else 0
        parse_repairs = tuple(getattr(proposal, "parse_repairs", ()) or ())
        self.artifacts.record_iteration(
            {
                "phase": "serving_candidate_proposal_received",
                "label": spec.name,
                "candidate_mode": candidate_mode,
                "candidate_name": str(proposal.payload.get("candidate_name") or ""),
                "operation_count": operation_count,
                "parse_repairs": list(parse_repairs),
            }
        )
        if parse_repairs:
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_payload_parse_repaired",
                    "label": spec.name,
                    "candidate_mode": candidate_mode,
                    "candidate_name": str(proposal.payload.get("candidate_name") or ""),
                    "repairs": list(parse_repairs),
                }
            )
        if serving_candidate_is_artifact_patch_mode(candidate_mode):
            proposal_payload = deepcopy(proposal.payload)
            proposal_payload["candidate_mode"] = candidate_mode
            artifact_patch_repair_summary: dict[str, Any] | None = None
            if serving_candidate_is_strict_artifact_patch_mode(candidate_mode):
                proposal_payload, artifact_patch_repair_summary = repair_artifact_patch_v2_payload(
                    self.artifacts.space_dir,
                    proposal_payload,
                )
                if artifact_patch_repair_summary.get("changed"):
                    self.artifacts.record_iteration(
                        {
                            "phase": "serving_candidate_artifact_patch_repaired",
                            "label": spec.name,
                            "candidate_mode": candidate_mode,
                            "candidate_name": str(proposal_payload.get("candidate_name") or ""),
                            "repair_summary": artifact_patch_repair_summary,
                        }
                    )
            candidate_config, summary = self._apply_artifact_patch_payload(
                payload=proposal_payload,
                current_config=current_config,
                structured_failure_context=structured_failure_context,
                failure_context=failure_context,
            )
            artifact_paths = self._write_serving_candidate_artifacts(
                spec_name=spec.name,
                proposal=proposal,
                proposal_payload=proposal_payload,
                failure_context=failure_context,
                structured_failure_context=structured_failure_context,
                trace_context_pack=trace_context_pack,
                historical_analyst_artifact=historical_analyst_artifact,
                data_probe_payload=data_probe_payload,
            )
            result_summary = {
                **summary,
                "strategy": AUTONOMOUS_STRATEGY_SERVING,
                "endpoint": serving_config.endpoint,
                "model": serving_config.model,
                "reasoning_effort": serving_config.reasoning_effort,
                "api_format": serving_config.api_format,
                "candidate_mode": candidate_mode,
                **artifact_paths,
            }
            if parse_repairs:
                result_summary["parse_repairs"] = list(parse_repairs)
            if artifact_patch_repair_summary is not None:
                result_summary["artifact_patch_repair"] = artifact_patch_repair_summary
            if replay_lesson is not None:
                result_summary["cross_run_replay"] = {
                    "source_run": replay_lesson.source_run,
                    "candidate_name": replay_lesson.candidate_name,
                    "payload_path": replay_lesson.payload_path,
                    "checkpoint_path": replay_lesson.checkpoint_path,
                    "final_full_rate": replay_lesson.final_full_rate,
                    "final_holdout_rate": replay_lesson.final_holdout_rate,
                }
            return candidate_config, result_summary
        require_guardrail_acknowledgement = prior_guard_regressions > 0
        benchmark_by_question_id = {
            str(row["id"]): {
                "question": str(row.get("question") or ""),
                "expected_sql": str(row.get("expected_sql") or ""),
            }
            for row in visible_payload.get("questions", [])
            if isinstance(row, dict) and row.get("id") and row.get("expected_sql")
        }
        proposal_payload, canonicalization_summary = canonicalize_tagged_example_sql(
            proposal.payload,
            benchmark_by_question_id=benchmark_by_question_id,
        )
        if candidate_mode == "typed_template":
            proposal_payload, typed_template_summary = materialize_typed_template_examples(
                proposal_payload,
                benchmark_by_question_id=benchmark_by_question_id,
                structured_failure_context=structured_failure_context,
                required_pass_guard_ids=required_pass_guard_ids,
                max_mutation_operations=max_mutation_operations,
            )
            canonicalization_summary.update(typed_template_summary)
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_typed_template_materialized",
                    "label": spec.name,
                    "candidate_mode": candidate_mode,
                    "candidate_name": str(proposal_payload.get("candidate_name") or ""),
                    **typed_template_summary,
                }
            )
        if replay_lesson is not None and serving_candidate_is_example_set_mode(candidate_mode):
            proposal_payload, repair_summary = repair_example_set_pass_guards(
                proposal_payload,
                benchmark_by_question_id=benchmark_by_question_id,
                required_pass_guard_ids=required_pass_guard_ids,
                max_mutation_operations=max_mutation_operations,
            )
            canonicalization_summary.update(repair_summary)
            if (
                repair_summary["repaired_pass_guard_ids"]
                or repair_summary["unrepaired_pass_guard_ids"]
                or repair_summary["pruned_duplicate_operation_count"]
            ):
                self.artifacts.record_iteration(
                    {
                        "phase": "serving_candidate_replay_guard_repair",
                        "label": spec.name,
                        "candidate_mode": candidate_mode,
                        "candidate_name": str(proposal_payload.get("candidate_name") or ""),
                        "repaired_pass_guard_ids": repair_summary["repaired_pass_guard_ids"],
                        "unrepaired_pass_guard_ids": repair_summary["unrepaired_pass_guard_ids"],
                        "pruned_duplicate_operation_count": repair_summary[
                            "pruned_duplicate_operation_count"
                        ],
                    }
                )
        expected_sql_by_question_id = {
            question_id: row["expected_sql"]
            for question_id, row in benchmark_by_question_id.items()
            if row.get("expected_sql")
        }
        structured_failures = (
            structured_failure_context.get("failures", [])
            if isinstance(structured_failure_context, dict)
            else []
        )
        failure_family_by_question_id = {
            str(failure["question_id"]): str(failure["likely_root_cause_family"])
            for failure in structured_failures
            if (
                isinstance(failure, dict)
                and failure.get("question_id")
                and failure.get("likely_root_cause_family")
            )
        }
        guard_regression_risk_by_question_id = {
            str(failure["question_id"]): str(failure["guard_regression_risk"])
            for failure in structured_failures
            if (
                isinstance(failure, dict)
                and failure.get("question_id")
                and failure.get("guard_regression_risk")
            )
        }
        risk_assessment = assess_serving_candidate_risk(
            proposal_payload,
            candidate_mode=candidate_mode,
            allowed_operation_groups=allowed_operation_groups,
            require_guardrail_acknowledgement=require_guardrail_acknowledgement,
            max_mutation_operations=max_mutation_operations,
            required_pass_guard_ids=required_pass_guard_ids,
            expected_sql_by_question_id=expected_sql_by_question_id,
            pass_guard_questions_by_id=baseline_pass_guard_questions(failure_context),
            failure_family_by_question_id=failure_family_by_question_id,
            guard_regression_risk_by_question_id=guard_regression_risk_by_question_id,
        )
        if any(value for value in canonicalization_summary.values()):
            risk_assessment["canonicalization"] = canonicalization_summary
        if bool(risk_assessment.get("accepted")):
            candidate_config, summary = apply_serving_candidate(
                current_config,
                proposal_payload,
            )
            summary["risk_assessment"] = risk_assessment
            summary["candidate_mode"] = candidate_mode
        else:
            candidate_config = deepcopy(current_config)
            operations = proposal_payload.get("operations")
            if not isinstance(operations, list):
                operations = []
            summary = {
                "changed": False,
                "candidate_name": str(proposal_payload.get("candidate_name") or "serving_candidate"),
                "rationale": str(proposal_payload.get("rationale") or ""),
                "expected_impact": str(proposal_payload.get("expected_impact") or ""),
                "risk": str(proposal_payload.get("risk") or ""),
                "owner_followups": [
                    str(item)
                    for item in proposal_payload.get("owner_followups", [])
                    if str(item).strip()
                ] if isinstance(proposal_payload.get("owner_followups"), list) else [],
                "operation_count": len(operations),
                "applied_operations": [],
                "skipped_operations": [
                    {
                        "index": index,
                        "op": (
                            str(operation.get("op") or "")
                            if isinstance(operation, dict)
                            else "invalid"
                        ),
                        "reason": "serving_candidate_risk_rejected",
                    }
                    for index, operation in enumerate(operations)
                ],
                "reason": "serving_candidate_risk_rejected",
                "risk_assessment": risk_assessment,
                "candidate_mode": candidate_mode,
            }

        stamp = time.strftime("%Y%m%dT%H%M%S")
        proposal_dir = self.artifacts.history_dir / "serving_candidates" / f"{stamp}.{spec.name}"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        request_path = proposal_dir / "request.json"
        response_path = proposal_dir / "response.json"
        payload_path = proposal_dir / "candidate_payload.json"
        raw_payload_path = proposal_dir / "raw_candidate_payload.json"
        context_path = proposal_dir / "failure_context.md"
        structured_context_path = proposal_dir / "structured_failure_context.json"
        trace_context_pack_path = proposal_dir / "failure_context_pack.json"
        self.artifacts.write_json_path(request_path, proposal.request)
        self.artifacts.write_json_path(response_path, proposal.response)
        self.artifacts.write_json_path(raw_payload_path, proposal.payload)
        self.artifacts.write_json_path(payload_path, proposal_payload)
        self.artifacts.write_text_path(context_path, failure_context)
        self.artifacts.write_json_path(structured_context_path, structured_failure_context)
        self.artifacts.write_json_path(trace_context_pack_path, trace_context_pack)

        result_summary = {
            **summary,
            "strategy": AUTONOMOUS_STRATEGY_SERVING,
            "endpoint": serving_config.endpoint,
            "model": serving_config.model,
            "reasoning_effort": serving_config.reasoning_effort,
            "api_format": serving_config.api_format,
            "candidate_mode": candidate_mode,
            "proposal_dir": str(proposal_dir),
            "request_path": str(request_path),
            "response_path": str(response_path),
            "payload_path": str(payload_path),
            "raw_payload_path": str(raw_payload_path),
            "failure_context_path": str(context_path),
            "structured_failure_context_path": str(structured_context_path),
            "trace_context_pack_path": str(trace_context_pack_path),
        }
        if parse_repairs:
            result_summary["parse_repairs"] = list(parse_repairs)
        if replay_lesson is not None:
            result_summary["cross_run_replay"] = {
                "source_run": replay_lesson.source_run,
                "candidate_name": replay_lesson.candidate_name,
                "payload_path": replay_lesson.payload_path,
                "checkpoint_path": replay_lesson.checkpoint_path,
                "final_full_rate": replay_lesson.final_full_rate,
                "final_holdout_rate": replay_lesson.final_holdout_rate,
            }
        return candidate_config, result_summary

    def _load_cross_run_replay_payload(
        self,
        *,
        cross_run_memory: Any,
        serving_iteration_index: int,
        optimization_summary: dict[str, Any] | None,
        expected_candidate_mode: str | None = None,
    ) -> tuple[dict[str, Any] | None, Any | None]:
        lesson = self._cross_run_replay_lesson_for_iteration(
            cross_run_memory=cross_run_memory,
            serving_iteration_index=serving_iteration_index,
            optimization_summary=optimization_summary,
        )
        if lesson is None:
            return None, None
        if not self._cross_run_replay_candidate_mode_compatible(
            lesson=lesson,
            expected_candidate_mode=expected_candidate_mode,
        ):
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_replay_unavailable",
                    "reason": "candidate_mode_incompatible_with_scheduled_mode",
                    "expected_candidate_mode": expected_candidate_mode,
                    "candidate_mode": getattr(lesson, "candidate_mode", None),
                    "source_run": getattr(lesson, "source_run", None),
                    "candidate_name": getattr(lesson, "candidate_name", None),
                    "payload_path": getattr(lesson, "payload_path", None),
                }
            )
            return None, None
        payload_path_text = getattr(lesson, "payload_path", None)
        if not payload_path_text:
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_replay_unavailable",
                    "reason": "best_prior_payload_missing",
                    "source_run": getattr(lesson, "source_run", None),
                    "candidate_name": getattr(lesson, "candidate_name", None),
                }
            )
            return None, None
        payload_path = Path(str(payload_path_text))
        if not payload_path.exists():
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_replay_unavailable",
                    "reason": "best_prior_payload_path_not_found",
                    "source_run": getattr(lesson, "source_run", None),
                    "candidate_name": getattr(lesson, "candidate_name", None),
                    "payload_path": str(payload_path),
                }
            )
            return None, None
        payload = _load_json_dict(payload_path)
        if payload is None or not isinstance(payload.get("operations"), list):
            self.artifacts.record_iteration(
                {
                    "phase": "serving_candidate_replay_unavailable",
                    "reason": "best_prior_payload_invalid",
                    "source_run": getattr(lesson, "source_run", None),
                    "candidate_name": getattr(lesson, "candidate_name", None),
                    "payload_path": str(payload_path),
                }
            )
            return None, None
        replay_payload = deepcopy(payload)
        if not str(replay_payload.get("candidate_name") or "").strip():
            replay_payload["candidate_name"] = (
                str(getattr(lesson, "candidate_name", "") or "cross_run_replay")
            )
        if not str(replay_payload.get("rationale") or "").strip():
            replay_payload["rationale"] = (
                "Replay the strongest prior accepted candidate payload from sibling run memory."
            )
        if not str(replay_payload.get("expected_impact") or "").strip():
            replay_payload["expected_impact"] = (
                "Recover the previously observed visible benchmark improvement under the current gates."
            )
        if not str(replay_payload.get("risk") or "").strip():
            replay_payload["risk"] = (
                "Guarded by the normal pass-guard and validation gates before acceptance."
            )
        if not isinstance(replay_payload.get("owner_followups"), list):
            replay_payload["owner_followups"] = []
        return replay_payload, lesson

    @staticmethod
    def _cross_run_replay_candidate_mode_compatible(
        *,
        lesson: Any,
        expected_candidate_mode: str | None,
    ) -> bool:
        expected = str(expected_candidate_mode or "").strip()
        if not expected:
            return True
        candidate_mode = str(getattr(lesson, "candidate_mode", "") or "").strip()
        return candidate_mode == expected

    def _cross_run_replay_lesson_for_iteration(
        self,
        *,
        cross_run_memory: Any,
        serving_iteration_index: int,
        optimization_summary: dict[str, Any] | None,
    ) -> Any | None:
        sequence = tuple(getattr(cross_run_memory, "best_prior_sequence", ()) or ())
        if sequence:
            sequence_index = serving_iteration_index - 1
            if sequence_index >= len(sequence):
                tail_lesson = sequence[-1]
                if not self._prior_cross_run_replay_accepted(
                    prior_lesson=tail_lesson,
                    optimization_summary=optimization_summary,
                ):
                    self.artifacts.record_iteration(
                        {
                            "phase": "serving_candidate_replay_unavailable",
                            "reason": "prior_sequence_tail_not_accepted",
                            "sequence_index": sequence_index,
                            "source_run": getattr(tail_lesson, "source_run", None),
                            "candidate_name": getattr(tail_lesson, "candidate_name", None),
                        }
                    )
                    return None
                fallback_lesson = self._cross_run_sequence_fallback_lesson(
                    cross_run_memory=cross_run_memory,
                    optimization_summary=optimization_summary,
                )
                if fallback_lesson is not None:
                    self.artifacts.record_iteration(
                        {
                            "phase": "serving_candidate_replay_sequence_fallback",
                            "sequence_index": sequence_index,
                            "source_run": getattr(fallback_lesson, "source_run", None),
                            "candidate_name": getattr(fallback_lesson, "candidate_name", None),
                            "payload_path": getattr(fallback_lesson, "payload_path", None),
                        }
                    )
                    return fallback_lesson
                self.artifacts.record_iteration(
                    {
                        "phase": "serving_candidate_replay_unavailable",
                        "reason": "best_prior_sequence_exhausted",
                        "sequence_index": sequence_index,
                    }
                )
                return None
            if sequence_index > 0 and not self._prior_cross_run_replay_accepted(
                prior_lesson=sequence[sequence_index - 1],
                optimization_summary=optimization_summary,
            ):
                self.artifacts.record_iteration(
                    {
                        "phase": "serving_candidate_replay_unavailable",
                        "reason": "prior_sequence_step_not_accepted",
                        "sequence_index": sequence_index,
                        "source_run": getattr(sequence[sequence_index], "source_run", None),
                        "candidate_name": getattr(sequence[sequence_index], "candidate_name", None),
                    }
                )
                return None
            return sequence[sequence_index]
        if serving_iteration_index == 1:
            return getattr(cross_run_memory, "best_prior_lesson", None)
        return None

    def _cross_run_sequence_fallback_lesson(
        self,
        *,
        cross_run_memory: Any,
        optimization_summary: dict[str, Any] | None,
    ) -> Any | None:
        accepted_names, accepted_payloads = self._accepted_cross_run_replay_markers(
            optimization_summary=optimization_summary,
        )
        accepted_lessons = tuple(getattr(cross_run_memory, "accepted_lessons", ()) or ())
        lessons_by_source: dict[str, list[Any]] = {}
        for lesson in accepted_lessons:
            source_run = str(getattr(lesson, "source_run", "") or "")
            if not source_run:
                continue
            lessons_by_source.setdefault(source_run, []).append(lesson)

        candidates: list[tuple[int, int, float, float, float, Any]] = []
        for lessons in lessons_by_source.values():
            ordered_lessons = sorted(
                lessons,
                key=lambda lesson: int(getattr(lesson, "attempt", 0) or 0),
            )
            matched_prefix_count = 0
            for lesson in ordered_lessons:
                if self._cross_run_lesson_was_accepted(
                    lesson=lesson,
                    accepted_names=accepted_names,
                    accepted_payloads=accepted_payloads,
                ):
                    matched_prefix_count += 1
                    continue
                if matched_prefix_count <= 0:
                    break
                payload_path = str(getattr(lesson, "payload_path", "") or "")
                if not payload_path:
                    break
                candidates.append(
                    (
                        matched_prefix_count,
                        len(ordered_lessons),
                        self._rank_optional_rate(getattr(lesson, "validation_rate", None)),
                        self._rank_optional_rate(getattr(lesson, "train_rate", None)),
                        self._rank_optional_rate(getattr(lesson, "full_delta", None)),
                        lesson,
                    )
                )
                break

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:5], reverse=True)
        return candidates[0][5]

    @staticmethod
    def _rank_optional_rate(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1.0

    def _accepted_cross_run_replay_markers(
        self,
        *,
        optimization_summary: dict[str, Any] | None,
    ) -> tuple[set[str], set[str]]:
        accepted_names: set[str] = set()
        accepted_payloads: set[str] = set()
        passes = (
            optimization_summary.get("passes")
            if isinstance(optimization_summary, dict)
            and isinstance(optimization_summary.get("passes"), list)
            else []
        )
        for entry in passes:
            if not isinstance(entry, dict) or entry.get("outcome") != "accepted":
                continue
            candidate_summary = (
                entry.get("candidate_summary")
                if isinstance(entry.get("candidate_summary"), dict)
                else {}
            )
            candidate_name = str(candidate_summary.get("candidate_name") or "")
            if candidate_name:
                accepted_names.add(candidate_name)
            replay = (
                candidate_summary.get("cross_run_replay")
                if isinstance(candidate_summary.get("cross_run_replay"), dict)
                else {}
            )
            replay_name = str(replay.get("candidate_name") or "")
            replay_payload = str(replay.get("payload_path") or "")
            if replay_name:
                accepted_names.add(replay_name)
            if replay_payload:
                accepted_payloads.add(replay_payload)
        return accepted_names, accepted_payloads

    @staticmethod
    def _cross_run_lesson_was_accepted(
        *,
        lesson: Any,
        accepted_names: set[str],
        accepted_payloads: set[str],
    ) -> bool:
        payload_path = str(getattr(lesson, "payload_path", "") or "")
        candidate_name = str(getattr(lesson, "candidate_name", "") or "")
        return bool(
            (payload_path and payload_path in accepted_payloads)
            or (candidate_name and candidate_name in accepted_names)
        )

    def _prior_cross_run_replay_accepted(
        self,
        *,
        prior_lesson: Any,
        optimization_summary: dict[str, Any] | None,
    ) -> bool:
        passes = (
            optimization_summary.get("passes")
            if isinstance(optimization_summary, dict)
            and isinstance(optimization_summary.get("passes"), list)
            else []
        )
        expected_payload_path = str(getattr(prior_lesson, "payload_path", "") or "")
        expected_candidate_name = str(getattr(prior_lesson, "candidate_name", "") or "")
        for entry in passes:
            if not isinstance(entry, dict) or entry.get("outcome") != "accepted":
                continue
            candidate_summary = (
                entry.get("candidate_summary")
                if isinstance(entry.get("candidate_summary"), dict)
                else {}
            )
            replay = (
                candidate_summary.get("cross_run_replay")
                if isinstance(candidate_summary.get("cross_run_replay"), dict)
                else {}
            )
            payload_matches = (
                bool(expected_payload_path)
                and str(replay.get("payload_path") or "") == expected_payload_path
            )
            name_matches = (
                bool(expected_candidate_name)
                and str(replay.get("candidate_name") or "") == expected_candidate_name
            )
            if payload_matches or name_matches:
                return True
        return False

    def _curation_candidate_specs(self, modes: list[str]) -> list[CandidateSpec]:
        return [
            CandidateSpec(
                name=mode,
                family="curation",
                description=f"Deterministic curation pass: {mode}",
                suppression_keys=(mode,),
            )
            for mode in modes
        ]

    def _default_candidate_specs(self) -> list[CandidateSpec]:
        specs = [
            CandidateSpec(
                name=CURATION_MODE_SAFE_DEFAULTS,
                family="curation",
                description="Safe metadata cleanup across descriptions, visibility, and toggles.",
                suppression_keys=(CURATION_MODE_SAFE_DEFAULTS,),
            )
        ]
        specs.extend(
            CandidateSpec(
                name=policy.name,
                family="policy",
                description=policy.description,
                suppression_keys=(policy.name,),
            )
            for policy in self.policy_candidates.values()
        )
        return specs

    def _centaur_prepass_candidate_specs(self) -> list[CandidateSpec]:
        current_config = assemble_config(self.artifacts.space_dir)
        specs: list[CandidateSpec] = []

        if inconsistent_filter_snippet_ids(current_config):
            policy = self.centaur_prepass_candidates["centaur_filter_snippet_audit"]
            specs.append(
                CandidateSpec(
                    name=policy.name,
                    family="centaur_prepass",
                    description=policy.description,
                    suppression_keys=(policy.name,),
                )
            )

        if instruction_contradiction_findings(current_config):
            policy = self.centaur_prepass_candidates["centaur_instruction_audit"]
            specs.append(
                CandidateSpec(
                    name=policy.name,
                    family="centaur_prepass",
                    description=policy.description,
                    suppression_keys=(policy.name,),
                )
            )

        if expanded_entity_matching_candidates(current_config):
            policy = self.centaur_prepass_candidates["centaur_expanded_entity_matching"]
            specs.append(
                CandidateSpec(
                    name=policy.name,
                    family="centaur_prepass",
                    description=policy.description,
                    suppression_keys=(policy.name,),
                )
            )

        return specs

    def _apply_candidate_spec(
        self,
        *,
        current_config: GenieSpaceConfig,
        spec: CandidateSpec,
        serving_config: ServingCandidateConfig | None = None,
    ) -> tuple[GenieSpaceConfig, dict[str, Any]]:
        if spec.family == "curation":
            candidate_config, curation_summary = curate_space_config(current_config, mode=spec.name)
            return candidate_config, curation_summary.to_dict()

        if spec.family == "policy":
            policy = self.policy_candidates.get(spec.name)
            if policy is None:
                raise WorkspaceFlowError(f"Unknown policy candidate: {spec.name}")
            visible_config = self._config_with_visible_benchmarks(current_config)
            candidate = policy.apply(
                visible_config,
                PolicyContext(
                    original_config=deepcopy(visible_config),
                    train_benchmark_ids=frozenset(question.id for question in self._load_train_questions()),
                ),
            )
            before = _editable_serialized_space(current_config)
            candidate.benchmarks = None
            after = _editable_serialized_space(candidate)
            return candidate, {
                "changed": before != after,
                "policy": spec.name,
                "description": spec.description,
            }

        if spec.family == "centaur_prepass":
            policy = self.centaur_prepass_candidates.get(spec.name)
            if policy is None:
                raise WorkspaceFlowError(f"Unknown centaur pre-pass candidate: {spec.name}")
            visible_config = self._config_with_visible_benchmarks(current_config)
            candidate = policy.apply(
                visible_config,
                PolicyContext(
                    original_config=deepcopy(visible_config),
                    train_benchmark_ids=frozenset(question.id for question in self._load_train_questions()),
                ),
            )
            before = _editable_serialized_space(current_config)
            candidate.benchmarks = None
            after = _editable_serialized_space(candidate)
            return candidate, {
                "changed": before != after,
                "policy": spec.name,
                "description": spec.description,
                "prepass": True,
            }

        if spec.family == "serving":
            if serving_config is None:
                raise WorkspaceFlowError("Serving-backed optimization requires a serving endpoint.")
            candidate, serving_summary = self._run_serving_candidate(
                current_config=current_config,
                spec=spec,
                serving_config=serving_config,
            )
            before = _editable_serialized_space(current_config)
            candidate.benchmarks = None
            after = _editable_serialized_space(candidate)
            return candidate, {
                **serving_summary,
                "changed": before != after,
            }

        raise WorkspaceFlowError(f"Unsupported candidate family: {spec.family}")

    def _persist_optimization_summary(self, summary: dict[str, Any]) -> None:
        existing_summary = _load_json_dict(self.artifacts.optimization_summary_path) or {}
        existing_suppressed = existing_summary.get("suppressed_candidates")
        current_suppressed = summary.get("suppressed_candidates")
        if isinstance(existing_suppressed, dict):
            merged_suppressed = dict(existing_suppressed)
            if isinstance(current_suppressed, dict):
                merged_suppressed.update(current_suppressed)
            summary["suppressed_candidates"] = merged_suppressed
        self.artifacts.write_json_path(self.artifacts.optimization_summary_path, summary)

    def _load_suppressed_candidates(self) -> dict[str, dict[str, Any]]:
        optimization_summary = _load_json_dict(self.artifacts.optimization_summary_path) or {}
        suppressed = optimization_summary.get("suppressed_candidates")
        if not isinstance(suppressed, dict):
            return {}
        return {
            str(key): value
            for key, value in suppressed.items()
            if key and isinstance(value, dict)
        }

    def _persist_suppressed_candidate(
        self,
        *,
        summary_path: Path,
        spec: CandidateSpec,
        reason: str,
        label: str,
        candidate_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        reason_code = _candidate_suppression_reason_code(reason)
        if reason_code is None or not spec.suppression_keys:
            return None

        entry = _candidate_suppression_entry(
            spec=spec,
            reason=reason,
            reason_code=reason_code,
            label=label,
            candidate_summary=candidate_summary,
        )
        optimization_summary = _load_json_dict(summary_path) or {}
        suppressed = optimization_summary.get("suppressed_candidates")
        if not isinstance(suppressed, dict):
            suppressed = {}
        for key in spec.suppression_keys:
            suppressed[key] = entry
        optimization_summary["suppressed_candidates"] = suppressed
        self.artifacts.write_json_path(summary_path, optimization_summary)
        return entry

    def _confirm_candidate_beat(
        self,
        *,
        pass_label: str,
        reference_train_rate: float | None,
        reference_validation_rate: float | None,
    ) -> dict[str, Any]:
        """Re-run the visible benchmark to confirm an accepted beat is not noise.

        Optional anti-noise insurance for greedy promotion. The candidate is already
        pushed to the clone, so this re-benchmarks train + validation
        ``self._confirm_on_beat`` times and confirms the beat only when no re-run
        regresses the visible reference. It never touches the hidden holdout, so the
        confirmation stays inside the visible split.
        """
        runs = max(0, int(getattr(self, "_confirm_on_beat", 0)))
        samples: list[dict[str, float]] = []
        for index in range(runs):
            train_run, _ = self._run_train_benchmark_internal(
                label=f"{pass_label}.confirm{index + 1}.train",
                persist_latest=False,
                min_pass_rate=None,
            )
            validation_summary = self.run_validation_benchmark(
                label=f"{pass_label}.confirm{index + 1}.validation",
                persist_latest=False,
                min_pass_rate=None,
            )
            train_rate = float(train_run.pass_rate)
            validation_rate = float(validation_summary.get("pass_rate", 0.0))
            samples.append({"train_rate": train_rate, "validation_rate": validation_rate})
            train_regressed = (
                reference_train_rate is not None
                and train_rate < reference_train_rate - _TRAIN_REGRESSION_EPSILON
            )
            validation_regressed = (
                reference_validation_rate is not None
                and validation_rate < reference_validation_rate - _VALIDATION_REGRESSION_EPSILON
            )
            if train_regressed or validation_regressed:
                return {
                    "confirmed": False,
                    "reason": (
                        f"confirm-on-beat re-run {index + 1}/{runs} regressed visible "
                        f"reference: train {train_rate:.1f}% / validation {validation_rate:.1f}%"
                    ),
                    "details": {"confirm_on_beat_runs": runs, "confirm_samples": samples},
                }
        return {
            "confirmed": True,
            "reason": "",
            "details": {"confirm_on_beat_runs": runs, "confirm_samples": samples},
        }

    def _reject_candidate_before_validation(
        self,
        *,
        spec: CandidateSpec,
        pass_label: str,
        pre_pass_config: GenieSpaceConfig,
        reason: str,
        train_rate: float,
        train_failed_ids: list[str],
        reference_train_rate: float | None,
        reference_validation_rate: float | None,
        candidate_summary: dict[str, Any] | None = None,
        rejection_class: str | None = None,
        gate_details: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float | None, float | None]:
        decompose_config(pre_pass_config, self.artifacts.space_dir, include_benchmarks=False)
        self.push()
        event = {
            "phase": "candidate_evaluated",
            "label": pass_label,
            "name": spec.name,
            "family": spec.family,
            "outcome": "rejected",
            "train_rate": train_rate,
            "validation_rate": None,
            "reason": reason,
            "train_failed_ids": train_failed_ids,
            "skipped_validation": True,
            "candidate_summary": candidate_summary,
        }
        if rejection_class is not None:
            event["rejection_class"] = rejection_class
        if gate_details is not None:
            event["gate_details"] = gate_details
        self.artifacts.record_iteration(event)
        result = {
            "name": spec.name,
            "family": spec.family,
            "description": spec.description,
            "outcome": "rejected",
            "reason": reason,
            "train_rate": train_rate,
            "validation_rate": None,
            "skipped_validation": True,
            "candidate_summary": candidate_summary,
        }
        if rejection_class is not None:
            result["rejection_class"] = rejection_class
        if gate_details is not None:
            result["gate_details"] = gate_details
        return (
            result,
            reference_train_rate,
            reference_validation_rate,
        )

    def _run_candidate_spec(
        self,
        *,
        spec: CandidateSpec,
        label: str,
        reference_train_rate: float | None,
        reference_validation_rate: float | None,
        gating_active: bool,
        serving_config: ServingCandidateConfig | None = None,
        summary_path: Path | None = None,
        suppressed_candidates: dict[str, dict[str, Any]] | None = None,
        mlflow_tracker: MlflowTrackerProtocol | None = None,
        sweep_index: int = 0,
    ) -> tuple[dict[str, Any], float | None, float | None]:
        suppressed_entry = _matching_suppression_entry(spec, suppressed_candidates)
        if suppressed_entry is not None:
            result = {
                "name": spec.name,
                "family": spec.family,
                "description": spec.description,
                "outcome": "skipped",
                "reason": str(suppressed_entry.get("reason") or "suppressed"),
                "suppressed": True,
                "suppression_reason_code": suppressed_entry.get("reason_code"),
                "suppression_label": suppressed_entry.get("label"),
                "candidate_summary": suppressed_entry.get("candidate_summary"),
            }
            self.artifacts.record_iteration(
                {
                    "phase": "candidate_evaluated",
                    "label": f"{label}.{spec.name}",
                    "name": spec.name,
                    "family": spec.family,
                    "outcome": "skipped",
                    "reason": result["reason"],
                    "suppressed": True,
                    "suppression_reason_code": result["suppression_reason_code"],
                }
            )
            return (
                result,
                reference_train_rate,
                reference_validation_rate,
            )

        try:
            if mlflow_tracker is not None:
                mlflow_tracker.start_candidate_run(
                    spec_name=spec.name,
                    family=spec.family,
                    sweep_index=sweep_index,
                )
            pre_pass_config = assemble_config(self.artifacts.space_dir)
            candidate_config, candidate_summary = self._apply_candidate_spec(
                current_config=pre_pass_config,
                spec=spec,
                serving_config=serving_config,
            )
            if not bool(candidate_summary.get("changed")):
                no_change_result = {
                    "name": spec.name,
                    "family": spec.family,
                    "description": spec.description,
                    "outcome": "skipped",
                    "reason": str(candidate_summary.get("reason", "no_changes")),
                    "train_rate": reference_train_rate,
                    "validation_rate": reference_validation_rate,
                    "candidate_summary": candidate_summary,
                }
                self.artifacts.record_iteration(
                    {
                        "phase": "candidate_evaluated",
                        "label": f"{label}.{spec.name}",
                        "name": spec.name,
                        "family": spec.family,
                        "outcome": no_change_result["outcome"],
                        "train_rate": reference_train_rate,
                        "validation_rate": reference_validation_rate,
                        "reason": no_change_result["reason"],
                        "candidate_summary": candidate_summary,
                    }
                )
                _no_change_result = (
                    no_change_result,
                    reference_train_rate,
                    reference_validation_rate,
                )
                if mlflow_tracker is not None:
                    mlflow_tracker.log_candidate_result(_no_change_result[0])
                return _no_change_result

            decompose_config(candidate_config, self.artifacts.space_dir, include_benchmarks=False)
            try:
                self.push()
            except WorkspaceFlowError as exc:
                decompose_config(pre_pass_config, self.artifacts.space_dir, include_benchmarks=False)
                self.push()
                _push_fail_result = (
                    {
                        "name": spec.name,
                        "family": spec.family,
                        "description": spec.description,
                        "outcome": "rejected",
                        "reason": f"push_failed: {exc}",
                    },
                    reference_train_rate,
                    reference_validation_rate,
                )
                if mlflow_tracker is not None:
                    mlflow_tracker.log_candidate_result(_push_fail_result[0])
                return _push_fail_result

            pass_label = f"{label}.{spec.name}"

            # --- Fail-fast: run failures first, abort early if none improve ---
            previous_run = self._load_reference_train_run()
            previous_validation_summary = self._load_validation_summary()
            baseline_run = _load_run(self.artifacts.baseline_train_path)
            previous_failure_ids: set[str] = set()
            if previous_run is not None:
                previous_failure_ids = {
                    result.question_id
                    for result in previous_run.results
                    if result.status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}
                }

            train_run, train_payload = self._run_train_benchmark_internal(
                label=f"{pass_label}.train",
                persist_latest=False,
                min_pass_rate=reference_train_rate if gating_active else None,
            )
            train_rate = train_run.pass_rate
            train_failed_ids = [
                result.question_id
                for result in train_run.results
                if result.status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}
            ]

            # Health-check gate: broken config — train crashed to 0% from >0% (any magnitude),
            # OR train sank to <=floor with a large pp drop from a healthy reference.
            _train_crashed_to_zero = (
                train_rate == 0.0
                and reference_train_rate is not None
                and reference_train_rate > 0.0
            )
            _train_collapsed = (
                train_rate <= _TRAIN_HEALTH_FLOOR_ABSOLUTE
                and reference_train_rate is not None
                and (reference_train_rate - train_rate) >= _TRAIN_HEALTH_MAX_DROP_PCT
            )
            if _train_crashed_to_zero or _train_collapsed:
                reason = (
                    f"train_health_failed: {reference_train_rate:.1f}% -> {train_rate:.1f}%"
                )
                _reject_result = self._reject_candidate_before_validation(
                    spec=spec,
                    pass_label=pass_label,
                    pre_pass_config=pre_pass_config,
                    reason=reason,
                    train_rate=train_rate,
                    train_failed_ids=train_failed_ids,
                    reference_train_rate=reference_train_rate,
                    reference_validation_rate=reference_validation_rate,
                    candidate_summary=candidate_summary,
                )
                if mlflow_tracker is not None:
                    mlflow_tracker.log_candidate_result(_reject_result[0])
                return _reject_result

            # Fail-fast gate 1: skip validation if train regressed
            if (
                gating_active
                and reference_train_rate is not None
                and train_rate < reference_train_rate - _TRAIN_REGRESSION_EPSILON
            ):
                reason = f"train regressed {reference_train_rate:.1f}% -> {train_rate:.1f}%"
                _reject_result = self._reject_candidate_before_validation(
                    spec=spec,
                    pass_label=pass_label,
                    pre_pass_config=pre_pass_config,
                    reason=reason,
                    train_rate=train_rate,
                    train_failed_ids=train_failed_ids,
                    reference_train_rate=reference_train_rate,
                    reference_validation_rate=reference_validation_rate,
                    candidate_summary=candidate_summary,
                )
                if mlflow_tracker is not None:
                    mlflow_tracker.log_candidate_result(_reject_result[0])
                return _reject_result

            train_diff = train_payload.get("diff")
            regressed_guard_ids = (
                [
                    str(value)
                    for value in train_diff.get("regressed", [])
                    if str(value).strip()
                ]
                if isinstance(train_diff, dict)
                else []
            )
            if gating_active and regressed_guard_ids:
                accepted_fix_regressed_ids = _accepted_fix_regression_ids(
                    baseline_run=baseline_run,
                    reference_run=previous_run,
                    current_run=train_run,
                )
                if accepted_fix_regressed_ids:
                    reason = "accepted fix regressed: " + ", ".join(accepted_fix_regressed_ids[:5])
                    rejection_class = "accepted_fix_regression"
                else:
                    reason = "pass guard regressed: " + ", ".join(regressed_guard_ids[:5])
                    rejection_class = "pass_guard_regression"
                _reject_result = self._reject_candidate_before_validation(
                    spec=spec,
                    pass_label=pass_label,
                    pre_pass_config=pre_pass_config,
                    reason=reason,
                    train_rate=train_rate,
                    train_failed_ids=train_failed_ids,
                    reference_train_rate=reference_train_rate,
                    reference_validation_rate=reference_validation_rate,
                    candidate_summary=candidate_summary,
                    rejection_class=rejection_class,
                    gate_details={
                        "regressed_guard_ids": regressed_guard_ids,
                        "accepted_fix_regressed_ids": accepted_fix_regressed_ids,
                    },
                )
                if mlflow_tracker is not None:
                    mlflow_tracker.log_candidate_result(_reject_result[0])
                return _reject_result

            # Fail-fast gate 2: skip validation if zero failures improved and train didn't improve
            if (
                gating_active
                and previous_failure_ids
                and reference_train_rate is not None
                and train_rate <= reference_train_rate
                and previous_failure_ids <= set(train_failed_ids)
            ):
                reason = (
                    "known failures did not improve; "
                    f"train remained at {train_rate:.1f}%"
                )
                _reject_result = self._reject_candidate_before_validation(
                    spec=spec,
                    pass_label=pass_label,
                    pre_pass_config=pre_pass_config,
                    reason=reason,
                    train_rate=train_rate,
                    train_failed_ids=train_failed_ids,
                    reference_train_rate=reference_train_rate,
                    reference_validation_rate=reference_validation_rate,
                    candidate_summary=candidate_summary,
                )
                if mlflow_tracker is not None:
                    mlflow_tracker.log_candidate_result(_reject_result[0])
                return _reject_result

            # Train passed fast gates — run validation
            validation_min_pass_rate = (
                reference_validation_rate + _VALIDATION_IMPROVEMENT_EPSILON
                if gating_active and reference_validation_rate is not None
                else None
            )
            validation_summary = self.run_validation_benchmark(
                label=f"{pass_label}.validation",
                persist_latest=False,
                min_pass_rate=validation_min_pass_rate,
            )
            validation_rate = float(validation_summary.get("pass_rate", 0.0))
            proxy_pressure = compute_visible_proxy_pressure(
                self._visible_proxy_records(
                    train_run=previous_run,
                    validation_summary=previous_validation_summary,
                ),
                self._visible_proxy_records(
                    train_run=train_run,
                    validation_summary=validation_summary,
                ),
                before_train_rate=reference_train_rate,
                after_train_rate=train_rate,
                before_validation_rate=reference_validation_rate,
                after_validation_rate=validation_rate,
            ).to_dict()
            self.artifacts.write_json_path(
                self.artifacts.visible_proxy_pressure_path,
                proxy_pressure,
            )
            validation_failed_ids = [
                str(value)
                for value in (
                    list(validation_summary.get("failed_ids", []))
                    + list(validation_summary.get("error_ids", []))
                )
                if value
            ]

            accepted = True
            reason = ""
            rejection_class: str | None = None
            gate_details: dict[str, Any] | None = None
            acceptance_class: str | None = None
            if gating_active:
                train_improved = (
                    reference_train_rate is not None
                    and train_rate > reference_train_rate + _TRAIN_IMPROVEMENT_EPSILON
                )
                validation_regressed = (
                    reference_validation_rate is not None
                    and validation_rate < reference_validation_rate - _VALIDATION_REGRESSION_EPSILON
                )
                validation_improved = (
                    reference_validation_rate is None
                    or validation_rate >= reference_validation_rate + _VALIDATION_IMPROVEMENT_EPSILON
                )
                validation_tied = (
                    reference_validation_rate is not None
                    and abs(validation_rate - reference_validation_rate) <= _VALIDATION_REGRESSION_EPSILON
                )
                if validation_regressed:
                    accepted = False
                    early_stop_reason = validation_summary.get("early_stop_reason")
                    if early_stop_reason:
                        gate_details = {
                            "early_stop_reason": early_stop_reason,
                            "required_passes": validation_summary.get("required_passes"),
                            "best_possible_pass_rate": validation_summary.get("best_possible_pass_rate"),
                            "skipped_ids": validation_summary.get("skipped_ids", []),
                        }
                    reason = (
                        "validation improvement impossible; "
                        if early_stop_reason
                        else "validation did not improve; "
                    ) + (
                        "validation regressed "
                        f"{reference_validation_rate:.1f}% -> {validation_rate:.1f}%"
                    )
                    rejection_class = (
                        "validation_improvement_impossible"
                        if early_stop_reason
                        else "validation_regression"
                    )
                elif not validation_improved:
                    if train_improved and validation_tied:
                        acceptance_class = "train_lift_validation_tie"
                        gate_details = {
                            "train_lift_without_validation_regression": True,
                            "reference_train_rate": reference_train_rate,
                            "reference_validation_rate": reference_validation_rate,
                        }
                    else:
                        accepted = False
                        early_stop_reason = validation_summary.get("early_stop_reason")
                        if early_stop_reason:
                            rejection_class = "validation_improvement_impossible"
                            gate_details = {
                                "early_stop_reason": early_stop_reason,
                                "required_passes": validation_summary.get("required_passes"),
                                "best_possible_pass_rate": validation_summary.get("best_possible_pass_rate"),
                                "skipped_ids": validation_summary.get("skipped_ids", []),
                            }
                        reason = (
                            "validation improvement impossible; "
                            if early_stop_reason
                            else ""
                        ) + (
                            "validation did not improve "
                            f"{reference_validation_rate:.1f}% -> {validation_rate:.1f}%"
                        )
                if accepted and "visible_score_lift_with_proxy_degradation" in (
                    proxy_pressure.get("warnings", []) or []
                ):
                    accepted = False
                    reason = "visible proxy guards degraded despite visible score lift"
                    rejection_class = "visible_proxy_regression"
                    gate_details = {
                        "visible_proxy_pressure": proxy_pressure,
                    }

            # Optional confirm-on-beat anti-noise insurance: re-run the visible
            # benchmark to verify an accepted beat reproduces before checkpointing.
            if accepted and getattr(self, "_confirm_on_beat", 0) > 0:
                confirm_outcome = self._confirm_candidate_beat(
                    pass_label=pass_label,
                    reference_train_rate=reference_train_rate,
                    reference_validation_rate=reference_validation_rate,
                )
                gate_details = {
                    **(gate_details or {}),
                    **(confirm_outcome.get("details") or {}),
                }
                if not confirm_outcome["confirmed"]:
                    accepted = False
                    acceptance_class = None
                    reason = confirm_outcome["reason"]
                    rejection_class = "confirm_on_beat_failed"

            checkpoint_path: str | None = None
            if accepted:
                self._write_train_artifacts(
                    payload=train_payload,
                    baseline=False,
                    persist_latest=True,
                )
                self._write_scored_summary_artifacts(
                    summary=validation_summary,
                    baseline=False,
                    persist_latest=True,
                    baseline_path=self.artifacts.baseline_validation_score_path,
                    latest_path=self.artifacts.latest_validation_score_path,
                    baseline_summary_path=self.artifacts.baseline_validation_summary_path,
                    latest_summary_path=self.artifacts.latest_validation_summary_path,
                )
                checkpoint = self.artifacts.create_checkpoint(
                    label=spec.name,
                    metadata={
                        "candidate_name": spec.name,
                        "family": spec.family,
                        "label": label,
                        "train_rate": train_rate,
                        "validation_rate": validation_rate,
                    },
                )
                checkpoint_path = str(checkpoint)
                updated_train_rate = train_rate
                updated_validation_rate = validation_rate
            else:
                decompose_config(pre_pass_config, self.artifacts.space_dir, include_benchmarks=False)
                self.push()
                updated_train_rate = reference_train_rate
                updated_validation_rate = reference_validation_rate

            result = {
                "name": spec.name,
                "family": spec.family,
                "description": spec.description,
                "outcome": "accepted" if accepted else "rejected",
                "reason": reason or None,
                "train_rate": train_rate,
                "validation_rate": validation_rate,
                "train_failed_ids": train_failed_ids,
                "validation_failed_ids": validation_failed_ids,
                "checkpoint_path": checkpoint_path,
                "candidate_summary": candidate_summary,
                "visible_proxy_pressure": proxy_pressure,
            }
            if rejection_class is not None:
                result["rejection_class"] = rejection_class
            if acceptance_class is not None:
                result["acceptance_class"] = acceptance_class
            if gate_details is not None:
                result["gate_details"] = gate_details
            self.artifacts.record_iteration(
                {
                    "phase": "candidate_evaluated",
                    "label": pass_label,
                    "name": spec.name,
                    "family": spec.family,
                    "outcome": result["outcome"],
                    "train_rate": train_rate,
                    "validation_rate": validation_rate,
                    "reason": reason or None,
                    "train_failed_ids": train_failed_ids,
                    "validation_failed_ids": validation_failed_ids,
                    "candidate_summary": candidate_summary,
                    "visible_proxy_pressure": proxy_pressure,
                    "rejection_class": rejection_class,
                    "acceptance_class": acceptance_class,
                    "gate_details": gate_details,
                }
            )
            if mlflow_tracker is not None:
                mlflow_tracker.log_candidate_result(result)
            return result, updated_train_rate, updated_validation_rate
        finally:
            if mlflow_tracker is not None:
                mlflow_tracker.end_candidate_run()

    def _run_candidate_sweep(
        self,
        *,
        specs: list[CandidateSpec],
        label: str,
        summary_path: Path,
        initial_reference_train_rate: float | None,
        initial_reference_validation_rate: float | None,
        gating_active: bool,
        metadata: dict[str, Any] | None = None,
        serving_config: ServingCandidateConfig | None = None,
        mlflow_tracker: MlflowTrackerProtocol | None = None,
        sweep_index: int = 0,
    ) -> dict[str, Any]:
        reference_train_rate = initial_reference_train_rate
        reference_validation_rate = initial_reference_validation_rate
        pass_outcomes: list[dict[str, Any]] = []
        accepted_count = 0
        rejected_count = 0
        skipped_count = 0
        persist_suppressions = summary_path == self.artifacts.optimization_summary_path
        suppressed_candidates = self._load_suppressed_candidates() if persist_suppressions else {}

        def _persist_loop_summary() -> None:
            payload = {
                "label": label,
                "candidate_names": [spec.name for spec in specs],
                "accepted": accepted_count,
                "rejected": rejected_count,
                "skipped": skipped_count,
                "gating_active": gating_active,
                "final_train_rate": reference_train_rate,
                "final_validation_rate": reference_validation_rate,
                "passes": pass_outcomes,
                **(metadata or {}),
            }
            if suppressed_candidates:
                payload["suppressed_candidates"] = suppressed_candidates
            self.artifacts.write_json_path(summary_path, payload)

        import logging as _logging
        _sweep_logger = _logging.getLogger(__name__)
        for spec_idx, spec in enumerate(specs, 1):
            _sweep_logger.info(
                "[%s] Candidate %d/%d: %s (%s)",
                label, spec_idx, len(specs), spec.name, spec.family,
            )
            result, reference_train_rate, reference_validation_rate = self._run_candidate_spec(
                spec=spec,
                label=label,
                reference_train_rate=reference_train_rate,
                reference_validation_rate=reference_validation_rate,
                gating_active=gating_active,
                serving_config=serving_config,
                summary_path=summary_path if persist_suppressions else None,
                suppressed_candidates=suppressed_candidates,
                mlflow_tracker=mlflow_tracker,
                sweep_index=sweep_index,
            )
            if persist_suppressions and result.get("outcome") == "rejected":
                suppression_entry = self._persist_suppressed_candidate(
                    summary_path=summary_path,
                    spec=spec,
                    reason=str(result.get("reason") or ""),
                    label=f"{label}.{spec.name}",
                    candidate_summary=(
                        result.get("candidate_summary")
                        if isinstance(result.get("candidate_summary"), dict)
                        else None
                    ),
                )
                if suppression_entry is not None:
                    for key in spec.suppression_keys:
                        suppressed_candidates[key] = suppression_entry
                    result["suppression_reason_code"] = suppression_entry.get("reason_code")
            pass_outcomes.append(result)
            outcome = result["outcome"]
            if outcome == "accepted":
                accepted_count += 1
            elif outcome == "rejected":
                rejected_count += 1
            else:
                skipped_count += 1
            _sweep_logger.info(
                "[%s] %s -> %s (train=%.1f%%, val=%s) [%d accepted, %d rejected, %d skipped]",
                label, spec.name, outcome,
                reference_train_rate or 0,
                f"{reference_validation_rate:.1f}%" if reference_validation_rate else "n/a",
                accepted_count, rejected_count, skipped_count,
            )
            _persist_loop_summary()

        result = {
            "label": label,
            "candidate_names": [spec.name for spec in specs],
            "accepted": accepted_count,
            "rejected": rejected_count,
            "skipped": skipped_count,
            "gating_active": gating_active,
            "final_train_rate": reference_train_rate,
            "final_validation_rate": reference_validation_rate,
            "passes": pass_outcomes,
            **(metadata or {}),
        }
        if suppressed_candidates:
            result["suppressed_candidates"] = suppressed_candidates
        self.artifacts.write_json_path(summary_path, result)
        return result

    def run_autonomous_optimization(
        self,
        *,
        label: str = "optimize",
        plateau_rounds: int = 3,
        max_iterations: int | None = None,
        candidate_budget: int | None = None,
        strategy: str = AUTONOMOUS_STRATEGY_CENTAUR,
        serving_endpoint: str | None = None,
        serving_model: str | None = None,
        serving_temperature: float | None = None,
        serving_max_tokens: int = 16000,
        serving_reasoning_effort: str | None = "low",
        serving_response_format: str = "json_schema",
        serving_api_format: str = "auto",
        serving_candidate_mode: str | None = None,
        greedy_best_ever: bool = False,
        confirm_on_beat: int = 0,
    ) -> dict[str, Any]:
        if strategy not in {
            AUTONOMOUS_STRATEGY_CENTAUR,
            AUTONOMOUS_STRATEGY_SERVING,
        }:
            raise WorkspaceFlowError(f"Unsupported autonomous strategy: {strategy}")
        # Round C promotion controls (default off → unchanged noise-aware behaviour).
        self._greedy_best_ever_promotion = bool(greedy_best_ever)
        self._confirm_on_beat = max(0, int(confirm_on_beat))
        serving_config: ServingCandidateConfig | None = None
        forced_serving_candidate_mode = (
            serving_candidate_mode.strip()
            if serving_candidate_mode is not None and serving_candidate_mode.strip()
            else None
        )
        if (
            forced_serving_candidate_mode is not None
            and serving_candidate_allowed_operation_groups(forced_serving_candidate_mode) is None
        ):
            raise WorkspaceFlowError(
                f"Unsupported serving candidate mode: {forced_serving_candidate_mode}"
            )
        if strategy == AUTONOMOUS_STRATEGY_SERVING:
            if not serving_endpoint or not serving_endpoint.strip():
                raise WorkspaceFlowError(
                    "Serving-backed autonomous optimization requires --serving-endpoint or MAXGENIE_SERVING_ENDPOINT."
                )
            try:
                serving_config = ServingCandidateConfig(
                    endpoint=serving_endpoint.strip(),
                    model=serving_model.strip() if serving_model and serving_model.strip() else None,
                    temperature=serving_temperature,
                    max_tokens=serving_max_tokens,
                    reasoning_effort=(
                        serving_reasoning_effort.strip()
                        if serving_reasoning_effort and serving_reasoning_effort.strip()
                        else None
                    ),
                    response_format=serving_response_format,
                    api_format=serving_api_format,
                )
            except ValueError as exc:
                raise WorkspaceFlowError(str(exc)) from exc

        tracker = self._build_mlflow_tracker()

        from datetime import datetime as _dt

        optimization_started_at = _dt.now()

        baseline_train = self._load_reference_train_run()
        baseline_validation = self._load_validation_summary()
        if baseline_train is None or baseline_validation is None:
            raise WorkspaceFlowError(
                "Baseline train and validation scores are required before autonomous optimization."
            )

        baseline_train_rate = baseline_train.pass_rate
        baseline_validation_rate = float(baseline_validation.get("pass_rate", 0.0))
        baseline_test_rate = self.artifacts.read_score_path(self.artifacts.baseline_test_score_path)
        baseline_full = _load_json_dict(self.artifacts.latest_full_summary_path) or {}
        evaluation_context = _benchmark_evaluation_context(
            self.comparator,
            match_threshold=self.match_threshold,
        )
        baseline_comparison_metrics = _aggregate_comparison_metrics(
            [
                _comparison_metrics_from_run(baseline_train),
                baseline_validation,
                baseline_full.get("test", {}) if isinstance(baseline_full, dict) else {},
                baseline_full.get("overall", {}) if isinstance(baseline_full, dict) else {},
            ]
        )
        baseline_checkpoint = self.artifacts.create_checkpoint(
            label="baseline",
            metadata={
                "label": label,
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": baseline_full.get("overall", {}).get("pass_rate"),
            },
        )

        summary: dict[str, Any] = {
            "label": label,
            "mode": "autonomous",
            "strategy": strategy,
            "started_at": optimization_started_at.isoformat(),
            "plateau_rounds": plateau_rounds,
            "max_iterations": max_iterations,
            "candidate_budget": candidate_budget,
            "serving_candidate_mode": forced_serving_candidate_mode,
            "evaluation_context": evaluation_context,
            "baseline_evaluation_context": evaluation_context,
            "evidence_protocol": _build_evidence_protocol(
                strategy=strategy,
                evaluation_context=evaluation_context,
            ),
            "rerun_plan": _build_rerun_plan(strategy=strategy),
            "baseline": {
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": baseline_full.get("overall", {}).get("pass_rate"),
                "comparison_metrics": baseline_comparison_metrics,
            },
            "baseline_checkpoint_path": str(baseline_checkpoint),
            "accepted": 0,
            "rejected": 0,
            "skipped": 0,
            "passes": [],
            "best_checkpoint_path": str(baseline_checkpoint),
            "best_train_rate": baseline_train_rate,
            "best_validation_rate": baseline_validation_rate,
            "stop_reason": None,
        }
        if serving_config is not None:
            summary["serving_endpoint"] = serving_config.endpoint
            summary["serving_model"] = serving_config.model
            summary["serving_temperature"] = serving_config.temperature
            summary["serving_reasoning_effort"] = serving_config.reasoning_effort
            summary["serving_response_format"] = serving_config.response_format
            summary["serving_api_format"] = serving_config.api_format
            summary["serving_candidate_mode"] = forced_serving_candidate_mode
        summary["protocol_preflight"] = self._write_protocol_preflight(
            mode="autonomous",
            strategy=strategy,
            summary=baseline_full,
            serving_endpoint=serving_config.endpoint if serving_config is not None else None,
            serving_model=serving_config.model if serving_config is not None else None,
            serving_reasoning_effort=(
                serving_config.reasoning_effort if serving_config is not None else None
            ),
        )
        self._persist_optimization_summary(summary)

        tracker.start_optimization_run(
            strategy=strategy,
            label=label,
            baseline_metrics={
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": (
                    baseline_full.get("overall", {}).get("pass_rate")
                    if isinstance(baseline_full, dict) else None
                ),
            },
            params={
                "max_iterations": max_iterations,
                "plateau_rounds": plateau_rounds,
                "candidate_budget": candidate_budget,
                "gating_active": True,
            },
        )

        try:
            plateau_count = 0
            sweep_index = 0
            attempted_count = 0

            import logging as _opt_logging
            _opt_logger = _opt_logging.getLogger(__name__)
            while (max_iterations is None or sweep_index < max_iterations) and plateau_count < plateau_rounds:
                sweep_index += 1
                max_iterations_label = str(max_iterations) if max_iterations is not None else "unbounded"
                _opt_logger.info(
                    "=== Sweep %d/%s (plateau %d/%d, best train=%.1f%%, best val=%.1f%%) ===",
                    sweep_index, max_iterations_label, plateau_count, plateau_rounds,
                    summary.get("best_train_rate") or 0,
                    summary.get("best_validation_rate") or 0,
                )
                if strategy == AUTONOMOUS_STRATEGY_SERVING:
                    specs = [
                        self._serving_candidate_spec(
                            sweep_index,
                            candidate_mode=forced_serving_candidate_mode,
                        )
                    ]
                elif strategy == AUTONOMOUS_STRATEGY_CENTAUR:
                    specs = self._centaur_prepass_candidate_specs() + self._default_candidate_specs()
                else:
                    specs = self._default_candidate_specs()
                if candidate_budget is not None:
                    remaining = candidate_budget - attempted_count
                    if remaining <= 0:
                        summary["stop_reason"] = "candidate_budget_reached"
                        break
                    specs = specs[:remaining]
                if not specs:
                    summary["stop_reason"] = "no_candidates_remaining"
                    break

                sweep_label = f"{label}.sweep{sweep_index:02d}"
                sweep_result = self._run_candidate_sweep(
                    specs=specs,
                    label=sweep_label,
                    summary_path=self.artifacts.optimization_summary_path,
                    initial_reference_train_rate=summary["best_train_rate"],
                    initial_reference_validation_rate=summary["best_validation_rate"],
                    gating_active=True,
                    metadata={
                        "mode": "autonomous",
                        "strategy": strategy,
                        "sweep_index": sweep_index,
                    },
                    serving_config=serving_config,
                    mlflow_tracker=tracker,
                    sweep_index=sweep_index,
                )
                attempted_count += len(specs)
                summary["accepted"] += int(sweep_result.get("accepted", 0))
                summary["rejected"] += int(sweep_result.get("rejected", 0))
                summary["skipped"] += int(sweep_result.get("skipped", 0))
                summary["passes"].extend(sweep_result.get("passes", []))
                accepted_passes = [
                    entry
                    for entry in sweep_result.get("passes", [])
                    if isinstance(entry, dict) and entry.get("outcome") == "accepted"
                ]
                if accepted_passes:
                    plateau_count = 0
                    last_accepted = accepted_passes[-1]
                    summary["best_train_rate"] = float(last_accepted.get("train_rate", summary["best_train_rate"]))
                    summary["best_validation_rate"] = float(
                        last_accepted.get("validation_rate", summary["best_validation_rate"])
                    )
                    if last_accepted.get("checkpoint_path"):
                        summary["best_checkpoint_path"] = str(last_accepted["checkpoint_path"])
                else:
                    plateau_count += 1
                summary["plateau_count"] = plateau_count
                summary["sweeps_completed"] = sweep_index
                summary["attempted_candidates"] = attempted_count
                self._persist_optimization_summary(summary)

            if summary.get("stop_reason") is None:
                if plateau_count >= plateau_rounds:
                    summary["stop_reason"] = "validation_plateau"
                elif max_iterations is not None and sweep_index >= max_iterations:
                    summary["stop_reason"] = "max_iterations_reached"
                else:
                    summary["stop_reason"] = "completed"

            if summary["accepted"] > 0:
                restored_checkpoint_path = self._restore_best_checkpoint_for_final_audit(summary)
                if restored_checkpoint_path is not None:
                    summary["final_audit_restored_checkpoint_path"] = restored_checkpoint_path
                    self._persist_optimization_summary(summary)
                final_summary = self.run_full_benchmark(label=f"{label}.final", baseline=False)
            else:
                final_summary = _load_json_dict(self.artifacts.latest_full_summary_path) or baseline_full

            optimization_completed_at = _dt.now()
            summary["completed_at"] = optimization_completed_at.isoformat()
            summary["elapsed_seconds"] = round((optimization_completed_at - optimization_started_at).total_seconds(), 1)
            summary["final"] = final_summary
            summary["final_evaluation_context"] = (
                final_summary.get("evaluation_context")
                if isinstance(final_summary, dict) and isinstance(final_summary.get("evaluation_context"), dict)
                else evaluation_context
            )
            summary["final_comparison_metrics"] = (
                final_summary.get("overall", {}).get("comparison_metrics")
                if isinstance(final_summary, dict)
                else None
            )
            if isinstance(final_summary, dict):
                protocol_preflight = self._write_protocol_preflight(
                    mode="autonomous",
                    strategy=strategy,
                    summary=final_summary,
                    serving_endpoint=serving_config.endpoint if serving_config is not None else None,
                    serving_model=serving_config.model if serving_config is not None else None,
                    serving_reasoning_effort=(
                        serving_config.reasoning_effort if serving_config is not None else None
                    ),
                )
                summary["protocol_preflight"] = protocol_preflight
                terminal_selection = _terminal_result_selection(
                    optimization_summary=summary,
                    final_summary=final_summary,
                    protocol_preflight=protocol_preflight,
                    noise_model=self._build_terminal_noise_model(summary, final_summary),
                    transfer_decisions=getattr(self, "_robustness_transfer_decisions", None),
                    greedy_best_ever=getattr(self, "_greedy_best_ever_promotion", False),
                )
                summary["terminal_selection"] = terminal_selection
                self.artifacts.write_json_path(
                    self.artifacts.terminal_selection_path,
                    terminal_selection,
                )
                self.artifacts.record_iteration(
                    {
                        "phase": "terminal_result_selection",
                        "label": label,
                        "selection": terminal_selection,
                    }
                )
                generalization_audit = _generalization_audit(
                    optimization_summary=summary,
                    final_summary=final_summary,
                )
                summary["generalization_audit"] = generalization_audit
                self.artifacts.write_json_path(
                    self.artifacts.generalization_audit_path,
                    generalization_audit,
                )
                self.artifacts.record_iteration(
                    {
                        "phase": "generalization_audit",
                        "label": label,
                        "audit": generalization_audit,
                    }
                )
            summary["last_stop_reason"] = summary.get("stop_reason")
            self.artifacts.record_iteration(
                {
                    "phase": "optimization_completed",
                    "label": label,
                    "summary": {
                        "accepted": summary["accepted"],
                        "rejected": summary["rejected"],
                        "skipped": summary["skipped"],
                        "stop_reason": summary["stop_reason"],
                        "best_validation_rate": summary["best_validation_rate"],
                        "final_full_pass_rate": (
                            final_summary.get("overall", {}).get("pass_rate")
                            if isinstance(final_summary, dict)
                            else None
                        ),
                    },
                }
            )
            self._persist_optimization_summary(summary)
            if isinstance(final_summary, dict):
                tracker.log_benchmark_summary(label="final", summary=final_summary)
            tracker.set_optimization_status(summary)
            return summary
        finally:
            tracker.end_optimization_run()

    def run_checkpoint_audit(
        self,
        *,
        label: str,
        checkpoint_path: Path,
        k_samples: int = 1,
    ) -> dict[str, Any]:
        from datetime import datetime as _dt

        started_at = _dt.now()
        checkpoint_path = Path(checkpoint_path)
        audit_k_samples = max(int(k_samples), 1)
        if not (checkpoint_path / "space").exists():
            raise WorkspaceFlowError(f"Checkpoint is missing space snapshot: {checkpoint_path}")

        baseline_train = self._load_reference_train_run()
        baseline_validation = self._load_validation_summary()
        if baseline_train is None or baseline_validation is None:
            raise WorkspaceFlowError(
                "Baseline train and validation scores are required before checkpoint audit."
            )

        baseline_train_rate = baseline_train.pass_rate
        baseline_validation_rate = float(baseline_validation.get("pass_rate", 0.0))
        baseline_test_rate = self.artifacts.read_score_path(self.artifacts.baseline_test_score_path)
        baseline_full = _load_json_dict(self.artifacts.baseline_full_summary_path)
        if not isinstance(baseline_full, dict):
            baseline_full = _load_json_dict(self.artifacts.latest_full_summary_path) or {}
        evaluation_context = _benchmark_evaluation_context(
            self.comparator,
            match_threshold=self.match_threshold,
        )
        baseline_comparison_metrics = _aggregate_comparison_metrics(
            [
                _comparison_metrics_from_run(baseline_train),
                baseline_validation,
                baseline_full.get("test", {}) if isinstance(baseline_full, dict) else {},
                baseline_full.get("overall", {}) if isinstance(baseline_full, dict) else {},
            ]
        )
        baseline_checkpoint = self.artifacts.create_checkpoint(
            label=f"{label}_starting_state",
            metadata={
                "label": label,
                "audit_checkpoint_path": str(checkpoint_path),
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": baseline_full.get("overall", {}).get("pass_rate"),
            },
        )
        summary: dict[str, Any] = {
            "label": label,
            "mode": "checkpoint_audit",
            "strategy": "checkpoint_audit",
            "audit_k_samples": audit_k_samples,
            "started_at": started_at.isoformat(),
            "evaluation_context": evaluation_context,
            "baseline_evaluation_context": evaluation_context,
            "evidence_protocol": _build_evidence_protocol(
                strategy="checkpoint_audit",
                evaluation_context=evaluation_context,
            ),
            "rerun_plan": _build_rerun_plan(strategy="checkpoint_audit"),
            "baseline": {
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": baseline_full.get("overall", {}).get("pass_rate"),
                "comparison_metrics": baseline_comparison_metrics,
            },
            "baseline_checkpoint_path": str(baseline_checkpoint),
            "audited_checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(checkpoint_path),
            "best_train_rate": baseline_train_rate,
            "best_validation_rate": baseline_validation_rate,
            "accepted": 0,
            "rejected": 0,
            "skipped": 0,
            "passes": [],
            "stop_reason": "checkpoint_audited",
        }
        self._persist_optimization_summary(summary)
        self.artifacts.record_iteration(
            {
                "phase": "checkpoint_audit_started",
                "label": label,
                "checkpoint_path": str(checkpoint_path),
                "baseline_checkpoint_path": str(baseline_checkpoint),
            }
        )

        self.restore_space_checkpoint(checkpoint_path)
        clone_info = self.push()
        self.artifacts.record_iteration(
            {
                "phase": "checkpoint_audit_restored",
                "label": label,
                "clone_space_id": clone_info.space_id,
                "checkpoint_path": str(checkpoint_path),
            }
        )

        if audit_k_samples > 1:
            # Noise characterisation: re-benchmark the restored state K times over the full
            # question set so the audit emits a multi_sample aggregate (sigma + per-question
            # pass fractions, including the hidden holdout) that measure-noise consumes and the
            # noise-aware terminal gate can read. run_population_benchmark preflights the
            # existing restored clone (it does not re-clone), so every repeat is an independent
            # Genie regeneration of this checkpoint's state.
            self.run_population_benchmark(
                label="checkpoint_audit_population",
                questions=self._load_all_benchmark_questions(),
                k_samples=audit_k_samples,
            )
            self.artifacts.record_iteration(
                {
                    "phase": "checkpoint_audit_population_sampled",
                    "label": label,
                    "k_samples": audit_k_samples,
                }
            )

        final_summary = self.run_full_benchmark(label=f"{label}.final", baseline=False)
        completed_at = _dt.now()
        summary["completed_at"] = completed_at.isoformat()
        summary["elapsed_seconds"] = round((completed_at - started_at).total_seconds(), 1)
        summary["final"] = final_summary
        summary["final_evaluation_context"] = (
            final_summary.get("evaluation_context")
            if isinstance(final_summary, dict) and isinstance(final_summary.get("evaluation_context"), dict)
            else evaluation_context
        )
        summary["final_comparison_metrics"] = (
            final_summary.get("overall", {}).get("comparison_metrics")
            if isinstance(final_summary, dict)
            else None
        )
        protocol_preflight = self._write_protocol_preflight(
            mode="checkpoint_audit",
            strategy="checkpoint_audit",
            summary=final_summary,
        )
        summary["protocol_preflight"] = protocol_preflight
        terminal_selection = _terminal_result_selection(
            optimization_summary=summary,
            final_summary=final_summary,
            protocol_preflight=protocol_preflight,
            noise_model=self._build_terminal_noise_model(summary, final_summary),
            transfer_decisions=getattr(self, "_robustness_transfer_decisions", None),
            greedy_best_ever=getattr(self, "_greedy_best_ever_promotion", False),
        )
        summary["terminal_selection"] = terminal_selection
        self.artifacts.write_json_path(
            self.artifacts.terminal_selection_path,
            terminal_selection,
        )
        self.artifacts.record_iteration(
            {
                "phase": "terminal_result_selection",
                "label": label,
                "selection": terminal_selection,
            }
        )
        generalization_audit = _generalization_audit(
            optimization_summary=summary,
            final_summary=final_summary,
        )
        summary["generalization_audit"] = generalization_audit
        self.artifacts.write_json_path(
            self.artifacts.generalization_audit_path,
            generalization_audit,
        )
        self.artifacts.record_iteration(
            {
                "phase": "generalization_audit",
                "label": label,
                "audit": generalization_audit,
            }
        )
        summary["last_stop_reason"] = summary.get("stop_reason")
        self.artifacts.record_iteration(
            {
                "phase": "checkpoint_audit_completed",
                "label": label,
                "summary": {
                    "stop_reason": summary["stop_reason"],
                    "final_full_pass_rate": final_summary.get("overall", {}).get("pass_rate"),
                    "terminal_selection_status": terminal_selection.get("status"),
                    "generalization_audit_status": generalization_audit.get("status"),
                },
            }
        )
        self._persist_optimization_summary(summary)
        return summary

    def run_patch_replay_audit(
        self,
        *,
        label: str,
        patch_path: Path,
    ) -> dict[str, Any]:
        from datetime import datetime as _dt

        started_at = _dt.now()
        patch_path = Path(patch_path)
        patch_payloads = load_artifact_patch_sequence(patch_path)
        if not patch_payloads:
            raise WorkspaceFlowError(f"Patch replay sequence is empty: {patch_path}")

        baseline_train = self._load_reference_train_run()
        baseline_validation = self._load_validation_summary()
        if baseline_train is None or baseline_validation is None:
            raise WorkspaceFlowError(
                "Baseline train and validation scores are required before patch replay audit."
            )

        baseline_train_rate = baseline_train.pass_rate
        baseline_validation_rate = float(baseline_validation.get("pass_rate", 0.0))
        baseline_test_rate = self.artifacts.read_score_path(self.artifacts.baseline_test_score_path)
        baseline_full = _load_json_dict(self.artifacts.baseline_full_summary_path)
        if not isinstance(baseline_full, dict):
            baseline_full = _load_json_dict(self.artifacts.latest_full_summary_path) or {}
        evaluation_context = _benchmark_evaluation_context(
            self.comparator,
            match_threshold=self.match_threshold,
        )
        baseline_comparison_metrics = _aggregate_comparison_metrics(
            [
                _comparison_metrics_from_run(baseline_train),
                baseline_validation,
                baseline_full.get("test", {}) if isinstance(baseline_full, dict) else {},
                baseline_full.get("overall", {}) if isinstance(baseline_full, dict) else {},
            ]
        )
        baseline_checkpoint = self.artifacts.create_checkpoint(
            label=f"{label}_starting_state",
            metadata={
                "label": label,
                "audit_patch_path": str(patch_path),
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": baseline_full.get("overall", {}).get("pass_rate"),
            },
        )
        summary: dict[str, Any] = {
            "label": label,
            "mode": "patch_replay_audit",
            "strategy": "patch_replay_audit",
            "started_at": started_at.isoformat(),
            "evaluation_context": evaluation_context,
            "baseline_evaluation_context": evaluation_context,
            "evidence_protocol": _build_evidence_protocol(
                strategy="patch_replay_audit",
                evaluation_context=evaluation_context,
            ),
            "rerun_plan": _build_rerun_plan(strategy="patch_replay_audit"),
            "baseline": {
                "train_rate": baseline_train_rate,
                "validation_rate": baseline_validation_rate,
                "test_rate": baseline_test_rate,
                "full_pass_rate": baseline_full.get("overall", {}).get("pass_rate"),
                "comparison_metrics": baseline_comparison_metrics,
            },
            "baseline_checkpoint_path": str(baseline_checkpoint),
            "audited_patch_path": str(patch_path),
            "accepted": 0,
            "rejected": 0,
            "skipped": 0,
            "passes": [],
            "stop_reason": "patch_replay_audited",
        }
        self._persist_optimization_summary(summary)
        self.artifacts.record_iteration(
            {
                "phase": "patch_replay_audit_started",
                "label": label,
                "patch_path": str(patch_path),
                "payload_count": len(patch_payloads),
                "baseline_checkpoint_path": str(baseline_checkpoint),
            }
        )

        current_config = assemble_config(self.artifacts.space_dir)
        for index, payload in enumerate(patch_payloads, start=1):
            payload = dict(payload)
            payload.setdefault("candidate_mode", "artifact_patch_v2")
            payload.setdefault("candidate_name", f"{label}_patch_{index:02d}")
            current_config, patch_summary = self._apply_artifact_patch_payload(
                payload=payload,
                current_config=current_config,
            )
            pass_entry = {
                "index": index,
                "candidate_name": patch_summary.get("candidate_name"),
                "candidate_mode": patch_summary.get("candidate_mode"),
                "outcome": "accepted" if patch_summary.get("changed") else "rejected",
                "reason": patch_summary.get("reason"),
                "file_edit_count": patch_summary.get("file_edit_count"),
                "candidate_summary": patch_summary,
            }
            summary["passes"].append(pass_entry)
            if not patch_summary.get("changed"):
                summary["rejected"] += 1
                self._persist_optimization_summary(summary)
                self.artifacts.record_iteration(
                    {
                        "phase": "patch_replay_payload_rejected",
                        "label": label,
                        "index": index,
                        "summary": pass_entry,
                    }
                )
                raise WorkspaceFlowError(
                    "Patch replay payload was rejected before benchmark audit: "
                    f"{patch_summary.get('candidate_name')}"
                )
            summary["accepted"] += 1
            self.artifacts.record_iteration(
                {
                    "phase": "patch_replay_payload_applied",
                    "label": label,
                    "index": index,
                    "summary": pass_entry,
                }
            )
            self._persist_optimization_summary(summary)

        patched_checkpoint = self.artifacts.create_checkpoint(
            label=f"{label}_patched_state",
            metadata={
                "label": label,
                "audit_patch_path": str(patch_path),
                "accepted_payloads": summary["accepted"],
            },
        )
        summary["best_checkpoint_path"] = str(patched_checkpoint)
        summary["best_train_rate"] = baseline_train_rate
        summary["best_validation_rate"] = baseline_validation_rate
        clone_info = self.push()
        self.artifacts.record_iteration(
            {
                "phase": "patch_replay_audit_pushed",
                "label": label,
                "clone_space_id": clone_info.space_id,
                "patched_checkpoint_path": str(patched_checkpoint),
            }
        )

        final_summary = self.run_full_benchmark(label=f"{label}.final", baseline=False)
        completed_at = _dt.now()
        summary["completed_at"] = completed_at.isoformat()
        summary["elapsed_seconds"] = round((completed_at - started_at).total_seconds(), 1)
        summary["final"] = final_summary
        summary["final_evaluation_context"] = (
            final_summary.get("evaluation_context")
            if isinstance(final_summary, dict) and isinstance(final_summary.get("evaluation_context"), dict)
            else evaluation_context
        )
        summary["final_comparison_metrics"] = (
            final_summary.get("overall", {}).get("comparison_metrics")
            if isinstance(final_summary, dict)
            else None
        )
        protocol_preflight = self._write_protocol_preflight(
            mode="patch_replay_audit",
            strategy="patch_replay_audit",
            summary=final_summary,
        )
        summary["protocol_preflight"] = protocol_preflight
        terminal_selection = _terminal_result_selection(
            optimization_summary=summary,
            final_summary=final_summary,
            protocol_preflight=protocol_preflight,
            noise_model=self._build_terminal_noise_model(summary, final_summary),
            transfer_decisions=getattr(self, "_robustness_transfer_decisions", None),
            greedy_best_ever=getattr(self, "_greedy_best_ever_promotion", False),
        )
        summary["terminal_selection"] = terminal_selection
        self.artifacts.write_json_path(
            self.artifacts.terminal_selection_path,
            terminal_selection,
        )
        self.artifacts.record_iteration(
            {
                "phase": "terminal_result_selection",
                "label": label,
                "selection": terminal_selection,
            }
        )
        generalization_audit = _generalization_audit(
            optimization_summary=summary,
            final_summary=final_summary,
        )
        summary["generalization_audit"] = generalization_audit
        self.artifacts.write_json_path(
            self.artifacts.generalization_audit_path,
            generalization_audit,
        )
        self.artifacts.record_iteration(
            {
                "phase": "generalization_audit",
                "label": label,
                "audit": generalization_audit,
            }
        )
        summary["last_stop_reason"] = summary.get("stop_reason")
        self.artifacts.record_iteration(
            {
                "phase": "patch_replay_audit_completed",
                "label": label,
                "summary": {
                    "stop_reason": summary["stop_reason"],
                    "final_full_pass_rate": final_summary.get("overall", {}).get("pass_rate"),
                    "terminal_selection_status": terminal_selection.get("status"),
                    "generalization_audit_status": generalization_audit.get("status"),
                },
            }
        )
        self._persist_optimization_summary(summary)
        return summary

    def curate_and_gate(
        self,
        *,
        modes: list[str],
        label: str = "curate_loop",
    ) -> dict[str, Any]:
        reference_train_run = self._load_reference_train_run()
        reference_train_rate = reference_train_run.pass_rate if reference_train_run is not None else None
        reference_validation_summary = self._load_validation_summary()
        reference_validation_rate = (
            float(reference_validation_summary["pass_rate"])
            if isinstance(reference_validation_summary, dict) and reference_validation_summary.get("pass_rate") is not None
            else None
        )
        gating_active = reference_validation_rate is not None
        result = self._run_candidate_sweep(
            specs=self._curation_candidate_specs(modes),
            label=label,
            summary_path=self.artifacts.curation_summary_path,
            initial_reference_train_rate=reference_train_rate,
            initial_reference_validation_rate=reference_validation_rate,
            gating_active=gating_active,
            metadata={
                "modes_tried": modes,
            },
        )
        mapped_result = {
            "label": result["label"],
            "modes_tried": modes,
            "accepted": result["accepted"],
            "rejected": result["rejected"],
            "skipped": result["skipped"],
            "gating_active": gating_active,
            "passes": [
                {
                    "mode": entry["name"],
                    "outcome": entry["outcome"],
                    "reason": entry.get("reason"),
                    "train_rate": entry.get("train_rate"),
                    "validation_rate": entry.get("validation_rate"),
                    "rejection_class": entry.get("rejection_class"),
                    "gate_details": entry.get("gate_details"),
                }
                for entry in result.get("passes", [])
            ],
        }
        self.artifacts.write_json_path(self.artifacts.curation_summary_path, mapped_result)
        return mapped_result

    def _resolve_query_history_warehouse_id(self, explicit_warehouse_id: str | None) -> str | None:
        if explicit_warehouse_id:
            return explicit_warehouse_id
        split_meta = self._load_split_meta()
        workspace_warehouse_id = split_meta.get("warehouse_id")
        if isinstance(workspace_warehouse_id, str) and workspace_warehouse_id.strip():
            return workspace_warehouse_id
        return discover_best_warehouse_id(self.genie_service.wc)

    def collect_query_history(
        self,
        *,
        label: str,
        warehouse_id: str | None = None,
        days: int = 30,
        limit: int = 200,
    ) -> dict[str, Any]:
        resolved_warehouse_id = self._resolve_query_history_warehouse_id(warehouse_id)
        clone_space_id = self.require_clone_id()
        clone_meta = self.artifacts.load_clone_meta() or {}
        source_space_id = str(clone_meta.get("source_space_id") or "")
        space_ids = [source_space_id, clone_space_id]

        if not resolved_warehouse_id:
            summary = {
                "status": "unavailable",
                "message": "No accessible SQL warehouse was found for query-history collection.",
                "warehouse_id": None,
                "space_ids": [space_id for space_id in space_ids if space_id],
                "days": days,
                "query_count": 0,
                "successful_query_count": 0,
                "failed_query_count": 0,
                "unique_query_family_count": 0,
                "feedback_thumbs_up": 0,
                "feedback_thumbs_down": 0,
                "recurring_success_patterns": [],
                "recurring_failure_patterns": [],
            }
            recommendations_payload = {
                "status": "unavailable",
                "message": summary["message"],
                "recommended_trusted_asset_candidates": [],
                "recommended_benchmark_candidates": [],
                "noisy_one_off_queries": 0,
                "repeated_query_families": 0,
            }
            self.artifacts.write_json_path(self.artifacts.query_history_summary_path, summary)
            self.artifacts.write_json_path(self.artifacts.query_history_path, {"entries": []})
            self.artifacts.write_json_path(
                self.artifacts.query_history_recommendations_path,
                recommendations_payload,
            )
            self.artifacts.record_iteration(
                {
                    "phase": "query_history",
                    "label": label,
                    "summary": summary,
                }
            )
            return summary

        collector = QueryHistoryCollector(self.genie_service.wc, resolved_warehouse_id)
        entries, summary = collector.collect(
            space_ids=space_ids,
            days=days,
            limit=limit,
        )
        summary_payload = summary.to_dict()
        recommendations_payload = build_query_history_recommendations(
            entries,
            summary,
            benchmark_questions=self._load_all_benchmark_questions(),
        ).to_dict()
        self.artifacts.write_json_path(
            self.artifacts.query_history_path,
            {
                "entries": [entry.__dict__ for entry in entries],
            },
        )
        self.artifacts.write_json_path(self.artifacts.query_history_summary_path, summary_payload)
        self.artifacts.write_json_path(
            self.artifacts.query_history_recommendations_path,
            recommendations_payload,
        )
        self.artifacts.record_iteration(
            {
                "phase": "query_history",
                "label": label,
                "summary": {
                    "status": summary.status,
                    "warehouse_id": summary.warehouse_id,
                    "query_count": summary.query_count,
                    "successful_query_count": summary.successful_query_count,
                    "failed_query_count": summary.failed_query_count,
                    "feedback_thumbs_up": summary.feedback_thumbs_up,
                    "feedback_thumbs_down": summary.feedback_thumbs_down,
                },
            }
        )
        return summary_payload

    def _current_space_diagnostics(self) -> SpaceDiagnostics | None:
        try:
            config = assemble_config(self.artifacts.space_dir)
        except Exception:
            return None

        diagnostics = analyze_space_config(config)
        self.artifacts.write_json_path(
            self.artifacts.space_diagnostics_path,
            diagnostics.to_dict(),
        )
        return diagnostics

    def _load_train_questions(self) -> list[BenchmarkQuestion]:
        return self.exporter.load_questions_from_file(self.artifacts.train_set_path)

    def _load_split_meta(self) -> dict[str, Any]:
        payload = _load_json_dict(self.artifacts.split_meta_path)
        if payload is None:
            raise WorkspaceFlowError("Workspace split metadata is missing. Run `maxgenie export` first.")
        return payload

    def _load_blind_ids(self, path: Path, label: str) -> list[str]:
        if not path.exists():
            return []
        payload = self.artifacts.read_json_path(path)
        if not isinstance(payload, dict):
            raise WorkspaceFlowError(f"Invalid {label} payload.")
        return [str(value) for value in payload.get("ids", [])]

    def _load_validation_ids(self) -> list[str]:
        return self._load_blind_ids(self.artifacts.validation_set_path, "validation_set.json")

    def _load_dev_ids(self) -> list[str]:
        return self._load_validation_ids()

    def _load_test_ids(self) -> list[str]:
        return self._load_blind_ids(self.artifacts.test_set_path, "test_set.json")

    def _load_validation_summary(self) -> dict[str, Any] | None:
        return _load_json_dict(self.artifacts.latest_validation_summary_path)

    def _load_dev_summary(self) -> dict[str, Any] | None:
        return self._load_validation_summary()

    def _load_validation_folds(self) -> dict[str, Any] | None:
        return _load_json_dict(self.artifacts.validation_folds_path)

    def _load_curation_summary(self) -> dict[str, Any] | None:
        return _load_json_dict(self.artifacts.curation_summary_path)

    def _load_query_history_summary(self) -> dict[str, Any] | None:
        return _load_json_dict(self.artifacts.query_history_summary_path)

    def _load_query_history_recommendations(self) -> dict[str, Any] | None:
        return _load_json_dict(self.artifacts.query_history_recommendations_path)

    def _load_all_benchmark_questions(self) -> list[BenchmarkQuestion]:
        payload = self.artifacts.load_benchmark_payload()
        if not isinstance(payload, dict):
            return []

        rows = payload.get("questions")
        if not isinstance(rows, list):
            return []

        questions: list[BenchmarkQuestion] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question_id = row.get("id")
            question_text = row.get("question")
            if not question_id or not question_text:
                continue
            questions.append(
                BenchmarkQuestion(
                    id=str(question_id),
                    question=str(question_text),
                    expected_sql=(
                        str(row["expected_sql"])
                        if row.get("expected_sql") is not None
                        else None
                    ),
                )
            )
        return questions

    def _load_fresh_full_summary(
        self,
        *,
        latest_train: BenchmarkRun | None,
        latest_dev_summary: dict[str, Any] | None,
        latest_test_pass_rate: float | None,
    ) -> dict[str, Any] | None:
        latest_full = _load_json_dict(self.artifacts.latest_full_summary_path)
        latest_full_mtime = _artifact_mtime(self.artifacts.latest_full_summary_path)
        component_mtime = _latest_component_mtime(
            [
                self.artifacts.latest_train_path,
                self.artifacts.latest_dev_summary_path,
                self.artifacts.latest_test_score_path,
            ]
        )
        if (
            isinstance(latest_full, dict)
            and latest_full_mtime is not None
            and (component_mtime is None or latest_full_mtime >= component_mtime)
        ):
            return latest_full

        derived = _derived_full_summary(
            latest_train=latest_train,
            latest_dev_summary=latest_dev_summary,
            latest_test_pass_rate=latest_test_pass_rate,
            latest_test_question_count=len(self._load_test_ids()),
        )
        if derived is not None:
            self.artifacts.write_json_path(self.artifacts.latest_full_summary_path, derived)
        return derived

    def _resolve_blind_questions(
        self,
        clone_config: GenieSpaceConfig,
        *,
        ids: list[str],
        label: str,
    ) -> list[BenchmarkQuestion]:
        blind_ids = set(ids)
        if not blind_ids:
            return []
        question_map = {
            question.id: question
            for question in extract_benchmark_questions_from_config(clone_config)
        }
        missing_ids = sorted(blind_ids - question_map.keys())
        if missing_ids:
            raise WorkspaceFlowError(
                f"Clone is missing benchmark questions for {label}: {', '.join(missing_ids)}"
            )
        return [question_map[question_id] for question_id in ids]

    def _resolve_validation_questions(self, clone_config: GenieSpaceConfig) -> list[BenchmarkQuestion]:
        return self._resolve_blind_questions(
            clone_config,
            ids=self._load_validation_ids(),
            label="validation ids",
        )

    def _resolve_dev_questions(self, clone_config: GenieSpaceConfig) -> list[BenchmarkQuestion]:
        return self._resolve_validation_questions(clone_config)

    def _resolve_test_questions(self, clone_config: GenieSpaceConfig) -> list[BenchmarkQuestion]:
        return self._resolve_blind_questions(
            clone_config,
            ids=self._load_test_ids(),
            label="held-out ids",
        )

    def _load_reference_train_run(self) -> BenchmarkRun | None:
        latest = _load_run(self.artifacts.latest_train_path)
        if latest is not None:
            return latest
        return _load_run(self.artifacts.baseline_train_path)

    def _ordered_train_questions(self) -> list[BenchmarkQuestion]:
        questions = self._load_train_questions()
        previous = self._load_reference_train_run()
        if previous is None:
            return questions

        question_map = {question.id: question for question in questions}
        failing_ids = [
            result.question_id
            for result in previous.results
            if result.status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}
            and result.question_id in question_map
        ]
        passing_ids = [
            result.question_id
            for result in previous.results
            if result.status == BenchmarkStatus.PASSED and result.question_id in question_map
        ]
        seen = set(failing_ids) | set(passing_ids)
        remaining_ids = [question.id for question in questions if question.id not in seen]
        ordered_ids = failing_ids + passing_ids + remaining_ids
        return [question_map[question_id] for question_id in ordered_ids]

    def _write_train_subset_definition(
        self,
        *,
        subset_name: str,
        questions: list[BenchmarkQuestion],
    ) -> dict[str, Any]:
        payload = {
            "name": subset_name,
            "question_count": len(questions),
            "ids": [question.id for question in questions],
            "questions": [_question_payload(question) for question in questions],
        }
        self.artifacts.write_json_path(self.artifacts.subset_path(subset_name), payload)
        return payload

    def _split_train_working_sets(
        self,
        *,
        questions: list[BenchmarkQuestion],
        reference_run: BenchmarkRun | None,
    ) -> tuple[list[BenchmarkQuestion], list[BenchmarkQuestion]]:
        question_map = {question.id: question for question in questions}
        known_failure_ids: list[str] = []
        if reference_run is not None:
            known_failure_ids = [
                result.question_id
                for result in reference_run.results
                if result.status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}
                and result.question_id in question_map
            ]

        known_failure_questions = [question_map[question_id] for question_id in known_failure_ids]
        known_failure_id_set = set(known_failure_ids)
        core_pass_guard_questions = [
            question
            for question in questions
            if question.id not in known_failure_id_set
        ]

        self._write_train_subset_definition(
            subset_name="known_failures",
            questions=known_failure_questions,
        )
        self._write_train_subset_definition(
            subset_name="core_pass_guard",
            questions=core_pass_guard_questions,
        )
        return known_failure_questions, core_pass_guard_questions

    def _run_train_subset_benchmark(
        self,
        *,
        label: str,
        subset_name: str,
        clone_space_id: str,
        config_hash: str,
        questions: list[BenchmarkQuestion],
        min_pass_rate: float | None = None,
    ) -> tuple[BenchmarkRun, dict[str, Any]]:
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_train_subset_start",
                "label": label,
                "subset_name": subset_name,
                "clone_space_id": clone_space_id,
                "question_count": len(questions),
                "question_ids": [question.id for question in questions],
                "min_pass_rate": min_pass_rate,
            }
        )
        run = self._run_benchmarks(
            space_id=clone_space_id,
            questions=questions,
            config_hash=config_hash,
            min_pass_rate=min_pass_rate,
            label=label,
            benchmark_phase="benchmark_train_subset",
        )
        effective_clone_space_id = run.space_id or clone_space_id
        payload = {
            **run.to_dict(),
            "subset_name": subset_name,
            "ordering": [question.id for question in questions],
            "config_hash": config_hash,
        }
        self.artifacts.write_history_snapshot("train", label, payload)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_train_subset",
                "label": label,
                "subset_name": subset_name,
                "clone_space_id": effective_clone_space_id,
                "requested_clone_space_id": (
                    clone_space_id if effective_clone_space_id != clone_space_id else None
                ),
                "summary": _run_summary_with_issue_classes(run),
                "question_ids": [question.id for question in questions],
                "early_stop_reason": (
                    "min_pass_rate_unreachable" if len(run.results) < len(questions) else None
                ),
            }
        )
        return run, payload

    def _run_benchmarks(
        self,
        *,
        space_id: str,
        questions: list[BenchmarkQuestion],
        config_hash: str,
        min_pass_rate: float | None = None,
        label: str | None = None,
        benchmark_phase: str = "benchmark_questions",
        sample_salt: str | None = None,
    ) -> BenchmarkRun:
        cache = BenchmarkCache.load(self.artifacts.benchmark_cache_path)
        cache_hash = _benchmark_cache_hash(
            config_hash=config_hash,
            match_threshold=self.match_threshold,
            comparator=self.comparator,
        )
        # Multi-sample repeats must be independent Genie draws, not cache hits.
        # Salting the cache key per sample keeps each repeat individually
        # resumable while preventing later samples from collapsing onto the
        # first sample's cached result. With sample_salt=None (the default,
        # single-sample path) the cache key is unchanged.
        if sample_salt:
            cache_hash = f"{cache_hash}::{sample_salt}"
        cache_save_every = 1
        started_at = time.monotonic()
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_questions_start",
                "label": label,
                "benchmark_phase": benchmark_phase,
                "clone_space_id": space_id,
                "question_count": len(questions),
                "question_ids": [question.id for question in questions],
                "cache_save_every": cache_save_every,
                "min_pass_rate": min_pass_rate,
            }
        )
        try:
            run = self.benchmark_service.run_benchmarks(
                space_id=space_id,
                questions=questions,
                delay_between_questions=self.delay_seconds,
                fail_fast=False,
                match_threshold=self.match_threshold,
                cache=cache,
                config_hash=cache_hash,
                cache_save_path=self.artifacts.benchmark_cache_path,
                cache_save_every=cache_save_every,
                result_comparator=self.comparator,
                min_pass_rate=min_pass_rate,
                require_result_based=self._require_result_based,
            )
        except Exception as exc:
            self.artifacts.record_iteration(
                {
                    "phase": "benchmark_questions_error",
                    "label": label,
                    "benchmark_phase": benchmark_phase,
                    "clone_space_id": space_id,
                    "question_count": len(questions),
                    "duration_seconds": round(time.monotonic() - started_at, 2),
                    "issue_class": _benchmark_issue_class(
                        BenchmarkResult(
                            question_id="benchmark_run",
                            question="benchmark_run",
                            status=BenchmarkStatus.ERROR,
                            error_message=str(exc),
                        )
                    ),
                    "error_message": str(exc),
                }
            )
            raise
        if _has_missing_space_error_results(run):
            recovery_checkpoint = self.artifacts.create_recovery_checkpoint(
                label=f"{label or benchmark_phase}.missing_clone",
                metadata={
                    "reason": "benchmark_missing_clone",
                    "label": label,
                    "benchmark_phase": benchmark_phase,
                    "requested_clone_space_id": space_id,
                    "question_ids": [question.id for question in questions],
                    "config_hash": config_hash,
                    "benchmark_cache_hash": cache_hash,
                },
            )
            self.artifacts.record_iteration(
                {
                    "phase": "benchmark_recovery_checkpoint_created",
                    "label": label,
                    "benchmark_phase": benchmark_phase,
                    "clone_space_id": space_id,
                    "checkpoint_path": str(recovery_checkpoint),
                }
            )
            recovered_info = self.recover_managed_clone_from_workspace(
                reason="benchmark_missing_clone",
                restore_best_checkpoint=False,
                restore_checkpoint_path=recovery_checkpoint,
                fallback_warehouse_id=assemble_config(self.artifacts.space_dir).warehouse_id,
            )
            for result in run.results:
                if result.error_message and _is_missing_space_error(RuntimeError(result.error_message)):
                    cache.delete(result.question_id, cache_hash)
            cache.save(self.artifacts.benchmark_cache_path)
            run = self.benchmark_service.run_benchmarks(
                space_id=recovered_info.space_id,
                questions=questions,
                delay_between_questions=self.delay_seconds,
                fail_fast=False,
                match_threshold=self.match_threshold,
                cache=cache,
                config_hash=cache_hash,
                cache_save_path=self.artifacts.benchmark_cache_path,
                cache_save_every=cache_save_every,
                result_comparator=self.comparator,
                min_pass_rate=min_pass_rate,
                require_result_based=self._require_result_based,
            )
        cache.save(self.artifacts.benchmark_cache_path)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_questions_complete",
                "label": label,
                "benchmark_phase": benchmark_phase,
                "clone_space_id": run.space_id or space_id,
                "requested_clone_space_id": space_id if run.space_id != space_id else None,
                "question_count": len(questions),
                "evaluated_question_count": len(run.results),
                "duration_seconds": round(time.monotonic() - started_at, 2),
                "summary": _run_summary_with_issue_classes(run),
                "early_stop_reason": (
                    "min_pass_rate_unreachable" if len(run.results) < len(questions) else None
                ),
                "min_pass_rate": min_pass_rate,
            }
        )
        return run

    def _run_benchmarks_multi(
        self,
        *,
        space_id: str,
        questions: list[BenchmarkQuestion],
        config_hash: str,
        k_samples: int = 1,
        min_pass_rate: float | None = None,
        label: str | None = None,
        benchmark_phase: str = "benchmark_questions",
    ) -> tuple[BenchmarkRun, list[BenchmarkRun], SampleAggregate]:
        """Run the question set ``k_samples`` times and aggregate the repeats.

        Each repeat uses a distinct cache salt so it is an independent Genie draw
        (not a cache hit) while remaining individually resumable. The first
        sample is the representative run reused for the comparability hashes and
        artifact writes, preserving the frozen-seed contract. With
        ``k_samples == 1`` this is exactly one ``_run_benchmarks`` call and the
        returned aggregate has ``sample_std == 0`` and ``n == 1``.
        """
        k = max(int(k_samples), 1)
        runs: list[BenchmarkRun] = []
        for sample_index in range(1, k + 1):
            salt = None if k == 1 else f"ms{sample_index:02d}"
            sample_label = (
                label if (k == 1 or not label) else f"{label}.sample{sample_index:02d}"
            )
            run = self._run_benchmarks(
                space_id=space_id,
                questions=questions,
                config_hash=config_hash,
                min_pass_rate=min_pass_rate,
                label=sample_label,
                benchmark_phase=benchmark_phase,
                sample_salt=salt,
            )
            runs.append(run)
        aggregate = aggregate_samples(runs)
        if k > 1:
            self.artifacts.record_iteration(
                {
                    "phase": "benchmark_multi_sample_complete",
                    "label": label,
                    "benchmark_phase": benchmark_phase,
                    "clone_space_id": runs[0].space_id or space_id,
                    "k_samples": k,
                    "aggregate": aggregate.as_dict(),
                }
            )
        return runs[0], runs, aggregate

    def _build_terminal_noise_model(
        self,
        optimization_summary: dict[str, Any],
        final_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Assemble a noise model for the terminal-selection gate, or ``None``.

        Sigma sources, combined by taking the maximum (the conservative MDD):
        * an explicit ``self._terminal_noise_sigma_override`` set from a measured
          noise study (e.g. the ``measure-noise`` command), and
        * live multi-sample aggregates captured during this run's baseline and
          final population benchmarks (``self._full_sample_aggregates``).

        Returns ``None`` when no positive sigma is available, in which case the
        gate keeps its epsilon behaviour and single-sample runs are unchanged.
        """
        aggregates = getattr(self, "_full_sample_aggregates", {}) or {}
        # Contract: exactly one population label per run contains "baseline" (e.g.
        # "population_baseline") and is the run baseline; the most recent other label
        # is the final candidate. If that contract were ever broken by storing two
        # baseline-labelled aggregates, the first wins (guarded below) and the last
        # non-baseline aggregate wins for the final — both deterministic by insertion
        # order, but callers must not rely on storing multiple baselines per run.
        baseline_agg = None
        final_agg = None
        for key, agg in aggregates.items():
            if "baseline" in key:
                if baseline_agg is None:
                    baseline_agg = agg
            else:
                final_agg = agg

        sigma_candidates: list[float] = []
        override = getattr(self, "_terminal_noise_sigma_override", None)
        override_active = override is not None and float(override) > 0
        if override_active:
            sigma_candidates.append(float(override))
        if baseline_agg is not None and baseline_agg.n > 1:
            sigma_candidates.append(float(baseline_agg.sample_std))
        if final_agg is not None and final_agg.n > 1:
            sigma_candidates.append(float(final_agg.sample_std))
        sigma_candidates = [s for s in sigma_candidates if s > 0]
        if not sigma_candidates:
            return None
        sigma_full = max(sigma_candidates)

        baseline = optimization_summary.get("baseline")
        baseline = baseline if isinstance(baseline, dict) else {}
        baseline_full_mean = (
            baseline_agg.mean_pass_rate
            if baseline_agg is not None
            else _optional_float(baseline.get("full_pass_rate"))
        )
        final_full_mean = (
            final_agg.mean_pass_rate
            if final_agg is not None
            else _summary_split_pass_rate(final_summary, "overall")
        )
        baseline_n = (
            baseline_agg.n
            if baseline_agg is not None
            else getattr(self, "_terminal_noise_baseline_n", None)
        )
        candidate_n = final_agg.n if final_agg is not None else 1
        return {
            "sigma_full": sigma_full,
            "baseline_full_mean": baseline_full_mean,
            "final_full_mean": final_full_mean,
            "baseline_holdout_mean": _optional_float(baseline.get("test_rate")),
            "final_holdout_mean": _summary_split_pass_rate(final_summary, "test"),
            "baseline_n": baseline_n,
            "candidate_n": candidate_n,
            "holdout_epsilon": _TERMINAL_SELECTION_EPSILON,
            "z": 2.0,
            "sigma_source": "override" if override_active else "live_samples",
        }

    def _write_train_artifacts(
        self,
        *,
        payload: dict[str, Any],
        baseline: bool,
        persist_latest: bool,
    ) -> None:
        if baseline:
            self.artifacts.write_json_path(self.artifacts.baseline_train_path, payload)
            self.artifacts.write_json_path(self.artifacts.latest_train_path, payload)
            return
        if persist_latest:
            self.artifacts.write_json_path(self.artifacts.latest_train_path, payload)

    def _write_scored_summary_artifacts(
        self,
        *,
        summary: dict[str, Any],
        baseline: bool,
        persist_latest: bool,
        baseline_path: Path,
        latest_path: Path,
        baseline_summary_path: Path | None,
        latest_summary_path: Path | None,
    ) -> None:
        if baseline:
            self.artifacts.write_score_path(baseline_path, float(summary["pass_rate"]))
            self.artifacts.write_score_path(latest_path, float(summary["pass_rate"]))
            if baseline_summary_path is not None:
                self.artifacts.write_json_path(baseline_summary_path, summary)
            if latest_summary_path is not None:
                self.artifacts.write_json_path(latest_summary_path, summary)
            return
        if persist_latest:
            self.artifacts.write_score_path(latest_path, float(summary["pass_rate"]))
            if latest_summary_path is not None:
                self.artifacts.write_json_path(latest_summary_path, summary)

    def _preflight_with_recovery(self) -> tuple[str, GenieSpaceConfig]:
        """Preflight the clone, retrying on transient auth failures."""
        import logging
        import time as _time
        logger = logging.getLogger(__name__)
        clone_space_id = self.require_clone_id()
        last_error: Exception | None = None

        # Retry up to 3 times with backoff — most "does not exist" errors
        # are transient auth/session issues, not actual deletions.
        for attempt in range(3):
            try:
                _, clone_config = self.preflight_space(clone_space_id)
                return clone_space_id, clone_config
            except Exception as exc:
                last_error = exc
                error_msg = str(exc)
                is_transient = "does not exist" in error_msg or "not accessible" in error_msg
                if not is_transient:
                    raise
                if attempt < 2:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        "Clone %s preflight failed (attempt %d/3): %s — retrying in %ds",
                        clone_space_id, attempt + 1, exc, wait,
                    )
                    _time.sleep(wait)
        if last_error is not None and _is_missing_space_chain(last_error):
            logger.warning(
                "Clone %s preflight failed after 3 attempts with a missing-space error. Recovering the clone.",
                clone_space_id,
            )
            recovered_info = self.recover_managed_clone_from_workspace(
                reason=f"preflight_missing_clone: {last_error}",
                restore_best_checkpoint=True,
            )
            _, clone_config = self.preflight_space(recovered_info.space_id)
            return recovered_info.space_id, clone_config
        if last_error is not None:
            logger.error(
                "Clone %s preflight failed after 3 attempts: %s — "
                "this is likely an auth/session expiration. "
                "Consider using a PAT (personal access token) instead of OAuth.",
                clone_space_id, last_error,
            )
            raise WorkspaceFlowError(
                f"Clone `{clone_space_id}` is not accessible after 3 retries. "
                f"Last error: {last_error}. "
                "If this is a long-running optimization, use a personal access token "
                "(PAT) instead of OAuth to avoid session expiration."
            ) from last_error
        raise WorkspaceFlowError("Unreachable")

    def _run_train_benchmark_internal(
        self,
        *,
        label: str,
        baseline: bool = False,
        persist_latest: bool = True,
        min_pass_rate: float | None = None,
    ) -> tuple[BenchmarkRun, dict[str, Any]]:
        clone_space_id, clone_config = self._preflight_with_recovery()
        questions = self._ordered_train_questions()
        config_hash = self.artifacts.config_hash(clone_config.to_serialized_space())
        previous_run = None if baseline else self._load_reference_train_run()
        known_failure_questions, core_pass_guard_questions = self._split_train_working_sets(
            questions=questions,
            reference_run=previous_run,
        )

        subset_runs: list[BenchmarkRun] = []
        subset_payloads: list[dict[str, Any]] = []
        if known_failure_questions:
            failure_run, failure_payload = self._run_train_subset_benchmark(
                label=f"{label}.known_failures",
                subset_name="known_failures",
                clone_space_id=clone_space_id,
                config_hash=config_hash,
                questions=known_failure_questions,
            )
            subset_runs.append(failure_run)
            clone_space_id = failure_run.space_id or clone_space_id
            subset_payloads.append(
                {
                    "name": "known_failures",
                    "question_count": len(known_failure_questions),
                    "summary": _run_summary_with_issue_classes(failure_run),
                    "question_ids": [question.id for question in known_failure_questions],
                    "clone_space_id": clone_space_id,
                }
            )

            if (
                not baseline
                and previous_run is not None
                and min_pass_rate is not None
                and previous_run.pass_rate >= min_pass_rate
                and failure_run.passed == 0
            ):
                payload = {
                    **failure_payload,
                    "diff": {
                        "fixed": [],
                        "regressed": [],
                        "still_failing": [question.id for question in known_failure_questions],
                        "new_failures": [],
                        "known_failures_after": [question.id for question in known_failure_questions],
                    },
                    "config_hash": config_hash,
                    "subsets": subset_payloads,
                    "early_stop_reason": "known_failures_unimproved",
                }
                self.artifacts.write_history_snapshot("train", label, payload)
                self.artifacts.record_iteration(
                    {
                        "phase": "benchmark_train",
                        "label": label,
                        "clone_space_id": clone_space_id,
                        "summary": _run_summary_with_issue_classes(failure_run),
                        "diff": payload["diff"],
                        "subsets": subset_payloads,
                        "early_stop_reason": "known_failures_unimproved",
                    }
                )
                return failure_run, payload

        if core_pass_guard_questions:
            core_pass_guard_min_pass_rate = 100.0 if previous_run is not None else None
            guard_run, _ = self._run_train_subset_benchmark(
                label=f"{label}.core_pass_guard",
                subset_name="core_pass_guard",
                clone_space_id=clone_space_id,
                config_hash=config_hash,
                questions=core_pass_guard_questions,
                min_pass_rate=core_pass_guard_min_pass_rate,
            )
            subset_runs.append(guard_run)
            clone_space_id = guard_run.space_id or clone_space_id
            subset_payloads.append(
                {
                    "name": "core_pass_guard",
                    "question_count": len(core_pass_guard_questions),
                    "summary": _run_summary_with_issue_classes(guard_run),
                    "question_ids": [question.id for question in core_pass_guard_questions],
                    "clone_space_id": clone_space_id,
                }
            )

        if not subset_runs:
            run = BenchmarkRun(space_id=clone_space_id)
            run.completed_at = run.started_at
        elif len(subset_runs) == 1 and len(subset_runs[0].results) == len(questions):
            run = subset_runs[0]
        else:
            run = _merge_benchmark_runs(
                ordered_questions=questions,
                runs=subset_runs,
                carry_forward_run=previous_run,
            )

        diff = _build_train_diff(previous_run, run)
        payload = {
            **run.to_dict(),
            "ordering": [question.id for question in questions],
            "diff": diff,
            "config_hash": config_hash,
            "subsets": subset_payloads,
        }
        self._write_train_artifacts(
            payload=payload,
            baseline=baseline,
            persist_latest=persist_latest,
        )
        self.artifacts.write_history_snapshot("train", label, payload)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_train",
                "label": label,
                "clone_space_id": clone_space_id,
                "summary": _run_summary_with_issue_classes(run),
                "diff": diff,
                "subsets": subset_payloads,
            }
        )
        return run, payload

    def run_train_benchmark(
        self,
        *,
        label: str,
        baseline: bool = False,
        persist_latest: bool = True,
    ) -> BenchmarkRun:
        run, _ = self._run_train_benchmark_internal(
            label=label,
            baseline=baseline,
            persist_latest=persist_latest,
        )
        return run

    def _run_blind_benchmark(
        self,
        *,
        label: str,
        baseline: bool,
        persist_latest: bool,
        clone_config: GenieSpaceConfig,
        questions: list[BenchmarkQuestion],
        baseline_path: Path,
        latest_path: Path,
        baseline_summary_path: Path | None,
        latest_summary_path: Path | None,
        snapshot_kind: str,
        iteration_phase: str,
        validation_strategy: str = "blind_split",
        min_pass_rate: float | None = None,
    ) -> dict[str, Any]:
        clone_space_id = self.require_clone_id()
        if not questions:
            summary = {
                "question_count": 0,
                "evaluated_question_count": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "failed_ids": [],
                "error_ids": [],
                "skipped_ids": [],
                "pass_rate": 0.0,
                "validation_strategy": validation_strategy,
                "comparison_metrics": _aggregate_comparison_metrics([]),
                "evaluation_context": _benchmark_evaluation_context(
                    self.comparator,
                    match_threshold=self.match_threshold,
                ),
            }
        else:
            config_hash = self.artifacts.config_hash(clone_config.to_serialized_space())
            self.artifacts.record_iteration(
                {
                    "phase": f"{iteration_phase}_start",
                    "label": label,
                    "clone_space_id": clone_space_id,
                    "question_count": len(questions),
                    "question_ids": [question.id for question in questions],
                    "validation_strategy": validation_strategy,
                }
            )
            run = self._run_benchmarks(
                space_id=clone_space_id,
                questions=questions,
                config_hash=config_hash,
                min_pass_rate=min_pass_rate,
                label=label,
                benchmark_phase=iteration_phase,
            )
            clone_space_id = run.space_id or clone_space_id
            evaluated_ids = {result.question_id for result in run.results}
            skipped_ids = [question.id for question in questions if question.id not in evaluated_ids]
            issue_details = _benchmark_issue_details(run.results)
            pass_rate = (run.passed / len(questions) * 100) if questions else 0.0
            summary = {
                "question_count": len(questions),
                "evaluated_question_count": len(run.results),
                "passed": run.passed,
                "failed": run.failed,
                "errors": run.errors,
                "skipped": len(skipped_ids),
                "failed_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.FAILED
                ],
                "error_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.ERROR
                ],
                "skipped_ids": skipped_ids,
                "pass_rate": round(pass_rate, 2),
                "validation_strategy": validation_strategy,
                "comparison_metrics": _comparison_metrics_from_run(run),
                "evaluation_context": _benchmark_evaluation_context(
                    self.comparator,
                    match_threshold=self.match_threshold,
                ),
            }
            if skipped_ids:
                summary["early_stop_reason"] = "min_pass_rate_unreachable"
                summary["min_pass_rate"] = min_pass_rate
                summary["best_possible_pass_rate"] = round(
                    ((run.passed + len(skipped_ids)) / len(questions) * 100) if questions else 0.0,
                    2,
                )
            if issue_details:
                summary["issue_class_counts"] = _benchmark_issue_class_counts(run.results)
                summary["issue_details"] = issue_details

        self._write_scored_summary_artifacts(
            summary=summary,
            baseline=baseline,
            persist_latest=persist_latest,
            baseline_path=baseline_path,
            latest_path=latest_path,
            baseline_summary_path=baseline_summary_path,
            latest_summary_path=latest_summary_path,
        )
        self.artifacts.write_history_snapshot(snapshot_kind, label, summary)
        self.artifacts.record_iteration(
            {
                "phase": iteration_phase,
                "label": label,
                "clone_space_id": clone_space_id,
                "summary": summary,
            }
        )
        return summary

    def _run_validation_fold_benchmark(
        self,
        *,
        label: str,
        baseline: bool,
        persist_latest: bool,
        clone_config: GenieSpaceConfig,
        folds_payload: dict[str, Any],
        min_pass_rate: float | None = None,
    ) -> dict[str, Any]:
        clone_space_id = self.require_clone_id()
        folds = folds_payload.get("folds", [])
        if not isinstance(folds, list):
            raise WorkspaceFlowError("Invalid validation_folds.json payload.")

        config_hash = self.artifacts.config_hash(clone_config.to_serialized_space())
        fold_questions: list[tuple[dict[str, Any], list[BenchmarkQuestion]]] = []
        for fold in folds:
            if not isinstance(fold, dict):
                raise WorkspaceFlowError("Invalid validation fold entry.")
            validation_ids = [str(value) for value in fold.get("validation_ids", [])]
            questions = self._resolve_blind_questions(
                clone_config,
                ids=validation_ids,
                label="validation fold ids",
            )
            fold_questions.append((fold, questions))

        required_passes = _required_pass_count(
            min_pass_rate=min_pass_rate,
            question_count=sum(len(questions) for _, questions in fold_questions),
        )
        fold_summaries: list[dict[str, Any]] = []
        total_passed = 0
        total_failed = 0
        total_errors = 0
        evaluated_questions = 0
        skipped_questions = 0
        failed_ids: list[str] = []
        error_ids: list[str] = []
        skipped_ids: list[str] = []
        early_stop_reason: str | None = None

        for fold_index, (fold, questions) in enumerate(fold_questions):
            validation_ids = [str(value) for value in fold.get("validation_ids", [])]
            remaining_after_fold = sum(
                len(later_questions)
                for _, later_questions in fold_questions[fold_index + 1:]
            )
            needed_from_fold = (
                max(0, required_passes - total_passed - remaining_after_fold)
                if required_passes is not None
                else None
            )
            if needed_from_fold is not None and needed_from_fold > len(questions):
                early_stop_reason = "validation_improvement_impossible"
                skipped_ids.extend(
                    question.id
                    for _, remaining_questions in fold_questions[fold_index:]
                    for question in remaining_questions
                )
                skipped_questions += sum(
                    len(remaining_questions)
                    for _, remaining_questions in fold_questions[fold_index:]
                )
                self.artifacts.record_iteration(
                    {
                        "phase": "benchmark_validation_fold_skipped",
                        "label": label,
                        "clone_space_id": clone_space_id,
                        "fold_index": int(fold.get("fold_index", len(fold_summaries) + 1)),
                        "reason": early_stop_reason,
                        "required_passes": required_passes,
                        "passed_so_far": total_passed,
                        "remaining_question_count": skipped_questions,
                        "min_pass_rate": min_pass_rate,
                    }
                )
                break
            fold_min_pass_rate = (
                (needed_from_fold / len(questions) * 100)
                if needed_from_fold is not None and questions and needed_from_fold > 0
                else None
            )
            self.artifacts.record_iteration(
                {
                    "phase": "benchmark_validation_fold_start",
                    "label": label,
                    "clone_space_id": clone_space_id,
                    "fold_index": int(fold.get("fold_index", len(fold_summaries) + 1)),
                    "question_count": len(questions),
                    "question_ids": [question.id for question in questions],
                    "validation_ids": validation_ids,
                    "min_pass_rate": fold_min_pass_rate,
                    "overall_min_pass_rate": min_pass_rate,
                    "required_passes": required_passes,
                }
            )
            run = self._run_benchmarks(
                space_id=clone_space_id,
                questions=questions,
                config_hash=config_hash,
                min_pass_rate=fold_min_pass_rate,
                label=f"{label}.fold_{int(fold.get('fold_index', len(fold_summaries) + 1)):02d}",
                benchmark_phase="benchmark_validation_fold",
            )
            clone_space_id = run.space_id or clone_space_id
            evaluated_ids = {result.question_id for result in run.results}
            fold_skipped_ids = [
                question.id for question in questions if question.id not in evaluated_ids
            ]
            issue_details = _benchmark_issue_details(run.results)
            fold_pass_rate = (run.passed / len(questions) * 100) if questions else 0.0
            fold_summary = {
                "fold_index": int(fold.get("fold_index", len(fold_summaries) + 1)),
                "question_count": len(questions),
                "evaluated_question_count": len(run.results),
                "passed": run.passed,
                "failed": run.failed,
                "errors": run.errors,
                "skipped": len(fold_skipped_ids),
                "pass_rate": round(fold_pass_rate, 2),
                "validation_ids": validation_ids,
                "failed_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.FAILED
                ],
                "error_ids": [
                    result.question_id
                    for result in run.results
                    if result.status == BenchmarkStatus.ERROR
                ],
                "skipped_ids": fold_skipped_ids,
                "comparison_metrics": _comparison_metrics_from_run(run),
            }
            if fold_skipped_ids:
                fold_summary["early_stop_reason"] = "min_pass_rate_unreachable"
                fold_summary["min_pass_rate"] = fold_min_pass_rate
                fold_summary["best_possible_pass_rate"] = round(
                    ((run.passed + len(fold_skipped_ids)) / len(questions) * 100) if questions else 0.0,
                    2,
                )
            if issue_details:
                fold_summary["issue_class_counts"] = _benchmark_issue_class_counts(run.results)
                fold_summary["issue_details"] = issue_details
            fold_summaries.append(fold_summary)
            evaluated_questions += len(run.results)
            total_passed += run.passed
            total_failed += run.failed
            total_errors += run.errors
            skipped_questions += len(fold_skipped_ids)
            failed_ids.extend(fold_summary["failed_ids"])
            error_ids.extend(fold_summary["error_ids"])
            skipped_ids.extend(fold_skipped_ids)

            if fold_skipped_ids:
                early_stop_reason = "validation_improvement_impossible"
                skipped_ids.extend(
                    question.id
                    for _, remaining_questions in fold_questions[fold_index + 1:]
                    for question in remaining_questions
                )
                skipped_questions += remaining_after_fold
                break

            best_possible_passes = total_passed + remaining_after_fold
            if required_passes is not None and best_possible_passes < required_passes:
                early_stop_reason = "validation_improvement_impossible"
                skipped_ids.extend(
                    question.id
                    for _, remaining_questions in fold_questions[fold_index + 1:]
                    for question in remaining_questions
                )
                skipped_questions += remaining_after_fold
                self.artifacts.record_iteration(
                    {
                        "phase": "benchmark_validation_fold_early_stop",
                        "label": label,
                        "clone_space_id": clone_space_id,
                        "fold_index": int(fold.get("fold_index", len(fold_summaries))),
                        "reason": early_stop_reason,
                        "required_passes": required_passes,
                        "passed_so_far": total_passed,
                        "best_possible_passes": best_possible_passes,
                        "remaining_question_count": remaining_after_fold,
                        "min_pass_rate": min_pass_rate,
                    }
                )
                break

        denominator = sum(len(questions) for _, questions in fold_questions)
        pass_rate = (total_passed / denominator * 100) if denominator else 0.0
        fold_pass_rates = [float(fold["pass_rate"]) for fold in fold_summaries]
        summary = {
            "validation_strategy": "small_sample_kfold",
            "fold_count": len(fold_summaries),
            "visible_question_count": int(folds_payload.get("visible_question_count", denominator)),
            "question_count": denominator,
            "evaluated_question_count": evaluated_questions,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "skipped": skipped_questions,
            "failed_ids": failed_ids,
            "error_ids": error_ids,
            "skipped_ids": skipped_ids,
            "pass_rate": round(pass_rate, 2),
            "mean_fold_pass_rate": round(sum(fold_pass_rates) / len(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "min_fold_pass_rate": round(min(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "max_fold_pass_rate": round(max(fold_pass_rates), 2) if fold_pass_rates else 0.0,
            "folds": fold_summaries,
            "comparison_metrics": _aggregate_comparison_metrics(
                [
                    fold_summary.get("comparison_metrics", {})
                    for fold_summary in fold_summaries
                    if isinstance(fold_summary, dict)
                ]
            ),
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
        }
        if early_stop_reason is not None:
            summary["early_stop_reason"] = early_stop_reason
            summary["min_pass_rate"] = min_pass_rate
            summary["required_passes"] = required_passes
            summary["best_possible_pass_rate"] = round(
                ((total_passed + skipped_questions) / denominator * 100) if denominator else 0.0,
                2,
            )
        issue_details = [
            detail
            for fold_summary in fold_summaries
            for detail in fold_summary.get("issue_details", [])
            if isinstance(detail, dict)
        ]
        if issue_details:
            issue_counts: dict[str, int] = {}
            for detail in issue_details:
                issue_class = str(detail["issue_class"])
                issue_counts[issue_class] = issue_counts.get(issue_class, 0) + 1
            summary["issue_class_counts"] = dict(sorted(issue_counts.items()))
            summary["issue_details"] = issue_details

        self._write_scored_summary_artifacts(
            summary=summary,
            baseline=baseline,
            persist_latest=persist_latest,
            baseline_path=self.artifacts.baseline_validation_score_path,
            latest_path=self.artifacts.latest_validation_score_path,
            baseline_summary_path=self.artifacts.baseline_validation_summary_path,
            latest_summary_path=self.artifacts.latest_validation_summary_path,
        )
        self.artifacts.write_history_snapshot("validation", label, summary)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_validation",
                "label": label,
                "clone_space_id": clone_space_id,
                "summary": summary,
            }
        )
        return summary

    def run_validation_benchmark(
        self,
        *,
        label: str,
        baseline: bool = False,
        persist_latest: bool = True,
        min_pass_rate: float | None = None,
    ) -> dict[str, Any]:
        clone_space_id, clone_config = self._preflight_with_recovery()
        clone_config = _config_with_local_benchmarks(self.artifacts, clone_config)
        folds_payload = self._load_validation_folds()
        if folds_payload and folds_payload.get("validation_strategy") == "small_sample_kfold":
            return self._run_validation_fold_benchmark(
                label=label,
                baseline=baseline,
                persist_latest=persist_latest,
                clone_config=clone_config,
                folds_payload=folds_payload,
                min_pass_rate=min_pass_rate,
            )
        questions = self._resolve_validation_questions(clone_config)
        return self._run_blind_benchmark(
            label=label,
            baseline=baseline,
            persist_latest=persist_latest,
            clone_config=clone_config,
            questions=questions,
            baseline_path=self.artifacts.baseline_validation_score_path,
            latest_path=self.artifacts.latest_validation_score_path,
            baseline_summary_path=self.artifacts.baseline_validation_summary_path,
            latest_summary_path=self.artifacts.latest_validation_summary_path,
            snapshot_kind="validation",
            iteration_phase="benchmark_validation",
            validation_strategy="blind_split",
            min_pass_rate=min_pass_rate,
        )

    def run_dev_benchmark(
        self,
        *,
        label: str,
        baseline: bool = False,
        persist_latest: bool = True,
    ) -> dict[str, Any]:
        return self.run_validation_benchmark(
            label=label,
            baseline=baseline,
            persist_latest=persist_latest,
        )

    def run_test_benchmark(self, *, label: str, baseline: bool = False) -> dict[str, Any]:
        clone_space_id, clone_config = self._preflight_with_recovery()
        clone_config = _config_with_local_benchmarks(self.artifacts, clone_config)
        questions = self._resolve_test_questions(clone_config)
        return self._run_blind_benchmark(
            label=label,
            baseline=baseline,
            persist_latest=True,
            clone_config=clone_config,
            questions=questions,
            baseline_path=self.artifacts.baseline_test_score_path,
            latest_path=self.artifacts.latest_test_score_path,
            baseline_summary_path=None,
            latest_summary_path=None,
            snapshot_kind="test",
            iteration_phase="benchmark_test",
            validation_strategy="final_holdout",
        )

    def precompute_answer_key(self, questions: Sequence[Any]) -> int:
        """Execute and persist gold result rows for every benchmark expected SQL once.

        Ensures the answer key holds gold rows for ALL questions (visible AND holdout) up
        front, independent of whether the baseline *generated* SQL succeeds. This matters
        because the dominant failure family is unbound-parameter SQL whose generated query
        raises before ``compare()`` can lazily cache the expected rows — so without this,
        those questions' gold rows (and the visible-failure expected-row preview fed to the
        proposer) would be missing. Holdout gold rows live only in this key (a gold store)
        and are never placed in proposer context.

        Returns the number of newly executed (previously uncached) expected SQLs. A no-op
        when no result comparator is wired (string-only mode has no answer key).
        """
        comparator = self.comparator
        if comparator is None:
            return 0
        precompute = getattr(comparator, "precompute_expected", None)
        if not callable(precompute):
            return 0
        expected_sqls = [
            str(question.expected_sql)
            for question in (questions or [])
            if getattr(question, "expected_sql", None)
        ]
        if not expected_sqls:
            return 0
        computed = precompute(expected_sqls)
        self.artifacts.record_iteration(
            {
                "phase": "answer_key_precompute",
                "total_questions": len(list(questions or [])),
                "distinct_expected_sqls": len(set(expected_sqls)),
                "newly_computed": computed,
            }
        )
        return computed

    def run_full_benchmark(self, *, label: str, baseline: bool = False) -> dict[str, Any]:
        train_run = self.run_train_benchmark(label=f"{label}.train", baseline=baseline)
        validation_summary = self.run_validation_benchmark(label=f"{label}.validation", baseline=baseline)
        test_summary = self.run_test_benchmark(label=f"{label}.test", baseline=baseline)
        if validation_summary.get("validation_strategy") == "small_sample_kfold":
            overall_total = int(validation_summary["question_count"]) + int(test_summary["question_count"])
            overall_passed = int(validation_summary["passed"]) + int(test_summary["passed"])
            overall_failed = int(validation_summary["failed"]) + int(test_summary["failed"])
            overall_errors = int(validation_summary["errors"]) + int(test_summary["errors"])
            overall_basis = "visible_kfold_plus_holdout"
            overall_comparison_metrics = _aggregate_comparison_metrics(
                [
                    validation_summary.get("comparison_metrics", {}),
                    test_summary.get("comparison_metrics", {}),
                ]
            )
        else:
            overall_total = (
                train_run.total + int(validation_summary["question_count"]) + int(test_summary["question_count"])
            )
            overall_passed = (
                train_run.passed + int(validation_summary["passed"]) + int(test_summary["passed"])
            )
            overall_failed = (
                train_run.failed + int(validation_summary["failed"]) + int(test_summary["failed"])
            )
            overall_errors = (
                train_run.errors + int(validation_summary["errors"]) + int(test_summary["errors"])
            )
            overall_basis = "train_plus_validation_plus_holdout"
            overall_comparison_metrics = _aggregate_comparison_metrics(
                [
                    _comparison_metrics_from_run(train_run),
                    validation_summary.get("comparison_metrics", {}),
                    test_summary.get("comparison_metrics", {}),
                ]
            )
        overall_rate = (overall_passed / overall_total * 100) if overall_total else 0.0

        summary = {
            "label": label,
            "clone_space_id": self.require_clone_id(),
            "overall_basis": overall_basis,
            "evaluation_context": _benchmark_evaluation_context(
                self.comparator,
                match_threshold=self.match_threshold,
            ),
            "overall": {
                "total": overall_total,
                "passed": overall_passed,
                "failed": overall_failed,
                "errors": overall_errors,
                "pass_rate": round(overall_rate, 2),
                "comparison_metrics": overall_comparison_metrics,
            },
            "train": _summarize_run(train_run),
            "validation": validation_summary,
            "test": test_summary,
        }
        self.artifacts.write_json_path(self.artifacts.latest_full_summary_path, summary)
        if baseline:
            self.artifacts.write_json_path(self.artifacts.baseline_full_summary_path, summary)
        self.artifacts.write_history_snapshot("full", label, summary)
        self.artifacts.record_iteration(
            {
                "phase": "benchmark_full",
                "label": label,
                "summary": summary,
            }
        )
        self._refresh_clone_title_with_full_score(summary)
        return summary

    def status(self) -> WorkspaceStatus:
        clone_meta = self.artifacts.load_clone_meta()
        workspace_meta = self.artifacts.load_workspace_meta() or {}
        clone_space_id = str(clone_meta["clone_space_id"]) if clone_meta else None
        source_space_id = str(clone_meta["source_space_id"]) if clone_meta else None
        input_space_id = (
            str(clone_meta["input_space_id"])
            if clone_meta and clone_meta.get("input_space_id")
            else (
                str(workspace_meta.get("input_space_id"))
                if workspace_meta.get("input_space_id") is not None
                else None
            )
        )
        managed_source_space_id = (
            str(clone_meta["managed_source_space_id"])
            if clone_meta and clone_meta.get("managed_source_space_id")
            else (
                str(workspace_meta.get("managed_source_space_id"))
                if workspace_meta.get("managed_source_space_id") is not None
                else None
            )
        )
        clone_accessible = False
        if clone_space_id:
            try:
                self.genie_service.get(clone_space_id)
                clone_accessible = True
            except Exception:
                clone_accessible = False

        baseline_train = _load_run(self.artifacts.baseline_train_path)
        latest_train = _load_run(self.artifacts.latest_train_path)
        latest_validation_summary = self._load_dev_summary()
        curation_summary = self._load_curation_summary()
        optimization_summary = _load_json_dict(self.artifacts.optimization_summary_path)
        query_history_summary = self._load_query_history_summary()
        query_history_recommendations = self._load_query_history_recommendations()
        git_state = workspace_meta.get("git_state", {})
        git_end_state = git_state.get("end") if isinstance(git_state, dict) else None
        git_start_state = git_state.get("start") if isinstance(git_state, dict) else None
        git_selected_state = git_end_state if isinstance(git_end_state, dict) else git_start_state
        latest_validation_pass_rate = self.artifacts.read_score_path(self.artifacts.latest_dev_score_path)
        latest_test_pass_rate = self.artifacts.read_score_path(self.artifacts.latest_test_score_path)
        latest_full = self._load_fresh_full_summary(
            latest_train=latest_train,
            latest_dev_summary=latest_validation_summary,
            latest_test_pass_rate=latest_test_pass_rate,
        )
        diagnostics = self._current_space_diagnostics()
        described_column_coverage_percent = None
        if diagnostics is not None:
            described_column_coverage_percent = _coverage_percent(
                diagnostics.metadata_coverage.described_columns,
                diagnostics.metadata_coverage.total_columns,
            )

        return WorkspaceStatus(
            input_space_id=input_space_id,
            source_space_id=source_space_id,
            managed_source_space_id=managed_source_space_id,
            clone_space_id=clone_space_id,
            clone_accessible=clone_accessible,
            baseline_train_pass_rate=baseline_train.pass_rate if baseline_train else None,
            latest_train_pass_rate=latest_train.pass_rate if latest_train else None,
            latest_validation_pass_rate=latest_validation_pass_rate,
            latest_test_pass_rate=latest_test_pass_rate,
            latest_full_pass_rate=(
                float(latest_full["overall"]["pass_rate"])
                if isinstance(latest_full, dict)
                else None
            ),
            latest_full_comparison_mode=(
                str(
                    latest_full.get("overall", {})
                    .get("comparison_metrics", {})
                    .get("comparison_mode")
                )
                if isinstance(latest_full, dict)
                and latest_full.get("overall", {})
                .get("comparison_metrics", {})
                .get("comparison_mode")
                is not None
                else None
            ),
            blocked_candidate_ids=self.detect_blocked_candidates(),
            external_table_reference_count=(
                len(diagnostics.external_table_references) if diagnostics is not None else None
            ),
            entity_matching_candidate_count=(
                len(diagnostics.entity_matching_candidates) if diagnostics is not None else None
            ),
            format_assistance_candidate_count=(
                len(diagnostics.format_assistance_candidates) if diagnostics is not None else None
            ),
            described_column_coverage_percent=described_column_coverage_percent,
            curation_candidate_status=(
                (
                    f"loop: {int(curation_summary.get('accepted', 0))} accepted"
                    f" / {int(curation_summary.get('rejected', 0))} rejected"
                )
                if isinstance(curation_summary, dict) and _is_gated_curation_summary(curation_summary)
                else (
                    str(curation_summary.get("candidate_status"))
                    if isinstance(curation_summary, dict) and curation_summary.get("candidate_status") is not None
                    else None
                )
            ),
            curation_change_count=(
                int(curation_summary.get("accepted", 0))
                if isinstance(curation_summary, dict) and _is_gated_curation_summary(curation_summary)
                else (
                    len(curation_summary.get("changes", []))
                    if isinstance(curation_summary, dict)
                    else None
                )
            ),
            hidden_column_count=(
                None
                if isinstance(curation_summary, dict) and _is_gated_curation_summary(curation_summary)
                else (
                    int(curation_summary.get("columns_hidden", 0))
                    if isinstance(curation_summary, dict)
                    else None
                )
            ),
            query_history_status=(
                str(query_history_summary.get("status"))
                if isinstance(query_history_summary, dict) and query_history_summary.get("status") is not None
                else None
            ),
            query_history_query_count=(
                int(query_history_summary.get("query_count", 0))
                if isinstance(query_history_summary, dict)
                else None
            ),
            query_history_feedback_thumbs_down=(
                int(query_history_summary.get("feedback_thumbs_down", 0))
                if isinstance(query_history_summary, dict)
                else None
            ),
            query_history_trusted_asset_candidates=(
                len(query_history_recommendations.get("recommended_trusted_asset_candidates", []))
                if isinstance(query_history_recommendations, dict)
                else None
            ),
            query_history_benchmark_candidates=(
                len(query_history_recommendations.get("recommended_benchmark_candidates", []))
                if isinstance(query_history_recommendations, dict)
                else None
            ),
            git_commit_sha=(
                str(git_selected_state.get("commit_sha"))
                if isinstance(git_selected_state, dict) and git_selected_state.get("commit_sha")
                else None
            ),
            git_dirty=(
                bool(git_selected_state.get("dirty"))
                if isinstance(git_selected_state, dict) and git_selected_state.get("dirty") is not None
                else None
            ),
            autonomous_strategy=(
                str(optimization_summary.get("strategy"))
                if isinstance(optimization_summary, dict) and optimization_summary.get("strategy") is not None
                else (
                    str(workspace_meta.get("autonomous_strategy"))
                    if workspace_meta.get("autonomous_strategy") is not None
                    else None
                )
            ),
            last_stop_reason=(
                str(optimization_summary.get("stop_reason"))
                if isinstance(optimization_summary, dict) and optimization_summary.get("stop_reason") is not None
                else None
            ),
        )

    def detect_blocked_candidates(self) -> list[str]:
        snapshots = sorted(self.artifacts.train_run_history_dir.glob("*.json"))
        if len(snapshots) < 3:
            return []

        last_three = snapshots[-3:]
        failure_sets: list[set[str]] = []
        for path in last_three:
            run = _load_run(path)
            if run is None:
                continue
            failure_sets.append(set(_ids_by_status(run, {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR})))
        if len(failure_sets) < 3:
            return []
        blocked = set.intersection(*failure_sets)
        return sorted(blocked)

    def write_comparable_run_manifest(
        self,
        *,
        latest_full: dict[str, Any] | None = None,
        workspace_meta: dict[str, Any] | None = None,
        clone_meta: dict[str, Any] | None = None,
        split_meta: dict[str, Any] | None = None,
    ) -> Path:
        latest_full = latest_full if isinstance(latest_full, dict) else {}
        workspace_meta = workspace_meta if isinstance(workspace_meta, dict) else (self.artifacts.load_workspace_meta() or {})
        clone_meta = clone_meta if isinstance(clone_meta, dict) else (self.artifacts.load_clone_meta() or {})
        split_meta = split_meta if isinstance(split_meta, dict) else (_load_json_dict(self.artifacts.split_meta_path) or {})

        overall = latest_full.get("overall") if isinstance(latest_full.get("overall"), dict) else {}
        comparison_metrics = (
            overall.get("comparison_metrics")
            if isinstance(overall.get("comparison_metrics"), dict)
            else {}
        )
        evaluation_context = (
            latest_full.get("evaluation_context")
            if isinstance(latest_full.get("evaluation_context"), dict)
            else {}
        )
        comparison_mode = (
            comparison_metrics.get("comparison_mode")
            or evaluation_context.get("preferred_comparison_mode")
            or ("result_based" if self.comparator is not None else "string_only")
        )
        expected_count = self._expected_protocol_benchmark_count(latest_full)
        scored_count = (
            comparison_metrics.get("scored_question_count")
            or overall.get("question_count")
            or overall.get("total")
        )
        result_based_count = comparison_metrics.get("result_based_question_count")
        git_state = workspace_meta.get("git_state", {}) if isinstance(workspace_meta.get("git_state"), dict) else {}
        git_end_state = git_state.get("end") if isinstance(git_state.get("end"), dict) else None
        git_start_state = git_state.get("start") if isinstance(git_state.get("start"), dict) else None
        git_selected_state = git_end_state or git_start_state or {}
        product_commit_sha = (
            workspace_meta.get("product_commit_sha")
            or git_selected_state.get("commit_sha")
        )
        product_branch = (
            workspace_meta.get("product_branch")
            or git_selected_state.get("branch")
        )
        product_dirty = workspace_meta.get("product_build_dirty")
        if product_dirty is None:
            product_dirty = git_selected_state.get("dirty")

        payload = {
            "schema_version": "maxgenie.comparable_run_manifest.v1",
            "source_space_id": split_meta.get("source_space_id") or clone_meta.get("input_space_id") or self.artifacts.space_id,
            "input_space_id": clone_meta.get("input_space_id") or workspace_meta.get("input_space_id"),
            "managed_source_space_id": clone_meta.get("managed_source_space_id") or workspace_meta.get("managed_source_space_id"),
            "clone_space_id": latest_full.get("clone_space_id") or clone_meta.get("clone_space_id") or workspace_meta.get("active_space_id"),
            "workspace_root": str(self.artifacts.workspace_dir),
            "product_commit_sha": product_commit_sha,
            "product_branch": product_branch,
            "product_build_dirty": product_dirty,
            "candidate_mode": workspace_meta.get("serving_candidate_mode"),
            "bootstrap_only": workspace_meta.get("bootstrap_only"),
            "strategy": workspace_meta.get("autonomous_strategy"),
            "serving_endpoint": workspace_meta.get("serving_endpoint"),
            "serving_model": workspace_meta.get("serving_model"),
            "serving_reasoning_effort": workspace_meta.get("serving_reasoning_effort"),
            "warehouse_id": evaluation_context.get("warehouse_id") or split_meta.get("warehouse_id"),
            "evaluator_mode": comparison_mode,
            "comparison_fingerprint": evaluation_context.get("comparison_fingerprint"),
            "match_threshold": evaluation_context.get("match_threshold"),
            "benchmark_payload_hash": _file_sha256(self.artifacts.benchmark_payload_path),
            "hidden_test_payload_hash": _file_sha256(self.artifacts.hidden_test_payload_path),
            "split_metadata_hash": _file_sha256(self.artifacts.split_meta_path),
            "split_strategy": split_meta.get("split_strategy"),
            "validation_strategy": split_meta.get("validation_strategy"),
            "split_seed": split_meta.get("split_seed"),
            "fresh_baseline": workspace_meta.get("fresh_baseline"),
            "observed_benchmark_reuse_policy": workspace_meta.get("observed_benchmark_reuse_policy"),
            "split_counts": {
                "train": _json_question_count(self.artifacts.train_set_path),
                "validation": _json_question_count(self.artifacts.validation_set_path),
                "holdout": _json_question_count(self.artifacts.test_set_path),
            },
            "evaluation_counts": {
                "expected": expected_count,
                "scored": scored_count,
                "result_based": result_based_count,
                "string_based": comparison_metrics.get("string_based_question_count"),
            },
            "scores": {
                "full": overall.get("pass_rate"),
                "train": (latest_full.get("train") or {}).get("pass_rate") if isinstance(latest_full.get("train"), dict) else None,
                "validation": (latest_full.get("validation") or {}).get("pass_rate") if isinstance(latest_full.get("validation"), dict) else None,
                "holdout": (latest_full.get("test") or {}).get("pass_rate") if isinstance(latest_full.get("test"), dict) else None,
            },
            "coverage": {
                "scored": scored_count,
                "expected": expected_count,
                "result_based": result_based_count,
                "comparison_method_counts": comparison_metrics.get("comparison_method_counts"),
            },
        }
        return self.artifacts.write_json_path(self.artifacts.comparable_run_manifest_path, payload)

    def _write_report_notebook(self, report_markdown: str) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": report_markdown.splitlines(keepends=True),
                }
            ],
            "metadata": {
                "language_info": {
                    "name": "python",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path = self.artifacts.final_report_notebook_path
        notebook_path_text = str(notebook_path)
        if not notebook_path_text.startswith("/Workspace/"):
            self.artifacts.write_json_path(notebook_path, notebook)
            return

        api_path = notebook_path_text[len("/Workspace") :]
        content = base64.b64encode(
            json.dumps(notebook, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        try:
            self.genie_service.wc.workspace.import_(
                api_path,
                content=content,
                format=workspace_service.ImportFormat.JUPYTER,
                overwrite=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to import final report notebook",
                extra={"path": notebook_path_text, "error": str(exc)},
            )
            self.artifacts.write_json_path(
                self.artifacts.results_dir / "final_report_notebook_import_error.json",
                {
                    "path": notebook_path_text,
                    "error": str(exc),
                },
            )

    def write_report(self) -> Path:
        latest_train = _load_run(self.artifacts.latest_train_path)
        baseline_train = _load_run(self.artifacts.baseline_train_path)
        clone_meta = self.artifacts.load_clone_meta() or {}
        workspace_meta = self.artifacts.load_workspace_meta() or {}
        diagnostics = self._current_space_diagnostics()
        curation_summary = self._load_curation_summary()
        advisory_summary = _load_json_dict(self.artifacts.advisory_best_practices_path)
        optimization_summary = _load_json_dict(self.artifacts.optimization_summary_path)
        protocol_preflight = _load_json_dict(self.artifacts.protocol_preflight_path)
        visible_proxy_pressure = _load_json_dict(self.artifacts.visible_proxy_pressure_path)
        generalization_audit = _load_json_dict(self.artifacts.generalization_audit_path)
        if generalization_audit is None and isinstance(optimization_summary, dict):
            embedded_audit = optimization_summary.get("generalization_audit")
            generalization_audit = embedded_audit if isinstance(embedded_audit, dict) else None
        query_history_summary = self._load_query_history_summary()
        query_history_recommendations = self._load_query_history_recommendations()
        latest_validation_summary = self._load_dev_summary()
        baseline_validation_summary = _load_json_dict(self.artifacts.baseline_validation_summary_path) or {}
        baseline_validation_pass_rate = self.artifacts.read_score_path(self.artifacts.baseline_dev_score_path)
        latest_validation_pass_rate = self.artifacts.read_score_path(self.artifacts.latest_dev_score_path)
        baseline_test_pass_rate = self.artifacts.read_score_path(self.artifacts.baseline_test_score_path)
        latest_test_pass_rate = self.artifacts.read_score_path(self.artifacts.latest_test_score_path)
        baseline_full = _load_json_dict(self.artifacts.baseline_full_summary_path) or {}
        latest_full = self._load_fresh_full_summary(
            latest_train=latest_train,
            latest_dev_summary=latest_validation_summary,
            latest_test_pass_rate=latest_test_pass_rate,
        ) or {}
        observed_seed = (
            _load_json_dict(self.artifacts.results_dir / "observed_benchmark_seed.json")
            or _load_json_dict(self.artifacts.results_dir / "latest_observed_benchmark_seed.json")
            or {}
        )
        observed_context = _load_json_dict(
            self.artifacts.results_dir / "latest_observed_benchmark_context.json"
        ) or {}
        is_advisory = is_advisory_best_practices(advisory_summary) or is_advisory_best_practices(workspace_meta)
        if is_advisory:
            latest_train = None
            baseline_train = None
            latest_validation_summary = {}
            baseline_validation_summary = {}
            baseline_validation_pass_rate = None
            latest_validation_pass_rate = None
            baseline_test_pass_rate = None
            latest_test_pass_rate = None
            baseline_full = {}
            latest_full = {}
            observed_seed = {}
            observed_context = {}
            optimization_summary = None
            protocol_preflight = None
            visible_proxy_pressure = None
            generalization_audit = None
            query_history_summary = None
            query_history_recommendations = None
        blocked_notes = self.artifacts.read_text_path(self.artifacts.blocked_failures_path).strip() if self.artifacts.blocked_failures_path.exists() else ""
        iteration_log_path = self.artifacts.history_dir / "iteration_log.jsonl"
        iteration_count = 0
        push_count = 0
        if iteration_log_path.exists():
            with iteration_log_path.open(encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    iteration_count += 1
                    payload = json.loads(line)
                    if payload.get("phase") == "push":
                        push_count += 1

        full_pass_rate_text = (
            "n/a"
            if not latest_full
            else f"{latest_full.get('overall', {}).get('pass_rate', 0.0):.2f}%"
        )
        validation_pass_rate_text = (
            "n/a" if latest_validation_pass_rate is None else f"{latest_validation_pass_rate:.2f}%"
        )
        test_pass_rate_text = (
            "n/a" if latest_test_pass_rate is None else f"{latest_test_pass_rate:.2f}%"
        )
        validation_mode = (
            str(latest_validation_summary.get("validation_strategy", "blind_split"))
            if latest_validation_summary
            else "blind_split"
        )
        split_meta = _load_json_dict(self.artifacts.split_meta_path) or {}
        split_strategy = str(split_meta.get("split_strategy", "unknown"))
        validation_questions = (
            int(latest_validation_summary.get("question_count", 0))
            if latest_validation_summary
            else len(self._load_dev_ids())
        )
        validation_delta_text = "n/a"
        final_test_delta_text = "n/a"
        validation_delta_value: float | None = None
        final_test_delta_value: float | None = None
        if latest_train and latest_validation_pass_rate is not None:
            validation_delta_value = latest_train.pass_rate - latest_validation_pass_rate
            validation_delta_text = f"{validation_delta_value:+.2f}%"
        if latest_train and latest_test_pass_rate is not None:
            final_test_delta_value = latest_train.pass_rate - latest_test_pass_rate
            final_test_delta_text = f"{final_test_delta_value:+.2f}%"

        latest_full_comparison_metrics = (
            latest_full.get("overall", {}).get("comparison_metrics", {})
            if isinstance(latest_full, dict)
            and isinstance(latest_full.get("overall"), dict)
            and isinstance(latest_full.get("overall", {}).get("comparison_metrics"), dict)
            else {}
        )
        latest_full_evaluation_context: dict[str, Any] = {}
        if isinstance(latest_full, dict) and isinstance(latest_full.get("evaluation_context"), dict):
            latest_full_evaluation_context = latest_full["evaluation_context"]
        elif isinstance(optimization_summary, dict):
            fallback_evaluation_context = (
                optimization_summary.get("final_evaluation_context")
                or optimization_summary.get("baseline_evaluation_context")
            )
            if isinstance(fallback_evaluation_context, dict):
                latest_full_evaluation_context = fallback_evaluation_context
        latest_full_comparison_mode = (
            str(latest_full_comparison_metrics.get("comparison_mode"))
            if latest_full_comparison_metrics.get("comparison_mode") is not None
            else (
                str(latest_full_evaluation_context.get("preferred_comparison_mode"))
                if latest_full_evaluation_context.get("preferred_comparison_mode") is not None
                else None
            )
        )
        latest_full_comparison_mode_text = latest_full_comparison_mode or "n/a"
        train_pass_rate_text = f"{latest_train.pass_rate if latest_train else 0.0:.2f}%"
        if is_advisory:
            full_pass_rate_text = "n/a (no benchmark)"
            latest_full_comparison_mode_text = NOT_BENCHMARK_VERIFIED_LABEL
            train_pass_rate_text = "n/a (no benchmark)"
            validation_pass_rate_text = "n/a (no benchmark)"
            test_pass_rate_text = "n/a (no benchmark)"
            split_strategy = "no_benchmark"
            validation_mode = "no_benchmark"
        validation_mode_text = (
            "no benchmark"
            if validation_mode == "no_benchmark"
            else ("grouped k-fold" if validation_mode == "small_sample_kfold" else "blind split")
        )

        def _overall_summary_text(summary: dict[str, Any]) -> str:
            overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
            if not isinstance(overall, dict) or overall.get("total") is None:
                return "n/a"
            total = int(overall.get("total", 0) or 0)
            passed = int(overall.get("passed", 0) or 0)
            pass_rate = float(overall.get("pass_rate", 0.0) or 0.0)
            return f"{passed}/{total} ({pass_rate:.2f}%)"

        def _overall_pass_rate(summary: dict[str, Any]) -> float | None:
            overall = summary.get("overall", {}) if isinstance(summary, dict) else {}
            if not isinstance(overall, dict) or overall.get("pass_rate") is None:
                return None
            return float(overall.get("pass_rate", 0.0) or 0.0)

        def _optional_rate_text(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}%"

        def _rate_delta_text(before: float | None, after: float | None) -> str:
            return "n/a" if before is None or after is None else f"{after - before:+.2f}%"

        def _split_rate_text(
            *,
            train_rate: float | None,
            validation_rate: float | None,
            test_rate: float | None,
        ) -> str:
            return (
                f"train `{_optional_rate_text(train_rate)}`, "
                f"validation `{_optional_rate_text(validation_rate)}`, "
                f"test `{_optional_rate_text(test_rate)}`"
            )

        def _observed_count_text(payload: dict[str, Any]) -> str | None:
            counts = payload.get("assessment_counts") if isinstance(payload, dict) else None
            if not isinstance(counts, dict) or not counts:
                return None
            return ", ".join(
                f"{key}={int(value)}"
                for key, value in sorted(counts.items())
                if isinstance(value, (int, float))
            )

        def _observed_score_text(payload: dict[str, Any]) -> str | None:
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list):
                return None
            total = int(payload.get("result_count") or payload.get("question_count") or len(results) or 0)
            if total <= 0:
                return None
            passed = 0
            for result in results:
                if not isinstance(result, dict):
                    continue
                score = _optional_float(result.get("score"))
                status = str(result.get("status") or result.get("assessment") or "").lower()
                if score is not None and score >= 1.0:
                    passed += 1
                elif status in {"good", "passed", "pass"}:
                    passed += 1
            return f"{passed}/{total} ({(passed / total) * 100.0:.2f}%)"

        active_clone_id = str(clone_meta.get("clone_space_id") or self.require_clone_id())
        evaluated_clone_id = (
            str(latest_full.get("clone_space_id"))
            if isinstance(latest_full, dict) and latest_full.get("clone_space_id")
            else active_clone_id
        )
        clone_title = (
            str(workspace_meta.get("clone_title"))
            if workspace_meta.get("clone_title") is not None
            else None
        )
        clone_space_url = (
            str(workspace_meta.get("clone_space_url"))
            if workspace_meta.get("clone_space_url") is not None
            else _genie_space_url_from_host(workspace_meta.get("workspace_host"), active_clone_id)
        )
        clone_lines = [
            f"- Evaluated clone: `{evaluated_clone_id}`",
        ]
        if clone_space_url:
            clone_lines.insert(
                0,
                f"- Optimized space: [open optimized space]({clone_space_url})",
            )
        if clone_title:
            clone_lines.insert(
                1 if clone_space_url else 0,
                f"- Optimized space title: `{clone_title}`",
            )
        if active_clone_id != evaluated_clone_id:
            clone_lines.append(f"- Active clone after recovery: `{active_clone_id}`")
        if clone_meta.get("clone_parent_path"):
            clone_lines.append(f"- Clone parent path: `{clone_meta.get('clone_parent_path')}`")
            clone_lines.append(
                f"- Clone parent path source: `{clone_meta.get('clone_parent_path_source') or 'unknown'}`"
            )

        lines = [
            "# MaxGenie Optimization Report",
            "",
            *([f"**{NO_BENCHMARK_WARNING}**", ""] if is_advisory else []),
            "## Snapshot",
            "",
            f"- Input space: `{clone_meta.get('input_space_id') or 'unknown'}`",
            f"- Managed source space: `{clone_meta.get('managed_source_space_id') or clone_meta.get('source_space_id') or 'unknown'}`",
            *clone_lines,
            f"- Full pass rate: `{full_pass_rate_text}`",
            f"- Full comparison mode: `{latest_full_comparison_mode_text}`",
            f"- Train pass rate: `{train_pass_rate_text}`",
            f"- Validation pass rate: `{validation_pass_rate_text}`",
            f"- Test pass rate: `{test_pass_rate_text}`",
            f"- Split strategy: `{split_strategy}`",
            f"- Validation mode: `{validation_mode_text}`",
            *(
                [
                    f"- Evidence classification: `{ADVISORY_BEST_PRACTICES_CLASSIFICATION}`",
                    f"- Benchmark verification: `{NOT_BENCHMARK_VERIFIED_LABEL}`",
                    "- Auto-promotion: `disabled`",
                ]
                if is_advisory
                else []
            ),
            f"- Pushes: `{push_count}`",
            f"- Logged events: `{iteration_count}`",
            "",
        ]

        if is_advisory:
            advisory_warning = (
                str(advisory_summary.get("warning"))
                if isinstance(advisory_summary, dict) and advisory_summary.get("warning")
                else NO_BENCHMARK_WARNING
            )
            suggested_next_step = (
                str(advisory_summary.get("suggested_next_step"))
                if isinstance(advisory_summary, dict) and advisory_summary.get("suggested_next_step")
                else (
                    "Add owner-approved benchmark questions, then rerun MaxGenie under the "
                    "result-based protocol before treating any changes as verified."
                )
            )
            lines.extend(
                [
                    "## No Benchmark Advisory",
                    "",
                    f"- Warning: {advisory_warning}",
                    f"- Evidence classification: `{ADVISORY_BEST_PRACTICES_CLASSIFICATION}`",
                    f"- Benchmark verification: `{NOT_BENCHMARK_VERIFIED_LABEL}`",
                    "- Score: `n/a`; no benchmark score was computed or inferred.",
                    "- Auto-promotion: `disabled`",
                    f"- Suggested next step: {suggested_next_step}",
                    "",
                ]
            )

        initial_rate = _overall_pass_rate(baseline_full)
        final_rate = _overall_pass_rate(latest_full)
        if baseline_full or latest_full:
            train_delta_text = _rate_delta_text(
                baseline_train.pass_rate if baseline_train else None,
                latest_train.pass_rate if latest_train else None,
            )
            validation_split_delta_text = _rate_delta_text(
                baseline_validation_pass_rate,
                latest_validation_pass_rate,
            )
            holdout_delta_text = _rate_delta_text(baseline_test_pass_rate, latest_test_pass_rate)
            baseline_source = (
                baseline_full.get("baseline_source")
                or observed_seed.get("source")
                or "benchmark_run"
            )
            observed_reference = observed_seed or observed_context
            observed_score_text = _observed_score_text(observed_reference)
            observed_eval_run_id = observed_reference.get("eval_run_id") if isinstance(observed_reference, dict) else None
            lines.extend(
                [
                    "## Initial To Final",
                    "",
                    f"- Initial benchmark source: `{baseline_source}`",
                    (
                        f"- Observed benchmark eval run: `{observed_seed.get('eval_run_id')}`"
                        if observed_seed.get("eval_run_id")
                        else None
                    ),
                    (
                        f"- Existing Genie UI eval score: `{observed_score_text}`"
                        + (f" from eval run `{observed_eval_run_id}`" if observed_eval_run_id else "")
                        if observed_score_text and baseline_source != "latest_genie_eval_run"
                        else None
                    ),
                    f"- Initial full score: `{_overall_summary_text(baseline_full)}`",
                    f"- Final full score: `{_overall_summary_text(latest_full)}`",
                    (
                        f"- Full-score delta: `{final_rate - initial_rate:+.2f}%`"
                        if initial_rate is not None and final_rate is not None
                        else "- Full-score delta: `n/a`"
                    ),
                    "- Initial split rates: "
                    + _split_rate_text(
                        train_rate=baseline_train.pass_rate if baseline_train else None,
                        validation_rate=baseline_validation_pass_rate,
                        test_rate=baseline_test_pass_rate,
                    ),
                    "- Final split rates: "
                    + _split_rate_text(
                        train_rate=latest_train.pass_rate if latest_train else None,
                        validation_rate=latest_validation_pass_rate,
                        test_rate=latest_test_pass_rate,
                    ),
                    "- Split-rate deltas: "
                    f"train `{train_delta_text}`, "
                    f"validation `{validation_split_delta_text}`, "
                    f"holdout `{holdout_delta_text}`",
                    (
                        f"- Observed assessment counts: `{_observed_count_text(observed_seed)}`"
                        if _observed_count_text(observed_seed)
                        else None
                    ),
                    (
                        "- Comparison note: `no net improvement` means final versus the fresh "
                        "MaxGenie result-based baseline from this run; an existing Genie UI eval "
                        "can show a different score because it is a separate benchmark signal."
                        if baseline_source == "benchmark_run"
                        else None
                    ),
                    "",
                ]
            )
            lines = [line for line in lines if line is not None]

        git_state = workspace_meta.get("git_state", {}) if isinstance(workspace_meta, dict) else {}
        git_end_state = git_state.get("end") if isinstance(git_state, dict) else None
        git_start_state = git_state.get("start") if isinstance(git_state, dict) else None
        git_selected_state = git_end_state if isinstance(git_end_state, dict) else git_start_state
        if isinstance(git_selected_state, dict):
            lines.extend(
                [
                    "## Code Revision",
                    "",
                    f"- Commit SHA: `{git_selected_state.get('commit_sha') or 'unknown'}`",
                    f"- Branch: `{git_selected_state.get('branch') or 'detached'}`",
                    f"- Dirty worktree: `{'yes' if git_selected_state.get('dirty') else 'no'}`",
                    f"- Tracked files changed: `{len(git_selected_state.get('changed_files', []))}`",
                    "",
                ]
            )

        if isinstance(optimization_summary, dict):
            lines.extend(
                [
                    "## Optimization Loop",
                    "",
                    f"- Mode: `{optimization_summary.get('mode', 'manual')}`",
                    f"- Strategy: `{optimization_summary.get('strategy', workspace_meta.get('autonomous_strategy', 'unknown'))}`",
                    f"- Accepted candidates: `{int(optimization_summary.get('accepted', 0))}`",
                    f"- Rejected candidates: `{int(optimization_summary.get('rejected', 0))}`",
                    f"- Skipped candidates: `{int(optimization_summary.get('skipped', 0))}`",
                    f"- Stop reason: `{optimization_summary.get('stop_reason', 'unknown')}`",
                    f"- Sweeps completed: `{int(optimization_summary.get('sweeps_completed', 0) or 0)}`",
                    f"- Candidate attempts: `{int(optimization_summary.get('attempted_candidates', 0) or 0)}`",
                    "",
                ]
            )
            lines = [line for line in lines if line is not None]
            terminal_selection = optimization_summary.get("terminal_selection")
            if isinstance(terminal_selection, dict):
                reasons = terminal_selection.get("reasons")
                reason_text = (
                    ", ".join(str(reason) for reason in reasons)
                    if isinstance(reasons, list) and reasons
                    else "none"
                )
                lines.extend(
                    [
                        "## Terminal Selection",
                        "",
                        f"- Status: `{terminal_selection.get('status', 'unknown')}`",
                        f"- Reasons: `{reason_text}`",
                        f"- Selected checkpoint: `{terminal_selection.get('selected_checkpoint_path') or 'n/a'}`",
                        f"- Final holdout rate: `{_optional_rate_text(_optional_float(terminal_selection.get('final_holdout_rate')))}`",
                        "",
                    ]
                )

        if isinstance(optimization_summary, dict) or isinstance(curation_summary, dict):
            lines.extend(["## What Changed", ""])
            if isinstance(curation_summary, dict):
                changes = curation_summary.get("changes", [])
                change_count = len(changes) if isinstance(changes, list) else 0
                candidate_status = str(curation_summary.get("candidate_status", "prepared"))
                if _is_gated_curation_summary(curation_summary):
                    lines.append(
                        "- Benchmark-gated curation loop: "
                        f"`{int(curation_summary.get('accepted', 0))}` accepted, "
                        f"`{int(curation_summary.get('rejected', 0))}` rejected, "
                        f"`{int(curation_summary.get('skipped', 0))}` skipped."
                    )
                elif change_count:
                    lines.append(
                        "- Automatic curation prepared "
                        f"`{change_count}` metadata changes with status `{candidate_status}`."
                    )
            if isinstance(optimization_summary, dict):
                passes = optimization_summary.get("passes", [])
                accepted_passes = [
                    item
                    for item in passes
                    if isinstance(item, dict) and item.get("outcome") == "accepted"
                ] if isinstance(passes, list) else []
                rejected_passes = [
                    item
                    for item in passes
                    if isinstance(item, dict) and item.get("outcome") == "rejected"
                ] if isinstance(passes, list) else []
                if accepted_passes:
                    lines.append("- Accepted optimizer candidates:")
                    for item in accepted_passes[:5]:
                        name = item.get("name", "unknown")
                        family = item.get("family", "unknown")
                        train_rate = _optional_rate_text(
                            float(item["train_rate"]) if item.get("train_rate") is not None else None
                        )
                        validation_rate = _optional_rate_text(
                            float(item["validation_rate"])
                            if item.get("validation_rate") is not None
                            else None
                        )
                        reason = item.get("reason") or "benchmark gate accepted"
                        lines.append(
                            f"- `{name}` (`{family}`): train `{train_rate}`, "
                            f"validation `{validation_rate}`; {reason}"
                        )
                    if len(accepted_passes) > 5:
                        lines.append(f"- `{len(accepted_passes) - 5}` more accepted candidates omitted")
                else:
                    lines.append(
                        "- Optimizer candidates accepted: `0`; the best checkpoint stayed at the baseline state."
                    )
                lines.append(
                    f"- Optimizer candidates rejected: `{len(rejected_passes)}`; "
                    f"skipped: `{int(optimization_summary.get('skipped', 0) or 0)}`."
                )
            lines.append("")

        if latest_full_evaluation_context or latest_full_comparison_metrics:
            average_comparison_score = latest_full_comparison_metrics.get("average_comparison_score")
            average_comparison_score_text = (
                f"{float(average_comparison_score):.4f}"
                if average_comparison_score is not None
                else "n/a"
            )
            expected_question_count = self._expected_protocol_benchmark_count(latest_full)
            scored_question_count = int(
                latest_full_comparison_metrics.get("scored_question_count", 0) or 0
            )
            result_based_question_count = int(
                latest_full_comparison_metrics.get("result_based_question_count", 0) or 0
            )
            fallback_methods_text = _fallback_method_counts_text(
                latest_full_comparison_metrics.get("comparison_method_counts")
            )
            lines.extend(
                [
                    "## Benchmark Evaluation",
                    "",
                    f"- Preferred comparison mode: `{latest_full_evaluation_context.get('preferred_comparison_mode', 'n/a')}`",
                    f"- Actual full comparison mode: `{latest_full_comparison_mode_text}`",
                    f"- Comparison fingerprint: `{latest_full_evaluation_context.get('comparison_fingerprint') or 'n/a'}`",
                    f"- Warehouse used for result comparison: `{latest_full_evaluation_context.get('warehouse_id') or 'n/a'}`",
                    f"- Match threshold: `{latest_full_evaluation_context.get('match_threshold') if latest_full_evaluation_context.get('match_threshold') is not None else 'n/a'}`",
                    f"- Expected benchmark questions: `{expected_question_count}`",
                    f"- Benchmark coverage: `{scored_question_count}/{expected_question_count}` scored; `{result_based_question_count}/{expected_question_count}` result-based",
                    f"- Scored questions: `{scored_question_count}`",
                    f"- Average comparison score: `{average_comparison_score_text}`",
                    f"- Result-based questions: `{result_based_question_count}`",
                    f"- String-based questions: `{int(latest_full_comparison_metrics.get('string_based_question_count', 0) or 0)}`",
                    f"- Methods observed: `{_comparison_method_counts_text(latest_full_comparison_metrics.get('comparison_method_counts'))}`",
                    f"- String fallback methods: `{fallback_methods_text}`",
                    "",
                ]
            )
            if fallback_methods_text != "none":
                lines.extend(
                    [
                        "> Result-comparator failures scored by string fallback are diagnostic only and do not count as result-based leaderboard coverage.",
                        "",
                    ]
                )

        if is_advisory:
            lines.extend(
                [
                    "## Protocol Eligibility",
                    "",
                    f"- Status: `{NOT_BENCHMARK_VERIFIED_STATUS}`",
                    "- Promotion eligible: `no`",
                    f"- Evidence classification: `{ADVISORY_BEST_PRACTICES_CLASSIFICATION}`",
                    "- Blocking reasons: `no_benchmark`",
                    "- Warnings: `best_practice_heuristics_only`",
                    "",
                ]
            )

        if isinstance(protocol_preflight, dict):
            eligibility = protocol_preflight.get("promotion_eligibility")
            eligibility = eligibility if isinstance(eligibility, dict) else {}
            errors = protocol_preflight.get("errors")
            warnings = protocol_preflight.get("warnings")
            error_text = (
                ", ".join(
                    str(issue.get("code"))
                    for issue in errors
                    if isinstance(issue, dict) and issue.get("code")
                )
                if isinstance(errors, list) and errors
                else "none"
            )
            warning_text = (
                ", ".join(
                    str(issue.get("code"))
                    for issue in warnings
                    if isinstance(issue, dict) and issue.get("code")
                )
                if isinstance(warnings, list) and warnings
                else "none"
            )
            lines.extend(
                [
                    "## Protocol Eligibility",
                    "",
                    f"- Status: `{protocol_preflight.get('status', 'unknown')}`",
                    f"- Promotion eligible: `{'yes' if eligibility.get('promotion_eligible') else 'no'}`",
                    f"- Evidence classification: `{'leaderboard' if eligibility.get('promotion_eligible') else 'diagnostic_non_comparable'}`",
                    f"- Blocking reasons: `{error_text}`",
                    f"- Warnings: `{warning_text}`",
                    "",
                ]
            )

        if isinstance(visible_proxy_pressure, dict):
            warnings = visible_proxy_pressure.get("warnings")
            warning_text = (
                ", ".join(str(value) for value in warnings)
                if isinstance(warnings, list) and warnings
                else "none"
            )
            lines.extend(
                [
                    "## Visible Proxy Pressure",
                    "",
                    f"- Pressure score: `{float(visible_proxy_pressure.get('pressure_score', 0.0) or 0.0):.2f}`",
                    f"- Guard retention: `{float(visible_proxy_pressure.get('guard_retention_rate', 0.0) or 0.0):.2f}%`",
                    f"- Family retention: `{float(visible_proxy_pressure.get('family_retention_rate', 0.0) or 0.0):.2f}%`",
                    f"- Regressed visible proxy guards: `{', '.join(str(value) for value in (visible_proxy_pressure.get('regressed_guard_ids', []) or [])) or 'none'}`",
                    f"- Degraded visible families: `{', '.join(str(value) for value in (visible_proxy_pressure.get('degraded_families', []) or [])) or 'none'}`",
                    f"- Warnings: `{warning_text}`",
                    "",
                ]
            )

        if isinstance(optimization_summary, dict):
            evidence_protocol = optimization_summary.get("evidence_protocol")
            rerun_plan = optimization_summary.get("rerun_plan")
            if isinstance(evidence_protocol, dict) or isinstance(rerun_plan, dict):
                lines.extend(["## Evidence Protocol", ""])
                if isinstance(evidence_protocol, dict):
                    local_validation = ", ".join(
                        str(value) for value in evidence_protocol.get("required_local_validation", [])
                    ) or "none"
                    required_reruns = ", ".join(
                        str(value) for value in evidence_protocol.get("required_reruns", [])
                    ) or "none"
                    comparison_fields = ", ".join(
                        str(value) for value in evidence_protocol.get("required_comparison_fields", [])
                    ) or "none"
                    lines.extend(
                        [
                            f"- Decision rule: {evidence_protocol.get('decision_rule', 'n/a')}",
                            f"- Strategy under test: `{evidence_protocol.get('strategy_under_test', 'unknown')}`",
                            f"- Preferred comparison mode: `{evidence_protocol.get('preferred_comparison_mode', 'n/a')}`",
                            f"- Required local validation: `{local_validation}`",
                            f"- Required reruns: `{required_reruns}`",
                            f"- Required comparison fields: `{comparison_fields}`",
                        ]
                    )
                if isinstance(rerun_plan, dict):
                    compare_against = ", ".join(
                        str(value) for value in rerun_plan.get("compare_against", [])
                    ) or "none"
                    required_artifacts = ", ".join(
                        str(value) for value in rerun_plan.get("required_artifacts", [])
                    ) or "none"
                    lines.extend(
                        [
                            f"- Recommended first rerun: `{rerun_plan.get('recommended_first_rerun', 'n/a')}`",
                            f"- First rerun reason: {rerun_plan.get('first_rerun_reason', 'n/a')}",
                            f"- Follow-up rerun: `{rerun_plan.get('follow_up_rerun', 'n/a')}`",
                            f"- Follow-up condition: {rerun_plan.get('follow_up_condition', 'n/a')}",
                            f"- Compare against: `{compare_against}`",
                            f"- Required artifacts: `{required_artifacts}`",
                        ]
                    )
                lines.append("")

        if latest_train and (latest_validation_pass_rate is not None or latest_test_pass_rate is not None):
            lines.extend(
                [
                    "## Generalization",
                    "",
                    f"- Train questions: `{latest_train.total}`",
                    f"- Validation questions: `{validation_questions}`",
                    f"- Held-out questions: `{len(self._load_test_ids())}`",
                    f"- Train -> validation delta: `{validation_delta_text}`",
                    f"- Train -> final test delta: `{final_test_delta_text}`",
                    "",
                ]
            )
            if latest_validation_summary and validation_mode == "small_sample_kfold":
                lines.extend(
                    [
                        f"- Validation folds: `{int(latest_validation_summary.get('fold_count', 0))}`",
                        f"- Mean fold pass rate: `{float(latest_validation_summary.get('mean_fold_pass_rate', 0.0)):.2f}%`",
                        f"- Worst fold pass rate: `{float(latest_validation_summary.get('min_fold_pass_rate', 0.0)):.2f}%`",
                        "",
                    ]
                )

        if latest_train and (latest_validation_pass_rate is not None or latest_test_pass_rate is not None):
            lines.extend(["## Assessment", ""])
            if (
                latest_validation_pass_rate is not None
                and latest_test_pass_rate is not None
                and latest_train.pass_rate >= 95.0
                and latest_validation_pass_rate >= 95.0
                and (latest_train.pass_rate - latest_test_pass_rate) >= 20.0
            ):
                lines.append("- The visible pool appears saturated, but the final blind holdout still lags materially. Further automated tuning is likely to overfit unless the owner adds better metadata, broader examples, or more representative benchmarks.")
            elif latest_test_pass_rate is not None and latest_train.pass_rate < latest_test_pass_rate:
                lines.append("- The final blind holdout is stronger than the train split, which suggests the current changes generalize well and the train set may be narrower than production usage.")
            else:
                lines.append("- MaxGenie still has room to improve automatically using the current assets and benchmark pool.")
            lines.append("")

        if isinstance(generalization_audit, dict):
            audit_metrics = generalization_audit.get("metrics")
            audit_metrics = audit_metrics if isinstance(audit_metrics, dict) else {}
            audit_decisions = generalization_audit.get("decisions")
            audit_decisions = audit_decisions if isinstance(audit_decisions, dict) else {}
            hidden_summary = generalization_audit.get("hidden_holdout_summary")
            hidden_summary = hidden_summary if isinstance(hidden_summary, dict) else {}
            flags = generalization_audit.get("flags")
            flag_text = (
                ", ".join(str(flag) for flag in flags)
                if isinstance(flags, list) and flags
                else "none"
            )
            issue_counts = hidden_summary.get("issue_class_counts")
            issue_text = (
                ", ".join(
                    f"{key}={int(value)}"
                    for key, value in sorted(issue_counts.items())
                    if isinstance(value, (int, float))
                )
                if isinstance(issue_counts, dict) and issue_counts
                else "none"
            )
            lines.extend(
                [
                    "## Generalization Audit",
                    "",
                    f"- Status: `{generalization_audit.get('status', 'unknown')}`",
                    f"- Flags: `{flag_text}`",
                    f"- May workspace incumbent candidate: `{'yes' if audit_decisions.get('may_workspace_incumbent') else 'no'}`",
                    f"- Production-golden candidate: `{'yes' if audit_decisions.get('production_golden_candidate') else 'no'}`",
                    f"- Full delta from run baseline: `{_optional_rate_text(_optional_float(audit_metrics.get('full_delta')))}`",
                    f"- Holdout delta from run baseline: `{_optional_rate_text(_optional_float(audit_metrics.get('holdout_delta')))}`",
                    f"- Validation -> holdout gap: `{_optional_rate_text(_optional_float(audit_metrics.get('validation_holdout_gap')))}`",
                    f"- Hidden holdout question count: `{int(hidden_summary.get('question_count', 0) or 0)}`",
                    f"- Hidden holdout issue families: `{issue_text}`",
                    f"- Hidden holdout policy: {generalization_audit.get('hidden_holdout_policy', 'n/a')}",
                    "",
                ]
            )
            owner_actions = generalization_audit.get("owner_actions")
            if isinstance(owner_actions, list) and owner_actions:
                lines.extend(["### Audit Actions", ""])
                for action in owner_actions[:5]:
                    lines.append(f"- {action}")
                lines.append("")

        if diagnostics is not None:
            metadata_coverage = diagnostics.metadata_coverage
            table_coverage_percent = _coverage_percent(
                metadata_coverage.described_tables,
                metadata_coverage.total_tables,
            )
            column_coverage_percent = _coverage_percent(
                metadata_coverage.described_columns,
                metadata_coverage.total_columns,
            )
            lines.extend(
                [
                    "## Space Diagnostics",
                    "",
                    f"- External table reference warnings: `{len(diagnostics.external_table_references)}`",
                    f"- Entity-matching candidates still disabled: `{len(diagnostics.entity_matching_candidates)}`",
                    f"- Format-assistance candidates still disabled: `{len(diagnostics.format_assistance_candidates)}`",
                    f"- Table description coverage: `{table_coverage_percent:.2f}%`"
                    if table_coverage_percent is not None
                    else "- Table description coverage: `n/a`",
                    f"- Column description coverage: `{column_coverage_percent:.2f}%`"
                    if column_coverage_percent is not None
                    else "- Column description coverage: `n/a`",
                    "",
                ]
            )
            if diagnostics.external_table_references:
                lines.extend(["### External Table References", ""])
                for finding in diagnostics.external_table_references[:5]:
                    lines.append(f"- `{finding.reference}` in `{finding.location}`")
                if len(diagnostics.external_table_references) > 5:
                    lines.append(
                        f"- `{len(diagnostics.external_table_references) - 5}` more external references omitted"
                    )
                lines.append("")
            if metadata_coverage.low_coverage_tables:
                lines.extend(["### Low-Coverage Tables", ""])
                for table in metadata_coverage.low_coverage_tables[:5]:
                    lines.append(
                        f"- `{table.identifier}`: `{table.described_columns}/{table.total_columns}` described columns"
                    )
                lines.append("")

        if isinstance(curation_summary, dict):
            if _is_gated_curation_summary(curation_summary):
                # Render benchmark-gated loop summary.
                accepted = int(curation_summary.get("accepted", 0))
                rejected = int(curation_summary.get("rejected", 0))
                skipped = int(curation_summary.get("skipped", 0))
                modes_tried = curation_summary.get("modes_tried", [])
                gating_active = bool(curation_summary.get("gating_active", True))
                gating_note = "" if gating_active else " (no reference scores — gating was inactive)"
                lines.extend(
                    [
                        "## Benchmark-Gated Curation Loop",
                        "",
                        f"- Modes tried: `{', '.join(modes_tried) if modes_tried else 'none'}`",
                        f"- Accepted: `{accepted}`  Rejected: `{rejected}`  Skipped: `{skipped}`{gating_note}",
                        "",
                    ]
                )
                passes = curation_summary.get("passes", [])
                if isinstance(passes, list) and passes:
                    lines.extend(["### Per-Mode Outcomes", ""])
                    for pass_outcome in passes:
                        if not isinstance(pass_outcome, dict):
                            continue
                        outcome = str(pass_outcome.get("outcome", "unknown")).upper()
                        mode = pass_outcome.get("mode", "?")
                        if pass_outcome.get("outcome") == "accepted":
                            train_r = pass_outcome.get("train_rate")
                            validation_r = pass_outcome.get("validation_rate", pass_outcome.get("dev_rate"))
                            detail = (
                                f"train={train_r:.1f}%  validation={validation_r:.1f}%"
                                if train_r is not None and validation_r is not None
                                else ""
                            )
                        elif pass_outcome.get("outcome") == "rejected":
                            detail = str(pass_outcome.get("reason", ""))
                        else:
                            detail = str(pass_outcome.get("reason", ""))
                        lines.append(f"- `[{outcome:8s}]` `{mode}`: {detail}")
                    lines.append("")
                    intent_audit_lines = self._artifact_patch_intent_audit_lines(
                        curation_summary
                    )
                    if intent_audit_lines:
                        lines.extend(["### Artifact Patch Intent Audit", ""])
                        lines.extend(intent_audit_lines)
                        lines.append("")
            else:
                # Render single-mode curation summary.
                candidate_status = str(curation_summary.get("candidate_status", "prepared"))
                lines.extend(
                    [
                        "## Automatic Curation",
                        "",
                        f"- Mode: `{curation_summary.get('mode', 'unknown')}`",
                        f"- Candidate status: `{candidate_status}`",
                        f"- Table descriptions added: `{int(curation_summary.get('table_descriptions_added', 0))}`",
                        f"- Column descriptions added: `{int(curation_summary.get('column_descriptions_added', 0))}`",
                        f"- Columns hidden as noise: `{int(curation_summary.get('columns_hidden', 0))}`",
                        f"- Format-assistance toggles enabled: `{int(curation_summary.get('format_assistance_enabled', 0))}`",
                        f"- Entity-matching toggles enabled: `{int(curation_summary.get('entity_matching_enabled', 0))}`",
                        f"- Synonyms added: `{int(curation_summary.get('synonyms_added', 0))}`",
                        "",
                    ]
                )
                rejection_reason = curation_summary.get("rejection_reason")
                acceptance_reason = curation_summary.get("acceptance_reason")
                if candidate_status == "accepted" and acceptance_reason:
                    lines.append(f"- Acceptance reason: {acceptance_reason}")
                    lines.append("")
                if candidate_status == "rejected" and rejection_reason:
                    lines.append(f"- Rejection reason: {rejection_reason}")
                    lines.append("")
                changes = curation_summary.get("changes", [])
                if isinstance(changes, list) and changes:
                    lines.extend(["### Notable Curation Actions", ""])
                    for change in changes[:8]:
                        if not isinstance(change, dict):
                            continue
                        target = (
                            f"`{change.get('table_identifier')}`"
                            if change.get("column_name") is None
                            else f"`{change.get('table_identifier')}.{change.get('column_name')}`"
                        )
                        lines.append(f"- {target}: {change.get('reason')}")
                    if len(changes) > 8:
                        lines.append(f"- `{len(changes) - 8}` more curation changes omitted")
                    lines.append("")

        if isinstance(query_history_summary, dict):
            lines.extend(
                [
                    "## Query History Signals",
                    "",
                    f"- Collection status: `{query_history_summary.get('status', 'unknown')}`",
                    f"- Warehouse used: `{query_history_summary.get('warehouse_id') or 'n/a'}`",
                    f"- Queries collected: `{int(query_history_summary.get('query_count', 0))}`",
                    f"- Successful queries: `{int(query_history_summary.get('successful_query_count', 0))}`",
                    f"- Failed queries: `{int(query_history_summary.get('failed_query_count', 0))}`",
                    f"- Thumbs up: `{int(query_history_summary.get('feedback_thumbs_up', 0))}`",
                    f"- Thumbs down: `{int(query_history_summary.get('feedback_thumbs_down', 0))}`",
                    "",
                ]
            )
            if query_history_summary.get("message"):
                lines.append(f"- Note: {query_history_summary.get('message')}")
                lines.append("")
            if isinstance(query_history_recommendations, dict):
                lines.extend(
                    [
                        f"- Trusted-asset candidates: `{len(query_history_recommendations.get('recommended_trusted_asset_candidates', []))}`",
                        f"- Benchmark candidates: `{len(query_history_recommendations.get('recommended_benchmark_candidates', []))}`",
                        f"- Noisy one-off queries: `{int(query_history_recommendations.get('noisy_one_off_queries', 0))}`",
                        f"- Repeated query families: `{int(query_history_recommendations.get('repeated_query_families', 0))}`",
                        "",
                    ]
                )
            success_patterns = query_history_summary.get("recurring_success_patterns", [])
            if isinstance(success_patterns, list) and success_patterns:
                lines.extend(["### Recurring Successful SQL Families", ""])
                for pattern in success_patterns[:5]:
                    if not isinstance(pattern, dict):
                        continue
                    lines.append(
                        f"- `{int(pattern.get('query_count', 0))}` runs, last seen `{pattern.get('last_seen_at')}`"
                    )
                lines.append("")
            failure_patterns = query_history_summary.get("recurring_failure_patterns", [])
            if isinstance(failure_patterns, list) and failure_patterns:
                lines.extend(["### Recurring Failed SQL Families", ""])
                for pattern in failure_patterns[:5]:
                    if not isinstance(pattern, dict):
                        continue
                    lines.append(
                        f"- `{int(pattern.get('query_count', 0))}` failed runs, last seen `{pattern.get('last_seen_at')}`"
                    )
                lines.append("")
            if (
                isinstance(query_history_recommendations, dict)
                and query_history_recommendations.get("recommended_trusted_asset_candidates")
            ):
                lines.extend(["### Trusted-Asset Candidates", ""])
                for recommendation in query_history_recommendations["recommended_trusted_asset_candidates"][:5]:
                    if not isinstance(recommendation, dict):
                        continue
                    line = (
                        f"- `{int(recommendation.get('query_count', 0))}` repeated successful runs, last seen"
                        f" `{recommendation.get('last_seen_at')}`, strategy"
                        f" `{recommendation.get('expansion_strategy', 'review')}`"
                    )
                    matched_ids = recommendation.get("matched_benchmark_ids") or []
                    if isinstance(matched_ids, list) and matched_ids:
                        line += f", matched benchmarks `{', '.join(str(value) for value in matched_ids[:3])}`"
                    lines.append(line)
                    owner_action = recommendation.get("owner_action")
                    if owner_action:
                        lines.append(f"- Action: {owner_action}")
                lines.append("")
            if (
                isinstance(query_history_recommendations, dict)
                and query_history_recommendations.get("recommended_benchmark_candidates")
            ):
                lines.extend(["### Benchmark Candidates From Query History", ""])
                for recommendation in query_history_recommendations["recommended_benchmark_candidates"][:5]:
                    if not isinstance(recommendation, dict):
                        continue
                    line = (
                        f"- `{int(recommendation.get('query_count', 0))}` repeated problem runs, last seen"
                        f" `{recommendation.get('last_seen_at')}`, strategy"
                        f" `{recommendation.get('expansion_strategy', 'review')}`"
                    )
                    matched_ids = recommendation.get("matched_benchmark_ids") or []
                    if isinstance(matched_ids, list) and matched_ids:
                        line += f", matched benchmarks `{', '.join(str(value) for value in matched_ids[:3])}`"
                    lines.append(line)
                    owner_action = recommendation.get("owner_action")
                    if owner_action:
                        lines.append(f"- Action: {owner_action}")
                lines.append("")

        outside_items: list[str] = []
        if isinstance(query_history_summary, dict) and query_history_summary.get("status") not in {
            None,
            "ok",
            "success",
            "available",
            "no_data",
        }:
            message = query_history_summary.get("message")
            outside_items.append(
                "Query history could not be collected from the workspace"
                + (f": {message}" if message else ".")
            )
        if latest_full_comparison_mode == "string_only":
            outside_items.append(
                "Full result-based comparison was not active; provide a usable SQL warehouse and access "
                "when exact answer-result comparison is required."
            )
        if latest_test_pass_rate is not None and len(self._load_test_ids()) <= 3:
            outside_items.append(
                "The final holdout is very small, so owner-provided benchmark expansion is needed before "
                "treating the generalization estimate as stable."
            )
        if diagnostics is not None and diagnostics.external_table_references:
            outside_items.append(
                "Some instructions or examples reference tables outside the declared space; those upstream "
                "or asset-scope decisions need owner review."
            )
        if diagnostics is not None and diagnostics.entity_matching_candidates:
            outside_items.append(
                "High-value categorical columns still need owner confirmation before prompt matching or "
                "entity matching can be enabled safely."
            )
        blocked_ids_for_outside = self.detect_blocked_candidates()
        if blocked_ids_for_outside or blocked_notes:
            outside_items.append(
                "Repeated benchmark failures remain after the automated loop; these likely need business "
                "mapping, join, metric, or upstream data clarification."
            )
        if outside_items:
            lines.extend(["## Outside MaxGenie Control", ""])
            for item in outside_items:
                lines.append(f"- {item}")
            lines.append("")

        if baseline_train and latest_train:
            train_delta = latest_train.pass_rate - baseline_train.pass_rate
            lines.extend(
                [
                    "## Change Since Baseline",
                    "",
                    f"- Baseline train pass rate: `{baseline_train.pass_rate:.2f}%`",
                    f"- Current train pass rate: `{latest_train.pass_rate:.2f}%`",
                    f"- Delta: `{train_delta:+.2f}%`",
                    "",
                ]
            )

        if latest_train and (latest_validation_pass_rate is not None or latest_test_pass_rate is not None):
            lines.extend(["## Next Focus", ""])
            focus_gaps = [
                gap
                for gap in (validation_delta_value, final_test_delta_value)
                if gap is not None
            ]
            largest_gap = max(focus_gaps) if focus_gaps else 0.0
            if largest_gap >= 20.0:
                lines.append("- Lowest hanging fruit: replace or trim narrow benchmark-shaped examples and favor broader instruction patterns that generalize across brands, markets, and time windows.")
                lines.append("- Prioritize examples that demonstrate reusable query structure rather than hard-coded benchmark phrasings or assumption columns.")
                lines.append("- Once train is saturated, judge new changes by validation lift first and use the final blind test sparingly at checkpoints.")
            else:
                lines.append("- Continue targeted structural cleanup in table descriptions, joins, and SQL snippets while watching for blind-set regression.")
            if any(gap >= 20.0 for gap in focus_gaps):
                lines.append("- Avoid adding more narrow SQL examples until the blind validation and final test gaps come down.")
            if diagnostics is not None and diagnostics.external_table_references:
                lines.append("- Remove references to tables that are not declared in the space before adding more examples or instructions.")
            if diagnostics is not None and diagnostics.entity_matching_candidates:
                lines.append("- Review prompt matching/entity matching for high-value categorical columns before adding more benchmark-shaped SQL examples.")
            if diagnostics is not None and diagnostics.format_assistance_candidates:
                lines.append("- Review format assistance for time and period columns so date-window questions rely less on brittle text instructions.")
            if isinstance(query_history_summary, dict) and int(query_history_summary.get("failed_query_count", 0)) > 0:
                lines.append("- Review recurring failed query-history families before adding more benchmark-shaped examples; those failures reflect real user demand rather than only synthetic benchmarks.")
            if isinstance(query_history_recommendations, dict) and query_history_recommendations.get("recommended_benchmark_candidates"):
                lines.append("- Use the query-history benchmark candidates to expand existing benchmark families with real Messages feed phrasings where possible; query history alone gives SQL demand signals, not reliable natural-language wording.")
            if isinstance(query_history_recommendations, dict) and query_history_recommendations.get("recommended_trusted_asset_candidates"):
                lines.append("- Promote only the most repeated successful query families into trusted assets or highly scoped examples; do not bulk-import raw history.")
            lines.append("")

        if diagnostics is not None:
            lines.extend(["## Owner Inputs Needed", ""])
            if diagnostics.metadata_coverage.described_columns == 0:
                lines.append("- Add business descriptions for the included tables and their important columns. Right now the space has no described columns, so Genie is learning mostly from examples instead of durable semantics.")
            if latest_test_pass_rate is not None and len(self._load_test_ids()) <= 3:
                lines.append("- Add more benchmark families or real user questions before trusting the generalization estimate too much. A 3-question final holdout is directionally useful but still small.")
            if diagnostics.entity_matching_candidates:
                entity_columns = list(
                    dict.fromkeys(candidate.column_name for candidate in diagnostics.entity_matching_candidates[:5])
                )
                lines.append("- Confirm whether these categorical fields should support prompt matching/entity matching: " + ", ".join(
                    f"`{column_name}`" for column_name in entity_columns
                ))
            if diagnostics.format_assistance_candidates:
                format_columns = list(
                    dict.fromkeys(candidate.column_name for candidate in diagnostics.format_assistance_candidates[:5])
                )
                lines.append("- Confirm the canonical date and time-bucket semantics for columns like " + ", ".join(
                    f"`{column_name}`" for column_name in format_columns
                ) + " so the space can apply period logic more reliably.")
            if diagnostics.external_table_references:
                lines.append("- Remove or justify any references to tables that are not declared inside the space before trusting future optimization runs.")
            if isinstance(query_history_summary, dict) and query_history_summary.get("status") == "no_data":
                lines.append("- Enable real-user traffic or collect a longer history window before relying on query-history-driven example suggestions.")
            if isinstance(query_history_summary, dict) and int(query_history_summary.get("feedback_thumbs_down", 0)) > 0:
                lines.append("- Inspect the thumbs-down conversations for real user intents that are not represented in the current benchmark families.")
            if isinstance(query_history_recommendations, dict) and query_history_recommendations.get("recommended_benchmark_candidates"):
                lines.append("- Query history identifies recurring SQL families, but the original user wording is still best sourced from the Messages feed before adding new benchmark phrasings.")
            if blocked_notes and blocked_notes != "# Blocked Failures":
                lines.append("- Review the blocked-failure notes for structural issues MaxGenie could not resolve automatically.")
            if lines[-1] != "":
                lines.append("")

        if latest_train:
            failures = [result for result in latest_train.results if result.status in {BenchmarkStatus.FAILED, BenchmarkStatus.ERROR}]
            if failures:
                lines.extend(["## Remaining Train Issues", ""])
                for result in failures[:8]:
                    if result.status == BenchmarkStatus.ERROR:
                        detail = result.error_message or "Unknown error"
                    elif result.comparison_score is not None:
                        detail = f"score={result.comparison_score:.3f}"
                    else:
                        detail = "score=N/A"
                    lines.append(f"- `{result.question_id}`: {detail}")
                lines.append("")

                near_misses = [result for result in failures if result.comparison_score is not None and result.comparison_score >= 0.7]
                structural = [result for result in failures if result.comparison_score is not None and result.comparison_score < 0.5]
                lines.extend(["## Recommendations", ""])
                if near_misses:
                    lines.append(f"- `{len(near_misses)}` near-miss failures look fixable with targeted synonyms, snippets, or instruction cleanup.")
                if structural:
                    lines.append(f"- `{len(structural)}` failures still look structural; review joins, table descriptions, and missing business mappings.")
                if not near_misses and not structural:
                    lines.append("- Review the latest train failures and blocked-failure notes before making another wide config change.")
                lines.append("")

        blocked_ids = self.detect_blocked_candidates()
        if blocked_ids or blocked_notes:
            lines.extend(["## Blocked Failures", ""])
            if blocked_ids:
                lines.append(f"- Repeated unresolved failures across the last 3 train runs: `{', '.join(blocked_ids)}`")
            if blocked_notes:
                for note_line in blocked_notes.splitlines()[1:]:
                    if note_line.strip():
                        lines.append(f"- {note_line.strip()}")
            lines.append("")

        if not is_advisory:
            self.write_comparable_run_manifest(
                latest_full=latest_full,
                workspace_meta=workspace_meta,
                clone_meta=clone_meta,
                split_meta=split_meta,
            )
        report_markdown = "\n".join(lines).rstrip() + "\n"
        report_path = self.artifacts.write_text_path(self.artifacts.final_report_path, report_markdown)
        self._write_report_notebook(report_markdown)
        return report_path
