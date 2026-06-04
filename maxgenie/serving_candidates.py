"""Serving-backed candidate proposals for the optimization loop."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import fnmatch
import json
import re
from typing import Any, Mapping, Sequence

from databricks.sdk import WorkspaceClient
import requests

from maxgenie.artifact_patch_semantics import (
    question_overlap_terms,
    sql_delta_intents_from_visible_failure,
)
from maxgenie.ids import coerce_artifact_id
from maxgenie.schemas import (
    GenieColumnConfig,
    GenieExampleSQL,
    GenieInstruction,
    GenieInstructions,
    GenieJoinSpecs,
    GenieSQLSnippet,
    GenieSQLSnippets,
    GenieSpaceConfig,
    GenieTableJoinSpec,
)


_VALID_SNIPPET_GROUPS = frozenset({"filters", "expressions", "measures"})
_VALID_RESPONSE_FORMATS = frozenset({"json_schema", "json_object", "none"})
_VALID_API_FORMATS = frozenset({"auto", "chat", "responses"})
_OPERATION_RISK_GROUPS = {
    "append_text_instruction": "text_instruction",
    "add_example_sql": "example_sql",
    "add_sql_snippet": "sql_snippet",
    "add_join_spec": "join_spec",
    "set_table_description": "metadata",
    "set_column_config": "metadata",
}
_MAX_METADATA_OPERATIONS_PER_CANDIDATE = 3
_MAX_TEXT_INSTRUCTION_CHARS = 900
_MAX_EXAMPLE_SET_OPERATIONS = 9
_MAX_EXAMPLE_SET_TARGET_FAILURES = 2
_BIND_PLACEHOLDER_RE = re.compile(r"(?<!:):[A-Za-z_][A-Za-z0-9_]*|\$\{[^}]+\}")
_SQL_STATEMENT_START_RE = re.compile(
    r"^\s*(?:--[^\n]*\n\s*)*(?:/\*.*?\*/\s*)*([A-Za-z]+)",
    re.DOTALL,
)
_SQL_PREDICATE_OPERATOR_RE = re.compile(
    r"(?:=|<>|!=|<=|>=|<|>|\bIN\s*\(|\bLIKE\b|\bBETWEEN\b|\bIS\s+(?:NOT\s+)?NULL\b)",
    re.IGNORECASE,
)
_SQL_SNIPPET_OPTION_SET_RE = re.compile(
    r"\b(?:choose|select|use)\s+(?:exactly\s+)?one\b|\bone\s+predicate\b|\balternative(?:s)?\b",
    re.IGNORECASE,
)
_DISALLOWED_SNIPPET_STATEMENT_STARTS = frozenset(
    {
        "alter",
        "create",
        "delete",
        "describe",
        "drop",
        "explain",
        "grant",
        "insert",
        "merge",
        "refresh",
        "revoke",
        "select",
        "show",
        "truncate",
        "update",
        "use",
        "with",
    }
)
_EXAMPLE_SET_CANDIDATE_MODES = frozenset(
    {"example_set", "benchmark_example_set", "typed_template"}
)
_CANDIDATE_MODE_SEQUENCE = (
    "artifact_patch_v2",
)
_RECOVERY_MODE_SEQUENCE = (
    "artifact_patch_v2",
)
_ACCEPTED_FIX_REGRESSION_RECOVERY_MODE_SEQUENCE = (
    "artifact_patch_v2",
)
# Opt-in adaptive recipe selection: map a *dominant* visible failure family to the
# narrow recipe mode whose single operation group most directly addresses it. Only
# families with one unambiguous best-fit mode are listed; everything else (e.g.
# ``runtime_or_generation_error`` and ``general_sql_mismatch``, whose suggested
# operations span several groups) deliberately falls through to the strict
# ``artifact_patch_v2`` default so the comparable backbone is never weakened. The
# family names are exactly those emitted by ``failure_context._classify_failure_family``.
_ADAPTIVE_MODE_BY_FAMILY = {
    "missing_generated_sql": "deterministic_guidance",
    "literal_or_parameter_mismatch": "typed_template",
    "percentage_or_comparison_metric": "typed_template",
    "time_filter_mismatch": "schema_time_window",
    "table_or_join_path_mismatch": "schema_metadata_aliases",
    "selected_metric_or_column_mismatch": "schema_metadata_aliases",
    "aggregation_or_grouping_mismatch": "schema_metadata_aliases",
}
_FREEDOM_ARTIFACT_PATCH_MODE = "freedom_artifact_patch_v1"
_FREEDOM_HISTORY_ARTIFACT_PATCH_MODE = "freedom_artifact_patch_with_history_v1"
_DATA_PROBE_ARTIFACT_PATCH_MODE = "data_probe_artifact_patch_v1"
_BROAD_ARTIFACT_PATCH_MODES = frozenset(
    {
        _FREEDOM_ARTIFACT_PATCH_MODE,
        _FREEDOM_HISTORY_ARTIFACT_PATCH_MODE,
        _DATA_PROBE_ARTIFACT_PATCH_MODE,
    }
)
_CANDIDATE_MODE_OPERATION_GROUPS = {
    "example_sql": frozenset({"example_sql"}),
    "example_set": frozenset({"example_sql"}),
    "benchmark_example_set": frozenset({"example_sql"}),
    "typed_template": frozenset({"example_sql"}),
    "schema_time_window": frozenset({"sql_snippet"}),
    "deterministic_guidance": frozenset({"text_instruction"}),
    "schema_metadata_aliases": frozenset({"metadata"}),
    "artifact_patch": frozenset({"artifact_patch"}),
    "artifact_patch_v2": frozenset({"artifact_patch"}),
    "skeleton_artifact_patch_v2": frozenset({"artifact_patch"}),
    "composite_artifact_patch_v2": frozenset({"artifact_patch"}),
    "executable_artifact_patch_v2": frozenset({"artifact_patch"}),
    _FREEDOM_ARTIFACT_PATCH_MODE: frozenset({"artifact_patch"}),
    _FREEDOM_HISTORY_ARTIFACT_PATCH_MODE: frozenset({"artifact_patch"}),
    _DATA_PROBE_ARTIFACT_PATCH_MODE: frozenset({"artifact_patch"}),
    "sql_snippet": frozenset({"sql_snippet"}),
    "metadata": frozenset({"metadata"}),
    "join_spec": frozenset({"join_spec"}),
    "text_instruction": frozenset({"text_instruction"}),
}
_STRICT_ARTIFACT_PATCH_MODES = frozenset(
    {
        "artifact_patch_v2",
        "skeleton_artifact_patch_v2",
        "composite_artifact_patch_v2",
        "executable_artifact_patch_v2",
        _FREEDOM_ARTIFACT_PATCH_MODE,
        _FREEDOM_HISTORY_ARTIFACT_PATCH_MODE,
        _DATA_PROBE_ARTIFACT_PATCH_MODE,
    }
)
_ARTIFACT_PATCH_OPERATION_PATH_HINTS = {
    "append_text_instruction": ("instructions/text_instruction.md",),
    "add_example_sql": ("instructions/examples/*.yml",),
    "add_sql_snippet": (
        "instructions/sql_snippets/filters/*.yml",
        "instructions/sql_snippets/expressions/*.yml",
        "instructions/sql_snippets/measures/*.yml",
    ),
    "add_join_spec": ("instructions/join_specs/*.yml",),
    "set_column_config": ("tables/*.yml", "metric_views/*.yml"),
    "set_table_description": ("tables/*.yml", "metric_views/*.yml"),
}
_EXECUTABLE_ARTIFACT_OPERATION_FAMILIES = frozenset(
    {"add_example_sql", "add_sql_snippet"}
)
_FILE_EDIT_NULLABLE_FIELDS = (
    "path",
    "relative_path",
    "content",
    "append_content",
    "find",
    "replace",
    "expected_sha256",
    "reason",
    "description",
    "comment",
)
_SKELETON_ARTIFACT_PATCH_MODE = "skeleton_artifact_patch_v2"
_SKELETON_REPLACEMENT_BUDGET_CHARS = 1200
_SKELETON_HIDDEN_CONTEXT_RE = re.compile(
    r"\b(?:hidden[_ -]?holdout|holdout\s+(?:question|sql|query)|final[_ -]?test|test[_ -]?ids)\b",
    re.IGNORECASE,
)
_SKELETON_METADATA_QUERY_PLAN_RE = re.compile(
    r"\b(?:select\b.+\bfrom|with\s+\w+\s+as|cte|all_sales|time_buckets|result\s+schema|output\s+columns|aggregation\s+instructions?)\b",
    re.IGNORECASE | re.DOTALL,
)
_CANDIDATE_MODE_GUIDANCE = {
    "example_sql": (
        "Use operation group example_sql only.",
        "Allowed operation: add_example_sql.",
        "Return exactly one literal example query, or no_change if no safe example exists.",
    ),
    "example_set": (
        "Use operation group example_sql only.",
        "Allowed operation: add_example_sql.",
        "Return 4 to 9 literal example queries as one guard-preserving set, or no_change if no safe set exists.",
        "This mode is for guard-preserving fixes where single examples or global instructions could regress working queries.",
        "Include one exact example for every visible train/validation baseline pass guard before adding any failed train examples.",
        "For each guard example, set reason to guard_preservation:<question_id> using the guard ID from the failure context.",
        "After all guard IDs are covered, use the remaining operation budget for at most two high-confidence failed train examples with reason target_train_failure:<question_id>.",
        "For benchmark-tagged examples, set id to the benchmark question_id and sql to null; MaxGenie will materialize the exact expected SQL.",
        "When a prior example_set was accepted, target remaining compatible failures from the latest Failed Questions section and the Still failing lines; do not repeat target_train_failure IDs that are no longer failing.",
        "Keep follow-up example_set candidates compact and benchmark-ID based: prefer id plus sql=null over copied SQL for every visible benchmark example.",
        "Prefer one target failure; use two only when the failed examples share the same root-cause pattern and do not conflict with any guard pattern.",
        "A bundle missing any visible baseline pass guard will be rejected before benchmarking.",
        "Do not copy long benchmark SQL into the response unless no benchmark ID is available; do not use bind placeholders.",
    ),
    "benchmark_example_set": (
        "Use operation group example_sql only.",
        "Allowed operation: add_example_sql.",
        "Return one compact benchmark-tagged example bundle, or no_change if no safe bundle exists.",
        "This mode is the structured alternative to global text-instruction patches.",
        "Every operation should be benchmark-tagged: set id to a visible benchmark question_id and sql to null so MaxGenie materializes the exact visible expected SQL.",
        "Include one guard_preservation:<question_id> example for every visible train/validation baseline pass guard before adding any failed train examples.",
        "After all guard IDs are covered, add at most two compatible target_train_failure:<question_id> examples from the current remaining train failures.",
        "Do not copy long benchmark SQL into the response; use IDs and reasons. Do not use hidden holdout text or SQL.",
        "Prefer one target failure; use two only when the failures share the same root-cause pattern and do not conflict with any guard pattern.",
    ),
    "typed_template": (
        "Use operation group example_sql only.",
        "Allowed operation: add_example_sql.",
        "Return compact benchmark-tagged target hints only; MaxGenie will deterministically materialize executable visible SQL and typed guidance.",
        "This mode is for recurring visible result-error families: unbound placeholders, period/time-window logic, and output-schema column mismatches.",
        "Set id to a visible benchmark question_id and sql to null for each target; do not copy SQL text.",
        "Prefer exactly one target_train_failure:<question_id> from the dominant visible family. Use two only when both share the same root-cause family and guard risk is not high.",
        "MaxGenie will inject required guard_preservation:<question_id> examples before the target examples and will reject the bundle if guards cannot be preserved.",
        "Do not use hidden holdout text or SQL. Return no_change if the visible failures do not support a safe typed template.",
    ),
    "schema_time_window": (
        "Use operation group sql_snippet only.",
        "Allowed operation: add_sql_snippet.",
        "Return one reusable SQL snippet, or no_change if no safe snippet exists.",
        "Always set display_name; if omitted, MaxGenie can derive it only from a clear description, question, id, or alias.",
        "This mode is the less benchmark-shaped successor to typed_template: do not add exact benchmark examples or copied visible SQL.",
        "Target recurring visible output-schema, metric-column, period-comparison, or time-window failures with a reusable filter, expression, or measure snippet.",
        "The snippet must be a valid SQL fragment for its group, not a full SELECT/WITH query or DML/DDL statement, and it must use literal-safe semantics with no bind placeholders.",
        "For filter snippets, return one concrete predicate/expression, not a menu or list of alternative predicates.",
        "Prefer snippets that preserve requested output aliases, date-grain/window labels, and current-versus-prior period semantics across a family of questions.",
        "Do not use hidden holdout text or SQL. Return no_change if the visible failures only support a narrow benchmark-specific example.",
    ),
    "deterministic_guidance": (
        "Use operation group text_instruction only.",
        "Allowed operation: append_text_instruction.",
        "Return exactly one concise instruction under 900 characters, or no_change if no safe durable guidance exists.",
        "This mode is for recurring visible result-error families after example and snippet lanes have overfit or stayed flat.",
        "Target one durable rule for executable SQL generation: literal values instead of bind placeholders, required output aliases, period-comparison semantics, or time-window labels.",
        "Do not add examples, SQL snippets, metadata, joins, file_edits, or copied benchmark SQL.",
        "Do not mention hidden holdout content. Return no_change if the visible failures require a benchmark-specific SQL answer rather than general guidance.",
    ),
    "schema_metadata_aliases": (
        "Use operation group metadata only.",
        "Allowed operations: set_column_config or set_table_description.",
        "Touch at most two existing tables or columns, or no_change if no metadata-only fix is safe.",
        "This mode is the structural follow-up after examples, snippets, and text guidance plateaued.",
        "Target visible recurring schema/alias/entity issues: ambiguous business columns, required output column aliases, literal entity values, table purpose, or metric-column meaning.",
        "Prefer column descriptions and synonyms over enabling entity matching; enable entity matching only when visible evidence shows entity resolution is the direct blocker for that column.",
        "Do not add examples, SQL snippets, joins, text instructions, file_edits, copied benchmark SQL, or hidden holdout content.",
        "Do not hide or exclude columns unless diagnostics show that the column is actively misleading visible failures.",
    ),
    "artifact_patch": (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return a coherent decomposed-space file patch across allowlisted files only.",
        "Allowed paths include instructions/text_instruction.md, instructions/examples/*.yml, instructions/sql_snippets/*/*.yml, instructions/join_specs/*.yml, tables/*.yml, metric_views/*.yml, and sample_questions.yml.",
        "Each file edit must include path, content, and expected_sha256 from the Decomposed Space Files section when the file already exists.",
        "Patch train and validation-visible patterns only; do not use hidden holdout text or SQL.",
        "Prefer a small coherent patch over many broad edits, and return no_change if a safe file patch is not available.",
    ),
    "artifact_patch_v2": (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return at most two decomposed-space file edits.",
        "Each edit must include path, expected_sha256 copied exactly from the current Decomposed Space Files payload, and a concise reason. Set unused nullable file-edit fields to null. Set relative_path to null when path is set; path and relative_path must not disagree.",
        "Each edit reason, rationale, or expected_impact must cite a visible-effect marker from structured context: target_train_failure:<question_id>, visible_family:<family>, failure_family:<family>, or guard_preservation:<question_id>.",
        "Use exactly one edit primitive per file edit: content, append_content, or find plus replace. For existing files, use compact edits: set content to null and use either append_content or exact find plus replace. Full content rewrites of existing files are rejected for visible-context causal-plan patches. Use full content only for small new files.",
        "Patch train and validation-visible patterns only; do not use hidden holdout text or SQL.",
        "If editing SQL examples or SQL guidance, make the resulting guidance produce executable Databricks SQL with no bind placeholders such as :time_grp_type, :brand_name, or ${value}; if editing SQL snippets, make them valid literal SQL fragments for their group, not full SELECT/WITH queries.",
        "Do not introduce colon-prefixed placeholder tokens anywhere in changed file content, even when warning against placeholders; describe them in prose instead.",
        "Metadata/table edits may clarify business meaning, synonyms, or entity hints only; do not encode generated query plans, CTE names, result-schema column lists, or aggregation instructions in metadata.",
        "If using full content for a new executable YAML file, it must parse as a YAML mapping before benchmarking: examples require non-empty question and sql fields; SQL snippets require non-empty string/list sql plus a string or repairable display_name. Do not put a top-level content: key inside that YAML; file_edits[].content is already the file body.",
        "If editing instructions/examples/*.yml, the example question must share meaningful non-generic terms with the selected visible target question; unrelated example questions are rejected before benchmarking.",
        "Do not rewrite global trusted-query routing text; local validation rejects trusted-query routing edits. Prefer a narrow executable-literal filter or example edit.",
        "Do not reuse hashes from earlier candidates; hashes change after accepted patches.",
        "Prefer one precise file-level change over a broad multi-file rewrite, and return no_change if a safe file patch is not available.",
    ),
    "skeleton_artifact_patch_v2": (
        "Use top-level file_edits only; set operations to an empty array.",
        "Use exactly one deterministic patch skeleton from the Deterministic Patch Skeleton Contract section.",
        "Return at most one file edit. The edit path, expected_sha256, edit mode, selected marker, family, and find anchor must come from the chosen skeleton exactly.",
        "Serving may fill only file_edits[0].replace plus the existing causal_patch_plan text fields; do not choose a different path, marker, family, edit mode, or hidden context.",
        "Use compact find/replace only: set content and append_content to null, copy find exactly from the skeleton find_anchor_json value, and keep replacement text within the skeleton replacement budget.",
        "The replacement must include a required visible expected token when the skeleton lists one, or the expected_visible_effect must name a listed visible intent label when no token is required.",
        "Do not introduce bind placeholders, hidden holdout references, metadata query plans, CTE names, result-schema column lists, or aggregation instructions.",
        "If no skeleton can safely improve a visible train failure while preserving pass guards, return no_change with operations=[], file_edits=[], and causal_patch_plan=null.",
    ),
    "composite_artifact_patch_v2": (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return a coordinated patch with two to four decomposed-space file edits; one-file patches are not valid in this mode.",
        "Each edit must include path, expected_sha256 copied exactly from the current Decomposed Space Files payload, and a concise reason tied to a visible failure family or pass guard.",
        "Use compact edits for existing files: set content to null and use append_content or exact find plus replace. Use full content only for small new files.",
        "The patch should combine compatible artifact families, for example one executable visible example, one narrow instruction/guidance edit, and one table or column metadata edit.",
        "Use only train failures, visible pass guards, aggregate visible failure families, and April/p02b pattern lessons. Do not use hidden holdout text, SQL, query text, or failed-ID lists.",
        "Reject yourself with no_change if the only plausible edit is a generic instructions/text_instruction.md rewrite.",
        "Do not introduce bind placeholders such as :time_grp_type, :brand_name, :metric_type, or ${value}; SQL examples must be executable Databricks SQL and SQL snippets must be fragments, not full SELECT/WITH queries.",
        "Avoid broad trusted-query routing rewrites. Prefer narrow executable examples, local guidance, and metadata changes that work together for one visible family.",
    ),
    "executable_artifact_patch_v2": (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return one to three strict decomposed-space file edits.",
        "At least one changed file must be an executable asset under instructions/examples/*.yml or instructions/sql_snippets/{filters,expressions,measures}/*.yml.",
        "Instruction, table, or metadata edits are allowed only as support for that executable asset; instruction-only and instruction-plus-metadata patches are invalid in this mode.",
        "Each edit must include path, expected_sha256 copied exactly from the current Decomposed Space Files payload for existing files, and a concise reason tied to a visible train failure or pass guard.",
        "Use compact edits for existing files: set content to null and use append_content or exact find plus replace. Use full content only for small new executable asset files.",
        "Full-content executable YAML files must parse as mappings before benchmarking: examples require non-empty question and sql fields; SQL snippets require non-empty string/list sql plus a string or repairable display_name. Do not put a top-level content: key inside the YAML file body.",
        "For SQL snippet files, include display_name. If omitted, MaxGenie can derive one from description, question, id, or alias, but explicit display_name is preferred.",
        "Examples must contain directly executable Databricks SQL with literal values and no bind placeholders. SQL snippets must be reusable fragments for their group, not full SELECT/WITH queries or DML/DDL statements.",
        "Do not create an executable example by copying a passing guard pattern and changing only a brand, market, metric, time, or entity literal; return no_change unless the visible target failure proves a reusable semantic rule.",
        "MaxGenie compares new executable example SQL against visible pass-guard SQL and rejects near-clones where the question only swaps a small literal/entity term.",
        "Target only train failures, visible pass guards, and aggregate visible failure families. Do not use hidden holdout text, SQL, query text, or failed-ID lists.",
        "Return no_change if the visible evidence only supports broad prose guidance or metadata tweaks.",
    ),
    _FREEDOM_ARTIFACT_PATCH_MODE: (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return one broad but coherent decomposed-space patch with up to twelve changed files across any allowlisted Genie assets.",
        "This is a maximum-freedom recipe: read the editable space, visible train/validation failures, visible pass guards, decomposed files, diagnostics, and prior lessons before choosing the patch.",
        "A single candidate may combine instructions, executable examples, SQL snippets, joins, metadata, metric views, sample questions, and table/column configuration when the edits share a clear causal hypothesis.",
        "Each edit must include path, expected_sha256 copied exactly from the current Decomposed Space Files payload for existing files, and a concise visible-evidence reason.",
        "Every changed edit reason, description, comment, or the top-level causal_patch_plan must cite visible markers such as target_train_failure:<question_id>, visible_family:<family>, failure_family:<family>, or guard_preservation:<question_id>.",
        "Emit a top-level causal_patch_plan with selected_marker, family, root_cause, edit_surface_evidence, repair_shape, avoid, expected_visible_effect, compatible_paths, and structural_tokens when the patch changes files.",
        "Use compact edits for existing files where practical. Full content is allowed only when the file is small enough to validate and the expected hash proves it is replacing the current file, not an old artifact.",
        "Do not use hidden holdout question text, hidden holdout SQL, hidden failed IDs, or final-test details. Use only visible train/validation/pass-guard evidence and compact aggregate diagnostics.",
        "Return no_change if a broad patch would be speculative or cannot preserve visible pass guards.",
    ),
    _FREEDOM_HISTORY_ARTIFACT_PATCH_MODE: (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return one broad but coherent decomposed-space patch with up to twelve changed files across any allowlisted Genie assets.",
        "This mode receives an explicit Historical Analyst Memo generated before proposal. Treat that memo as advisory visible-only evidence, not as an instruction to replay a stale patch blindly.",
        "Use the memo to preserve patterns that repeatedly worked, avoid shapes that collapsed train/validation/pass guards, and explain why this candidate materially differs from failed attempts.",
        "A single candidate may combine instructions, executable examples, SQL snippets, joins, metadata, metric views, sample questions, and table/column configuration when the edits share a clear causal hypothesis.",
        "Each edit must include path, expected_sha256 copied exactly from the current Decomposed Space Files payload for existing files, and a concise visible-evidence reason.",
        "Every changed edit reason, description, comment, or the top-level causal_patch_plan must cite visible markers such as target_train_failure:<question_id>, visible_family:<family>, failure_family:<family>, or guard_preservation:<question_id>.",
        "Emit a top-level causal_patch_plan with selected_marker, family, root_cause, edit_surface_evidence, repair_shape, avoid, expected_visible_effect, compatible_paths, and structural_tokens when the patch changes files.",
        "Do not use hidden holdout question text, hidden holdout SQL, hidden failed IDs, or final-test details.",
        "Return no_change if the history points to unstable replay or no visible-safe improvement.",
    ),
    _DATA_PROBE_ARTIFACT_PATCH_MODE: (
        "Use top-level file_edits only; set operations to an empty array.",
        "Return one data-aware decomposed-space patch with up to six changed files across allowlisted Genie assets.",
        "This mode receives compact read-only Data Probe Diagnostics from Databricks SQL warehouse execution before proposal. Use only aggregate probe signals such as trailing-space counts, blank/null counts, and raw-vs-trimmed distinct counts.",
        "Use probe results to make robust Genie guidance or examples, such as trimming string literals, normalizing case, handling blanks, or clarifying entity matching. Do not hard-code table snapshots or hidden benchmark answers.",
        "Each edit must include path, expected_sha256 copied exactly from the current Decomposed Space Files payload for existing files, and a concise visible-evidence reason.",
        "Every changed edit reason, description, comment, or the top-level causal_patch_plan must cite visible markers such as target_train_failure:<question_id>, visible_family:<family>, failure_family:<family>, or guard_preservation:<question_id>.",
        "Emit a top-level causal_patch_plan with selected_marker, family, root_cause, edit_surface_evidence, repair_shape, avoid, expected_visible_effect, compatible_paths, and structural_tokens when the patch changes files.",
        "Do not request row-level samples, write data, mutate tables, create views, or inspect hidden holdout question text or SQL.",
        "Return no_change if probe diagnostics do not support a visible-safe patch.",
    ),
    "sql_snippet": (
        "Use operation group sql_snippet only.",
        "Allowed operation: add_sql_snippet.",
        "Return one reusable filter, expression, or measure snippet, or no_change if no safe snippet exists.",
    ),
    "metadata": (
        "Use operation group metadata only.",
        "Allowed operations: set_table_description or set_column_config.",
        "Touch at most three tables or columns, or no_change if no metadata-only fix is safe.",
    ),
    "join_spec": (
        "Use operation group join_spec only.",
        "Allowed operation: add_join_spec.",
        "Return one reusable join between existing tables or metric views, or no_change if no safe join exists.",
    ),
    "text_instruction": (
        "Use operation group text_instruction only.",
        "Allowed operation: append_text_instruction.",
        "Return one concise instruction under 900 characters, or no_change if text guidance is unsafe.",
    ),
}


@dataclass(frozen=True)
class ServingCandidateConfig:
    """Configuration for workspace serving endpoint candidate proposals."""

    endpoint: str
    model: str | None = None
    temperature: float | None = None
    max_tokens: int = 16000
    reasoning_effort: str | None = "low"
    response_format: str = "json_schema"
    api_format: str = "auto"
    request_timeout_seconds: int = 900
    retry_timeout_seconds: int = 1800
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("Serving endpoint is required.")
        if self.response_format not in _VALID_RESPONSE_FORMATS:
            raise ValueError(
                "Unsupported serving response format. Choose json_schema, json_object, or none."
            )
        if self.api_format not in _VALID_API_FORMATS:
            raise ValueError("Unsupported serving API format. Choose auto, chat, or responses.")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Serving temperature must be between 0.0 and 2.0.")
        if self.request_timeout_seconds < 60:
            raise ValueError("Serving request timeout must be at least 60 seconds.")
        if self.retry_timeout_seconds < self.request_timeout_seconds:
            raise ValueError("Serving retry timeout must be greater than or equal to request timeout.")


@dataclass(frozen=True)
class ServingCandidateProposal:
    """Raw and parsed candidate proposal returned by the serving endpoint."""

    request: dict[str, Any]
    response: dict[str, Any]
    content: str
    payload: dict[str, Any]
    parse_repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServingCandidateAnalysis:
    """Raw historical analysis returned by the serving endpoint."""

    request: dict[str, Any]
    response: dict[str, Any]
    content: str


class ServingCandidateResponseError(ValueError):
    """Raised when a serving endpoint returns unusable candidate JSON."""

    def __init__(
        self,
        message: str,
        *,
        request: dict[str, Any],
        response: dict[str, Any],
        content: str,
    ) -> None:
        super().__init__(message)
        self.request = request
        self.response = response
        self.content = content


def dominant_visible_failure_family(
    failures: Sequence[Mapping[str, Any]] | None,
) -> str | None:
    """Return the most common ``likely_root_cause_family`` across visible failures.

    Ties are broken deterministically by family name so the selection is stable
    across runs. Returns ``None`` when no failure carries a usable family, which
    callers treat as "no adaptive signal -> keep the comparable default mode".
    """
    if not failures:
        return None
    counts: dict[str, int] = {}
    for failure in failures:
        if not isinstance(failure, Mapping):
            continue
        family = str(failure.get("likely_root_cause_family") or "").strip()
        if not family:
            continue
        counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda family: (-counts[family], family))


def adaptive_candidate_mode_for_family(
    family: str | None,
    *,
    default_mode: str = "artifact_patch_v2",
) -> str:
    """Map a dominant failure family to its best-fit narrow recipe mode.

    Families with a single unambiguous best-fit mode (see ``_ADAPTIVE_MODE_BY_FAMILY``)
    return that mode; every other family -- including an empty/unknown family --
    returns ``default_mode`` so ambiguous cases stay on the strict comparable path.
    """
    if not family:
        return default_mode
    return _ADAPTIVE_MODE_BY_FAMILY.get(family.strip(), default_mode)


def serving_candidate_mode_for_iteration(
    iteration_index: int,
    *,
    prior_pass_guard_regressions: int = 0,
    prior_accepted_fix_regressions: int = 0,
    prior_rejected_candidates: int = 0,
    prior_invalid_responses: int = 0,
    prior_skipped_candidates: int = 0,
    prior_rejected_mode_counts: Mapping[str, int] | None = None,
    adaptive_modes: bool = False,
    dominant_failure_family: str | None = None,
) -> str:
    """Return the default production candidate mode for serving proposals.

    Diagnostic modes remain available by forcing ``serving_candidate_mode`` at the
    call site. Unforced production runs stay on strict artifact patches so result
    evidence is comparable across iterations.

    When ``adaptive_modes`` is enabled (opt-in), a *fresh* scheduling iteration with
    no prior regressions or rejections to react to may instead select the narrow
    recipe mode that best fits ``dominant_failure_family``. The recovery branches
    below keep priority and are untouched, and an unmapped/empty family falls back
    to the same ``artifact_patch_v2`` default -- so the comparable backbone is only
    ever changed deliberately, never silently.
    """
    if prior_invalid_responses >= 1:
        return "artifact_patch_v2"
    if prior_accepted_fix_regressions >= 1:
        if prior_rejected_mode_counts:
            return _least_rejected_mode(
                _ACCEPTED_FIX_REGRESSION_RECOVERY_MODE_SEQUENCE,
                prior_rejected_mode_counts,
            )
        return "artifact_patch_v2"
    if prior_pass_guard_regressions >= 1:
        if prior_rejected_mode_counts:
            return _least_rejected_mode(_RECOVERY_MODE_SEQUENCE, prior_rejected_mode_counts)
        return "artifact_patch_v2"
    if prior_rejected_candidates >= 1 or prior_skipped_candidates >= 1:
        if prior_rejected_mode_counts:
            return _least_rejected_mode(_RECOVERY_MODE_SEQUENCE, prior_rejected_mode_counts)
        return "artifact_patch_v2"
    if iteration_index < 1:
        iteration_index = 1
    if adaptive_modes:
        return adaptive_candidate_mode_for_family(
            dominant_failure_family,
            default_mode=_CANDIDATE_MODE_SEQUENCE[
                (iteration_index - 1) % len(_CANDIDATE_MODE_SEQUENCE)
            ],
        )
    return _CANDIDATE_MODE_SEQUENCE[(iteration_index - 1) % len(_CANDIDATE_MODE_SEQUENCE)]


def _least_rejected_mode(sequence: tuple[str, ...], counts: Mapping[str, int]) -> str:
    return min(sequence, key=lambda mode: (int(counts.get(mode, 0)), sequence.index(mode)))


def serving_candidate_allowed_operation_groups(candidate_mode: str | None) -> frozenset[str] | None:
    """Return operation groups allowed for a serving candidate mode."""
    if candidate_mode is None:
        return None
    return _CANDIDATE_MODE_OPERATION_GROUPS.get(candidate_mode)


def serving_candidate_max_mutation_operations(candidate_mode: str | None) -> int | None:
    """Return the mutation count cap for a serving candidate mode."""
    if candidate_mode in _EXAMPLE_SET_CANDIDATE_MODES:
        return _MAX_EXAMPLE_SET_OPERATIONS
    if serving_candidate_is_artifact_patch_mode(candidate_mode):
        return None
    if candidate_mode == "schema_metadata_aliases":
        return 2
    if candidate_mode == "metadata":
        return _MAX_METADATA_OPERATIONS_PER_CANDIDATE
    if candidate_mode in _CANDIDATE_MODE_OPERATION_GROUPS:
        return 1
    return None


def serving_candidate_is_example_set_mode(candidate_mode: str | None) -> bool:
    """Return whether a candidate mode uses guard-preserving example bundles."""
    return candidate_mode in _EXAMPLE_SET_CANDIDATE_MODES


def serving_candidate_is_artifact_patch_mode(candidate_mode: str | None) -> bool:
    """Return whether a candidate mode uses decomposed file edits."""
    return candidate_mode in {
        "artifact_patch",
        "artifact_patch_v2",
        "skeleton_artifact_patch_v2",
        "composite_artifact_patch_v2",
        "executable_artifact_patch_v2",
        *_BROAD_ARTIFACT_PATCH_MODES,
    }


def serving_candidate_is_strict_artifact_patch_mode(candidate_mode: str | None) -> bool:
    """Return whether artifact patch validation should use v2 strictness."""
    return candidate_mode in {
        "artifact_patch_v2",
        "skeleton_artifact_patch_v2",
        "composite_artifact_patch_v2",
        "executable_artifact_patch_v2",
        *_BROAD_ARTIFACT_PATCH_MODES,
    }


def serving_candidate_is_broad_artifact_patch_mode(candidate_mode: str | None) -> bool:
    """Return whether a candidate mode allows broad visible-evidence file patches."""
    return candidate_mode in _BROAD_ARTIFACT_PATCH_MODES


def serving_candidate_uses_historical_analysis(candidate_mode: str | None) -> bool:
    """Return whether a candidate mode should run a separate history analysis call."""
    return candidate_mode == _FREEDOM_HISTORY_ARTIFACT_PATCH_MODE


def serving_candidate_uses_data_probe(candidate_mode: str | None) -> bool:
    """Return whether a candidate mode should run read-only data-shape probes."""
    return candidate_mode == _DATA_PROBE_ARTIFACT_PATCH_MODE


def serving_candidate_artifact_patch_file_edit_limits(
    candidate_mode: str | None,
) -> tuple[int | None, int | None]:
    """Return min/max changed file edit limits for artifact patch modes."""
    if candidate_mode in {
        _FREEDOM_ARTIFACT_PATCH_MODE,
        _FREEDOM_HISTORY_ARTIFACT_PATCH_MODE,
    }:
        return None, 12
    if candidate_mode == _DATA_PROBE_ARTIFACT_PATCH_MODE:
        return None, 6
    if candidate_mode == "composite_artifact_patch_v2":
        return 2, 4
    if candidate_mode == "executable_artifact_patch_v2":
        return 1, 3
    if candidate_mode == "skeleton_artifact_patch_v2":
        return 1, 1
    if candidate_mode == "artifact_patch_v2":
        return None, 2
    return None, None


def serving_candidate_artifact_patch_requires_executable_asset(
    candidate_mode: str | None,
) -> bool:
    """Return whether a patch mode must edit an executable example/snippet asset."""
    return candidate_mode == "executable_artifact_patch_v2"


def serving_candidate_is_executable_artifact_path(relative_path: str) -> bool:
    """Return whether a decomposed file path is an executable example or snippet asset."""
    return bool(
        re.match(
            r"^instructions/(?:examples/[^/]+\.ya?ml|sql_snippets/(?:filters|expressions|measures)/[^/]+\.ya?ml)$",
            relative_path,
        )
    )


def count_prior_pass_guard_regressions(failure_context: str) -> int:
    """Count prior attempts that explicitly regressed baseline pass guards."""
    count = 0
    for line in failure_context.splitlines():
        if (
            "Regressed pass guards before rollback:" in line
            or "Reason: pass guard regressed:" in line
        ):
            count += 1
    return count


def count_prior_accepted_fix_regressions(failure_context: str) -> int:
    """Count prior attempts that explicitly regressed a previously accepted fix."""
    count = 0
    for line in failure_context.splitlines():
        if "Reason: accepted fix regressed:" in line:
            count += 1
    return count


def count_prior_rejected_candidates(failure_context: str) -> int:
    """Count prior candidate attempts that were benchmark-rejected."""
    count = 0
    for line in failure_context.splitlines():
        stripped = line.strip()
        if stripped.startswith("- **") and "rejected" in stripped:
            count += 1
    return count


def count_prior_rejected_candidate_modes(failure_context: str) -> dict[str, int]:
    """Count benchmark-rejected serving attempts by candidate mode."""
    counts: dict[str, int] = {}
    inside_rejected_attempt = False
    for line in failure_context.splitlines():
        stripped = line.strip()
        if stripped.startswith("- **"):
            inside_rejected_attempt = "→ rejected" in stripped
            continue
        if not inside_rejected_attempt:
            continue
        match = re.search(r"Candidate mode:\s*`([^`]+)`", stripped)
        if match:
            mode = match.group(1)
            counts[mode] = counts.get(mode, 0) + 1
            inside_rejected_attempt = False
            continue
        if stripped.startswith("## "):
            inside_rejected_attempt = False
    return counts


def count_prior_invalid_responses(failure_context: str) -> int:
    """Count prior serving attempts that returned malformed candidate JSON."""
    return failure_context.count("invalid_serving_response_json")


def count_prior_skipped_candidates(failure_context: str) -> int:
    """Count prior candidates that produced no applied candidate."""
    count = 0
    for line in failure_context.splitlines():
        stripped = line.strip()
        if stripped.startswith("- **") and "→ skipped" in stripped:
            count += 1
    return count


def baseline_pass_guard_ids(failure_context: str) -> tuple[str, ...]:
    """Extract visible baseline pass-guard question IDs from failure context."""
    guard_ids: list[str] = []
    inside_guard = False
    for line in failure_context.splitlines():
        if line.startswith("### Guard:"):
            inside_guard = True
            continue
        if not inside_guard:
            continue
        match = re.search(r"- ID:\s*`([^`]+)`", line)
        if match:
            guard_ids.append(match.group(1))
            inside_guard = False
            continue
        if line.startswith("### "):
            inside_guard = False
    return tuple(dict.fromkeys(guard_ids))


def baseline_pass_guard_questions(failure_context: str) -> dict[str, str]:
    """Extract visible baseline pass-guard question text keyed by question ID."""
    guards: dict[str, str] = {}
    current_question: str | None = None
    for line in failure_context.splitlines():
        if line.startswith("### Guard:"):
            current_question = line.removeprefix("### Guard:").strip()
            continue
        if current_question is None:
            continue
        match = re.search(r"- ID:\s*`([^`]+)`", line)
        if match:
            guards[match.group(1)] = current_question
            current_question = None
            continue
        if line.startswith("### "):
            current_question = None
    return guards


def _candidate_response_schema() -> dict[str, Any]:
    file_edit_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "path": {"type": ["string", "null"]},
            "relative_path": {"type": ["string", "null"]},
            "content": {"type": ["string", "null"]},
            "append_content": {"type": ["string", "null"]},
            "find": {"type": ["string", "null"]},
            "replace": {"type": ["string", "null"]},
            "expected_sha256": {"type": ["string", "null"]},
            "reason": {"type": ["string", "null"]},
            "description": {"type": ["string", "null"]},
            "comment": {"type": ["string", "null"]},
        },
        "required": [
            "path",
            "relative_path",
            "content",
            "append_content",
            "find",
            "replace",
            "expected_sha256",
            "reason",
            "description",
            "comment",
        ],
    }
    operation_properties: dict[str, Any] = {
        "op": {
            "type": "string",
            "enum": [
                "append_text_instruction",
                "add_example_sql",
                "add_sql_snippet",
                "add_join_spec",
                "set_table_description",
                "set_column_config",
                "no_change",
            ],
        },
        "content": {"type": ["string", "null"]},
        "id": {"type": ["string", "null"]},
        "question": {"type": ["string", "array", "null"], "items": {"type": "string"}},
        "sql": {"type": ["string", "array", "null"], "items": {"type": "string"}},
        "usage_guidance": {"type": ["string", "array", "null"], "items": {"type": "string"}},
        "group": {"type": ["string", "null"], "enum": ["filters", "expressions", "measures", None]},
        "display_name": {"type": ["string", "null"]},
        "alias": {"type": ["string", "null"]},
        "synonyms": {"type": ["array", "null"], "items": {"type": "string"}},
        "comment": {"type": ["string", "array", "null"], "items": {"type": "string"}},
        "instruction": {"type": ["string", "array", "null"], "items": {"type": "string"}},
        "left_identifier": {"type": ["string", "null"]},
        "left_alias": {"type": ["string", "null"]},
        "right_identifier": {"type": ["string", "null"]},
        "right_alias": {"type": ["string", "null"]},
        "table_identifier": {"type": ["string", "null"]},
        "column_name": {"type": ["string", "null"]},
        "description": {"type": ["string", "array", "null"], "items": {"type": "string"}},
        "exclude": {"type": ["boolean", "null"]},
        "enable_format_assistance": {"type": ["boolean", "null"]},
        "enable_entity_matching": {"type": ["boolean", "null"]},
        "reason": {"type": ["string", "null"]},
    }
    operation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": operation_properties,
        "required": list(operation_properties),
    }
    causal_patch_plan_schema = {
        "type": ["object", "null"],
        "additionalProperties": False,
        "properties": {
            "selected_marker": {"type": ["string", "null"]},
            "family": {"type": ["string", "null"]},
            "root_cause": {"type": ["string", "null"]},
            "edit_surface_evidence": {"type": ["string", "null"]},
            "repair_shape": {"type": ["string", "null"]},
            "avoid": {"type": ["string", "null"]},
            "expected_visible_effect": {"type": ["string", "null"]},
            "compatible_paths": {"type": ["array", "null"], "items": {"type": "string"}},
            "structural_tokens": {"type": ["array", "null"], "items": {"type": "string"}},
        },
        "required": [
            "selected_marker",
            "family",
            "root_cause",
            "edit_surface_evidence",
            "repair_shape",
            "avoid",
            "expected_visible_effect",
            "compatible_paths",
            "structural_tokens",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_name": {"type": "string"},
            "rationale": {"type": "string"},
            "expected_impact": {"type": "string"},
            "risk": {"type": "string"},
            "owner_followups": {"type": "array", "items": {"type": "string"}},
            "causal_patch_plan": causal_patch_plan_schema,
            "file_edits": {"type": ["array", "null"], "items": file_edit_schema},
            "operations": {"type": "array", "items": operation_schema},
        },
        "required": [
            "candidate_name",
            "rationale",
            "expected_impact",
            "risk",
            "owner_followups",
            "causal_patch_plan",
            "file_edits",
            "operations",
        ],
    }


@contextmanager
def _temporary_api_timeouts(api_client: Any, *, request_timeout_seconds: int, retry_timeout_seconds: int) -> Any:
    """Temporarily extend SDK HTTP timeouts for slow reasoning endpoints."""
    base_client = getattr(api_client, "_api_client", None)
    targets: list[tuple[Any, str, Any]] = []
    for target, attr, requested_value in (
        (base_client, "_http_timeout_seconds", float(request_timeout_seconds)),
        (base_client, "_retry_timeout_seconds", int(retry_timeout_seconds)),
    ):
        if target is None or not hasattr(target, attr):
            continue
        current_value = getattr(target, attr)
        targets.append((target, attr, current_value))
        if current_value is None or current_value < requested_value:
            setattr(target, attr, requested_value)
    try:
        yield
    finally:
        for target, attr, original_value in targets:
            setattr(target, attr, original_value)


def _ensure_instruction_container(config: GenieSpaceConfig) -> GenieInstructions:
    if config.instructions is None:
        config.instructions = GenieInstructions(text_instructions=[GenieInstruction(content=[])])
    if not config.instructions.text_instructions:
        config.instructions.text_instructions = [GenieInstruction(content=[])]
    if config.instructions.text_instructions[0].content is None:
        config.instructions.text_instructions[0].content = []
    return config.instructions


def _ensure_sql_snippets(config: GenieSpaceConfig) -> GenieSQLSnippets:
    instructions = _ensure_instruction_container(config)
    if instructions.sql_snippets is None:
        instructions.sql_snippets = GenieSQLSnippets()
    return instructions.sql_snippets


def _normalize_lines(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [] if value == "" else value.splitlines()
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_optional_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else None


def _operation_sql_text(operation: dict[str, Any]) -> str:
    value = operation.get("sql")
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _contains_bind_placeholder(sql_text: str) -> bool:
    return bool(_BIND_PLACEHOLDER_RE.search(sql_text))


def _sql_snippet_statement_start(sql_text: str) -> str | None:
    match = _SQL_STATEMENT_START_RE.search(sql_text)
    if match is None:
        return None
    return match.group(1).lower()


def _sql_snippet_shape_rejection_reason(sql_text: str) -> str | None:
    if _contains_bind_placeholder(sql_text):
        return "sql_snippet_contains_placeholder"
    first_word = _sql_snippet_statement_start(sql_text)
    if first_word in _DISALLOWED_SNIPPET_STATEMENT_STARTS:
        return f"sql_snippet_full_statement_not_allowed:{first_word}"
    return None


def _sql_snippet_option_set_rejection_reason(
    operation: Mapping[str, Any],
    sql_lines: list[str],
) -> str | None:
    if str(operation.get("group") or "").strip() != "filters":
        return None
    nonempty_lines = [line.strip().rstrip(";") for line in sql_lines if line.strip()]
    if len(nonempty_lines) < 2:
        return None

    guidance_text = " ".join(
        str(operation.get(field_name) or "")
        for field_name in (
            "description",
            "display_name",
            "usage_guidance",
            "instruction",
            "comment",
            "reason",
        )
    )
    if _SQL_SNIPPET_OPTION_SET_RE.search(guidance_text):
        return "sql_snippet_filter_has_multiple_alternatives"

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
    if all(independent_predicate_lines):
        return "sql_snippet_filter_has_multiple_alternatives"
    return None


def _operation_field_text(operation: dict[str, Any], fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field_name in fields:
        value = operation.get(field_name)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _operation_mentioned_question_ids(
    operation: dict[str, Any],
    question_ids: Mapping[str, Any],
) -> list[str]:
    operation_text = _operation_field_text(
        operation,
        (
            "reason",
            "id",
            "description",
            "usage_guidance",
            "question",
            "comment",
        ),
    )
    return [question_id for question_id in question_ids if question_id in operation_text]


def _normalize_example_sql_for_gate(sql: str) -> str:
    normalized = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    normalized = re.sub(r"--[^\r\n]*", " ", normalized)
    normalized = normalized.lower().replace("`", "")
    return re.sub(r"\s+", " ", normalized).strip(" ;\n\r\t")


def _valid_id(value: Any) -> str:
    return coerce_artifact_id(value)


def _truncate_json(payload: Any, *, limit: int = 12000) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated, {len(text)} characters total"


def _proposal_safe_trace_context_pack(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(_proposal_safe_trace_value(dict(payload), sensitive_context=False))


def _proposal_safe_trace_value(value: Any, *, sensitive_context: bool) -> Any:
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized_key = key.lower()
            child_sensitive = (
                sensitive_context
                or normalized_key in {
                    "blind_holdout",
                    "final_test",
                    "full",
                    "hidden",
                    "hidden_holdout",
                    "holdout",
                    "overall",
                    "test",
                }
                or "hidden" in normalized_key
                or "holdout" in normalized_key
            )
            if child_sensitive and (
                normalized_key
                in {
                    "error_question_ids",
                    "error_ids",
                    "failed_question_ids",
                    "failed_ids",
                    "hidden_ids",
                    "holdout_ids",
                    "ids",
                    "question_ids",
                    "skipped_question_ids",
                    "skipped_ids",
                    "test_ids",
                }
                or normalized_key.endswith("_ids")
            ):
                safe[key] = []
                safe[f"{key}_policy"] = (
                    "omitted because this section can include hidden holdout question ids"
                )
                continue
            if child_sensitive and normalized_key in {
                "benchmark_sql",
                "expected_query",
                "expected_sql",
                "generated_query",
                "generated_sql",
                "query",
                "query_text",
                "question",
                "question_text",
                "sql",
                "sql_text",
                "statement",
                "text",
            }:
                safe[key] = None
                safe[f"{key}_policy"] = (
                    "omitted because this section can include hidden holdout payloads"
                )
                continue
            safe[key] = _proposal_safe_trace_value(item, sensitive_context=child_sensitive)
        return safe
    if isinstance(value, list):
        return [
            _proposal_safe_trace_value(item, sensitive_context=sensitive_context)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _proposal_safe_trace_value(item, sensitive_context=sensitive_context)
            for item in value
        )
    return deepcopy(value)


def _truncate_text(text: str, *, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated, {len(text)} characters total"


def _format_percent(value: Any) -> str:
    if isinstance(value, bool):
        return "n/a"
    if isinstance(value, int | float):
        return f"{float(value):.2f}%"
    return "n/a"


def _short_id_list(values: list[Any], *, limit: int = 5) -> str:
    ids = [str(value) for value in values if str(value).strip()]
    if not ids:
        return "(none)"
    suffix = "" if len(ids) <= limit else f"; +{len(ids) - limit} more"
    return ", ".join(f"`{value}`" for value in ids[:limit]) + suffix


def _structured_failures_by_id(
    structured_failure_context: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(structured_failure_context, Mapping):
        return {}
    failures = structured_failure_context.get("failures")
    if not isinstance(failures, list):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        question_id = str(failure.get("question_id") or "")
        if question_id:
            output[question_id] = failure
    return output


def _failure_cluster_lines(
    structured_failure_context: Mapping[str, Any] | None,
) -> list[str]:
    failures_by_id = _structured_failures_by_id(structured_failure_context)
    if not failures_by_id:
        return ["- No structured remaining-failure clusters were available; use the markdown context."]

    family_ids = (
        structured_failure_context.get("failure_families")
        if isinstance(structured_failure_context, Mapping)
        else None
    )
    grouped: dict[str, list[str]] = {}
    if isinstance(family_ids, dict):
        for raw_family, raw_ids in family_ids.items():
            if isinstance(raw_ids, list):
                grouped[str(raw_family)] = [str(value) for value in raw_ids if str(value).strip()]
    if not grouped:
        for question_id, failure in failures_by_id.items():
            family = str(failure.get("likely_root_cause_family") or "unknown")
            grouped.setdefault(family, []).append(question_id)

    lines: list[str] = []
    for family, ids in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        family_failures = [failures_by_id[question_id] for question_id in ids if question_id in failures_by_id]
        operation_families: list[str] = []
        risk_levels: list[str] = []
        missing_tokens: list[str] = []
        extra_tokens: list[str] = []
        confidences: list[float] = []
        for failure in family_failures:
            for operation_family in failure.get("suggested_operation_families", []) or []:
                operation_families.append(str(operation_family))
            risk = failure.get("guard_regression_risk")
            if risk:
                risk_levels.append(str(risk))
            confidence = failure.get("confidence")
            if isinstance(confidence, int | float) and not isinstance(confidence, bool):
                confidences.append(float(confidence))
            for token in failure.get("missing_expected_tokens", []) or []:
                missing_tokens.append(str(token))
            for token in failure.get("extra_generated_tokens", []) or []:
                extra_tokens.append(str(token))
        confidence_text = (
            f"{sum(confidences) / len(confidences):.2f}" if confidences else "n/a"
        )
        lines.append(
            f"- `{family}` ids: {_short_id_list(ids)}; "
            f"suggested ops: {_short_id_list(list(dict.fromkeys(operation_families)), limit=4)}; "
            f"guard risk: {_short_id_list(list(dict.fromkeys(risk_levels)), limit=3)}; "
            f"confidence: {confidence_text}."
        )
        if missing_tokens or extra_tokens:
            lines.append(
                "  - Structural hints: "
                f"missing expected tokens {_short_id_list(list(dict.fromkeys(missing_tokens)), limit=6)}; "
                f"extra generated tokens {_short_id_list(list(dict.fromkeys(extra_tokens)), limit=6)}."
            )
    return lines


def _top_structural_tokens(
    failures: list[dict[str, Any]],
    key: str,
    *,
    limit: int = 6,
) -> list[str]:
    counts: dict[str, int] = {}
    for failure in failures:
        values = failure.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            token = str(value).strip()
            if not token:
                continue
            counts[token] = counts.get(token, 0) + 1
    return [
        token
        for token, _count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:limit]
    ]


def _family_failures(
    failures: list[dict[str, Any]],
    family: str,
) -> list[dict[str, Any]]:
    return [
        failure
        for failure in failures
        if str(failure.get("likely_root_cause_family") or "") == family
    ]


def _generalized_visible_family_guidance_lines(
    structured_failure_context: Mapping[str, Any] | None,
) -> list[str]:
    failures_by_id = _structured_failures_by_id(structured_failure_context)
    if not failures_by_id:
        return [
            "- No structured visible failure-family guidance is available; rely on compact failure evidence and preserve pass guards."
        ]
    failures = list(failures_by_id.values())
    lines: list[str] = []

    literal_failures = _family_failures(failures, "literal_or_parameter_mismatch")
    if literal_failures:
        placeholders = [
            token
            for token in _top_structural_tokens(literal_failures, "extra_generated_tokens")
            if token.startswith(":") or token.startswith("${")
        ]
        detail = (
            f" Repeated placeholders: {_short_id_list(placeholders)}."
            if placeholders
            else ""
        )
        lines.append(
            "- Literalization family: generated SQL is using bind-style placeholders where visible expected SQL uses concrete literals."
            f"{detail} Prefer durable guidance or snippets that produce executable literal filters, not more copied benchmark examples."
        )

    comparison_failures = _family_failures(failures, "percentage_or_comparison_metric")
    if comparison_failures:
        missing = _top_structural_tokens(comparison_failures, "missing_expected_tokens")
        lines.append(
            "- Period-comparison family: preserve both current and comparison-period measures, denominator semantics, and percent/change aliases."
            f" Repeated missing expected tokens: {_short_id_list(missing)}."
        )

    time_failures = _family_failures(failures, "time_filter_mismatch")
    if time_failures:
        missing = _top_structural_tokens(time_failures, "missing_expected_tokens")
        extra = _top_structural_tokens(time_failures, "extra_generated_tokens")
        lines.append(
            "- Time-window family: align weekly/monthly/quarterly bucket filters and output aliases to the visible expected pattern."
            f" Missing tokens: {_short_id_list(missing)}; extra generated tokens: {_short_id_list(extra)}."
        )

    output_failures = _family_failures(failures, "selected_metric_or_column_mismatch")
    if output_failures:
        missing = _top_structural_tokens(output_failures, "missing_expected_tokens")
        extra = _top_structural_tokens(output_failures, "extra_generated_tokens")
        lines.append(
            "- Output-schema family: preserve expected selected columns and aliases; do not swap lookup/existence checks or alternate time descriptors for the requested result schema."
            f" Missing tokens: {_short_id_list(missing)}; extra generated tokens: {_short_id_list(extra)}."
        )

    runtime_failures = _family_failures(failures, "runtime_or_generation_error")
    if runtime_failures:
        methods = list(
            dict.fromkeys(
                str(failure.get("comparison_method") or failure.get("issue_class") or "unknown")
                for failure in runtime_failures
            )
        )
        lines.append(
            "- Runtime-generation family: fix invalid or non-select generated SQL before adding broader examples."
            f" Observed methods/classes: {_short_id_list(methods)}."
        )

    return lines or [
        "- No dominant generalized visible family emerged; target one visible failure cluster and preserve every pass guard."
    ]


def _pass_guard_summary_lines(
    structured_failure_context: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(structured_failure_context, Mapping):
        return [
            "- No structured pass guards were available; preserve all currently passing visible queries."
        ]
    pass_guards = structured_failure_context.get("pass_guards")
    if not isinstance(pass_guards, list) or not pass_guards:
        return [
            "- No structured pass guards were available; preserve all currently passing visible queries."
        ]

    lines: list[str] = []
    for guard in pass_guards[:8]:
        if not isinstance(guard, dict):
            continue
        question_id = str(guard.get("question_id") or "?")
        source = str(guard.get("source") or "unknown")
        latest_status = str(guard.get("latest_status") or "unknown")
        reason = str(guard.get("guard_reason") or source)
        accepted_candidate = guard.get("accepted_candidate")
        suffix = f"; accepted candidate `{accepted_candidate}`" if accepted_candidate else ""
        lines.append(
            f"- `{question_id}` source=`{source}` latest=`{latest_status}` "
            f"reason=`{reason}`{suffix}."
        )
    return lines or [
        "- No structured pass guards were available; preserve all currently passing visible queries."
    ]


def _artifact_patch_visible_effect_contract_lines(
    *,
    structured_failure_context: Mapping[str, Any] | None,
    candidate_mode: str | None,
) -> list[str]:
    if candidate_mode not in _STRICT_ARTIFACT_PATCH_MODES:
        return []
    if not isinstance(structured_failure_context, Mapping):
        return []

    failures_by_id = _structured_failures_by_id(structured_failure_context)
    pass_guards = structured_failure_context.get("pass_guards")
    pass_guards = pass_guards if isinstance(pass_guards, list) else []
    families: set[str] = set()
    raw_families = structured_failure_context.get("failure_families")
    if isinstance(raw_families, Mapping):
        families.update(str(family) for family in raw_families if str(family).strip())
    tokens_by_family: dict[str, list[str]] = {}
    for failure in failures_by_id.values():
        family = str(failure.get("likely_root_cause_family") or "").strip()
        if family:
            families.add(family)
            tokens_by_family.setdefault(family, []).extend(
                _artifact_patch_contract_tokens(failure)
            )
    if not failures_by_id and not families and not pass_guards:
        return []

    lines: list[str] = [
        "### Artifact Patch Visible-Effect Contract",
        "Every changed strict file edit must cite one exact marker below in its own reason, description, or comment; top-level rationale/expected_impact is not enough for per-edit causal attribution.",
        "Use `target_train_failure:<question_id>` for the primary visible train failure, `visible_family:<family>` or `failure_family:<family>` for a family-level edit, and `guard_preservation:<question_id>` only for explicit guard-preserving edits.",
        "Guard-preservation markers are support markers only; a default strict patch still needs one target failure or visible family marker as its improvement plan.",
        "If the chosen target has guard_risk=`high` and guard-preservation markers are listed, cite at least one `guard_preservation:<question_id>` support marker in rationale, expected_impact, risk, or an edit reason.",
        "If the chosen family plan has guard_risk=`high` and guard-preservation markers are listed, cite at least one `guard_preservation:<question_id>` support marker in rationale, expected_impact, risk, or an edit reason.",
        "For text-guidance edits, mention at least one listed structural token when tokens are shown for the cited target or family.",
        "Do not invent marker IDs. If no marker below fits the patch, return no_change.",
        "",
        "Allowed target failure markers:",
    ]
    if failures_by_id:
        for question_id, failure in sorted(failures_by_id.items())[:8]:
            family = str(failure.get("likely_root_cause_family") or "unknown")
            risk = str(failure.get("guard_regression_risk") or "unknown")
            suggested = [
                str(value)
                for value in failure.get("suggested_operation_families", []) or []
                if str(value).strip()
            ]
            tokens = _artifact_patch_contract_tokens(failure)
            token_text = f" tokens={_short_id_list(tokens, limit=6)}" if tokens else ""
            lines.append(
                f"- `target_train_failure:{question_id}` family=`{family}` "
                f"guard_risk=`{risk}` suggested_ops={_short_id_list(suggested, limit=4)}"
                f"{token_text}."
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "Allowed visible family markers:"])
    if families:
        for family in sorted(families)[:8]:
            family_tokens = list(dict.fromkeys(tokens_by_family.get(family, [])))
            token_text = (
                f" tokens={_short_id_list(family_tokens, limit=6)}"
                if family_tokens
                else ""
            )
            lines.append(
                f"- `visible_family:{family}` or `failure_family:{family}`{token_text}."
            )
    else:
        lines.append("- (none)")

    lines.extend(["", "Allowed guard preservation markers:"])
    guard_lines = 0
    for guard in pass_guards[:8]:
        if not isinstance(guard, Mapping):
            continue
        question_id = str(guard.get("question_id") or "").strip()
        if not question_id:
            continue
        source = str(guard.get("source") or "unknown")
        latest_status = str(guard.get("latest_status") or "unknown")
        evidence_line = _artifact_patch_guard_evidence_line(guard)
        evidence_text = f" evidence=({evidence_line})" if evidence_line else ""
        lines.append(
            f"- `guard_preservation:{question_id}` source=`{source}` "
            f"latest=`{latest_status}`{evidence_text}."
        )
        guard_lines += 1
    if guard_lines == 0:
        lines.append("- (none)")

    lines.append("")
    return lines


def _artifact_patch_path_hints_for_operations(
    operation_families: list[str],
) -> list[str]:
    hints: list[str] = []
    for operation_family in operation_families:
        for hint in _ARTIFACT_PATCH_OPERATION_PATH_HINTS.get(operation_family, ()):
            hints.append(hint)
    return list(dict.fromkeys(hints))


def _artifact_patch_target_plan_lines(
    *,
    structured_failure_context: Mapping[str, Any] | None,
    decomposed_space_payload: Mapping[str, Any] | None,
    candidate_mode: str | None,
    failure_context: str = "",
) -> list[str]:
    if candidate_mode not in _STRICT_ARTIFACT_PATCH_MODES:
        return []
    failures_by_id = _structured_failures_by_id(structured_failure_context)
    if not failures_by_id:
        return []

    risk_rank = {"low": 0, "medium": 1, "high": 2}

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, str]:
        question_id, failure = item
        risk = str(failure.get("guard_regression_risk") or "unknown").lower()
        confidence = failure.get("confidence")
        confidence_value = (
            float(confidence)
            if isinstance(confidence, int | float) and not isinstance(confidence, bool)
            else 0.0
        )
        return (risk_rank.get(risk, 3), -confidence_value, question_id)

    repeated_new_executable_collapse = (
        _artifact_patch_has_repeated_new_executable_collapse(failure_context)
    )

    lines: list[str] = [
        "### Strict Artifact Patch Target Plan",
        "Choose exactly one plan below before creating file_edits. Mutating default artifact_patch_v2 responses must include top-level causal_patch_plan; put the chosen plan there with selected_marker, family, root_cause, edit_surface_evidence, repair_shape, avoid, expected_visible_effect, compatible_paths, and structural_tokens. selected_marker must be an exact visible marker string such as target_train_failure:<id>, visible_family:<family>, or failure_family:<family>, not a raw question ID. family must be non-empty and match the chosen marker or target family. root_cause must be non-empty and name one root_cause_anchors value from the chosen plan. edit_surface_evidence must be non-empty and name one edit_surface_anchors value from the chosen plan to explain why the selected artifact path is the right edit surface. compatible_paths must be non-empty and copied from the chosen plan. If existing_edit_files are listed, prefer one of those exact paths; if new_file_allowed=false, do not invent a new path. When prior lessons report repeated new executable artifact collapse, executable example/snippet plans set new_file_allowed=false and must use listed existing edit files or return no_change. Patch archetype lines are guidance only; encode the chosen shape in repair_shape, edit_surface_evidence, expected_visible_effect, and file_edits, not as extra JSON fields. sql_delta_intents are normalized visible SQL/result delta labels; when listed, expected_visible_effect should name one chosen intent and the concrete token/result evidence it will change. repair_shape and avoid must be non-empty and should follow that plan's Causal repair recipe and Avoid guidance. Copy structural_tokens only from the chosen plan's shared_token_hints when shared_token_hints are listed; otherwise copy structural_tokens only from that plan's token_hints. For target-level executable examples or SQL snippets, prefer expected_token_hints over generated_token_hints; if expected_token_hints are listed, include one expected token in structural_tokens and changed content. For family-level executable examples or SQL snippets, prefer shared_expected_token_hints over shared_generated_token_hints; if shared_expected_token_hints are listed, include one shared expected token in structural_tokens and changed content. For target-level literal_or_parameter_mismatch executable examples or SQL snippets, choose the visible expected literal from expected_token_hints for structural_tokens and put that literal in the changed content; using only the generated placeholder or column token is rejected. Make expected_visible_effect name the concrete visible SQL/result delta the patch should change, including one shared_token_hints value when shared_token_hints are listed, otherwise including one token_hints value when token_hints are listed, and include one listed structural_tokens value in the new content of each changed edit. Each changed file_edits[].reason, description, or comment must use that plan's marker, stay within compatible artifact paths, and mention one listed token when editing text instructions. If no plan fits, return no_change with causal_patch_plan set to null.",
    ]
    for index, (question_id, failure) in enumerate(
        sorted(failures_by_id.items(), key=sort_key)[:4],
        start=1,
    ):
        family = str(failure.get("likely_root_cause_family") or "unknown")
        risk = str(failure.get("guard_regression_risk") or "unknown")
        suggested = [
            str(value)
            for value in failure.get("suggested_operation_families", []) or []
            if str(value).strip()
        ]
        path_hints = _artifact_patch_path_hints_for_operations(suggested)
        expected_tokens = [
            str(value).strip()
            for value in failure.get("missing_expected_tokens", []) or []
            if str(value).strip()
        ]
        generated_tokens = [
            str(value).strip()
            for value in failure.get("extra_generated_tokens", []) or []
            if str(value).strip()
        ]
        tokens = _artifact_patch_contract_tokens(failure)
        question_terms = _artifact_patch_target_question_terms(failure)
        sql_delta_intents = _artifact_patch_sql_delta_intents(failure)
        existing_edit_files = _artifact_patch_existing_edit_files_for_plan(
            decomposed_space_payload=decomposed_space_payload,
            path_hints=path_hints,
            tokens=tokens,
        )
        target_file_excerpts = _artifact_patch_existing_edit_file_excerpts_for_plan(
            decomposed_space_payload=decomposed_space_payload,
            existing_edit_files=existing_edit_files,
            tokens=tokens,
        )
        prefer_existing_executable_file = (
            repeated_new_executable_collapse
            and bool(set(suggested) & _EXECUTABLE_ARTIFACT_OPERATION_FAMILIES)
        )
        repair_recipe = _artifact_patch_repair_recipe(
            family=family,
            operation_families=suggested,
            tokens=tokens,
        )
        patch_archetype = _artifact_patch_patch_archetype(
            family=family,
            operation_families=suggested,
            expected_tokens=expected_tokens,
            generated_tokens=generated_tokens,
            prefer_existing_executable_file=prefer_existing_executable_file,
            has_existing_edit_files=bool(existing_edit_files),
        )
        patch_template = _artifact_patch_patch_template_hint(
            operation_families=suggested,
            prefer_existing_executable_file=prefer_existing_executable_file,
            has_existing_edit_files=bool(existing_edit_files),
        )
        recipe_anchors = artifact_patch_causal_recipe_anchors(
            family=family,
            operation_families=suggested,
            tokens=tokens,
        )
        new_file_allowed = _artifact_patch_plan_allows_new_file(
            suggested
        ) and not prefer_existing_executable_file
        lines.append(
            f"- Plan {index}: marker=`target_train_failure:{question_id}`; "
            f"family=`{family}`; guard_risk=`{risk}`; "
            f"suggested_ops={_short_id_list(suggested, limit=4)}; "
            f"compatible_paths={_short_id_list(path_hints, limit=5)}; "
            f"existing_edit_files={_short_id_list(existing_edit_files, limit=5)}; "
            f"new_file_allowed=`{str(new_file_allowed).lower()}`; "
            f"target_question_terms={_short_id_list(question_terms, limit=8)}; "
            f"sql_delta_intents={_short_id_list(sql_delta_intents, limit=6)}; "
            f"expected_token_hints={_short_id_list(expected_tokens, limit=6)}; "
            f"generated_token_hints={_short_id_list(generated_tokens, limit=6)}; "
            f"token_hints={_short_id_list(tokens, limit=6)}; "
            f"root_cause_anchors={_short_id_list(recipe_anchors['root_cause'], limit=6)}; "
            f"edit_surface_anchors={_short_id_list(recipe_anchors['edit_surface'], limit=6)}."
        )
        evidence_line = _artifact_patch_target_evidence_line(failure)
        if evidence_line:
            lines.append(f"  - Visible target evidence: {evidence_line}")
        lines.append(f"  - Patch archetype: {patch_archetype}")
        lines.append(f"  - Patch template hint: {patch_template}")
        if target_file_excerpts:
            lines.append("  - Current edit-file excerpts for compact find/replace:")
            for excerpt in target_file_excerpts:
                lines.append(
                    "    - "
                    f"path=`{excerpt['path']}`; "
                    f"expected_sha256=`{excerpt['expected_sha256']}`; "
                    f"excerpt=`{excerpt['excerpt']}`"
                )
        lines.append(
            f"  - Causal repair recipe: {repair_recipe['repair']} "
            f"Avoid: {repair_recipe['avoid']}"
        )
        if risk.lower() == "high":
            lines.append(
                "  - High guard risk: include a visible guard_preservation:<question_id> support marker if any guard markers are listed."
            )
    family_plans = _artifact_patch_family_target_plan_summaries(
        structured_failure_context=structured_failure_context,
        failures_by_id=failures_by_id,
    )
    if family_plans:
        lines.append("- Family-level plans below are only for repeated visible failures with the same family and compatible edit surface.")
    for index, family_plan in enumerate(family_plans[:3], start=1):
        family = family_plan["family"]
        suggested = family_plan["suggested_ops"]
        tokens = family_plan["tokens"]
        path_hints = _artifact_patch_path_hints_for_operations(suggested)
        if not path_hints:
            continue
        existing_edit_files = _artifact_patch_existing_edit_files_for_plan(
            decomposed_space_payload=decomposed_space_payload,
            path_hints=path_hints,
            tokens=tokens,
        )
        target_file_excerpts = _artifact_patch_existing_edit_file_excerpts_for_plan(
            decomposed_space_payload=decomposed_space_payload,
            existing_edit_files=existing_edit_files,
            tokens=tokens,
        )
        prefer_existing_executable_file = (
            repeated_new_executable_collapse
            and bool(set(suggested) & _EXECUTABLE_ARTIFACT_OPERATION_FAMILIES)
        )
        recipe_anchors = artifact_patch_causal_recipe_anchors(
            family=family,
            operation_families=suggested,
            tokens=tokens,
        )
        repair_recipe = _artifact_patch_repair_recipe(
            family=family,
            operation_families=suggested,
            tokens=tokens,
        )
        patch_archetype = _artifact_patch_patch_archetype(
            family=family,
            operation_families=suggested,
            expected_tokens=family_plan["expected_tokens"],
            generated_tokens=family_plan["generated_tokens"],
            shared_expected_tokens=family_plan["shared_expected_tokens"],
            shared_generated_tokens=family_plan["shared_generated_tokens"],
            prefer_existing_executable_file=prefer_existing_executable_file,
            has_existing_edit_files=bool(existing_edit_files),
        )
        patch_template = _artifact_patch_patch_template_hint(
            operation_families=suggested,
            prefer_existing_executable_file=prefer_existing_executable_file,
            has_existing_edit_files=bool(existing_edit_files),
        )
        new_file_allowed = _artifact_patch_plan_allows_new_file(
            suggested
        ) and not prefer_existing_executable_file
        lines.append(
            f"- Family Plan {index}: marker=`visible_family:{family}`; "
            f"family=`{family}`; member_ids={_short_id_list(family_plan['member_ids'], limit=5)}; "
            f"guard_risk=`{family_plan['guard_risk']}`; "
            f"high_guard_risk_member_ids={_short_id_list(family_plan['high_guard_risk_member_ids'], limit=5)}; "
            f"suggested_ops={_short_id_list(suggested, limit=4)}; "
            f"compatible_paths={_short_id_list(path_hints, limit=5)}; "
            f"existing_edit_files={_short_id_list(existing_edit_files, limit=5)}; "
            f"new_file_allowed=`{str(new_file_allowed).lower()}`; "
            f"member_question_terms={_short_id_list(family_plan['question_terms'], limit=10)}; "
            f"sql_delta_intents={_short_id_list(family_plan['sql_delta_intents'], limit=8)}; "
            f"expected_token_hints={_short_id_list(family_plan['expected_tokens'], limit=8)}; "
            f"generated_token_hints={_short_id_list(family_plan['generated_tokens'], limit=8)}; "
            f"token_hints={_short_id_list(tokens, limit=8)}; "
            f"shared_expected_token_hints={_short_id_list(family_plan['shared_expected_tokens'], limit=6)}; "
            f"shared_generated_token_hints={_short_id_list(family_plan['shared_generated_tokens'], limit=6)}; "
            f"shared_token_hints={_short_id_list(family_plan['shared_tokens'], limit=6)}; "
            f"root_cause_anchors={_short_id_list(recipe_anchors['root_cause'], limit=6)}; "
            f"edit_surface_anchors={_short_id_list(recipe_anchors['edit_surface'], limit=6)}."
        )
        evidence_line = _artifact_patch_family_evidence_line(family_plan)
        if evidence_line:
            lines.append(f"  - Visible family evidence: {evidence_line}")
        lines.append(f"  - Patch archetype: {patch_archetype}")
        lines.append(f"  - Patch template hint: {patch_template}")
        member_evidence_lines = _artifact_patch_family_member_evidence_lines(
            family_plan
        )
        if member_evidence_lines:
            lines.append("  - Visible shared-member evidence:")
            lines.extend(f"    - {line}" for line in member_evidence_lines)
        if family_plan["shared_tokens"]:
            lines.append(
                "  - Shared family token evidence: choose at least one shared_token_hints value for structural_tokens and expected_visible_effect."
            )
        if family_plan["shared_expected_tokens"]:
            lines.append(
                "  - Shared expected-token evidence: choose at least one shared_expected_token_hints value for executable structural_tokens and changed content."
            )
        if target_file_excerpts:
            lines.append("  - Current edit-file excerpts for compact find/replace:")
            for excerpt in target_file_excerpts:
                lines.append(
                    "    - "
                    f"path=`{excerpt['path']}`; "
                    f"expected_sha256=`{excerpt['expected_sha256']}`; "
                    f"excerpt=`{excerpt['excerpt']}`"
                )
        lines.append(
            f"  - Causal repair recipe: {repair_recipe['repair']} "
            f"Avoid: {repair_recipe['avoid']}"
        )
        if family_plan["guard_risk"] == "high":
            lines.append(
                "  - High family guard risk: include a visible guard_preservation:<question_id> support marker if any guard markers are listed."
            )
    if candidate_mode == "executable_artifact_patch_v2":
        lines.append(
            "- Executable artifact patch mode requires at least one compatible executable asset edit under instructions/examples or instructions/sql_snippets."
        )
    lines.append("")
    return lines


def _artifact_patch_self_checklist_lines(candidate_mode: str | None) -> list[str]:
    if candidate_mode not in _STRICT_ARTIFACT_PATCH_MODES:
        return []
    return [
        "### Strict Artifact Patch Self-Checklist",
        "Before returning JSON, internally check every item below. Do not add checklist fields to the response; encode the result through causal_patch_plan and file_edits.",
        "- Hidden holdout exclusion: no hidden holdout question text, SQL, query text, or failed IDs appear in rationale, causal_patch_plan, operations, file_edits, or comments.",
        "- Selected marker attribution: causal_patch_plan.selected_marker is one exact visible marker, and every changed file_edit reason, description, or comment repeats that same marker.",
        "- Patch shape: file_edits follow the chosen plan's patch archetype and patch template hint; existing files use compact append_content or exact find/replace edits, not full-file rewrites.",
        "- Executable YAML shape: full-content instructions/examples/*.yml and instructions/sql_snippets/*/*.yml edits parse as YAML mappings with required non-empty fields before benchmarking; do not wrap the YAML in a top-level content: key.",
        "- Hash and path safety: existing-file edits copy the current expected_sha256, unused nullable fields are null, and path/relative_path do not disagree.",
        "- Expected-token safety: executable examples/snippets include the expected_token_hints or shared_expected_token_hints value in structural_tokens and changed content when listed.",
        "- Pass-guard safety: high guard-risk target/family plans include a visible guard_preservation:<question_id> support marker when guard markers are available.",
        "- Non-clone safety: executable examples/snippets are not pass-guard clones with only brand, market, metric, time, or entity literals swapped.",
        "- Causal specificity: if any item cannot be satisfied from visible evidence, return no_change with causal_patch_plan set to null.",
        "",
    ]


def _artifact_patch_family_target_plan_summaries(
    *,
    structured_failure_context: Mapping[str, Any] | None,
    failures_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(structured_failure_context, Mapping):
        return []
    raw_family_ids = structured_failure_context.get("failure_families")
    grouped_ids: dict[str, list[str]] = {}
    if isinstance(raw_family_ids, Mapping):
        for raw_family, raw_ids in raw_family_ids.items():
            family = str(raw_family).strip()
            if not family or not isinstance(raw_ids, list):
                continue
            grouped_ids[family] = [
                str(value).strip()
                for value in raw_ids
                if str(value).strip() in failures_by_id
            ]
    if not grouped_ids:
        for question_id, failure in failures_by_id.items():
            family = str(failure.get("likely_root_cause_family") or "").strip()
            if family:
                grouped_ids.setdefault(family, []).append(question_id)

    plans: list[dict[str, Any]] = []
    for family, member_ids in grouped_ids.items():
        member_ids = list(dict.fromkeys(member_ids))
        if len(member_ids) < 2:
            continue
        failures = [
            failures_by_id[question_id]
            for question_id in member_ids
            if question_id in failures_by_id
        ]
        op_counts: dict[str, int] = {}
        for failure in failures:
            failure_ops = {
                str(value).strip()
                for value in failure.get("suggested_operation_families", []) or []
                if str(value).strip()
            }
            for operation_family in failure_ops:
                op_counts[operation_family] = op_counts.get(operation_family, 0) + 1
        suggested_ops = sorted(
            operation_family
            for operation_family, count in op_counts.items()
            if count >= 2
        )
        path_hints = _artifact_patch_path_hints_for_operations(suggested_ops)
        if not path_hints:
            continue
        failures = [
            failure
            for failure in failures
            if set(failure.get("suggested_operation_families", []) or [])
            & set(suggested_ops)
        ]
        member_ids = [
            question_id
            for question_id in member_ids
            if failures_by_id.get(question_id) in failures
        ]
        if len(member_ids) < 2:
            continue
        risk_rank = {"low": 0, "medium": 1, "high": 2}
        guard_risks = [
            str(failure.get("guard_regression_risk") or "").strip().lower()
            for failure in failures
            if str(failure.get("guard_regression_risk") or "").strip()
        ]
        guard_risk = (
            max(guard_risks, key=lambda risk: risk_rank.get(risk, -1))
            if guard_risks
            else "unknown"
        )
        if guard_risk not in risk_rank:
            guard_risk = "unknown"
        high_guard_risk_member_ids = [
            question_id
            for question_id in member_ids
            if str(
                failures_by_id[question_id].get("guard_regression_risk") or ""
            ).strip().lower()
            == "high"
        ]
        tokens = sorted(
            {
                token
                for failure in failures
                for token in _artifact_patch_contract_tokens(failure)
            }
        )
        expected_tokens = sorted(
            {
                str(token).strip()
                for failure in failures
                for token in failure.get("missing_expected_tokens", []) or []
                if str(token).strip()
            }
        )
        generated_tokens = sorted(
            {
                str(token).strip()
                for failure in failures
                for token in failure.get("extra_generated_tokens", []) or []
                if str(token).strip()
            }
        )
        token_counts: dict[str, int] = {}
        expected_token_counts: dict[str, int] = {}
        generated_token_counts: dict[str, int] = {}
        for failure in failures:
            failure_tokens = set(_artifact_patch_contract_tokens(failure))
            for token in failure_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            expected_failure_tokens = {
                str(token).strip()
                for token in failure.get("missing_expected_tokens", []) or []
                if str(token).strip()
            }
            for token in expected_failure_tokens:
                expected_token_counts[token] = expected_token_counts.get(token, 0) + 1
            generated_failure_tokens = {
                str(token).strip()
                for token in failure.get("extra_generated_tokens", []) or []
                if str(token).strip()
            }
            for token in generated_failure_tokens:
                generated_token_counts[token] = generated_token_counts.get(token, 0) + 1
        shared_tokens = sorted(
            token for token, count in token_counts.items() if count >= 2
        )
        shared_expected_tokens = sorted(
            token for token, count in expected_token_counts.items() if count >= 2
        )
        shared_generated_tokens = sorted(
            token for token, count in generated_token_counts.items() if count >= 2
        )
        question_terms = sorted(
            {
                term
                for failure in failures
                for term in _artifact_patch_target_question_terms(failure, limit=12)
            }
        )
        sql_delta_intents = sorted(
            {
                intent
                for failure in failures
                for intent in _artifact_patch_sql_delta_intents(failure)
            }
        )
        plans.append(
            {
                "family": family,
                "member_ids": member_ids,
                "failures": failures,
                "guard_risk": guard_risk,
                "high_guard_risk_member_ids": high_guard_risk_member_ids,
                "suggested_ops": suggested_ops,
                "tokens": tokens,
                "expected_tokens": expected_tokens,
                "generated_tokens": generated_tokens,
                "shared_tokens": shared_tokens,
                "shared_expected_tokens": shared_expected_tokens,
                "shared_generated_tokens": shared_generated_tokens,
                "question_terms": question_terms,
                "sql_delta_intents": sql_delta_intents,
            }
        )
    return sorted(plans, key=lambda plan: (-len(plan["member_ids"]), plan["family"]))


def _artifact_patch_family_member_evidence_lines(
    family_plan: Mapping[str, Any],
) -> list[str]:
    failures = family_plan.get("failures")
    member_ids = family_plan.get("member_ids")
    if not isinstance(failures, list) or not isinstance(member_ids, list):
        return []

    lines: list[str] = []
    for question_id, failure in zip(member_ids, failures, strict=False):
        if not isinstance(failure, Mapping):
            continue
        evidence_line = _artifact_patch_target_evidence_line(failure)
        if not evidence_line:
            continue
        lines.append(f"id=`{question_id}`; {evidence_line}")
        if len(lines) >= 3:
            break
    return lines


def _artifact_patch_family_evidence_line(family_plan: Mapping[str, Any]) -> str:
    failures = family_plan.get("failures")
    if not isinstance(failures, list):
        return ""
    fields = [
        f"failure_count=`{len(failures)}`",
        f"member_ids={_short_id_list(list(family_plan.get('member_ids', [])), limit=5)}",
    ]
    sample_questions: list[str] = []
    methods: list[str] = []
    for failure in failures[:3]:
        if not isinstance(failure, Mapping):
            continue
        question = _compact_inline_text(failure.get("question"), limit=120)
        if question:
            sample_questions.append(question)
        method = _compact_inline_text(failure.get("comparison_method"), limit=80)
        if method and method not in methods:
            methods.append(method)
    if sample_questions:
        fields.append(f"sample_questions=\"{' | '.join(sample_questions)}\"")
    if methods:
        fields.append(f"methods={_short_id_list(methods, limit=4)}")
    return "; ".join(fields)


# Targeted failures get the full expected/generated SQL and the full execution error
# (no ~260-char clip): the proposer needs to see the exact divergence — for example an
# unbound ``:parameter`` and the error line that names it — to fix the failure properly.
# Bounds are generous so real Genie SQL/errors are rendered in full while still guarding
# against a pathological multi-kilobyte payload.
_TARGET_EVIDENCE_SQL_LIMIT = 4000
_TARGET_EVIDENCE_ERROR_LIMIT = 2000


def _artifact_patch_target_evidence_line(failure: Mapping[str, Any]) -> str:
    fields: list[str] = []
    question = _compact_inline_text(failure.get("question"), limit=180)
    if question:
        fields.append(f"question=\"{question}\"")
    issue_class = _compact_inline_text(failure.get("issue_class"), limit=80)
    if issue_class:
        fields.append(f"issue=`{issue_class}`")
    method = _compact_inline_text(failure.get("comparison_method"), limit=80)
    if method:
        fields.append(f"method=`{method}`")
    expected_sql = _compact_inline_text(failure.get("expected_sql"), limit=_TARGET_EVIDENCE_SQL_LIMIT)
    if expected_sql:
        fields.append(f"expected_sql=`{expected_sql}`")
    generated_sql = _compact_inline_text(failure.get("generated_sql"), limit=_TARGET_EVIDENCE_SQL_LIMIT)
    if generated_sql:
        fields.append(f"generated_sql=`{generated_sql}`")
    error_text = _compact_inline_text(
        failure.get("error_message") or failure.get("comparison_details"),
        limit=_TARGET_EVIDENCE_ERROR_LIMIT,
    )
    if error_text:
        fields.append(f"error=`{error_text}`")
    # Visible expected answer-key rows (precomputed): the target result the generated
    # SQL must reproduce. Attached upstream from the answer key for visible questions
    # only; holdout expected rows are never looked up here.
    expected_preview = _compact_inline_text(
        failure.get("expected_result_preview"), limit=_TARGET_EVIDENCE_ERROR_LIMIT
    )
    if expected_preview:
        fields.append(f"expected_result={expected_preview}")
    return "; ".join(fields)


def _artifact_patch_guard_evidence_line(guard: Mapping[str, Any]) -> str:
    fields: list[str] = []
    question = _compact_inline_text(guard.get("question"), limit=140)
    if question:
        fields.append(f"question=\"{question}\"")
    expected_sql = _compact_inline_text(guard.get("expected_sql"), limit=220)
    if expected_sql:
        fields.append(f"expected_sql=`{expected_sql}`")
    latest_status = _compact_inline_text(guard.get("latest_status"), limit=60)
    if latest_status:
        fields.append(f"latest=`{latest_status}`")
    return "; ".join(fields)


def _artifact_patch_target_question_terms(
    failure: Mapping[str, Any],
    *,
    limit: int = 8,
) -> list[str]:
    question = str(failure.get("question") or "")
    if not question.strip():
        return []
    return sorted(question_overlap_terms(question))[:limit]


def _artifact_patch_sql_delta_intents(failure: Mapping[str, Any]) -> list[str]:
    return sql_delta_intents_from_visible_failure(failure)


def _artifact_patch_existing_edit_file_excerpts_for_plan(
    *,
    decomposed_space_payload: Mapping[str, Any] | None,
    existing_edit_files: list[str],
    tokens: list[str],
    limit: int = 2,
) -> list[dict[str, str]]:
    if not isinstance(decomposed_space_payload, Mapping) or not existing_edit_files:
        return []
    raw_files = decomposed_space_payload.get("files")
    if not isinstance(raw_files, list):
        return []

    files_by_path: dict[str, Mapping[str, Any]] = {}
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            continue
        path = str(raw_file.get("path") or "").strip()
        if path:
            files_by_path[path] = raw_file

    excerpts: list[dict[str, str]] = []
    for path in existing_edit_files:
        raw_file = files_by_path.get(path)
        if raw_file is None:
            continue
        content = str(raw_file.get("content") or "")
        excerpt = _artifact_patch_file_excerpt(content, tokens=tokens)
        if not excerpt:
            continue
        digest = str(raw_file.get("expected_sha256") or raw_file.get("sha256") or "").strip()
        excerpts.append(
            {
                "path": path,
                "expected_sha256": digest,
                "excerpt": excerpt,
            }
        )
        if len(excerpts) >= limit:
            break
    return excerpts


def _artifact_patch_file_excerpt(
    content: str,
    *,
    tokens: list[str],
    line_window: int = 4,
    limit: int = 360,
) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    normalized_tokens = [
        token
        for token in (
            _artifact_patch_normalized_visible_token(value) for value in tokens
        )
        if token
    ]
    start = 0
    if normalized_tokens:
        for index, line in enumerate(lines):
            normalized_line = _artifact_patch_normalized_token_text(line)
            if any(token in normalized_line for token in normalized_tokens):
                start = max(0, index - 1)
                break
    excerpt = "\n".join(lines[start : start + line_window])
    return _compact_inline_text(excerpt, limit=limit)


@dataclass(frozen=True)
class ArtifactPatchSkeleton:
    """A deterministic file-edit envelope for bounded serving fill-in."""

    skeleton_id: str
    selected_marker: str
    family: str
    path: str
    expected_sha256: str
    edit_mode: str
    find_anchor: str
    anchor_excerpt: str
    replacement_budget_chars: int
    required_visible_tokens: tuple[str, ...]
    required_visible_intent_labels: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]
    pass_guard_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton_id": self.skeleton_id,
            "selected_marker": self.selected_marker,
            "family": self.family,
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "edit_mode": self.edit_mode,
            "find_anchor": self.find_anchor,
            "anchor_excerpt": self.anchor_excerpt,
            "replacement_budget_chars": self.replacement_budget_chars,
            "required_visible_tokens": list(self.required_visible_tokens),
            "required_visible_intent_labels": list(
                self.required_visible_intent_labels
            ),
            "forbidden_patterns": list(self.forbidden_patterns),
            "pass_guard_notes": list(self.pass_guard_notes),
        }


def build_artifact_patch_skeletons(
    *,
    structured_failure_context: Mapping[str, Any] | None,
    decomposed_space_payload: Mapping[str, Any] | None,
    candidate_mode: str | None,
    limit: int = 4,
) -> list[ArtifactPatchSkeleton]:
    """Build deterministic visible-only patch skeletons for serving fill-in."""

    if candidate_mode != _SKELETON_ARTIFACT_PATCH_MODE:
        return []
    if not isinstance(decomposed_space_payload, Mapping):
        return []
    failures_by_id = _structured_failures_by_id(structured_failure_context)
    if not failures_by_id:
        return []
    files_by_path = _decomposed_files_by_path(decomposed_space_payload)
    if not files_by_path:
        return []

    risk_rank = {"low": 0, "medium": 1, "high": 2}

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, str]:
        question_id, failure = item
        risk = str(failure.get("guard_regression_risk") or "unknown").lower()
        confidence = failure.get("confidence")
        confidence_value = (
            float(confidence)
            if isinstance(confidence, int | float) and not isinstance(confidence, bool)
            else 0.0
        )
        return (risk_rank.get(risk, 3), -confidence_value, question_id)

    skeletons: list[ArtifactPatchSkeleton] = []
    for question_id, failure in sorted(failures_by_id.items(), key=sort_key):
        family = str(failure.get("likely_root_cause_family") or "unknown").strip()
        suggested_ops = [
            str(value).strip()
            for value in failure.get("suggested_operation_families", []) or []
            if str(value).strip()
        ]
        path_hints = _artifact_patch_path_hints_for_operations(suggested_ops)
        if not path_hints:
            continue
        expected_tokens = [
            str(value).strip()
            for value in failure.get("missing_expected_tokens", []) or []
            if str(value).strip()
        ]
        tokens = _artifact_patch_contract_tokens(failure)
        visible_intents = _artifact_patch_sql_delta_intents(failure)
        if not expected_tokens and not visible_intents:
            continue
        existing_files = _artifact_patch_existing_edit_files_for_plan(
            decomposed_space_payload=decomposed_space_payload,
            path_hints=path_hints,
            tokens=tokens,
            limit=6,
        )
        for path in existing_files:
            raw_file = files_by_path.get(path)
            if raw_file is None:
                continue
            content = str(raw_file.get("content") or "")
            expected_sha256 = str(
                raw_file.get("expected_sha256") or raw_file.get("sha256") or ""
            ).strip()
            if not content.strip() or not expected_sha256:
                continue
            find_anchor = _artifact_patch_skeleton_find_anchor(
                content=content,
                path=path,
                tokens=expected_tokens or tokens,
            )
            if not find_anchor:
                continue
            skeletons.append(
                ArtifactPatchSkeleton(
                    skeleton_id=f"skel_{len(skeletons) + 1:02d}_{question_id[:8]}",
                    selected_marker=f"target_train_failure:{question_id}",
                    family=family,
                    path=path,
                    expected_sha256=expected_sha256,
                    edit_mode="find_replace",
                    find_anchor=find_anchor,
                    anchor_excerpt=_artifact_patch_file_excerpt(
                        content,
                        tokens=expected_tokens or tokens,
                        limit=260,
                    ),
                    replacement_budget_chars=_SKELETON_REPLACEMENT_BUDGET_CHARS,
                    required_visible_tokens=tuple(expected_tokens[:3]),
                    required_visible_intent_labels=tuple(visible_intents[:4]),
                    forbidden_patterns=tuple(
                        _artifact_patch_skeleton_forbidden_patterns(path)
                    ),
                    pass_guard_notes=tuple(
                        _artifact_patch_skeleton_pass_guard_notes(
                            structured_failure_context
                        )
                    ),
                )
            )
            break
        if len(skeletons) >= limit:
            break
    return skeletons


def validate_artifact_patch_skeleton_binding(
    payload: Mapping[str, Any],
    *,
    skeletons: list[ArtifactPatchSkeleton],
) -> dict[str, Any]:
    """Validate that a patch response stays bound to one deterministic skeleton."""

    result: dict[str, Any] = {
        "applied": True,
        "accepted": True,
        "reasons": [],
        "skeleton_count": len(skeletons),
        "selected_skeleton_id": None,
        "matched_skeleton": None,
        "policy": {
            "exact_path_binding": True,
            "exact_hash_binding": True,
            "exact_find_anchor_binding": True,
            "replacement_budget_enforced": True,
            "hidden_holdout_content": "omitted",
        },
    }
    reasons: list[str] = []
    if not skeletons:
        reasons.append("skeleton_artifact_patch_no_available_skeleton")
        result["accepted"] = False
        result["reasons"] = reasons
        return result

    operations = payload.get("operations")
    if operations not in (None, []):
        reasons.append("skeleton_artifact_patch_operations_must_be_empty")

    file_edits = payload.get("file_edits")
    edits = [edit for edit in file_edits if isinstance(edit, dict)] if isinstance(file_edits, list) else []
    if len(edits) != 1:
        reasons.append(f"skeleton_artifact_patch_requires_one_file_edit:{len(edits)}")
        result["accepted"] = False
        result["reasons"] = list(dict.fromkeys(reasons))
        return result

    edit = edits[0]
    plan = payload.get("causal_patch_plan")
    plan = plan if isinstance(plan, Mapping) else {}
    edit_path = str(edit.get("path") or edit.get("relative_path") or "").strip()
    edit_hash = str(edit.get("expected_sha256") or "").strip()
    edit_find = edit.get("find")
    edit_replace = edit.get("replace")
    selected_marker = str(plan.get("selected_marker") or "").strip()
    selected_family = str(plan.get("family") or "").strip()

    if edit.get("content") is not None or edit.get("append_content") is not None:
        reasons.append("skeleton_artifact_patch_requires_find_replace")
    if not isinstance(edit_find, str) or not edit_find:
        reasons.append("skeleton_artifact_patch_missing_find_anchor")
    if not isinstance(edit_replace, str) or not edit_replace:
        reasons.append("skeleton_artifact_patch_missing_replace")
    if not selected_marker or not selected_family:
        reasons.append("skeleton_artifact_patch_missing_causal_plan_binding")

    matched = [
        skeleton
        for skeleton in skeletons
        if edit_path == skeleton.path
        and edit_hash == skeleton.expected_sha256
        and edit_find == skeleton.find_anchor
        and selected_marker == skeleton.selected_marker
        and selected_family == skeleton.family
    ]
    if not matched:
        reasons.append("skeleton_artifact_patch_no_matching_skeleton")
        result["candidate_binding"] = {
            "path": edit_path,
            "expected_sha256": edit_hash,
            "find_anchor": edit_find if isinstance(edit_find, str) else None,
            "selected_marker": selected_marker,
            "family": selected_family,
        }
        result["available_skeletons"] = [
            {
                "skeleton_id": skeleton.skeleton_id,
                "path": skeleton.path,
                "expected_sha256": skeleton.expected_sha256,
                "selected_marker": skeleton.selected_marker,
                "family": skeleton.family,
                "find_anchor": skeleton.find_anchor,
            }
            for skeleton in skeletons
        ]
    else:
        skeleton = matched[0]
        result["selected_skeleton_id"] = skeleton.skeleton_id
        result["matched_skeleton"] = skeleton.to_dict()
        replace_text = edit_replace if isinstance(edit_replace, str) else ""
        if len(replace_text) > skeleton.replacement_budget_chars:
            reasons.append("skeleton_artifact_patch_replacement_too_large")
        if _BIND_PLACEHOLDER_RE.search(replace_text):
            reasons.append("skeleton_artifact_patch_replace_contains_placeholder")
        if _SKELETON_HIDDEN_CONTEXT_RE.search(json.dumps(payload, ensure_ascii=False)):
            reasons.append("skeleton_artifact_patch_hidden_holdout_reference")
        if (
            _artifact_patch_path_is_metadata(skeleton.path)
            and _SKELETON_METADATA_QUERY_PLAN_RE.search(replace_text)
        ):
            reasons.append("skeleton_artifact_patch_metadata_query_plan_guidance")
        if skeleton.required_visible_tokens:
            normalized_replace = _artifact_patch_normalized_token_text(replace_text)
            if not any(
                token
                and token in normalized_replace
                for token in (
                    _artifact_patch_normalized_visible_token(value)
                    for value in skeleton.required_visible_tokens
                )
            ):
                reasons.append("skeleton_artifact_patch_missing_required_visible_token")
        elif skeleton.required_visible_intent_labels:
            effect_text = _artifact_patch_normalized_token_text(
                " ".join(
                    str(plan.get(field) or "")
                    for field in (
                        "expected_visible_effect",
                        "repair_shape",
                        "root_cause",
                    )
                )
            )
            if not any(
                _artifact_patch_normalized_token_text(label) in effect_text
                for label in skeleton.required_visible_intent_labels
            ):
                reasons.append("skeleton_artifact_patch_missing_required_intent_label")

    result["accepted"] = not reasons
    result["reasons"] = list(dict.fromkeys(reasons))
    return result


def _artifact_patch_skeleton_contract_lines(
    *,
    structured_failure_context: Mapping[str, Any] | None,
    decomposed_space_payload: Mapping[str, Any] | None,
    candidate_mode: str | None,
) -> list[str]:
    if candidate_mode != _SKELETON_ARTIFACT_PATCH_MODE:
        return []

    skeletons = build_artifact_patch_skeletons(
        structured_failure_context=structured_failure_context,
        decomposed_space_payload=decomposed_space_payload,
        candidate_mode=candidate_mode,
    )
    lines = [
        "### Deterministic Patch Skeleton Contract",
        "Use exactly one skeleton below. Local code selected the visible marker, family, path, hash, edit mode, and exact find anchor before this request.",
        "Serving may fill only `file_edits[0].replace` and the existing causal_patch_plan text fields. Do not choose a different path, marker, family, edit mode, or find anchor.",
        "Return no_change with `operations=[]`, `file_edits=[]`, and `causal_patch_plan=null` if no skeleton can be filled safely.",
    ]
    if not skeletons:
        lines.extend(
            [
                "- No usable deterministic skeleton was found from visible evidence and current decomposed files.",
                "",
            ]
        )
        return lines

    for skeleton in skeletons:
        lines.append(
            f"- skeleton_id=`{skeleton.skeleton_id}`; "
            f"selected_marker=`{skeleton.selected_marker}`; "
            f"family=`{skeleton.family}`; "
            f"path=`{skeleton.path}`; "
            f"expected_sha256=`{skeleton.expected_sha256}`; "
            f"edit_mode=`{skeleton.edit_mode}`; "
            f"replacement_budget_chars=`{skeleton.replacement_budget_chars}`; "
            f"find_anchor_json={json.dumps(skeleton.find_anchor, ensure_ascii=False)}; "
            f"required_visible_tokens={_short_id_list(list(skeleton.required_visible_tokens), limit=4)}; "
            f"required_visible_intent_labels={_short_id_list(list(skeleton.required_visible_intent_labels), limit=4)}; "
            f"forbidden_patterns={_short_id_list(list(skeleton.forbidden_patterns), limit=5)}."
        )
        if skeleton.anchor_excerpt:
            lines.append(f"  - Anchor excerpt: `{skeleton.anchor_excerpt}`")
        if skeleton.pass_guard_notes:
            lines.append(
                "  - Visible pass-guard notes: "
                + " | ".join(skeleton.pass_guard_notes[:3])
            )
    lines.append("")
    return lines


def _decomposed_files_by_path(
    decomposed_space_payload: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw_files = decomposed_space_payload.get("files")
    if not isinstance(raw_files, list):
        return {}
    files_by_path: dict[str, Mapping[str, Any]] = {}
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            continue
        path = str(raw_file.get("path") or "").strip()
        if path:
            files_by_path[path] = raw_file
    return files_by_path


def _artifact_patch_skeleton_find_anchor(
    *,
    content: str,
    path: str,
    tokens: list[str],
) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    token_anchors = _artifact_patch_skeleton_anchor_candidates(
        lines,
        path=path,
        tokens=tokens,
    )
    for candidate in token_anchors:
        anchor = candidate.rstrip("\n")
        if len(anchor.strip()) < 3:
            continue
        if content.count(anchor) == 1:
            return anchor
    for index in range(max(len(lines) - 1, 0)):
        block = "\n".join(lines[index : index + 2]).rstrip("\n")
        if len(block.strip()) >= 6 and content.count(block) == 1:
            return block
    return ""


def _artifact_patch_skeleton_anchor_candidates(
    lines: list[str],
    *,
    path: str,
    tokens: list[str],
) -> list[str]:
    normalized_tokens = [
        token
        for token in (
            _artifact_patch_normalized_visible_token(value) for value in tokens
        )
        if token
    ]
    candidates: list[str] = []
    for line in lines:
        normalized_line = _artifact_patch_normalized_token_text(line)
        if any(token in normalized_line for token in normalized_tokens):
            candidates.append(line)
    field_patterns = (
        (r"^\s*(?:question|sql|usage_guidance|display_name|description):",)
        if serving_candidate_is_executable_artifact_path(path)
        else (
            r"^\s*(?:description|comment|instruction|synonyms|columns?|column_name):",
        )
    )
    if path == "instructions/text_instruction.md":
        field_patterns = (r"\S",)
    for pattern in field_patterns:
        for line in lines:
            if re.search(pattern, line):
                candidates.append(line)
    candidates.extend(line for line in lines if line.strip())
    return list(dict.fromkeys(candidates))


def _artifact_patch_skeleton_forbidden_patterns(path: str) -> list[str]:
    patterns = ["bind_placeholders", "hidden_holdout_content"]
    if _artifact_patch_path_is_metadata(path):
        patterns.append("metadata_query_plan_or_result_schema")
    if serving_candidate_is_executable_artifact_path(path):
        patterns.extend(["invalid_executable_yaml", "guard_clone"])
    return patterns


def _artifact_patch_skeleton_pass_guard_notes(
    structured_failure_context: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(structured_failure_context, Mapping):
        return []
    pass_guards = structured_failure_context.get("pass_guards")
    if not isinstance(pass_guards, list):
        return []
    notes: list[str] = []
    for guard in pass_guards[:4]:
        if not isinstance(guard, Mapping):
            continue
        question_id = str(guard.get("question_id") or "").strip()
        latest = str(guard.get("latest_status") or "unknown").strip()
        if question_id:
            notes.append(f"preserve guard_preservation:{question_id} latest={latest}")
    return notes


def _artifact_patch_path_is_metadata(path: str) -> bool:
    return path.startswith("tables/") or path.startswith("metric_views/")


def _artifact_patch_existing_edit_files_for_plan(
    *,
    decomposed_space_payload: Mapping[str, Any] | None,
    path_hints: list[str],
    tokens: list[str],
    limit: int = 8,
) -> list[str]:
    """Return concrete existing file paths that match a target plan's edit surface."""

    if not isinstance(decomposed_space_payload, Mapping) or not path_hints:
        return []
    raw_files = decomposed_space_payload.get("files")
    if not isinstance(raw_files, list):
        return []

    matches: list[tuple[int, str]] = []
    for file_index, raw_file in enumerate(raw_files):
        if not isinstance(raw_file, Mapping):
            continue
        path = str(raw_file.get("path") or "").strip()
        if not path:
            continue
        if not any(_artifact_patch_path_matches_hint(path, hint) for hint in path_hints):
            continue
        content = str(raw_file.get("content") or "")
        token_rank = 0 if _artifact_patch_file_mentions_any_token(path, content, tokens) else 1
        matches.append((token_rank * 10000 + file_index, path))

    return [path for _rank, path in sorted(matches)[:limit]]


def _artifact_patch_path_matches_hint(path: str, hint: str) -> bool:
    normalized_path = path.strip().lstrip("/")
    normalized_hint = hint.strip().lstrip("/")
    if not normalized_path or not normalized_hint:
        return False
    if "*" in normalized_hint:
        return fnmatch.fnmatchcase(normalized_path, normalized_hint)
    return normalized_path == normalized_hint or normalized_path.startswith(
        normalized_hint.rstrip("/") + "/"
    )


def _artifact_patch_file_mentions_any_token(
    path: str,
    content: str,
    tokens: list[str],
) -> bool:
    if not tokens:
        return False
    haystack = _artifact_patch_normalized_token_text(f"{path}\n{content}")
    return any(
        token
        for token in (_artifact_patch_normalized_visible_token(value) for value in tokens)
        if token and token in haystack
    )


def _artifact_patch_normalized_visible_token(value: str) -> str:
    token = value.strip().strip("`'\"")
    token = token.removeprefix(":").removeprefix("${").removesuffix("}")
    return _artifact_patch_normalized_token_text(token)


def _artifact_patch_normalized_token_text(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", value.lower())


def _compact_inline_text(value: Any, *, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(value or "").strip()).replace("`", "'")
    if not compact:
        return ""
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 16, 0)].rstrip() + " ... (truncated)"


def _artifact_patch_plan_allows_new_file(operation_families: list[str]) -> bool:
    return bool(
        {
            "add_example_sql",
            "add_sql_snippet",
            "add_join_spec",
        }.intersection(operation_families)
    )


def _artifact_patch_repair_recipe(
    *,
    family: str,
    operation_families: list[str],
    tokens: list[str],
) -> dict[str, str]:
    token_text = (
        f" Mention visible tokens {_short_id_list(tokens, limit=4)} in the changed asset."
        if tokens
        else ""
    )
    ops = set(operation_families)
    if family == "literal_or_parameter_mismatch":
        if "add_sql_snippet" in ops:
            repair = (
                "Add one literal SQL fragment under instructions/sql_snippets/filters "
                "or a tiny example that forces executable literal filters."
            )
        elif "add_example_sql" in ops:
            repair = (
                "Add one narrow visible example asset for the target entity/literal pattern, "
                "using executable SQL and the target marker."
            )
        else:
            repair = (
                "Clarify the concrete entity or literal mapping in the narrowest compatible asset."
            )
        avoid = (
            "bind placeholders, global trusted-query routing rewrites, and pass-guard clones "
            "that only swap brand, market, metric, time, or entity literals."
        )
    elif family == "selected_metric_or_column_mismatch":
        repair = (
            "Prefer metadata or a reusable measure/expression snippet that names the missing "
            "output column or alias and keeps the expected result schema intact."
        )
        avoid = (
            "lookup/existence examples, alternate time descriptor columns, and examples copied "
            "from passing guards with only literals changed."
        )
    elif family == "percentage_or_comparison_metric":
        repair = (
            "Patch the denominator/current-vs-prior rule in one compatible text, example, or "
            "measure asset while preserving both comparison-period aliases."
        )
        avoid = "current-period-only metrics, broad prose rewrites, and copied benchmark SQL."
    elif family == "time_filter_mismatch":
        repair = (
            "Patch one compatible time-grain/window asset so generated SQL uses the visible "
            "expected bucket and period labels."
        )
        avoid = "changing default windows globally or adding full-query snippets."
    elif family == "aggregation_or_grouping_mismatch":
        repair = (
            "Patch the aggregate or grouping definition in metadata or a reusable measure "
            "fragment before adding any example."
        )
        avoid = "single literal examples that hide the aggregate or GROUP BY mismatch."
    elif family == "table_or_join_path_mismatch":
        repair = (
            "Patch a join spec or table description that fixes the visible table path decision."
        )
        avoid = "example-only patches that bypass the wrong join path."
    elif family == "runtime_or_generation_error":
        repair = (
            "Fix the invalid reference or unsupported syntax in the smallest compatible asset, "
            "or return no_change if the bad SQL cannot be localized."
        )
        avoid = "benchmark-specific SQL answers and broad instruction rewrites."
    elif family == "missing_generated_sql":
        repair = (
            "Return no_change unless visible diagnostics identify a concrete editable asset "
            "that prevents SQL generation."
        )
        avoid = "guessing examples or snippets without visible generated-SQL evidence."
    else:
        repair = "Patch one compatible asset that directly explains the visible SQL delta."
        avoid = "multi-family edits, generic prose, and hidden-holdout inference."
    return {"repair": f"{repair}{token_text}", "avoid": avoid}


def _artifact_patch_patch_archetype(
    *,
    family: str,
    operation_families: list[str],
    expected_tokens: list[str],
    generated_tokens: list[str],
    shared_expected_tokens: list[str] | None = None,
    shared_generated_tokens: list[str] | None = None,
    prefer_existing_executable_file: bool = False,
    has_existing_edit_files: bool = False,
) -> str:
    """Return a concrete prompt-side patch shape for a visible target plan."""

    ops = set(operation_families)
    required_expected_tokens = shared_expected_tokens or expected_tokens
    wrong_generated_tokens = shared_generated_tokens or generated_tokens
    expected_text = (
        f" using expected token(s) {_short_id_list(required_expected_tokens, limit=3)}"
        if required_expected_tokens
        else ""
    )
    generated_text = (
        f"; do not anchor only to generated token(s) {_short_id_list(wrong_generated_tokens, limit=3)}"
        if wrong_generated_tokens
        else ""
    )
    if prefer_existing_executable_file and not has_existing_edit_files:
        return (
            "do not create a new executable example or SQL-snippet file after prior "
            "new executable train-collapse; no listed existing executable edit file is "
            "available, so return no_change unless a non-executable compatible plan is selected"
        )
    if "add_sql_snippet" in ops:
        if prefer_existing_executable_file:
            return (
                "compactly edit one listed existing SQL-snippet asset with exact "
                f"find/replace around a display_name or SQL fragment{expected_text}"
                f"{generated_text}; do not create a new SQL-snippet file and do not "
                "rewrite the whole file"
            )
        if family == "literal_or_parameter_mismatch":
            return (
                "create one SQL-snippet filter asset with display_name and one "
                f"literal-safe predicate or fragment{expected_text}{generated_text}; "
                "no bind placeholders and no full SELECT/WITH query"
            )
        if family == "selected_metric_or_column_mismatch":
            return (
                "create one SQL-snippet expression or measure asset with display_name "
                f"that exposes the expected output alias or metric{expected_text}"
                f"{generated_text}; no alternate descriptor lookup"
            )
        if family == "time_filter_mismatch":
            return (
                "create one SQL-snippet filter asset with display_name for the visible "
                f"time bucket or period label{expected_text}{generated_text}; no "
                "global default-window rewrite"
            )
        return (
            "create one reusable SQL-snippet asset with display_name and one narrow "
            f"fragment tied to the visible SQL delta{expected_text}{generated_text}"
        )
    if "add_example_sql" in ops:
        if prefer_existing_executable_file:
            return (
                "compactly edit one listed existing example asset with exact "
                f"find/replace around the question, SQL, or usage guidance{expected_text}"
                f"{generated_text}; do not create a new example file and do not clone "
                "a passing guard with only literals swapped"
            )
        return (
            "create one executable example asset whose question overlaps the visible "
            f"target/member terms and whose SQL contains the expected signal{expected_text}"
            f"{generated_text}; do not clone a passing guard with only literals swapped"
        )
    if {"set_column_config", "set_table_description"} & ops:
        return (
            "compactly edit one listed existing table or metric metadata file to add "
            f"column/table description or synonym evidence{expected_text}{generated_text}; "
            "do not create a new metadata file"
        )
    if "append_text_instruction" in ops:
        return (
            "append one short durable rule to the listed text-guidance asset that names "
            f"the visible expected signal{expected_text}{generated_text}; avoid broad "
            "routing or default-behavior rewrites"
        )
    if "add_join_spec" in ops:
        return (
            "create one join-spec asset that fixes the visible table path decision "
            f"and cites the selected marker{expected_text}{generated_text}"
        )
    return (
        "edit one compatible artifact only when the visible SQL/result delta can be "
        f"localized to that artifact{expected_text}{generated_text}; otherwise return no_change"
    )


def _artifact_patch_patch_template_hint(
    *,
    operation_families: list[str],
    prefer_existing_executable_file: bool = False,
    has_existing_edit_files: bool = False,
) -> str:
    """Return a compact file-edit template hint for the selected edit surface."""

    ops = set(operation_families)
    if prefer_existing_executable_file and not has_existing_edit_files:
        return (
            "no listed existing executable edit file is available after repeated new "
            "executable collapse; return no_change instead of creating a new example "
            "or SQL-snippet file"
        )
    if "add_sql_snippet" in ops:
        if prefer_existing_executable_file:
            return (
                "existing instructions/sql_snippets YAML only: use exact `find` plus "
                "`replace` around one display_name or `sql` fragment; include current "
                "`expected_sha256`, set `content` null, and do not create a new SQL-snippet file"
            )
        return (
            "new `instructions/sql_snippets/<filters|expressions|measures>/<name>.yml` "
            "with `display_name: <short name>` and `sql: |` containing one SQL fragment; "
            "set `content` for the new file and keep `expected_sha256` null"
        )
    if "add_example_sql" in ops:
        if prefer_existing_executable_file:
            return (
                "existing instructions/examples YAML only: use exact `find` plus "
                "`replace` around one question, `sql`, or usage_guidance fragment; include "
                "current `expected_sha256`, set `content` null, and do not create a new example file"
            )
        return (
            "new `instructions/examples/<name>.yml` with one overlapping `question`, "
            "`sql: |` executable Databricks SQL, optional `usage_guidance`, and no bind "
            "placeholders; set `content` for the new file and keep `expected_sha256` null"
        )
    if {"set_column_config", "set_table_description"} & ops:
        return (
            "existing table or metric YAML only: use exact `find` plus `replace` around "
            "a column/table description, synonym, or semantic hint; include current "
            "`expected_sha256` and do not rewrite the whole file"
        )
    if "append_text_instruction" in ops:
        return (
            "existing `instructions/text_instruction.md` only: use `append_content` "
            "for one short durable rule; include current `expected_sha256` and do not "
            "replace the whole file"
        )
    if "add_join_spec" in ops:
        return (
            "new `instructions/join_specs/<name>.yml` with the minimal left/right "
            "table identifiers and aliases needed for the visible join path; set "
            "`content` for the new file and keep `expected_sha256` null"
        )
    return (
        "one compatible file edit only: prefer exact `find` plus `replace` for existing "
        "files, use full `content` only for a small new allowlisted file, and set unused "
        "nullable fields to null"
    )


def artifact_patch_causal_recipe_anchors(
    *,
    family: str,
    operation_families: list[str],
    tokens: list[str],
) -> dict[str, list[str]]:
    """Return visible-only text anchors expected in causal patch plan intent fields."""

    repair_anchors_by_family = {
        "literal_or_parameter_mismatch": [
            "literal",
            "filter",
            "entity",
            "example",
            "snippet",
            "executable",
        ],
        "selected_metric_or_column_mismatch": [
            "metadata",
            "measure",
            "expression",
            "column",
            "alias",
            "schema",
        ],
        "percentage_or_comparison_metric": [
            "denominator",
            "current",
            "prior",
            "comparison",
            "percentage",
            "alias",
        ],
        "time_filter_mismatch": [
            "time",
            "window",
            "grain",
            "bucket",
            "period",
            "label",
        ],
        "aggregation_or_grouping_mismatch": [
            "aggregate",
            "group",
            "grouping",
            "measure",
        ],
        "table_or_join_path_mismatch": ["join", "table", "path"],
        "runtime_or_generation_error": [
            "invalid",
            "syntax",
            "reference",
            "smallest",
            "sql",
        ],
        "missing_generated_sql": ["no_change", "diagnostic", "generation"],
    }
    avoid_anchors_by_family = {
        "literal_or_parameter_mismatch": [
            "bind",
            "placeholder",
            "trusted-query",
            "routing",
            "guard",
            "clone",
            "literal",
        ],
        "selected_metric_or_column_mismatch": [
            "lookup",
            "existence",
            "alternate",
            "descriptor",
            "guard",
            "literal",
        ],
        "percentage_or_comparison_metric": [
            "current-period-only",
            "broad",
            "prose",
            "copied",
            "benchmark",
        ],
        "time_filter_mismatch": [
            "default",
            "window",
            "global",
            "full-query",
            "snippet",
        ],
        "aggregation_or_grouping_mismatch": [
            "single literal",
            "aggregate",
            "group",
        ],
        "table_or_join_path_mismatch": ["example-only", "wrong join", "join path"],
        "runtime_or_generation_error": [
            "benchmark-specific",
            "broad instruction",
            "unsupported",
        ],
        "missing_generated_sql": ["guessing", "without visible", "diagnostic"],
    }
    root_cause_anchors_by_family = {
        "literal_or_parameter_mismatch": [
            "literal",
            "parameter",
            "placeholder",
            "bind",
            "entity",
            "value",
        ],
        "selected_metric_or_column_mismatch": [
            "metric",
            "column",
            "measure",
            "alias",
            "schema",
        ],
        "percentage_or_comparison_metric": [
            "percentage",
            "comparison",
            "denominator",
            "current",
            "prior",
        ],
        "time_filter_mismatch": [
            "time",
            "window",
            "period",
            "date",
            "grain",
            "bucket",
        ],
        "aggregation_or_grouping_mismatch": [
            "aggregate",
            "group",
            "grouping",
            "measure",
        ],
        "table_or_join_path_mismatch": ["table", "join", "path"],
        "runtime_or_generation_error": [
            "runtime",
            "syntax",
            "reference",
            "invalid",
            "sql",
        ],
        "missing_generated_sql": ["missing", "generated", "sql", "generation"],
    }
    repair_anchors = list(
        repair_anchors_by_family.get(
            family,
            ["sql", "delta", "compatible", "asset"],
        )
    )
    avoid_anchors = list(
        avoid_anchors_by_family.get(
            family,
            ["multi-family", "generic", "hidden-holdout"],
        )
    )
    root_cause_anchors = list(
        root_cause_anchors_by_family.get(
            family,
            ["sql", "mismatch", "delta", "visible"],
        )
    )
    operation_anchors = {
        "append_text_instruction": ["instruction", "text", "guidance"],
        "add_example_sql": ["example"],
        "add_sql_snippet": ["snippet", "fragment", "filter", "expression", "measure"],
        "add_join_spec": ["join"],
        "set_column_config": ["metadata", "column"],
        "set_table_description": ["metadata", "table", "description"],
    }
    edit_surface_anchors_by_operation = {
        "append_text_instruction": ["instruction", "guidance"],
        "add_example_sql": ["example", "executable"],
        "add_sql_snippet": ["snippet", "fragment", "filter", "expression", "measure"],
        "add_join_spec": ["join", "join spec"],
        "set_column_config": ["metadata", "column", "schema"],
        "set_table_description": ["metadata", "table", "description"],
    }
    edit_surface_anchors_by_path_hint = {
        "instructions/text_instruction.md": ["instruction", "guidance"],
        "instructions/examples/*.yml": ["example", "executable"],
        "instructions/sql_snippets/filters/*.yml": ["snippet", "filter"],
        "instructions/sql_snippets/expressions/*.yml": ["snippet", "expression"],
        "instructions/sql_snippets/measures/*.yml": ["snippet", "measure"],
        "instructions/join_specs/*.yml": ["join", "join spec"],
        "tables/*.yml": ["metadata", "table", "column", "schema"],
        "metric_views/*.yml": ["metadata", "metric view", "column", "schema"],
    }
    edit_surface_anchors_by_family = {
        "literal_or_parameter_mismatch": ["example", "snippet", "filter", "instruction"],
        "selected_metric_or_column_mismatch": ["metadata", "column", "schema", "measure"],
        "percentage_or_comparison_metric": ["snippet", "measure", "example", "instruction"],
        "time_filter_mismatch": ["snippet", "filter", "example", "instruction"],
        "aggregation_or_grouping_mismatch": ["snippet", "measure", "metadata", "example"],
        "table_or_join_path_mismatch": ["join", "metadata", "table"],
        "runtime_or_generation_error": ["example", "snippet", "instruction"],
        "missing_generated_sql": ["diagnostic", "no_change"],
    }
    edit_surface_anchors: list[str] = []
    for operation_family in operation_families:
        repair_anchors.extend(operation_anchors.get(operation_family, []))
        edit_surface_anchors.extend(
            edit_surface_anchors_by_operation.get(operation_family, [])
        )
        for path_hint in _ARTIFACT_PATCH_OPERATION_PATH_HINTS.get(operation_family, ()):
            edit_surface_anchors.extend(
                edit_surface_anchors_by_path_hint.get(path_hint, [])
            )
    if not edit_surface_anchors:
        edit_surface_anchors.extend(
            edit_surface_anchors_by_family.get(
                family,
                ["artifact", "path", "surface"],
            )
        )
    repair_anchors.extend(tokens)
    root_cause_anchors.extend(tokens)
    return {
        "repair_shape": list(dict.fromkeys(anchor for anchor in repair_anchors if anchor)),
        "avoid": list(dict.fromkeys(anchor for anchor in avoid_anchors if anchor)),
        "root_cause": list(dict.fromkeys(anchor for anchor in root_cause_anchors if anchor)),
        "edit_surface": list(
            dict.fromkeys(anchor for anchor in edit_surface_anchors if anchor)
        ),
    }


def _artifact_patch_contract_tokens(failure: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in ("missing_expected_tokens", "extra_generated_tokens"):
        values = failure.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            token = str(value).strip()
            if token and len(token) >= 3:
                tokens.append(token)
    return list(dict.fromkeys(tokens))


def _generalization_proxy_guard_lines(
    structured_failure_context: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(structured_failure_context, Mapping):
        return []
    pass_guards = structured_failure_context.get("pass_guards")
    if not isinstance(pass_guards, list):
        return []
    proxy_guards: list[str] = []
    for guard in pass_guards:
        if not isinstance(guard, dict):
            continue
        source = str(guard.get("source") or "")
        if source != "baseline_validation":
            continue
        question_id = str(guard.get("question_id") or "?")
        latest_status = str(guard.get("latest_status") or "unknown")
        question = str(guard.get("question") or "(unknown question)")
        proxy_guards.append(
            f"- `{question_id}` latest=`{latest_status}` question=\"{_truncate_text(question, limit=180)}\"."
        )
    return proxy_guards[:4]


def _prior_candidate_signal_lines(failure_context: str) -> list[str]:
    rejected_modes = count_prior_rejected_candidate_modes(failure_context)
    rejected_count = count_prior_rejected_candidates(failure_context)
    skipped_count = count_prior_skipped_candidates(failure_context)
    invalid_count = count_prior_invalid_responses(failure_context)
    pass_guard_regressions = count_prior_pass_guard_regressions(failure_context)
    accepted_fix_regressions = count_prior_accepted_fix_regressions(failure_context)

    lines: list[str] = [
        (
            f"- Prior rejected candidates: `{rejected_count}`; skipped/no-change candidates: "
            f"`{skipped_count}`; malformed serving responses: `{invalid_count}`."
        ),
        (
            f"- Prior pass-guard regressions: `{pass_guard_regressions}`; "
            f"accepted-fix regressions: `{accepted_fix_regressions}`."
        ),
    ]
    if rejected_modes:
        mode_text = ", ".join(
            f"`{mode}`={count}" for mode, count in sorted(rejected_modes.items())
        )
        lines.append(f"- Rejected counts by candidate mode: {mode_text}.")
    else:
        lines.append("- Rejected counts by candidate mode: none recorded.")
    return lines


def _extract_markdown_h3_lines(text: str, heading: str) -> list[str]:
    marker = f"### {heading}"
    output: list[str] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == marker:
            inside = True
            continue
        if inside and (stripped.startswith("### ") or stripped.startswith("## ")):
            break
        if inside and stripped:
            output.append(line)
    return output


def _artifact_patch_rejected_new_executable_asset_count(lines: list[str]) -> int:
    count = 0
    inside_train_collapse = False
    for line in lines:
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
            path = match.group("path").rstrip(".,;")
            mode = match.group("mode").rstrip(".,;")
            if mode not in {"replace_file", "create_file"}:
                continue
            if path.startswith("instructions/examples/") or path.startswith(
                "instructions/sql_snippets/"
            ):
                count += 1
        inside_train_collapse = False
    return count


def _artifact_patch_has_repeated_new_executable_collapse(
    failure_context: str,
) -> bool:
    rejected = _extract_markdown_h3_lines(
        failure_context, "Rejected Patterns To Avoid Or Revise"
    )
    return _artifact_patch_rejected_new_executable_asset_count(rejected) >= 2


def _prior_candidate_lesson_lines(failure_context: str) -> list[str]:
    accepted = _extract_markdown_h3_lines(
        failure_context, "Accepted Patterns To Preserve"
    )
    rejected = _extract_markdown_h3_lines(
        failure_context, "Rejected Patterns To Avoid Or Revise"
    )
    skipped = _extract_markdown_h3_lines(
        failure_context, "Skipped Or Risk-Rejected Patterns To Avoid"
    )
    if not accepted and not rejected and not skipped:
        return []

    def compact(lines: list[str], *, max_items: int) -> list[str]:
        compacted: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            compacted.append(
                f"- {_truncate_text(stripped.removeprefix('-').strip(), limit=260)}"
            )
            if len(compacted) >= max_items:
                break
        return compacted

    def file_edit_shapes(lines: list[str], *, max_items: int) -> list[str]:
        shapes: list[str] = []
        for line in lines:
            match = re.search(r"file_edit\s+path=[^\n]+", line)
            if match is None:
                continue
            shapes.append(f"- {_truncate_text(match.group(0).strip(), limit=260)}")
            if len(shapes) >= max_items:
                break
        return list(dict.fromkeys(shapes))

    def file_surface_for_path(path: str) -> str:
        if path.startswith("instructions/examples/"):
            return "examples"
        if path.startswith("instructions/sql_snippets/"):
            return "sql_snippets"
        if path == "instructions/text_instruction.md":
            return "text_instruction"
        if path.startswith("tables/") or path.startswith("metric_views/"):
            return "metadata"
        if path.startswith("instructions/join_specs/"):
            return "join_specs"
        return "other_files"

    def file_surface_lessons(
        accepted_lines: list[str],
        avoided_lines: list[str],
        *,
        max_items: int,
    ) -> list[str]:
        entries: list[tuple[str, str, str]] = []
        for disposition, lines in (
            ("preserve", accepted_lines),
            ("avoid_or_revise", avoided_lines),
        ):
            for line in lines:
                match = re.search(r"file_edit\s+path=(?P<path>[^\s`]+)[^\n]*", line)
                if match is None:
                    continue
                path = match.group("path").rstrip(".,;")
                shape = _truncate_text(match.group(0).strip(), limit=220)
                entries.append((disposition, file_surface_for_path(path), shape))
        deduped = list(dict.fromkeys(entries))
        surface_order = {
            "examples": 0,
            "sql_snippets": 1,
            "text_instruction": 2,
            "metadata": 3,
            "join_specs": 4,
            "other_files": 5,
        }
        deduped.sort(key=lambda item: (item[0] != "preserve", surface_order[item[1]], item[2]))
        return [
            f"- {disposition} {surface}: {shape}"
            for disposition, surface, shape in deduped[:max_items]
        ]

    output: list[str] = [
        "### Prior Candidate Lessons To Preserve Or Avoid",
        "These visible train/validation lessons are hard synthesis constraints; preserve accepted shapes and avoid rejected or skipped shapes unless the new causal plan explains the material difference.",
    ]
    rejected_text = "\n".join([*rejected, *skipped]).lower()
    if "train_health_failed" in rejected_text or "-> 0.0%" in rejected_text:
        output.append(
            "- Prior train-health collapse detected: do not repeat broad/global text, metadata, or patch shapes from rejected attempts; choose a narrower visible target plan or return no_change."
        )
    if _artifact_patch_rejected_new_executable_asset_count(rejected) >= 2:
        output.append(
            "- Repeated new executable artifact collapse detected: do not create another new instructions/examples or instructions/sql_snippets file; use compact edits to existing files, a clearly different non-executable surface, or return no_change."
        )
    accepted_lines = compact(accepted, max_items=4)
    if accepted_lines:
        output.append("- Accepted patterns:")
        output.extend(f"  {line}" for line in accepted_lines)
    rejected_lines = compact([*rejected, *skipped], max_items=8)
    if rejected_lines:
        output.append("- Rejected or skipped anti-patterns:")
        output.extend(f"  {line}" for line in rejected_lines)
    surface_lesson_lines = file_surface_lessons(
        accepted,
        [*rejected, *skipped],
        max_items=8,
    )
    if surface_lesson_lines:
        output.append("- File-surface lessons from prior patches:")
        output.extend(f"  {line}" for line in surface_lesson_lines)
    rejected_shape_lines = file_edit_shapes([*rejected, *skipped], max_items=6)
    if rejected_shape_lines:
        output.append("- File-edit shapes to avoid or materially revise:")
        output.extend(f"  {line}" for line in rejected_shape_lines)
    return output


def _artifact_patch_parameter_domain_lines(
    structured_failure_context: Mapping[str, Any] | None,
) -> list[str]:
    """Render probed literal domains for the parameter-driving columns, if available.

    These are column *domains* probed from the warehouse (e.g. the distinct values of
    ``time_grp_type``), attached upstream only for visible context. They give the proposer
    real literals to bind an unbound ``:parameter`` to, which is the dominant failure lever.
    """
    if not isinstance(structured_failure_context, Mapping):
        return []
    domains = structured_failure_context.get("parameter_domain_values")
    if not isinstance(domains, Mapping) or not domains:
        return []
    column_lines: list[str] = []
    for column in sorted(domains):
        values = domains.get(column)
        if not isinstance(values, list) or not values:
            continue
        rendered = ", ".join(_compact_inline_text(str(value), limit=80) for value in values[:25])
        more = f" (+{len(values) - 25} more)" if len(values) > 25 else ""
        column_lines.append(f"- `{column}`: {rendered}{more}")
    if not column_lines:
        return []
    return [
        "### Parameter Literal Domains (probed from the warehouse)",
        (
            "Distinct literal values for the parameter-driving columns. When a generated "
            "query fails with an unbound `:parameter`, bind the marker to one of these "
            "literals (or add a literal example) instead of leaving it unbound."
        ),
        *column_lines,
        "",
    ]


def _artifact_patch_cross_round_attribution_lines(
    structured_failure_context: Mapping[str, Any] | None,
    *,
    max_questions: int = 12,
    max_edits_per_question: int = 6,
) -> list[str]:
    """Render which prior edits flipped each visible question's pass/fail state.

    Built from train-visible iteration diffs only (no hidden holdout id appears). Lets the
    proposer reuse edits that previously fixed a question and avoid edits that regressed one.
    """
    if not isinstance(structured_failure_context, Mapping):
        return []
    attribution = structured_failure_context.get("per_question_attribution")
    if not isinstance(attribution, Mapping) or not attribution:
        return []
    question_lines: list[str] = []
    for question_id in list(attribution)[:max_questions]:
        edits = attribution.get(question_id)
        if not isinstance(edits, list) or not edits:
            continue
        rendered_edits: list[str] = []
        for edit in edits[:max_edits_per_question]:
            if not isinstance(edit, Mapping):
                continue
            effect = str(edit.get("effect") or "?")
            label = _compact_inline_text(str(edit.get("label") or "?"), limit=80)
            status = "accepted" if edit.get("accepted") else "rejected"
            rendered_edits.append(f"{effect} by `{label}` ({status})")
        if rendered_edits:
            question_lines.append(f"- `{question_id}`: " + "; ".join(rendered_edits))
    if not question_lines:
        return []
    return [
        "### Cross-Round Edit Attribution (visible questions)",
        (
            "Which prior edits flipped each visible question's pass/fail. Prefer the shape "
            "of an edit that previously fixed a question; do not repeat an edit that "
            "regressed one. Train-visible only — no hidden holdout question appears here."
        ),
        *question_lines,
        "",
    ]


def _candidate_context_summary_lines(
    *,
    structured_failure_context: Mapping[str, Any] | None,
    failure_context: str,
    optimization_summary: Mapping[str, Any] | None,
    candidate_mode: str | None,
    decomposed_space_payload: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(structured_failure_context, Mapping):
        return []
    summary = optimization_summary if isinstance(optimization_summary, Mapping) else {}
    accepted = summary.get("accepted")
    rejected = summary.get("rejected")
    skipped = summary.get("skipped")
    return [
        "## Candidate Context Summary",
        "",
        "Use this compact structured summary before reading the full artifacts below.",
        (
            "- Current visible state: "
            f"train={_format_percent(structured_failure_context.get('train_pass_rate'))}; "
            f"best_train={_format_percent(structured_failure_context.get('best_train_rate'))}; "
            f"best_validation={_format_percent(structured_failure_context.get('best_validation_rate'))}; "
            f"remaining_train_failures={structured_failure_context.get('failed_question_count', 'n/a')}; "
            f"candidate_mode=`{candidate_mode or 'unspecified'}`."
        ),
        (
            "- Optimization counters: "
            f"accepted=`{accepted if accepted is not None else 'n/a'}`; "
            f"rejected=`{rejected if rejected is not None else 'n/a'}`; "
            f"skipped=`{skipped if skipped is not None else 'n/a'}`."
        ),
        "",
        "### Remaining Failure Clusters",
        *_failure_cluster_lines(structured_failure_context),
        "",
        "### Generalized Visible-Family Guidance",
        "These signals are computed only from train-visible failures and visible pass guards; hidden holdout question text and SQL are omitted.",
        *_generalized_visible_family_guidance_lines(structured_failure_context),
        "",
        *_artifact_patch_parameter_domain_lines(structured_failure_context),
        "### Pass Guard Constraints",
        *_pass_guard_summary_lines(structured_failure_context),
        "",
        *_artifact_patch_visible_effect_contract_lines(
            structured_failure_context=structured_failure_context,
            candidate_mode=candidate_mode,
        ),
        *_artifact_patch_target_plan_lines(
            structured_failure_context=structured_failure_context,
            decomposed_space_payload=decomposed_space_payload,
            candidate_mode=candidate_mode,
            failure_context=failure_context,
        ),
        *_artifact_patch_skeleton_contract_lines(
            structured_failure_context=structured_failure_context,
            decomposed_space_payload=decomposed_space_payload,
            candidate_mode=candidate_mode,
        ),
        *_artifact_patch_self_checklist_lines(candidate_mode),
        *(
            [
                "### Visible Generalization Proxy Guards",
                "These are validation-visible pass guards, not hidden holdout content. Use them as proxy patterns that a candidate must preserve.",
                *_generalization_proxy_guard_lines(structured_failure_context),
                "",
            ]
            if _generalization_proxy_guard_lines(structured_failure_context)
            else []
        ),
        *_artifact_patch_cross_round_attribution_lines(structured_failure_context),
        "### Prior Candidate Signals",
        *_prior_candidate_lesson_lines(failure_context),
        *_prior_candidate_signal_lines(failure_context),
        "",
        (
            "Target exactly one remaining failure cluster. Preserve every listed pass guard; in "
            "example-set modes, cover pass guards with guard_preservation:<question_id> operations "
            "before target_train_failure operations."
        ),
        "",
    ]


def _extract_markdown_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_heading = re.search(r"\n## [^\n]+", text[start + len(marker):])
    if next_heading is None:
        return text[start:].strip()
    end = start + len(marker) + next_heading.start()
    return text[start:end].strip()


def _find_table_like(config: GenieSpaceConfig, table_identifier: str) -> Any | None:
    for table in (config.data_sources.tables or []) if config.data_sources else []:
        if table.identifier == table_identifier:
            return table
    for metric_view in (config.data_sources.metric_views or []) if config.data_sources else []:
        if metric_view.identifier == table_identifier:
            return metric_view
    return None


def _find_column(config: GenieSpaceConfig, table_identifier: str, column_name: str) -> GenieColumnConfig | None:
    table_like = _find_table_like(config, table_identifier)
    if table_like is None:
        return None
    for column in table_like.column_configs or []:
        if column.column_name == column_name:
            return column
    return None


def canonicalize_tagged_example_sql(
    payload: dict[str, Any],
    *,
    benchmark_by_question_id: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace tagged example SQL with exact visible benchmark SQL.

    The serving proposal still chooses which benchmark IDs to use. This helper
    removes copy/paste drift from long SQL examples before the risk gate and
    clone application.
    """
    candidate = deepcopy(payload)
    operations = candidate.get("operations")
    if not isinstance(operations, list):
        return candidate, {"canonicalized_example_sql_ids": [], "skipped_example_sql_ids": []}

    canonicalized_ids: list[str] = []
    skipped_ids: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if str(operation.get("op") or "").strip() != "add_example_sql":
            continue
        matched_ids = _operation_mentioned_question_ids(operation, benchmark_by_question_id)
        if not matched_ids:
            continue
        question_id = matched_ids[0]
        benchmark = benchmark_by_question_id.get(question_id) or {}
        expected_sql = str(benchmark.get("expected_sql") or "").strip()
        if not expected_sql:
            skipped_ids.append(question_id)
            continue
        question = str(benchmark.get("question") or "").strip()
        if question:
            operation["question"] = question
        operation["sql"] = expected_sql
        canonicalized_ids.append(question_id)

    return candidate, {
        "canonicalized_example_sql_ids": list(dict.fromkeys(canonicalized_ids)),
        "skipped_example_sql_ids": list(dict.fromkeys(skipped_ids)),
    }


def repair_example_set_pass_guards(
    payload: dict[str, Any],
    *,
    benchmark_by_question_id: Mapping[str, Mapping[str, str]],
    required_pass_guard_ids: frozenset[str] | set[str] | tuple[str, ...] | list[str] | None,
    max_mutation_operations: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill stale replay example sets with currently required guard examples."""
    candidate = deepcopy(payload)
    operations = candidate.get("operations")
    if not isinstance(operations, list):
        return candidate, {
            "repaired_pass_guard_ids": [],
            "unrepaired_pass_guard_ids": [],
            "pruned_duplicate_operation_count": 0,
        }

    required_guards = tuple(
        dict.fromkeys(str(item) for item in (required_pass_guard_ids or []) if str(item).strip())
    )
    if not required_guards:
        return candidate, {
            "repaired_pass_guard_ids": [],
            "unrepaired_pass_guard_ids": [],
            "pruned_duplicate_operation_count": 0,
        }

    covered_guards: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if str(operation.get("op") or "").strip() != "add_example_sql":
            continue
        operation_text = _operation_field_text(
            operation,
            (
                "reason",
                "description",
                "usage_guidance",
                "question",
            ),
        )
        for guard_id in required_guards:
            if guard_id in operation_text:
                covered_guards.add(guard_id)

    missing_guards = [guard_id for guard_id in required_guards if guard_id not in covered_guards]
    if not missing_guards:
        return candidate, {
            "repaired_pass_guard_ids": [],
            "unrepaired_pass_guard_ids": [],
            "pruned_duplicate_operation_count": 0,
        }

    pruned_duplicate_count = _prune_duplicate_guard_operations(operations)
    mutation_count = _mutation_operation_count(operations)
    available_slots = (
        len(missing_guards)
        if max_mutation_operations is None
        else max(max_mutation_operations - mutation_count, 0)
    )

    repaired_ids: list[str] = []
    unrepaired_ids: list[str] = []
    for guard_id in missing_guards:
        benchmark = benchmark_by_question_id.get(guard_id) or {}
        expected_sql = str(benchmark.get("expected_sql") or "").strip()
        if available_slots <= 0 or not expected_sql:
            unrepaired_ids.append(guard_id)
            continue
        question = str(benchmark.get("question") or "").strip() or guard_id
        operations.append(
            {
                "op": "add_example_sql",
                "id": guard_id,
                "question": question,
                "sql": expected_sql,
                "usage_guidance": (
                    "Preserve this visible pass-guard benchmark pattern before targeting "
                    "remaining failures."
                ),
                "reason": f"guard_preservation:{guard_id}",
            }
        )
        available_slots -= 1
        repaired_ids.append(guard_id)

    return candidate, {
        "repaired_pass_guard_ids": repaired_ids,
        "unrepaired_pass_guard_ids": unrepaired_ids,
        "pruned_duplicate_operation_count": pruned_duplicate_count,
    }


_TYPED_TEMPLATE_FAMILY_PRIORITY = (
    "literal_or_parameter_mismatch",
    "percentage_or_comparison_metric",
    "time_filter_mismatch",
    "selected_metric_or_column_mismatch",
)


def materialize_typed_template_examples(
    payload: dict[str, Any],
    *,
    benchmark_by_question_id: Mapping[str, Mapping[str, str]],
    structured_failure_context: Mapping[str, Any] | None,
    required_pass_guard_ids: frozenset[str] | set[str] | tuple[str, ...] | list[str] | None,
    max_mutation_operations: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize a compact typed-template proposal into executable visible examples.

    The serving endpoint chooses a visible target family/question. This helper
    fills concrete SQL only from the visible benchmark payload, injects all
    required visible pass guards first, and adds deterministic usage guidance.
    Hidden holdout payloads are never available to this function.
    """
    candidate = deepcopy(payload)
    original_operations = candidate.get("operations")
    operations = original_operations if isinstance(original_operations, list) else []

    target_ids = _typed_template_target_ids(
        operations=operations,
        structured_failure_context=structured_failure_context,
        benchmark_by_question_id=benchmark_by_question_id,
    )
    required_guards = tuple(
        dict.fromkeys(str(item) for item in (required_pass_guard_ids or []) if str(item).strip())
    )
    operation_budget = max_mutation_operations if max_mutation_operations is not None else 999

    materialized_operations: list[dict[str, Any]] = []
    skipped_ids: list[str] = []
    for guard_id in required_guards:
        operation = _typed_template_operation(
            question_id=guard_id,
            benchmark_by_question_id=benchmark_by_question_id,
            structured_failure_context=structured_failure_context,
            reason=f"guard_preservation:{guard_id}",
            guard=True,
        )
        if operation is None:
            skipped_ids.append(guard_id)
            continue
        materialized_operations.append(operation)

    target_slots = max(operation_budget - len(materialized_operations), 0)
    materialized_target_ids: list[str] = []
    for target_id in target_ids:
        if target_id in required_guards or target_id in materialized_target_ids:
            continue
        if target_slots <= 0:
            skipped_ids.append(target_id)
            continue
        operation = _typed_template_operation(
            question_id=target_id,
            benchmark_by_question_id=benchmark_by_question_id,
            structured_failure_context=structured_failure_context,
            reason=f"target_train_failure:{target_id}",
            guard=False,
        )
        if operation is None:
            skipped_ids.append(target_id)
            continue
        materialized_operations.append(operation)
        materialized_target_ids.append(target_id)
        target_slots -= 1

    candidate["operations"] = materialized_operations or [
        {
            "op": "no_change",
            "reason": "typed_template_no_visible_materializable_target",
        }
    ]
    candidate["file_edits"] = None

    return candidate, {
        "typed_template_materialized_guard_ids": [
            operation["id"]
            for operation in materialized_operations
            if str(operation.get("reason") or "").startswith("guard_preservation:")
        ],
        "typed_template_materialized_target_ids": materialized_target_ids,
        "typed_template_skipped_ids": list(dict.fromkeys(skipped_ids)),
        "typed_template_family_by_target_id": {
            question_id: _failure_family_for_question_id(
                question_id,
                structured_failure_context=structured_failure_context,
            )
            for question_id in materialized_target_ids
        },
    }


def _typed_template_target_ids(
    *,
    operations: list[Any],
    structured_failure_context: Mapping[str, Any] | None,
    benchmark_by_question_id: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    hinted_ids: list[str] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        if str(operation.get("op") or "").strip() not in {"add_example_sql", "no_change"}:
            continue
        for question_id in _operation_mentioned_question_ids(operation, benchmark_by_question_id):
            if question_id not in hinted_ids:
                hinted_ids.append(question_id)
    if hinted_ids:
        return tuple(hinted_ids[:_MAX_EXAMPLE_SET_TARGET_FAILURES])

    failures_by_id = _structured_failures_by_id(structured_failure_context)
    ranked: list[tuple[int, int, str, str]] = []
    for question_id, failure in failures_by_id.items():
        if question_id not in benchmark_by_question_id:
            continue
        family = str(failure.get("likely_root_cause_family") or "")
        try:
            family_rank = _TYPED_TEMPLATE_FAMILY_PRIORITY.index(family)
        except ValueError:
            continue
        risk = str(failure.get("guard_regression_risk") or "").lower()
        risk_rank = {"low": 0, "medium": 1, "high": 2}.get(risk, 1)
        ranked.append((family_rank, risk_rank, question_id, family))

    if not ranked:
        return ()
    ranked.sort()
    first_family = ranked[0][3]
    output: list[str] = []
    for _family_rank, risk_rank, question_id, family in ranked:
        if family != first_family:
            continue
        if output and risk_rank >= 2:
            continue
        output.append(question_id)
        if len(output) >= _MAX_EXAMPLE_SET_TARGET_FAILURES:
            break
    return tuple(output)


def _typed_template_operation(
    *,
    question_id: str,
    benchmark_by_question_id: Mapping[str, Mapping[str, str]],
    structured_failure_context: Mapping[str, Any] | None,
    reason: str,
    guard: bool,
) -> dict[str, Any] | None:
    benchmark = benchmark_by_question_id.get(question_id) or {}
    expected_sql = str(benchmark.get("expected_sql") or "").strip()
    if not expected_sql:
        return None
    question = str(benchmark.get("question") or "").strip() or question_id
    family = _failure_family_for_question_id(
        question_id,
        structured_failure_context=structured_failure_context,
    )
    guidance = (
        "Preserve this visible pass-guard typed template before targeting failures."
        if guard
        else _typed_template_usage_guidance(family)
    )
    return {
        "op": "add_example_sql",
        "id": question_id,
        "question": question,
        "sql": expected_sql,
        "usage_guidance": guidance,
        "reason": reason,
    }


def _failure_family_for_question_id(
    question_id: str,
    *,
    structured_failure_context: Mapping[str, Any] | None,
) -> str:
    failures_by_id = _structured_failures_by_id(structured_failure_context)
    failure = failures_by_id.get(question_id) or {}
    return str(failure.get("likely_root_cause_family") or "unknown")


def _typed_template_usage_guidance(family: str) -> str:
    if family == "literal_or_parameter_mismatch":
        return (
            "Typed template: preserve the concrete literal brand, metric, market, hierarchy, "
            "and time-bucket filters from this visible SQL. Never emit unbound :param placeholders."
        )
    if family == "percentage_or_comparison_metric":
        return (
            "Typed template: preserve current-versus-previous period CTEs, denominator semantics, "
            "and percentage/change output aliases from this visible SQL."
        )
    if family == "time_filter_mismatch":
        return (
            "Typed template: preserve the exact weekly/monthly/quarterly bucket columns, labels, "
            "and period filters from this visible SQL."
        )
    if family == "selected_metric_or_column_mismatch":
        return (
            "Typed template: preserve the selected output columns and aliases from this visible SQL; "
            "do not replace requested facts with lookup/existence checks."
        )
    return (
        "Typed template: reuse this visible executable SQL shape only for the same visible "
        "failure family while preserving pass guards."
    )


def _mutation_operation_count(operations: list[Any]) -> int:
    return sum(
        1
        for operation in operations
        if isinstance(operation, dict)
        and str(operation.get("op") or "").strip()
        and str(operation.get("op") or "").strip() != "no_change"
    )


def _prune_duplicate_guard_operations(operations: list[Any]) -> int:
    seen_signatures: set[tuple[str, str, str]] = set()
    keep_operations: list[Any] = []
    pruned_count = 0
    for operation in operations:
        if (
            isinstance(operation, dict)
            and str(operation.get("op") or "").strip() == "add_example_sql"
            and str(operation.get("reason") or "").startswith("guard_preservation:")
        ):
            signature = (
                str(operation.get("reason") or "").strip(),
                _operation_field_text(operation, ("question",)).strip().lower(),
                _normalize_example_sql_for_gate(_operation_sql_text(operation)),
            )
            if signature in seen_signatures:
                pruned_count += 1
                continue
            seen_signatures.add(signature)
        keep_operations.append(operation)
    if pruned_count:
        operations[:] = keep_operations
    return pruned_count


def _append_text_instruction(config: GenieSpaceConfig, operation: dict[str, Any]) -> tuple[bool, str]:
    content = str(operation.get("content") or "").strip()
    if not content:
        return False, "missing_content"
    instructions = _ensure_instruction_container(config)
    lines = instructions.text_instructions[0].content or []
    joined = "\n".join(lines)
    if content in joined:
        return False, "duplicate_content"
    if lines and lines[-1].strip():
        lines.append("")
    lines.extend(content.splitlines())
    instructions.text_instructions[0].content = lines
    return True, "appended_text_instruction"


def _add_example_sql(config: GenieSpaceConfig, operation: dict[str, Any]) -> tuple[bool, str]:
    question_lines = _normalize_lines(operation.get("question"))
    sql_lines = _normalize_lines(operation.get("sql"))
    if not question_lines or not sql_lines:
        return False, "missing_question_or_sql"
    instructions = _ensure_instruction_container(config)
    existing = list(instructions.example_question_sqls or [])
    signature = ("\n".join(question_lines).strip().lower(), "\n".join(sql_lines).strip().lower())
    existing_signatures = {
        ("\n".join(item.question or []).strip().lower(), "\n".join(item.sql or []).strip().lower())
        for item in existing
    }
    if signature in existing_signatures:
        return False, "duplicate_example"
    existing.append(
        GenieExampleSQL(
            id=_valid_id(operation.get("id")),
            question=question_lines,
            sql=sql_lines,
            usage_guidance=_normalize_lines(operation.get("usage_guidance")),
        )
    )
    instructions.example_question_sqls = existing
    return True, "added_example_sql"


def _add_sql_snippet(config: GenieSpaceConfig, operation: dict[str, Any]) -> tuple[bool, str]:
    group = str(operation.get("group") or "").strip()
    if group not in _VALID_SNIPPET_GROUPS:
        return False, "invalid_snippet_group"
    display_name = _sql_snippet_display_name(operation)
    sql_lines = _normalize_lines(operation.get("sql"))
    if not display_name or not sql_lines:
        return False, "missing_display_name_or_sql"
    operation["display_name"] = display_name
    shape_rejection = _sql_snippet_shape_rejection_reason("\n".join(sql_lines))
    if shape_rejection:
        return False, shape_rejection
    option_set_rejection = _sql_snippet_option_set_rejection_reason(operation, sql_lines)
    if option_set_rejection:
        return False, option_set_rejection
    snippets = _ensure_sql_snippets(config)
    existing = list(getattr(snippets, group) or [])
    signature = (display_name.lower(), "\n".join(sql_lines).strip().lower())
    existing_signatures = {
        ((item.display_name or "").strip().lower(), "\n".join(item.sql or []).strip().lower())
        for item in existing
    }
    if signature in existing_signatures:
        return False, "duplicate_snippet"
    existing.append(
        GenieSQLSnippet(
            id=_valid_id(operation.get("id")),
            alias=str(operation.get("alias") or "").strip() or None,
            display_name=display_name,
            sql=sql_lines,
            synonyms=_normalize_optional_list(operation.get("synonyms")),
            comment=_normalize_lines(operation.get("comment")),
            instruction=_normalize_lines(operation.get("instruction")),
        )
    )
    setattr(snippets, group, existing)
    return True, "added_sql_snippet"


def _sql_snippet_display_name(operation: Mapping[str, Any]) -> str:
    explicit = str(operation.get("display_name") or "").strip()
    if explicit:
        return _truncate_display_name(explicit)

    for key in ("description", "question", "id", "alias"):
        lines = _normalize_lines(operation.get(key))
        if not lines:
            continue
        candidate = lines[0].replace("_", " ").strip()
        if candidate:
            return _truncate_display_name(candidate)
    return ""


def _truncate_display_name(value: str, *, max_chars: int = 96) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _add_join_spec(config: GenieSpaceConfig, operation: dict[str, Any]) -> tuple[bool, str]:
    left_identifier = str(operation.get("left_identifier") or "").strip()
    right_identifier = str(operation.get("right_identifier") or "").strip()
    left_alias = str(operation.get("left_alias") or "").strip()
    right_alias = str(operation.get("right_alias") or "").strip()
    sql_lines = _normalize_lines(operation.get("sql"))
    if not left_identifier or not right_identifier or not left_alias or not right_alias or not sql_lines:
        return False, "missing_join_fields"
    if _find_table_like(config, left_identifier) is None or _find_table_like(config, right_identifier) is None:
        return False, "unknown_join_table"
    instructions = _ensure_instruction_container(config)
    existing = list(instructions.join_specs or [])
    signature = (left_identifier, right_identifier, tuple(sql_lines))
    existing_signatures = {
        (item.left.identifier, item.right.identifier, tuple(item.sql or []))
        for item in existing
    }
    if signature in existing_signatures:
        return False, "duplicate_join"
    existing.append(
        GenieJoinSpecs(
            id=_valid_id(operation.get("id")),
            left=GenieTableJoinSpec(identifier=left_identifier, alias=left_alias),
            right=GenieTableJoinSpec(identifier=right_identifier, alias=right_alias),
            sql=sql_lines,
            comment=_normalize_lines(operation.get("comment")),
            instruction=_normalize_lines(operation.get("instruction")),
        )
    )
    instructions.join_specs = existing
    return True, "added_join_spec"


def _set_table_description(config: GenieSpaceConfig, operation: dict[str, Any]) -> tuple[bool, str]:
    table_identifier = str(operation.get("table_identifier") or "").strip()
    description = _normalize_lines(operation.get("description"))
    if not table_identifier or not description:
        return False, "missing_table_description"
    table_like = _find_table_like(config, table_identifier)
    if table_like is None:
        return False, "unknown_table"
    if table_like.description == description:
        return False, "table_description_unchanged"
    table_like.description = description
    return True, "set_table_description"


def _set_column_config(config: GenieSpaceConfig, operation: dict[str, Any]) -> tuple[bool, str]:
    table_identifier = str(operation.get("table_identifier") or "").strip()
    column_name = str(operation.get("column_name") or "").strip()
    if not table_identifier or not column_name:
        return False, "missing_column_reference"
    column = _find_column(config, table_identifier, column_name)
    if column is None:
        return False, "unknown_column"

    changed = False
    description = _normalize_lines(operation.get("description"))
    if description is not None and column.description != description:
        column.description = description
        changed = True
    synonyms = _normalize_optional_list(operation.get("synonyms"))
    if synonyms is not None:
        merged = list(dict.fromkeys([*(column.synonyms or []), *synonyms]))
        if column.synonyms != merged:
            column.synonyms = merged
            changed = True
    for key in ("exclude", "enable_format_assistance", "enable_entity_matching"):
        value = operation.get(key)
        if value is None:
            continue
        bool_value = bool(value)
        if getattr(column, key) != bool_value:
            setattr(column, key, bool_value)
            changed = True
    return (changed, "set_column_config" if changed else "column_config_unchanged")


_OPERATION_APPLIERS = {
    "append_text_instruction": _append_text_instruction,
    "add_example_sql": _add_example_sql,
    "add_sql_snippet": _add_sql_snippet,
    "add_join_spec": _add_join_spec,
    "set_table_description": _set_table_description,
    "set_column_config": _set_column_config,
}


_PASS_GUARD_OVERLAP_STOPWORDS = frozenset(
    {
        "and",
        "available",
        "brand",
        "business",
        "change",
        "compare",
        "current",
        "data",
        "difference",
        "filter",
        "for",
        "from",
        "give",
        "group",
        "how",
        "level",
        "list",
        "market",
        "metric",
        "name",
        "national",
        "only",
        "percentage",
        "previous",
        "region",
        "sales",
        "show",
        "speciality",
        "the",
        "total",
        "view",
        "week",
        "weeks",
        "what",
        "when",
        "with",
    }
)


def _significant_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9][a-z0-9_+]{2,}", text.lower()))
    return {
        term
        for term in terms
        if term not in _PASS_GUARD_OVERLAP_STOPWORDS and not term.isdigit()
    }


def _snippet_pass_guard_overlap_reasons(
    operation: dict[str, Any],
    *,
    pass_guard_questions_by_id: Mapping[str, str],
) -> list[str]:
    if str(operation.get("op") or "").strip() != "add_sql_snippet":
        return []
    if str(operation.get("group") or "").strip() != "filters":
        return []
    snippet_terms = _significant_terms(
        _operation_field_text(
            operation,
            (
                "display_name",
                "synonyms",
                "sql",
                "usage_guidance",
                "question",
                "description",
                "column_name",
            ),
        )
    )
    if not snippet_terms:
        return []
    reasons: list[str] = []
    for guard_id, guard_question in pass_guard_questions_by_id.items():
        overlap = sorted(snippet_terms & _significant_terms(guard_question))
        if overlap:
            reasons.append(f"sql_snippet_overlaps_pass_guard:{guard_id}:{overlap[0]}")
    return reasons


def _metadata_alias_rejection_reasons(operation: dict[str, Any]) -> list[str]:
    """Reject high-blast-radius metadata edits in the guarded alias lane."""
    op_name = str(operation.get("op") or "").strip()
    if op_name not in {"set_column_config", "set_table_description"}:
        return []

    reasons: list[str] = []
    if op_name == "set_column_config" and operation.get("exclude") is True:
        reasons.append("metadata_aliases_cannot_exclude_columns")

    metadata_text = _operation_field_text(
        operation,
        (
            "description",
            "synonyms",
            "reason",
            "comment",
            "instruction",
        ),
    )
    if _contains_bind_placeholder(metadata_text):
        reasons.append("metadata_aliases_contains_placeholder")
    if re.search(r"\bselect\b.+\bfrom\b", metadata_text, re.IGNORECASE | re.DOTALL):
        reasons.append("metadata_aliases_contains_sql_payload")
    return reasons


def assess_serving_candidate_risk(
    payload: dict[str, Any],
    *,
    candidate_mode: str | None = None,
    allowed_operation_groups: frozenset[str] | set[str] | None = None,
    require_guardrail_acknowledgement: bool = False,
    max_mutation_operations: int | None = None,
    required_pass_guard_ids: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    expected_sql_by_question_id: Mapping[str, str] | None = None,
    pass_guard_questions_by_id: Mapping[str, str] | None = None,
    failure_family_by_question_id: Mapping[str, str] | None = None,
    guard_regression_risk_by_question_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reject high-blast-radius serving proposals before they touch a clone."""
    operations = payload.get("operations")
    if not isinstance(operations, list):
        operations = []

    mutation_operations: list[dict[str, Any]] = []
    no_change_count = 0
    reasons: list[str] = []
    groups: list[str] = []
    for raw_operation in operations:
        if not isinstance(raw_operation, dict):
            reasons.append("invalid_operation")
            continue
        op_name = str(raw_operation.get("op") or "").strip()
        if op_name == "no_change":
            no_change_count += 1
            continue
        group = _OPERATION_RISK_GROUPS.get(op_name)
        if group is None:
            reasons.append(f"unsupported_operation:{op_name or 'missing'}")
            continue
        mutation_operations.append(raw_operation)
        groups.append(group)

    unique_groups = sorted(set(groups))
    mutation_count = len(mutation_operations)
    if no_change_count and mutation_count:
        reasons.append("mixed_no_change_with_mutations")
    if len(unique_groups) > 1:
        reasons.append("mixed_operation_groups:" + ",".join(unique_groups))
    if allowed_operation_groups is not None and mutation_count:
        disallowed_groups = sorted(set(unique_groups) - set(allowed_operation_groups))
        if disallowed_groups:
            reasons.append("disallowed_operation_groups:" + ",".join(disallowed_groups))
    if require_guardrail_acknowledgement and mutation_count:
        acknowledgement_text = " ".join(
            str(value or "")
            for value in (
                payload.get("candidate_name"),
                payload.get("rationale"),
                payload.get("expected_impact"),
                payload.get("risk"),
            )
        ).lower()
        guardrail_markers = ("guard", "pass guard", "pass-guard", "preserve", "regress")
        if not any(marker in acknowledgement_text for marker in guardrail_markers):
            reasons.append("missing_guardrail_acknowledgement")
    if max_mutation_operations is not None:
        if mutation_count > max_mutation_operations:
            reasons.append(f"too_many_operations:{mutation_count}")
    elif unique_groups == ["metadata"] and mutation_count > _MAX_METADATA_OPERATIONS_PER_CANDIDATE:
        reasons.append(f"too_many_metadata_operations:{mutation_count}")
    elif unique_groups and unique_groups[0] != "metadata" and mutation_count > 1:
        reasons.append(f"too_many_{unique_groups[0]}_operations:{mutation_count}")

    for raw_operation in mutation_operations:
        op_name = str(raw_operation.get("op") or "").strip()
        if pass_guard_questions_by_id:
            reasons.extend(
                _snippet_pass_guard_overlap_reasons(
                    raw_operation,
                    pass_guard_questions_by_id=pass_guard_questions_by_id,
                )
            )
        if candidate_mode == "schema_metadata_aliases":
            reasons.extend(_metadata_alias_rejection_reasons(raw_operation))
        if op_name == "add_example_sql":
            sql_text = _operation_sql_text(raw_operation)
            matched_ids = (
                _operation_mentioned_question_ids(
                    raw_operation,
                    expected_sql_by_question_id,
                )
                if expected_sql_by_question_id
                else []
            )
            if not sql_text.strip() and not matched_ids:
                reasons.append("example_sql_missing_sql_or_benchmark_id")
            if re.search(r"(?<!:):[A-Za-z_][A-Za-z0-9_]*|\$\{", sql_text):
                reasons.append("example_sql_contains_placeholder")
            if expected_sql_by_question_id:
                for question_id in matched_ids[:1]:
                    expected_sql = expected_sql_by_question_id.get(question_id)
                    if expected_sql and _normalize_example_sql_for_gate(sql_text) != _normalize_example_sql_for_gate(expected_sql):
                        reasons.append(f"example_sql_mismatch:{question_id}")
        if op_name == "add_sql_snippet":
            sql_text = _operation_sql_text(raw_operation)
            shape_rejection = _sql_snippet_shape_rejection_reason(sql_text)
            if shape_rejection:
                reasons.append(shape_rejection)
            sql_lines = _normalize_lines(raw_operation.get("sql")) or []
            option_set_rejection = _sql_snippet_option_set_rejection_reason(
                raw_operation,
                sql_lines,
            )
            if option_set_rejection:
                reasons.append(option_set_rejection)
        if op_name != "append_text_instruction":
            continue
        content = str(raw_operation.get("content") or "").strip()
        if len(content) > _MAX_TEXT_INSTRUCTION_CHARS:
            reasons.append(f"text_instruction_too_long:{len(content)}")

    target_failure_ids: list[str] = []
    for raw_operation in mutation_operations:
        if str(raw_operation.get("op") or "").strip() != "add_example_sql":
            continue
        reason = str(raw_operation.get("reason") or "")
        if not reason.startswith("target_train_failure:"):
            continue
        target_id = reason.split(":", 1)[1].strip()
        if target_id:
            target_failure_ids.append(target_id)

    unique_target_failure_ids = tuple(dict.fromkeys(target_failure_ids))
    if len(unique_target_failure_ids) > 1 and failure_family_by_question_id:
        target_families = {
            question_id: str(failure_family_by_question_id.get(question_id) or "")
            for question_id in unique_target_failure_ids
            if failure_family_by_question_id.get(question_id)
        }
        if len(set(target_families.values())) > 1:
            reason_detail = ",".join(
                f"{question_id}={family}"
                for question_id, family in sorted(target_families.items())
            )
            reasons.append(f"mixed_target_failure_families:{reason_detail}")
    if len(unique_target_failure_ids) > 1 and guard_regression_risk_by_question_id:
        high_risk_target_ids = [
            question_id
            for question_id in unique_target_failure_ids
            if str(guard_regression_risk_by_question_id.get(question_id) or "").lower() == "high"
        ]
        if high_risk_target_ids:
            reasons.append("multiple_high_guard_risk_targets:" + ",".join(high_risk_target_ids[:5]))

    required_guards = tuple(dict.fromkeys(str(item) for item in (required_pass_guard_ids or []) if str(item).strip()))
    if required_guards and mutation_count:
        covered_guards: set[str] = set()
        for raw_operation in mutation_operations:
            if str(raw_operation.get("op") or "").strip() != "add_example_sql":
                continue
            reason = str(raw_operation.get("reason") or "")
            operation_text = " ".join(
                str(raw_operation.get(key) or "")
                for key in (
                    "reason",
                    "description",
                    "usage_guidance",
                    "question",
                )
            )
            for guard_id in required_guards:
                if guard_id in operation_text:
                    covered_guards.add(guard_id)
        missing_guards = [guard_id for guard_id in required_guards if guard_id not in covered_guards]
        if missing_guards:
            reasons.append("missing_pass_guard_examples:" + ",".join(missing_guards[:5]))
        minimum_example_set_operations = len(required_guards) + 1
        if mutation_count < minimum_example_set_operations:
            reasons.append(
                f"too_few_example_set_operations:{mutation_count}<{minimum_example_set_operations}"
            )
        if len(target_failure_ids) > _MAX_EXAMPLE_SET_TARGET_FAILURES:
            reasons.append(
                f"too_many_target_failure_examples:{len(target_failure_ids)}>{_MAX_EXAMPLE_SET_TARGET_FAILURES}"
            )

    accepted = not reasons
    return {
        "accepted": accepted,
        "reason": None if accepted else "serving_candidate_risk_rejected",
        "reasons": reasons,
        "operation_count": len(operations),
        "mutation_operation_count": mutation_count,
        "operation_groups": unique_groups,
        "policy": {
            "one_operation_group": True,
            "max_metadata_operations": _MAX_METADATA_OPERATIONS_PER_CANDIDATE,
            "max_text_instruction_chars": _MAX_TEXT_INSTRUCTION_CHARS,
            "allowed_operation_groups": (
                sorted(allowed_operation_groups)
                if allowed_operation_groups is not None
                else None
            ),
            "require_guardrail_acknowledgement": require_guardrail_acknowledgement,
            "max_mutation_operations": max_mutation_operations,
            "required_pass_guard_ids": list(required_guards),
            "expected_sql_checked_ids": sorted(expected_sql_by_question_id or {}),
            "pass_guard_overlap_checked_ids": sorted(pass_guard_questions_by_id or {}),
            "failure_family_checked_ids": sorted(failure_family_by_question_id or {}),
            "guard_regression_risk_checked_ids": sorted(guard_regression_risk_by_question_id or {}),
        },
    }


def apply_serving_candidate(
    config: GenieSpaceConfig,
    payload: dict[str, Any],
) -> tuple[GenieSpaceConfig, dict[str, Any]]:
    """Apply a structured serving proposal to a config."""
    candidate = deepcopy(config)
    operations = payload.get("operations")
    if not isinstance(operations, list):
        operations = []

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, raw_operation in enumerate(operations):
        if not isinstance(raw_operation, dict):
            skipped.append({"index": index, "reason": "invalid_operation"})
            continue
        operation = dict(raw_operation)
        op_name = str(operation.get("op") or "").strip()
        if op_name == "no_change":
            skipped.append({"index": index, "op": op_name, "reason": str(operation.get("reason") or "no_change")})
            continue
        applier = _OPERATION_APPLIERS.get(op_name)
        if applier is None:
            skipped.append({"index": index, "op": op_name, "reason": "unsupported_operation"})
            continue
        changed, reason = applier(candidate, operation)
        target = applied if changed else skipped
        entry = {"index": index, "op": op_name, "reason": reason}
        if operation.get("display_name"):
            entry["display_name"] = operation["display_name"]
        if operation.get("table_identifier"):
            entry["table_identifier"] = operation["table_identifier"]
        if operation.get("column_name"):
            entry["column_name"] = operation["column_name"]
        target.append(entry)

    summary = {
        "changed": bool(applied),
        "candidate_name": str(payload.get("candidate_name") or "serving_candidate"),
        "rationale": str(payload.get("rationale") or ""),
        "expected_impact": str(payload.get("expected_impact") or ""),
        "risk": str(payload.get("risk") or ""),
        "owner_followups": [
            str(item) for item in payload.get("owner_followups", []) if str(item).strip()
        ] if isinstance(payload.get("owner_followups"), list) else [],
        "operation_count": len(operations),
        "applied_operations": applied,
        "skipped_operations": skipped,
    }
    if not applied:
        summary["reason"] = "no_valid_serving_operations"
    return candidate, summary


def _extract_content(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = response.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content_item in item.get("content", []) or []:
                    if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                        parts.append(str(content_item.get("text") or ""))
            elif item.get("type") == "custom_tool_call" and item.get("input"):
                parts.append(str(item["input"]))
        if any(part.strip() for part in parts):
            return "\n".join(part for part in parts if part)
        if response.get("object") == "response":
            incomplete_details = response.get("incomplete_details")
            incomplete_reason = (
                incomplete_details.get("reason")
                if isinstance(incomplete_details, dict)
                else None
            )
            status = response.get("status")
            raise ValueError(
                "Serving response did not include candidate JSON output_text"
                f" (status={status or 'unknown'}, reason={incomplete_reason or 'unknown'})."
            )
    if isinstance(response.get("candidate"), dict):
        return json.dumps(response["candidate"], ensure_ascii=False)
    if isinstance(response.get("outputs"), list) and response["outputs"]:
        first_output = response["outputs"][0]
        if isinstance(first_output, dict):
            if first_output.get("content"):
                return str(first_output["content"])
            return json.dumps(first_output, ensure_ascii=False)
        return str(first_output)
    predictions = response.get("predictions")
    if isinstance(predictions, list) and predictions:
        first_prediction = predictions[0]
        if isinstance(first_prediction, dict):
            if "choices" in first_prediction:
                return _extract_content(first_prediction)
            if first_prediction.get("content"):
                return str(first_prediction["content"])
            return json.dumps(first_prediction, ensure_ascii=False)
        return str(first_prediction)
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("text"):
                            parts.append(str(item["text"]))
                    if parts:
                        return "\n".join(parts)
            if first_choice.get("text"):
                return str(first_choice["text"])
    if response.get("content"):
        return str(response["content"])
    return json.dumps(response, ensure_ascii=False)


def _parse_json_payload_with_repairs(content: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    stripped = content.strip()
    repairs: list[str] = []
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        repairs.append("stripped_markdown_fence")
    candidates = [stripped]
    balanced = _extract_balanced_json_object(stripped)
    balanced_trailing_comma_repaired: str | None = None
    if balanced and balanced not in candidates:
        candidates.append(balanced)
        balanced_trailing_comma_repaired = _strip_trailing_json_commas(balanced)
        if balanced_trailing_comma_repaired not in candidates:
            candidates.append(balanced_trailing_comma_repaired)
    trailing_comma_repaired = _strip_trailing_json_commas(stripped)
    if trailing_comma_repaired not in candidates:
        candidates.append(trailing_comma_repaired)
    balanced_repaired = _balance_json_delimiters(trailing_comma_repaired)
    if balanced_repaired and balanced_repaired not in candidates:
        candidates.append(balanced_repaired)

    last_error: Exception | None = None
    for index, candidate in enumerate(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise ValueError("Serving response JSON must be an object.")
        if index > 0:
            if candidate == balanced:
                repairs.append("extracted_balanced_json_object")
            if candidate == balanced_trailing_comma_repaired:
                repairs.append("extracted_balanced_json_object")
            if candidate in {trailing_comma_repaired, balanced_trailing_comma_repaired}:
                repairs.append("removed_trailing_json_commas")
            if candidate == balanced_repaired:
                repairs.append("balanced_json_delimiters")
        return payload, tuple(dict.fromkeys(repairs))

    try:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            if last_error is not None:
                raise last_error
            raise ValueError("Serving response JSON did not contain an object.")
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        if last_error is not None:
            raise last_error
        raise
    if not isinstance(payload, dict):
        raise ValueError("Serving response JSON must be an object.")
    return payload, ("regex_json_object_extract",)


def _parse_json_payload(content: str) -> dict[str, Any]:
    payload, _repairs = _parse_json_payload_with_repairs(content)
    return payload


def _normalize_candidate_payload_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Backfill schema defaults that older/non-strict responses can omit."""
    normalized = dict(payload)
    normalized.setdefault("causal_patch_plan", None)
    file_edits = normalized.get("file_edits")
    if isinstance(file_edits, list):
        normalized["file_edits"] = [
            _normalize_file_edit_payload_defaults(edit) for edit in file_edits
        ]
    return normalized


def _normalize_file_edit_payload_defaults(edit: Any) -> Any:
    if not isinstance(edit, Mapping):
        return edit
    normalized = dict(edit)
    for field_name in _FILE_EDIT_NULLABLE_FIELDS:
        normalized.setdefault(field_name, None)
    return normalized


def _strip_trailing_json_commas(value: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", value)


def _extract_balanced_json_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return None


def _balance_json_delimiters(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in value[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                return None
            stack.pop()
    if in_string or not stack:
        return None
    return value[start:] + "".join(reversed(stack))


def _response_format_payload(mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "maxgenie_candidate",
                "strict": True,
                "schema": _candidate_response_schema(),
            },
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def _responses_text_payload(mode: str) -> dict[str, Any] | None:
    response_format = _response_format_payload(mode)
    if response_format is None:
        return None
    if mode == "json_schema":
        json_schema = response_format["json_schema"]
        return {
            "format": {
                "type": "json_schema",
                "name": json_schema["name"],
                "strict": json_schema.get("strict", True),
                "schema": json_schema["schema"],
            }
        }
    return {"format": response_format}


def _normalize_responses_reasoning_effort(
    value: str | None,
    *,
    model_name: str | None = None,
) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"xhigh", "extra_high", "extra-high", "max", "maximum"}:
        normalized = "xhigh"
    if normalized == "xhigh" and "pro" in (model_name or "").strip().lower():
        return "high"
    if normalized in {"low", "medium", "high", "xhigh"}:
        return normalized
    return normalized


def _uses_responses_api(config: ServingCandidateConfig) -> bool:
    if config.api_format == "responses":
        return True
    if config.api_format == "chat":
        return False
    model_name = (config.model or config.endpoint).strip().lower()
    return model_name.startswith("databricks-gpt-5")


def _uses_adaptive_thinking_controls(config: ServingCandidateConfig) -> bool:
    """True when the target endpoint requires adaptive thinking controls.

    These endpoints reject ``reasoning_effort`` and the legacy enabled-thinking
    shape; they use ``thinking.type=adaptive`` plus ``output_config.effort`` and
    cannot combine thinking with a forced ``response_format`` tool.
    """
    model_name = (config.model or config.endpoint).strip().lower()
    return "claude" in model_name


def _thinking_effort(value: str | None) -> str | None:
    """Translate a reasoning/thinking-effort string to a valid ``output_config.effort``.

    ``xhigh``/``extra_high``/``max``/``maximum`` map to ``high``; ``low``/``medium``/
    ``high`` pass through; ``none``/``null``/``omit``/``unset``/empty/None disable
    thinking (return None). Unknown values fall back to ``medium`` rather than passing
    an effort the endpoint would reject.
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "omit", "unset", "off", "disabled"}:
        return None
    if normalized in {"xhigh", "extra_high", "extra-high", "max", "maximum"}:
        return "high"
    if normalized in {"low", "medium", "high"}:
        return normalized
    return "medium"


def _apply_chat_reasoning_and_format(
    body: dict[str, Any],
    config: ServingCandidateConfig,
    *,
    apply_response_format: bool,
) -> None:
    """Apply model-aware thinking/reasoning controls and response_format to a chat body.

    Adaptive-thinking endpoints translate the configured effort to
    ``thinking.type=adaptive`` plus ``output_config.effort``, never send
    ``reasoning_effort``, and skip forced ``response_format`` while thinking is
    enabled. Other chat models keep the ``reasoning_effort`` plus
    ``response_format`` behavior.
    """
    if _uses_adaptive_thinking_controls(config):
        effort = _thinking_effort(config.reasoning_effort)
        if effort is not None:
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
            return  # forced response_format cannot coexist with adaptive thinking
        if apply_response_format:
            response_format = _response_format_payload(config.response_format)
            if response_format is not None:
                body["response_format"] = response_format
        return

    if config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    if apply_response_format:
        response_format = _response_format_payload(config.response_format)
        if response_format is not None:
            body["response_format"] = response_format


def build_serving_candidate_history_analysis_prompt(
    *,
    current_config: GenieSpaceConfig,
    failure_context: str,
    structured_failure_context: Mapping[str, Any] | None = None,
    trace_context_pack: Mapping[str, Any] | None = None,
    visible_proxy_guard_payload: Mapping[str, Any] | None = None,
    decomposed_space_payload: dict[str, Any] | None = None,
    visible_benchmark_payload: dict[str, Any],
    optimization_summary: dict[str, Any] | None,
    candidate_mode: str,
) -> str:
    """Build a visible-only historical analysis prompt before broad proposal."""
    return "\n".join(
        [
            "Analyze the visible MaxGenie optimization history before a broad candidate is proposed.",
            "Return concise plain text, not JSON. Do not propose file_edits directly.",
            "Use only train, visible-validation, visible pass-guard, decomposed file, diagnostics, and prior-attempt evidence.",
            "Do not use, infer, request, or summarize hidden holdout question text, hidden holdout SQL, hidden failed IDs, or final-test details.",
            "",
            "Focus on:",
            "- patterns that improved visible train/validation without guard regressions",
            "- patterns that caused train collapse, pass-guard regression, malformed payloads, or risk rejection",
            "- edit surfaces that still look promising from visible evidence",
            "- explicit warnings for the next proposer",
            "- a short recommendation for whether the broad freedom recipe should patch, narrow scope, or return no_change",
            "",
            f"## Candidate Mode\n{candidate_mode}",
            "",
            "## Current Editable Space",
            _truncate_json(current_config.to_serialized_space(), limit=60000),
            "",
            "## Decomposed Space Files",
            _truncate_json(decomposed_space_payload or {}, limit=60000),
            "",
            "## Structured Visible Failure Context",
            _truncate_json(structured_failure_context or {}, limit=60000),
            "",
            "## Visible Proxy Guards",
            _truncate_json(visible_proxy_guard_payload or {}, limit=30000),
            "",
            "## Visible Benchmarks",
            _truncate_json(visible_benchmark_payload, limit=30000),
            "",
            "## Compact Trace Memory",
            _truncate_json(_proposal_safe_trace_context_pack(trace_context_pack), limit=50000),
            "",
            "## Optimization Summary",
            _truncate_json(optimization_summary or {}, limit=30000),
            "",
            "## Latest Failure Context",
            _truncate_text(failure_context, limit=80000),
        ]
    )


def build_serving_candidate_prompt(
    *,
    current_config: GenieSpaceConfig,
    failure_context: str,
    structured_failure_context: Mapping[str, Any] | None = None,
    trace_context_pack: Mapping[str, Any] | None = None,
    visible_proxy_guard_payload: Mapping[str, Any] | None = None,
    historical_analyst_memo: str | None = None,
    data_probe_payload: Mapping[str, Any] | None = None,
    visible_benchmark_payload: dict[str, Any],
    diagnostics_payload: dict[str, Any] | None,
    optimization_summary: dict[str, Any] | None,
    candidate_mode: str | None = None,
    decomposed_space_payload: dict[str, Any] | None = None,
) -> str:
    """Build the candidate proposal prompt from workspace artifacts."""
    config_payload = current_config.to_serialized_space()
    guard_context = _extract_markdown_section(failure_context, "Baseline Pass Guard Patterns")
    mode_guidance = _CANDIDATE_MODE_GUIDANCE.get(candidate_mode or "")
    broad_patch_mode = serving_candidate_is_broad_artifact_patch_mode(candidate_mode)
    mode_section = (
        [
            "## Required Candidate Mode",
            f"- Mode: {candidate_mode}",
            *[f"- {line}" for line in mode_guidance],
            "- If this mode cannot safely improve a visible train failure without pass-guard risk, return no_change.",
            *(
                [
                    "- For artifact_patch_v2 no_change, set operations to [], file_edits to [], and causal_patch_plan to null. Strict artifact-patch modes, including skeleton and broad recipe modes, use the same empty patch contract. Do not include a no_change operation and do not set file_edits to null.",
                ]
                if candidate_mode in _STRICT_ARTIFACT_PATCH_MODES
                else []
            ),
            "",
        ]
        if candidate_mode and mode_guidance
        else []
    )
    guard_section = (
        [
            "## Required Baseline Pass Guard Context",
            "This section is prioritized and intentionally repeated outside the truncated failure context.",
            "Use it for complete guard examples in recovery example-set modes; do not claim guard IDs are unavailable when listed here.",
            _truncate_text(guard_context, limit=80000),
            "",
        ]
        if guard_context
        else []
    )
    candidate_context_summary = _candidate_context_summary_lines(
        structured_failure_context=structured_failure_context,
        failure_context=failure_context,
        optimization_summary=optimization_summary,
        candidate_mode=candidate_mode,
        decomposed_space_payload=decomposed_space_payload,
    )
    trace_context_section = (
        [
            "## Compact Trace Memory",
            _truncate_json(_proposal_safe_trace_context_pack(trace_context_pack)),
            "",
        ]
        if isinstance(trace_context_pack, Mapping)
        else []
    )
    historical_analyst_section = (
        [
            "## Historical Analyst Memo",
            "This memo is generated by a separate visible-only serving call before proposal. Use it as advisory context. Do not infer hidden holdout details from it.",
            _truncate_text(historical_analyst_memo, limit=20000),
            "",
        ]
        if historical_analyst_memo and historical_analyst_memo.strip()
        else []
    )
    data_probe_section = (
        [
            "## Data Probe Diagnostics",
            "These are compact aggregate read-only SQL diagnostics gathered inside the Databricks job. They are not hidden holdout evidence.",
            _truncate_json(dict(data_probe_payload)),
            "",
        ]
        if isinstance(data_probe_payload, Mapping)
        else []
    )
    visible_proxy_guard_section = (
        [
            "## Visible Family Proxy Guards",
            "This guard set is derived only from train and visible-validation pass records. Hidden holdout text and SQL are omitted.",
            "Preserve these guard IDs and families when proposing a visible failure fix; prefer no_change over a narrow train lift that sacrifices these proxies.",
            _truncate_json(dict(visible_proxy_guard_payload)),
            "",
        ]
        if isinstance(visible_proxy_guard_payload, Mapping)
        else []
    )
    decomposed_space_section = (
        [
            "## Decomposed Space Files",
            "Use these allowlisted files and current hashes only when candidate_mode is an artifact patch mode. Copy expected_sha256 exactly from the edited file entry for every existing edited file; do not invent hashes or reuse hashes from prior candidates.",
            _truncate_json(decomposed_space_payload),
            "",
        ]
        if isinstance(decomposed_space_payload, dict)
        else []
    )
    return "\n".join(
        [
            "You are optimizing one Databricks Genie space by proposing exactly one structured candidate.",
            "Return only JSON matching the requested schema. Do not include markdown.",
            "",
            "The optimizer will apply your operations, push the clone, run benchmark gates, and reject regressions.",
            "Use only train and visible-pool evidence. Do not request or infer hidden holdout content.",
            "Prefer structural changes that generalize: joins, descriptions, entity matching, SQL snippets, examples, then text instructions.",
            "If no safe improvement is available, return no_change using the required candidate-mode contract and explain why.",
            "",
            "## Candidate Blast-Radius Rules",
            *(
                [
                    "- Broad artifact-patch modes may bundle multiple compatible edits and failure families when the causal_patch_plan explains why the bundle is more reliable than isolated edits.",
                    "- Every broad edit still needs visible attribution and pass-guard preservation; hidden holdout text, SQL, and failed-ID lists remain excluded.",
                ]
                if broad_patch_mode
                else [
                    "- Pick exactly one target failure family; example-set modes may include at most two failed examples and only when they are mutually compatible with the same preserved guard patterns.",
                ]
            ),
            "- Use exactly one operation group per candidate: metadata OR SQL examples OR one SQL snippet OR one join spec OR one concise text instruction OR one decomposed file patch.",
            *(
                [
                    "- In broad artifact-patch modes, a decomposed file patch may combine instructions, examples, snippets, joins, metadata, metric views, and sample questions when the edits are explicitly related.",
                ]
                if broad_patch_mode
                else [
                    "- Do not combine global text instructions with SQL examples, snippets, joins, or metadata in the same candidate.",
                ]
            ),
            "- Artifact patch mode is the only mode that may use file_edits; all other modes must set file_edits to null.",
            "- Include top-level causal_patch_plan for strict artifact_patch_v2 candidates; set it to null for non-patch modes or no_change. The plan must diagnose root_cause from visible evidence and justify the selected artifact edit surface before describing the edit.",
            "- Treat Baseline Pass Guard Patterns and Prior Optimization Attempts as hard constraints.",
            "- Treat visible validation pass guards as generalization proxy guards; do not sacrifice them for a narrow train-only fix.",
            "- If prior attempts regressed pass guards, explain in `risk` how this candidate preserves those guards; otherwise return no_change.",
            "- Do not repeat an operation family or target failure pattern that previously fixed one visible failure but regressed pass guards.",
            "- Outside example-set modes, add_example_sql must return exactly one literal example query. It must be directly executable Databricks SQL with no bind placeholders.",
            "- In example-set modes only, add one example for every visible baseline pass guard, then use remaining operation budget for mutually compatible failed train examples.",
            "- In example-set modes, every guard example must set reason to guard_preservation:<question_id>; every failure example must set reason to target_train_failure:<question_id>.",
            "- In example-set modes, include at most two target_train_failure examples. Prefer one target unless two failures share the same root cause and are compatible with every guard.",
            "- For benchmark-tagged add_example_sql operations, set id to the benchmark question_id and sql to null; MaxGenie will fill the exact expected SQL from the visible benchmark payload.",
            "- After an accepted example_set, the next example_set should target remaining compatible failed benchmark IDs from the current state, not previously fixed target_train_failure IDs.",
            "- Do not copy long benchmark SQL into example_set responses; the response must stay compact and valid JSON.",
            "- SQL examples and SQL guidance must describe directly executable Databricks SQL. SQL snippets must be valid SQL fragments for their group, not full SELECT/WITH queries or DML/DDL statements. Do not introduce bind placeholders like :time_grp_type, :brand_name, :metric_type, or ${parameter}; if a visible expected SQL uses literals, preserve literals rather than parameters.",
            "- Do not rewrite global trusted-query routing instructions; strict artifact_patch_v2 validation rejects trusted-query routing edits. Prefer narrow executable-literal filter guidance or benchmark-tagged examples.",
            "- For append_text_instruction, keep the content under 900 characters and avoid changing defaults for unrelated questions.",
            "- For metadata-only candidates, touch at most three columns/tables and enable entity matching only when the failure context shows entity resolution is the direct blocker.",
            *(
                [
                    "- Prefer a broad bundle only when it is more likely to generalize than separate benchmark-shaped edits; otherwise return no_change.",
                ]
                if broad_patch_mode
                else [
                    "- Prefer no_change over a broad candidate. The optimizer will reject broad proposals before running benchmarks.",
                ]
            ),
            "",
            *mode_section,
            *candidate_context_summary,
            *trace_context_section,
            *historical_analyst_section,
            *data_probe_section,
            *guard_section,
            *visible_proxy_guard_section,
            *decomposed_space_section,
            "## Current Editable Space",
            _truncate_json(config_payload),
            "",
            "## Visible Benchmarks",
            _truncate_json(visible_benchmark_payload),
            "",
            "## Latest Failure Context",
            _truncate_text(failure_context),
            "",
            "## Diagnostics",
            _truncate_json(diagnostics_payload or {}),
            "",
            "## Optimization Summary",
            _truncate_json(optimization_summary or {}),
            "",
            "## Operation Guidance",
            "- artifact_patch mode: set operations to [] and return file_edits for allowlisted decomposed files only. Keep the patch coherent and small.",
            "- artifact_patch_v2 mode: set operations to [], return at most two file_edits, use exactly one edit primitive per file edit, set unused nullable file-edit fields to null, keep path and relative_path consistent with no disagreement, copy expected_sha256 exactly from the current Decomposed Space Files entry for existing files, include causal_patch_plan for the chosen target plan with root_cause and edit_surface_evidence grounded in visible evidence, include the selected visible-effect marker such as target_train_failure:<question_id>, visible_family:<family>, or failure_family:<family> in each changed edit reason, description, or comment, include one listed structural_tokens value in each changed edit's new content when structural_tokens are present, for target-level executable examples/snippets prefer expected_token_hints and include an expected token in structural_tokens and changed content, for family-level executable examples/snippets prefer shared_expected_token_hints over shared_generated_token_hints and include a shared expected token when listed, use the expected visible literal rather than only the generated placeholder or column token for literal mismatches, use compact append_content or exact find/replace edits for existing files because full-file content rewrites are rejected for visible-context causal-plan patches, do not rewrite trusted-query routing instructions, and do not add new bind placeholders.",
            "- skeleton_artifact_patch_v2 mode: set operations to [], return at most one file_edit, bind it to exactly one deterministic skeleton, copy path/expected_sha256/find from that skeleton exactly, set content and append_content to null, fill only replace plus existing causal_patch_plan text fields, include required visible tokens or intent labels, and return no_change if no skeleton can be filled safely.",
            "- composite_artifact_patch_v2 mode: set operations to [], return two to four coordinated file_edits across compatible artifact families, copy expected_sha256 exactly for existing files, use compact deltas, tie each edit reason to visible family/guard evidence, and do not return a one-file generic instruction rewrite.",
            "- executable_artifact_patch_v2 mode: set operations to [], return one to three file_edits, include at least one changed instructions/examples/*.yml or instructions/sql_snippets/{filters,expressions,measures}/*.yml asset, and return no_change rather than another prose-only or metadata-only patch.",
            "- executable_artifact_patch_v2 examples must not be literal-swap clones of passing guards; do not copy a passing guard and change only brand, market, metric, time, or entity literals.",
            "- New executable example SQL that is structurally near-identical to visible pass-guard SQL while only the question literal/entity changes will be rejected before benchmarking.",
            "- freedom_artifact_patch_v1 mode: set operations to [], return up to twelve hash-checked file_edits across allowlisted decomposed assets when a broad visible-evidence patch is more plausible than isolated edits.",
            "- freedom_artifact_patch_with_history_v1 mode: use the separate Historical Analyst Memo plus current visible evidence to return up to twelve hash-checked file_edits, or no_change if history shows the broad patch is unstable.",
            "- data_probe_artifact_patch_v1 mode: use compact read-only Data Probe Diagnostics plus current visible evidence to return up to six hash-checked file_edits, or no_change when probes do not support a patch.",
            "- append_text_instruction: add concise durable guidance only when structured assets cannot express the fix.",
            "- add_example_sql: outside example-set modes, add one literal example only when it encodes a reusable pattern found in failures; if tagged with a benchmark question_id, set id to that ID and sql to null.",
            "- example-set modes: add one compact benchmark-tagged bundle that covers every visible baseline pass guard plus at most two compatible currently failing visible train failures. Use ids and reasons, not copied SQL.",
            "- add_sql_snippet: add reusable filters, expressions, or measures with valid SQL fragments.",
            "- add_join_spec: add joins only between existing tables or metric views.",
            "- set_table_description and set_column_config: improve metadata and matching toggles for known tables/columns.",
            "- no_change: use when the evidence points to data quality, missing permissions, or no safe candidate.",
        ]
    )


class ServingCandidateClient:
    """Client for workspace serving endpoint candidate proposals."""

    def __init__(self, *, wc: WorkspaceClient, config: ServingCandidateConfig):
        self.wc = wc
        self.config = config

    def analyze_history(
        self,
        *,
        current_config: GenieSpaceConfig,
        failure_context: str,
        structured_failure_context: Mapping[str, Any] | None = None,
        trace_context_pack: Mapping[str, Any] | None = None,
        visible_proxy_guard_payload: Mapping[str, Any] | None = None,
        decomposed_space_payload: dict[str, Any] | None = None,
        visible_benchmark_payload: dict[str, Any],
        optimization_summary: dict[str, Any] | None,
        candidate_mode: str,
    ) -> ServingCandidateAnalysis:
        """Ask the endpoint for a visible-only history memo before proposal."""
        prompt = build_serving_candidate_history_analysis_prompt(
            current_config=current_config,
            failure_context=failure_context,
            structured_failure_context=structured_failure_context,
            trace_context_pack=trace_context_pack,
            visible_proxy_guard_payload=visible_proxy_guard_payload,
            decomposed_space_payload=decomposed_space_payload,
            visible_benchmark_payload=visible_benchmark_payload,
            optimization_summary=optimization_summary,
            candidate_mode=candidate_mode,
        )
        max_tokens = min(self.config.max_tokens, 8000)
        if _uses_responses_api(self.config):
            body: dict[str, Any] = {
                "model": self.config.model or self.config.endpoint,
                "instructions": (
                    "You analyze visible-only MaxGenie optimization history. "
                    "Return concise plain text and do not include hidden holdout content."
                ),
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": max_tokens,
                **dict(self.config.extra_body),
            }
            reasoning_effort = _normalize_responses_reasoning_effort(
                self.config.reasoning_effort,
                model_name=self.config.model or self.config.endpoint,
            )
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}
            with _temporary_api_timeouts(
                self.wc.api_client,
                request_timeout_seconds=self.config.request_timeout_seconds,
                retry_timeout_seconds=self.config.retry_timeout_seconds,
            ):
                try:
                    response = self.wc.api_client.do(
                        "POST",
                        "/serving-endpoints/responses",
                        body=body,
                    )
                except requests.exceptions.RequestException as exc:
                    raise ServingCandidateResponseError(
                        f"Serving endpoint request failed: {exc}",
                        request=body,
                        response={},
                        content="",
                    ) from exc
        else:
            body = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You analyze visible-only MaxGenie optimization history. "
                            "Return concise plain text and do not include hidden holdout content."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
            }
            if self.config.temperature is not None:
                body["temperature"] = self.config.temperature
            body.update(dict(self.config.extra_body))
            if self.config.model:
                body["model"] = self.config.model
            _apply_chat_reasoning_and_format(body, self.config, apply_response_format=False)
            with _temporary_api_timeouts(
                self.wc.api_client,
                request_timeout_seconds=self.config.request_timeout_seconds,
                retry_timeout_seconds=self.config.retry_timeout_seconds,
            ):
                try:
                    response = self.wc.api_client.do(
                        "POST",
                        f"/serving-endpoints/{self.config.endpoint}/invocations",
                        body=body,
                    )
                except requests.exceptions.RequestException as exc:
                    raise ServingCandidateResponseError(
                        f"Serving endpoint request failed: {exc}",
                        request=body,
                        response={},
                        content="",
                    ) from exc
        return ServingCandidateAnalysis(
            request=body,
            response=response,
            content=_extract_content(response),
        )

    def propose(
        self,
        *,
        current_config: GenieSpaceConfig,
        failure_context: str,
        structured_failure_context: Mapping[str, Any] | None = None,
        trace_context_pack: Mapping[str, Any] | None = None,
        visible_proxy_guard_payload: Mapping[str, Any] | None = None,
        historical_analyst_memo: str | None = None,
        data_probe_payload: Mapping[str, Any] | None = None,
        visible_benchmark_payload: dict[str, Any],
        diagnostics_payload: dict[str, Any] | None,
        optimization_summary: dict[str, Any] | None,
        candidate_mode: str | None = None,
        decomposed_space_payload: dict[str, Any] | None = None,
    ) -> ServingCandidateProposal:
        prompt = build_serving_candidate_prompt(
            current_config=current_config,
            failure_context=failure_context,
            structured_failure_context=structured_failure_context,
            trace_context_pack=trace_context_pack,
            visible_proxy_guard_payload=visible_proxy_guard_payload,
            historical_analyst_memo=historical_analyst_memo,
            data_probe_payload=data_probe_payload,
            visible_benchmark_payload=visible_benchmark_payload,
            decomposed_space_payload=decomposed_space_payload,
            diagnostics_payload=diagnostics_payload,
            optimization_summary=optimization_summary,
            candidate_mode=candidate_mode,
        )
        if _uses_responses_api(self.config):
            body: dict[str, Any] = {
                "model": self.config.model or self.config.endpoint,
                "instructions": (
                    "You produce strict JSON candidate operations for Databricks Genie space optimization."
                ),
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": self.config.max_tokens,
                **dict(self.config.extra_body),
            }
            reasoning_effort = _normalize_responses_reasoning_effort(
                self.config.reasoning_effort,
                model_name=self.config.model or self.config.endpoint,
            )
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}
            text_payload = _responses_text_payload(self.config.response_format)
            if text_payload is not None:
                body["text"] = text_payload
            with _temporary_api_timeouts(
                self.wc.api_client,
                request_timeout_seconds=self.config.request_timeout_seconds,
                retry_timeout_seconds=self.config.retry_timeout_seconds,
            ):
                try:
                    response = self.wc.api_client.do(
                        "POST",
                        "/serving-endpoints/responses",
                        body=body,
                    )
                except requests.exceptions.RequestException as exc:
                    raise ServingCandidateResponseError(
                        f"Serving endpoint request failed: {exc}",
                        request=body,
                        response={},
                        content="",
                    ) from exc
        else:
            body = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You produce strict JSON candidate operations for Databricks Genie space optimization.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.config.max_tokens,
            }
            if self.config.temperature is not None:
                body["temperature"] = self.config.temperature
            body.update(dict(self.config.extra_body))
            if self.config.model:
                body["model"] = self.config.model
            _apply_chat_reasoning_and_format(body, self.config, apply_response_format=True)

            with _temporary_api_timeouts(
                self.wc.api_client,
                request_timeout_seconds=self.config.request_timeout_seconds,
                retry_timeout_seconds=self.config.retry_timeout_seconds,
            ):
                response = self.wc.api_client.do(
                    "POST",
                    f"/serving-endpoints/{self.config.endpoint}/invocations",
                    body=body,
                )
        try:
            content = _extract_content(response)
        except ValueError as exc:
            raise ServingCandidateResponseError(
                f"Serving response did not contain usable candidate content: {exc}",
                request=body,
                response=response,
                content="",
            ) from exc
        try:
            payload, parse_repairs = _parse_json_payload_with_repairs(content)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ServingCandidateResponseError(
                f"Serving response did not contain valid candidate JSON: {exc}",
                request=body,
                response=response,
                content=content,
            ) from exc
        payload = _normalize_candidate_payload_defaults(payload)
        return ServingCandidateProposal(
            request=body,
            response=response,
            content=content,
            payload=payload,
            parse_repairs=parse_repairs,
        )
