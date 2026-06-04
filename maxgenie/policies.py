"""Optimization policy transformations for Genie space configs."""

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from maxgenie.schemas import (
    GenieBenchmarkQuestion,
    GenieExampleSQL,
    GenieInstruction,
    GenieInstructions,
    GenieSQLSnippet,
    GenieSQLSnippets,
    GenieSpaceConfig,
)
from maxgenie.space_diagnostics import (
    entity_matching_candidates,
    expanded_entity_matching_candidates,
    filter_sql_snippet_candidates,
    format_assistance_candidates,
    inconsistent_filter_snippet_ids,
    instruction_contradiction_findings,
    join_spec_candidates,
    measure_sql_snippet_candidates,
    noise_column_candidates,
)


@dataclass(frozen=True)
class PolicyContext:
    """Context available to policy transforms."""

    original_config: GenieSpaceConfig
    train_benchmark_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class CandidatePolicy:
    """Named optimization policy."""

    name: str
    description: str
    apply: Callable[[GenieSpaceConfig, PolicyContext], GenieSpaceConfig]


def _ensure_instruction_container(config: GenieSpaceConfig) -> GenieInstructions:
    if not config.instructions:
        config.instructions = GenieInstructions(text_instructions=[GenieInstruction(content=[])])
    if not config.instructions.text_instructions:
        config.instructions.text_instructions = [GenieInstruction(content=[])]
    if not config.instructions.text_instructions[0].content:
        config.instructions.text_instructions[0].content = []
    return config.instructions


def _find_column(
    config: GenieSpaceConfig,
    *,
    table_identifier: str,
    column_name: str,
):
    for table in (config.data_sources.tables or []) if config.data_sources else []:
        if table.identifier != table_identifier:
            continue
        for column in table.column_configs or []:
            if column.column_name == column_name:
                return column
    for metric_view in (config.data_sources.metric_views or []) if config.data_sources else []:
        if metric_view.identifier != table_identifier:
            continue
        for column in metric_view.column_configs or []:
            if column.column_name == column_name:
                return column
    return None


def _ensure_sql_snippets(config: GenieSpaceConfig) -> GenieSQLSnippets:
    instructions = _ensure_instruction_container(config)
    if instructions.sql_snippets is None:
        instructions.sql_snippets = GenieSQLSnippets()
    return instructions.sql_snippets


def _append_unique_snippets(
    existing: list[GenieSQLSnippet] | None,
    generated: list[GenieSQLSnippet],
) -> list[GenieSQLSnippet]:
    merged = list(existing or [])
    seen = {
        (
            (snippet.display_name or "").strip().lower(),
            "\n".join(snippet.sql or []).strip().lower(),
        )
        for snippet in merged
    }
    for snippet in generated:
        signature = (
            (snippet.display_name or "").strip().lower(),
            "\n".join(snippet.sql or []).strip().lower(),
        )
        if signature in seen:
            continue
        merged.append(snippet)
        seen.add(signature)
    return merged


def _benchmark_examples(
    config: GenieSpaceConfig,
    allowed_ids: frozenset[str] | None = None,
) -> list[GenieExampleSQL]:
    """Extract benchmark Q->SQL as example pairs.

    Args:
        config: Space config containing benchmarks.
        allowed_ids: If provided, only include benchmarks with these IDs (train set).
    """
    examples: list[GenieExampleSQL] = []
    for benchmark in config.benchmarks.questions if config.benchmarks and config.benchmarks.questions else []:
        if allowed_ids is not None and benchmark.id not in allowed_ids:
            continue
        question_text = benchmark.question[0] if benchmark.question else ""
        expected_sql = _benchmark_sql(benchmark)
        if not question_text or not expected_sql:
            continue
        examples.append(GenieExampleSQL(id=benchmark.id, question=[question_text], sql=[expected_sql]))
    return examples


def _benchmark_sql(benchmark: GenieBenchmarkQuestion) -> str | None:
    for answer in benchmark.answer or []:
        if answer.format.upper() == "SQL":
            return "".join(answer.content or [])
    return None


def _is_comparison_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "compare",
            "difference",
            "percentage",
            "%",
            " vs ",
            "versus",
            "change",
        )
    )


def policy_instruction_coherence(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Normalize instructions and remove conflicting exact-vs-like guidance."""
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)
    content = instructions.text_instructions[0].content

    filtered: list[str] = []
    skip_like_block = False
    for line in content:
        lowered = line.lower()
        if "**10. like operator" in lowered:
            skip_like_block = True
            continue
        if skip_like_block:
            if lowered.strip() == "":
                skip_like_block = False
            continue
        if "use case-insensitive exact matching" in lowered:
            continue
        filtered.append(line)

    canonical_lines = [
        "\n",
        "**Canonical Time Mapping (deterministic):**\\r\\n",
        "- current 6 months -> time_grp_type=mth_grp, time_grp_value=C6M\\r\\n",
        "- previous/prior 6 months -> comparison_time_grp_type=mth_grp, comparison_time_grp_value=P6M\\r\\n",
        "- current 6 completed months -> time_grp_type=mth_grp, time_grp_value=C6CM\\r\\n",
        "\n",
        "**Canonical Filter Policy:**\\r\\n",
        "- Default to case-insensitive exact equality/IN matching for brand/market/team filters.\\r\\n",
        "- Use LIKE only when the user explicitly asks for contains/starts with/partial matching.\\r\\n",
    ]
    joined = "\n".join(filtered)
    if "current 6 months -> time_grp_type=mth_grp, time_grp_value=C6M" not in joined:
        filtered.extend(canonical_lines)

    instructions.text_instructions[0].content = filtered
    return candidate


def policy_centaur_filter_snippet_audit(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Remove filter snippets that reference suspicious columns missing from the schema."""
    candidate = deepcopy(config)
    snippets = candidate.instructions.sql_snippets if candidate.instructions else None
    filters = list(snippets.filters or []) if snippets else []
    if not filters:
        return candidate

    inconsistent_ids = set(inconsistent_filter_snippet_ids(candidate))
    if not inconsistent_ids:
        return candidate

    snippets.filters = [snippet for snippet in filters if snippet.id not in inconsistent_ids]
    return candidate


def policy_centaur_instruction_audit(config: GenieSpaceConfig, ctx: PolicyContext) -> GenieSpaceConfig:
    """Normalize contradictory instruction guidance and legacy time-bucket references."""
    findings = instruction_contradiction_findings(config)
    candidate = deepcopy(config)
    if not findings:
        return candidate

    instructions = _ensure_instruction_container(candidate)
    normalized_lines = [
        line.replace("tb_wk_id", "wk_grp").replace("tb_mth_id", "mth_grp").replace("mth_grp_desc", "mth_grp")
        for line in (instructions.text_instructions[0].content or [])
    ]
    instructions.text_instructions[0].content = normalized_lines
    return policy_instruction_coherence(candidate, ctx)


def policy_hide_noise_columns(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Hide internal or operational columns that add prompt noise."""
    candidate = deepcopy(config)
    for recommendation in noise_column_candidates(candidate):
        column = _find_column(
            candidate,
            table_identifier=recommendation.table_identifier,
            column_name=recommendation.column_name,
        )
        if column is None:
            continue
        column.exclude = True
        column.enable_entity_matching = False
        column.enable_format_assistance = False
    return candidate


def policy_entity_matching_candidates(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Enable entity matching on recommended categorical columns."""
    candidate = deepcopy(config)
    for recommendation in entity_matching_candidates(candidate):
        column = _find_column(
            candidate,
            table_identifier=recommendation.table_identifier,
            column_name=recommendation.column_name,
        )
        if column is None:
            continue
        column.enable_format_assistance = True
        column.enable_entity_matching = True
    return candidate


def policy_centaur_expanded_entity_matching(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Enable entity matching for a broader diagnostic-driven candidate set."""
    candidate = deepcopy(config)
    for recommendation in expanded_entity_matching_candidates(candidate):
        column = _find_column(
            candidate,
            table_identifier=recommendation.table_identifier,
            column_name=recommendation.column_name,
        )
        if column is None:
            continue
        column.enable_format_assistance = True
        column.enable_entity_matching = True
    return candidate


def policy_format_assistance_candidates(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Enable format assistance on recommended time-like columns."""
    candidate = deepcopy(config)
    for recommendation in format_assistance_candidates(candidate):
        column = _find_column(
            candidate,
            table_identifier=recommendation.table_identifier,
            column_name=recommendation.column_name,
        )
        if column is None:
            continue
        column.enable_format_assistance = True
    return candidate


def _split_compound_join_sql(sql_items: list[str]) -> list[str]:
    """Split any compound AND-joined join condition into individual conditions.

    The Databricks export proto rejects a single string containing ' AND ' as a
    join condition (of class java.lang.String).  Each join predicate must be a
    separate list element.
    """
    result: list[str] = []
    for item in sql_items:
        parts = re.split(r"\s+AND\s+", item, flags=re.IGNORECASE)
        result.extend(part.strip() for part in parts if part.strip())
    return result


def policy_join_spec_generation(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Generate conservative join specs from shared join keys."""
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)
    existing = list(instructions.join_specs or [])
    existing_pairs = {
        frozenset({join_spec.left.identifier.lower(), join_spec.right.identifier.lower()})
        for join_spec in existing
    }
    for join_spec in join_spec_candidates(candidate):
        pair = frozenset({join_spec.left.identifier.lower(), join_spec.right.identifier.lower()})
        if pair in existing_pairs:
            continue
        # Proto guard: split any compound ON clause into individual list elements.
        safe_sql = _split_compound_join_sql(join_spec.sql or [])
        safe_spec = join_spec.model_copy(update={"sql": safe_sql})
        existing.append(safe_spec)
        existing_pairs.add(pair)
    instructions.join_specs = existing
    return candidate


def policy_sql_expression_measures(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Add reusable measure expressions for metric-like columns."""
    candidate = deepcopy(config)
    sql_snippets = _ensure_sql_snippets(candidate)
    sql_snippets.measures = _append_unique_snippets(
        sql_snippets.measures,
        measure_sql_snippet_candidates(candidate),
    )
    return candidate


def _strip_filter_alias(snippets: list[GenieSQLSnippet]) -> list[GenieSQLSnippet]:
    """Remove the 'alias' field from filter snippets.

    The Databricks export proto's Filter message has no 'alias' field; including
    it causes: 'Cannot find field: alias in message databricks.datarooms.export.Filter'.
    """
    result: list[GenieSQLSnippet] = []
    for snippet in snippets:
        if snippet.alias is not None:
            snippet = snippet.model_copy(update={"alias": None})
        result.append(snippet)
    return result


def policy_sql_expression_filters(config: GenieSpaceConfig, _: PolicyContext) -> GenieSpaceConfig:
    """Add reusable boolean filters for common flag-like columns."""
    candidate = deepcopy(config)
    sql_snippets = _ensure_sql_snippets(candidate)
    # Proto guard: Filter message does not accept 'alias'; strip it before merging.
    safe_candidates = _strip_filter_alias(filter_sql_snippet_candidates(candidate))
    sql_snippets.filters = _append_unique_snippets(
        sql_snippets.filters,
        safe_candidates,
    )
    return candidate


def policy_merge_benchmark_examples(config: GenieSpaceConfig, ctx: PolicyContext) -> GenieSpaceConfig:
    """Append exact benchmark Q->SQL examples to trusted examples."""
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)

    existing = list(instructions.example_question_sqls or [])
    existing_ids = {item.id for item in existing}
    for example in _benchmark_examples(candidate, ctx.train_benchmark_ids):
        if example.id not in existing_ids:
            existing.append(example)
            existing_ids.add(example.id)

    instructions.example_question_sqls = existing
    return candidate


def policy_benchmark_only_examples(config: GenieSpaceConfig, ctx: PolicyContext) -> GenieSpaceConfig:
    """Use only exact benchmark Q->SQL examples."""
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)
    instructions.example_question_sqls = _benchmark_examples(candidate, ctx.train_benchmark_ids)
    return candidate


# Common Genie parameter names that should be literalized
_GENIE_PARAMS = frozenset({
    "time_grp_type",
    "time_grp_value",
    "comparison_time_grp_type",
    "comparison_time_grp_value",
    "metric_name",
    "metric_type",
    "hierarchy_level",
})

# Default literal values when cannot be inferred from context
_DEFAULT_LITERALS = {
    "time_grp_type": "wk_grp",
    "time_grp_value": "C5W",
    "comparison_time_grp_type": "wk_grp",
    "comparison_time_grp_value": "P5W",
    "metric_name": "TRX",
    "metric_type": "TRX",
    "hierarchy_level": "NATN",
}


def _extract_literals_from_sql(sql: str) -> dict[str, str]:
    """Extract literal values from benchmark SQL that can replace parameters.

    Looks for patterns like `wk_grp = 'C5W'`, `hierarchy_level = 'NATN'`, etc.
    """
    literals: dict[str, str] = {}

    # Pattern: column_name = 'VALUE' or column_name IN ('VALUE')
    patterns = [
        (r"wk_grp\s*=\s*'([^']+)'", "time_grp_value"),
        (r"mth_grp\s*=\s*'([^']+)'", "time_grp_value"),
        (r"qtr_grp\s*=\s*'([^']+)'", "time_grp_value"),
        (r"hierarchy_level\s*=\s*'([^']+)'", "hierarchy_level"),
        (r"hierarchy_level\s+IN\s*\(\s*'([^']+)'", "hierarchy_level"),
        (r"'(TRX|NRX|NBRX|TRX EU|NRX EU)'", "metric_type"),
    ]

    for pattern, param_name in patterns:
        match = re.search(pattern, sql, re.IGNORECASE)
        if match:
            literals[param_name] = match.group(1)

    # Infer time_grp_type from time_grp_value
    if "time_grp_value" in literals:
        val = literals["time_grp_value"].upper()
        if val.endswith("W"):
            literals["time_grp_type"] = "wk_grp"
        elif val.endswith("M") or val.endswith("CM"):
            literals["time_grp_type"] = "mth_grp"
        elif val.endswith("Q"):
            literals["time_grp_type"] = "qtr_grp"

    return literals


def _literalize_sql(sql: str, literals: dict[str, str]) -> str:
    """Replace :param placeholders with literal values in SQL."""
    result = sql
    for param_name, value in literals.items():
        # Replace :param_name with 'value' (quoted string)
        result = re.sub(
            rf"(?<![:\w]):({param_name})(?![A-Za-z0-9_])",
            f"'{value}'",
            result,
            flags=re.IGNORECASE,
        )
    return result


def policy_literal_examples(config: GenieSpaceConfig, ctx: PolicyContext) -> GenieSpaceConfig:
    """Convert parameterized examples to literal form and add no-unbound-params instruction.

    This addresses a repeated unbound-parameter failure mode: when space examples
    contain :param placeholders, Genie may generate SQL with parameters that
    cannot be executed. Converting to literals guides Genie toward executable SQL.
    """
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)

    # Build literal mappings from benchmark expected SQL
    benchmark_literals: dict[str, str] = {}
    for benchmark in candidate.benchmarks.questions if candidate.benchmarks and candidate.benchmarks.questions else []:
        sql = _benchmark_sql(benchmark)
        if sql:
            benchmark_literals.update(_extract_literals_from_sql(sql))

    # Merge with defaults for any missing values
    final_literals = {**_DEFAULT_LITERALS, **benchmark_literals}

    # Literalize existing example SQLs
    existing = list(instructions.example_question_sqls or [])
    literalized: list[GenieExampleSQL] = []

    for example in existing:
        sql_text = "\n".join(example.sql or [])
        if any(f":{param}" in sql_text for param in _GENIE_PARAMS):
            new_sql = _literalize_sql(sql_text, final_literals)
            literalized.append(
                GenieExampleSQL(
                    id=example.id,
                    question=example.question,
                    sql=[new_sql],
                )
            )
        else:
            literalized.append(example)

    instructions.example_question_sqls = literalized

    # Add instruction to never emit unbound parameters
    content = instructions.text_instructions[0].content
    no_unbound_marker = "NEVER emit unbound :param placeholders"
    if not any(no_unbound_marker in line for line in content):
        content.extend([
            "\n",
            "**Critical: No Unbound Parameters**\\r\\n",
            f"{no_unbound_marker} in final SQL output.\\r\\n",
            "All parameter references (:time_grp_type, :time_grp_value, :metric_name, etc.) "
            "must be replaced with literal values before returning SQL.\\r\\n",
            "Default literals: time_grp_type=wk_grp, time_grp_value=C5W, hierarchy_level=NATN, metric_type=TRX.\\r\\n",
        ])

    return candidate


def policy_blend_non_comparison_examples(config: GenieSpaceConfig, ctx: PolicyContext) -> GenieSpaceConfig:
    """Keep generic non-comparison examples + benchmark exact examples."""
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)

    original_examples = (
        ctx.original_config.instructions.example_question_sqls
        if ctx.original_config.instructions and ctx.original_config.instructions.example_question_sqls
        else []
    )

    generic = []
    for item in original_examples:
        text = " | ".join(item.question or [])
        if _is_comparison_text(text):
            continue
        generic.append(deepcopy(item))

    merged = []
    seen_ids: set[str] = set()
    for item in generic + _benchmark_examples(candidate, ctx.train_benchmark_ids):
        if item.id in seen_ids:
            continue
        merged.append(item)
        seen_ids.add(item.id)

    instructions.example_question_sqls = merged
    return candidate


def policy_comparison_enforcement(config: GenieSpaceConfig, ctx: PolicyContext) -> GenieSpaceConfig:
    """Add explicit comparison guidance and synonym variants for comparison benchmarks."""
    candidate = deepcopy(config)
    instructions = _ensure_instruction_container(candidate)

    content = instructions.text_instructions[0].content
    marker = (
        "For percentage difference / % difference / current-vs-previous comparisons, "
        "always use the trusted comparison SQL template with both current and comparison time buckets."
    )
    if not any(marker in line for line in content):
        content.extend(
            [
                "\n",
                "**Comparison Query Enforcement:**\\r\\n",
                marker + "\\r\\n",
                "Do not replace trusted comparison templates with simplified current_period/comparison_period shortcuts.\\r\\n",
                "For current 6 months vs previous 6 months, use C6M vs P6M.\\r\\n",
                "For current 13 weeks vs previous 13 weeks, use C13W vs P13W.\\r\\n",
            ]
        )

    existing = list(instructions.example_question_sqls or [])
    question_texts = {" ".join(item.question or []).strip().lower() for item in existing}

    for benchmark in candidate.benchmarks.questions if candidate.benchmarks and candidate.benchmarks.questions else []:
        if ctx.train_benchmark_ids is not None and benchmark.id not in ctx.train_benchmark_ids:
            continue
        question = benchmark.question[0].strip() if benchmark.question else ""
        if not question or not _is_comparison_text(question):
            continue

        sql = _benchmark_sql(benchmark)
        if not sql:
            continue

        variants = {
            question,
            question.replace("% difference", "percentage difference"),
            question.replace("percentage difference", "% difference"),
            question.replace(" vs ", " versus "),
        }

        for variant in variants:
            cleaned = " ".join(variant.split()).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in question_texts:
                continue
            existing.append(
                GenieExampleSQL(
                    id=uuid.uuid4().hex,
                    question=[cleaned],
                    sql=[sql],
                )
            )
            question_texts.add(lowered)

    instructions.example_question_sqls = existing
    return candidate


def centaur_prepass_policies() -> list[CandidatePolicy]:
    """Diagnostic-driven structural passes unique to centaur."""
    return [
        CandidatePolicy(
            name="centaur_filter_snippet_audit",
            description="Remove filter snippets that reference suspicious unknown columns.",
            apply=policy_centaur_filter_snippet_audit,
        ),
        CandidatePolicy(
            name="centaur_instruction_audit",
            description="Normalize contradictory instruction guidance and legacy time-bucket references.",
            apply=policy_centaur_instruction_audit,
        ),
        CandidatePolicy(
            name="centaur_expanded_entity_matching",
            description="Expand entity matching to broader business-label columns before the main sweep.",
            apply=policy_centaur_expanded_entity_matching,
        ),
    ]


def default_policies() -> list[CandidatePolicy]:
    """Evidence-based default policies for production use.

    Only includes policies with demonstrated benefit on held-out test sets.
    Based on result-based head-to-head evidence from the canonical race protocol.

    See .omc/policy_classification.md for full evidence and rationale.
    """
    return [
        CandidatePolicy(
            name="literal_examples",
            description="Convert parameterized examples to literal form and forbid unbound params.",
            apply=policy_literal_examples,
        ),
    ]


def deprecated_policies() -> list[CandidatePolicy]:
    """Policies removed from production defaults due to no benefit or harm.

    These policies either:
    - Caused train/val overfitting that broke test generalization
    - Had no measurable impact (always skipped)
    - Caused train regression when applied

    Retained for research/experimentation. Use --policies flag to enable.
    See .omc/policy_classification.md for evidence on each policy.
    """
    return [
        CandidatePolicy(
            name="hide_noise_columns",
            description="Hide internal IDs, hashes, and ETL-style columns that add prompt noise.",
            apply=policy_hide_noise_columns,
        ),
        CandidatePolicy(
            name="entity_matching_candidates",
            description="Enable entity matching on recommended categorical columns.",
            apply=policy_entity_matching_candidates,
        ),
        CandidatePolicy(
            name="format_assistance_candidates",
            description="Enable format assistance on recommended time-like columns.",
            apply=policy_format_assistance_candidates,
        ),
        CandidatePolicy(
            name="join_spec_generation",
            description="Generate conservative join specs from shared join-key columns.",
            apply=policy_join_spec_generation,
        ),
        CandidatePolicy(
            name="sql_expression_measures",
            description="Add reusable measure expressions for metric-like columns.",
            apply=policy_sql_expression_measures,
        ),
        CandidatePolicy(
            name="sql_expression_filters",
            description="Add reusable boolean filters for common flag-like columns.",
            apply=policy_sql_expression_filters,
        ),
        CandidatePolicy(
            name="merge_benchmark_examples",
            description="Append exact benchmark Q->SQL trusted examples.",
            apply=policy_merge_benchmark_examples,
        ),
        CandidatePolicy(
            name="instruction_coherence",
            description="Remove conflicting filter guidance and enforce deterministic time mappings.",
            apply=policy_instruction_coherence,
        ),
        CandidatePolicy(
            name="benchmark_only_examples",
            description="Use only benchmark-aligned examples to reduce trusted-query competition.",
            apply=policy_benchmark_only_examples,
        ),
        CandidatePolicy(
            name="blend_non_comparison_examples",
            description="Reintroduce only non-comparison generic examples with benchmark examples.",
            apply=policy_blend_non_comparison_examples,
        ),
        CandidatePolicy(
            name="comparison_enforcement",
            description="Strengthen percentage/compare intent routing and add comparison synonyms.",
            apply=policy_comparison_enforcement,
        ),
    ]
