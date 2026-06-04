"""Diagnostics and linting for Genie space assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from maxgenie.schemas import (
    GenieColumnConfig,
    GenieExampleSQL,
    GenieInstruction,
    GenieJoinSpecs,
    GenieMetricView,
    GenieSQLSnippet,
    GenieSampleQuestion,
    GenieSpaceConfig,
    GenieTableJoinSpec,
    GenieTableConfig,
)

_THREE_PART_IDENTIFIER_RE = re.compile(
    r"\b([A-Za-z_][\w$]*\.[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*)\b"
)
_ENTITY_MATCHING_HINTS = frozenset(
    {
        "brand",
        "channel",
        "city",
        "country",
        "district",
        "formulary",
        "geographical",
        "geography",
        "hierarchy",
        "manager",
        "market",
        "nation",
        "payer",
        "plan",
        "position",
        "region",
        "segment",
        "speciality",
        "specialty",
        "state",
        "status",
        "sub",
        "team",
        "territory",
        "tier",
    }
)
_EXPANDED_ENTITY_MATCHING_HINTS = frozenset(
    {
        *_ENTITY_MATCHING_HINTS,
        "account",
        "business",
        "class",
        "customer",
        "division",
        "franchise",
        "group",
        "name",
        "network",
        "organization",
        "product",
        "provider",
        "segment",
        "source",
        "type",
        "unit",
    }
)
_METRIC_HINTS = frozenset(
    {
        "amount",
        "count",
        "metric",
        "nbrx",
        "nrx",
        "pct",
        "percentage",
        "prescriptions",
        "sales",
        "share",
        "total",
        "trx",
        "units",
        "value",
        "volume",
    }
)
_FORMAT_ASSISTANCE_HINTS = frozenset(
    {
        "calendar",
        "cal",
        "calander",
        "cycle",
        "data",
        "date",
        "dt",
        "grp",
        "month",
        "mth",
        "period",
        "qtr",
        "quarter",
        "time",
        "timestamp",
        "week",
        "wk",
        "year",
        "yr",
    }
)
_NOISE_HINTS = frozenset(
    {
        "audit",
        "batch",
        "checksum",
        "created",
        "deleted",
        "etl",
        "hash",
        "ingest",
        "ingested",
        "lineage",
        "load",
        "loaded",
        "modified",
        "partition",
        "row",
        "surrogate",
        "technical",
        "updated",
    }
)
_GENERIC_JOIN_COLUMN_HINTS = frozenset(
    {
        "date",
        "dt",
        "key",
        "month",
        "period",
        "qtr",
        "quarter",
        "time",
        "week",
        "wk",
        "year",
        "yr",
    }
)
_SQL_IDENTIFIER_RE = re.compile(r"(?:\b[A-Za-z_][\w$]*\.)?([A-Za-z_][\w$]*)\b")
_SQL_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "between",
        "by",
        "case",
        "cast",
        "coalesce",
        "current_date",
        "date",
        "else",
        "end",
        "false",
        "from",
        "in",
        "interval",
        "is",
        "like",
        "lower",
        "not",
        "null",
        "on",
        "or",
        "select",
        "string",
        "then",
        "true",
        "when",
        "where",
    }
)


@dataclass(frozen=True)
class ExternalTableReference:
    """Reference to a table not declared in the Genie space."""

    reference: str
    location: str


@dataclass(frozen=True)
class ColumnRecommendation:
    """Recommended metadata change for a column."""

    table_identifier: str
    column_name: str
    reason: str


@dataclass(frozen=True)
class TableCoverage:
    """Description coverage for one table or metric view."""

    identifier: str
    described_columns: int
    total_columns: int


@dataclass(frozen=True)
class MetadataCoverage:
    """Description coverage for the space."""

    total_tables: int
    described_tables: int
    total_columns: int
    described_columns: int
    low_coverage_tables: list[TableCoverage]


@dataclass(frozen=True)
class SpaceDiagnostics:
    """High-level diagnostics for a Genie space."""

    metadata_coverage: MetadataCoverage
    external_table_references: list[ExternalTableReference]
    entity_matching_candidates: list[ColumnRecommendation]
    format_assistance_candidates: list[ColumnRecommendation]

    def to_dict(self) -> dict[str, object]:
        """Serialize diagnostics for reports and persisted artifacts."""
        return asdict(self)


def _join_lines(value: list[str] | None) -> str:
    if not value:
        return ""
    return "\n".join(part for part in value if part)


def _has_meaningful_text(value: list[str] | None) -> bool:
    return bool(_join_lines(value).strip())


def _column_tokens(column_name: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", column_name.lower())
        if token
    }


def _looks_like_identifier(column_name: str) -> bool:
    lowered = column_name.lower()
    return lowered == "id" or lowered.endswith("_id") or lowered.endswith("id")


def _looks_like_noise_column(column_name: str) -> bool:
    tokens = _column_tokens(column_name)
    lowered = column_name.lower()
    if lowered.endswith(("_id", "_key")) and not (
        tokens & (_FORMAT_ASSISTANCE_HINTS | _ENTITY_MATCHING_HINTS)
    ):
        return True
    if {"hash", "checksum"} & tokens:
        return True
    if {"etl", "ingest", "ingested", "loaded", "load", "batch", "partition"} & tokens:
        return True
    if {"created", "updated", "modified", "deleted"} & tokens and (
        {"at", "ts", "timestamp", "date", "dt", "time"} & tokens
    ):
        return True
    if {"row"} & tokens and {"num", "number", "id"} & tokens:
        return True
    if _looks_like_identifier(column_name) and not (
        tokens & (_FORMAT_ASSISTANCE_HINTS | _ENTITY_MATCHING_HINTS | _METRIC_HINTS)
    ):
        return True
    return False


def _table_like_iterable(config: GenieSpaceConfig) -> list[GenieTableConfig | GenieMetricView]:
    items: list[GenieTableConfig | GenieMetricView] = []
    if config.data_sources:
        items.extend(config.data_sources.tables or [])
        items.extend(config.data_sources.metric_views or [])
    return items


def _table_short_name(identifier: str) -> str:
    return identifier.rsplit(".", 1)[-1]


def _table_alias(identifier: str) -> str:
    parts = [token for token in re.split(r"[^a-z0-9]+", _table_short_name(identifier).lower()) if token]
    if not parts:
        return "t"
    alias = "".join(part[0] for part in parts[:3]) or parts[0][:3]
    return alias[:8]


def _allowed_table_identifiers(config: GenieSpaceConfig) -> set[str]:
    return {
        item.identifier.lower()
        for item in _table_like_iterable(config)
        if item.identifier
    }


def _iter_text_locations(config: GenieSpaceConfig) -> list[tuple[str, str]]:
    locations: list[tuple[str, str]] = []
    if config.description:
        locations.append(("space.description", config.description))

    if config.config and config.config.sample_questions:
        for index, question in enumerate(config.config.sample_questions):
            locations.append(
                (f"config.sample_questions[{index}]", _join_lines(question.question))
            )

    for table_index, table in enumerate(_table_like_iterable(config)):
        table_prefix = f"data_sources[{table_index}]"
        locations.append((f"{table_prefix}.description", _join_lines(table.description)))
        for column_index, column in enumerate(table.column_configs or []):
            locations.append(
                (
                    f"{table_prefix}.column_configs[{column_index}].description",
                    _join_lines(column.description),
                )
            )

    instructions = config.instructions
    if instructions is None:
        return locations

    for index, instruction in enumerate(instructions.text_instructions or []):
        locations.append((f"instructions.text_instructions[{index}]", _join_lines(instruction.content)))

    for index, example in enumerate(instructions.example_question_sqls or []):
        locations.extend(_example_text_locations(index, example))

    for index, join_spec in enumerate(instructions.join_specs or []):
        locations.extend(_join_spec_text_locations(index, join_spec))

    snippets = instructions.sql_snippets
    if snippets:
        for group_name, group in (
            ("filters", snippets.filters or []),
            ("expressions", snippets.expressions or []),
            ("measures", snippets.measures or []),
        ):
            for index, snippet in enumerate(group):
                locations.extend(_snippet_text_locations(group_name, index, snippet))

    return [(location, text) for location, text in locations if text.strip()]


def _example_text_locations(index: int, example: GenieExampleSQL) -> list[tuple[str, str]]:
    return [
        (f"instructions.example_question_sqls[{index}].question", _join_lines(example.question)),
        (f"instructions.example_question_sqls[{index}].sql", _join_lines(example.sql)),
        (
            f"instructions.example_question_sqls[{index}].usage_guidance",
            _join_lines(example.usage_guidance),
        ),
    ]


def _join_spec_text_locations(index: int, join_spec: GenieJoinSpecs) -> list[tuple[str, str]]:
    return [
        (f"instructions.join_specs[{index}].sql", _join_lines(join_spec.sql)),
        (f"instructions.join_specs[{index}].comment", _join_lines(join_spec.comment)),
        (
            f"instructions.join_specs[{index}].instruction",
            _join_lines(join_spec.instruction),
        ),
    ]


def _snippet_text_locations(
    group_name: str,
    index: int,
    snippet: GenieSQLSnippet,
) -> list[tuple[str, str]]:
    return [
        (f"instructions.sql_snippets.{group_name}[{index}].sql", _join_lines(snippet.sql)),
        (
            f"instructions.sql_snippets.{group_name}[{index}].comment",
            _join_lines(snippet.comment),
        ),
        (
            f"instructions.sql_snippets.{group_name}[{index}].instruction",
            _join_lines(snippet.instruction),
        ),
    ]


def _external_table_references(config: GenieSpaceConfig) -> list[ExternalTableReference]:
    allowed = _allowed_table_identifiers(config)
    findings: list[ExternalTableReference] = []
    seen: set[tuple[str, str]] = set()

    for location, text in _iter_text_locations(config):
        for reference in _THREE_PART_IDENTIFIER_RE.findall(text):
            normalized_reference = reference.lower()
            if normalized_reference in allowed:
                continue
            key = (normalized_reference, location)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                ExternalTableReference(
                    reference=reference,
                    location=location,
                )
            )

    return sorted(findings, key=lambda item: (item.reference.lower(), item.location))


def _recommend_entity_matching(column: GenieColumnConfig) -> bool:
    if column.enable_entity_matching or column.exclude:
        return False
    tokens = _column_tokens(column.column_name)
    if _looks_like_identifier(column.column_name):
        return False
    if tokens & _FORMAT_ASSISTANCE_HINTS:
        return False
    if tokens & _METRIC_HINTS and not {"name", "desc", "status"} & tokens:
        return False
    if tokens & _ENTITY_MATCHING_HINTS:
        return True
    return bool(column.synonyms)


def _recommend_format_assistance(column: GenieColumnConfig) -> bool:
    if column.enable_format_assistance or column.exclude:
        return False
    tokens = _column_tokens(column.column_name)
    return bool(tokens & _FORMAT_ASSISTANCE_HINTS)


def _matching_recommendations(config: GenieSpaceConfig) -> tuple[list[ColumnRecommendation], list[ColumnRecommendation]]:
    entity_matching: list[ColumnRecommendation] = []
    format_assistance: list[ColumnRecommendation] = []

    for table in _table_like_iterable(config):
        for column in table.column_configs or []:
            if _recommend_entity_matching(column):
                entity_matching.append(
                    ColumnRecommendation(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        reason="Categorical business value looks likely to benefit from prompt matching/entity matching.",
                    )
                )
            if _recommend_format_assistance(column):
                format_assistance.append(
                    ColumnRecommendation(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        reason="Time or period-like column looks likely to benefit from format assistance.",
                    )
                )

    entity_matching.sort(key=lambda item: (item.table_identifier, item.column_name))
    format_assistance.sort(key=lambda item: (item.table_identifier, item.column_name))
    return entity_matching, format_assistance


def entity_matching_candidates(config: GenieSpaceConfig) -> list[ColumnRecommendation]:
    """Return columns that should likely enable entity matching."""
    entity_matching, _ = _matching_recommendations(config)
    return entity_matching


def format_assistance_candidates(config: GenieSpaceConfig) -> list[ColumnRecommendation]:
    """Return columns that should likely enable format assistance."""
    _, format_assistance = _matching_recommendations(config)
    return format_assistance


def expanded_entity_matching_candidates(config: GenieSpaceConfig) -> list[ColumnRecommendation]:
    """Return a broader entity-matching candidate set for centaur pre-pass policies."""
    base_candidates = entity_matching_candidates(config)
    seen = {(item.table_identifier, item.column_name) for item in base_candidates}
    recommendations = list(base_candidates)

    for table in _table_like_iterable(config):
        for column in table.column_configs or []:
            if column.enable_entity_matching or column.exclude:
                continue
            if _looks_like_identifier(column.column_name) or _looks_like_noise_column(column.column_name):
                continue
            tokens = _column_tokens(column.column_name)
            if tokens & _FORMAT_ASSISTANCE_HINTS:
                continue
            if tokens & _METRIC_HINTS and not ({"name", "desc", "description", "status"} & tokens):
                continue
            lowered = column.column_name.lower()
            if not (
                tokens & _EXPANDED_ENTITY_MATCHING_HINTS
                or lowered.endswith(("_name", "_desc", "_description", "_type", "_group"))
                or bool(column.synonyms)
            ):
                continue
            key = (table.identifier, column.column_name)
            if key in seen:
                continue
            seen.add(key)
            recommendations.append(
                ColumnRecommendation(
                    table_identifier=table.identifier,
                    column_name=column.column_name,
                    reason="Broader categorical business field looks likely to benefit from centaur entity-matching expansion.",
                )
            )

    recommendations.sort(key=lambda item: (item.table_identifier, item.column_name))
    return recommendations


def instruction_contradiction_findings(config: GenieSpaceConfig) -> list[str]:
    """Detect contradictory or legacy instruction guidance worth normalizing."""
    instructions = config.instructions.text_instructions if config.instructions else None
    if not instructions:
        return []

    content = "\n".join(
        line
        for instruction in instructions
        for line in (instruction.content or [])
        if line
    )
    lowered = content.lower()
    findings: list[str] = []

    has_like_guidance = "like operator" in lowered or " use like" in lowered or "use like " in lowered
    has_exact_guidance = (
        "exact matching" in lowered
        or "exact equality" in lowered
        or "case-insensitive exact" in lowered
    )
    if has_like_guidance and has_exact_guidance:
        findings.append("like_vs_exact_matching")

    if any(token in lowered for token in ("tb_wk_id", "tb_mth_id", "mth_grp_desc")):
        findings.append("legacy_time_bucket_columns")

    return findings


def _known_column_names(config: GenieSpaceConfig) -> set[str]:
    return {
        column.column_name.lower()
        for table in _table_like_iterable(config)
        for column in (table.column_configs or [])
        if column.column_name
    }


def _snippet_sql_identifiers(snippet: GenieSQLSnippet) -> set[str]:
    identifiers = {
        token.lower()
        for token in _SQL_IDENTIFIER_RE.findall("\n".join(snippet.sql or []))
        if token and token.lower() not in _SQL_KEYWORDS
    }
    return identifiers


def inconsistent_filter_snippet_ids(config: GenieSpaceConfig) -> list[str]:
    """Return existing filter snippet IDs that do not appear to reference current schema columns."""
    snippets = config.instructions.sql_snippets if config.instructions else None
    if snippets is None or not snippets.filters:
        return []

    known_columns = _known_column_names(config)
    inconsistent: list[str] = []
    for snippet in snippets.filters or []:
        identifiers = _snippet_sql_identifiers(snippet)
        if not identifiers:
            continue
        referenced_columns = identifiers & known_columns
        suspicious_identifiers = {
            token
            for token in identifiers
            if token.startswith("is_")
            or token.endswith(("_flag", "_indicator", "_status", "_name", "_type", "_group"))
            or token.startswith("tb_")
        }
        if suspicious_identifiers and not referenced_columns:
            inconsistent.append(snippet.id)
    return inconsistent


def noise_column_candidates(config: GenieSpaceConfig) -> list[ColumnRecommendation]:
    """Return internal or noisy columns that should likely be hidden."""
    recommendations: list[ColumnRecommendation] = []
    for table in _table_like_iterable(config):
        for column in table.column_configs or []:
            if column.exclude or not _looks_like_noise_column(column.column_name):
                continue
            recommendations.append(
                ColumnRecommendation(
                    table_identifier=table.identifier,
                    column_name=column.column_name,
                    reason="Column looks internal or operational rather than user-facing business data.",
                )
            )
    recommendations.sort(key=lambda item: (item.table_identifier, item.column_name))
    return recommendations


def _existing_join_pairs(config: GenieSpaceConfig) -> set[frozenset[str]]:
    join_specs = config.instructions.join_specs if config.instructions else None
    pairs: set[frozenset[str]] = set()
    for join_spec in join_specs or []:
        pairs.add(
            frozenset(
                {
                    join_spec.left.identifier.lower(),
                    join_spec.right.identifier.lower(),
                }
            )
        )
    return pairs


def _looks_like_join_column(column_name: str) -> bool:
    lowered = column_name.lower()
    tokens = _column_tokens(lowered)
    if lowered.endswith(("_id", "_key")) or lowered.startswith("tb_"):
        return True
    if _looks_like_noise_column(column_name):
        return False
    return bool(tokens & _GENERIC_JOIN_COLUMN_HINTS)


def _join_column_priority(column_name: str) -> tuple[int, int, str]:
    lowered = column_name.lower()
    tokens = _column_tokens(lowered)
    return (
        0 if lowered.startswith("tb_") or lowered.endswith(("_id", "_key")) else 1,
        0 if tokens & _FORMAT_ASSISTANCE_HINTS else 1,
        lowered,
    )


def join_spec_candidates(config: GenieSpaceConfig) -> list[GenieJoinSpecs]:
    """Generate conservative join specs from shared join-like column names."""
    items = _table_like_iterable(config)
    existing_pairs = _existing_join_pairs(config)
    candidates: list[GenieJoinSpecs] = []

    for left_index, left_table in enumerate(items):
        left_columns = {
            column.column_name.lower(): column.column_name
            for column in left_table.column_configs or []
            if not column.exclude
        }
        for right_table in items[left_index + 1 :]:
            pair_key = frozenset({left_table.identifier.lower(), right_table.identifier.lower()})
            if pair_key in existing_pairs:
                continue

            right_columns = {
                column.column_name.lower(): column.column_name
                for column in right_table.column_configs or []
                if not column.exclude
            }
            shared = [
                left_columns[name]
                for name in set(left_columns) & set(right_columns)
                if _looks_like_join_column(left_columns[name])
            ]
            if not shared:
                continue

            join_columns = sorted(shared, key=_join_column_priority)[:3]
            left_alias = _table_alias(left_table.identifier)
            right_alias = _table_alias(right_table.identifier)
            if left_alias == right_alias:
                left_alias = f"{left_alias}1"
                right_alias = f"{right_alias}2"

            join_sql = " AND ".join(
                f"{left_alias}.{column_name} = {right_alias}.{column_name}"
                for column_name in join_columns
            )
            candidates.append(
                GenieJoinSpecs(
                    left=GenieTableJoinSpec(identifier=left_table.identifier, alias=left_alias),
                    right=GenieTableJoinSpec(identifier=right_table.identifier, alias=right_alias),
                    sql=[join_sql],
                    comment=[
                        "Generated from shared join-like columns: "
                        + ", ".join(f"`{column_name}`" for column_name in join_columns)
                    ],
                    instruction=[
                        "Use this join when combining "
                        f"`{_table_short_name(left_table.identifier)}` and "
                        f"`{_table_short_name(right_table.identifier)}`."
                    ],
                )
            )

    candidates.sort(
        key=lambda item: (
            item.left.identifier.lower(),
            item.right.identifier.lower(),
            "\n".join(item.sql or []),
        )
    )
    return candidates


def _display_name(column_name: str) -> str:
    acronym_map = {
        "nrx": "NRx",
        "nbrx": "NBRx",
        "trx": "TRx",
        "pct": "Percentage",
        "id": "ID",
        "etl": "ETL",
    }
    parts = [token for token in re.split(r"[^a-z0-9]+", column_name.lower()) if token]
    rendered = [acronym_map.get(token, token.capitalize()) for token in parts]
    return " ".join(rendered) or column_name


def _existing_snippet_signatures(
    config: GenieSpaceConfig,
    *,
    snippet_type: str,
) -> set[tuple[str, str]]:
    snippets = config.instructions.sql_snippets if config.instructions else None
    if snippets is None:
        return set()
    groups = {
        "filters": snippets.filters or [],
        "expressions": snippets.expressions or [],
        "measures": snippets.measures or [],
    }
    return {
        (
            (snippet.display_name or "").strip().lower(),
            "\n".join(snippet.sql or []).strip().lower(),
        )
        for snippet in groups[snippet_type]
    }


def _metric_synonyms(column_name: str, *, aggregate_name: str) -> list[str]:
    display_name = _display_name(column_name).lower()
    synonyms = [display_name]
    if aggregate_name == "avg":
        synonyms.extend([f"average {display_name}", f"avg {display_name}"])
    else:
        synonyms.extend([f"total {display_name}", f"sum of {display_name}"])
    return list(dict.fromkeys(synonyms))


def measure_sql_snippet_candidates(config: GenieSpaceConfig) -> list[GenieSQLSnippet]:
    """Generate reusable measure expressions for metric-like columns."""
    existing = _existing_snippet_signatures(config, snippet_type="measures")
    candidates: list[GenieSQLSnippet] = []
    seen = set(existing)

    for table in _table_like_iterable(config):
        for column in table.column_configs or []:
            if column.exclude:
                continue
            tokens = _column_tokens(column.column_name)
            if not tokens & _METRIC_HINTS or _looks_like_identifier(column.column_name):
                continue
            aggregate_name = (
                "avg"
                if tokens & {"pct", "percentage", "share", "rate", "avg", "average"}
                else "sum"
            )
            expression = f"{aggregate_name.upper()}({column.column_name})"
            display_name = _display_name(column.column_name)
            if aggregate_name == "avg" and not display_name.lower().startswith(("average ", "avg ")):
                display_name = f"Average {display_name}"
            if aggregate_name == "sum" and not display_name.lower().startswith(("total ", "sum ")):
                display_name = f"Total {display_name}"
            signature = (display_name.strip().lower(), expression.lower())
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(
                GenieSQLSnippet(
                    alias=re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_") or None,
                    sql=[expression],
                    display_name=display_name,
                    synonyms=_metric_synonyms(column.column_name, aggregate_name=aggregate_name),
                    comment=[f"Generated from `{table.identifier}.{column.column_name}`."],
                    instruction=[f"Use for {display_name.lower()} requests."],
                )
            )

    candidates.sort(key=lambda item: item.display_name.lower())
    return candidates[:12]


def filter_sql_snippet_candidates(config: GenieSpaceConfig) -> list[GenieSQLSnippet]:
    """Generate conservative boolean filters for flag-like columns."""
    existing = _existing_snippet_signatures(config, snippet_type="filters")
    candidates: list[GenieSQLSnippet] = []
    seen = set(existing)

    for table in _table_like_iterable(config):
        for column in table.column_configs or []:
            if column.exclude:
                continue
            tokens = _column_tokens(column.column_name)
            lowered = column.column_name.lower()
            if not (
                tokens & {"flag", "indicator"}
                or lowered.startswith("is_")
                or {"active", "enabled", "target"} & tokens
            ):
                continue

            display_name = _display_name(column.column_name)
            expression = (
                f"LOWER(CAST(COALESCE({column.column_name}, '') AS STRING)) IN "
                "('1', 'y', 'yes', 'true', 'active', 'enabled')"
            )
            signature = (display_name.strip().lower(), expression.lower())
            if signature in seen:
                continue
            seen.add(signature)
            synonyms = [display_name.lower()]
            if "active" in tokens:
                synonyms.extend(["active", "currently active"])
            if "target" in tokens:
                synonyms.extend(["target", "targeted"])
            if {"flag", "indicator"} & tokens:
                synonyms.append(f"{display_name.lower()} true")
            candidates.append(
                GenieSQLSnippet(
                    alias=re.sub(r"[^a-z0-9]+", "_", display_name.lower()).strip("_") or None,
                    sql=[expression],
                    display_name=display_name,
                    synonyms=list(dict.fromkeys(synonyms)),
                    comment=[f"Generated from `{table.identifier}.{column.column_name}`."],
                    instruction=[f"Apply when the user asks for {display_name.lower()} rows."],
                )
            )

    candidates.sort(key=lambda item: item.display_name.lower())
    return candidates[:8]


def _metadata_coverage(config: GenieSpaceConfig) -> MetadataCoverage:
    table_items = _table_like_iterable(config)
    described_tables = sum(1 for table in table_items if _has_meaningful_text(table.description))
    total_columns = 0
    described_columns = 0
    low_coverage_tables: list[TableCoverage] = []

    for table in table_items:
        columns = table.column_configs or []
        column_total = len(columns)
        column_described = sum(1 for column in columns if _has_meaningful_text(column.description))
        total_columns += column_total
        described_columns += column_described
        if column_total > 0 and column_described < column_total:
            low_coverage_tables.append(
                TableCoverage(
                    identifier=table.identifier,
                    described_columns=column_described,
                    total_columns=column_total,
                )
            )

    low_coverage_tables.sort(key=lambda item: (item.described_columns / item.total_columns, item.identifier))
    return MetadataCoverage(
        total_tables=len(table_items),
        described_tables=described_tables,
        total_columns=total_columns,
        described_columns=described_columns,
        low_coverage_tables=low_coverage_tables[:5],
    )


def analyze_space_config(config: GenieSpaceConfig) -> SpaceDiagnostics:
    """Analyze a Genie space for metadata and structural issues."""
    entity_recommendations = entity_matching_candidates(config)
    format_recommendations = format_assistance_candidates(config)
    return SpaceDiagnostics(
        metadata_coverage=_metadata_coverage(config),
        external_table_references=_external_table_references(config),
        entity_matching_candidates=entity_recommendations,
        format_assistance_candidates=format_recommendations,
    )
