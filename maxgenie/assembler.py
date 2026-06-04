from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from maxgenie.ids import coerce_artifact_id
from maxgenie.schemas import (
    GenieBenchmarkAnswer,
    GenieBenchmarkQuestion,
    GenieBenchmarks,
    GenieColumnConfig,
    GenieConfig,
    GenieDataSources,
    GenieExampleSQL,
    GenieInstruction,
    GenieInstructions,
    GenieJoinSpecs,
    GenieMetricView,
    GenieSQLFunction,
    GenieSQLSnippet,
    GenieSQLSnippets,
    GenieSampleQuestion,
    GenieSpaceConfig,
    GenieTableConfig,
)

_TEXT_INSTRUCTION_META_PATH = "text_instruction.meta.yml"
_MISSING = object()


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_yaml_if_present(path: Path) -> Any:
    if not path.exists():
        return _MISSING
    try:
        return _load_yaml(path)
    except FileNotFoundError:
        return _MISSING


def _read_text_if_present(path: Path) -> object | str:
    if not path.exists():
        return _MISSING
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _MISSING



def _as_dict(data: Any, path: Path) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data



def _as_list(data: Any, path: Path) -> list[Any]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data).__name__}")
    return data



def _normalize_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Expected string or list[str], got {type(value).__name__}")



def _normalize_lines(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [] if value == "" else value.split("\n")
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"Expected string or list[str], got {type(value).__name__}")



def _read_meta(space_dir: Path) -> dict[str, Any]:
    return _as_dict(_load_yaml(space_dir / "meta.yml"), space_dir / "meta.yml")


# Field-name aliases a candidate proposer may emit for required snippet/example/join
# fields. Resolving a recognized alias onto its canonical name is a parse-and-repair
# step so an aliased required field becomes a repair instead of a missing-required
# crash at model construction. Each alias is an unambiguous compound name that is not
# itself a valid Genie field, and resolution only fires when the canonical field is
# absent, so genuine source content is never rewritten.
_REQUIRED_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sql": ("sql_query", "sql_text", "sql_statement", "query_sql", "query_text"),
    "display_name": ("display", "displayname", "display_label"),
    "question": ("nl_question", "user_question", "nl_prompt"),
}


def _is_empty_field(value: Any) -> bool:
    return value is None or value == "" or value == []


def _resolve_required_field_aliases(
    data: dict[str, Any], canonical_fields: tuple[str, ...]
) -> dict[str, Any]:
    """Map recognized aliases of required fields onto their canonical names."""
    resolved = dict(data)
    for canonical in canonical_fields:
        if not _is_empty_field(resolved.get(canonical)):
            continue
        for alias in _REQUIRED_FIELD_ALIASES.get(canonical, ()):
            if alias in resolved and not _is_empty_field(resolved[alias]):
                resolved[canonical] = resolved[alias]
                break
    return resolved


def _stable_id_seed(data: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(data),
        sort_keys=True,
        allow_unicode=False,
        width=10_000,
    )


def _coerce_present_id(data: dict[str, Any]) -> dict[str, Any]:
    if "id" in data:
        data["id"] = coerce_artifact_id(data.get("id"), seed=_stable_id_seed(data))
    return data



def _normalize_column_config(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    normalized["description"] = _normalize_lines(normalized.get("description"))
    normalized["synonyms"] = _normalize_string_list(normalized.get("synonyms"))
    return normalized



def _normalize_table_like_config(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    normalized["description"] = _normalize_lines(normalized.get("description"))
    column_configs = normalized.get("column_configs")
    if column_configs is not None:
        normalized["column_configs"] = [
            _normalize_column_config(_as_dict(item, Path("column_configs")))
            for item in _as_list(column_configs, Path("column_configs"))
        ]
    return normalized



def _read_tables(tables_dir: Path) -> list[GenieTableConfig] | None:
    if not tables_dir.exists():
        return None

    tables: list[GenieTableConfig] = []
    for path in sorted(tables_dir.glob("*.yml")):
        data = _normalize_table_like_config(_as_dict(_load_yaml(path), path))
        tables.append(GenieTableConfig(**data))
    return tables



def _read_metric_views(metric_views_dir: Path) -> list[GenieMetricView] | None:
    if not metric_views_dir.exists():
        return None

    metric_views: list[GenieMetricView] = []
    for path in sorted(metric_views_dir.glob("*.yml")):
        data = _normalize_table_like_config(_as_dict(_load_yaml(path), path))
        metric_views.append(GenieMetricView(**data))
    return metric_views



def _read_text_instruction(instructions_dir: Path) -> GenieInstruction | None:
    text_path = instructions_dir / "text_instruction.md"
    raw_text = _read_text_if_present(text_path)
    if raw_text is _MISSING:
        return None

    instruction_data: dict[str, Any] = {
        "content": [] if raw_text == "" else raw_text.split("\n"),
    }

    meta_path = instructions_dir / _TEXT_INSTRUCTION_META_PATH
    meta_data = _load_yaml_if_present(meta_path)
    if meta_data is not _MISSING:
        meta = _as_dict(meta_data, meta_path)
        if meta.get("id"):
            instruction_data["id"] = meta["id"]

    _coerce_present_id(instruction_data)
    return GenieInstruction(**instruction_data)



def _normalize_example_sql(data: dict[str, Any]) -> dict[str, Any]:
    normalized = _resolve_required_field_aliases(dict(data), ("question", "sql"))
    _coerce_present_id(normalized)
    normalized["question"] = _normalize_lines(normalized.get("question"))
    normalized["sql"] = _normalize_lines(normalized.get("sql"))
    normalized["usage_guidance"] = _normalize_lines(normalized.get("usage_guidance"))
    return normalized



def _read_examples(examples_dir: Path) -> list[GenieExampleSQL] | None:
    if not examples_dir.exists():
        return None

    examples: list[GenieExampleSQL] = []
    for path in sorted(examples_dir.glob("*.yml")):
        data = _normalize_example_sql(_as_dict(_load_yaml(path), path))
        examples.append(GenieExampleSQL(**data))
    return examples



def _normalize_join_spec(data: dict[str, Any]) -> dict[str, Any]:
    normalized = _resolve_required_field_aliases(dict(data), ("sql",))
    _coerce_present_id(normalized)
    normalized["sql"] = _normalize_lines(normalized.get("sql"))
    normalized["comment"] = _normalize_lines(normalized.get("comment"))
    normalized["instruction"] = _normalize_lines(normalized.get("instruction"))
    return normalized



def _read_join_specs(join_specs_dir: Path) -> list[GenieJoinSpecs] | None:
    if not join_specs_dir.exists():
        return None

    join_specs: list[GenieJoinSpecs] = []
    for path in sorted(join_specs_dir.glob("*.yml")):
        data = _normalize_join_spec(_as_dict(_load_yaml(path), path))
        join_specs.append(GenieJoinSpecs(**data))
    return join_specs



def _normalize_sql_snippet(data: dict[str, Any]) -> dict[str, Any]:
    normalized = _resolve_required_field_aliases(dict(data), ("sql", "display_name"))
    _coerce_present_id(normalized)
    normalized["sql"] = _normalize_lines(normalized.get("sql"))
    normalized["synonyms"] = _normalize_string_list(normalized.get("synonyms"))
    normalized["comment"] = _normalize_lines(normalized.get("comment"))
    normalized["instruction"] = _normalize_lines(normalized.get("instruction"))
    return normalized



def _read_snippet_group(snippet_group_dir: Path) -> list[GenieSQLSnippet] | None:
    if not snippet_group_dir.exists():
        return None

    snippets: list[GenieSQLSnippet] = []
    for path in sorted(snippet_group_dir.glob("*.yml")):
        data = _normalize_sql_snippet(_as_dict(_load_yaml(path), path))
        snippets.append(GenieSQLSnippet(**data))
    return snippets



def _read_snippets(snippets_dir: Path) -> GenieSQLSnippets | None:
    if not snippets_dir.exists():
        return None

    filters_dir = snippets_dir / "filters"
    expressions_dir = snippets_dir / "expressions"
    measures_dir = snippets_dir / "measures"

    filters = _read_snippet_group(filters_dir)
    expressions = _read_snippet_group(expressions_dir)
    measures = _read_snippet_group(measures_dir)

    if not any(
        path.exists()
        for path in (snippets_dir, filters_dir, expressions_dir, measures_dir)
    ):
        return None

    return GenieSQLSnippets(
        filters=filters,
        expressions=expressions,
        measures=measures,
    )



def _read_sql_functions(space_dir: Path) -> list[GenieSQLFunction] | None:
    path = space_dir / "instructions" / "sql_functions.yml"
    data = _load_yaml_if_present(path)
    if data is _MISSING:
        return None

    return [
        GenieSQLFunction(**_coerce_present_id(_as_dict(item, path)))
        for item in _as_list(data, path)
    ]



def _normalize_sample_question(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    _coerce_present_id(normalized)
    normalized["question"] = _normalize_lines(normalized.get("question"))
    return normalized



def _read_sample_questions(space_dir: Path) -> list[GenieSampleQuestion] | None:
    path = space_dir / "sample_questions.yml"
    data = _load_yaml_if_present(path)
    if data is _MISSING:
        return None

    return [
        GenieSampleQuestion(**_normalize_sample_question(_as_dict(item, path)))
        for item in _as_list(data, path)
    ]



def _normalize_benchmark_answer(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    normalized["content"] = _normalize_lines(normalized.get("content"))
    return normalized



def _normalize_benchmark_question(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    _coerce_present_id(normalized)
    normalized["question"] = _normalize_lines(normalized.get("question"))
    answers = normalized.get("answer")
    if answers is not None:
        normalized["answer"] = [
            _normalize_benchmark_answer(_as_dict(item, Path("answer")))
            for item in _as_list(answers, Path("answer"))
        ]
    return normalized



def _read_benchmarks(space_dir: Path) -> GenieBenchmarks | None:
    path = space_dir / "benchmarks.yml"
    data = _load_yaml_if_present(path)
    if data is _MISSING:
        return None

    data = _as_dict(data, path)
    questions = data.get("questions")
    if questions is None:
        return GenieBenchmarks(questions=[])

    return GenieBenchmarks(
        questions=[
            GenieBenchmarkQuestion(**_normalize_benchmark_question(_as_dict(item, path)))
            for item in _as_list(questions, path)
        ]
    )



def assemble_config(space_dir: Path) -> GenieSpaceConfig:
    """Read decomposed asset files and rebuild a valid GenieSpaceConfig."""
    meta = _read_meta(space_dir)
    tables = _read_tables(space_dir / "tables")
    metric_views = _read_metric_views(space_dir / "metric_views")

    instructions_dir = space_dir / "instructions"
    text_instruction = _read_text_instruction(instructions_dir)
    examples = _read_examples(instructions_dir / "examples")
    join_specs = _read_join_specs(instructions_dir / "join_specs")
    sql_snippets = _read_snippets(instructions_dir / "sql_snippets")
    sql_functions = _read_sql_functions(space_dir)
    sample_questions = _read_sample_questions(space_dir)
    benchmarks = _read_benchmarks(space_dir)

    config = None
    if sample_questions is not None:
        config = GenieConfig(sample_questions=sample_questions)

    data_sources = None
    if tables is not None or metric_views is not None:
        data_sources = GenieDataSources(
            tables=tables,
            metric_views=metric_views,
        )

    text_instructions = [text_instruction] if text_instruction is not None else None
    instructions = None
    if any(
        value is not None
        for value in (text_instructions, examples, join_specs, sql_functions, sql_snippets)
    ):
        instructions = GenieInstructions(
            text_instructions=text_instructions,
            example_question_sqls=examples,
            join_specs=join_specs,
            sql_functions=sql_functions,
            sql_snippets=sql_snippets,
        )

    return GenieSpaceConfig(
        version=meta.get("version", 1),
        space_id=meta.get("space_id"),
        title=meta.get("title"),
        description=meta.get("description"),
        warehouse_id=meta.get("warehouse_id"),
        parent_path=meta.get("parent_path"),
        config=config,
        data_sources=data_sources,
        instructions=instructions,
        benchmarks=benchmarks,
    )
