from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from maxgenie.schemas import GenieSpaceConfig

_TEXT_INSTRUCTION_META_PATH = "text_instruction.meta.yml"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False, allow_unicode=True)



def _join_lines(value: Any) -> Any:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return value



def _slugify(value: str, *, max_length: int = 50, fallback: str = "item") -> str:
    slug = _SLUG_PATTERN.sub("_", value.lower()).strip("_")
    if not slug:
        return fallback
    if len(slug) <= max_length:
        return slug
    return slug[:max_length].rstrip("_") or fallback



def _safe_identifier_stem(identifier: str, *, fallback: str) -> str:
    short_name = identifier.rsplit(".", 1)[-1]
    return _slugify(short_name, max_length=80, fallback=fallback)



def _disambiguated_stems(items: list[tuple[str, str]], *, max_length: int | None = None) -> dict[str, str]:
    counts = Counter(stem for _, stem in items)
    stems: dict[str, str] = {}
    for unique_key, stem in items:
        if counts[stem] == 1:
            stems[unique_key] = stem
            continue

        suffix = hashlib.sha1(unique_key.encode("utf-8")).hexdigest()[:8]
        base = stem
        if max_length is not None:
            reserved = len(suffix) + 2
            base = base[: max_length - reserved].rstrip("_")
        stems[unique_key] = f"{base}__{suffix}"
    return stems



def _clear_glob(directory: Path, pattern: str) -> None:
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        path.unlink()



def _clear_managed_paths(space_dir: Path) -> None:
    _clear_glob(space_dir / "tables", "*.yml")
    _clear_glob(space_dir / "metric_views", "*.yml")
    _clear_glob(space_dir / "instructions" / "examples", "*.yml")
    _clear_glob(space_dir / "instructions" / "join_specs", "*.yml")
    _clear_glob(space_dir / "instructions" / "sql_snippets" / "filters", "*.yml")
    _clear_glob(space_dir / "instructions" / "sql_snippets" / "expressions", "*.yml")
    _clear_glob(space_dir / "instructions" / "sql_snippets" / "measures", "*.yml")

    for path in (
        space_dir / "meta.yml",
        space_dir / "sample_questions.yml",
        space_dir / "benchmarks.yml",
        space_dir / "instructions" / "text_instruction.md",
        space_dir / "instructions" / _TEXT_INSTRUCTION_META_PATH,
        space_dir / "instructions" / "sql_functions.yml",
    ):
        if path.exists():
            path.unlink()



def _table_like_payload(table_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(table_data)
    payload["description"] = _join_lines(payload.get("description"))
    column_configs = payload.get("column_configs")
    if column_configs is not None:
        payload["column_configs"] = [
            {
                **column,
                "description": _join_lines(column.get("description")),
            }
            for column in column_configs
        ]
    return payload



def _example_payload(example_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(example_data)
    payload["question"] = _join_lines(payload.get("question"))
    payload["sql"] = _join_lines(payload.get("sql"))
    payload["usage_guidance"] = _join_lines(payload.get("usage_guidance"))
    return payload



def _join_spec_payload(join_spec_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(join_spec_data)
    payload["sql"] = _join_lines(payload.get("sql"))
    payload["comment"] = _join_lines(payload.get("comment"))
    payload["instruction"] = _join_lines(payload.get("instruction"))
    return payload



def _snippet_payload(snippet_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(snippet_data)
    payload["sql"] = _join_lines(payload.get("sql"))
    payload["comment"] = _join_lines(payload.get("comment"))
    payload["instruction"] = _join_lines(payload.get("instruction"))
    return payload



def _sample_question_payload(question_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(question_data)
    payload["question"] = _join_lines(payload.get("question"))
    return payload



def _benchmark_payload(benchmark_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(benchmark_data)
    payload["question"] = _join_lines(payload.get("question"))

    answers = payload.get("answer")
    if answers is not None:
        payload["answer"] = [
            {
                **answer,
                "content": _join_lines(answer.get("content")),
            }
            for answer in answers
        ]
    return payload



def decompose_config(
    config: GenieSpaceConfig,
    space_dir: Path,
    *,
    include_benchmarks: bool = True,
) -> None:
    """Write a GenieSpaceConfig as decomposed asset files."""
    serialized = config.to_serialized_space()
    space_dir.mkdir(parents=True, exist_ok=True)
    _clear_managed_paths(space_dir)

    tables_dir = space_dir / "tables"
    metric_views_dir = space_dir / "metric_views"
    instructions_dir = space_dir / "instructions"
    examples_dir = instructions_dir / "examples"
    join_specs_dir = instructions_dir / "join_specs"
    snippets_dir = instructions_dir / "sql_snippets"
    filters_dir = snippets_dir / "filters"
    expressions_dir = snippets_dir / "expressions"
    measures_dir = snippets_dir / "measures"

    for directory in (
        tables_dir,
        metric_views_dir,
        instructions_dir,
        examples_dir,
        join_specs_dir,
        filters_dir,
        expressions_dir,
        measures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    meta = {
        "title": config.title,
        "description": config.description,
        "warehouse_id": config.warehouse_id,
        "version": config.version,
        "space_id": config.space_id,
    }
    if config.parent_path is not None:
        meta["parent_path"] = config.parent_path
    _write_yaml(space_dir / "meta.yml", meta)

    data_sources = serialized.get("data_sources") or {}
    table_items = data_sources.get("tables", [])
    table_stems = _disambiguated_stems(
        [
            (table["identifier"], _safe_identifier_stem(table["identifier"], fallback="table"))
            for table in table_items
        ]
    )
    for table in table_items:
        _write_yaml(
            tables_dir / f"{table_stems[table['identifier']]}.yml",
            _table_like_payload(table),
        )

    metric_view_items = data_sources.get("metric_views", [])
    metric_view_stems = _disambiguated_stems(
        [
            (metric_view["identifier"], _safe_identifier_stem(metric_view["identifier"], fallback="metric_view"))
            for metric_view in metric_view_items
        ]
    )
    for metric_view in metric_view_items:
        _write_yaml(
            metric_views_dir / f"{metric_view_stems[metric_view['identifier']]}.yml",
            _table_like_payload(metric_view),
        )

    instructions = serialized.get("instructions") or {}
    text_instructions = instructions.get("text_instructions", [])
    if text_instructions:
        text_instruction = text_instructions[0]
        (instructions_dir / "text_instruction.md").write_text(
            _join_lines(text_instruction.get("content")) or "",
            encoding="utf-8",
        )
        _write_yaml(
            instructions_dir / _TEXT_INSTRUCTION_META_PATH,
            {"id": text_instruction.get("id")},
        )

    example_items = instructions.get("example_question_sqls", [])
    example_stems = _disambiguated_stems(
        [(example["id"], example["id"][:8]) for example in example_items],
        max_length=40,
    )
    for example in example_items:
        _write_yaml(
            examples_dir / f"example_{example_stems[example['id']]}.yml",
            _example_payload(example),
        )

    join_spec_items = instructions.get("join_specs", [])
    join_spec_stems = _disambiguated_stems(
        [(join_spec["id"], join_spec["id"][:8]) for join_spec in join_spec_items],
        max_length=40,
    )
    for join_spec in join_spec_items:
        _write_yaml(
            join_specs_dir / f"join_{join_spec_stems[join_spec['id']]}.yml",
            _join_spec_payload(join_spec),
        )

    sql_snippets = instructions.get("sql_snippets") or {}
    for snippet_type, snippet_dir in (
        ("filters", filters_dir),
        ("expressions", expressions_dir),
        ("measures", measures_dir),
    ):
        snippet_items = sql_snippets.get(snippet_type, [])
        snippet_stems = _disambiguated_stems(
            [
                (
                    snippet["id"],
                    _slugify(snippet.get("display_name", "snippet"), fallback="snippet"),
                )
                for snippet in snippet_items
            ],
            max_length=50,
        )
        for snippet in snippet_items:
            _write_yaml(
                snippet_dir / f"{snippet_stems[snippet['id']]}.yml",
                _snippet_payload(snippet),
            )

    if "sql_functions" in instructions:
        _write_yaml(
            instructions_dir / "sql_functions.yml",
            instructions.get("sql_functions", []),
        )

    if "config" in serialized:
        config_payload = serialized.get("config") or {}
        sample_questions = config_payload.get("sample_questions", [])
        _write_yaml(
            space_dir / "sample_questions.yml",
            [_sample_question_payload(question) for question in sample_questions],
        )

    if include_benchmarks and "benchmarks" in serialized:
        benchmarks = serialized.get("benchmarks") or {}
        questions = benchmarks.get("questions", [])
        _write_yaml(
            space_dir / "benchmarks.yml",
            {"questions": [_benchmark_payload(question) for question in questions]},
        )
