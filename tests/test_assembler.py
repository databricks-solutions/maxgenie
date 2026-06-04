"""Tests for decomposed config round-trips."""

from pathlib import Path
import re

import yaml

from maxgenie.assembler import assemble_config
from maxgenie.decomposer import decompose_config
from maxgenie.schemas import (
    GenieExampleSQL,
    GenieInstruction,
    GenieInstructions,
    GenieJoinSpecs,
    GenieSpaceConfig,
    GenieTableJoinSpec,
)


def test_decompose_roundtrip_preserves_colliding_example_and_join_ids(tmp_path: Path):
    config = GenieSpaceConfig(
        title="Collision Test",
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(id="f" * 32, content=["Keep all assets"])],
            example_question_sqls=[
                GenieExampleSQL(
                    id="12345678aaaaaaaaaaaaaaaaaaaaaaaa",
                    question=["Q1"],
                    sql=["SELECT 1"],
                ),
                GenieExampleSQL(
                    id="12345678bbbbbbbbbbbbbbbbbbbbbbbb",
                    question=["Q2"],
                    sql=["SELECT 2"],
                ),
            ],
            join_specs=[
                GenieJoinSpecs(
                    id="abcdef12aaaaaaaaaaaaaaaaaaaaaaaa",
                    left=GenieTableJoinSpec(identifier="left_a", alias="la"),
                    right=GenieTableJoinSpec(identifier="right_a", alias="ra"),
                    sql=["la.id = ra.id"],
                ),
                GenieJoinSpecs(
                    id="abcdef12bbbbbbbbbbbbbbbbbbbbbbbb",
                    left=GenieTableJoinSpec(identifier="left_b", alias="lb"),
                    right=GenieTableJoinSpec(identifier="right_b", alias="rb"),
                    sql=["lb.id = rb.id"],
                ),
            ],
        ),
    )

    decompose_config(config, tmp_path / "space", include_benchmarks=False)
    restored = assemble_config(tmp_path / "space")

    assert restored.instructions is not None
    assert restored.instructions.example_question_sqls is not None
    assert restored.instructions.join_specs is not None
    assert len(restored.instructions.example_question_sqls) == 2
    assert len(restored.instructions.join_specs) == 2
    assert {
        example.id for example in restored.instructions.example_question_sqls
    } == {
        "12345678aaaaaaaaaaaaaaaaaaaaaaaa",
        "12345678bbbbbbbbbbbbbbbbbbbbbbbb",
    }
    assert {
        join.id for join in restored.instructions.join_specs
    } == {
        "abcdef12aaaaaaaaaaaaaaaaaaaaaaaa",
        "abcdef12bbbbbbbbbbbbbbbbbbbbbbbb",
    }


def _write_space_with_snippet(space_dir: Path, snippet: dict) -> None:
    (space_dir).mkdir(parents=True, exist_ok=True)
    (space_dir / "meta.yml").write_text(
        yaml.safe_dump({"title": "Alias Repair", "version": 1}), encoding="utf-8"
    )
    measures_dir = space_dir / "instructions" / "sql_snippets" / "measures"
    measures_dir.mkdir(parents=True, exist_ok=True)
    (measures_dir / "snippet.yml").write_text(
        yaml.safe_dump(snippet), encoding="utf-8"
    )


def _write_space_with_example(space_dir: Path, example: dict) -> None:
    (space_dir).mkdir(parents=True, exist_ok=True)
    (space_dir / "meta.yml").write_text(
        yaml.safe_dump({"title": "Alias Repair", "version": 1}), encoding="utf-8"
    )
    examples_dir = space_dir / "instructions" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    (examples_dir / "example.yml").write_text(
        yaml.safe_dump(example), encoding="utf-8"
    )


def test_assemble_repairs_aliased_required_snippet_fields(tmp_path: Path):
    # The proposer emitted an aliased required field (`sql_query` instead of `sql`,
    # `display` instead of `display_name`). Without repair, model construction would
    # raise a missing-required ValidationError; with repair the snippet assembles.
    space_dir = tmp_path / "space"
    _write_space_with_snippet(
        space_dir,
        {"sql_query": "SUM(net_sales)", "display": "Total Net Sales"},
    )

    restored = assemble_config(space_dir)

    assert restored.instructions is not None
    assert restored.instructions.sql_snippets is not None
    measures = restored.instructions.sql_snippets.measures
    assert measures is not None and len(measures) == 1
    assert measures[0].sql == ["SUM(net_sales)"]
    assert measures[0].display_name == "Total Net Sales"


def test_assemble_ignores_unsupported_snippet_field(tmp_path: Path):
    # An unsupported/unexpected field must not crash assembly; it is dropped.
    space_dir = tmp_path / "space"
    _write_space_with_snippet(
        space_dir,
        {
            "sql": "SUM(net_sales)",
            "display_name": "Total Net Sales",
            "unsupported_field": {"nested": "value"},
        },
    )

    restored = assemble_config(space_dir)

    measures = restored.instructions.sql_snippets.measures
    assert measures is not None and len(measures) == 1
    assert measures[0].sql == ["SUM(net_sales)"]
    assert not hasattr(measures[0], "unsupported_field")


def test_assemble_repairs_invalid_sql_snippet_id(tmp_path: Path):
    space_dir = tmp_path / "space"
    _write_space_with_snippet(
        space_dir,
        {
            "id": "plain_average_fare_avg_fare",
            "sql": "AVG(fare_amount)",
            "display_name": "Average fare",
        },
    )

    restored = assemble_config(space_dir)

    measures = restored.instructions.sql_snippets.measures
    assert measures is not None and len(measures) == 1
    assert re.fullmatch(r"[0-9a-f]{32}", measures[0].id)
    assert measures[0].id != "plain_average_fare_avg_fare"


def test_assemble_repairs_aliased_required_example_fields(tmp_path: Path):
    space_dir = tmp_path / "space"
    _write_space_with_example(
        space_dir,
        {"nl_question": "What were total sales last week?", "sql_query": "SELECT 1"},
    )

    restored = assemble_config(space_dir)

    examples = restored.instructions.example_question_sqls
    assert examples is not None and len(examples) == 1
    assert examples[0].question == ["What were total sales last week?"]
    assert examples[0].sql == ["SELECT 1"]


def test_assemble_does_not_rewrite_present_canonical_field(tmp_path: Path):
    # When the canonical field is present, an unrelated alias key is ignored (dropped),
    # never overwriting genuine content.
    space_dir = tmp_path / "space"
    _write_space_with_snippet(
        space_dir,
        {
            "sql": "SUM(net_sales)",
            "display_name": "Canonical Name",
            "display": "Alias Should Not Win",
        },
    )

    restored = assemble_config(space_dir)

    measures = restored.instructions.sql_snippets.measures
    assert measures is not None and len(measures) == 1
    assert measures[0].display_name == "Canonical Name"


def test_assemble_treats_stale_optional_sql_functions_file_as_absent(
    tmp_path: Path,
    monkeypatch,
):
    space_dir = tmp_path / "space"
    instructions_dir = space_dir / "instructions"
    instructions_dir.mkdir(parents=True)
    (space_dir / "meta.yml").write_text(
        yaml.safe_dump({"title": "Race Test", "version": 1}), encoding="utf-8"
    )
    sql_functions_path = instructions_dir / "sql_functions.yml"
    sql_functions_path.write_text("[]\n", encoding="utf-8")
    original_open = Path.open

    def flaky_open(self, *args, **kwargs):
        if self == sql_functions_path:
            raise FileNotFoundError(str(self))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)

    restored = assemble_config(space_dir)

    assert restored.title == "Race Test"
    assert restored.instructions is None


def test_assemble_treats_stale_optional_text_instruction_as_absent(
    tmp_path: Path,
    monkeypatch,
):
    space_dir = tmp_path / "space"
    instructions_dir = space_dir / "instructions"
    instructions_dir.mkdir(parents=True)
    (space_dir / "meta.yml").write_text(
        yaml.safe_dump({"title": "Race Test", "version": 1}), encoding="utf-8"
    )
    text_path = instructions_dir / "text_instruction.md"
    text_path.write_text("Use clear SQL.", encoding="utf-8")
    original_read_text = Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self == text_path:
            raise FileNotFoundError(str(self))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    restored = assemble_config(space_dir)

    assert restored.title == "Race Test"
    assert restored.instructions is None
