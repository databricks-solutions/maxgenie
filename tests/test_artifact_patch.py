from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pytest
import yaml

from maxgenie.artifact_patch import (
    apply_artifact_patch_candidate,
    build_artifact_patch_sequence_from_space_dirs,
    load_artifact_patch_sequence,
    repair_artifact_patch_v2_payload,
    write_artifact_patch_sequence,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_apply_artifact_patch_candidate_writes_valid_edit_atomically(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("old instruction\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": "new instruction\n",
                    "expected_sha256": _sha256("old instruction\n"),
                }
            ]
        },
    )

    assert summary["changed"] is True
    assert summary["changed_count"] == 1
    assert summary["rejected_count"] == 0
    assert summary["changed_edits"][0]["path"] == "instructions/text_instruction.md"
    assert summary["changed_edits"][0]["action"] == "updated"
    assert instruction_path.read_text(encoding="utf-8") == "new instruction\n"
    assert not list(instruction_path.parent.glob("*.tmp"))


def test_apply_artifact_patch_candidate_supports_dry_run(tmp_path: Path):
    space_dir = tmp_path / "space"
    example_path = space_dir / "instructions" / "examples" / "example_new.yml"

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/examples/example_new.yml",
                    "content": "id: abc\nquestion: Q\nsql: SELECT 1\n",
                }
            ]
        },
        dry_run=True,
    )

    assert summary["changed"] is True
    assert summary["dry_run"] is True
    assert summary["changed_edits"][0]["action"] == "created"
    assert not example_path.exists()


def test_apply_artifact_patch_candidate_rejects_unsafe_paths_transactionally(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("keep me\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": "do not write\n",
                },
                {
                    "path": "../outside.yml",
                    "content": "bad\n",
                },
                {
                    "path": "benchmarks.yml",
                    "content": "questions: []\n",
                },
                {
                    "path": "/tmp/absolute.yml",
                    "content": "bad\n",
                },
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["changed_count"] == 0
    assert summary["rejected_count"] == 3
    assert {edit["reason"] for edit in summary["rejected_edits"]} == {
        "path_traversal_not_allowed",
        "path_not_allowlisted",
        "absolute_path_not_allowed",
    }
    assert summary["skipped_edits"][0]["reason"] == "payload_has_rejections"
    assert instruction_path.read_text(encoding="utf-8") == "keep me\n"


def test_apply_artifact_patch_candidate_rejects_binary_looking_content(tmp_path: Path):
    summary = apply_artifact_patch_candidate(
        tmp_path / "space",
        {
            "file_edits": [
                {
                    "path": "tables/sales.yml",
                    "content": "identifier: t\x00bad\n",
                }
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"] == [
        {
            "index": 0,
            "path": "tables/sales.yml",
            "reason": "binary_content_not_allowed",
        }
    ]


def test_apply_artifact_patch_candidate_rejects_expected_sha256_mismatch(tmp_path: Path):
    space_dir = tmp_path / "space"
    table_path = space_dir / "tables" / "sales.yml"
    table_path.parent.mkdir(parents=True)
    table_path.write_text("identifier: sales\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "tables/sales.yml",
                    "content": "identifier: sales\ndescription: new\n",
                    "expected_sha256": _sha256("different\n"),
                }
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"][0]["reason"] == "expected_sha256_mismatch"
    assert table_path.read_text(encoding="utf-8") == "identifier: sales\n"


def test_apply_artifact_patch_candidate_skips_unchanged_content(tmp_path: Path):
    space_dir = tmp_path / "space"
    snippet_path = space_dir / "instructions" / "sql_snippets" / "filters" / "brand.yml"
    snippet_path.parent.mkdir(parents=True)
    snippet_path.write_text("display_name: Brand\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "relative_path": "instructions/sql_snippets/filters/brand.yml",
                    "content": "display_name: Brand\n",
                }
            ]
        },
    )

    assert summary["changed"] is False
    assert summary["skipped_count"] == 1
    assert summary["skipped_edits"][0]["reason"] == "content_unchanged"


def test_apply_artifact_patch_v2_style_guards_require_reason_and_existing_hash(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("existing\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": "updated\n",
                }
            ]
        },
        require_existing_sha256=True,
        require_edit_reason=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"] == [
        {
            "index": 0,
            "path": "instructions/text_instruction.md",
            "reason": "edit_reason_required",
        }
    ]
    assert instruction_path.read_text(encoding="utf-8") == "existing\n"

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": "updated\n",
                    "reason": "Tighten visible period-comparison instruction.",
                }
            ]
        },
        require_existing_sha256=True,
        require_edit_reason=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"][0]["reason"] == "expected_sha256_required_for_existing_file"


def test_apply_artifact_patch_v2_style_records_compact_file_delta(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("old\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": "old\nnew\n",
                    "expected_sha256": _sha256("old\n"),
                    "reason": "Add one visible-safe instruction line.",
                    "description": "target_train_failure:visible-target",
                    "comment": "guard_preservation:visible-guard",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
    )

    assert summary["changed"] is True
    changed = summary["changed_edits"][0]
    assert changed["bytes_delta"] == len("old\nnew\n") - len("old\n")
    assert changed["lines_delta"] == 1
    assert changed["reason"] == "Add one visible-safe instruction line."
    assert changed["description"] == "target_train_failure:visible-target"
    assert changed["comment"] == "guard_preservation:visible-guard"


def test_apply_artifact_patch_v2_style_rejects_new_bind_placeholders_in_executable_sql(tmp_path: Path):
    # A genuinely new executable bind in a SQL artifact is still rejected.
    space_dir = tmp_path / "space"
    example_path = space_dir / "instructions" / "examples" / "example_period.yml"
    example_path.parent.mkdir(parents=True)
    example_path.write_text(
        "id: example_period\nsql: SELECT trx FROM t WHERE wk_grp = 'C5W'\n",
        encoding="utf-8",
    )

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/examples/example_period.yml",
                    "content": "id: example_period\nsql: SELECT trx FROM t WHERE wk_grp = :time_grp_type\n",
                    "expected_sha256": _sha256(
                        "id: example_period\nsql: SELECT trx FROM t WHERE wk_grp = 'C5W'\n"
                    ),
                    "reason": "Add period filter.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"][0]["reason"] == "new_bind_placeholders_not_allowed"
    assert summary["rejected_edits"][0]["placeholders"] == [":time_grp_type"]


def test_apply_artifact_patch_v2_style_allows_bind_placeholder_in_prose_instruction(tmp_path: Path):
    # A candidate must be able to TEACH literalization in prose by naming the
    # placeholder it forbids — the guard must not fire on natural-language files.
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text(
        "Use literal SQL values from visible examples.\n",
        encoding="utf-8",
    )

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": (
                        "Use literal SQL values from visible examples.\n"
                        "Never emit an unbound :time_grp_type; use a string literal like 'C5W'.\n"
                    ),
                    "expected_sha256": _sha256(
                        "Use literal SQL values from visible examples.\n"
                    ),
                    "reason": "Teach period-grain literalization.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert not summary["rejected_edits"]
    assert ":time_grp_type" in instruction_path.read_text(encoding="utf-8")


def test_apply_artifact_patch_v2_style_ignores_bind_placeholder_in_sql_comment(tmp_path: Path):
    # A placeholder mentioned only in a SQL comment is never executed → allowed.
    space_dir = tmp_path / "space"
    example_path = space_dir / "instructions" / "examples" / "example_note.yml"
    example_path.parent.mkdir(parents=True)
    example_path.write_text(
        "id: example_note\nsql: SELECT trx FROM t\n",
        encoding="utf-8",
    )

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/examples/example_note.yml",
                    "content": "id: example_note\nsql: SELECT trx FROM t -- never bind :time_grp_type\n",
                    "expected_sha256": _sha256("id: example_note\nsql: SELECT trx FROM t\n"),
                    "reason": "Annotate the example.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert not summary["rejected_edits"]


def test_apply_artifact_patch_v2_style_ignores_bind_placeholder_inside_string_literal(tmp_path: Path):
    # A :param inside a SQL string literal is literal text, not a runtime bind.
    space_dir = tmp_path / "space"
    example_path = space_dir / "instructions" / "examples" / "example_str.yml"
    example_path.parent.mkdir(parents=True)
    example_path.write_text("id: example_str\nsql: SELECT trx FROM t\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/examples/example_str.yml",
                    "content": "id: example_str\nsql: SELECT trx FROM t WHERE note = 'prefer a literal over :time_grp_type'\n",
                    "expected_sha256": _sha256("id: example_str\nsql: SELECT trx FROM t\n"),
                    "reason": "Add note literal.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert not summary["rejected_edits"]


def test_apply_artifact_patch_v2_style_rejects_new_dollar_brace_placeholder_in_table_sql(tmp_path: Path):
    # ${var} style in an executable tables/*.yml artifact is still rejected.
    space_dir = tmp_path / "space"
    table_path = space_dir / "tables" / "fact.yml"
    table_path.parent.mkdir(parents=True)
    table_path.write_text("name: fact\nsql: SELECT trx FROM t\n", encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "tables/fact.yml",
                    "content": "name: fact\nsql: SELECT trx FROM t WHERE region = ${region}\n",
                    "expected_sha256": _sha256("name: fact\nsql: SELECT trx FROM t\n"),
                    "reason": "Add region filter.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"][0]["reason"] == "new_bind_placeholders_not_allowed"
    assert "${region}" in summary["rejected_edits"][0]["placeholders"]


def test_apply_artifact_patch_v2_style_rejects_new_bind_after_apostrophe_in_comment(tmp_path: Path):
    # Regression: an apostrophe inside a -- comment must not cross-pair with a later
    # string literal and swallow the newline, which would hide a genuine new bind on
    # the following line. The new bind in executable SQL must still be rejected.
    space_dir = tmp_path / "space"
    example_path = space_dir / "instructions" / "examples" / "example_cmt.yml"
    example_path.parent.mkdir(parents=True)
    before = "id: example_cmt\nsql: SELECT trx FROM t\n"
    example_path.write_text(before, encoding="utf-8")

    after = (
        "id: example_cmt\n"
        "sql: SELECT trx FROM t -- it's a note\n"
        "  WHERE note = 'hello' AND region = :real_bind\n"
    )
    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/examples/example_cmt.yml",
                    "content": after,
                    "expected_sha256": _sha256(before),
                    "reason": "Add a filter.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"][0]["reason"] == "new_bind_placeholders_not_allowed"
    assert ":real_bind" in summary["rejected_edits"][0]["placeholders"]


def test_apply_artifact_patch_v2_style_ignores_double_dash_inside_string_literal(tmp_path: Path):
    # A '--' inside a string literal is not a comment, and a :param inside that string
    # is literal text — neither should trip the guard.
    space_dir = tmp_path / "space"
    example_path = space_dir / "instructions" / "examples" / "example_dd.yml"
    example_path.parent.mkdir(parents=True)
    before = "id: example_dd\nsql: SELECT trx FROM t\n"
    example_path.write_text(before, encoding="utf-8")

    after = "id: example_dd\nsql: SELECT trx FROM t WHERE label = 'a -- :notabind b'\n"
    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/examples/example_dd.yml",
                    "content": after,
                    "expected_sha256": _sha256(before),
                    "reason": "Add a label filter.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert not summary["rejected_edits"]


def test_apply_artifact_patch_v2_style_accepts_compact_append_content(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    original = "Use literal SQL values from visible examples.\n"
    instruction_path.write_text(original, encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": None,
                    "append_content": "Preserve regional pass guards.\n",
                    "expected_sha256": _sha256(original),
                    "reason": "Append one visible-safe guard instruction.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert summary["changed_edits"][0]["edit_mode"] == "append_content"
    assert instruction_path.read_text(encoding="utf-8") == (
        original + "Preserve regional pass guards.\n"
    )


def test_apply_artifact_patch_v2_style_accepts_compact_find_replace(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    original = "Use brand filters when supplied.\nKeep market filters exact.\n"
    instruction_path.write_text(original, encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": None,
                    "find": "Use brand filters when supplied.\n",
                    "replace": "Use case-insensitive brand filters when supplied.\n",
                    "expected_sha256": _sha256(original),
                    "reason": "Narrow brand filter guidance.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert summary["changed_edits"][0]["edit_mode"] == "find_replace"
    assert instruction_path.read_text(encoding="utf-8") == (
        "Use case-insensitive brand filters when supplied.\n"
        "Keep market filters exact.\n"
    )


def test_apply_artifact_patch_v2_style_rejects_multiple_edit_modes(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    original = "Use brand filters when supplied.\n"
    instruction_path.write_text(original, encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": "Use rewritten brand filters.\n",
                    "append_content": "Preserve region filters.\n",
                    "expected_sha256": _sha256(original),
                    "reason": "Narrow brand filter guidance.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"] == [
        {
            "index": 0,
            "path": "instructions/text_instruction.md",
            "reason": "multiple_edit_content_modes",
            "edit_modes": ["content", "append_content"],
        }
    ]
    assert instruction_path.read_text(encoding="utf-8") == original


def test_apply_artifact_patch_v2_style_rejects_conflicting_path_aliases(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    original = "Use brand filters when supplied.\n"
    instruction_path.write_text(original, encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "relative_path": "sample_questions.yml",
                    "content": None,
                    "append_content": "Preserve market filters.\n",
                    "expected_sha256": _sha256(original),
                    "reason": "Path aliases must target one file.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"] == [
        {
            "index": 0,
            "path": "instructions/text_instruction.md",
            "relative_path": "sample_questions.yml",
            "reason": "path_relative_path_mismatch",
        }
    ]
    assert instruction_path.read_text(encoding="utf-8") == original


def test_apply_artifact_patch_v2_style_rejects_ambiguous_find_replace(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    original = "Use exact filters.\nUse exact filters.\n"
    instruction_path.write_text(original, encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": None,
                    "find": "Use exact filters.\n",
                    "replace": "Use exact visible filters.\n",
                    "expected_sha256": _sha256(original),
                    "reason": "Ambiguous replacement should not be guessed.",
                }
            ]
        },
        max_file_edits=2,
        require_existing_sha256=True,
        require_edit_reason=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"][0]["reason"] == "find_text_not_unique"
    assert instruction_path.read_text(encoding="utf-8") == original


def test_apply_artifact_patch_v2_style_caps_file_edit_count(tmp_path: Path):
    summary = apply_artifact_patch_candidate(
        tmp_path / "space",
        {
            "file_edits": [
                {
                    "path": "instructions/examples/a.yml",
                    "content": "id: a\n",
                    "reason": "a",
                },
                {
                    "path": "instructions/examples/b.yml",
                    "content": "id: b\n",
                    "reason": "b",
                },
                {
                    "path": "instructions/examples/c.yml",
                    "content": "id: c\n",
                    "reason": "c",
                },
            ]
        },
        max_file_edits=2,
        require_edit_reason=True,
    )

    assert summary["changed"] is False
    assert summary["rejected_edits"] == [
        {
            "path": None,
            "reason": "too_many_file_edits",
            "max_file_edits": 2,
            "actual_file_edits": 3,
        }
    ]


def test_apply_artifact_patch_v2_style_accepts_composite_three_file_patch(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    table_path = space_dir / "tables" / "sales.yml"
    example_path = space_dir / "instructions" / "examples" / "literal_brand.yml"
    instruction_path.parent.mkdir(parents=True)
    table_path.parent.mkdir(parents=True)
    example_path.parent.mkdir(parents=True)
    original_instruction = "Use exact table names.\n"
    original_table = "identifier: sales\n"
    instruction_path.write_text(original_instruction, encoding="utf-8")
    table_path.write_text(original_table, encoding="utf-8")

    summary = apply_artifact_patch_candidate(
        space_dir,
        {
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "content": None,
                    "append_content": "Use literal period labels such as C13W when visible examples require them.\n",
                    "expected_sha256": _sha256(original_instruction),
                    "reason": "visible_family:literal_time_window_guidance",
                },
                {
                    "path": "tables/sales.yml",
                    "content": None,
                    "append_content": "description: Sales facts with literal brand and time-window dimensions.\n",
                    "expected_sha256": _sha256(original_table),
                    "reason": "visible_family:schema_context",
                },
                {
                    "path": "instructions/examples/literal_brand.yml",
                    "content": "id: literal_brand\nsql: SELECT 'FARXIGA' AS brand_name\n",
                    "reason": "visible_family:executable_literal_example",
                },
            ]
        },
        max_file_edits=4,
        require_existing_sha256=True,
        require_edit_reason=True,
        reject_new_bind_placeholders=True,
    )

    assert summary["changed"] is True
    assert summary["changed_count"] == 3
    assert summary["rejected_count"] == 0
    assert "C13W" in instruction_path.read_text(encoding="utf-8")
    assert "Sales facts" in table_path.read_text(encoding="utf-8")
    assert "SELECT 'FARXIGA'" in example_path.read_text(encoding="utf-8")


def test_build_artifact_patch_sequence_chunks_hash_checked_file_edits(tmp_path: Path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    for root in (before, after):
        (root / "instructions" / "examples").mkdir(parents=True)
        (root / "tables").mkdir()
    (before / "instructions" / "text_instruction.md").parent.mkdir(parents=True, exist_ok=True)
    (before / "instructions" / "text_instruction.md").write_text("old\n", encoding="utf-8")
    (before / "instructions" / "examples" / "a.yml").write_text("id: a\n", encoding="utf-8")
    (after / "instructions" / "text_instruction.md").write_text("new\n", encoding="utf-8")
    (after / "instructions" / "examples" / "a.yml").write_text("id: a\nsql: SELECT 1\n", encoding="utf-8")
    (after / "tables" / "sales.yml").write_text("identifier: sales\n", encoding="utf-8")

    sequence = build_artifact_patch_sequence_from_space_dirs(
        before,
        after,
        candidate_name_prefix="control",
        max_file_edits=2,
    )

    assert sequence["candidate_mode"] == "artifact_patch_v2"
    assert sequence["file_edit_count"] == 3
    assert sequence["payload_count"] == 2
    assert all(len(payload["file_edits"]) <= 2 for payload in sequence["payloads"])
    edits = [
        edit
        for payload in sequence["payloads"]
        for edit in payload["file_edits"]
    ]
    updated = {edit["path"]: edit for edit in edits}
    assert updated["instructions/text_instruction.md"]["expected_sha256"] == _sha256("old\n")
    assert "expected_sha256" not in updated["tables/sales.yml"]
    assert all(edit["reason"] for edit in edits)


def test_artifact_patch_sequence_round_trips_from_directory(tmp_path: Path):
    sequence = {
        "schema_version": "maxgenie.artifact_patch_sequence.v1",
        "payloads": [
            {
                "candidate_name": "one",
                "candidate_mode": "artifact_patch_v2",
                "file_edits": [],
            }
        ],
    }
    sequence_path = write_artifact_patch_sequence(tmp_path / "sequence", sequence)

    assert sequence_path.name == "patch_sequence.json"
    assert load_artifact_patch_sequence(sequence_path)[0]["candidate_name"] == "one"
    assert load_artifact_patch_sequence(tmp_path / "sequence")[0]["candidate_mode"] == "artifact_patch_v2"


def test_build_artifact_patch_sequence_rejects_deletes(tmp_path: Path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    (before / "instructions").mkdir(parents=True)
    after.mkdir()
    (before / "instructions" / "text_instruction.md").write_text("old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not support file deletes"):
        build_artifact_patch_sequence_from_space_dirs(before, after)


def test_build_artifact_patch_sequence_rejects_unsupported_changed_files(tmp_path: Path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    before.mkdir()
    after.mkdir()
    (before / "metadata.json").write_text('{"old": true}\n', encoding="utf-8")
    (after / "metadata.json").write_text('{"new": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported file changes"):
        build_artifact_patch_sequence_from_space_dirs(before, after)


def test_repair_artifact_patch_v2_payload_fills_safe_deterministic_fields(tmp_path: Path):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("old\n", encoding="utf-8")

    repaired, summary = repair_artifact_patch_v2_payload(
        space_dir,
        {
            "candidate_name": "patch",
            "operations": [{"op": "no_change"}],
            "file_edits": [
                {
                    "relative_path": "instructions/text_instruction.md",
                    "content": "new\n",
                }
            ],
        },
    )

    assert repaired["operations"] == []
    assert repaired["file_edits"][0]["path"] == "instructions/text_instruction.md"
    assert repaired["file_edits"][0]["append_content"] is None
    assert repaired["file_edits"][0]["find"] is None
    assert repaired["file_edits"][0]["replace"] is None
    assert repaired["file_edits"][0]["description"] is None
    assert repaired["file_edits"][0]["comment"] is None
    assert repaired["file_edits"][0]["expected_sha256"] == _sha256("old\n")
    assert repaired["file_edits"][0]["reason"] == "deterministic_artifact_patch_v2_repair"
    assert summary["changed"] is True
    assert summary["normalized_operations"] is True
    assert summary["filled_nullable_file_edit_fields"] == [
        {
            "path": "instructions/text_instruction.md",
            "fields": [
                "append_content",
                "find",
                "replace",
                "expected_sha256",
                "reason",
                "description",
                "comment",
            ],
        }
    ]
    assert summary["filled_expected_sha256_paths"] == ["instructions/text_instruction.md"]
    assert summary["filled_reason_paths"] == ["instructions/text_instruction.md"]


def test_repair_artifact_patch_v2_payload_normalizes_no_change_file_edits(tmp_path: Path):
    space_dir = tmp_path / "space"
    space_dir.mkdir()

    repaired, summary = repair_artifact_patch_v2_payload(
        space_dir,
        {
            "candidate_name": "no_change",
            "operations": [{"op": "no_change", "reason": "insufficient evidence"}],
            "file_edits": None,
        },
    )

    assert repaired["operations"] == []
    assert repaired["file_edits"] == []
    assert summary["normalized_operations"] is True
    assert summary["changed"] is True


def test_repair_artifact_patch_v2_payload_skips_conflicting_path_aliases(
    tmp_path: Path,
):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("old\n", encoding="utf-8")

    repaired, summary = repair_artifact_patch_v2_payload(
        space_dir,
        {
            "candidate_name": "patch",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "relative_path": "sample_questions.yml",
                    "append_content": "new\n",
                }
            ],
        },
    )

    assert "expected_sha256" not in repaired["file_edits"][0]
    assert "reason" not in repaired["file_edits"][0]
    assert summary["changed"] is False
    assert summary["skipped_edits"] == [
        {
            "index": 0,
            "path": "instructions/text_instruction.md",
            "relative_path": "sample_questions.yml",
            "reason": "path_relative_path_mismatch",
        }
    ]


def test_repair_artifact_patch_v2_payload_promotes_description_to_reason(
    tmp_path: Path,
):
    space_dir = tmp_path / "space"
    instruction_path = space_dir / "instructions" / "text_instruction.md"
    instruction_path.parent.mkdir(parents=True)
    instruction_path.write_text("old\n", encoding="utf-8")

    repaired, summary = repair_artifact_patch_v2_payload(
        space_dir,
        {
            "candidate_name": "patch",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/text_instruction.md",
                    "append_content": "new\n",
                    "description": "target_train_failure:visible-target",
                    "comment": "guard_preservation:visible-guard",
                }
            ],
        },
    )

    assert repaired["file_edits"][0]["reason"] == "target_train_failure:visible-target"
    assert repaired["file_edits"][0]["description"] == "target_train_failure:visible-target"
    assert repaired["file_edits"][0]["comment"] == "guard_preservation:visible-guard"
    assert summary["changed"] is True
    assert summary["filled_reason_paths"] == ["instructions/text_instruction.md"]


def test_repair_artifact_patch_v2_payload_derives_sql_snippet_display_name(
    tmp_path: Path,
):
    space_dir = tmp_path / "space"
    space_dir.mkdir()

    repaired, summary = repair_artifact_patch_v2_payload(
        space_dir,
        {
            "candidate_name": "snippet_patch",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/expressions/literal_time_bucket_selector.yml",
                    "content": "\n".join(
                        [
                            "description: Use literal SQL constants, not bind parameters, when selecting week/month time bucket ids for market-speciality queries.",
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
        },
    )

    content = repaired["file_edits"][0]["content"]
    assert summary["changed"] is True
    assert summary["filled_sql_snippet_display_name_paths"] == [
        "instructions/sql_snippets/expressions/literal_time_bucket_selector.yml"
    ]
    assert content.startswith(
        "display_name: Use literal SQL constants, not bind parameters, when selecting week/month time bucket ids for...\n"
    )


def test_repair_artifact_patch_v2_payload_repairs_invalid_artifact_id(
    tmp_path: Path,
):
    space_dir = tmp_path / "space"
    space_dir.mkdir()

    repaired, summary = repair_artifact_patch_v2_payload(
        space_dir,
        {
            "candidate_name": "snippet_patch",
            "operations": [],
            "file_edits": [
                {
                    "path": "instructions/sql_snippets/measures/plain_average_fare.yml",
                    "content": "\n".join(
                        [
                            "id: plain_average_fare_avg_fare",
                            "display_name: Average fare",
                            "sql: AVG(fare_amount)",
                            "",
                        ]
                    ),
                    "reason": "target_train_failure:average_fare",
                }
            ],
        },
    )

    content = repaired["file_edits"][0]["content"]
    loaded = yaml.safe_load(content)
    assert summary["changed"] is True
    assert summary["repaired_artifact_id_paths"] == [
        {
            "path": "instructions/sql_snippets/measures/plain_average_fare.yml",
            "item": "$",
            "old_id": "plain_average_fare_avg_fare",
            "new_id": loaded["id"],
        }
    ]
    assert re.fullmatch(r"[0-9a-f]{32}", loaded["id"])
