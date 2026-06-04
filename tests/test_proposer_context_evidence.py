"""Tests for max (visible-only) proposer context evidence.

Targeted failures must receive the full generated SQL and the full execution error
(no ~260-char clip), so the proposer can see and fix the exact divergence — most
importantly the unbound ``:parameter`` that drives the dominant failure family.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxgenie.serving_candidates import _artifact_patch_target_evidence_line
from maxgenie.workspace_flow import (
    OptimizationWorkspace,
    _format_expected_result_preview,
)


def test_target_evidence_line_keeps_full_sql_and_full_error():
    long_sql = (
        "SELECT " + ", ".join(f"col_{i}" for i in range(80)) + " FROM sales_fact "
        "WHERE time_grp_type = :time_grp_type AND metric_type = :metric_type"
    )
    assert len(long_sql) > 400  # comfortably beyond the old 260-char clip

    failure = {
        "question": "Show weekly TRx for the brand",
        "issue_class": "result_unbound_parameters_error",
        "comparison_method": "result_unbound_parameters_error",
        "expected_sql": "SELECT 1",
        "generated_sql": long_sql,
        "error_message": (
            "[UNBOUND_SQL_PARAMETER] Found the unbound parameter: time_grp_type "
            "at line 1, col 142. Bind it or remove the parameter marker."
        ),
    }

    line = _artifact_patch_target_evidence_line(failure)

    # Full SQL preserved end-to-end (the trailing :metric_type survives, no truncation).
    assert ":metric_type" in line
    assert "(truncated)" not in line
    # Full execution error included, naming the exact unbound parameter and line.
    assert "UNBOUND_SQL_PARAMETER" in line
    assert "time_grp_type at line 1, col 142" in line


def test_target_evidence_line_falls_back_to_comparison_details_for_error():
    failure = {
        "question": "Total sales",
        "issue_class": "result_mismatch",
        "generated_sql": "SELECT total FROM t",
        "comparison_details": "Cell match rate: 0.200 (1/5 cells, threshold=0.8)",
    }

    line = _artifact_patch_target_evidence_line(failure)

    assert "Cell match rate: 0.200" in line


def test_format_expected_result_preview_compact_and_truncates():
    cols = ["brand", "trx"]
    rows = [["AZ-1", 1], ["AZ-2", 2], ["AZ-3", 3]]
    preview = _format_expected_result_preview(cols, rows, max_rows=2)
    assert preview == "cols=[brand, trx] rows=[(AZ-1, 1); (AZ-2, 2)] (+1 more rows)"
    # NULLs render explicitly; empty input -> empty string (nothing to attach).
    assert _format_expected_result_preview(["a"], [[None]]) == "cols=[a] rows=[(NULL)]"
    assert _format_expected_result_preview([], []) == ""


def test_target_evidence_line_includes_expected_result_when_present():
    failure = {
        "question": "Weekly TRx for the brand",
        "generated_sql": "SELECT trx FROM sales",
        "expected_result_preview": "cols=[brand, trx] rows=[(AZ-1, 42)]",
    }

    line = _artifact_patch_target_evidence_line(failure)

    # The visible answer-key target rows reach the proposer verbatim (no backtick wrap).
    assert "expected_result=cols=[brand, trx] rows=[(AZ-1, 42)]" in line


def test_attach_visible_expected_rows_attaches_preview_from_answer_key():
    queried: list[str] = []

    def _lookup(sql: str):
        queried.append(sql)
        if sql == "SELECT brand, trx FROM sales":
            return (["brand", "trx"], [["AZ-1", 42], ["AZ-2", 17]])
        return None

    stub = SimpleNamespace(comparator=SimpleNamespace(expected_rows_for_sql=_lookup))
    context = {
        "failures": [
            {"question_id": "q1", "expected_sql": "SELECT brand, trx FROM sales"},
            {"question_id": "q2", "expected_sql": "SELECT 1"},  # absent from answer key
            {"question_id": "q3"},  # no expected_sql -> never looked up
        ]
    }

    OptimizationWorkspace._attach_visible_expected_rows(stub, context)

    preview = context["failures"][0]["expected_result_preview"]
    assert "brand, trx" in preview
    assert "AZ-1" in preview
    # No answer-key row -> no preview; no SQL -> not even queried.
    assert "expected_result_preview" not in context["failures"][1]
    assert "expected_result_preview" not in context["failures"][2]
    assert queried == ["SELECT brand, trx FROM sales", "SELECT 1"]


def test_attach_visible_expected_rows_is_read_only_and_visible_scoped():
    # The comparator exposes ONLY the read-only lookup. If _attach tried to execute SQL
    # or enumerate the full answer key (which could include holdout), it would fail loudly
    # rather than silently leak. It must query exactly the visible failures it was given.
    queried: list[str] = []

    def _lookup(sql: str):
        queried.append(sql)
        return (["c"], [[1]])

    stub = SimpleNamespace(comparator=SimpleNamespace(expected_rows_for_sql=_lookup))
    context = {"failures": [{"question_id": "v1", "expected_sql": "SELECT c"}]}

    OptimizationWorkspace._attach_visible_expected_rows(stub, context)

    assert queried == ["SELECT c"]
    assert context["failures"][0]["expected_result_preview"] == "cols=[c] rows=[(1)]"


def test_attach_visible_expected_rows_noops_without_comparator():
    stub = SimpleNamespace(comparator=None)
    context = {"failures": [{"question_id": "v1", "expected_sql": "SELECT c"}]}

    OptimizationWorkspace._attach_visible_expected_rows(stub, context)

    assert "expected_result_preview" not in context["failures"][0]
