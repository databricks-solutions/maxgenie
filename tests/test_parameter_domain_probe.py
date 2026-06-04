"""Tests for STEP 1e part (3): parameter literal/domain probing (visible-only).

The proposer's dominant lever is literalizing unbound ``:parameter`` markers. These tests
cover table mining from visible SQL, the read-only warehouse probe (first table that has
the column wins, errors swallowed), the attach/cache method, and the proposer-context
rendering — all without a live warehouse. Tables are mined only from visible SQL and the
probe returns column domains (never holdout answer rows), so nothing here leaks holdout.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

from maxgenie.serving_candidates import _artifact_patch_parameter_domain_lines
from maxgenie.workspace_flow import (
    OptimizationWorkspace,
    _extract_sql_table_identifiers,
    _ordered_candidate_tables,
    _probe_parameter_domain_values,
)


def test_extract_sql_table_identifiers_from_from_and_join():
    sql = (
        "SELECT * FROM cat.sch.sales_fact s "
        "JOIN `cat`.`sch`.`brand_dim` b ON s.id = b.id "
        "LEFT JOIN cat.sch.sales_fact again ON 1=1"
    )
    # Backticks stripped, qualified names captured, duplicates collapsed.
    assert _extract_sql_table_identifiers(sql) == ["cat.sch.sales_fact", "cat.sch.brand_dim"]


def test_extract_sql_table_identifiers_handles_empty():
    assert _extract_sql_table_identifiers(None) == []
    assert _extract_sql_table_identifiers("") == []


def test_ordered_candidate_tables_ranks_by_frequency_then_first_seen():
    sqls = [
        "SELECT 1 FROM a.b.dim",
        "SELECT 1 FROM a.b.fact JOIN a.b.dim ON 1=1",
        "SELECT 1 FROM a.b.fact",
        "SELECT 1 FROM a.b.fact",
    ]
    # fact=3, dim=2 -> fact ranks first by count.
    assert _ordered_candidate_tables(sqls) == ["a.b.fact", "a.b.dim"]


def test_ordered_candidate_tables_caps_table_count():
    sqls = [f"SELECT 1 FROM c.s.t{i}" for i in range(10)]
    assert len(_ordered_candidate_tables(sqls, max_tables=3)) == 3


def test_probe_parameter_domain_values_first_table_with_column_wins():
    calls: list[str] = []

    def _exec(sql: str):
        calls.append(sql)
        if "time_grp_type" in sql and "FROM c.s.fact" in sql:
            return (["time_grp_type"], [["WEEKLY"], ["MONTHLY"], ["WEEKLY"]])
        raise RuntimeError("[UNRESOLVED_COLUMN] time_grp_type not in this table")

    out = _probe_parameter_domain_values(_exec, ["c.s.dim", "c.s.fact"], columns=["time_grp_type"])

    # Deduped, order preserved; only the table that has the column contributes.
    assert out == {"time_grp_type": ["WEEKLY", "MONTHLY"]}
    # Tried dim first (errored), then fact (hit), then stopped -> exactly two probes.
    assert len(calls) == 2


def test_probe_parameter_domain_values_swallows_errors_and_skips_missing():
    def _exec(sql: str):
        raise RuntimeError("column does not exist anywhere")

    assert _probe_parameter_domain_values(_exec, ["c.s.t"], columns=["metric_type"]) == {}


def test_probe_parameter_domain_values_caps_values():
    def _exec(sql: str):
        return (["market_scope"], [[f"M{i}"] for i in range(100)])

    out = _probe_parameter_domain_values(_exec, ["c.s.t"], columns=["market_scope"], value_limit=10)
    assert len(out["market_scope"]) == 10


def test_visible_sqls_for_domain_probe_collects_failures_and_guards():
    stub = SimpleNamespace()
    context = {
        "failures": [
            {"expected_sql": "SELECT a", "generated_sql": "SELECT b"},
            {"generated_sql": "SELECT c"},
            "not-a-dict",
        ],
        "pass_guards": [{"expected_sql": "SELECT d"}],
    }
    out = OptimizationWorkspace._visible_sqls_for_domain_probe(stub, context)
    assert out == ["SELECT a", "SELECT b", "SELECT c", "SELECT d"]


def test_attach_parameter_domain_values_mines_visible_sql_and_caches():
    probed: list[str] = []

    def _exec(sql: str):
        probed.append(sql)
        if "time_grp_type" in sql:
            return (["time_grp_type"], [["WEEKLY"], ["MONTHLY"]])
        raise RuntimeError("unresolved column")

    stub = SimpleNamespace(
        comparator=SimpleNamespace(execute_readonly=_exec),
        _parameter_domain_values_cache=None,
    )
    stub._visible_sqls_for_domain_probe = types.MethodType(
        OptimizationWorkspace._visible_sqls_for_domain_probe, stub
    )
    context = {
        "failures": [
            {"expected_sql": "SELECT 1 FROM c.s.fact WHERE x=1", "generated_sql": "SELECT 1 FROM c.s.fact"}
        ],
        "pass_guards": [{"expected_sql": "SELECT 1 FROM c.s.fact"}],
    }

    OptimizationWorkspace._attach_parameter_domain_values(stub, context)
    assert context["parameter_domain_values"]["time_grp_type"] == ["WEEKLY", "MONTHLY"]

    # Cached after the first probe -> a second attach never re-queries the warehouse.
    before = len(probed)
    context2 = {"failures": [{"expected_sql": "SELECT 1 FROM c.s.fact"}], "pass_guards": []}
    OptimizationWorkspace._attach_parameter_domain_values(stub, context2)
    assert len(probed) == before
    assert context2["parameter_domain_values"]["time_grp_type"] == ["WEEKLY", "MONTHLY"]


def test_attach_parameter_domain_values_noops_without_comparator():
    stub = SimpleNamespace(comparator=None, _parameter_domain_values_cache=None)
    context = {"failures": [{"expected_sql": "SELECT 1 FROM c.s.fact"}]}
    OptimizationWorkspace._attach_parameter_domain_values(stub, context)
    assert "parameter_domain_values" not in context


def test_parameter_domain_lines_render_block_when_present():
    ctx = {
        "parameter_domain_values": {
            "time_grp_type": ["WEEKLY", "MONTHLY"],
            "metric_type": ["TRX", "NRX"],
        }
    }
    text = "\n".join(_artifact_patch_parameter_domain_lines(ctx))
    assert "Parameter Literal Domains" in text
    assert "`time_grp_type`: WEEKLY, MONTHLY" in text
    assert "`metric_type`: TRX, NRX" in text


def test_parameter_domain_lines_empty_without_domains():
    assert _artifact_patch_parameter_domain_lines({}) == []
    assert _artifact_patch_parameter_domain_lines({"parameter_domain_values": {}}) == []
    assert _artifact_patch_parameter_domain_lines(None) == []
