"""Tests for the STEP 1d wiring fix: precompute the full answer key before baseline.

``compare()`` lazily caches expected rows only for questions whose *generated* SQL
succeeds, so the dominant unbound-parameter failures (generated raises first) never get
gold rows. ``precompute_answer_key`` proactively executes ALL expected SQLs once via the
comparator so the gold store covers visible AND holdout up front. The holdout gold rows
stay in the key (a gold store), never in proposer context.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxgenie.workspace_flow import OptimizationWorkspace


class _Q:
    def __init__(self, expected_sql):
        self.expected_sql = expected_sql


def _make_stub(precompute_return=15):
    calls = {}

    def _precompute(sqls):
        calls["sqls"] = list(sqls)
        return precompute_return

    iterations = []
    stub = SimpleNamespace(
        comparator=SimpleNamespace(precompute_expected=_precompute),
        artifacts=SimpleNamespace(record_iteration=lambda event: iterations.append(event)),
    )
    return stub, calls, iterations


def test_precompute_answer_key_executes_all_expected_sqls_once():
    stub, calls, iterations = _make_stub(precompute_return=15)
    questions = [_Q(f"SELECT {i}") for i in range(15)]

    computed = OptimizationWorkspace.precompute_answer_key(stub, questions)

    assert computed == 15
    # Every expected SQL is handed to the comparator, not just the generated-success ones.
    assert calls["sqls"] == [f"SELECT {i}" for i in range(15)]
    assert iterations and iterations[0]["phase"] == "answer_key_precompute"
    assert iterations[0]["total_questions"] == 15
    assert iterations[0]["distinct_expected_sqls"] == 15
    assert iterations[0]["newly_computed"] == 15


def test_precompute_answer_key_skips_questions_without_expected_sql():
    stub, calls, _ = _make_stub()
    questions = [_Q("SELECT 1"), _Q(None), _Q("")]

    OptimizationWorkspace.precompute_answer_key(stub, questions)

    assert calls["sqls"] == ["SELECT 1"]


def test_precompute_answer_key_noop_without_comparator():
    stub = SimpleNamespace(comparator=None)
    assert OptimizationWorkspace.precompute_answer_key(stub, [_Q("SELECT 1")]) == 0


def test_precompute_answer_key_noop_on_empty_questions():
    stub, calls, iterations = _make_stub()
    assert OptimizationWorkspace.precompute_answer_key(stub, []) == 0
    assert "sqls" not in calls
    assert iterations == []
