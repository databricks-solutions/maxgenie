from unittest.mock import MagicMock

from databricks.sdk import WorkspaceClient

from maxgenie.result_comparator import (
    CONTENT_MATCH_THRESHOLD,
    INLINE_WAIT_TIMEOUT,
    ResultComparator,
    _compare_content_based,
    _content_column_role,
    _extract_sql_named_placeholders,
    _format_row_level_diff,
    _match_content_columns,
    _rows_match_alias_tolerant,
)


def test_execute_sql_uses_supported_inline_wait_timeout():
    wc = MagicMock(spec=WorkspaceClient)
    wc.api_client.do.return_value = {
        "status": {"state": "SUCCEEDED"},
        "manifest": {"schema": {"columns": [{"name": "value"}]}},
        "result": {"data_array": [[1]]},
    }

    comparator = ResultComparator(wc=wc, warehouse_id="wh-123")

    columns, rows = comparator._execute_sql("SELECT 1")

    assert columns == ["value"]
    assert rows == [[1]]
    _, _, kwargs = wc.api_client.do.mock_calls[0]
    assert kwargs["body"]["wait_timeout"] == INLINE_WAIT_TIMEOUT
    assert INLINE_WAIT_TIMEOUT == "50s"


def test_execute_sql_allows_leading_comments():
    wc = MagicMock(spec=WorkspaceClient)
    wc.api_client.do.return_value = {
        "status": {"state": "SUCCEEDED"},
        "manifest": {"schema": {"columns": [{"name": "value"}]}},
        "result": {"data_array": [[1]]},
    }

    comparator = ResultComparator(wc=wc, warehouse_id="wh-123")

    columns, rows = comparator._execute_sql("-- explain intent\n/* block */\nSELECT 1")

    assert columns == ["value"]
    assert rows == [[1]]


# ---------------------------------------------------------------------------
# Helpers for alias-tolerant tests
# ---------------------------------------------------------------------------

def _make_comparator_with_two_results(gen_cols, gen_rows, exp_cols, exp_rows):
    """Build a ResultComparator whose _execute_sql returns canned results."""
    wc = MagicMock(spec=WorkspaceClient)
    # First call → generated SQL result, second call → expected SQL result
    wc.api_client.do.side_effect = [
        {
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": c} for c in gen_cols]}},
            "result": {"data_array": gen_rows},
        },
        {
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": c} for c in exp_cols]}},
            "result": {"data_array": exp_rows},
        },
    ]
    return ResultComparator(wc=wc, warehouse_id="wh-test")


# ---------------------------------------------------------------------------
# Alias-tolerant integration tests (via ResultComparator.compare)
# ---------------------------------------------------------------------------

def test_alias_tolerant_match_when_columns_rename_but_values_align():
    """Same numeric values, different column names → result_match_alias_tolerant."""
    comparator = _make_comparator_with_two_results(
        gen_cols=["total_trx"],
        gen_rows=[["100"], ["200"], ["300"]],
        exp_cols=["trx"],
        exp_rows=[["100"], ["200"], ["300"]],
    )
    passed, score, method, _ = comparator.compare("SELECT total_trx FROM t", "SELECT trx FROM t")
    assert method == "result_match_alias_tolerant"
    assert passed is True
    assert score == 1.0


def test_alias_tolerant_still_mismatch_when_row_values_differ():
    """Column count matches but one cell value is outside tolerance → result_column_mismatch."""
    comparator = _make_comparator_with_two_results(
        gen_cols=["sum_trx"],
        gen_rows=[["100"], ["999"]],
        exp_cols=["trx"],
        exp_rows=[["100"], ["200"]],
    )
    passed, score, method, _ = comparator.compare("SELECT sum_trx FROM t", "SELECT trx FROM t")
    assert method == "result_column_mismatch"
    assert passed is False


def test_alias_tolerant_still_mismatch_when_column_counts_differ():
    """Different column counts → alias fallback is skipped → result_column_mismatch."""
    comparator = _make_comparator_with_two_results(
        gen_cols=["a", "b"],
        gen_rows=[["1", "2"]],
        exp_cols=["x"],
        exp_rows=[["1"]],
    )
    passed, score, method, _ = comparator.compare("SELECT a, b FROM t", "SELECT x FROM t")
    assert method == "result_column_mismatch"
    assert passed is False


# ---------------------------------------------------------------------------
# Row-level diff for mismatch failures (STEP 1e part 1: visible-only proposer context)
# ---------------------------------------------------------------------------

def test_format_row_level_diff_compact():
    out = _format_row_level_diff(["trx"], [["100"], ["200"]], ["sum_trx"], [["100"], ["999"]])
    assert out.startswith("row-level diff: ")
    assert "exp_cols=[trx]" in out
    assert "gen_cols=[sum_trx]" in out
    assert "r0 exp=(100) gen=(100)" in out
    assert "r1 exp=(200) gen=(999)" in out


def test_format_row_level_diff_truncates_rows_and_marks_missing():
    out = _format_row_level_diff(["x"], [["1"], ["2"], ["3"]], ["x"], [["1"]], max_rows=2)
    assert "r0 exp=(1) gen=(1)" in out
    assert "r1 exp=(2) gen=(missing)" in out
    assert "(+1 more rows)" in out


def test_compare_column_mismatch_details_include_row_level_diff():
    comparator = _make_comparator_with_two_results(
        gen_cols=["a", "b"],
        gen_rows=[["1", "2"]],
        exp_cols=["x"],
        exp_rows=[["1"]],
    )
    passed, _score, method, details = comparator.compare("SELECT a, b FROM t", "SELECT x FROM t")
    assert method == "result_column_mismatch"
    assert passed is False
    assert "Column mismatch:" in details
    assert "row-level diff:" in details
    assert "gen_cols=[a, b]" in details


def test_compare_cell_mismatch_details_include_row_level_diff():
    comparator = _make_comparator_with_two_results(
        gen_cols=["x"],
        gen_rows=[["9"], ["9"], ["9"]],
        exp_cols=["x"],
        exp_rows=[["1"], ["2"], ["3"]],
    )
    passed, _score, method, details = comparator.compare("SELECT x FROM t", "SELECT x FROM t")
    assert method == "result_mismatch"
    assert passed is False
    assert "Cell match rate:" in details
    assert "row-level diff:" in details
    assert "r0 exp=(1) gen=(9)" in details


# ---------------------------------------------------------------------------
# Unit tests for the _rows_match_alias_tolerant helper
# ---------------------------------------------------------------------------

def test_rows_match_alias_tolerant_numeric_near_equal_within_eps():
    """Numeric values differing by less than 1e-6 are treated as equal."""
    left = [[1.0000003]]
    right = [[1.0000000]]
    # difference is 3e-7, within abs eps 1e-6
    assert _rows_match_alias_tolerant(left, right) is True


def test_rows_match_alias_tolerant_numeric_outside_eps():
    """Numeric values differing by more than eps are not equal."""
    left = [[1.001]]
    right = [[1.0]]
    assert _rows_match_alias_tolerant(left, right) is False


def test_rows_match_alias_tolerant_row_order_insensitive():
    """Rows in a different order but with the same values should match."""
    left = [["300"], ["100"], ["200"]]
    right = [["100"], ["200"], ["300"]]
    assert _rows_match_alias_tolerant(left, right) is True


def test_rows_match_alias_tolerant_row_count_mismatch():
    """Different number of rows → False."""
    left = [["1"], ["2"]]
    right = [["1"]]
    assert _rows_match_alias_tolerant(left, right) is False


def test_strict_name_match_still_reports_result_match_not_alias_tolerant():
    """When column names are identical, result_match should be returned, not the alias path."""
    comparator = _make_comparator_with_two_results(
        gen_cols=["trx"],
        gen_rows=[["100"]],
        exp_cols=["trx"],
        exp_rows=[["100"]],
    )
    passed, score, method, _ = comparator.compare("SELECT trx FROM t", "SELECT trx FROM t")
    assert method == "result_match"
    assert passed is True


# ---------------------------------------------------------------------------
# _extract_sql_named_placeholders tests
# ---------------------------------------------------------------------------

def test_extract_placeholders_finds_single_name():
    result = _extract_sql_named_placeholders("SELECT * FROM t WHERE brand = :brand_name")
    assert result == {"brand_name"}


def test_extract_placeholders_ignores_postgres_cast():
    result = _extract_sql_named_placeholders("SELECT col::text FROM t")
    assert result == set()


def test_extract_placeholders_skips_numeric_prefix():
    result = _extract_sql_named_placeholders("SELECT :123 FROM t")
    assert result == set()


def test_extract_placeholders_returns_unique_set():
    result = _extract_sql_named_placeholders(
        "SELECT * FROM t WHERE a = :brand AND b = :brand AND c = :region"
    )
    assert result == {"brand", "region"}


# ---------------------------------------------------------------------------
# parameter_values binding tests
# ---------------------------------------------------------------------------

_SUCCEEDED_RESPONSE = {
    "status": {"state": "SUCCEEDED"},
    "manifest": {"schema": {"columns": []}},
    "result": {"data_array": []},
}


def _make_capturing_wc():
    """Return a fake WorkspaceClient whose api_client.do records calls."""
    captured: list[tuple] = []

    class _FakeApiClient:
        def do(self, method, path, body=None, **kwargs):
            captured.append((method, path, body))
            return _SUCCEEDED_RESPONSE

    class _FakeWC:
        api_client = _FakeApiClient()

    return _FakeWC(), captured


def test_execute_sql_binds_parameters_when_values_provided():
    wc, captured = _make_capturing_wc()
    comparator = ResultComparator(wc, warehouse_id="w", parameter_values={"brand_name": "Farxiga"})
    comparator._execute_sql("SELECT * FROM t WHERE brand = :brand_name")
    _, _, body = captured[0]
    assert body["parameters"] == [{"name": "brand_name", "value": "Farxiga", "type": "STRING"}]


def test_execute_sql_omits_parameters_when_values_none():
    wc, captured = _make_capturing_wc()
    comparator = ResultComparator(wc, warehouse_id="w")
    comparator._execute_sql("SELECT * FROM t WHERE brand = :brand_name")
    _, _, body = captured[0]
    assert "parameters" not in body


def test_execute_sql_partial_binding_only_sends_known_names():
    wc, captured = _make_capturing_wc()
    comparator = ResultComparator(wc, warehouse_id="w", parameter_values={"a": "1"})
    comparator._execute_sql("SELECT * FROM t WHERE x = :a AND y = :b")
    _, _, body = captured[0]
    assert body["parameters"] == [{"name": "a", "value": "1", "type": "STRING"}]


def test_cache_fingerprint_differs_with_parameter_values():
    wc1 = MagicMock(spec=WorkspaceClient)
    wc2 = MagicMock(spec=WorkspaceClient)
    comparator_none = ResultComparator(wc1, warehouse_id="w")
    comparator_with = ResultComparator(wc2, warehouse_id="w", parameter_values={"x": "1"})
    assert comparator_none.cache_fingerprint() != comparator_with.cache_fingerprint()


def test_cache_fingerprint_differs_when_parameter_value_changes():
    wc = MagicMock(spec=WorkspaceClient)
    a = ResultComparator(wc, warehouse_id="w", parameter_values={"brand": "Farxiga"})
    b = ResultComparator(wc, warehouse_id="w", parameter_values={"brand": "Ozempic"})
    assert a.cache_fingerprint() != b.cache_fingerprint()


def test_execute_sql_omits_parameters_when_sql_has_no_placeholders():
    wc, captured = _make_capturing_wc()
    comparator = ResultComparator(wc, warehouse_id="w", parameter_values={"x": "1"})
    comparator._execute_sql("SELECT 1")
    _, _, body = captured[0]
    assert "parameters" not in body


# ---------------------------------------------------------------------------
# Precomputed answer key.
# ---------------------------------------------------------------------------
def test_answer_key_resolves_expected_once_and_reuses():
    wc = MagicMock(spec=WorkspaceClient)
    comparator = ResultComparator(wc=wc, warehouse_id="wh")
    calls: list[str] = []

    def fake_exec(sql):
        calls.append(sql)
        return ["trx"], [["10"]]

    comparator._execute_sql = fake_exec  # type: ignore[method-assign]
    first = comparator.compare("SELECT trx FROM gen_a", "SELECT trx FROM answer")
    second = comparator.compare("SELECT trx FROM gen_b", "SELECT trx FROM answer")

    assert first[0] is True and second[0] is True
    # Expected (answer) SQL executed exactly once; generated executed per candidate.
    assert calls.count("SELECT trx FROM answer") == 1
    assert sum(1 for c in calls if c.startswith("SELECT trx FROM gen_")) == 2


def test_answer_key_persists_and_reloads_without_executing(tmp_path):
    wc = MagicMock(spec=WorkspaceClient)
    path = tmp_path / ".answer_key.json"
    comparator = ResultComparator(wc=wc, warehouse_id="wh", answer_key_path=path)
    comparator._execute_sql = lambda sql: (["trx"], [["10"]])  # type: ignore[method-assign]
    comparator.compare("SELECT trx FROM gen", "SELECT trx FROM answer")
    assert path.exists()

    reloaded = ResultComparator(wc=wc, warehouse_id="wh", answer_key_path=path)

    def _no_exec(sql):
        raise AssertionError("answer key is precomputed; must not execute expected SQL")

    reloaded._execute_sql = _no_exec  # type: ignore[method-assign]
    assert reloaded.expected_rows_for_sql("SELECT trx FROM answer") == (["trx"], [["10"]])


def test_expected_rows_for_sql_returns_none_without_executing():
    wc = MagicMock(spec=WorkspaceClient)
    comparator = ResultComparator(wc=wc, warehouse_id="wh")

    def _no_exec(sql):
        raise AssertionError("lookup must not execute SQL on a miss")

    comparator._execute_sql = _no_exec  # type: ignore[method-assign]
    assert comparator.expected_rows_for_sql("SELECT 1") is None


def test_precompute_expected_executes_each_distinct_once():
    wc = MagicMock(spec=WorkspaceClient)
    comparator = ResultComparator(wc=wc, warehouse_id="wh")
    calls: list[str] = []

    def fake_exec(sql):
        calls.append(sql)
        return ["c"], [[sql]]

    comparator._execute_sql = fake_exec  # type: ignore[method-assign]
    # The third entry normalizes to the first (whitespace/case), so it is not re-run.
    computed = comparator.precompute_expected(["SELECT a", "SELECT b", "select  a "])

    assert computed == 2
    assert len(calls) == 2
    assert comparator.expected_rows_for_sql("SELECT a") == (["c"], [["SELECT a"]])


# ===========================================================================
# Content-based ("fair") comparator — opt-in value-first ruler (Phase A).
# Designed on general principles (ignore constant label columns, match columns
# by value, mandatory measures/dimensions), validated against — never fit to —
# the holdout questions Q2 (must PASS) and Q1 (must still FAIL).
# ===========================================================================


def _content_comparator(gen_cols, gen_rows, exp_cols, exp_rows):
    """Build a content_match=True comparator returning canned gen then exp results."""
    wc = MagicMock(spec=WorkspaceClient)
    wc.api_client.do.side_effect = [
        {
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": c} for c in gen_cols]}},
            "result": {"data_array": gen_rows},
        },
        {
            "status": {"state": "SUCCEEDED"},
            "manifest": {"schema": {"columns": [{"name": c} for c in exp_cols]}},
            "result": {"data_array": exp_rows},
        },
    ]
    return ResultComparator(wc=wc, warehouse_id="wh-test", content_match=True)


# --- direct helper unit tests -------------------------------------------------

def test_content_column_role_classifies_constant_measure_dimension():
    assert _content_column_role(["TRX", "TRX", "TRX"]) == "constant"
    assert _content_column_role(["10", "20", "30"]) == "measure"
    assert _content_column_role(["east", "west", "north"]) == "dimension"
    assert _content_column_role([]) == "constant"  # no values → trivially constant
    assert _content_column_role([None, None]) == "constant"


def test_match_content_columns_aligns_rename_by_value():
    exp_cols = ["region", "trx"]
    exp_rows = [["east", "10"], ["west", "20"]]
    gen_cols = ["region", "sum_trx"]
    gen_rows = [["east", "10"], ["west", "20"]]
    matches = _match_content_columns(exp_cols, exp_rows, gen_cols, gen_rows)
    assert matches[0] == 0  # region matched by name
    assert matches[1] == 1  # trx → sum_trx matched by value-multiset


# --- layer (i): ignore constant label columns --------------------------------

def test_content_layer_i_ignores_missing_constant_label_column():
    # Expected has a constant label column 'metric_type' absent from generated;
    # every varying value matches → PASS (the label is optional).
    comparator = _content_comparator(
        gen_cols=["region", "trx"],
        gen_rows=[["east", "10"], ["west", "20"], ["north", "30"]],
        exp_cols=["region", "trx", "metric_type"],
        exp_rows=[["east", "10", "TRX"], ["west", "20", "TRX"], ["north", "30", "TRX"]],
    )
    passed, score, method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is True
    assert method == "result_match_content"
    assert score > 0.5  # not a hard zero despite the column-set delta


# --- layer (ii): match renamed varying columns by value ----------------------

def test_content_layer_ii_matches_renamed_varying_column_by_value():
    comparator = _content_comparator(
        gen_cols=["region", "total_trx"],  # measure renamed
        gen_rows=[["east", "10"], ["west", "20"], ["north", "30"]],
        exp_cols=["region", "trx"],
        exp_rows=[["east", "10"], ["west", "20"], ["north", "30"]],
    )
    passed, score, method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is True
    assert score == 1.0


# --- layer (iii): mandatory measures/dimensions are required -----------------

def test_content_layer_iii_missing_mandatory_measure_fails():
    comparator = _content_comparator(
        gen_cols=["region", "trx"],  # mandatory varying measure 'nrx' absent
        gen_rows=[["east", "10"], ["west", "20"], ["north", "30"]],
        exp_cols=["region", "trx", "nrx"],
        exp_rows=[["east", "10", "1"], ["west", "20", "2"], ["north", "30", "3"]],
    )
    passed, _score, method, details = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False
    assert method == "result_content_mandatory_missing"
    assert "nrx" in details


# --- layer (iv): union partial credit (gradient), not a hard zero ------------

def test_content_layer_iv_partial_overlap_is_a_gradient_not_a_cliff():
    # Column sets differ (legacy would return a hard 0.0); content gives partial
    # credit so the optimizer can tell "almost right" from "totally wrong".
    comparator = _content_comparator(
        gen_cols=["region", "trx", "junk"],
        gen_rows=[["east", "10", "9"], ["west", "20", "9"], ["north", "30", "9"]],
        exp_cols=["region", "trx", "nrx"],
        exp_rows=[["east", "10", "1"], ["west", "20", "2"], ["north", "30", "3"]],
    )
    passed, score, _method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False  # mandatory 'nrx' unmatched
    assert 0.0 < score < 1.0  # but a usable gradient, not a cliff


# --- layer (v): pass rule — coverage / cardinality gate ----------------------

def test_content_layer_v_low_row_coverage_fails():
    comparator = _content_comparator(
        gen_cols=["region", "trx"],
        gen_rows=[["e", "1"]],  # only 1 of 5 expected rows
        exp_cols=["region", "trx"],
        exp_rows=[["e", "1"], ["w", "2"], ["n", "3"], ["s", "4"], ["c", "5"]],
    )
    passed, _score, _method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False


# --- negative control: fair, not lenient -------------------------------------

def test_content_negative_control_wrong_values_fail():
    # Identical shape, deliberately wrong measure values → must FAIL.
    comparator = _content_comparator(
        gen_cols=["region", "trx", "nrx"],
        gen_rows=[["east", "999", "111"], ["west", "888", "222"], ["north", "777", "333"]],
        exp_cols=["region", "trx", "nrx"],
        exp_rows=[["east", "10", "1"], ["west", "20", "2"], ["north", "30", "3"]],
    )
    passed, score, _method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False
    assert score < CONTENT_MATCH_THRESHOLD


# --- sanity: exact match still scores 1.0 ------------------------------------

def test_content_exact_match_scores_one():
    comparator = _content_comparator(
        gen_cols=["region", "trx"],
        gen_rows=[["east", "10"], ["west", "20"]],
        exp_cols=["region", "trx"],
        exp_rows=[["east", "10"], ["west", "20"]],
    )
    passed, score, method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is True
    assert score == 1.0
    assert method == "result_match_content"


# --- flag off by default: legacy behavior is unchanged -----------------------

def test_content_match_off_by_default_keeps_legacy_hard_zero():
    # Same Q2-like column delta, but content_match defaults False → legacy 0.0.
    comparator = _make_comparator_with_two_results(
        gen_cols=["region", "trx"],
        gen_rows=[["east", "10"]],
        exp_cols=["region", "trx", "metric_type"],
        exp_rows=[["east", "10", "TRX"]],
    )
    passed, score, method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert method == "result_column_mismatch"
    assert passed is False
    assert score == 0.0


def test_cache_fingerprint_distinguishes_content_match():
    wc = MagicMock(spec=WorkspaceClient)
    off = ResultComparator(wc, warehouse_id="w")
    on = ResultComparator(wc, warehouse_id="w", content_match=True)
    assert off.cache_fingerprint() != on.cache_fingerprint()
    assert "content=on" in on.cache_fingerprint()
    assert "content" not in off.cache_fingerprint()  # off path byte-identical to legacy


# --- validation pins: the real holdout questions -----------------------------
# Faithful reconstructions of the seed-42 holdout from the champion test-run
# evidence (history/test_runs/...checkpoint_audit.final.test.json). Q2 differs
# from gold only by a renamed constant column (time_grp_desc→wk_grp_desc) and a
# missing constant label (metric_type); Q1 returns a single total instead of the
# multi-row decomposition (the mandatory measure columns are absent).

_HOLDOUT_EXP_COLS = [
    "time_bucket_id", "time_grp_desc", "market_name", "sub_market_name", "brand_name",
    "team_name", "hierarchy_level", "region", "metric_type", "brand_metric_value",
    "total_market_metric_value", "metric_market_share_pct", "assumption_time_period",
    "assumption_hierarchy_level", "assumption_market_scope", "assumption_team_scope",
    "assumption_metric_type",
]
# Champion Q2 output: time_grp_desc renamed to wk_grp_desc, metric_type dropped.
_Q2_GEN_COLS = [
    "time_bucket_id", "wk_grp_desc", "market_name", "sub_market_name", "brand_name",
    "team_name", "hierarchy_level", "region", "brand_metric_value",
    "total_market_metric_value", "metric_market_share_pct", "assumption_time_period",
    "assumption_hierarchy_level", "assumption_market_scope", "assumption_team_scope",
    "assumption_metric_type",
]


def _q2_exp_row(tbid, team, region, bmv, tmmv, mspct):
    return [
        tbid, "Current 5 Week", "hyperkalemia market", "hyperkalemia submarket", "lokelma",
        team, "REGN", region, "NBRX", bmv, tmmv, mspct, "P1", "REGN", "mkt", "team", "NBRX",
    ]


def _q2_gen_row(tbid, team, region, bmv, tmmv, mspct):
    # metric_type dropped; time_grp_desc carried under wk_grp_desc; region lower-cased.
    return [
        tbid, "Current 5 Week", "hyperkalemia market", "hyperkalemia submarket", "lokelma",
        team, "REGN", region.lower(), bmv, tmmv, mspct, "P1", "REGN", "mkt", "team", "NBRX",
    ]


def test_content_holdout_q2_passes_at_fair_score():
    exp_rows = [
        _q2_exp_row("W001", "spec_cvrm_1_total", "Atlantic Coast", "100.0", "1000.0", "10.0"),
        _q2_exp_row("W001", "spec_cvrm_total", "Atlantic Coast", "200.0", "1000.0", "20.0"),
        _q2_exp_row("W002", "spec_cvrm_2_total", "Pacific", "300.0", "1200.0", "25.0"),
        _q2_exp_row("W002", "spec_cvrm_3_total", "Pacific", "400.0", "1200.0", "33.0"),
    ]
    # Generated rows deliberately in a different order to prove order-independence.
    gen_rows = [
        _q2_gen_row("W002", "spec_cvrm_2_total", "Pacific", "300.0", "1200.0", "25.0"),
        _q2_gen_row("W001", "spec_cvrm_1_total", "Atlantic Coast", "100.0", "1000.0", "10.0"),
        _q2_gen_row("W002", "spec_cvrm_3_total", "Pacific", "400.0", "1200.0", "33.0"),
        _q2_gen_row("W001", "spec_cvrm_total", "Atlantic Coast", "200.0", "1000.0", "20.0"),
    ]
    comparator = _content_comparator(_Q2_GEN_COLS, gen_rows, _HOLDOUT_EXP_COLS, exp_rows)
    passed, score, method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is True
    assert method == "result_match_content"
    # 16 of 17 columns match on every row (only the constant metric_type label is
    # missing) → ~0.94, the fair score the root-cause study predicted.
    assert 0.93 < score < 0.95


def test_content_holdout_q2_is_a_hard_zero_under_legacy_ruler():
    # Same Q2 data under the legacy comparator → the false-negative the fix removes.
    exp_rows = [_q2_exp_row("W001", "spec_cvrm_1_total", "Atlantic Coast", "100.0", "1000.0", "10.0")]
    gen_rows = [_q2_gen_row("W001", "spec_cvrm_1_total", "Atlantic Coast", "100.0", "1000.0", "10.0")]
    legacy = _make_comparator_with_two_results(_Q2_GEN_COLS, gen_rows, _HOLDOUT_EXP_COLS, exp_rows)
    passed, score, method, _ = legacy.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False
    assert score == 0.0
    assert method == "result_column_mismatch"


def test_content_holdout_q1_still_fails_genuine_shape_miss():
    # Champion Q1: a single national total instead of the 6-row team decomposition.
    exp_rows = [
        _q1_exp_row("spec_cvrm_1a_pr", "111.0", "5000.0", "2.2"),
        _q1_exp_row("spec_cvrm_2a_pr", "222.0", "5000.0", "4.4"),
        _q1_exp_row("pc_cvrm_1a_pr", "333.0", "5000.0", "6.6"),
        _q1_exp_row("spec_cvrm_1_total", "444.0", "5000.0", "8.8"),
        _q1_exp_row("spec_cvrm_total", "555.0", "5000.0", "11.1"),
        _q1_exp_row("pc_cvrm_total", "666.0", "5000.0", "13.3"),
    ]
    gen_cols = ["market_name", "sub_market_name", "brand_name", "hierarchy_level", "region", "total_trx"]
    gen_rows = [["hyperkalemia market", "hyperkalemia submarket", "lokelma", "NATN", "ALL_REGIONS", "2922925.7"]]
    comparator = _content_comparator(gen_cols, gen_rows, _HOLDOUT_EXP_COLS, exp_rows)
    passed, score, method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False
    assert method == "result_content_mandatory_missing"
    assert score < 0.2  # the mandatory measure columns are absent


def _q1_exp_row(team, bmv, tmmv, mspct):
    return [
        "W001", "Current 13 Week", "hyperkalemia market", "hyperkalemia submarket", "lokelma",
        team, "NATN", "ALL_REGIONS", "TRX", bmv, tmmv, mspct, "P1", "NATN", "mkt", "team", "TRX",
    ]


# --- degenerate-shape guards (single-row / all-constant) ---------------------
# A 1-row or all-constant result makes every column look "constant", so role
# classification cannot tell an answer column from a label. The pass gate must
# then fall back to the overall score, never pass vacuously.

def test_content_single_row_exact_match_passes():
    comparator = _content_comparator(
        gen_cols=["region", "total_trx"],
        gen_rows=[["national", "2922925.7"]],
        exp_cols=["region", "total_trx"],
        exp_rows=[["national", "2922925.7"]],
    )
    passed, score, _method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is True
    assert score == 1.0


def test_content_single_row_wrong_value_fails():
    comparator = _content_comparator(
        gen_cols=["region", "total_trx"],
        gen_rows=[["national", "999.0"]],
        exp_cols=["region", "total_trx"],
        exp_rows=[["national", "2922925.7"]],
    )
    passed, score, _method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False
    assert score < 1.0


def test_content_all_constant_multirow_wrong_value_fails():
    comparator = _content_comparator(
        gen_cols=["metric_type", "total_trx"],
        gen_rows=[["TRX", "999"], ["TRX", "999"], ["TRX", "999"]],
        exp_cols=["metric_type", "total_trx"],
        exp_rows=[["TRX", "10"], ["TRX", "10"], ["TRX", "10"]],
    )
    passed, score, _method, _ = comparator.compare("SELECT * FROM g", "SELECT * FROM e")
    assert passed is False
    assert score < CONTENT_MATCH_THRESHOLD


def test_content_score_is_monotone_more_correct_rows_score_higher():
    exp_cols = ["region", "trx"]
    exp_rows = [["east", "10"], ["west", "20"], ["north", "30"]]
    one = _content_comparator(
        gen_cols=["region", "trx"],
        gen_rows=[["east", "10"], ["west", "999"], ["north", "999"]],
        exp_cols=exp_cols,
        exp_rows=exp_rows,
    ).compare("SELECT * FROM g", "SELECT * FROM e")
    two = _content_comparator(
        gen_cols=["region", "trx"],
        gen_rows=[["east", "10"], ["west", "20"], ["north", "999"]],
        exp_cols=exp_cols,
        exp_rows=exp_rows,
    ).compare("SELECT * FROM g", "SELECT * FROM e")
    assert two[1] > one[1]  # more-correct ⇒ strictly higher score (a usable slope)


def test_content_duplicate_dim_key_alignment_is_order_independent():
    # A repeated dimension key with several measure rows must score identically
    # regardless of row order on either side (no FIFO order-dependence).
    exp_cols = ["region", "amount"]
    exp_rows = [["east", 10], ["east", 20], ["west", 30]]
    gen_cols = ["region", "amount"]
    gen_a = [["east", 10], ["east", 20], ["west", 999]]
    gen_b = [["east", 20], ["east", 10], ["west", 999]]  # same multiset, reordered within 'east'

    _, score_a, _, _ = _compare_content_based(gen_cols, gen_a, exp_cols, exp_rows)
    _, score_b, _, _ = _compare_content_based(gen_cols, gen_b, exp_cols, exp_rows)

    assert score_a == score_b
    assert abs(score_a - 5.0 / 6.0) < 1e-9  # 5 of 6 cells match for both orderings


def test_content_zero_columns_is_mismatch():
    # Defensive: rows present but no columns must not pass vacuously at score 1.0.
    passed, score, method, _ = _compare_content_based([], [[]], [], [[]])

    assert passed is False
    assert score == 0.0
    assert method == "result_content_mismatch"


def test_content_padded_duplicate_rows_fail():
    # A row-doubled answer (correct rows duplicated) is wrong-cardinality and must not
    # pass as a content match, even though every expected row matches.
    exp_cols = ["region", "amount"]
    exp_rows = [["east", 10], ["west", 30]]
    gen_cols = ["region", "amount"]
    gen_rows = [["east", 10], ["east", 10], ["west", 30], ["west", 30]]  # row_ratio == 2.0

    passed, _, method, _ = _compare_content_based(gen_cols, gen_rows, exp_cols, exp_rows)

    assert passed is False
    assert method != "result_match_content"


def test_content_spurious_extra_row_fails():
    # Correct rows plus one spurious extra row (same key, wrong measure) is
    # wrong-cardinality and must fail regardless of which row sorts into the pair slot.
    exp_cols = ["region", "amount"]
    exp_rows = [["east", 10], ["west", 30]]
    gen_cols = ["region", "amount"]
    gen_rows = [["east", 10], ["west", 30], ["east", 999]]

    passed, _, method, _ = _compare_content_based(gen_cols, gen_rows, exp_cols, exp_rows)

    assert passed is False
    assert method != "result_match_content"


def test_content_dropped_constant_column_still_passes():
    # Dropping an all-constant label column must NOT fail the candidate (constant
    # columns are optional). Pins that the surplus-row cardinality penalty does not
    # bleed into the dropped-constant-column case, preserving the comparator's
    # core fairness goal.
    exp_cols = ["region", "amount", "metric_type"]
    exp_rows = [["east", 10, "NRx"], ["west", 30, "NRx"]]
    gen_cols = ["region", "amount"]
    gen_rows = [["east", 10], ["west", 30]]

    passed, _, method, _ = _compare_content_based(gen_cols, gen_rows, exp_cols, exp_rows)

    assert passed is True
    assert method == "result_match_content"
