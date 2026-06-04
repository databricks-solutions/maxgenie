"""Result-based SQL comparison by executing queries on a warehouse.

Instead of comparing SQL strings (which misses semantically equivalent queries),
this module executes both generated and expected SQL on the warehouse and
compares actual result sets.

"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)

MAX_ROWS = 1000
POLL_INTERVAL = 2.0
MAX_POLL_ATTEMPTS = 60
INLINE_WAIT_TIMEOUT = "50s"
REL_TOLERANCE = 0.001
ABS_TOLERANCE = 0.01
MISMATCH_THRESHOLD = 0.20

# Tolerances for alias-tolerant positional row comparison (independent of the
# per-cell tolerances used in the strict-name-match path above).
_ALIAS_TOLERANT_ABS_EPS = 1e-6
_ALIAS_TOLERANT_REL_EPS = 1e-6

# Content-based comparison (opt-in via content_match=True). A fair, value-first
# ruler: ignore label/annotation columns that are constant across all rows, match
# columns by VALUE (not name) so renames align, and score by a union cell-match
# rate so a near-miss reads as a slope (a gradient the optimizer can climb) instead
# of an all-or-nothing cliff. Pass requires every MANDATORY (varying measure/
# dimension) expected column to be matched and correct; CONSTANT label columns
# (e.g. metric_type echoed on every row) are optional. CONTENT_MATCH_THRESHOLD is
# set to 0.85 to reconcile the reported match_threshold (0.85) with the legacy cell
# path's effective 0.80 gate (1 - MISMATCH_THRESHOLD); the legacy path is unchanged
# so the pre-content leaderboard stays comparable.
CONTENT_COVERAGE_THRESHOLD = 0.85
CONTENT_MATCH_THRESHOLD = 0.85

# Matches :name placeholders but not ::cast syntax, abc:foo, or :123
_PLACEHOLDER_RE = re.compile(r"(?<![:\w]):([A-Za-z_][A-Za-z0-9_]*)")


def _extract_sql_named_placeholders(sql: str) -> set[str]:
    """Return the set of named placeholder identifiers found in *sql*.

    Matches ``:name`` style placeholders used by the Databricks Statement API
    while ignoring PostgreSQL ``::cast`` syntax, word-prefixed colons, and
    purely numeric tokens like ``:123``.
    """
    return set(_PLACEHOLDER_RE.findall(sql))


def _answer_key_sql_fingerprint(expected_sql: str) -> str:
    """Stable key for an expected (answer) SQL, tolerant to whitespace/case.

    The precomputed answer key is addressed by this fingerprint so the same
    expected SQL resolves to the same persisted expected result rows across the
    whole run, regardless of trivial formatting differences.
    """
    normalized = re.sub(r"\s+", " ", (expected_sql or "").strip()).lower().rstrip(";")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class ResultComparator:
    """Compare SQL results by executing both queries on the warehouse."""

    def __init__(
        self,
        wc: WorkspaceClient,
        warehouse_id: str,
        parameter_values: dict[str, Any] | None = None,
        answer_key_path: str | Path | None = None,
        content_match: bool = False,
    ):
        self.wc = wc
        self.warehouse_id = warehouse_id
        self._parameter_values = dict(parameter_values) if parameter_values else None
        # Opt-in fair, value-first comparison. Off by default so the legacy
        # byte-exact column-set behavior (and the pre-content leaderboard) is
        # unchanged until this ruler is proven and promoted to the default.
        self._content_match = bool(content_match)
        # Precomputed hidden answer key: expected (answer) result rows are executed
        # once and persisted, then every candidate is diffed against the same rows.
        # Keyed by a whitespace/case-tolerant fingerprint of the expected SQL so the
        # key is stable across the run and is also queryable by the proposer context
        # (visible questions only — holdout rows are never surfaced there).
        self._answer_key_path = Path(answer_key_path) if answer_key_path else None
        self._expected_cache: dict[str, dict[str, Any]] = {}
        self._load_answer_key()

    def cache_fingerprint(self) -> str:
        """Return a stable cache fingerprint for result-comparison semantics."""
        if self._parameter_values:
            param_hash = hashlib.sha1(
                ",".join(
                    f"{k}={self._parameter_values[k]}"
                    for k in sorted(self._parameter_values)
                ).encode()
            ).hexdigest()[:8]
        else:
            param_hash = "none"
        # Append the content marker only when enabled so the legacy fingerprint
        # (content off) is byte-identical to pre-content runs and their cached
        # scores stay reusable; the content ruler gets its own distinct key.
        content_marker = "|content=on" if self._content_match else ""
        return (
            "result-comparator"
            f"|wait={INLINE_WAIT_TIMEOUT}"
            f"|rows={MAX_ROWS}"
            f"|mismatch={MISMATCH_THRESHOLD}"
            f"|rel_tol={REL_TOLERANCE}"
            f"|abs_tol={ABS_TOLERANCE}"
            f"{content_marker}"
            f"|params={param_hash}"
        )

    def _load_answer_key(self) -> None:
        if self._answer_key_path is None or not self._answer_key_path.exists():
            return
        try:
            payload = json.loads(self._answer_key_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load answer key %s: %s", self._answer_key_path, exc)
            return
        expected = payload.get("expected") if isinstance(payload, dict) else None
        if isinstance(expected, dict):
            self._expected_cache = {
                str(key): {"cols": list(value.get("cols", [])), "rows": list(value.get("rows", []))}
                for key, value in expected.items()
                if isinstance(value, dict)
            }

    def _persist_answer_key(self) -> None:
        if self._answer_key_path is None:
            return
        try:
            self._answer_key_path.parent.mkdir(parents=True, exist_ok=True)
            self._answer_key_path.write_text(
                json.dumps({"version": 1, "expected": self._expected_cache}, default=str, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not persist answer key %s: %s", self._answer_key_path, exc)

    def _resolve_expected(self, expected_sql: str) -> tuple[list[str], list[list[Any]]]:
        """Return expected (answer) rows for *expected_sql*, executing once and caching.

        On a cache miss the expected SQL is executed on the warehouse, stored in the
        precomputed answer key, and persisted, so every later candidate is diffed
        against the same expected rows.
        """
        key = _answer_key_sql_fingerprint(expected_sql)
        cached = self._expected_cache.get(key)
        if cached is not None:
            return list(cached["cols"]), list(cached["rows"])
        exp_cols, exp_rows = self._execute_sql(expected_sql)
        self._expected_cache[key] = {"cols": exp_cols, "rows": exp_rows}
        self._persist_answer_key()
        return exp_cols, exp_rows

    def expected_rows_for_sql(
        self, expected_sql: str
    ) -> tuple[list[str], list[list[Any]]] | None:
        """Return already-computed expected rows for *expected_sql*, or ``None``.

        Read-only lookup for proposer context: it never executes SQL, so it only
        exposes rows that the answer key already holds. Callers must restrict this to
        visible questions; holdout expected rows must never reach proposer context.
        """
        cached = self._expected_cache.get(_answer_key_sql_fingerprint(expected_sql))
        if cached is None:
            return None
        return list(cached["cols"]), list(cached["rows"])

    def precompute_expected(self, expected_sqls: Iterable[str]) -> int:
        """Execute and persist expected rows for each distinct expected SQL once.

        Returns the number of newly executed (previously uncached) expected SQLs.
        Failures (e.g. an expected SQL that cannot execute) are skipped so one bad
        answer SQL never blocks precomputing the rest of the key.
        """
        computed = 0
        for expected_sql in expected_sqls:
            if not expected_sql or not str(expected_sql).strip():
                continue
            key = _answer_key_sql_fingerprint(expected_sql)
            if key in self._expected_cache:
                continue
            try:
                exp_cols, exp_rows = self._execute_sql(expected_sql)
            except Exception as exc:  # noqa: BLE001 - one bad answer SQL must not block the rest
                logger.warning("Answer-key precompute skipped one expected SQL: %s", exc)
                continue
            self._expected_cache[key] = {"cols": exp_cols, "rows": exp_rows}
            computed += 1
        if computed:
            self._persist_answer_key()
        return computed

    def _execute_sql(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        """Execute SQL on the warehouse and return (column_names, rows).

        Validates SQL is read-only (SELECT or WITH only) and limits to MAX_ROWS.
        """
        stripped = sql.strip().rstrip(";").strip()
        stripped = _strip_leading_sql_comments(stripped)
        first_word = stripped.split()[0].upper() if stripped else ""
        if first_word not in ("SELECT", "WITH"):
            raise ValueError(f"Only SELECT/WITH queries allowed, got: {first_word}")

        limited_sql = f"({stripped}) LIMIT {MAX_ROWS}" if "limit" not in stripped.lower() else stripped

        body: dict[str, Any] = {
            "warehouse_id": self.warehouse_id,
            "statement": limited_sql,
            # Databricks SQL statements only accept 0 or 5-50 second inline waits.
            "wait_timeout": INLINE_WAIT_TIMEOUT,
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
        }

        if self._parameter_values:
            placeholder_names = _extract_sql_named_placeholders(limited_sql)
            bound_names = placeholder_names & self._parameter_values.keys()
            if bound_names:
                body["parameters"] = [
                    {"name": n, "value": str(self._parameter_values[n]), "type": "STRING"}
                    for n in sorted(bound_names)
                ]

        response = self.wc.api_client.do(
            "POST",
            "/api/2.0/sql/statements",
            body=body,
        )

        status = response.get("status", {}).get("state", "")
        if status == "SUCCEEDED":
            return self._extract_results(response)

        # Poll if pending
        statement_id = response.get("statement_id")
        if not statement_id:
            raise RuntimeError(f"SQL execution failed: {response}")

        for _ in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL)
            poll = self.wc.api_client.do("GET", f"/api/2.0/sql/statements/{statement_id}")
            state = poll.get("status", {}).get("state", "")
            if state == "SUCCEEDED":
                return self._extract_results(poll)
            if state in ("FAILED", "CANCELED", "CLOSED"):
                error = poll.get("status", {}).get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"SQL execution {state}: {error}")

        raise RuntimeError("SQL execution timed out")

    @staticmethod
    def _extract_results(response: dict) -> tuple[list[str], list[list[Any]]]:
        """Extract column names and data rows from API response."""
        manifest = response.get("manifest", {})
        columns = [col["name"] for col in manifest.get("schema", {}).get("columns", [])]

        result = response.get("result", {})
        data_array = result.get("data_array", [])

        return columns, data_array

    def compare(
        self,
        generated_sql: str,
        expected_sql: str,
    ) -> tuple[bool, float, str, str]:
        """Execute both SQLs and compare results.

        Returns (passed, score, method, details). On execution failure this re-raises;
        the caller scores it as a result-based failure. There is no string fallback —
        a warehouse-backed run must never silently degrade to string comparison.
        """
        try:
            gen_cols, gen_rows = self._execute_sql(generated_sql)
            # Diff against the precomputed answer key (executed once and persisted);
            # falls back to executing the expected SQL on the first, uncached miss.
            exp_cols, exp_rows = self._resolve_expected(expected_sql)
        except Exception as exc:
            logger.warning("Result comparison failed; re-raising as a result-based error: %s", exc)
            raise

        # Fair, value-first content comparison (opt-in). Generalizes the legacy
        # exact-column-set path: ignores constant label columns, aligns renamed
        # columns by value, and returns a continuous union match-rate score.
        if self._content_match:
            return _compare_content_based(gen_cols, gen_rows, exp_cols, exp_rows)

        # Column comparison (case-insensitive, order-independent)
        gen_col_set = {c.lower() for c in gen_cols}
        exp_col_set = {c.lower() for c in exp_cols}

        if gen_col_set != exp_col_set:
            missing = exp_col_set - gen_col_set
            extra = gen_col_set - exp_col_set
            details = f"Column mismatch: missing={missing}, extra={extra}"
            # Alias-tolerant fallback: same column COUNT and same row VALUES but
            # different column names (e.g. total_trx vs trx vs sum_trx).
            if len(gen_cols) == len(exp_cols) and _rows_match_alias_tolerant(gen_rows, exp_rows):
                return True, 1.0, "result_match_alias_tolerant", details
            row_diff = _format_row_level_diff(exp_cols, exp_rows, gen_cols, gen_rows)
            return False, 0.0, "result_column_mismatch", f"{details} | {row_diff}"

        # Row count comparison
        gen_count = len(gen_rows)
        exp_count = len(exp_rows)
        if exp_count == 0 and gen_count == 0:
            return True, 1.0, "result_match", "Both queries returned 0 rows"

        if exp_count > 0:
            ratio = gen_count / exp_count if exp_count > 0 else float("inf")
            if ratio > 2.0 or ratio < 0.5:
                details = f"Row count mismatch: generated={gen_count}, expected={exp_count}"
                return False, 0.0, "result_row_mismatch", details

        # Cell-by-cell comparison
        # Align columns: reorder generated rows to match expected column order
        exp_col_lower = [c.lower() for c in exp_cols]
        gen_col_map = {c.lower(): i for i, c in enumerate(gen_cols)}

        col_indices = []
        for col in exp_col_lower:
            if col in gen_col_map:
                col_indices.append(gen_col_map[col])
            else:
                col_indices.append(None)

        total_cells = 0
        matched_cells = 0
        min_rows = min(gen_count, exp_count)

        for row_idx in range(min_rows):
            exp_row = exp_rows[row_idx]
            gen_row = gen_rows[row_idx]
            for col_idx, gen_col_idx in enumerate(col_indices):
                if gen_col_idx is None:
                    total_cells += 1
                    continue
                total_cells += 1
                exp_val = exp_row[col_idx] if col_idx < len(exp_row) else None
                gen_val = gen_row[gen_col_idx] if gen_col_idx < len(gen_row) else None
                if _values_match(gen_val, exp_val):
                    matched_cells += 1

        # Add unmatched rows as mismatches
        extra_rows = abs(gen_count - exp_count)
        total_cells += extra_rows * len(exp_cols)

        match_rate = matched_cells / total_cells if total_cells > 0 else 1.0

        if match_rate >= (1.0 - MISMATCH_THRESHOLD):
            details = f"Cell match rate: {match_rate:.3f} ({matched_cells}/{total_cells} cells)"
            return True, match_rate, "result_match", details
        else:
            details = f"Cell match rate: {match_rate:.3f} ({matched_cells}/{total_cells} cells, threshold={1.0 - MISMATCH_THRESHOLD})"
            row_diff = _format_row_level_diff(exp_cols, exp_rows, gen_cols, gen_rows)
            return False, match_rate, "result_mismatch", f"{details} | {row_diff}"

    def execute_readonly(
        self,
        sql: str,
    ) -> tuple[list[str], list[list[Any]]]:
        """Execute one read-only diagnostic query with the comparator safeguards."""
        return self._execute_sql(sql)


def _format_row_level_diff(
    exp_cols: list[str],
    exp_rows: list[list[Any]],
    gen_cols: list[str],
    gen_rows: list[list[Any]],
    *,
    max_rows: int = 5,
    max_cols: int = 8,
) -> str:
    """Compact row-level diff for a mismatch failure (expected vs generated).

    Shows the expected/generated column lists and a few aligned rows so the proposer can
    see the exact divergence at the cell level. Bounded in rows and per-row width. Used
    only on the failing column/cell-mismatch branches; the expected rows shown here are the
    same visible answer-key rows already surfaced to the proposer (no new disclosure), and
    a holdout question's diff never reaches proposer context.
    """

    def _fmt_row(row: list[Any]) -> str:
        cells = ["NULL" if value is None else str(value) for value in row[:max_cols]]
        suffix = ", ..." if len(row) > max_cols else ""
        return "(" + ", ".join(cells) + suffix + ")"

    parts = [
        f"exp_cols=[{', '.join(str(col) for col in exp_cols)}]",
        f"gen_cols=[{', '.join(str(col) for col in gen_cols)}]",
    ]
    total = max(len(exp_rows), len(gen_rows))
    shown = min(max_rows, total)
    for index in range(shown):
        exp_render = _fmt_row(exp_rows[index]) if index < len(exp_rows) else "(missing)"
        gen_render = _fmt_row(gen_rows[index]) if index < len(gen_rows) else "(missing)"
        parts.append(f"r{index} exp={exp_render} gen={gen_render}")
    if total > shown:
        parts.append(f"(+{total - shown} more rows)")
    return "row-level diff: " + "; ".join(parts)


def _rows_match_alias_tolerant(left_rows: list[list[Any]], right_rows: list[list[Any]]) -> bool:
    """Return True when left and right row sets contain the same values positionally.

    Both row lists are sorted by the stringified tuple of their values before
    comparison, making the check order-insensitive for rows.  Column order IS
    significant (position-based, name-insensitive).

    Numeric comparison tolerates absolute OR relative difference <= the module
    constants _ALIAS_TOLERANT_ABS_EPS / _ALIAS_TOLERANT_REL_EPS.
    Non-numeric comparison is stringified-equal after .strip().
    """
    if len(left_rows) != len(right_rows):
        return False

    def _sort_key(row: list[Any]) -> tuple:
        return tuple(str(v) for v in row)

    sorted_left = sorted(left_rows, key=_sort_key)
    sorted_right = sorted(right_rows, key=_sort_key)

    for l_row, r_row in zip(sorted_left, sorted_right):
        if len(l_row) != len(r_row):
            return False
        for l_val, r_val in zip(l_row, r_row):
            if not _alias_tolerant_values_match(l_val, r_val):
                return False
    return True


def _alias_tolerant_values_match(left: Any, right: Any) -> bool:
    """Cell-level comparison used by _rows_match_alias_tolerant."""
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        l_f = float(left)
        r_f = float(right)
        abs_diff = abs(l_f - r_f)
        if abs_diff <= _ALIAS_TOLERANT_ABS_EPS:
            return True
        if r_f != 0.0:
            return abs_diff / abs(r_f) <= _ALIAS_TOLERANT_REL_EPS
        return False
    except (ValueError, TypeError):
        pass
    return str(left).strip() == str(right).strip()


def _values_match(generated: Any, expected: Any) -> bool:
    """Compare two cell values with tolerance for floats and case-insensitive strings."""
    if generated is None and expected is None:
        return True
    if generated is None or expected is None:
        return False

    # Try numeric comparison with tolerance
    try:
        gen_f = float(generated)
        exp_f = float(expected)
        if exp_f == 0.0:
            return abs(gen_f) <= ABS_TOLERANCE
        return abs(gen_f - exp_f) / max(abs(exp_f), 1e-10) <= REL_TOLERANCE or abs(gen_f - exp_f) <= ABS_TOLERANCE
    except (ValueError, TypeError):
        pass

    # String comparison (case-insensitive, whitespace-normalized)
    gen_str = str(generated).strip().lower()
    exp_str = str(expected).strip().lower()
    return gen_str == exp_str


def _normalize_content_cell(value: Any) -> Any:
    """Normalize a cell for role classification and value-multiset column matching.

    Numbers are rounded to a stable grid and tagged ``num``; everything else is
    lower/trimmed and tagged ``str`` (so ``"100"`` and ``100`` collapse together and
    ``"Atlantic Coast"`` == ``"atlantic coast"``). This is used only to *classify*
    columns and *align* renamed columns by their value distribution — the final cell
    equality still goes through the tolerant :func:`_values_match`.
    """
    if value is None:
        return None
    try:
        return ("num", round(float(value), 6))
    except (ValueError, TypeError):
        return ("str", str(value).strip().lower())


def _content_column_values(rows: list[list[Any]], idx: int) -> list[Any]:
    return [row[idx] if idx < len(row) else None for row in rows]


def _content_column_role(values: list[Any]) -> str:
    """Classify a column from its values across all rows.

    ``constant`` — at most one distinct non-null value (a label/annotation/echo
    column such as ``metric_type`` repeated on every row); treated as OPTIONAL.
    ``measure`` — varies and every distinct value is numeric (an answer column).
    ``dimension`` — varies and is non-numeric (a categorical grouping column).
    ``measure`` and ``dimension`` columns are MANDATORY: a question is only answered
    if its varying answer/grouping columns are present and correct.
    """
    distinct = {_normalize_content_cell(v) for v in values}
    distinct.discard(None)
    if len(distinct) <= 1:
        return "constant"
    if all(isinstance(d, tuple) and d[0] == "num" for d in distinct):
        return "measure"
    return "dimension"


def _content_column_multiset(rows: list[list[Any]], idx: int) -> Counter:
    return Counter(_normalize_content_cell(v) for v in _content_column_values(rows, idx))


def _match_content_columns(
    exp_cols: list[str],
    exp_rows: list[list[Any]],
    gen_cols: list[str],
    gen_rows: list[list[Any]],
) -> dict[int, int | None]:
    """Map each expected column index to a generated column index (or ``None``).

    First match by name (case-insensitive); then, for the remainder, match by the
    column's value-multiset so a renamed column carrying the same data still aligns
    (e.g. expected ``time_grp_desc`` ↔ generated ``wk_grp_desc``). Greedy and
    one-to-one; a column matched by name is never re-used by the value pass. When two
    columns share an identical value-multiset the value binding is order-dependent,
    but identical multisets produce identical cell scores under any binding, so the
    final score is unaffected.
    """
    matches: dict[int, int | None] = {}
    used: set[int] = set()
    gen_by_name: dict[str, int] = {}
    for gi, gc in enumerate(gen_cols):
        gen_by_name.setdefault(gc.lower(), gi)

    for ei, ec in enumerate(exp_cols):
        gi = gen_by_name.get(ec.lower())
        if gi is not None and gi not in used:
            matches[ei] = gi
            used.add(gi)

    gen_multisets = {
        gi: _content_column_multiset(gen_rows, gi)
        for gi in range(len(gen_cols))
        if gi not in used
    }
    for ei in range(len(exp_cols)):
        if ei in matches:
            continue
        exp_ms = _content_column_multiset(exp_rows, ei)
        found: int | None = None
        for gi, gen_ms in gen_multisets.items():
            if gi in used:
                continue
            if gen_ms == exp_ms:
                found = gi
                break
        matches[ei] = found
        if found is not None:
            used.add(found)
    return matches


def _compare_content_based(
    gen_cols: list[str],
    gen_rows: list[list[Any]],
    exp_cols: list[str],
    exp_rows: list[list[Any]],
) -> tuple[bool, float, str, str]:
    """Fair, value-first comparison. Returns (passed, score, method, details).

    Pass rule: every MANDATORY (varying measure/dimension) expected column is matched
    (by name or value) AND the generated row cardinality is sane AND expected-row
    coverage and the mandatory cell-match rate clear their thresholds. CONSTANT label
    columns do not gate the pass. The score is a continuous union cell-match rate over
    all expected columns, so a near-miss is a usable gradient, not a hard zero.
    """
    gen_count, exp_count = len(gen_rows), len(exp_rows)
    if exp_count == 0 and gen_count == 0:
        return True, 1.0, "result_match_content", "Both queries returned 0 rows"
    if exp_count == 0:
        return False, 0.0, "result_content_mismatch", f"Expected 0 rows, generated {gen_count}"
    if not exp_cols:
        # Defensive: a real SELECT answer always has >=1 column. With rows but no
        # columns, total_cells would be 0 and the score would default to a vacuous 1.0.
        return False, 0.0, "result_content_mismatch", "Expected result has no columns"

    exp_roles = [
        _content_column_role(_content_column_values(exp_rows, i)) for i in range(len(exp_cols))
    ]
    matches = _match_content_columns(exp_cols, exp_rows, gen_cols, gen_rows)

    mandatory = [i for i in range(len(exp_cols)) if exp_roles[i] != "constant"]
    unmatched_mandatory = [i for i in mandatory if matches.get(i) is None]

    # Align rows order-independently. Prefer keying on matched DIMENSION columns
    # (the grouping keys that identify a row): align each expected row to the
    # generated row with the same key, then score its cells. This keeps the score
    # a monotone gradient even when some rows are wrong (a value-sort positional
    # alignment can mis-pair a wrong row with the wrong partner). Fall back to a
    # value-sort positional alignment only when there is no shared dimension key
    # (e.g. an all-measure or single-row result). Order-sensitive ranking/top-N
    # questions are out of scope here and a documented follow-up.
    matched_pairs = [(ei, gi) for ei, gi in matches.items() if gi is not None]
    dim_pairs = [
        (ei, matches[ei])
        for ei in range(len(exp_cols))
        if exp_roles[ei] == "dimension" and matches.get(ei) is not None
    ]

    def _row_key(row: list[Any], pairs: list[tuple[int, int]], *, exp_side: bool) -> tuple:
        parts = []
        for exp_idx, gen_idx in pairs:
            idx = exp_idx if exp_side else gen_idx
            value = row[idx] if idx < len(row) else None
            parts.append(str(_normalize_content_cell(value)))
        return tuple(parts)

    def _measure_signature(row: list[Any], *, exp_side: bool) -> tuple:
        # Normalized values of the matched NON-dimension columns, in expected-column
        # order. Within a dimension-key bucket every row shares the dimension values,
        # so sorting by this signature pairs rows order-independently.
        parts = []
        for ei in range(len(exp_cols)):
            gi = matches.get(ei)
            if gi is None or exp_roles[ei] == "dimension":
                continue
            idx = ei if exp_side else gi
            parts.append(str(_normalize_content_cell(row[idx] if idx < len(row) else None)))
        return tuple(parts)

    aligned_pairs: list[tuple[list[Any], list[Any] | None]] = []
    if dim_pairs:
        gen_by_key: dict[tuple, list[list[Any]]] = {}
        for gen_row in gen_rows:
            gen_by_key.setdefault(_row_key(gen_row, dim_pairs, exp_side=False), []).append(gen_row)
        exp_by_key: dict[tuple, list[list[Any]]] = {}
        for exp_row in exp_rows:
            exp_by_key.setdefault(_row_key(exp_row, dim_pairs, exp_side=True), []).append(exp_row)
        # Pair rows WITHIN each dimension-key bucket order-independently: sort both
        # sides by their matched-measure signature and pair positionally, so a repeated
        # dimension key with several measure rows scores the same regardless of row
        # order on either side. Rows whose key has no counterpart pair with None (no
        # credit); surplus generated rows are penalized by the union term below.
        for key in exp_by_key:
            exp_bucket = sorted(exp_by_key[key], key=lambda r: _measure_signature(r, exp_side=True))
            gen_bucket = sorted(
                gen_by_key.get(key, []), key=lambda r: _measure_signature(r, exp_side=False)
            )
            for idx, exp_row in enumerate(exp_bucket):
                gen_row = gen_bucket[idx] if idx < len(gen_bucket) else None
                aligned_pairs.append((exp_row, gen_row))
    else:
        exp_sorted = (
            sorted(exp_rows, key=lambda r: _row_key(r, matched_pairs, exp_side=True))
            if matched_pairs
            else list(exp_rows)
        )
        gen_sorted = (
            sorted(gen_rows, key=lambda r: _row_key(r, matched_pairs, exp_side=False))
            if matched_pairs
            else list(gen_rows)
        )
        for idx in range(len(exp_sorted)):
            gen_row = gen_sorted[idx] if idx < len(gen_sorted) else None
            aligned_pairs.append((exp_sorted[idx], gen_row))

    total_cells = 0
    matched_cells = 0
    mandatory_total = 0
    mandatory_matched = 0
    aligned_rows = 0
    for exp_row, gen_row in aligned_pairs:
        if gen_row is not None:
            aligned_rows += 1
        for ei in range(len(exp_cols)):
            gi = matches.get(ei)
            is_mandatory = exp_roles[ei] != "constant"
            total_cells += 1
            if is_mandatory:
                mandatory_total += 1
            if gi is None or gen_row is None:
                continue
            exp_val = exp_row[ei] if ei < len(exp_row) else None
            gen_val = gen_row[gi] if gi < len(gen_row) else None
            if _values_match(gen_val, exp_val):
                matched_cells += 1
                if is_mandatory:
                    mandatory_matched += 1

    # Union basis: extra generated rows beyond the expected cardinality are misses.
    # They count against BOTH the overall score and the mandatory-column rate, so a
    # candidate cannot pass by padding or duplicating rows (a wrong-cardinality answer)
    # while every expected row happens to match. This penalizes surplus ROWS without
    # re-penalizing dropped CONSTANT columns (which stay optional), because the surplus
    # only adds to mandatory_total when mandatory columns exist.
    if gen_count > exp_count:
        surplus_rows = gen_count - exp_count
        total_cells += surplus_rows * len(exp_cols)
        mandatory_total += surplus_rows * len(mandatory)

    score = matched_cells / total_cells if total_cells else 1.0
    mandatory_rate = mandatory_matched / mandatory_total if mandatory_total else 1.0
    coverage = aligned_rows / exp_count if exp_count else 1.0
    row_ratio = gen_count / exp_count if exp_count else float("inf")

    # Pass gate. When there ARE mandatory (varying measure/dimension) columns, gate
    # on their match rate so constant label columns stay optional. When there are
    # NONE — a single-row or all-constant expected result, where role classification
    # cannot distinguish an answer column from a label — gate on the overall score
    # instead, so a wrong scalar/constant answer cannot pass vacuously.
    quality_ok = (
        mandatory_rate >= CONTENT_MATCH_THRESHOLD
        if mandatory_total
        else score >= CONTENT_MATCH_THRESHOLD
    )
    passed = (
        not unmatched_mandatory
        and 0.5 <= row_ratio <= 2.0
        and coverage >= CONTENT_COVERAGE_THRESHOLD
        and quality_ok
    )

    if passed:
        method = "result_match_content"
    elif unmatched_mandatory:
        method = "result_content_mandatory_missing"
    else:
        method = "result_content_mismatch"

    missing_names = [exp_cols[i] for i in unmatched_mandatory]
    details = (
        f"content: score={score:.3f} mandatory_rate={mandatory_rate:.3f} "
        f"coverage={coverage:.3f} rows(gen/exp)={gen_count}/{exp_count} "
        f"unmatched_mandatory={missing_names}"
    )
    if not passed:
        details = f"{details} | {_format_row_level_diff(exp_cols, exp_rows, gen_cols, gen_rows)}"
    return passed, score, method, details


_SQL_EXECUTION_FAILURE_MARKERS = (
    "sqlstate",
    "sql_execution_exception",
    "unbound_sql_parameter",
    "unbound parameter",
    "parse_syntax_error",
)


def is_sql_execution_failure_text(text: str | None) -> bool:
    """True when an error message indicates the generated SQL failed to execute.

    Lets a Genie message failure caused by broken generated SQL be scored as a
    result-based failure (counted), while genuine infra/transient failures (auth,
    timeout, space gone) stay uncounted runtime errors.
    """
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _SQL_EXECUTION_FAILURE_MARKERS)


def classify_sql_error_text(text: str | None) -> str:
    """Classify a SQL execution error message into a result-based failure kind.

    Returns 'unbound_parameters', 'non_select', or 'other'.
    """
    lowered = (text or "").lower()
    if "unbound_sql_parameter" in lowered or "unbound parameter" in lowered:
        return "unbound_parameters"
    if "only select/with queries allowed" in lowered:
        return "non_select"
    return "other"


def classify_comparator_error(exc: BaseException) -> str:
    """Classify why the result comparator could not run.

    Returns one of:
      - 'unbound_parameters' — the expected SQL references Genie runtime params.
      - 'non_select'         — the expected SQL isn't a read-only SELECT/WITH.
      - 'other'              — anything else (network, timeout, permission, etc.)
    """
    return classify_sql_error_text(str(exc))


def _strip_leading_sql_comments(sql: str) -> str:
    """Remove leading SQL comments so read-only validation sees the first statement token."""
    remaining = sql.lstrip()
    while True:
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            if newline == -1:
                return ""
            remaining = remaining[newline + 1 :].lstrip()
            continue
        if remaining.startswith("/*"):
            end = remaining.find("*/")
            if end == -1:
                return ""
            remaining = remaining[end + 2 :].lstrip()
            continue
        return remaining
