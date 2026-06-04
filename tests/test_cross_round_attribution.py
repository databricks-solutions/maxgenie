"""Tests for STEP 1e part (4): richer per-question cross-round edit attribution.

Maps each visible (train) question to the prior edits that flipped its pass/fail, so the
proposer can reuse fixes and avoid regressions. Train-visible only — holdout ids never
appear. Covers the pure inversion, the end-to-end structured-context field, and the
proposer-context rendering.
"""

from __future__ import annotations

import json
from pathlib import Path

from maxgenie.failure_context import (
    _per_question_edit_attribution,
    build_structured_failure_context,
)
from maxgenie.serving_candidates import _artifact_patch_cross_round_attribution_lines


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_per_question_edit_attribution_inverts_label_diffs():
    attempts = [
        {"phase": "candidate_evaluated", "label": "sweep.iter01", "outcome": "accepted"},
        {"phase": "candidate_evaluated", "label": "sweep.iter02", "outcome": "rejected"},
    ]
    train_diffs_by_label = {
        "baseline.train": {"fixed": [], "regressed": []},
        "sweep.iter01.train": {"fixed": ["q1", "q2"], "regressed": []},
        "sweep.iter02.train": {"fixed": [], "regressed": ["q1"]},
    }

    out = _per_question_edit_attribution(attempts, train_diffs_by_label)

    assert out["q1"] == [
        {"label": "sweep.iter01", "effect": "fixed", "accepted": True},
        {"label": "sweep.iter02", "effect": "regressed", "accepted": False},
    ]
    assert out["q2"] == [{"label": "sweep.iter01", "effect": "fixed", "accepted": True}]
    # baseline diffs are skipped (reference, not an edit).
    assert all(edit["label"] != "baseline" for edits in out.values() for edit in edits)


def test_per_question_edit_attribution_handles_missing_attempt_record():
    # A label with a train diff but no candidate_evaluated attempt -> accepted defaults False.
    out = _per_question_edit_attribution([], {"orphan.train": {"fixed": ["qX"]}})
    assert out["qX"] == [{"label": "orphan", "effect": "fixed", "accepted": False}]


def test_build_structured_failure_context_includes_per_question_attribution(tmp_path: Path):
    results_dir = tmp_path / "results"
    history_dir = tmp_path / "history"
    benchmarks_dir = tmp_path / "benchmarks"
    for directory in (results_dir, history_dir, benchmarks_dir):
        directory.mkdir()

    _write_json(
        results_dir / "latest_train.json",
        {
            "results": [
                {
                    "question_id": "q-fail",
                    "question": "Visible failing question",
                    "status": "failed",
                    "expected_sql": "SELECT 1 FROM visible_table",
                    "generated_sql": "SELECT 2 FROM visible_table",
                }
            ]
        },
    )
    (history_dir / "iteration_log.jsonl").write_text(
        json.dumps({"phase": "candidate_evaluated", "label": "sweep.iter01", "outcome": "accepted"})
        + "\n"
        + json.dumps(
            {
                "phase": "benchmark_train",
                "label": "sweep.iter01.train",
                "diff": {"fixed": ["q-fail"], "regressed": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        benchmarks_dir / "test_set.json",
        [{"id": "hidden-1", "question": "Secret holdout", "expected_sql": "SELECT secret FROM hidden_table"}],
    )

    result = build_structured_failure_context(
        results_dir=results_dir,
        history_dir=history_dir,
        benchmarks_dir=benchmarks_dir,
    )

    assert result["per_question_attribution"]["q-fail"] == [
        {"label": "sweep.iter01", "effect": "fixed", "accepted": True}
    ]
    # Holdout exclusion still holds with attribution attached.
    serialized = json.dumps(result)
    assert "hidden-1" not in serialized
    assert "Secret holdout" not in serialized
    assert "hidden_table" not in serialized


def test_cross_round_attribution_lines_render_when_present():
    ctx = {
        "per_question_attribution": {
            "q1": [
                {"label": "sweep.iter01", "effect": "fixed", "accepted": True},
                {"label": "sweep.iter02", "effect": "regressed", "accepted": False},
            ]
        }
    }

    text = "\n".join(_artifact_patch_cross_round_attribution_lines(ctx))

    assert "Cross-Round Edit Attribution" in text
    assert "`q1`:" in text
    assert "fixed by `sweep.iter01` (accepted)" in text
    assert "regressed by `sweep.iter02` (rejected)" in text


def test_cross_round_attribution_lines_empty_without_data():
    assert _artifact_patch_cross_round_attribution_lines({}) == []
    assert _artifact_patch_cross_round_attribution_lines({"per_question_attribution": {}}) == []
    assert _artifact_patch_cross_round_attribution_lines(None) == []
