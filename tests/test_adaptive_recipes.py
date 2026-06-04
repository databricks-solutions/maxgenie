"""Tests for opt-in adaptive recipe-mode selection.

The default (adaptive off) must keep every iteration on the strict, comparable
``artifact_patch_v2`` path. With adaptive selection enabled, a fresh iteration
maps the dominant visible failure family to its best-fit narrow recipe mode, but
recovery iterations and ambiguous families still fall back to the default so the
apple-to-apple measurement backbone is never weakened silently.
"""

from __future__ import annotations

from maxgenie.serving_candidates import (
    adaptive_candidate_mode_for_family,
    dominant_visible_failure_family,
    serving_candidate_mode_for_iteration,
)


def _failure(family: str, qid: str = "Q") -> dict:
    return {"question_id": qid, "likely_root_cause_family": family}


# --------------------------------------------------------------------------- #
# dominant_visible_failure_family
# --------------------------------------------------------------------------- #
def test_dominant_family_none_for_empty():
    assert dominant_visible_failure_family(None) is None
    assert dominant_visible_failure_family([]) is None


def test_dominant_family_picks_most_common():
    failures = [
        _failure("literal_or_parameter_mismatch", "Q1"),
        _failure("literal_or_parameter_mismatch", "Q2"),
        _failure("time_filter_mismatch", "Q3"),
    ]
    assert dominant_visible_failure_family(failures) == "literal_or_parameter_mismatch"


def test_dominant_family_tie_breaks_alphabetically():
    # One each -> deterministic alphabetical tie-break ("time..." < "unbound...").
    failures = [
        _failure("time_filter_mismatch", "Q1"),
        _failure("table_or_join_path_mismatch", "Q2"),
    ]
    # "table_or_join_path_mismatch" sorts before "time_filter_mismatch".
    assert dominant_visible_failure_family(failures) == "table_or_join_path_mismatch"


def test_dominant_family_ignores_blank_and_nonmapping():
    failures = [
        {"question_id": "Q1", "likely_root_cause_family": ""},
        "not-a-mapping",
        _failure("missing_generated_sql", "Q2"),
    ]
    assert dominant_visible_failure_family(failures) == "missing_generated_sql"


# --------------------------------------------------------------------------- #
# adaptive_candidate_mode_for_family
# --------------------------------------------------------------------------- #
def test_mapped_families_return_narrow_modes():
    assert adaptive_candidate_mode_for_family("literal_or_parameter_mismatch") == "typed_template"
    assert adaptive_candidate_mode_for_family("percentage_or_comparison_metric") == "typed_template"
    assert adaptive_candidate_mode_for_family("time_filter_mismatch") == "schema_time_window"
    assert adaptive_candidate_mode_for_family("missing_generated_sql") == "deterministic_guidance"
    assert (
        adaptive_candidate_mode_for_family("table_or_join_path_mismatch")
        == "schema_metadata_aliases"
    )
    assert (
        adaptive_candidate_mode_for_family("selected_metric_or_column_mismatch")
        == "schema_metadata_aliases"
    )
    assert (
        adaptive_candidate_mode_for_family("aggregation_or_grouping_mismatch")
        == "schema_metadata_aliases"
    )


def test_unmapped_or_empty_family_returns_default():
    assert adaptive_candidate_mode_for_family(None) == "artifact_patch_v2"
    assert adaptive_candidate_mode_for_family("") == "artifact_patch_v2"
    # Ambiguous / multi-group families deliberately stay on the comparable default.
    assert adaptive_candidate_mode_for_family("runtime_or_generation_error") == "artifact_patch_v2"
    assert adaptive_candidate_mode_for_family("general_sql_mismatch") == "artifact_patch_v2"


def test_default_mode_override_is_respected():
    assert (
        adaptive_candidate_mode_for_family("general_sql_mismatch", default_mode="example_set")
        == "example_set"
    )


# --------------------------------------------------------------------------- #
# serving_candidate_mode_for_iteration -- default off preserves legacy behaviour
# --------------------------------------------------------------------------- #
def test_adaptive_off_always_uses_strict_default():
    # Even with a strong adaptive signal, adaptive_modes=False keeps the default.
    mode = serving_candidate_mode_for_iteration(
        1,
        adaptive_modes=False,
        dominant_failure_family="literal_or_parameter_mismatch",
    )
    assert mode == "artifact_patch_v2"


def test_adaptive_off_is_default_argument():
    # No adaptive kwargs at all -> unchanged legacy signature behaviour.
    assert serving_candidate_mode_for_iteration(1) == "artifact_patch_v2"


# --------------------------------------------------------------------------- #
# serving_candidate_mode_for_iteration -- adaptive on
# --------------------------------------------------------------------------- #
def test_adaptive_on_maps_fresh_iteration_to_family_mode():
    assert (
        serving_candidate_mode_for_iteration(
            1, adaptive_modes=True, dominant_failure_family="literal_or_parameter_mismatch"
        )
        == "typed_template"
    )
    assert (
        serving_candidate_mode_for_iteration(
            2, adaptive_modes=True, dominant_failure_family="time_filter_mismatch"
        )
        == "schema_time_window"
    )
    assert (
        serving_candidate_mode_for_iteration(
            3, adaptive_modes=True, dominant_failure_family="table_or_join_path_mismatch"
        )
        == "schema_metadata_aliases"
    )


def test_adaptive_on_unmapped_family_falls_back_to_default():
    assert (
        serving_candidate_mode_for_iteration(
            1, adaptive_modes=True, dominant_failure_family="general_sql_mismatch"
        )
        == "artifact_patch_v2"
    )
    assert (
        serving_candidate_mode_for_iteration(1, adaptive_modes=True, dominant_failure_family=None)
        == "artifact_patch_v2"
    )


def test_recovery_branches_keep_priority_over_adaptive():
    # A malformed prior response forces artifact_patch_v2 regardless of family.
    assert (
        serving_candidate_mode_for_iteration(
            1,
            adaptive_modes=True,
            dominant_failure_family="literal_or_parameter_mismatch",
            prior_invalid_responses=1,
        )
        == "artifact_patch_v2"
    )
    # A prior rejected candidate also pins recovery to artifact_patch_v2.
    assert (
        serving_candidate_mode_for_iteration(
            1,
            adaptive_modes=True,
            dominant_failure_family="time_filter_mismatch",
            prior_rejected_candidates=1,
        )
        == "artifact_patch_v2"
    )
