from maxgenie.artifact_patch_semantics import (
    question_overlap_terms,
    sql_delta_intents_from_visible_failure,
)


def test_question_overlap_terms_filters_generic_question_words():
    assert question_overlap_terms("Show current and prior NRX for Farxiga") == {
        "current",
        "farxiga",
        "nrx",
        "prior",
    }


def test_sql_delta_intents_from_visible_failure_labels_literalization():
    assert sql_delta_intents_from_visible_failure(
        {
            "likely_root_cause_family": "literal_or_parameter_mismatch",
            "comparison_method": "result_unbound_parameters_error",
            "missing_expected_tokens": ["'farxiga'"],
            "extra_generated_tokens": [":brand_name"],
        }
    ) == [
        "literalize_expected_value",
        "remove_bind_placeholder",
        "produce_executable_sql",
    ]


def test_sql_delta_intents_from_visible_failure_labels_column_repair():
    assert sql_delta_intents_from_visible_failure(
        {
            "likely_root_cause_family": "selected_metric_or_column_mismatch",
            "comparison_method": "result_column_mismatch",
            "missing_expected_tokens": ["prior_nrx"],
            "extra_generated_tokens": ["current_nrx"],
        }
    ) == [
        "restore_expected_result_columns",
        "remove_unrequested_result_columns",
    ]
