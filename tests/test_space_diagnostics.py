"""Tests for Genie space diagnostics and linting."""

from maxgenie.schemas import (
    GenieColumnConfig,
    GenieConfig,
    GenieDataSources,
    GenieExampleSQL,
    GenieInstruction,
    GenieInstructions,
    GenieSampleQuestion,
    GenieSpaceConfig,
    GenieTableConfig,
)
from maxgenie.space_diagnostics import (
    analyze_space_config,
    filter_sql_snippet_candidates,
    join_spec_candidates,
    measure_sql_snippet_candidates,
    noise_column_candidates,
)


def test_diagnostics_detect_external_table_references():
    config = GenieSpaceConfig(
        description="Space only uses catalog.analytics.allowed_sales",
        config=GenieConfig(
            sample_questions=[
                GenieSampleQuestion(question=["How does catalog.analytics.other_table compare?"])
            ]
        ),
        data_sources=GenieDataSources(
            tables=[GenieTableConfig(identifier="catalog.analytics.allowed_sales")]
        ),
        instructions=GenieInstructions(
            text_instructions=[
                GenieInstruction(
                    content=[
                        "Always prefer catalog.analytics.other_table for historical backfills."
                    ]
                )
            ],
            example_question_sqls=[
                GenieExampleSQL(
                    question=["Show sales"],
                    sql=[
                        "SELECT * FROM catalog.analytics.allowed_sales "
                        "JOIN catalog.analytics.shadow_sales USING (day_id)"
                    ],
                )
            ],
        ),
    )

    diagnostics = analyze_space_config(config)
    references = {(item.reference, item.location) for item in diagnostics.external_table_references}

    assert ("catalog.analytics.other_table", "config.sample_questions[0]") in references
    assert ("catalog.analytics.other_table", "instructions.text_instructions[0]") in references
    assert ("catalog.analytics.shadow_sales", "instructions.example_question_sqls[0].sql") in references
    assert not any(reference == "catalog.analytics.allowed_sales" for reference, _ in references)


def test_diagnostics_report_matching_candidates_and_metadata_gaps():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    description=None,
                    column_configs=[
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="region_name"),
                        GenieColumnConfig(column_name="week_id"),
                        GenieColumnConfig(column_name="new_brand_prescriptions"),
                        GenieColumnConfig(
                            column_name="described_metric",
                            description=["Documented column"],
                        ),
                    ],
                )
            ]
        )
    )

    diagnostics = analyze_space_config(config)

    assert diagnostics.metadata_coverage.total_tables == 1
    assert diagnostics.metadata_coverage.described_tables == 0
    assert diagnostics.metadata_coverage.total_columns == 5
    assert diagnostics.metadata_coverage.described_columns == 1
    assert diagnostics.metadata_coverage.low_coverage_tables[0].identifier == "catalog.analytics.sales"

    entity_columns = {item.column_name for item in diagnostics.entity_matching_candidates}
    format_columns = {item.column_name for item in diagnostics.format_assistance_candidates}

    assert "brand_name" in entity_columns
    assert "region_name" in entity_columns
    assert "week_id" not in entity_columns
    assert "new_brand_prescriptions" not in entity_columns
    assert "week_id" in format_columns


def test_diagnostics_do_not_treat_calander_time_columns_as_entity_matching():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(
                            column_name="iqvia_calander_week",
                            synonyms=["calendar week", "iqvia week"],
                        ),
                        GenieColumnConfig(
                            column_name="iqvia_calander_month",
                            synonyms=["calendar month", "iqvia month"],
                        ),
                    ],
                )
            ]
        )
    )

    diagnostics = analyze_space_config(config)

    entity_columns = {item.column_name for item in diagnostics.entity_matching_candidates}
    format_columns = {item.column_name for item in diagnostics.format_assistance_candidates}

    assert "iqvia_calander_week" not in entity_columns
    assert "iqvia_calander_month" not in entity_columns
    assert "iqvia_calander_week" in format_columns
    assert "iqvia_calander_month" in format_columns


def test_noise_join_and_sql_snippet_candidates_are_generated():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="sales_id"),
                        GenieColumnConfig(column_name="customer_id"),
                        GenieColumnConfig(column_name="target_indicator"),
                        GenieColumnConfig(column_name="total_sales_amount"),
                    ],
                ),
                GenieTableConfig(
                    identifier="catalog.analytics.customers",
                    column_configs=[
                        GenieColumnConfig(column_name="customer_id"),
                        GenieColumnConfig(column_name="region_name"),
                    ],
                ),
            ]
        )
    )

    hidden_columns = {item.column_name for item in noise_column_candidates(config)}
    joins = join_spec_candidates(config)
    measures = measure_sql_snippet_candidates(config)
    filters = filter_sql_snippet_candidates(config)

    assert "sales_id" in hidden_columns
    assert joins
    assert {joins[0].left.identifier, joins[0].right.identifier} == {
        "catalog.analytics.customers",
        "catalog.analytics.sales",
    }
    assert "customer_id" in joins[0].sql[0]
    assert any(snippet.display_name == "Total Sales Amount" for snippet in measures)
    assert any(snippet.display_name == "Target Indicator" for snippet in filters)
