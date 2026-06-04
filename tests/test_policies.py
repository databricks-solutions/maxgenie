import re

from maxgenie.policies import (
    PolicyContext,
    centaur_prepass_policies,
    default_policies,
    policy_benchmark_only_examples,
    policy_centaur_expanded_entity_matching,
    policy_centaur_filter_snippet_audit,
    policy_centaur_instruction_audit,
    policy_comparison_enforcement,
    policy_entity_matching_candidates,
    policy_format_assistance_candidates,
    policy_hide_noise_columns,
    policy_join_spec_generation,
    policy_literal_examples,
    policy_merge_benchmark_examples,
    policy_sql_expression_filters,
    policy_sql_expression_measures,
)
from maxgenie.schemas import (
    GenieBenchmarkAnswer,
    GenieBenchmarkQuestion,
    GenieBenchmarks,
    GenieColumnConfig,
    GenieConfig,
    GenieDataSources,
    GenieExampleSQL,
    GenieInstruction,
    GenieInstructions,
    GenieSQLSnippet,
    GenieSQLSnippets,
    GenieSampleQuestion,
    GenieSpaceConfig,
    GenieTableConfig,
)
from maxgenie.space_diagnostics import (
    expanded_entity_matching_candidates,
    inconsistent_filter_snippet_ids,
    instruction_contradiction_findings,
)


def _base_config() -> GenieSpaceConfig:
    return GenieSpaceConfig(
        title="test-space",
        config=GenieConfig(sample_questions=[GenieSampleQuestion(question=["sample"])]),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact matching"])],
            example_question_sqls=[
                GenieExampleSQL(
                    id="11111111111111111111111111111111",
                    question=["show sales of brand"],
                    sql=["SELECT 1"],
                )
            ],
        ),
        benchmarks=GenieBenchmarks(
            questions=[
                GenieBenchmarkQuestion(
                    id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    question=["percentage difference for Lokelma current vs prior"],
                    answer=[GenieBenchmarkAnswer(format="SQL", content=["SELECT 2"])],
                )
            ]
        ),
    )


def test_policy_merge_benchmark_examples_adds_missing_benchmark_example():
    config = _base_config()
    updated = policy_merge_benchmark_examples(config, PolicyContext(original_config=config))

    ids = {item.id for item in updated.instructions.example_question_sqls}
    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in ids
    assert len(updated.instructions.example_question_sqls) == 2


def test_policy_benchmark_only_examples_replaces_examples():
    config = _base_config()
    updated = policy_benchmark_only_examples(config, PolicyContext(original_config=config))

    assert len(updated.instructions.example_question_sqls) == 1
    assert updated.instructions.example_question_sqls[0].id == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_policy_comparison_enforcement_adds_variants_with_valid_ids():
    config = _base_config()
    updated = policy_comparison_enforcement(config, PolicyContext(original_config=config))

    variants = updated.instructions.example_question_sqls
    assert len(variants) >= 2

    hex32 = re.compile(r"^[0-9a-f]{32}$")
    for item in variants:
        assert hex32.match(item.id)


def test_policy_high_priority_transforms_add_structural_assets():
    config = GenieSpaceConfig(
        title="test-space",
        config=GenieConfig(sample_questions=[GenieSampleQuestion(question=["sample"])]),
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="sales_id"),
                        GenieColumnConfig(column_name="customer_id"),
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="target_indicator"),
                        GenieColumnConfig(column_name="total_sales_amount"),
                        GenieColumnConfig(column_name="data_date"),
                    ],
                ),
                GenieTableConfig(
                    identifier="catalog.analytics.customers",
                    column_configs=[
                        GenieColumnConfig(column_name="customer_id"),
                    ],
                ),
            ]
        ),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact matching"])]
        ),
    )

    context = PolicyContext(original_config=config)

    hidden = policy_hide_noise_columns(config, context)
    assert hidden.data_sources.tables[0].column_configs[0].exclude is True

    matched = policy_entity_matching_candidates(config, context)
    assert matched.data_sources.tables[0].column_configs[2].enable_entity_matching is True

    formatted = policy_format_assistance_candidates(config, context)
    assert formatted.data_sources.tables[0].column_configs[5].enable_format_assistance is True

    joined = policy_join_spec_generation(config, context)
    assert joined.instructions.join_specs
    assert "customer_id" in joined.instructions.join_specs[0].sql[0]

    measured = policy_sql_expression_measures(config, context)
    assert measured.instructions.sql_snippets is not None
    assert measured.instructions.sql_snippets.measures
    assert any(
        snippet.display_name == "Total Sales Amount"
        for snippet in measured.instructions.sql_snippets.measures
    )

    filtered = policy_sql_expression_filters(config, context)
    assert filtered.instructions.sql_snippets is not None
    assert filtered.instructions.sql_snippets.filters
    assert any(
        snippet.display_name == "Target Indicator"
        for snippet in filtered.instructions.sql_snippets.filters
    )


def test_default_policies_contains_only_literal_examples():
    """Production default is a single evidence-based policy."""
    names = [policy.name for policy in default_policies()]
    assert names == ["literal_examples"]


def test_centaur_prepass_policies_apply_structural_diagnostics():
    config = GenieSpaceConfig(
        title="test-space",
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="territory_group"),
                        GenieColumnConfig(column_name="wk_grp"),
                        GenieColumnConfig(column_name="mth_grp"),
                    ],
                )
            ]
        ),
        instructions=GenieInstructions(
            text_instructions=[
                GenieInstruction(
                    content=[
                        "Use LIKE operator for brand matching.",
                        "Prefer case-insensitive exact matching for brand filters.",
                        "Map tb_wk_id and tb_mth_id carefully.",
                        "current_6_months_c6m should rely on mth_grp_desc.",
                    ]
                )
            ],
            sql_snippets=GenieSQLSnippets(
                filters=[
                    GenieSQLSnippet(
                        id="deadbeefdeadbeefdeadbeefdeadbeef",
                        display_name="Legacy Target",
                        sql=["tb_mth_id = 'C6M' AND target_indicator = 'Y'"],
                    )
                ]
            ),
        ),
    )
    context = PolicyContext(original_config=config)

    audited_filters = policy_centaur_filter_snippet_audit(config, context)
    assert audited_filters.instructions.sql_snippets is not None
    assert audited_filters.instructions.sql_snippets.filters == []

    audited_instructions = policy_centaur_instruction_audit(config, context)
    content = "\n".join(audited_instructions.instructions.text_instructions[0].content)
    assert "tb_wk_id" not in content
    assert "tb_mth_id" not in content
    assert "mth_grp_desc" not in content
    assert "Canonical Filter Policy" in content

    expanded = policy_centaur_expanded_entity_matching(config, context)
    sales_columns = {
        column.column_name: column
        for column in expanded.data_sources.tables[0].column_configs
    }
    assert sales_columns["territory_group"].enable_entity_matching is True
    assert sales_columns["territory_group"].enable_format_assistance is True

    prepass_names = [policy.name for policy in centaur_prepass_policies()]
    assert prepass_names == [
        "centaur_filter_snippet_audit",
        "centaur_instruction_audit",
        "centaur_expanded_entity_matching",
    ]

    findings = instruction_contradiction_findings(config)
    assert "like_vs_exact_matching" in findings
    assert "legacy_time_bucket_columns" in findings

    expanded_candidates = {item.column_name for item in expanded_entity_matching_candidates(config)}
    assert "territory_group" in expanded_candidates
    assert inconsistent_filter_snippet_ids(config) == ["deadbeefdeadbeefdeadbeefdeadbeef"]


def test_join_spec_generation_guards_compound_on_clause():
    """Proto guard: compound AND join conditions must be split into a list of individual predicates."""
    # Two tables sharing three join-like columns forces join_spec_candidates to emit
    # a compound ' AND '-joined sql string (e.g. "fsx1.col_a = fsx2.col_a AND ...").
    config = GenieSpaceConfig(
        title="test-space",
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.fact_sales_x1",
                    column_configs=[
                        GenieColumnConfig(column_name="timebucket_month_id"),
                        GenieColumnConfig(column_name="timebucket_week_id"),
                        GenieColumnConfig(column_name="data_date"),
                        GenieColumnConfig(column_name="sales_amount"),
                    ],
                ),
                GenieTableConfig(
                    identifier="catalog.analytics.fact_sales_x2",
                    column_configs=[
                        GenieColumnConfig(column_name="timebucket_month_id"),
                        GenieColumnConfig(column_name="timebucket_week_id"),
                        GenieColumnConfig(column_name="data_date"),
                        GenieColumnConfig(column_name="units_sold"),
                    ],
                ),
            ]
        ),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact matching"])]
        ),
    )
    context = PolicyContext(original_config=config)
    result = policy_join_spec_generation(config, context)

    assert result.instructions.join_specs
    for join_spec in result.instructions.join_specs:
        for sql_item in join_spec.sql:
            # No individual sql element should contain ' AND ' — each condition is its own entry.
            assert " AND " not in sql_item.upper(), (
                f"Compound AND clause found in join spec sql element: {sql_item!r}"
            )
    # There should be multiple sql items (one per join predicate) rather than one compound string.
    first_spec = result.instructions.join_specs[0]
    assert len(first_spec.sql) > 1, (
        "Expected compound join condition to be split into multiple sql list elements"
    )


def test_sql_expression_filters_guards_alias_field():
    """Proto guard: filter snippets must not carry an 'alias' field (not in Filter proto message)."""
    config = GenieSpaceConfig(
        title="test-space",
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="target_indicator"),
                        GenieColumnConfig(column_name="is_active"),
                    ],
                )
            ]
        ),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact matching"])]
        ),
    )
    context = PolicyContext(original_config=config)
    result = policy_sql_expression_filters(config, context)

    assert result.instructions.sql_snippets is not None
    filters = result.instructions.sql_snippets.filters or []
    assert filters, "Expected at least one filter snippet to be generated"
    for snippet in filters:
        assert snippet.alias is None, (
            f"Filter snippet {snippet.display_name!r} must not have alias set "
            f"(proto_guard: filter contains forbidden 'alias' field), got: {snippet.alias!r}"
        )


def test_policy_literal_examples_replaces_params_and_adds_instruction():
    """Policy converts :param placeholders to literals and adds no-unbound-params instruction."""
    config = GenieSpaceConfig(
        title="test-space",
        config=GenieConfig(sample_questions=[GenieSampleQuestion(question=["sample"])]),
        instructions=GenieInstructions(
            text_instructions=[GenieInstruction(content=["Use exact matching"])],
            example_question_sqls=[
                GenieExampleSQL(
                    id="parameterized_example",
                    question=["show sales for current period"],
                    sql=["SELECT * FROM sales WHERE wk_grp = :time_grp_value AND hierarchy_level = :hierarchy_level"],
                ),
                GenieExampleSQL(
                    id="literal_example",
                    question=["show brand totals"],
                    sql=["SELECT brand_name, SUM(sales) FROM sales GROUP BY brand_name"],
                ),
            ],
        ),
        benchmarks=GenieBenchmarks(
            questions=[
                GenieBenchmarkQuestion(
                    id="benchmark_with_literals",
                    question=["TRx for Lokelma in Current 5 weeks"],
                    answer=[GenieBenchmarkAnswer(format="SQL", content=["SELECT * FROM sales WHERE wk_grp = 'C5W' AND hierarchy_level = 'NATN'"])],
                )
            ]
        ),
    )

    updated = policy_literal_examples(config, PolicyContext(original_config=config))

    # Check that parameterized example was literalized
    examples = {ex.id: ex for ex in updated.instructions.example_question_sqls}
    param_example_sql = "\n".join(examples["parameterized_example"].sql)
    assert ":time_grp_value" not in param_example_sql
    assert ":hierarchy_level" not in param_example_sql
    assert "'C5W'" in param_example_sql
    assert "'NATN'" in param_example_sql

    # Check that literal example was unchanged
    literal_example_sql = "\n".join(examples["literal_example"].sql)
    assert "GROUP BY brand_name" in literal_example_sql

    # Check that instruction was added
    content = "\n".join(updated.instructions.text_instructions[0].content)
    assert "NEVER emit unbound :param placeholders" in content
