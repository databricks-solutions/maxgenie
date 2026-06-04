"""Tests for expanded schema models and constraint enforcement."""

import json

from maxgenie.schemas import (
    GenieBenchmarkAnswer,
    GenieBenchmarkQuestion,
    GenieBenchmarks,
    GenieColumnConfig,
    GenieConfig,
    GenieDataSources,
    GenieExampleSQL,
    GenieExampleSQLParameter,
    GenieInstruction,
    GenieInstructions,
    GenieJoinSpecs,
    GenieMetricView,
    GenieSQLFunction,
    GenieSQLSnippet,
    GenieSQLSnippets,
    GenieSampleQuestion,
    GenieSpaceConfig,
    GenieTableConfig,
    GenieTableJoinSpec,
    _enforce_constraints,
)


def _full_config() -> GenieSpaceConfig:
    """Build a config that exercises all 10 sections."""
    return GenieSpaceConfig(
        version=2,
        title="Test Space",
        config=GenieConfig(
            sample_questions=[
                GenieSampleQuestion(id="b" * 32, question=["Q B"]),
                GenieSampleQuestion(id="a" * 32, question=["Q A"]),
            ]
        ),
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.schema.orders",
                    description=["Orders table"],
                    column_configs=[
                        GenieColumnConfig(
                            column_name="z_col",
                            description=["Z column"],
                            synonyms=["zee"],
                            enable_entity_matching=True,
                        ),
                        GenieColumnConfig(
                            column_name="a_col",
                            enable_format_assistance=True,
                        ),
                    ],
                ),
            ],
            metric_views=[
                GenieMetricView(
                    identifier="catalog.schema.monthly_metrics",
                    description=["Monthly rollup"],
                ),
            ],
        ),
        instructions=GenieInstructions(
            text_instructions=[
                GenieInstruction(id="i" * 32, content=["Use CURRENT_DATE"]),
            ],
            example_question_sqls=[
                GenieExampleSQL(
                    id="e" * 32,
                    question=["Top 10"],
                    sql=["SELECT * LIMIT 10"],
                    parameters=[
                        GenieExampleSQLParameter(name="limit", type_hint="INT"),
                    ],
                    usage_guidance=["Use for top-N queries"],
                ),
            ],
            join_specs=[
                GenieJoinSpecs(
                    id="j" * 32,
                    left=GenieTableJoinSpec(identifier="orders", alias="o"),
                    right=GenieTableJoinSpec(identifier="products", alias="p"),
                    sql=["o.product_id = p.id"],
                    comment=["FK relationship"],
                ),
            ],
            sql_functions=[
                GenieSQLFunction(id="f" * 32, identifier="catalog.schema.my_udf"),
            ],
            sql_snippets=GenieSQLSnippets(
                filters=[
                    GenieSQLSnippet(
                        id="sf" * 16,
                        sql=["status = 'active'"],
                        display_name="Active only",
                        synonyms=["current"],
                    ),
                ],
                expressions=[
                    GenieSQLSnippet(
                        id="se" * 16,
                        sql=["CASE WHEN amount > 100 THEN 'high' ELSE 'low' END"],
                        display_name="Amount tier",
                    ),
                ],
                measures=[
                    GenieSQLSnippet(
                        id="sm" * 16,
                        sql=["SUM(amount)"],
                        display_name="Total amount",
                    ),
                ],
            ),
        ),
        benchmarks=GenieBenchmarks(
            questions=[
                GenieBenchmarkQuestion(
                    id="bq" * 16,
                    question=["Total sales"],
                    answer=[GenieBenchmarkAnswer(format="SQL", content=["SELECT SUM(sales) FROM orders"])],
                ),
            ]
        ),
    )


def test_schema_roundtrip_all_sections():
    """Config -> serialized_space -> Config preserves all sections."""
    config = _full_config()
    serialized = config.to_serialized_space()
    restored = GenieSpaceConfig.from_serialized_space(serialized)

    assert restored.version == 2
    assert restored.data_sources is not None
    assert len(restored.data_sources.tables) == 1
    assert len(restored.data_sources.metric_views) == 1
    assert restored.instructions is not None
    assert len(restored.instructions.text_instructions) == 1
    assert len(restored.instructions.example_question_sqls) == 1
    assert len(restored.instructions.join_specs) == 1
    assert len(restored.instructions.sql_functions) == 1
    assert restored.instructions.sql_snippets is not None
    assert len(restored.instructions.sql_snippets.filters) == 1
    assert len(restored.instructions.sql_snippets.expressions) == 1
    assert len(restored.instructions.sql_snippets.measures) == 1
    assert restored.benchmarks is not None
    assert len(restored.benchmarks.questions) == 1


def test_column_config_new_fields():
    """Column config supports description, synonyms, exclude."""
    col = GenieColumnConfig(
        column_name="brand",
        description=["Brand name"],
        synonyms=["make", "manufacturer"],
        exclude=False,
        enable_entity_matching=True,
    )
    data = col.model_dump(exclude_none=True)
    assert data["synonyms"] == ["make", "manufacturer"]
    assert data["description"] == ["Brand name"]
    assert data["exclude"] is False


def test_example_sql_parameters():
    """Example SQL supports parameters and usage_guidance."""
    ex = GenieExampleSQL(
        question=["Sales by region"],
        sql=["SELECT region, SUM(sales) FROM orders GROUP BY region"],
        parameters=[
            GenieExampleSQLParameter(name="region", type_hint="STRING", description="Filter region"),
        ],
        usage_guidance=["Use for regional breakdowns"],
    )
    data = ex.model_dump(exclude_none=True)
    assert len(data["parameters"]) == 1
    assert data["parameters"][0]["name"] == "region"
    assert data["usage_guidance"] == ["Use for regional breakdowns"]


def test_example_sql_parameters_accept_native_serialized_shapes():
    """Genie exports parameter descriptions/defaults as list/dict payloads."""
    config = GenieSpaceConfig.from_serialized_space(
        {
            "version": 2,
            "instructions": {
                "example_question_sqls": [
                    {
                        "id": "e" * 32,
                        "question": ["Trips by pickup ZIP"],
                        "sql": ["SELECT * FROM samples.nyctaxi.trips WHERE pickup_zip = :pickup_zip"],
                        "parameters": [
                            {
                                "name": "pickup_zip",
                                "type_hint": "INTEGER",
                                "description": ["The pickup ZIP code."],
                                "default_value": {"values": ["10001"]},
                            }
                        ],
                    }
                ]
            },
        }
    )

    parameter = config.instructions.example_question_sqls[0].parameters[0]
    assert parameter.description == ["The pickup ZIP code."]
    assert parameter.default_value == {"values": ["10001"]}
    serialized_parameter = (
        config.to_serialized_space()["instructions"]["example_question_sqls"][0]["parameters"][0]
    )
    assert serialized_parameter["default_value"] == {"values": ["10001"]}


def test_enforce_constraints_sorts_by_id():
    """Constraint enforcement sorts ID-bearing arrays."""
    data = {
        "config": {
            "sample_questions": [
                {"id": "z" * 32, "question": ["Z"]},
                {"id": "a" * 32, "question": ["A"]},
            ]
        },
        "instructions": {
            "example_question_sqls": [
                {"id": "m" * 32, "question": ["M"], "sql": ["SELECT m"]},
                {"id": "b" * 32, "question": ["B"], "sql": ["SELECT b"]},
            ],
        },
    }
    _enforce_constraints(data)
    assert data["config"]["sample_questions"][0]["id"] == "a" * 32
    assert data["instructions"]["example_question_sqls"][0]["id"] == "b" * 32


def test_enforce_constraints_sorts_column_configs_by_name():
    data = {
        "data_sources": {
            "tables": [{
                "identifier": "t",
                "column_configs": [
                    {"column_name": "z_col"},
                    {"column_name": "a_col"},
                ],
            }]
        }
    }
    _enforce_constraints(data)
    cols = data["data_sources"]["tables"][0]["column_configs"]
    assert cols[0]["column_name"] == "a_col"
    assert cols[1]["column_name"] == "z_col"


def test_enforce_constraints_limits_text_instructions():
    data = {
        "instructions": {
            "text_instructions": [
                {"id": "1" * 32, "content": ["First"]},
                {"id": "2" * 32, "content": ["Second"]},
            ],
        }
    }
    _enforce_constraints(data)
    assert len(data["instructions"]["text_instructions"]) == 1


def test_enforce_constraints_removes_empty_snippets():
    data = {
        "instructions": {
            "sql_snippets": {
                "filters": [
                    {"id": "a" * 32, "sql": ["status = 'active'"], "display_name": "Active"},
                    {"id": "b" * 32, "sql": [], "display_name": "Empty"},
                ],
            }
        }
    }
    _enforce_constraints(data)
    assert len(data["instructions"]["sql_snippets"]["filters"]) == 1


def test_enforce_constraints_filters_nulls():
    data = {
        "config": {
            "sample_questions": [
                {"id": "a" * 32, "question": ["Q"]},
                None,
                {"id": "b" * 32, "question": ["Q2"]},
            ]
        }
    }
    _enforce_constraints(data)
    assert len(data["config"]["sample_questions"]) == 2
    assert all(q is not None for q in data["config"]["sample_questions"])
