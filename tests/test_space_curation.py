"""Tests for deterministic Genie space curation."""

import pytest

from maxgenie.schemas import (
    GenieColumnConfig,
    GenieDataSources,
    GenieSpaceConfig,
    GenieTableConfig,
)
from maxgenie.space_curation import (
    CURATION_MODE_ENTITY_MATCHING,
    CURATION_MODE_FORMAT_ASSISTANCE,
    CURATION_MODE_HIDE_NOISE,
    CURATION_MODE_SAFE_DEFAULTS,
    curate_space_config,
)


def test_curate_space_config_enriches_metadata_and_reduces_noise():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.fact_sales_cardiovascular",
                    column_configs=[
                        GenieColumnConfig(column_name="az_brand_id"),
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="geographical_region_name"),
                        GenieColumnConfig(column_name="timebucket_week_id"),
                        GenieColumnConfig(column_name="new_prescriptions", enable_format_assistance=True),
                    ],
                )
            ]
        )
    )

    curated, summary = curate_space_config(config)
    table = curated.data_sources.tables[0]
    columns = {column.column_name: column for column in table.column_configs}

    # Table should receive a domain-agnostic description derived from its name and columns.
    assert table.description is not None
    assert len(table.description) == 1
    desc = table.description[0]
    assert "fact" in desc.lower() or "fact sales cardiovascular" in desc.lower()
    assert "brand" in desc.lower()  # label column hint from brand_name
    assert columns["az_brand_id"].exclude is True
    assert columns["az_brand_id"].enable_format_assistance is False
    assert columns["az_brand_id"].enable_entity_matching is False
    assert columns["brand_name"].enable_format_assistance is True
    assert columns["brand_name"].enable_entity_matching is True
    assert "brand" in (columns["brand_name"].synonyms or [])
    assert columns["geographical_region_name"].enable_entity_matching is True
    assert columns["timebucket_week_id"].enable_format_assistance is True
    assert columns["timebucket_week_id"].enable_entity_matching is False
    assert columns["new_prescriptions"].enable_format_assistance is False
    assert columns["new_prescriptions"].enable_entity_matching is False
    assert columns["new_prescriptions"].description == ["New prescriptions metric (NRx)."]

    assert summary.changed is True
    assert summary.table_descriptions_added == 1
    assert summary.column_descriptions_added == 5
    assert summary.columns_hidden == 1
    assert summary.entity_matching_enabled >= 2
    assert summary.synonyms_added >= 1


def test_curate_space_config_safe_defaults_skip_matching_and_hide_noise():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.fact_sales_cardiovascular",
                    column_configs=[
                        GenieColumnConfig(column_name="az_brand_id"),
                        GenieColumnConfig(column_name="brand_name"),
                    ],
                )
            ]
        )
    )

    curated, summary = curate_space_config(config, mode=CURATION_MODE_SAFE_DEFAULTS)
    columns = {column.column_name: column for column in curated.data_sources.tables[0].column_configs}

    assert summary.mode == CURATION_MODE_SAFE_DEFAULTS
    assert columns["az_brand_id"].exclude is None
    assert columns["brand_name"].enable_entity_matching is False
    assert summary.columns_hidden == 0
    assert summary.entity_matching_enabled == 0
    assert summary.synonyms_added >= 1
    assert summary.column_descriptions_added == 2


def test_curate_space_config_hide_noise_only_changes_excludes():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.fact_sales_cardiovascular",
                    column_configs=[
                        GenieColumnConfig(column_name="az_brand_id"),
                        GenieColumnConfig(column_name="brand_name"),
                    ],
                )
            ]
        )
    )

    curated, summary = curate_space_config(config, mode=CURATION_MODE_HIDE_NOISE)
    columns = {column.column_name: column for column in curated.data_sources.tables[0].column_configs}

    assert summary.mode == CURATION_MODE_HIDE_NOISE
    assert columns["az_brand_id"].exclude is True
    assert columns["brand_name"].description is None
    assert columns["brand_name"].synonyms is None
    assert summary.columns_hidden == 1
    assert summary.column_descriptions_added == 0
    assert summary.synonyms_added == 0


def test_curate_space_config_entity_matching_only_skips_format_assistance():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="week_id"),
                    ],
                )
            ]
        )
    )

    curated, summary = curate_space_config(config, mode=CURATION_MODE_ENTITY_MATCHING)
    columns = {column.column_name: column for column in curated.data_sources.tables[0].column_configs}

    assert summary.mode == CURATION_MODE_ENTITY_MATCHING
    assert columns["brand_name"].enable_entity_matching is True
    assert columns["week_id"].enable_format_assistance is False
    assert summary.entity_matching_enabled == 1
    assert summary.format_assistance_enabled == 0


def test_curate_space_config_format_assistance_only_skips_entity_matching():
    config = GenieSpaceConfig(
        data_sources=GenieDataSources(
            tables=[
                GenieTableConfig(
                    identifier="catalog.analytics.sales",
                    column_configs=[
                        GenieColumnConfig(column_name="brand_name"),
                        GenieColumnConfig(column_name="week_id"),
                    ],
                )
            ]
        )
    )

    curated, summary = curate_space_config(config, mode=CURATION_MODE_FORMAT_ASSISTANCE)
    columns = {column.column_name: column for column in curated.data_sources.tables[0].column_configs}

    assert summary.mode == CURATION_MODE_FORMAT_ASSISTANCE
    assert columns["brand_name"].enable_entity_matching is False
    assert columns["brand_name"].enable_format_assistance is True
    assert columns["week_id"].enable_format_assistance is True
    assert summary.entity_matching_enabled == 0
    assert summary.format_assistance_enabled == 2


def test_curate_space_config_rejects_unknown_mode():
    config = GenieSpaceConfig()

    with pytest.raises(ValueError, match="Unsupported curation mode"):
        curate_space_config(config, mode="unsupported")
