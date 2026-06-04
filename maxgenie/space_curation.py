"""Deterministic curation passes for Genie space assets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re

from maxgenie.schemas import (
    GenieColumnConfig,
    GenieMetricView,
    GenieSpaceConfig,
    GenieTableConfig,
)
from maxgenie.space_diagnostics import SpaceDiagnostics, analyze_space_config

CURATION_MODE_ALL = "all"
CURATION_MODE_SAFE_DEFAULTS = "safe_defaults"
CURATION_MODE_DESCRIPTIONS = "descriptions"
CURATION_MODE_SYNONYMS = "synonyms"
CURATION_MODE_MATCHING = "matching"
CURATION_MODE_ENTITY_MATCHING = "entity_matching"
CURATION_MODE_FORMAT_ASSISTANCE = "format_assistance"
CURATION_MODE_HIDE_NOISE = "hide_noise"
SUPPORTED_CURATION_MODES = frozenset(
    {
        CURATION_MODE_ALL,
        CURATION_MODE_SAFE_DEFAULTS,
        CURATION_MODE_DESCRIPTIONS,
        CURATION_MODE_SYNONYMS,
        CURATION_MODE_MATCHING,
        CURATION_MODE_ENTITY_MATCHING,
        CURATION_MODE_FORMAT_ASSISTANCE,
        CURATION_MODE_HIDE_NOISE,
    }
)

# Ordered sequence used by the benchmark-gated curation loop.
# Each mode is tried and benchmarked independently; "all" and "safe_defaults"
# are composites and are excluded here.
NARROW_CURATION_MODES: tuple[str, ...] = (
    CURATION_MODE_DESCRIPTIONS,
    CURATION_MODE_HIDE_NOISE,
    CURATION_MODE_SYNONYMS,
    CURATION_MODE_ENTITY_MATCHING,
    CURATION_MODE_FORMAT_ASSISTANCE,
    CURATION_MODE_MATCHING,
)

_ENTITY_HINTS = frozenset(
    {
        "brand",
        "businessunit",
        "channel",
        "district",
        "formulary",
        "geographical",
        "geography",
        "hierarchy",
        "manager",
        "market",
        "nation",
        "payer",
        "plan",
        "position",
        "region",
        "segment",
        "segmentation",
        "speciality",
        "specialty",
        "status",
        "sub",
        "target",
        "team",
        "territory",
        "tier",
    }
)
_FORMAT_HINTS = frozenset(
    {
        "calendar",
        "cal",
        "calander",
        "cycle",
        "date",
        "dt",
        "month",
        "mth",
        "period",
        "qtr",
        "quarter",
        "timebucket",
        "week",
        "wk",
        "year",
        "yr",
    }
)
_METRIC_HINTS = frozenset(
    {
        "amount",
        "count",
        "nbrx",
        "new",
        "nrx",
        "prescriptions",
        "sales",
        "share",
        "total",
        "trx",
        "units",
        "value",
        "volume",
    }
)
_COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "brand_name": ("brand", "product"),
    "market_name": ("market",),
    "sub_market_name": ("sub market", "submarket"),
    "market_speciality_name": ("speciality", "specialty"),
    "team_name": ("team",),
    "channel_description": ("channel",),
    "payer_type": ("payer", "payer type"),
    "geographical_region_name": ("region", "regional"),
    "geographical_territory_name": ("territory",),
    "geographical_district_name": ("district",),
    "geographical_nation_name": ("nation", "national"),
    "geographical_businessunit_name": ("business unit",),
    "formulary_status": ("formulary status",),
    "formulary_position": ("formulary position",),
    "formulary_tier": ("formulary tier",),
    "formulary_manager": ("formulary manager",),
    "formulary_influence": ("formulary influence",),
    "new_prescriptions": ("nrx", "new rx"),
    "new_brand_prescriptions": ("nbrx", "new to brand rx", "new brand rx"),
    "total_prescriptions": ("trx", "total rx"),
    "iqvia_calander_week": ("calendar week", "iqvia week"),
    "iqvia_calander_month": ("calendar month", "iqvia month"),
    "timebucket_week_id": ("week id",),
    "timebucket_month_id": ("month id",),
}
_EXACT_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "brand_name": "Brand name used for brand-level filtering, grouping, and comparisons.",
    "market_name": "Market name used for market-level filtering and grouping.",
    "sub_market_name": "Sub-market name used for narrower market segmentation.",
    "market_speciality_name": "Specialty segment used for specialty-level market analysis.",
    "team_name": "Sales team name used for team-level analysis.",
    "channel_description": "Human-readable sales channel used for filtering and grouping.",
    "payer_type": "Payer type used for payer-level filtering and analysis.",
    "payer_type_id": "Identifier for the payer type.",
    "plan_account_name": "Plan account name used for plan-level filtering and grouping.",
    "pharmacy_benefit_manager_name": "Pharmacy benefit manager name used for payer and access analysis.",
    "segmentation_name": "Segmentation category used to interpret segmentation values.",
    "segmentation_value_name": "Segmentation value used for filtered segmentation analysis.",
    "target_indicator": "Target flag that identifies targeted accounts or entities.",
    "target_view_description": "Description of the target view represented by the row.",
    "hierarchy_level": "Hierarchy level that indicates the geographic rollup represented by the row.",
    "formulary_status": "Formulary status used for access and coverage analysis.",
    "formulary_position": "Formulary position used for access and coverage analysis.",
    "formulary_tier": "Formulary tier used for access and coverage analysis.",
    "formulary_manager": "Formulary manager associated with the row.",
    "formulary_influence": "Formulary influence segment associated with the row.",
    "geographical_region_name": "Region name used for regional filtering and rollups.",
    "geographical_territory_name": "Territory name used for field-level geographic analysis.",
    "geographical_district_name": "District name used for geographic filtering and grouping.",
    "geographical_nation_name": "Nation name used for national rollups.",
    "geographical_businessunit_name": "Business unit name used for high-level commercial grouping.",
    "new_prescriptions": "New prescriptions metric (NRx).",
    "new_brand_prescriptions": "New-to-brand prescriptions metric (NBRx).",
    "total_prescriptions": "Total prescriptions metric (TRx).",
    "new_prescriptions_units": "New prescriptions volume expressed in units.",
    "new_brand_prescriptions_month": "Monthly new-to-brand prescriptions metric.",
    "total_prescriptions_units": "Total prescriptions volume expressed in units.",
    "data_date": "Data date representing the observation date for the aggregated record.",
    "data_dt": "Calendar date represented by the time bucket row.",
    "sls_dt": "Sales date represented by the time bucket row.",
    "iqvia_calander_week": "IQVIA calendar week label used for weekly grouping and filtering.",
    "iqvia_calander_month": "IQVIA calendar month label used for monthly grouping and filtering.",
    "iqvia_cal_wk": "IQVIA calendar week used for weekly grouping and filtering.",
    "iqvia_cal_mth": "IQVIA calendar month used for monthly grouping and filtering.",
    "timebucket_week_id": "Time bucket week identifier used for weekly relative-period logic.",
    "timebucket_month_id": "Time bucket month identifier used for monthly relative-period logic.",
    "tb_wk_id": "Time bucket week identifier used for shared time-dimension joins.",
    "tb_mth_id": "Time bucket month identifier used for shared time-dimension joins.",
    "tb_qtr_id": "Time bucket quarter identifier used for shared time-dimension joins.",
    "tb_yr_id": "Time bucket year identifier used for shared time-dimension joins.",
    "tb_wk_key": "Internal key for the shared time-bucket week.",
    "tb_mth_key": "Internal key for the shared time-bucket month.",
    "wk_grp": "Week-group bucket used for grouped weekly analysis.",
    "wk_grp_desc": "Description of the grouped weekly bucket.",
    "mth_grp": "Month-group bucket used for grouped monthly analysis.",
    "mth_grp_desc": "Description of the grouped monthly bucket.",
    "qtr_grp": "Quarter-group bucket used for grouped quarterly analysis.",
    "qtr_grp_desc": "Description of the grouped quarterly bucket.",
    "yr_grp": "Year-group bucket used for grouped yearly analysis.",
    "yr_grp_desc": "Description of the grouped yearly bucket.",
    "qtr": "Calendar quarter value used for quarterly grouping and filtering.",
    "yr": "Calendar year value used for yearly grouping and filtering.",
    "cycle_id": "Commercial cycle identifier used for cycle-based period analysis.",
    "cycl_id": "Commercial cycle identifier used in the shared time-bucket dimension.",
    "dataset": "Internal dataset marker used for data lineage and partitioning.",
}


@dataclass(frozen=True)
class CurationChange:
    """One deterministic curation action."""

    table_identifier: str
    column_name: str | None
    field: str
    new_value: str | bool | list[str] | None
    reason: str


@dataclass(frozen=True)
class SpaceCurationSummary:
    """Summary of deterministic curation changes."""

    mode: str
    changed: bool
    table_descriptions_added: int
    column_descriptions_added: int
    columns_hidden: int
    format_assistance_enabled: int
    format_assistance_disabled: int
    entity_matching_enabled: int
    entity_matching_disabled: int
    synonyms_added: int
    before_diagnostics: dict[str, object]
    after_diagnostics: dict[str, object]
    changes: list[CurationChange]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _column_tokens(column_name: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", column_name.lower())
        if token
    }


def _has_text(value: list[str] | None) -> bool:
    return bool(value and any(part.strip() for part in value))


def _looks_like_identifier(column_name: str) -> bool:
    lowered = column_name.lower()
    return lowered == "id" or lowered.endswith("_id") or lowered.endswith("_key")


def _is_time_like(column_name: str) -> bool:
    return bool(_column_tokens(column_name) & _FORMAT_HINTS)


def _is_metric_like(column_name: str) -> bool:
    tokens = _column_tokens(column_name)
    return bool(tokens & _METRIC_HINTS and not {"name", "status", "tier"} & tokens)


def _is_label_like(column_name: str) -> bool:
    lowered = column_name.lower()
    if lowered.endswith("_name") or lowered.endswith("_desc") or lowered.endswith("_description"):
        return True
    return bool(_column_tokens(column_name) & _ENTITY_HINTS and not _looks_like_identifier(column_name))


def _canonical_stem(column_name: str) -> str:
    lowered = column_name.lower()
    lowered = re.sub(r"^az_", "", lowered)
    lowered = re.sub(r"^(geographical|timebucket|tb)_", "", lowered)
    lowered = re.sub(r"_(id|key|name|desc|description)$", "", lowered)
    lowered = lowered.replace("_", " ").strip()
    return lowered


def _display_stems(columns: list[GenieColumnConfig]) -> set[str]:
    stems: set[str] = set()
    for column in columns:
        if _is_label_like(column.column_name):
            stems.add(_canonical_stem(column.column_name))
    return stems


def _should_hide_column(column_name: str, available_display_stems: set[str]) -> tuple[bool, str]:
    lowered = column_name.lower()
    stem = _canonical_stem(column_name)

    if lowered == "dataset":
        return True, "Internal dataset marker is not useful for natural-language analytics."
    if lowered.endswith("_key"):
        return True, "Internal key column adds noise without helping end-user questions."
    if _looks_like_identifier(column_name) and not _is_time_like(column_name) and stem in available_display_stems:
        return True, "Surrogate identifier is redundant because a readable business label is already present."
    return False, ""


def _desired_format_assistance(column_name: str, *, hidden: bool) -> bool:
    if hidden:
        return False
    if _is_metric_like(column_name):
        return False
    return _is_time_like(column_name) or _is_label_like(column_name)


def _desired_entity_matching(column_name: str, *, hidden: bool) -> bool:
    if hidden or _is_time_like(column_name) or _is_metric_like(column_name) or _looks_like_identifier(column_name):
        return False
    return _is_label_like(column_name)


def _suggest_synonyms(column_name: str) -> list[str]:
    return list(_COLUMN_SYNONYMS.get(column_name.lower(), ()))


def _humanize_tokens(column_name: str) -> str:
    parts = [part for part in column_name.lower().split("_") if part]
    if not parts:
        return column_name
    return " ".join(parts)


def _table_role(table_identifier: str) -> str:
    lowered = table_identifier.lower()
    if "time_bucket" in lowered or "calendar" in lowered or "time_bucket_dim" in lowered:
        return "time_dimension"
    if "segementation" in lowered or "segmentation" in lowered:
        return "segmented_fact"
    if "fact" in lowered or "sales" in lowered or "universe" in lowered:
        return "fact"
    return "dimension"


def _describe_table(
    table_identifier: str,
    columns: list[GenieColumnConfig],
) -> str:
    role = _table_role(table_identifier)
    table_name = _humanize_tokens(table_identifier.rsplit(".", 1)[-1])
    if role == "time_dimension":
        return (
            "Shared time-bucket dimension with calendar dates, weeks, months, quarters, "
            "years, and grouped period labels used for relative-period filtering."
        )
    metric_columns = [column.column_name for column in columns if _is_metric_like(column.column_name)]
    label_columns = [column.column_name for column in columns if _is_label_like(column.column_name)]
    if role in ("segmented_fact", "fact"):
        dimension_hint = ""
        if label_columns:
            top_labels = [_humanize_tokens(c.rsplit("_name", 1)[0] if c.endswith("_name") else c) for c in label_columns[:4]]
            dimension_hint = f" by {', '.join(top_labels)}"
        return f"Aggregated fact table for {table_name}{dimension_hint}."
    if metric_columns:
        return (
            f"Business table for {table_name} "
            "with reusable metrics and dimensions for Genie analysis."
        )
    return f"Business table for {table_name} used in the Genie space."


def _generic_column_description(column_name: str, table_identifier: str) -> str:
    humanized = _humanize_tokens(column_name)
    role = _table_role(table_identifier)
    lowered = column_name.lower()

    if lowered in _EXACT_COLUMN_DESCRIPTIONS:
        return _EXACT_COLUMN_DESCRIPTIONS[lowered]
    if role == "time_dimension" and _is_time_like(column_name):
        return f"Time-bucket attribute for {humanized.lower()} used in period filters and chronological grouping."
    if _is_metric_like(column_name):
        return f"Business metric for {humanized.lower()}."
    if lowered.endswith("_name"):
        return f"Human-readable {humanized[:-5].lower()} used for filtering and grouping."
    if lowered.endswith("_description") or lowered.endswith("_desc"):
        return f"Description of {humanized.replace(' description', '').replace(' desc', '').lower()}."
    if lowered.endswith("_id"):
        return f"Identifier for {humanized[:-3].lower()}."
    if lowered.endswith("_key"):
        return f"Internal key for {humanized[:-4].lower()}."
    if _is_label_like(column_name):
        return f"Business attribute for {humanized.lower()} used in filtering and grouping."
    return f"Business attribute for {humanized.lower()}."


def _append_missing_synonyms(existing: list[str] | None, suggested: list[str]) -> tuple[list[str] | None, int]:
    if not suggested:
        return existing, 0
    current = list(existing or [])
    normalized = {item.lower(): item for item in current}
    added = 0
    for synonym in suggested:
        if synonym.lower() in normalized:
            continue
        current.append(synonym)
        normalized[synonym.lower()] = synonym
        added += 1
    return (current or None), added


def _table_like_iterable(
    config: GenieSpaceConfig,
) -> list[GenieTableConfig | GenieMetricView]:
    items: list[GenieTableConfig | GenieMetricView] = []
    if config.data_sources:
        items.extend(config.data_sources.tables or [])
        items.extend(config.data_sources.metric_views or [])
    return items


def _mode_flags(mode: str) -> dict[str, bool]:
    if mode not in SUPPORTED_CURATION_MODES:
        supported = ", ".join(sorted(SUPPORTED_CURATION_MODES))
        raise ValueError(f"Unsupported curation mode `{mode}`. Supported values: {supported}.")
    if mode == CURATION_MODE_ALL:
        return {
            "descriptions": True,
            "synonyms": True,
            "entity_matching": True,
            "format_assistance": True,
            "hide_noise": True,
        }
    if mode == CURATION_MODE_SAFE_DEFAULTS:
        return {
            "descriptions": True,
            "synonyms": True,
            "entity_matching": False,
            "format_assistance": False,
            "hide_noise": False,
        }
    if mode == CURATION_MODE_MATCHING:
        return {
            "descriptions": False,
            "synonyms": False,
            "entity_matching": True,
            "format_assistance": True,
            "hide_noise": False,
        }
    return {
        "descriptions": mode == CURATION_MODE_DESCRIPTIONS,
        "synonyms": mode == CURATION_MODE_SYNONYMS,
        "entity_matching": mode == CURATION_MODE_ENTITY_MATCHING,
        "format_assistance": mode == CURATION_MODE_FORMAT_ASSISTANCE,
        "hide_noise": mode == CURATION_MODE_HIDE_NOISE,
    }


def curate_space_config(
    config: GenieSpaceConfig,
    *,
    mode: str = CURATION_MODE_ALL,
) -> tuple[GenieSpaceConfig, SpaceCurationSummary]:
    """Apply deterministic curation rules to a Genie space config."""
    flags = _mode_flags(mode)
    curated = deepcopy(config)
    before = analyze_space_config(config)
    changes: list[CurationChange] = []
    table_descriptions_added = 0
    column_descriptions_added = 0
    columns_hidden = 0
    format_assistance_enabled = 0
    format_assistance_disabled = 0
    entity_matching_enabled = 0
    entity_matching_disabled = 0
    synonyms_added = 0

    for table in _table_like_iterable(curated):
        columns = table.column_configs or []
        if flags["descriptions"] and not _has_text(table.description):
            description = _describe_table(table.identifier, columns)
            table.description = [description]
            table_descriptions_added += 1
            changes.append(
                CurationChange(
                    table_identifier=table.identifier,
                    column_name=None,
                    field="description",
                    new_value=description,
                    reason="Missing table description was filled with a deterministic business summary.",
                )
            )

        available_display_stems = _display_stems(columns)
        for column in columns:
            if flags["descriptions"] and not _has_text(column.description):
                description = _generic_column_description(column.column_name, table.identifier)
                column.description = [description]
                column_descriptions_added += 1
                changes.append(
                    CurationChange(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        field="description",
                        new_value=description,
                        reason="Missing column description was filled from deterministic naming heuristics.",
                    )
                )

            should_hide, hide_reason = _should_hide_column(column.column_name, available_display_stems)
            if flags["hide_noise"] and should_hide and column.exclude is not True:
                column.exclude = True
                columns_hidden += 1
                changes.append(
                    CurationChange(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        field="exclude",
                        new_value=True,
                        reason=hide_reason,
                    )
                )
            hidden = bool(column.exclude)

            target_format = _desired_format_assistance(column.column_name, hidden=hidden)
            if flags["format_assistance"] and column.enable_format_assistance != target_format:
                column.enable_format_assistance = target_format
                if target_format:
                    format_assistance_enabled += 1
                    reason = "Column looks like a time or business filter attribute that benefits from representative values."
                else:
                    format_assistance_disabled += 1
                    reason = "Column is hidden, metric-like, or too technical to benefit from representative values."
                changes.append(
                    CurationChange(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        field="enable_format_assistance",
                        new_value=target_format,
                        reason=reason,
                    )
                )

            target_entity = _desired_entity_matching(column.column_name, hidden=hidden)
            if flags["entity_matching"] and column.enable_entity_matching != target_entity:
                column.enable_entity_matching = target_entity
                if target_entity:
                    entity_matching_enabled += 1
                    reason = "Column looks like a high-value categorical business attribute that benefits from entity matching."
                else:
                    entity_matching_disabled += 1
                    reason = "Column is hidden, metric-like, time-like, or identifier-heavy, so entity matching would add noise."
                changes.append(
                    CurationChange(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        field="enable_entity_matching",
                        new_value=target_entity,
                        reason=reason,
                    )
                )

            merged_synonyms, added_synonyms = _append_missing_synonyms(
                column.synonyms,
                _suggest_synonyms(column.column_name),
            )
            if flags["synonyms"] and added_synonyms:
                column.synonyms = merged_synonyms
                synonyms_added += added_synonyms
                changes.append(
                    CurationChange(
                        table_identifier=table.identifier,
                        column_name=column.column_name,
                        field="synonyms",
                        new_value=merged_synonyms,
                        reason="High-value business column received deterministic synonyms for common user phrasing.",
                    )
                )

    after = analyze_space_config(curated)
    summary = SpaceCurationSummary(
        mode=mode,
        changed=bool(changes),
        table_descriptions_added=table_descriptions_added,
        column_descriptions_added=column_descriptions_added,
        columns_hidden=columns_hidden,
        format_assistance_enabled=format_assistance_enabled,
        format_assistance_disabled=format_assistance_disabled,
        entity_matching_enabled=entity_matching_enabled,
        entity_matching_disabled=entity_matching_disabled,
        synonyms_added=synonyms_added,
        before_diagnostics=before.to_dict(),
        after_diagnostics=after.to_dict(),
        changes=changes,
    )
    return curated, summary
