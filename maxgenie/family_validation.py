"""Visible family-aware validation helpers.

These helpers intentionally operate only on visible train/validation records.
Hidden holdout/test payloads are excluded before classification, guard
construction, or reporting.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


__all__ = [
    "FamilyClassification",
    "ProxyGuardRequirement",
    "ProxyGuardSet",
    "ProxyPressureReport",
    "build_proxy_guard_requirements",
    "classify_benchmark_family",
    "compute_visible_proxy_pressure",
]


_HIDDEN_SOURCES = frozenset({"hidden", "holdout", "test", "final_test", "blind_holdout"})
_VISIBLE_SOURCES = frozenset({"train", "validation", "dev", "visible_validation"})

_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "runtime_or_generation_error",
        (
            "error",
            "exception",
            "failed to generate",
            "missing generated",
            "no generated",
            "timeout",
        ),
    ),
    (
        "period_comparison",
        (
            "% difference",
            "comparison",
            "compare",
            "difference",
            "growth",
            "percent",
            "percentage",
            "prior",
            "previous",
            "versus",
            "vs",
        ),
    ),
    (
        "time_window",
        (
            "13 week",
            "6 month",
            "current",
            "date",
            "month",
            "period",
            "quarter",
            "time bucket",
            "week",
            "ytd",
        ),
    ),
    (
        "metric_selection",
        (
            "nbrx",
            "new brand",
            "nrx",
            "rx",
            "sales",
            "trx",
            "volume",
        ),
    ),
    (
        "geography_breakdown",
        (
            "area",
            "business unit",
            "businessunit",
            "geographical",
            "geography",
            "national",
            "region",
            "territory",
        ),
    ),
    (
        "segment_or_specialty",
        (
            "hcp",
            "market specialty",
            "payer",
            "plan",
            "segment",
            "speciality",
            "specialty",
        ),
    ),
    (
        "market_product_filter",
        (
            "brand",
            "farxiga",
            "hf+ckd",
            "hyperkalemia",
            "lokelma",
            "market",
            "product",
            "sglt2",
        ),
    ),
    (
        "aggregation_or_grouping",
        (
            "aggregation",
            "average",
            "breakdown",
            "by ",
            "group",
            "list",
            "rank",
            "sum",
            "top",
        ),
    ),
    (
        "table_or_join_path",
        (
            "fact",
            "join",
            "path",
            "relationship",
            "table",
        ),
    ),
)

_ISSUE_CLASS_FAMILY_HINTS = {
    "aggregation": "aggregation_or_grouping",
    "column": "metric_selection",
    "comparison": "period_comparison",
    "generation": "runtime_or_generation_error",
    "group": "aggregation_or_grouping",
    "incomplete": "metric_selection",
    "join": "table_or_join_path",
    "metric": "metric_selection",
    "missing": "runtime_or_generation_error",
    "period": "time_window",
    "region": "geography_breakdown",
    "runtime": "runtime_or_generation_error",
    "segment": "segment_or_specialty",
    "specialty": "segment_or_specialty",
    "table": "table_or_join_path",
    "time": "time_window",
}

_PRIMARY_PRIORITY = (
    "runtime_or_generation_error",
    "period_comparison",
    "time_window",
    "table_or_join_path",
    "aggregation_or_grouping",
    "geography_breakdown",
    "segment_or_specialty",
    "metric_selection",
    "market_product_filter",
    "general_sql",
)


@dataclass(frozen=True)
class FamilyClassification:
    """Broad visible benchmark family classification."""

    question_id: str
    primary_family: str
    families: tuple[str, ...]
    signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyGuardRequirement:
    """One visible train/validation guard required to preserve a family."""

    question_id: str
    source: str
    primary_family: str
    families: tuple[str, ...]
    reason: str
    question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProxyGuardSet:
    """Visible-only proxy guard requirements for train/validation families."""

    guards: tuple[ProxyGuardRequirement, ...]
    required_guard_ids: tuple[str, ...]
    required_families: tuple[str, ...]
    guard_ids_by_family: dict[str, tuple[str, ...]]
    ignored_hidden_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "guards": [guard.to_dict() for guard in self.guards],
            "required_guard_ids": list(self.required_guard_ids),
            "required_families": list(self.required_families),
            "guard_ids_by_family": {
                family: list(ids) for family, ids in self.guard_ids_by_family.items()
            },
            "ignored_hidden_count": self.ignored_hidden_count,
        }


@dataclass(frozen=True)
class ProxyPressureReport:
    """Visible proxy pressure produced by a candidate evaluation."""

    train_delta: float | None
    validation_delta: float | None
    visible_scores_improved: bool
    pressure_score: float
    guard_retention_rate: float
    family_retention_rate: float
    required_guard_count: int
    retained_guard_count: int
    required_family_count: int
    retained_family_count: int
    regressed_guard_ids: tuple[str, ...]
    degraded_families: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_benchmark_family(record: Mapping[str, Any]) -> FamilyClassification:
    """Classify a visible benchmark/failure into broad families.

    The classifier uses question text, question IDs, issue_class, and token-hint
    fields only. SQL payloads are intentionally ignored.
    """
    question_id = _question_id(record)
    issue_class = _clean_text(record.get("issue_class") or record.get("root_cause") or "")
    question = _clean_text(record.get("question") or record.get("question_text") or "")
    token_hints = _token_hint_text(record)
    searchable = " ".join(part for part in (question_id.lower(), issue_class, question, token_hints) if part)

    families: set[str] = set()
    signals: list[str] = []

    for issue_hint, family in _ISSUE_CLASS_FAMILY_HINTS.items():
        if issue_hint in issue_class:
            families.add(family)
            signals.append(f"issue_class:{issue_hint}")

    for family, patterns in _FAMILY_PATTERNS:
        for pattern in patterns:
            if pattern in searchable:
                families.add(family)
                signals.append(f"{family}:{pattern.strip()}")
                break

    if not families:
        families.add("general_sql")
        signals.append("fallback:general_sql")

    ordered_families = tuple(family for family in _PRIMARY_PRIORITY if family in families)
    primary_family = ordered_families[0]
    return FamilyClassification(
        question_id=question_id,
        primary_family=primary_family,
        families=ordered_families,
        signals=tuple(dict.fromkeys(signals)),
    )


def build_proxy_guard_requirements(
    records: Sequence[Mapping[str, Any]],
    *,
    max_guards_per_family: int = 2,
    include_question_text: bool = True,
) -> ProxyGuardSet:
    """Build visible train/validation proxy pass guards.

    Only passed train/validation records can become requirements. Test/holdout
    records are counted as ignored and never included in the returned payload.
    """
    if max_guards_per_family < 1:
        raise ValueError("max_guards_per_family must be at least 1")

    guards: list[ProxyGuardRequirement] = []
    guard_ids_by_family: dict[str, list[str]] = {}
    ignored_hidden_count = 0
    seen_ids: set[str] = set()

    for record in records:
        source = _visible_source(record)
        if source is None:
            ignored_hidden_count += int(_is_hidden_source(record))
            continue
        if not _is_passed(record):
            continue

        classification = classify_benchmark_family(record)
        question_id = classification.question_id
        if not question_id or question_id in seen_ids:
            continue

        has_capacity = any(
            len(guard_ids_by_family.get(family, [])) < max_guards_per_family
            for family in classification.families
        )
        if not has_capacity:
            continue

        seen_ids.add(question_id)
        guard = ProxyGuardRequirement(
            question_id=question_id,
            source=source,
            primary_family=classification.primary_family,
            families=classification.families,
            reason=f"visible_{source}_pass_proxy_guard",
            question=_safe_question(record) if include_question_text else None,
        )
        guards.append(guard)
        for family in classification.families:
            family_ids = guard_ids_by_family.setdefault(family, [])
            if len(family_ids) < max_guards_per_family:
                family_ids.append(question_id)

    sorted_family_ids = {
        family: tuple(ids)
        for family, ids in sorted(guard_ids_by_family.items(), key=lambda item: item[0])
    }
    required_guard_ids = tuple(guard.question_id for guard in guards)
    return ProxyGuardSet(
        guards=tuple(guards),
        required_guard_ids=required_guard_ids,
        required_families=tuple(sorted_family_ids),
        guard_ids_by_family=sorted_family_ids,
        ignored_hidden_count=ignored_hidden_count,
    )


def compute_visible_proxy_pressure(
    before_records: Sequence[Mapping[str, Any]],
    after_records: Sequence[Mapping[str, Any]],
    *,
    before_train_rate: float | None = None,
    after_train_rate: float | None = None,
    before_validation_rate: float | None = None,
    after_validation_rate: float | None = None,
    proxy_guards: ProxyGuardSet | None = None,
) -> ProxyPressureReport:
    """Compute visible proxy pressure after a candidate.

    Pressure increases when required visible proxy guards or family coverage
    degrade. Warnings become explicit when train/validation improved while those
    visible proxies degraded.
    """
    requirements = proxy_guards or build_proxy_guard_requirements(before_records)
    after_visible = [record for record in after_records if _visible_source(record) is not None]
    after_passed_ids = {_question_id(record) for record in after_visible if _is_passed(record)}
    after_passed_families = _passed_families(after_visible)

    regressed_guard_ids = tuple(
        guard_id for guard_id in requirements.required_guard_ids if guard_id not in after_passed_ids
    )
    degraded_families = tuple(
        family for family in requirements.required_families if family not in after_passed_families
    )

    required_guard_count = len(requirements.required_guard_ids)
    retained_guard_count = required_guard_count - len(regressed_guard_ids)
    required_family_count = len(requirements.required_families)
    retained_family_count = required_family_count - len(degraded_families)
    guard_retention_rate = _rate(retained_guard_count, required_guard_count)
    family_retention_rate = _rate(retained_family_count, required_family_count)
    train_delta = _delta(before_train_rate, after_train_rate)
    validation_delta = _delta(before_validation_rate, after_validation_rate)
    visible_scores_improved = _visible_scores_improved(train_delta, validation_delta)

    pressure_score = round(((1.0 - guard_retention_rate) * 60.0) + ((1.0 - family_retention_rate) * 40.0), 2)
    warnings = _pressure_warnings(
        visible_scores_improved=visible_scores_improved,
        regressed_guard_ids=regressed_guard_ids,
        degraded_families=degraded_families,
        required_guard_count=required_guard_count,
    )

    return ProxyPressureReport(
        train_delta=train_delta,
        validation_delta=validation_delta,
        visible_scores_improved=visible_scores_improved,
        pressure_score=pressure_score,
        guard_retention_rate=round(guard_retention_rate * 100.0, 2),
        family_retention_rate=round(family_retention_rate * 100.0, 2),
        required_guard_count=required_guard_count,
        retained_guard_count=retained_guard_count,
        required_family_count=required_family_count,
        retained_family_count=retained_family_count,
        regressed_guard_ids=regressed_guard_ids,
        degraded_families=degraded_families,
        warnings=warnings,
    )


def _pressure_warnings(
    *,
    visible_scores_improved: bool,
    regressed_guard_ids: tuple[str, ...],
    degraded_families: tuple[str, ...],
    required_guard_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if required_guard_count == 0:
        warnings.append("no_visible_proxy_guards_available")
    if regressed_guard_ids:
        warnings.append("proxy_guard_retention_degraded:" + ",".join(regressed_guard_ids[:5]))
    if degraded_families:
        warnings.append("family_coverage_degraded:" + ",".join(degraded_families[:5]))
    if visible_scores_improved and (regressed_guard_ids or degraded_families):
        warnings.append("visible_score_lift_with_proxy_degradation")
    return tuple(warnings)


def _passed_families(records: Sequence[Mapping[str, Any]]) -> set[str]:
    families: set[str] = set()
    for record in records:
        if not _is_passed(record):
            continue
        families.update(classify_benchmark_family(record).families)
    return families


def _visible_scores_improved(train_delta: float | None, validation_delta: float | None) -> bool:
    deltas = [delta for delta in (train_delta, validation_delta) if delta is not None]
    return bool(deltas) and any(delta > 0 for delta in deltas) and all(delta >= 0 for delta in deltas)


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(float(after) - float(before), 2)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def _visible_source(record: Mapping[str, Any]) -> str | None:
    source = _source(record)
    if source in _HIDDEN_SOURCES:
        return None
    if source in _VISIBLE_SOURCES:
        return "validation" if source in {"dev", "visible_validation"} else source
    if source == "":
        return "train"
    return None


def _is_hidden_source(record: Mapping[str, Any]) -> bool:
    source = _source(record)
    return source in _HIDDEN_SOURCES


def _source(record: Mapping[str, Any]) -> str:
    raw = (
        record.get("split")
        or record.get("source")
        or record.get("subset")
        or record.get("set")
        or ""
    )
    return _clean_text(raw).replace("-", "_").replace(" ", "_")


def _is_passed(record: Mapping[str, Any]) -> bool:
    status = _clean_text(record.get("status") or record.get("latest_status") or record.get("outcome") or "")
    if status in {"passed", "pass", "success", "succeeded"}:
        return True
    if status in {"failed", "fail", "error", "errored", "regressed"}:
        return False
    passed = record.get("passed")
    if isinstance(passed, bool):
        return passed
    return False


def _question_id(record: Mapping[str, Any]) -> str:
    return str(record.get("question_id") or record.get("id") or record.get("benchmark_id") or "").strip()


def _safe_question(record: Mapping[str, Any]) -> str | None:
    question = str(record.get("question") or record.get("question_text") or "").strip()
    return question or None


def _token_hint_text(record: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for key in (
        "token_hints",
        "tokens",
        "missing_expected_tokens",
        "extra_generated_tokens",
        "comparison_details",
    ):
        value = record.get(key)
        if isinstance(value, Mapping):
            chunks.extend(str(item) for pair in value.items() for item in pair)
        elif isinstance(value, (list, tuple, set, frozenset)):
            chunks.extend(str(item) for item in value)
        elif value is not None:
            chunks.append(str(value))
    return _clean_text(" ".join(chunks))


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())
