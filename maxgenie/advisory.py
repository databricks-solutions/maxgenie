"""Labels and warnings for non-benchmark advisory runs."""

from __future__ import annotations

ADVISORY_BEST_PRACTICES_CLASSIFICATION = "advisory_best_practices"
NOT_BENCHMARK_VERIFIED_STATUS = "not_benchmark_verified"
NOT_BENCHMARK_VERIFIED_LABEL = "not benchmark-verified"
NO_BENCHMARK_WARNING = (
    "WARNING: This Genie space has no benchmark. Changes are best-practice heuristics, "
    "NOT evidence-based, and must be reviewed before adoption. MaxGenie did not compute "
    "or infer a benchmark score."
)


def is_advisory_best_practices(payload: object) -> bool:
    """Return whether an artifact or metadata payload represents a no-benchmark advisory run."""
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("evidence_classification") == ADVISORY_BEST_PRACTICES_CLASSIFICATION
        or payload.get("mode") == ADVISORY_BEST_PRACTICES_CLASSIFICATION
        or payload.get("no_benchmark_advisory") is True
    )
