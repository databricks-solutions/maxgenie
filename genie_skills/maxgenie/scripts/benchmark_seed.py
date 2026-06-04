"""Helpers for writing benchmark observations from the workspace runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .runner import _ensure_import_paths, _ensure_maxgenie_package
except ImportError:  # pragma: no cover - workspace or direct script fallback
    try:
        from scripts.runner import _ensure_import_paths, _ensure_maxgenie_package
    except ModuleNotFoundError:
        from runner import _ensure_import_paths, _ensure_maxgenie_package


def write_benchmark_seed(
    *,
    curated_questions: Any,
    benchmark_results: Any,
    path: str | Path = "runs/observed/benchmark_seed.json",
    match_threshold: float = 0.85,
) -> Path:
    """Normalize visible benchmark questions/results into a runner seed file."""
    _ensure_import_paths()
    _ensure_maxgenie_package()

    from maxgenie.observed_benchmarks import write_observed_benchmark_seed

    return write_observed_benchmark_seed(
        curated_questions=curated_questions,
        benchmark_results=benchmark_results,
        path=path,
        match_threshold=match_threshold,
    )
