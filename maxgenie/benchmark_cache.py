"""Cache for benchmark results across repeated CLI invocations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from maxgenie.benchmark_service import BenchmarkResult, BenchmarkStatus

logger = logging.getLogger(__name__)


def _serialize_result(result: BenchmarkResult) -> dict[str, object]:
    return {
        "question_id": result.question_id,
        "question": result.question,
        "status": result.status.value,
        "generated_sql": result.generated_sql,
        "expected_sql": result.expected_sql,
        "error_message": result.error_message,
        "duration_seconds": result.duration_seconds,
        "conversation_id": result.conversation_id,
        "message_id": result.message_id,
        "comparison_score": result.comparison_score,
        "comparison_method": result.comparison_method,
        "comparison_details": result.comparison_details,
    }


def _deserialize_result(payload: dict[str, object]) -> BenchmarkResult:
    return BenchmarkResult(
        question_id=str(payload["question_id"]),
        question=str(payload["question"]),
        status=BenchmarkStatus(str(payload["status"])),
        generated_sql=payload.get("generated_sql"),
        expected_sql=payload.get("expected_sql"),
        error_message=payload.get("error_message"),
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
        conversation_id=payload.get("conversation_id"),
        message_id=payload.get("message_id"),
        comparison_score=payload.get("comparison_score"),
        comparison_method=payload.get("comparison_method"),
        comparison_details=payload.get("comparison_details"),
    )


@dataclass
class BenchmarkCache:
    """Cache benchmark results keyed by `(question_id, config_hash)`."""

    _cache: dict[tuple[str, str], BenchmarkResult] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    def get(self, question_id: str, config_hash: str) -> BenchmarkResult | None:
        key = (question_id, config_hash)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, question_id: str, config_hash: str, result: BenchmarkResult) -> None:
        self._cache[(question_id, config_hash)] = result

    def delete(self, question_id: str, config_hash: str) -> None:
        self._cache.pop((question_id, config_hash), None)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        return len(self._cache)

    def log_stats(self, label: str = "") -> None:
        total = self._hits + self._misses
        if total == 0:
            return
        hit_rate = self._hits / total * 100
        prefix = f"[{label}] " if label else ""
        logger.info(
            "%sCache stats: %d hits, %d misses (%.1f%% hit rate), %d entries",
            prefix,
            self._hits,
            self._misses,
            hit_rate,
            self.size,
        )

    def save(self, path: str | Path) -> Path:
        payload = {
            "hits": self._hits,
            "misses": self._misses,
            "entries": [
                {
                    "question_id": question_id,
                    "config_hash": config_hash,
                    "result": _serialize_result(result),
                }
                for (question_id, config_hash), result in sorted(self._cache.items())
            ],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        return target

    def copy_from(self, source: str | Path | "BenchmarkCache") -> int:
        """Merge cache entries from another cache or cache file.

        Returns:
            Number of entries copied into this cache.
        """
        incoming = source if isinstance(source, BenchmarkCache) else self.load(source)
        copied = 0
        for key, result in incoming._cache.items():
            if self._cache.get(key) == result:
                continue
            self._cache[key] = result
            copied += 1
        return copied

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkCache":
        target = Path(path)
        cache = cls()
        if not target.exists():
            return cache

        with target.open(encoding="utf-8") as file:
            payload = json.load(file)

        for entry in payload.get("entries", []):
            question_id = str(entry["question_id"])
            config_hash = str(entry["config_hash"])
            result = _deserialize_result(entry["result"])
            cache._cache[(question_id, config_hash)] = result

        cache._hits = int(payload.get("hits", 0))
        cache._misses = int(payload.get("misses", 0))
        return cache
