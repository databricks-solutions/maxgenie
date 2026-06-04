"""Tests for BenchmarkCache."""

from pathlib import Path

from maxgenie.benchmark_cache import BenchmarkCache
from maxgenie.benchmark_service import BenchmarkResult, BenchmarkStatus


def _make_result(question_id: str, status: BenchmarkStatus = BenchmarkStatus.PASSED) -> BenchmarkResult:
    return BenchmarkResult(
        question_id=question_id,
        question=f"Q {question_id}",
        status=status,
    )


def test_cache_miss_returns_none():
    cache = BenchmarkCache()
    assert cache.get("q1", "hash_abc") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_hit_returns_stored_result():
    cache = BenchmarkCache()
    result = _make_result("q1")
    cache.put("q1", "hash_abc", result)

    cached = cache.get("q1", "hash_abc")
    assert cached is result
    assert cache.hits == 1


def test_cache_different_hash_is_miss():
    cache = BenchmarkCache()
    result = _make_result("q1")
    cache.put("q1", "hash_abc", result)

    assert cache.get("q1", "hash_xyz") is None
    assert cache.misses == 1


def test_cache_same_question_different_config_stores_both():
    cache = BenchmarkCache()
    r1 = _make_result("q1", BenchmarkStatus.PASSED)
    r2 = _make_result("q1", BenchmarkStatus.FAILED)
    cache.put("q1", "hash_v1", r1)
    cache.put("q1", "hash_v2", r2)

    assert cache.get("q1", "hash_v1").status == BenchmarkStatus.PASSED
    assert cache.get("q1", "hash_v2").status == BenchmarkStatus.FAILED
    assert cache.size == 2


def test_cache_stats():
    cache = BenchmarkCache()
    r = _make_result("q1")
    cache.put("q1", "h1", r)

    cache.get("q1", "h1")  # hit
    cache.get("q1", "h1")  # hit
    cache.get("q2", "h1")  # miss

    assert cache.hits == 2
    assert cache.misses == 1


def test_cache_save_and_load_roundtrip(tmp_path: Path):
    cache = BenchmarkCache()
    cache.put("q1", "h1", _make_result("q1", BenchmarkStatus.PASSED))
    cache.put("q2", "h2", _make_result("q2", BenchmarkStatus.FAILED))

    cache_path = tmp_path / ".benchmark_cache.json"
    cache.save(cache_path)

    restored = BenchmarkCache.load(cache_path)
    assert restored.size == 2
    assert restored.get("q1", "h1").status == BenchmarkStatus.PASSED
    assert restored.get("q2", "h2").status == BenchmarkStatus.FAILED


def test_cache_copy_from_merges_entries(tmp_path: Path):
    source = BenchmarkCache()
    source.put("q1", "h1", _make_result("q1", BenchmarkStatus.PASSED))
    source_path = tmp_path / "source_cache.json"
    source.save(source_path)

    target = BenchmarkCache()
    target.put("q2", "h2", _make_result("q2", BenchmarkStatus.FAILED))

    copied = target.copy_from(source_path)

    assert copied == 1
    assert target.size == 2
    assert target.get("q1", "h1").status == BenchmarkStatus.PASSED
    assert target.get("q2", "h2").status == BenchmarkStatus.FAILED
