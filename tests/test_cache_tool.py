"""Tests for CacheTool."""

import json
import time
import pytest

from agent_friend.tools.cache import CacheTool


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cache(tmp_path):
    """Return a CacheTool backed by a temp directory."""
    return CacheTool(cache_path=str(tmp_path / "cache.json"))


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(cache):
    assert cache.name == "cache"


def test_description(cache):
    assert "cache" in cache.description.lower()


def test_definitions_returns_five_tools(cache):
    defs = cache.definitions()
    assert len(defs) == 5


def test_definition_names(cache):
    names = {d["name"] for d in cache.definitions()}
    assert names == {"cache_get", "cache_set", "cache_delete", "cache_clear", "cache_stats"}


# ── cache_set / cache_get ─────────────────────────────────────────────────────


def test_set_and_get(cache):
    cache.cache_set("k", "v")
    assert cache.cache_get("k") == "v"


def test_get_missing_key(cache):
    assert cache.cache_get("nonexistent") is None


def test_set_overwrites(cache):
    cache.cache_set("k", "first")
    cache.cache_set("k", "second")
    assert cache.cache_get("k") == "second"


def test_set_returns_confirmation_with_ttl(cache):
    result = cache.cache_set("k", "v", ttl_seconds=60)
    assert "k" in result
    assert "60" in result


def test_set_returns_confirmation_no_expiry(cache):
    result = cache.cache_set("k", "v", ttl_seconds=None)
    assert "no expiry" in result


def test_set_json_value(cache):
    payload = json.dumps({"temp": 72, "sky": "clear"})
    cache.cache_set("weather", payload)
    retrieved = cache.cache_get("weather")
    assert json.loads(retrieved) == {"temp": 72, "sky": "clear"}


# ── TTL expiry ────────────────────────────────────────────────────────────────


def test_expired_entry_returns_none(cache):
    cache.cache_set("k", "v", ttl_seconds=0)  # expires immediately
    time.sleep(0.05)
    assert cache.cache_get("k") is None


def test_non_expired_entry_accessible(cache):
    cache.cache_set("k", "v", ttl_seconds=3600)
    assert cache.cache_get("k") == "v"


def test_no_expiry_survives(cache):
    cache.cache_set("k", "v", ttl_seconds=None)
    assert cache.cache_get("k") == "v"


def test_expired_entry_removed_from_file(cache, tmp_path):
    cache.cache_set("k", "v", ttl_seconds=0)
    time.sleep(0.05)
    cache.cache_get("k")  # triggers cleanup
    with open(cache.cache_path) as f:
        data = json.load(f)
    assert "k" not in data


# ── cache_delete ──────────────────────────────────────────────────────────────


def test_delete_existing_key(cache):
    cache.cache_set("k", "v")
    result = cache.cache_delete("k")
    assert "deleted" in result
    assert cache.cache_get("k") is None


def test_delete_nonexistent_key(cache):
    result = cache.cache_delete("missing")
    assert "not found" in result


# ── cache_clear ───────────────────────────────────────────────────────────────


def test_clear_removes_all(cache):
    cache.cache_set("a", "1")
    cache.cache_set("b", "2")
    result = cache.cache_clear()
    assert "2" in result
    assert cache.cache_get("a") is None
    assert cache.cache_get("b") is None


def test_clear_empty_cache(cache):
    result = cache.cache_clear()
    assert "0" in result


# ── cache_stats ───────────────────────────────────────────────────────────────


def test_stats_empty(cache):
    stats = json.loads(cache.cache_stats())
    assert stats["entries"] == 0
    assert stats["session_hits"] == 0
    assert stats["session_misses"] == 0


def test_stats_after_set(cache):
    cache.cache_set("a", "1")
    cache.cache_set("b", "2")
    stats = json.loads(cache.cache_stats())
    assert stats["entries"] == 2


def test_stats_tracks_hits(cache):
    cache.cache_set("k", "v")
    cache.cache_get("k")
    cache.cache_get("k")
    stats = json.loads(cache.cache_stats())
    assert stats["session_hits"] == 2


def test_stats_tracks_misses(cache):
    cache.cache_get("missing_1")
    cache.cache_get("missing_2")
    stats = json.loads(cache.cache_stats())
    assert stats["session_misses"] == 2


def test_stats_counts_expired_separately(cache):
    cache.cache_set("active", "v", ttl_seconds=3600)
    cache.cache_set("expired", "v", ttl_seconds=0)
    time.sleep(0.05)
    stats = json.loads(cache.cache_stats())
    assert stats["entries"] == 1
    assert stats["expired_entries"] == 1


# ── persistence ───────────────────────────────────────────────────────────────


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "cache.json")
    c1 = CacheTool(cache_path=path)
    c1.cache_set("key", "persisted", ttl_seconds=None)

    c2 = CacheTool(cache_path=path)
    assert c2.cache_get("key") == "persisted"


def test_multiple_keys_independent(cache):
    cache.cache_set("x", "X")
    cache.cache_set("y", "Y")
    assert cache.cache_get("x") == "X"
    assert cache.cache_get("y") == "Y"


# ── tool definition schemas ───────────────────────────────────────────────────


def test_cache_get_schema(cache):
    defs = {d["name"]: d for d in cache.definitions()}
    schema = defs["cache_get"]["input_schema"]
    assert "key" in schema["properties"]
    assert schema["required"] == ["key"]


def test_cache_set_schema(cache):
    defs = {d["name"]: d for d in cache.definitions()}
    schema = defs["cache_set"]["input_schema"]
    assert "key" in schema["properties"]
    assert "value" in schema["properties"]
    assert "ttl_seconds" in schema["properties"]
    assert set(schema["required"]) == {"key", "value"}


def test_cache_clear_schema_no_required(cache):
    defs = {d["name"]: d for d in cache.definitions()}
    schema = defs["cache_clear"]["input_schema"]
    assert schema.get("required", []) == []
