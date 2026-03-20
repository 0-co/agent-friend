"""Tests for EventBusTool."""

import json
import pytest

from agent_friend.tools.event_bus import EventBusTool


@pytest.fixture
def tool():
    return EventBusTool()


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "event_bus"


def test_description(tool):
    desc = tool.description.lower()
    assert "event" in desc or "pub" in desc


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "bus_subscribe", "bus_unsubscribe", "bus_publish",
        "bus_history", "bus_topics", "bus_subscribers",
        "bus_stats", "bus_clear",
    }


# ── bus_subscribe ─────────────────────────────────────────────────────────────


def test_subscribe(tool):
    result = json.loads(tool.bus_subscribe("events", "handler_a"))
    assert result["subscribed"] is True
    assert result["topic"] == "events"


def test_subscribe_duplicate(tool):
    tool.bus_subscribe("t", "sub")
    result = json.loads(tool.bus_subscribe("t", "sub"))
    assert result["subscribed"] is False


def test_subscribe_wildcard(tool):
    result = json.loads(tool.bus_subscribe("*", "logger"))
    assert result["subscribed"] is True


def test_subscribe_max_topics():
    t = EventBusTool(max_topics=2)
    t.bus_subscribe("a", "s")
    t.bus_subscribe("b", "s")
    result = json.loads(t.bus_subscribe("c", "s"))
    assert "error" in result


def test_subscribe_max_per_topic():
    t = EventBusTool(max_subscribers_per_topic=2)
    t.bus_subscribe("t", "a")
    t.bus_subscribe("t", "b")
    result = json.loads(t.bus_subscribe("t", "c"))
    assert "error" in result


# ── bus_unsubscribe ────────────────────────────────────────────────────────────


def test_unsubscribe(tool):
    tool.bus_subscribe("t", "h")
    result = json.loads(tool.bus_unsubscribe("t", "h"))
    assert result["unsubscribed"] is True


def test_unsubscribe_not_subscribed(tool):
    result = json.loads(tool.bus_unsubscribe("t", "ghost"))
    assert result["unsubscribed"] is False


def test_unsubscribe_then_no_longer_notified(tool):
    tool.bus_subscribe("t", "h")
    tool.bus_unsubscribe("t", "h")
    result = json.loads(tool.bus_publish("t", "data"))
    assert "h" not in result["notified"]


# ── bus_publish ────────────────────────────────────────────────────────────────


def test_publish_notifies_subscribers(tool):
    tool.bus_subscribe("news", "reader_a")
    tool.bus_subscribe("news", "reader_b")
    result = json.loads(tool.bus_publish("news", {"headline": "agent ships tool"}))
    assert result["published"] is True
    assert set(result["notified"]) == {"reader_a", "reader_b"}


def test_publish_event_id_increments(tool):
    tool.bus_subscribe("t", "h")
    r1 = json.loads(tool.bus_publish("t", "first"))
    r2 = json.loads(tool.bus_publish("t", "second"))
    assert r2["event_id"] == r1["event_id"] + 1


def test_publish_no_subscribers_still_works(tool):
    result = json.loads(tool.bus_publish("silent", "hello"))
    assert result["published"] is True
    assert result["notified"] == []


def test_publish_complex_data(tool):
    tool.bus_subscribe("t", "h")
    data = {"url": "https://x.com", "tags": [1, 2, 3], "meta": {"k": "v"}}
    result = json.loads(tool.bus_publish("t", data))
    assert result["published"] is True


def test_publish_null_data(tool):
    tool.bus_subscribe("t", "h")
    result = json.loads(tool.bus_publish("t"))
    assert result["published"] is True


# ── wildcard subscriptions ────────────────────────────────────────────────────


def test_wildcard_receives_all_topics(tool):
    tool.bus_subscribe("*", "auditor")
    tool.bus_publish("orders", {"amount": 100})
    tool.bus_publish("alerts", "disk full")

    stats = json.loads(tool.bus_stats())
    assert stats["subscriber_counts"]["auditor"] == 2


def test_wildcard_not_double_notified(tool):
    """If a subscriber is in both the specific topic and wildcard, notify once."""
    tool.bus_subscribe("news", "reader")
    tool.bus_subscribe("*", "reader")
    result = json.loads(tool.bus_publish("news", "headline"))
    assert result["notified"].count("reader") == 1


# ── bus_history ────────────────────────────────────────────────────────────────


def test_history_stores_events(tool):
    tool.bus_publish("t", "first")
    tool.bus_publish("t", "second")
    history = json.loads(tool.bus_history("t"))
    assert len(history) == 2
    assert history[0]["data"] == "first"
    assert history[1]["data"] == "second"


def test_history_n_limit(tool):
    for i in range(10):
        tool.bus_publish("t", i)
    history = json.loads(tool.bus_history("t", n=3))
    assert len(history) == 3
    # Last 3 events
    assert history[-1]["data"] == 9


def test_history_empty_topic(tool):
    result = json.loads(tool.bus_history("nonexistent"))
    assert result == []


def test_history_max_history():
    t = EventBusTool(max_history=3)
    for i in range(5):
        t.bus_publish("t", i)
    hist = json.loads(t.bus_history("t", n=10))
    # Only last 3 kept
    assert len(hist) == 3
    assert hist[-1]["data"] == 4


def test_history_has_timestamp(tool):
    tool.bus_publish("t", "x")
    hist = json.loads(tool.bus_history("t"))
    assert "timestamp" in hist[0]
    assert hist[0]["timestamp"] > 0


def test_history_has_event_id(tool):
    tool.bus_publish("t", "x")
    hist = json.loads(tool.bus_history("t"))
    assert "event_id" in hist[0]


# ── bus_topics ─────────────────────────────────────────────────────────────────


def test_topics_empty(tool):
    result = json.loads(tool.bus_topics())
    assert result == []


def test_topics_shows_subscribers(tool):
    tool.bus_subscribe("a", "s1")
    tool.bus_subscribe("b", "s2")
    result = json.loads(tool.bus_topics())
    names = {t["topic"] for t in result}
    assert names == {"a", "b"}


def test_topics_shows_event_count(tool):
    tool.bus_publish("t", "x")
    tool.bus_publish("t", "y")
    topics = json.loads(tool.bus_topics())
    t_entry = next(e for e in topics if e["topic"] == "t")
    assert t_entry["event_count"] == 2


# ── bus_subscribers ────────────────────────────────────────────────────────────


def test_subscribers_list(tool):
    tool.bus_subscribe("t", "a")
    tool.bus_subscribe("t", "b")
    result = json.loads(tool.bus_subscribers("t"))
    assert set(result["subscribers"]) == {"a", "b"}


def test_subscribers_empty_topic(tool):
    result = json.loads(tool.bus_subscribers("empty"))
    assert result["subscribers"] == []


# ── bus_stats ─────────────────────────────────────────────────────────────────


def test_stats_call_counts(tool):
    tool.bus_subscribe("t", "h")
    tool.bus_publish("t", "x")
    tool.bus_publish("t", "y")
    stats = json.loads(tool.bus_stats())
    assert stats["subscriber_counts"]["h"] == 2


def test_stats_total_events(tool):
    tool.bus_publish("a", 1)
    tool.bus_publish("b", 2)
    tool.bus_publish("c", 3)
    stats = json.loads(tool.bus_stats())
    assert stats["total_events"] == 3


# ── bus_clear ─────────────────────────────────────────────────────────────────


def test_clear_topic(tool):
    tool.bus_subscribe("t", "h")
    tool.bus_publish("t", "x")
    tool.bus_clear(topic="t")
    assert json.loads(tool.bus_history("t")) == []
    # subscriber removed too
    assert json.loads(tool.bus_subscribers("t"))["subscribers"] == []


def test_clear_all(tool):
    tool.bus_subscribe("a", "s")
    tool.bus_publish("a", 1)
    tool.bus_subscribe("b", "s")
    tool.bus_publish("b", 2)
    tool.bus_clear()
    assert json.loads(tool.bus_topics()) == []
    stats = json.loads(tool.bus_stats())
    assert stats["total_events"] == 0


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_subscribe(tool):
    result = json.loads(tool.execute("bus_subscribe", {"topic": "t", "subscriber": "s"}))
    assert result["subscribed"] is True


def test_execute_unsubscribe(tool):
    tool.execute("bus_subscribe", {"topic": "t", "subscriber": "s"})
    result = json.loads(tool.execute("bus_unsubscribe", {"topic": "t", "subscriber": "s"}))
    assert result["unsubscribed"] is True


def test_execute_publish(tool):
    result = json.loads(tool.execute("bus_publish", {"topic": "t", "data": 42}))
    assert result["published"] is True


def test_execute_history(tool):
    tool.execute("bus_publish", {"topic": "t", "data": "hi"})
    result = json.loads(tool.execute("bus_history", {"topic": "t"}))
    assert len(result) == 1


def test_execute_topics(tool):
    result = json.loads(tool.execute("bus_topics", {}))
    assert isinstance(result, list)


def test_execute_subscribers(tool):
    result = json.loads(tool.execute("bus_subscribers", {"topic": "t"}))
    assert "subscribers" in result


def test_execute_stats(tool):
    result = json.loads(tool.execute("bus_stats", {}))
    assert "total_events" in result


def test_execute_clear(tool):
    result = json.loads(tool.execute("bus_clear", {}))
    assert result["cleared"] is True


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
