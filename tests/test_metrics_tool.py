"""Tests for MetricsTool."""

import json
import time

import pytest

from agent_friend.tools.metrics import MetricsTool


@pytest.fixture()
def tool():
    return MetricsTool()


# ---------------------------------------------------------------------------
# BaseTool contract
# ---------------------------------------------------------------------------

class TestBaseContract:
    def test_name(self, tool):
        assert tool.name == "metrics"

    def test_description(self, tool):
        assert len(tool.description) > 10

    def test_definitions(self, tool):
        defs = tool.definitions()
        assert isinstance(defs, list)
        assert len(defs) >= 9

    def test_definitions_keys(self, tool):
        for d in tool.definitions():
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d


# ---------------------------------------------------------------------------
# metric_increment
# ---------------------------------------------------------------------------

class TestMetricIncrement:
    def test_increment_default(self, tool):
        result = tool.metric_increment("api_calls")
        assert result["type"] == "counter"
        assert result["count"] == 1
        assert result["total"] == 1.0
        assert result["last"] == 1.0

    def test_increment_custom_value(self, tool):
        result = tool.metric_increment("bytes", 512.0)
        assert result["total"] == 512.0
        assert result["count"] == 1

    def test_increment_multiple(self, tool):
        tool.metric_increment("calls")
        tool.metric_increment("calls")
        result = tool.metric_increment("calls", 3.0)
        assert result["count"] == 3
        assert result["total"] == 5.0

    def test_increment_min_max(self, tool):
        tool.metric_increment("val", 10.0)
        tool.metric_increment("val", 2.0)
        result = tool.metric_increment("val", 7.0)
        assert result["min"] == 2.0
        assert result["max"] == 10.0

    def test_increment_with_tags(self, tool):
        result = tool.metric_increment("calls", tags={"env": "prod"})
        assert result["tags"]["env"] == "prod"

    def test_increment_type_conflict(self, tool):
        tool.metric_gauge("x", 5.0)
        result = tool.metric_increment("x")
        assert "error" in result

    def test_increment_negative(self, tool):
        result = tool.metric_increment("errors", -1.0)
        assert result["total"] == -1.0


# ---------------------------------------------------------------------------
# metric_gauge
# ---------------------------------------------------------------------------

class TestMetricGauge:
    def test_gauge_set(self, tool):
        result = tool.metric_gauge("queue_depth", 42.0)
        assert result["type"] == "gauge"
        assert result["value"] == 42.0

    def test_gauge_overwrite(self, tool):
        tool.metric_gauge("load", 0.5)
        result = tool.metric_gauge("load", 0.9)
        assert result["value"] == 0.9

    def test_gauge_zero(self, tool):
        result = tool.metric_gauge("active", 0)
        assert result["value"] == 0

    def test_gauge_negative(self, tool):
        result = tool.metric_gauge("delta", -5.0)
        assert result["value"] == -5.0

    def test_gauge_with_tags(self, tool):
        result = tool.metric_gauge("memory_mb", 256, tags={"host": "server1"})
        assert result["tags"]["host"] == "server1"

    def test_gauge_type_conflict(self, tool):
        tool.metric_increment("x")
        result = tool.metric_gauge("x", 1.0)
        assert "error" in result


# ---------------------------------------------------------------------------
# metric_timer
# ---------------------------------------------------------------------------

class TestMetricTimer:
    def test_timer_start_returns_id(self, tool):
        result = tool.metric_timer_start("search")
        assert "timer_id" in result
        assert result["name"] == "search"
        assert result["status"] == "running"

    def test_timer_stop_records_duration(self, tool):
        r = tool.metric_timer_start("op")
        time.sleep(0.01)
        stop = tool.metric_timer_stop(r["timer_id"])
        assert stop["type"] == "timer"
        assert stop["elapsed_ms"] >= 5.0
        assert stop["count"] == 1

    def test_timer_stop_unknown_id(self, tool):
        result = tool.metric_timer_stop("nonexistent-id")
        assert "error" in result

    def test_timer_multiple_stops(self, tool):
        r1 = tool.metric_timer_start("batch")
        r2 = tool.metric_timer_start("batch")
        tool.metric_timer_stop(r1["timer_id"])
        stop = tool.metric_timer_stop(r2["timer_id"])
        assert stop["count"] == 2

    def test_timer_min_max_avg(self, tool):
        for _ in range(3):
            r = tool.metric_timer_start("op")
            tool.metric_timer_stop(r["timer_id"])
        result = tool.metric_get("op")
        assert result["min_ms"] is not None
        assert result["max_ms"] >= result["min_ms"]
        assert result["avg_ms"] > 0

    def test_timer_type_conflict(self, tool):
        tool.metric_increment("x")
        r = tool.metric_timer_start("x")
        result = tool.metric_timer_stop(r["timer_id"])
        assert "error" in result


# ---------------------------------------------------------------------------
# metric_get
# ---------------------------------------------------------------------------

class TestMetricGet:
    def test_get_counter(self, tool):
        tool.metric_increment("hits", 5.0)
        result = tool.metric_get("hits")
        assert result["name"] == "hits"
        assert result["type"] == "counter"

    def test_get_gauge(self, tool):
        tool.metric_gauge("temp", 98.6)
        result = tool.metric_get("temp")
        assert result["value"] == 98.6

    def test_get_missing(self, tool):
        result = tool.metric_get("doesnt_exist")
        assert "error" in result


# ---------------------------------------------------------------------------
# metric_list
# ---------------------------------------------------------------------------

class TestMetricList:
    def test_list_empty(self, tool):
        result = tool.metric_list()
        assert result == []

    def test_list_multiple(self, tool):
        tool.metric_increment("a")
        tool.metric_gauge("b", 1.0)
        result = tool.metric_list()
        names = [r["name"] for r in result]
        assert "a" in names
        assert "b" in names

    def test_list_sorted(self, tool):
        tool.metric_increment("z")
        tool.metric_increment("a")
        tool.metric_increment("m")
        names = [r["name"] for r in tool.metric_list()]
        assert names == sorted(names)

    def test_list_types_correct(self, tool):
        tool.metric_increment("counter_m")
        tool.metric_gauge("gauge_m", 1.0)
        for r in tool.metric_list():
            if r["name"] == "counter_m":
                assert r["type"] == "counter"
            elif r["name"] == "gauge_m":
                assert r["type"] == "gauge"


# ---------------------------------------------------------------------------
# metric_summary
# ---------------------------------------------------------------------------

class TestMetricSummary:
    def test_summary_empty(self, tool):
        assert tool.metric_summary() == {}

    def test_summary_contains_all(self, tool):
        tool.metric_increment("a")
        tool.metric_gauge("b", 2.0)
        summary = tool.metric_summary()
        assert "a" in summary
        assert "b" in summary

    def test_summary_is_copy(self, tool):
        tool.metric_increment("x")
        summary = tool.metric_summary()
        summary["x"]["count"] = 999
        assert tool.metric_get("x")["count"] == 1


# ---------------------------------------------------------------------------
# metric_reset
# ---------------------------------------------------------------------------

class TestMetricReset:
    def test_reset_single(self, tool):
        tool.metric_increment("a")
        tool.metric_increment("b")
        result = tool.metric_reset("a")
        assert result["reset_count"] == 1
        assert "a" not in tool.metric_summary()
        assert "b" in tool.metric_summary()

    def test_reset_all(self, tool):
        tool.metric_increment("a")
        tool.metric_gauge("b", 1.0)
        result = tool.metric_reset()
        assert result["reset_count"] == 2
        assert tool.metric_summary() == {}

    def test_reset_missing(self, tool):
        result = tool.metric_reset("nonexistent")
        assert "error" in result
        assert result["reset_count"] == 0

    def test_reset_clears_timers(self, tool):
        r = tool.metric_timer_start("op")
        tool.metric_reset()
        stop = tool.metric_timer_stop(r["timer_id"])
        assert "error" in stop


# ---------------------------------------------------------------------------
# metric_export
# ---------------------------------------------------------------------------

class TestMetricExport:
    def test_export_json_empty(self, tool):
        result = tool.metric_export("json")
        assert json.loads(result) == {}

    def test_export_json(self, tool):
        tool.metric_increment("calls", 5.0)
        result = tool.metric_export("json")
        parsed = json.loads(result)
        assert "calls" in parsed
        assert parsed["calls"]["total"] == 5.0

    def test_export_prometheus_counter(self, tool):
        tool.metric_increment("api_calls", 3.0)
        result = tool.metric_export("prometheus")
        assert "api_calls_total 3.0" in result
        assert "# TYPE api_calls counter" in result

    def test_export_prometheus_gauge(self, tool):
        tool.metric_gauge("queue_depth", 7.0)
        result = tool.metric_export("prometheus")
        assert "queue_depth 7.0" in result
        assert "# TYPE queue_depth gauge" in result

    def test_export_default_is_json(self, tool):
        tool.metric_increment("x")
        result = tool.metric_export()
        parsed = json.loads(result)
        assert "x" in parsed

    def test_export_prometheus_name_sanitize(self, tool):
        tool.metric_gauge("my-metric.value", 1.0)
        result = tool.metric_export("prometheus")
        assert "my_metric_value" in result


# ---------------------------------------------------------------------------
# execute dispatch
# ---------------------------------------------------------------------------

class TestExecuteDispatch:
    def test_execute_increment(self, tool):
        out = json.loads(tool.execute("metric_increment", {"name": "hits"}))
        assert out["type"] == "counter"

    def test_execute_gauge(self, tool):
        out = json.loads(tool.execute("metric_gauge", {"name": "depth", "value": 5.0}))
        assert out["value"] == 5.0

    def test_execute_timer_start_stop(self, tool):
        r = json.loads(tool.execute("metric_timer_start", {"name": "op"}))
        assert "timer_id" in r
        stop = json.loads(tool.execute("metric_timer_stop", {"timer_id": r["timer_id"]}))
        assert stop["count"] == 1

    def test_execute_get(self, tool):
        tool.metric_increment("x")
        out = json.loads(tool.execute("metric_get", {"name": "x"}))
        assert out["type"] == "counter"

    def test_execute_list(self, tool):
        tool.metric_increment("a")
        out = json.loads(tool.execute("metric_list", {}))
        assert any(r["name"] == "a" for r in out)

    def test_execute_summary(self, tool):
        tool.metric_gauge("g", 1.0)
        out = json.loads(tool.execute("metric_summary", {}))
        assert "g" in out

    def test_execute_reset(self, tool):
        tool.metric_increment("x")
        out = json.loads(tool.execute("metric_reset", {}))
        assert out["reset_count"] >= 1

    def test_execute_export(self, tool):
        tool.metric_increment("x")
        out = json.loads(tool.execute("metric_export", {"format": "json"}))
        assert "output" in out

    def test_execute_unknown(self, tool):
        out = json.loads(tool.execute("unknown_fn", {}))
        assert "error" in out
