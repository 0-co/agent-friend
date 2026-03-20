"""Tests for DateTimeTool."""

import json
from datetime import datetime, timezone

import pytest

from agent_friend.tools.datetime_tool import DateTimeTool, _parse_dt, _resolve_tz


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tool():
    return DateTimeTool()


# ── basic properties ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "datetime"

def test_description_nonempty(tool):
    assert len(tool.description) > 10

def test_definitions_list(tool):
    defs = tool.definitions()
    assert isinstance(defs, list)
    assert len(defs) == 8

def test_definitions_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {"now", "parse", "format_dt", "diff", "add_duration",
                     "convert_timezone", "to_timestamp", "from_timestamp"}


# ── now() ─────────────────────────────────────────────────────────────────────

def test_now_returns_iso_string(tool):
    result = tool.now()
    dt = datetime.fromisoformat(result)
    assert dt.tzinfo is not None

def test_now_utc_contains_z_or_offset(tool):
    result = tool.now("UTC")
    assert "+00:00" in result or result.endswith("Z")

def test_now_with_timezone(tool):
    result = tool.now(timezone="America/New_York")
    dt = datetime.fromisoformat(result)
    assert dt.tzinfo is not None

def test_now_different_timezones_differ(tool):
    utc = tool.now("UTC")
    tokyo = tool.now("Asia/Tokyo")
    # Tokyo is UTC+9, so offsets will differ
    assert utc != tokyo

def test_now_invalid_timezone(tool):
    result = tool.execute("now", {"timezone": "Fake/Timezone"})
    assert result.startswith("Error:")


# ── parse() ───────────────────────────────────────────────────────────────────

def test_parse_iso_8601(tool):
    result = tool.parse("2026-03-12T15:30:00")
    dt = datetime.fromisoformat(result)
    assert dt.year == 2026 and dt.month == 3 and dt.day == 12

def test_parse_date_only(tool):
    result = tool.parse("2026-03-12")
    assert "2026-03-12" in result

def test_parse_natural_language(tool):
    result = tool.parse("March 12, 2026")
    assert "2026-03-12" in result

def test_parse_slashes(tool):
    result = tool.parse("03/12/2026")
    assert "2026" in result

def test_parse_returns_aware_datetime(tool):
    result = tool.parse("2026-03-12T10:00:00")
    dt = datetime.fromisoformat(result)
    assert dt.tzinfo is not None

def test_parse_invalid_string(tool):
    result = tool.execute("parse", {"text": "not a date at all xyz"})
    assert result.startswith("Error:")


# ── format_dt() ───────────────────────────────────────────────────────────────

def test_format_default(tool):
    result = tool.format_dt("2026-03-12T00:00:00+00:00")
    assert result == "2026-03-12 00:00:00"

def test_format_custom(tool):
    result = tool.format_dt("2026-03-12T00:00:00+00:00", fmt="%B %d, %Y")
    assert result == "March 12, 2026"

def test_format_year_only(tool):
    result = tool.format_dt("2026-03-12T00:00:00+00:00", fmt="%Y")
    assert result == "2026"

def test_format_with_timezone(tool):
    # UTC noon = NY morning (-5 in March)
    result = tool.format_dt(
        "2026-03-12T12:00:00+00:00",
        fmt="%H",
        timezone="America/New_York",
    )
    # Eastern Standard Time (before DST) = UTC-5
    assert result in ("07", "08")  # depending on DST


# ── diff() ────────────────────────────────────────────────────────────────────

def test_diff_seconds(tool):
    result = tool.diff("2026-03-12T00:00:00", "2026-03-12T00:01:00", unit="seconds")
    assert float(result) == 60.0

def test_diff_minutes(tool):
    result = tool.diff("2026-03-12T00:00:00", "2026-03-12T01:00:00", unit="minutes")
    assert float(result) == 60.0

def test_diff_hours(tool):
    result = tool.diff("2026-03-12T00:00:00", "2026-03-13T00:00:00", unit="hours")
    assert float(result) == 24.0

def test_diff_days(tool):
    result = tool.diff("2026-03-12", "2026-04-01", unit="days")
    assert float(result) == 20.0

def test_diff_order_independent(tool):
    # a before b and b before a should give same result
    r1 = tool.diff("2026-03-12", "2026-03-15", unit="days")
    r2 = tool.diff("2026-03-15", "2026-03-12", unit="days")
    assert r1 == r2

def test_diff_invalid_unit(tool):
    result = tool.execute("diff", {"a": "2026-03-12", "b": "2026-03-15", "unit": "weeks"})
    assert result.startswith("Error:")


# ── add_duration() ────────────────────────────────────────────────────────────

def test_add_days(tool):
    result = tool.add_duration("2026-03-12T00:00:00", days=7)
    dt = datetime.fromisoformat(result)
    assert dt.day == 19

def test_add_hours(tool):
    result = tool.add_duration("2026-03-12T00:00:00", hours=3)
    dt = datetime.fromisoformat(result)
    assert dt.hour == 3

def test_subtract_days(tool):
    result = tool.add_duration("2026-03-12T00:00:00", days=-1)
    dt = datetime.fromisoformat(result)
    assert dt.day == 11

def test_add_multiple_units(tool):
    result = tool.add_duration("2026-03-12T00:00:00", days=1, hours=2, minutes=30)
    dt = datetime.fromisoformat(result)
    assert dt.day == 13 and dt.hour == 2 and dt.minute == 30

def test_add_zero_returns_same_date(tool):
    result = tool.add_duration("2026-03-12T00:00:00")
    dt = datetime.fromisoformat(result)
    assert dt.day == 12


# ── convert_timezone() ────────────────────────────────────────────────────────

def test_convert_utc_to_tokyo(tool):
    result = tool.convert_timezone("2026-03-12T00:00:00", to_tz="Asia/Tokyo")
    dt = datetime.fromisoformat(result)
    # Tokyo is UTC+9
    assert dt.hour == 9

def test_convert_preserves_instant(tool):
    utc_str = "2026-03-12T12:00:00+00:00"
    tokyo = tool.convert_timezone(utc_str, to_tz="Asia/Tokyo")
    # Convert back
    back = tool.convert_timezone(tokyo, to_tz="UTC")
    # Both should represent the same instant (UTC hour 12)
    dt = datetime.fromisoformat(back)
    assert dt.hour == 12

def test_convert_invalid_tz(tool):
    result = tool.execute("convert_timezone", {"dt_str": "2026-03-12", "to_tz": "Bad/TZ"})
    assert result.startswith("Error:")


# ── to_timestamp() / from_timestamp() ────────────────────────────────────────

def test_to_timestamp_epoch(tool):
    result = tool.to_timestamp("1970-01-01T00:00:00+00:00")
    assert result == "0"

def test_to_timestamp_is_integer_string(tool):
    result = tool.to_timestamp("2026-03-12T00:00:00+00:00")
    assert result.isdigit()

def test_from_timestamp_epoch(tool):
    result = tool.from_timestamp("0")
    assert "1970" in result

def test_from_timestamp_roundtrip(tool):
    original = "2026-03-12T12:00:00+00:00"
    ts = tool.to_timestamp(original)
    back = tool.from_timestamp(ts, timezone="UTC")
    # Hours should match
    dt = datetime.fromisoformat(back)
    assert dt.hour == 12 and dt.year == 2026


# ── execute() dispatch ────────────────────────────────────────────────────────

def test_execute_now(tool):
    result = tool.execute("now", {})
    dt = datetime.fromisoformat(result)
    assert dt.year == 2026

def test_execute_parse(tool):
    result = tool.execute("parse", {"text": "2026-03-12"})
    assert "2026" in result

def test_execute_format_dt(tool):
    result = tool.execute("format_dt", {"dt_str": "2026-03-12T00:00:00", "fmt": "%Y"})
    assert result == "2026"

def test_execute_diff(tool):
    result = tool.execute("diff", {"a": "2026-03-12", "b": "2026-03-13"})
    assert float(result) == 86400.0

def test_execute_add_duration(tool):
    result = tool.execute("add_duration", {"dt_str": "2026-03-12T00:00:00", "days": 1})
    assert "2026-03-13" in result

def test_execute_convert_timezone(tool):
    result = tool.execute("convert_timezone", {
        "dt_str": "2026-03-12T00:00:00",
        "to_tz": "America/Los_Angeles",
    })
    dt = datetime.fromisoformat(result)
    assert dt.tzinfo is not None

def test_execute_to_timestamp(tool):
    result = tool.execute("to_timestamp", {"dt_str": "1970-01-01T00:00:00+00:00"})
    assert result == "0"

def test_execute_from_timestamp(tool):
    result = tool.execute("from_timestamp", {"timestamp": "0"})
    assert "1970" in result

def test_execute_unknown_tool(tool):
    result = tool.execute("fly_to_moon", {})
    assert "Unknown" in result


# ── schema validation ─────────────────────────────────────────────────────────

def test_each_definition_has_required_keys(tool):
    for defn in tool.definitions():
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn

def test_each_definition_has_type_object(tool):
    for defn in tool.definitions():
        assert defn["input_schema"]["type"] == "object"
