"""Tests for FormatTool."""

import json
import pytest

from agent_friend.tools.format_tool import FormatTool


@pytest.fixture
def tool():
    return FormatTool()


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "format"


def test_description(tool):
    assert "format" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 10


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "format_bytes", "format_duration", "format_number",
        "format_percent", "format_currency", "format_ordinal",
        "format_plural", "format_truncate", "format_pad", "format_table",
    }


# ── format_bytes ──────────────────────────────────────────────────────────────


def test_bytes_b(tool):
    assert tool.format_bytes(500) == "500.0 B"


def test_bytes_kb(tool):
    assert tool.format_bytes(1500) == "1.5 KB"


def test_bytes_mb(tool):
    assert "MB" in tool.format_bytes(1_234_567)


def test_bytes_gb(tool):
    assert "GB" in tool.format_bytes(1_000_000_000)


def test_bytes_zero(tool):
    assert tool.format_bytes(0) == "0.0 B"


def test_bytes_binary(tool):
    result = tool.format_bytes(1024, binary=True)
    assert "KiB" in result


def test_bytes_decimals(tool):
    result = tool.format_bytes(1_500_000, decimals=2)
    assert "." in result


def test_bytes_negative(tool):
    result = tool.format_bytes(-1024)
    assert result.startswith("-")


def test_bytes_invalid(tool):
    result = tool.format_bytes("not a number")
    assert "error" in result


# ── format_duration ───────────────────────────────────────────────────────────


def test_duration_ms(tool):
    assert tool.format_duration(0.5) == "500ms"


def test_duration_s(tool):
    assert tool.format_duration(45) == "45s"


def test_duration_m(tool):
    assert tool.format_duration(90) == "1m 30s"


def test_duration_h(tool):
    assert tool.format_duration(3661) == "1h 1m 1s"


def test_duration_d(tool):
    assert tool.format_duration(86400) == "1d 0h 0m 0s"


def test_duration_zero(tool):
    assert tool.format_duration(0) == "0ms"


def test_duration_verbose(tool):
    result = tool.format_duration(90, verbose=True)
    assert "minute" in result and "second" in result


def test_duration_verbose_singular(tool):
    result = tool.format_duration(61, verbose=True)
    assert "1 minute" in result
    assert "1 second" in result


def test_duration_negative(tool):
    assert tool.format_duration(-5) == "0s"


def test_duration_invalid(tool):
    result = tool.format_duration("oops")
    assert "error" in result


# ── format_number ─────────────────────────────────────────────────────────────


def test_number_basic(tool):
    assert tool.format_number(1234567.89) == "1,234,567.89"


def test_number_no_decimals(tool):
    assert tool.format_number(1234567, decimals=0) == "1,234,567"


def test_number_small(tool):
    assert tool.format_number(42) == "42.00"


def test_number_negative(tool):
    result = tool.format_number(-1234)
    assert result.startswith("-")


def test_number_custom_sep(tool):
    result = tool.format_number(1234567, thousands_sep=".", decimal_sep=",")
    assert "1.234.567" in result


def test_number_zero(tool):
    assert tool.format_number(0) == "0.00"


def test_number_invalid(tool):
    result = tool.format_number("bad")
    assert "error" in result


# ── format_percent ────────────────────────────────────────────────────────────


def test_percent_ratio(tool):
    assert tool.format_percent(0.8734) == "87.3%"


def test_percent_already_percent(tool):
    # value > 1 → treated as already a percentage
    assert tool.format_percent(87.34) == "87.3%"


def test_percent_zero(tool):
    assert tool.format_percent(0) == "0.0%"


def test_percent_one_hundred(tool):
    assert tool.format_percent(1.0) == "100.0%"


def test_percent_include_sign(tool):
    result = tool.format_percent(0.05, include_sign=True)
    assert result.startswith("+")


def test_percent_decimals(tool):
    result = tool.format_percent(0.8734, decimals=2)
    assert "87.34%" == result


def test_percent_invalid(tool):
    result = tool.format_percent("x")
    assert "error" in result


# ── format_currency ───────────────────────────────────────────────────────────


def test_currency_usd(tool):
    result = tool.format_currency(1234.5)
    assert "$" in result and "1,234.50" in result


def test_currency_eur(tool):
    result = tool.format_currency(1234.5, currency="EUR")
    assert "€" in result


def test_currency_jpy(tool):
    result = tool.format_currency(1235, currency="JPY", decimals=0)
    assert "¥" in result and "1,235" in result


def test_currency_negative(tool):
    result = tool.format_currency(-99.99)
    assert result.startswith("-")


def test_currency_unknown(tool):
    result = tool.format_currency(50, currency="XYZ")
    assert "XYZ" in result


def test_currency_invalid(tool):
    result = tool.format_currency("abc")
    assert "error" in result


# ── format_ordinal ────────────────────────────────────────────────────────────


def test_ordinal_1st(tool):
    assert tool.format_ordinal(1) == "1st"


def test_ordinal_2nd(tool):
    assert tool.format_ordinal(2) == "2nd"


def test_ordinal_3rd(tool):
    assert tool.format_ordinal(3) == "3rd"


def test_ordinal_4th(tool):
    assert tool.format_ordinal(4) == "4th"


def test_ordinal_11th(tool):
    assert tool.format_ordinal(11) == "11th"


def test_ordinal_12th(tool):
    assert tool.format_ordinal(12) == "12th"


def test_ordinal_21st(tool):
    assert tool.format_ordinal(21) == "21st"


def test_ordinal_22nd(tool):
    assert tool.format_ordinal(22) == "22nd"


def test_ordinal_negative(tool):
    assert tool.format_ordinal(-3).endswith("rd")


def test_ordinal_invalid(tool):
    result = tool.format_ordinal("x")
    assert "error" in result


# ── format_plural ─────────────────────────────────────────────────────────────


def test_plural_singular(tool):
    assert tool.format_plural(1, "item") == "1 item"


def test_plural_plural(tool):
    assert tool.format_plural(3, "item") == "3 items"


def test_plural_custom(tool):
    assert tool.format_plural(1, "mouse", "mice") == "1 mouse"
    assert tool.format_plural(2, "mouse", "mice") == "2 mice"


def test_plural_no_count(tool):
    assert tool.format_plural(3, "item", include_count=False) == "items"


def test_plural_zero(tool):
    assert tool.format_plural(0, "item") == "0 items"


def test_plural_invalid_count(tool):
    result = tool.format_plural("x", "item")
    assert "error" in result


# ── format_truncate ───────────────────────────────────────────────────────────


def test_truncate_short(tool):
    assert tool.format_truncate("hello") == "hello"


def test_truncate_exact(tool):
    assert tool.format_truncate("hello", max_length=5) == "hello"


def test_truncate_long(tool):
    result = tool.format_truncate("hello world", max_length=8)
    assert len(result) == 8
    assert result.endswith("…")


def test_truncate_custom_suffix(tool):
    result = tool.format_truncate("hello world", max_length=8, suffix="...")
    assert result.endswith("...")


def test_truncate_non_string(tool):
    result = tool.format_truncate(123)
    assert "error" in result


# ── format_pad ────────────────────────────────────────────────────────────────


def test_pad_left(tool):
    result = tool.format_pad("hi", 10)
    assert len(result) == 10
    assert result.startswith("hi")


def test_pad_right(tool):
    result = tool.format_pad("hi", 10, align="right")
    assert result.endswith("hi")


def test_pad_center(tool):
    result = tool.format_pad("hi", 10, align="center")
    assert len(result) == 10
    assert "hi" in result


def test_pad_no_padding_needed(tool):
    result = tool.format_pad("hello", 3)
    assert result == "hello"


def test_pad_custom_fill(tool):
    result = tool.format_pad("hi", 6, fill="*")
    assert "*" in result


def test_pad_invalid_align(tool):
    result = tool.format_pad("hi", 10, align="diagonal")
    assert "error" in result


# ── format_table ──────────────────────────────────────────────────────────────


def test_table_basic(tool):
    data = json.dumps([{"name": "Alice", "score": 90}, {"name": "Bob", "score": 75}])
    result = tool.format_table(data)
    assert "Alice" in result
    assert "Bob" in result
    assert "name" in result
    assert "score" in result


def test_table_has_borders(tool):
    data = json.dumps([{"x": 1}])
    result = tool.format_table(data)
    assert "|" in result
    assert "+" in result


def test_table_columns_order(tool):
    data = json.dumps([{"b": 2, "a": 1}])
    result = tool.format_table(data, columns=["a", "b"])
    a_pos = result.index("a")
    b_pos = result.index("b")
    assert a_pos < b_pos


def test_table_empty(tool):
    result = tool.format_table("[]")
    assert result == "(empty)"


def test_table_invalid_json(tool):
    result = tool.format_table("not json")
    assert "Error" in result or "error" in result


# ── execute dispatch ──────────────────────────────────────────────────────────


def test_execute_bytes(tool):
    result = tool.execute("format_bytes", {"value": 1500})
    assert "KB" in result


def test_execute_duration(tool):
    result = tool.execute("format_duration", {"seconds": 90})
    assert "1m" in result


def test_execute_number(tool):
    result = tool.execute("format_number", {"value": 1234567})
    assert "1,234,567" in result


def test_execute_percent(tool):
    result = tool.execute("format_percent", {"value": 0.5})
    assert "50" in result


def test_execute_currency(tool):
    result = tool.execute("format_currency", {"value": 100})
    assert "$" in result


def test_execute_ordinal(tool):
    result = tool.execute("format_ordinal", {"n": 1})
    assert result == "1st"


def test_execute_plural(tool):
    result = tool.execute("format_plural", {"count": 2, "singular": "cat"})
    assert result == "2 cats"


def test_execute_truncate(tool):
    result = tool.execute("format_truncate", {"text": "hello world", "max_length": 5})
    assert len(result) == 5


def test_execute_pad(tool):
    result = tool.execute("format_pad", {"text": "hi", "width": 10})
    assert len(result) == 10


def test_execute_table(tool):
    data = json.dumps([{"k": "v"}])
    result = tool.execute("format_table", {"data": data})
    assert "k" in result


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
