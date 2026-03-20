"""Tests for token_estimate() on Tool and Toolkit, and token_report()."""

import json

import pytest
from typing import Optional

from agent_friend.tools.function_tool import FunctionTool, tool
from agent_friend.toolkit import Toolkit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get weather for a city.

    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return {"city": city, "unit": unit, "temp": 20}


@tool(name="add_nums", description="Add two numbers together")
def add(a: int, b: int) -> int:
    return a + b


@tool
def no_params() -> str:
    """A tool with no parameters."""
    return "ok"


@tool
def many_params(
    query: str,
    limit: int = 10,
    offset: int = 0,
    verbose: bool = False,
    format: str = "json",
    tags: Optional[str] = None,
) -> str:
    """Search with many parameters.

    Args:
        query: The search query
        limit: Maximum number of results
        offset: Result offset for pagination
        verbose: Whether to include detailed output
        format: Output format (json or csv)
        tags: Comma-separated tag filter
    """
    return query


# ---------------------------------------------------------------------------
# BaseTool.token_estimate() via FunctionTool
# ---------------------------------------------------------------------------


class TestToolTokenEstimate:
    def test_returns_int(self):
        result = get_weather._agent_tool.token_estimate(format="openai")
        assert isinstance(result, int)

    def test_positive_for_nonempty_tool(self):
        result = get_weather._agent_tool.token_estimate(format="openai")
        assert result > 0

    def test_all_formats_supported(self):
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            result = get_weather._agent_tool.token_estimate(format=fmt)
            assert isinstance(result, int)
            assert result > 0

    def test_default_format_is_openai(self):
        default = get_weather._agent_tool.token_estimate()
        explicit = get_weather._agent_tool.token_estimate(format="openai")
        assert default == explicit

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            get_weather._agent_tool.token_estimate(format="invalid")

    def test_more_params_means_more_tokens(self):
        small = no_params._agent_tool.token_estimate(format="openai")
        large = many_params._agent_tool.token_estimate(format="openai")
        assert large > small

    def test_matches_manual_calculation(self):
        """Verify the heuristic: len(json) / 4."""
        schema = get_weather._agent_tool.to_openai()
        expected = int(len(json.dumps(schema, separators=(",", ":"))) / 4)
        assert get_weather._agent_tool.token_estimate(format="openai") == expected

    def test_anthropic_format(self):
        schema = get_weather._agent_tool.to_anthropic()
        expected = int(len(json.dumps(schema, separators=(",", ":"))) / 4)
        assert get_weather._agent_tool.token_estimate(format="anthropic") == expected

    def test_google_format(self):
        schema = get_weather._agent_tool.to_google()
        expected = int(len(json.dumps(schema, separators=(",", ":"))) / 4)
        assert get_weather._agent_tool.token_estimate(format="google") == expected

    def test_mcp_format(self):
        schema = get_weather._agent_tool.to_mcp()
        expected = int(len(json.dumps(schema, separators=(",", ":"))) / 4)
        assert get_weather._agent_tool.token_estimate(format="mcp") == expected

    def test_json_schema_format(self):
        schema = get_weather._agent_tool.to_json_schema()
        expected = int(len(json.dumps(schema, separators=(",", ":"))) / 4)
        assert get_weather._agent_tool.token_estimate(format="json_schema") == expected


# ---------------------------------------------------------------------------
# @tool proxy: token_estimate accessible on decorated function
# ---------------------------------------------------------------------------


class TestToolDecoratorProxy:
    def test_decorated_function_has_token_estimate(self):
        assert hasattr(get_weather, "token_estimate")
        assert callable(get_weather.token_estimate)

    def test_proxy_matches_underlying_tool(self):
        assert get_weather.token_estimate(format="openai") == \
            get_weather._agent_tool.token_estimate(format="openai")

    def test_proxy_all_formats(self):
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            assert get_weather.token_estimate(format=fmt) == \
                get_weather._agent_tool.token_estimate(format=fmt)

    def test_proxy_default_format(self):
        assert get_weather.token_estimate() == \
            get_weather._agent_tool.token_estimate()


# ---------------------------------------------------------------------------
# FunctionTool direct usage
# ---------------------------------------------------------------------------


class TestFunctionToolTokenEstimate:
    def test_function_tool_token_estimate(self):
        def fn(x: str) -> str:
            """A helper."""
            return x

        ft = FunctionTool(fn, "helper", "A helper function")
        result = ft.token_estimate(format="openai")
        assert isinstance(result, int)
        assert result > 0

    def test_function_tool_all_formats(self):
        def fn(x: str, y: int = 0) -> str:
            return str(x)

        ft = FunctionTool(fn, "test_tool", "Test tool")
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            result = ft.token_estimate(format=fmt)
            assert isinstance(result, int)
            assert result > 0


# ---------------------------------------------------------------------------
# Toolkit.token_estimate()
# ---------------------------------------------------------------------------


class TestToolkitTokenEstimate:
    def test_returns_int(self):
        kit = Toolkit([get_weather, add])
        result = kit.token_estimate(format="openai")
        assert isinstance(result, int)

    def test_positive_for_nonempty_toolkit(self):
        kit = Toolkit([get_weather, add])
        assert kit.token_estimate(format="openai") > 0

    def test_empty_toolkit(self):
        kit = Toolkit([])
        assert kit.token_estimate(format="openai") == 0

    def test_all_formats_supported(self):
        kit = Toolkit([get_weather, add])
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            result = kit.token_estimate(format=fmt)
            assert isinstance(result, int)
            assert result > 0

    def test_default_format_is_openai(self):
        kit = Toolkit([get_weather, add])
        assert kit.token_estimate() == kit.token_estimate(format="openai")

    def test_invalid_format_raises(self):
        kit = Toolkit([get_weather])
        with pytest.raises(ValueError, match="Unknown format"):
            kit.token_estimate(format="invalid")

    def test_more_tools_means_more_tokens(self):
        small = Toolkit([get_weather])
        large = Toolkit([get_weather, add, many_params])
        assert large.token_estimate(format="openai") > small.token_estimate(format="openai")

    def test_matches_manual_calculation(self):
        kit = Toolkit([get_weather, add])
        schema = kit.to_openai()
        expected = int(len(json.dumps(schema, separators=(",", ":"))) / 4)
        assert kit.token_estimate(format="openai") == expected

    def test_toolkit_estimate_is_whole_list(self):
        """Toolkit estimates the full JSON array, not sum of individual tools."""
        kit = Toolkit([get_weather, add])
        # The toolkit serializes the entire list at once
        whole = kit.token_estimate(format="openai")
        # Individual tool estimates summed would be slightly different
        # (because the list brackets and commas are counted once in the whole)
        assert isinstance(whole, int)
        assert whole > 0


# ---------------------------------------------------------------------------
# Toolkit.token_report()
# ---------------------------------------------------------------------------


class TestToolkitTokenReport:
    def test_returns_dict(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        assert isinstance(report, dict)

    def test_has_required_keys(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        assert "estimates" in report
        assert "most_expensive" in report
        assert "least_expensive" in report
        assert "tool_count" in report

    def test_estimates_has_all_formats(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            assert fmt in report["estimates"]
            assert isinstance(report["estimates"][fmt], int)

    def test_most_expensive_is_valid_format(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        assert report["most_expensive"] in report["estimates"]

    def test_least_expensive_is_valid_format(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        assert report["least_expensive"] in report["estimates"]

    def test_most_expensive_is_actually_largest(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        most = report["most_expensive"]
        for fmt, count in report["estimates"].items():
            assert report["estimates"][most] >= count

    def test_least_expensive_is_actually_smallest(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        least = report["least_expensive"]
        for fmt, count in report["estimates"].items():
            assert report["estimates"][least] <= count

    def test_tool_count(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        assert report["tool_count"] == 2

    def test_tool_count_matches_len(self):
        kit = Toolkit([get_weather, add, many_params])
        report = kit.token_report()
        assert report["tool_count"] == len(kit)

    def test_empty_toolkit_report(self):
        kit = Toolkit([])
        report = kit.token_report()
        assert report["tool_count"] == 0
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            assert report["estimates"][fmt] == 0

    def test_estimates_match_individual_calls(self):
        kit = Toolkit([get_weather, add])
        report = kit.token_report()
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            assert report["estimates"][fmt] == kit.token_estimate(format=fmt)

    def test_single_tool_report(self):
        kit = Toolkit([no_params])
        report = kit.token_report()
        assert report["tool_count"] == 1
        assert all(v >= 0 for v in report["estimates"].values())


# ---------------------------------------------------------------------------
# Cross-format comparison sanity checks
# ---------------------------------------------------------------------------


class TestCrossFormatComparison:
    def test_openai_has_wrapper_overhead(self):
        """OpenAI wraps in {"type":"function","function":{...}} so should
        generally be larger than anthropic/mcp for the same tool."""
        kit = Toolkit([get_weather, add])
        openai_est = kit.token_estimate(format="openai")
        anthropic_est = kit.token_estimate(format="anthropic")
        # OpenAI adds "type":"function" wrapper
        assert openai_est > anthropic_est

    def test_google_uppercases_types(self):
        """Google format uppercases types (STRING vs string) which slightly
        changes the character count but should still be in the same ballpark."""
        kit = Toolkit([get_weather, add])
        google_est = kit.token_estimate(format="google")
        assert google_est > 0

    def test_all_formats_in_same_order_of_magnitude(self):
        """All formats should produce estimates within 5x of each other."""
        kit = Toolkit([get_weather, add, many_params])
        estimates = [kit.token_estimate(format=fmt)
                     for fmt in ("openai", "anthropic", "google", "mcp", "json_schema")]
        assert max(estimates) <= 5 * min(estimates)
