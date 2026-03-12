"""Tests for framework adapter methods and docstring parameter parsing."""

import pytest
from typing import Optional

from agent_friend.tools.base import BaseTool
from agent_friend.tools.function_tool import (
    FunctionTool,
    tool,
    _parse_docstring_params,
    _build_input_schema,
)
from agent_friend.toolkit import Toolkit


# ---------------------------------------------------------------------------
# Fixtures / helpers
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
def no_docstring(x: str) -> str:
    return x


@tool
def no_type_hints(x, y):
    """Do something."""
    return str(x) + str(y)


@tool
def optional_params(query: str, limit: Optional[int] = None, verbose: bool = False) -> str:
    """Search with optional params.

    Args:
        query: The search query
        limit: Maximum number of results
        verbose: Whether to include detailed output
    """
    return query


# ---------------------------------------------------------------------------
# Docstring parameter parsing
# ---------------------------------------------------------------------------


class TestDocstringParsing:
    def test_basic_google_style(self):
        def fn(city: str, unit: str = "celsius") -> dict:
            """Get weather for a city.

            Args:
                city: The city name
                unit: Temperature unit (celsius or fahrenheit)
            """
        result = _parse_docstring_params(fn)
        assert result == {
            "city": "The city name",
            "unit": "Temperature unit (celsius or fahrenheit)",
        }

    def test_no_docstring(self):
        def fn(x: str) -> str:
            return x
        assert _parse_docstring_params(fn) == {}

    def test_docstring_without_args_section(self):
        def fn(x: str) -> str:
            """Just a description, no Args section."""
            return x
        assert _parse_docstring_params(fn) == {}

    def test_parameters_section_header(self):
        def fn(x: str) -> str:
            """Do something.

            Parameters:
                x: The input value
            """
            return x
        result = _parse_docstring_params(fn)
        assert result == {"x": "The input value"}

    def test_multiline_description(self):
        def fn(x: str) -> str:
            """Do something.

            Args:
                x: A long description that
                    spans multiple lines
            """
            return x
        result = _parse_docstring_params(fn)
        assert "long description" in result["x"]
        assert "multiple lines" in result["x"]

    def test_args_section_followed_by_returns(self):
        def fn(x: str) -> str:
            """Do something.

            Args:
                x: Input value

            Returns:
                The result
            """
            return x
        result = _parse_docstring_params(fn)
        assert result == {"x": "Input value"}
        assert "Returns" not in result

    def test_empty_docstring(self):
        def fn(x: str) -> str:
            """"""
            return x
        assert _parse_docstring_params(fn) == {}


class TestBuildInputSchemaWithDescriptions:
    def test_descriptions_included_in_schema(self):
        def fn(city: str, unit: str = "celsius") -> dict:
            """Get weather.

            Args:
                city: The city name
                unit: Temperature unit
            """
        schema = _build_input_schema(fn)
        assert schema["properties"]["city"]["description"] == "The city name"
        assert schema["properties"]["unit"]["description"] == "Temperature unit"

    def test_no_descriptions_when_no_docstring(self):
        def fn(x: str) -> str:
            return x
        schema = _build_input_schema(fn)
        assert "description" not in schema["properties"]["x"]

    def test_type_and_description_coexist(self):
        def fn(count: int) -> str:
            """Do something.

            Args:
                count: Number of items
            """
        schema = _build_input_schema(fn)
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["count"]["description"] == "Number of items"


# ---------------------------------------------------------------------------
# to_anthropic()
# ---------------------------------------------------------------------------


class TestToAnthropic:
    def test_returns_list(self):
        result = get_weather.to_anthropic()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_correct_keys(self):
        defn = get_weather.to_anthropic()[0]
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn

    def test_name_and_description(self):
        defn = get_weather.to_anthropic()[0]
        assert defn["name"] == "get_weather"
        assert "weather" in defn["description"].lower()

    def test_input_schema_has_properties(self):
        defn = get_weather.to_anthropic()[0]
        schema = defn["input_schema"]
        assert schema["type"] == "object"
        assert "city" in schema["properties"]
        assert "unit" in schema["properties"]

    def test_same_as_definitions(self):
        assert get_weather.to_anthropic() == get_weather.definitions()


# ---------------------------------------------------------------------------
# to_openai()
# ---------------------------------------------------------------------------


class TestToOpenAI:
    def test_returns_list(self):
        result = get_weather.to_openai()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_wrapper_structure(self):
        defn = get_weather.to_openai()[0]
        assert defn["type"] == "function"
        assert "function" in defn

    def test_function_keys(self):
        func = get_weather.to_openai()[0]["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func

    def test_function_name(self):
        func = get_weather.to_openai()[0]["function"]
        assert func["name"] == "get_weather"

    def test_function_description(self):
        func = get_weather.to_openai()[0]["function"]
        assert "weather" in func["description"].lower()

    def test_parameters_match_input_schema(self):
        func = get_weather.to_openai()[0]["function"]
        anthropic_schema = get_weather.to_anthropic()[0]["input_schema"]
        assert func["parameters"] == anthropic_schema

    def test_custom_name_tool(self):
        func = add.to_openai()[0]["function"]
        assert func["name"] == "add_nums"
        assert func["description"] == "Add two numbers together"


# ---------------------------------------------------------------------------
# to_google()
# ---------------------------------------------------------------------------


class TestToGoogle:
    def test_returns_list(self):
        result = get_weather.to_google()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_correct_keys(self):
        defn = get_weather.to_google()[0]
        assert "name" in defn
        assert "description" in defn
        assert "parameters" in defn

    def test_parameters_type_is_object(self):
        defn = get_weather.to_google()[0]
        assert defn["parameters"]["type"] == "OBJECT"

    def test_property_types_are_uppercase(self):
        defn = get_weather.to_google()[0]
        props = defn["parameters"]["properties"]
        assert props["city"]["type"] == "STRING"
        assert props["unit"]["type"] == "STRING"

    def test_integer_type_uppercase(self):
        defn = add.to_google()[0]
        props = defn["parameters"]["properties"]
        assert props["a"]["type"] == "INTEGER"
        assert props["b"]["type"] == "INTEGER"

    def test_required_list(self):
        defn = get_weather.to_google()[0]
        assert "city" in defn["parameters"]["required"]

    def test_property_descriptions(self):
        defn = get_weather.to_google()[0]
        props = defn["parameters"]["properties"]
        assert props["city"]["description"] == "The city name"

    def test_boolean_type(self):
        defn = optional_params.to_google()[0]
        props = defn["parameters"]["properties"]
        assert props["verbose"]["type"] == "BOOLEAN"


# ---------------------------------------------------------------------------
# to_mcp()
# ---------------------------------------------------------------------------


class TestToMCP:
    def test_returns_list(self):
        result = get_weather.to_mcp()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_correct_keys(self):
        defn = get_weather.to_mcp()[0]
        assert "name" in defn
        assert "description" in defn
        assert "inputSchema" in defn

    def test_uses_camel_case_input_schema(self):
        defn = get_weather.to_mcp()[0]
        assert "inputSchema" in defn
        assert "input_schema" not in defn

    def test_input_schema_matches(self):
        mcp = get_weather.to_mcp()[0]
        anthropic = get_weather.to_anthropic()[0]
        assert mcp["inputSchema"] == anthropic["input_schema"]

    def test_name_matches(self):
        defn = get_weather.to_mcp()[0]
        assert defn["name"] == "get_weather"


# ---------------------------------------------------------------------------
# to_json_schema()
# ---------------------------------------------------------------------------


class TestToJsonSchema:
    def test_returns_list(self):
        result = get_weather.to_json_schema()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_has_title(self):
        schema = get_weather.to_json_schema()[0]
        assert schema["title"] == "get_weather"

    def test_has_description(self):
        schema = get_weather.to_json_schema()[0]
        assert "weather" in schema["description"].lower()

    def test_has_type_and_properties(self):
        schema = get_weather.to_json_schema()[0]
        assert schema["type"] == "object"
        assert "city" in schema["properties"]

    def test_has_required(self):
        schema = get_weather.to_json_schema()[0]
        assert "city" in schema["required"]


# ---------------------------------------------------------------------------
# @tool proxy methods
# ---------------------------------------------------------------------------


class TestToolProxyMethods:
    def test_decorated_function_has_to_anthropic(self):
        assert hasattr(get_weather, "to_anthropic")
        assert callable(get_weather.to_anthropic)

    def test_decorated_function_has_to_openai(self):
        assert hasattr(get_weather, "to_openai")
        assert callable(get_weather.to_openai)

    def test_decorated_function_has_to_google(self):
        assert hasattr(get_weather, "to_google")
        assert callable(get_weather.to_google)

    def test_decorated_function_has_to_mcp(self):
        assert hasattr(get_weather, "to_mcp")
        assert callable(get_weather.to_mcp)

    def test_decorated_function_has_to_json_schema(self):
        assert hasattr(get_weather, "to_json_schema")
        assert callable(get_weather.to_json_schema)

    def test_decorated_function_has_definitions(self):
        assert hasattr(get_weather, "definitions")
        assert callable(get_weather.definitions)

    def test_proxy_matches_underlying_tool(self):
        assert get_weather.to_openai() == get_weather._agent_tool.to_openai()
        assert get_weather.to_google() == get_weather._agent_tool.to_google()
        assert get_weather.to_mcp() == get_weather._agent_tool.to_mcp()

    def test_function_still_callable(self):
        result = get_weather("London")
        assert result == {"city": "London", "unit": "celsius", "temp": 20}

    def test_function_with_no_docstring_has_methods(self):
        assert hasattr(no_docstring, "to_openai")
        defn = no_docstring.to_openai()[0]
        assert defn["function"]["name"] == "no_docstring"


# ---------------------------------------------------------------------------
# Optional parameters
# ---------------------------------------------------------------------------


class TestOptionalParameters:
    def test_optional_not_required_in_anthropic(self):
        defn = optional_params.to_anthropic()[0]
        required = defn["input_schema"].get("required", [])
        assert "query" in required
        assert "limit" not in required
        assert "verbose" not in required

    def test_optional_not_required_in_openai(self):
        func = optional_params.to_openai()[0]["function"]
        required = func["parameters"].get("required", [])
        assert "query" in required
        assert "limit" not in required

    def test_optional_not_required_in_google(self):
        defn = optional_params.to_google()[0]
        required = defn["parameters"].get("required", [])
        assert "query" in required
        assert "limit" not in required

    def test_optional_descriptions_in_google(self):
        defn = optional_params.to_google()[0]
        props = defn["parameters"]["properties"]
        assert props["query"]["description"] == "The search query"
        assert props["limit"]["description"] == "Maximum number of results"
        assert props["verbose"]["description"] == "Whether to include detailed output"


# ---------------------------------------------------------------------------
# Functions with no type hints
# ---------------------------------------------------------------------------


class TestNoTypeHints:
    def test_defaults_to_string(self):
        defn = no_type_hints.to_anthropic()[0]
        props = defn["input_schema"]["properties"]
        assert props["x"]["type"] == "string"
        assert props["y"]["type"] == "string"

    def test_openai_works(self):
        result = no_type_hints.to_openai()
        assert len(result) == 1
        assert result[0]["type"] == "function"

    def test_google_uppercase_string(self):
        defn = no_type_hints.to_google()[0]
        props = defn["parameters"]["properties"]
        assert props["x"]["type"] == "STRING"


# ---------------------------------------------------------------------------
# FunctionTool direct usage
# ---------------------------------------------------------------------------


class TestFunctionToolAdapters:
    def test_function_tool_to_openai(self):
        def fn(x: str) -> str:
            """A helper."""
            return x

        ft = FunctionTool(fn, "helper", "A helper function")
        result = ft.to_openai()
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "helper"

    def test_function_tool_to_google(self):
        def fn(n: int) -> int:
            return n

        ft = FunctionTool(fn, "doubler", "Double a number")
        result = ft.to_google()
        assert result[0]["parameters"]["properties"]["n"]["type"] == "INTEGER"

    def test_function_tool_to_mcp(self):
        def fn(q: str) -> str:
            return q

        ft = FunctionTool(fn, "search", "Search")
        result = ft.to_mcp()
        assert "inputSchema" in result[0]
        assert result[0]["name"] == "search"


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------


class TestToolkit:
    def test_toolkit_from_decorated_functions(self):
        kit = Toolkit([get_weather, add])
        assert len(kit) == 2

    def test_toolkit_repr(self):
        kit = Toolkit([get_weather, add])
        assert repr(kit) == "Toolkit(2 tools)"

    def test_toolkit_to_anthropic(self):
        kit = Toolkit([get_weather, add])
        result = kit.to_anthropic()
        assert len(result) == 2
        names = {d["name"] for d in result}
        assert "get_weather" in names
        assert "add_nums" in names

    def test_toolkit_to_openai(self):
        kit = Toolkit([get_weather, add])
        result = kit.to_openai()
        assert len(result) == 2
        for entry in result:
            assert entry["type"] == "function"
            assert "function" in entry

    def test_toolkit_to_google(self):
        kit = Toolkit([get_weather, add])
        result = kit.to_google()
        assert len(result) == 2
        for entry in result:
            assert entry["parameters"]["type"] == "OBJECT"

    def test_toolkit_to_mcp(self):
        kit = Toolkit([get_weather, add])
        result = kit.to_mcp()
        assert len(result) == 2
        for entry in result:
            assert "inputSchema" in entry

    def test_toolkit_to_json_schema(self):
        kit = Toolkit([get_weather, add])
        result = kit.to_json_schema()
        assert len(result) == 2
        titles = {s["title"] for s in result}
        assert "get_weather" in titles
        assert "add_nums" in titles

    def test_toolkit_with_base_tool(self):
        def fn(x: str) -> str:
            return x

        ft = FunctionTool(fn, "raw_tool", "A raw function tool")
        kit = Toolkit([ft, get_weather])
        assert len(kit) == 2
        result = kit.to_openai()
        names = {d["function"]["name"] for d in result}
        assert "raw_tool" in names
        assert "get_weather" in names

    def test_toolkit_with_plain_callable(self):
        def plain_fn(msg: str) -> str:
            """Send a message."""
            return msg

        kit = Toolkit([plain_fn])
        assert len(kit) == 1
        result = kit.to_anthropic()
        assert result[0]["name"] == "plain_fn"
        assert result[0]["description"] == "Send a message."

    def test_toolkit_with_mixed_inputs(self):
        """Test Toolkit with BaseTool, @tool-decorated, and plain callable."""
        def raw_fn(z: float) -> str:
            """Raw function."""
            return str(z)

        ft = FunctionTool(raw_fn, "manual", "Manual tool")

        def plain(w: str) -> str:
            """Plain callable."""
            return w

        kit = Toolkit([ft, get_weather, plain])
        assert len(kit) == 3
        result = kit.to_openai()
        names = {d["function"]["name"] for d in result}
        assert names == {"manual", "get_weather", "plain"}

    def test_toolkit_rejects_non_callable(self):
        with pytest.raises(TypeError, match="Expected BaseTool"):
            Toolkit([42])

    def test_toolkit_empty(self):
        kit = Toolkit([])
        assert len(kit) == 0
        assert kit.to_openai() == []
        assert repr(kit) == "Toolkit(0 tools)"


# ---------------------------------------------------------------------------
# Imports from top-level package
# ---------------------------------------------------------------------------


class TestImports:
    def test_toolkit_importable_from_agent_friend(self):
        from agent_friend import Toolkit as T
        assert T is Toolkit

    def test_version_bumped(self):
        import agent_friend
        assert agent_friend.__version__ == "0.49.0"
