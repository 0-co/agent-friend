"""Tests for @tool decorator and FunctionTool."""

import pytest
from typing import Optional

from agent_friend.tools.function_tool import (
    FunctionTool,
    tool,
    _build_input_schema,
    _python_type_to_json_schema,
    _is_optional,
)


# ── @tool decorator ──────────────────────────────────────────────────────────


def test_tool_no_args_adds_attribute():
    @tool
    def get_name() -> str:
        """Return a name."""
        return "Alice"

    assert hasattr(get_name, "_agent_tool")
    assert isinstance(get_name._agent_tool, FunctionTool)


def test_tool_with_name_and_description():
    @tool(name="custom_name", description="Custom desc")
    def my_func(x: str) -> str:
        return x

    assert my_func._agent_tool.name == "custom_name"
    assert my_func._agent_tool.description == "Custom desc"


def test_tool_decorator_preserves_callable():
    @tool
    def double(n: int) -> int:
        """Double a number."""
        return n * 2

    assert double(5) == 10


def test_tool_uses_function_name_by_default():
    @tool
    def fetch_weather(city: str) -> str:
        """Get weather."""
        return city

    assert fetch_weather._agent_tool.name == "fetch_weather"


def test_tool_uses_docstring_by_default():
    @tool
    def my_tool(q: str) -> str:
        """Search for relevant information."""
        return q

    assert my_tool._agent_tool.description == "Search for relevant information."


def test_tool_custom_name_overrides_function_name():
    @tool(name="renamed")
    def original_name(x: str) -> str:
        return x

    assert original_name._agent_tool.name == "renamed"


def test_tool_custom_description_overrides_docstring():
    @tool(description="Override description")
    def fn(x: str) -> str:
        """Original docstring."""
        return x

    assert fn._agent_tool.description == "Override description"


def test_tool_no_docstring_falls_back_to_name():
    @tool
    def nodoc(x: str) -> str:
        return x

    assert nodoc._agent_tool.description == "nodoc"


# ── FunctionTool properties ──────────────────────────────────────────────────


def test_function_tool_name():
    def fn(x: str) -> str:
        return x

    t = FunctionTool(fn, "my_tool", "A tool")
    assert t.name == "my_tool"


def test_function_tool_description():
    def fn(x: str) -> str:
        return x

    t = FunctionTool(fn, "my_tool", "A tool")
    assert t.description == "A tool"


def test_function_tool_definitions_structure():
    def fn(city: str) -> str:
        return city

    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    t = FunctionTool(fn, "weather", "Get weather", schema)
    defs = t.definitions()

    assert len(defs) == 1
    assert defs[0]["name"] == "weather"
    assert defs[0]["description"] == "Get weather"
    assert defs[0]["input_schema"] == schema


def test_function_tool_execute_str_result():
    def fn(text: str) -> str:
        return f"echo: {text}"

    t = FunctionTool(fn, "echo", "Echo text")
    result = t.execute("echo", {"text": "hello"})
    assert result == "echo: hello"


def test_function_tool_execute_converts_non_str_to_str():
    def fn(n: int) -> int:
        return n * 2

    t = FunctionTool(fn, "double", "Double")
    result = t.execute("double", {"n": 5})
    assert result == "10"


def test_function_tool_execute_with_multiple_args():
    def fn(a: int, b: int) -> int:
        return a + b

    t = FunctionTool(fn, "add", "Add two numbers")
    result = t.execute("add", {"a": 3, "b": 4})
    assert result == "7"


# ── Schema generation ────────────────────────────────────────────────────────


def test_schema_str_param_type():
    def fn(city: str) -> str:
        return city

    schema = _build_input_schema(fn)
    assert schema["properties"]["city"]["type"] == "string"


def test_schema_int_param_type():
    def fn(count: int) -> str:
        return str(count)

    schema = _build_input_schema(fn)
    assert schema["properties"]["count"]["type"] == "integer"


def test_schema_float_param_type():
    def fn(price: float) -> str:
        return str(price)

    schema = _build_input_schema(fn)
    assert schema["properties"]["price"]["type"] == "number"


def test_schema_bool_param_type():
    def fn(verbose: bool) -> str:
        return str(verbose)

    schema = _build_input_schema(fn)
    assert schema["properties"]["verbose"]["type"] == "boolean"


def test_schema_required_str_param():
    def fn(query: str) -> str:
        return query

    schema = _build_input_schema(fn)
    assert "query" in schema.get("required", [])


def test_schema_optional_param_not_required():
    def fn(query: str, limit: Optional[int] = None) -> str:
        return query

    schema = _build_input_schema(fn)
    assert "query" in schema.get("required", [])
    assert "limit" not in schema.get("required", [])


def test_schema_default_value_not_required():
    def fn(query: str, max_results: int = 5) -> str:
        return query

    schema = _build_input_schema(fn)
    assert "query" in schema.get("required", [])
    assert "max_results" not in schema.get("required", [])


def test_schema_no_params():
    def fn() -> str:
        return "ok"

    schema = _build_input_schema(fn)
    assert schema["type"] == "object"
    assert schema["properties"] == {}


def test_schema_self_param_excluded():
    class MyClass:
        def method(self, x: str) -> str:
            return x

    schema = _build_input_schema(MyClass.method)
    assert "self" not in schema["properties"]
    assert "x" in schema["properties"]


def test_is_optional_with_optional():
    assert _is_optional(Optional[int]) is True


def test_is_optional_with_plain_type():
    assert _is_optional(str) is False


def test_python_type_to_json_schema_dict():
    assert _python_type_to_json_schema(dict)["type"] == "object"


def test_python_type_to_json_schema_list():
    assert _python_type_to_json_schema(list)["type"] == "array"


def test_python_type_to_json_schema_unknown_defaults_to_string():
    class CustomType:
        pass

    result = _python_type_to_json_schema(CustomType)
    assert result["type"] == "string"


# ── Friend integration ───────────────────────────────────────────────────────


def test_friend_accepts_decorated_function():
    from agent_friend import Friend

    @tool
    def custom_greeting(name: str) -> str:
        """Return a greeting."""
        return f"Hello, {name}!"

    # Should not raise
    friend = Friend(api_key="sk-test", tools=[custom_greeting])
    assert len(friend._tools) == 1
    assert friend._tools[0].name == "custom_greeting"


def test_friend_mixed_string_and_decorated_tool():
    from agent_friend import Friend

    @tool(name="add_nums", description="Add two numbers")
    def add(a: int, b: int) -> int:
        return a + b

    friend = Friend(api_key="sk-test", tools=["memory", add])
    assert len(friend._tools) == 2


def test_friend_rejects_plain_callable():
    from agent_friend import Friend

    def plain_fn(x: str) -> str:
        return x

    with pytest.raises(TypeError, match="@tool"):
        Friend(api_key="sk-test", tools=[plain_fn])


def test_friend_tool_definition_reflects_custom_schema():
    from agent_friend import Friend

    @tool(name="lookup", description="Look up a record")
    def lookup(record_id: int, include_deleted: bool = False) -> str:
        return str(record_id)

    friend = Friend(api_key="sk-test", tools=[lookup])
    defs = friend._build_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "lookup"
    assert defs[0]["input_schema"]["properties"]["record_id"]["type"] == "integer"
    assert "record_id" in defs[0]["input_schema"]["required"]
    assert "include_deleted" not in defs[0]["input_schema"].get("required", [])
