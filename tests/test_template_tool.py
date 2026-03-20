"""Tests for TemplateTool."""

import json

import pytest

from agent_friend.tools.template import TemplateTool


@pytest.fixture()
def tool():
    return TemplateTool()


# ---------------------------------------------------------------------------
# BaseTool contract
# ---------------------------------------------------------------------------

class TestBaseContract:
    def test_name(self, tool):
        assert tool.name == "template"

    def test_description(self, tool):
        assert len(tool.description) > 10

    def test_definitions(self, tool):
        defs = tool.definitions()
        assert isinstance(defs, list)
        assert len(defs) >= 8

    def test_definitions_keys(self, tool):
        for d in tool.definitions():
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d


# ---------------------------------------------------------------------------
# template_render
# ---------------------------------------------------------------------------

class TestTemplateRender:
    def test_simple_substitution(self, tool):
        result = tool.template_render("Hello, ${name}!", {"name": "Alice"})
        assert result["rendered"] == "Hello, Alice!"

    def test_multiple_variables(self, tool):
        result = tool.template_render("${greeting}, ${name}!", {"greeting": "Hi", "name": "Bob"})
        assert result["rendered"] == "Hi, Bob!"

    def test_returns_length(self, tool):
        result = tool.template_render("hello", {})
        assert result["length"] == 5

    def test_no_variables(self, tool):
        result = tool.template_render("plain text", {})
        assert result["rendered"] == "plain text"

    def test_numeric_value(self, tool):
        result = tool.template_render("Count: ${n}", {"n": 42})
        assert result["rendered"] == "Count: 42"

    def test_missing_variable_strict(self, tool):
        result = tool.template_render("Hello, ${name}!", {}, strict=True)
        assert "error" in result

    def test_missing_variable_non_strict(self, tool):
        result = tool.template_render("Hello, ${name}!", {}, strict=False)
        assert result["rendered"] == "Hello, ${name}!"

    def test_extra_variables_ignored(self, tool):
        result = tool.template_render("Hello, ${name}!", {"name": "Alice", "extra": "ignored"})
        assert result["rendered"] == "Hello, Alice!"

    def test_dollar_sign_variable(self, tool):
        result = tool.template_render("$name is here", {"name": "Bob"})
        assert result["rendered"] == "Bob is here"

    def test_repeated_variable(self, tool):
        result = tool.template_render("${x} and ${x}", {"x": "yes"})
        assert result["rendered"] == "yes and yes"


# ---------------------------------------------------------------------------
# template_save / template_render_named
# ---------------------------------------------------------------------------

class TestTemplateSaveAndRenderNamed:
    def test_save_and_render(self, tool):
        tool.template_save("greeting", "Hello, ${name}!")
        result = tool.template_render_named("greeting", {"name": "World"})
        assert result["rendered"] == "Hello, World!"

    def test_save_returns_variables(self, tool):
        result = tool.template_save("t", "Hi ${a} and ${b}")
        assert "a" in result["variables"]
        assert "b" in result["variables"]

    def test_save_empty_name(self, tool):
        result = tool.template_save("", "hello")
        assert "error" in result

    def test_render_named_not_found(self, tool):
        result = tool.template_render_named("nonexistent", {})
        assert "error" in result

    def test_save_overwrites(self, tool):
        tool.template_save("t", "version 1: ${x}")
        tool.template_save("t", "version 2: ${y}")
        result = tool.template_render_named("t", {"y": "ok"})
        assert "version 2" in result["rendered"]

    def test_render_named_strict_missing(self, tool):
        tool.template_save("t", "${a} and ${b}")
        result = tool.template_render_named("t", {"a": "1"}, strict=True)
        assert "error" in result

    def test_render_named_non_strict(self, tool):
        tool.template_save("t", "${a} and ${b}")
        result = tool.template_render_named("t", {"a": "1"}, strict=False)
        assert "1" in result["rendered"]


# ---------------------------------------------------------------------------
# template_variables
# ---------------------------------------------------------------------------

class TestTemplateVariables:
    def test_extract_braces(self, tool):
        result = tool.template_variables("Hello, ${name}!")
        assert "name" in result["variables"]

    def test_extract_bare(self, tool):
        result = tool.template_variables("$name is $age")
        assert "name" in result["variables"]
        assert "age" in result["variables"]

    def test_no_variables(self, tool):
        result = tool.template_variables("plain text")
        assert result["variables"] == []
        assert result["count"] == 0

    def test_deduplication(self, tool):
        result = tool.template_variables("${x} and ${x} again")
        assert result["variables"] == ["x"]
        assert result["count"] == 1

    def test_sorted(self, tool):
        result = tool.template_variables("${z} ${a} ${m}")
        assert result["variables"] == sorted(result["variables"])


# ---------------------------------------------------------------------------
# template_validate
# ---------------------------------------------------------------------------

class TestTemplateValidate:
    def test_valid(self, tool):
        result = tool.template_validate("${a} ${b}", {"a": 1, "b": 2})
        assert result["valid"] is True
        assert result["missing"] == []
        assert result["extra"] == []

    def test_missing(self, tool):
        result = tool.template_validate("${a} ${b}", {"a": 1})
        assert result["valid"] is False
        assert "b" in result["missing"]

    def test_extra(self, tool):
        result = tool.template_validate("${a}", {"a": 1, "b": 2})
        assert result["valid"] is True
        assert "b" in result["extra"]

    def test_no_variables_empty_dict(self, tool):
        result = tool.template_validate("plain text", {})
        assert result["valid"] is True

    def test_returns_required_and_provided(self, tool):
        result = tool.template_validate("${x}", {"x": 1, "y": 2})
        assert "x" in result["required"]
        assert "x" in result["provided"]
        assert "y" in result["provided"]


# ---------------------------------------------------------------------------
# template_list / template_get / template_delete
# ---------------------------------------------------------------------------

class TestTemplateListGetDelete:
    def test_list_empty(self, tool):
        assert tool.template_list() == []

    def test_list_multiple(self, tool):
        tool.template_save("a", "x=${x}")
        tool.template_save("b", "y=${y}")
        names = [r["name"] for r in tool.template_list()]
        assert "a" in names
        assert "b" in names

    def test_list_sorted(self, tool):
        tool.template_save("z", "")
        tool.template_save("a", "")
        tool.template_save("m", "")
        names = [r["name"] for r in tool.template_list()]
        assert names == sorted(names)

    def test_get_existing(self, tool):
        tool.template_save("greet", "Hello, ${name}!")
        result = tool.template_get("greet")
        assert result["template"] == "Hello, ${name}!"
        assert "name" in result["variables"]

    def test_get_missing(self, tool):
        result = tool.template_get("nonexistent")
        assert "error" in result

    def test_delete_existing(self, tool):
        tool.template_save("t", "hi")
        result = tool.template_delete("t")
        assert result["deleted"] == "t"
        assert "error" in tool.template_get("t")

    def test_delete_missing(self, tool):
        result = tool.template_delete("nonexistent")
        assert "error" in result


# ---------------------------------------------------------------------------
# execute dispatch
# ---------------------------------------------------------------------------

class TestExecuteDispatch:
    def test_execute_render(self, tool):
        out = json.loads(tool.execute("template_render", {
            "template": "Hi ${name}", "variables": {"name": "X"}
        }))
        assert out["rendered"] == "Hi X"

    def test_execute_save(self, tool):
        out = json.loads(tool.execute("template_save", {
            "name": "t", "template": "${a}"
        }))
        assert out["saved"] is True

    def test_execute_render_named(self, tool):
        tool.template_save("t", "val=${v}")
        out = json.loads(tool.execute("template_render_named", {
            "name": "t", "variables": {"v": "42"}
        }))
        assert out["rendered"] == "val=42"

    def test_execute_variables(self, tool):
        out = json.loads(tool.execute("template_variables", {"template": "${x}"}))
        assert "x" in out["variables"]

    def test_execute_validate(self, tool):
        out = json.loads(tool.execute("template_validate", {
            "template": "${a}", "variables": {"a": 1}
        }))
        assert out["valid"] is True

    def test_execute_list(self, tool):
        out = json.loads(tool.execute("template_list", {}))
        assert isinstance(out, list)

    def test_execute_get(self, tool):
        tool.template_save("t", "hi")
        out = json.loads(tool.execute("template_get", {"name": "t"}))
        assert out["template"] == "hi"

    def test_execute_delete(self, tool):
        tool.template_save("t", "hi")
        out = json.loads(tool.execute("template_delete", {"name": "t"}))
        assert out["deleted"] == "t"

    def test_execute_unknown(self, tool):
        out = json.loads(tool.execute("unknown", {}))
        assert "error" in out
