"""Tests for the agent-friend optimize CLI subcommand and optimize module."""

import json
import io
import os
import sys
import tempfile

import pytest

from agent_friend.optimize import (
    Suggestion,
    analyze_tools,
    generate_optimize_report,
    generate_json_output,
    run_optimize,
    _check_verbose_prefix,
    _check_long_description,
    _check_long_param_descriptions,
    _check_redundant_param_descriptions,
    _check_missing_descriptions,
    _check_duplicate_param_descriptions,
    _check_deep_nesting,
    _measure_nesting,
)


# ---------------------------------------------------------------------------
# Sample tool definitions
# ---------------------------------------------------------------------------

CLEAN_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
}

VERBOSE_PREFIX_TOOL = {
    "name": "search_db",
    "description": "This tool allows you to search the database for records.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
    },
}

LONG_DESC_TOOL = {
    "name": "analyze_data",
    "description": "A" * 250,
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "Input data"},
        },
    },
}

LONG_PARAM_DESC_TOOL = {
    "name": "create_user",
    "description": "Create a new user.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bio": {
                "type": "string",
                "description": "A" * 120,
            },
        },
    },
}

REDUNDANT_PARAM_TOOL = {
    "name": "search",
    "description": "Search for items.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The query."},
            "name": {"type": "string", "description": "The name"},
        },
    },
}

MISSING_DESC_TOOL = {
    "name": "do_something",
    "description": "",
    "input_schema": {
        "type": "object",
        "properties": {
            "config": {"type": "object", "properties": {}},
        },
    },
}

DEEP_NESTED_TOOL = {
    "name": "update_config",
    "description": "Update configuration.",
    "input_schema": {
        "type": "object",
        "properties": {
            "settings": {
                "type": "object",
                "description": "Settings object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "description": "Nested settings",
                        "properties": {
                            "deep": {
                                "type": "object",
                                "description": "Deep settings",
                                "properties": {
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

# Tools for cross-tool duplicate test
DUPE_TOOL_1 = {
    "name": "search_users",
    "description": "Search users.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string"},
        },
    },
}

DUPE_TOOL_2 = {
    "name": "search_products",
    "description": "Search products.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string"},
        },
    },
}

DUPE_TOOL_3 = {
    "name": "search_orders",
    "description": "Search orders.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string"},
        },
    },
}

# OpenAI format sample
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "verbose_tool",
        "description": "Use this tool to fetch data from the API.",
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string", "description": "API endpoint"},
            },
        },
    },
}

# MCP format sample
MCP_TOOL = {
    "name": "verbose_mcp",
    "description": "A tool that processes input data.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"},
        },
    },
}


# ---------------------------------------------------------------------------
# Rule 1: Verbose description prefixes
# ---------------------------------------------------------------------------


class TestVerbosePrefix:
    def test_detects_this_tool_allows(self):
        s = _check_verbose_prefix("t", "This tool allows you to search the database.")
        assert s is not None
        assert s.rule == "verbose_prefix"
        assert s.token_savings > 0

    def test_detects_use_this_tool_to(self):
        s = _check_verbose_prefix("t", "Use this tool to fetch records.")
        assert s is not None
        assert s.rule == "verbose_prefix"

    def test_detects_a_tool_that(self):
        s = _check_verbose_prefix("t", "A tool that processes data.")
        assert s is not None

    def test_detects_this_function(self):
        s = _check_verbose_prefix("t", "This function retrieves data.")
        assert s is not None

    def test_detects_allows_the_user_to(self):
        s = _check_verbose_prefix("t", "Allows the user to edit documents.")
        assert s is not None

    def test_detects_used_to(self):
        s = _check_verbose_prefix("t", "Used to transform input.")
        assert s is not None

    def test_detects_this_is_a_tool_that(self):
        s = _check_verbose_prefix("t", "This is a tool that does things.")
        assert s is not None

    def test_no_match_for_clean_description(self):
        s = _check_verbose_prefix("t", "Search the database for records.")
        assert s is None

    def test_case_insensitive(self):
        s = _check_verbose_prefix("t", "this tool allows you to search.")
        assert s is not None

    def test_suggested_text_capitalizes(self):
        s = _check_verbose_prefix("t", "This tool allows you to search items.")
        assert s is not None
        assert "Search" in s.message


# ---------------------------------------------------------------------------
# Rule 2: Long descriptions
# ---------------------------------------------------------------------------


class TestLongDescription:
    def test_flags_long_description(self):
        s = _check_long_description("t", "A" * 250)
        assert s is not None
        assert s.rule == "long_description"
        assert "250 chars" in s.message
        assert s.token_savings > 0

    def test_passes_short_description(self):
        s = _check_long_description("t", "Short description.")
        assert s is None

    def test_boundary_200_chars(self):
        s = _check_long_description("t", "A" * 200)
        assert s is None

    def test_boundary_201_chars(self):
        s = _check_long_description("t", "A" * 201)
        assert s is not None


# ---------------------------------------------------------------------------
# Rule 3: Long parameter descriptions
# ---------------------------------------------------------------------------


class TestLongParamDescription:
    def test_flags_long_param(self):
        schema = {
            "properties": {
                "x": {"type": "string", "description": "B" * 120},
            },
        }
        results = _check_long_param_descriptions("t", schema)
        assert len(results) == 1
        assert results[0].rule == "long_param_description"
        assert "120 chars" in results[0].message

    def test_passes_short_param(self):
        schema = {
            "properties": {
                "x": {"type": "string", "description": "Short."},
            },
        }
        results = _check_long_param_descriptions("t", schema)
        assert len(results) == 0

    def test_boundary_100_chars(self):
        schema = {
            "properties": {
                "x": {"type": "string", "description": "C" * 100},
            },
        }
        results = _check_long_param_descriptions("t", schema)
        assert len(results) == 0

    def test_boundary_101_chars(self):
        schema = {
            "properties": {
                "x": {"type": "string", "description": "C" * 101},
            },
        }
        results = _check_long_param_descriptions("t", schema)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Rule 4: Redundant parameter descriptions
# ---------------------------------------------------------------------------


class TestRedundantParamDescription:
    def test_exact_match(self):
        schema = {
            "properties": {
                "query": {"type": "string", "description": "query"},
            },
        }
        results = _check_redundant_param_descriptions("t", schema)
        assert len(results) == 1
        assert results[0].rule == "redundant_param_description"

    def test_with_article_the(self):
        schema = {
            "properties": {
                "query": {"type": "string", "description": "The query."},
            },
        }
        results = _check_redundant_param_descriptions("t", schema)
        assert len(results) == 1

    def test_with_article_a(self):
        schema = {
            "properties": {
                "name": {"type": "string", "description": "A name"},
            },
        }
        results = _check_redundant_param_descriptions("t", schema)
        assert len(results) == 1

    def test_underscore_to_space(self):
        schema = {
            "properties": {
                "user_name": {"type": "string", "description": "The user name."},
            },
        }
        results = _check_redundant_param_descriptions("t", schema)
        assert len(results) == 1

    def test_meaningful_description_passes(self):
        schema = {
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute against the database"},
            },
        }
        results = _check_redundant_param_descriptions("t", schema)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Rule 5: Missing descriptions
# ---------------------------------------------------------------------------


class TestMissingDescriptions:
    def test_missing_tool_description(self):
        schema = {"properties": {"x": {"type": "string"}}}
        results = _check_missing_descriptions("t", "", schema)
        assert any(s.rule == "missing_description" for s in results)

    def test_missing_complex_param_description(self):
        schema = {
            "properties": {
                "config": {"type": "object", "properties": {}},
            },
        }
        results = _check_missing_descriptions("t", "Good description", schema)
        assert any(s.rule == "missing_param_description" for s in results)

    def test_missing_array_param_description(self):
        schema = {
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }
        results = _check_missing_descriptions("t", "Good description", schema)
        assert any(s.rule == "missing_param_description" for s in results)

    def test_simple_type_no_desc_ok(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
            },
        }
        results = _check_missing_descriptions("t", "Good description", schema)
        assert not any(s.rule == "missing_param_description" for s in results)

    def test_present_description_passes(self):
        schema = {"properties": {"x": {"type": "string"}}}
        results = _check_missing_descriptions("t", "Good tool.", schema)
        assert not any(s.rule == "missing_description" for s in results)


# ---------------------------------------------------------------------------
# Rule 6: Duplicate cross-tool parameter descriptions
# ---------------------------------------------------------------------------


class TestDuplicateParamDescriptions:
    def test_three_tools_same_param(self):
        tools_data = [
            ("tool1", "desc", {"properties": {"q": {"type": "string", "description": "Search text"}}}),
            ("tool2", "desc", {"properties": {"q": {"type": "string", "description": "Search text"}}}),
            ("tool3", "desc", {"properties": {"q": {"type": "string", "description": "Search text"}}}),
        ]
        results = _check_duplicate_param_descriptions(tools_data)
        assert len(results) == 1
        assert results[0].rule == "duplicate_param_description"
        assert "3 tools" in results[0].message

    def test_two_tools_not_flagged(self):
        tools_data = [
            ("tool1", "desc", {"properties": {"q": {"type": "string", "description": "Search text"}}}),
            ("tool2", "desc", {"properties": {"q": {"type": "string", "description": "Search text"}}}),
        ]
        results = _check_duplicate_param_descriptions(tools_data)
        assert len(results) == 0

    def test_different_descriptions_not_flagged(self):
        tools_data = [
            ("tool1", "desc", {"properties": {"q": {"type": "string", "description": "Search text"}}}),
            ("tool2", "desc", {"properties": {"q": {"type": "string", "description": "Filter text"}}}),
            ("tool3", "desc", {"properties": {"q": {"type": "string", "description": "Query text"}}}),
        ]
        results = _check_duplicate_param_descriptions(tools_data)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Rule 7: Deep nesting
# ---------------------------------------------------------------------------


class TestDeepNesting:
    def test_flags_deep_nesting(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "object",
                                    "properties": {
                                        "d": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        s = _check_deep_nesting("t", schema)
        assert s is not None
        assert s.rule == "deep_nesting"
        assert s.token_savings > 0

    def test_flat_schema_passes(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }
        s = _check_deep_nesting("t", schema)
        assert s is None

    def test_two_levels_passes(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "string"},
                    },
                },
                "c": {
                    "type": "object",
                    "properties": {
                        "d": {"type": "integer"},
                    },
                },
            },
        }
        s = _check_deep_nesting("t", schema)
        assert s is None

    def test_measure_nesting_depth(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "string"},
                    },
                },
            },
        }
        assert _measure_nesting(schema) == 2

    def test_array_with_nested_objects(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nested": {
                                "type": "object",
                                "properties": {
                                    "deep": {
                                        "type": "object",
                                        "properties": {
                                            "value": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        s = _check_deep_nesting("t", schema)
        assert s is not None


# ---------------------------------------------------------------------------
# analyze_tools() integration
# ---------------------------------------------------------------------------


class TestAnalyzeTools:
    def test_clean_schema_no_suggestions(self):
        suggestions, stats = analyze_tools(CLEAN_TOOL)
        assert len(suggestions) == 0
        assert stats["tools_analyzed"] == 1
        assert stats["current_tokens"] > 0

    def test_verbose_prefix_detected(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        rules = [s.rule for s in suggestions]
        assert "verbose_prefix" in rules

    def test_long_description_detected(self):
        suggestions, stats = analyze_tools(LONG_DESC_TOOL)
        rules = [s.rule for s in suggestions]
        assert "long_description" in rules

    def test_long_param_description_detected(self):
        suggestions, stats = analyze_tools(LONG_PARAM_DESC_TOOL)
        rules = [s.rule for s in suggestions]
        assert "long_param_description" in rules

    def test_redundant_param_detected(self):
        suggestions, stats = analyze_tools(REDUNDANT_PARAM_TOOL)
        rules = [s.rule for s in suggestions]
        assert "redundant_param_description" in rules

    def test_missing_description_detected(self):
        suggestions, stats = analyze_tools(MISSING_DESC_TOOL)
        rules = [s.rule for s in suggestions]
        assert "missing_description" in rules

    def test_deep_nesting_detected(self):
        suggestions, stats = analyze_tools(DEEP_NESTED_TOOL)
        rules = [s.rule for s in suggestions]
        assert "deep_nesting" in rules

    def test_cross_tool_duplicates_detected(self):
        suggestions, stats = analyze_tools([DUPE_TOOL_1, DUPE_TOOL_2, DUPE_TOOL_3])
        rules = [s.rule for s in suggestions]
        assert "duplicate_param_description" in rules

    def test_empty_list(self):
        suggestions, stats = analyze_tools([])
        assert len(suggestions) == 0
        assert stats["tools_analyzed"] == 0

    def test_savings_are_summed(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        assert stats["estimated_savings"] == sum(s.token_savings for s in suggestions)


# ---------------------------------------------------------------------------
# Input format tests
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_openai_format(self):
        suggestions, stats = analyze_tools(OPENAI_TOOL)
        rules = [s.rule for s in suggestions]
        assert "verbose_prefix" in rules
        assert stats["tools_analyzed"] == 1

    def test_anthropic_format(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        assert stats["tools_analyzed"] == 1

    def test_mcp_format(self):
        suggestions, stats = analyze_tools(MCP_TOOL)
        rules = [s.rule for s in suggestions]
        assert "verbose_prefix" in rules
        assert stats["tools_analyzed"] == 1

    def test_simple_format(self):
        tool = {
            "name": "simple_tool",
            "description": "This function does something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                },
            },
        }
        suggestions, stats = analyze_tools(tool)
        rules = [s.rule for s in suggestions]
        assert "verbose_prefix" in rules

    def test_json_schema_format(self):
        tool = {
            "type": "object",
            "title": "schema_tool",
            "description": "Good description.",
            "properties": {
                "x": {"type": "string", "description": "Good param"},
            },
        }
        suggestions, stats = analyze_tools(tool)
        assert stats["tools_analyzed"] == 1


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReport:
    def test_empty_report(self):
        report = generate_optimize_report(
            [], {"tools_analyzed": 0, "current_tokens": 0, "estimated_savings": 0},
            use_color=False,
        )
        assert "No tools found" in report

    def test_clean_report(self):
        report = generate_optimize_report(
            [], {"tools_analyzed": 3, "current_tokens": 100, "estimated_savings": 0},
            use_color=False,
        )
        assert "No optimization suggestions" in report
        assert "Tools analyzed: 3" in report

    def test_report_with_suggestions(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        report = generate_optimize_report(suggestions, stats, use_color=False)
        assert "=== agent-friend optimize ===" in report
        assert "search_db" in report
        assert "Summary:" in report
        assert "Suggestions:" in report

    def test_report_no_ansi_when_disabled(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        report = generate_optimize_report(suggestions, stats, use_color=False)
        assert "\033[" not in report

    def test_report_shows_savings_summary(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        report = generate_optimize_report(suggestions, stats, use_color=False)
        assert "Estimated total savings:" in report
        assert "Current total:" in report
        assert "Optimized total:" in report

    def test_cross_tool_section(self):
        suggestions, stats = analyze_tools([DUPE_TOOL_1, DUPE_TOOL_2, DUPE_TOOL_3])
        report = generate_optimize_report(suggestions, stats, use_color=False)
        assert "Cross-tool:" in report


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_output_structure(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        output = generate_json_output(suggestions, stats)
        data = json.loads(output)
        assert "suggestions" in data
        assert "stats" in data
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) > 0

    def test_json_suggestion_fields(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        output = generate_json_output(suggestions, stats)
        data = json.loads(output)
        s = data["suggestions"][0]
        assert "tool" in s
        assert "rule" in s
        assert "message" in s
        assert "token_savings" in s

    def test_json_stats_fields(self):
        suggestions, stats = analyze_tools(VERBOSE_PREFIX_TOOL)
        output = generate_json_output(suggestions, stats)
        data = json.loads(output)
        assert "tools_analyzed" in data["stats"]
        assert "current_tokens" in data["stats"]
        assert "estimated_savings" in data["stats"]

    def test_empty_json(self):
        output = generate_json_output(
            [], {"tools_analyzed": 0, "current_tokens": 0, "estimated_savings": 0},
        )
        data = json.loads(output)
        assert data["suggestions"] == []


# ---------------------------------------------------------------------------
# run_optimize() — file and stdin handling
# ---------------------------------------------------------------------------


class TestRunOptimize:
    def test_file_input(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(VERBOSE_PREFIX_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_optimize(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert "search_db" in out
        finally:
            os.unlink(path)

    def test_stdin_input(self, monkeypatch, capsys):
        data = json.dumps(VERBOSE_PREFIX_TOOL)
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        code = run_optimize("-", use_color=False)
        assert code == 0
        out = capsys.readouterr().out
        assert "search_db" in out

    def test_file_not_found(self, capsys):
        code = run_optimize("/nonexistent/file.json", use_color=False)
        assert code == 1
        err = capsys.readouterr().err
        assert "file not found" in err

    def test_invalid_json(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{not valid json}")
            f.flush()
            path = f.name

        try:
            code = run_optimize(path, use_color=False)
            assert code == 1
            err = capsys.readouterr().err
            assert "invalid JSON" in err
        finally:
            os.unlink(path)

    def test_empty_file(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            code = run_optimize(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert "No tools found" in out
        finally:
            os.unlink(path)

    def test_json_flag(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(VERBOSE_PREFIX_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_optimize(path, use_color=False, json_output=True)
            assert code == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert "suggestions" in data
            assert len(data["suggestions"]) > 0
        finally:
            os.unlink(path)

    def test_undetectable_format(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"foo": "bar"}, f)
            f.flush()
            path = f.name

        try:
            code = run_optimize(path, use_color=False)
            assert code == 1
            err = capsys.readouterr().err
            assert "Cannot detect" in err
        finally:
            os.unlink(path)

    def test_clean_tool_no_suggestions(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(CLEAN_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_optimize(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert "No optimization suggestions" in out
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_optimize_help(self, monkeypatch):
        """Verify optimize --help doesn't crash."""
        monkeypatch.setattr("sys.argv", ["agent-friend", "optimize", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from agent_friend.cli import main
            main()
        assert exc_info.value.code == 0

    def test_optimize_with_file(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(VERBOSE_PREFIX_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "optimize", path, "--no-color"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            assert "search_db" in out
        finally:
            os.unlink(path)

    def test_optimize_with_json_flag(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(VERBOSE_PREFIX_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "optimize", path, "--json"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert "suggestions" in data
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Suggestion class
# ---------------------------------------------------------------------------


class TestSuggestion:
    def test_to_dict_basic(self):
        s = Suggestion("tool1", "rule1", "msg", 5)
        d = s.to_dict()
        assert d["tool"] == "tool1"
        assert d["rule"] == "rule1"
        assert d["message"] == "msg"
        assert d["token_savings"] == 5
        assert "detail" not in d

    def test_to_dict_with_detail(self):
        s = Suggestion("tool1", "rule1", "msg", 5, detail="extra info")
        d = s.to_dict()
        assert d["detail"] == "extra info"


# ---------------------------------------------------------------------------
# Multiple rules on one tool
# ---------------------------------------------------------------------------


class TestMultipleRules:
    def test_tool_with_many_issues(self):
        """A tool that triggers multiple rules at once."""
        tool = {
            "name": "bad_tool",
            "description": "This tool allows you to " + "x" * 200,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query."},
                    "data": {"type": "object", "properties": {}},
                },
            },
        }
        suggestions, stats = analyze_tools(tool)
        rules = set(s.rule for s in suggestions)
        assert "verbose_prefix" in rules
        assert "long_description" in rules
        assert "redundant_param_description" in rules
        assert "missing_param_description" in rules
        assert len(suggestions) >= 4
