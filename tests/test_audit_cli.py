"""Tests for the agent-friend audit CLI subcommand and audit module."""

import json
import os
import sys
import tempfile

import pytest

from agent_friend.audit import detect_format, parse_tools, generate_report, generate_json_report, run_audit
from agent_friend.tools.function_tool import FunctionTool


# ---------------------------------------------------------------------------
# Sample tool definitions in each format
# ---------------------------------------------------------------------------

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
}

ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "description": "Temperature units"},
        },
        "required": ["city"],
    },
}

MCP_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "description": "Temperature units"},
        },
        "required": ["city"],
    },
}

JSON_SCHEMA_TOOL = {
    "type": "object",
    "title": "get_weather",
    "description": "Get current weather for a city.",
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "description": "Temperature units"},
    },
    "required": ["city"],
}

SIMPLE_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "description": "Temperature units"},
        },
        "required": ["city"],
    },
}

# A tool with a long description (>200 chars) for recommendation testing
LONG_DESC_TOOL = {
    "name": "search_web",
    "description": "A" * 250,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
    },
}


# ---------------------------------------------------------------------------
# detect_format()
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_openai_format(self):
        assert detect_format(OPENAI_TOOL) == "openai"

    def test_anthropic_format(self):
        assert detect_format(ANTHROPIC_TOOL) == "anthropic"

    def test_mcp_format(self):
        assert detect_format(MCP_TOOL) == "mcp"

    def test_json_schema_format(self):
        assert detect_format(JSON_SCHEMA_TOOL) == "json_schema"

    def test_simple_format(self):
        assert detect_format(SIMPLE_TOOL) == "simple"

    def test_simple_without_parameters(self):
        """A tool with just name + description should still be detected as simple."""
        tool = {"name": "ping", "description": "Check connectivity"}
        assert detect_format(tool) == "simple"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Cannot detect tool format"):
            detect_format({"foo": "bar"})

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="Cannot detect tool format"):
            detect_format({})

    def test_openai_takes_priority_over_type_object(self):
        """OpenAI format has type='function', not type='object'."""
        tool = {"type": "function", "function": {"name": "x", "description": "y"}}
        assert detect_format(tool) == "openai"

    def test_anthropic_takes_priority_over_simple(self):
        """Anthropic has input_schema; simple has parameters."""
        tool = {
            "name": "x",
            "description": "y",
            "input_schema": {"type": "object", "properties": {}},
            "parameters": {"type": "object"},
        }
        assert detect_format(tool) == "anthropic"

    def test_mcp_takes_priority_over_simple(self):
        """MCP has inputSchema; simple has parameters."""
        tool = {
            "name": "x",
            "description": "y",
            "inputSchema": {"type": "object", "properties": {}},
            "parameters": {"type": "object"},
        }
        assert detect_format(tool) == "mcp"


# ---------------------------------------------------------------------------
# parse_tools()
# ---------------------------------------------------------------------------


class TestParseTools:
    def test_single_openai_tool(self):
        tools = parse_tools(OPENAI_TOOL)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert tools[0].description == "Get current weather for a city."

    def test_single_anthropic_tool(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_single_mcp_tool(self):
        tools = parse_tools(MCP_TOOL)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_single_json_schema_tool(self):
        tools = parse_tools(JSON_SCHEMA_TOOL)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_single_simple_tool(self):
        tools = parse_tools(SIMPLE_TOOL)
        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_array_of_tools(self):
        data = [ANTHROPIC_TOOL, LONG_DESC_TOOL]
        tools = parse_tools(data)
        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "search_web"

    def test_empty_array(self):
        tools = parse_tools([])
        assert tools == []

    def test_returns_function_tools(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        assert all(isinstance(t, FunctionTool) for t in tools)

    def test_preserves_schema(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        defs = tools[0].definitions()
        assert "properties" in defs[0]["input_schema"]
        assert "city" in defs[0]["input_schema"]["properties"]

    def test_openai_extracts_parameters(self):
        tools = parse_tools(OPENAI_TOOL)
        defs = tools[0].definitions()
        assert "city" in defs[0]["input_schema"]["properties"]

    def test_json_schema_extracts_properties(self):
        tools = parse_tools(JSON_SCHEMA_TOOL)
        defs = tools[0].definitions()
        assert "city" in defs[0]["input_schema"]["properties"]
        assert defs[0]["input_schema"].get("required") == ["city"]

    def test_invalid_data_type_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON object or array"):
            parse_tools("not a dict or list")

    def test_invalid_data_int_raises(self):
        with pytest.raises(ValueError, match="Expected a JSON object or array"):
            parse_tools(42)

    def test_tools_have_token_estimate(self):
        """Parsed tools should support the token_estimate method."""
        tools = parse_tools(ANTHROPIC_TOOL)
        est = tools[0].token_estimate(format="openai")
        assert isinstance(est, int)
        assert est > 0

    def test_tools_export_to_all_formats(self):
        """Parsed tools should be exportable to all 5 formats."""
        tools = parse_tools(MCP_TOOL)
        t = tools[0]
        assert len(t.to_openai()) > 0
        assert len(t.to_anthropic()) > 0
        assert len(t.to_google()) > 0
        assert len(t.to_mcp()) > 0
        assert len(t.to_json_schema()) > 0


# ---------------------------------------------------------------------------
# generate_report()
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_empty_tools(self):
        report = generate_report([], use_color=False)
        assert "No tools found" in report

    def test_single_tool_report(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "get_weather" in report
        assert "tokens" in report

    def test_multiple_tools_report(self):
        data = [ANTHROPIC_TOOL, LONG_DESC_TOOL]
        tools = parse_tools(data)
        report = generate_report(tools, use_color=False)
        assert "get_weather" in report
        assert "search_web" in report
        assert "Total (2 tools)" in report

    def test_single_tool_plural(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "Total (1 tool)" in report

    def test_format_comparison_present(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "Format comparison" in report
        assert "openai" in report
        assert "anthropic" in report
        assert "google" in report
        assert "mcp" in report
        assert "json_schema" in report

    def test_cheapest_annotation(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "<- cheapest" in report

    def test_long_description_recommendation(self):
        tools = parse_tools(LONG_DESC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "consider trimming" in report
        assert "search_web" in report
        assert "250 chars" in report

    def test_no_recommendation_for_short_descriptions(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "consider trimming" not in report

    def test_description_length_shown(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        desc_len = len("Get current weather for a city.")
        assert f"{desc_len} chars" in report

    def test_no_color_mode(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "\033[" not in report

    def test_header_present(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "agent-friend audit" in report
        assert "tool token cost report" in report

    def test_report_contains_tilde_tokens(self):
        """Token estimates should use ~ prefix."""
        tools = parse_tools(ANTHROPIC_TOOL)
        report = generate_report(tools, use_color=False)
        assert "~" in report


# ---------------------------------------------------------------------------
# run_audit() — file and stdin handling
# ---------------------------------------------------------------------------


class TestRunAudit:
    def test_file_input(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([ANTHROPIC_TOOL], f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert "get_weather" in out
        finally:
            os.unlink(path)

    def test_file_not_found(self, capsys):
        code = run_audit("/nonexistent/file.json", use_color=False)
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
            code = run_audit(path, use_color=False)
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
            code = run_audit(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert "No tools found" in out
        finally:
            os.unlink(path)

    def test_stdin_input(self, monkeypatch, capsys):
        import io

        data = json.dumps([ANTHROPIC_TOOL])
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        code = run_audit("-", use_color=False)
        assert code == 0
        out = capsys.readouterr().out
        assert "get_weather" in out

    def test_stdin_none_path(self, monkeypatch, capsys):
        import io

        data = json.dumps(ANTHROPIC_TOOL)
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        code = run_audit(None, use_color=False)
        assert code == 0
        out = capsys.readouterr().out
        assert "get_weather" in out

    def test_undetectable_format(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"foo": "bar"}, f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False)
            assert code == 1
            err = capsys.readouterr().err
            assert "Cannot detect" in err
        finally:
            os.unlink(path)

    def test_openai_array_input(self, capsys):
        data = [
            OPENAI_TOOL,
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "body"],
                    },
                },
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert "get_weather" in out
            assert "send_email" in out
            assert "Total (2 tools)" in out
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: all 5 input formats produce valid reports
# ---------------------------------------------------------------------------


class TestAllFormats:
    @pytest.mark.parametrize(
        "tool_data,expected_name",
        [
            (OPENAI_TOOL, "get_weather"),
            (ANTHROPIC_TOOL, "get_weather"),
            (MCP_TOOL, "get_weather"),
            (JSON_SCHEMA_TOOL, "get_weather"),
            (SIMPLE_TOOL, "get_weather"),
        ],
        ids=["openai", "anthropic", "mcp", "json_schema", "simple"],
    )
    def test_format_produces_report(self, tool_data, expected_name, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(tool_data, f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            assert expected_name in out
            assert "Format comparison" in out
        finally:
            os.unlink(path)

    @pytest.mark.parametrize(
        "tool_data",
        [OPENAI_TOOL, ANTHROPIC_TOOL, MCP_TOOL, JSON_SCHEMA_TOOL, SIMPLE_TOOL],
        ids=["openai", "anthropic", "mcp", "json_schema", "simple"],
    )
    def test_all_formats_parse_to_function_tool(self, tool_data):
        tools = parse_tools(tool_data)
        assert len(tools) == 1
        assert isinstance(tools[0], FunctionTool)
        assert tools[0].name == "get_weather"

    @pytest.mark.parametrize(
        "tool_data",
        [OPENAI_TOOL, ANTHROPIC_TOOL, MCP_TOOL, JSON_SCHEMA_TOOL, SIMPLE_TOOL],
        ids=["openai", "anthropic", "mcp", "json_schema", "simple"],
    )
    def test_all_formats_have_positive_token_estimates(self, tool_data):
        tools = parse_tools(tool_data)
        for fmt in ("openai", "anthropic", "google", "mcp", "json_schema"):
            est = tools[0].token_estimate(format=fmt)
            assert est > 0, f"Expected positive token estimate for {fmt}"


# ---------------------------------------------------------------------------
# CLI integration (subprocess-like via main())
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_audit_help(self, monkeypatch):
        """Verify audit --help doesn't crash."""
        monkeypatch.setattr("sys.argv", ["agent-friend", "audit", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from agent_friend.cli import main
            main()
        assert exc_info.value.code == 0

    def test_audit_with_file(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(ANTHROPIC_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "audit", path, "--no-color"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            assert "get_weather" in out
        finally:
            os.unlink(path)

    def test_audit_with_no_color(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(ANTHROPIC_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "audit", "--no-color", path]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            assert "\033[" not in out
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# generate_json_report()
# ---------------------------------------------------------------------------


class TestGenerateJsonReport:
    def test_empty_tools(self):
        result = generate_json_report([])
        assert result["tool_count"] == 0
        assert result["total_tokens"] == 0
        assert result["tools"] == []

    def test_single_tool(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        result = generate_json_report(tools)
        assert result["tool_count"] == 1
        assert result["total_tokens"] > 0
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["tokens"] > 0
        assert result["tools"][0]["description_length"] == len("Get current weather for a city.")

    def test_multiple_tools(self):
        data = [ANTHROPIC_TOOL, LONG_DESC_TOOL]
        tools = parse_tools(data)
        result = generate_json_report(tools)
        assert result["tool_count"] == 2
        assert result["total_tokens"] > 0
        # Sorted by tokens descending
        assert result["tools"][0]["tokens"] >= result["tools"][1]["tokens"]

    def test_format_estimates_present(self):
        tools = parse_tools(ANTHROPIC_TOOL)
        result = generate_json_report(tools)
        fmts = result["format_estimates"]
        assert "openai" in fmts
        assert "anthropic" in fmts
        assert "google" in fmts
        assert "mcp" in fmts
        assert "json_schema" in fmts
        assert all(v > 0 for v in fmts.values())


# ---------------------------------------------------------------------------
# run_audit() — JSON output and threshold
# ---------------------------------------------------------------------------


class TestRunAuditJson:
    def test_json_output(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([ANTHROPIC_TOOL], f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False, json_output=True)
            assert code == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["tool_count"] == 1
            assert data["total_tokens"] > 0
        finally:
            os.unlink(path)

    def test_json_output_empty(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("")
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False, json_output=True)
            assert code == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["tool_count"] == 0
        finally:
            os.unlink(path)


class TestRunAuditThreshold:
    def test_under_threshold(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([ANTHROPIC_TOOL], f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False, threshold=10000)
            assert code == 0
        finally:
            os.unlink(path)

    def test_over_threshold(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([ANTHROPIC_TOOL], f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False, threshold=1)
            assert code == 2
            err = capsys.readouterr().err
            assert "Threshold exceeded" in err
        finally:
            os.unlink(path)

    def test_threshold_with_json(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([ANTHROPIC_TOOL], f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False, json_output=True, threshold=1)
            assert code == 2
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["total_tokens"] > 1
        finally:
            os.unlink(path)

    def test_no_threshold(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump([ANTHROPIC_TOOL], f)
            f.flush()
            path = f.name

        try:
            code = run_audit(path, use_color=False, threshold=None)
            assert code == 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Context window impact
# ---------------------------------------------------------------------------


class TestContextWindowImpact:
    """Tests for the context window impact section in reports."""

    def test_report_includes_context_window(self):
        tools = parse_tools([OPENAI_TOOL])
        report = generate_report(tools, use_color=False)
        assert "Context window impact:" in report
        assert "GPT-4o (128K)" in report
        assert "Claude (200K)" in report

    def test_json_report_includes_context_pct(self):
        tools = parse_tools([OPENAI_TOOL])
        result = generate_json_report(tools)
        assert "context_window_pct" in result
        assert "GPT-4o (128K)" in result["context_window_pct"]
        assert isinstance(result["context_window_pct"]["GPT-4o (128K)"], float)

    def test_empty_tools_no_crash(self):
        report = generate_report([], use_color=False)
        assert "No tools found" in report

    def test_warning_for_high_impact(self):
        """Many tools should trigger the 'check your budget' warning for GPT-4 (8K)."""
        big_tools = []
        for i in range(20):
            big_tools.append({
                "name": f"tool_{i}",
                "description": "A tool " * 30,
                "inputSchema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            })
        tools = parse_tools(big_tools)
        report = generate_report(tools, use_color=False)
        assert "check your budget" in report
