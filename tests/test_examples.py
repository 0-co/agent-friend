"""Tests for the agent-friend examples module and --example CLI flag."""

import json
import os
import sys
import tempfile

import pytest

from agent_friend.examples import get_example, list_examples, get_example_info
from agent_friend.grade import run_grade, grade_tools
from agent_friend.audit import run_audit, parse_tools
from agent_friend.validate import run_validate, validate_tools
from agent_friend.optimize import run_optimize, analyze_tools


# ---------------------------------------------------------------------------
# examples module tests
# ---------------------------------------------------------------------------

class TestListExamples:
    def test_returns_list(self) -> None:
        result = list_examples()
        assert isinstance(result, list)

    def test_contains_known_examples(self) -> None:
        result = list_examples()
        assert "notion" in result
        assert "github" in result
        assert "filesystem" in result
        assert "slack" in result
        assert "puppeteer" in result

    def test_sorted(self) -> None:
        result = list_examples()
        assert result == sorted(result)

    def test_at_least_five(self) -> None:
        result = list_examples()
        assert len(result) >= 5


class TestGetExampleInfo:
    def test_returns_dict(self) -> None:
        result = get_example_info()
        assert isinstance(result, dict)

    def test_has_descriptions(self) -> None:
        result = get_example_info()
        for name, desc in result.items():
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_matches_list_examples(self) -> None:
        info = get_example_info()
        names = list_examples()
        assert sorted(info.keys()) == names


class TestGetExample:
    def test_notion_loads(self) -> None:
        data = get_example("notion")
        assert isinstance(data, list)
        assert len(data) > 0

    def test_notion_has_22_tools(self) -> None:
        data = get_example("notion")
        assert len(data) == 22

    def test_notion_is_mcp_format(self) -> None:
        data = get_example("notion")
        # MCP format has "name" and "inputSchema"
        first = data[0]
        assert "name" in first
        assert "inputSchema" in first

    def test_github_loads(self) -> None:
        data = get_example("github")
        assert isinstance(data, list)
        assert len(data) > 0

    def test_github_is_mcp_format(self) -> None:
        data = get_example("github")
        first = data[0]
        assert "name" in first
        assert "inputSchema" in first

    def test_filesystem_loads(self) -> None:
        data = get_example("filesystem")
        assert isinstance(data, list)
        assert len(data) > 0

    def test_filesystem_is_mcp_format(self) -> None:
        data = get_example("filesystem")
        first = data[0]
        assert "name" in first
        assert "inputSchema" in first

    def test_slack_loads(self) -> None:
        data = get_example("slack")
        assert isinstance(data, list)
        assert len(data) == 8

    def test_slack_is_mcp_format(self) -> None:
        data = get_example("slack")
        first = data[0]
        assert "name" in first
        assert "inputSchema" in first

    def test_puppeteer_loads(self) -> None:
        data = get_example("puppeteer")
        assert isinstance(data, list)
        assert len(data) == 7

    def test_puppeteer_is_mcp_format(self) -> None:
        data = get_example("puppeteer")
        first = data[0]
        assert "name" in first
        assert "inputSchema" in first

    def test_unknown_example_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown example"):
            get_example("nonexistent")

    def test_unknown_example_lists_available(self) -> None:
        with pytest.raises(ValueError, match="notion"):
            get_example("nonexistent")

    def test_all_examples_load(self) -> None:
        """Every listed example should load without error."""
        for name in list_examples():
            data = get_example(name)
            assert isinstance(data, list)
            assert len(data) > 0

    def test_all_examples_are_valid_json(self) -> None:
        """Every example should contain valid tool definitions."""
        for name in list_examples():
            data = get_example(name)
            for tool in data:
                assert isinstance(tool, dict)
                assert "name" in tool


# ---------------------------------------------------------------------------
# Integration: --example with grade
# ---------------------------------------------------------------------------

class TestGradeWithExample:
    def _write_example_temp(self, name: str) -> str:
        """Load example and write to temp file, return path."""
        data = get_example(name)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_notion_grade_f(self) -> None:
        """Notion example should grade poorly (F or D)."""
        data = get_example("notion")
        report = grade_tools(data)
        assert report["tool_count"] == 22
        # Notion has many issues, should be F
        assert report["overall_grade"] == "F"

    def test_notion_grade_via_file(self, capsys) -> None:
        path = self._write_example_temp("notion")
        try:
            exit_code = run_grade(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] == 22
            assert data["overall_grade"] == "F"
        finally:
            os.unlink(path)

    def test_github_grades(self, capsys) -> None:
        path = self._write_example_temp("github")
        try:
            exit_code = run_grade(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] > 0
            assert data["overall_grade"] in [
                "A+", "A", "A-", "B+", "B", "B-",
                "C+", "C", "C-", "D+", "D", "D-", "F",
            ]
        finally:
            os.unlink(path)

    def test_filesystem_grades(self, capsys) -> None:
        path = self._write_example_temp("filesystem")
        try:
            exit_code = run_grade(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: --example with audit
# ---------------------------------------------------------------------------

class TestAuditWithExample:
    def _write_example_temp(self, name: str) -> str:
        data = get_example(name)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_notion_audit(self, capsys) -> None:
        path = self._write_example_temp("notion")
        try:
            exit_code = run_audit(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] == 22
            assert data["total_tokens"] > 0
        finally:
            os.unlink(path)

    def test_github_audit(self, capsys) -> None:
        path = self._write_example_temp("github")
        try:
            exit_code = run_audit(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: --example with validate
# ---------------------------------------------------------------------------

class TestValidateWithExample:
    def _write_example_temp(self, name: str) -> str:
        data = get_example(name)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_notion_validate(self, capsys) -> None:
        path = self._write_example_temp("notion")
        try:
            exit_code = run_validate(path, use_color=False, json_output=True)
            # Notion has validation issues (name_valid warnings), so exit code may be 0 or 1
            assert exit_code in (0, 1)
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] == 22
        finally:
            os.unlink(path)

    def test_notion_has_name_warnings(self) -> None:
        """Notion tools use hyphens in names which triggers name_valid warnings."""
        data = get_example("notion")
        issues, stats = validate_tools(data)
        # Should have warnings for hyphenated names
        name_issues = [i for i in issues if i.check == "name_valid"]
        assert len(name_issues) > 0

    def test_filesystem_validate(self, capsys) -> None:
        path = self._write_example_temp("filesystem")
        try:
            exit_code = run_validate(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["tool_count"] > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Integration: --example with optimize
# ---------------------------------------------------------------------------

class TestOptimizeWithExample:
    def _write_example_temp(self, name: str) -> str:
        data = get_example(name)
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_notion_optimize(self, capsys) -> None:
        path = self._write_example_temp("notion")
        try:
            exit_code = run_optimize(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["stats"]["tools_analyzed"] == 22
        finally:
            os.unlink(path)

    def test_notion_has_optimization_suggestions(self) -> None:
        """Notion tools should have optimization suggestions."""
        data = get_example("notion")
        suggestions, stats = analyze_tools(data)
        # Notion has long descriptions and other issues
        assert len(suggestions) > 0

    def test_filesystem_optimize(self, capsys) -> None:
        path = self._write_example_temp("filesystem")
        try:
            exit_code = run_optimize(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["stats"]["tools_analyzed"] > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI _resolve_file_or_example tests
# ---------------------------------------------------------------------------

class TestResolveFileOrExample:
    def test_example_creates_temp_file(self) -> None:
        from agent_friend.cli import _resolve_file_or_example

        class FakeArgs:
            example = "notion"
            file = "-"

        path = _resolve_file_or_example(FakeArgs())
        assert os.path.exists(path)
        # Read and verify it's valid JSON with tools
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 22
        os.unlink(path)

    def test_no_example_returns_file(self) -> None:
        from agent_friend.cli import _resolve_file_or_example

        class FakeArgs:
            example = None
            file = "/some/path.json"

        result = _resolve_file_or_example(FakeArgs())
        assert result == "/some/path.json"

    def test_no_example_no_file_returns_stdin(self) -> None:
        from agent_friend.cli import _resolve_file_or_example

        class FakeArgs:
            example = None
            file = "-"

        result = _resolve_file_or_example(FakeArgs())
        assert result == "-"

    def test_example_github_creates_valid_file(self) -> None:
        from agent_friend.cli import _resolve_file_or_example

        class FakeArgs:
            example = "github"
            file = "-"

        path = _resolve_file_or_example(FakeArgs())
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        os.unlink(path)


# ---------------------------------------------------------------------------
# JSON file integrity tests
# ---------------------------------------------------------------------------

class TestExampleFileIntegrity:
    def test_notion_json_parseable(self) -> None:
        """notion.json should be valid JSON."""
        data = get_example("notion")
        # Re-serialize and re-parse to verify
        raw = json.dumps(data)
        reparsed = json.loads(raw)
        assert reparsed == data

    def test_github_json_parseable(self) -> None:
        data = get_example("github")
        raw = json.dumps(data)
        reparsed = json.loads(raw)
        assert reparsed == data

    def test_filesystem_json_parseable(self) -> None:
        data = get_example("filesystem")
        raw = json.dumps(data)
        reparsed = json.loads(raw)
        assert reparsed == data

    def test_slack_json_parseable(self) -> None:
        data = get_example("slack")
        raw = json.dumps(data)
        reparsed = json.loads(raw)
        assert reparsed == data

    def test_puppeteer_json_parseable(self) -> None:
        data = get_example("puppeteer")
        raw = json.dumps(data)
        reparsed = json.loads(raw)
        assert reparsed == data

    def test_all_tools_have_required_fields(self) -> None:
        """Every tool in every example should have name, description, inputSchema."""
        for name in list_examples():
            data = get_example(name)
            for i, tool in enumerate(data):
                assert "name" in tool, "{name}[{i}] missing 'name'".format(name=name, i=i)
                assert "description" in tool, "{name}[{i}] missing 'description'".format(name=name, i=i)
                assert "inputSchema" in tool, "{name}[{i}] missing 'inputSchema'".format(name=name, i=i)

    def test_all_input_schemas_are_objects(self) -> None:
        """Every inputSchema should be type: object."""
        for name in list_examples():
            data = get_example(name)
            for tool in data:
                schema = tool["inputSchema"]
                assert schema.get("type") == "object", (
                    "{name}/{tool_name} inputSchema is not type:object".format(
                        name=name, tool_name=tool["name"],
                    )
                )

    def test_all_examples_parseable_by_audit(self) -> None:
        """All examples should be parseable by the audit module's parse_tools."""
        for name in list_examples():
            data = get_example(name)
            tools = parse_tools(data)
            assert len(tools) == len(data), (
                "{name}: parse_tools returned {got}, expected {expected}".format(
                    name=name, got=len(tools), expected=len(data),
                )
            )
