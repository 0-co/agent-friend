"""Tests for the agent-friend grade CLI subcommand and grade module."""

import json
import os
import sys
import tempfile
from typing import Any

import pytest

from agent_friend.grade import (
    score_to_grade,
    compute_correctness_score,
    compute_efficiency_score,
    compute_quality_score,
    compute_overall_score,
    grade_tools,
    generate_grade_report,
    run_grade,
    _grade_color,
)


# ---------------------------------------------------------------------------
# Sample tool definitions
# ---------------------------------------------------------------------------

CLEAN_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}

CLEAN_ANTHROPIC_TOOL = {
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

CLEAN_MCP_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
}

# Tool with validation errors
BROKEN_TOOL = {
    "name": "bad_tool",
    "description": "A tool with problems.",
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "badtype"},
            "y": {"type": "object"},  # no properties = warn
        },
        "required": ["x", "missing_param"],
    },
}

# Tool with verbose description (triggers optimize suggestion)
VERBOSE_TOOL = {
    "name": "search_db",
    "description": "This tool allows you to search the database for records.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
    },
}

# Tool with very long description (triggers both optimize suggestions)
LONG_DESC_TOOL = {
    "name": "analyze",
    "description": "A" * 250,
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {"type": "string", "description": "Input data"},
        },
    },
}


# ---------------------------------------------------------------------------
# score_to_grade tests
# ---------------------------------------------------------------------------

class TestScoreToGrade:
    def test_a_plus(self) -> None:
        assert score_to_grade(100) == "A+"
        assert score_to_grade(97) == "A+"
        assert score_to_grade(99.5) == "A+"

    def test_a(self) -> None:
        assert score_to_grade(93) == "A"
        assert score_to_grade(96) == "A"

    def test_a_minus(self) -> None:
        assert score_to_grade(90) == "A-"
        assert score_to_grade(92) == "A-"

    def test_b_plus(self) -> None:
        assert score_to_grade(87) == "B+"
        assert score_to_grade(89) == "B+"

    def test_b(self) -> None:
        assert score_to_grade(83) == "B"
        assert score_to_grade(86) == "B"

    def test_b_minus(self) -> None:
        assert score_to_grade(80) == "B-"
        assert score_to_grade(82) == "B-"

    def test_c_plus(self) -> None:
        assert score_to_grade(77) == "C+"
        assert score_to_grade(79) == "C+"

    def test_c(self) -> None:
        assert score_to_grade(73) == "C"
        assert score_to_grade(76) == "C"

    def test_c_minus(self) -> None:
        assert score_to_grade(70) == "C-"
        assert score_to_grade(72) == "C-"

    def test_d_plus(self) -> None:
        assert score_to_grade(67) == "D+"
        assert score_to_grade(69) == "D+"

    def test_d(self) -> None:
        assert score_to_grade(63) == "D"
        assert score_to_grade(66) == "D"

    def test_d_minus(self) -> None:
        assert score_to_grade(60) == "D-"
        assert score_to_grade(62) == "D-"

    def test_f(self) -> None:
        assert score_to_grade(59) == "F"
        assert score_to_grade(0) == "F"
        assert score_to_grade(30) == "F"

    def test_boundary_values(self) -> None:
        """Each threshold boundary should map to the higher grade."""
        assert score_to_grade(96.9) == "A"
        assert score_to_grade(92.9) == "A-"
        assert score_to_grade(89.9) == "B+"


# ---------------------------------------------------------------------------
# Correctness score tests
# ---------------------------------------------------------------------------

class TestCorrectnessScore:
    def test_perfect(self) -> None:
        assert compute_correctness_score(0, 0) == 100

    def test_one_error(self) -> None:
        assert compute_correctness_score(1, 0) == 75

    def test_one_warning(self) -> None:
        assert compute_correctness_score(0, 1) == 90

    def test_mixed(self) -> None:
        assert compute_correctness_score(1, 2) == 55

    def test_floor_at_zero(self) -> None:
        assert compute_correctness_score(5, 5) == 0
        assert compute_correctness_score(10, 0) == 0

    def test_exactly_zero(self) -> None:
        # 4 errors = 100 - 100 = 0
        assert compute_correctness_score(4, 0) == 0

    def test_negative_floors(self) -> None:
        # Would be -25 without floor
        assert compute_correctness_score(5, 0) == 0


# ---------------------------------------------------------------------------
# Efficiency score tests
# ---------------------------------------------------------------------------

class TestEfficiencyScore:
    def test_very_low_tokens(self) -> None:
        assert compute_efficiency_score(10) == 100
        assert compute_efficiency_score(0) == 100

    def test_boundary_50(self) -> None:
        assert compute_efficiency_score(49) == 100
        assert compute_efficiency_score(50) == 100  # < 50 check is strict less-than

    def test_midpoint(self) -> None:
        # At 275 tokens: (275-50)/(500-50) = 225/450 = 0.5 -> 100 - 50 = 50
        assert compute_efficiency_score(275) == 50

    def test_high_tokens(self) -> None:
        assert compute_efficiency_score(500) == 0
        assert compute_efficiency_score(1000) == 0

    def test_linear_scale(self) -> None:
        # At 50: score should be 100
        # At 500: score should be 0
        # At 275: score should be ~50
        s275 = compute_efficiency_score(275)
        assert 45 <= s275 <= 55

    def test_floor_at_zero(self) -> None:
        assert compute_efficiency_score(999) == 0


# ---------------------------------------------------------------------------
# Quality score tests
# ---------------------------------------------------------------------------

class TestQualityScore:
    def test_no_suggestions(self) -> None:
        assert compute_quality_score(0) == 100

    def test_one_suggestion(self) -> None:
        assert compute_quality_score(1) == 85

    def test_multiple_suggestions(self) -> None:
        assert compute_quality_score(3) == 55

    def test_floor_at_zero(self) -> None:
        assert compute_quality_score(7) == 0
        assert compute_quality_score(10) == 0

    def test_exactly_zero(self) -> None:
        # 6 suggestions = 100 - 90 = 10
        assert compute_quality_score(6) == 10
        # 7 suggestions = 100 - 105 -> floor to 0
        assert compute_quality_score(7) == 0


# ---------------------------------------------------------------------------
# Overall score tests
# ---------------------------------------------------------------------------

class TestOverallScore:
    def test_perfect(self) -> None:
        score = compute_overall_score(100, 100, 100)
        assert score == 100.0

    def test_zero(self) -> None:
        score = compute_overall_score(0, 0, 0)
        assert score == 0.0

    def test_weights(self) -> None:
        # 40% correctness + 30% efficiency + 30% quality
        score = compute_overall_score(100, 0, 0)
        assert score == 40.0

        score = compute_overall_score(0, 100, 0)
        assert score == 30.0

        score = compute_overall_score(0, 0, 100)
        assert score == 30.0

    def test_mixed(self) -> None:
        score = compute_overall_score(80, 60, 70)
        # 80*0.4 + 60*0.3 + 70*0.3 = 32 + 18 + 21 = 71
        assert score == 71.0


# ---------------------------------------------------------------------------
# grade_tools integration tests
# ---------------------------------------------------------------------------

class TestGradeTools:
    def test_clean_single_tool(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        assert report["tool_count"] == 1
        assert report["correctness"]["errors"] == 0
        assert report["correctness"]["warnings"] == 0
        assert report["correctness"]["score"] == 100
        assert report["detected_format"] == "anthropic"
        assert report["overall_score"] > 0

    def test_clean_openai_format(self) -> None:
        report = grade_tools(CLEAN_OPENAI_TOOL)
        assert report["detected_format"] == "openai"
        assert report["correctness"]["errors"] == 0

    def test_clean_mcp_format(self) -> None:
        report = grade_tools(CLEAN_MCP_TOOL)
        assert report["detected_format"] == "mcp"
        assert report["correctness"]["errors"] == 0

    def test_broken_tool_lowers_correctness(self) -> None:
        report = grade_tools(BROKEN_TOOL)
        assert report["correctness"]["errors"] > 0
        assert report["correctness"]["score"] < 100

    def test_verbose_tool_lowers_quality(self) -> None:
        report = grade_tools(VERBOSE_TOOL)
        assert report["quality"]["suggestions"] > 0
        assert report["quality"]["score"] < 100

    def test_array_of_tools(self) -> None:
        tools = [CLEAN_ANTHROPIC_TOOL, VERBOSE_TOOL]
        report = grade_tools(tools)
        assert report["tool_count"] == 2

    def test_empty_list(self) -> None:
        report = grade_tools([])
        assert report["tool_count"] == 0
        assert report["total_tokens"] == 0

    def test_overall_grade_present(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        assert report["overall_grade"] in [
            "A+", "A", "A-", "B+", "B", "B-",
            "C+", "C", "C-", "D+", "D", "D-", "F",
        ]

    def test_all_dimensions_present(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        assert "correctness" in report
        assert "efficiency" in report
        assert "quality" in report
        assert "overall_score" in report
        assert "overall_grade" in report
        assert "tool_count" in report
        assert "total_tokens" in report
        assert "detected_format" in report


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestGenerateGradeReport:
    def test_report_contains_grade(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "Overall Grade:" in text
        assert report["overall_grade"] in text

    def test_report_contains_score(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "/100" in text

    def test_report_contains_dimensions(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "Correctness" in text
        assert "Efficiency" in text
        assert "Quality" in text

    def test_report_contains_summary_line(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "Tools:" in text
        assert "Format:" in text
        assert "Tokens:" in text

    def test_report_json_hint(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "--json" in text

    def test_empty_report(self) -> None:
        report = {
            "overall_score": 0,
            "overall_grade": "F",
            "correctness": {"score": 0, "grade": "F", "errors": 0, "warnings": 0},
            "efficiency": {"score": 0, "grade": "F", "avg_tokens_per_tool": 0},
            "quality": {"score": 0, "grade": "F", "suggestions": 0},
            "tool_count": 0,
            "total_tokens": 0,
            "detected_format": "unknown",
        }
        text = generate_grade_report(report, use_color=False)
        assert "No tools found" in text

    def test_report_contains_leaderboard(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "Leaderboard:" in text
        assert "out of" in text and "popular MCP servers" in text
        assert "Your server" in text
        assert "Full leaderboard:" in text

    def test_no_color(self) -> None:
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        text = generate_grade_report(report, use_color=False)
        assert "\033[" not in text


# ---------------------------------------------------------------------------
# _grade_color tests
# ---------------------------------------------------------------------------

class TestGradeColor:
    def test_a_grades_green(self) -> None:
        assert _grade_color("A+", "g", "y", "r") == "g"
        assert _grade_color("A", "g", "y", "r") == "g"
        assert _grade_color("A-", "g", "y", "r") == "g"

    def test_b_grades_green(self) -> None:
        assert _grade_color("B+", "g", "y", "r") == "g"
        assert _grade_color("B", "g", "y", "r") == "g"
        assert _grade_color("B-", "g", "y", "r") == "g"

    def test_c_grades_yellow(self) -> None:
        assert _grade_color("C+", "g", "y", "r") == "y"
        assert _grade_color("C", "g", "y", "r") == "y"
        assert _grade_color("C-", "g", "y", "r") == "y"

    def test_d_grades_red(self) -> None:
        assert _grade_color("D+", "g", "y", "r") == "r"
        assert _grade_color("D", "g", "y", "r") == "r"
        assert _grade_color("D-", "g", "y", "r") == "r"

    def test_f_grade_red(self) -> None:
        assert _grade_color("F", "g", "y", "r") == "r"


# ---------------------------------------------------------------------------
# run_grade integration tests
# ---------------------------------------------------------------------------

class TestRunGrade:
    def _write_temp(self, data: Any) -> str:
        """Write data to a temp file and return the path."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_basic_run(self) -> None:
        path = self._write_temp(CLEAN_ANTHROPIC_TOOL)
        try:
            exit_code = run_grade(path, use_color=False)
            assert exit_code == 0
        finally:
            os.unlink(path)

    def test_json_output(self, capsys) -> None:
        path = self._write_temp(CLEAN_ANTHROPIC_TOOL)
        try:
            exit_code = run_grade(path, use_color=False, json_output=True)
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert "overall_score" in data
            assert "overall_grade" in data
            assert "correctness" in data
            assert "efficiency" in data
            assert "quality" in data
            assert "leaderboard_rank" in data
            assert "leaderboard_total" in data
            assert "leaderboard_url" in data
            assert data["leaderboard_total"] >= 100
            assert isinstance(data["leaderboard_rank"], int)
        finally:
            os.unlink(path)

    def test_threshold_pass(self) -> None:
        path = self._write_temp(CLEAN_ANTHROPIC_TOOL)
        try:
            exit_code = run_grade(path, use_color=False, threshold=50)
            assert exit_code == 0
        finally:
            os.unlink(path)

    def test_threshold_fail(self) -> None:
        path = self._write_temp(BROKEN_TOOL)
        try:
            exit_code = run_grade(path, use_color=False, threshold=95)
            assert exit_code == 2
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        exit_code = run_grade("/nonexistent/file.json", use_color=False)
        assert exit_code == 1

    def test_invalid_json(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("not valid json{{{")
        try:
            exit_code = run_grade(path, use_color=False)
            assert exit_code == 1
        finally:
            os.unlink(path)

    def test_empty_file(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("")
        try:
            exit_code = run_grade(path, use_color=False)
            assert exit_code == 0
        finally:
            os.unlink(path)

    def test_array_input(self) -> None:
        tools = [CLEAN_ANTHROPIC_TOOL, VERBOSE_TOOL]
        path = self._write_temp(tools)
        try:
            exit_code = run_grade(path, use_color=False)
            assert exit_code == 0
        finally:
            os.unlink(path)

    def test_openai_format(self) -> None:
        path = self._write_temp(CLEAN_OPENAI_TOOL)
        try:
            exit_code = run_grade(path, use_color=False, json_output=True)
            assert exit_code == 0
        finally:
            os.unlink(path)

    def test_mcp_format(self) -> None:
        path = self._write_temp(CLEAN_MCP_TOOL)
        try:
            exit_code = run_grade(path, use_color=False, json_output=True)
            assert exit_code == 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_many_errors_floors_correctness(self) -> None:
        """Many errors should floor correctness at 0, not go negative."""
        # Build a tool with many validation errors
        tool = {
            "name": "bad",
            "description": "",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "badtype1"},
                    "b": {"type": "badtype2"},
                    "c": {"type": "badtype3"},
                    "d": {"type": "badtype4"},
                    "e": {"type": "badtype5"},
                },
                "required": ["missing1", "missing2", "missing3"],
            },
        }
        report = grade_tools(tool)
        assert report["correctness"]["score"] == 0
        assert report["overall_score"] >= 0

    def test_many_suggestions_floors_quality(self) -> None:
        """Many optimization suggestions should floor quality at 0."""
        # A tool with lots of verbose patterns
        tools = []
        for i in range(10):
            tools.append({
                "name": "tool_{i}".format(i=i),
                "description": "This tool allows you to " + "x" * 250,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "param": {
                            "type": "string",
                            "description": "y" * 150,
                        },
                    },
                },
            })
        report = grade_tools(tools)
        assert report["quality"]["score"] == 0

    def test_single_tool_dict(self) -> None:
        """Single tool dict (not wrapped in array) should work."""
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        assert report["tool_count"] == 1

    def test_all_perfect_is_a_plus(self) -> None:
        """A clean, minimal tool should score A+ or close to it."""
        report = grade_tools(CLEAN_ANTHROPIC_TOOL)
        # Correctness should be 100 (no errors/warnings)
        assert report["correctness"]["score"] == 100
        # Quality should be 100 (no suggestions for a clean tool)
        assert report["quality"]["score"] == 100
        # Overall should be high
        assert report["overall_score"] >= 90

    def test_json_output_is_valid_json(self, capsys) -> None:
        """JSON output should be parseable."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(CLEAN_ANTHROPIC_TOOL, f)
        try:
            run_grade(path, use_color=False, json_output=True)
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            # Verify structure
            assert isinstance(data["overall_score"], (int, float))
            assert isinstance(data["overall_grade"], str)
            assert isinstance(data["correctness"]["score"], int)
            assert isinstance(data["efficiency"]["score"], int)
            assert isinstance(data["quality"]["score"], int)
        finally:
            os.unlink(path)

    def test_threshold_exactly_at_score(self) -> None:
        """Threshold equal to score should pass (not strictly less)."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(CLEAN_ANTHROPIC_TOOL, f)
        try:
            # Get the score first
            report = grade_tools(CLEAN_ANTHROPIC_TOOL)
            score = report["overall_score"]
            # Threshold at exactly the score should pass
            exit_code = run_grade(path, use_color=False, threshold=int(score))
            assert exit_code == 0
        finally:
            os.unlink(path)


