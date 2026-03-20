"""grade.py — Combined schema quality report card.

Runs validate + audit + optimize on tool definitions and produces a single
letter-grade report. Think of it as a final exam for your tool schemas.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from .audit import detect_format, parse_tools
from .validate import validate_tools
from .optimize import analyze_tools as optimize_analyze
from .toolkit import Toolkit


# ---------------------------------------------------------------------------
# Letter grade mapping
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS = [
    (97, "A+"),
    (93, "A"),
    (90, "A-"),
    (87, "B+"),
    (83, "B"),
    (80, "B-"),
    (77, "C+"),
    (73, "C"),
    (70, "C-"),
    (67, "D+"),
    (63, "D"),
    (60, "D-"),
]


def score_to_grade(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def compute_correctness_score(errors: int, warnings: int) -> int:
    """Compute correctness score: 100 - 25*errors - 10*warnings, floor 0."""
    score = 100 - (25 * errors) - (10 * warnings)
    return max(0, score)


def compute_efficiency_score(avg_tokens_per_tool: float) -> int:
    """Compute efficiency score: 100 if avg < 50, linear to 0 at 500+."""
    if avg_tokens_per_tool < 50:
        return 100
    if avg_tokens_per_tool >= 500:
        return 0
    # Linear interpolation: 100 at 50, 0 at 500
    score = 100 - ((avg_tokens_per_tool - 50) / (500 - 50)) * 100
    return max(0, int(round(score)))


def compute_quality_score(suggestion_count: int) -> int:
    """Compute quality score: 100 - 15*suggestions, floor 0."""
    score = 100 - (15 * suggestion_count)
    return max(0, score)


def compute_overall_score(
    correctness: int,
    efficiency: int,
    quality: int,
) -> float:
    """Compute weighted overall score (40/30/30)."""
    return (correctness * 0.4) + (efficiency * 0.3) + (quality * 0.3)


# ---------------------------------------------------------------------------
# Grade report data
# ---------------------------------------------------------------------------

def grade_tools(data: Any) -> Dict[str, Any]:
    """Run all three analyses and compute grade report data.

    Parameters
    ----------
    data:
        Parsed JSON data (dict or list of tool definitions).

    Returns
    -------
    Dict with all grade report data including scores, grade, and details.
    """
    # Run validate
    issues, validate_stats = validate_tools(data)
    errors = validate_stats.get("errors", 0)
    warnings = validate_stats.get("warnings", 0)
    tool_count = validate_stats.get("tool_count", 0)

    # Run audit (parse tools for token info)
    tools = parse_tools(data) if tool_count > 0 else []
    total_tokens = 0
    avg_tokens = 0.0
    detected_format = "unknown"

    if tools:
        kit = Toolkit(tools)
        total_tokens = kit.token_estimate(format="anthropic")
        avg_tokens = total_tokens / len(tools) if tools else 0.0

        # Detect format from first item
        items = [data] if isinstance(data, dict) else data
        if items:
            try:
                detected_format = detect_format(items[0])
            except ValueError:
                detected_format = "unknown"

    # Run optimize
    suggestions, optimize_stats = optimize_analyze(data) if tool_count > 0 else ([], {})
    suggestion_count = len(suggestions)

    # Compute scores
    correctness = compute_correctness_score(errors, warnings)
    efficiency = compute_efficiency_score(avg_tokens)
    quality = compute_quality_score(suggestion_count)
    overall = compute_overall_score(correctness, efficiency, quality)

    return {
        "overall_score": round(overall, 1),
        "overall_grade": score_to_grade(overall),
        "correctness": {
            "score": correctness,
            "grade": score_to_grade(correctness),
            "errors": errors,
            "warnings": warnings,
        },
        "efficiency": {
            "score": efficiency,
            "grade": score_to_grade(efficiency),
            "avg_tokens_per_tool": round(avg_tokens, 1),
        },
        "quality": {
            "score": quality,
            "grade": score_to_grade(quality),
            "suggestions": suggestion_count,
        },
        "tool_count": tool_count,
        "total_tokens": total_tokens,
        "detected_format": detected_format,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_grade_report(
    report: Dict[str, Any],
    *,
    use_color: bool = True,
) -> str:
    """Generate a formatted grade report card.

    Returns the report as a string (with ANSI escapes if use_color is True).
    """
    if use_color and sys.stderr.isatty():
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        GRAY = "\033[90m"
        RESET = "\033[0m"
    else:
        BOLD = CYAN = GREEN = YELLOW = RED = GRAY = RESET = ""

    lines: List[str] = []
    lines.append("")
    lines.append("{bold}agent-friend grade{reset} — schema quality report card".format(
        bold=BOLD, reset=RESET,
    ))

    tool_count = report.get("tool_count", 0)

    if tool_count == 0:
        lines.append("")
        lines.append("  {gray}No tools found in input.{reset}".format(
            gray=GRAY, reset=RESET,
        ))
        lines.append("")
        return "\n".join(lines)

    overall_grade = report["overall_grade"]
    overall_score = report["overall_score"]

    # Pick color for overall grade
    grade_color = _grade_color(overall_grade, GREEN, YELLOW, RED)

    lines.append("")
    lines.append("  Overall Grade: {color}{bold}{grade}{reset}".format(
        color=grade_color, bold=BOLD, grade=overall_grade, reset=RESET,
    ))
    lines.append("  Score: {color}{score}/100{reset}".format(
        color=grade_color, score=overall_score, reset=RESET,
    ))

    lines.append("")

    # Correctness row
    c = report["correctness"]
    c_color = _grade_color(c["grade"], GREEN, YELLOW, RED)
    c_detail = "{errors} error{es}, {warnings} warning{ws}".format(
        errors=c["errors"],
        es="s" if c["errors"] != 1 else "",
        warnings=c["warnings"],
        ws="s" if c["warnings"] != 1 else "",
    )
    lines.append("  {label:<14s}{color}{grade:<4s}{reset}({score}/100)  {gray}{detail}{reset}".format(
        label="Correctness",
        color=c_color,
        grade=c["grade"],
        reset=RESET,
        score=c["score"],
        gray=GRAY,
        detail=c_detail,
    ))

    # Efficiency row
    e = report["efficiency"]
    e_color = _grade_color(e["grade"], GREEN, YELLOW, RED)
    e_detail = "avg {avg} tokens/tool".format(
        avg=int(e["avg_tokens_per_tool"]),
    )
    lines.append("  {label:<14s}{color}{grade:<4s}{reset}({score}/100)  {gray}{detail}{reset}".format(
        label="Efficiency",
        color=e_color,
        grade=e["grade"],
        reset=RESET,
        score=e["score"],
        gray=GRAY,
        detail=e_detail,
    ))

    # Quality row
    q = report["quality"]
    q_color = _grade_color(q["grade"], GREEN, YELLOW, RED)
    q_detail = "{n} suggestion{s}".format(
        n=q["suggestions"],
        s="s" if q["suggestions"] != 1 else "",
    )
    lines.append("  {label:<14s}{color}{grade:<4s}{reset}({score}/100)  {gray}{detail}{reset}".format(
        label="Quality",
        color=q_color,
        grade=q["grade"],
        reset=RESET,
        score=q["score"],
        gray=GRAY,
        detail=q_detail,
    ))

    lines.append("")
    lines.append("  {gray}Tools: {tools} | Format: {fmt} | Tokens: {tokens}{reset}".format(
        gray=GRAY,
        tools=report["tool_count"],
        fmt=report["detected_format"],
        tokens=report["total_tokens"],
        reset=RESET,
    ))

    # Leaderboard ranking
    from .leaderboard_data import get_leaderboard_position, LEADERBOARD_URL
    rank, total, servers_above, servers_below = get_leaderboard_position(overall_score)
    lines.append("")
    lines.append("  Leaderboard: {cyan}#{rank}{reset} out of {total} popular MCP servers".format(
        cyan=CYAN, rank=rank, reset=RESET, total=total,
    ))
    for name, s in servers_above:
        lines.append("    {gray}{arrow} {name} ({score}){reset}".format(
            gray=GRAY, arrow="\u2191", name=name, score=s, reset=RESET,
        ))
    lines.append("    {arrow} Your server ({score})".format(
        arrow="\u2192", score=overall_score,
    ))
    for name, s in servers_below:
        lines.append("    {gray}{arrow} {name} ({score}){reset}".format(
            gray=GRAY, arrow="\u2193", name=name, score=s, reset=RESET,
        ))
    lines.append("  Full leaderboard: {cyan}{url}{reset}".format(
        cyan=CYAN, url=LEADERBOARD_URL, reset=RESET,
    ))

    lines.append("")
    lines.append("  {gray}Use --json for machine-readable output.{reset}".format(
        gray=GRAY, reset=RESET,
    ))
    lines.append("")

    return "\n".join(lines)


def _grade_color(grade: str, green: str, yellow: str, red: str) -> str:
    """Pick a color based on the letter grade."""
    if grade.startswith("A") or grade.startswith("B"):
        return green
    if grade.startswith("C"):
        return yellow
    return red


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_grade(
    file_path: Optional[str] = None,
    use_color: bool = True,
    json_output: bool = False,
    threshold: Optional[int] = None,
) -> int:
    """Run the grade command. Returns exit code.

    Exit codes:
        0 = success
        1 = file read / parse error
        2 = threshold not met

    Parameters
    ----------
    file_path:
        Path to a JSON file, or "-" for stdin, or None to read from stdin.
    use_color:
        Whether to use ANSI color codes in output.
    json_output:
        If True, output JSON instead of colored text.
    threshold:
        If set, exit with code 2 when overall score is below this value.
    """
    # Read input
    try:
        if file_path is None or file_path == "-":
            raw = sys.stdin.read()
        else:
            with open(file_path, "r") as f:
                raw = f.read()
    except FileNotFoundError:
        print("Error: file not found: {path}".format(path=file_path), file=sys.stderr)
        return 1
    except Exception as e:
        print("Error reading input: {err}".format(err=e), file=sys.stderr)
        return 1

    raw = raw.strip()
    if not raw:
        empty_report = {
            "overall_score": 0,
            "overall_grade": "F",
            "correctness": {"score": 0, "grade": "F", "errors": 0, "warnings": 0},
            "efficiency": {"score": 0, "grade": "F", "avg_tokens_per_tool": 0},
            "quality": {"score": 0, "grade": "F", "suggestions": 0},
            "tool_count": 0,
            "total_tokens": 0,
            "detected_format": "unknown",
        }
        if json_output:
            print(json.dumps(empty_report, indent=2))
        else:
            print(generate_grade_report(empty_report, use_color=use_color))
        return 0

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Error: invalid JSON: {err}".format(err=e), file=sys.stderr)
        return 1

    # Run grade analysis
    try:
        report = grade_tools(data)
    except ValueError as e:
        print("Error: {err}".format(err=e), file=sys.stderr)
        return 1

    # Add leaderboard position
    from .leaderboard_data import get_leaderboard_position, LEADERBOARD_URL
    rank, total, above, below = get_leaderboard_position(report["overall_score"])
    report["leaderboard_rank"] = rank
    report["leaderboard_total"] = total
    report["leaderboard_url"] = LEADERBOARD_URL

    # Output
    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print(generate_grade_report(report, use_color=use_color))

    # Threshold check
    if threshold is not None:
        if report["overall_score"] < threshold:
            print(
                "Threshold not met: {score} < {threshold}".format(
                    score=report["overall_score"],
                    threshold=threshold,
                ),
                file=sys.stderr,
            )
            return 2

    return 0
