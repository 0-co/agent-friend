#!/usr/bin/env python3
"""Notion MCP Quality Dashboard — I used Notion MCP to audit Notion MCP.

Takes MCP tool schemas, runs agent-friend's grade analysis, then pushes results
into a Notion database via Notion's own MCP server (stdio). The meta-angle:
Notion's MCP server (Grade F, 19.8/100) is the transport layer for reporting
its own quality failures.

Requires: mcp pip package, npx, NOTION_API_KEY + NOTION_PARENT_PAGE_ID env vars.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_friend.grade import grade_tools
from agent_friend.validate import validate_tools
from agent_friend.audit import parse_tools, generate_json_report

log = logging.getLogger("notion-dashboard")

def _reconstruct_mcp(ft: Any) -> Dict[str, Any]:
    """Rebuild MCP-format dict from a FunctionTool for single-tool grading."""
    schema = ft.input_schema if hasattr(ft, "input_schema") else {"type": "object", "properties": {}}
    return {"name": ft.name, "description": ft.description, "inputSchema": schema}


def analyze_tool_schemas(data: Any) -> Dict[str, Any]:
    """Run grade + validate + audit and return overall report + per-tool breakdown."""
    overall = grade_tools(data)
    issues, stats = validate_tools(data)
    tools = parse_tools(data) if stats.get("tool_count", 0) > 0 else []
    audit = generate_json_report(tools)

    # Per-tool issue counts and worst severity
    issue_counts: Dict[str, int] = {}
    issue_severities: Dict[str, str] = {}
    for issue in issues:
        issue_counts[issue.tool] = issue_counts.get(issue.tool, 0) + 1
        if issue.severity == "error":
            issue_severities[issue.tool] = "Critical"
        elif issue.tool not in issue_severities or issue_severities[issue.tool] == "Low":
            issue_severities[issue.tool] = "Medium"

    token_lookup = {t["name"]: t["tokens"] for t in audit.get("tools", [])}

    per_tool: List[Dict[str, Any]] = []
    for ft in tools:
        try:
            tg = grade_tools([_reconstruct_mcp(ft)])
        except Exception:
            tg = {"overall_score": 0.0, "overall_grade": "F"}
        per_tool.append({
            "name": ft.name,
            "grade": tg["overall_grade"],
            "score": tg["overall_score"],
            "tokens": token_lookup.get(ft.name, 0),
            "issues": issue_counts.get(ft.name, 0),
            "severity": issue_severities.get(ft.name, "Low"),
        })
    return {"overall": overall, "tools": per_tool}

def print_dry_run(analysis: Dict[str, Any], server_name: str) -> None:
    overall = analysis["overall"]
    tools = analysis["tools"]
    print("=== DRY RUN: MCP Quality Dashboard ===\n"
          "Database: 'MCP Quality Dashboard'\n"
          "Server: {}\n"
          "Overall: {} ({}/100)\n"
          "Tools: {}  |  Total tokens: {}\n".format(
              server_name, overall["overall_grade"], overall["overall_score"],
              len(tools), overall["total_tokens"]))
    print("{:<30s} {:>5s} {:>6s} {:>7s} {:>6s} {:>10s}".format(
        "Tool", "Grade", "Score", "Tokens", "Issues", "Severity"))
    print("-" * 70)
    for t in tools:
        print("{:<30s} {:>5s} {:>6.1f} {:>7d} {:>6d} {:>10s}".format(
            t["name"][:30], t["grade"], t["score"],
            t["tokens"], t["issues"], t["severity"]))
    print("\nWould create 1 database + {} pages in Notion.".format(len(tools)))

# Notion MCP interaction via stdio protocol

_GRADE_OPTIONS = [
    {"name": g, "color": c} for g, c in [
        ("A+", "green"), ("A", "green"), ("A-", "green"),
        ("B+", "blue"), ("B", "blue"), ("B-", "blue"),
        ("C+", "yellow"), ("C", "yellow"), ("C-", "yellow"),
        ("D+", "orange"), ("D", "orange"), ("D-", "orange"),
        ("F", "red"),
    ]
]

_SEVERITY_OPTIONS = [
    {"name": "Critical", "color": "red"}, {"name": "High", "color": "orange"},
    {"name": "Medium", "color": "yellow"}, {"name": "Low", "color": "green"},
]

# Notion database column definitions
_DB_PROPERTIES = {
    "Tool Name": {"title": {}},
    "Grade": {"select": {"options": _GRADE_OPTIONS}},
    "Score": {"number": {"format": "number"}},
    "Token Count": {"number": {"format": "number"}},
    "Issues": {"number": {"format": "number"}},
    "Severity": {"select": {"options": _SEVERITY_OPTIONS}},
    "Server Name": {"rich_text": {}},
    "Audit Date": {"date": {}},
}


def _extract_id(result: Any) -> str:
    """Extract entity ID from MCP tool call result (Notion returns JSON text blocks)."""
    for block in result.content:
        if hasattr(block, "text"):
            try:
                data = json.loads(block.text)
                if "id" in data:
                    return data["id"]
            except (json.JSONDecodeError, TypeError):
                continue
    raise RuntimeError("Failed to extract ID from MCP response")


async def create_dashboard(
    analysis: Dict[str, Any], server_name: str, parent_page_id: str,
) -> str:
    """Connect to Notion MCP server via stdio and create the quality dashboard."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={"NOTION_API_KEY": os.environ["NOTION_API_KEY"],
             "PATH": os.environ.get("PATH", "")},
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            log.info("Connected to Notion MCP server")

            # Create database under the parent page
            result = await session.call_tool("create-a-database", arguments={
                "parent": {"type": "page_id", "page_id": parent_page_id},
                "title": [{"type": "text", "text": {"content": "MCP Quality Dashboard"}}],
                "properties": _DB_PROPERTIES,
            })
            db_id = _extract_id(result)
            log.info("Created database: %s", db_id)

            # Insert one row per tool
            today = date.today().isoformat()
            for tool in analysis["tools"]:
                props = {
                    "Tool Name": {"title": [{"text": {"content": tool["name"]}}]},
                    "Grade": {"select": {"name": tool["grade"]}},
                    "Score": {"number": tool["score"]},
                    "Token Count": {"number": tool["tokens"]},
                    "Issues": {"number": tool["issues"]},
                    "Severity": {"select": {"name": tool["severity"]}},
                    "Server Name": {"rich_text": [{"text": {"content": server_name}}]},
                    "Audit Date": {"date": {"start": today}},
                }
                await session.call_tool("post-page", arguments={
                    "parent": {"database_id": db_id},
                    "properties": props,
                })
                log.info("Inserted: %s", tool["name"])

            return db_id

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an MCP Quality Dashboard in Notion from tool schemas.")
    parser.add_argument("schemas", help="Path to JSON file with MCP tool schemas")
    parser.add_argument("--server-name", default="Unknown MCP Server",
                        help="Name of the MCP server being audited")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without connecting to Notion")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output analysis as JSON (implies --dry-run)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # Load schemas
    try:
        with open(args.schemas, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: file not found: {}".format(args.schemas), file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print("Error: invalid JSON: {}".format(e), file=sys.stderr)
        return 1

    log.info("Analyzing %s", args.schemas)
    analysis = analyze_tool_schemas(data)

    if args.json_output:
        print(json.dumps(analysis, indent=2))
        return 0
    if args.dry_run:
        print_dry_run(analysis, args.server_name)
        return 0

    # Live mode: validate env vars
    for var in ("NOTION_API_KEY", "NOTION_PARENT_PAGE_ID"):
        if not os.environ.get(var):
            print("Error: {} environment variable is required".format(var), file=sys.stderr)
            return 1

    try:
        db_id = asyncio.run(create_dashboard(
            analysis, args.server_name, os.environ["NOTION_PARENT_PAGE_ID"]))
    except Exception as e:
        print("Error creating Notion dashboard: {}".format(e), file=sys.stderr)
        log.exception("Full traceback")
        return 1

    overall = analysis["overall"]
    print("Dashboard created: {}".format(db_id))
    print("Overall grade: {} ({}/100)".format(overall["overall_grade"], overall["overall_score"]))
    print("Tools analyzed: {}".format(len(analysis["tools"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
