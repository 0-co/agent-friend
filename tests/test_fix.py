"""Tests for the agent-friend fix CLI subcommand and fix module."""

import json
import io
import os
import sys
import tempfile

import pytest

from agent_friend.fix import (
    Change,
    fix_tools,
    run_fix,
    generate_fix_report,
    generate_diff_report,
    _fix_names,
    _fix_undefined_schemas,
    _fix_verbose_prefixes,
    _fix_redundant_params,
    _fix_long_descriptions,
    _fix_long_param_descriptions,
    _truncate_at_sentence,
    _token_estimate,
)


# ---------------------------------------------------------------------------
# Sample tool definitions
# ---------------------------------------------------------------------------

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

KEBAB_NAME_TOOL = {
    "name": "create-page",
    "description": "Create a new page.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Page title"},
        },
    },
}

UPPERCASE_NAME_TOOL = {
    "name": "GET_Weather",
    "description": "Get the weather.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
    },
}

UNDEFINED_SCHEMA_TOOL = {
    "name": "create_page",
    "description": "Create a page.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "options": {"type": "object"},
            "title": {"type": "string", "description": "Page title"},
        },
    },
}

VERBOSE_PREFIX_TOOL = {
    "name": "search_pages",
    "description": "This tool allows you to search for pages in the database.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
    },
}

REDUNDANT_PARAM_TOOL = {
    "name": "search",
    "description": "Search for items.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The query."},
            "name": {"type": "string", "description": "The name"},
            "filter": {"type": "string", "description": "SQL filter expression"},
        },
    },
}

LONG_DESC_TOOL = {
    "name": "get_database",
    "description": (
        "Retrieve the full database schema including all properties, relations, "
        "and configuration settings. This endpoint returns detailed information "
        "about each property including its type, options, and relationships to "
        "other databases. Use this to understand the structure before queries."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "database_id": {"type": "string", "description": "Database ID"},
        },
    },
}

LONG_PARAM_DESC_TOOL = {
    "name": "create_entry",
    "description": "Create a new entry.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The full content of the entry including all formatted text, "
                    "inline references, and metadata that should be stored with the entry."
                ),
            },
        },
    },
}

# Multi-issue tool
MULTI_ISSUE_TOOL = {
    "name": "create-item",
    "description": "This tool allows you to create a new item in the database with various properties and settings. " + "A" * 150,
    "inputSchema": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The name."},
            "config": {"type": "object"},
        },
    },
}


# ---------------------------------------------------------------------------
# Tools in different formats
# ---------------------------------------------------------------------------

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": "create-record",
        "description": "This tool allows you to create a record.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "object"},
                "name": {"type": "string", "description": "The name"},
            },
        },
    },
}

ANTHROPIC_TOOL = {
    "name": "update-page",
    "description": "A tool that updates a page.",
    "input_schema": {
        "type": "object",
        "properties": {
            "page_id": {"type": "string", "description": "The page id."},
            "settings": {"type": "object"},
        },
    },
}

MCP_TOOL = {
    "name": "delete-block",
    "description": "Used to delete a block from the page.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "block_id": {"type": "string", "description": "Block identifier"},
        },
    },
}

GOOGLE_TOOL = {
    "type": "object",
    "title": "search-items",
    "description": "This function searches for items.",
    "properties": {
        "query": {"type": "string", "description": "The query."},
    },
}

SIMPLE_TOOL = {
    "name": "list-users",
    "description": "Allows the user to list all users.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "The limit"},
        },
    },
}


# ---------------------------------------------------------------------------
# Fix Rule 1: fix_names
# ---------------------------------------------------------------------------


class TestFixNames:
    def test_kebab_to_snake(self):
        tool = {"name": "create-page", "description": "Create.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_names(tool, "mcp")
        assert len(changes) == 1
        assert tool["name"] == "create_page"
        assert changes[0].rule == "fix_names"
        assert "create-page -> create_page" in changes[0].message

    def test_uppercase_to_lower(self):
        tool = {"name": "GET_Weather", "description": "Get.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_names(tool, "mcp")
        assert len(changes) == 1
        assert tool["name"] == "get_weather"

    def test_combined_kebab_and_upper(self):
        tool = {"name": "Get-User-Data", "description": "Get.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_names(tool, "mcp")
        assert len(changes) == 1
        assert tool["name"] == "get_user_data"

    def test_clean_name_no_change(self):
        tool = {"name": "get_weather", "description": "Get.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_names(tool, "mcp")
        assert len(changes) == 0
        assert tool["name"] == "get_weather"

    def test_openai_format(self):
        tool = {"type": "function", "function": {"name": "create-record", "description": "Create.", "parameters": {"type": "object", "properties": {}}}}
        changes = _fix_names(tool, "openai")
        assert len(changes) == 1
        assert tool["function"]["name"] == "create_record"

    def test_json_schema_format(self):
        tool = {"type": "object", "title": "Search-Items", "description": "Search.", "properties": {}}
        changes = _fix_names(tool, "json_schema")
        assert len(changes) == 1
        assert tool["title"] == "search_items"


# ---------------------------------------------------------------------------
# Fix Rule 2: fix_undefined_schemas
# ---------------------------------------------------------------------------


class TestFixUndefinedSchemas:
    def test_adds_properties_to_object(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config": {"type": "object"},
                },
            },
        }
        changes = _fix_undefined_schemas(tool, "mcp")
        assert len(changes) == 1
        assert tool["inputSchema"]["properties"]["config"]["properties"] == {}
        assert "config" in changes[0].message

    def test_already_has_properties(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config": {"type": "object", "properties": {"key": {"type": "string"}}},
                },
            },
        }
        changes = _fix_undefined_schemas(tool, "mcp")
        assert len(changes) == 0

    def test_non_object_type_ignored(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
        }
        changes = _fix_undefined_schemas(tool, "mcp")
        assert len(changes) == 0

    def test_openai_format(self):
        tool = {
            "type": "function",
            "function": {
                "name": "test",
                "description": "Test.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "opts": {"type": "object"},
                    },
                },
            },
        }
        changes = _fix_undefined_schemas(tool, "openai")
        assert len(changes) == 1
        assert tool["function"]["parameters"]["properties"]["opts"]["properties"] == {}


# ---------------------------------------------------------------------------
# Fix Rule 3: fix_verbose_prefixes
# ---------------------------------------------------------------------------


class TestFixVerbosePrefixes:
    def test_strips_this_tool_allows(self):
        tool = {"name": "search", "description": "This tool allows you to search the database.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Search the database."

    def test_strips_use_this_tool_to(self):
        tool = {"name": "fetch", "description": "Use this tool to fetch records.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Fetch records."

    def test_strips_a_tool_that(self):
        tool = {"name": "proc", "description": "A tool that processes data.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Processes data."

    def test_strips_this_function(self):
        tool = {"name": "get", "description": "This function retrieves data.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Retrieves data."

    def test_strips_allows_the_user_to(self):
        tool = {"name": "edit", "description": "Allows the user to edit documents.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Edit documents."

    def test_strips_used_to(self):
        tool = {"name": "transform", "description": "Used to transform input.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Transform input."

    def test_strips_this_is_a_tool_that(self):
        tool = {"name": "do", "description": "This is a tool that does things.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Does things."

    def test_clean_description_no_change(self):
        tool = {"name": "search", "description": "Search the database.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 0
        assert tool["description"] == "Search the database."

    def test_case_insensitive(self):
        tool = {"name": "search", "description": "this tool allows you to search.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"] == "Search."

    def test_capitalizes_first_letter(self):
        tool = {"name": "search", "description": "This tool allows you to search items.", "inputSchema": {"type": "object", "properties": {}}}
        changes = _fix_verbose_prefixes(tool, "mcp")
        assert tool["description"] == "Search items."

    def test_openai_format(self):
        tool = {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Use this tool to fetch data.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        changes = _fix_verbose_prefixes(tool, "openai")
        assert len(changes) == 1
        assert tool["function"]["description"] == "Fetch data."


# ---------------------------------------------------------------------------
# Fix Rule 4: fix_redundant_params
# ---------------------------------------------------------------------------


class TestFixRedundantParams:
    def test_removes_exact_match(self):
        tool = {
            "name": "search",
            "description": "Search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "query"},
                },
            },
        }
        changes = _fix_redundant_params(tool, "mcp")
        assert len(changes) == 1
        assert "description" not in tool["inputSchema"]["properties"]["query"]

    def test_removes_with_article(self):
        tool = {
            "name": "search",
            "description": "Search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query."},
                },
            },
        }
        changes = _fix_redundant_params(tool, "mcp")
        assert len(changes) == 1
        assert "description" not in tool["inputSchema"]["properties"]["query"]

    def test_removes_underscore_name(self):
        tool = {
            "name": "tool",
            "description": "A tool.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_name": {"type": "string", "description": "The user name."},
                },
            },
        }
        changes = _fix_redundant_params(tool, "mcp")
        assert len(changes) == 1

    def test_keeps_meaningful_description(self):
        tool = {
            "name": "search",
            "description": "Search.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                },
            },
        }
        changes = _fix_redundant_params(tool, "mcp")
        assert len(changes) == 0
        assert tool["inputSchema"]["properties"]["query"]["description"] == "SQL query to execute"

    def test_multiple_params(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name"},
                    "id": {"type": "string", "description": "The ID"},
                    "filter": {"type": "string", "description": "SQL filter expression"},
                },
            },
        }
        changes = _fix_redundant_params(tool, "mcp")
        assert len(changes) == 2  # name and id are redundant, filter is not


# ---------------------------------------------------------------------------
# Fix Rule 5: fix_long_descriptions
# ---------------------------------------------------------------------------


class TestFixLongDescriptions:
    def test_truncates_long_description(self):
        tool = {
            "name": "test",
            "description": "A" * 250,
            "inputSchema": {"type": "object", "properties": {}},
        }
        changes = _fix_long_descriptions(tool, "mcp")
        assert len(changes) == 1
        assert len(tool["description"]) <= 200
        assert "250 -> " in changes[0].message

    def test_keeps_short_description(self):
        tool = {
            "name": "test",
            "description": "Short description.",
            "inputSchema": {"type": "object", "properties": {}},
        }
        changes = _fix_long_descriptions(tool, "mcp")
        assert len(changes) == 0

    def test_boundary_200_chars(self):
        tool = {
            "name": "test",
            "description": "A" * 200,
            "inputSchema": {"type": "object", "properties": {}},
        }
        changes = _fix_long_descriptions(tool, "mcp")
        assert len(changes) == 0

    def test_truncates_at_sentence_boundary(self):
        desc = "First sentence. " + "A" * 200
        tool = {
            "name": "test",
            "description": desc,
            "inputSchema": {"type": "object", "properties": {}},
        }
        changes = _fix_long_descriptions(tool, "mcp")
        assert len(changes) == 1
        # Should truncate at the period after "First sentence."
        assert tool["description"].endswith(".")
        assert len(tool["description"]) <= 200

    def test_truncates_with_ellipsis_no_sentence(self):
        desc = "A" * 250  # No sentence boundary
        tool = {
            "name": "test",
            "description": desc,
            "inputSchema": {"type": "object", "properties": {}},
        }
        changes = _fix_long_descriptions(tool, "mcp")
        assert len(changes) == 1
        assert tool["description"].endswith("...")
        assert len(tool["description"]) == 200


# ---------------------------------------------------------------------------
# Fix Rule 6: fix_long_param_descriptions
# ---------------------------------------------------------------------------


class TestFixLongParamDescriptions:
    def test_truncates_long_param(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "B" * 120},
                },
            },
        }
        changes = _fix_long_param_descriptions(tool, "mcp")
        assert len(changes) == 1
        assert len(tool["inputSchema"]["properties"]["content"]["description"]) <= 80

    def test_keeps_short_param(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short."},
                },
            },
        }
        changes = _fix_long_param_descriptions(tool, "mcp")
        assert len(changes) == 0

    def test_boundary_80_chars(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": "C" * 80},
                },
            },
        }
        changes = _fix_long_param_descriptions(tool, "mcp")
        assert len(changes) == 0

    def test_boundary_81_chars(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": "C" * 81},
                },
            },
        }
        changes = _fix_long_param_descriptions(tool, "mcp")
        assert len(changes) == 1


# ---------------------------------------------------------------------------
# truncate_at_sentence helper
# ---------------------------------------------------------------------------


class TestTruncateAtSentence:
    def test_short_text_unchanged(self):
        assert _truncate_at_sentence("Short.", 200) == "Short."

    def test_truncates_at_period(self):
        text = "First sentence. Second sentence that is very long." + "x" * 200
        result = _truncate_at_sentence(text, 50)
        assert result == "First sentence. Second sentence that is very long."

    def test_truncates_at_question_mark(self):
        text = "Is this a question? " + "x" * 200
        result = _truncate_at_sentence(text, 30)
        assert result == "Is this a question?"

    def test_truncates_at_exclamation(self):
        text = "Warning! " + "x" * 200
        result = _truncate_at_sentence(text, 20)
        assert result == "Warning!"

    def test_no_sentence_boundary_adds_ellipsis(self):
        text = "A" * 250
        result = _truncate_at_sentence(text, 200)
        assert result.endswith("...")
        assert len(result) == 200


# ---------------------------------------------------------------------------
# fix_tools() integration
# ---------------------------------------------------------------------------


class TestFixTools:
    def test_clean_tool_no_changes(self):
        fixed, changes = fix_tools(CLEAN_MCP_TOOL)
        assert len(changes) == 0
        assert fixed == CLEAN_MCP_TOOL

    def test_kebab_name_fixed(self):
        fixed, changes = fix_tools(KEBAB_NAME_TOOL)
        rules = [c.rule for c in changes]
        assert "fix_names" in rules
        assert fixed["name"] == "create_page"

    def test_verbose_prefix_fixed(self):
        fixed, changes = fix_tools(VERBOSE_PREFIX_TOOL)
        rules = [c.rule for c in changes]
        assert "fix_verbose_prefixes" in rules
        assert not fixed["description"].startswith("This tool allows you to")

    def test_redundant_params_fixed(self):
        fixed, changes = fix_tools(REDUNDANT_PARAM_TOOL)
        rules = [c.rule for c in changes]
        assert "fix_redundant_params" in rules
        # "The query." should be removed, but "SQL filter expression" should stay
        assert "description" not in fixed["inputSchema"]["properties"]["query"]
        assert "description" not in fixed["inputSchema"]["properties"]["name"]
        assert fixed["inputSchema"]["properties"]["filter"]["description"] == "SQL filter expression"

    def test_undefined_schema_fixed(self):
        fixed, changes = fix_tools(UNDEFINED_SCHEMA_TOOL)
        rules = [c.rule for c in changes]
        assert "fix_undefined_schemas" in rules
        assert fixed["inputSchema"]["properties"]["options"]["properties"] == {}

    def test_long_description_fixed(self):
        fixed, changes = fix_tools(LONG_DESC_TOOL)
        rules = [c.rule for c in changes]
        assert "fix_long_descriptions" in rules
        assert len(fixed["description"]) <= 200

    def test_long_param_description_fixed(self):
        fixed, changes = fix_tools(LONG_PARAM_DESC_TOOL)
        rules = [c.rule for c in changes]
        assert "fix_long_param_descriptions" in rules
        assert len(fixed["inputSchema"]["properties"]["content"]["description"]) <= 80

    def test_multi_issue_tool(self):
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL)
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_verbose_prefixes" in rules
        assert "fix_long_descriptions" in rules
        assert "fix_redundant_params" in rules
        assert "fix_undefined_schemas" in rules

    def test_does_not_mutate_original(self):
        import copy
        original = copy.deepcopy(MULTI_ISSUE_TOOL)
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL)
        assert MULTI_ISSUE_TOOL == original

    def test_empty_list(self):
        fixed, changes = fix_tools([])
        assert len(changes) == 0
        assert fixed == []

    def test_single_dict_input(self):
        fixed, changes = fix_tools(KEBAB_NAME_TOOL)
        assert isinstance(fixed, dict)
        assert fixed["name"] == "create_page"

    def test_list_input(self):
        fixed, changes = fix_tools([KEBAB_NAME_TOOL, VERBOSE_PREFIX_TOOL])
        assert isinstance(fixed, list)
        assert len(fixed) == 2
        assert fixed[0]["name"] == "create_page"

    def test_only_filter_names(self):
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL, only=["names"])
        rules = set(c.rule for c in changes)
        assert rules == {"fix_names"}

    def test_only_filter_prefixes(self):
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL, only=["prefixes"])
        rules = set(c.rule for c in changes)
        assert rules == {"fix_verbose_prefixes"}

    def test_only_filter_multiple(self):
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL, only=["names", "schemas"])
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_undefined_schemas" in rules
        # Should not have prefix or redundant changes
        assert "fix_verbose_prefixes" not in rules
        assert "fix_redundant_params" not in rules

    def test_only_filter_full_rule_name(self):
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL, only=["fix_names"])
        rules = set(c.rule for c in changes)
        assert rules == {"fix_names"}

    def test_already_clean_tools(self):
        clean = {
            "name": "get_weather",
            "description": "Get the weather.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name to query"},
                },
            },
        }
        fixed, changes = fix_tools(clean)
        assert len(changes) == 0
        assert fixed == clean


# ---------------------------------------------------------------------------
# Input format tests
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_openai_format(self):
        fixed, changes = fix_tools(OPENAI_TOOL)
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_verbose_prefixes" in rules
        assert "fix_undefined_schemas" in rules
        assert "fix_redundant_params" in rules
        # Verify the structure is preserved
        assert "function" in fixed
        assert fixed["type"] == "function"
        assert fixed["function"]["name"] == "create_record"

    def test_anthropic_format(self):
        fixed, changes = fix_tools(ANTHROPIC_TOOL)
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_verbose_prefixes" in rules
        assert "fix_undefined_schemas" in rules
        assert "input_schema" in fixed
        assert fixed["name"] == "update_page"

    def test_mcp_format(self):
        fixed, changes = fix_tools(MCP_TOOL)
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_verbose_prefixes" in rules
        assert "inputSchema" in fixed
        assert fixed["name"] == "delete_block"

    def test_google_json_schema_format(self):
        fixed, changes = fix_tools(GOOGLE_TOOL)
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_verbose_prefixes" in rules
        assert "fix_redundant_params" in rules
        assert fixed["title"] == "search_items"

    def test_simple_format(self):
        fixed, changes = fix_tools(SIMPLE_TOOL)
        rules = set(c.rule for c in changes)
        assert "fix_names" in rules
        assert "fix_verbose_prefixes" in rules
        assert fixed["name"] == "list_users"


# ---------------------------------------------------------------------------
# Format preservation
# ---------------------------------------------------------------------------


class TestFormatPreservation:
    def test_openai_structure_preserved(self):
        fixed, _ = fix_tools(OPENAI_TOOL)
        assert fixed["type"] == "function"
        assert "function" in fixed
        assert "name" in fixed["function"]
        assert "description" in fixed["function"]
        assert "parameters" in fixed["function"]

    def test_anthropic_structure_preserved(self):
        fixed, _ = fix_tools(ANTHROPIC_TOOL)
        assert "name" in fixed
        assert "description" in fixed
        assert "input_schema" in fixed

    def test_mcp_structure_preserved(self):
        fixed, _ = fix_tools(MCP_TOOL)
        assert "name" in fixed
        assert "description" in fixed
        assert "inputSchema" in fixed

    def test_json_schema_structure_preserved(self):
        fixed, _ = fix_tools(GOOGLE_TOOL)
        assert "title" in fixed
        assert "type" in fixed
        assert fixed["type"] == "object"
        assert "properties" in fixed


# ---------------------------------------------------------------------------
# Fixes don't break validate
# ---------------------------------------------------------------------------


class TestFixesPassValidate:
    def test_fixed_tools_pass_validate(self):
        """Run validate on fixed output to ensure no new issues introduced."""
        from agent_friend.validate import validate_tools

        # Fix a multi-issue tool
        fixed, changes = fix_tools(MULTI_ISSUE_TOOL)
        assert len(changes) > 0

        # Validate the fixed output
        issues, stats = validate_tools(fixed)
        # The fixed tool should have fewer/no issues for name_valid
        name_issues = [i for i in issues if i.check == "name_valid"]
        assert len(name_issues) == 0

    def test_fixed_notion_passes_name_check(self):
        """Notion tools have kebab names; fixing should eliminate name_valid warnings."""
        from agent_friend.validate import validate_tools
        from agent_friend.examples import get_example

        notion_tools = get_example("notion")
        fixed, changes = fix_tools(notion_tools)

        issues, stats = validate_tools(fixed)
        name_issues = [i for i in issues if i.check == "name_valid"]
        assert len(name_issues) == 0


# ---------------------------------------------------------------------------
# Bundled examples
# ---------------------------------------------------------------------------


class TestBundledExamples:
    def test_notion_example_has_fixes(self):
        """Notion example should have many fixes (it grades F)."""
        from agent_friend.examples import get_example

        notion_tools = get_example("notion")
        fixed, changes = fix_tools(notion_tools)
        assert len(changes) > 0

        # Should have name fixes (all notion tools use kebab-case)
        name_changes = [c for c in changes if c.rule == "fix_names"]
        assert len(name_changes) > 0

    def test_slack_example_few_fixes(self):
        """Slack example grades A+, should have very few fixes."""
        from agent_friend.examples import get_example

        slack_tools = get_example("slack")
        fixed, changes = fix_tools(slack_tools)
        # Slack is already well-formed, so few or no changes
        # (it might still have some long descriptions)
        name_changes = [c for c in changes if c.rule == "fix_names"]
        assert len(name_changes) == 0  # slack uses snake_case already

    def test_filesystem_example(self):
        from agent_friend.examples import get_example

        fs_tools = get_example("filesystem")
        fixed, changes = fix_tools(fs_tools)
        # Just verify it runs without error
        assert isinstance(fixed, list)

    def test_puppeteer_example(self):
        from agent_friend.examples import get_example

        puppeteer_tools = get_example("puppeteer")
        fixed, changes = fix_tools(puppeteer_tools)
        assert isinstance(fixed, list)

    def test_github_example(self):
        from agent_friend.examples import get_example

        github_tools = get_example("github")
        fixed, changes = fix_tools(github_tools)
        assert isinstance(fixed, list)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_token_estimate_basic(self):
        data = {"name": "test", "description": "Test tool."}
        tokens = _token_estimate(data)
        assert tokens > 0
        assert tokens == len(json.dumps(data)) // 4

    def test_fixes_reduce_tokens(self):
        """Fixing a verbose tool should reduce token count."""
        original_tokens = _token_estimate(VERBOSE_PREFIX_TOOL)
        fixed, changes = fix_tools(VERBOSE_PREFIX_TOOL)
        fixed_tokens = _token_estimate(fixed)
        assert fixed_tokens < original_tokens


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReport:
    def test_empty_report(self):
        report = generate_fix_report([], 0, 0, 0, "test.json", use_color=False)
        assert "No tools found" in report

    def test_clean_report(self):
        report = generate_fix_report([], 3, 100, 100, "test.json", "mcp", use_color=False)
        assert "No fixes needed" in report

    def test_report_with_changes(self):
        changes = [
            Change("create_page", "fix_names", "create-page -> create_page (name)"),
            Change("search", "fix_verbose_prefixes", 'Stripped "This tool allows you to" from search description'),
        ]
        report = generate_fix_report(
            changes, 5, 500, 450, "tools.json", "mcp", use_color=False,
        )
        assert "agent-friend fix" in report
        assert "create-page -> create_page" in report
        assert "2 fixes applied" in report
        assert "500 -> 450 tokens" in report

    def test_report_no_ansi_when_disabled(self):
        changes = [Change("test", "fix_names", "test change")]
        report = generate_fix_report(changes, 1, 100, 90, "test.json", use_color=False)
        assert "\033[" not in report

    def test_diff_report(self):
        original = {"name": "create-page", "description": "Old desc."}
        fixed = {"name": "create_page", "description": "New desc."}
        changes = [Change("create_page", "fix_names", "name fix")]
        diff = generate_diff_report(original, fixed, changes, use_color=False)
        assert "Diff:" in diff


# ---------------------------------------------------------------------------
# Change class
# ---------------------------------------------------------------------------


class TestChange:
    def test_to_dict(self):
        c = Change("tool1", "fix_names", "test message")
        d = c.to_dict()
        assert d["tool"] == "tool1"
        assert d["rule"] == "fix_names"
        assert d["message"] == "test message"


# ---------------------------------------------------------------------------
# run_fix() — file and stdin handling
# ---------------------------------------------------------------------------


class TestRunFix:
    def test_file_input(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_fix(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            # Should output fixed JSON to stdout
            fixed = json.loads(out)
            assert fixed["name"] == "create_page"
        finally:
            os.unlink(path)

    def test_stdin_input(self, monkeypatch, capsys):
        data = json.dumps(KEBAB_NAME_TOOL)
        monkeypatch.setattr("sys.stdin", io.StringIO(data))
        code = run_fix("-", use_color=False)
        assert code == 0
        out = capsys.readouterr().out
        fixed = json.loads(out)
        assert fixed["name"] == "create_page"

    def test_file_not_found(self, capsys):
        code = run_fix("/nonexistent/file.json", use_color=False)
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
            code = run_fix(path, use_color=False)
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
            code = run_fix(path, use_color=False)
            assert code == 0
        finally:
            os.unlink(path)

    def test_json_flag_outputs_only_json(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_fix(path, use_color=False, json_output=True)
            assert code == 0
            out = capsys.readouterr().out
            # Should be valid JSON only
            fixed = json.loads(out)
            assert fixed["name"] == "create_page"
            # stderr should NOT have the report
            err = capsys.readouterr().err
            assert "agent-friend fix" not in err
        finally:
            os.unlink(path)

    def test_dry_run_no_json_output(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_fix(path, use_color=False, dry_run=True)
            assert code == 0
            out = capsys.readouterr().out
            # stdout should be empty (no JSON output in dry-run)
            assert out.strip() == ""
        finally:
            os.unlink(path)

    def test_diff_flag(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_fix(path, use_color=False, diff=True)
            assert code == 0
            err = capsys.readouterr().err
            assert "Diff:" in err
        finally:
            os.unlink(path)

    def test_only_flag(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(MULTI_ISSUE_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_fix(path, use_color=False, json_output=True, only=["names"])
            assert code == 0
            out = capsys.readouterr().out
            fixed = json.loads(out)
            # Name should be fixed
            assert "_" in fixed["name"]  # kebab converted
            # But verbose prefix should NOT be fixed (only names selected)
            assert fixed["description"].startswith("This tool allows you to")
        finally:
            os.unlink(path)

    def test_clean_tool_no_changes(self, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(CLEAN_MCP_TOOL, f)
            f.flush()
            path = f.name

        try:
            code = run_fix(path, use_color=False)
            assert code == 0
            out = capsys.readouterr().out
            fixed = json.loads(out)
            assert fixed == CLEAN_MCP_TOOL
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_fix_help(self, monkeypatch):
        """Verify fix --help doesn't crash."""
        monkeypatch.setattr("sys.argv", ["agent-friend", "fix", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            from agent_friend.cli import main
            main()
        assert exc_info.value.code == 0

    def test_fix_with_file(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "fix", path, "--no-color"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            fixed = json.loads(out)
            assert fixed["name"] == "create_page"
        finally:
            os.unlink(path)

    def test_fix_with_json_flag(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "fix", path, "--json"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            fixed = json.loads(out)
            assert fixed["name"] == "create_page"
        finally:
            os.unlink(path)

    def test_fix_with_dry_run(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(KEBAB_NAME_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "fix", path, "--dry-run", "--no-color"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            assert out.strip() == ""  # No JSON in stdout during dry-run
        finally:
            os.unlink(path)

    def test_fix_with_example_flag(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.argv", ["agent-friend", "fix", "--example", "notion", "--json"]
        )
        with pytest.raises(SystemExit) as exc_info:
            from agent_friend.cli import main
            main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        fixed = json.loads(out)
        assert isinstance(fixed, list)
        # All notion tools should have snake_case names after fix
        for tool in fixed:
            assert "-" not in tool["name"]

    def test_fix_with_only_flag(self, monkeypatch, capsys):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(MULTI_ISSUE_TOOL, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setattr(
                "sys.argv", ["agent-friend", "fix", path, "--json", "--only", "names,schemas"]
            )
            with pytest.raises(SystemExit) as exc_info:
                from agent_friend.cli import main
                main()
            assert exc_info.value.code == 0
            out = capsys.readouterr().out
            fixed = json.loads(out)
            # Name should be fixed
            assert "-" not in fixed["name"]
            # Undefined schema should be fixed
            assert "properties" in fixed["inputSchema"]["properties"]["config"]
            # But verbose prefix should NOT be fixed
            assert "This tool allows you to" in fixed["description"]
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_tools_list(self):
        fixed, changes = fix_tools([])
        assert fixed == []
        assert len(changes) == 0

    def test_non_dict_non_list_input(self):
        fixed, changes = fix_tools("not a dict or list")
        assert fixed == "not a dict or list"
        assert len(changes) == 0

    def test_tool_with_no_description(self):
        tool = {
            "name": "create-item",
            "description": "",
            "inputSchema": {"type": "object", "properties": {}},
        }
        fixed, changes = fix_tools(tool)
        # Should still fix the name
        assert fixed["name"] == "create_item"

    def test_tool_with_no_schema(self):
        tool = {
            "name": "list-items",
            "description": "List items.",
        }
        # This might not detect as any format, but shouldn't crash
        fixed, changes = fix_tools(tool)
        # At minimum, shouldn't raise

    def test_undetectable_format_skipped(self):
        tool = {"foo": "bar"}
        fixed, changes = fix_tools(tool)
        assert len(changes) == 0

    def test_param_with_no_description_key(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": {"type": "string"},
                },
            },
        }
        fixed, changes = fix_tools(tool)
        # Redundant check should not crash on params without description
        redundant = [c for c in changes if c.rule == "fix_redundant_params"]
        assert len(redundant) == 0

    def test_empty_description_not_crashed_by_prefix_fix(self):
        tool = {
            "name": "test",
            "description": "",
            "inputSchema": {"type": "object", "properties": {}},
        }
        fixed, changes = fix_tools(tool)
        prefix_changes = [c for c in changes if c.rule == "fix_verbose_prefixes"]
        assert len(prefix_changes) == 0

    def test_properties_not_dict(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": "invalid",
            },
        }
        # Should not crash
        fixed, changes = fix_tools(tool)

    def test_param_schema_not_dict(self):
        tool = {
            "name": "test",
            "description": "Test.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x": "not a dict",
                },
            },
        }
        # Should not crash
        fixed, changes = fix_tools(tool)
