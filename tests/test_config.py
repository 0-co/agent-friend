"""Tests for agent-friend config loading."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.config import (
    FriendConfig,
    load_from_dict,
    load_from_yaml,
    _parse_simple_yaml,
    _coerce_value,
)


class TestFriendConfig(unittest.TestCase):
    def test_defaults(self):
        config = FriendConfig()
        self.assertEqual(config.model, "claude-haiku-4-5-20251001")
        self.assertEqual(config.max_context_messages, 20)
        self.assertIsNone(config.budget_usd)
        self.assertIsNone(config.api_key)

    def test_resolve_provider_anthropic_default(self):
        config = FriendConfig(model="claude-sonnet-4-6")
        self.assertEqual(config.resolve_provider(), "anthropic")

    def test_resolve_provider_openai_from_model(self):
        config = FriendConfig(model="gpt-4o")
        self.assertEqual(config.resolve_provider(), "openai")

    def test_resolve_provider_openai_from_model_mini(self):
        config = FriendConfig(model="gpt-4o-mini")
        self.assertEqual(config.resolve_provider(), "openai")

    def test_resolve_provider_explicit_override(self):
        config = FriendConfig(model="claude-sonnet-4-6", provider="openai")
        self.assertEqual(config.resolve_provider(), "openai")

    def test_resolve_api_key_explicit(self):
        config = FriendConfig(api_key="sk-test-key")
        self.assertEqual(config.resolve_api_key(), "sk-test-key")

    def test_resolve_api_key_from_env_anthropic(self):
        config = FriendConfig(model="claude-haiku-4-5-20251001")
        with unittest.mock.patch.dict(
            os.environ, {"ANTHROPIC_API_KEY": "env-anthropic-key"}
        ):
            self.assertEqual(config.resolve_api_key(), "env-anthropic-key")

    def test_resolve_api_key_from_env_openai(self):
        config = FriendConfig(model="gpt-4o")
        with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
            self.assertEqual(config.resolve_api_key(), "env-openai-key")

    def test_resolve_api_key_none_when_not_set(self):
        config = FriendConfig(model="claude-haiku-4-5-20251001")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with unittest.mock.patch.dict(os.environ, env, clear=True):
            result = config.resolve_api_key()
            self.assertIsNone(result)


import unittest.mock


class TestLoadFromDict(unittest.TestCase):
    def test_basic_dict(self):
        config = load_from_dict({"seed": "You are a helper.", "model": "gpt-4o"})
        self.assertEqual(config.seed, "You are a helper.")
        self.assertEqual(config.model, "gpt-4o")

    def test_tools_list(self):
        config = load_from_dict({"tools": ["search", "code"]})
        self.assertEqual(config.tools, ["search", "code"])

    def test_budget_usd(self):
        config = load_from_dict({"budget_usd": 5.0})
        self.assertEqual(config.budget_usd, 5.0)

    def test_unknown_keys_ignored(self):
        config = load_from_dict({"seed": "Hi", "unknown_key": "ignored"})
        self.assertEqual(config.seed, "Hi")

    def test_empty_dict_uses_defaults(self):
        config = load_from_dict({})
        self.assertEqual(config.model, "claude-haiku-4-5-20251001")

    def test_max_context_messages(self):
        config = load_from_dict({"max_context_messages": 50})
        self.assertEqual(config.max_context_messages, 50)


class TestLoadFromYaml(unittest.TestCase):
    def _write_yaml(self, content: str) -> str:
        tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        tmpfile.write(content)
        tmpfile.close()
        return tmpfile.name

    def test_basic_yaml(self):
        path = self._write_yaml("seed: You are a test agent.\nmodel: gpt-4o\n")
        try:
            config = load_from_yaml(path)
            self.assertEqual(config.seed, "You are a test agent.")
            self.assertEqual(config.model, "gpt-4o")
        finally:
            os.unlink(path)

    def test_yaml_with_tools_list(self):
        content = "seed: Helper.\ntools:\n- search\n- code\n"
        path = self._write_yaml(content)
        try:
            config = load_from_yaml(path)
            self.assertIn("search", config.tools)
            self.assertIn("code", config.tools)
        finally:
            os.unlink(path)

    def test_yaml_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_from_yaml("/nonexistent/path.yaml")

    def test_yaml_with_budget(self):
        path = self._write_yaml("seed: Helper.\nbudget_usd: 2.5\n")
        try:
            config = load_from_yaml(path)
            self.assertAlmostEqual(config.budget_usd, 2.5)
        finally:
            os.unlink(path)


class TestSimpleYamlParser(unittest.TestCase):
    def test_simple_key_value(self):
        result = _parse_simple_yaml("seed: You are a helper.\nmodel: gpt-4o\n")
        self.assertEqual(result["seed"], "You are a helper.")
        self.assertEqual(result["model"], "gpt-4o")

    def test_list_items(self):
        result = _parse_simple_yaml("tools:\n- search\n- code\n")
        self.assertEqual(result["tools"], ["search", "code"])

    def test_comment_lines_skipped(self):
        result = _parse_simple_yaml("# comment\nseed: Hello.\n")
        self.assertEqual(result["seed"], "Hello.")

    def test_blank_lines_skipped(self):
        result = _parse_simple_yaml("\n\nseed: Hello.\n\n")
        self.assertEqual(result["seed"], "Hello.")

    def test_numeric_value(self):
        result = _parse_simple_yaml("budget_usd: 5.0\n")
        self.assertAlmostEqual(result["budget_usd"], 5.0)

    def test_integer_value(self):
        result = _parse_simple_yaml("max_context_messages: 20\n")
        self.assertEqual(result["max_context_messages"], 20)


class TestCoerceValue(unittest.TestCase):
    def test_true_variants(self):
        self.assertTrue(_coerce_value("true"))
        self.assertTrue(_coerce_value("yes"))
        self.assertTrue(_coerce_value("True"))

    def test_false_variants(self):
        self.assertFalse(_coerce_value("false"))
        self.assertFalse(_coerce_value("no"))

    def test_none_variants(self):
        self.assertIsNone(_coerce_value("null"))
        self.assertIsNone(_coerce_value("~"))
        self.assertIsNone(_coerce_value("none"))

    def test_integer(self):
        self.assertEqual(_coerce_value("42"), 42)

    def test_float(self):
        self.assertAlmostEqual(_coerce_value("3.14"), 3.14)

    def test_string(self):
        self.assertEqual(_coerce_value("hello"), "hello")

    def test_quoted_string(self):
        self.assertEqual(_coerce_value('"hello world"'), "hello world")
        self.assertEqual(_coerce_value("'hello world'"), "hello world")


if __name__ == "__main__":
    unittest.main()
