"""Tests for BitNet provider integration."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.providers.bitnet import BitNetProvider
from agent_friend.providers.openai import OpenAIProvider
from agent_friend.config import FriendConfig
from agent_friend.friend import _calculate_cost


# ---------------------------------------------------------------------------
# BitNetProvider tests
# ---------------------------------------------------------------------------

class TestBitNetProvider(unittest.TestCase):
    def test_default_model(self):
        p = BitNetProvider()
        self.assertEqual(p.DEFAULT_MODEL, "bitnet-b1.58-2B-4T")

    def test_base_url_default(self):
        p = BitNetProvider()
        self.assertEqual(p._base_url, "http://localhost:8080/v1")

    def test_base_url_from_env(self):
        with patch.dict("os.environ", {"BITNET_HOST": "http://myhost:9090"}):
            p = BitNetProvider()
            self.assertEqual(p._base_url, "http://myhost:9090/v1")

    def test_base_url_no_double_v1(self):
        p = BitNetProvider(base_url="http://localhost:8080/v1")
        self.assertEqual(p._base_url, "http://localhost:8080/v1")

    def test_base_url_explicit(self):
        p = BitNetProvider(base_url="http://remote:3000")
        self.assertEqual(p._base_url, "http://remote:3000/v1")

    def test_api_key_is_dummy(self):
        p = BitNetProvider()
        self.assertEqual(p.api_key, "bitnet")

    def test_api_key_ignored(self):
        p = BitNetProvider(api_key="real-key-ignored")
        self.assertEqual(p.api_key, "bitnet")

    def test_inherits_from_openai(self):
        p = BitNetProvider()
        self.assertIsInstance(p, OpenAIProvider)

    def test_convert_tools_works(self):
        p = BitNetProvider()
        tools = [{"name": "test", "description": "Test", "input_schema": {"type": "object", "properties": {}}}]
        result = p._convert_tools(tools)
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "test")

    def test_complete_delegates_to_openai(self):
        p = BitNetProvider()
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Hello from BitNet"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response = MagicMock()
        response.choices = [choice]
        response.model = "bitnet-b1.58-2B-4T"
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = response
        p._client = mock_client

        result = p.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
            model="bitnet-b1.58-2B-4T",
        )
        self.assertEqual(result.text, "Hello from BitNet")
        self.assertEqual(result.model, "bitnet-b1.58-2B-4T")


# ---------------------------------------------------------------------------
# Config resolution tests for BitNet
# ---------------------------------------------------------------------------

class TestConfigBitNetResolution(unittest.TestCase):
    def _make_config(self, **kwargs):
        return FriendConfig(**kwargs)

    def test_bitnet_model_resolves_to_bitnet(self):
        c = self._make_config(model="bitnet-b1.58-2B-4T")
        self.assertEqual(c.resolve_provider(), "bitnet")

    def test_bitnet_prefix_resolves_to_bitnet(self):
        c = self._make_config(model="bitnet-something-else")
        self.assertEqual(c.resolve_provider(), "bitnet")

    def test_explicit_bitnet_provider(self):
        c = self._make_config(provider="bitnet", model="anything")
        self.assertEqual(c.resolve_provider(), "bitnet")

    def test_api_key_returns_bitnet_string(self):
        c = self._make_config(model="bitnet-b1.58-2B-4T")
        self.assertEqual(c.resolve_api_key(), "bitnet")

    def test_explicit_provider_api_key(self):
        c = self._make_config(provider="bitnet")
        self.assertEqual(c.resolve_api_key(), "bitnet")


# ---------------------------------------------------------------------------
# BitNet cost calculation tests
# ---------------------------------------------------------------------------

class TestBitNetCost(unittest.TestCase):
    def test_bitnet_2b_is_free(self):
        cost = _calculate_cost(1_000_000, 1_000_000, "bitnet-b1.58-2B-4T")
        self.assertEqual(cost, 0.0)

    def test_bitnet_zero_tokens_zero_cost(self):
        cost = _calculate_cost(0, 0, "bitnet-b1.58-2B-4T")
        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
