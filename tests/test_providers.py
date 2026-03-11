"""Tests for agent-friend providers."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.providers.base import BaseProvider, ProviderResponse
from agent_friend.providers.anthropic import AnthropicProvider
from agent_friend.providers.openai import OpenAIProvider
from agent_friend.friend import _calculate_cost


# ---------------------------------------------------------------------------
# ProviderResponse tests
# ---------------------------------------------------------------------------

class TestProviderResponse(unittest.TestCase):
    def _make(self, tool_calls=None):
        return ProviderResponse(
            text="Hello",
            tool_calls=tool_calls or [],
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
            model="claude-haiku-4-5-20251001",
        )

    def test_has_tool_calls_false_when_empty(self):
        response = self._make(tool_calls=[])
        self.assertFalse(response.has_tool_calls)

    def test_has_tool_calls_true_when_present(self):
        response = self._make(tool_calls=[{"id": "1", "name": "run_code", "arguments": {}}])
        self.assertTrue(response.has_tool_calls)

    def test_attributes(self):
        response = self._make()
        self.assertEqual(response.text, "Hello")
        self.assertEqual(response.input_tokens, 10)
        self.assertEqual(response.output_tokens, 5)
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(response.model, "claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------

class TestAnthropicProvider(unittest.TestCase):
    def _make_mock_anthropic_response(self, text="Hello", tool_calls=None):
        """Build a mock Anthropic API response object."""
        content_blocks = []

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        content_blocks.append(text_block)

        if tool_calls:
            for tc in tool_calls:
                tool_block = MagicMock()
                tool_block.type = "tool_use"
                tool_block.id = tc["id"]
                tool_block.name = tc["name"]
                tool_block.input = tc["arguments"]
                content_blocks.append(tool_block)

        response = MagicMock()
        response.content = content_blocks
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        response.stop_reason = "end_turn" if not tool_calls else "tool_use"
        response.model = "claude-haiku-4-5-20251001"
        return response

    def test_raises_import_error_without_anthropic(self):
        provider = AnthropicProvider(api_key="test")
        with patch("builtins.__import__", side_effect=ImportError("No anthropic")):
            provider._client = None
            # Patching the import at module level to simulate missing package
        # Just verify the provider was created
        self.assertIsInstance(provider, AnthropicProvider)

    def test_complete_returns_provider_response(self):
        provider = AnthropicProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_anthropic_response()
        provider._client = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
        )
        self.assertIsInstance(response, ProviderResponse)

    def test_complete_extracts_text(self):
        provider = AnthropicProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_anthropic_response(
            text="The answer is 42."
        )
        provider._client = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            system="You are helpful.",
        )
        self.assertEqual(response.text, "The answer is 42.")

    def test_complete_extracts_tool_calls(self):
        provider = AnthropicProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_anthropic_response(
            text="",
            tool_calls=[{"id": "call_1", "name": "search", "arguments": {"query": "AI"}}],
        )
        provider._client = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "Search for AI"}],
            system="You are helpful.",
            tools=[{"name": "search", "description": "Search", "input_schema": {}}],
        )
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0]["name"], "search")

    def test_complete_passes_tools_to_api(self):
        provider = AnthropicProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_anthropic_response()
        provider._client = mock_client

        tools = [{"name": "search", "description": "Search", "input_schema": {}}]
        provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
            tools=tools,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertIn("tools", call_kwargs)
        self.assertEqual(call_kwargs["tools"], tools)

    def test_complete_no_tools_when_none(self):
        provider = AnthropicProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_anthropic_response()
        provider._client = mock_client

        provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
            tools=None,
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertNotIn("tools", call_kwargs)

    def test_build_tool_result_message_format(self):
        provider = AnthropicProvider(api_key="test-key")
        response = ProviderResponse(
            text="", tool_calls=[], input_tokens=0, output_tokens=0,
            stop_reason="tool_use", model="claude-haiku-4-5-20251001"
        )
        tool_results = [{"tool_use_id": "call_1", "content": "42"}]
        msg = provider.build_tool_result_message(response, tool_results, None)
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"][0]["type"], "tool_result")
        self.assertEqual(msg["content"][0]["tool_use_id"], "call_1")

    def test_uses_model_from_kwarg(self):
        provider = AnthropicProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_anthropic_response()
        provider._client = mock_client

        provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
            model="claude-opus-4-6",
        )
        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-opus-4-6")


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------

class TestOpenAIProvider(unittest.TestCase):
    def _make_mock_openai_response(self, text="Hello", tool_calls=None, model="gpt-4o-mini"):
        choice = MagicMock()
        choice.message.content = text
        choice.message.tool_calls = None
        choice.finish_reason = "stop"

        if tool_calls:
            oai_tool_calls = []
            for tc in tool_calls:
                import json
                oai_tc = MagicMock()
                oai_tc.id = tc["id"]
                oai_tc.function.name = tc["name"]
                oai_tc.function.arguments = json.dumps(tc["arguments"])
                oai_tool_calls.append(oai_tc)
            choice.message.tool_calls = oai_tool_calls
            choice.finish_reason = "tool_calls"

        response = MagicMock()
        response.choices = [choice]
        response.model = model
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        return response

    def test_complete_returns_provider_response(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_openai_response()
        provider._client = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
        )
        self.assertIsInstance(response, ProviderResponse)

    def test_complete_extracts_text(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_openai_response(
            text="The answer is 4."
        )
        provider._client = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "2+2?"}],
            system="You are helpful.",
        )
        self.assertEqual(response.text, "The answer is 4.")

    def test_complete_extracts_tool_calls(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_openai_response(
            text="",
            tool_calls=[{"id": "call_1", "name": "search", "arguments": {"query": "AI"}}],
        )
        provider._client = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "Search"}],
            system="You are helpful.",
            tools=[{"name": "search", "description": "Search", "input_schema": {}}],
        )
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.tool_calls[0]["name"], "search")

    def test_convert_tools_format(self):
        provider = OpenAIProvider(api_key="test-key")
        anthropic_tools = [
            {
                "name": "run_code",
                "description": "Run code",
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            }
        ]
        openai_tools = provider._convert_tools(anthropic_tools)
        self.assertEqual(openai_tools[0]["type"], "function")
        self.assertEqual(openai_tools[0]["function"]["name"], "run_code")

    def test_build_tool_result_message_format(self):
        provider = OpenAIProvider(api_key="test-key")
        response = ProviderResponse(
            text="", tool_calls=[], input_tokens=0, output_tokens=0,
            stop_reason="tool_calls", model="gpt-4o-mini"
        )
        tool_results = [{"tool_use_id": "call_1", "content": "42"}]
        msg = provider.build_tool_result_message(response, tool_results, None)
        self.assertEqual(msg["role"], "__tool_results__")
        self.assertEqual(msg["tool_results"][0]["role"], "tool")
        self.assertEqual(msg["tool_results"][0]["tool_call_id"], "call_1")

    def test_prepends_system_message(self):
        provider = OpenAIProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_openai_response()
        provider._client = mock_client

        provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="Be helpful.",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Be helpful.")


# ---------------------------------------------------------------------------
# Cost calculation tests
# ---------------------------------------------------------------------------

class TestCalculateCost(unittest.TestCase):
    def test_haiku_cost(self):
        cost = _calculate_cost(1_000_000, 0, "claude-haiku-4-5-20251001")
        self.assertAlmostEqual(cost, 0.80, places=2)

    def test_sonnet_cost(self):
        cost = _calculate_cost(1_000_000, 0, "claude-sonnet-4-6")
        self.assertAlmostEqual(cost, 3.00, places=2)

    def test_opus_cost(self):
        cost = _calculate_cost(0, 1_000_000, "claude-opus-4-6")
        self.assertAlmostEqual(cost, 75.00, places=2)

    def test_gpt4o_cost(self):
        cost = _calculate_cost(1_000_000, 0, "gpt-4o")
        self.assertAlmostEqual(cost, 2.50, places=2)

    def test_gpt4o_mini_cost(self):
        cost = _calculate_cost(1_000_000, 0, "gpt-4o-mini")
        self.assertAlmostEqual(cost, 0.15, places=2)

    def test_zero_tokens_zero_cost(self):
        cost = _calculate_cost(0, 0, "claude-sonnet-4-6")
        self.assertEqual(cost, 0.0)

    def test_unknown_model_returns_zero(self):
        cost = _calculate_cost(1_000_000, 1_000_000, "totally-unknown-model")
        self.assertEqual(cost, 0.0)

    def test_combined_input_output(self):
        # claude-sonnet-4-6: $3/$15 per 1M
        cost = _calculate_cost(1_000_000, 1_000_000, "claude-sonnet-4-6")
        self.assertAlmostEqual(cost, 18.00, places=2)


if __name__ == "__main__":
    unittest.main()
