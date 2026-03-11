"""Tests for the Friend class."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend import Friend, ChatResponse, BudgetExceeded
from agent_friend.providers.base import ProviderResponse


def _make_provider_response(
    text="Hello!",
    tool_calls=None,
    input_tokens=10,
    output_tokens=5,
    stop_reason="end_turn",
    model="claude-haiku-4-5-20251001",
):
    return ProviderResponse(
        text=text,
        tool_calls=tool_calls or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason,
        model=model,
    )


class TestFriendInit(unittest.TestCase):
    def test_default_init(self):
        friend = Friend.__new__(Friend)
        friend._config = MagicMock()
        friend._provider = None
        friend._tools = []
        friend._conversation = []
        friend._total_cost_usd = 0.0
        self.assertEqual(friend._total_cost_usd, 0.0)
        self.assertEqual(friend._conversation, [])

    def test_init_with_seed(self):
        friend = Friend(seed="You are a pirate.")
        self.assertEqual(friend._config.seed, "You are a pirate.")

    def test_init_with_model(self):
        friend = Friend(model="claude-sonnet-4-6")
        self.assertEqual(friend._config.model, "claude-sonnet-4-6")

    def test_init_with_budget(self):
        friend = Friend(budget_usd=2.50)
        self.assertEqual(friend._config.budget_usd, 2.50)

    def test_init_no_tools(self):
        friend = Friend()
        self.assertEqual(friend._tools, [])

    def test_init_with_string_tools(self):
        friend = Friend(tools=["code", "search"])
        self.assertEqual(len(friend._tools), 2)

    def test_init_invalid_tool_name_raises(self):
        with self.assertRaises(ValueError):
            Friend(tools=["nonexistent_tool"])

    def test_init_invalid_tool_type_raises(self):
        with self.assertRaises(TypeError):
            Friend(tools=[42])

    def test_total_cost_usd_property(self):
        friend = Friend()
        self.assertEqual(friend.total_cost_usd, 0.0)


class TestFriendChat(unittest.TestCase):
    def _make_friend_with_mock_provider(self, responses):
        friend = Friend()
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = responses
        friend._provider = mock_provider
        return friend

    def test_chat_returns_chat_response(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response("I am fine.")]
        )
        result = friend.chat("How are you?")
        self.assertIsInstance(result, ChatResponse)

    def test_chat_response_has_text(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response("The answer is 4.")]
        )
        result = friend.chat("2+2?")
        self.assertEqual(result.text, "The answer is 4.")

    def test_chat_response_has_token_counts(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response(input_tokens=20, output_tokens=10)]
        )
        result = friend.chat("Hello")
        self.assertEqual(result.input_tokens, 20)
        self.assertEqual(result.output_tokens, 10)

    def test_chat_response_has_model(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response(model="claude-sonnet-4-6")]
        )
        result = friend.chat("Hello")
        self.assertEqual(result.model, "claude-sonnet-4-6")

    def test_chat_response_has_cost(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response(input_tokens=1_000_000, output_tokens=0, model="claude-sonnet-4-6")]
        )
        result = friend.chat("Hello")
        # claude-sonnet-4-6: $3.00 per 1M input tokens
        self.assertAlmostEqual(result.cost_usd, 3.00, places=2)

    def test_chat_appends_to_conversation(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response("Hi there")]
        )
        friend.chat("Hello")
        # After chat: user message + assistant message
        self.assertEqual(len(friend._conversation), 2)
        self.assertEqual(friend._conversation[0]["role"], "user")
        self.assertEqual(friend._conversation[1]["role"], "assistant")

    def test_chat_response_has_empty_tool_calls_when_no_tools_used(self):
        friend = self._make_friend_with_mock_provider(
            [_make_provider_response()]
        )
        result = friend.chat("Hello")
        self.assertEqual(result.tool_calls, [])


class TestFriendMultiTurn(unittest.TestCase):
    def _make_friend_with_mock_provider(self, responses):
        friend = Friend()
        mock_provider = MagicMock()
        mock_provider.complete.side_effect = responses
        friend._provider = mock_provider
        return friend

    def test_multiturn_history_preserved(self):
        friend = self._make_friend_with_mock_provider(
            [
                _make_provider_response("I am Alice's assistant."),
                _make_provider_response("Your name is Alice."),
            ]
        )
        friend.chat("My name is Alice")
        result = friend.chat("What is my name?")

        # Conversation should have 4 messages: user+assistant + user+assistant
        self.assertEqual(len(friend._conversation), 4)
        self.assertEqual(result.text, "Your name is Alice.")

    def test_multiturn_provider_called_with_history(self):
        friend = self._make_friend_with_mock_provider(
            [
                _make_provider_response("Got it."),
                _make_provider_response("You said hello."),
            ]
        )
        friend.chat("Hello")
        friend.chat("What did I say?")

        second_call_messages = friend._provider.complete.call_args_list[1][1]["messages"]
        # Second call should include first exchange
        self.assertGreater(len(second_call_messages), 1)


class TestFriendReset(unittest.TestCase):
    def test_reset_clears_conversation(self):
        friend = Friend()
        mock_provider = MagicMock()
        mock_provider.complete.return_value = _make_provider_response("OK")
        friend._provider = mock_provider

        friend.chat("Hello")
        self.assertEqual(len(friend._conversation), 2)

        friend.reset()
        self.assertEqual(len(friend._conversation), 0)

    def test_reset_does_not_affect_cost_tracker(self):
        friend = Friend()
        mock_provider = MagicMock()
        mock_provider.complete.return_value = _make_provider_response()
        friend._provider = mock_provider

        friend.chat("Hi")
        cost_before = friend.total_cost_usd
        friend.reset()
        # Cost accumulates across resets — memory is separate from conversation
        self.assertEqual(friend.total_cost_usd, cost_before)


class TestFriendToolCallLoop(unittest.TestCase):
    def test_tool_call_loop_executes_and_continues(self):
        """When the LLM requests a tool, the loop executes it and continues."""
        from agent_friend.tools.code import CodeTool

        tool_use_response = ProviderResponse(
            text="",
            tool_calls=[{"id": "call_1", "name": "run_code", "arguments": {"code": "print(42)"}}],
            input_tokens=10,
            output_tokens=5,
            stop_reason="tool_use",
            model="claude-haiku-4-5-20251001",
        )
        final_response = _make_provider_response("The answer is 42.")

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = [tool_use_response, final_response]

        friend = Friend(tools=[CodeTool()])
        friend._provider = mock_provider

        result = friend.chat("Run print(42)")
        self.assertEqual(result.text, "The answer is 42.")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["name"], "run_code")

    def test_tool_call_not_found_returns_error_message(self):
        """Unknown tool name returns error string, does not crash."""
        tool_use_response = ProviderResponse(
            text="",
            tool_calls=[{"id": "call_x", "name": "nonexistent_tool", "arguments": {}}],
            input_tokens=5,
            output_tokens=3,
            stop_reason="tool_use",
            model="claude-haiku-4-5-20251001",
        )
        final_response = _make_provider_response("I see.")

        mock_provider = MagicMock()
        mock_provider.complete.side_effect = [tool_use_response, final_response]

        friend = Friend()
        friend._provider = mock_provider

        result = friend.chat("Use unknown tool")
        self.assertEqual(result.text, "I see.")


class TestFriendBudget(unittest.TestCase):
    def test_budget_exceeded_raises(self):
        """BudgetExceeded is raised when spending limit is hit."""
        # Use a tiny budget and enough tokens to exceed it
        friend = Friend(budget_usd=0.000001, model="claude-sonnet-4-6")
        mock_provider = MagicMock()
        # 1M input tokens at claude-sonnet-4-6 = $3.00 >> $0.000001
        mock_provider.complete.return_value = _make_provider_response(
            input_tokens=1_000_000, output_tokens=0, model="claude-sonnet-4-6"
        )
        friend._provider = mock_provider

        with self.assertRaises(BudgetExceeded) as context:
            friend.chat("Hello")

        self.assertIn("Budget exceeded", str(context.exception))

    def test_budget_not_exceeded_within_limit(self):
        """No exception when cost is within budget."""
        friend = Friend(budget_usd=100.0, model="claude-haiku-4-5-20251001")
        mock_provider = MagicMock()
        mock_provider.complete.return_value = _make_provider_response(
            input_tokens=10, output_tokens=5, model="claude-haiku-4-5-20251001"
        )
        friend._provider = mock_provider

        result = friend.chat("Hello")
        self.assertIsInstance(result, ChatResponse)


class TestFriendStream(unittest.TestCase):
    def test_stream_yields_text(self):
        friend = Friend()
        mock_provider = MagicMock()
        mock_provider.complete.return_value = _make_provider_response("Hello world today")
        friend._provider = mock_provider

        chunks = list(friend.stream("Hi"))
        full_text = "".join(chunks)
        self.assertIn("Hello", full_text)
        self.assertIn("world", full_text)


class TestFriendFromConfig(unittest.TestCase):
    def test_from_config_creates_friend(self):
        friend = Friend.from_config({
            "seed": "You are a pirate.",
            "model": "claude-sonnet-4-6",
            "tools": [],
        })
        self.assertIsInstance(friend, Friend)
        self.assertEqual(friend._config.seed, "You are a pirate.")

    def test_from_config_with_tools(self):
        friend = Friend.from_config({
            "seed": "Helper.",
            "tools": ["code", "search"],
        })
        self.assertEqual(len(friend._tools), 2)

    def test_from_config_unknown_keys_ignored(self):
        friend = Friend.from_config({
            "seed": "Helper.",
            "unknown_key_xyz": "value",
        })
        self.assertIsInstance(friend, Friend)


class TestFriendFromYaml(unittest.TestCase):
    def test_from_yaml_loads_config(self):
        import tempfile
        import os
        yaml_content = "seed: You are a test agent.\nmodel: claude-sonnet-4-6\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmpfile:
            tmpfile.write(yaml_content)
            tmp_path = tmpfile.name
        try:
            friend = Friend.from_yaml(tmp_path)
            self.assertEqual(friend._config.seed, "You are a test agent.")
            self.assertEqual(friend._config.model, "claude-sonnet-4-6")
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            Friend.from_yaml("/nonexistent/path/config.yaml")


if __name__ == "__main__":
    unittest.main()
