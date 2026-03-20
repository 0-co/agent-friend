"""Tests for agent-friend EmailTool."""

import os
import sys
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.tools.email import EmailTool


class TestEmailToolInstantiation(unittest.TestCase):
    def test_basic_instantiation(self):
        t = EmailTool(inbox="test@agentmail.to")
        self.assertEqual(t.inbox, "test@agentmail.to")

    def test_default_vault_path(self):
        t = EmailTool(inbox="test@agentmail.to")
        self.assertIn("vault-agentmail", t.vault_path)

    def test_custom_vault_path(self):
        t = EmailTool(inbox="test@agentmail.to", vault_path="/custom/path")
        self.assertEqual(t.vault_path, "/custom/path")

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"AGENTMAIL_API_KEY": "am-test-key"}):
            t = EmailTool(inbox="test@agentmail.to")
            self.assertEqual(t.api_key, "am-test-key")

    def test_explicit_api_key(self):
        t = EmailTool(inbox="test@agentmail.to", api_key="am-explicit")
        self.assertEqual(t.api_key, "am-explicit")


class TestEmailToolDefinitions(unittest.TestCase):
    def setUp(self):
        self.tool = EmailTool(inbox="test@agentmail.to")

    def test_has_four_definitions(self):
        defs = self.tool.definitions()
        self.assertEqual(len(defs), 4)

    def test_definition_names(self):
        names = {d["name"] for d in self.tool.definitions()}
        self.assertEqual(names, {"email_list", "email_read", "email_send", "email_threads"})

    def test_email_send_has_send_param(self):
        send_def = next(d for d in self.tool.definitions() if d["name"] == "email_send")
        props = send_def["input_schema"]["properties"]
        self.assertIn("send", props)
        self.assertEqual(props["send"]["type"], "boolean")

    def test_email_send_required_fields(self):
        send_def = next(d for d in self.tool.definitions() if d["name"] == "email_send")
        required = send_def["input_schema"]["required"]
        self.assertIn("to", required)
        self.assertIn("subject", required)
        self.assertIn("body", required)


class TestEmailToolDraftMode(unittest.TestCase):
    def setUp(self):
        self.tool = EmailTool(inbox="test@agentmail.to")

    def test_send_false_returns_draft(self):
        result = self.tool.execute("email_send", {
            "to": "alice@example.com",
            "subject": "Hello",
            "body": "Test body",
            "send": False,
        })
        self.assertIn("DRAFT", result)
        self.assertIn("not sent", result)
        self.assertIn("alice@example.com", result)

    def test_send_default_is_draft(self):
        """Default (no send arg) should be draft mode, not send."""
        result = self.tool.execute("email_send", {
            "to": "alice@example.com",
            "subject": "Hello",
            "body": "Test body",
            # no "send" key
        })
        self.assertIn("DRAFT", result)
        self.assertIn("not sent", result)

    def test_draft_shows_recipient(self):
        result = self.tool.execute("email_send", {
            "to": "bob@example.com",
            "subject": "Test Subject",
            "body": "Hello Bob",
        })
        self.assertIn("bob@example.com", result)
        self.assertIn("Test Subject", result)
        self.assertIn("Hello Bob", result)


class TestEmailToolCallRouting(unittest.TestCase):
    def setUp(self):
        self.tool = EmailTool(inbox="test@agentmail.to")

    def test_unknown_tool_returns_error(self):
        result = self.tool.execute("email_unknown", {})
        self.assertIn("Unknown", result)

    def test_list_routes_to_list_messages(self):
        with patch.object(self.tool, "_list_messages", return_value="list result") as mock:
            result = self.tool.execute("email_list", {})
            mock.assert_called_once()
            self.assertEqual(result, "list result")

    def test_read_routes_to_read_message(self):
        with patch.object(self.tool, "_read_message", return_value="read result") as mock:
            result = self.tool.execute("email_read", {"message_id": "123"})
            mock.assert_called_once()
            self.assertEqual(result, "read result")

    def test_threads_routes_to_list_threads(self):
        with patch.object(self.tool, "_list_threads", return_value="threads result") as mock:
            result = self.tool.execute("email_threads", {})
            mock.assert_called_once()
            self.assertEqual(result, "threads result")


class TestEmailToolListMessages(unittest.TestCase):
    def setUp(self):
        self.tool = EmailTool(inbox="test@agentmail.to")

    def _mock_messages(self):
        return {
            "messages": [
                {
                    "message_id": "msg-1",
                    "from": "Alice <alice@example.com>",
                    "subject": "Hello",
                    "timestamp": "2026-03-11T12:00:00Z",
                    "preview": "Hi there!",
                    "labels": ["received", "unread"],
                },
                {
                    "message_id": "msg-2",
                    "from": "Bob <bob@example.com>",
                    "subject": "Re: Question",
                    "timestamp": "2026-03-10T10:00:00Z",
                    "preview": "Sure, that works.",
                    "labels": ["received"],
                },
            ]
        }

    def test_list_shows_messages(self):
        with patch.object(self.tool, "_call", return_value=self._mock_messages()):
            result = self.tool._list_messages({"limit": 10})
            self.assertIn("Alice", result)
            self.assertIn("Hello", result)

    def test_list_marks_unread(self):
        with patch.object(self.tool, "_call", return_value=self._mock_messages()):
            result = self.tool._list_messages({"limit": 10})
            self.assertIn("UNREAD", result)

    def test_list_empty_inbox(self):
        with patch.object(self.tool, "_call", return_value={"messages": []}):
            result = self.tool._list_messages({})
            self.assertIn("empty", result.lower())

    def test_list_unread_only_filters(self):
        with patch.object(self.tool, "_call", return_value=self._mock_messages()):
            result = self.tool._list_messages({"unread_only": True})
            self.assertIn("Alice", result)
            self.assertNotIn("Bob", result)

    def test_list_error_handling(self):
        with patch.object(self.tool, "_call", side_effect=RuntimeError("API error")):
            result = self.tool._list_messages({})
            self.assertIn("Failed", result)


class TestEmailToolReadMessage(unittest.TestCase):
    def setUp(self):
        self.tool = EmailTool(inbox="test@agentmail.to")

    def test_read_formats_message(self):
        mock_msg = {
            "from": "Alice <alice@example.com>",
            "to": ["test@agentmail.to"],
            "subject": "Hello",
            "timestamp": "2026-03-11T12:00:00Z",
            "body": {"text": "This is the email body."},
        }
        with patch.object(self.tool, "_call", return_value=mock_msg):
            result = self.tool._read_message({"message_id": "msg-1"})
            self.assertIn("Alice", result)
            self.assertIn("Hello", result)
            self.assertIn("This is the email body.", result)


if __name__ == "__main__":
    unittest.main()
