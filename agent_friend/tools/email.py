"""email.py — AgentMail email tool for agent-friend.

Reads and sends email via AgentMail (https://agentmail.to/).
Requires a vault-agentmail wrapper OR AGENTMAIL_API_KEY env var.

Safety model:
- Reading messages: always allowed
- Sending email: requires explicit send=True argument
  (LLM must pass send=True, default is draft-only)

Hard limits (callers are responsible for enforcement):
- 5 outbound emails/day max
- 1 cold outreach/day max
- Zero bulk email ever

Usage:
    tool = EmailTool(inbox="user@agentmail.to", vault_path="/home/vault/bin/vault-agentmail")
    # or with direct API key:
    tool = EmailTool(inbox="user@agentmail.to", api_key="am-key-...")
"""

import json
import os
import subprocess
import urllib.request
import urllib.parse
from typing import Any, Dict, List, Optional

from .base import BaseTool


AGENTMAIL_BASE = "https://api.agentmail.to"


class EmailTool(BaseTool):
    """Email tool for agent-friend using AgentMail.

    Parameters
    ----------
    inbox:       AgentMail inbox address (e.g. "agent@agentmail.to")
    vault_path:  Path to vault-agentmail wrapper script (preferred)
    api_key:     AgentMail API key (fallback if no vault wrapper)
    max_list:    Maximum messages to list at once (default 10)
    """

    def __init__(
        self,
        inbox: str,
        vault_path: Optional[str] = None,
        api_key: Optional[str] = None,
        max_list: int = 10,
    ) -> None:
        self.inbox = inbox
        self.vault_path = vault_path or "/home/vault/bin/vault-agentmail"
        self.api_key = api_key or os.environ.get("AGENTMAIL_API_KEY")
        self.max_list = max_list

    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        return f"Read and send email via AgentMail inbox ({self.inbox})."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "email_list",
                "description": "List recent email messages in the inbox.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of messages to list (default 10, max 25).",
                        },
                        "unread_only": {
                            "type": "boolean",
                            "description": "If true, only show unread messages.",
                        },
                    },
                },
            },
            {
                "name": "email_read",
                "description": "Read the full body of a specific email message.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Message ID from email_list.",
                        },
                    },
                    "required": ["message_id"],
                },
            },
            {
                "name": "email_send",
                "description": (
                    "Send an email. By default returns a draft for review. "
                    "Set send=true to actually send. "
                    "IMPORTANT: Never send without user confirmation. Follow email limits."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject line.",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body (plain text).",
                        },
                        "send": {
                            "type": "boolean",
                            "description": "Set to true to actually send. Default is draft mode (shows email without sending).",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
            },
            {
                "name": "email_threads",
                "description": "List email conversation threads.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of threads to list (default 5).",
                        },
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "email_list":
            return self._list_messages(arguments)
        elif tool_name == "email_read":
            return self._read_message(arguments)
        elif tool_name == "email_send":
            return self._send_email(arguments)
        elif tool_name == "email_threads":
            return self._list_threads(arguments)
        return f"Unknown email tool: {tool_name}"

    def _call(self, method: str, endpoint: str, body: Optional[Dict] = None) -> Any:
        """Call the AgentMail API via vault wrapper or direct HTTP."""
        # Try vault wrapper first (sudo works even if the file isn't readable by agent)
        try:
            args = ["sudo", "-u", "vault", self.vault_path, method, endpoint]
            if body:
                args.append(json.dumps(body))
            result = subprocess.run(args, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            # Fall through to direct HTTP if subprocess fails for any reason
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

        # Fallback: direct HTTP with API key
        if not self.api_key:
            raise RuntimeError(
                "No vault wrapper or AGENTMAIL_API_KEY. "
                "Set AGENTMAIL_API_KEY or use vault-agentmail wrapper."
            )
        url = f"{AGENTMAIL_BASE}{endpoint}"
        req = urllib.request.Request(
            url,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        if body:
            req.data = json.dumps(body).encode()
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def _list_messages(self, arguments: Dict[str, Any]) -> str:
        limit = min(int(arguments.get("limit", self.max_list)), 25)
        unread_only = arguments.get("unread_only", False)

        try:
            endpoint = f"/v0/inboxes/{self.inbox}/messages?limit={limit}"
            data = self._call("GET", endpoint)
            messages = data.get("messages", [])
        except Exception as error:
            return f"Failed to list messages: {error}"

        if not messages:
            return "Inbox is empty."

        lines = []
        for msg in messages:
            labels = msg.get("labels", [])
            if unread_only and "unread" not in labels:
                continue
            unread = "[UNREAD] " if "unread" in labels else ""
            from_addr = msg.get("from", "unknown")
            subject = msg.get("subject", "(no subject)")
            timestamp = msg.get("timestamp", "")[:10]
            msg_id = msg.get("message_id", "?")
            preview = msg.get("preview", "").strip().replace("\n", " ")[:80]
            lines.append(f"{unread}From: {from_addr}")
            lines.append(f"  Subject: {subject} | Date: {timestamp}")
            lines.append(f"  Preview: {preview}")
            lines.append(f"  ID: {msg_id}")
            lines.append("")

        if not lines:
            return "No unread messages." if unread_only else "No messages."

        return f"Inbox ({self.inbox}) — {len(messages)} messages:\n\n" + "\n".join(lines).rstrip()

    def _read_message(self, arguments: Dict[str, Any]) -> str:
        message_id = arguments["message_id"]
        # message_id needs URL encoding for the path
        encoded_id = urllib.parse.quote(message_id, safe="")

        try:
            endpoint = f"/v0/inboxes/{self.inbox}/messages/{encoded_id}"
            msg = self._call("GET", endpoint)
        except Exception as error:
            return f"Failed to read message: {error}"

        from_addr = msg.get("from", "unknown")
        to = msg.get("to", [])
        subject = msg.get("subject", "(no subject)")
        timestamp = msg.get("timestamp", "")
        body = msg.get("body", {})
        text = body.get("text", "") if isinstance(body, dict) else str(body)
        if not text:
            text = msg.get("preview", "(no body)")

        return (
            f"From: {from_addr}\n"
            f"To: {', '.join(to) if isinstance(to, list) else to}\n"
            f"Subject: {subject}\n"
            f"Date: {timestamp}\n"
            f"\n{text}"
        )

    def _send_email(self, arguments: Dict[str, Any]) -> str:
        to = arguments["to"]
        subject = arguments["subject"]
        body = arguments["body"]
        actually_send = arguments.get("send", False)

        draft_preview = (
            f"📧 DRAFT EMAIL\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Body:\n{body}\n"
        )

        if not actually_send:
            return (
                draft_preview
                + "\n[Draft only — not sent. Set send=true to send.]"
            )

        # Actually send
        try:
            payload = {
                "to": [to] if isinstance(to, str) else to,
                "subject": subject,
                "body": body,
            }
            endpoint = f"/inboxes/{self.inbox}/messages/send"
            result = self._call("POST", endpoint, payload)
            msg_id = result.get("message_id", "?")
            return f"Email sent successfully.\nMessage ID: {msg_id}\n\n{draft_preview}"
        except Exception as error:
            return f"Failed to send email: {error}"

    def _list_threads(self, arguments: Dict[str, Any]) -> str:
        limit = min(int(arguments.get("limit", 5)), 25)

        try:
            endpoint = f"/v0/inboxes/{self.inbox}/threads?limit={limit}"
            data = self._call("GET", endpoint)
            threads = data.get("threads", [])
        except Exception as error:
            return f"Failed to list threads: {error}"

        if not threads:
            return "No threads found."

        lines = [f"Threads in {self.inbox}:\n"]
        for t in threads:
            subject = t.get("subject", "(no subject)")
            count = t.get("message_count", "?")
            thread_id = t.get("thread_id", "?")
            lines.append(f"• {subject} ({count} messages) — ID: {thread_id}")

        return "\n".join(lines)
