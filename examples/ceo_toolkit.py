#!/usr/bin/env python3
"""CEO Toolkit — real-world @tool usage for company operations.

This is agent-friend dogfooding itself. These tools wrap the vault-*
CLI commands used daily by the AI CEO at twitch.tv/0coceo.

Usage:
    from examples.ceo_toolkit import kit
    kit.to_openai()      # All tools in OpenAI format
    kit.to_anthropic()   # All tools in Anthropic format
    kit.to_mcp()         # All tools in MCP format
    kit.token_report()   # How many tokens do these cost?
"""

import json
import subprocess
from typing import Optional

from agent_friend import tool, Toolkit


def _vault(cmd: str, method: str, endpoint: str, body: Optional[str] = None) -> dict:
    """Run a vault-* command and return parsed JSON."""
    args = ["sudo", "-u", "vault", f"/home/vault/bin/vault-{cmd}", method, endpoint]
    if body:
        args.append(body)
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"Exit code {result.returncode}"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip()}


@tool
def twitch_stream_status() -> dict:
    """Check if the Twitch stream is currently live."""
    return _vault("twitch", "GET", "/streams?user_id=1455485722")


@tool
def twitch_followers() -> dict:
    """Get current Twitch follower count and list."""
    return _vault("twitch", "GET", "/channels/followers?broadcaster_id=1455485722")


@tool
def twitch_chat(message: str) -> dict:
    """Send a message to Twitch chat.

    Args:
        message: Chat message text
    """
    body = json.dumps({
        "broadcaster_id": "1455485722",
        "sender_id": "1455485722",
        "message": message
    })
    return _vault("twitch", "POST", "/chat/messages", body)


@tool
def bluesky_post(text: str) -> dict:
    """Post to Bluesky (@0coceo.bsky.social). Max 300 graphemes.

    Args:
        text: Post text (max 300 chars)
    """
    import datetime
    record = {
        "repo": "did:plc:ak33o45ans6qtlhxxulcd4ko",
        "collection": "app.bsky.feed.post",
        "record": {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    }
    result = subprocess.run(
        ["sudo", "-u", "vault", "/home/vault/bin/vault-bsky",
         "com.atproto.repo.createRecord", json.dumps(record)],
        capture_output=True, text=True, timeout=30
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip(), "error": result.stderr.strip()}


@tool
def devto_article_status(article_id: int) -> dict:
    """Get a Dev.to article's status (views, reactions, published state).

    Args:
        article_id: Dev.to article ID
    """
    return _vault("devto", "GET", f"/articles/{article_id}")


@tool
def discord_post(channel_id: str, message: str) -> dict:
    """Post a message to a Discord channel.

    Args:
        channel_id: Discord channel ID
        message: Message content
    """
    body = json.dumps({"content": message})
    result = subprocess.run(
        ["sudo", "-u", "vault", "/home/vault/bin/vault-discord",
         "-s", "-X", "POST",
         f"https://discord.com/api/v10/channels/{channel_id}/messages",
         "-H", "Content-Type: application/json", "-d", body],
        capture_output=True, text=True, timeout=30
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"raw": result.stdout.strip()}


@tool
def github_repo_stats(repo: str = "0-co/agent-friend") -> dict:
    """Get GitHub repo stats (stars, forks, issues).

    Args:
        repo: GitHub repo in owner/name format
    """
    result = subprocess.run(
        ["sudo", "-u", "vault", "/home/vault/bin/vault-gh",
         "api", f"repos/{repo}"],
        capture_output=True, text=True, timeout=30
    )
    try:
        data = json.loads(result.stdout)
        return {
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "watchers": data.get("subscribers_count"),
        }
    except (json.JSONDecodeError, KeyError):
        return {"raw": result.stdout.strip()}


# Bundle all tools into a Toolkit
kit = Toolkit([
    twitch_stream_status,
    twitch_followers,
    twitch_chat,
    bluesky_post,
    devto_article_status,
    discord_post,
    github_repo_stats,
])


if __name__ == "__main__":
    # Dogfood: show the token cost of these tools
    print("CEO Toolkit — token cost report\n")
    print(f"Tools: {len(kit)}")
    print()

    # Show in each format
    for fmt in ["openai", "anthropic", "google", "mcp", "json_schema"]:
        method = getattr(kit, f"to_{fmt}")
        data = method()
        est = sum(len(json.dumps(t)) // 4 for t in data) if isinstance(data, list) else len(json.dumps(data)) // 4
        print(f"  {fmt:15s} ~{est} tokens")

    print()
    print("OpenAI format sample:")
    print(json.dumps(kit.to_openai()[0], indent=2))
