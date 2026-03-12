"""
agent-friend — a composable personal AI agent library.

Quick start:

    from agent_friend import Friend

    friend = Friend(
        seed="You are a helpful assistant.",
        api_key="sk-...",
    )
    response = friend.chat("What is 2+2?")
    print(response.text)

With tools:

    friend = Friend(
        seed="You are a helpful assistant with tools.",
        tools=["search", "code", "memory"],
        model="claude-sonnet-4-6",
        budget_usd=1.0,
    )
"""

from .friend import Friend, ChatResponse, BudgetExceeded
from .tools import MemoryTool, CodeTool, SearchTool, BrowserTool, EmailTool, FileTool, FetchTool, VoiceTool, RSSFeedTool, SchedulerTool, DatabaseTool

__all__ = [
    "Friend",
    "ChatResponse",
    "BudgetExceeded",
    "MemoryTool",
    "CodeTool",
    "SearchTool",
    "BrowserTool",
    "EmailTool",
    "FileTool",
    "FetchTool",
    "VoiceTool",
    "RSSFeedTool",
    "SchedulerTool",
    "DatabaseTool",
]

__version__ = "0.8.0"
