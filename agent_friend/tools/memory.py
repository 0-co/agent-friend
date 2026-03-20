"""memory.py — SQLite-backed persistent memory tool for agent-friend."""

import sqlite3
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTool


DEFAULT_DB_PATH = "~/.agent_friend/memory.db"


class MemoryTool(BaseTool):
    """Persistent memory using SQLite with FTS5 full-text search.

    Provides remember/recall/forget operations for the agent.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(Path(db_path).expanduser())
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and tables if they do not exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(key, value, content='memories', content_rowid='rowid')
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, key, value)
                    VALUES (new.rowid, new.key, new.value);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, value)
                    VALUES ('delete', old.rowid, old.key, old.value);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, key, value)
                    VALUES ('delete', old.rowid, old.key, old.value);
                    INSERT INTO memories_fts(rowid, key, value)
                    VALUES (new.rowid, new.key, new.value);
                END
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Persistent memory — store and retrieve facts across conversations."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "remember",
                "description": "Store a fact in persistent memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "A short label for the fact (e.g. 'user_name', 'project_goal').",
                        },
                        "value": {
                            "type": "string",
                            "description": "The fact to store.",
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "recall",
                "description": "Search persistent memory for relevant facts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — keywords or phrase to look for.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "forget",
                "description": "Remove a stored fact by key.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "The key of the fact to remove.",
                        },
                    },
                    "required": ["key"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "remember":
            return self._remember(arguments["key"], arguments["value"])
        if tool_name == "recall":
            return self._recall(arguments["query"])
        if tool_name == "forget":
            return self._forget(arguments["key"])
        return f"Unknown memory tool: {tool_name}"

    def _remember(self, key: str, value: str) -> str:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now, now),
            )
        return f"Remembered: {key}"

    def _recall(self, query: str) -> str:
        with self._connect() as conn:
            # FTS5 search across key and value columns
            cursor = conn.execute(
                """
                SELECT m.key, m.value
                FROM memories m
                JOIN memories_fts fts ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT 10
                """,
                (query,),
            )
            rows = cursor.fetchall()

        if not rows:
            # Fall back to LIKE search when FTS finds nothing
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT key, value FROM memories
                    WHERE key LIKE ? OR value LIKE ?
                    LIMIT 10
                    """,
                    (f"%{query}%", f"%{query}%"),
                )
                rows = cursor.fetchall()

        if not rows:
            return "No memories found."

        lines = [f"{key}: {value}" for key, value in rows]
        return "\n".join(lines)

    def _forget(self, key: str) -> str:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            deleted = cursor.rowcount
        if deleted:
            return f"Forgot: {key}"
        return f"No memory found with key: {key}"
