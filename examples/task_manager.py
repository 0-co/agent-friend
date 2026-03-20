"""task_manager.py — a conversational task manager using DatabaseTool.

Shows how an agent can create and query a SQLite database to track tasks.
No external APIs needed for the database — uses stdlib sqlite3.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python3 examples/task_manager.py

Or use the Python API directly (no LLM required):
    from agent_friend import DatabaseTool
    db = DatabaseTool()
    db.create_table("tasks", "id INTEGER PRIMARY KEY, title TEXT, done INTEGER DEFAULT 0")
    db.insert("tasks", {"title": "Buy groceries", "done": 0})
    print(db.query("SELECT * FROM tasks WHERE done = 0"))
"""

import os
import sys
from agent_friend import Friend, DatabaseTool


def demo_python_api():
    """Show the Python API without an LLM — just SQLite."""
    print("=== Python API Demo (no LLM) ===\n")
    db = DatabaseTool(db_path="/tmp/demo_tasks.db")

    # Create a table
    db.create_table(
        "tasks",
        "id INTEGER PRIMARY KEY, title TEXT NOT NULL, priority INTEGER DEFAULT 1, done INTEGER DEFAULT 0",
    )

    # Insert some tasks
    tasks = [
        {"title": "Write tests for DatabaseTool", "priority": 2, "done": 1},
        {"title": "Ship agent-friend v0.8", "priority": 3, "done": 1},
        {"title": "Write article053", "priority": 3, "done": 0},
        {"title": "Hit 20 stars on GitHub", "priority": 2, "done": 0},
        {"title": "Reach Twitch affiliate", "priority": 1, "done": 0},
    ]
    for task in tasks:
        db.insert("tasks", task)

    # Query pending tasks by priority
    pending = db.query(
        "SELECT id, title, priority FROM tasks WHERE done = 0 ORDER BY priority DESC"
    )
    print("Pending tasks (by priority):")
    for row in pending:
        stars = "★" * row["priority"]
        print(f"  {row['id']}. {row['title']} {stars}")

    # Mark one done
    db.run("UPDATE tasks SET done = 1 WHERE id = ?", [3])
    print("\nMarked 'Write article053' as done.")

    # Count
    stats = db.query("SELECT SUM(done) AS done, COUNT(*) AS total FROM tasks")
    s = stats[0]
    print(f"Progress: {s['done']}/{s['total']} tasks complete\n")

    # Clean up
    db.run("DROP TABLE tasks")


def demo_agent_api():
    """Show the agent API — conversational task management."""
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("Skipping agent demo — no API key found.")
        print("Set OPENROUTER_API_KEY (free at openrouter.ai) to run this demo.")
        return

    print("=== Agent API Demo ===\n")
    manager = Friend(
        seed=(
            "You are a task manager. Use the database tool to create and manage tasks.\n"
            "The tasks table has: id INTEGER PRIMARY KEY, title TEXT NOT NULL, "
            "done INTEGER DEFAULT 0.\n"
            "Always show results after each action."
        ),
        tools=["database"],
        api_key=api_key,
        model="google/gemini-2.0-flash-exp:free",
        budget_usd=0.05,
    )

    # Create table and add tasks
    r = manager.chat(
        "Create a tasks table with columns: id INTEGER PRIMARY KEY, title TEXT NOT NULL, done INTEGER DEFAULT 0. "
        "Then add these 3 tasks: 'Review PR', 'Update docs', 'Ship next version'."
    )
    print(r.text)

    # Query
    r = manager.chat("Show me all incomplete tasks.")
    print("\n" + r.text)

    # Complete one
    r = manager.chat("Mark 'Review PR' as done and show me the updated list.")
    print("\n" + r.text)


if __name__ == "__main__":
    demo_python_api()
    demo_agent_api()
