"""data_analyst.py — Agent that reads and analyzes CSV data using TableTool.

Run:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/data_analyst.py
"""

import csv
import os
import tempfile

from agent_friend import Friend, TableTool

# Create sample data
SALES_DATA = """region,product,revenue,units,month
North,Widget A,1200,15,Jan
South,Widget B,850,10,Jan
North,Widget B,2100,25,Feb
East,Widget A,975,12,Jan
South,Widget A,1500,18,Feb
East,Widget B,620,8,Feb
West,Widget A,3200,40,Feb
West,Widget B,1100,14,Jan
"""


def main():
    # Write sample CSV to a temp file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    tmp.write(SALES_DATA)
    tmp.close()
    csv_path = tmp.name

    print(f"Analyzing: {csv_path}\n")

    # Python API — no LLM needed for basic operations
    table = TableTool()

    rows = table.read(csv_path)
    print(f"Total rows: {len(rows)}")
    print(f"Columns: {table.columns(csv_path)}")

    total_revenue = table.aggregate(csv_path, "revenue", "sum")
    avg_revenue = table.aggregate(csv_path, "revenue", "avg")
    print(f"\nTotal revenue: ${total_revenue}")
    print(f"Average revenue: ${avg_revenue}")

    north = table.filter_rows(csv_path, "region", "eq", "North")
    print(f"\nNorth region rows: {len(north)}")
    for row in north:
        print(f"  {row}")

    high_value = table.filter_rows(csv_path, "revenue", "gt", "1000")
    print(f"\nHigh-value sales (>$1000): {len(high_value)} of {len(rows)}")

    # With the full agent (requires API key)
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nSet OPENROUTER_API_KEY to run the full agent demo.")
        os.unlink(csv_path)
        return

    analyst = Friend(
        seed=(
            "You are a data analyst. When given a CSV file path, "
            "use the table tool to read it and answer questions with specific numbers."
        ),
        tools=["table", "code"],
        api_key=api_key,
        model="google/gemini-2.0-flash-exp:free",
        budget_usd=0.05,
    )

    print("\n--- Agent Analysis ---")
    response = analyst.chat(
        f"Read {csv_path} and tell me: "
        "(1) which region had the highest total revenue, "
        "(2) which product sold more units overall, "
        "(3) how many sales were over $1000."
    )
    print(response.text)

    os.unlink(csv_path)


if __name__ == "__main__":
    main()
