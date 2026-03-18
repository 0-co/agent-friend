"""Leaderboard benchmark data -- 50 popular MCP servers graded."""

# (server_id, display_name, score) sorted by score descending
LEADERBOARD = [
    ("postgres", "PostgreSQL MCP Server", 100.0),
    ("sqlite", "SQLite MCP Server", 99.7),
    ("e2b", "E2B MCP Server", 99.1),
    ("slack", "Slack MCP Server", 97.3),
    ("shadcn", "shadcn-ui MCP", 93.4),
    ("git", "Git MCP Server", 93.1),
    ("puppeteer", "Puppeteer MCP Server", 91.2),
    ("browsermcp", "BrowserMCP", 89.2),
    ("whatsapp", "WhatsApp MCP", 87.4),
    ("fastapi-mcp", "FastAPI-MCP", 85.6),
    ("ghidra", "GhidraMCP", 84.4),
    ("brave", "Brave Search MCP Server", 82.6),
    ("dbhub", "dbhub", 82.3),
    ("time", "Time MCP Server", 81.7),
    ("xiaohongshu", "Xiaohongshu MCP Server", 80.2),
    ("sequentialthinking", "Sequential Thinking", 79.9),
    ("googlemaps", "Google Maps MCP Server", 79.9),
    ("github", "GitHub MCP Server", 79.6),
    ("memory", "Memory MCP Server", 78.4),
    ("sentry", "Sentry MCP Server", 76.6),
    ("fetch", "Fetch MCP Server", 74.4),
    ("obsidian", "Obsidian MCP", 73.5),
    ("browserbase", "Browserbase MCP", 69.6),
    ("playwright", "Playwright MCP Server", 67.0),
    ("serena", "Serena", 67.0),
    ("filesystem", "Filesystem MCP Server", 64.9),
    ("chrome-devtools", "Chrome DevTools MCP Server", 64.9),
    ("genai-toolbox", "Google genai-toolbox", 64.3),
    ("excel", "Excel MCP Server", 63.8),
    ("stripe", "Stripe Agent Toolkit MCP Server", 62.5),
    ("atlassian", "Atlassian MCP Server", 62.2),
    ("figma", "Figma-Context-MCP", 61.9),
    ("magic-mcp", "magic-mcp", 58.3),
    ("chart", "Chart MCP Server", 56.5),
    ("google-workspace", "Google Workspace MCP", 54.8),
    ("blender", "Blender MCP Server", 54.2),
    ("exa", "Exa MCP Server", 53.0),
    ("aws", "AWS MCP Server", 52.2),
    ("github-official", "GitHub MCP Server (Official)", 52.1),
    ("cloudflare", "Cloudflare Radar MCP Server", 51.4),
    ("pal", "PAL MCP Server", 49.0),
    ("tavily", "Tavily MCP Server", 48.1),
    ("n8n-mcp", "n8n-mcp", 47.7),
    ("mcp-chrome", "mcp-chrome", 44.9),
    ("ga4", "Google Analytics (GA4) MCP Server", 40.0),
    ("context7", "Context7 MCP Server", 39.5),
    ("firecrawl", "Firecrawl MCP Server", 35.8),
    ("desktop-commander", "Desktop Commander MCP Server", 30.8),
    ("grafana", "Grafana MCP", 21.9),
    ("notion", "Notion MCP Server", 19.8),
]

LEADERBOARD_URL = "https://0-co.github.io/company/leaderboard.html"


def get_leaderboard_position(score):
    """Return (rank, total, servers_above, servers_below) for a given score.

    rank: 1-indexed position (1 = best)
    total: total servers in leaderboard
    servers_above: list of (name, score) for the 2 servers immediately above
    servers_below: list of (name, score) for the 2 servers immediately below
    """
    scores = [s[2] for s in LEADERBOARD]
    total = len(scores)

    # Find rank (1-indexed, higher score = better rank)
    # Count how many leaderboard entries have a strictly higher score,
    # then add 1. Ties share the same rank position.
    rank = 1
    for s in scores:
        if s > score:
            rank += 1

    # Get neighboring servers
    # "above" = those with higher scores (better rank)
    idx = rank - 1  # 0-indexed insertion point
    servers_above = [(LEADERBOARD[i][1], LEADERBOARD[i][2]) for i in range(max(0, idx - 2), idx)]
    servers_below = [(LEADERBOARD[i][1], LEADERBOARD[i][2]) for i in range(idx, min(total, idx + 2))]

    return rank, total, servers_above, servers_below
