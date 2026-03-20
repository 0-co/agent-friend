"""Run agent-friend as an MCP stdio server: python -m agent_friend

When invoked as a module (python -m agent_friend), start the MCP server.
This is how Glama and other MCP clients call package-based servers.
For the interactive CLI, use the `agent-friend` command directly.
"""
from agent_friend.mcp_server import main

main()
