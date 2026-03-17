FROM python:3.12-slim

WORKDIR /app

# Install MCP SDK (only external dependency for the MCP server)
RUN pip install --no-cache-dir "mcp>=1.25,<2"

# Copy the agent-friend package and MCP server
COPY agent_friend/ agent_friend/
COPY mcp_server.py .

# MCP stdio requires unbuffered output
ENV PYTHONUNBUFFERED=1

# Server communicates via stdin/stdout
CMD ["python", "mcp_server.py"]
