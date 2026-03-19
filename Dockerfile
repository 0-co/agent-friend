FROM python:3.12-slim

WORKDIR /app

# Copy package files
COPY pyproject.toml README.md ./
COPY agent_friend/ agent_friend/
COPY mcp_server.py .

# Install agent-friend + MCP SDK
RUN pip install --no-cache-dir "mcp>=1.25,<2" && pip install --no-cache-dir -e .

# MCP stdio requires unbuffered output
ENV PYTHONUNBUFFERED=1

# Server communicates via stdin/stdout
CMD ["agent-friend-mcp"]
