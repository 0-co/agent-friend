FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir mcp
ENTRYPOINT ["python3", "mcp_server.py"]
