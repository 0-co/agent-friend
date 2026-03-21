# agent-friend — MCP Schema Linter

Real-time MCP server schema grading in VS Code. Letter grades A+ through F. Catch token bloat, missing descriptions, and schema errors before deployment.

## The problem

The most popular MCP servers consume 20,000+ tokens before your agent does any work. Every tool description, every parameter without a type, every missing `minLength` — it all ends up in the context window on every single call. At GPT-4 pricing, a single poorly-designed MCP server can cost $47/session.

This extension shows you the grade in real-time, in your status bar, as you edit.

## What it does

- Detects MCP schema JSON files automatically (looks for a tools array with `name` + `inputSchema`)
- Runs `agent-friend validate` and `agent-friend grade` on open and save
- Displays the grade in the status bar: `MCP: A+ (95/100)`
- Shows individual issues as VS Code diagnostics (squiggles + Problems panel)
- Highlights the specific tool that has the issue

## Status bar icons

| Icon | Grade |
|---|---|
| `$(check)` | A or B — good schema |
| `$(warning)` | C — some issues |
| `$(error)` | D or F — significant problems |

## How it works

1. You open or save a `.json` file
2. The extension checks whether it looks like an MCP schema (tools array with `name` and `inputSchema`)
3. If yes: runs `agent-friend validate --json <file>` and `agent-friend grade --json <file>`
4. Validate issues appear as diagnostics. Grade appears in the status bar.
5. If the file is not an MCP schema, the extension stays completely silent.

## Requirements

You need `agent-friend` installed:

```
pip install agent-friend
```

If it's not found, the extension will show a message with a "Copy" button to copy the install command.

## Commands

- **agent-friend: Grade MCP Schema** (`agent-friend.grade`) — manually trigger grading on the active file. Also accessible by clicking the status bar item.

## Screenshot

![Status bar showing MCP grade and Problems panel with issues](https://raw.githubusercontent.com/0-co/agent-friend/main/docs/vscode-screenshot.png)

## Links

- [agent-friend on GitHub](https://github.com/0-co/agent-friend)
- [agent-friend on PyPI](https://pypi.org/project/agent-friend/)
- [MCP Server Leaderboard](https://0-co.github.io/company/leaderboard.html) — 200+ graded servers

## License

MIT
