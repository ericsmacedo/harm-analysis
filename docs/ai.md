# AI assistants (MCP)

`harm-analysis` includes an [MCP](https://modelcontextprotocol.io) server, so AI
assistants such as Claude or GitHub Copilot can run the analyses directly on your
files. It runs locally, on your machine, and exposes two tools:

- `run_harm_analysis`: signals with an injected tone (SNR, THD, SINAD, THD+N, ...).
- `run_spec_analysis`: detect DC, tones and noise.

Each takes the sampling frequency `fs` and either `file_path` (a text file of
samples) or `samples` (a list of numbers), and returns the same fields as
`HarmResult` / `SpecResult`. Values that are not finite are returned as `null`.

## Install

The server needs the `mcp` extra:

```bash
pip install "harm-analysis[mcp]"
```

This installs the `harm-analysis-mcp` command. You can also run it without
installing anything, with [uv](https://docs.astral.sh/uv/):
`uvx --from "harm-analysis[mcp]" harm-analysis-mcp`.

## Configure your assistant

The assistant starts the server itself. Add it to your client's configuration;
check your client's documentation for the current details.

**Claude Code**:

```bash
claude mcp add harm-analysis -- uvx --from "harm-analysis[mcp]" harm-analysis-mcp
```

**Claude Desktop** (Settings, Developer, Edit Config, then restart the app):

```json
{
  "mcpServers": {
    "harm-analysis": {
      "command": "uvx",
      "args": ["--from", "harm-analysis[mcp]", "harm-analysis-mcp"]
    }
  }
}
```

**GitHub Copilot in VS Code** (`.vscode/mcp.json`; the tools are available in
agent mode):

```json
{
  "servers": {
    "harm-analysis": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "harm-analysis[mcp]", "harm-analysis-mcp"]
    }
  }
}
```

If the client can't find `uvx`, use its full path: desktop apps often don't see
your shell's `PATH`. Pass absolute file paths to the tools.

## Skill

The repository also contains a skill,
[`skills/harm-analysis/SKILL.md`](https://github.com/ericsmacedo/harm-analysis/blob/main/skills/harm-analysis/SKILL.md),
that tells an assistant when to use which analysis, how to read the results and
which pitfalls to check (for example, a stronger interferer being taken as the
fundamental). It works with or without the MCP server. For Claude Code, copy the
`harm-analysis` folder to `~/.claude/skills/` (all projects) or to
`.claude/skills/` in a project.
