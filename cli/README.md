# @aiedx/firstbreakai

The official CLI and local MCP server for [First Break AI](https://cohort.bubblnet.com) — **the first fully agentic AI cohort**.

First Break AI is a cohort **run by AI agents**. AI agents answer your questions, validate your code against rubrics, and track your progress across the website, your terminal, and your IDE. The agentic backbone is powered by [FetchLens.ai](https://fetchlens.ai).

This package is how you put those **AI agents in your own terminal and IDE**: check your dev environment, track progress, validate exercises, scaffold a blog, and get AI-agent help — from the command line or any MCP-capable AI assistant (Cursor, Claude, Codex).

## Install

```bash
npm install -g @aiedx/firstbreakai
```

Or run without installing:

```bash
npx @aiedx/firstbreakai doctor
```

**Requires:** Node.js >= 18

## How AI agents run the cohort

First Break AI is **completely agentic** — every part of the cohort is powered by AI agents, not static videos and quizzes:

- **AI agents answer your questions** — the on-site assistant and `firstbreakai ask` know the syllabus, lessons, and your current step.
- **AI agents validate your code** — `firstbreakai validate` runs deterministic checks against the rubric and reports exactly what passed.
- **AI agents track your progress** — log in with Discord and your progress syncs across the CLI, the website, and the MCP server.
- **AI agents live in your IDE** — run `firstbreakai mcp` (or use the remote MCP server) so Cursor, Claude Desktop, Claude Code, and OpenAI Codex can call cohort tools directly.
- **Powered by [FetchLens.ai](https://fetchlens.ai)** — the agentic AI platform that provides the MCP server backbone, the AI-agent assistant, and authenticated progress tracking.

## Commands

| Command | Description |
|---------|-------------|
| `firstbreakai doctor` | Check your dev environment (Git, Python, Quarto, HF CLI, Node.js, Cursor) |
| `firstbreakai status` | Show your cohort progress |
| `firstbreakai done <step>` | Mark a step as complete |
| `firstbreakai init` | Scaffold a new Quarto blog |
| `firstbreakai open <page>` | Open a cohort page in your browser |
| `firstbreakai next` | Open the next incomplete step |
| `firstbreakai ask "question"` | Ask the FBA AI assistant |
| `firstbreakai login` | Authenticate via Discord |
| `firstbreakai validate <step>` | Run local validation checks for a step |
| `firstbreakai mcp` | Start the MCP server for AI IDE integration |

## Quick start

```bash
# 1. Check your environment
firstbreakai doctor

# 2. Scaffold a blog
mkdir my-ai-blog && cd my-ai-blog
firstbreakai init

# 3. Preview it
quarto preview

# 4. Mark step 1 as done
firstbreakai done 1

# 5. See what's next
firstbreakai next
```

## MCP server — give your AI agent cohort superpowers

Run First Break AI as a local [Model Context Protocol](https://modelcontextprotocol.io) server so the **AI agent inside your IDE** (Cursor, Claude Desktop, Claude Code, OpenAI Codex) can call cohort tools directly — no context switching, fully agentic.

Add to your `.cursor/mcp.json` or Claude Desktop config:

```json
{
  "mcpServers": {
    "firstbreakai": {
      "command": "npx",
      "args": ["@aiedx/firstbreakai", "mcp"]
    }
  }
}
```

This exposes these agentic cohort tools to your AI agent:

- `cohort_doctor` — environment check
- `cohort_status` — progress overview
- `cohort_done` — mark step complete
- `cohort_validate` — run validation checks
- `cohort_open` — open cohort pages
- `cohort_next` — open next step
- `cohort_ask` — ask the FBA AI assistant

## Validation

`firstbreakai validate <step>` runs deterministic, local checks — no LLM, no network required.

**Step 1** (Quarto blog): checks `_quarto.yml`, `.qmd` files, Git repo, GitHub remote.

**Step 2** (Run a model locally): checks `run.c`/binary, `config.json`, tokenizer, model weights.

If you're logged in (`firstbreakai login`), results are also saved to your profile on the FBA server.

## Links

- [Cohort homepage](https://cohort.bubblnet.com)
- [Roadmap](https://cohort.bubblnet.com/roadmap)
- [Lessons](https://cohort.bubblnet.com/lessons/)
- [Checklist](https://cohort.bubblnet.com/checklist)
- [GitHub](https://github.com/thefirehacker/firstbreakai)

## License

MIT
