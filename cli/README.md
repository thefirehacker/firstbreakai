# @aiedx/firstbreakai

CLI and local MCP server for the [First Break AI](https://cohort.bubblnet.com) cohort.

Check your dev environment, track progress, validate exercises, scaffold a blog, and get AI-powered help — all from your terminal or any MCP-capable IDE.

## Install

```bash
npm install -g @aiedx/firstbreakai
```

Or run without installing:

```bash
npx @aiedx/firstbreakai doctor
```

**Requires:** Node.js >= 18

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

## MCP server (for Cursor, Claude Desktop, etc.)

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

This exposes these MCP tools to your AI assistant:

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
- [Discord](https://discord.gg/hRPese4H3F)
- [GitHub](https://github.com/thefirehacker/firstbreakai)

## License

MIT
