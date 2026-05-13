---
name: First Break AI MCP Server
overview: Build an MCP server inside this repo as a Cloudflare Pages Function at `cohort.bubblnet.com/mcp`, using the official Cloudflare Agents SDK (`createMcpHandler`) and the MCP TypeScript SDK. Phase 1 ships 5 read-only tools wrapping public cohort content and HuggingFace data, with no auth, deployed via the existing GitHub Actions pipeline. Phase 2 adds cohort progress tracking (D1 + HF token auth + auto-verified self-reports + Reddit-style points/streaks/trophies) without any Discord dependency. Compatible with Cursor, Claude Code, Claude Desktop, HuggingChat, and VS Code / GitHub Copilot out of the box.
todos:
  - id: deps
    content: Add agents, @modelcontextprotocol/sdk, zod to package.json
    status: pending
  - id: config
    content: Add tsconfig.json and wrangler.jsonc for Pages Functions TS
    status: pending
  - id: index-builder
    content: Write scripts/build-content-index.mjs that walks .qmd files and writes public/mcp-content-index.json
    status: pending
  - id: quarto-hook
    content: Append build-content-index.mjs to _quarto.yml post-render
    status: pending
  - id: mcp-handler
    content: Create functions/mcp/[[path]].ts with createMcpHandler and 5 tools
    status: pending
  - id: middleware-check
    content: Verify functions/_middleware.ts does not interfere with /mcp; scope it if needed
    status: pending
  - id: smoke-test
    content: Deploy to preview, hit /mcp with a test client, verify all 5 tools respond
    status: pending
  - id: lesson-update
    content: Add a short section to lesson-1 (or new lesson-2) showing students how to add the MCP server URL in their tool
    status: pending
  - id: phase2-d1
    content: "Phase 2 — Provision D1 database + binding in wrangler.jsonc; create migrations/ folder with students + events schema"
    status: pending
  - id: phase2-auth
    content: "Phase 2 — Add HF token passthrough (Authorization: Bearer header → huggingface.co/api/whoami → cache hf_username on context)"
    status: pending
  - id: phase2-verify
    content: "Phase 2 — Build functions/lib/verify.ts with verify_blog_post, verify_hf_model, verify_github_pr (auto-verification before points awarded)"
    status: pending
  - id: phase2-write-tools
    content: "Phase 2 — Add write tools: submit_artifact, complete_lesson, set_privacy"
    status: pending
  - id: phase2-read-tools
    content: "Phase 2 — Add read tools: my_progress, leaderboard, cohort_stats, student_profile"
    status: pending
  - id: phase2-trophies
    content: "Phase 2 — Define trophies table + milestone triggers (first model, lesson-1 done, 7-day streak, etc.)"
    status: pending
  - id: phase2-footer
    content: "Phase 2 — Add per-response footer (Streak / Points / Rank) injected into every tool result"
    status: pending
isProject: false
---

# First Break AI MCP Server — Plan

## Why this approach

The repo is already a Cloudflare Pages site with one Pages Function ([functions/_middleware.ts](functions/_middleware.ts)). Adding an MCP server is **one new file** in `functions/`, plus a small content index built at render time. No Python, no Gradio, no separate hosting account, no new domain.

Cloudflare's Agents SDK now ships `createMcpHandler` — a stateless Streamable-HTTP MCP server (the current MCP spec since March 2025) that runs natively in Pages Functions / Workers. It is the correct primitive for this.

## Architecture

```mermaid
flowchart LR
    Cursor["Cursor"] --> CF
    ClaudeCode["Claude Code"] --> CF
    ClaudeDesktop["Claude Desktop"] --> CF
    HuggingChat["HuggingChat"] --> CF
    VSCode["VS Code Copilot"] --> CF
    CF["cohort.bubblnet.com/mcp<br/>Pages Function"] --> Tools["Tool handlers"]
    Tools --> Index["public/mcp-content-index.json<br/>built from .qmd files"]
    Tools --> HF["HuggingFace API<br/>/croissant, /api/datasets/..."]
    Tools --> GH["GitHub API<br/>(read-only, public repos)"]
```

## Client compatibility (one URL, all clients)

- **Cursor:** add server URL in MCP settings → done
- **Claude Code:** add to `.claude/settings.json` with `"url": "https://cohort.bubblnet.com/mcp"`
- **Claude Desktop:** custom MCP server via settings UI
- **HuggingChat:** "Add MCP server" dialog (the one in your screenshot) — paste URL
- **VS Code / GitHub Copilot:** `.vscode/mcp.json` with `"type": "http"`
- **ChatGPT:** not supported (different protocol — OpenAPI Actions, not MCP). Out of scope.

## Phase 1 tools (read-only, no auth)

All tools wrap **public** data, so no auth is needed. Per earlier discussion, defer auth until Phase 2 when write/private tools are added.

- `search_lesson(query: string)` — semantic-ish keyword search across all `.qmd` files in [lessons/](lessons/), [blog/](blog/), [office-hours/](office-hours/). Returns matching paragraphs with source path + URL.
- `get_lesson(slug: string)` — fetch full markdown of a lesson by slug (e.g. `lesson-1-huggingface-beyond-upload`).
- `inspect_dataset_schema(hf_dataset_id: string)` — hits `https://huggingface.co/api/datasets/<id>/croissant`, parses Croissant JSON-LD, returns clean column-name → type table. The exact thing demonstrated in Lesson 1.
- `inspect_binary_dataset(hf_dataset_id: string)` — for repos like `kjj0/fineweb10B-gpt2` where Croissant is empty, lists `.bin` files and points at the linked source code repo for binary format. Implements the "follow the trail" pattern from Lesson 1.
- `get_homework(lesson_number: number)` — extracts the `## Homework` section from a lesson `.qmd`. Useful for "what is the homework for lesson 1?".

## Files to add / change

### New files

- [functions/mcp/[[path]].ts](functions/mcp/%5B%5Bpath%5D%5D.ts) — Pages Function catch-all that mounts `createMcpHandler`. Defines the 5 tools above using `zod` schemas.
- [scripts/build-content-index.mjs](scripts/build-content-index.mjs) — runs at render time, walks `lessons/`, `blog/`, `office-hours/`, parses each `.qmd` (frontmatter + section headings + paragraphs), writes `public/mcp-content-index.json`. The MCP function imports this JSON at request time (Pages Functions can fetch static assets via `env.ASSETS.fetch`).
- [tsconfig.json](tsconfig.json) — minimal config for `functions/*.ts` (Pages Functions can compile TS without one, but having it helps editor and prevents drift).
- [wrangler.jsonc](wrangler.jsonc) — Pages compatibility date + flags (so `nodejs_compat` is on if needed for any deps).

### Modified files

- [package.json](package.json) — add three deps:
  - `agents` (Cloudflare Agents SDK — provides `createMcpHandler`)
  - `@modelcontextprotocol/sdk`
  - `zod`
- [_quarto.yml](_quarto.yml) — append `scripts/build-content-index.mjs` to `project.post-render` so the content index is rebuilt every render.
- [functions/_middleware.ts](functions/_middleware.ts) — verify Fetchlens (`observeOnly: true`) does not interfere with `/mcp` POST requests. If it adds latency to streaming, scope middleware to non-`/mcp` paths via an early return.
- [.github/workflows/publish.yml](.github/workflows/publish.yml) — confirm `functions/` is included in the deploy artifact (currently `_cf_deploy` is just `docs/`; `functions/` lives at the project root and Wrangler picks it up automatically — verify on first deploy).

## Skeleton for the main handler

Conceptual shape, not final code:

```typescript
// functions/mcp/[[path]].ts
import { createMcpHandler } from "agents/mcp";
import { z } from "zod";

const handler = createMcpHandler((server) => {
  server.tool(
    "search_lesson",
    "Search across First Break AI lessons, blog posts, and office hours.",
    { query: z.string() },
    async ({ query }) => {
      const idx = await env.ASSETS.fetch("/mcp-content-index.json").then(r => r.json());
      // simple ranked keyword search
      return { content: [{ type: "text", text: formatHits(idx, query) }] };
    }
  );

  server.tool(
    "inspect_dataset_schema",
    "Returns column names and types for any HuggingFace dataset by parsing its Croissant manifest.",
    { hf_dataset_id: z.string() },
    async ({ hf_dataset_id }) => {
      const r = await fetch(`https://huggingface.co/api/datasets/${hf_dataset_id}/croissant`);
      // walk JSON for extract.column entries (same logic from lesson 1)
      return { content: [{ type: "text", text: schemaTable(await r.json()) }] };
    }
  );

  // ... 3 more tools
});

export const onRequest = handler;
```

## Auth posture

**Phase 1: no auth.** All Phase 1 tools call public endpoints. Friction-free onboarding for cohort students.

**Phase 2 (deferred):** Add HuggingFace token passthrough — students send `Authorization: Bearer <hf-token>` in the MCP "HTTP Headers" field; server validates by calling `huggingface.co/api/whoami`. Then we can add tools like `get_my_uploaded_models()`, `submit_homework_link(...)` that touch private/user-specific data.

---

## Phase 2: Cohort progress + Reddit-style points

Self-reported homework, auto-verified server-side, gamified with points/streaks/trophies. Pure MCP, no Discord required (but Discord bot can later share the same backend).

### Why pure MCP fits

Students live inside their AI tool. `submit_artifact("blog_post", "https://...")` from inside Cursor or Claude Code has zero context-switch cost. Verification + leaderboards happen on the server, not in the AI prompt.

### Storage: Cloudflare D1

D1 is SQLite at the edge — same Cloudflare account, no new service. Bind it in [wrangler.jsonc](wrangler.jsonc), expose it as `env.DB` inside the MCP handler.

```sql
-- migrations/0001_init.sql
CREATE TABLE students (
  hf_username      TEXT PRIMARY KEY,
  discord_handle   TEXT,
  github_handle    TEXT,
  joined_at        INTEGER NOT NULL,
  privacy          TEXT DEFAULT 'private'  -- 'public' | 'anonymous' | 'private'
);

CREATE TABLE events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  hf_username   TEXT NOT NULL,
  type          TEXT NOT NULL,   -- 'lesson_complete' | 'blog_post' | 'model_upload' | 'pr_merged' | 'office_hours_attended'
  payload       TEXT NOT NULL,   -- JSON
  points        INTEGER NOT NULL,
  verified      INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL,
  FOREIGN KEY (hf_username) REFERENCES students(hf_username)
);

CREATE TABLE trophies (
  hf_username   TEXT NOT NULL,
  trophy_id     TEXT NOT NULL,   -- 'first_model', 'lesson_1_done', '7_day_streak', etc.
  awarded_at    INTEGER NOT NULL,
  PRIMARY KEY (hf_username, trophy_id)
);

CREATE INDEX events_user_time ON events(hf_username, created_at);
CREATE INDEX events_type_time ON events(type, created_at);
```

### Identity = HF token passthrough

Students send `Authorization: Bearer <hf-token>` (one of the very tokens they already create in Lesson 1). The MCP middleware:

1. Reads the header
2. Calls `https://huggingface.co/api/whoami-v2` to validate
3. Caches `hf_username` on the request context for the tool handler
4. Auto-creates a `students` row on first call (privacy defaults to `private`)

No login flow, no separate auth server.

### Auto-verification (the trust layer)

Self-report is the *interface*; verification is the *backend*. Points are only written after server-side checks pass. Logic lives in [functions/lib/verify.ts](functions/lib/verify.ts):

- `verify_blog_post(url)` — `fetch(url)`, look for required cohort backlink (`cohort.bubblnet.com/lessons/...`), confirm `<time>` tag is recent.
- `verify_hf_model(repo_id, hf_username)` — call HF API, confirm repo exists and `author == hf_username`.
- `verify_github_pr(pr_url, github_handle)` — call GitHub API, confirm PR is merged and author matches.

If verification fails: event recorded with `verified = 0`, no points awarded, error returned.

### Reddit-style mechanics

Three distinct primitives, all mapped:

- **Points (cumulative, never decrease):**
  - Complete lesson: 100
  - Publish homework blog: 50
  - Upload model to HF: 200
  - Merged PR to firstbreakai repo: 150
  - Attend office hours: 75
  - Daily streak bonus: +10/day, multiplier at 7/14/30 days
- **Trophies (one-time milestones):** `first_model`, `lesson_1_done`, `7_day_streak`, `30_day_streak`, `first_pr_merged`, `cohort_graduate`. Inserted into `trophies` table by triggers in tool handlers.
- **Streak (consecutive days with at least one verified event):** computed on read from `events` table; reset on miss.

### New write tools (Phase 2)

- `submit_artifact(type, url, description?)` — generic submission. Routes to the right verifier, records event, awards points if verified.
- `complete_lesson(lesson_number, blog_post_url)` — convenience wrapper; verifies the blog post and marks lesson done.
- `set_privacy(mode: "public" | "anonymous" | "private")` — controls leaderboard visibility.
- `link_handles(discord_handle?, github_handle?)` — student attaches their Discord/GitHub names to their HF username. Required before GH-PR verification works.

### New read tools (Phase 2)

- `my_progress()` — current student's points, streak, trophies, last 10 events.
- `leaderboard(period: "week" | "month" | "all_time")` — top 10 by points; respects each student's privacy setting.
- `cohort_stats()` — totals across cohort (models uploaded, blog posts, PRs merged, etc.).
- `student_profile(hf_username)` — public profile if that student is `public`; anonymous summary if `anonymous`; "not visible" if `private`.

### Per-response footer (gamification surface)

Every tool result gets a small footer appended by the server:

```
Streak: 7 | Points: 425 | Rank: #12 of 47 | Next trophy: 14-day streak (7 days to go)
```

This means *every* MCP call — even a read like `search_lesson` — reminds students of their progress. Cheap, sticky, no new UI.

### Privacy posture

Default: `private`. Three opt-in modes:

- `public` — appears on leaderboard with HF handle.
- `anonymous` — appears as "Student #047" with points, no handle.
- `private` — never appears; only `my_progress()` works.

### Discord bot is later, shares the same D1

```mermaid
flowchart LR
    Cursor --> MCP
    ClaudeCode --> MCP
    DiscordUser["Discord user"] --> Bot["Discord bot (later)"]
    MCP["MCP server (this repo)"] --> D1[("D1 database")]
    Bot --> D1
    Verify["verify.ts"] --> HF["HF API"]
    Verify --> GH["GitHub API"]
    MCP --> Verify
```

Discord bot becomes the **social layer** (announce milestones, post weekly leaderboard, react to streaks). MCP server is the **developer layer** (submit, query). Single source of truth in D1.

### Cohort-narrative payoff

Lesson 1 ends with *"could an AI agent figure out how to use this without a human in the loop?"*. Phase 2 is the live answer: a student's AI agent can submit homework, check streaks, see the leaderboard — entirely without a browser. The supply-chain idea, applied to the cohort itself.

---

## Setup snippets students will paste

Once deployed, the cohort gets one URL and these snippets:

**Cursor / Claude Desktop / HuggingChat:** paste `https://cohort.bubblnet.com/mcp` in the "Add MCP Server" dialog.

**Claude Code** — `.claude/settings.json`:
```json
{
  "mcpServers": {
    "firstbreakai": { "url": "https://cohort.bubblnet.com/mcp" }
  }
}
```

**VS Code / GitHub Copilot** — `.vscode/mcp.json`:
```json
{
  "servers": {
    "firstbreakai": { "type": "http", "url": "https://cohort.bubblnet.com/mcp" }
  }
}
```

## Risks / open items

- **Middleware compatibility:** [functions/_middleware.ts](functions/_middleware.ts) wraps every request with Fetchlens analytics. Need to verify it does not break MCP streaming responses. Mitigation: add path-based early return for `/mcp` if it does.
- **`docs/` vs `functions/` deploy:** the deploy artifact is currently `_cf_deploy` (= `docs/`). Cloudflare Pages picks up `functions/` separately from the project root — the [.github/workflows/publish.yml](.github/workflows/publish.yml) checks the repo out in the deploy job, so this should work. Verify on first deploy with a smoke test endpoint.
- **MCP spec drift:** Streamable HTTP became the spec March 2025, SSE deprecated. Cloudflare's `createMcpHandler` uses Streamable HTTP by default but supports SSE fallback for older clients. No action needed — just be aware.
- **GitHub-hosted MCP:** GitHub itself runs an official MCP server, but for *consuming* MCP, students use VS Code Copilot config (covered above). There is no separate "GitHub" install path for our server.

## What this delivers

A working MCP server at `cohort.bubblnet.com/mcp` that any student in any AI-coding tool can add with one URL and start asking questions like "search the lessons for Croissant" or "show me the schema for FineWeb" — answered using the same code paths the lesson teaches them. The server itself becomes a Phase-2 cohort project (students can submit tools as PRs).