---
name: AI Cohort Site and Name
overview: Host the free AI open-source cohort "First Break AI" in a new GitHub Pages repo so community members can contribute; outline structure (checklist, setup, roadmap) with CONTRIBUTING.md and cross-links to thefirehacker.
todos: []
isProject: false
---

# First Break AI: Site Choice, Name, and Structure

## 1. New site vs this repo — recommendation

**Recommendation: Use a new repo (and new GitHub Pages site) for the cohort.**

Your instinct is right: **letting the most active and interested community members contribute** is a strong reason to separate the cohort from your personal blog.

### Why a new repo fits better

- **Safe contribution surface** — Contributors can open PRs for syllabus, exercises, roadmap, and fixes without touching your personal blog (thefirehacker). You keep the blog repo as yours; the cohort repo becomes the community’s.
- **Clear ownership** — “This repo is First Break AI; contribute here” is simple. People see a CONTRIBUTING.md, open issues for ideas, and submit PRs for new modules or improvements without needing access to your main site.
- **Community identity** — A dedicated repo (and URL like `firstbreak-ai.github.io` or under your org) makes the program feel like a shared project, not a subsection of your blog. Active members can become collaborators or maintainers over time.
- **Separation of concerns** — Your blog stays about you and AIEDX/BubblSpace. The cohort site stays about the learning path, checklist, and roadmap. You can still link from the cohort site to your TIL and blog posts for deeper dives.

### What you give up (and how to soften it)

- **Two sites to maintain** — Mitigation: keep the cohort site minimal (landing + checklist + setup + roadmap + maybe a “Resources” page that links to your blog/TIL). You don’t need to duplicate content; link out to thefirehacker for detailed posts.
- **New repo and Pages setup** — One-time cost: create repo, add a simple Quarto website (or even a single `index.html` + a few markdown pages), enable GitHub Pages. You can reuse the same pattern as this blog’s `[publish.yml](.github/workflows/publish.yml)` if you use Quarto in the new repo.

### Suggested setup for the new repo

- **Repo name** — `firstbreak-ai` or `first-break-ai` (matches the name; easy to find).
- **Site** — Quarto website (familiar to you, same stack as this blog) or a simple docs site (e.g. MkDocs or plain Markdown → HTML). Quarto keeps the “build a Quarto blog” milestone consistent for participants.
- **Contribution path** — Add a `CONTRIBUTING.md` (how to suggest changes, PR process, where to add new roadmap items or modules). Optionally: GitHub Discussions or a “Suggest a topic” issue template so active members can propose and vote on content.
- **Cross-links** — Cohort “Resources” or “Deep dives” section links to thefirehacker TIL and blog; your blog can have a single “Join First Break AI” call-to-action that points to the new site.

---

## 2. Cohort name and description

**Decided: First Break AI**

Name communicates the promise clearly: the program that gets you your first break in AI. Target audience: self-learners (B.E. / M.Tech / MBA) looking for first break or employment; learning is self- and community-driven.

**One-liner (tagline / hero):**  
First Break AI — your first break in AI. Free, open cohort to upskill in training, inference, and AI product building, and showcase what you build.

**Short (repo description / meta):**  
First Break AI is a free, open-to-all cohort for self-learners who want their first break in AI. Learn training and inference, build AI products, and grow a public portfolio — with the community, in the open.

**Medium (landing / About):**  
First Break AI is a free, community-driven cohort for anyone who wants their first break in AI. Whether you're from engineering, an MBA, or self-taught, we focus on what matters: running and training models, understanding inference, and shipping AI-powered products. Most learning is self-directed and peer-supported; the roadmap, checklist, and resources live in the open so you can contribute and others can follow. The goal is simple: upskill, build, showcase — and get that first role or first break in AI.

---

## 3. Where the checklist and roadmap live (new repo)

In the **new cohort repo**, keep one source of truth for the program.

- **Landing page** — `index.qmd` (or `index.html`): cohort name, one-line pitch, “Who it’s for,” and links to:
  - **Checklist** (accounts to create, who to follow).
  - **AI setup** (Cursor, ChatGPT, Open Router).
  - **Roadmap** (phases and learning objectives).
- **Content structure** — Prefer multiple pages from the start so contributors can PR individual sections:
  - `checklist.qmd`, `setup.qmd`, `roadmap.qmd`, and optionally `resources.qmd` (links to your thefirehacker TIL/blog and other references).
  - A simple nav in `_quarto.yml` (or equivalent) tying them together.
- **Contribution** — `CONTRIBUTING.md` with: how to propose new roadmap items or modules, how to fix typos/content, and (optional) issue templates for “Suggest a topic” or “Add a resource.” This makes it clear that active community members are welcome to help shape the program.
- **Cross-links** — From the cohort site, link to thefirehacker for deep dives (e.g. “For more on DDP, see [TIL: DDP from scratch](https://thefirehacker.github.io/til/ddp-python-basics)”). From thefirehacker blog, add a single “Join First Break AI” link in the nav or hero that points to the new site.

---

## 4. Disruptor-minded strategy (condensed)

- **Vision** — A world where anyone can upskill in AI training, inference, and product building through open-source tools and community, and showcase that publicly.
- **Core insight** — Most “AI upskilling” is either high-cost or shallow (prompt-only). A free, open-to-all cohort that goes from “run a model locally” to “build an AI product” and uses your own blog/TIL as the platform fills that gap.
- **Unfair advantage** — Your existing content (TIL, blog) and BubblSpace as a real product to “sign in and see.” The cohort site links to that content; the **new repo** lets the community own and extend the learning path while you stay the anchor.

---

## 5. Summary and next steps

- **Site**: Use a **new repo** (e.g. `firstbreak-ai`) with its own GitHub Pages site so community members can contribute to syllabus, roadmap, and resources without touching your personal blog.
- **Name**: **First Break AI** — your first break in AI; free, open, community-driven. Use the one-liner for hero/OG, short for repo/meta, medium for landing About.
- **Structure**: New repo with landing page, `checklist.qmd`, `setup.qmd`, `roadmap.qmd`, optional `resources.qmd`, and `CONTRIBUTING.md`. Link from the cohort site to thefirehacker for deep dives; add a “Join First Break AI” link from your blog to the new site.

When you’re ready to implement: (1) create the new repo and enable GitHub Pages, (2) add a minimal Quarto website with the checklist/setup/roadmap content and First Break AI description, (3) add `CONTRIBUTING.md` and optionally issue templates, (4) add a link from thefirehacker to the cohort site.