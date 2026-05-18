# SEO & Indexing Efforts Log — cohort.bubblnet.com

> Single source of truth for all SEO work on `cohort.bubblnet.com`. Owner: TheFireHacker. Last updated: **18 May 2026**.

This document records every technical fix, every Google Search Console interaction, every backlink built, the current state, and explicit instructions for any human or AI agent who needs to continue the work.

**If you are an AI agent reading this for the first time:** start with [Section 10 — AI Agent Instructions](#section-10--ai-agent-instructions). It tells you what to do, what NOT to do, and how to update this file. Then read the [Snapshot](#section-2--snapshot) to understand the current state.

**Companion docs:** [SEO-Explained.md](../SEO-Explained.md) (plain-English clean-URL walkthrough), [DEVELOPMENT.md](../DEVELOPMENT.md) (deploy + infra), [README.md](../README.md) (course overview).

---

## Table of contents

1. [Header & conventions](#section-1--header--conventions)
2. [Snapshot](#section-2--snapshot) — current state at a glance
3. [Site Setup](#section-3--site-setup) — technical SEO foundations
4. [Indexing Infrastructure](#section-4--indexing-infrastructure) — post-render scripts
5. [Redirect Bridge](#section-5--redirect-bridge) — old domain + apex + parent
6. [Backlink & Authority Strategy](#section-6--backlink--authority-strategy)
7. [GSC Activity Log](#section-7--gsc-activity-log)
8. [Weekly Metrics Timeline](#section-8--weekly-metrics-timeline)
9. [Pending High-Leverage Actions](#section-9--pending-high-leverage-actions)
10. [AI Agent Instructions](#section-10--ai-agent-instructions)
11. [File Reference Map](#section-11--file-reference-map)
12. [Glossary](#section-12--glossary)

---

## Section 1 — Header & conventions

- **Target domain:** `https://cohort.bubblnet.com/`
- **Migrated from:** `https://thefirehacker.github.io/firstbreakai/` (April 2026)
- **Sister/bridge domain:** `https://www.bubblspace.com/firstbreakai` (Next.js landing, indexed)
- **Parent domain:** `https://bubblnet.com/` (currently content-empty, 301s to cohort)
- **Stack:** Quarto (static) → Cloudflare Pages
- **CI:** [.github/workflows/publish.yml](../.github/workflows/publish.yml) on push to `main`
- **Analytics:** Google Analytics 4 — property `G-VB604Y51JX`
- **Search Console properties:**
  - `https://cohort.bubblnet.com/` (URL-prefix property) — primary
  - `https://thefirehacker.github.io/firstbreakai/` (URL-prefix property) — bridge, mostly inactive

**Conventions for this doc:**
- Dates in `YYYY-MM-DD` format.
- File links are relative to repo root, written as `[file](relative/path)`.
- Each entry has a status: `Done` / `In progress` / `Pending` / `Blocked` / `Won't do` / `Counterproductive (don't retry)`.
- "Leverage" = expected SEO impact: `High` / `Medium` / `Low`.
- When updating, **append** to logs; do not overwrite history. Update the Snapshot in place.

---

## Section 2 — Snapshot

**As of 18 May 2026:**

| Metric | Value | Trend |
| --- | --- | --- |
| Pages indexed in Google | **0** | flat |
| Pages "Crawled - currently not indexed" | 11 | growing |
| Latest validation cycle | **Failed** (started 13 May, failed 16 May) | — |
| Validation: pending URLs | 7 | stale (last crawled 24 Apr – 3 May) |
| Validation: failed URLs | 3 | crawled fresh 15 May, Google still declined |
| Sitemap status | Processed successfully | 26 URLs, last read 11 May |
| GA traffic (28 days) | 558 active users, 6.2K events, 2.2K views | +169% users vs prev period |
| Site uptime | 100% | Cloudflare Pages |
| TTFB (homepage) | 0.35s | excellent |
| Average page word count | ~6.3K (homepage), 17K (lesson-1) | not thin |

**Currently failed URLs (validation 16 May 2026):**

- `https://cohort.bubblnet.com/` — homepage
- `https://cohort.bubblnet.com/roadmap`
- `https://cohort.bubblnet.com/lessons/lesson-1-huggingface-beyond-upload`

**Currently pending URLs (validation, last crawled Apr 24 – May 3):**

- `https://cohort.bubblnet.com/office-hours/`
- `https://cohort.bubblnet.com/about`
- `https://cohort.bubblnet.com/lessons/lesson-0-welcome`
- `https://cohort.bubblnet.com/blog/`
- `https://cohort.bubblnet.com/lessons/`
- `https://cohort.bubblnet.com/office-hours/2026-04-24`
- `https://cohort.bubblnet.com/journey/`

**Diagnosis as of this snapshot:** Site is technically perfect (clean URLs, valid sitemap, no noindex anywhere, rich JSON-LD, fast TTFB, generous word counts, unique titles). The blocker is **trust budget**: subdomain on a content-empty parent, in a competitive niche, with mostly nofollow inbound links. See [Section 6](#section-6--backlink--authority-strategy) and [Section 9](#section-9--pending-high-leverage-actions).

---

## Section 3 — Site Setup

Every technical SEO foundation that's in place. All entries verified working as of 18 May 2026 unless noted.

### 3.1 Canonical URLs

- **Status:** Done
- **Implementation:** Quarto Lua filter at [canonical.lua](../canonical.lua) injects `<link rel="canonical" href="https://cohort.bubblnet.com/...">` matching the clean URL on every page.
- **Verified:** `curl -s https://cohort.bubblnet.com/ | grep canonical` → `<link rel="canonical" href="https://cohort.bubblnet.com/">`
- **All canonicals point to clean URLs** (no `.html` extensions, no `/index.html`).

### 3.2 Sitemap

- **Status:** Done, healthy
- **URL:** `https://cohort.bubblnet.com/sitemap.xml`
- **URLs in sitemap:** 26 (as of last render)
- **Post-processed** by [scripts/rewrite-sitemap.mjs](../scripts/rewrite-sitemap.mjs) so `<loc>` entries match canonicals (no `.html`, no `/index.html`).
- **Submitted to GSC:** 11 May 2026 — status "Sitemap processed successfully", 21 discovered pages reported at submit time (more discovered as content grew).
- **Verified:** `curl -I https://cohort.bubblnet.com/sitemap.xml` → HTTP 200, `content-type: application/xml`

### 3.3 robots.txt

- **Status:** Done
- **File:** [robots.txt](../robots.txt)
- **Contents:**
  ```
  User-agent: *
  Allow: /
  Disallow: /public/audio/
  Sitemap: https://cohort.bubblnet.com/sitemap.xml
  ```
- **`Disallow: /public/audio/`** added because audio files are hosted on R2 (firstbreakai.bubblnet.com) but the path existed locally and Googlebot was wasting crawl budget on missing files.

### 3.4 JSON-LD structured data

- **Status:** Done (basic), see [Section 9 #1, #2](#section-9--pending-high-leverage-actions) for gaps
- **File:** [includes/schema.html](../includes/schema.html) — injected via `_quarto.yml` `include-in-header`
- **Schemas present on homepage:** `WebSite`, `Course`, `PodcastSeries`, `FAQPage`
- **Schemas missing (HIGH-leverage gap):** `Organization` with `parentOrganization` pointing to bubblspace.com, business contact info, NPM/PyPI `sameAs` links. See [Section 9](#section-9--pending-high-leverage-actions) action #1.

### 3.5 Open Graph + Twitter Card

- **Status:** Done
- **Configured in:** [_quarto.yml](../_quarto.yml) under `open-graph:` and `twitter-card:`
- **OG image:** `public/images/Fort-Journey-01.png`
- **Twitter creator/site:** `@thefirehacker`

### 3.6 Page titles & descriptions

- **Status:** Done
- **Per-page title** from `.qmd` front-matter; site suffix " – First Break AI" appended automatically.
- **Per-page description** from `.qmd` front-matter `description:` field.
- **Verified unique** for homepage, `/roadmap`, `/lessons/lesson-1-huggingface-beyond-upload` via curl.

### 3.7 Content depth

- **Status:** Done (not thin)
- **Word counts (production, May 2026):**
  - Homepage: 6,360 words
  - Roadmap: 6,265 words
  - Lesson 1: 17,223 words
- Google's "thin content" threshold is typically <300 words. We are far above.

### 3.8 Page performance

- **Status:** Done
- **Homepage TTFB:** 0.35s (Cloudflare cached)
- **Homepage size:** 78 KB HTML
- **Largest image:** `firstbreakai-anatomy-poster-1500.webp` (212 KB, lazy-loaded, below the fold). Full PNG (4.2 MB) only fetched on click-to-enlarge.
- **CLS:** poster image has explicit `width="1332" height="1868"` to reserve layout.

---

## Section 4 — Indexing Infrastructure

The two post-render Node scripts that align the rendered site with Cloudflare's clean-URL behavior. **Don't break these.**

### 4.1 [scripts/rewrite-sitemap.mjs](../scripts/rewrite-sitemap.mjs)

- **Status:** Done
- **What it does:** Reads `docs/sitemap.xml` and rewrites `<loc>` entries:
  - `https://cohort.bubblnet.com/about.html` → `https://cohort.bubblnet.com/about`
  - `https://cohort.bubblnet.com/index.html` → `https://cohort.bubblnet.com/`
- **When it runs:** Every render (local preview AND CI). It is **not gated** by any env var.
- **Why:** Without this, sitemap URLs would 301-redirect to canonical, Google records "sitemap doesn't cover canonical URL", and indexing stalls with "No referring sitemaps detected".

### 4.2 [scripts/rewrite-html-links.mjs](../scripts/rewrite-html-links.mjs)

- **Status:** Done
- **What it does:** Walks `docs/**/*.html` and strips `.html` from internal `href` attributes. `href="/about.html"` → `href="/about"`. Also converts `href="…/index.html"` → `href="…/"`.
- **When it runs:** Only when env var `QUARTO_PROJECT_RENDER_ALL=1` is set, i.e. only during a full `quarto render` (CI). Skipped during `quarto preview` and single-file renders.
- **Why gated:** Quarto's local preview server doesn't auto-resolve clean URLs the way Cloudflare does. Without the guard, every internal link 404'd locally.
- **Why correct:** `docs/` is `.gitignore`d, so local builds never leak to production. CI always does a full render and gets the rewrite. See [SEO-Explained.md](../SEO-Explained.md).
- **Impact:** Eliminated ~470 internal 301/308 redirect chains site-wide.

### 4.3 Wiring in [_quarto.yml](../_quarto.yml)

```yaml
project:
  post-render:
    - scripts/rewrite-sitemap.mjs
    - scripts/rewrite-html-links.mjs
```

### 4.4 CI workflow [.github/workflows/publish.yml](../.github/workflows/publish.yml)

- Triggered on push to `main`
- Runs bare `quarto render` (full project render — this sets `QUARTO_PROJECT_RENDER_ALL=1`, which triggers the link rewriter)
- Uploads `docs/` as artifact
- Deploys via `cloudflare/wrangler-action@v3`
- **Do not change to single-file render.** Doing so would skip the link rewriter and ship `.html` hrefs to production.

---

## Section 5 — Redirect Bridge

### 5.1 Apex domain → cohort subdomain

- **Status:** Done
- **Mechanism:** Cloudflare Redirect Rule on `bubblnet.com` account, named "Redirect bubblnet.com and www to cohort"
- **Rule:** `http.host eq "bubblnet.com"` OR `http.host eq "www.bubblnet.com"` → 301 redirect (dynamic, preserves path) to `https://cohort.bubblnet.com{path}`
- **Verified:** `curl -I https://bubblnet.com/` → HTTP 301, `location: https://cohort.bubblnet.com/`

### 5.2 Old GitHub Pages → new domain (homepage)

- **Status:** Done (mediumstrong signal)
- **URL:** `https://thefirehacker.github.io/firstbreakai/`
- **Mechanism:** HTTP 200 with `<meta http-equiv="refresh" content="0; url=https://cohort.bubblnet.com/">` + JS `location.replace(...)`
- **Behavior:**
  - Browser users: instant redirect to new domain
  - Googlebot: meta-refresh treated as a 301-equivalent
- **Verified:** `curl https://thefirehacker.github.io/firstbreakai/` returns redirect HTML.

### 5.3 Old GitHub Pages → new domain (inner pages, via 404.html)

- **Status:** Done (weakerlinkequity signal)
- **Mechanism:** GitHub Pages serves a custom `404.html` for any path not found. The 404 body contains JS:
  ```js
  var oldPrefix = "/firstbreakai";
  var p = location.pathname;
  if (p.indexOf(oldPrefix) === 0) p = p.slice(oldPrefix.length) || "/";
  location.replace("https://cohort.bubblnet.com" + p + location.search + location.hash);
  ```
- **Behavior:**
  - Browser users: path-preserving redirect (e.g. `/firstbreakai/about.html` → `https://cohort.bubblnet.com/about.html`)
  - Googlebot: sees HTTP 404 first, weaker signal than a real 301
- **Caveat:** HTTP 404 status means Google de-indexes old URLs (good for cleanup) but transfers minimal link equity (bad for authority pass-through). Acceptable for a young site without much historical link equity.

### 5.4 Change of Address tool in GSC

- **Status:** **BLOCKED — DO NOT RETRY**
- **Why:** The old GSC property is a URL-prefix property on `github.io`. Change of Address requires a domain-level property. We don't own `github.io`, so Step 2 of the Change of Address wizard is permanently greyed out.
- **Workaround:** Rely on the 301-equivalents in 5.2 and 5.3 plus passive Google re-crawl. No further action possible at this level.

### 5.5 Sitemap on old GitHub Pages property

- **Status:** **Counterproductive — DO NOT RETRY**
- **What happened:** Submitted `/sitemap.xml` to `thefirehacker.github.io/firstbreakai/` property. Returned "Couldn't fetch" because the path no longer serves a sitemap file (or returns 404).
- **Why we don't want to fix this:** Even if we hosted a sitemap there, listing old URLs would only signal "pay attention to old domain" — opposite of intent. Submitting new domain URLs is rejected (cross-domain).

---

## Section 6 — Backlink & Authority Strategy

### 6.1 What's been done

| Source | Type | Authority pass | Status |
| --- | --- | --- | --- |
| `bubblspace.com/firstbreakai` Next.js landing | Dofollow, sister-domain | Medium (bubblspace also young) | **Done — indexed quickly** |
| Reddit posts | nofollow ugc | ~zero | Ongoing (engagement, not SEO) |
| Discord server | private, not crawled | zero | Ongoing |
| `thefirehacker.github.io` profile | Dofollow but low-authority subdomain | Weak | Done |
| GitHub repo README + DEVELOPMENT.md | Dofollow on github.com | Weak (GitHub mutes authority pass) | Done |
| YouTube playlist + Lesson 0 video description | nofollow | zero (but brand signal) | Done |

### 6.2 Comparison: why fetchlens.ai got indexed in 2 days and we haven't

`fetchlens.ai` was created on **9 April 2026** — younger than the cohort subdomain. It was indexed within 2 days. What it has that we don't:

| Signal | fetchlens.ai | cohort.bubblnet.com |
| --- | --- | --- |
| Apex vs subdomain | Apex (`.ai`) | Subdomain on dormant parent |
| `parentOrganization` JSON-LD | Yes → `https://bubblspace.com` | **No** |
| NPM/PyPI backlinks | 3 (`@aiedx/fetchlens-next`, `@aiedx/fetchlens-cloudflare`, `@aiedx/fetchlens-core`) | **None** |
| Visible business signals (address, phone, email) | Yes (Mumbai address, phone, email in footer) | **None visible** |
| Organization schema | Yes | **No** |
| Parent domain has content | n/a (apex) | **No** (bubblnet.com is just a redirect) |

**Conclusion:** The trust-graph signal is the gap, not site quality. See [Section 9](#section-9--pending-high-leverage-actions) for the action items that close this gap.

### 6.3 What's been verified about traffic

- Real users: 558 active users / 28 days (~20/day), trending up 169% vs previous period.
- Real engagement: 6.2K events / 28 days, 2.2K views.
- Conclusion: traffic is real but modest. Google does not weight traffic volume as an indexing signal — it weights source authority. We need authoritative inbound links, not more traffic.

---

## Section 7 — GSC Activity Log

Append-only chronological log of every Google Search Console interaction. Newest at the bottom.

| Date | Property | Action | Target | Outcome |
| --- | --- | --- | --- | --- |
| 2026-04-24 | cohort.bubblnet.com | Auto-discovery crawl | `/journey/` | Crawled - currently not indexed |
| 2026-04-25 | cohort.bubblnet.com | Auto-discovery crawl | `/office-hours/2026-04-24` | Crawled - currently not indexed |
| 2026-04-26 | cohort.bubblnet.com | Auto-discovery crawl | `/lessons/` | Crawled - currently not indexed |
| 2026-05-03 | cohort.bubblnet.com | Auto-discovery crawl | `/office-hours/`, `/about`, `/lessons/lesson-0-welcome`, `/blog/` | Crawled - currently not indexed |
| 2026-05-05 | thefirehacker.github.io/firstbreakai | Sitemap submission | `/sitemap.xml` | **Failed: Couldn't fetch** (no sitemap at that path post-migration) |
| 2026-05-07 | cohort.bubblnet.com | Validate Fix clicked on "Redirect error" issue | `/about/`, `/journey` | Pending |
| 2026-05-08 | cohort.bubblnet.com | Validate Fix clicked on "Crawled - currently not indexed" | various | Failed 9 May (no signal change since prior crawl) |
| 2026-05-11 | cohort.bubblnet.com | Sitemap re-submitted (after delete) | `/sitemap.xml` | **Success** — 21 pages discovered |
| 2026-05-13 | cohort.bubblnet.com | Validate Fix clicked on "Crawled - currently not indexed" | 10 URLs | **Failed 16 May 2026** (current state) |
| 2026-05-15 | cohort.bubblnet.com | Auto-recrawl | `/`, `/roadmap`, `/lessons/lesson-1-huggingface-beyond-upload` | Crawled - currently not indexed (the 3 failed URLs) |
| 2026-05-16 | cohort.bubblnet.com | Previous validation cycle | various | **Passed** (older batch resolved) |
| 2026-05-16 | thefirehacker.github.io/firstbreakai | Change of Address tool attempted | n/a | **Blocked** — Step 2 greyed out, URL-prefix property cannot trigger Change of Address |
| 2026-05-18 | — | This document created | — | — |

**Pattern noted:** Repeated "Validate Fix" clicks without underlying changes are tracked by GSC and reduce future validation priority. **Stop clicking it.** See [Section 10](#section-10--ai-agent-instructions) DO NOT list.

---

## Section 8 — Weekly Metrics Timeline

Update this table every Monday (or whenever new GSC data is reviewed). **Append rows, never overwrite.** Pull values from GSC → Pages → Indexing report.

| Week starting | Indexed | Not indexed | Pending | Failed | Sitemap URLs | GA users (28d) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-18 | 0 | 11 | 7 | 3 | 26 | 558 | Baseline. Validation failed 16 May. No high-leverage actions taken yet. |
| 2026-05-25 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| 2026-06-01 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| 2026-06-08 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| 2026-06-15 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |

**Trigger thresholds:**

- **If indexed > 0** for the first time → log which URL got indexed and what signal preceded it.
- **If indexed drops** → escalate to user immediately (see [Section 10](#section-10--ai-agent-instructions)).
- **If failed URLs change** → note which URLs joined/left the failed list.
- **If sitemap URL count changes** → confirm it's expected (new content) and not a build regression.

---

## Section 9 — Pending High-Leverage Actions

Prioritized backlog. Each item has leverage, effort, blocker, and acceptance criteria. **When an action ships, move it to a "Completed actions" subsection at the bottom of this section with the date.**

### 9.1 Add `parentOrganization` + `sameAs` JSON-LD on homepage

- **Leverage:** HIGH
- **Effort:** 30 minutes
- **Why:** Tells Google explicitly that cohort.bubblnet.com is owned by AIEDX (which owns bubblspace.com). Connects the cohort to the existing org graph, closing the #1 gap vs fetchlens.ai.
- **File:** [includes/schema.html](../includes/schema.html)
- **Implementation sketch:**
  ```json
  {
    "@context": "https://schema.org",
    "@type": "Organization",
    "name": "First Break AI",
    "url": "https://cohort.bubblnet.com",
    "parentOrganization": {
      "@type": "Organization",
      "name": "AIEDX Private Limited",
      "url": "https://bubblspace.com"
    },
    "sameAs": [
      "https://github.com/thefirehacker/firstbreakai",
      "https://www.bubblspace.com/firstbreakai",
      "https://thefirehacker.github.io/"
    ]
  }
  ```
- **Acceptance:** `curl -s https://cohort.bubblnet.com/ | grep parentOrganization` returns a match after next deploy.
- **Status:** Pending

### 9.2 Add business `Organization` JSON-LD with address/phone/email

- **Leverage:** HIGH
- **Effort:** 15 minutes
- **Why:** Visible "real business" signals (address, phone, email) help Google's spam/trust classifiers. Fetchlens has these prominently in their footer.
- **File:** [includes/schema.html](../includes/schema.html) (extend the Organization schema from #9.1)
- **Add:** `address`, `telephone`, `email`, `contactPoint` fields using AIEDX Private Limited's Mumbai office details (already present on fetchlens.ai footer for reference).
- **Acceptance:** Schema validates in Google Rich Results Test.
- **Status:** Pending

### 9.3 Put a real page on `bubblnet.com/`

- **Leverage:** HIGH
- **Effort:** 1–2 hours
- **Why:** Parent `bubblnet.com` is a 4-year-old domain that has never had content and only started redirecting in April 2026. This is a "dormant-to-active" pattern Google scrutinizes. A real landing page with links to cohort + firstbreakai gives the subdomain a parent to inherit from.
- **Where to host:** Could be a single static HTML file deployed to Cloudflare Pages on a separate project, or replace the Redirect Rule with a real Pages deployment that has a homepage and redirects everything else.
- **Content sketch:** "Bubblnet — projects by AIEDX". Short intro, links to:
  - First Break AI cohort → `https://cohort.bubblnet.com/`
  - FirstBreakAI landing → `https://www.bubblspace.com/firstbreakai`
  - About AIEDX → `https://bubblspace.com`
- **Caveat:** This may require updating the Cloudflare Redirect Rule from "redirect all" to "redirect inner paths only, leave root serving the landing page".
- **Acceptance:** `curl https://bubblnet.com/` returns HTTP 200 with real content and a link to the cohort.
- **Status:** Pending

### 9.4 Publish stub NPM package linking to cohort

- **Leverage:** MEDIUM-HIGH
- **Effort:** 1 hour
- **Why:** NPM is a Tier-1 trust domain. One dofollow link from `npmjs.com` is worth ~50 dofollow links from random blogs. Fetchlens has 3 such links.
- **What:** A tiny NPM package — e.g. `@firstbreakai/setup-cohort` — a CLI that scaffolds the Quarto blog from Lesson 0. Even a stub `README` with a link to `https://cohort.bubblnet.com/` is enough; functionality is secondary for SEO purposes (but make it useful too).
- **Alternatives:** PyPI package with the same idea would also work.
- **Acceptance:** Package live at `npmjs.com/package/@firstbreakai/...` with link to cohort homepage in description.
- **Status:** Pending

### 9.5 Show HN / ProductHunt launch post

- **Leverage:** HIGH (one-shot)
- **Effort:** 30 minutes to write, 1 hour to monitor
- **Why:** A front-page HN mention is worth more than months of organic outreach. Even a mid-tier submission gets multiple dofollow links from blog rollups.
- **Hook:** "Show HN: I built a free open AI cohort with AI-narrated podcast for each scene." Lead with the Three.js journey storyboard or the Lesson 1 / HuggingFace deep-dive.
- **When:** Once items 9.1–9.3 are live (so first-time visitors see the polished version).
- **Acceptance:** Submitted to HN or PH, link captured in [Section 7](#section-7--gsc-activity-log).
- **Status:** Pending

### 9.6 Pitch one AI newsletter

- **Leverage:** HIGH
- **Effort:** 1 hour to draft
- **Why:** One mention in a recognized AI newsletter (The Batch, Latent Space, Import AI, AlphaSignal, TLDR AI, Last Week in AI) is a high-authority brand signal that compounds for months.
- **Pitch angle:** Lesson 1 (HuggingFace Beyond Upload) — 17K-word novel technical angle ("an open model is not a file; it's a supply chain"). That's pitch-worthy on its own.
- **Acceptance:** Pitch sent, response tracked.
- **Status:** Pending

### 9.7 Add cohort link to thefirehacker.github.io homepage

- **Leverage:** LOW (but trivial effort)
- **Effort:** 5 minutes
- **Why:** Another dofollow link from a related personal domain. Weak signal but additive.
- **Acceptance:** `thefirehacker.github.io` homepage prominently links to `cohort.bubblnet.com`.
- **Status:** Pending

### Completed actions

> (Empty as of 2026-05-18. Move shipped items here with date and brief outcome.)

---

## Section 10 — AI Agent Instructions

**Read this section before doing anything.** This is the operating manual for any agent (human or AI) maintaining SEO for this site.

### DO

- **Update [Section 2 — Snapshot](#section-2--snapshot)** every time you have fresh GSC data. Overwrite values in place; the Snapshot is meant to reflect *current* state.
- **Append a row to [Section 8 — Weekly Metrics Timeline](#section-8--weekly-metrics-timeline)** every Monday with the week's GSC numbers.
- **Append entries to [Section 7 — GSC Activity Log](#section-7--gsc-activity-log)** whenever an action is taken in Search Console (sitemap submit, validate fix, request indexing, removals, etc.). Date + property + action + target + outcome.
- **Move shipped items** from [Section 9](#section-9--pending-high-leverage-actions) to its "Completed actions" subsection with date and outcome.
- **Add new pending actions** to [Section 9](#section-9--pending-high-leverage-actions) using the same template (leverage, effort, why, acceptance, status).
- **Reference repo files** with relative markdown links like `[file](../path/to/file)`. Verify the link resolves before committing.
- **Cross-link** to [SEO-Explained.md](../SEO-Explained.md) and [DEVELOPMENT.md](../DEVELOPMENT.md) rather than duplicating their content here.
- **Update the "Last updated" date** at the top of this file on any meaningful edit.

### DO NOT

- **Do not click "Validate Fix"** in GSC on indexing issues unless something material has changed since the previous click (e.g. you just shipped action 9.1 or 9.2). Repeated empty clicks reduce future validation priority. The GSC tooltip says: "Issues only re-checked when explicitly clicked Validate Fix." Each unjustified click counts against you.
- **Do not re-submit the cohort sitemap** unless URL count materially changes. The current submission (11 May) is healthy. Re-submitting an unchanged sitemap accomplishes nothing.
- **Do not re-try Change of Address** in GSC for the old domain. It is permanently blocked by GitHub Pages being a URL-prefix property. We do not own `github.io`. See [Section 5.4](#54-change-of-address-tool-in-gsc).
- **Do not re-submit a sitemap on the old GitHub Pages GSC property.** GSC requires same-host sitemaps; the old path no longer serves one; cross-domain sitemap submissions are rejected. See [Section 5.5](#55-sitemap-on-old-github-pages-property).
- **Do not add 4xx-returning URLs to the sitemap.** Verify with `curl -I` first.
- **Do not modify the `QUARTO_PROJECT_RENDER_ALL` guard** in [scripts/rewrite-html-links.mjs](../scripts/rewrite-html-links.mjs). It exists so local previews don't 404; removing it breaks local dev.
- **Do not change CI to use a single-file `quarto render <file>`.** It must remain bare `quarto render` so the env var fires and the link rewriter runs. See [.github/workflows/publish.yml](../.github/workflows/publish.yml).
- **Do not promise indexing timelines** to the user. Realistic range for a new subdomain in this niche is 8–16 weeks of active authority-building. Faster only with a major mention (HN front page, recognized newsletter).
- **Do not interpret "Crawled - currently not indexed" as a technical bug.** It is a Google trust-judgment. Re-audit the technical state if helpful (Section 3), but the fix is in Section 9, not in code.

### ESCALATE TO USER WHEN

- **Indexed page count drops** from a positive number back toward zero.
- **A new validation cycle fails** unexpectedly (without a clear cause logged in [Section 7](#section-7--gsc-activity-log)).
- **"Soft 404" or "Page with redirect" errors** appear in GSC Indexing → Pages report.
- **Someone proposes** Change of Address, sitemap resubmission on old property, or adding NoIndex anywhere — flag the proposal and reference the relevant DO NOT entry above before acting.
- **Cloudflare deployment fails** or the link rewriter doesn't run (look for `[links] total: N href(s) rewritten` in the CI build log; absence means the env var didn't fire).
- **The sitemap URL count** changes by more than 5 in a single deploy without a corresponding content addition.

### Standard update workflow (weekly)

1. Open GSC → Pages report. Note `indexed` and `not indexed` counts.
2. Open GSC → Sitemaps. Confirm latest sitemap status and discovered count.
3. Open GA → last 28 days. Note active users and views.
4. Append a row to [Section 8](#section-8--weekly-metrics-timeline) with these numbers.
5. Overwrite [Section 2 — Snapshot](#section-2--snapshot) with the same numbers + any state changes (failed/pending URL lists, new validation cycle status).
6. If actions were taken this week, append rows to [Section 7](#section-7--gsc-activity-log) and move completed items in [Section 9](#section-9--pending-high-leverage-actions).
7. Update the "Last updated" date at the top of this file.
8. Commit with message: `docs(seo): weekly update YYYY-MM-DD`.

---

## Section 11 — File Reference Map

Every file in this repo that touches SEO. Touch with care.

| File | Role | When to touch |
| --- | --- | --- |
| [canonical.lua](../canonical.lua) | Quarto filter injecting `<link rel="canonical">` on every page | Only if canonical URL strategy changes |
| [scripts/rewrite-sitemap.mjs](../scripts/rewrite-sitemap.mjs) | Strips `.html`/`index.html` from sitemap `<loc>` entries | Only if Quarto's sitemap format changes |
| [scripts/rewrite-html-links.mjs](../scripts/rewrite-html-links.mjs) | Strips `.html` from internal `href`s in CI | **Don't remove the env var guard** |
| [_quarto.yml](../_quarto.yml) | Wires post-render scripts, configures OG/Twitter/canonical | Add new post-render hooks here; don't remove existing ones |
| [.github/workflows/publish.yml](../.github/workflows/publish.yml) | CI build + Cloudflare deploy | Keep `run: quarto render` bare (no args) |
| [robots.txt](../robots.txt) | Crawler directives + sitemap pointer | Only if new path needs Disallow |
| [includes/schema.html](../includes/schema.html) | JSON-LD structured data on homepage | Add Organization, parentOrganization, sameAs here (action 9.1) |
| [SEO-Explained.md](../SEO-Explained.md) | Plain-English clean-URL walkthrough | Reference, don't duplicate |
| [DEVELOPMENT.md](../DEVELOPMENT.md) | Deploy + infra docs | Cross-link from here |
| [.gitignore](../.gitignore) | `docs/` is here — local builds never ship | Don't unignore `docs/` |

---

## Section 12 — Glossary

Terms a new agent needs to understand. Skim before asking.

- **Canonical URL** — The preferred URL for a page, declared via `<link rel="canonical">`. Tells Google "if you see this content at multiple URLs, this is the one to index."
- **Clean URL** — A URL without a `.html` extension or `/index.html` suffix (e.g. `/about` not `/about.html`).
- **CLS** — Cumulative Layout Shift, a Core Web Vital. Fixed by setting `width`/`height` on images.
- **Crawled - currently not indexed** — Google fetched the page, decided not to put it in the index. **Not a technical bug.** Usually a trust/authority judgment.
- **Dofollow / nofollow** — Whether a link passes SEO authority. `rel="nofollow"` (and `ugc`, `sponsored`) tell Google "don't pass authority through this link." Reddit and most social platforms add these automatically.
- **Domain property vs URL-prefix property** — In GSC. Domain properties cover all subdomains+protocols of a domain you own. URL-prefix properties cover one specific origin+path. **Change of Address requires a domain property.** GitHub Pages users cannot use Change of Address because `github.io` is not theirs.
- **JSON-LD** — JavaScript Object Notation for Linked Data. Structured data Google reads to understand entity relationships (Organization, Course, FAQ, etc.).
- **LCP** — Largest Contentful Paint, a Core Web Vital. Time until the biggest visible element loads.
- **Meta-refresh redirect** — `<meta http-equiv="refresh" content="0; url=...">`. Google treats it like a 301 if the page returns HTTP 200. If the page returns HTTP 404, the redirect signal is weakened.
- **`parentOrganization`** — schema.org property linking a child organization to its parent. Critical for trust-graph signal (e.g. cohort → AIEDX → bubblspace).
- **Post-render hook** — A Quarto config in `_quarto.yml` that runs scripts after the site is built. We use it for sitemap and link rewriting.
- **`QUARTO_PROJECT_RENDER_ALL`** — Environment variable Quarto sets to `1` only during full project renders (bare `quarto render`). We use it to gate the link rewriter so it only runs in CI, not local preview.
- **`sameAs`** — schema.org property listing other URLs that represent the same entity. Used to link an Organization to its NPM packages, GitHub profile, Twitter, etc.
- **Sitemap** — XML file at `/sitemap.xml` listing all canonical URLs. Tells Google what to crawl.
- **Soft 404** — A page that returns HTTP 200 but has thin/missing content, so Google treats it like a 404. Avoid.
- **Trust budget** — Informal term for Google's per-site indexing willingness. Driven by domain authority, source quality of inbound links, age, and signal consistency. The current blocker for cohort.bubblnet.com.
- **Validation cycle (GSC)** — Triggered by clicking "Validate Fix" on an indexing issue. Google re-checks the affected URLs and either passes or fails. **Repeated empty clicks reduce future validation priority.**

---

*End of document. Last updated: 2026-05-18. Next update due: 2026-05-25 (weekly cadence) or sooner if material change.*
