# SEO & Indexing Efforts Log — cohort.bubblnet.com

> Single source of truth for all SEO work on `cohort.bubblnet.com`. Owner: TheFireHacker. Last updated: **18 May 2026 (afternoon — major trust-signal shipment)**.

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

**As of 18 May 2026 (afternoon):**

| Metric | Value | Trend |
| --- | --- | --- |
| Pages indexed in Google | **0** | flat — but trust signals just changed materially |
| Pages "Crawled - currently not indexed" | 11 | unchanged from morning |
| Latest validation cycle | Failed 16 May 2026 | superseded by new shipment — see below |
| Sitemap status | Processed successfully | 26 URLs, last read 11 May |
| GA traffic (28 days) | 558 active users, 6.2K events, 2.2K views | +169% users vs prev period |
| Site uptime | 100% | Cloudflare Pages |
| TTFB (homepage) | 0.35s | excellent |
| **`bubblnet.com/` parent landing page** | **LIVE (HTTP 200)** | **NEW — shipped 18 May PM** |
| **`Organization` + `parentOrganization` JSON-LD on cohort** | **LIVE** | **NEW — shipped 18 May PM** |
| **`Organization` + `parentOrganization` JSON-LD on bubblnet.com** | **LIVE** | **NEW — shipped 18 May PM** |
| GSC: bubblnet.com domain property | Being added | NEW — 18 May PM |

**What changed today (afternoon, 18 May 2026):**

1. **Apex `bubblnet.com` is no longer a redirect shell.** Old Cloudflare Redirect Rule disabled. A real static landing page now served via a separate Cloudflare Worker, attached as a custom domain on that Worker. Page introduces Bubblnet as the AIEDX project network and links to First Break AI cohort, BubblSpace, and FetchLens.
2. **Trust-graph JSON-LD shipped on both ends.** Both `cohort.bubblnet.com` and `bubblnet.com` now carry `Organization` + `parentOrganization` schema connecting them to AIEDX / BubblSpace. `sameAs` links also added.
3. **Dofollow link from new parent → cohort verified** (`<a href="https://cohort.bubblnet.com/">Visit First Break AI Cohort</a>`).
4. **GSC**: `bubblnet.com` being added as a **Domain property** (covers root + all subdomains). Existing cohort URL-prefix property retained.

**Currently failed URLs (validation 16 May 2026 — pre-shipment):**

- `https://cohort.bubblnet.com/` — homepage
- `https://cohort.bubblnet.com/roadmap`
- `https://cohort.bubblnet.com/lessons/lesson-1-huggingface-beyond-upload`

**Currently pending URLs (validation, last crawled Apr 24 – May 3 — pre-shipment):**

- `https://cohort.bubblnet.com/office-hours/`
- `https://cohort.bubblnet.com/about`
- `https://cohort.bubblnet.com/lessons/lesson-0-welcome`
- `https://cohort.bubblnet.com/blog/`
- `https://cohort.bubblnet.com/lessons/`
- `https://cohort.bubblnet.com/office-hours/2026-04-24`
- `https://cohort.bubblnet.com/journey/`

**Diagnosis as of this snapshot:** The trust-budget gap identified in the morning has been substantially closed. The biggest 3 missing signals (parent JSON-LD on cohort, parent JSON-LD on bubblnet, real content on bubblnet.com) all shipped together. **Expected re-crawl response window: 1–3 weeks.** No further structural fixes are needed; the remaining lift is external authority (NPM, HN, newsletter — Section 9.4–9.6).

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

- **Status:** Done — **expanded 18 May 2026 PM** with trust-graph schema
- **File:** [includes/schema.html](../includes/schema.html) — injected via `_quarto.yml` `include-in-header`
- **Schemas present on homepage:** `WebSite`, `Course`, `PodcastSeries`, `FAQPage`, **`Organization` with `parentOrganization` pointing to AIEDX / BubblSpace**, **`sameAs` links to GitHub repo, bubblspace.com/firstbreakai, thefirehacker.github.io**.
- **Mirror schema on `bubblnet.com`:** `Organization` with `parentOrganization` (AIEDX), `sameAs` to cohort + bubblspace + fetchlens, `contactPoint` with general email. See [Section 5.1](#51-apex-bubblnetcom--real-parent-landing-page).
- **Still missing (lower-leverage):** Business `Organization` schema with full address/phone/email on the cohort homepage (Section 9.2 — partial mitigation via `bubblnet.com` `contactPoint`).

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

### 5.1 Apex `bubblnet.com` → real parent landing page

- **Status:** **REPLACED 18 May 2026 PM.** Previously a 301 redirect (Cloudflare Redirect Rule). Now serves a real static HTML landing page via a Cloudflare Worker (static assets).
- **Why changed:** A content-empty parent that only 301s gives the cohort subdomain nothing to inherit from. Google flags "dormant parent" patterns. A real parent page with unique content + outbound dofollow link to the cohort gives the trust graph somewhere to root.
- **Mechanism:**
  - Separate Cloudflare Worker (not the `firstbreakai` Pages project) with static HTML asset.
  - Worker has `bubblnet.com` attached as a custom domain.
  - Old Redirect Rule disabled (kept in account history for rollback).
- **Page content:** "Bubblnet — Projects by AIEDX". Three project cards: First Break AI (→ `cohort.bubblnet.com`), BubblSpace (→ `bubblspace.com`), FetchLens (→ `fetchlens.ai`). Footer with `mailto:contact@bubblspace.com`.
- **JSON-LD on the page:** `Organization` with `parentOrganization` (AIEDX Private Limited / bubblspace.com), `sameAs` to cohort + bubblspace + bubblspace/firstbreakai + fetchlens.ai, `contactPoint` with email.
- **Canonical:** `<link rel="canonical" href="https://bubblnet.com/">` (self-canonical).
- **Verified:**
  - `curl -I https://bubblnet.com/` → `HTTP/2 200`, `content-type: text/html`, `server: cloudflare`, `cf-cache-status: HIT`.
  - `curl -s https://bubblnet.com/ | grep cohort.bubblnet.com` → includes `<a href="https://cohort.bubblnet.com/">Visit First Break AI Cohort</a>`.
  - `curl -s https://bubblnet.com/ | grep parentOrganization` → match present.
- **`www.bubblnet.com`:** Not yet configured (shows "Not secure"). Low priority — see [Section 5.7](#57-known-not-yet-configured-www-hosts).

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

### 5.6 Cloudflare custom-domain topology (current state — updated 18 May 2026 PM)

- **Status:** Done (current config is correct)
- **`firstbreakai` Pages project:**
  - Custom domains attached: only `cohort.bubblnet.com` (Active, healthy)
  - **DO NOT attach `bubblnet.com` / `www.bubblnet.com` here.** They belong on the separate Worker. Attaching here would serve cohort content from two hostnames and split signals.
- **Separate Cloudflare Worker (Bubblnet parent landing):**
  - Custom domains attached: `bubblnet.com` (Active)
  - Serves the static HTML landing page described in [Section 5.1](#51-apex-bubblnetcom--real-parent-landing-page).
  - **Different content** from the Pages project — the only correct way to give the apex its own origin.
- **Why this split matters for SEO:** Two distinct origins (`cohort.bubblnet.com` = cohort site; `bubblnet.com` = parent index), each with its own canonical, JSON-LD, and outbound links. Google sees a real parent → child relationship instead of a redirect shell. This mirrors how `fetchlens.ai` is structured.

### 5.7 Known not-yet-configured `www` hosts

- **Status:** Pending (LOW priority — not blocking indexing)
- **Not configured today:**
  - `www.bubblnet.com` — shows "Not secure"
  - `www.cohort.bubblnet.com` — also not configured
- **Why low priority:** Canonical targets are the bare hosts (`https://bubblnet.com/`, `https://cohort.bubblnet.com/`). Without inbound traffic to the `www` variants, missing config is invisible.
- **Optional cleanup (when convenient):**
  - `www.bubblnet.com` → 301 → `https://bubblnet.com/`
  - `www.cohort.bubblnet.com` → 301 → `https://cohort.bubblnet.com/`
- **Acceptance:** `curl -I https://www.bubblnet.com/` returns HTTP 301 to apex with valid TLS.

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

### 6.2 Comparison: fetchlens.ai vs cohort.bubblnet.com (updated 18 May 2026 PM)

`fetchlens.ai` was created on **9 April 2026** and indexed within 2 days. Gaps that existed this morning have largely closed by this afternoon.

| Signal | fetchlens.ai | cohort.bubblnet.com (was → now) |
| --- | --- | --- |
| Apex vs subdomain | Apex (`.ai`) | Subdomain on parent — but parent now serves real content |
| `parentOrganization` JSON-LD | Yes → `https://bubblspace.com` | **Was: No → Now: Yes** (shipped 18 May PM) |
| Organization schema on site | Yes | **Was: No → Now: Yes** (shipped 18 May PM) |
| Parent domain has content | n/a (apex) | **Was: redirect shell → Now: real landing page** (shipped 18 May PM) |
| `parentOrganization` JSON-LD on parent | n/a | **Now: Yes** (on `bubblnet.com`) |
| NPM/PyPI backlinks | 3 packages | **Still: None** (Section 9.4 pending) |
| Visible business signals on cohort | Address + phone + email in footer | Email only (via `bubblnet.com` `contactPoint`) — Section 9.2 still partial |
| External authority mentions (HN, newsletter) | Yes | **None** (Section 9.5, 9.6 pending) |

**Conclusion as of this update:** The structural / on-page trust-graph gap is closed. What remains is **external authority** — NPM package + one HN/PH/newsletter mention. Those are the only Section 9 items still moving the needle.

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
| 2026-05-18 (AM) | — | This document created | — | — |
| 2026-05-18 (PM) | cohort.bubblnet.com (deployment) | `Organization` + `parentOrganization` + `sameAs` JSON-LD added | site-wide via `includes/schema.html` | Shipped (action 9.1 ✅) |
| 2026-05-18 (PM) | Cloudflare (account-level) | Disabled Redirect Rule "bubblnet.com → cohort" | apex + www | Done (kept in history for rollback) |
| 2026-05-18 (PM) | Cloudflare Worker (new) | Deployed Bubblnet parent landing page; attached `bubblnet.com` custom domain | `bubblnet.com` | HTTP 200 verified (action 9.3 ✅) |
| 2026-05-18 (PM) | bubblnet.com | Verified `<a href="https://cohort.bubblnet.com/">` link present | `curl` grep | Pass |
| 2026-05-18 (PM) | bubblnet.com | Verified `parentOrganization` JSON-LD present | `curl` grep | Pass |
| 2026-05-18 (PM) | GSC | Added `bubblnet.com` as **Domain property** (in progress) | bubblnet.com domain | Started; covers root + all subdomains |

**Pattern noted:** Repeated "Validate Fix" clicks without underlying changes are tracked by GSC and reduce future validation priority. **Today is the exception:** with the 18 May PM shipment of new JSON-LD + new parent page, *one* Request Indexing pass on `https://cohort.bubblnet.com/` and `https://bubblnet.com/` is now justified. After that, wait again. See [Section 10](#section-10--ai-agent-instructions).

---

## Section 8 — Weekly Metrics Timeline

Update this table every Monday (or whenever new GSC data is reviewed). **Append rows, never overwrite.** Pull values from GSC → Pages → Indexing report.

| Week starting | Indexed | Not indexed | Pending | Failed | Sitemap URLs | GA users (28d) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-18 | 0 | 11 | 7 | 3 | 26 | 558 | **Baseline + trust-signal shipment.** PM 18 May: shipped 9.1 (parent JSON-LD on cohort) and 9.3 (real `bubblnet.com/` landing page). Expected re-crawl response 1–3 weeks. |
| 2026-05-25 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | First post-shipment data point. Watch for indexed > 0 on cohort homepage or bubblnet.com root. |
| 2026-06-01 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| 2026-06-08 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| 2026-06-15 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | If still 0 indexed by this week, ship Section 9.4 (NPM stub) and 9.5 (Show HN) — external authority lift. |

**Trigger thresholds:**

- **If indexed > 0** for the first time → log which URL got indexed and what signal preceded it.
- **If indexed drops** → escalate to user immediately (see [Section 10](#section-10--ai-agent-instructions)).
- **If failed URLs change** → note which URLs joined/left the failed list.
- **If sitemap URL count changes** → confirm it's expected (new content) and not a build regression.

---

## Section 9 — Pending High-Leverage Actions

Prioritized backlog. Each item has leverage, effort, blocker, and acceptance criteria. **When an action ships, move it to a "Completed actions" subsection at the bottom of this section with the date.**

### 9.1 ~~Add `parentOrganization` + `sameAs` JSON-LD on homepage~~ ✅ SHIPPED 18 May 2026 PM

> **Moved to [Completed actions](#completed-actions)** — see entry C-1.

### 9.2 Add business `Organization` JSON-LD with address/phone/email

- **Leverage:** MEDIUM (partially mitigated 18 May PM by `contactPoint` email on `bubblnet.com`)
- **Effort:** 15 minutes
- **Why:** Visible "real business" signals (full address, phone, email) help Google's spam/trust classifiers. Fetchlens has these prominently in their footer. We currently have email only via the parent landing page.
- **File:** [includes/schema.html](../includes/schema.html) — extend the Organization schema shipped in 9.1.
- **Add:** `address` (Mumbai), `telephone`, `email` fields on the cohort `Organization` schema. Also consider a visible footer block on the cohort homepage with the same info.
- **Acceptance:** Schema validates in Google Rich Results Test with no warnings; `curl -s https://cohort.bubblnet.com/ | grep -E "address|telephone"` returns matches.
- **Status:** Pending

### 9.3 ~~Put a real page on `bubblnet.com/`~~ ✅ SHIPPED 18 May 2026 PM

> **Moved to [Completed actions](#completed-actions)** — see entry C-2.

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

#### C-1 — `parentOrganization` + `sameAs` JSON-LD on cohort (was 9.1)

- **Shipped:** 2026-05-18 PM
- **Where:** [includes/schema.html](../includes/schema.html), site-wide via Quarto include
- **What:** Added `Organization` with `parentOrganization` → AIEDX / bubblspace, plus `sameAs` to GitHub repo, bubblspace/firstbreakai, thefirehacker.github.io
- **Verified:** `curl -s https://cohort.bubblnet.com/ | grep parentOrganization` returns match
- **SEO purpose closed:** Trust-graph connection between cohort and parent org

#### C-2 — Real landing page on `bubblnet.com/` (was 9.3)

- **Shipped:** 2026-05-18 PM
- **Where:** Separate Cloudflare Worker (not the `firstbreakai` Pages project), `bubblnet.com` attached as custom domain
- **What:** Static HTML page titled "Bubblnet — Projects by AIEDX". Three project cards (First Break AI / BubblSpace / FetchLens), each linking to its destination. Footer with `mailto:contact@bubblspace.com`. JSON-LD with `Organization`, `parentOrganization`, `sameAs`, `contactPoint`.
- **Verified:**
  - `curl -I https://bubblnet.com/` → HTTP 200, served by Cloudflare
  - `curl -s https://bubblnet.com/ | grep cohort.bubblnet.com` → dofollow link to cohort present
  - `curl -s https://bubblnet.com/ | grep parentOrganization` → JSON-LD present
- **SEO purpose closed:** Parent no longer a redirect shell; provides real trust origin + dofollow link to cohort
- **Side effect:** Old Redirect Rule (`bubblnet.com` → cohort) disabled. Documented in [Section 5.1](#51-apex-bubblnetcom--real-parent-landing-page).

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
- **After a material on-page shipment** (new schema, new content, new redirect topology, parent-domain change): do **one** Request Indexing pass on the affected canonical URL(s). Log the request in Section 7. Then stop and wait.

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
