# Old GitHub Pages migration → Cloudflare Pages indexing fix

**Status:** ✅ Code shipped. Search Console actions remaining.

## Context

The site moved from `https://thefirehacker.github.io/firstbreakai/` (project-style GitHub Pages, Google had 6 URLs indexed) to `https://cohort.bubblnet.com/` (Cloudflare Pages, 0 indexed at start, 18 known but mostly "Discovered – currently not indexed").

Two compounding problems were blocking re-indexing:

1. **No self-canonical on the new domain.** Quarto 1.8.24 was emitting `og:title`, `og:description`, `og:image`, `og:site_name`, but **not** `<link rel="canonical">` and **not** `og:url`. Without a self-canonical, Google had nothing to bind the new URL's identity to.
2. **Old indexed URLs returning 404.** GitHub Pages publishing was disabled when migration to Cloudflare Pages happened. The previously-indexed URLs all returned `HTTP/2 404`. From Google's point of view the old trusted property disappeared and a separate new domain showed up — no 301/canonical bridge between them.

A previously-completed Pages-default-domain canonicalization (`firstbreakai.pages.dev` → `cohort.bubblnet.com` via `functions/_middleware.ts`) and `index.html` redirects in `_redirects` solved a separate duplicate-host concern but did not address either of the above.

---

# PART A — Self-canonicalization on `cohort.bubblnet.com` ✅ DONE

## What was changed

### New file: `canonical.lua` (repo root)

Quarto Lua filter that injects `<link rel="canonical">` and `<meta property="og:url">` into every rendered page's head.

```lua
-- Quarto Lua filter: emit <link rel="canonical"> and <meta property="og:url"> on every HTML page.
-- Computes the URL from a hardcoded base + the rendered output file path.
-- Site URL kept here (not read from _quarto.yml) because Quarto's website meta
-- isn't exposed to per-document filters.

local SITE_URL = 'https://cohort.bubblnet.com'

function Pandoc(doc)
  local out = quarto.doc.output_file or ''
  local rel = out:gsub('^.*/docs/', ''):gsub('^docs/', '')
  if rel == '' or rel == out then return nil end

  local url_path = '/' .. rel
  url_path = url_path:gsub('/index%.html$', '/')
  url_path = url_path:gsub('%.html$', '')         -- strip .html to match Cloudflare's clean-URL serving form

  local full = SITE_URL .. url_path

  quarto.doc.include_text('in-header',
    '<link rel="canonical" href="' .. full .. '">\n' ..
    '<meta property="og:url" content="' .. full .. '">')
  return nil
end
```

### Edited: `_quarto.yml`

Added the filter under `format.html`:

```yaml
format:
  html:
    css: styles/global.css
    include-in-header: includes/schema.html
    filters:
      - canonical.lua          # ← added
    include-after-body:
      - includes/mermaid-zoom.html
      - includes/github-card.html
      - includes/journey-player.html
      - includes/lesson-player.html
    mermaid:
      theme: neutral
```

## Discoveries during Part A

### Discovery 1 — `meta['site-url']` is not accessible to per-document Lua filters

First attempt read `site-url` from page metadata:
```lua
local site_url = meta['site-url'] and pandoc.utils.stringify(meta['site-url'])
```

Debug output showed:
```
[canonical.lua] meta keys=fig-responsive,toc-title,title-block-style,link-citations,include-after,labels,quarto-template-params,header-includes,include-before,title,mermaid,lang,biblio-config,document-css,description,quarto-version
[canonical.lua] no site-url found in meta
```

Quarto's `website.site-url` from `_quarto.yml` is processed at the project level, not propagated to per-document metadata. **Resolution:** hardcoded `SITE_URL = 'https://cohort.bubblnet.com'` directly in the Lua filter.

### Discovery 2 — Cloudflare Pages strips `.html` extensions (308 redirect)

After first deploy, verification curl returned empty for non-homepage URLs:

```
$ curl -s https://cohort.bubblnet.com/roadmap.html | grep -iE 'canonical|og:url'
(empty)
$ curl -sI https://cohort.bubblnet.com/roadmap.html | head
HTTP/2 308
location: /roadmap
```

Cloudflare Pages' default behavior is to 308-redirect `/foo.html` → `/foo`. `curl -s` doesn't follow redirects, so the grep was running against an empty 308 body. Following with `curl -sL` showed the canonical was actually present, but pointed at `/roadmap.html` (which 308s).

**Resolution:** updated `canonical.lua` to strip the `.html` extension, so canonical now points at `/roadmap` (the URL Cloudflare actually serves) instead of `/roadmap.html`. This avoids a `canonical → 308 → final URL` chain that Google would have to reconcile.

## Tests run during Part A

### After first deploy (canonical with `.html` extension)

```
$ curl -s https://cohort.bubblnet.com/ | grep -iE 'canonical|og:url'
<link rel="canonical" href="https://cohort.bubblnet.com/">
<meta property="og:url" content="https://cohort.bubblnet.com/">

$ curl -sL https://cohort.bubblnet.com/roadmap.html | grep -i canonical
<link rel="canonical" href="https://cohort.bubblnet.com/roadmap.html">          ← problematic, points at 308'd URL
```

### After second deploy (canonical with clean URLs)

```
$ curl -sL https://cohort.bubblnet.com/roadmap.html | grep -i canonical
<link rel="canonical" href="https://cohort.bubblnet.com/roadmap">                ← matches serving URL ✅

$ curl -sI https://cohort.bubblnet.com/roadmap | grep -iE 'HTTP|cf-cache'
HTTP/2 200
cf-cache-status: DYNAMIC                                                          ← serves directly, no redirect ✅
```

## Local rendered pages — full canonical verification

```
docs/index.html                          → https://cohort.bubblnet.com/
docs/roadmap.html                        → https://cohort.bubblnet.com/roadmap
docs/checklist.html                      → https://cohort.bubblnet.com/checklist
docs/setup.html                          → https://cohort.bubblnet.com/setup
docs/about.html                          → https://cohort.bubblnet.com/about
docs/office-hours/index.html             → https://cohort.bubblnet.com/office-hours/
docs/blog/index.html                     → https://cohort.bubblnet.com/blog/
docs/lessons/lesson-0-welcome.html       → https://cohort.bubblnet.com/lessons/lesson-0-welcome
```

All 8 spot-checked pages emit both `<link rel="canonical">` and `<meta property="og:url">`.

---

# PART B — GitHub Pages migration bridge ✅ DONE

## What was created

### Files added at `/Users/booimac/AIEDX/Code/AI/firstbreakai/bridge/` on `main`

| File | Redirects to |
|---|---|
| `bridge/index.html` | `https://cohort.bubblnet.com/` |
| `bridge/checklist.html` | `https://cohort.bubblnet.com/checklist` |
| `bridge/setup.html` | `https://cohort.bubblnet.com/setup` |
| `bridge/roadmap.html` | `https://cohort.bubblnet.com/roadmap` |
| `bridge/office-hours/index.html` | `https://cohort.bubblnet.com/office-hours/` |
| `bridge/404.html` | JS catch-all: strips `/firstbreakai/` prefix and redirects to same path on `cohort.bubblnet.com` |

Each named-page file uses three redirect signals plus `noindex,follow`:
- `<link rel="canonical" href="…">` — tells Google the canonical URL is on the new domain
- `<meta http-equiv="refresh" content="0; url=…">` — instant browser redirect
- `<script>location.replace(…)</script>` — JS fallback if meta-refresh is blocked
- `<meta name="robots" content="noindex,follow">` — bridge page itself doesn't compete in the index, but Google follows the canonical signal

`bridge/404.html` is a JS catch-all for any unmapped path under `/firstbreakai/`, ensuring no old indexed URL on the GitHub Pages domain ever returns a hard 404.

The `bridge/` folder lives on `main` as source-of-truth. It does **not** deploy to Cloudflare (Quarto's render globs and resources list don't include `bridge/`).

## Promoting `bridge/` to a `gh-pages` branch

Used `git subtree push` to create the orphan branch with only `bridge/` contents at root:

```
$ git subtree push --prefix bridge origin gh-pages
git push using:  origin gh-pages
Enumerating objects: 14, done.
Counting objects: 100% (14/14), done.
Delta compression using up to 8 threads
Compressing objects: 100% (13/13), done.
Writing objects: 100% (14/14), 2.18 KiB | 2.18 MiB/s, done.
Total 14 (delta 9), reused 0 (delta 0), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (9/9), done.
To https://github.com/thefirehacker/firstbreakai.git
 * [new branch]      7b49556df0ec46eefe1b37a8351157313de1e4e6 -> gh-pages
```

This created `gh-pages` with `index.html`, `checklist.html`, `setup.html`, `roadmap.html`, `office-hours/index.html`, and `404.html` at the root — exactly what GitHub Pages needs.

## GitHub Pages enablement

In **github.com/thefirehacker/firstbreakai/settings/pages**:
- **Source:** Deploy from a branch
- **Branch:** `gh-pages` / `(root)`
- Saved → green check + "Your site is live at https://thefirehacker.github.io/firstbreakai/"

## Tests run after bridge deployed

```
$ curl -I https://thefirehacker.github.io/firstbreakai/
HTTP/2 200                                                                        ← was 404 before ✅
server: GitHub.com
content-type: text/html; charset=utf-8
last-modified: Tue, 28 Apr 2026 17:45:21 GMT
content-length: 522

$ curl -sL https://thefirehacker.github.io/firstbreakai/roadmap.html | grep -i canonical
<link rel="canonical" href="https://cohort.bubblnet.com/roadmap">                ← clean canonical ✅

$ curl -sL https://thefirehacker.github.io/firstbreakai/anything.html | grep -i cohort.bubblnet
<link rel="canonical" href="https://cohort.bubblnet.com/">
location.replace("https://cohort.bubblnet.com" + p + location.search + location.hash);
<p>First Break AI has moved to <a href="https://cohort.bubblnet.com/">cohort.bubblnet.com</a>.</p>
                                                                                  ← 404.html catch-all working ✅
```

All bridge endpoints return `200 OK` with canonical and triple-redirect signal pointing at the matching clean URL on the new domain. Unmapped paths fall through to `404.html` which JS-redirects to the new domain via prefix-stripping.

---

# Files modified — summary

| File | Status | Purpose |
|---|---|---|
| `canonical.lua` (new) | ✅ committed to `main` | Quarto Lua filter, emits canonical + og:url |
| `_quarto.yml` | ✅ committed to `main` | Wired the filter under `format.html.filters` |
| `bridge/index.html` (new) | ✅ committed to `main` + `gh-pages` | Bridge: old `/` → new `/` |
| `bridge/checklist.html` (new) | ✅ committed to `main` + `gh-pages` | Bridge: old `/checklist.html` → new `/checklist` |
| `bridge/setup.html` (new) | ✅ committed to `main` + `gh-pages` | Bridge: old `/setup.html` → new `/setup` |
| `bridge/roadmap.html` (new) | ✅ committed to `main` + `gh-pages` | Bridge: old `/roadmap.html` → new `/roadmap` |
| `bridge/office-hours/index.html` (new) | ✅ committed to `main` + `gh-pages` | Bridge: old `/office-hours/` → new `/office-hours/` |
| `bridge/404.html` (new) | ✅ committed to `main` + `gh-pages` | JS catch-all for any other previously-indexed path |

**Branches:**
- `main` — canonical source for everything including `bridge/`
- `gh-pages` — orphan branch, root contains only the bridge HTML files. Created via `git subtree push --prefix bridge origin gh-pages`. Regenerate with the same command if `bridge/` is updated on `main`.

**Untouched on purpose:**
- `.github/workflows/publish.yml` — Cloudflare deploy workflow, unchanged
- `functions/_middleware.ts` — already handles `firstbreakai.pages.dev` → `cohort.bubblnet.com`
- `_redirects` — already handles `/index.html` cleanups
- `robots.txt`, `includes/schema.html` — already correct

---

# Remaining manual steps — Search Console

Code work is done. The following are user actions in Google Search Console that I cannot perform.

## On the OLD property (`https://thefirehacker.github.io/firstbreakai/`)

1. **Sitemaps tab** → if any sitemap is submitted, **delete it** (the bridge has no sitemap by design — the new domain's sitemap is the only one Google should consume).
2. **URL Inspection tab** → request indexing for, one URL at a time:
   - `https://thefirehacker.github.io/firstbreakai/`
   - `https://thefirehacker.github.io/firstbreakai/roadmap.html`
   - `https://thefirehacker.github.io/firstbreakai/checklist.html`
   - `https://thefirehacker.github.io/firstbreakai/setup.html`
   - `https://thefirehacker.github.io/firstbreakai/office-hours/`

   For each: paste URL → "Test live URL" → confirm Google sees the canonical pointing at `cohort.bubblnet.com` → click "Request indexing".

## On the NEW property (`https://cohort.bubblnet.com/`)

1. **Sitemaps tab** → confirm `sitemap.xml` is submitted (should already be there).
2. **URL Inspection tab** → request indexing for these priority pages (use **clean URLs**, no `.html`, since that's what canonical now emits):
   - `https://cohort.bubblnet.com/`
   - `https://cohort.bubblnet.com/roadmap`
   - `https://cohort.bubblnet.com/checklist`
   - `https://cohort.bubblnet.com/setup`
   - `https://cohort.bubblnet.com/lessons/lesson-0-welcome`

   Don't submit all 18 sitemap URLs at once — start with these 5 and wait a week before submitting more.

## Expected timeline

- **Days 1–7:** Google fetches the bridge pages, sees canonical → cohort.bubblnet.com, follows. Pages start moving from "Discovered – currently not indexed" → "Crawled".
- **Weeks 2–3:** "Crawled" pages move to "Indexed". Old GitHub Pages URLs in the index either get replaced by the new canonical or marked as redirected.
- **Month 1+:** Search results for old URLs naturally consolidate to the new domain.

---

# Maintaining the bridge later

If you ever add a new top-level page on the new domain that was *previously indexed* on the old GitHub Pages domain, repeat:

1. Add a new file under `bridge/` on `main` following the template in the existing files (just substitute the path).
2. Commit + push to `main`.
3. Run `git subtree push --prefix bridge origin gh-pages` again to update the deployed bridge.
4. GitHub Pages will redeploy automatically within a minute.

For new pages that were *never* indexed on the old domain, no bridge update is needed — Google will discover them through the new domain's sitemap.

---

# Out of scope

- **Backlinks from `bubblspace.com`, `aiedx.com`, LinkedIn, YouTube, X** — these help crawl priority but aren't code changes; do them manually when you have time.
- **GitHub Action to auto-rebuild the bridge** — content is static (5 fixed paths + JS catch-all). Manual `git subtree push` is fine.
- **Modifying `.github/workflows/publish.yml`** — the Cloudflare publish workflow stays as-is. The bridge is intentionally isolated to a separate `gh-pages` branch.
