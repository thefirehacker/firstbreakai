# Plan: Old GitHub Pages migration → Cloudflare Pages indexing fix

## Context

The site moved from `https://thefirehacker.github.io/firstbreakai/` (project-style GitHub Pages, was indexed by Google with 6 URLs) to `https://cohort.bubblnet.com/` (Cloudflare Pages, currently 0 indexed, 18 known but mostly "Discovered – currently not indexed").

Two compounding problems are blocking re-indexing:

1. **No self-canonical on the new domain.** Quarto 1.8.24 is currently emitting `og:title`, `og:description`, `og:image`, etc., but **not** `<link rel="canonical">` and **not** `og:url`. Verified by:
   ```
   curl -s https://cohort.bubblnet.com/ | grep -i canonical    # empty
   curl -s https://cohort.bubblnet.com/ | grep -i og:url       # empty
   ```
   Without a self-canonical, Google has nothing to bind the new URL's identity to.

2. **Old indexed URLs are 404.** GitHub Pages was disabled when the migration to Cloudflare Pages happened. The previously-indexed URLs (`/`, `/checklist.html`, `/setup.html`, `/roadmap.html`, `/office-hours/`) all return `HTTP/2 404`. From Google's point of view the old trusted property disappeared and a separate new domain showed up — no 301/canonical bridge between them.

The previously-completed Pages-default-domain canonicalization (`firstbreakai.pages.dev` → `cohort.bubblnet.com` via `functions/_middleware.ts`) and `index.html` redirects in `_redirects` solve a separate duplicate-host concern. They do **not** address either problem above.

---

## Plan overview — two parts, ship in order

| Part | What | Who |
|---|---|---|
| **A** | Add a Lua filter so every Quarto page emits `<link rel="canonical">` + `og:url` | Me (code) |
| **B** | Restore the old GitHub Pages URLs as a redirect bridge to the new domain | You (manual, GitHub-side) + Me (code in `bridge/` folder) |

**Ship A first**, verify it's live, then do B. The bridge in B is pointless if the new pages don't self-canonicalize.

---

# PART A — Make new pages self-canonicalize

## A.1 — Code I will write (when you approve)

### New file: `canonical.lua` (repo root)

A Quarto Lua filter that injects canonical + og:url into every page's head, computing the URL from `_quarto.yml`'s `site-url` plus the actual output file path.

```lua
function Meta(meta)
  local site_url = meta['site-url'] and pandoc.utils.stringify(meta['site-url']) or nil
  if not site_url then return nil end
  site_url = site_url:gsub('/+$', '')

  local out = quarto.doc.output_file or ''
  local rel_out = out:gsub('^.*/docs/', ''):gsub('^docs/', '')
  local url_path = '/' .. rel_out
  url_path = url_path:gsub('/index%.html$', '/')

  local full = site_url .. url_path
  quarto.doc.include_text('in-header',
    '<link rel="canonical" href="' .. full .. '">\n' ..
    '<meta property="og:url" content="' .. full .. '">')
  return nil
end
```

### Edit: `_quarto.yml`

Add the filter under `format.html`:

```yaml
format:
  html:
    css: styles/global.css
    include-in-header: includes/schema.html
    filters:
      - canonical.lua          # ← new line
    include-after-body:
      - includes/mermaid-zoom.html
      - includes/github-card.html
      - includes/journey-player.html
      - includes/lesson-player.html
    mermaid:
      theme: neutral
```

## A.2 — Manual steps you'll do for Part A

> Nothing manual on GitHub/Cloudflare side. Just run a render + push:

1. After I make the two changes above, run:
   ```
   quarto render
   ```
2. Sanity-check locally:
   ```
   grep -i canonical docs/index.html docs/roadmap.html docs/checklist.html
   ```
   Each should show: `<link rel="canonical" href="https://cohort.bubblnet.com/...">`
3. Commit and push to your Cloudflare Pages production branch.
4. Wait for Cloudflare deploy to complete, then:
   ```
   curl -s https://cohort.bubblnet.com/ | grep -iE 'canonical|og:url'
   curl -s https://cohort.bubblnet.com/roadmap.html | grep -iE 'canonical|og:url'
   ```
   Both should now show canonical + og:url. **Only when these are green do we move to Part B.**

---

# PART B — Restore old GitHub Pages as a redirect bridge

## B.1 — Code I will write (when you approve)

I will create a `bridge/` folder at the repo root containing the redirect HTML files. **This folder will not be deployed to Cloudflare** — it's only the source for the GitHub Pages bridge. The existing Quarto render globs (`*.qmd`, `blog/*.qmd`, etc.) won't pick up `.html` files in `bridge/`, and `bridge/` is not listed in `_quarto.yml` resources, so it won't end up in `docs/`.

Files I'll create in `bridge/`:

| File | Redirects to |
|---|---|
| `bridge/index.html` | `https://cohort.bubblnet.com/` |
| `bridge/checklist.html` | `https://cohort.bubblnet.com/checklist.html` |
| `bridge/setup.html` | `https://cohort.bubblnet.com/setup.html` |
| `bridge/roadmap.html` | `https://cohort.bubblnet.com/roadmap.html` |
| `bridge/office-hours/index.html` | `https://cohort.bubblnet.com/office-hours/` |
| `bridge/404.html` | catch-all: strips `/firstbreakai/` prefix in JS, redirects to same path on `cohort.bubblnet.com` |

Each named-page file uses three redirect signals (canonical + meta-refresh + JS) and `<meta name="robots" content="noindex,follow">` so the bridge page itself doesn't compete with the new URL in the index, but Google still follows the canonical signal.

Template (PATH gets substituted per file):

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Moved — First Break AI</title>
  <link rel="canonical" href="https://cohort.bubblnet.com/PATH">
  <meta http-equiv="refresh" content="0; url=https://cohort.bubblnet.com/PATH">
  <meta name="robots" content="noindex,follow">
  <script>location.replace("https://cohort.bubblnet.com/PATH" + location.search + location.hash);</script>
</head>
<body>
  <p>This page has moved to <a href="https://cohort.bubblnet.com/PATH">cohort.bubblnet.com</a>.</p>
</body>
</html>
```

`bridge/404.html` body:
```html
<script>
  const old = "/firstbreakai";
  let p = location.pathname;
  if (p.startsWith(old)) p = p.slice(old.length) || "/";
  location.replace("https://cohort.bubblnet.com" + p + location.search + location.hash);
</script>
```

I will **not** create a `bridge/sitemap.xml` — the new domain's sitemap is the only one Google should consume.

## B.2 — Manual steps you'll do for Part B

These are the GitHub-side actions. I cannot do them for you because they require web auth on github.com and `git checkout --orphan` is something you should run yourself in your own checkout.

### Manual Step B.2.1 — Create the `gh-pages` branch and publish the bridge

After I've created the `bridge/` folder on `main`, run these commands in your local checkout:

```bash
# 0. Make sure you're on main with the bridge/ folder present
git checkout main
git pull
ls bridge/   # confirm the files are there

# 1. Create an orphan branch (no history from main)
git checkout --orphan gh-pages

# 2. Wipe the working tree clean (this only stages deletions; the bridge/ folder
#    is still on disk because it was just copied from main's index)
git rm -rf --cached .
# Remove everything except the bridge folder we want to promote
find . -mindepth 1 -maxdepth 1 ! -name bridge ! -name .git -exec rm -rf {} +

# 3. Promote bridge/* to the branch root
mv bridge/* .
mv bridge/.* . 2>/dev/null || true  # ignore if no dotfiles
rmdir bridge

# 4. Commit and push
git add .
git commit -m "Add migration redirects to cohort.bubblnet.com"
git push -u origin gh-pages

# 5. Switch back to your working branch
git checkout 1.5.1_CohortStart_Lesson01_Onwards
```

### Manual Step B.2.2 — Enable GitHub Pages on the gh-pages branch

1. Open https://github.com/thefirehacker/firstbreakai/settings/pages
2. Under **Build and deployment**:
   - **Source:** *Deploy from a branch*
   - **Branch:** `gh-pages` / `(root)`
3. Click **Save**.
4. Wait ~1 minute. The page will refresh with a green check + "Your site is live at https://thefirehacker.github.io/firstbreakai/".

### Manual Step B.2.3 — Verify the bridge

```bash
# Old GitHub Pages URLs should now return 200 (not 404), with redirect signals in the body
curl -I  https://thefirehacker.github.io/firstbreakai/
curl -s  https://thefirehacker.github.io/firstbreakai/roadmap.html | grep -iE 'canonical|refresh'
curl -s  https://thefirehacker.github.io/firstbreakai/checklist.html | grep -iE 'canonical|refresh'

# Catch-all: any unmapped path should JS-redirect via 404.html
curl -s  https://thefirehacker.github.io/firstbreakai/some-random-page.html | grep -iE 'cohort.bubblnet'

# New domain should still work and now have canonical (from Part A)
curl -I https://cohort.bubblnet.com/
curl -s https://cohort.bubblnet.com/ | grep -i canonical
```

Expected: old URLs return `200 OK` with `<link rel="canonical" href="https://cohort.bubblnet.com/...">` in the body. New URLs return `200` with self-canonical.

### Manual Step B.2.4 — Search Console actions

In **https://search.google.com/search-console**:

**On the OLD property** (`https://thefirehacker.github.io/firstbreakai/`):
1. Sitemaps → if any sitemap is submitted, **delete it** (the bridge has no sitemap by design).
2. URL Inspection → request indexing for each of:
   - `https://thefirehacker.github.io/firstbreakai/`
   - `https://thefirehacker.github.io/firstbreakai/roadmap.html`
   - `https://thefirehacker.github.io/firstbreakai/checklist.html`
   - `https://thefirehacker.github.io/firstbreakai/setup.html`
   - `https://thefirehacker.github.io/firstbreakai/office-hours/`

**On the NEW property** (`https://cohort.bubblnet.com/`):
1. Sitemaps → confirm `sitemap.xml` is submitted (already is).
2. URL Inspection → request indexing for:
   - `https://cohort.bubblnet.com/`
   - `https://cohort.bubblnet.com/roadmap.html`
   - `https://cohort.bubblnet.com/checklist.html`
   - `https://cohort.bubblnet.com/setup.html`
   - `https://cohort.bubblnet.com/lessons/lesson-0-welcome.html`

Don't request all 18 URLs at once — start with these 5 priority pages. Submit one batch, then wait a week before submitting more.

---

## Summary of who does what

### I will (code changes, after you approve):
- Create `canonical.lua` at repo root
- Edit `_quarto.yml` to add the `filters: [canonical.lua]` line
- Create `bridge/index.html`, `bridge/checklist.html`, `bridge/setup.html`, `bridge/roadmap.html`, `bridge/office-hours/index.html`, `bridge/404.html`

### You will (manual):
- **Part A — local + Cloudflare side:**
  - Run `quarto render` and push to your Cloudflare Pages branch
  - Verify canonical tags appear on the live new domain
- **Part B — GitHub side:**
  - Run the git commands in B.2.1 to create the `gh-pages` orphan branch and push the bridge files
  - Enable GitHub Pages on the `gh-pages` branch in repo Settings (B.2.2)
  - Verify old URLs now return 200 with canonical pointing to the new domain (B.2.3)
  - Submit URL inspection / re-indexing requests in Search Console (B.2.4)

---

## Out of scope

- **Backlinks from `bubblspace.com`, `aiedx.com`, LinkedIn, YouTube, X** — these help crawl priority but aren't code changes; do them when you can.
- **GitHub Action to auto-rebuild the bridge** — content is static (5 fixed paths + JS catch-all). If you ever add new top-level URLs that were previously indexed, just edit the bridge once.
- **Modifying `.github/workflows/publish.yml`** — the Cloudflare publish workflow stays as-is. The bridge is intentionally isolated to a separate `gh-pages` branch.
