# First Break AI.

Free, open cohort for your first break in AI. Live site: [cohort.bubblnet.com](https://cohort.bubblnet.com).

## Deploying the site

Built with [Quarto](https://quarto.org/) and deployed to **Cloudflare Pages**. The [`.github/workflows/publish.yml`](.github/workflows/publish.yml) workflow renders the site and deploys on push to `main`.

Local preview:

```bash
quarto preview --port 5942
```

Using a fixed port keeps the R2 CORS policy stable (see below).

### Cloudflare Pages and SEO (clean URLs)

Cloudflare Pages strips `.html` extensions and serves canonical "clean" URLs (e.g. `/about` instead of `/about.html`). Requests for the `.html` version 308-redirect to the clean URL. Two post-render scripts in [`_quarto.yml`](_quarto.yml) keep the generated site aligned with that behavior:

| Script | What it does | When it runs |
| --- | --- | --- |
| [`scripts/rewrite-sitemap.mjs`](scripts/rewrite-sitemap.mjs) | Strips `.html` and `index.html` from `<loc>` entries in `sitemap.xml` so URLs match canonical tags. | Every render. |
| [`scripts/rewrite-html-links.mjs`](scripts/rewrite-html-links.mjs) | Rewrites internal `href` attributes in every `docs/**/*.html` file to drop `.html`, eliminating ~470 internal redirect chains. | Only when `QUARTO_PROJECT_RENDER_ALL=1` (i.e. a full `quarto render`). |

#### Local preview vs Cloudflare deploy

| | Internal `href`s | Sitemap | Server behavior |
| --- | --- | --- | --- |
| `quarto preview` (local) | keep `.html` | clean URLs | Quarto preview server does **not** auto-resolve clean URLs, so links must keep `.html` to avoid 404s. |
| `quarto render` in CI (Cloudflare) | rewritten to clean URLs | clean URLs | Cloudflare serves clean URLs natively; Googlebot sees a fully consistent site (canonical = sitemap = href). |

#### Why local has no SEO risk

- [`docs/` is gitignored](.gitignore) — the rendered site is **never committed**. Cloudflare always serves what CI builds, not what you build locally.
- The CI workflow runs `quarto render` (a full project render), which sets `QUARTO_PROJECT_RENDER_ALL=1` automatically. The link rewriter fires, and production ships clean URLs.
- A local `quarto preview` skips the link rewriter so navigation works in the browser, but that output never reaches the live site.

If you ever change CI to render a single file (`quarto render some-page.qmd`), the env var won't be set and that page would ship with `.html` `href`s. Keep the workflow on a bare `quarto render`.

## Journey Player: Audio and Transcripts

The Journey page (`journey/index.qmd`) features a scroll-driven audio player with chapter markers and interactive transcripts. Audio and transcript files are hosted on **Cloudflare R2**.

### File hosting (Cloudflare R2)

All `.mp3` and `.json` transcript files live in an R2 bucket mapped to the custom domain `firstbreakai.bubblnet.com`. The base URL is set once at the top of `journey/index.qmd`:

```html
<script>var AUDIO_BASE = 'https://firstbreakai.bubblnet.com';</script>
```

Upload new files via the Cloudflare dashboard: **R2 → your bucket → Upload**.

### Adding a new audio episode

1. Upload `scene-N.mp3` to R2.
2. Optionally drop a local copy in `public/audio/` for offline preview (already gitignored).
3. In `journey/index.qmd`, inside the target `<section class="journey-scene">`, add:

```html
<audio preload="auto" data-scene="N"
  data-transcript="scene-N_eng.json"
  data-chapters='[{"t": 0, "title": "Intro"}, {"t": 60, "title": "Topic A"}]'>
  <source data-r2="scene-N.mp3" src="../public/audio/scene-N.mp3" type="audio/mpeg">
</audio>
```

- `data-r2` — filename only; the player prepends `AUDIO_BASE` at runtime.
- `src` — local fallback used when `AUDIO_BASE` is empty.

### Chapters (seek markers)

`data-chapters` is a JSON array on the `<audio>` element. Each entry needs two fields:

```json
{"t": 46, "title": "Ship something real"}
```

- `t` — start time in seconds.
- `title` — label shown in the chapter dropdown and as a marker on the seek bar.

### Adding an interactive transcript

1. Upload `scene-N_eng.json` to R2 (same bucket as audio).
2. Set `data-transcript="scene-N_eng.json"` on the `<audio>` element.
3. Add the transcript panel HTML inside the scene section:

```html
<div class="scene-transcript-wrap">
  <button class="transcript-toggle" type="button" id="transcriptToggleN">Show Transcript</button>
  <div class="scene-transcript" id="sceneTranscriptN"></div>
</div>
```

Replace every `N` with the scene number (e.g. `transcriptToggle2`, `sceneTranscript2`).

**Expected JSON shape:**

```json
{
  "segments": [
    {
      "start_time": 0,
      "end_time": 12.5,
      "text": "Hello and welcome...",
      "speaker": { "id": "speaker_0" }
    }
  ]
}
```

- `speaker_0` displays as **Speaker A**; any other id displays as **Speaker B**.
- `start_time` / `end_time` are in seconds.

### Quick checklist (per scene)

- [ ] Upload `.mp3` to R2
- [ ] Upload `.json` transcript to R2 (if applicable)
- [ ] Add/update `<audio>` in `journey/index.qmd` with `data-r2`, `data-chapters`, `data-transcript`
- [ ] Add transcript panel HTML with matching IDs (`transcriptToggleN`, `sceneTranscriptN`)
- [ ] Preview locally with `quarto preview --port 5942`

### CORS

The R2 bucket CORS policy allows specific origins to fetch audio and transcript files:

- `https://cohort.bubblnet.com` — production site
- `https://firstbreakai.pages.dev` — Cloudflare Pages default domain
- `http://localhost:5942` — local Quarto preview

Update the policy via **Cloudflare Dashboard → R2 → bucket → Settings → CORS Policy**.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes and new roadmap items.
