# First Break AI

Free, open cohort for your first break in AI. See the live site: [thefirehacker.github.io/firstbreakai](https://thefirehacker.github.io/firstbreakai).

## Deploying the site (GitHub Pages)

The site is built with [Quarto](https://quarto.org/) and deployed via GitHub Actions.

**If you get a 404:** In the repo go to **Settings → Pages**. Under "Build and deployment", set **Source** to **"GitHub Actions"** (not "Deploy from a branch"). Save. Push a commit to `main` to trigger the workflow; the site will update at `https://thefirehacker.github.io/firstbreakai/`.

- With **GitHub Actions**: the `publish.yml` workflow runs on push to `main`, renders the Quarto site into `docs/`, and deploys it. No need to commit built files.
- The repo root contains only source (`.qmd`); the built HTML is in `docs/` and is gitignored.

## Local preview

```bash
quarto render
quarto preview
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes and new roadmap items.
