# Contributing to First Break AI

First Break AI is a community-driven cohort. We welcome contributions to the syllabus, roadmap, checklist, and resources.

## How to contribute

- **Typos and small fixes** — Open a pull request that edits the relevant `.qmd` file (e.g. `checklist.qmd`, `setup.qmd`, `roadmap.qmd`).
- **New roadmap items or modules** — Open an issue with the label or title "Roadmap" describing the topic and learning objectives, or submit a PR that adds a new section to `roadmap.qmd`.
- **New resources or links** — Propose in an issue or add a "Resources" section/link in a PR. We prefer linking out to [thefirehacker](https://thefirehacker.github.io/) and other references rather than duplicating long content on this site.
- **Suggest a topic** — Open an issue with your idea; we can discuss and then add it to the roadmap or checklist.

## PR process

1. Fork the repo and create a branch (e.g. `fix/typo-setup` or `content/new-roadmap-phase`).
2. Edit the Quarto source (`.qmd`) or add new pages as needed.
3. Run `quarto render` locally if you have Quarto installed (optional but helpful).
4. Open a PR with a short description. We’ll review and merge.

## Where things live

| Content        | File           |
|----------------|----------------|
| Landing page   | `index.qmd`    |
| Checklist      | `checklist.qmd`|
| AI setup       | `setup.qmd`    |
| Roadmap        | `roadmap.qmd`  |

Site config and nav are in `_quarto.yml`. Do not change the repo name or GitHub Pages setup without discussion.

Thank you for helping others get their first break in AI.
