---
name: Sponsors and enterprise pages
overview: Add a /sponsors page (cloud provider sponsorship pitch with partner tiers and CTA) and a /teams page (paid enterprise cohort offering with pricing tiers and contact form) to the cohort site, linked from the footer.
todos:
  - id: sponsors-page
    content: Create sponsors.qmd with partner tiers, compute mapping table, and contact CTA
    status: completed
  - id: teams-page
    content: Create teams.qmd with pricing tiers, free vs paid comparison, and contact CTA
    status: completed
  - id: site-integration
    content: Add footer links in _quarto.yml and brief mentions in index.qmd
    status: completed
isProject: false
---

# Add Sponsors and Enterprise Pages

## Page 1: `/sponsors` — Partner with First Break AI

New file: [sponsors.qmd](sponsors.qmd)

### Content structure

- **Hero:** "Partner with First Break AI" — one line pitch: sponsor the cohort that teaches AI engineers from scratch using your infrastructure
- **Why sponsor:** reach, audience profile (builders learning inference + training + product), curriculum integration (not just a logo — your platform becomes part of the lesson), content co-marketing
- **Where credits are used:** a clear table mapping roadmap steps to compute needs:
  - Step 2: CPU-only (no credits needed)
  - Step 3: Inference API credits (Together, Fireworks, etc.)
  - Step 4: GPU training credits (Lambda, Modal, RunPod, Google Cloud)
  - Step 5: Deployment credits (Modal, cloud hosting)
- **Sponsor tiers** (three levels):
  - **Compute Partner:** credits for participants, logo on site + lessons that use the platform, "Powered by" badge
  - **Infrastructure Partner:** above + dedicated lesson/tutorial using their platform, co-branded blog post
  - **Founding Partner:** above + named cohort step ("Step 4, powered by [Provider]"), joint office hours session, case study
- **Current sponsors:** placeholder section ("Sponsor spots for Cohort 01 are open") — ready for logos
- **CTA:** "Interested? Email contact@bubblspace.com" with a mailto link — or a simple Tally/Typeform embed if preferred

### Design

- Use the same `page-layout: full` + `content-section` pattern as [index.qmd](index.qmd)
- Reuse `.cta-banner` for the contact CTA
- Keep it clean and professional — this page faces DevRel and partnerships teams

---

## Page 2: `/teams` — First Break AI for Teams

New file: [teams.qmd](teams.qmd)

### Content structure

- **Hero:** "First Break AI for Teams" — upskill your engineering team on inference, training, and AI product building
- **What's included (vs free):** a clear comparison:
  - Free cohort: self-paced, community Discord, open content, public office hours
  - Teams: everything in free + instructor-led sessions, private Slack/Teams channel, team progress dashboard, custom project scoping, priority Q&A, PO/invoice billing, completion certificates
- **Three pricing tiers:**
  - **Individual Pro** ($299): certificate + priority support + 1x 1:1 session with cohort lead
  - **Team** (5-20 seats, $999/seat): above + private channel + team dashboard + custom exercises
  - **Enterprise** (20+ seats): custom pricing, SOW, dedicated onboarding, joint roadmap planning
- **What teams build:** a brief section showing the portfolio outcome — blog, trained model, deployed product, open-source contribution
- **Social proof placeholder:** "Teams from [logos] are learning with First Break AI" — empty for now but structured for future logos
- **FAQ:** 2-3 enterprise-specific questions (billing, cancellation, custom content)
- **CTA:** "Get started — contact@bubblspace.com" or a Tally form for "Request a quote"

---

## Site integration

### Footer update in [_quarto.yml](_quarto.yml)

Add both pages to the footer center section (alongside Terms, Privacy, etc.):

```yaml
center:
  - text: "Sponsors"
    href: sponsors.qmd
  - text: "Teams"
    href: teams.qmd
  - text: "Terms"
    href: terms.qmd
  # ... rest unchanged
```

Do NOT add to the main navbar — these are secondary pages. The navbar is already full. Footer placement keeps them discoverable without cluttering navigation.

### Homepage mention

Add a brief line in the [index.qmd](index.qmd) content section, near the "Who it's for" area:

- "**For teams:** [First Break AI for Teams](teams.qmd) — instructor-led, with team dashboards and enterprise billing."
- "**For sponsors:** [Partner with us](sponsors.qmd) — provide compute credits and reach AI engineers learning your platform."

---

## Files changed

- **New:** [sponsors.qmd](sponsors.qmd)
- **New:** [teams.qmd](teams.qmd)
- **Edit:** [_quarto.yml](_quarto.yml) — add footer links
- **Edit:** [index.qmd](index.qmd) — add brief mentions with links
