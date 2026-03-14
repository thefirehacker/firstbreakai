---
name: Project Watch section
overview: "Add a new top-level \"Project Watch\" section to First Break AI with its own nav item, card-based index page, and two full annotated blog posts: (1) Unsloth monkey patching / Daniel Han, and (2) Karpathy's Autoresearch."
todos:
  - id: pw-css
    content: Add Project Watch card color variants to styles/blog.css
    status: completed
  - id: pw-nav
    content: Add Project Watch to _quarto.yml navbar
    status: completed
  - id: pw-index
    content: Create project-watch/index.qmd with card layout
    status: completed
  - id: pw-unsloth
    content: Create project-watch/unsloth-monkey-patching.qmd full blog post
    status: completed
  - id: pw-autoresearch
    content: Create project-watch/autoresearch.qmd full blog post
    status: completed
isProject: false
---

# Project Watch: learn by studying real AI projects

## What is Project Watch

A new top-level section on the First Break AI site where cohort members study real, notable AI projects in depth. Each entry is a full annotated walkthrough: what the project does, why the engineering decisions matter, annotated source code, diagrams, and exercises -- with explicit links back to what learners covered in the roadmap.

This is "learn by reading the masters" -- curated and annotated for people getting their first break in AI.

---

## 1. Site structure changes

### New folder: `project-watch/`

```
firstbreakai/
├── project-watch/
│   ├── index.qmd                         # Card-based index (reuse blog card CSS)
│   ├── unsloth-monkey-patching.qmd       # Post 1: Daniel Han / Unsloth
│   └── autoresearch.qmd                  # Post 2: Karpathy's Autoresearch
```

### Navbar update: [_quarto.yml](_quarto.yml)

Add a new nav item after "Blog":

```yaml
- href: project-watch/index.qmd
  text: Project Watch
```

### CSS: reuse existing [styles/blog.css](styles/blog.css)

The `.blog-card` styles already work for any card grid. Add a couple of new color variants for Project Watch cards (e.g. `.blog-card-pw-1`, `.blog-card-pw-2`) -- or just use the existing step colors. No new CSS file needed; extend `blog.css` with a small block.

---

## 2. Project Watch index page

**File: `project-watch/index.qmd`**

- Front matter with `css: ../styles/blog.css`
- Short intro: "Project Watch is where we study real AI projects in depth. Each entry is a full annotated walkthrough -- what the project does, why the engineering decisions matter, and how it connects to what you're learning."
- Two cards using the same `.blog-cards` / `.blog-card` grid:
  - **Card 1: Unsloth Monkey Patching** -- "How Daniel Han makes LLM inference 2x faster by replacing HuggingFace internals at runtime. Fused GPU kernels, practical GPU optimization, and why it matters for every model you'll ever serve." Links to `unsloth-monkey-patching.qmd`.
  - **Card 2: Autoresearch** -- "Karpathy's autonomous research harness. How the community turned a toy training loop into an agentic ResearchOps platform. Agents, search policy, verification, and the future of AI R&D." Links to `autoresearch.qmd`.
- "Back to Roadmap" link at bottom.

---

## 3. Post 1: Unsloth Monkey Patching (Daniel Han)

**File: `project-watch/unsloth-monkey-patching.qmd`**

**Source material:** [temp/DanielHan-Unsloth-Monkey-Patching/Monley-Patching-LLM.md](temp/DanielHan-Unsloth-Monkey-Patching/Monley-Patching-LLM.md) -- contains the full analysis, Qwen3 code, explanation of three levels of patching, fused kernel examples, and teaching structure.

**Structure (full blog post):**

- **Roadmap nav** at top (like Step 2 post): links back to roadmap, shows which steps this connects to (Steps 2, 3, 4)
- **Table of contents**
- **What is Unsloth** -- one-paragraph context (Daniel Han, what it does, why it matters)
- **What is monkey patching** -- the core idea: overwrite `.forward()` at runtime to reroute execution
  - Mermaid diagram: HF class load -> overwrite methods -> normal API call executes custom path
- **Minimal example** -- the one-liner Qwen2 flash attention patch (from the reference)
- **End-to-end example: Qwen3 patch** -- the full stack (attention, decoder layer, model, CausalLM, PEFT, generation prep)
  - Mermaid diagram: patch routing map showing all 6+ assignments
  - Annotated code from the reference material
- **What the Qwen3 fast path actually changes** -- QKV path, Q/K norm, fast RoPE, attention dispatch, prefill vs decode split, KV cache handling
- **Fused GPU kernels** -- where the real speedups come from
  - RMSNorm (best first example), RoPE, SwiGLU, GEGLU, cross-entropy
  - Why each matters; connection back to Step 2 concepts (learners already saw RMSNorm, RoPE, attention in `run.c`)
- **Trade-offs** -- version fragility, warm-up cost, static-shape assumptions, harder debugging
- **Connection to your learning** -- table mapping Unsloth concepts back to roadmap steps
- **Exercises** -- e.g. "Find the RMSNorm Triton kernel in Unsloth's repo," "Compare the Qwen3 fast forward to the vanilla HF forward," "Write a minimal monkey patch that swaps one method"

---

## 4. Post 2: Autoresearch (Karpathy)

**File: `project-watch/autoresearch.qmd`**

**Source material:** [temp/Autoresearch/Autoresearch-Analysis.md](temp/Autoresearch/Autoresearch-Analysis.md) (conversation analysis) and [temp/Autoresearch/auto-research-deep-research-report.md](temp/Autoresearch/auto-research-deep-research-report.md) (structured deep research report).

**Structure (full blog post):**

- **Roadmap nav** at top: connects to Steps 3, 5 (inference engines, building AI products)
- **Table of contents**
- **What is Autoresearch** -- Karpathy's repo: a closed-loop research harness (agent edits `train.py`, fixed 5-min eval, git as ledger)
  - Mermaid diagram: the autoresearch loop (propose code change -> run train.py -> measure val_bpb -> keep/revert -> repeat)
- **Why it matters** -- not AutoML. The search object is "a patch to executable research code," not a point in a parameter space. Agents navigating "soft program-space moves."
- **The architecture** -- `program.md` (agent instructions), `train.py` (editable code), `prepare.py` (fixed eval), git as memory
  - Mermaid diagram: human/agent split (human edits .md prompt, agent edits .py, evaluation is fixed)
- **What the community is actually building** -- four directions from the analysis:
  1. Testbed for autonomous model improvement
  2. Scaffold for research orchestration (memory, multi-agent, dashboards)
  3. Portability benchmark (MLX, CUDA, multi-GPU, Colab)
  4. "Research taste" + verification tooling
- **The "autoresearch as a pattern" insight** -- people are generalizing the loop beyond ML training (autocontext, distributed agents, skill factories, quant strategies)
  - Mermaid diagram: autoresearch pattern generalized (fixed eval + tight loop + agent edits + git ledger = reusable for any domain)
- **Why hype cooled but the project didn't stall** -- the five reasons from the analysis (novelty ended, repo intentionally minimal, search-policy wall, infra PRs aren't viral, forks fragment attention)
- **Market directions** -- ResearchOps, hardware-specific tuning, agent skill factories, verification tooling
- **Connection to your learning** -- how autoresearch relates to what you learned in Steps 2-3 (inference, training loops, eval metrics) and previews Steps 4-5 (training, building products)
- **Exercises** -- e.g. "Fork autoresearch and run one 5-minute experiment," "Read program.md and identify the agent's constraints," "Propose one change to train.py and predict whether it will improve val_bpb"

---

## 5. Summary of files to create/edit


| File                                        | Action                                                        |
| ------------------------------------------- | ------------------------------------------------------------- |
| `project-watch/index.qmd`                   | New: card-based index with two project cards                  |
| `project-watch/unsloth-monkey-patching.qmd` | New: full annotated blog post from reference material         |
| `project-watch/autoresearch.qmd`            | New: full annotated blog post from reference material         |
| `styles/blog.css`                           | Edit: add `.blog-card-pw-1`, `.blog-card-pw-2` color variants |
| `_quarto.yml`                               | Edit: add "Project Watch" nav item                            |


