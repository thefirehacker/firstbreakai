---
name: Rewrite Project Watch Posts
overview: Rewrite both Project Watch posts from encyclopedia-style summaries into lesson-based, code-tracing learning journeys that match the quality and structure of the Step 2 blog. Unsloth becomes a guided source-code reading journey; Autoresearch becomes a build-then-fork hands-on journey.
todos:
  - id: rewrite-unsloth
    content: Full rewrite of project-watch/unsloth-monkey-patching.qmd as a lesson-based code-reading journey (Lessons 0-9)
    status: pending
  - id: rewrite-autoresearch
    content: Full rewrite of project-watch/autoresearch.qmd as a build-then-fork journey (Parts 1-3, Lessons 0-13)
    status: pending
isProject: false
---

# Rewrite Project Watch Posts as Learning Journeys

## The problem with the current posts

Both [unsloth-monkey-patching.qmd](project-watch/unsloth-monkey-patching.qmd) and [autoresearch.qmd](project-watch/autoresearch.qmd) read like Wikipedia articles: they describe what the projects do, show some code snippets, and tack on exercises at the end. A learner reads them and thinks "cool" but doesn't build real understanding.

The [Step 2 blog post](blog/qwen3-run-locally.qmd) sets the standard: lesson-by-lesson progression, "run it first," real code tracing, Mermaid diagrams at each stage, and practical checkpoints. Both Project Watch posts need to match this.

## Design principle

Each lesson answers: **"What problem is the author facing, and how did they solve it?"** The learner walks in Daniel Han's / Karpathy's shoes, facing the same constraints, seeing the same code, making the same decisions.

---

## Post 1: Unsloth (code-reading journey)

**Goal:** Learner reads and fully understands the Unsloth Qwen3 source by tracing every function call step-by-step. "Code reading as a skill."

**File:** [project-watch/unsloth-monkey-patching.qmd](project-watch/unsloth-monkey-patching.qmd) (full rewrite)

### Structure

The post opens with the same front matter / roadmap connections but switches to lesson-based structure:

- **Opening:** "By the end of this post you will have..." (concrete outcomes like Step 2). Show the "before/after" speed difference. Show the HF code path vs. Unsloth code path side by side.
- **Lesson 0: The problem Daniel saw.** HuggingFace is slow. Why? Show the default `Qwen3Attention.forward()` from HF transformers. Count the kernel launches. Diagram: "6 separate GPU kernel launches for one RMSNorm." This is the problem.
- **Lesson 1: What is monkey patching? (the 3-line version).** The TinyZero example. But DON'T just show it -- walk through what happens in Python's object model. Show `id(Qwen2FlashAttention2.forward)` before and after. Checkpoint: "You now understand the mechanism."
- **Lesson 2: Open the real file.** Guide the learner to open `unsloth/models/qwen3.py` on GitHub. Trace the imports at the top -- what is Unsloth importing from HuggingFace? What is it importing from its own codebase? Annotated reading of lines 1-120 from the reference material.
- **Lesson 3: The pre_patch() method -- where the switch happens.** Read `FastQwen3Model.pre_patch()` line by line. Each `.forward =` assignment is a decision. Why does Daniel patch all three attention variants to the same function? Why does he also patch the RotaryEmbedding? Diagram: the "before" execution path vs. the "after" execution path.
- **Lesson 4: Trace `Qwen3Attention_fast_forward` -- the training path.** Walk through the actual function code from the reference material (lines 120-228). Key moments to annotate:
  - `apply_qkv` -- why custom QKV projection?
  - `fast_rms_layernorm` on Q and K -- connect to Step 2 `rmsnorm()` in C
  - `fast_rope_embedding` -- connect to Step 2 RoPE rotation
  - `run_attention` with backend dispatch -- why choose between SDPA/Flash/VarLen?
  - Each annotated code block has a "What Daniel is solving here" callout.
- **Lesson 5: Trace `Qwen3Attention_fast_forward_inference` -- the decode path.** Walk through the inference fast path (lines 234-463). Key moments:
  - The docstring explaining QK^T chunking
  - Pre-allocated paged KV cache (`self.paged_attention`)
  - Why prefill and decode need separate paths
  - Manual RoPE application (in-place math)
  - Sliding window handling
  - Connect to Step 2's KV cache explanation
- **Lesson 6: Follow one call down -- the RMSNorm Triton kernel.** Guide the learner to `unsloth/kernels/rms_layernorm.py`. Show the `@triton.jit` kernel. Explain what "fused" means concretely: one kernel body doing square, sum, mean, rsqrt, and multiply. Diagram: "unfused 6 launches vs fused 1 launch." This is where the speedup actually lives.
- **Lesson 7: The full stack -- from `from_pretrained()` to Triton.** Put it all together. Trace the complete call chain: `from_pretrained()` -> `model_patcher=FastQwen3Model` -> `pre_patch()` -> `Qwen3Attention.forward = ...` -> `fast_rms_layernorm()` -> Triton kernel. One diagram showing the full depth.
- **Lesson 8: Trade-offs and version fragility.** Show the version checks. Show what breaks when HF updates. Show the warm-up cost. This is the honest engineering reality.
- **Lesson 9: Code reading exercises.** NOT tacked-on exercises. Progressive challenges that require the learner to actually open files and answer specific questions:
  1. Find `pre_patch()` -- how many `.forward` assignments? List them.
  2. In `Qwen3Attention_fast_forward`, find where Q and K get normalized. What function is called? Where is that function defined?
  3. Open HF's `modeling_qwen3.py` side-by-side. Find three concrete lines that differ between HF's `forward()` and Unsloth's `fast_forward()`.
  4. In `rms_layernorm.py`, find the `@triton.jit` kernel. Identify the "fuse boundary" -- where multiple math ops run in one kernel.
  5. Write a 3-line monkey patch that prints "PATCHED!" before calling the original `RMSNorm.forward()`. Run a forward pass. Does it print?

### Key content from reference material to use

The file [temp/DanielHan-Unsloth-Monkey-Patching/Monley-Patching-LLM.md](temp/DanielHan-Unsloth-Monkey-Patching/Monley-Patching-LLM.md) has:

- Full annotated `Qwen3Attention_fast_forward` source (lines 120-228)
- Full annotated `Qwen3Attention_fast_forward_inference` source (lines 234-463)
- Full `FastQwen3Model.pre_patch()` and `from_pretrained()` (lines 466-530)
- Detailed explanations of each fast path change (sections 4.1-4.6)
- Fused kernel explanations (sections 6.1-6.6)
- Trade-offs analysis (section 10)
- Teaching structure recommendations (sections 11-14)

All of this should be woven into lessons, not dumped as reference sections.

---

## Post 2: Autoresearch (build-then-fork journey)

**Goal:** Learner builds a mini autoresearch from scratch to understand the pattern, then forks the real repo and runs experiments.

**File:** [project-watch/autoresearch.qmd](project-watch/autoresearch.qmd) (full rewrite)

### Structure

- **Opening:** "By the end of this post you will have..." (concrete outcomes). Show Karpathy's autoresearch running -- the git log of commits/reverts, the val_bpb improving over time. This is the destination.
- **Part 1: Build your own autoresearch from scratch**
  - **Lesson 0: The question Karpathy asked.** "What if AI could do research on its own code?" Frame the insight: classical AutoML searches a human-defined grid. Autoresearch searches over code diffs. Show the difference with a diagram.
  - **Lesson 1: Build the editable artifact.** Write a tiny `my_train.py` -- a 20-line training script for something simple (a tiny neural net fitting a sine wave, or a simple sorting algorithm -- something that runs in seconds, not minutes). The learner writes this file.
  - **Lesson 2: Build the fixed evaluation.** Write `my_eval.py` -- measures a single metric (loss, accuracy, speed). The key constraint: the evaluation is FIXED. The agent cannot change it. Explain why this matters (anti-Goodharting). The learner writes this file.
  - **Lesson 3: Build the loop.** Write `my_loop.sh` (or Python script) that:
    1. Asks an LLM (via API or local model) to propose a code change to `my_train.py`
    2. Applies the change
    3. Runs `my_train.py`
    4. Measures the metric with `my_eval.py`
    5. If improved: `git commit`. If not: `git revert`.
    6. Repeat.
    Walk through each step. The learner builds this incrementally. Git-as-memory is the key insight -- show why this is better than "just overwrite the file."
  - **Lesson 4: Run it and watch.** The learner runs their loop for 5-10 iterations. What happens? Did it improve? Did it break? Did it propose garbage? This is where the learner encounters the REAL problems: hallucinated code, regression, shallow search. These aren't abstract -- the learner just experienced them.
  - **Lesson 5: What went wrong? (The taste problem).** After running the loop, the learner has seen failures. Now explain WHY: the agent has no memory, no taste, no triage. This motivates everything Karpathy designed.
- **Part 2: Open Karpathy's version**
  - **Lesson 6: Read `program.md`.** Guide the learner to open the real file. Compare it to what they built. Key questions: What constraints did Karpathy add that you didn't? Where does "research taste" live? Annotated reading.
  - **Lesson 7: Read `prepare.py`.** Find `TIME_BUDGET = 300`. Find `evaluate_bpb`. Compare to `my_eval.py`. Why is the evaluation deterministic? What would happen if the agent could modify it?
  - **Lesson 8: Read `train.py`.** This is a real character-level language model. Connect to Step 2: "you saw attention, RoPE, KV cache in C -- here they are in PyTorch." What is the agent allowed to change in this file? What makes this file small enough for an LLM to reason about?
  - **Lesson 9: Trace one experiment.** Look at Karpathy's session reports (linked from README). Pick one change. What did the agent propose? What was the diff? Did val_bpb improve? Was the commit kept or reverted? The learner traces a real experiment.
  - **Lesson 10: The human/agent split.** Now that the learner has seen both their loop and Karpathy's, the architecture becomes clear. Diagram: human controls evaluation + constraints, agent controls code + experiments. Why this split matters for safety, reliability, and scaling.
- **Part 3: Fork, run, and design**
  - **Lesson 11: Fork and run (if GPU available).** Step-by-step fork instructions. Run one 5-minute experiment. Analyze the result. Write a one-paragraph analysis.
  - **Lesson 12: Design your own loop.** Pick a domain outside ML training (sorting algorithm, CSS layout, prompt template, API configuration). Design an autoresearch-style loop: what is the editable artifact? What is the fixed eval? What is the time budget? What goes in program.md? Write it as a one-page design doc.
  - **Lesson 13: The pattern beyond Karpathy.** Now show how the community generalized it: autocontext, distributed agents, skill factories. The learner has enough foundation to understand WHY these extensions exist. Diagram: the core loop as a reusable pattern.

### Key content from reference material to use

- [temp/Autoresearch/Autoresearch-Analysis.md](temp/Autoresearch/Autoresearch-Analysis.md): Community analysis, 4 usage directions, market directions, tensor's argument
- [temp/Autoresearch/auto-research-deep-research-report.md](temp/Autoresearch/auto-research-deep-research-report.md): PR/issue clustering, cross-correlation of GitHub and Twitter themes, detailed market analysis, "minimal research JVM" framing
- The current post's community analysis sections should be condensed into Lesson 13, not spread across the post as the primary content

---

## What does NOT change

- [project-watch/index.qmd](project-watch/index.qmd) -- card layout stays the same
- [styles/blog.css](styles/blog.css) -- styling stays the same
- [_quarto.yml](_quarto.yml) -- no changes needed
- Front matter format stays consistent (title, description, date, categories)
- Roadmap connections table stays (but moves to a callout or top block, not a standalone section)

## Estimated size

- Unsloth post: ~800-1000 lines (comparable to Step 2's ~1300 lines, but denser since it's code reading not code writing)
- Autoresearch post: ~900-1100 lines (Part 1 build-from-scratch is the longest section)

