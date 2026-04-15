---
name: podcast-transcript-writer
description: >
  Write teaching-focused podcast transcripts for the First Break AI journey storyboard.
  Use this skill whenever the user asks to write, create, draft, or improve a podcast
  transcript or script for any scene of the First Break AI journey, or when they mention
  "scene transcript", "podcast script", "journey audio", or "NotebookLM script". Also
  trigger when they ask to write dialogue for the storyboard or create audio content
  for the /journey/ page.
---

# First Break AI — Podcast Transcript Writer

You are writing teaching-focused podcast transcripts for the First Break AI journey
storyboard. Each transcript becomes audio via NotebookLM and plays as users scroll
through the /journey/ page.

## What makes a good transcript

The transcripts are **teaching conversations**, not hype pieces. Two hosts discuss
the actual technical content from the program — blog posts, office hours sessions,
and the roadmap. The listener should learn something real from every segment.

**What to do:**
- Teach specific concepts with the exact analogies, code examples, and explanations
  from the published content
- Reference real moments from office hours (discussions, questions learners asked,
  insights that emerged)
- Explain the "why" behind each concept — why it matters, why it's in this order
- Let concepts breathe — don't rush through technical explanations
- Connect each topic to what comes before and after in the roadmap

**What NOT to do:**
- Don't hype or use marketing language ("amazing", "incredible", "game-changing")
- Don't just name-drop topics — explain them in enough depth that the listener learns
- Don't summarize when you should teach
- Don't skip the technical substance to save time

## The two hosts

**Host A (The Guide):** Knows the program deeply. Teaches clearly using examples.
Explains concepts step by step. References specific blog lessons, office hours
discussions, and code examples. Confident but not condescending.

**Host B (The Learner):** Asks genuine questions that deepen understanding. Pushes
for clarity when something is vague. Occasionally summarizes to check understanding.
Represents the target audience — smart but potentially new to ML. Their questions
should be the questions the listener is thinking.

The dynamic should feel like a knowledgeable friend walking someone through the
material. Not a lecture, not a sales pitch — a real conversation where both
participants are thinking.

## Research process

Before writing a transcript, you need to deeply understand the source material.
This is the most important step — shallow research produces shallow transcripts.

### Step 1: Identify the scene and its roadmap step

The journey has 6 scenes mirroring the roadmap:

| Scene | Title | Roadmap Step | Key content |
|-------|-------|-------------|-------------|
| 1 | Ship Something Real | Step 1 | Quarto, GitHub, AI IDE, cohort philosophy |
| 2 | See Inside the Machine | Step 2 | Qwen3 in pure C, tokenization, attention, KV cache |
| 3 | Think at Production Scale | Step 3 | vLLM, batching, quantization, benchmarks |
| 4 | Train Your Own | Step 4 | PyTorch, DDP, speedrun, parallelism ladder |
| 5 | Ship a Product | Step 5 | RAG, agents, tool use, deployment |
| 6 | Prove It | Step 6 | Capstone, OSS contribution, portfolio |

### Step 2: Read ALL relevant source material

For the requested scene, read these files completely (not skimming):

**Always read:**
- `roadmap.qmd` — the full section for this step (tasks, learning objectives, resources)
- `index.qmd` — homepage philosophy and framing

**Read the relevant blog posts** (in `blog/`):
- Scene 1: No dedicated blog post yet; draw from roadmap + office hours
- Scene 2: `blog/qwen3-run-locally.qmd` (9 lessons, ~1300 lines — read ALL of it),
  `blog/model-formats-gguf-safetensors.qmd` (7 lessons)
- Scene 3-6: Check `blog/` for any new posts related to the step

**Read the relevant office hours** (in `office-hours/`):
- Check the roadmap to see which sessions cover this step
- Read the full .qmd write-ups for those sessions
- Also check `audio-transcripts/` for raw Gemini meeting notes (.docx) — these
  contain real conversational moments, specific quotes, and teaching dynamics that
  the polished write-ups may omit. Use `textutil -convert txt -stdout` to read them.

### Step 3: Extract teaching substance

From each source, pull out:

1. **Specific technical concepts** — with the exact explanations and analogies used
2. **Code examples** — specific functions, line numbers, code snippets that illustrate concepts
3. **Real conversational moments** — questions learners asked, "aha" moments, confusions
   that led to better explanations, exact quotes that would work in a podcast
4. **The learning progression** — why concepts are taught in this specific order
5. **Connections to other steps** — how this builds on previous steps and sets up next ones
6. **Exercises or experiments** — things learners are asked to try

## Writing the transcript

### Format

Write in markdown with speaker labels:

```markdown
**Host A:** [dialogue]

**Host B:** [dialogue]
```

Use `---` horizontal rules to separate major topic shifts.

### Structure

A good transcript follows this arc:

1. **Opening** (~1 min) — Ground the listener: what are we covering and why does it
   matter in the journey? Connect to the previous scene if applicable.

2. **Core teaching sections** (~6-8 min) — The substantive middle. Each section should:
   - Introduce the concept clearly
   - Use the specific analogy or example from the source material
   - Have Host B ask the question the listener would ask
   - Go deep enough that the listener actually learns something
   - Connect to the next concept naturally

3. **Synthesis** (~1 min) — Pull the pieces together. What has the listener learned?
   What can they now do or understand?

4. **Bridge to next scene** (~30 sec) — Natural transition that makes the listener
   want to continue to the next scene. Not a cliffhanger — a genuine "here's what's
   next and why it matters."

### Target length

- Scenes 1, 3, 4: ~1800-2200 words (8-10 minutes of audio)
- Scene 2: ~2500-3000 words (10-13 minutes — this is the most content-dense scene)
- Scenes 5, 6: TBD when content exists

### Dialogue guidelines

- Host B's questions should advance the explanation, not just say "wow" or "interesting"
- When Host A explains something technical, they should use the SPECIFIC analogy from
  the source material (e.g., the "film script format" analogy for chat templates, the
  "trophy/suitcase" example for attention)
- Include moments where Host B summarizes to check understanding — and Host A confirms
  or corrects. This mirrors real learning.
- Don't be afraid of technical depth. The audience is smart. Explain through analogy
  first, then precision.
- Keep filler to zero. Every line should either teach something, ask a question that
  advances the teaching, or connect concepts.

## Output format

Save the transcript to:
`AI-Prompts/journey_prompts/scene-{N}-transcript.md`

Include at the top of the file:
1. A comment block with NotebookLM instructions (how to upload and what customize
   prompt to use)
2. A comment block listing which source files were read and what key content was
   extracted

Include at the bottom:
1. NotebookLM iteration prompts — specific suggestions for refining the generated
   audio if quality isn't right

## Quality checklist

Before delivering the transcript, verify:

- [ ] Every major concept from the roadmap step is covered
- [ ] At least 3 specific analogies/examples from the blog posts are used verbatim
- [ ] At least 2 real moments from office hours are referenced (quotes, questions, discussions)
- [ ] The learning progression is clear — concepts build on each other
- [ ] Host B asks at least 5 substantive questions (not "wow, really?")
- [ ] The connection to previous and next scenes is explicit
- [ ] No marketing language — this teaches, it doesn't sell
- [ ] Target word count is met (not too short, not padded)
