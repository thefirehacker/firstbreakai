# NotebookLM Prompts: First Break AI Journey — Scenes 1-4 (v3)

<!-- ============================================================
  WHY THIS FILE EXISTS
  ============================================================
  First Break AI is building a scroll-driven narrative storyboard
  (/journey/) where AI-generated podcast episodes auto-play as
  users scroll through 6 scenes mirroring the roadmap.

  This v3 was built after deep-reading the RAW Gemini meeting
  transcripts from the March 27 and April 10 office hours — not
  just the polished blog write-ups. The transcripts revealed:
  - Real participant names and dynamics (FireHacker teaches,
    Learner01 asks sharp questions and pushes back)
  - Live "aha" moments (Learner01 coining "context grasping")
  - Candid humor ("I forgot the word hyperparameter")
  - Analogies used in real-time that differ from blog versions
  - Topics that never made it into the blog write-ups
  - Emotional moments about building in India, affordable
    training, and career paths

  These prompts reference specific conversational moments,
  timestamps, and quotes from the live sessions. This makes
  NotebookLM produce episodes that feel like you're sitting
  in on the best parts of the actual office hours.

  SOURCES TO LOAD IN NOTEBOOKLM BEFORE GENERATING:
    1. https://thefirsthacker.github.io/firstbreakai/roadmap/
    2. https://thefirsthacker.github.io/firstbreakai/blog/qwen3-run-locally/
    3. https://thefirsthacker.github.io/firstbreakai/blog/model-formats-gguf-safetensors/
    4. https://thefirsthacker.github.io/firstbreakai/office-hours/2026-03-13/
    5. https://thefirsthacker.github.io/firstbreakai/office-hours/2026-03-27/
    6. https://thefirsthacker.github.io/firstbreakai/office-hours/2026-04-10/
    7. https://thefirsthacker.github.io/firstbreakai/ (homepage)
    8. The March 27 Gemini meeting notes (.docx) — raw transcript
    9. The April 10 Gemini meeting notes (.docx) — raw transcript

  HOW TO USE:
    1. Load all sources above into a NotebookLM notebook
    2. For each scene, go to Audio Overview > Customize
    3. Paste the scene prompt into the customization field
    4. Generate the Deep Dive audio
    5. Use Interactive Mode to iterate
    6. Export and save as:
       - public/audio/scene-1-ship-something-real.mp3
       - public/audio/scene-2-see-inside-the-machine.mp3
       - public/audio/scene-3-think-at-production-scale.mp3
       - public/audio/scene-4-train-your-own.mp3
============================================================ -->

---

## Scene 1: "Ship Something Real" (~5 min)

<!-- ============================================================
  CONTENT SOURCES:
  - Homepage: "No prerequisites, no applications..."
  - Roadmap Step 1: Quarto blog, GitHub, AI IDE
  - Session 1 (13 Mar): GitHub PRs, merge conflicts, rebase debate
  - Session 3 (10 Apr) RAW TRANSCRIPT: Learner01's stock game demo,
    Mini Militia LAN party concept, web-first no-backend design,
    friend's pitch to financial influencers as validation
  - Session 3 RAW TRANSCRIPT: Claude Code leak — FireHacker's
    excitement ("some aspects are really cool"), Learner01's mature
    security refusal ("I'll wait somebody will put something
    which is reliable")
  - Session 1 (13 Mar): Cohort-based vs self-paced dropout stats

  KEY LIVE MOMENTS TO WEAVE IN:
  - Learner01's game uses a "Mini Militia" model — local hotspot,
    room codes, temporary nicknames, no accounts
  - A friend who runs a content boosting company pitched the
    game to financial influencers at an event — real validation
  - Learner01 explicitly refused to implement leaked Claude Code
    repos because of malware risk — security maturity
  - FireHacker described Claude Code's architecture: "it compacts
    memory in a single file... skills and sub-agents where you
    can spawn more than one agent looking at three different
    code bases"
  - Learner01's insight: "the sub-agents consume slower... lower
    memory... they go to smaller models for menial work"
============================================================ -->

> Create a Deep Dive episode (~5 minutes) called **"Scene 1: Ship Something Real."**
>
> This is the opening episode of a narrative podcast series for **First Break AI** — a free, community-driven cohort helping people get their first break in AI. The cohort launches May 1, 2026. Pre-cohort office hours have been running every Friday since March, with a small group led by FireHacker and joined by learners like Learner01.
>
> **The opening question:**
> Start with what every career switcher asks: "Where do I even start with AI?" Most programs say "take this course." First Break AI says "Ship something. Today. Before you understand anything about transformers or attention or training loops — ship something with your name on it."
>
> **Who this is for (use the homepage's exact language):**
> "No prerequisites, no applications. Follow the roadmap, build in the open, and let your work speak for itself." Students, professionals, career switchers, the simply curious. No gatekeeping. The only question: will you build?
>
> **What Step 1 asks you to do:**
> Set up a Quarto blog, host it on GitHub Pages, and use an AI IDE (Cursor or Claude Code) from day one. Your first deliverable is a live website — about-me page, a blog post, a "Today I Learned" entry. Your portfolio starts on day one, not after 12 weeks of lectures.
>
> **The stock game story — a real learner shipping a real thing (from the April 10 office hours):**
> This is the best example of the "ship something real" philosophy in action. Learner01, one of the learners, presented a web-based multiplayer stock trading game he was building. The concept: inspired by a physical board game with 8 listed companies, fluctuating prices, and player cards worth +/- 10 points that affect stock prices. Power cards like "debenture" and "freeze." The strategic element: you can see what your opponents are buying and react — if someone buys 50,000 shares of SBI and you have a -5 card, you know exactly what to do.
>
> But what made the discussion remarkable wasn't the game mechanics — it was the design philosophy. Learner01 chose web-first over a native app for easier user testing. No traditional backend — it's LAN-based, like the mobile game Mini Militia where one device hosts and others join with a room code. When the host closes the window, all data clears. Temporary nicknames instead of full account systems. Minimal persistence. The whole point: keep friction low so you can playtest with friends immediately.
>
> And here's the validation moment: a friend who runs a content boosting company — collaborates with influencers, does event setups — recently pitched the game to financial influencers at an event. People loved it. Real-world feedback before writing a single line of backend infrastructure.
>
> FireHacker's response captured the philosophy perfectly: it's not about what framework you use. It's about what you ship first and what you learn from putting it in front of people.
>
> **The GitHub workflow (from the March 13 office hours):**
> The first session went deep on real engineering workflows — not "git add, git commit, git push" tutorials. Pull requests, code review, merge conflicts (when Git shows you those `<<<<<<< HEAD` markers because two people edited the same lines). The merge vs rebase debate: merge creates merge commits and is safer for collaboration; rebase replays commits for cleaner history but rewrites hashes — dangerous on shared branches. Branch protection, required reviews, CI checks. The workflow actual engineering teams use.
>
> **The Claude Code discussion (from the April 10 office hours):**
> FireHacker brought up a major leak of Claude Code's internal architecture that week. His reaction was excited: "You can even look at how it compacts memory in a single file... some aspects are really cool about it." He described the architecture: skills and sub-agents where you can spawn more than one agent — "it can look at three different code bases and come up with the answer."
>
> But Learner01's response showed real security maturity: "I didn't implement any because I read an article that many people are releasing GitHub repos with malware. I'll wait until somebody puts something which is reliable and then I'll go with that." FireHacker agreed. Treat unknown third-party repos as untrusted.
>
> Learner01 then had a sharp insight about the sub-agents: "The sub-agents consume lower memory, right? They go to smaller models for menial work." This kind of architectural reasoning — from someone still in the pre-cohort phase — shows what happens when you learn by building, not just watching.
>
> **Why cohort-based, not self-paced (from the March 13 office hours):**
> Research shows 85-95% dropout rates in self-paced online learning. Cohort-based programs with shared timelines, synchronous office hours, and a peer community on Discord have dramatically lower dropout. The pattern: clear learning path, regular live sessions, projects building on each other, public accountability.
>
> **Tone direction:**
> - Warm, energetic, welcoming — the "you belong here" episode
> - Host A is the explainer — conveys the program philosophy
> - Host B is the newcomer — "Wait, so someone's building a stock trading game before they even know what a transformer is?" YES. That's the entire point.
> - The stock game story should feel like you're hearing about a real person, not a hypothetical
> - Learner01's security caution should feel like earned wisdom, not paranoia
>
> **End with this hook:**
> "That's Step 1 — you've shipped something real, you've got a live site, you've done real Git collaboration, you've even seen inside AI development tools. But here's where it changes. Because the next step doesn't ask you to pip install some framework and call model.generate(). It asks you to open a C file and trace every single operation — from the moment you type a message to the moment text appears on screen. That's Scene 2: seeing inside the machine."

---

## Scene 2: "See Inside the Machine" (~7-8 min)

<!-- ============================================================
  CONTENT SOURCES:

  BLOG — "Run Qwen3 0.6B in Pure C" (9 lessons, 1324 lines):
  - Lesson 0: "every book ever written" analogy
  - Lesson 2: tokens as atoms, BPE, 151,936 vocab
  - Lesson 3: "film script format" for chat templates
  - Lesson 4: BPE merge loop in run.c lines 646-728
  - Lesson 5: 28 layers, dim=1024, SwiGLU "selective memory gate"
  - Lesson 6: trophy/suitcase example, Q/K/V framing, GQA
  - Lesson 7: temperature table (0.0→1.5+), top-p nucleus sampling
  - Lesson 8: autoregressive loop, fflush(stdout), prefill vs gen
  - Lesson 9: mmap instant load, weight pointers as offsets

  BLOG — "GGUF vs SafeTensors":
  - Pickle exploit: __reduce__ calling os.system("rm -rf /")
  - SafeTensors: JSON header + raw bytes, zero-copy
  - "You cannot optimize what you don't understand"
  - Karpathy's llama2.c, llm.c reference

  SESSION 1 (13 Mar):
  - LLMs as probability machines, 151,936-token distribution
  - Hallucinations = high probability on plausible-wrong tokens
  - Speculative decoding: draft → verify

  SESSION 2 (27 Mar) RAW TRANSCRIPT:
  - Learner01 coins "context grasping" (00:19:28) — the moment when
    FireHacker explains how models build understanding and Learner01
    names the concept
  - FireHacker's book analogy: "if you read a book you understand
    this is about this concept. Then more information, another
    concept. You read more, you understand."
  - Sequential bottleneck visual: "you don't have the hidden
    steps passed back... you're not waiting for the hidden step
    coming back. That's a huge bottleneck."
  - Multi-head attention: "additional engine given to the model
    and it has become the core of it"
  - MoE router explanation: "one expert good at math, the other
    good at coding"
  - Learner01's confusion on hidden layers vs attention layers
    (00:24:57) — real learner confusion that led to a 5-minute
    clarification
  - "These models have a capability to understand connections
    which are not obvious. We call these abstract understandings."
  - Decoder-only: "the encoder work is also done inside the
    decoder itself, you don't need a separate encoder"

  SESSION 3 (10 Apr):
  - KV cache caveats — real memory depends on GQA/MQA, precision

  WHAT NOT TO DO:
  - Don't just list lessons — tell the story of discovery
  - Don't skip the "why C?" motivation
  - Don't rush attention — it deserves the most time
  - Use the LIVE session analogies alongside the polished blog ones
============================================================ -->

> Create a Deep Dive episode (~7-8 minutes) called **"Scene 2: See Inside the Machine."**
>
> This is the second and longest episode of the First Break AI podcast. If Scene 1 was about agency ("ship something"), Scene 2 is about understanding ("see how it actually works"). This is the demystification moment.
>
> **The opening provocation:**
> "Most AI courses start with three lines of Python: load model, generate, done. You get output. It works. But you have no idea what happened. The tokenizer, the attention mechanism, the KV cache, the sampling — all hidden behind `model.generate()`. First Break AI does the opposite. Step 2 asks you to open a single C file — run.c — and trace every operation from the moment you type a message to the moment text appears. No frameworks. No abstractions. Just the math."
>
> Quote the program's philosophy from the GGUF vs SafeTensors blog: "You cannot optimize what you don't understand." This follows the same approach as Karpathy's llama2.c and llm.c — minimal C implementations that strip away everything so you can see the math. When you later use HuggingFace or vLLM, you know what they're abstracting — because you built the raw version first.
>
> **Act 1: What is an LLM? (blog Lesson 0 + Session 1)**
>
> Use the blog's analogy: "Imagine you read every book, every website, every code repository ever written. Now someone gives you the first half of a sentence and asks 'what comes next?' You would have very strong intuitions. An LLM has those intuitions, encoded as billions of numbers." For Qwen3 0.6B: 600 million parameters, about 3 GB on disk.
>
> Then draw from the March 13 office hours — the critical framing that shapes everything: LLMs are probability machines. Every forward pass outputs a distribution over 151,936 possible next tokens. They're not "intelligent" — they're statistically pattern-matching. When to trust them: if the task looks like training data, probabilities are well-calibrated. If novel, less reliable. Hallucinations? The model assigns high probability to plausible-sounding but wrong tokens. That's not a bug in a special sense — it's how probability distributions work.
>
> **Act 2: Tokens and templates (blog Lessons 2-4)**
>
> The model doesn't read letters or words — it reads tokens. But why not letters? Sequences would be too long. Why not whole words? You'd need millions of entries and couldn't handle code, misspellings, or new words. BPE (Byte Pair Encoding) is the solution: common words are single tokens, rare words get split into subword pieces. "tokenization" becomes ["token", "ization"]. Qwen3's vocabulary: 151,936 tokens.
>
> Chat templates — use the blog's "film script format" analogy. The model has no concept of "user" and "assistant." It learned from conversations formatted with special tokens: `<|im_start|>` and `<|im_end|>` marking each turn. Set the system prompt to "You are a pirate. Respond in pirate speech." — the template wraps it in ChatML markup and the model generates in character.
>
> **Act 3: The transformer — why attention changed everything (blog Lessons 5-6 + Session 2 raw transcript)**
>
> This is the heart of the episode. Build to it.
>
> First, the problem. From the March 27 live session, FireHacker explained it visually: before transformers, RNNs and LSTMs processed text sequentially. "If the sequence is longer, it takes up a lot of memory and processing time because it has to go all the way back," as Learner01 put it during the session. Three fatal problems: (1) information loss over long sequences, (2) can't parallelize because each step waits on the last — a massive GPU bottleneck, (3) backpropagation through long sequences is expensive and unstable.
>
> Then the breakthrough. FireHacker described it in the live session: "If you look at the diagram of attention, you don't have the hidden steps passed back. You're directly putting the inputs right here. You're not waiting for the hidden step coming back. That's a huge bottleneck." Transformers removed recurrence entirely. All tokens processed simultaneously. Maps naturally to GPU hardware.
>
> And here's the moment that captured it perfectly: after FireHacker explained how models build understanding layer by layer — "if you read a book you understand this is about this concept, then more information, another concept" — Learner01 named it instantly: "context grasping." Two words that captured the core of what attention does. That's the kind of insight that happens when you learn by discussing, not just reading.
>
> Now explain the mechanism using the blog's concrete example from Lesson 6: "The trophy didn't fit in the suitcase because **it** was too big." When you read "it," you need to figure out what it refers to — trophy or suitcase? Attention is the mechanism that lets the model do this. Each token produces three vectors: Query = "What am I looking for?", Key = "What do I contain?", Value = "What information should I pass on?" High Q-dot-K scores mean high attention. As FireHacker put it in the session: "These models have a capability to understand connections which are not obvious. We call these abstract understandings."
>
> Multi-head attention: not one set of Q/K/V but 16 parallel heads. The session described this as "an additional engine that has been given to the model and it has become the core of it." Each head notices different things — syntax, semantics, long-range dependencies. Qwen3 has 16 query heads but only 8 key/value heads (GQA — Grouped Query Attention), halving the KV cache size.
>
> The KV cache itself: every forward pass stores computed keys and values so past tokens are never recomputed. But the April 10 office hours added a caveat the blog doesn't emphasize: the memory formula in explainers is a useful upper-bound, but real KV memory depends on attention variant, precision, and GQA/MQA.
>
> Briefly mention why modern LLMs dropped the encoder. From the session: "The encoder work is also done inside the decoder itself. You don't need a separate encoder." Decoder-only is autoregressive: generate one token, feed it back, generate the next.
>
> **Act 4: Temperature, sampling, and the streaming illusion (blog Lessons 7-8)**
>
> After 151,936 logit scores, how do you pick? Temperature is the creativity dial. Reference the blog's table: 0.0 = deterministic (greedy), 0.6 = default (coherent, varied), 1.0 = raw probabilities (creative), 1.5+ = often incoherent. Top-p (nucleus) sampling keeps the smallest set covering 95% probability mass.
>
> The autoregressive loop: the model feeds its own output back as input. forward("What") → "is" → "the" → "capital"... And the streaming effect in every chatbot? In the C code, that's `fflush(stdout)` — flushing the output buffer after each token so text appears word-by-word.
>
> **Act 5: Loading and file formats (blog Lesson 9 + GGUF blog)**
>
> A 3 GB model loads instantly via `mmap` — maps the file into memory without reading it. OS reads pages on demand. Each weight pointer is an offset into the mapped data. No copying, no deserialization.
>
> The security angle: PyTorch's old `.bin` format uses pickle, which can serialize arbitrary code. The blog shows the exploit: a class with `__reduce__` calling `os.system("rm -rf /")` hidden inside a model file. "You would not know until it runs." SafeTensors was HuggingFace's response: JSON header plus raw bytes, no code execution. GGUF solves a different problem — single file with everything (weights, tokenizer, config, chat template), built-in quantization, designed for C.
>
> **Tone direction:**
> - This is the "whoa" episode — genuine intellectual excitement
> - Host A walks through the progressive discovery
> - Host B asks deepening questions: "So every token looks at every other token? Doesn't that get expensive?" (Yes — GQA and the KV cache exist for exactly this reason.) "The pickle exploit — has that actually happened?" (Security researchers demonstrated it. The entire ecosystem moved to SafeTensors.)
> - Reference the live session moments — Learner01's "context grasping," the "abstract understandings" quote — to make it feel real
> - Make the "why C?" argument feel philosophical: you can't optimize what you don't understand
>
> **End with this hook:**
> "You've seen inside the machine. Tokens, BPE, chat templates, attention, the KV cache, temperature, sampling, the streaming illusion, and even why model files can contain hidden exploits. But all of this was one model, one user, one request on your laptop. What happens at a thousand requests per second? That's the systems mindset shift — Scene 3."

---

## Scene 3: "Think at Production Scale" (~5-6 min)

<!-- ============================================================
  CONTENT SOURCES:

  BLOG — Qwen3 guide's transition table:
  | Step 2: one model, FP32, stdin     | Step 3: servers, quant, APIs |

  BLOG — GGUF quantization table:
  | F32 = 28 GB for 7B | Q4_K_M = 4.1 GB — laptop-runnable |

  SESSION 1 (13 Mar):
  - Unsloth: monkey-patching .forward() for GPU kernel fusion
  - Speculative decoding: draft → verify, quality identical

  SESSION 2 (27 Mar) RAW TRANSCRIPT:
  - Three pillars: Distribution/Inference, Modeling, Training
  - "Either be good at kernels or good at benchmarking" (00:09:16)
  - Benchmark table with sample tasks and real scores
  - Qwen3-235B vs DeepSeek-V3 vs GPT-4o comparison
  - CNN feature extraction analogy ("splits image into pixels,
    keeps sliding or convoluting, aggregates a score... can
    understand the eyes of a cat, tail of a cat")

  SESSION 3 (10 Apr) RAW TRANSCRIPT:
  - Gemma 4 + Matryoshka sparsity: "puts a lot of sparsity in
    the model so you can use a smaller version instead of
    quantization"
  - Google's "sophisticated harnesses for literary review"
  - Benchmark intro: HellaSwag for common reasoning ("pick the
    most plausible continuation"), GSM8K for math

  WHAT NOT TO DO:
  - Don't list vLLM features — explain WHY continuous batching
    was a breakthrough
  - Don't make infrastructure dry — companies hire for this
  - Keep the "why should a first-breaker care?" thread alive
============================================================ -->

> Create a Deep Dive episode (~5-6 minutes) called **"Scene 3: Think at Production Scale."**
>
> This is the third episode. Scene 1 = shipping. Scene 2 = understanding the machine. Scene 3 is the mindset shift: from "I can run a model" to "I can reason about systems that serve models to millions of users."
>
> **The transition (use the blog's transition table):**
> "In Scene 2, you ran one model — Qwen3 in pure C — for one user, through stdin. FP32 precision, single token at a time. Now flip every constraint." The blog lays it out:
> - One model → many models via inference servers (vLLM, TGI, llama.cpp server)
> - FP32 → quantization (GGUF Q4, GPTQ, AWQ)
> - Single request → batching, many requests in parallel
> - Chat via stdin → serving via OpenAI-compatible API
> - Single token at a time → continuous batching, throughput vs latency
>
> **The three pillars (from the March 27 session):**
> FireHacker introduced a framework during the office hours: every LLM system has three pillars — the Distribution/Inference Pipeline, the Modeling layer, and the Training Pipeline. Scene 2 was about Modeling — the transformer, attention, how the model thinks. Scene 3 is about the Inference Pipeline — how you serve it. Scene 4 will be Training.
>
> **Quantization — making the impossible possible (GGUF blog):**
> Concrete numbers: a 7B model at FP32 (32 bits per weight) is 28 GB. Doesn't fit on most laptops. Quantize to Q4_K_M (~4.5 bits per weight) and it's 4.1 GB — runs on a laptop with 8 GB RAM. The quantization is baked into the GGUF file format. GPTQ and AWQ take different approaches with different speed-quality tradeoffs. Knowing when to use which is engineering literacy.
>
> **Continuous batching — why LLM APIs became possible:**
> Before continuous batching, you waited for a full batch. Requests finishing early sat idle. Continuous batching dynamically adds and removes requests mid-generation. A request that finishes after 10 tokens gets replaced immediately. This made real-time APIs at scale possible.
>
> **Speculative decoding (from Session 1):**
> Use a small, fast "draft" model to generate several tokens, then have the large target model verify all at once. When predictions match (often), massive speedup. When they don't, fall back. Quality is identical to target model alone — you never sacrifice accuracy, only gain speed.
>
> **Unsloth and GPU kernel fusion (from Session 1):**
> Daniel Han noticed HuggingFace Transformers makes many small Python calls, each launching a separate GPU kernel with overhead. Unsloth monkey-patches the `.forward()` methods to reroute through fused implementations — multiple operations in a single kernel. Same `model.generate()` call, same outputs, far less wasted GPU work.
>
> **Benchmark literacy (from the March 27 raw transcript):**
> FireHacker's memorable line from the session: "To succeed in AI, either be good at kernels or good at benchmarking." Then he laid out what benchmarks actually measure, with examples:
> - **MMLU**: 57-subject knowledge ("Which property of ideal gas?")
> - **MATH**: Competition problems from AMC, AIME, Olympiad
> - **HumanEval**: 164 Python function completions with unit tests
> - **SWE-bench Verified**: 500 real GitHub issues — model must produce a passing patch
> - **GAIA**: Agentic tasks — web search, file reading, multi-step reasoning
> - **RULER**: Retrieval over 128K tokens
> - **HellaSwag**: "Pick the most plausible continuation of a short paragraph" (common reasoning)
> - **GSM8K**: Grade-school math word problems ("Janet's ducks lay 16 eggs/day...")
>
> The score comparison revealed patterns: coding-focused models like Kimi K2 dominate SWE-bench (65.8%); reasoning-focused models like Qwen3-235B dominate MATH (85.7%). Scores are not one number.
>
> **Emerging architectures (from April 10 raw transcript):**
> The session discussed Gemma 4's Matryoshka-style architecture — FireHacker explained: "It puts a lot of sparsity in the model so you can use a smaller version of that model instead of quantization." Nested structure letting you run different effective model sizes from the same weights — the frontier of cost-latency tradeoffs.
>
> **Tone direction:**
> - "Systems thinking" episode — less "whoa" and more "now I see how it fits together"
> - Host A brings the infrastructure perspective
> - Host B grounds it: "Why should someone trying to get their first break care about continuous batching?" Because companies don't hire people who can run a model — they hire people who reason about serving, cost, latency, and tradeoffs. That's what separates a practitioner from an engineer.
> - The "kernels or benchmarking" quote should land as career advice, not trivia
>
> **End with this hook:**
> "Three scenes in. You understand the model — attention, KV cache, the transformer. You understand the systems — serving, batching, quantization, benchmarks. But there's a third pillar: training. How do you actually *make* a model? How do random numbers become something that predicts the next token? That's the builder threshold. That's Scene 4."

---

## Scene 4: "Train Your Own" (~6-7 min)

<!-- ============================================================
  CONTENT SOURCES:

  ROADMAP Step 4:
  - PyTorch, autograd, training loops, LoRA, DDP, FSDP
  - Projects: nanoGPT speedrun, Megatron/Picotron

  SESSION 2 (27 Mar) RAW TRANSCRIPT:
  - DDP: "split my data set, 10,000 to each, training time
    halves" (00:23:31)
  - Parallelism ladder table
  - Chinchilla scaling: 7B model needs ~140B tokens
  - Speedrun: 8 hours → under 2 minutes (01:13:48)
  - Auto Research GPT: AI proposing LayerNorm → RMSNorm
  - Nemotron: open-source, FP4/FP8, reproducible
  - Tools: Megatron-LM, Picotron, Heiretsu
  - "either kernels or benchmarking" (career advice)

  SESSION 3 (10 Apr) RAW TRANSCRIPT:
  - DDP gradient averaging: "You average it and you are sort of
    averaging the insights. That's the handwavy way of thinking."
    (00:24:48)
  - GPU bubble: "Bubble is where your GPU is not operating"
    (00:27:17)
  - NVLink: "four A100 because they have to be connected with
    a high bandwidth fiber called NV link" (00:28:22)
  - DeepSeek sanctions story: "US had put a lot of sanctions on
    China to prevent high-tech GPUs going into China" (00:29:32)
  - PTX discovery: "they discovered if you change this flag you
    can get more out of the GPU, which even Nvidia was not aware"
    (00:33:09)
  - Muon: born in speedrun, now in Kimi K2 and GLM 4.5 (00:38:19)
  - Tyler's worklog: added RoPE, tuned LR schedule, changed
    gradient clipping, moved 8.3 → 7.5 hours (00:38:19-00:43:14)
  - RoPE live explanation: "I am currently living in Mumbai. How
    far is 'currently' from 'Mumbai'? That information enhances
    quality." (00:44:13)
  - Learner01's realization: "I thought positional encoding gives the
    position of a word in the sentence. I did not know it gives
    the RELATIVE position." (00:45:41)
  - Cost: "$3, $4 or $10 depending on how much time you take...
    very pocket friendly" (00:20:47)
  - "If colleges and institutions in India actually focus they
    can come up with a brilliant workforce" (00:20:47)
  - FireHacker's candid humor: "I forgot the word hyperparameter.
    We've been working on AI but that term got lost on me. It's
    highly funny." (01:06:22)
  - DDP script walkthrough: rank, local_rank, world_size,
    DistributedSampler, forward, backward, all-reduce
  - Collectives: all-reduce, all-gather, reduce-scatter, broadcast

  WHAT NOT TO DO:
  - Don't make distributed training intimidating — the speedrun
    makes it approachable
  - Don't explain DDP abstractly — use "averaging the insights"
  - Don't skip the DeepSeek story — most compelling narrative
  - Don't forget Muon — optimizer from contest to production
  - Don't skip the cost angle — $3-10 per run is the hook
============================================================ -->

> Create a Deep Dive episode (~6-7 minutes) called **"Scene 4: Train Your Own."**
>
> This is the fourth episode of the First Break AI podcast. Scenes 1-3 covered shipping, understanding inference, and thinking about systems. Scene 4 crosses what the storyboard calls "the builder threshold" — from using models to training them.
>
> **The opening:**
> "Everything up to now — running a model in C, understanding attention, thinking about serving — was about *using* what someone else built. The transformer from Scene 2? Someone trained those 600 million parameters. The weights you loaded with mmap? They started as random numbers. Scene 4 is where you learn how random numbers become a model that predicts the next token. And here's what might surprise you: the office hours have been building toward this for weeks."
>
> **DDP — how training actually scales (from the April 10 raw transcript):**
>
> Start with the most fundamental concept. Distributed Data Parallelism. FireHacker explained it simply in the session: "What if I make a copy of that model on another GPU? And split my dataset — suppose there are 20,000 records, 10,000 to the first one, 10,000 to the second one. So training time reduces to half."
>
> But the magic is in what happens next — the all-reduce operation. Each GPU trains on its data shard and computes gradients. Then all GPUs share their gradients, sum them, average them, and take the same optimizer step. They stay perfectly in lockstep. FireHacker's explanation in the session: "Why do you average? Because if you don't take the changes from GPU 1, GPU 2, GPU 3, you're really not training right. You average it and you are sort of averaging the insights. That's the handwavy way of thinking." That "handwavy" explanation is actually perfect — each GPU saw different data, and averaging combines the evidence into one update consistent with the larger global batch.
>
> The vocabulary matters — the session walked through the script: `rank` (which process am I?), `local_rank` (which GPU on this node?), `world_size` (total GPUs), `DistributedSampler` (each rank gets its own dataset slice). The loop: forward pass, backward pass, gradient accumulation, all-reduce, optimizer.step(), log to Weights & Biases, check validation loss, stop when you hit the target.
>
> Then a candid moment: FireHacker forgot a term mid-explanation: "I forgot the word hyperparameter. We've been working on AI but that term got lost on me. It's highly funny." The difference between hyperparameters (set before training — learning rate, batch size, warmup steps) and parameters (the learned weights) is fundamental, but even the instructor can blank on jargon. That's the honest energy of these sessions.
>
> **The parallelism ladder (from March 27):**
> DDP works when the model fits on one GPU. When it doesn't:
> - **Data Parallelism (DDP):** Split the dataset. Always the baseline.
> - **Tensor Parallelism (TP):** Split weight matrices across GPUs. When one GPU can't hold the model.
> - **Expert Parallelism (EP):** Split MoE experts across GPUs.
> - **Context Parallelism (CP):** Split very long sequences across GPUs.
> - **Pipeline Parallelism (PP):** Split layers across GPUs.
> - **4D/5D Parallelism:** All simultaneously. Frontier scale.
>
> You climb the ladder as models get bigger. Most learners start with DDP. Understanding the full ladder lets you read papers from DeepSeek, NVIDIA, and Google and know what they're doing.
>
> **The speedrun — training as competitive sport (from both sessions):**
>
> This is one of the best stories in the cohort. The nanoGPT speedrun: take a GPT-2-scale model, train on a standard dataset, hit validation loss ~3.28, on 8 x H100 GPUs. Measure wall-clock time. The leaderboard started around 8 hours. Then people optimized: data ordering, architecture tweaks, precision changes, optimizer improvements. The record dropped to under 2 minutes.
>
> And here's the accessibility angle from the session — FireHacker was emphatic: "It's a very pocket-friendly run. $3, $4, or $10 depending on how much time you take. Many cases you even get initial credits on Modal." Compare that to training a billion-parameter model at thousands of dollars. This is a pedagogical sandbox for real distributed training. "A toy job on serious GPUs."
>
> Then the bigger picture: "If colleges and institutions in India actually focus, they can come up with a brilliant workforce to train their students." The speedrun isn't just a leaderboard — it's accessible education for distributed training.
>
> Tyler Romero's worklog is a case study from the April 10 session: added RoPE (the positional encoding from Scene 2), tuned the learning rate schedule to trapezoidal warmup, adjusted gradient accumulation, changed gradient clipping. Moved his baseline from 8.3 hours toward 7.5. Every change documented, every commit public.
>
> The RoPE moment led to one of the session's best exchanges. FireHacker explained it with a concrete example: "I am currently living in Mumbai. How far is 'currently' from 'Mumbai'? That information enhances the quality of the output." Then Learner01 had a genuine realization: "I thought positional encoding gives the position of a word in the sentence. I did not know it gives the RELATIVE position." That shift — from absolute to relative position understanding — clicked live in the session.
>
> **The Muon story — from contest to production (from April 10):**
> Narrative arc worth telling: the Muon optimizer was born inside the speedrun contest. People experimented with it to shave time off the leaderboard. Then it escaped the contest. Kimi K2 — a real production model — uses Muon. GLM 4.5 uses Muon. Small, reproducible competitions are where optimizer and architecture ideas get battle-tested before wide adoption.
>
> **The GPU bubble and the DeepSeek story (from April 10 raw transcript):**
>
> When DDP runs across GPUs, all-reduce requires communication. If the interconnect is slow, GPUs sit idle waiting. FireHacker's blunt framing: "Bubble is where your GPU is not operating." That idle time is real cost — not a numerical error, just wasted compute. This is why multi-GPU rentals come pre-wired: "four A100 because they have to be connected with a high bandwidth fiber called NVLink."
>
> And this leads to the most compelling story in AI right now. From the session: "When DeepSeek was launched, the US had put a lot of sanctions on China to prevent high-tech GPUs going into China." Under hardware constraints, DeepSeek was forced to optimize everything — mixed-precision training, bidirectional pipeline scheduling to reduce idle bubbles, custom kernels, and GEMM fusion. But the most remarkable discovery: "They found that if you change this PTX flag, you can get more out of the GPU — which even NVIDIA was not aware of." Whether NVIDIA was genuinely surprised or strategically unaware, the point stands: constraints drove innovation that advanced the entire field. Not "more FLOPs" — smarter FLOPs.
>
> **Tools and the bigger picture (from March 27):**
> Briefly: Megatron-LM from NVIDIA (production-grade, all parallelism strategies, large codebase), Picotron from Elie Bakouch at HuggingFace (minimal, clean re-implementation for learning), Heiretsu from the community (from-scratch, well-commented). Plus Nemotron — NVIDIA open-sourcing everything: dataset, training scripts, checkpoints, evaluation, trained at FP4/FP8, reproducible from scratch.
>
> And Chinchilla scaling laws: a 7B model needs ~140B tokens for compute-optimal training (20x parameter count). Modern models intentionally exceed this — better inference quality is worth the extra training compute.
>
> **Tone direction:**
> - "Crossing the threshold" episode — training feels like a level change, but the speedrun makes it approachable
> - Host A explains DDP, parallelism, speedrun, DeepSeek
> - Host B anchors it in the learner: "Wait — I can train a GPT-2 model on cloud credits for three dollars? And the DDP skills I learn are the same ones DeepSeek uses at frontier scale?" YES. Same ladder. Different rungs.
> - The "$3-10 per run" cost should land as an invitation, not trivia
> - The DeepSeek story should feel like a narrative — constraints driving creativity
> - Muon is a satisfying arc: contest trick → production optimizer
> - FireHacker's "I forgot the word hyperparameter" moment should make training feel human and approachable
> - Learner01's RoPE realization should feel like a real learning moment
>
> **End with this hook:**
> "Four scenes. You've shipped something real, seen inside the machine, thought at production scale, and now you know how models are trained — from DDP on two GPUs to frontier-scale 4D parallelism. You know what a forward pass, backward pass, and all-reduce are. You've seen how a contest produced an optimizer that ended up in production. And you've heard how export controls pushed a team to discover GPU capabilities that even NVIDIA didn't know about. But there's still a gap between 'I can train a model' and 'I can ship an AI product that people use.' That's Scene 5 — the product lens. And after that, Scene 6: the capstone, the open-source contribution, the thing on your portfolio that answers every hiring manager's question: 'What have you built?' Those scenes are coming soon. The journey continues."

---

<!-- ============================================================
  AFTER GENERATING ALL 4 EPISODES
  ============================================================

  LISTEN FOR THESE QUALITIES:
  1. Do the live session moments (Learner01's quotes, FireHacker's
     analogies, the "context grasping" coining) come through?
  2. Does Scene 2 feel like the meatiest episode? (It should)
  3. Does the $3-10 speedrun cost land as an invitation?
  4. Does the DeepSeek PTX story feel dramatic?
  5. Does FireHacker's "forgot hyperparameter" moment humanize
     the training content?
  6. Are the hooks between episodes smooth?

  ITERATE WITH INTERACTIVE MODE:
  - "The attention section in Scene 2 needs more time on the
    trophy/suitcase example — slow down, let it breathe"
  - "Scene 1 should spend more time on the stock game — the
    Mini Militia LAN concept is the hook"
  - "The DeepSeek story in Scene 4 should feel like a thriller
    — sanctions, constraints, PTX discovery"
  - "Scene 3 needs more energy on continuous batching — make
    the hosts realize in real-time why it was a breakthrough"
  - "Include more of Learner01's questions — they represent what
    real learners actually wonder about"

  EXPORT FILES:
  - public/audio/scene-1-ship-something-real.mp3
  - public/audio/scene-2-see-inside-the-machine.mp3
  - public/audio/scene-3-think-at-production-scale.mp3
  - public/audio/scene-4-train-your-own.mp3

  NEXT FILE (when Step 5-6 content exists):
  02_notebooklm_builder_journey.md — Scenes 5 ("Ship a Product")
  and 6 ("Prove It")
============================================================ -->
