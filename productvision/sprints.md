# First Break AI — Product Vision & Sprints

## Sprint 1: Foundation — First Break AI

**Status:** Shipped

The core cohort platform. A Quarto-based website with everything a learner needs to go from zero to their first break in AI.

**What was built:**
- 6-step roadmap (first commit → capstone)
- Blog with step-by-step guides (Qwen3 in pure C, GGUF vs SafeTensors)
- Project Watch (Unsloth monkey patching, Auto Research GPT)
- Office Hours with expanded write-ups from live sessions
- Warm paper aesthetic, Mermaid diagrams with fullscreen zoom, GitHub repo cards
- Google Analytics, OG/Twitter cards, JSON-LD schema, Fetchlens analytics
- GitHub Actions → GitHub Pages deployment

**Core belief:** Free, open, community-driven. No prerequisites, no gatekeeping. Learn by doing, build in the open.

---

## Sprint 2: The Journey — Narrative-Driven Storyboard

**Status:** In progress

The first adult AI learning cohort with a scroll-driven, AI-podcast-powered storyboard. Turns the 6-step roadmap into an immersive hero's journey — the site's centerpiece feature.

**What was built:**
- `/journey/` storyboard page with 6 full-viewport scenes mirroring roadmap steps
- Scroll-driven audio playback (IntersectionObserver triggers play/pause as scenes enter viewport)
- Sticky mini-player with seek bar, play/pause, scene navigation
- Side progress rail with numbered dots for jumping between scenes
- Tap-to-begin gate for mobile autoplay unlock
- Homepage redesign: "Start the Journey" CTA, 6-scene preview strip
- AI content disclaimers (gate screen, footer, homepage, office hours index)
- Updated OG metadata and JSON-LD (PodcastSeries schema)
- NotebookLM podcast generation pipeline (manual, from cohort content)

**Core insight:** Nobody in adult AI education does narrative-driven learning. CodeFly gets 94% completion rates with story-based K-12 coding. Khanmigo (AI tutor) failed because passive tutoring doesn't motivate — narrative pull does.

**Defensibility:**
- Content moat: every office hours session generates new storyboard material
- Format moat: scroll-driven narrative requires both content depth and production quality
- Community moat: "I'm on Scene 3" is more visceral than "I'm on Step 3"
- AI-native moat: episodes regenerate as the roadmap evolves

---

## Sprint 3: Interactive Transcript Reader + Video Sync

**Status:** Planned

Inspired by Obsidian Reader's interactive transcript feature (kepano, Apr 2026). Each journey scene evolves from "narrative text + audio" into a **synced video + interactive transcript** experience.

**The concept:**
- Split-panel layout per scene: video on one side, timestamped transcript on the other
- Transcript auto-scrolls and highlights the current paragraph as the video plays
- Click any paragraph to scrub to that timestamp in the video
- Works with YouTube embeds (iframe API) or self-hosted video
- Transcript text is searchable, highlightable, and linkable

**Why this is a 10x upgrade:**
- Learners who prefer reading scan the transcript. Learners who prefer watching get the video. Both interact with the same content simultaneously.
- The transcript already exists — it's the input to the AI podcast generation. Zero extra content work.
- Turns passive watching into active exploration (click, scrub, jump, re-read)
- Enables "notebook-style" learning: pause, highlight a concept, jump back

**Technical approach:**
- YouTube iframe API (`onStateChange`, `getCurrentTime()`) for playback state
- Timestamped transcript paragraphs with `data-start` and `data-end` attributes
- `timeupdate` event listener maps current time → active paragraph → toggle `.active` class
- Click handler on paragraphs calls `player.seekTo(timestamp)`
- CSS: sticky video panel, scrollable transcript, highlighted active line
- ~100 lines of JS on top of existing journey-player.html

**Additional Sprint 3 ideas to explore:**
- HeyGen AI avatar video clips per scene (short 2-3 min narrator segments)
- Full-length YouTube episodes with chapter markers matching scenes
- ElevenLabs-polished audio with emotion/pacing control (Audio Tags)
- Curriculum-aware AI chat agent embedded in the journey page (knows which scene you're on, scaffolds help based on introduced concepts)
- `localStorage`-based progress tracking ("resume where you left off")
- Keyboard shortcuts for transcript navigation (J/K to jump paragraphs, Space to play/pause)

---

## The Arc

| Sprint | What changes | User experience |
|--------|-------------|-----------------|
| 1 | Static content site | Read guides, follow roadmap |
| 2 | Narrative storyboard + AI podcast | Scroll through an immersive journey with audio |
| 3 | Interactive video + synced transcript | Watch, read, scrub, highlight — active learning |

Each sprint layers on the previous. The roadmap content stays the same; the delivery format gets progressively richer and more interactive.
