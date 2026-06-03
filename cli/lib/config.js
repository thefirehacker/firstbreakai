import { homedir } from 'node:os';
import { join } from 'node:path';

export const COHORT_NAME = 'First Break AI';
export const COHORT_URL = 'https://cohort.bubblnet.com';
export const MCP_WORKER_URL = 'https://fba-mcp.throbbing-thunder-4d33.workers.dev';
export const DISCORD_INVITE = 'https://discord.gg/hRPese4H3F';
export const REPO_URL = 'https://github.com/thefirehacker/firstbreakai';

export const PROGRESS_FILE = join(homedir(), '.firstbreakai.json');

export const PAGES = {
  home:       `${COHORT_URL}/`,
  roadmap:    `${COHORT_URL}/roadmap`,
  checklist:  `${COHORT_URL}/checklist`,
  lessons:    `${COHORT_URL}/lessons/`,
  setup:      `${COHORT_URL}/setup`,
  blog:       `${COHORT_URL}/blog/`,
  discord:    DISCORD_INVITE,
  repo:       REPO_URL,
  office:     `${COHORT_URL}/office-hours/`,
};

export const STEPS = [
  {
    id: 0,
    title: 'Join Discord & create accounts',
    url: `${COHORT_URL}/checklist`,
    lessons: [],
  },
  {
    id: 1,
    title: 'First use of AI for coding — Quarto blog + GitHub Pages',
    url: `${COHORT_URL}/roadmap#step-1`,
    lessons: [
      { id: '0', title: 'Welcome', url: `${COHORT_URL}/lessons/lesson-0-welcome` },
    ],
  },
  {
    id: 2,
    title: 'Run a model locally — Qwen3 0.6B, pure C inference',
    url: `${COHORT_URL}/roadmap#step-2`,
    lessons: [
      { id: '1', title: 'HuggingFace Beyond Upload', url: `${COHORT_URL}/lessons/lesson-1-huggingface-beyond-upload` },
      { id: '1b', title: 'Qwen3 Fundamentals', url: `${COHORT_URL}/lessons/lesson-1b-qwen3-fundamentals` },
    ],
  },
  {
    id: 3,
    title: 'Inference deep dive (coming soon)',
    url: `${COHORT_URL}/roadmap#step-3`,
    lessons: [],
  },
  {
    id: 4,
    title: 'Training fundamentals (coming soon)',
    url: `${COHORT_URL}/roadmap#step-4`,
    lessons: [],
  },
  {
    id: 5,
    title: 'Build an AI product (coming soon)',
    url: `${COHORT_URL}/roadmap#step-5`,
    lessons: [],
  },
  {
    id: 6,
    title: 'Capstone / open-source contribution (coming soon)',
    url: `${COHORT_URL}/roadmap#step-6`,
    lessons: [],
  },
];
