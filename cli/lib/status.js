import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import chalk from 'chalk';
import { PROGRESS_FILE, STEPS, MCP_WORKER_URL } from './config.js';

export function loadProgress() {
  try {
    return JSON.parse(readFileSync(PROGRESS_FILE, 'utf-8'));
  } catch {
    return { version: 1, learner_id: null, access_token: null, steps: {} };
  }
}

export function saveProgress(progress) {
  mkdirSync(dirname(PROGRESS_FILE), { recursive: true });
  writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2) + '\n');
}

async function fetchServerProgress(progress) {
  if (!progress.access_token) return null;
  try {
    const res = await fetch(`${MCP_WORKER_URL}/auth/me`, {
      headers: { 'Authorization': `Bearer ${progress.access_token}` },
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) return await res.json();
  } catch { /* offline or error — ignore */ }
  return null;
}

export default async function status() {
  const progress = loadProgress();

  console.log(chalk.bold('\n  First Break AI — Your Progress\n'));

  let synced = false;
  if (progress.access_token) {
    const serverData = await fetchServerProgress(progress);
    if (serverData) {
      synced = true;
      if (serverData.discord_username) {
        console.log(chalk.dim(`  Logged in as: ${serverData.discord_username} (${progress.learner_id})`));
      } else {
        console.log(chalk.dim(`  Logged in as: ${progress.learner_id}`));
      }
      if (typeof serverData.current_step === 'number' && serverData.current_step > 0) {
        for (let i = 0; i < serverData.current_step; i++) {
          if (!progress.steps[i]?.done) {
            progress.steps[i] = { done: true, at: new Date().toISOString(), synced: true };
          }
        }
        saveProgress(progress);
      }
      console.log(chalk.dim('  (synced with server)\n'));
    } else {
      if (progress.learner_id) console.log(chalk.dim(`  Logged in as: ${progress.learner_id} (offline)\n`));
    }
  } else {
    console.log(chalk.dim('  Not logged in. Run "firstbreakai login" to sync across devices.\n'));
  }

  for (const step of STEPS) {
    const entry = progress.steps[step.id];
    const done = entry?.done === true;
    const icon = done ? chalk.green('✓') : chalk.dim('○');
    const title = done ? chalk.strikethrough(step.title) : step.title;
    const date = done && entry.at
      ? chalk.dim(` (${new Date(entry.at).toLocaleDateString()})`)
      : '';

    console.log(`  ${icon} Step ${step.id}: ${title}${date}`);

    for (const lesson of step.lessons) {
      console.log(`      ${chalk.dim('·')} ${chalk.dim(lesson.title)}`);
    }
  }

  const completed = STEPS.filter((s) => progress.steps[s.id]?.done).length;
  console.log(chalk.dim(`\n  ${completed}/${STEPS.length} steps completed`));
  console.log(chalk.dim('  Roadmap: https://cohort.bubblnet.com/roadmap\n'));
}
