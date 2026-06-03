import { readFileSync } from 'node:fs';
import chalk from 'chalk';
import { PROGRESS_FILE, STEPS } from './config.js';

export function loadProgress() {
  try {
    return JSON.parse(readFileSync(PROGRESS_FILE, 'utf-8'));
  } catch {
    return { version: 1, learner_id: null, access_token: null, steps: {} };
  }
}

export default async function status() {
  const progress = loadProgress();

  console.log(chalk.bold('\n  First Break AI — Your Progress\n'));

  if (progress.learner_id) {
    console.log(chalk.dim(`  Logged in as: ${progress.learner_id}\n`));
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
