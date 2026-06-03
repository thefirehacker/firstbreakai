import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import chalk from 'chalk';
import { PROGRESS_FILE, STEPS } from './config.js';
import { loadProgress } from './status.js';

export default async function done(args) {
  const stepId = parseInt(args[0], 10);

  if (isNaN(stepId) || !STEPS.find((s) => s.id === stepId)) {
    console.error(`Usage: firstbreakai done <step>\nValid steps: ${STEPS.map((s) => s.id).join(', ')}`);
    process.exit(1);
  }

  const step = STEPS.find((s) => s.id === stepId);
  const progress = loadProgress();

  if (progress.steps[stepId]?.done) {
    console.log(chalk.yellow(`\n  Step ${stepId} is already marked as done.\n`));
    return;
  }

  progress.steps[stepId] = { done: true, at: new Date().toISOString() };

  mkdirSync(dirname(PROGRESS_FILE), { recursive: true });
  writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2) + '\n');

  console.log(chalk.green(`\n  ✓ Step ${stepId}: ${step.title} — marked complete!`));

  const nextStep = STEPS.find((s) => !progress.steps[s.id]?.done);
  if (nextStep) {
    console.log(chalk.dim(`  Next up: Step ${nextStep.id} — ${nextStep.title}`));
    console.log(chalk.dim(`  ${nextStep.url}\n`));
  } else {
    console.log(chalk.bold('  🎉 All steps complete — congratulations!\n'));
  }
}
