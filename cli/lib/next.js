import openUrl from 'open';
import chalk from 'chalk';
import { STEPS } from './config.js';
import { loadProgress } from './status.js';

export default async function next() {
  const progress = loadProgress();

  const nextStep = STEPS.find((s) => !progress.steps[s.id]?.done);

  if (!nextStep) {
    console.log(chalk.green('\n  All steps complete! Nothing left to do.'));
    console.log(chalk.dim('  Roadmap: https://cohort.bubblnet.com/roadmap\n'));
    return;
  }

  console.log(chalk.bold(`\n  Next: Step ${nextStep.id} — ${nextStep.title}`));
  console.log(chalk.dim(`  Opening ${nextStep.url} ...\n`));
  await openUrl(nextStep.url);
}
