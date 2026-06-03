import openUrl from 'open';
import chalk from 'chalk';
import { PAGES, COHORT_URL } from './config.js';

export default async function open(args) {
  const page = args[0]?.toLowerCase();

  if (!page) {
    const pageList = Object.keys(PAGES).join(', ');
    console.log(`\nUsage: firstbreakai open <page>\n\nPages: ${pageList}\n`);
    return;
  }

  const url = PAGES[page];
  if (!url) {
    console.error(`Unknown page: "${page}"`);
    console.error(`Available pages: ${Object.keys(PAGES).join(', ')}`);
    process.exit(1);
  }

  console.log(chalk.dim(`  Opening ${url} ...`));
  await openUrl(url);
}
