import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import chalk from 'chalk';
import { PROGRESS_FILE } from './config.js';

const require = createRequire(import.meta.url);
const pkg = require('../package.json');
const ONE_DAY = 86_400_000;

export async function checkForUpdate() {
  try {
    const progress = JSON.parse(readFileSync(PROGRESS_FILE, 'utf-8'));
    if (progress._lastUpdateCheck && Date.now() - progress._lastUpdateCheck < ONE_DAY) {
      if (progress._latestVersion && progress._latestVersion !== pkg.version) {
        printNotice(progress._latestVersion);
      }
      return;
    }
  } catch { /* no progress file yet */ }

  try {
    const res = await fetch('https://registry.npmjs.org/@aiedx/firstbreakai/latest', {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return;
    const data = await res.json();
    const latest = data.version;

    try {
      const progress = JSON.parse(readFileSync(PROGRESS_FILE, 'utf-8'));
      progress._lastUpdateCheck = Date.now();
      progress._latestVersion = latest;
      writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2) + '\n');
    } catch { /* can't cache, that's fine */ }

    if (latest && latest !== pkg.version) {
      printNotice(latest);
    }
  } catch { /* network error — skip silently */ }
}

function printNotice(latest) {
  console.log(chalk.yellow(`\n  Update available: ${pkg.version} → ${latest}`));
  console.log(chalk.dim('  Run: npm install -g @aiedx/firstbreakai'));
}
