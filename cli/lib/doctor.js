import { execSync } from 'node:child_process';
import chalk from 'chalk';

const checks = [
  {
    name: 'Git',
    cmd: 'git --version',
    extract: (out) => out.match(/git version ([\d.]+)/)?.[1],
    hint: 'Install Git: https://git-scm.com/downloads',
  },
  {
    name: 'Python',
    cmd: 'python3 --version',
    extract: (out) => out.match(/Python ([\d.]+)/)?.[1],
    hint: 'Install Python 3: https://python.org/downloads',
  },
  {
    name: 'Quarto',
    cmd: 'quarto --version',
    extract: (out) => out.trim(),
    hint: 'Install Quarto: https://quarto.org/docs/get-started/',
  },
  {
    name: 'HuggingFace CLI',
    cmd: 'huggingface-cli version',
    extract: (out) => out.trim(),
    hint: 'Install: pip install huggingface_hub[cli]',
  },
  {
    name: 'Node.js',
    cmd: 'node --version',
    extract: (out) => out.trim().replace(/^v/, ''),
    hint: 'Install Node.js >= 18: https://nodejs.org',
  },
];

function tryRun(cmd) {
  try {
    return execSync(cmd, { encoding: 'utf-8', timeout: 10_000, stdio: ['pipe', 'pipe', 'pipe'] });
  } catch {
    return null;
  }
}

function detectEditor() {
  if (process.env.TERM_PROGRAM === 'vscode' || process.env.VSCODE_PID) {
    return { name: 'VS Code / Cursor', detected: true };
  }
  if (tryRun('cursor --version')) return { name: 'Cursor', detected: true };
  if (tryRun('code --version')) return { name: 'VS Code', detected: true };
  return { name: 'Cursor / VS Code', detected: false };
}

export default async function doctor() {
  console.log(chalk.bold('\n  First Break AI — Environment Check\n'));

  let allPassed = true;

  for (const check of checks) {
    const output = tryRun(check.cmd);
    if (output) {
      const version = check.extract(output) || 'found';
      console.log(`  ${chalk.green('✓')} ${check.name} ${chalk.dim(`v${version}`)}`);
    } else {
      allPassed = false;
      console.log(`  ${chalk.red('✗')} ${check.name} — ${chalk.yellow('not found')}`);
      console.log(`    ${chalk.dim(check.hint)}`);
    }
  }

  const editor = detectEditor();
  if (editor.detected) {
    console.log(`  ${chalk.green('✓')} ${editor.name} ${chalk.dim('detected')}`);
  } else {
    allPassed = false;
    console.log(`  ${chalk.red('✗')} ${editor.name} — ${chalk.yellow('not detected')}`);
    console.log(`    ${chalk.dim('Install Cursor: https://cursor.com or VS Code: https://code.visualstudio.com')}`);
  }

  console.log();
  if (allPassed) {
    console.log(chalk.green('  All checks passed — you\'re ready for the cohort!'));
  } else {
    console.log(chalk.yellow('  Some tools are missing. Install them before starting.'));
  }
  console.log(chalk.dim('  Checklist: https://cohort.bubblnet.com/checklist\n'));
}
