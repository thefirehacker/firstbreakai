import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join } from 'node:path';
import chalk from 'chalk';
import { STEPS, MCP_WORKER_URL } from './config.js';
import { loadProgress } from './status.js';

const rubrics = {
  1: [
    {
      name: '_quarto.yml exists',
      check: () => existsSync('_quarto.yml'),
    },
    {
      name: '_quarto.yml has type: website',
      check: () => {
        try {
          const content = readFileSync('_quarto.yml', 'utf-8');
          return /type:\s*website/i.test(content);
        } catch { return false; }
      },
    },
    {
      name: 'At least one .qmd file exists',
      check: () => readdirSync('.').some((f) => f.endsWith('.qmd')),
    },
    {
      name: 'Git repository initialized',
      check: () => existsSync('.git'),
    },
    {
      name: 'Git remote origin points to GitHub',
      check: () => {
        try {
          const out = execSync('git remote -v', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
          return /origin\s+.*github\.com/i.test(out);
        } catch { return false; }
      },
    },
    {
      name: 'At least one commit exists',
      check: () => {
        try {
          execSync('git log --oneline -1', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
          return true;
        } catch { return false; }
      },
    },
  ],
  2: [
    {
      name: 'run.c or run binary exists',
      check: () => existsSync('run.c') || existsSync('run'),
    },
    {
      name: 'Binary is executable',
      check: () => {
        try {
          if (existsSync('run')) {
            const st = statSync('run');
            return (st.mode & 0o111) !== 0;
          }
          return false;
        } catch { return false; }
      },
    },
    {
      name: 'config.json exists (model config)',
      check: () => existsSync('config.json'),
    },
    {
      name: 'Tokenizer file exists',
      check: () => existsSync('tokenizer.json') || existsSync('tokenizer.model'),
    },
    {
      name: 'Model weights file exists',
      check: () => readdirSync('.').some((f) => f.endsWith('.bin') || f.endsWith('.safetensors')),
    },
  ],
};

export default async function validate(args) {
  const stepId = parseInt(args[0], 10);

  if (isNaN(stepId) || !STEPS.find((s) => s.id === stepId)) {
    console.error(`Usage: firstbreakai validate <step>\nValid steps: ${STEPS.map((s) => s.id).join(', ')}`);
    process.exit(1);
  }

  const checks = rubrics[stepId];
  if (!checks) {
    console.log(chalk.yellow(`\n  No validation rubric for Step ${stepId} yet (coming soon).\n`));
    return;
  }

  const step = STEPS.find((s) => s.id === stepId);
  console.log(chalk.bold(`\n  Validating Step ${stepId}: ${step.title}\n`));

  let allPassed = true;
  const results = [];

  for (const check of checks) {
    let passed = false;
    try { passed = check.check(); } catch { passed = false; }

    results.push({ name: check.name, passed });

    if (passed) {
      console.log(`  ${chalk.green('✓')} ${check.name}`);
    } else {
      allPassed = false;
      console.log(`  ${chalk.red('✗')} ${check.name}`);
    }
  }

  console.log();
  if (allPassed) {
    console.log(chalk.green('  All checks passed!'));
    await submitToServer(stepId, results);
  } else {
    const passed = results.filter((r) => r.passed).length;
    console.log(chalk.yellow(`  ${passed}/${results.length} checks passed. Fix the failing ones and try again.`));
  }
  console.log();
}

async function submitToServer(stepId, results) {
  const progress = loadProgress();
  if (!progress.access_token) {
    console.log(chalk.dim('  Run "firstbreakai login" to save results to your profile.'));
    return;
  }

  console.log(chalk.dim('  Submitting results to server...'));

  try {
    const body = {
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'validate',
        arguments: {
          step: stepId,
          artifact: JSON.stringify(results),
        },
        ...(progress.session_id && { meta: { session_id: progress.session_id } }),
      },
    };

    const res = await fetch(`${MCP_WORKER_URL}/mcp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${progress.access_token}`,
      },
      body: JSON.stringify(body),
    });

    if (res.ok) {
      console.log(chalk.green('  ✓ Results saved to your profile.'));
    } else {
      console.log(chalk.yellow(`  Could not save results (${res.status}). They're stored locally.`));
    }
  } catch {
    console.log(chalk.yellow('  Could not reach server. Results stored locally only.'));
  }
}
