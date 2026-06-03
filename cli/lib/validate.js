import { existsSync, readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import chalk from 'chalk';
import { STEPS, MCP_WORKER_URL, PROGRESS_FILE } from './config.js';
import { loadProgress } from './status.js';

let _serverProfile = undefined;
async function getServerProfile() {
  if (_serverProfile !== undefined) return _serverProfile;
  const progress = loadProgress();
  if (!progress.access_token) { _serverProfile = null; return null; }
  try {
    const res = await fetch(`${MCP_WORKER_URL}/auth/me`, {
      headers: { 'Authorization': `Bearer ${progress.access_token}` },
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) { _serverProfile = await res.json(); return _serverProfile; }
  } catch { /* offline */ }
  _serverProfile = null;
  return null;
}

const rubrics = {
  0: [
    {
      name: 'Logged in via Discord',
      hint: 'Run "firstbreakai login" to authenticate with Discord',
      check: () => {
        const progress = loadProgress();
        return !!(progress.access_token && progress.learner_id);
      },
    },
    {
      name: 'Member of FBA Discord server',
      hint: 'Join the Discord: https://discord.gg/hRPese4H3F — then run "firstbreakai login"',
      check: async () => {
        const profile = await getServerProfile();
        return profile?.in_guild === true || profile?.in_guild === 1;
      },
    },
    {
      name: 'Git installed',
      hint: 'Install Git: https://git-scm.com/downloads',
      check: () => {
        try { execSync('git --version', { stdio: 'pipe' }); return true; } catch { return false; }
      },
    },
    {
      name: 'Python installed',
      hint: 'Install Python 3: https://python.org/downloads',
      check: () => {
        try { execSync('python3 --version', { stdio: 'pipe' }); return true; } catch { return false; }
      },
    },
  ],
  1: [
    {
      name: '_quarto.yml exists',
      hint: 'Run "firstbreakai init" to scaffold a Quarto blog',
      check: () => existsSync('_quarto.yml'),
    },
    {
      name: '_quarto.yml has type: website',
      hint: 'Add "type: website" under "project:" in your _quarto.yml',
      check: () => {
        try {
          const content = readFileSync('_quarto.yml', 'utf-8');
          return /type:\s*website/i.test(content);
        } catch { return false; }
      },
    },
    {
      name: 'At least one .qmd file exists',
      hint: 'Create a .qmd file, or run "firstbreakai init"',
      check: () => readdirSync('.').some((f) => f.endsWith('.qmd')),
    },
    {
      name: 'Git repository initialized',
      hint: 'Run: git init',
      check: () => existsSync('.git'),
    },
    {
      name: 'Git remote origin points to GitHub',
      hint: 'Run: git remote add origin https://github.com/YOUR_USER/YOUR_REPO',
      check: () => {
        try {
          const out = execSync('git remote -v', { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
          return /origin\s+.*github\.com/i.test(out);
        } catch { return false; }
      },
    },
    {
      name: 'At least one commit exists',
      hint: 'Run: git add . && git commit -m "Initial commit"',
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
      hint: 'Download from the Lesson 1 guide: https://cohort.bubblnet.com/lessons/lesson-1-huggingface-beyond-upload',
      check: () => existsSync('run.c') || existsSync('run'),
    },
    {
      name: 'Binary is executable',
      hint: 'Compile with: gcc -O2 -o run run.c -lm  then: chmod +x run',
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
      hint: 'Run: huggingface-cli download Qwen/Qwen3-0.6B config.json --local-dir .',
      check: () => existsSync('config.json'),
    },
    {
      name: 'Tokenizer file exists',
      hint: 'Run: huggingface-cli download Qwen/Qwen3-0.6B tokenizer.json --local-dir .',
      check: () => existsSync('tokenizer.json') || existsSync('tokenizer.model'),
    },
    {
      name: 'Model weights file exists',
      hint: 'Run: huggingface-cli download Qwen/Qwen3-0.6B --local-dir .  (downloads .safetensors)',
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
    try { passed = await check.check(); } catch { passed = false; }

    results.push({ name: check.name, passed });

    if (passed) {
      console.log(`  ${chalk.green('✓')} ${check.name}`);
    } else {
      allPassed = false;
      console.log(`  ${chalk.red('✗')} ${check.name}`);
      if (check.hint) console.log(`    ${chalk.dim(check.hint)}`);
    }
  }

  console.log();
  if (allPassed) {
    console.log(chalk.green('  All checks passed!'));
    await markStepDone(stepId, step);
    await submitToServer(stepId, results);
  } else {
    const passed = results.filter((r) => r.passed).length;
    console.log(chalk.yellow(`  ${passed}/${results.length} checks passed. Fix the failing ones and try again.`));
  }
  console.log();
}

async function markStepDone(stepId, step) {
  const progress = loadProgress();
  if (progress.steps[stepId]?.done) return;
  progress.steps[stepId] = { done: true, at: new Date().toISOString() };
  mkdirSync(dirname(PROGRESS_FILE), { recursive: true });
  writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2) + '\n');
  console.log(chalk.green(`  ✓ Step ${stepId}: ${step.title} — marked complete!`));
  const nextStep = STEPS.find((s) => !progress.steps[s.id]?.done);
  if (nextStep) {
    console.log(chalk.dim(`  Next up: Step ${nextStep.id} — ${nextStep.title}`));
  }
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
