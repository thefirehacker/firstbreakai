import chalk from 'chalk';
import { MCP_WORKER_URL } from './config.js';
import { loadProgress } from './status.js';

export default async function ask(args) {
  const question = args.join(' ').trim();
  if (!question) {
    console.error('Usage: firstbreakai ask "your question here"');
    process.exit(1);
  }

  const progress = loadProgress();
  const headers = { 'Content-Type': 'application/json' };
  if (progress.access_token) {
    headers['Authorization'] = `Bearer ${progress.access_token}`;
  }

  const body = {
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: {
      name: 'ask',
      arguments: { question },
      ...(progress.session_id && { meta: { session_id: progress.session_id } }),
    },
  };

  console.log(chalk.dim('\n  Asking the FBA assistant...\n'));

  let res;
  try {
    res = await fetch(`${MCP_WORKER_URL}/mcp`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
  } catch (err) {
    console.error(chalk.red('  Could not reach the FBA server. Check your internet connection.'));
    console.error(chalk.dim(`  ${err.message}\n`));
    process.exit(1);
  }

  if (!res.ok) {
    console.error(chalk.red(`  Server error: ${res.status} ${res.statusText}`));
    process.exit(1);
  }

  const json = await res.json();

  if (json.error) {
    console.error(chalk.red(`  Error: ${json.error.message || JSON.stringify(json.error)}`));
    process.exit(1);
  }

  const result = json.result;
  if (!result) {
    console.error(chalk.red('  No result returned.'));
    process.exit(1);
  }

  const content = result.content || result;
  if (Array.isArray(content)) {
    for (const block of content) {
      if (block.type === 'text') {
        console.log(block.text);
      }
    }
  } else if (typeof content === 'string') {
    console.log(content);
  } else {
    console.log(JSON.stringify(content, null, 2));
  }

  if (result.sessionId) {
    progress.session_id = result.sessionId;
    const { writeFileSync } = await import('node:fs');
    const { PROGRESS_FILE } = await import('./config.js');
    writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2) + '\n');
  }

  console.log();
}
