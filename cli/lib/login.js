import { createServer } from 'node:http';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import openUrl from 'open';
import chalk from 'chalk';
import { MCP_WORKER_URL, PROGRESS_FILE } from './config.js';
import { loadProgress } from './status.js';

export default async function login() {
  const progress = loadProgress();

  if (progress.access_token && progress.learner_id) {
    console.log(chalk.yellow(`\n  Already logged in as ${progress.learner_id}.`));
    console.log(chalk.dim('  To re-login, delete ~/.firstbreakai.json and try again.\n'));
    return;
  }

  const port = await findFreePort();

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      const url = new URL(req.url, `http://localhost:${port}`);

      if (url.pathname !== '/callback') {
        res.writeHead(404);
        res.end('Not found');
        return;
      }

      const token = url.searchParams.get('token');
      const learnerId = url.searchParams.get('learner_id');
      const error = url.searchParams.get('error');

      if (error) {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end('<html><body><h2>Login failed</h2><p>You can close this tab.</p></body></html>');
        server.close();
        console.error(chalk.red(`\n  Login failed: ${error}\n`));
        reject(new Error(error));
        return;
      }

      if (!token) {
        res.writeHead(400);
        res.end('Missing token');
        return;
      }

      progress.access_token = token;
      progress.learner_id = learnerId || null;
      mkdirSync(dirname(PROGRESS_FILE), { recursive: true });
      writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2) + '\n');

      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(`
        <html><body style="font-family:system-ui;text-align:center;padding:4rem">
          <h2>Logged in to First Break AI!</h2>
          <p>You can close this tab and return to your terminal.</p>
        </body></html>
      `);

      server.close();
      console.log(chalk.green('\n  ✓ Logged in successfully!'));
      if (learnerId) console.log(chalk.dim(`  Learner ID: ${learnerId}`));
      console.log();
      resolve();
    });

    server.listen(port, '127.0.0.1', () => {
      const authUrl = `${MCP_WORKER_URL}/auth/discord?cli_port=${port}`;
      console.log(chalk.bold('\n  Opening Discord login in your browser...'));
      console.log(chalk.dim(`  If it doesn't open, visit: ${authUrl}\n`));
      console.log(chalk.dim('  Waiting for login...\n'));
      openUrl(authUrl);
    });

    setTimeout(() => {
      server.close();
      console.error(chalk.yellow('\n  Login timed out after 2 minutes. Try again.\n'));
      reject(new Error('timeout'));
    }, 120_000);
  });
}

async function findFreePort() {
  return new Promise((resolve) => {
    const srv = createServer();
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}
