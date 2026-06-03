#!/usr/bin/env node

import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkg = createRequire(import.meta.url)('../package.json');

const HELP = `
firstbreakai v${pkg.version} — CLI for the First Break AI cohort
https://cohort.bubblnet.com

Usage:
  firstbreakai <command> [options]

Commands:
  doctor            Check your dev environment (Git, Python, Quarto, etc.)
  status            Show your cohort progress
  done <step>       Mark a step as complete
  init              Scaffold a new Quarto blog in the current directory
  open [page]       Open a cohort page in your browser
  next              Open the next incomplete step
  ask "<question>"  Ask the FBA AI assistant a question
  login             Authenticate via Discord
  whoami            Show your login status and profile
  validate <step>   Run local checks for a step
  mcp               Start the MCP server (for Cursor, Claude Desktop, etc.)
  help              Show this help message
  version           Show version

Examples:
  firstbreakai doctor
  firstbreakai done 1
  firstbreakai ask "How do I download a model from HuggingFace?"
  firstbreakai validate 1

Learn more: https://cohort.bubblnet.com/roadmap
`;

const [command, ...args] = process.argv.slice(2);

if (!command || command === 'help' || command === '--help' || command === '-h') {
  console.log(HELP);
  process.exit(0);
}

if (command === 'version' || command === '--version' || command === '-v') {
  console.log(pkg.version);
  process.exit(0);
}

const commands = {
  doctor:   () => import('../lib/doctor.js'),
  status:   () => import('../lib/status.js'),
  done:     () => import('../lib/done.js'),
  init:     () => import('../lib/init.js'),
  open:     () => import('../lib/open.js'),
  next:     () => import('../lib/next.js'),
  ask:      () => import('../lib/ask.js'),
  login:    () => import('../lib/login.js'),
  whoami:   () => import('../lib/whoami.js'),
  validate: () => import('../lib/validate.js'),
  mcp:      () => import('../lib/mcp-server.js'),
};

if (!commands[command]) {
  console.error(`Unknown command: ${command}\nRun "firstbreakai help" to see available commands.`);
  process.exit(1);
}

try {
  const mod = await commands[command]();
  await mod.default(args);
} catch (err) {
  console.error(err.message || err);
  process.exit(1);
}

if (command === 'doctor' || command === 'help' || command === 'status') {
  import('../lib/update-check.js')
    .then((m) => m.checkForUpdate())
    .catch(() => {});
}
