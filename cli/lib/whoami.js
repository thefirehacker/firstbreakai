import chalk from 'chalk';
import { MCP_WORKER_URL } from './config.js';
import { loadProgress } from './status.js';

export default async function whoami() {
  const progress = loadProgress();

  if (!progress.access_token || !progress.learner_id) {
    console.log(chalk.yellow('\n  Not logged in.'));
    console.log(chalk.dim('  Run "firstbreakai login" to authenticate via Discord.\n'));
    return;
  }

  console.log(chalk.bold('\n  First Break AI — Account\n'));
  console.log(`  Learner ID:  ${chalk.cyan(progress.learner_id)}`);

  try {
    const res = await fetch(`${MCP_WORKER_URL}/auth/me`, {
      headers: { 'Authorization': `Bearer ${progress.access_token}` },
    });

    if (res.ok) {
      const me = await res.json();
      if (me.discord_username) console.log(`  Discord:     ${chalk.cyan(me.discord_username)}`);
      if (me.email) console.log(`  Email:       ${chalk.dim(me.email)}`);
      if (me.slug) console.log(`  Profile:     ${chalk.dim(`https://cohort.bubblnet.com/students/${me.slug}`)}`);
      console.log(`  In guild:    ${me.in_guild ? chalk.green('yes') : chalk.yellow('no')}`);
      console.log(`  Step:        ${me.current_step ?? 0}`);
    } else if (res.status === 401) {
      console.log(chalk.yellow('  Token expired. Run "firstbreakai login" to re-authenticate.'));
    } else {
      console.log(chalk.dim('  Could not fetch profile from server.'));
    }
  } catch {
    console.log(chalk.dim('  Offline — showing local info only.'));
  }

  console.log();
}
