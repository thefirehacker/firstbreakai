import { existsSync, cpSync, readdirSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import chalk from 'chalk';

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEMPLATES_DIR = resolve(__dirname, '..', 'templates');

export default async function init() {
  const dest = process.cwd();

  if (existsSync(join(dest, '_quarto.yml'))) {
    console.log(chalk.yellow('\n  _quarto.yml already exists in this directory.'));
    console.log(chalk.dim('  This looks like an existing Quarto project. No changes made.\n'));
    return;
  }

  const files = readdirSync(TEMPLATES_DIR);
  for (const file of files) {
    const src = join(TEMPLATES_DIR, file);
    const target = join(dest, file);
    if (existsSync(target)) {
      console.log(chalk.yellow(`  Skipped ${file} (already exists)`));
    } else {
      cpSync(src, target);
      console.log(chalk.green(`  ✓ Created ${file}`));
    }
  }

  console.log(chalk.bold('\n  Blog scaffolded! Next steps:'));
  console.log(chalk.dim('  1. Run: quarto preview'));
  console.log(chalk.dim('  2. Edit first-post.qmd with your own content'));
  console.log(chalk.dim('  3. git init && git add . && git commit -m "Initial blog"'));
  console.log(chalk.dim('  4. Push to GitHub and enable GitHub Pages'));
  console.log(chalk.dim(`\n  Full guide: https://cohort.bubblnet.com/lessons/lesson-0-welcome\n`));
}
