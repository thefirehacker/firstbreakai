#!/usr/bin/env node
// Rewrite internal href attributes in all docs/*.html files so they use
// clean URLs instead of .html extensions.  Cloudflare Pages serves clean
// URLs and 308-redirects .html requests, so leaving .html in hrefs means
// every internal link triggers a redirect — wasting crawl budget and
// creating URL-signal inconsistency (links say .html, canonicals and
// sitemap say clean).

import fs from 'node:fs';
import path from 'node:path';

const DOCS_DIR = 'docs';

function findHtmlFiles(dir) {
  const results = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findHtmlFiles(full));
    } else if (entry.name.endsWith('.html')) {
      results.push(full);
    }
  }
  return results;
}

function rewriteLinks(html) {
  let count = 0;
  const result = html
    // Pass 1: index.html → / (must run before the generic .html strip)
    .replace(/href="(?!https?:\/\/|mailto:|tel:|#|javascript:|data:)([^"]*?)index\.html([#?][^"]*)?"/g,
      (_match, prefix, suffix) => {
        count++;
        return `href="${prefix}${suffix || ''}"`;
      })
    // Pass 2: .html → (strip extension)
    .replace(/href="(?!https?:\/\/|mailto:|tel:|#|javascript:|data:)([^"]*?)\.html([#?][^"]*)?"/g,
      (_match, prefix, suffix) => {
        count++;
        return `href="${prefix}${suffix || ''}"`;
      });
  return { result, count };
}

// Only rewrite during a full project render (CI deploy / `quarto render`).
// Single-file renders and `quarto preview` re-renders skip this step so that
// internal links keep their `.html` extensions for local browsing — Quarto's
// preview server doesn't auto-resolve clean URLs the way Cloudflare Pages does.
if (process.env.QUARTO_PROJECT_RENDER_ALL !== '1') {
  console.log('[links] skipping rewrite (not a full project render — set QUARTO_PROJECT_RENDER_ALL=1 to force)');
  process.exit(0);
}

if (!fs.existsSync(DOCS_DIR)) {
  console.warn(`[links] ${DOCS_DIR}/ not found, skipping rewrite`);
  process.exit(0);
}

const files = findHtmlFiles(DOCS_DIR);
let totalRewrites = 0;

for (const file of files) {
  const before = fs.readFileSync(file, 'utf8');
  const { result: after, count } = rewriteLinks(before);
  if (count > 0) {
    fs.writeFileSync(file, after);
    totalRewrites += count;
    console.log(`[links] ${file}: ${count} href(s) rewritten`);
  }
}

if (totalRewrites === 0) {
  console.log('[links] no internal .html hrefs found, nothing to rewrite');
} else {
  console.log(`[links] total: ${totalRewrites} href(s) rewritten across ${files.length} file(s)`);
}
