#!/usr/bin/env node
// Rewrite docs/sitemap.xml so <loc> entries match the canonical URLs
// emitted by canonical.lua and the clean URLs Cloudflare Pages serves
// (i.e. /index.html -> / and /foo/bar.html -> /foo/bar). Without this,
// every sitemap entry 308-redirects to its canonical, Google records
// the sitemap as not covering the canonical URL, and indexing stalls
// with "No referring sitemaps detected".

import fs from 'node:fs';
import path from 'node:path';

const sitemapPath = path.join('docs', 'sitemap.xml');

if (!fs.existsSync(sitemapPath)) {
  console.warn(`[sitemap] ${sitemapPath} not found, skipping rewrite`);
  process.exit(0);
}

const before = fs.readFileSync(sitemapPath, 'utf8');
const after = before
  .replace(/<loc>([^<]+)\/index\.html<\/loc>/g, '<loc>$1/</loc>')
  .replace(/<loc>([^<]+)\.html<\/loc>/g, '<loc>$1</loc>');

if (before === after) {
  console.log('[sitemap] no .html suffixes found, nothing to rewrite');
  process.exit(0);
}

fs.writeFileSync(sitemapPath, after);

const rewritten = (after.match(/<loc>/g) || []).length;
console.log(`[sitemap] rewrote ${rewritten} <loc> entries to clean URLs`);
