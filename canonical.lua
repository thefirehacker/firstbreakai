-- Quarto Lua filter: emit <link rel="canonical"> and <meta property="og:url"> on every HTML page.
-- Computes the URL from a hardcoded base + the rendered output file path.
-- Site URL kept here (not read from _quarto.yml) because Quarto's website meta
-- isn't exposed to per-document filters.

local SITE_URL = 'https://cohort.bubblnet.com'

function Pandoc(doc)
  local out = quarto.doc.output_file or ''
  local rel = out:gsub('^.*/docs/', ''):gsub('^docs/', '')
  if rel == '' or rel == out then return nil end

  local url_path = '/' .. rel
  url_path = url_path:gsub('/index%.html$', '/')

  local full = SITE_URL .. url_path

  quarto.doc.include_text('in-header',
    '<link rel="canonical" href="' .. full .. '">\n' ..
    '<meta property="og:url" content="' .. full .. '">')
  return nil
end
