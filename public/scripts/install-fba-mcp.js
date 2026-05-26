/* fba-mcp-install v2 — toast only, no alert */
(function () {
  'use strict';

  function getUrl() {
    return window.FBA_MCP_URL || 'https://fba-mcp.throbbing-thunder-4d33.workers.dev/mcp';
  }

  function showToast(message) {
    var toast = document.getElementById('fba-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'fba-toast';
      toast.className = 'fba-toast';
      toast.setAttribute('role', 'status');
      toast.setAttribute('aria-live', 'polite');
      var msg = document.createElement('span');
      msg.className = 'fba-toast-msg';
      toast.appendChild(msg);
      document.body.appendChild(toast);
    }
    var el = toast.querySelector('.fba-toast-msg') || toast;
    el.textContent = message;
    toast.classList.remove('visible');
    void toast.offsetWidth;
    toast.classList.add('visible');
    if (toast._hideTimer) clearTimeout(toast._hideTimer);
    toast._hideTimer = setTimeout(function () {
      toast.classList.remove('visible');
    }, 2800);
  }

  function fallbackCopy(text) {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      var ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (err) {
      return false;
    }
  }

  function copyText(text) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      return navigator.clipboard.writeText(text).catch(function () {
        return fallbackCopy(text) ? Promise.resolve() : Promise.reject();
      });
    }
    return fallbackCopy(text) ? Promise.resolve() : Promise.reject();
  }

  function claudeDesktopConfig() {
    return JSON.stringify({
      mcpServers: {
        'fba-cohort': { command: 'npx', args: ['-y', 'mcp-remote', getUrl()] }
      }
    }, null, 2);
  }

  function claudeCodeConfig() {
    return JSON.stringify({
      mcpServers: {
        'fba-cohort': { type: 'http', url: getUrl() }
      }
    }, null, 2);
  }

  function cursorConfig() {
    return JSON.stringify({
      mcpServers: {
        'fba-cohort': { url: getUrl() }
      }
    }, null, 2);
  }

  function codexToml() {
    return '[mcp_servers.fba-cohort]\nurl = "' + getUrl() + '"\n';
  }

  function onCopySuccess(target, kind) {
    var msg;
    if (kind === 'config') {
      msg = 'Copied ' + target + ' config — paste into your MCP settings.';
    } else if (kind === 'toml') {
      msg = 'Copied Codex snippet — paste into ~/.codex/config.toml';
    } else {
      msg = 'Copied install URL for ' + target + ' — paste into the MCP panel.';
    }
    showToast(msg);
  }

  function handleCopy(btn) {
    var target = btn.getAttribute('data-target') || 'your AI client';
    var text = getUrl();
    var kind = 'url';

    if (btn.hasAttribute('data-mcp-claude-desktop-config')) {
      text = claudeDesktopConfig();
      kind = 'config';
      target = 'Claude Desktop';
    } else if (btn.hasAttribute('data-mcp-claude-code-config')) {
      text = claudeCodeConfig();
      kind = 'config';
      target = 'Claude Code';
    } else if (btn.hasAttribute('data-mcp-cursor-config')) {
      text = cursorConfig();
      kind = 'config';
      target = 'Cursor';
    } else if (btn.hasAttribute('data-mcp-codex-toml')) {
      text = codexToml();
      kind = 'toml';
      target = 'Codex';
    }

    copyText(text).then(
      function () { onCopySuccess(target, kind); },
      function () {
        showToast('Could not copy automatically — select text in the prompt.');
        window.prompt('Copy this for ' + target + ':', text);
      }
    );
  }

  function syncEndpointField() {
    var input = document.getElementById('fba-mcp-url-display');
    if (!input || input.tagName !== 'INPUT') return;
    input.value = getUrl();
    input.addEventListener('focus', function () { input.select(); });
    input.addEventListener('click', function () { input.select(); });
  }

  function init() {
    syncEndpointField();
    document.addEventListener('click', function (e) {
      var btn = e.target.closest(
        '[data-mcp-copy-url], [data-mcp-claude-desktop-config], [data-mcp-claude-code-config], [data-mcp-cursor-config], [data-mcp-codex-toml]'
      );
      if (!btn) return;
      e.preventDefault();
      e.stopPropagation();
      handleCopy(btn);
    });

    var openBtn = document.getElementById('fba-open-widget');
    if (openBtn) {
      openBtn.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var w = document.querySelector('.fba-lens-fab');
        if (w) w.click();
        else showToast('Use the Open Cohort Lens button at the bottom-right.');
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
