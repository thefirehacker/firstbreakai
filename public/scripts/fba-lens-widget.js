/*
 * FBA Cohort Lens widget — cohort-hosted UI for FetchLens widget/mcp endpoint.
 * Modal forms (no window.prompt), resizable panel, per-message copy.
 */
(function () {
  'use strict';

  if (window.__fbaLensLoaded) return;
  window.__fbaLensLoaded = true;

  var script = document.currentScript;
  var ENDPOINT =
    (script && script.dataset.endpoint) ||
    'https://fba-mcp.throbbing-thunder-4d33.workers.dev/widget/mcp';
  var BUTTON_LABEL =
    (script && script.dataset.buttonLabel) || 'Open Cohort Lens';
  var STORAGE_KEY = 'fba-lens-session';
  var TOKEN_KEY = 'fba-lens-token';
  var SIZE_KEY = 'fba-lens-panel-size';
  var AUTH_ORIGIN = ENDPOINT.replace(/\/widget\/mcp$/, '').replace(/\/mcp$/, '');

  var sessionId = null;
  var authToken = null;
  try {
    sessionId = localStorage.getItem(STORAGE_KEY);
  } catch (_) {}
  try {
    authToken = localStorage.getItem(TOKEN_KEY);
  } catch (_) {}

  var panelSizes = {
    s: { w: 360, h: 420 },
    m: { w: 420, h: 520 },
    l: { w: 520, h: 600 },
    xl: { w: 640, h: 680 },
  };

  function el(tag, className, attrs) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs.html;
        else node.setAttribute(k, attrs[k]);
      });
    }
    return node;
  }

  function copyText(text, btn) {
    var done = function () {
      if (!btn) return;
      var prev = btn.getAttribute('aria-label') || btn.title;
      btn.setAttribute('aria-label', 'Copied');
      btn.textContent = '✓';
      setTimeout(function () {
        btn.textContent = '⎘';
        if (prev) btn.setAttribute('aria-label', prev);
      }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(function () {
        fallbackCopy(text);
        done();
      });
    } else {
      fallbackCopy(text);
      done();
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;left:-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
    } catch (_) {}
    document.body.removeChild(ta);
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  var mqMobile = window.matchMedia('(max-width: 640px)');
  var mqCompact = window.matchMedia('(max-width: 1100px), (max-height: 720px)');

  // --- FAB ---
  var fab = el('button', 'fba-lens-fab', { type: 'button', 'aria-label': BUTTON_LABEL });
  var fabLong = el('span', 'fba-lens-fab-label fba-lens-fab-label--long', { text: BUTTON_LABEL });
  var fabShort = el('span', 'fba-lens-fab-label fba-lens-fab-label--short', { text: 'Ask' });
  fab.appendChild(fabLong);
  fab.appendChild(fabShort);

  // --- Panel ---
  var panel = el('div', 'fba-lens-panel', { role: 'dialog', 'aria-label': 'Cohort Lens' });

  var header = el('div', 'fba-lens-header');
  header.appendChild(el('span', 'fba-lens-title', { text: 'Cohort Lens' }));

  var headerActions = el('div', 'fba-lens-header-actions');
  var sizeGroup = el('div', 'fba-lens-size-group');
  var expandBtn = el('button', 'fba-lens-icon-btn fba-lens-expand-btn', {
    type: 'button',
    'aria-label': 'Expand chat panel',
    title: 'Expand',
    text: '⤢',
  });
  var sizeBtns = {};
  ['s', 'm', 'l', 'xl'].forEach(function (size) {
    var b = el('button', 'fba-lens-icon-btn', {
      type: 'button',
      'aria-label': 'Panel size ' + size.toUpperCase(),
      title: 'Size ' + size.toUpperCase(),
      text: size.toUpperCase(),
    });
    b.dataset.size = size;
    sizeBtns[size] = b;
    sizeGroup.appendChild(b);
  });
  headerActions.appendChild(sizeGroup);
  headerActions.appendChild(expandBtn);
  var closeBtn = el('button', 'fba-lens-icon-btn', {
    type: 'button',
    'aria-label': 'Close chat',
    title: 'Close',
    text: '×',
  });
  headerActions.appendChild(closeBtn);
  header.appendChild(headerActions);

  var log = el('div', 'fba-lens-log', { 'aria-live': 'polite' });

  var form = el('form', 'fba-lens-form');
  var input = el('input', 'fba-lens-input', {
    type: 'text',
    placeholder: 'Ask anything about the cohort…',
    'aria-label': 'Message',
    required: 'required',
  });
  var sendBtn = el('button', 'fba-lens-send', { type: 'submit', text: 'Send' });
  form.appendChild(input);
  form.appendChild(sendBtn);

  var actions = el('div', 'fba-lens-actions');

  var resizeHint = el('span', 'fba-lens-resize-hint', { text: 'Drag corner to resize' });

  // --- Modal ---
  var modal = el('div', 'fba-lens-modal', { role: 'dialog', 'aria-modal': 'true' });
  var modalCard = el('div', 'fba-lens-modal-card');
  var modalTitle = el('h3', 'fba-lens-modal-title');
  var modalDesc = el('p', 'fba-lens-modal-desc');
  var modalFields = el('div', 'fba-lens-modal-fields');
  var modalCancel = el('button', 'fba-lens-modal-cancel', { type: 'button', text: 'Cancel' });
  var modalSubmit = el('button', 'fba-lens-modal-submit', { type: 'button', text: 'Continue' });
  var modalActions = el('div', 'fba-lens-modal-actions');
  modalActions.appendChild(modalCancel);
  modalActions.appendChild(modalSubmit);
  modalCard.appendChild(modalTitle);
  modalCard.appendChild(modalDesc);
  modalCard.appendChild(modalFields);
  modalCard.appendChild(modalActions);
  modal.appendChild(modalCard);

  panel.appendChild(header);
  panel.appendChild(log);
  panel.appendChild(form);
  panel.appendChild(actions);
  panel.appendChild(resizeHint);
  panel.appendChild(modal);

  var modalResolve = null;

  function showModal(opts) {
    modalTitle.textContent = opts.title || '';
    modalDesc.textContent = opts.description || '';
    modalFields.innerHTML = '';
    modalSubmit.textContent = opts.submitLabel || 'Continue';
    (opts.fields || []).forEach(function (field) {
      var wrap = el('div', 'fba-lens-modal-field');
      wrap.appendChild(el('label', '', { text: field.label, for: field.id }));
      var control;
      if (field.type === 'textarea') {
        control = el('textarea', '', { id: field.id, placeholder: field.placeholder || '' });
      } else {
        control = el('input', '', {
          id: field.id,
          type: field.inputType || 'text',
          placeholder: field.placeholder || '',
        });
        if (field.required) control.required = true;
      }
      wrap.appendChild(control);
      modalFields.appendChild(wrap);
    });
    modal.classList.add('is-visible');
    var first = modalFields.querySelector('input, textarea');
    if (first) setTimeout(function () { first.focus(); }, 50);
    return new Promise(function (resolve) {
      modalResolve = resolve;
    });
  }

  function hideModal(result) {
    modal.classList.remove('is-visible');
    if (modalResolve) {
      modalResolve(result);
      modalResolve = null;
    }
  }

  modalCancel.addEventListener('click', function () {
    hideModal(null);
  });

  modal.addEventListener('click', function (e) {
    if (e.target === modal) hideModal(null);
  });

  modalSubmit.addEventListener('click', function () {
    var inputs = modalFields.querySelectorAll('input, textarea');
    var values = [];
    for (var i = 0; i < inputs.length; i++) {
      if (inputs[i].required && !inputs[i].value.trim()) {
        inputs[i].focus();
        return;
      }
      values.push(inputs[i].value.trim());
    }
    hideModal(values);
  });

  document.addEventListener('keydown', function (e) {
    if (!modal.classList.contains('is-visible')) return;
    if (e.key === 'Escape') hideModal(null);
  });

  var layoutMode = 'desktop';

  function isSheet() {
    return layoutMode === 'sheet';
  }

  function isCompact() {
    return layoutMode === 'compact';
  }

  function setBodyLock(locked) {
    document.documentElement.classList.toggle('fba-lens-open', locked);
    document.body.classList.toggle('fba-lens-open', locked);
  }

  function updateLayoutMode() {
    var next = mqMobile.matches ? 'sheet' : mqCompact.matches ? 'compact' : 'desktop';
    panel.classList.toggle('is-sheet', next === 'sheet');
    panel.classList.toggle('is-compact', next === 'compact');
    if (next === layoutMode) {
      if (panel.classList.contains('is-open')) setBodyLock(next === 'sheet');
      return;
    }
    layoutMode = next;
    if (panel.classList.contains('is-open')) {
      setBodyLock(next === 'sheet');
    }
    if (next !== 'desktop') {
      panel.style.width = '';
      panel.style.height = '';
      panel.classList.remove('is-expanded');
      expandBtn.setAttribute('aria-pressed', 'false');
      expandBtn.textContent = '⤢';
      expandBtn.setAttribute('aria-label', 'Expand chat panel');
    } else {
      var saved = null;
      try {
        saved = localStorage.getItem(SIZE_KEY);
      } catch (_) {}
      applyPanelSize(saved && panelSizes[saved] ? saved : 'm', true);
    }
  }

  function applyPanelSize(size, force) {
    if (!force && (isSheet() || isCompact())) return;
    var dim = panelSizes[size] || panelSizes.m;
    var maxW = Math.min(dim.w, window.innerWidth - 24);
    var maxH = Math.min(dim.h, window.innerHeight - 96);
    panel.style.width = maxW + 'px';
    panel.style.height = maxH + 'px';
    Object.keys(sizeBtns).forEach(function (k) {
      sizeBtns[k].setAttribute('aria-pressed', k === size ? 'true' : 'false');
    });
    try {
      localStorage.setItem(SIZE_KEY, size);
    } catch (_) {}
  }

  Object.keys(sizeBtns).forEach(function (size) {
    sizeBtns[size].addEventListener('click', function () {
      applyPanelSize(size, true);
    });
  });

  expandBtn.addEventListener('click', function () {
    var expanded = panel.classList.toggle('is-expanded');
    expandBtn.setAttribute('aria-pressed', expanded ? 'true' : 'false');
    expandBtn.setAttribute('aria-label', expanded ? 'Shrink chat panel' : 'Expand chat panel');
    expandBtn.textContent = expanded ? '⤡' : '⤢';
  });

  try {
    if (!mqMobile.matches && !mqCompact.matches) {
      var saved = localStorage.getItem(SIZE_KEY);
      applyPanelSize(saved && panelSizes[saved] ? saved : 'm', true);
    }
  } catch (_) {
    applyPanelSize('m', true);
  }

  updateLayoutMode();
  if (mqMobile.addEventListener) {
    mqMobile.addEventListener('change', updateLayoutMode);
    mqCompact.addEventListener('change', updateLayoutMode);
  } else {
    mqMobile.addListener(updateLayoutMode);
    mqCompact.addListener(updateLayoutMode);
  }

  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      updateLayoutMode();
      if (layoutMode === 'desktop' && panel.classList.contains('is-open')) {
        var pressed = Object.keys(sizeBtns).find(function (k) {
          return sizeBtns[k].getAttribute('aria-pressed') === 'true';
        });
        if (pressed) applyPanelSize(pressed, true);
      }
    }, 120);
  });

  function appendMessage(kind, text) {
    var kindClass =
      kind === 'user'
        ? 'fba-lens-msg--user'
        : kind === 'sys'
          ? 'fba-lens-msg--sys'
          : kind === 'loading'
            ? 'fba-lens-msg--loading fba-lens-msg--lens'
            : 'fba-lens-msg--lens';
    var div = el('div', 'fba-lens-msg ' + kindClass);
    var body = el('div', 'fba-lens-msg-body');
    body.textContent = text;
    div.appendChild(body);

    if (kind !== 'loading') {
      var copyBtn = el('button', 'fba-lens-msg-copy', {
        type: 'button',
        'aria-label': 'Copy message',
        title: 'Copy',
        text: '⎘',
      });
      copyBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        copyText(body.innerText || body.textContent || '', copyBtn);
      });
      div.appendChild(copyBtn);
    }

    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return { root: div, body: body };
  }

  function setOpen(open) {
    if (open) {
      updateLayoutMode();
      panel.classList.add('is-open');
      setBodyLock(isSheet());
      if (!log.childElementCount) {
        appendMessage('sys', 'Ask me anything about the cohort — lessons, roadmap, enrollment, or next steps.');
        renderQuickActions();
      }
      if (!isSheet()) {
        try {
          input.focus({ preventScroll: true });
        } catch (_) {
          input.focus();
        }
      }
    } else {
      panel.classList.remove('is-open');
      setBodyLock(false);
      hideModal(null);
    }
  }

  fab.addEventListener('click', function () {
    setOpen(!panel.classList.contains('is-open'));
  });

  closeBtn.addEventListener('click', function () {
    setOpen(false);
  });

  function renderQuickActions() {
    actions.innerHTML = '';
    var items = [];

    if (authToken) {
      items.push({ label: 'My Profile', action: function () { callTool('profile', { action: 'get', session_id: sessionId || undefined }); } });
      items.push({ label: 'Logout', action: logoutFlow, cls: 'fba-lens-chip--danger' });
    } else {
      items.push({ label: 'Login with Discord', action: loginFlow, cls: 'fba-lens-chip--discord' });
      items.push({ label: 'Enroll me', action: enrollFlow });
    }

    items.push({
      label: 'What is FBA?',
      action: function () {
        ask('What is First Break AI and how does it work?');
      },
    });
    items.push({
      label: 'Next step for me',
      action: function () {
        callTool('next', { session_id: sessionId || '' });
      },
    });
    items.push({ label: 'Validate my Quarto blog', action: validateFlow });

    items.forEach(function (a) {
      var b = el('button', 'fba-lens-chip' + (a.cls ? ' ' + a.cls : ''), { type: 'button', text: a.label });
      b.addEventListener('click', a.action);
      actions.appendChild(b);
    });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var q = input.value.trim();
    if (!q) return;
    input.value = '';
    ask(q);
  });

  function ask(question) {
    appendMessage('user', question);
    callTool('ask', {
      question: question,
      page_url: location.href,
      session_id: sessionId || undefined,
    });
  }

  function enrollFlow() {
    showModal({
      title: 'Join First Break AI',
      description: 'We will send a Discord invite and welcome email to this address.',
      submitLabel: 'Enroll',
      fields: [
        {
          id: 'fba-enroll-email',
          label: 'Email',
          inputType: 'email',
          placeholder: 'you@example.com',
          required: true,
        },
        {
          id: 'fba-enroll-intent',
          label: 'What do you want to learn or build? (optional)',
          type: 'textarea',
          placeholder: 'One sentence helps us point you at the right next step.',
        },
      ],
    }).then(function (values) {
      if (!values) return;
      var email = values[0];
      var intent = values[1] || '';
      if (!email) return;
      appendMessage('user', 'Enroll me with ' + email);
      callTool('do', {
        intent: 'enroll',
        email: email,
        stated_intent: intent,
        session_id: sessionId || undefined,
      });
    });
  }

  function validateFlow() {
    showModal({
      title: 'Validate your Quarto blog',
      description: 'Paste the live, public URL of your cohort blog post or site.',
      submitLabel: 'Validate',
      fields: [
        {
          id: 'fba-validate-url',
          label: 'Blog URL',
          inputType: 'url',
          placeholder: 'https://yourname.github.io/...',
          required: true,
        },
      ],
    }).then(function (values) {
      if (!values || !values[0]) return;
      var url = values[0];
      appendMessage('user', 'Validate: ' + url);
      callTool('validate', {
        rubric_id: 'step-1-quarto-blog',
        artifact: url,
        session_id: sessionId || undefined,
      });
    });
  }

  function loginFlow() {
    var w = 500, h = 700;
    var left = (screen.width - w) / 2, top = (screen.height - h) / 2;
    var popup = window.open(
      AUTH_ORIGIN + '/auth/discord',
      'fba-discord-login',
      'width=' + w + ',height=' + h + ',left=' + left + ',top=' + top
    );
    var interval = setInterval(function () {
      if (!popup || popup.closed) {
        clearInterval(interval);
        try { authToken = localStorage.getItem(TOKEN_KEY); } catch (_) {}
        if (authToken) {
          appendMessage('sys', 'Logged in via Discord.');
          renderQuickActions();
        }
        return;
      }
    }, 500);

    window.addEventListener('message', function handler(e) {
      if (e.data && e.data.type === 'fba-auth' && e.data.token) {
        window.removeEventListener('message', handler);
        clearInterval(interval);
        authToken = e.data.token;
        try { localStorage.setItem(TOKEN_KEY, authToken); } catch (_) {}
        appendMessage('sys', 'Logged in via Discord.');
        renderQuickActions();
        if (popup && !popup.closed) popup.close();
      }
    });
  }

  function logoutFlow() {
    var headers = { 'content-type': 'application/json' };
    if (authToken) headers['authorization'] = 'Bearer ' + authToken;
    fetch(AUTH_ORIGIN + '/auth/logout', { method: 'POST', headers: headers }).catch(function () {});
    authToken = null;
    try { localStorage.removeItem(TOKEN_KEY); } catch (_) {}
    appendMessage('sys', 'Logged out.');
    renderQuickActions();
  }

  function callTool(name, args) {
    var loading = appendMessage('loading', 'Thinking…');
    var headers = { 'content-type': 'application/json', 'x-mcp-client': 'fba-widget/2.0' };
    if (authToken) headers['authorization'] = 'Bearer ' + authToken;
    fetch(ENDPOINT, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: { name: name, arguments: args },
      }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (payload) {
        var sc = payload && payload.result && payload.result.structuredContent;
        if (sc && sc.sessionId) {
          sessionId = sc.sessionId;
          try {
            localStorage.setItem(STORAGE_KEY, sessionId);
          } catch (_) {}
        }
        renderToolResult(loading, name, sc || payload);
      })
      .catch(function (err) {
        loading.body.textContent =
          'Network error: ' + (err && err.message ? err.message : String(err));
        loading.root.classList.remove('fba-lens-msg--loading');
        loading.root.classList.add('fba-lens-msg--lens');
      });
  }

  function renderToolResult(loading, name, sc) {
    loading.root.classList.remove('fba-lens-msg--loading');
    loading.root.classList.add('fba-lens-msg--lens');

    if (!sc) {
      loading.body.textContent = 'Empty response from Lens.';
      return;
    }
    if (sc.ok === false) {
      loading.body.textContent = '⚠ ' + (sc.error || 'Tool returned an error.');
      if (sc.followUp && sc.followUp.question) {
        var follow = appendMessage('lens', sc.followUp.question);
        follow.body.style.fontStyle = 'italic';
      }
      return;
    }

    var data = sc.data || {};
    loading.body.innerHTML = '';

    if (name === 'ask') {
      var p = el('div', '', { text: data.answer || '(no answer)' });
      loading.body.appendChild(p);
      if (data.citations && data.citations.length) {
        var c = el('div', 'fba-lens-citations', {
          text: 'Sources: ' + data.citations.join(', '),
        });
        loading.body.appendChild(c);
      }
    } else if (name === 'do') {
      loading.body.textContent = data.message || 'Done.';
    } else if (name === 'find') {
      (data.results || []).forEach(function (r) {
        var row = el('div', 'fba-lens-find-row');
        row.innerHTML =
          '<strong>' +
          escapeHtml(r.title) +
          '</strong> <em style="color:#7a6f62;font-size:11px">' +
          escapeHtml(r.type) +
          '</em><br/><span style="color:#4a4035">' +
          escapeHtml(r.summary) +
          '</span>';
        loading.body.appendChild(row);
      });
    } else if (name === 'validate') {
      loading.body.textContent =
        (data.passed ? '✓ Passed — ' : '✗ Not yet — ') + (data.reason || '');
    } else if (name === 'next') {
      loading.body.textContent =
        data.question_for_user || data.step_name || 'Next step ready.';
    } else if (name === 'profile') {
      if (data.discord_username) {
        var html = '';
        if (data.avatar_url) html += '<img src="' + escapeHtml(data.avatar_url) + '" style="width:32px;height:32px;border-radius:50%;vertical-align:middle;margin-right:8px" />';
        html += '<strong>' + escapeHtml(data.discord_username) + '</strong>';
        html += '<br/><span style="color:#7a6f62;font-size:12px">Step ' + (data.current_step || 0) + ' &bull; ' + (data.in_guild ? 'In FBA Discord' : 'Not in Discord yet') + '</span>';
        if (data.blog_url) html += '<br/><span style="font-size:12px">Blog: ' + escapeHtml(data.blog_url) + '</span>';
        if (data.profile_url) html += '<br/><span style="font-size:12px">Profile: ' + escapeHtml(data.profile_url) + '</span>';
        loading.body.innerHTML = html;
      } else {
        loading.body.textContent = data.message || JSON.stringify(data);
      }
    } else {
      loading.body.textContent = JSON.stringify(data, null, 2);
    }

    if (sc.followUp && sc.followUp.question) {
      var follow = appendMessage('lens', sc.followUp.question);
      follow.body.style.fontStyle = 'italic';
    }
  }

  function mount() {
    document.body.appendChild(panel);
    document.body.appendChild(fab);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
