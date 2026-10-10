/* ReelRead — client app
 *
 * Plain JS, no framework, no build step. Loaded with `defer` from index.html.
 *
 * Security rules (from SPEC "Frontend rules"):
 *   - Server data is never put into innerHTML. Every node is built with
 *     createElement / textContent (or append(string), which makes a Text node).
 *   - img.src is only set for Wikimedia CDN thumbnails (upload./thumb.wikimedia.org) and
 *     a.href only for https://en.wikipedia.org/ links.
 *   - No inline scripts, styles or handlers (CSP).
 *
 * Map of this file:
 *   1. Config + small helpers      5. Feed loading (pages, prefetch, errors)
 *   2. API wrapper                 6. Card + quiz rendering
 *   3. Screens + boot              7. Dwell tracking + event queue
 *   4. Onboarding / topic picker   8. Keyboard, lifecycle, service worker
 */
'use strict';

(() => {
  // ------------------------------------------------------------------
  // 1. Config + small helpers
  // ------------------------------------------------------------------
  const FEED_PAGE = 10;          // items per GET /api/feed
  const PREFETCH_AT = 3;         // fetch the next page when this many items are left
  const FLUSH_EVERY_MS = 10000;  // flush the event queue at least this often
  const FLUSH_AT = 10;           // ...or as soon as this many events are queued
  const BATCH_MAX = 50;          // server accepts at most 50 events per request
  const QUEUE_CAP = 500;         // drop the oldest events beyond this (offline for a long time)
  const VIEW_THRESHOLD = 0.6;    // a card counts as "being read" while >= 60% visible
  const MAX_TOPICS = 5;

  // Wikimedia serves thumbnails from either CDN host.
  const IMG_PREFIXES = ['https://upload.wikimedia.org/', 'https://thumb.wikimedia.org/'];
  const LINK_PREFIX = 'https://en.wikipedia.org/';

  // Fallback labels if /api/topics hasn't loaded. Also the whitelist of ids
  // allowed into data-topic (which drives the topic colour in CSS).
  const KNOWN_TOPICS = {
    tech: 'Tech', sports: 'Sports', ai_ml: 'AI & ML', finance: 'Finance', politics: 'Politics',
  };

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const $ = (id) => document.getElementById(id);

  /** Create an element. `props.text` sets textContent; other keys become attributes.
   *  Children may be nodes or strings (strings become Text nodes — never parsed as HTML). */
  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'class') el.className = value;
        else if (key === 'text') el.textContent = value;
        else el.setAttribute(key, value === true ? '' : String(value));
      }
    }
    for (const child of children) {
      if (child !== null && child !== undefined) el.append(child);
    }
    return el;
  }

  /** Small inline SVG icon built with DOM calls (static paths, no server data). */
  function icon(paths) {
    const NS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    for (const d of paths) {
      const p = document.createElementNS(NS, 'path');
      p.setAttribute('d', d);
      svg.append(p);
    }
    return svg;
  }
  const HEART = ['M12 20.3s-7.4-4.5-9.2-9.1C1.7 8.3 3.5 4.9 6.9 4.6c2.1-.2 3.9.9 5.1 2.7 1.2-1.8 3-2.9 5.1-2.7 3.4.3 5.2 3.7 4.1 6.6-1.8 4.6-9.2 9.1-9.2 9.1z'];
  const RECALL = ['M3.5 12a8.5 8.5 0 1 0 2.6-6.1', 'M3.5 4.5v4h4', 'M12 8v4.2l2.8 1.8'];

  const isSafeImg = (u) => typeof u === 'string' && IMG_PREFIXES.some((p) => u.startsWith(p));
  const isSafeLink = (u) => typeof u === 'string' && u.startsWith(LINK_PREFIX);
  const topicKey = (id) => (Object.prototype.hasOwnProperty.call(KNOWN_TOPICS, id) ? id : 'other');
  const topicLabel = (id) => state.topicLabels[id] || KNOWN_TOPICS[id] || String(id);

  let toastTimer = null;
  function toast(message, ms = 3500) {
    const el = $('toast');
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }

  // ------------------------------------------------------------------
  // App state
  // ------------------------------------------------------------------
  const state = {
    topicLabels: {},          // id -> label, from /api/topics
    myTopics: [],             // the user's current topics (from /api/me or /api/session)
    seenCards: new Set(),     // card ids already in the feed this session (de-dupe)
    seenQuizzes: new Set(),   // quiz card_ids already shown this session
    items: [],                // feed item elements, in order (excludes the status slot)
    activeIndex: 0,           // index into items of the card currently on screen
    loading: false,           // a /api/feed request is in flight
    exhausted: false,         // server has nothing new for us
    emptyPages: 0,            // consecutive pages that added nothing new
    retryTimer: null,         // pending retry after a 429
    backoffMs: 0,
    feedGen: 0,               // bumped on reset so stale responses are ignored
    statusEl: null,           // trailing slot: skeleton / error / end-of-feed
    onStatus: false,          // the reader is currently looking at the status slot
  };

  // ------------------------------------------------------------------
  // 2. API wrapper
  // ------------------------------------------------------------------
  /** fetch JSON from our own API. Resolves to {status, ok, data, res};
   *  rejects only on network failure. */
  async function api(path, { method = 'GET', body } = {}) {
    const opts = { method, credentials: 'same-origin', headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    let data = null;
    if (res.status !== 204) {
      try { data = await res.json(); } catch { data = null; }
    }
    return { status: res.status, ok: res.ok, data, res };
  }

  /** Seconds from a Retry-After header, as ms (or null). */
  function retryAfterMs(res) {
    const s = Number(res && res.headers.get('Retry-After'));
    return Number.isFinite(s) && s > 0 ? Math.min(s, 120) * 1000 : null;
  }

  // ------------------------------------------------------------------
  // 3. Screens + boot
  // ------------------------------------------------------------------
  const SCREENS = ['boot', 'error', 'onboard', 'feed'];
  function showScreen(name) {
    for (const s of SCREENS) $('screen-' + s).hidden = s !== name;
  }

  let errorRetry = null;
  function showFatalError(title, body, onRetry) {
    $('error-title').textContent = title;
    $('error-body').textContent = body;
    errorRetry = onRetry;
    showScreen('error');
    $('error-retry').focus();
  }

  async function boot() {
    showScreen('boot');
    let me;
    try {
      me = await api('/api/me');
    } catch {
      showFatalError("Can't reach ReelRead", 'Check your connection, then try again.', boot);
      return;
    }
    if (me.status === 401) {
      openPicker({ firstRun: true });
      return;
    }
    if (!me.ok || !me.data || !Array.isArray(me.data.topics)) {
      showFatalError('Something went wrong', 'The server sent an unexpected reply. Try again in a moment.', boot);
      return;
    }
    state.myTopics = me.data.topics;
    loadTopicLabels(); // not awaited: only improves chip labels
    startFeed();
  }

  async function loadTopicLabels() {
    try {
      const r = await api('/api/topics');
      if (r.ok && Array.isArray(r.data)) {
        for (const t of r.data) {
          if (t && typeof t.id === 'string' && typeof t.label === 'string') state.topicLabels[t.id] = t.label;
        }
        return r.data;
      }
    } catch { /* fall back to KNOWN_TOPICS labels */ }
    return null;
  }

  // ------------------------------------------------------------------
  // 4. Onboarding / topic picker
  // ------------------------------------------------------------------
  const picker = {
    firstRun: true,
    selected: new Set(),
    busy: false,
  };

  async function openPicker({ firstRun }) {
    picker.firstRun = firstRun;
    picker.busy = false;
    $('onboard-cancel').hidden = firstRun;
    $('onboard-start').textContent = firstRun ? 'Start reading' : 'Save topics';
    $('onboard-title').textContent = firstRun
      ? 'Short reads from Wikipedia, one swipe at a time.'
      : 'Change your topics';
    $('onboard-sub').textContent = firstRun
      ? 'Pick up to five topics. The feed learns from what you linger on.'
      : 'Saving starts a fresh feed with your new mix.';
    setPickerStatus('Loading topics…');
    $('topic-picker').replaceChildren();
    $('onboard-start').disabled = true;
    showScreen('onboard');

    // Pre-fill from the server when changing topics.
    let prefill = [];
    if (!firstRun) {
      try {
        const me = await api('/api/me');
        if (me.status === 401) { picker.firstRun = true; $('onboard-cancel').hidden = true; }
        else if (me.ok && me.data && Array.isArray(me.data.topics)) prefill = me.data.topics;
      } catch { prefill = state.myTopics; }
    }
    picker.selected = new Set(prefill);

    const topics = await loadTopicLabels();
    if (!topics) {
      setPickerStatus("Couldn't load topics. ", true, 'Try again', () => openPicker({ firstRun: picker.firstRun }));
      return;
    }
    renderPicker(topics);
  }

  function setPickerStatus(text, isError = false, actionLabel, onAction) {
    const el = $('onboard-status');
    el.replaceChildren(text);
    el.classList.toggle('is-error', isError);
    if (actionLabel) {
      const b = h('button', { type: 'button', class: 'btn btn-ghost', text: actionLabel });
      b.addEventListener('click', onAction);
      el.append(b);
    }
  }

  function renderPicker(topics) {
    const box = $('topic-picker');
    box.replaceChildren();
    for (const t of topics) {
      if (!t || typeof t.id !== 'string') continue;
      const count = Number(t.count) || 0;
      if (count === 0) picker.selected.delete(t.id);
      const btn = h('button', {
        type: 'button',
        class: 'topic-choice',
        'data-topic': topicKey(t.id),
        'aria-pressed': picker.selected.has(t.id) ? 'true' : 'false',
        disabled: count === 0,
      },
      h('span', { class: 'topic-choice-dot', 'aria-hidden': 'true' }),
      h('span', { class: 'topic-choice-label', text: topicLabel(t.id) }),
      h('span', { class: 'topic-choice-count', text: count === 0 ? 'No cards yet' : `${count.toLocaleString()} cards` }));
      btn.dataset.id = t.id;
      btn.dataset.count = String(count);
      btn.addEventListener('click', () => toggleTopic(btn));
      box.append(btn);
    }
    syncPicker();
    const first = box.querySelector('.topic-choice:not(:disabled)');
    if (first && !picker.firstRun) first.focus();
  }

  function toggleTopic(btn) {
    const id = btn.dataset.id;
    if (picker.selected.has(id)) picker.selected.delete(id);
    else if (picker.selected.size < MAX_TOPICS) picker.selected.add(id);
    btn.setAttribute('aria-pressed', picker.selected.has(id) ? 'true' : 'false');
    syncPicker();
  }

  /** Enable/disable chips and the Start button to match the selection. */
  function syncPicker() {
    const n = picker.selected.size;
    for (const btn of $('topic-picker').children) {
      const empty = btn.dataset.count === '0';
      btn.disabled = empty || (n >= MAX_TOPICS && !picker.selected.has(btn.dataset.id));
    }
    $('onboard-start').disabled = n === 0 || picker.busy;
    setPickerStatus(n === 0 ? 'Pick at least one topic.' : `${n} of ${MAX_TOPICS} picked`);
  }

  async function saveTopics() {
    if (picker.busy || picker.selected.size === 0) return;
    picker.busy = true;
    $('onboard-start').disabled = true;
    setPickerStatus('Saving…');
    let r;
    try {
      r = await api('/api/session', { method: 'POST', body: { topics: [...picker.selected] } });
    } catch {
      r = null;
    }
    picker.busy = false;
    if (!r) { setPickerStatus("Can't reach ReelRead. Check your connection and try again.", true); $('onboard-start').disabled = false; return; }
    if (r.status === 429) { setPickerStatus('Too many tries. Wait a minute, then try again.', true); $('onboard-start').disabled = false; return; }
    if (r.status === 422) { setPickerStatus("One of those topics isn't available. Pick again.", true); $('onboard-start').disabled = false; return; }
    if (!r.ok) { setPickerStatus('Saving failed. Try again.', true); $('onboard-start').disabled = false; return; }

    state.myTopics = (r.data && Array.isArray(r.data.topics)) ? r.data.topics : [...picker.selected];
    resetFeed();
    startFeed();
  }

  $('onboard-start').addEventListener('click', saveTopics);
  $('onboard-cancel').addEventListener('click', () => { showScreen('feed'); $('btn-topics').focus(); });
  $('btn-topics').addEventListener('click', () => openPicker({ firstRun: false }));
  $('error-retry').addEventListener('click', () => { if (errorRetry) errorRetry(); });

  // ------------------------------------------------------------------
  // 5. Feed loading
  // ------------------------------------------------------------------
  const feedEl = $('feed');

  function startFeed() {
    showScreen('feed');
    if (!state.statusEl) {
      state.statusEl = h('section', { class: 'item' });
      feedEl.append(state.statusEl);
      observer.observe(state.statusEl);
      renderStatus('loading');
    }
    if (state.items.length === 0) loadMore();
    feedEl.focus({ preventScroll: true });
  }

  /** Throw away the current feed (used after changing topics). Flushes dwell first. */
  function resetFeed() {
    finalizeAllTimers();
    flushEvents();
    observer.disconnect();     // re-used: observe() is called again for new items
    textObserver.disconnect();
    feedEl.replaceChildren();
    feedEl.scrollTop = 0;
    clearTimeout(state.retryTimer);
    Object.assign(state, {
      items: [], activeIndex: 0, loading: false, exhausted: false, emptyPages: 0,
      retryTimer: null, backoffMs: 0, statusEl: null, onStatus: false, feedGen: state.feedGen + 1,
    });
  }

  async function loadMore() {
    if (state.loading || state.exhausted || state.retryTimer) return;
    state.loading = true;
    const gen = state.feedGen;
    let r;
    try {
      r = await api(`/api/feed?n=${FEED_PAGE}`);
    } catch {
      if (gen !== state.feedGen) return;
      state.loading = false;
      renderStatus('error');
      return;
    }
    if (gen !== state.feedGen) return; // the feed was reset while we waited
    state.loading = false;

    if (r.status === 401) {          // cookie gone: back to onboarding
      resetFeed();
      openPicker({ firstRun: true });
      return;
    }
    if (r.status === 429) {          // rate limited: back off, then retry quietly
      state.backoffMs = Math.min(Math.max(state.backoffMs * 2, 2000), 60000);
      const wait = retryAfterMs(r.res) || state.backoffMs;
      if (state.items.length - state.activeIndex <= 1) toast('Loading more cards in a moment…', Math.min(wait, 5000));
      state.retryTimer = setTimeout(() => { state.retryTimer = null; loadMore(); }, wait);
      return;
    }
    if (!r.ok || !r.data || !Array.isArray(r.data.items)) {
      renderStatus('error');
      return;
    }
    state.backoffMs = 0;

    const added = appendItems(r.data.items);
    if (added > 0) {
      state.emptyPages = 0;
      // Scroll snapping keeps the previously snapped element on screen. If the
      // reader was waiting on the skeleton (or this is the first page), that
      // would jump past the new cards, so snap to the first new card instead.
      if (state.onStatus || state.items.length === added) {
        state.items[state.items.length - added].scrollIntoView({ block: 'start', behavior: 'auto' });
      }
      renderStatus('loading');
      checkPrefetch(); // in case the page was tiny
      return;
    }
    // Nothing new. Either there's nothing at all, or the server started over
    // and sent cards we've already shown. Try once more, then stop.
    state.emptyPages += 1;
    if (r.data.items.length === 0 || state.emptyPages >= 2) {
      state.exhausted = true;
      renderStatus(state.items.length === 0 ? 'empty' : 'end');
    } else {
      loadMore();
    }
  }

  /** Prefetch when the reader is PREFETCH_AT items from the end. */
  function checkPrefetch() {
    const remaining = state.items.length - 1 - state.activeIndex;
    if (remaining <= PREFETCH_AT) loadMore();
  }

  /** Fill the trailing status slot: 'loading' skeleton, 'error', 'end' or 'empty'. */
  function renderStatus(kind) {
    const el = state.statusEl;
    if (!el) return;
    el.dataset.status = kind;
    el.className = 'item';
    el.removeAttribute('aria-busy');

    if (kind === 'loading') {
      el.classList.add('item-skeleton');
      el.setAttribute('aria-busy', 'true');
      el.setAttribute('aria-label', 'Loading more cards');
      el.replaceChildren(
        h('div', { class: 'card-media' }),
        h('div', { class: 'card-body' },
          h('span', { class: 'sk sk-chip' }),
          h('span', { class: 'sk sk-title' }),
          h('span', { class: 'sk sk-line' }),
          h('span', { class: 'sk sk-line' }),
          h('span', { class: 'sk sk-line' }),
          h('span', { class: 'sk sk-line short' })),
      );
      return;
    }

    const copy = {
      error: ["Can't load more cards", 'Check your connection, then try again.', 'Try again'],
      end: ["You've read everything here", 'Add more topics to keep going.', 'Change topics'],
      empty: ['No cards for these topics yet', 'Pick different topics, or check back after the next update.', 'Change topics'],
    }[kind];
    const btn = h('button', { type: 'button', class: 'btn btn-primary', text: copy[2] });
    btn.addEventListener('click', () => {
      if (kind === 'error') { renderStatus('loading'); loadMore(); }
      else openPicker({ firstRun: false });
    });
    el.classList.add('item-message');
    el.removeAttribute('aria-label');
    el.replaceChildren(h('div', { class: 'message' },
      h('h2', { class: 'message-title', text: copy[0] }),
      h('p', { class: 'message-body', text: copy[1] }),
      btn));
  }

  /** Validate, de-duplicate and render a page of items. Returns how many were added. */
  function appendItems(items) {
    let added = 0;
    for (const item of items) {
      let el = null;
      if (item && item.kind === 'card' && item.card && Number.isInteger(item.card.id)) {
        if (state.seenCards.has(item.card.id)) continue;
        state.seenCards.add(item.card.id);
        el = buildCard(item.card);
      } else if (item && item.kind === 'quiz' && item.quiz && Number.isInteger(item.quiz.card_id)) {
        if (state.seenQuizzes.has(item.quiz.card_id)) continue;
        el = buildQuiz(item.quiz);
        if (el) state.seenQuizzes.add(item.quiz.card_id);
      }
      if (!el) continue; // unknown or malformed item
      el.dataset.index = String(state.items.length);
      state.items.push(el);
      feedEl.insertBefore(el, state.statusEl);
      observer.observe(el);
      added += 1;
    }
    return added;
  }

  // ------------------------------------------------------------------
  // 6. Card + quiz rendering
  // ------------------------------------------------------------------
  function buildCard(card) {
    const title = typeof card.title === 'string' && card.title.trim() ? card.title : 'Untitled';
    const extract = typeof card.extract === 'string' ? card.extract : '';
    const topics = Array.isArray(card.topics) ? card.topics.filter((t) => typeof t === 'string') : [];
    const primary = topicKey(topics[0]);

    const el = h('article', { class: 'item item-card', 'aria-label': title, 'data-topic': primary });
    el.dataset.cardId = String(card.id);

    // Media: the photo if it's a Wikimedia upload, otherwise the typographic header.
    let media;
    if (isSafeImg(card.thumbnail)) {
      const img = h('img', { alt: '', decoding: 'async', loading: 'lazy' });
      img.src = card.thumbnail; // checked by isSafeImg above
      media = h('div', { class: 'card-media' }, img);
      img.addEventListener('error', () => media.replaceWith(buildArt(title, primary)), { once: true });
    } else {
      media = buildArt(title, primary);
    }

    const chips = h('ul', { class: 'chips', 'aria-label': 'Topics' });
    for (const t of topics) chips.append(h('li', { class: 'chip', 'data-topic': topicKey(t), text: topicLabel(t) }));

    // Extract: one <p> per paragraph. Scrolls inside the card when long.
    const text = h('div', { class: 'card-text', tabindex: '0', role: 'region', 'aria-label': 'Summary' });
    for (const para of extract.split(/\n+/)) {
      if (para.trim()) text.append(h('p', { text: para.trim() }));
    }
    text.addEventListener('scroll', () => updateClip(text), { passive: true });
    textObserver.observe(text);

    let link;
    if (isSafeLink(card.url)) {
      link = h('a', { class: 'card-link', target: '_blank', rel: 'noopener noreferrer', text: 'Read on Wikipedia' });
      link.href = card.url; // checked by isSafeLink above
    } else {
      link = h('span', { class: 'card-link-none', text: 'Link unavailable' });
    }

    const like = h('button', {
      type: 'button', class: 'like', 'aria-label': 'Like', 'aria-pressed': card.liked === true ? 'true' : 'false',
    }, icon(HEART));
    like.addEventListener('click', () => toggleLike(card.id, like));

    el.append(media, h('div', { class: 'card-body' },
      chips.childElementCount ? chips : null,
      h('h2', { class: 'card-title', text: title }),
      text,
      h('div', { class: 'card-foot' }, link, like)));
    return el;
  }

  /** Header for cards without a photo: ruled catalog card + big initial. */
  function buildArt(title, topic) {
    const letter = (title.match(/[\p{L}\p{N}]/u) || ['¶'])[0].toUpperCase();
    return h('div', { class: 'card-media card-art', 'data-topic': topic },
      h('span', { class: 'card-art-letter', 'aria-hidden': 'true', text: letter }));
  }

  /** Fade the bottom of the extract while there is more text below. */
  function updateClip(text) {
    const more = text.scrollHeight - text.scrollTop - text.clientHeight > 4;
    text.classList.toggle('is-clipped', more);
  }
  const textObserver = new ResizeObserver((entries) => {
    for (const e of entries) updateClip(e.target);
  });

  /** Optimistic like/unlike: flip the UI now, send the event, revert on failure.
   *  Requests for one button are chained so they reach the server in order. */
  function toggleLike(cardId, btn) {
    const liked = btn.getAttribute('aria-pressed') !== 'true';
    btn.setAttribute('aria-pressed', liked ? 'true' : 'false');
    btn.setAttribute('aria-label', liked ? 'Liked' : 'Like');
    btn.classList.remove('pop');
    if (liked && !reducedMotion.matches) { void btn.offsetWidth; btn.classList.add('pop'); }

    const send = async () => {
      let ok = false;
      try {
        const r = await api('/api/events', { method: 'POST', body: { events: [{ card_id: cardId, type: liked ? 'like' : 'unlike' }] } });
        ok = r.ok;
      } catch { ok = false; }
      if (!ok && btn.getAttribute('aria-pressed') === (liked ? 'true' : 'false')) {
        btn.setAttribute('aria-pressed', liked ? 'false' : 'true');
        btn.setAttribute('aria-label', liked ? 'Like' : 'Liked');
        toast(liked ? "Couldn't save that like. Try again." : "Couldn't remove that like. Try again.");
      }
    };
    btn._likeChain = (btn._likeChain || Promise.resolve()).then(send);
  }

  function buildQuiz(quiz) {
    const options = quiz.options;
    if (typeof quiz.question !== 'string' || !Array.isArray(options) || options.length !== 4
        || !options.every((o) => typeof o === 'string')) return null;

    const el = h('article', { class: 'item item-quiz', 'aria-label': 'Quick recall question' });
    const root = h('div', { class: 'quiz' });

    // Question text with each ___ drawn as a blank line.
    const q = h('p', { class: 'quiz-question' });
    quiz.question.split('___').forEach((part, i) => {
      if (i > 0) q.append(h('span', { class: 'blank' }, h('span', { class: 'sr-only', text: 'blank' })));
      q.append(part);
    });

    const feedback = h('p', { class: 'quiz-feedback', role: 'status', 'aria-live': 'polite' });
    const list = h('ul', { class: 'quiz-options' });
    const buttons = options.map((opt, i) => {
      const b = h('button', { type: 'button', class: 'quiz-option' },
        h('span', { class: 'quiz-option-text', text: opt }),
        h('span', { class: 'quiz-option-mark', 'aria-hidden': 'true' }));
      b.addEventListener('click', () => answerQuiz(quiz, i, buttons, root, feedback));
      list.append(h('li', null, b));
      return b;
    });

    root.append(
      h('p', { class: 'quiz-head' }, icon(RECALL), 'Quick recall'),
      q, list, feedback);
    el.append(root);
    return el;
  }

  async function answerQuiz(quiz, choice, buttons, root, feedback) {
    if (root.classList.contains('is-answered')) return;
    for (const b of buttons) b.disabled = true;
    buttons[choice].classList.add('is-chosen');
    feedback.className = 'quiz-feedback';
    feedback.textContent = 'Checking…';

    const unlock = (msg) => {
      for (const b of buttons) b.disabled = false;
      buttons[choice].classList.remove('is-chosen');
      feedback.className = 'quiz-feedback is-error';
      feedback.textContent = msg;
    };

    let r;
    try {
      r = await api('/api/quiz/answer', { method: 'POST', body: { card_id: quiz.card_id, choice } });
    } catch {
      unlock("Couldn't check your answer. Tap an option to try again.");
      return;
    }
    if (r.status === 429) { unlock('Too many answers at once. Try again in a moment.'); return; }
    if (r.status === 404) {
      feedback.className = 'quiz-feedback is-error';
      feedback.textContent = 'This question is no longer available.';
      return; // stays locked
    }
    const answer = r.data && r.data.answer_index;
    if (!r.ok || !Number.isInteger(answer) || answer < 0 || answer > 3) {
      unlock("Couldn't check your answer. Tap an option to try again.");
      return;
    }

    const correct = r.data.correct === true;
    root.classList.add('is-answered');
    buttons[answer].classList.add('is-correct');
    buttons[answer].lastChild.textContent = '✓';
    buttons[answer].setAttribute('aria-label', `${quiz.options[answer]}, correct answer`);
    if (!correct) {
      buttons[choice].classList.add('is-wrong');
      buttons[choice].lastChild.textContent = '✕';
      buttons[choice].setAttribute('aria-label', `${quiz.options[choice]}, your answer`);
    }
    feedback.className = 'quiz-feedback ' + (correct ? 'is-good' : 'is-bad');
    feedback.textContent = correct ? 'Nice — you remembered' : `It was ${quiz.options[answer]}`;
  }

  // ------------------------------------------------------------------
  // 7. Dwell tracking + event queue
  // ------------------------------------------------------------------
  // timers: card element -> { cardId, accum (ms), since (performance.now() or null) }
  // A card is in this map only while it is >= 60% visible.
  const timers = new Map();

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const el = entry.target;
      const visible = entry.isIntersecting && entry.intersectionRatio >= VIEW_THRESHOLD;

      if (el === state.statusEl) {
        state.onStatus = visible;
        // Reader reached the skeleton at the end: make sure a load is underway.
        if (visible && el.dataset.status === 'loading') loadMore();
        continue;
      }
      if (visible) {
        state.activeIndex = Number(el.dataset.index) || 0;
        if (el.dataset.cardId) startTimer(el);
        checkPrefetch();
      } else if (timers.has(el)) {
        stopTimer(el);
      }
    }
  }, { root: feedEl, threshold: VIEW_THRESHOLD });

  function startTimer(el) {
    if (timers.has(el)) return;
    timers.set(el, {
      cardId: Number(el.dataset.cardId),
      accum: 0,
      since: document.visibilityState === 'visible' ? performance.now() : null,
    });
  }

  // Cards that already had dwell sent this session. A view split by hide/show
  // can leave a short trailing piece; under SKIP_MS the ranker would read that
  // as a skip (-0.5), so later pieces shorter than SKIP_MS are dropped.
  const SKIP_MS = 1500;
  const dwellSent = new Set();

  function queueView(cardId, dwell) {
    if (dwell <= 0) return;
    if (dwellSent.has(cardId) && dwell < SKIP_MS) return; // leftover piece, not a skip
    dwellSent.add(cardId);
    queueEvent({ card_id: cardId, type: 'view', dwell_ms: dwell });
  }

  /** Card left the screen: queue its view with the total visible time. */
  function stopTimer(el) {
    const t = timers.get(el);
    if (!t) return;
    timers.delete(el);
    if (t.since !== null) t.accum += performance.now() - t.since;
    queueView(t.cardId, Math.round(t.accum));
  }

  function pauseTimers() {
    const now = performance.now();
    for (const t of timers.values()) {
      if (t.since !== null) { t.accum += now - t.since; t.since = null; }
    }
  }
  /** Page is being hidden: send the dwell so far for cards still on screen.
   *  Mobile browsers often kill background tabs without firing pagehide, so
   *  this is our last reliable chance. The timer stays in the map (the card is
   *  still on screen) but accum is reset to 0, so that time is never sent again;
   *  anything read after returning goes out later as a second view event. */
  function emitOpenTimers() {
    pauseTimers();
    for (const t of timers.values()) {
      const dwell = Math.round(t.accum);
      t.accum = 0;
      queueView(t.cardId, dwell);
    }
  }
  function resumeTimers() {
    const now = performance.now();
    for (const t of timers.values()) if (t.since === null) t.since = now;
  }
  function finalizeAllTimers() {
    for (const el of [...timers.keys()]) stopTimer(el);
  }

  let queue = [];
  let flushing = false;

  function queueEvent(ev) {
    queue.push(ev);
    if (queue.length > QUEUE_CAP) queue.splice(0, queue.length - QUEUE_CAP);
    if (queue.length >= FLUSH_AT) flushEvents();
  }

  /** Send queued events in batches of <= 50.
   *  beacon=true is for page hide/unload: sendBeacon survives the page going away. */
  async function flushEvents(beacon = false) {
    if (queue.length === 0) return;

    if (beacon) {
      while (queue.length) {
        const json = JSON.stringify({ events: queue.splice(0, BATCH_MAX) });
        let sent = false;
        try {
          sent = navigator.sendBeacon('/api/events', new Blob([json], { type: 'application/json' }));
        } catch { sent = false; }
        if (!sent) {
          fetch('/api/events', {
            method: 'POST', credentials: 'same-origin', keepalive: true,
            headers: { 'Content-Type': 'application/json' }, body: json,
          }).catch(() => {});
        }
      }
      return;
    }

    if (flushing) return;
    flushing = true;
    try {
      while (queue.length) {
        const batch = queue.splice(0, BATCH_MAX);
        let res = null;
        try {
          res = await fetch('/api/events', {
            method: 'POST', credentials: 'same-origin', keepalive: true,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ events: batch }),
          });
        } catch { res = null; }
        // Network error, rate limit or server error: put the batch back and wait
        // for the next tick. Other 4xx (401/413/422) won't succeed on retry: drop.
        if (!res || res.status === 429 || res.status >= 500) {
          queue = batch.concat(queue);
          break;
        }
      }
    } finally {
      flushing = false;
    }
  }

  setInterval(() => flushEvents(), FLUSH_EVERY_MS);

  // ------------------------------------------------------------------
  // 8. Keyboard, lifecycle, service worker
  // ------------------------------------------------------------------
  /** Move to the item at index i (status slot counts as the one after the last). */
  function goTo(i) {
    if (i < 0) return;
    const target = state.items[i] || (i === state.items.length ? state.statusEl : null);
    if (target) target.scrollIntoView({ behavior: reducedMotion.matches ? 'auto' : 'smooth', block: 'start' });
  }

  document.addEventListener('keydown', (e) => {
    if ($('screen-feed').hidden || e.altKey || e.ctrlKey || e.metaKey) return;
    const t = e.target;
    if (t instanceof Element && t.closest('input, textarea, select, [contenteditable="true"]')) return;

    let dir = 0;
    if (e.key === 'ArrowDown' || e.key === 'j' || e.key === 'PageDown') dir = 1;
    else if (e.key === 'ArrowUp' || e.key === 'k' || e.key === 'PageUp') dir = -1;
    else return;

    // Arrow keys inside a long, focused extract scroll the text first.
    if (t instanceof Element && t.classList.contains('card-text') && e.key.startsWith('Arrow')) {
      const canScroll = dir > 0 ? t.scrollTop + t.clientHeight < t.scrollHeight - 1 : t.scrollTop > 0;
      if (canScroll) return;
    }
    e.preventDefault();
    goTo(state.activeIndex + dir);
  });

  // Page hidden (app switch, tab change, screen off): stop the clocks, queue
  // the open card's dwell so far, then beacon everything. When we come back,
  // timers restart for cards that are still >= 60% visible.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      emitOpenTimers();
      flushEvents(true);
    } else {
      resumeTimers();
    }
  });

  // Page going away (or into the back/forward cache): close the open card's view.
  window.addEventListener('pagehide', () => {
    finalizeAllTimers();
    flushEvents(true);
  });
  window.addEventListener('pageshow', (e) => {
    if (!e.persisted) return;
    const el = state.items[state.activeIndex];
    if (el && el.dataset.cardId && !$('screen-feed').hidden) startTimer(el);
  });

  // Back online after a failed load: try again.
  window.addEventListener('online', () => {
    if (state.statusEl && state.statusEl.dataset.status === 'error') {
      renderStatus('loading');
      loadMore();
    }
    flushEvents();
  });

  window.addEventListener('resize', () => {
    for (const el of feedEl.querySelectorAll('.card-text')) updateClip(el);
  });

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js').catch(() => { /* offline shell is optional */ });
    });
  }

  boot();
})();
