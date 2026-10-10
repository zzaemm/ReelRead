/* ReelRead service worker
 *
 * - App shell (html/css/js/manifest/icons): cache-first.
 * - /api/*: network-only, never cached (we simply don't intercept it).
 * - Anything else (e.g. Wikimedia thumbnails): left to the browser.
 *
 * Shell is cache-first, so changes to the shell only reach users when
 * CACHE_VERSION changes. Bump it on every frontend release.
 */
'use strict';

const CACHE_VERSION = 'v2';
const CACHE_NAME = `reelread-shell-${CACHE_VERSION}`;

const SHELL = [
  '/',
  '/index.html',
  '/style.css',
  '/app.js',
  '/manifest.webmanifest',
  '/icons/icon.svg',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting()),
  );
});

// Delete caches from older versions.
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k.startsWith('reelread-') && k !== CACHE_NAME).map((k) => caches.delete(k)),
      ))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;   // thumbnails etc.: browser default
  if (url.pathname.startsWith('/api/')) return;       // network-only, never cached

  if (SHELL.includes(url.pathname)) {
    event.respondWith(
      caches.match(url.pathname).then((hit) => hit || fetch(req)),
    );
  }
  // Other same-origin paths fall through to the network.
});
