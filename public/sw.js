const CACHE_NAME = 'intraday-ai-v3';

self.addEventListener('install', (e) => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(clients.claim()));

self.addEventListener('fetch', (e) => {
  if (e.request.url.includes('raw.githubusercontent.com') || e.request.url.includes('/data_store/')) {
    e.respondWith(fetch(e.request, { cache: 'no-store' }));
    return;
  }
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});