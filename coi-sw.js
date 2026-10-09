// 1) Adds cross-origin isolation headers so ONNX Runtime can use WASM threads on hosts that can't set headers (GitHub Pages).
//    COEP 'credentialless' keeps no-cors third-party loads (Google Fonts) working; browsers without it stay single-threaded.
// 2) Keeps model and runtime files in a versioned cache, so repeat visits don't re-download ~115 MB. Bump VERSION when models change.
const VERSION='td-v2';
const BIG=/\/(models|ort|ort_gpu)\/[^/]+$/;
self.addEventListener('install',()=>self.skipWaiting());
self.addEventListener('activate',e=>e.waitUntil((async()=>{for(const k of await caches.keys())if(k!==VERSION)await caches.delete(k);await self.clients.claim();})()));
function isolate(res){if(res.status===0)return res;const h=new Headers(res.headers);h.set('Cross-Origin-Opener-Policy','same-origin');h.set('Cross-Origin-Embedder-Policy','credentialless');h.set('Cross-Origin-Resource-Policy','same-origin');
 return new Response(res.body,{status:res.status,statusText:res.statusText,headers:h});}
self.addEventListener('fetch',e=>{const r=e.request;if(r.method!=='GET'||(r.cache==='only-if-cached'&&r.mode!=='same-origin'))return;
 const u=new URL(r.url);if(u.origin!==self.location.origin)return;
 if(BIG.test(u.pathname)){e.respondWith((async()=>{const c=await caches.open(VERSION);const key=u.origin+u.pathname;let res=await c.match(key);
  if(!res){res=await fetch(key);if(res.ok)await c.put(key,res.clone());}return isolate(res);})());return;}
 e.respondWith(fetch(r).then(isolate));});
