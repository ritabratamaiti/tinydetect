// Pick the ONNX Runtime build: WebGPU-capable browsers get the WebGPU build, others the smaller WASM-only build.
(async()=>{
 if('serviceWorker' in navigator&&!self.crossOriginIsolated&&(location.protocol==='https:'||location.hostname==='127.0.0.1'||location.hostname==='localhost')){
  try{const reg=await navigator.serviceWorker.register('coi-sw.js');
   if(!navigator.serviceWorker.controller&&!sessionStorage.getItem('coi-reload')){sessionStorage.setItem('coi-reload','1');await navigator.serviceWorker.ready;location.reload();return;}}catch(e){console.warn('coi sw',e);}}
 let gpu=false;try{gpu=!!(navigator.gpu&&await navigator.gpu.requestAdapter());}catch(e){}
 if(new URLSearchParams(location.search).get('ep')==='wasm')gpu=false;
 window.ORT_DIR=gpu?'ort_gpu/':'ort/';window.USE_WEBGPU=gpu;
 const load=src=>new Promise((ok,err)=>{const s=document.createElement('script');s.src=src;s.onload=ok;s.onerror=()=>err(new Error('failed to load '+src));document.body.appendChild(s);});
 await load(window.ORT_DIR+(gpu?'ort.webgpu.min.js':'ort.wasm.min.js'));await load('bpe.js');await load('app.js');
})();
