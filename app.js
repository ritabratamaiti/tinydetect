// TinyDetect: fully client-side. Custom bits: pruned WordPiece vocab + int4 (per-row scaled) embeddings decoded in JS.
ort.env.wasm.wasmPaths=new URL('ort/',location.href).href; ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||1):1;
const $=id=>document.getElementById(id);
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));$('p-text').style.display=b.dataset.t==='text'?'':'none';$('p-image').style.display=b.dataset.t==='image'?'':'none';if(b.dataset.t==='image')loadImage();});
function f16(h){const s=(h&0x8000)?-1:1,e=(h>>10)&31,m=h&1023;if(e===0)return s*Math.pow(2,-14)*(m/1024);if(e===31)return m?NaN:s*Infinity;return s*Math.pow(2,e-15)*(1+m/1024);}
async function bin(u){const r=await fetch(u);if(!r.ok)throw new Error(u+' '+r.status);return new Uint8Array(await r.arrayBuffer());}
// ---------- BERT uncased tokenizer (basic + wordpiece) over a pruned vocab
let TX=null;
function isPunct(c){const cp=c.codePointAt(0);if((cp>=33&&cp<=47)||(cp>=58&&cp<=64)||(cp>=91&&cp<=96)||(cp>=123&&cp<=126))return true;return /\p{P}/u.test(c);}
function basic(text){text=text.replace(/[\u0000\ufffd]|[\p{Cc}\p{Cf}]/gu,m=>/[\t\n\r]/.test(m)?' ':''); // strips zero-width chars like HF does (defeats the zero-width-space attack)
 text=text.replace(/([\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF])/g,' $1 ');text=text.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu,'');
 const out=[];for(const w of text.split(/\s+/)){if(!w)continue;let cur='';for(const ch of w){if(isPunct(ch)){if(cur)out.push(cur);out.push(ch);cur='';}else cur+=ch;}if(cur)out.push(cur);}return out;}
function wordpiece(w,V){if(w.length>100)return['[UNK]'];const out=[];let s=0;while(s<w.length){let e=w.length,sub=null;while(s<e){let t=w.slice(s,e);if(s>0)t='##'+t;if(V.has(t)){sub=t;break;}e--;}if(sub===null)return['[UNK]'];out.push(sub);s=e;}return out;}
async function loadText(){
 const [voc,emb,sc]=await Promise.all([fetch('models/text_vocab.json').then(r=>r.json()),bin('models/text_emb_int4.bin'),bin('models/text_emb_scale_f16.bin')]);
 const V=new Map(voc.tokens.map((t,i)=>[t,i]));const s16=new Uint16Array(sc.buffer);const scale=Float32Array.from(s16,f16);
 const sess=await ort.InferenceSession.create('models/text_body_int8.onnx',{executionProviders:['wasm']});
 TX={V,emb,scale,dim:voc.dim,unk:voc.unk,cls:voc.cls,sep:voc.sep,max:voc.max_len,sess};
}
function embed(ids){const {emb,scale,dim}=TX;const half=dim/2;const out=new Float32Array(ids.length*dim);
 ids.forEach((id,r)=>{const s=scale[id],o=r*dim,b=id*half;for(let j=0;j<half;j++){const v=emb[b+j];out[o+2*j]=((v&15)-8)*s;out[o+2*j+1]=((v>>4)-8)*s;}});return out;}
async function scoreText(text){
 const pieces=[];for(const w of basic(text))for(const p of wordpiece(w,TX.V))pieces.push(TX.V.has(p)?TX.V.get(p):TX.unk);
 if(pieces.length<8)throw new Error('Please paste a longer passage.');
 const W=TX.max-2,wins=[];for(let i=0;i<pieces.length&&wins.length<6;i+=W)wins.push(pieces.slice(i,i+W));
 if(wins.length>1&&wins[wins.length-1].length<40)wins.pop();
 const probs=[];for(const w of wins){const ids=[TX.cls,...w,TX.sep];const e=new ort.Tensor('float32',embed(ids),[1,ids.length,TX.dim]);const m=new ort.Tensor('int64',BigInt64Array.from(ids,()=>1n),[1,ids.length]);
  const o=await TX.sess.run({inputs_embeds:e,attention_mask:m});probs.push(o.probs.data[1]);}
 return {p:probs.reduce((a,b)=>a+b,0)/probs.length,n:pieces.length,w:probs.length};}
// ---------- image
let IM=null,imgLoading=null;
function loadImage(){if(!imgLoading)imgLoading=ort.InferenceSession.create('models/image_int8.onnx',{executionProviders:['wasm']}).then(s=>{IM=s;$('st-image').textContent='Model ready.';});return imgLoading;}
async function scoreImage(img){await loadImage();const c=document.createElement('canvas');c.width=c.height=224;const g=c.getContext('2d');g.imageSmoothingQuality='high';g.drawImage(img,0,0,224,224);
 const d=g.getImageData(0,0,224,224).data,N=224*224,x=new Float32Array(3*N);for(let i=0;i<N;i++){x[i]=d[4*i+2]/255;x[N+i]=d[4*i+1]/255;x[2*N+i]=d[4*i]/255;}
 const o=await IM.run({pixel_values:new ort.Tensor('float32',x,[1,3,224,224])});const l=o.logits.data;const m=Math.max(l[0],l[1]);const a=Math.exp(l[0]-m),b=Math.exp(l[1]-m);return b/(a+b);}
// ---------- UI
function show(el,p,extra){const pct=Math.round(p*100);const v=p>=0.8?['Likely AI-generated','var(--ai)']:p>=0.5?['Leaning AI-generated','#b45309']:p>=0.25?['Uncertain','var(--mut)']:['Likely human-made','var(--hu)'];
 el.style.display='block';el.innerHTML='<div class="bar"><div class="mark" style="left:calc('+pct+'% - 2px)"></div></div><div class="lab"><span>human</span><span>AI</span></div><div class="verdict" style="color:'+v[1]+'">'+v[0]+' · '+pct+'%</div><div class="note">'+extra+'</div>';}
$('gotext').onclick=async()=>{const t=$('txt').value.trim();if(!t)return;$('gotext').disabled=true;const t0=performance.now();
 try{const r=await scoreText(t);show($('r-text'),r.p,r.n+' tokens, '+r.w+' window(s), '+Math.round(performance.now()-t0)+' ms on your device. Short or heavily edited text is much less reliable.');}catch(e){$('st-text').textContent=e.message;}$('gotext').disabled=false;};
const drop=$('drop'),file=$('file');drop.onclick=()=>file.click();['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hov');}));['dragleave','drop'].forEach(e=>drop.addEventListener(e,()=>drop.classList.remove('hov')));
drop.addEventListener('drop',ev=>{ev.preventDefault();if(ev.dataTransfer.files[0])handle(ev.dataTransfer.files[0]);});file.onchange=()=>file.files[0]&&handle(file.files[0]);
function handle(f){const u=URL.createObjectURL(f);const img=$('imgprev');img.onload=async()=>{$('st-image').textContent='Analysing…';const t0=performance.now();try{const p=await scoreImage(img);$('st-image').textContent='';show($('r-image'),p,'Ran in '+Math.round(performance.now()-t0)+' ms on your device. Screenshots, heavy edits and new image generators can fool it.');}catch(e){$('st-image').textContent=e.message;}};img.src=u;}
// ---------- boot
(async()=>{try{const t0=performance.now();await loadText();$('gotext').disabled=false;$('gotext').textContent='Check text';$('st-text').textContent='Text model loaded in '+Math.round(performance.now()-t0)+' ms.';}catch(e){$('gotext').textContent='Model failed to load';$('st-text').textContent=e.message;}
 try{const R=await fetch('results.json').then(r=>r.json());$('pills').innerHTML=R.pills.map(p=>'<span class="pill">'+p+'</span>').join('');$('stats').innerHTML=R.html;}catch(e){$('stats').textContent='';}})();
