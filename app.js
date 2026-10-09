// TinyDetect: fully client-side. Text = Fakespot RoBERTa (tuned, int4 embeddings + mixed-precision body); image = Community Forensics ViT-S.
ort.env.wasm.wasmPaths=new URL(window.ORT_DIR||'ort/',location.href).href; ort.env.wasm.numThreads=self.crossOriginIsolated?Math.min(4,navigator.hardwareConcurrency||1):1;
const EP=window.USE_WEBGPU?['webgpu','wasm']:['wasm'];
const $=id=>document.getElementById(id);
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===b));$('p-text').style.display=b.dataset.t==='text'?'':'none';$('p-image').style.display=b.dataset.t==='image'?'':'none';if(b.dataset.t==='image')loadImage().then(()=>$('st-image').textContent='');});
function f16(h){const s=(h&0x8000)?-1:1,e=(h>>10)&31,m=h&1023;if(e===0)return s*Math.pow(2,-14)*(m/1024);if(e===31)return m?NaN:s*Infinity;return s*Math.pow(2,e-15)*(1+m/1024);}
async function bin(u){const r=await fetch(u);if(!r.ok)throw new Error(u+' '+r.status);return new Uint8Array(await r.arrayBuffer());}
// Port of clean_text() from fakespot-ai/roberta-base-ai-text-detection-v1 utils.py (Apache-2.0): the detector was trained on text cleaned this way.
function unescapeHTML(s){const t=document.createElement('textarea');t.innerHTML=s;return t.value;}
function cleanMarkdown(s){s=s.replace(/```[\s\S]*?```/g,'');s=s.replace(/`[^`]*`/g,'');s=s.replace(/!\[[^\n]*?\]\([^\n]*?\)/g,'');s=s.replace(/\[([^\]]+)\]\([^\n]*?\)/g,'$1');
 s=s.replace(/(\*\*|__)([^\n]*?)\1/g,'$2');s=s.replace(/(\*|_)([^\n]*?)\1/g,'$2');s=s.replace(/#+ /g,'');s=s.replace(/^>.*$/gm,'');s=s.replace(/^(\s*[-*+]|\d+\.)\s+/gm,'');s=s.replace(/^\s*[-*_]{3,}\s*$/gm,'');
 s=s.replace(/\|[^\n]*?\|/g,'');s=s.replace(/<[^\n]*?>/g,'');return unescapeHTML(s);}
function cleanText(t){t=cleanMarkdown(t);t=t.replace(/\n/g,' ').replace(/\t/g,' ').split('^M').join(' ').replace(/\r/g,' ').split(' ,').join(',');return t.replace(/ +/g,' ');}
let TX=null;
// Text: Fakespot RoBERTa-base detector (Apache-2.0), lightly tuned. Byte-level BPE in JS; int4 word embeddings
// (one fp16 scale per 32 values) decoded in JS and fed to the ONNX body as inputs_embeds.
const TXT={body:'models/text_body.onnx',emb:'models/text_emb_int4.bin',scale:'models/text_emb_scale_f16.bin',bpe:'models/text_bpe.json',dim:768,block:32,win:510,maxWin:4,a:1,b:0};
async function loadText(){
 const [bpe,emb,sc]=await Promise.all([fetch(TXT.bpe).then(r=>r.json()),bin(TXT.emb),bin(TXT.scale)]);
 const tok=makeBPE(bpe);const s16=new Uint16Array(sc.buffer,sc.byteOffset,sc.byteLength/2);const scale=Float32Array.from(s16,f16);
 const sess=await ort.InferenceSession.create(TXT.body,{executionProviders:EP});
 TX={tok,emb,scale,sess};
 const w=new ort.Tensor('float32',new Float32Array(8*TXT.dim),[1,8,TXT.dim]);await sess.run({inputs_embeds:w,attention_mask:new ort.Tensor('int64',new BigInt64Array(8).fill(1n),[1,8])});// warm-up
}
function embed(ids){const D=TXT.dim,B=TXT.block,nb=D/B,half=D/2,{emb,scale}=TX;const out=new Float32Array(ids.length*D);
 ids.forEach((id,r)=>{const o=r*D,base=id*half,sb=id*nb;for(let j=0;j<half;j++){const v=emb[base+j],k=2*j,s=scale[sb+((k/B)|0)];out[o+k]=((v&15)-8)*s;out[o+k+1]=((v>>4)-8)*s;}});return out;}
async function scoreText(text){
 const ids=TX.tok.encode(cleanText(text));if(ids.length<8)throw new Error('Please paste a longer passage.');
 const W=TXT.win,wins=[];for(let i=0;i<ids.length&&wins.length<TXT.maxWin;i+=W)wins.push(ids.slice(i,i+W));
 if(wins.length>1&&wins[wins.length-1].length<64)wins.pop();
 const logits=[];for(const w of wins){const x=[TX.tok.bos,...w,TX.tok.eos];
  const o=await TX.sess.run({inputs_embeds:new ort.Tensor('float32',embed(x),[1,x.length,TXT.dim]),attention_mask:new ort.Tensor('int64',new BigInt64Array(x.length).fill(1n),[1,x.length])});
  const p=Math.min(Math.max(o.probs.data[1],1e-6),1-1e-6);logits.push(Math.log(p/(1-p)));}
 const l=logits.reduce((a,b)=>a+b,0)/logits.length;return {p:1/(1+Math.exp(-(TXT.a*l+TXT.b))),n:ids.length,w:logits.length};}
// ---------- image
let IM=null,imgLoading=null;
// Image: Community Forensics ViT-S/16 @384 (MIT), 8-bit block weights. Preprocess = resize short side to 440, centre crop 384, ImageNet norm.
const IMG={file:'models/image_cf384_nb8.onnx',size:384,resize:440,mean:[0.485,0.456,0.406],std:[0.229,0.224,0.225],a:0.5872,b:2.9691};
function loadImage(){if(!imgLoading)imgLoading=ort.InferenceSession.create(IMG.file,{executionProviders:EP}).then(async s=>{IM=s;const S=IMG.size;await s.run({pixel_values:new ort.Tensor('float32',new Float32Array(3*S*S),[1,3,S,S])});});/* warm-up run compiles WebGPU shaders before the first real check */return imgLoading;}
async function scoreImage(img){await loadImage();const S=IMG.size,w=img.naturalWidth,h=img.naturalHeight,k=IMG.resize/Math.min(w,h),cw=S/k,ch=S/k;
 const c=document.createElement('canvas');c.width=c.height=S;const g=c.getContext('2d');g.imageSmoothingEnabled=true;g.imageSmoothingQuality='high';g.drawImage(img,(w-cw)/2,(h-ch)/2,cw,ch,0,0,S,S);
 const d=g.getImageData(0,0,S,S).data,N=S*S,x=new Float32Array(3*N),[m0,m1,m2]=IMG.mean,[s0,s1,s2]=IMG.std;for(let j=0;j<N;j++){x[j]=(d[4*j]/255-m0)/s0;x[N+j]=(d[4*j+1]/255-m1)/s1;x[2*N+j]=(d[4*j+2]/255-m2)/s2;}
 const o=await IM.run({pixel_values:new ort.Tensor('float32',x,[1,3,S,S])});const l=o.logits.data[0];return 1/(1+Math.exp(-(IMG.a*l+IMG.b)));}
// ---------- UI
const EX={human:"That was a dismal revelation to me; for my memory was never loaded with anything but blank cartridges. However, I did not feel discouraged long. I judged that it was best to make some allowances, for doubtless Mr. Bixby was 'stretching.' Presently he pulled a rope and struck a few strokes on the big bell. The stars were all gone now, and the night was as black as ink. I could hear the wheels churn along the bank, but I was not entirely certain that I could see the shore.",
ai:"Remote work has fundamentally changed how teams collaborate. While many organizations initially viewed it as a temporary measure, it has evolved into a long-term strategy for attracting and retaining talent. Employees value the flexibility to structure their day around both professional and personal commitments, and studies consistently show that productivity can remain high when clear expectations are established. However, remote work also presents challenges, including feelings of isolation and difficulties in maintaining company culture."};
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{$('txt').value=EX[c.dataset.ex];$('gotext').click();});
function show(el,p,meta){const pct=Math.round(p*100);const v=p>=0.9?['AI-generated','var(--red)']:p>=0.6?['Possibly AI','var(--amb)']:p>=0.3?['Unclear','var(--mut)']:['Human','var(--grn)'];
 el.style.display='block';el.innerHTML='<div class="verdict"><span class="vlabel" style="color:'+v[1]+'">'+v[0]+'</span><span class="vpct">'+pct+'% AI likelihood</span></div><div class="meter"><div class="fill" style="width:'+Math.max(3,pct)+'%;background:'+v[1]+'"></div></div><div class="scale"><span>Human</span><span>AI</span></div><div class="small">'+meta+'</div>';}
$('gotext').onclick=async()=>{const t=$('txt').value.trim();if(!t)return;$('gotext').disabled=true;const t0=performance.now();
 try{const r=await scoreText(t);show($('r-text'),r.p,'Checked '+r.n+' tokens in '+Math.round(performance.now()-t0)+' ms. Short or heavily edited text is harder to judge.');}catch(e){$('st-text').textContent=e.message;}$('gotext').disabled=false;};
const drop=$('drop'),file=$('file');drop.onclick=()=>file.click();['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hov');}));['dragleave','drop'].forEach(e=>drop.addEventListener(e,()=>drop.classList.remove('hov')));
drop.addEventListener('drop',ev=>{ev.preventDefault();if(ev.dataTransfer.files[0])handle(ev.dataTransfer.files[0]);});file.onchange=()=>file.files[0]&&handle(file.files[0]);
function handle(f){const u=URL.createObjectURL(f);const img=$('imgprev');img.style.display='block';img.onload=async()=>{$('st-image').textContent='Checking…';const t0=performance.now();try{const p=await scoreImage(img);$('st-image').textContent='';show($('r-image'),p,'Checked in '+Math.round(performance.now()-t0)+' ms. Screenshots, crops and heavy edits are harder to judge.');}catch(e){$('st-image').textContent=e.message;}};img.src=u;}
// ---------- boot
(async()=>{try{const t0=performance.now();await loadText();$('gotext').disabled=false;$('st-text').textContent='Ready';}catch(e){$('st-text').textContent='Model failed to load: '+e.message;}
 try{const R=await fetch('results.json').then(r=>r.json());$('bench').innerHTML=R.html;}catch(e){}})();
