const fs=require('fs');const P="C:\\Projects\\TinyDetect";const log=m=>fs.appendFileSync(P+'/logs/data.log',new Date().toISOString().slice(11,19)+' '+m+'\n');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function rows(ds,cfg,split,off,len){for(let t=0;t<4;t++){try{const r=await fetch('https://datasets-server.huggingface.co/rows?dataset='+ds+'&config='+cfg+'&split='+split+'&offset='+off+'&length='+len);if(r.ok)return (await r.json()).rows.map(x=>x.row);await sleep(1500*(t+1));}catch(e){await sleep(1500);}}return [];}
async function pool(items,n,fn){let i=0;await Promise.all(Array.from({length:n},async()=>{while(i<items.length){const k=i++;await fn(items[k],k);}}));}
(async()=>{
 // TEXT: RAID train, 2.27M rows of 'raid/train'. sample 120 random windows of 50
 const N=2270000;const offs=Array.from({length:140},(_,k)=>Math.floor(((k*7919)%140)/140*N+Math.random()*(N/140-60)));
 const out=[];await pool(offs,8,async o=>{const r=await rows('liamdugan/raid','raid','train',o,50);for(const x of r){if(!x.generation||x.generation.length<200)continue;out.push({t:x.generation.slice(0,2000),y:x.model==='human'?0:1,m:x.model,a:x.attack,d:x.domain});}});
 // extra human: RAID human rows are rarer; pull dmitva human_text (student essays) too
 const hoffs=Array.from({length:30},(_,k)=>Math.floor(Math.random()*990000));await pool(hoffs,6,async o=>{const r=await rows('dmitva/human_ai_generated_text','default','train',o,40);for(const x of r){if(x.human_text&&x.human_text.length>200)out.push({t:x.human_text.slice(0,2000),y:0,m:'human',a:'none',d:'essay_dmitva'});if(x.ai_text&&x.ai_text.length>200&&Math.random()<0.5)out.push({t:x.ai_text.slice(0,2000),y:1,m:'dmitva_ai',a:'none',d:'essay_dmitva'});}});
 fs.writeFileSync(P+'/data/text_raw.jsonl',out.map(x=>JSON.stringify(x)).join('\n'));
 const hc=out.filter(x=>x.y===0).length;log('TEXT done n='+out.length+' human='+hc+' ai='+(out.length-hc));
 // IMAGES: Hemg train 152710 rows (label 0=AiArtData,1=RealArt) -> ai=1 for us
 fs.mkdirSync(P+'/data/img/train',{recursive:true});fs.mkdirSync(P+'/data/img/test',{recursive:true});
 const meta=[];const iw=Array.from({length:60},(_,k)=>Math.floor(Math.random()*152600));let cnt=0;
 await pool(iw,6,async o=>{const r=await rows('Hemg/AI-Generated-vs-Real-Images-Datasets','default','train',o,60);await pool(r,6,async x=>{try{const res=await fetch(x.image.src);if(!res.ok)return;const b=Buffer.from(await res.arrayBuffer());const f='h'+(cnt++)+'.jpg';fs.writeFileSync(P+'/data/img/train/'+f,b);meta.push({f:'train/'+f,y:x.label===0?1:0,src:'hemg'});}catch(e){}});});
 log('IMG hemg n='+meta.length);
 // test set: eddyfox 2k (labels ai=0, real=1 -> ai=1 for us), different source = generalisation check
 for(const sp of ['test','train']){let off=0;while(off<1000){const r=await rows('eddyfox8812/ai-vs-real-2k-images','default',sp,off,100);if(!r.length)break;await pool(r,6,async x=>{try{const res=await fetch(x.image.src);if(!res.ok)return;const b=Buffer.from(await res.arrayBuffer());const f='e'+(cnt++)+'.jpg';fs.writeFileSync(P+'/data/img/test/'+f,b);meta.push({f:'test/'+f,y:x.label===0?1:0,src:'eddyfox'});}catch(e){}});off+=100;if(meta.filter(m=>m.src==='eddyfox').length>=600)break;}if(meta.filter(m=>m.src==='eddyfox').length>=600)break;}
 fs.writeFileSync(P+'/data/img_meta.json',JSON.stringify(meta));log('IMG done total='+meta.length+' eddyfox='+meta.filter(m=>m.src==='eddyfox').length);log('ALLDONE');
})().catch(e=>log('ERR '+e.stack));