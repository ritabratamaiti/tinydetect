const fs=require('fs');const P="C:\\Projects\\TinyDetect";const log=m=>fs.appendFileSync(P+'/logs/human2.log',new Date().toISOString().slice(11,19)+' '+m+'\n');const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const fresh=JSON.parse(fs.readFileSync(P+'/fresh/human_text.json','utf8')).map(x=>x.t.slice(0,80).toLowerCase());
const leak=t=>{const l=t.toLowerCase();return fresh.some(f=>l.includes(f.slice(10,60)));};
async function rows(ds,cfg,split,off,len){for(let k=0;k<4;k++){try{const r=await fetch('https://datasets-server.huggingface.co/rows?dataset='+ds+'&config='+cfg+'&split='+split+'&offset='+off+'&length='+len);if(r.ok)return (await r.json()).rows.map(x=>x.row);log(ds+' '+r.status);await sleep(2500*(k+1));}catch(e){await sleep(2000);}}return [];}
async function split(ds){const j=await (await fetch('https://datasets-server.huggingface.co/splits?dataset='+ds)).json();const s=(j.splits||[]).find(x=>x.split==='train')||(j.splits||[])[0];return s;}
async function size(ds,cfg){try{const j=await (await fetch('https://datasets-server.huggingface.co/size?dataset='+ds)).json();const c=(j.size.configs||[]).find(x=>x.config===cfg);return (c||j.size.dataset).num_rows;}catch(e){return 100000;}}
const out=[];
async function take(ds,dom,field,want,cfgOverride){const s=await split(ds);if(!s){log('nosplit '+ds);return;}const cfg=cfgOverride||s.config;const N=await size(ds,cfg);let got=0,tries=0;while(got<want&&tries<40){tries++;const off=Math.floor(Math.random()*Math.max(1,N-100));const r=await rows(ds,cfg,s.split,off,100);for(const x of r){let t=typeof field==='function'?field(x):x[field];if(!t)continue;t=String(t).replace(/\s+/g,' ').trim();if(t.length<400)continue;if(leak(t))continue;out.push({t:t.slice(0,2000),y:0,m:'human',a:'none',d:dom});got++;if(got>=want)break;}await sleep(300);}log(dom+' '+got);}
(async()=>{try{
 await take('Salesforce/wikitext','wikipedia_2016','text',450,'wikitext-103-raw-v1');
 await take('vblagoje/cc_news','news_2017','text',450);
 await take('gfissore/arxiv-abstracts-2021','arxiv_abstract','abstract',350);
 await take('Yelp/yelp_review_full','yelp_review','text',350);
 await take('webis/tldr-17','reddit_2016','content',350);
 fs.writeFileSync(P+'/data/text_human_extra.jsonl',out.map(x=>JSON.stringify(x)).join('\n'));log('HUMAN2DONE '+out.length);
}catch(e){log('ERR '+e.stack)}})();