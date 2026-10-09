// GPT-2 / RoBERTa byte-level BPE tokenizer (matches Hugging Face "ByteLevel" + BPE, add_prefix_space=false).
(function(root){
function byteToUnicode(){const bs=[];for(let i=33;i<=126;i++)bs.push(i);for(let i=161;i<=172;i++)bs.push(i);for(let i=174;i<=255;i++)bs.push(i);const cs=bs.slice();let n=0;for(let b=0;b<256;b++)if(!bs.includes(b)){bs.push(b);cs.push(256+n);n++;}const m={};bs.forEach((b,i)=>m[b]=String.fromCharCode(cs[i]));return m;}
const PAT=/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
function make(data){const B2U=byteToUnicode(),enc=new TextEncoder(),V=new Map(data.tokens.map((t,i)=>[t,i])),R=new Map(data.merges.map((m,i)=>[m,i])),cache=new Map();
 function bpe(w){if(cache.has(w))return cache.get(w);let parts=Array.from(w);while(parts.length>1){let best=-1,bi=-1;for(let i=0;i<parts.length-1;i++){const r=R.get(parts[i]+' '+parts[i+1]);if(r!==undefined&&(best<0||r<best)){best=r;bi=i;}}if(bi<0)break;const a=parts[bi],b=parts[bi+1],out=[];for(let i=0;i<parts.length;){if(i<parts.length-1&&parts[i]===a&&parts[i+1]===b){out.push(a+b);i+=2;}else{out.push(parts[i]);i++;}}parts=out;}if(cache.size<20000)cache.set(w,parts);return parts;}
 function encode(text){const ids=[];for(const m of text.matchAll(PAT)){const w=Array.from(enc.encode(m[0]),b=>B2U[b]).join('');for(const p of bpe(w)){const id=V.get(p);ids.push(id===undefined?V.get('<unk>'):id);}}return ids;}
 return {encode,size:data.tokens.length,bos:V.get('<s>'),eos:V.get('</s>'),pad:V.get('<pad>')};}
root.makeBPE=make;if(typeof module!=='undefined')module.exports=make;})(typeof self!=='undefined'?self:globalThis);
