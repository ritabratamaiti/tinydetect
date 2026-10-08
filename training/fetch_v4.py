import os, json, random, time, urllib.request, urllib.parse, io
P=r'C:\Projects\TinyDetect'; random.seed(7)
def log(*a):
    s=time.strftime('%H:%M:%S')+' '+' '.join(map(str,a)); print(s,flush=True); open(P+r'\logs\v4data.log','a').write(s+'\n')
def rows(ds,cfg,split,off,n=100,tries=5):
    u='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=ds,config=cfg,split=split,offset=off,length=n))
    for t in range(tries):
        try: return json.load(urllib.request.urlopen(u,timeout=60))['rows']
        except Exception as e: time.sleep(3*(t+1))
    return []
FRESH_WIKI={'Bicycle','Bread','Monsoon','Tea','Jazz','Lighthouse','Volcano','Chess','Coffee','Bridge','Penguin','Opera'}
FRESH_BOOKS=['Pride and Prejudice','Moby','Tale of Two Cities','Frankenstein','Sherlock','Dorian Gray','Dracula','Huckleberry','Jane Eyre','Great Expectations','Emma','Heart of Darkness','Mississippi']
out=[]
# 1) HC3 paired human vs ChatGPT answers (casual + formal, same questions)
for cfg,N in [('reddit_eli5',17112),('open_qa',1187),('finance',3933),('medicine',1248),('wiki_csai',842)]:
    want=260 if cfg=='reddit_eli5' else 120; got=[0,0]
    for _ in range(30):
        if min(got)>=want: break
        for r in rows('Hello-SimpleAI/HC3',cfg,'train',random.randint(0,max(0,N-100))):
            x=r['row']
            for t in x.get('human_answers') or []:
                if got[0]<want and 300<=len(t)<=2500: out.append(dict(t=t.strip()[:2000],y=0,m='human',a='none',d='hc3_'+cfg)); got[0]+=1; break
            for t in x.get('chatgpt_answers') or []:
                if got[1]<want and 300<=len(t)<=2500: out.append(dict(t=t.strip()[:2000],y=1,m='chatgpt',a='none',d='hc3_'+cfg)); got[1]+=1; break
    log('hc3',cfg,got)
# 2) GPT-wiki-intro: same title, human vs GPT
got=[0,0]
for _ in range(20):
    if min(got)>=500: break
    for r in rows('aadityaubhat/GPT-wiki-intro','default','train',random.randint(0,149000)):
        x=r['row']
        if x.get('title') in FRESH_WIKI: continue
        if got[0]<500 and len(x.get('wiki_intro',''))>=300: out.append(dict(t=x['wiki_intro'][:2000],y=0,m='human',a='none',d='wiki_pairs')); got[0]+=1
        if got[1]<500 and len(x.get('generated_intro',''))>=300: out.append(dict(t=x['generated_intro'][:2000],y=1,m='curie',a='none',d='wiki_pairs')); got[1]+=1
log('gpt-wiki-intro',got)
# 3) Human book prose from Project Gutenberg (balances 2,208 AI 'books')
nb=0; seen=0
for _ in range(60):
    if nb>=900: break
    rr=rows('manu/project_gutenberg','default','en',random.randint(0,60000),n=4)
    for r in rr:
        x=r['row']; txt=x.get('text') or ''; head=txt[:3000]
        if any(b.lower() in head.lower() for b in FRESH_BOOKS): continue
        paras=[' '.join(p.split()) for p in txt.split('\n\n')]
        paras=[p for p in paras if 400<=len(p)<=1800 and not p.isupper() and 'Gutenberg' not in p and 'CHAPTER' not in p]
        if len(paras)<20: continue
        random.shuffle(paras); seen+=1
        for p in paras[:12]: out.append(dict(t=p,y=0,m='human',a='none',d='gutenberg')); nb+=1
log('gutenberg paras',nb,'from books',seen)
open(P+r'\data\text_v4_extra.jsonl','w',encoding='utf8').write('\n'.join(json.dumps(x) for x in out)); log('TEXTV4DONE',len(out))
# 4) images: diverse generators + diverse real photos
meta=[]
def save(src,dirn,y,ds,cfg,split,offs,per):
    n=0
    for o in offs:
        for r in rows(ds,cfg,split,o,n=per):
            im=next((v for v in r['row'].values() if isinstance(v,dict) and v.get('src')),None)
            if not im: continue
            try:
                b=urllib.request.urlopen(im['src'],timeout=60).read()
                if len(b)<3000: continue
                fn=f'{src}_{o}_{r["row_idx"]}.jpg'; open(os.path.join(P,'data','img',dirn,fn),'wb').write(b)
                meta.append(dict(f=dirn+'/'+fn,y=y,src=src)); n+=1
            except Exception: pass
        time.sleep(0.5)
    log('img',src,n)
save('mjv6','gen',1,'Photoroom/midjourney-v6-recap','default','train',[random.randint(60000,1200000) for _ in range(30)],12)
save('dalle3','gen',1,'OpenDatasets/dalle-3-dataset','default','train',[random.randint(3000,18900) for _ in range(25)],12)
save('flux','gen',1,'ash12321/flux-1-dev-generated-10k','default','train',[random.randint(0,9900) for _ in range(25)],12)
save('coco','real2',0,'detection-datasets/coco','default','train',[random.randint(0,117000) for _ in range(40)],14)
save('flickr','real2',0,'nlphuji/flickr30k','TEST','test',[random.randint(0,30900) for _ in range(25)],12)
open(P+r'\data\img_meta_v4.jsonl','w').write('\n'.join(json.dumps(m) for m in meta)); log('IMGV4DONE',len(meta))