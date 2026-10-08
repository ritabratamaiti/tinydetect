import json, random, time, urllib.request, urllib.parse, collections
P=r'C:\Projects\TinyDetect'; random.seed(11)
def rows(off,n=100):
    u='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset='Jinyan1/COLING_2025_MGT_en',config='default',split='train',offset=off,length=n))
    for t in range(5):
        try: return json.load(urllib.request.urlopen(u,timeout=60))['rows']
        except Exception: time.sleep(3*(t+1))
    return []
sz=json.load(urllib.request.urlopen('https://datasets-server.huggingface.co/size?dataset=Jinyan1/COLING_2025_MGT_en'))['size']['dataset']['num_rows']
fresh=[x['t'][5:50].lower() for f in ['fresh','fresh2'] for x in json.load(open(P+'\\'+f+r'\human_text.json',encoding='utf8'))]
cnt=collections.Counter(); out=[]; models=collections.Counter()
for k in range(90):
    for r in rows(random.randint(0,sz-100)):
        x=r['row']; t=(x.get('text') or '').strip(); y=int(x['label']); ss=x.get('sub_source') or x.get('source')
        if len(t)<250 or len(t)>3000: continue
        if any(f in t.lower() for f in fresh): continue
        key=(ss,y)
        if cnt[key]>=120: continue
        cnt[key]+=1; models[x.get('model')]+=1; out.append(dict(t=t[:2000],y=y,m=x.get('model'),a='none',d='mgt_'+str(ss)))
    if k%15==0: print(k,len(out),flush=True)
print('sources',sorted(((k[0],k[1],v) for k,v in cnt.items()),key=lambda z:z[0])[:80],flush=True)
print('models',models.most_common(25),flush=True)
open(P+r'\data\text_mgt.jsonl','w',encoding='utf8').write('\n'.join(json.dumps(x) for x in out)); print('MGTDONE',len(out),sum(o['y'] for o in out),flush=True)