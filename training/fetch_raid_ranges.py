import os, re, csv, io, json, random, urllib.request, collections
os.environ['HF_HUB_DISABLE_XET']='1'
csv.field_size_limit(10**9)
U='https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv'; P=r'C:\Projects\TinyDetect'
UUID=re.compile(r'\n([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}),')
fresh=[x['t'][:60].lower() for x in json.load(open(P+r'\fresh\human_text.json',encoding='utf8'))]
want={'reddit','reviews','recipes','wiki','poetry'}; cnt=collections.Counter(); out=[]
starts=[int(5.2e9+i*(11.7e9-5.2e9)/26) for i in range(26)]
for s in starts:
    try:
        req=urllib.request.Request(U,headers={'Range':f'bytes={s}-{s+9_000_000}'}); data=urllib.request.urlopen(req,timeout=60).read().decode('utf8','ignore')
    except Exception as e: print('range err',s,e,flush=True); continue
    m=UUID.search(data)
    if not m: continue
    chunk=data[m.start()+1:]; last=list(UUID.finditer(chunk)); chunk=chunk[:last[-1].start()] if last else chunk
    for row in csv.reader(io.StringIO(chunk)):
        if len(row)<11: continue
        dom,model,attack,gen=row[7],row[3],row[6],row[10]
        if dom not in want or not gen or len(gen)<300: continue
        if attack not in ('none','paraphrase','synonym','article_deletion','number','upper_lower'): continue
        y=0 if model=='human' else 1; key=(dom,y)
        if cnt[key]>= (300 if y==1 else 250): continue
        if any(f[5:50] in gen.lower() for f in fresh): continue
        cnt[key]+=1; out.append({'t':gen[:2000],'y':y,'m':model,'a':attack,'d':'raid_'+dom})
    print(s, dict(cnt), flush=True)
json.dump(dict((f'{k[0]}/{k[1]}',v) for k,v in cnt.items()),open(P+r'\logs\raid_extra_counts.json','w'))
open(P+r'\data\text_raid_extra.jsonl','w',encoding='utf8').write('\n'.join(json.dumps(x) for x in out)); print('RAIDXDONE',len(out),flush=True)