import json, random, time, re, collections, os
os.environ['HF_HUB_DISABLE_XET']='1'
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
P=r'C:\Projects\TinyDetect'; random.seed(5)
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
f=hf_hub_download('allenai/WildChat-1M','data/train-00007-of-00014.parquet',repo_type='dataset'); log('downloaded',f)
t=pq.read_table(f,columns=['model','language','conversation']).to_pylist(); log('rows',len(t))
ASK=re.compile(r'\b(write|draft|compose|reply|respond|rewrite|make)\b',re.I)
GENRE=re.compile(r'\b(e-?mail|message|text|reply|post|comment|review|tweet|slack|letter|note|caption|bio|forum|reddit|response|announcement|update|story|blog|description|paragraph)\b',re.I)
CODE=re.compile(r'\x60\x60\x60|def |function\(|#include|</?\w+>|\\begin|\[Your|\[Name|\[Recipient')
random.shuffle(t); out=[]; g=collections.Counter()
for x in t:
    c=x['conversation']
    if x['language']!='English' or not c or len(c)<2: continue
    u=c[0]['content'] or ''; a=(c[1]['content'] or '').strip()
    if not (ASK.search(u[:300]) and GENRE.search(u[:300])) or CODE.search(a): continue
    w=len(a.split())
    if w<45 or w>450: continue
    m=GENRE.search(u[:300]).group(1).lower().replace('e-mail','email')
    if g[m]>=160: continue
    g[m]+=1; out.append(dict(t=a[:2000],y=1,m=x['model'],a='none',d='casual_'+m))
log('wild',len(out),g.most_common())
hum=[]
try:
    fs=[s.rfilename for s in HfApi().dataset_info('Yale-LILY/aeslc',revision='refs/convert/parquet').siblings if s.rfilename.endswith('.parquet') and 'train' in s.rfilename]; log(fs)
    for fn in fs:
        for r in pq.read_table(hf_hub_download('Yale-LILY/aeslc',fn,repo_type='dataset',revision='refs/convert/parquet')).to_pylist():
            b=(r.get('email_body') or '').strip(); w=len(b.split())
            if 45<=w<=450: hum.append(dict(t=b[:2000],y=0,m='human',a='none',d='casual_email'))
    random.shuffle(hum); hum=hum[:900]
except Exception as e: log('aeslc fail',repr(e))
log('aeslc',len(hum))
open(P+r'\data\text_wild.jsonl','w',encoding='utf8').write('\n'.join(json.dumps(x) for x in out+hum)); log('WILDDONE',len(out),len(hum))