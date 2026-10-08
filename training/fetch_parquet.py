import os, io, json, random
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
P=r'C:\Projects\TinyDetect'; out=os.path.join(P,'data','img','train'); os.makedirs(out,exist_ok=True)
meta=[json.loads(l) for l in open(os.path.join(P,'data','img_meta.jsonl')) if l.strip()]
have={m['f'] for m in meta}; meta=[m for m in meta if os.path.exists(os.path.join(P,'data','img',*m['f'].split('/'))) and m['src']=='hemg' and not m['f'].startswith('train/p')]
random.seed(1); n=0; per={0:0,1:0}
for sh in ['data/train-00004-of-00006-1101eaf5152e1c5f.parquet','data/train-00001-of-00006-8ad2d550254dea81.parquet']:
    f=hf_hub_download('Hemg/AI-Generated-vs-Real-Images-Datasets',sh,repo_type='dataset'); t=pq.read_table(f)
    labs=t.column('label').to_pylist(); print(sh,'rows',len(labs),'label0',labs.count(0),'label1',labs.count(1),flush=True)
    idx=list(range(len(labs))); random.shuffle(idx); imgs=t.column('image')
    for i in idx:
        y=1 if labs[i]==0 else 0
        if per[y]>=1500: continue
        b=imgs[i].as_py()['bytes']
        if not b or len(b)<400: continue
        fn=f'q{n}.jpg'; open(os.path.join(out,fn),'wb').write(b); meta.append({'f':'train/'+fn,'y':y,'src':'hemg'}); per[y]+=1; n+=1
    print('per',per,flush=True)
    if per[0]>=1500 and per[1]>=1500: break
with open(os.path.join(P,'data','img_meta_hemg.jsonl'),'w') as fh:
    for m in meta: fh.write(json.dumps(m)+'\n')
print('PARQUETDONE added',n,'total',len(meta),flush=True)