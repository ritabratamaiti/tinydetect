import os, json, random; os.environ['HF_HUB_DISABLE_XET']='1'; os.environ['HF_HUB_OFFLINE']='1'
import pyarrow.parquet as pq, collections
from huggingface_hub import hf_hub_download
P=r'C:\Projects\TinyDetect'; random.seed(3)
t=pq.read_table(hf_hub_download('browndw/human-ai-parallel-corpus-mini','hape_mini-text.parquet',repo_type='dataset')).to_pylist()
out=[]
for r in t:
    src,g=r['doc_id'].split('_')[:2]; y=0 if src=='human' else 1
    if y==1 and random.random()<0.5: continue   # 1:1 human:AI per genre
    w=r['text'].split(); n=random.choice([120,180,250])
    for i in range(0,min(len(w),2*n),n):
        c=' '.join(w[i:i+n])
        if len(c.split())>=60: out.append(dict(t=c,y=y,m=src,a='none',d='hape_'+g))
print(collections.Counter((o['d'],o['y']) for o in out))
open(P+r'\data\text_hape.jsonl','w',encoding='utf8').write('\n'.join(json.dumps(o) for o in out)); print('HAPEDONE',len(out))