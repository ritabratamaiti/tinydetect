import os, json, random, io, time
os.environ['HF_HUB_DISABLE_XET']='1'
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
P=r'C:\Projects\TinyDetect'; random.seed(9)
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
f=hf_hub_download('detection-datasets/coco','data/train-00011-of-00040-b8e4a8a2d2f5bb1f.parquet' if False else [s for s in __import__('huggingface_hub').HfApi().list_repo_files('detection-datasets/coco',repo_type='dataset') if s.startswith('data/train-00013')][0],repo_type='dataset'); log('dl',f)
pf=pq.ParquetFile(f); log(pf.schema_arrow)
n=0; os.makedirs(P+r'\data\img\real2',exist_ok=True)
for b in pf.iter_batches(batch_size=64,columns=['image']):
    for im in b.column('image').to_pylist():
        if random.random()>0.4: continue
        open(P+rf'\data\img\real2\coco_{n}.jpg','wb').write(im['bytes']); n+=1
        if n>=900: break
    if n>=900: break
meta=[]
for fn in os.listdir(P+r'\data\img\gen'): meta.append(dict(f='gen/'+fn,y=1,src=fn.split('_')[0]))
for fn in os.listdir(P+r'\data\img\real2'): meta.append(dict(f='real2/'+fn,y=0,src='coco'))
open(P+r'\data\img_meta_v4.jsonl','w').write('\n'.join(json.dumps(m) for m in meta)); log('COCODONE',n,len(meta))