# Score compressed text-detector variants (embedding precision x body precision) against the fp32 model.
import os, sys, json, time, random
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score
P=r'C:\Projects\TinyDetect'; O=sys.argv[1]; MAXT=int(os.environ.get('MAXT','512'))
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
tk=AutoTokenizer.from_pretrained(O+r'\tok')
from safetensors.numpy import load_file
import glob
st=[f for f in glob.glob(O+r'\..\..\models\*') if False]
from transformers import AutoModelForSequenceClassification
meta=json.load(open(O+r'\meta.json'))
src=meta['src']; m=AutoModelForSequenceClassification.from_pretrained(src)
WE=m.roberta.embeddings.word_embeddings.weight.detach().numpy().astype(np.float32); del m
def q_row(W):
    s=np.abs(W).max(1,keepdims=True)/7.0; s[s==0]=1e-8; return (np.clip(np.round(W/s),-8,7)*s).astype(np.float32)
def q_block(W,b=32):
    V,D=W.shape; X=W.reshape(V,D//b,b); s=np.abs(X).max(2,keepdims=True)/7.0; s[s==0]=1e-8; return (np.clip(np.round(X/s),-8,7)*s).reshape(V,D).astype(np.float32)
EMB={'fp32':WE,'int4row':q_row(WE),'int4b32':q_block(WE,32)}
for k,v in EMB.items(): log('emb',k,'rel err',round(float(np.abs(v-WE).mean()/np.abs(WE).mean()),4))
sets={}
for F in ['fresh','fresh2']:
    h=json.load(open(P+'\\'+F+r'\human_text.json',encoding='utf8')); a=json.load(open(P+'\\'+F+r'\ai_text.json',encoding='utf8'))
    sets[F]=([x['t'] for x in h]+[x['t'] for x in a],np.array([0]*len(h)+[1]*len(a)))
if os.path.exists(P+r'\models\fakespot_tuned\wise.json'):
    pass
import sys as _s; _s.path.insert(0,P+r'\scripts'); from fakespot_utils import clean_text
CLEAN=os.environ.get('CLEAN','0')=='1'
def probs(sess,E,texts):
    out=[]
    for t in texts:
        if CLEAN: t=clean_text(t)
        ids=tk(t,truncation=True,max_length=MAXT)['input_ids']
        e=E[np.array(ids)][None]; am=np.ones((1,len(ids)),np.int64)
        out.append(float(sess.run(None,{'inputs_embeds':e,'attention_mask':am})[0][0,meta['ai_index']]))
    return np.array(out)
so=ort.SessionOptions(); so.intra_op_num_threads=int(os.environ.get('THREADS','6'))
bodies=[b for b in os.environ.get('BODIES','fp32,nb8,hqq4,nb4,fp16').split(',') if os.path.exists(O+f'\\body_{b}.onnx')]
combos=[('fp32','fp32')]+[(b,'int4b32') for b in bodies]
combos=list(dict.fromkeys(combos))
res={}; ref=None
for body,emb in combos:
    t0=time.time(); s=ort.InferenceSession(O+f'\\body_{body}.onnx',so,providers=['CPUExecutionProvider']); r={}
    for F,(tx,y) in sets.items():
        p=probs(s,EMB[emb],tx); r[F+'_p']=p.tolist()
        r[F]=dict(auc=round(float(roc_auc_score(y,p)),3),h_flag=int((p[y==0]>=0.6).sum()),a_caught=int((p[y==1]>=0.6).sum()))
    if ref is None: ref=r
    r['max_dp']=round(float(max(np.abs(np.array(r[F+'_p'])-np.array(ref[F+'_p'])).max() for F in sets)),4)
    res[f'{body}+{emb}']=r; log(f'{body:5s}+{emb:8s}','maxdp',r['max_dp'],{F:r[F] for F in sets},round(time.time()-t0),'s')
    json.dump(res,open(O+(r'\variants_clean'+os.environ.get('TAG','')+'.json' if CLEAN else r'\variants.json'),'w'))
log('VARDONE')