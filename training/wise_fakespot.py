# WiSE-FT: blend tuned and original Fakespot weights; pick alpha on the validation split + fresh set 1 only.
import os, json, time, copy
os.environ.setdefault('HF_HUB_OFFLINE','1')
P=r'C:\Projects\TinyDetect'
src=open(P+r'\scripts\tune_fakespot.py',encoding='utf8').read()
exec(src[:src.index("dev='cuda'")])          # rebuilds tr/va with the same seed and code
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
dev='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' else 'cpu'; torch.set_num_threads(8); tk=AutoTokenizer.from_pretrained(RID); L=256
orig=AutoModelForSequenceClassification.from_pretrained(RID).state_dict()
tuned=AutoModelForSequenceClassification.from_pretrained(OUT).state_dict()
m=AutoModelForSequenceClassification.from_pretrained(RID).to(dev).eval(); AI=1
@torch.no_grad()
def predict(texts,bs=32):
    out=[]
    for i in range(0,len(texts),bs):
        e=tk(texts[i:i+bs],truncation=True,max_length=L,padding=True,return_tensors='pt').to(dev)
        out+=torch.softmax(m(**e).logits.float(),-1)[:,AI].tolist()
    return np.array(out)
vt=[r['t'] for r in va]; vy=np.array([r['y'] for r in va])
FR={}
for F_ in ['fresh','fresh2']:
    h=json.load(open(P+'\\'+F_+r'\human_text.json',encoding='utf8')); a=json.load(open(P+'\\'+F_+r'\ai_text.json',encoding='utf8'))
    FR[F_]=([x['t'] for x in h]+[x['t'] for x in a],np.array([0]*len(h)+[1]*len(a)))
res={}
for al in [0.0,0.2,0.35,0.5,0.65,0.8,1.0]:
    sd={k:(orig[k].float()*(1-al)+tuned[k].float()*al).to(orig[k].dtype) if orig[k].is_floating_point() else tuned[k] for k in orig}
    m.load_state_dict(sd)
    p=predict(vt); h=p[vy==0]; a=p[vy==1]; thr=np.quantile(h,0.97)
    r=dict(val_auc=round(float(roc_auc_score(vy,p)),4),val_fpr06=round(float((h>=0.6).mean()),3),val_tpr06=round(float((a>=0.6).mean()),3),val_tpr_at_fpr3=round(float((a>thr).mean()),3))
    for F_,(tx,y) in FR.items():
        q=predict(tx); r[F_]=dict(auc=round(float(roc_auc_score(y,q)),3),h_flag=int((q[y==0]>=0.6).sum()),h_human=int((q[y==0]<0.3).sum()),a_caught=int((q[y==1]>=0.6).sum()),a_missed=int((q[y==1]<0.3).sum()))
        r[F_+'_scores']=[round(float(v),4) for v in q]
    r['val_scores']=[round(float(v),4) for v in p]
    res[str(al)]=r; print(time.strftime('%H:%M:%S'),'alpha',al,{k:v for k,v in r.items() if not k.endswith('scores')},flush=True)
json.dump(dict(res=res,val_y=vy.tolist()),open(OUT+r'\wise.json','w'))
print('WISEDONE',flush=True)