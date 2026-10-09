# Light tuning of fakespot-ai/roberta-base-ai-text-detection-v1 (Apache-2.0) to stop flagging formal human prose.
import os, json, random, time, math, copy
os.environ.setdefault('HF_HUB_DISABLE_XET','1'); os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score
P=r'C:\Projects\TinyDetect'; RID='fakespot-ai/roberta-base-ai-text-detection-v1'; OUT=P+r'\models\fakespot_tuned'; os.makedirs(OUT,exist_ok=True)
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
random.seed(1); torch.manual_seed(1)
def load(f): 
    p=P+'\\data\\'+f
    return [json.loads(l) for l in open(p,encoding='utf8') if l.strip()] if os.path.exists(p) else []
# test-set texts must never appear in tuning data
fresh=set()
for F_ in ['fresh','fresh2']:
    for fn in ['human_text.json','ai_text.json']:
        for x in json.load(open(P+'\\'+F_+'\\'+fn,encoding='utf8')): fresh.add(x['t'][:60].lower())
def clean(rows): return [r for r in rows if len(r['t'].split())>=40 and r['t'][:60].lower() not in fresh and not any(f in r['t'].lower() for f in list(fresh)[:0])]
def take(rows,y,n,doms=None):
    r=[x for x in rows if x['y']==y and (doms is None or any(x['d'].startswith(d) for d in doms))]; random.shuffle(r); return r[:n]
v4=load('text_v4_extra.jsonl'); wild=load('text_wild.jsonl'); hape=load('text_hape.jsonl'); gem=load('text_gen_gemma-3-1b-it.jsonl'); mgt=load('text_mgt.jsonl'); hx=load('text_human_extra.jsonl')
for r in hx: r.setdefault('y',0); r.setdefault('d','human_extra')
log('sources v4',len(v4),'wild',len(wild),'hape',len(hape),'gemma',len(gem),'mgt',len(mgt),'human_extra',len(hx))
log('v4 domains',sorted(set(r['d'] for r in v4)))
H= take(v4,0,1400)+take(wild,0,400)+take(hape,0,700)+take(hx,0,800)+take(mgt,0,500)
A= take(v4,1,1000)+take(wild,1,600)+take(hape,1,700)+take(gem,1,500)+take(mgt,1,700)
rows=clean(H)+clean(A); random.shuffle(rows)
nv=int(.15*len(rows)); va=rows[:nv]; tr=rows[nv:]
log('tune rows',len(tr),'val',len(va),'human frac tr',round(sum(r['y']==0 for r in tr)/len(tr),3))
dev='cuda'; tk=AutoTokenizer.from_pretrained(RID)
m=AutoModelForSequenceClassification.from_pretrained(RID).to(dev); ref=copy.deepcopy(m).eval()
for p in ref.parameters(): p.requires_grad=False
AI=[int(k) for k,v in m.config.id2label.items() if 'ai' in str(v).lower()][0]; log('AI label idx',AI,m.config.id2label)
L=int(os.environ.get('MAXLEN','256'))
def enc(b): return tk([r['t'] for r in b],truncation=True,max_length=L,padding=True,return_tensors='pt').to(dev)
@torch.no_grad()
def predict(model,texts,bs=32):
    model.eval(); out=[]
    for i in range(0,len(texts),bs):
        e=tk(texts[i:i+bs],truncation=True,max_length=L,padding=True,return_tensors='pt').to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16): out+=torch.softmax(model(**e).logits.float(),-1)[:,AI].tolist()
    return np.array(out)
def fresh_eval(model):
    r={}
    for F_ in ['fresh','fresh2']:
        h=json.load(open(P+'\\'+F_+r'\human_text.json',encoding='utf8')); a=json.load(open(P+'\\'+F_+r'\ai_text.json',encoding='utf8'))
        ph=predict(model,[x['t'] for x in h]); pa=predict(model,[x['t'] for x in a])
        r[F_]=dict(auc=round(float(roc_auc_score([0]*len(ph)+[1]*len(pa),list(ph)+list(pa))),3),h_flag=int((ph>=0.6).sum()),h_human=int((ph<0.3).sum()),a_caught=int((pa>=0.6).sum()),a_missed=int((pa<0.3).sum()))
    return r
def val_eval(model):
    p=predict(model,[r['t'] for r in va]); y=np.array([r['y'] for r in va]); h=p[y==0]; a=p[y==1]
    thr=np.quantile(h,0.97)   # threshold flagging 3% of validation humans
    return dict(auc=round(float(roc_auc_score(y,p)),4),fpr_at_06=round(float((h>=0.6).mean()),3),tpr_at_06=round(float((a>=0.6).mean()),3),tpr_at_fpr3=round(float((a>thr).mean()),3))
v0=val_eval(m); f0=fresh_eval(m); log('BEFORE val',v0,'fresh',f0)
EP=int(os.environ.get('EP','2')); bs=16; acc=2; lr=float(os.environ.get('LR','1e-5'))
opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01); steps=EP*math.ceil(len(tr)/bs)//acc
sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=steps+5,pct_start=0.1)
best=(v0['tpr_at_fpr3'],-1); hist=[dict(step=0,val=v0,fresh=f0)]
step=0; evals=int(os.environ.get('EVALS','6')); every=max(1,steps//evals)
for ep in range(EP):
    random.shuffle(tr); m.train()
    for i in range(0,len(tr),bs):
        b=tr[i:i+bs]; e=enc(b); y=torch.tensor([AI if r['y']==1 else 1-AI for r in b],device=dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lg=m(**e).logits.float()
            with torch.no_grad(): rl=ref(**e).logits.float()
        w=torch.tensor([1.0 if r['y']==1 else 2.0 for r in b],device=dev)
        ce=(F.cross_entropy(lg,y,reduction='none')*w).mean()
        aim=torch.tensor([r['y']==1 for r in b],device=dev)
        kd=(F.kl_div(F.log_softmax(lg[aim],-1),F.softmax(rl[aim],-1),reduction='batchmean') if aim.any() else lg.sum()*0)
        loss=(ce+0.5*kd)/acc; loss.backward()
        if (i//bs+1)%acc==0:
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad(); step+=1
            if step%every==0:
                v=val_eval(m); f=fresh_eval(m); hist.append(dict(step=step,val=v,fresh=f)); log('step',step,'/',steps,'loss',round(loss.item()*acc,4),'val',v,'fresh',f)
                if v['tpr_at_fpr3']>best[0]: best=(v['tpr_at_fpr3'],step); m.save_pretrained(OUT); tk.save_pretrained(OUT); log('  saved best (val tpr@fpr3 =',v['tpr_at_fpr3'],')')
json.dump(dict(before=dict(val=v0,fresh=f0),hist=hist,best_step=best[1],maxlen=L,lr=lr,ep=EP,n_tr=len(tr),n_val=len(va)),open(OUT+r'\tune_log.json','w'),indent=1)
log('TUNEDONE best step',best[1],'val tpr@fpr3',best[0])