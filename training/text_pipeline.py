# Text: teacher -> goal-aware distillation into bert-mini -> vocab prune + int4 embeddings -> ONNX int8 body
import os, json, random, math, time, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import roc_auc_score, roc_curve
P=r'C:\Projects\TinyDetect'; dev='cuda'; random.seed(0); torch.manual_seed(0)
def log(*a):
    s=time.strftime('%H:%M:%S')+' '+' '.join(map(str,a)); print(s,flush=True); open(P+r'\logs\text.log','a').write(s+'\n')
rows=[json.loads(l) for l in open(P+r'\data\text_raw.jsonl',encoding='utf8')]
random.shuffle(rows); n=len(rows); te=rows[:int(.18*n)]; tr=rows[int(.18*n):]
log('train',len(tr),'test',len(te),'human_tr',sum(r['y']==0 for r in tr))
# ---- teacher soft labels
tt=AutoTokenizer.from_pretrained('Oxidane/tmr-ai-text-detector'); tm=AutoModelForSequenceClassification.from_pretrained('Oxidane/tmr-ai-text-detector').to(dev).half().eval()
@torch.no_grad()
def teach(rs,bs=32):
    out=[]
    for i in range(0,len(rs),bs):
        b=tt([r['t'] for r in rs[i:i+bs]],truncation=True,max_length=256,padding=True,return_tensors='pt').to(dev)
        out+=torch.softmax(tm(**b).logits.float(),-1)[:,1].tolist()
    return out
t0=time.time(); ptr=teach(tr); pte=teach(te); log('teacher done',round(time.time()-t0),'s')
del tm; torch.cuda.empty_cache()
def metrics(y,p,name):
    y=np.array(y);p=np.array(p);auc=roc_auc_score(y,p);fpr,tpr,th=roc_curve(y,p)
    t1=float(np.interp(0.01,fpr,tpr)); h=p[y==0]; fp5=float((h>0.5).mean())
    log(f'{name}: AUROC={auc:.4f} TPR@1%FPR={t1:.3f} FPR@0.5={fp5:.3f}'); return dict(auc=auc,tpr1=t1,fpr05=fp5)
res={'teacher':metrics([r['y'] for r in te],pte,'TEACHER roberta-base 499MB')}
# ---- student
st=AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
sm=BertForSequenceClassification.from_pretrained('google/bert_uncased_L-4_H-256_A-4',num_labels=2).to(dev)
L=256
def enc(rs): return st([r['t'] for r in rs],truncation=True,max_length=L,padding=True,return_tensors='pt')
opt=torch.optim.AdamW(sm.parameters(),lr=1e-4,weight_decay=0.01)
EPOCHS=int(os.environ.get('EPOCHS','4')); bs=32; steps=EPOCHS*math.ceil(len(tr)/bs); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-4,total_steps=steps,pct_start=0.1)
T=2.0; W_HUMAN=3.0  # goal-aware: a false accusation (human flagged) costs 3x a miss
idx=list(range(len(tr)))
for ep in range(EPOCHS):
    random.shuffle(idx); sm.train(); tot=0
    for i in range(0,len(idx),bs):
        bi=idx[i:i+bs]; b=enc([tr[j] for j in bi]).to(dev)
        y=torch.tensor([tr[j]['y'] for j in bi],device=dev); pt=torch.tensor([ptr[j] for j in bi],device=dev)
        lg=sm(**b).logits; ls=torch.log_softmax(lg/T,-1); q=torch.stack([1-pt,pt],-1).clamp(1e-4,1)
        q=torch.softmax(torch.log(q)/T,-1)
        kd=F.kl_div(ls,q,reduction='batchmean')*T*T
        w=torch.where(y==0,torch.full_like(y,W_HUMAN,dtype=torch.float),torch.ones_like(y,dtype=torch.float))
        ce=(F.cross_entropy(lg,y,reduction='none')*w).mean()
        # FPR hinge: push human scores below 0.3 margin
        pai=torch.softmax(lg,-1)[:,1]; hinge=(F.relu(pai-0.3)*(y==0).float()).sum()/max(1,(y==0).sum().item())
        loss=0.5*kd+0.5*ce+1.0*hinge
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(sm.parameters(),1.0); opt.step(); sched.step(); tot+=loss.item()
    log('epoch',ep,'loss',round(tot/(len(idx)/bs),4))
@torch.no_grad()
def spred(model,rs,embed_override=None,bs=64):
    model.eval(); out=[]
    for i in range(0,len(rs),bs):
        b=enc(rs[i:i+bs]).to(dev)
        if embed_override is not None:
            e=embed_override[b['input_ids']]; o=model(inputs_embeds=e,attention_mask=b['attention_mask'],token_type_ids=b['token_type_ids'])
        else: o=model(**b)
        out+=torch.softmax(o.logits.float(),-1)[:,1].tolist()
    return out
yte=[r['y'] for r in te]
res['student_fp32']=metrics(yte,spred(sm,te),'STUDENT bert-mini fp32'); torch.save(sm.state_dict(),P+r'\models\text_student.pt')
# ---- custom quant: vocab prune + int4 per-row embeddings (simulate)
used=set(st.all_special_ids)
for i in range(0,len(rows),256):
    for ids in st([r['t'] for r in rows[i:i+256]],truncation=True,max_length=L)['input_ids']: used.update(ids)
keep=sorted(used); log('vocab kept',len(keep),'of',st.vocab_size)
W=sm.bert.embeddings.word_embeddings.weight.detach().float().cpu()
Wk=W[keep]; scale=Wk.abs().amax(1,keepdim=True)/7.0; q4=torch.clamp(torch.round(Wk/scale),-8,7).to(torch.int8)
deq=torch.zeros_like(W); deq[keep]=q4.float()*scale; # pruned rows -> unk
unk=st.unk_token_id; mask=torch.ones(W.shape[0],dtype=torch.bool); mask[keep]=False; deq[mask]=deq[unk].clone()
res['student_int4emb']=metrics(yte,spred(sm,te,embed_override=deq.to(dev)),'STUDENT + pruned vocab + int4 embeddings')
# per-attack breakdown (student int4)
ps=spred(sm,te,embed_override=deq.to(dev)); by={}
for r,p in zip(te,ps): by.setdefault(r['a'],[]).append((r['y'],p))
for a,v in sorted(by.items()):
    yy=[x[0] for x in v];pp=[x[1] for x in v]
    if len(set(yy))==2: log(f'  attack={a} n={len(v)} AUROC={roc_auc_score(yy,pp):.3f}')
    else: log(f'  attack={a} n={len(v)} (single class) mean_p={np.mean(pp):.3f}')
# ---- export
os.makedirs(P+r'\web\models',exist_ok=True)
packed=np.zeros((len(keep),W.shape[1]//2),dtype=np.uint8); qn=(q4.numpy()+8).astype(np.uint8); packed=(qn[:,0::2]|(qn[:,1::2]<<4)).astype(np.uint8)
packed.tofile(P+r'\web\models\text_emb_int4.bin'); scale.squeeze(1).numpy().astype(np.float16).tofile(P+r'\web\models\text_emb_scale_f16.bin')
vocab=st.convert_ids_to_tokens(keep); json.dump({'keep_ids':keep,'tokens':vocab,'unk':keep.index(unk),'cls':keep.index(st.cls_token_id),'sep':keep.index(st.sep_token_id),'dim':int(W.shape[1]),'max_len':L},open(P+r'\web\models\text_vocab.json','w'))
class Body(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,inputs_embeds,attention_mask): return torch.softmax(s.m(inputs_embeds=inputs_embeds,attention_mask=attention_mask,token_type_ids=torch.zeros_like(attention_mask)).logits,-1)
smc=sm.float().cpu().eval(); body=Body(smc)
ex=(torch.randn(1,16,W.shape[1]),torch.ones(1,16,dtype=torch.long))
torch.onnx.export(body,ex,P+r'\models\text_body.onnx',input_names=['inputs_embeds','attention_mask'],output_names=['probs'],dynamic_axes={'inputs_embeds':{1:'L'},'attention_mask':{1:'L'}},opset_version=17,dynamo=False)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(P+r'\models\text_body.onnx',P+r'\web\models\text_body_int8.onnx',weight_type=QuantType.QInt8)
sz=lambda f:os.path.getsize(f)/1e6
tot=sz(P+r'\web\models\text_body_int8.onnx')+sz(P+r'\web\models\text_emb_int4.bin')+sz(P+r'\web\models\text_emb_scale_f16.bin')+sz(P+r'\web\models\text_vocab.json')
res['size_MB']={'teacher':499,'student_total':round(tot,2)}; log('TOTAL text model download MB',round(tot,2))
# verify onnx int8 matches
import onnxruntime as ort
s=ort.InferenceSession(P+r'\web\models\text_body_int8.onnx'); pq=[]
remap={k:i for i,k in enumerate(keep)}; deqk=(q4.float()*scale).numpy()
for r in te:
    ids=st(r['t'],truncation=True,max_length=L)['input_ids']; ids=[i if i in remap else unk for i in ids]
    e=deqk[[remap[i] for i in ids]][None].astype(np.float32); pq.append(float(s.run(None,{'inputs_embeds':e,'attention_mask':np.ones((1,len(ids)),dtype=np.int64)})[0][0,1]))
res['student_final_onnx']=metrics(yte,pq,'FINAL shipped (int4 emb + int8 ONNX body)')
json.dump(res,open(P+r'\logs\text_results.json','w'),indent=1); log('TEXTDONE')
