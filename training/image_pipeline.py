# Image: teacher ensemble (Swin-B + ViT) -> goal-aware distillation into MobileViT-xxs -> static int8 QDQ ONNX
import os, json, random, time, io, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from PIL import Image
from transformers import AutoModelForImageClassification
from transformers import MobileViTForImageClassification
from sklearn.metrics import roc_auc_score, roc_curve
P=r'C:\Projects\TinyDetect'; dev='cuda'; random.seed(0); torch.manual_seed(0); np.random.seed(0)
def log(*a):
    s=time.strftime('%H:%M:%S')+' '+' '.join(map(str,a)); print(s,flush=True); open(P+r'\logs\image.log','a').write(s+'\n')
meta=[json.loads(l) for l in open(P+r'\data\img_meta.jsonl') if l.strip()]+[json.loads(l) for l in open(P+r'\data\img_meta_hemg.jsonl') if l.strip()]+([json.loads(l) for l in open(P+r'\data\img_meta_v4.jsonl') if l.strip()] if os.path.exists(P+r'\data\img_meta_v4.jsonl') else [])
def load(f):
    try: return Image.open(os.path.join(P,'data','img',*f.split('/'))).convert('RGB')
    except Exception: return None
items=[]
for m in meta:
    if m['src']=='hemg' and not (m['f'].startswith('train/q')): continue
    im=load(m['f'])
    if im is not None: items.append((im.resize((224,224),Image.BICUBIC),m['y'],m['src'],m.get('split','')))
eddy_tr=[x for x in items if x[2]=='eddyfox' and x[3]=='train']; eddy_te=[x for x in items if x[2]=='eddyfox' and x[3]=='test']
hem=[x for x in items if x[2]=='hemg']; random.shuffle(hem); nh=int(.2*len(hem))
v4=[x for x in items if x[2] not in ('hemg','eddyfox')]
te_in=eddy_te; edd=hem[:nh]   # 'in' = held-out eddyfox photos; 'ood' = different-source low-res thumbnails
tr=[x[:3] for x in eddy_tr+hem[nh:]+v4]; te_in=[x[:3] for x in te_in]; edd=[x[:3] for x in edd]
log('train',len(tr),'test_in',len(te_in),'test_ood(thumbnails)',len(edd),'ai_frac_tr',round(np.mean([y for _,y,_ in tr]),3))
# ---- teachers
T1='Smogy/SMOGY-Ai-images-detector'; T2='jacoballessio/ai-image-detect-distilled'
from huggingface_hub import hf_hub_download
def prep_fn(name):
    c=json.load(open(hf_hub_download(name,'preprocessor_config.json')))
    mean=np.array(c.get('image_mean',[0.5]*3),dtype=np.float32); std=np.array(c.get('image_std',[0.5]*3),dtype=np.float32)
    sz=c.get('size',224); sz=sz.get('height',sz.get('shortest_edge',224)) if isinstance(sz,dict) else sz
    log(name,'prep mean',mean.tolist(),'std',std.tolist(),'size',sz)
    def f(im):
        if im.size!=(sz,sz): im=im.resize((sz,sz),Image.BICUBIC)
        a=(np.asarray(im,dtype=np.float32)/255.-mean)/std; return torch.from_numpy(a.transpose(2,0,1))
    return f
def tprobs(name,ai_idx,ims,bs=32):
    pf=prep_fn(name); m=AutoModelForImageClassification.from_pretrained(name).to(dev).half().eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(ims),bs):
            b=torch.stack([pf(x[0]) for x in ims[i:i+bs]]).to(dev).half()
            out+=torch.softmax(m(pixel_values=b).logits.float(),-1)[:,ai_idx].tolist()
    del m; torch.cuda.empty_cache(); return np.array(out)
def metrics(y,p,name):
    y=np.array(y);p=np.array(p);auc=roc_auc_score(y,p);fpr,tpr,_=roc_curve(y,p);t5=float(np.interp(0.05,fpr,tpr));fp=float((p[y==0]>0.5).mean())
    log(f'{name}: AUROC={auc:.4f} TPR@5%FPR={t5:.3f} FPR@0.5(real flagged)={fp:.3f}'); return dict(auc=auc,tpr5=t5,fpr05=fp)
allims=tr+te_in+edd; t0=time.time()
pa=tprobs(T1,0,allims); pb=tprobs(T2,0,allims); log('teachers done',round(time.time()-t0),'s')
ens=(pa+pb)/2; ntr=len(tr); nin=len(te_in)
res={}
for nm,p in [('swin_b_347MB',pa),('vit_58MB',pb),('ensemble',ens)]:
    res[nm+'_in']=metrics([y for _,y,_ in te_in],p[ntr:ntr+nin],f'TEACHER {nm} in-dist')
    res[nm+'_ood']=metrics([y for _,y,_ in edd],p[ntr+nin:],f'TEACHER {nm} OOD')
soft=pa[:ntr]  # ViT teacher is worse than chance on this data, so distil from Swin-B alone
# ---- student
# MobileViT preprocessing: [0,1] range, BGR channel order (mirrored exactly in the browser)
def totensor(im): a=np.asarray(im,dtype=np.float32)[:,:,::-1]/255.; return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
def aug(im):
    if random.random()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random()<0.5:
        b=io.BytesIO(); im.save(b,'JPEG',quality=random.randint(40,95)); b.seek(0); im=Image.open(b).convert('RGB')
    if random.random()<0.3:
        s=random.randint(112,200); im=im.resize((s,s),Image.BILINEAR).resize((224,224),Image.BILINEAR)
    return im
sm=MobileViTForImageClassification.from_pretrained('apple/mobilevit-xx-small',num_labels=2,ignore_mismatched_sizes=True).to(dev)
class Wrap(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x): return s.m(pixel_values=x).logits
sm=Wrap(sm)
EP=int(os.environ.get('IEPOCHS','8')); bs=48; opt=torch.optim.AdamW(sm.parameters(),lr=1e-3,weight_decay=0.02)
steps=EP*((ntr+bs-1)//bs); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=steps,pct_start=0.15)
W_REAL=3.0; Tk=2.0; idx=list(range(ntr))
for ep in range(EP):
    random.shuffle(idx); sm.train(); tot=0
    for i in range(0,ntr,bs):
        bi=idx[i:i+bs]; x=torch.stack([totensor(aug(tr[j][0])) for j in bi]).to(dev)
        y=torch.tensor([tr[j][1] for j in bi],device=dev); pt=torch.tensor(soft[bi],device=dev,dtype=torch.float)
        lg=sm(x); q=torch.stack([1-pt,pt],-1).clamp(1e-4,1); q=torch.softmax(torch.log(q)/Tk,-1)
        kd=F.kl_div(torch.log_softmax(lg/Tk,-1),q,reduction='batchmean')*Tk*Tk
        w=torch.where(y==0,torch.full_like(y,W_REAL,dtype=torch.float),torch.ones_like(y,dtype=torch.float))
        ce=(F.cross_entropy(lg,y,reduction='none')*w).mean()
        pai=torch.softmax(lg,-1)[:,1]; hinge=(F.relu(pai-0.3)*(y==0).float()).sum()/max(1,(y==0).sum().item())
        loss=0.4*kd+0.6*ce+0.5*hinge; opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot+=loss.item()
    log('epoch',ep,'loss',round(tot/((ntr+bs-1)//bs),4))
@torch.no_grad()
def sp(model,ims,bs=64):
    model.eval(); o=[]
    for i in range(0,len(ims),bs): o+=torch.softmax(model(torch.stack([totensor(x[0]) for x in ims[i:i+bs]]).to(dev)).float(),-1)[:,1].tolist()
    return o
res['student_fp32_in']=metrics([y for _,y,_ in te_in],sp(sm,te_in),'STUDENT mobilevit-xxs fp32 in-dist')
res['student_fp32_ood']=metrics([y for _,y,_ in edd],sp(sm,edd),'STUDENT mobilevit-xxs fp32 OOD')
# ---- export fp32 (weight-only int8 is applied by quant_image.py / wq_image.py)
smc=sm.float().cpu().eval(); torch.onnx.export(smc,torch.randn(1,3,224,224),P+r'\models\image_fp32.onnx',input_names=['pixel_values'],output_names=['logits'],opset_version=17,dynamo=False)
log('IMGTRAINDONE')
