# Score candidate detectors on both fresh sets with the same bands as the site.
import os, sys, json, glob, time
os.environ.setdefault('HF_HUB_DISABLE_XET','1'); os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch, numpy as np
from PIL import Image
P=r'C:\Projects\TinyDetect'; dev='cuda' if torch.cuda.is_available() else 'cpu'
def band(p): return 'AI' if p>=0.9 else 'possAI' if p>=0.6 else 'unclear' if p>=0.3 else 'human'
def summ(ys,ps):
    ys=np.array(ys); ps=np.array(ps); h=ps[ys==0]; a=ps[ys==1]
    try:
        from sklearn.metrics import roc_auc_score; auc=roc_auc_score(ys,ps)
    except Exception: auc=float('nan')
    return dict(n_h=len(h),h_human=int((h<0.3).sum()),h_flag=int((h>=0.6).sum()),n_a=len(a),a_caught=int((a>=0.6).sum()),a_missed=int((a<0.3).sum()),auc=round(float(auc),3))
def text_sets():
    out={}
    for F in ['fresh','fresh2']:
        h=json.load(open(P+'\\'+F+r'\human_text.json',encoding='utf8')); a=json.load(open(P+'\\'+F+r'\ai_text.json',encoding='utf8'))
        out[F]=[(x['t'],0) for x in h]+[(x['t'],1) for x in a]
    return out
def img_sets():
    return {F:[(f,0 if os.path.basename(f).startswith('real') else 1) for f in sorted(glob.glob(P+'\\'+F+r'\img\*.jpg'))] for F in ['fresh','fresh2']}

def text_model(rid):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tk=AutoTokenizer.from_pretrained(rid); m=AutoModelForSequenceClassification.from_pretrained(rid).to(dev).eval()
    lab={int(k):str(v).lower() for k,v in (m.config.id2label or {}).items()}
    ai=[i for i,v in lab.items() if any(s in v for s in ['ai','chatgpt','fake','machine','generated'])]
    ai=ai[0] if ai else (1 if m.config.num_labels>1 else 0)
    def f(t):
        e=tk(t,truncation=True,max_length=512,return_tensors='pt').to(dev)
        with torch.no_grad(): lg=m(**e).logits[0].float()
        return torch.sigmoid(lg[0]).item() if lg.numel()==1 else torch.softmax(lg,-1)[ai].item()
    return f, dict(labels=lab,ai_idx=ai,params=sum(p.numel() for p in m.parameters()))

def commfor(rid,size):
    import timm
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    v=timm.create_model(f'vit_small_patch16_{size}',pretrained=False,num_classes=1)
    sd=load_file(hf_hub_download(rid,'model.safetensors')); sd={k[4:] if k.startswith('vit.') else k:v_ for k,v_ in sd.items()}
    print('load',v.load_state_dict(sd,strict=True)); v=v.to(dev).eval()
    mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
    rs=256 if size==224 else 440
    def f(path):
        im=Image.open(path).convert('RGB'); w,h=im.size; s=rs/min(w,h); im=im.resize((max(rs,round(w*s)),max(rs,round(h*s))),Image.BILINEAR)
        w,h=im.size; l=(w-size)//2; t=(h-size)//2; im=im.crop((l,t,l+size,t+size))
        x=(np.asarray(im,np.float32)/255.-mean)/std; x=torch.from_numpy(x.transpose(2,0,1))[None].to(dev)
        with torch.no_grad(): return torch.sigmoid(v(x)[0,0]).item()
    return f, dict(params=sum(p.numel() for p in v.parameters()))

def hf_image(rid):
    from transformers import AutoModelForImageClassification
    import json as _j
    from huggingface_hub import hf_hub_download
    m=AutoModelForImageClassification.from_pretrained(rid).to(dev).eval()
    pc=_j.load(open(hf_hub_download(rid,'preprocessor_config.json')))
    sz=pc.get('size',224); S=sz if isinstance(sz,int) else (sz.get('height') or sz.get('shortest_edge') or 224)
    mean=np.array(pc.get('image_mean',[0.5]*3),np.float32); std=np.array(pc.get('image_std',[0.5]*3),np.float32)
    lab={int(k):str(v).lower() for k,v in m.config.id2label.items()}
    ai=[i for i,v in lab.items() if any(s in v for s in ['artificial','ai','fake','sd','dalle','generated'])]
    def f(path):
        im=Image.open(path).convert('RGB').resize((S,S),Image.BICUBIC)
        x=(np.asarray(im,np.float32)/255.-mean)/std; x=torch.from_numpy(x.transpose(2,0,1))[None].to(dev)
        with torch.no_grad(): pr=torch.softmax(m(pixel_values=x).logits[0].float(),-1)
        return pr[ai].sum().item()
    return f, dict(labels=lab,ai_idx=ai,params=sum(p.numel() for p in m.parameters()))

res={}; rp=P+r'\logs\bench_candidates.json'
if os.path.exists(rp): res=json.load(open(rp))
for spec in sys.argv[1:]:
    kind,rid=spec.split(':',1); t0=time.time()
    try:
        if kind=='text': f,info=text_model(rid); sets=text_sets()
        elif kind=='commfor224': f,info=commfor(rid,224); sets=img_sets()
        elif kind=='commfor384': f,info=commfor(rid,384); sets=img_sets()
        else: f,info=hf_image(rid); sets=img_sets()
        r=dict(info={k:v for k,v in info.items() if k!='labels'},labels=str(info.get('labels','')))
        for F,items in sets.items():
            ps=[f(x) for x,_ in items]; ys=[y for _,y in items]; r[F]=summ(ys,ps); r[F+'_scores']=[round(p,4) for p in ps]
        res[kind+':'+rid]=r; print('RESULT',kind,rid,'params',info.get('params'),'|fresh',r['fresh'],'|fresh2',r['fresh2'],round(time.time()-t0),'s',flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print('FAIL',spec,repr(e)[:300],flush=True)
    json.dump(res,open(rp,'w'))
    torch.cuda.empty_cache()
print('BENCHDONE',flush=True)