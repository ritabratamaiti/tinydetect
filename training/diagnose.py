import json, os, glob, re, zlib, numpy as np, torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForImageClassification
P=r'C:\Projects\TinyDetect'; dev='cuda'
F=json.load(open(P+r'\fresh\results.json'))
H={x['t'][:50]:x for x in json.load(open(P+r'\fresh\human_text.json',encoding='utf8'))+json.load(open(P+r'\fresh\ai_text.json',encoding='utf8'))}
texts=json.load(open(P+r'\fresh\human_text.json',encoding='utf8'))+json.load(open(P+r'\fresh\ai_text.json',encoding='utf8'))
tt=AutoTokenizer.from_pretrained('Oxidane/tmr-ai-text-detector'); tm=AutoModelForSequenceClassification.from_pretrained('Oxidane/tmr-ai-text-detector').to(dev).eval()
st=AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
from transformers import BertForSequenceClassification
sm=BertForSequenceClassification.from_pretrained('google/bert_uncased_L-4_H-256_A-4',num_labels=2); sm.load_state_dict(torch.load(P+r'\models\text_student.pt')); sm=sm.to(dev).eval()
def feats(t):
    w=t.split(); sents=[s for s in re.split(r'[.!?]+\s',t) if s.strip()]; sl=[len(s.split()) for s in sents]
    return dict(words=len(w), comp=round(len(zlib.compress(t.encode()))/len(t.encode()),3), sent_mean=round(np.mean(sl),1), sent_std=round(np.std(sl),1), lower_start=round(np.mean([s.strip()[0].islower() for s in sents if s.strip()]),2), contractions=len(re.findall(r"\b\w+'(t|s|re|ve|ll|d|m)\b",t.lower())))
rows=[]
with torch.no_grad():
    for i,r in enumerate(texts):
        y=0 if i<24 else 1
        b=tt(r['t'],truncation=True,max_length=512,return_tensors='pt').to(dev); pt=torch.softmax(tm(**b).logits,-1)[0,1].item()
        b2=st(r['t'],truncation=True,max_length=256,return_tensors='pt').to(dev); ps=torch.softmax(sm(**b2).logits,-1)[0,1].item()
        rows.append(dict(y=y,src=r['src'],teacher=round(pt,3),student=round(ps,3),**feats(r['t'])))
for r in sorted(rows,key=lambda r:(r['y'],r['student'])): print(r)
# images
sw=AutoModelForImageClassification.from_pretrained('Smogy/SMOGY-Ai-images-detector').to(dev).eval()
mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
print('--- images (teacher swin p_ai, student p_ai, size, mode)')
st_img={x['src']:x['p'] for x in F['image']}
for f in sorted(glob.glob(P+r'\fresh\img\*.jpg')):
    n=os.path.basename(f); im=Image.open(f); sz=im.size; m=im.mode; im=im.convert('RGB').resize((224,224),Image.BICUBIC)
    a=torch.from_numpy(((np.asarray(im,np.float32)/255-mean)/std).transpose(2,0,1))[None].to(dev)
    with torch.no_grad(): pt=torch.softmax(sw(pixel_values=a).logits,-1)[0,0].item()
    ps=st_img.get(n)
    y=0 if n.startswith('real') else 1
    if (y==1 and ps<0.6) or (y==0 and ps>=0.3) or n in ('ai_mj_0.jpg','real_5.jpg','ai_dalle3_0.jpg','real_0.jpg'):
        print(n,'y',y,'teacher',round(pt,3),'student',round(ps,3),sz,m, os.path.getsize(f)//1024,'KB')