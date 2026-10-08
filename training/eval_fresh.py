import json, os, glob, numpy as np, onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer
P=r'C:\Projects\TinyDetect'; W=P+r'\web\models'
voc=json.load(open(W+r'\text_vocab.json')); keep=voc['keep_ids']; remap={k:i for i,k in enumerate(keep)}; dim=voc['dim']
emb=np.fromfile(W+r'\text_emb_int4.bin',dtype=np.uint8).reshape(len(keep),dim//2); sc=np.fromfile(W+r'\text_emb_scale_f16.bin',dtype=np.float16).astype(np.float32)
lo=(emb&15).astype(np.float32)-8; hi=(emb>>4).astype(np.float32)-8; E=np.empty((len(keep),dim),np.float32); E[:,0::2]=lo; E[:,1::2]=hi; E*=sc[:,None]
tok=AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4'); ts=ort.InferenceSession(W+r'\text_body_int8.onnx')
def ptext(t):
    ids=[remap.get(i,voc['unk']) for i in tok(t,add_special_tokens=False)['input_ids']]
    Wn=voc['max_len']-2; wins=[ids[i:i+Wn] for i in range(0,len(ids),Wn)][:6]
    if len(wins)>1 and len(wins[-1])<40: wins.pop()
    ps=[]
    for w in wins:
        x=[voc['cls']]+w+[voc['sep']]; ps.append(float(ts.run(None,{'inputs_embeds':E[x][None],'attention_mask':np.ones((1,len(x)),np.int64)})[0][0,1]))
    return float(np.mean(ps))
isess=ort.InferenceSession(W+r'\image_int8.onnx')
def pimg(f):
    im=Image.open(f).convert('RGB').resize((224,224),Image.BICUBIC); a=np.asarray(im,dtype=np.float32)[:,:,::-1]/255.
    l=isess.run(None,{'pixel_values':np.ascontiguousarray(a.transpose(2,0,1))[None]})[0][0]; e=np.exp(l-l.max()); return float(e[1]/e.sum())
def band(p): return 'AI' if p>=0.9 else 'possibly AI' if p>=0.6 else 'unclear' if p>=0.3 else 'human'
out={'text':[],'image':[]}
for f,y in [(P+r'\fresh\human_text.json',0),(P+r'\fresh\ai_text.json',1)]:
    for r in json.load(open(f,encoding='utf8')):
        p=ptext(r['t']); out['text'].append({'src':r['src'],'y':y,'p':p})
for f in (sorted(glob.glob(P+r'\fresh\img\*.jpg')) if os.environ.get('IMG','1')=='1' else []):
    y=0 if os.path.basename(f).startswith('real') else 1
    try: out['image'].append({'src':os.path.basename(f),'y':y,'p':pimg(f)})
    except Exception as e: print('bad',f,e)
for k in ['text','image']:
    rs=out[k]; h=[r for r in rs if r['y']==0]; a=[r for r in rs if r['y']==1]
    print(f'== {k}: human/real n={len(h)} flagged(p>=0.5)={sum(r["p"]>=0.5 for r in h)}  | AI n={len(a)} caught(p>=0.5)={sum(r["p"]>=0.5 for r in a)}')
    for r in sorted(rs,key=lambda r:(r['y'],r['p'])): print(f'   y={r["y"]} p={r["p"]:.3f} {band(r["p"]):8s} {r["src"]}')
import json as _j
for k in ['text','image']:
    rs=out[k]; out[k+'_summary']={'human_n':sum(r['y']==0 for r in rs),'ai_n':sum(r['y']==1 for r in rs)}
    for b in ['AI','possibly AI','unclear','human']:
        out[k+'_summary']['human_'+b]=sum(r['y']==0 and band(r['p'])==b for r in rs); out[k+'_summary']['ai_'+b]=sum(r['y']==1 and band(r['p'])==b for r in rs)
    print(k,out[k+'_summary'])
json.dump(out,open(P+r'\fresh\results.json','w'),indent=1)