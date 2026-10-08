import os, json, random, re, numpy as np, torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
P=r'C:\Projects\TinyDetect'; random.seed(0)
def log(*a):
    import time; s=time.strftime('%H:%M:%S')+' '+' '.join(map(str,a)); print(s,flush=True); open(P+r'\logs\image.log','a').write(s+'\n')
meta=[json.loads(l) for l in open(P+r'\data\img_meta.jsonl') if l.strip()]+[json.loads(l) for l in open(P+r'\data\img_meta_hemg.jsonl') if l.strip()]
items=[]
for m in meta:
    if m['src']=='hemg' and not (m['f'].startswith('train/q')): continue
    try: im=Image.open(os.path.join(P,'data','img',*m['f'].split('/'))).convert('RGB')
    except Exception: continue
    items.append((im.resize((224,224),Image.BICUBIC),m['y'],m['src'],m.get('split','')))
eddy_tr=[x for x in items if x[2]=='eddyfox' and x[3]=='train']; eddy_te=[x for x in items if x[2]=='eddyfox' and x[3]=='test']
hem=[x for x in items if x[2]=='hemg']; random.shuffle(hem); nh=int(.2*len(hem))
te_in=eddy_te; edd=hem[:nh]; tr=eddy_tr+hem[nh:]
def totensor(im): a=np.asarray(im,dtype=np.float32)[:,:,::-1]/255.; return np.ascontiguousarray(a.transpose(2,0,1))
def metrics(y,p,name):
    y=np.array(y);p=np.array(p);auc=roc_auc_score(y,p);fpr,tpr,_=roc_curve(y,p);t5=float(np.interp(0.05,fpr,tpr));fp=float((p[y==0]>0.5).mean())
    log(f'{name}: AUROC={auc:.4f} TPR@5%FPR={t5:.3f} FPR@0.5(real flagged)={fp:.3f}'); return dict(auc=auc,tpr5=t5,fpr05=fp)
import subprocess,sys; print(subprocess.run([sys.executable,P+r'\scripts\wq_image.py'],capture_output=True,text=True).stdout,flush=True)
class R(CalibrationDataReader):
    def __init__(s):
        real=[x for x in tr if x[1]==0]; ai=[x for x in tr if x[1]==1]
        s.it=iter([{'pixel_values':totensor(x[0])[None]} for x in random.sample(real,150)+random.sample(ai,150)])
    def get_next(s): return next(s.it,None)
os.makedirs(P+r'\web\models',exist_ok=True)

def run(path,ims):
    s=ort.InferenceSession(path); out=[]
    for x in ims:
        l=s.run(None,{'pixel_values':totensor(x[0])[None]})[0][0]; e=np.exp(l-l.max()); out.append(float(e[1]/e.sum()))
    return out
res={}
txt=open(P+r'\logs\image.log').read()
for nm in ['swin_b_347MB','vit_58MB','ensemble']:
    for sp,tag in [('in','in-dist'),('ood','OOD')]:
        m=re.search(r'TEACHER '+nm+' '+tag+r': AUROC=([\d.]+) TPR@5%FPR=([\d.]+) FPR@0.5\(real flagged\)=([\d.]+)',txt)
        if m: res[nm+'_'+sp]=dict(auc=float(m.group(1)),tpr5=float(m.group(2)),fpr05=float(m.group(3)))
for sp,tag in [('in','in-dist'),('ood','OOD')]:
    m=re.search(r'STUDENT mobilevit-xxs fp32 '+tag+r': AUROC=([\d.]+) TPR@5%FPR=([\d.]+) FPR@0.5\(real flagged\)=([\d.]+)',txt)
    res['student_fp32_'+sp]=dict(auc=float(m.group(1)),tpr5=float(m.group(2)),fpr05=float(m.group(3)))
yin=[x[1] for x in te_in]; yood=[x[1] for x in edd]
res['onnx_fp32_in']=metrics(yin,run(P+r'\models\image_fp32.onnx',te_in),'ONNX fp32 in-dist (sanity)')
res['final_int8_in']=metrics(yin,run(P+r'\web\models\image_int8.onnx',te_in),'FINAL shipped weight-only int8 in-dist')
res['final_int8_ood']=metrics(yood,run(P+r'\web\models\image_int8.onnx',edd),'FINAL shipped weight-only int8 OOD')
res['size_MB']={'teacher_swin':347,'teacher_vit':58,'student_fp32':round(os.path.getsize(P+r'\models\image_fp32.onnx')/1e6,2),'student_int8':round(os.path.getsize(P+r'\web\models\image_int8.onnx')/1e6,2)}
log('sizes',res['size_MB']); json.dump(res,open(P+r'\logs\image_results.json','w'),indent=1); log('IMAGEDONE')