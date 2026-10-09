# Weight-quantisation sweep for the Community Forensics ViT-S/16 @384 ONNX model.
import os, json, glob, time, gzip, random, sys
import numpy as np, onnx, onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import matmul_nbits_quantizer as mq
from onnxruntime.quantization import CalibrationDataReader
P=r'C:\Projects\TinyDetect'; O=P+r'\models\commfor'; SRC=O+r'\cf384_fp32.onnx'; S=384
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
random.seed(0)
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
def prep(path):
    im=Image.open(path).convert('RGB'); w,h=im.size; s=440/min(w,h); im=im.resize((max(440,round(w*s)),max(440,round(h*s))),Image.BILINEAR)
    w,h=im.size; l=(w-S)//2; t=(h-S)//2; im=im.crop((l,t,l+S,t+S))
    return ((np.asarray(im,np.float32)/255.-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
def fresh(F): return [(f,0 if os.path.basename(f).startswith('real') else 1) for f in sorted(glob.glob(P+'\\'+F+r'\img\*.jpg'))]
meta=[json.loads(l) for l in open(P+r'\data\img_meta_v4.jsonl') if l.strip()]
ai=[m for m in meta if m['y']==1]; rl=[m for m in meta if m['y']==0]; random.shuffle(ai); random.shuffle(rl)
pth=lambda m:P+'\\data\\img\\'+m['f'].replace('/','\\')
calib=[(pth(m),m['y']) for m in ai[:250]+rl[:250]]
gptq_imgs=[pth(m) for m in ai[250:282]+rl[250:282]]   # separate images for GPTQ, never scored
SETS={'calib':calib,'fresh':fresh('fresh'),'fresh2':fresh('fresh2')}
X={k:[prep(f) for f,_ in v] for k,v in SETS.items()}; Y={k:[y for _,y in v] for k,v in SETS.items()}
log('prepped')
from sklearn.metrics import roc_auc_score
def score(path,ref=None):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    s=ort.InferenceSession(path,so,providers=['CPUExecutionProvider']); inp=s.get_inputs()[0].name; r={}
    for k in SETS:
        lg=np.array([float(s.run(None,{inp:x})[0].reshape(-1)[0]) for x in X[k]]); r[k+'_logits']=lg.tolist()
        ps=1/(1+np.exp(-lg)); ys=np.array(Y[k]); r[k]=dict(auc=round(float(roc_auc_score(ys,ps)),4),real_flag=int((ps[ys==0]>=0.6).sum()),ai_caught=int((ps[ys==1]>=0.6).sum()))
    if ref: r['mean_abs_logit_diff']=round(float(np.mean([abs(a-b) for k in SETS for a,b in zip(r[k+'_logits'],ref[k+'_logits'])])),3)
    so1=ort.SessionOptions(); so1.intra_op_num_threads=1; s1=ort.InferenceSession(path,so1,providers=['CPUExecutionProvider']); x=X['fresh2'][0]
    for _ in range(2): s1.run(None,{inp:x})
    t=time.time(); [s1.run(None,{inp:x}) for _ in range(8)]; r['ms_1thread']=round((time.time()-t)/8*1000,1)
    r['MB']=round(os.path.getsize(path)/1e6,2); r['MB_gzip']=round(len(gzip.compress(open(path,'rb').read(),6))/1e6,2)
    return r
class Reader(CalibrationDataReader):
    def __init__(self): self.it=iter([{'pixel_values':prep(f)} for f in gptq_imgs])
    def get_next(self): return next(self.it,None)
def names(pred): 
    m=onnx.load(SRC); return [n.name for n in m.graph.node if n.op_type=='MatMul' and pred(n.name) and any(i.startswith('onnx::') or 'weight' in i or i in {t.name for t in m.graph.initializer} for i in n.input[1:])]
init_names={t.name for t in onnx.load(SRC).graph.initializer}
W=[n.name for n in onnx.load(SRC).graph.node if n.op_type=='MatMul' and n.input[1] in init_names]
MLP=[n for n in W if '/mlp/' in n]; ATT=[n for n in W if '/attn/' in n]
EDGE=[n for n in W if 'blocks.0/' in n or 'blocks.11/' in n]
log('weight matmuls',len(W),'mlp',len(MLP),'attn',len(ATT))
def q(dst,bits=4,block=32,sym=True,algo=None,include=None,exclude=None,src=SRC,acc=None):
    m=onnx.load(src)
    if include is not None: exclude=[n for n in W if n not in include]   # nodes_to_include is OR-ed with op type in ORT, so use exclude
    qz=mq.MatMulNBitsQuantizer(m,bits=bits,block_size=block,is_symmetric=sym,accuracy_level=acc,nodes_to_exclude=exclude,algo_config=algo)
    qz.process(); qz.model.save_model_to_file(dst,use_external_data_format=False); return dst
def chain(dst,steps):
    cur=SRC
    for i,(kw) in enumerate(steps):
        out=dst if i==len(steps)-1 else dst+f'.step{i}.onnx'; q(out,src=cur,**kw); cur=out
    return dst
EXP={
 'nb8_b32':lambda d:q(d,bits=8,block=32),
 'nb8_b128':lambda d:q(d,bits=8,block=128),
 'nb4_asym_b32':lambda d:q(d,sym=False),
 'nb4_b16':lambda d:q(d,block=16),
 'hqq4_b32':lambda d:q(d,algo=mq.HQQWeightOnlyQuantConfig(block_size=32,bits=4)),
 'gptq4_b32':lambda d:q(d,algo=mq.GPTQWeightOnlyQuantConfig(calibration_data_reader=Reader(),block_size=32,perchannel=False)),
 'mix_mlp4hqq_att8':lambda d:chain(d,[dict(algo=mq.HQQWeightOnlyQuantConfig(block_size=32,bits=4),include=MLP),dict(bits=8,block=32,include=ATT)]),
 'mix_mlp4hqq_att8_edge8':lambda d:chain(d,[dict(algo=mq.HQQWeightOnlyQuantConfig(block_size=32,bits=4),include=[n for n in MLP if n not in EDGE]),dict(bits=8,block=32,include=ATT+[n for n in MLP if n in EDGE])]),
 'fp16':lambda d:(lambda m:onnx.save(__import__('onnxruntime.transformers.float16',fromlist=['x']).convert_float_to_float16(m,keep_io_types=True),d))(onnx.load(SRC)),
 'nb8_b32_acc4':lambda d:q(d,bits=8,block=32,acc=4),
}
only=sys.argv[1:] or list(EXP)
rp=O+r'\quant_sweep.json'; res=json.load(open(rp)) if os.path.exists(rp) else {}
if 'fp32' not in res: res['fp32']=score(SRC); log('fp32',{k:res['fp32'][k] for k in ['calib','fresh','fresh2','ms_1thread','MB']})
ref=res['fp32']
for name in only:
    d=O+f'\\cf384_{name}.onnx'
    try:
        t=time.time(); EXP[name](d); r=score(d,ref); res[name]=r
        log('Q',name,'MB',r['MB'],'gz',r['MB_gzip'],'ms1',r['ms_1thread'],'dlogit',r['mean_abs_logit_diff'],{k:r[k] for k in SETS},round(time.time()-t),'s')
    except Exception as e:
        import traceback; traceback.print_exc(); log('QFAIL',name,repr(e)[:300])
    for f in glob.glob(d+'.step*'): os.remove(f)
    json.dump(res,open(rp,'w'))
log('QSWEEPDONE')