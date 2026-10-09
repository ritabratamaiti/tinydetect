# Community Forensics ViT-S/16 (MIT) -> ONNX -> compressed variants, each scored against fp32.
import os, json, glob, time, gzip, random
os.environ.setdefault('HF_HUB_DISABLE_XET','1'); os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np, torch, timm, onnx
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from timm.layers import resample_abs_pos_embed
import onnxruntime as ort
P=r'C:\Projects\TinyDetect'; O=P+r'\models\commfor'; os.makedirs(O,exist_ok=True)
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
random.seed(0)
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
def prep(path,size):
    rs=round(size*440/384)
    im=Image.open(path).convert('RGB'); w,h=im.size; s=rs/min(w,h); im=im.resize((max(rs,round(w*s)),max(rs,round(h*s))),Image.BILINEAR)
    w,h=im.size; l=(w-size)//2; t=(h-size)//2; im=im.crop((l,t,l+size,t+size))
    return ((np.asarray(im,np.float32)/255.-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
# image sets
def fresh(F): return [(f,0 if os.path.basename(f).startswith('real') else 1) for f in sorted(glob.glob(P+'\\'+F+r'\img\*.jpg'))]
meta=[json.loads(l) for l in open(P+r'\data\img_meta_v4.jsonl') if l.strip()]
ai=[m for m in meta if m['y']==1]; rl=[m for m in meta if m['y']==0]; random.shuffle(ai); random.shuffle(rl)
calib=[(P+'\\data\\img\\'+m['f'].replace('/','\\'),m['y']) for m in ai[:250]+rl[:250]]
SETS={'calib':calib,'fresh':fresh('fresh'),'fresh2':fresh('fresh2')}
def auc(ys,ps):
    from sklearn.metrics import roc_auc_score; return round(float(roc_auc_score(ys,ps)),4)
def bands(ys,ps):
    ys=np.array(ys); ps=np.array(ps); h=ps[ys==0]; a=ps[ys==1]
    return dict(real_flag=int((h>=0.6).sum()),real_n=len(h),ai_caught=int((a>=0.6).sum()),ai_n=len(a))
def build(size):
    m=timm.create_model('vit_small_patch16_384',pretrained=False,num_classes=1,img_size=size)
    sd=load_file(hf_hub_download('OwensLab/commfor-model-384','model.safetensors')); sd={k[4:] if k.startswith('vit.') else k:v for k,v in sd.items()}
    if size!=384: sd['pos_embed']=resample_abs_pos_embed(sd['pos_embed'],(size//16,size//16),num_prefix_tokens=1)
    m.load_state_dict(sd,strict=True); return m.eval()
# 1) resolution sweep in PyTorch (fewer patches = faster)
res={}; cache={}
dev='cuda'
for size in [384,320,288,256,224]:
    m=build(size).to(dev); r={}
    for k,items in SETS.items():
        X=[prep(f,size) for f,_ in items]; cache[(size,k)]=X
        with torch.no_grad(): lg=[m(torch.from_numpy(x).to(dev))[0,0].item() for x in X]
        r[k+'_logits']=lg; ps=1/(1+np.exp(-np.array(lg))); ys=[y for _,y in items]; r[k]=dict(auc=auc(ys,ps),**bands(ys,ps))
    res[f'torch_{size}']=r; log('TORCH',size,{k:r[k] for k in SETS})
    del m; torch.cuda.empty_cache()
json.dump(res,open(O+r'\sweep.json','w'))
# 2) ONNX variants at the chosen sizes
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.transformers import optimizer as tro
from onnxruntime.transformers.float16 import convert_float_to_float16
def wq8(src,dst):
    from onnx import numpy_helper, helper
    m=onnx.load(src); g=m.graph; cons={}
    for n in g.node:
        for i,x in enumerate(n.input): cons.setdefault(x,[]).append((n,i))
    ni=[]; nn=[]
    for t in list(g.initializer):
        w=numpy_helper.to_array(t)
        if w.dtype!=np.float32 or w.ndim<2 or w.size<1024: continue
        u=cons.get(t.name,[])
        if not u: continue
        n0,i0=u[0]
        if n0.op_type=='Conv' and i0==1: ax=0
        elif n0.op_type=='MatMul' and i0==1: ax=w.ndim-1
        elif n0.op_type=='Gemm' and i0==1: tb=[a.i for a in n0.attribute if a.name=='transB']; ax=0 if (tb and tb[0]==1) else 1
        else: continue
        red=tuple(i for i in range(w.ndim) if i!=ax); s=np.abs(w).max(axis=red)/127.; s[s==0]=1e-8; shp=[1]*w.ndim; shp[ax]=-1
        q=np.clip(np.round(w/s.reshape(shp)),-127,127).astype(np.int8); g.initializer.remove(t)
        ni+=[numpy_helper.from_array(q,t.name+'_q'),numpy_helper.from_array(s.astype(np.float32),t.name+'_s'),numpy_helper.from_array(np.zeros_like(s,dtype=np.int8),t.name+'_z')]
        nn.append(helper.make_node('DequantizeLinear',[t.name+'_q',t.name+'_s',t.name+'_z'],[t.name],axis=ax,name=t.name+'_dq'))
    g.initializer.extend(ni)
    for nd in reversed(nn): g.node.insert(0,nd)
    onnx.save(m,dst)
def nbits(src,dst,block=32,sym=True):
    m=onnx.load(src); q=MatMulNBitsQuantizer(m,block_size=block,is_symmetric=sym); q.process(); q.model.save_model_to_file(dst,use_external_data_format=False)
def fuse(src,dst):
    o=tro.optimize_model(src,model_type='vit',num_heads=6,hidden_size=384,opt_level=1,use_gpu=False); o.save_model_to_file(dst)
def fp16(src,dst):
    m=onnx.load(src); m=convert_float_to_float16(m,keep_io_types=True); onnx.save(m,dst)
def gzsz(p): return round(len(gzip.compress(open(p,'rb').read(),6))/1e6,2)
def run(path,size,threads=4):
    so=ort.SessionOptions(); so.intra_op_num_threads=threads; so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s=ort.InferenceSession(path,so,providers=['CPUExecutionProvider']); inp=s.get_inputs()[0].name
    r={}
    for k,items in SETS.items():
        lg=[float(s.run(None,{inp:x})[0].reshape(-1)[0]) for x in cache[(size,k)]]; r[k+'_logits']=lg
        ps=1/(1+np.exp(-np.array(lg))); ys=[y for _,y in items]; r[k]=dict(auc=auc(ys,ps),**bands(ys,ps))
    x=cache[(size,'fresh2')][0]
    for _ in range(3): s.run(None,{inp:x})
    so1=ort.SessionOptions(); so1.intra_op_num_threads=1; s1=ort.InferenceSession(path,so1,providers=['CPUExecutionProvider'])
    for _ in range(2): s1.run(None,{inp:x})
    t=time.time(); [s1.run(None,{inp:x}) for _ in range(10)]; r['ms_1thread']=round((time.time()-t)*100,1)
    t=time.time(); [s.run(None,{inp:x}) for _ in range(10)]; r['ms_4thread']=round((time.time()-t)*100,1)
    return r
for size in [384,320,288]:
    m=build(size).cpu(); base=O+f'\\cf{size}_fp32.onnx'
    torch.onnx.export(m,torch.randn(1,3,size,size),base,input_names=['pixel_values'],output_names=['logits'],opset_version=17,dynamo=False)
    variants={'fp32':base}
    try: fuse(base,O+f'\\cf{size}_fused.onnx'); variants['fused']=O+f'\\cf{size}_fused.onnx'
    except Exception as e: log('fuse fail',repr(e)[:200])
    wq8(base,O+f'\\cf{size}_wq8.onnx'); variants['wq8']=O+f'\\cf{size}_wq8.onnx'
    quantize_dynamic(base,O+f'\\cf{size}_dyn8.onnx',weight_type=QuantType.QInt8,per_channel=True); variants['dyn8']=O+f'\\cf{size}_dyn8.onnx'
    nbits(base,O+f'\\cf{size}_nb4.onnx'); variants['nb4']=O+f'\\cf{size}_nb4.onnx'
    nbits(base,O+f'\\cf{size}_nb4b128.onnx',block=128); variants['nb4b128']=O+f'\\cf{size}_nb4b128.onnx'
    if 'fused' in variants:
        nbits(variants['fused'],O+f'\\cf{size}_fused_nb4.onnx'); variants['fused_nb4']=O+f'\\cf{size}_fused_nb4.onnx'
    fp16(base,O+f'\\cf{size}_fp16.onnx'); variants['fp16']=O+f'\\cf{size}_fp16.onnx'
    ref=None
    for name,path in variants.items():
        try:
            r=run(path,size)
            if name=='fp32': ref=r
            r['max_logit_diff_vs_fp32']=round(float(max(abs(a-b) for k in SETS for a,b in zip(r[k+'_logits'],ref[k+'_logits']))),4)
            r['MB']=round(os.path.getsize(path)/1e6,2); r['MB_gzip']=gzsz(path)
            res[f'onnx_{size}_{name}']=r; log('ONNX',size,name,'MB',r['MB'],'gz',r['MB_gzip'],'ms1',r['ms_1thread'],'ms4',r['ms_4thread'],'diff',r['max_logit_diff_vs_fp32'],{k:r[k] for k in SETS})
        except Exception as e: log('VARIANT FAIL',size,name,repr(e)[:300])
        json.dump(res,open(O+r'\sweep.json','w'))
log('OPTDONE')