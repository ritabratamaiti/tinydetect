# NOTE: onnxruntime's nodes_to_include is OR-ed with op_types_to_quantize (so it selects every MatMul); select via nodes_to_exclude instead.
# Per-layer sensitivity: quantise one MatMul at a time to int4 and measure output drift; keep the worst at 8-bit.
import os, sys, json, time
import numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import matmul_nbits_quantizer as mq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P=r'C:\Projects\TinyDetect'; O=sys.argv[1]; K=[int(k) for k in os.environ.get('KEEP8','12,24,36').split(',')]
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
meta=json.load(open(O+r'\meta.json')); tk=AutoTokenizer.from_pretrained(O+r'\tok')
WE=AutoModelForSequenceClassification.from_pretrained(meta['src']).roberta.embeddings.word_embeddings.weight.detach().numpy()
V,D=WE.shape; X=WE.reshape(V,D//32,32); s=np.abs(X).max(2,keepdims=True)/7.0; s[s==0]=1e-8; E=(np.clip(np.round(X/s),-8,7)*s).reshape(V,D).astype(np.float32)
# probe texts: held-out-style, not the fresh sets (fresh sets stay for final scoring)
rows=[json.loads(l) for l in open(P+r'\data\text_hape.jsonl',encoding='utf8')][:40]+[json.loads(l) for l in open(P+r'\data\text_v4_extra.jsonl',encoding='utf8')][::90][:40]
feeds=[]
for r in rows:
    ids=tk(r['t'],truncation=True,max_length=256)['input_ids']; feeds.append({'inputs_embeds':E[np.array(ids)][None],'attention_mask':np.ones((1,len(ids)),np.int64)})
so=ort.SessionOptions(); so.intra_op_num_threads=int(os.environ.get('THREADS','4'))
def logits(path):
    s=ort.InferenceSession(path,so,providers=['CPUExecutionProvider']); p=np.array([s.run(None,f)[0][0,1] for f in feeds]); return np.log(p/(1-p+1e-9)+1e-9)
base=logits(O+r'\body_fp32.onnx')
src=onnx.load(O+r'\body_fp32.onnx'); inits={t.name for t in src.graph.initializer}
W=[n.name for n in src.graph.node if n.op_type=='MatMul' and n.input[1] in inits]; log('weight matmuls',len(W))
sens={}
for i,name in enumerate(W):
    q=mq.MatMulNBitsQuantizer(onnx.load(O+r'\body_fp32.onnx'),bits=4,block_size=32,is_symmetric=True,nodes_to_exclude=[n for n in W if n!=name]); q.process(); q.model.save_model_to_file(O+r'\_one.onnx',use_external_data_format=False)
    nq=[n.op_type for n in onnx.load(O+r'\_one.onnx').graph.node].count('MatMulNBits'); assert nq==1, f'expected 1 quantised MatMul, got {nq}'
    d=float(np.abs(logits(O+r'\_one.onnx')-base).mean()); sens[name]=d
    if i%6==5: log(i+1,'/',len(W),name,round(d,4))
os.remove(O+r'\_one.onnx'); json.dump(sens,open(O+r'\sensitivity.json','w'),indent=1)
order=sorted(sens,key=sens.get,reverse=True); log('most sensitive',[(n.split('/')[-2] if '/' in n else n,round(sens[n],3)) for n in order[:10]])
for k in K:
    keep=order[:k]; rest=[n for n in W if n not in keep]
    q=mq.MatMulNBitsQuantizer(onnx.load(O+r'\body_fp32.onnx'),bits=4,block_size=32,is_symmetric=True,nodes_to_exclude=keep); q.process(); q.model.save_model_to_file(O+f'\\_s1.onnx',use_external_data_format=False)
    q=mq.MatMulNBitsQuantizer(onnx.load(O+f'\\_s1.onnx'),bits=8,block_size=32,is_symmetric=True); q.process(); q.model.save_model_to_file(O+f'\\body_mix{k}.onnx',use_external_data_format=False)
    os.remove(O+f'\\_s1.onnx')
    m=onnx.load(O+f'\\body_mix{k}.onnx'); bits=[a.i for n in m.graph.node if n.op_type=='MatMulNBits' for a in n.attribute if a.name=='bits']
    log('mix',k,'size',round(os.path.getsize(O+f'\\body_mix{k}.onnx')/1e6,1),'MB  nbits ops: 8-bit',bits.count(8),'4-bit',bits.count(4),' probe drift',round(float(np.abs(logits(O+f'\\body_mix{k}.onnx')-base).mean()),4))
log('MIXDONE')