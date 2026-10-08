# Custom weight-only per-channel int8 for ONNX: weights stored int8 + fp32 scale, DequantizeLinear in-graph, activations stay fp32.
import onnx, numpy as np, json, os, re, random
from onnx import numpy_helper, helper, TensorProto
P=r'C:\Projects\TinyDetect'
m=onnx.load(P+r'\models\image_fp32.onnx'); g=m.graph
cons={}
for n in g.node:
    for i,x in enumerate(n.input): cons.setdefault(x,[]).append((n,i))
new_inits=[]; new_nodes=[]; done=0; skipped=0; before=0; after=0
inits={t.name:t for t in g.initializer}
for name,t in list(inits.items()):
    w=numpy_helper.to_array(t)
    before+=w.nbytes
    if w.dtype!=np.float32 or w.ndim<2 or w.size<1024: after+=w.nbytes; continue
    uses=cons.get(name,[])
    if not uses: after+=w.nbytes; continue
    n0,i0=uses[0]
    if n0.op_type=='Conv' and i0==1: axis=0
    elif n0.op_type=='MatMul' and i0==1: axis=w.ndim-1
    elif n0.op_type=='Gemm' and i0==1:
        tb=[a.i for a in n0.attribute if a.name=='transB']; axis=0 if (tb and tb[0]==1) else 1
    else: skipped+=1; after+=w.nbytes; continue
    red=tuple(i for i in range(w.ndim) if i!=axis)
    s=np.abs(w).max(axis=red)/127.0; s[s==0]=1e-8
    shp=[1]*w.ndim; shp[axis]=-1
    q=np.clip(np.round(w/s.reshape(shp)),-127,127).astype(np.int8)
    g.initializer.remove(t)
    qn,sn,zn=name+'_q',name+'_s',name+'_z'
    new_inits+= [numpy_helper.from_array(q,qn),numpy_helper.from_array(s.astype(np.float32),sn),numpy_helper.from_array(np.zeros_like(s,dtype=np.int8),zn)]
    new_nodes.append(helper.make_node('DequantizeLinear',[qn,sn,zn],[name],axis=axis,name=name+'_dq'))
    after+=q.nbytes+s.nbytes*1+s.size; done+=1
g.initializer.extend(new_inits)
for nd in reversed(new_nodes): g.node.insert(0,nd)
onnx.checker.check_model(m)
os.makedirs(P+r'\web\models',exist_ok=True); onnx.save(m,P+r'\web\models\image_int8.onnx')
print('quantized',done,'skipped',skipped,'MB before',round(before/1e6,2),'after~',round(after/1e6,2),'file',round(os.path.getsize(P+r'\web\models\image_int8.onnx')/1e6,2),flush=True)
