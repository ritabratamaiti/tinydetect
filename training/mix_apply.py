# Apply the sensitivity ranking from fs_orig to another export dir: top-K MatMuls at 8-bit, rest int4 (selection via nodes_to_exclude).
import os, sys, json, onnx
from onnxruntime.quantization import matmul_nbits_quantizer as mq
P=r'C:\Projects\TinyDetect'; O=sys.argv[1]; Ks=[int(k) for k in sys.argv[2].split(',')]
sens=json.load(open(P+r'\models\fs_orig\sensitivity.json')); order=sorted(sens,key=sens.get,reverse=True)
for K in Ks:
    keep=order[:K]
    q=mq.MatMulNBitsQuantizer(onnx.load(O+r'\body_fp32.onnx'),bits=4,block_size=32,is_symmetric=True,nodes_to_exclude=keep); q.process(); q.model.save_model_to_file(O+r'\_s1.onnx',use_external_data_format=False)
    q=mq.MatMulNBitsQuantizer(onnx.load(O+r'\_s1.onnx'),bits=8,block_size=32,is_symmetric=True); q.process(); q.model.save_model_to_file(O+f'\\body_mix{K}.onnx',use_external_data_format=False); os.remove(O+r'\_s1.onnx')
    m=onnx.load(O+f'\\body_mix{K}.onnx'); bits=[a.i for n in m.graph.node if n.op_type=='MatMulNBits' for a in n.attribute if a.name=='bits']
    assert bits.count(8)==K and bits.count(4)==72-K, (bits.count(8),bits.count(4))
    print('mix',K,round(os.path.getsize(O+f'\\body_mix{K}.onnx')/1e6,1),'MB ok',flush=True)
print('MIXAPPLYDONE',flush=True)