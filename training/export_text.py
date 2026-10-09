# Export a RoBERTa-family detector for the browser: int4 word embeddings (decoded in JS) + ONNX body taking inputs_embeds.
import os, sys, json, time, gzip
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np, torch, onnx, onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import matmul_nbits_quantizer as mq
P=r'C:\Projects\TinyDetect'; SRC=sys.argv[1]; O=sys.argv[2]; os.makedirs(O,exist_ok=True)
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
tk=AutoTokenizer.from_pretrained(SRC); m=AutoModelForSequenceClassification.from_pretrained(SRC).eval().float()
WE=m.roberta.embeddings.word_embeddings.weight.detach().numpy()   # [V,768]
class Body(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,inputs_embeds,attention_mask):
        lg=s.m(inputs_embeds=inputs_embeds,attention_mask=attention_mask).logits
        return torch.softmax(lg,-1)
b=Body(m).eval()
ids=tk('A quick check of the export path with a short sentence.',return_tensors='pt')
with torch.no_grad():
    ref=torch.softmax(m(**ids).logits,-1); emb=m.roberta.embeddings.word_embeddings(ids['input_ids']); alt=b(emb,ids['attention_mask'])
log('inputs_embeds parity max diff',float((ref-alt).abs().max()))
fp=O+r'\body_fp32.onnx'
torch.onnx.export(b,(emb,ids['attention_mask']),fp,input_names=['inputs_embeds','attention_mask'],output_names=['probs'],dynamic_axes={'inputs_embeds':{1:'T'},'attention_mask':{1:'T'},'probs':{}},opset_version=17,dynamo=False)
log('exported',round(os.path.getsize(fp)/1e6,1),'MB (includes unused word-embedding table:',round(WE.nbytes/1e6,1),'MB)')
# drop the unused word-embedding initializer from the graph (inputs_embeds bypasses it)
mm=onnx.load(fp); names={n for nd in mm.graph.node for n in nd.input}
dead=[t for t in mm.graph.initializer if t.name not in names]
for t in dead: mm.graph.initializer.remove(t)
onnx.save(mm,fp); log('removed',len(dead),'unused initializers; body now',round(os.path.getsize(fp)/1e6,1),'MB')
# int4 word embeddings: one fp16 absmax scale per block of 32 values, packed two per byte (low nibble first), values 0..15 centred at 8
B=32; V_,D_=WE.shape; Xb=WE.reshape(V_,D_//B,B); s=np.abs(Xb).max(2,keepdims=True)/7.0; s[s==0]=1e-8
q=(np.clip(np.round(Xb/s),-8,7).astype(np.int16)+8).reshape(V_,D_)
packed=(q[:,0::2]|(q[:,1::2]<<4)).astype(np.uint8)
packed.tofile(O+r'\emb_int4.bin'); s.astype(np.float16).reshape(-1).tofile(O+r'\emb_scale_f16.bin')
deq=((q.reshape(V_,D_//B,B).astype(np.float32)-8)*s.astype(np.float16).astype(np.float32)).reshape(V_,D_); np.save(O+r'\emb_deq.npy',deq)
log('emb int4 (block 32)',round(packed.nbytes/1e6,2),'MB + scales',round(s.size*2/1e6,2),'MB; rel err',float(np.abs(deq-WE).mean()/np.abs(WE).mean()))
# quantised bodies
inits={t.name for t in onnx.load(fp).graph.initializer}
def nb(dst,bits,algo=None):
    if os.path.exists(dst): return
    q_=mq.MatMulNBitsQuantizer(onnx.load(fp),bits=bits,block_size=32,is_symmetric=True,algo_config=algo); q_.process(); q_.model.save_model_to_file(dst,use_external_data_format=False)
nb(O+r'\body_nb8.onnx',8); log('nb8',round(os.path.getsize(O+r'\body_nb8.onnx')/1e6,1),'MB')
nb(O+r'\body_hqq4.onnx',4,mq.HQQWeightOnlyQuantConfig(block_size=32,bits=4)); log('hqq4',round(os.path.getsize(O+r'\body_hqq4.onnx')/1e6,1),'MB')
nb(O+r'\body_nb4.onnx',4); log('nb4',round(os.path.getsize(O+r'\body_nb4.onnx')/1e6,1),'MB')
from onnxruntime.transformers.float16 import convert_float_to_float16
onnx.save(convert_float_to_float16(onnx.load(fp),keep_io_types=True),O+r'\body_fp16.onnx'); log('fp16',round(os.path.getsize(O+r'\body_fp16.onnx')/1e6,1),'MB')
json.dump(dict(emb_block=32,src=SRC,vocab=int(WE.shape[0]),dim=int(WE.shape[1]),pad=int(tk.pad_token_id),cls=int(tk.cls_token_id),sep=int(tk.sep_token_id),unk=int(tk.unk_token_id),max_len=512,ai_index=1),open(O+r'\meta.json','w'))
tk.save_pretrained(O+r'\tok')
log('EXPORTDONE')