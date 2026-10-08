import json,torch,os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P=r'C:\Projects\TinyDetect'; m='Oxidane/tmr-ai-text-detector'
tk=AutoTokenizer.from_pretrained(m); md=AutoModelForSequenceClassification.from_pretrained(m).eval()
for F in ['fresh','fresh2']:
    for fn in ['ai_text.json','human_text.json']:
        for x in json.load(open(P+'\\'+F+'\\'+fn,encoding='utf8')):
            e=tk(x['t'],truncation=True,max_length=512,return_tensors='pt')
            with torch.no_grad(): p=torch.softmax(md(**e).logits,-1)[0,1].item()
            print(F,fn[:2],round(p,3),x.get('src',x.get('s','')),len(x['t'].split()),flush=True)
print('TDONE')