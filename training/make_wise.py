import os, sys
os.environ.setdefault('HF_HUB_OFFLINE','1')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
P=r'C:\Projects\TinyDetect'; RID='fakespot-ai/roberta-base-ai-text-detection-v1'; A=float(sys.argv[1]); OUT=P+rf'\models\fakespot_wise{int(A*100)}'
o=AutoModelForSequenceClassification.from_pretrained(RID); t=AutoModelForSequenceClassification.from_pretrained(P+r'\models\fakespot_tuned')
so,st=o.state_dict(),t.state_dict()
o.load_state_dict({k:(so[k].float()*(1-A)+st[k].float()*A).to(so[k].dtype) if so[k].is_floating_point() else st[k] for k in so})
o.save_pretrained(OUT); AutoTokenizer.from_pretrained(RID).save_pretrained(OUT); print('saved',OUT,flush=True)