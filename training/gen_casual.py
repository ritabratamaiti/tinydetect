import os, json, random, re, time, sys
os.environ['HF_HUB_DISABLE_XET']='1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
P=r'C:\Projects\TinyDetect'; random.seed(int(os.environ.get('SEED','1')))
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
GENRES=['a quick Slack message to your team','a short email to a coworker','a casual email to a friend','a text message to a friend','a Reddit comment giving advice','a Reddit post asking for help','a forum reply','a Hacker News comment','a short product review','a restaurant review','a reply to a customer support ticket','an FAQ answer','a recipe intro for a food blog','a short sports recap','a short news brief','a personal blog paragraph','a LinkedIn post','a tweet thread','a group chat message','a note to a neighbour','a short movie review','a message to a landlord','a reply on a parenting forum','a comment on a YouTube video','a short app store review','a quick update email to a client','a short explanation for a beginner','a travel tip post','a short reflection journal entry','a Discord message to your gaming group']
TOPICS=['being late to a meeting','a broken dishwasher','switching to a standing desk','a weekend hiking trip','learning to bake sourdough','a new coffee machine','moving apartments','a flaky internet connection','a team lunch','a delayed package','the local football match','a houseplant that keeps dying','a noisy upstairs neighbour','choosing a used car','a job interview next week','a birthday party','learning Python','a power cut last night','running a first 10k','a dentist appointment','a lost cat','a budget spreadsheet','a camping stove','a rainy holiday','a book club pick','a bug in the checkout page','a gym membership','a cancelled flight','a garden vegetable patch','a podcast recommendation','a school fundraiser','a cheap laptop','a parking ticket','a phone battery dying fast','a new puppy','a home office setup','a weekend farmers market','a broken bike chain','a pizza place that closed','a concert ticket mixup']
STY=['Keep it casual and short.','Write it like a real person would, informal.','Be brief and friendly.','Keep it under 120 words.','Make it sound natural, not formal.','']
mid=sys.argv[1]; n=int(sys.argv[2]); tag=mid.split('/')[-1]
tk=AutoTokenizer.from_pretrained(mid); tk.padding_side='left'
if tk.pad_token is None: tk.pad_token=tk.eos_token
m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16).cuda().eval()
PRE=re.compile(r'^(sure|okay|ok|here(\'s| is)|absolutely|certainly|of course)[^\n]*\n+',re.I)
out=[]; bs=24
while len(out)<n:
    ps=[f'Write {random.choice(GENRES)} about {random.choice(TOPICS)}. {random.choice(STY)} Only output the text itself.' for _ in range(bs)]
    kw=dict(enable_thinking=False) if 'Qwen3' in mid else {}
    txt=[tk.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True,**kw) for p in ps]
    e=tk(txt,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
    with torch.no_grad(): g=m.generate(**e,max_new_tokens=260,do_sample=True,temperature=0.9,top_p=0.95,pad_token_id=tk.pad_token_id)
    for p,s in zip(ps,tk.batch_decode(g[:,e['input_ids'].shape[1]:],skip_special_tokens=True)):
        s=re.sub(r'<think>.*?</think>','',s,flags=re.S).strip(); s=PRE.sub('',s).strip().strip('"“”‘’\'').strip()
        s=re.sub(r'^\*\*?[^\n]{0,60}\*\*?\n+','',s).strip()
        w=len(s.split())
        if 40<=w<=320 and '[' not in s[:400]: out.append(dict(t=s,y=1,m=tag,a='none',d='gencasual'))
    log(tag,len(out))
open(P+rf'\data\text_gen_{tag}.jsonl','w',encoding='utf8').write('\n'.join(json.dumps(o) for o in out[:n])); log('GENDONE',tag)