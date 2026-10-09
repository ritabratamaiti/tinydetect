# Choose the WiSE alpha from the tuning validation split only, broken down by source group (casual AI vs formal human etc).
import os, json
P=r'C:\Projects\TinyDetect'
src=open(P+r'\scripts\tune_fakespot.py',encoding='utf8').read()
exec(src[:src.index("dev='cuda'")])
import numpy as np
W=json.load(open(P+r'\models\fakespot_tuned\wise.json')); y=np.array(W['val_y'])
assert len(va)==len(y) and all(int(r['y'])==int(t) for r,t in zip(va,y)), 'validation split mismatch'
def grp(r):
    d=r['d']
    if r['y']==1 and (d.startswith('gencasual') or d.startswith('casual_') or d in ('mgt_reddit','mgt_eli5','mgt_cmv','mgt_yelp','mgt_tldr','mgt_reddit_eli5') or d in ('hape_blog','hape_spok','hape_tvm')): return 'AI casual'
    if r['y']==1: return 'AI formal'
    if d.startswith(('gutenberg','wiki','hape_acad','hape_news','hape_fic','human_extra','mgt_wikipedia','mgt_arxiv','mgt_peerread','mgt_sci')): return 'human formal'
    return 'human casual'
G=np.array([grp(r) for r in va])
print('group sizes',{g:int((G==g).sum()) for g in sorted(set(G))})
print('%-6s'%'alpha'+''.join('%14s'%g for g in ['AI casual','AI formal','human formal','human casual'])+'   (AI: caught >=0.6 | human: flagged >=0.6)')
for a,r in W['res'].items():
    p=np.array(r['val_scores']); row='%-6s'%a
    for g in ['AI casual','AI formal','human formal','human casual']:
        m=G==g; row+='%14s'%(f'{(p[m]>=0.6).mean()*100:.1f}%')
    print(row)
print('PICKDONE')