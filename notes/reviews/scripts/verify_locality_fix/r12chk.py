import pandas as pd, numpy as np, yaml, ast, re
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
R='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/'
C='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/configs/'
r=pd.read_csv(R+'r12.csv', dtype={'S':str})
cfg=yaml.safe_load(open(C+'stall_diagnosis.yaml'))['cases']
flag=set()
print('cases',len(cfg))
for c in cfg:
    m=re.search(r'stalled seeds \[([^\]]*)\]', c.get('note',''))
    seeds=[int(x) for x in m.group(1).split(',')] if m and m.group(1).strip() else []
    S=''.join(map(str,c['S']))
    for s in c['sigmas']:
        for sd in seeds: flag.add((c['kind']+str(c['n']),S,c['start'],c.get('chi',2),float(s),sd))
    print(c['kind'],c['n'],S,c['start'],c.get('chi',2),c['sigmas'],c['seeds'],c.get('note'))
r['flagged']=[(t,S,st,ch,float(sg),sd) in flag for t,S,st,ch,sg,sd in zip(r.tree,r.S,r.start,r.chi,r.sigma,r.seed)]
print('flagged in r12:', r.flagged.sum(), ' total flagged from notes:', len(flag))
print(pd.crosstab([r.start, r.flagged], r.outcome))
print('dead but not flagged:'); print(r[(r.outcome=='dead')&~r.flagged][['tree','S','start','chi','sigma','seed','t_end','end_amp_final','n_unfav']])
print('flagged but escaped:'); print(r[(r.outcome=='escaped')&r.flagged][['tree','S','start','chi','sigma','seed','t_end','end_amp_final','n_unfav']])
print('dead t_end range', r[r.outcome=='dead'].t_end.min(), r[r.outcome=='dead'].t_end.max(), 'end amp range', r[r.outcome=='dead'].end_amp_final.min(), r[r.outcome=='dead'].end_amp_final.max())
print('pinned:'); print(r[r.outcome=='pinned'][['tree','S','start','chi','sigma','seed','t_end','end_amp_final','interior_amp_final']])
b=r[r.start=='small_balanced']
print('balanced flagged', b.flagged.sum()); print(b[b.flagged][['tree','S','sigma','seed','outcome','t_end','end_amp_final']])
f=r[r.start!='small_balanced']
f['pred_dead']=f.n_unfav==f.n_ends
print(pd.crosstab(f.pred_dead,f.outcome))
m4=f[(f.tree=='mps4')&(f.S=='03')&(f.chi==2)]
print('mps4 03 chi2:', len(m4), 'correct', ((m4.pred_dead)==(m4.outcome=='dead')).sum(), pd.crosstab(m4.pred_dead,m4.outcome))
