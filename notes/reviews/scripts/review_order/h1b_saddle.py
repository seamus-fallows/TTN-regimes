import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=pd.read_csv('experiments/order/results/h1b_saddle_b4.csv')
lp=lambda r:(1-r)*r**(1/3)
print(d.groupby(['rho','sigma']).agg(n=('seed','size'),t1=('t1','mean'),t1sd=('t1','std'),lam=('lam_min','mean'),lamsd=('lam_min','std'),a2min=('a2_min_after_t1','max'),loss_after=('loss_after_rotation','mean'),reup=('a2_reup','mean'),st=('status','min')).to_string())
# per-seed slopes for different sigma ranges
for smax in [1e-2,1e-3,1e-4]:
    rows=[]
    for (rho,seed),g in d.groupby(['rho','seed']):
        g=g[g.sigma<=smax]
        b,a=np.polyfit(np.log(1/g.sigma),g.t1,1)
        rows.append(dict(rho=rho,seed=seed,slope=b))
    r=pd.DataFrame(rows).groupby('rho').slope.agg(['mean','std','count'])
    r['se']=r['std']/np.sqrt(r['count']); r['pred']=[1/lp(x) for x in r.index]; r['ratio']=r['mean']/r['pred']; r['z']=(r['mean']-r['pred'])/r['se']
    print('sigma<=',smax); print(r.round(4))
# successive differences t1(sigma)-t1(sigma*10) per seed / ln10 vs pred
d=d.sort_values(['rho','seed','sigma'])
rows=[]
for (rho,seed),g in d.groupby(['rho','seed']):
    g=g.sort_values('sigma')
    s=g.sigma.values; t=g.t1.values
    loc=-(np.diff(t)/np.diff(np.log(s)))
    rows.append(dict(rho=rho,seed=seed,**{f'{s[i]:.0e}-{s[i+1]:.0e}':loc[i]*lp(rho) for i in range(len(loc))}))
print('local slope x |lam_pred| (1 = prediction):')
print(pd.DataFrame(rows).groupby('rho').mean().drop(columns='seed').round(4).to_string())
