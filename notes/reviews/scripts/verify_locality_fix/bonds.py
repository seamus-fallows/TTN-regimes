import pandas as pd, numpy as np, json, yaml, sys
R='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/'
C='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/configs/'
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
b=pd.read_csv(R+'bond_events2_twocomp.csv')
print('events',len(b))
# config filter
def keys(cfgs):
    ks=set()
    for p in cfgs:
        for c in yaml.safe_load(open(C+p))['cases']:
            for seed in c['seeds']:
                for s in c['sigmas']:
                    ks.add(json.dumps([c.get('name'),c.get('S'),c['kind'],c.get('n'),c.get('start','small'),c.get('d',2),c.get('chi',2),c.get('weights',[1.0,0.5]),c.get('teacher_seed',1000),seed,float(s)]))
    return ks
ks=keys(['twocomp.yaml'])
b['inconfig']=b.key.apply(lambda k: json.dumps(json.loads(k)) in ks)
print('in twocomp config:', b.inconfig.sum(), 'not:', (~b.inconfig).sum())
print((~b.inconfig).groupby([b.tree,b.case,b.start]).size())
res=b[b.rho>=3]
print('resolved (rho>=3):',len(res),' unresolved:',(b.rho<3).sum(), ' nan rho:', b.rho.isna().sum())
for f in ['0.03','0.05','0.1']:
    print(f, 'match resolved %.4f'%res['match_f'+f].mean(), ' unresolved %.4f'%b[~(b.rho>=3)]['match_f'+f].mean(), ' all %.4f'%b['match_f'+f].mean())
res_c=b[(b.rho>=3)&b.inconfig]
print('resolved in-config', len(res_c), 'match f0.05 %.4f'%res_c['match_f0.05'].mean())
# which events
print('mismatches resolved by k_star/start:'); print(res[~res['match_f0.05']].groupby(['k_star','start']).size())
print('unresolved by k_star/start:'); print(b[~(b.rho>=3)].groupby(['k_star','start']).size())
# spread
for kk in [2,3,4,5,6,7,8]:
    s=b[(b.k_star==kk)].spread_over_width.dropna()
    if len(s): print('k',kk,'n',len(s),'median spread %.2e'%s.median(),' 95%% %.3f'%s.quantile(.95), ' n_cross median', b[b.k_star==kk].n_cross.median())
s3=b[b.k_star>=3].spread_over_width.dropna()
print('k>=3 spread median %.2e 95%% %.3f n=%d'%(s3.median(), s3.quantile(.95), len(s3)))
s3r=b[(b.k_star>=3)&(b.rho>=3)].spread_over_width.dropna()
print('k>=3 resolved spread median %.2e 95%% %.3f n=%d'%(s3r.median(), s3r.quantile(.95), len(s3r)))
print('all spread median %.2e 95%% %.3f'%(b.spread_over_width.median(), b.spread_over_width.quantile(.95)))
# n_cross vs predicted edges count
b['npred']=b.predicted.apply(lambda s: len(json.loads(s)))
print('n_cross == npred frac', (b.n_cross==b.npred).mean())
print('k=2: npred', b[b.k_star==2].npred.unique())
# leak
print(b[['leak_window','leak_run']].describe())
g=b.groupby('sigma').leak_run.agg(['median','max','count'])
print(g)
x=np.log(b.sigma); y=np.log(b.leak_run)
m=np.isfinite(y)&(b.leak_run>0)
A=np.vstack([x[m],np.ones(m.sum())]).T
print('pooled fit leak_run ~ sigma^p, p=',np.linalg.lstsq(A,y[m],rcond=None)[0])
gm=b[b.leak_run>0].groupby('sigma').leak_run.median()
A=np.vstack([np.log(gm.index),np.ones(len(gm))]).T
print('median-by-sigma fit p=',np.linalg.lstsq(A,np.log(gm.values),rcond=None)[0])
print('max leak_run/sigma^2 =', (b.leak_run/b.sigma**2).max(), ' ; leak_run/(sigma^2*0.5) max', (b.leak_run/(0.5*b.sigma**2)).max())
print('leak_run max for sigma<=0.03:', b[b.sigma<=0.03].leak_run.max())
print('leak_window vs leak_run: frac window>=run*0.99', (b.leak_window>=0.99*b.leak_run).mean())
