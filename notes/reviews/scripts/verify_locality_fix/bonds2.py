import pandas as pd, numpy as np, json
R='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/'
b=pd.read_csv(R+'bond_events2_twocomp.csv')
for f in ['0.03','0.05','0.1']:
    extra=0; missing=0
    for o,p in zip(b['observed_f'+f], b.predicted):
        o=set(json.loads(o)); p=set(json.loads(p))
        extra+= len(o-p)>0; missing+= len(p-o)>0
    print('f',f,'events with extra (non-Steiner) rising edges:',extra,' events with missing Steiner edges:',missing)
# spread by k and sigma range
b['sbin']=pd.cut(b.sigma,[0,0.01,0.03,0.1,0.3,1])
print(b[b.k_star>=3].groupby(['k_star']).spread_over_width.describe(percentiles=[.5,.9,.95]).round(4))
print(b[b.k_star>=3].groupby(['sbin']).spread_over_width.describe(percentiles=[.5,.9,.95]).round(4))
print('frac spread exactly 0 for k=3:', (b[b.k_star==3].spread_over_width==0).mean())
# leak: fit per case median like agent?
m=b[b.leak_run>0]
g=m.groupby(['tree','case','start'])
ps=[]
for k,gg in g:
    gm=gg.groupby('sigma').leak_run.median()
    if len(gm)>=3:
        A=np.vstack([np.log(gm.index),np.ones(len(gm))]).T
        ps.append(np.linalg.lstsq(A,np.log(gm.values),rcond=None)[0][0])
print('per-case slope of median leak_run vs sigma: median %.3f mean %.3f n %d'%(np.median(ps),np.mean(ps),len(ps)))
print('zero leak_run events:', (b.leak_run==0).sum())
# multicomp per event
e=pd.read_csv(R+'bond_events2_multicomp_events.csv')
print('multicomp events', len(e), 'match f0.05', e['match_f0.05'].mean())
print(e.groupby(['tree','case','start'])['match_f0.05'].agg(['mean','size']).round(3).to_string())
