import pandas as pd, numpy as np
pd.set_option('display.width',300); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',500)
R='experiments/s2s_core/results/'
d=pd.read_csv(R+'main_runs.csv')
def fits(df, only_s2s=False, smax=None):
    rows=[]
    c=df[df['class']=='complete']
    if only_s2s: c=c[c.s2s==True]
    if smax: c=c[c.sigma<=smax]
    for key,g in c.groupby(['sweep','tree','chi','teacher','init','seed']):
        modal=g.order.mode().iloc[0]; g=g[g.order==modal]
        if len(g)<4: continue
        x=np.log(g.sigma.values)
        r=dict(zip(['sweep','tree','chi','teacher','init','seed'],key)); r['order']=modal; r['n']=len(g)
        for nm,y in (('gap1',g.t2-g.t1),('gap2',g.t3-g.t2)):
            r[nm]=-np.polyfit(x,np.log(y.values),1)[0]
        rows.append(r)
    f=pd.DataFrame(rows)
    se=lambda v: v.std(ddof=1)/np.sqrt(len(v))
    return f.groupby(['sweep','tree','chi','teacher','init']).agg(n=('seed','size'),gap1=('gap1','mean'),gap1_se=('gap1',se),gap2=('gap2','mean'),gap2_se=('gap2',se))
b=d[(d.tree.isin(['binary4','mps4']))&(d.sweep=='main')]
print('all complete (as agent):'); print(fits(b).round(3))
print('S2S runs only:'); print(fits(b,only_s2s=True).round(3))
print('sigma<=0.02:'); print(fits(b,smax=0.02).round(3))
print('greedy-order seeds only (binary4 close):')
bc=b[(b.tree=='binary4')&(b.teacher=='close')&(b.order=='1-2-3')]
print(fits(bc).round(3))
