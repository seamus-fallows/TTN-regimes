import pandas as pd, numpy as np
R='experiments/s2s_core/results/'
L=pd.read_csv(R+'lo_chi.csv')
L=L[L.weights.astype(str)=='1']
rng=np.random.default_rng(0)
def slope(df,chis,col='T'):
    x=np.log(chis); y=np.log([df[df.chi==c][col].median() for c in chis]); return np.polyfit(x,y,1)[0]
for tree,chis in []:
    g=L[L.tree==tree]
    s=slope(g,chis); sk=slope(g,chis,'t_KH')
    bs=[]
    for b in range(500):
        gb=pd.concat([g[g.chi==c].sample(frac=1,replace=True,random_state=int(rng.integers(1e9))) for c in chis])
        bs.append(slope(gb,chis))
    print(tree,chis,'slope T %.3f +- %.3f'%(s,np.std(bs)),'slope tKH %.3f'%sk)
# mps8 chi 2->8 ratio
g=L[L.tree=='mps8']; print('mps8 chi2/chi8 median T ratio',g[g.chi==2]['T'].median()/g[g.chi==8]['T'].median(), 'tKH',g[g.chi==2].t_KH.median()/g[g.chi==8].t_KH.median())
d=pd.read_csv(R+'lo_d.csv')
for (tree,dd),g in d.groupby(['tree','d']):
    print(tree,dd,'chi',g.chi.iloc[0],'median T %.4f'%g['T'].median(),'sd lnT %.3f'%np.log(g['T']).std(),'tKH med %.4f'%g.t_KH.median(),'frac_dev_max %.4g'%g.frac_dev_max.max(),'n',len(g))
