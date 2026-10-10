import sys, numpy as np, pandas as pd
sys.path.insert(0,'C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality')
from analysis import plateau_table, summarize, run_classes
SP='C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_locality_fix'
C='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/configs/'
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
print(run_classes([SP+'/out/k5'], config=[C+'k5_extension.yaml']))
df=plateau_table([SP+'/out/k5'], config=[C+'k5_extension.yaml'])
s=summarize(df)
print(s[['tree','case','start','n_sigma','seeds','fit_seeds','p_fit','p_se','p_median','hess_min_small','hess_exp']].round(4).to_string())
for (t,c,st),g in df[~df.rotation].groupby(['tree','case','start']):
    ps=[]
    for seed,gs in g.groupby('seed'):
        gs=gs[gs.gap>0].sort_values('sigma')
        ps.append((seed, gs.sigma.nunique(), round(np.polyfit(np.log(1/gs.sigma.values),np.log(gs.gap.values),1)[0],3) if gs.sigma.nunique()>=2 else None))
    print(t,c,st,ps)
