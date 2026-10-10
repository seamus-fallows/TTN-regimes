import sys, numpy as np, pandas as pd
sys.path.insert(0,'C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality')
from analysis import plateau_table, summarize, run_classes
SP='C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_locality_fix'
C='C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/configs/'
cfg=[C+'khigh_extension.yaml', C+'lambda2_K.yaml']
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
cls=run_classes([SP+'/out/spot'], config=cfg)
print(cls)
df=plateau_table([SP+'/out/spot'], config=cfg)
s=summarize(df)
cols=['tree','case','start','w','n_sigma','decades','seeds','fit_seeds','p_fit','p_se','p_median','p_local','p_local_se','logslope','logslope_se','rms_log_law','rms_power_law','hess_min_small','hess_exp']
print(s[cols].round(4).to_string())
ref=pd.read_csv('C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality/results/extensions_table.csv')
print(ref[cols].round(4).to_string())
# per-seed exponents
for (t,c,st,w),g in df[~df.rotation].groupby(['tree','case','start','w']):
    ps=[]
    for seed,gs in g.groupby('seed'):
        gs=gs[gs.gap>0].sort_values('sigma')
        if gs.sigma.nunique()>=3:
            x=np.log(1/gs.sigma.values)
            ps.append((seed, len(gs), round(np.polyfit(x,np.log(gs.gap.values),1)[0],3), round(np.polyfit(x,gs.gap.values,1)[0],4)))
    print(t,c,st,w,ps)
