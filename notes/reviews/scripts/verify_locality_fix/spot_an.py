import sys, numpy as np, pandas as pd, glob, json
sys.path.insert(0,'C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/locality')
from analysis import plateau_table, summarize, load_records, classify
SP='C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/verify_locality_fix'
recs=load_records([SP+'/out/spot'])
from collections import Counter
print('records',len(recs), Counter((r.get('kind'),r.get('n'),tuple(r.get('S') or []),classify(r)) for r in recs))
df=plateau_table([SP+'/out/spot'])
pd.set_option('display.width',250)
for (tree,case,start,w),g in df.groupby(['tree','case','start','w']):
    g=g[~g.rotation]
    print('==',tree,case,start,w)
    for seed,gs in g.groupby('seed'):
        gs=gs[gs.gap>0].sort_values('sigma')
        x=np.log(1/gs.sigma.values); y=np.log(gs.gap.values)
        p=np.polyfit(x,y,1)[0] if len(gs)>=3 else np.nan
        pl=np.polyfit(x[:3],y[:3],1)[0] if len(gs)>=3 else np.nan
        lg=np.polyfit(x,gs.gap.values,1)[0]
        print('  seed',seed,'n_sigma',len(gs),'p=%.3f p_local3=%.3f logslope=%.4f'%(p,pl,lg), ' hess_min@smallest=%.5g'%gs.hess_min.values[0] if 'hess_min' in gs else '')
s=summarize(df)
print(s[['tree','case','start','w','fit_seeds','p_fit','p_se','p_local','logslope','logslope_se','rms_log_law','rms_power_law','hess_min_small','hess_exp']].round(4).to_string())
