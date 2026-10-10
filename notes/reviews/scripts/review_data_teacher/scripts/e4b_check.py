import numpy as np, pandas as pd
import importlib.util, sys
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/results/"
spec = importlib.util.spec_from_file_location("rf", "C:/Users/seamu/AppData/Local/Temp/claude/c--Users-seamu-Iliad-ML-TTN-regimes/ccfcd9a3-98b9-48c2-8218-49a3ab02b926/scratchpad/review_data_teacher/scripts/refit_law.py")
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
o = pd.read_csv(R + 'e4b_order.csv')
print(o.groupby(['teacher', 'sigma']).agg(n=('seed', 'size'), dt2_mean=('dt2', 'mean'), dt2_sem=('dt2', 'sem'), dt2_med=('dt2', 'median'),
      dt2_min=('dt2', 'min'), ok=('theory_ok', 'mean'), s2t1=('s2_at_t1', 'median'), plev=('plateau_level', 'median'), plev_min=('plateau_level', 'min')).to_string())
th = pd.read_csv(R + 'e4b_theory.csv')
for tag, g in o.groupby('teacher'):
    rf.fits(g, tag)
    # local slopes between consecutive sigmas
    m = g.groupby('sigma')['dt2'].mean()
    x = -np.log(m.index.values)
    print('   local slopes (large->small sigma):', np.round(np.diff(m.values[::-1]) / np.diff(x[::-1]), 2))
