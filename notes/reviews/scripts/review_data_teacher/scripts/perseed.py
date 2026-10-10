"""Per-seed slopes (same init direction at every sigma): A_seed from Dt_2 vs log(1/sigma) per seed,
then mean +- SEM over seeds; compare with the residual-based SE of the note's fits."""
import numpy as np, pandas as pd
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/results/"

def per_seed(df, key, val, sig_max=None, cats=None):
    g = df[df[key] == val]
    if sig_max: g = g[g.sigma <= sig_max]
    if cats is not None: g = g[g.cat.isin(cats)]
    out = []
    for s, gs in g.groupby('seed'):
        if gs.sigma.nunique() < 4: continue
        x = -np.log(gs.sigma.values); y = gs.dt2.values
        out.append(np.polyfit(x, y, 1)[0])
    out = np.array(out)
    return len(out), out.mean(), out.std(ddof=1) / np.sqrt(len(out)), np.round(np.sort(out), 1)

a = pd.read_csv(R + 'e4a_sigma_law.csv')
a['cat'] = np.where(a.plateau_rel < 0.05, 'none', np.where(a.plateau_rel > 0.5, 'minor', 'major'))
for c, smax in [(0.2, None), (0.2, 0.12), (0.45, 0.19)]:
    n, m, se, arr = per_seed(a, 'overlap', c, smax)
    print(f'e4a c={c} sigma<={smax}: per-seed A = {m:.2f} +- {se:.2f} (n={n}) {arr}')
    n, m, se, arr = per_seed(a, 'overlap', c, smax, ['major'])
    print(f'   major-only seeds (>=4 sigmas): A = {m:.2f} +- {se:.2f} (n={n}) {arr}')
b = pd.read_csv(R + 'e2_sigma_law.csv')
n, m, se, arr = per_seed(b, 'theta_deg', 20)
print(f'e2 theta=20: per-seed A = {m:.2f} +- {se:.2f} (n={n}) {arr}')
o = pd.read_csv(R + 'e4b_order.csv')
for t in sorted(o.teacher.unique()):
    n, m, se, arr = per_seed(o, 'teacher', t)
    print(f'e4b {t}: per-seed A = {m:.2f} +- {se:.2f} (n={n}) {arr}')
