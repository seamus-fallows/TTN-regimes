"""Paired (per-seed) log-vs-power discrimination: for each seed fit Dt_2 = A x + B (x = log 1/sigma)
and Dt_2 = C sigma^-p, both by least squares in log space, and compare residual rms per seed."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/results/"

def shape_test(g):
    wins, rl, rp = 0, [], []
    for s, gs in g.groupby('seed'):
        gs = gs.sort_values('sigma')
        if len(gs) < 4 or (gs.dt2 <= 0).any(): continue
        x = -np.log(gs.sigma.values); ly = np.log(gs.dt2.values)
        # power law: linear in log space
        cp = np.polyfit(x, ly, 1); r_p = ly - np.polyval(cp, x)
        # log law fitted in log space
        A0, B0 = np.polyfit(x, gs.dt2.values, 1)
        f = lambda q: np.log(np.maximum(q[0] * x + q[1], 1e-9)) - ly
        q = least_squares(f, [A0, B0]).x
        r_l = f(q)
        rl.append(np.sqrt(np.mean(r_l**2))); rp.append(np.sqrt(np.mean(r_p**2)))
        wins += np.sqrt(np.mean(r_l**2)) < np.sqrt(np.mean(r_p**2))
    return wins, len(rl), np.median(rl), np.median(rp)

a = pd.read_csv(R + 'e4a_sigma_law.csv')
a['cat'] = np.where(a.plateau_rel < 0.05, 'none', np.where(a.plateau_rel > 0.5, 'minor', 'major'))
for c in (0.2, 0.45):
    g = a[(a.overlap == c) & (a.sigma < (0.3 if c == 0.45 else 1))]
    print(f'e4a c={c} all: log better in %d/%d seeds; median rms log %.3f power %.3f' % shape_test(g))
    print(f'e4a c={c} major: log better in %d/%d seeds; median rms log %.3f power %.3f' % shape_test(g[g.cat == 'major']))
b = pd.read_csv(R + 'e2_sigma_law.csv')
print('e2 theta=20: log better in %d/%d seeds; median rms log %.3f power %.3f' % shape_test(b[b.theta_deg == 20]))
o = pd.read_csv(R + 'e4b_order.csv')
for t, g in o.groupby('teacher'):
    print(f'e4b {t}: log better in %d/%d seeds; median rms log %.3f power %.3f' % shape_test(g))
