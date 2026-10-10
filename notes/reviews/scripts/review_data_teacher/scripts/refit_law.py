"""Re-fit the plateau laws from the committed per-run CSVs: categories of detected plateau,
log vs power fits in the same (log) space, chi^2 with per-sigma SEM, leave-one-out slopes."""
import sys
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
R = "c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/results/"

def lsq(x, y, w=None):
    A = np.vstack([x, np.ones_like(x)]).T
    if w is None: w = np.ones_like(y)
    W = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    r = y - A @ c
    dof = max(len(x) - 2, 1)
    cov = (np.sum(w * r**2) / dof) * np.linalg.inv((A * w[:, None]).T @ A)
    return c, np.sqrt(np.diag(cov)), r

def fits(g, label):
    m = g.groupby('sigma')['dt2'].agg(['mean', 'sem', 'size', 'median'])
    x = -np.log(m.index.values)
    # log law on means (as in the note), unweighted
    (A, B), (Ase, _), r_lin = lsq(x, m['mean'].values)
    # weighted by 1/sem^2 (chi^2)
    (Aw, Bw), (Awse, _), r_w = lsq(x, m['mean'].values, 1 / m['sem'].values**2)
    chi2_log = np.sum((r_w / m['sem'].values) ** 2)
    # power law on means in log space; weights from relative sem
    ly = np.log(m['mean'].values); lse = m['sem'].values / m['mean'].values
    (p, lc), (pse, _), r_p = lsq(-x, ly, 1 / lse**2)
    chi2_pow = np.sum((r_p / lse) ** 2)
    # log-law residuals expressed in log space too (fair comparison)
    rel_log = np.log(m['mean'].values) - np.log(A * x + B)
    rel_pow_unw = lsq(-x, ly)[2]
    # median-based A
    (Am, _), (Amse, _), _ = lsq(x, m['median'].values)
    print(f"{label}: n_runs={len(g)} sigmas={len(m)} | A(mean)={A:.2f}+-{Ase:.2f}  A(w)={Aw:.2f}+-{Awse:.2f} chi2_log={chi2_log:.2f} (dof {len(m)-2}) | "
          f"p={-p:.3f}+-{pse:.3f} chi2_pow={chi2_pow:.2f} | rms(log-space): log-law {np.sqrt(np.mean(rel_log**2)):.4f} power {np.sqrt(np.mean(rel_pow_unw**2)):.4f} | A(median)={Am:.2f}")
    return m

if __name__ == '__main__':
    d = pd.read_csv(R + 'e4a_sigma_law.csv')
    d['cat'] = np.where(d.plateau_rel < 0.05, 'none', np.where(d.plateau_rel > 0.5, 'minor', 'major'))
    print(pd.crosstab([d.overlap, d.sigma], d.cat))
    for c, g in d.groupby('overlap'):
        g = g[g.dt2 > 0]
        if c == 0.45: g = g[g.sigma < 0.3]
        fits(g, f"c={c} all (as note)")
        fits(g[g.cat == 'major'], f"c={c} major only")
    e = pd.read_csv(R + 'e2_sigma_law.csv')
    for th, g in e.groupby('theta_deg'):
        fits(g, f"theta={th}")
