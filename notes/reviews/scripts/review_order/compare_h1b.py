"""Row-by-row comparison of the reviewer's h1b reruns (data/review_order/res/h1b_*.csv) with the
committed experiments/order/results/h1b_*.csv (same rho, sigma, seed), plus per-seed slopes of
t_corr vs ln(1/sigma) from the rerun."""
import numpy as np, pandas as pd
R = 'c:/Users/seamu/Iliad-ML/TTN_regimes/'
cols = ['t1', 't2', 'lam_min', 'lam_min_raw', 'lam_pred', 'a2_min_after_t1', 'a2_reup', 'loss_after_rotation', 'final_loss', 'stalled']
for name in ['h1b_saddle_b4', 'h1b_saddle_mps4', 'h1b_natural_b4']:
    try:
        mine = pd.read_csv(R + f'data/review_order/res/{name}.csv')
    except FileNotFoundError:
        print(name, 'not rerun yet'); continue
    ref = pd.read_csv(R + f'experiments/order/results/{name}.csv')
    m = mine.merge(ref, on=['rho', 'sigma', 'seed'], suffixes=('', '_ref'))
    print(f'== {name}: {len(mine)} rerun rows, {len(m)} matched to committed rows')
    for c in cols:
        if c in m and c + '_ref' in m:
            a, b = m[c].astype(float), m[c + '_ref'].astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)
            print(f'  {c:22s} finite pairs {ok.sum():3d}, both-nonfinite {int((~np.isfinite(a) & ~np.isfinite(b)).sum())}, '
                  f'max rel diff {rel.max() if len(rel) else float("nan"):.2e}')
    show = [c for c in ['rho', 'sigma', 'seed', 't1', 't1_ref', 't2', 't2_ref', 'lam_min', 'lam_min_ref', 'a2_min_after_t1', 'a2_reup', 'a2_reup_ref', 'stalled'] if c in m]
    print(m[show].sort_values(['rho', 'sigma', 'seed']).to_string(index=False, float_format=lambda v: f'{v:.6g}'))
    if name == 'h1b_saddle_b4':
        for rho, g in mine.groupby('rho'):
            lam = (1 - rho) * rho ** (1 / 3)
            sl_all, sl_small = [], []
            for seed, gs in g.groupby('seed'):
                x, y = np.log(1 / gs.sigma.values), gs.t1.values
                sl_all.append(np.polyfit(x, y, 1)[0])
                k = gs.sigma.values <= 1e-4
                if k.sum() >= 2:
                    sl_small.append(np.polyfit(x[k], y[k], 1)[0])
            print(f'  rho={rho}: per-seed slope dt_corr/dln(1/sigma) all sigma {np.mean(sl_all):.4f} +- {np.std(sl_all, ddof=1)/np.sqrt(len(sl_all)):.4f}; '
                  f'sigma<=1e-4 {np.mean(sl_small):.4f} +- {np.std(sl_small, ddof=1)/np.sqrt(len(sl_small)):.4f}; 1/|lambda| = {1/lam:.4f}')
    if name == 'h1b_natural_b4':
        w = mine[mine.wrong]
        rot = w[w.a2_min_after_t1 < 0.1]
        sc = (rot.t1 - rot.t2) * (1 - rot.rho) * rot.rho ** (1 / 3)
        print('  natural: rotated', len(rot), 'of', len(w), '; |lam|(t1-t2) by sigma:', sc.groupby(rot.sigma).agg(['count', 'mean', 'std']).to_string())
    if name == 'h1b_saddle_mps4':
        for seed, gs in mine.groupby('seed'):
            x = np.log(gs.sigma.values)
            print(f'  seed {seed}: lam exponent {np.polyfit(x, np.log(np.abs(gs.lam_min.values)), 1)[0]:.3f}, t1 exponent {np.polyfit(x, np.log(gs.t1.values), 1)[0]:.3f}, '
                  f'min a2 after t1 {gs.a2_min_after_t1.min():.4f}')
