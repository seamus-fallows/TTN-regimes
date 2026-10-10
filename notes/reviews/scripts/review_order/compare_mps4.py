"""Compare: (1) reviewer's independent MPS4 runs (400 seeds) with the committed main_summary.csv and
sigma_table.csv; (2) seed by seed with the reviewer's rerun of the agent's sweep.py (data/review_order/raw)."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import norm
R = 'c:/Users/seamu/Iliad-ML/TTN_regimes/'
tree, d, chi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])     # e.g. mps4 2 4
main_f = R + f'data/review_order/indep/{tree}_d{d}_chi{chi}_main.jsonl'
sig_f = R + f'data/review_order/indep/{tree}_d{d}_chi{chi}_sigma.jsonl'
recs = [json.loads(l) for l in open(main_f)]
print(f'independent {tree} d={d} chi={chi}: {len(recs)} seeds')
allr = {r['seed']: r for r in recs}
srecs = []
tau = np.array([r['tau'] for r in recs])
lt = np.log(tau).ravel()
sd = lt.std(ddof=1)
pooled = np.log(tau.ravel())
diff = pooled[None, :] - pooled[:, None]
off = ~np.eye(len(pooled), dtype=bool)
summ = pd.read_csv(R + 'experiments/order/results/main_summary.csv')
bonds = None
rows = []
for rho in sorted({x['rho'] for r in recs for x in r['runs']}):
    win = np.array([[x['winner'] for x in r['runs'] if x['rho'] == rho and x['sigma'] == 0.01][0] for r in recs])
    pg = np.mean(win == 0)
    pred = (tau[:, 0] < tau[:, 1] / rho).astype(int)
    agree = np.mean(pred == (win == 0))
    curve = np.mean(diff[off] > np.log(rho))
    ln = norm.cdf(np.log(1 / rho) / (np.sqrt(2) * sd))
    c = summ[(summ.tree == tree) & (summ.d == d) & (summ.chi == chi) & (summ.rho == rho)]
    rows.append(dict(rho=rho, n=len(win), p_greedy=pg, committed=c.p_greedy.iloc[0] if len(c) else np.nan,
                     race_agree=agree, race_agree_committed=c.race_agree.iloc[0] if len(c) else np.nan,
                     race_curve=curve, race_curve_committed=c.race_curve.iloc[0] if len(c) else np.nan,
                     sd_log_tau=sd, sd_committed=c.sd_log_tau.iloc[0] if len(c) else np.nan,
                     lognormal=ln))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.5g}'))
try:
    srecs = [json.loads(l) for l in open(sig_f)]
    tab = pd.read_csv(R + 'experiments/order/results/sigma_table.csv')
    out = []
    for rho in sorted({x['rho'] for r in srecs for x in r['runs']}):
        per_seed = {}
        for sig in sorted({x['sigma'] for r in srecs for x in r['runs']}):
            wrong = {r['seed']: int([x['winner'] for x in r['runs'] if x['rho'] == rho and x['sigma'] == sig][0] != 0) for r in srecs}
            for s, w in wrong.items():
                per_seed.setdefault(s, []).append(w)
            c = tab[(tab.sweep == f'm4_sigma') & (tab.rho == rho) & (tab.sigma == sig)] if tree == 'mps4' else tab.iloc[0:0]
            out.append(dict(rho=rho, sigma=sig, n=len(wrong), wrong=sum(wrong.values()),
                            committed=f"{int(c['sum'].iloc[0])}/{int(c['count'].iloc[0])}" if len(c) else ''))
        # sigma 0.01 from the main file
        if rho in {x['rho'] for x in recs[0]['runs']}:
            for s in per_seed:
                if s in allr:
                    per_seed[s].append(int([x['winner'] for x in allr[s]['runs'] if x['rho'] == rho and x['sigma'] == 0.01][0] != 0))
            c = tab[(tab.sweep == 'm4_sigma') & (tab.rho == rho) & (tab.sigma == 0.01)] if tree == 'mps4' else tab.iloc[0:0]
            w01 = sum(int([x['winner'] for x in allr[s]['runs'] if x['rho'] == rho and x['sigma'] == 0.01][0] != 0) for s in per_seed if s in allr)
            out.append(dict(rho=rho, sigma=0.01, n=len(per_seed), wrong=w01,
                            committed=f"{int(c['sum'].iloc[0])}/{int(c['count'].iloc[0])}" if len(c) else ''))
        stable = np.mean([len(set(v)) == 1 for v in per_seed.values()])
        out.append(dict(rho=rho, sigma='all', n=len(per_seed), wrong=np.nan, committed=f'seed-stable fraction {stable:.4f}'))
    print(pd.DataFrame(out).to_string(index=False))
except FileNotFoundError:
    pass
# seed-by-seed vs the agent-code rerun
try:
    agent = {}
    for l in open(R + 'data/review_order/raw/m4_sigma/runs.jsonl'):
        r = json.loads(l)
        if (f"{r['kind']}{r['n']}", r['d'], r['chi']) == (tree, d, chi):
            agent[r['seed']] = r
    common = sorted(set(agent) & set(allr))
    dt, dtau, agree, n = [], [], 0, 0
    srec = {r['seed']: r for r in srecs}
    for s in common:
        a = agent[s]
        dtau.append(np.max(np.abs(np.array(a['tau']) / np.array(allr[s]['tau']) - 1)))
        for x in a['runs']:
            mine = [y for y in allr[s]['runs'] + srec.get(s, {'runs': []})['runs'] if y['rho'] == x['rho'] and y['sigma'] == x['sigma']]
            if mine:
                n += 1
                agree += int(mine[0]['winner'] == x['winner'])
                dt.append(abs(mine[0]['t1'] / x['t1'] - 1))
    print(f'seed-by-seed vs agent-code rerun: {len(common)} seeds, {n} GF runs, winner agreement {agree}/{n}, '
          f'max |t1 ratio - 1| = {max(dt):.2e}, max |tau ratio - 1| = {max(dtau):.2e}')
except FileNotFoundError:
    pass
