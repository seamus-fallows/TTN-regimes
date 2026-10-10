"""binary4 (m = 2): reviewer's independent runs (all 1000 seeds) vs committed sigma_table.csv counts;
own binomial-ML power-law fit P(wrong) = C sigma^gamma over sigma <= 0.01 with a seed bootstrap;
seed-by-seed comparison with the reviewer's rerun of the agent's sweep (raw/b4_sigma)."""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
R = 'c:/Users/seamu/Iliad-ML/TTN_regimes/'
recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
tab = pd.read_csv(R + 'experiments/order/results/sigma_table.csv')
rows = {}
for r in recs:
    for x in r['runs']:
        rows.setdefault((r['chi'], x['rho'], x['sigma']), {})[r['seed']] = int(x['winner'] != 0)
out = []
for (chi, rho, sig), d in sorted(rows.items()):
    c = tab[(tab.sweep == 'b4_sigma') & (tab.chi == chi) & (tab.rho == rho) & (tab.sigma == sig)]
    out.append(dict(chi=chi, rho=rho, sigma=sig, n=len(d), wrong=sum(d.values()),
                    committed=f"{int(c['sum'].iloc[0])}/{int(c['count'].iloc[0])}" if len(c) else ''))
print(pd.DataFrame(out).to_string(index=False))

def fit(sig, k, n):
    ls = np.log(sig)
    def nll(p):
        q = np.clip(np.exp(p[0] + p[1] * ls), 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(q) + (n - k) * np.log(1 - q))
    best = min((minimize(nll, [lc, g], method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-12))
                for lc in (-1.0, 0.0) for g in (0.1, 0.3, 0.6)), key=lambda r: r.fun)
    return best.x[1]
for chi, rho in sorted({(k[0], k[1]) for k in rows}):
    sigs = sorted(s for (c, r, s) in rows if c == chi and r == rho and s <= 0.01)
    if len(sigs) < 2:
        continue
    seeds = sorted(set.intersection(*[set(rows[(chi, rho, s)]) for s in sigs]))
    M = np.array([[rows[(chi, rho, s)][sd] for s in sigs] for sd in seeds])
    g = fit(np.array(sigs), M.sum(0), np.full(len(sigs), len(seeds)))
    rng = np.random.default_rng(1)
    boot = [fit(np.array(sigs), M[rng.integers(0, len(seeds), len(seeds))].sum(0), np.full(len(sigs), len(seeds))) for _ in range(200)]
    print(f'chi={chi} rho={rho}: sigma {sigs}, {len(seeds)} seeds: gamma = {g:.4f} +- {np.std(boot, ddof=1):.4f} '
          f'(pred (1-rho)/rho = {(1-rho)/rho:.4f}); subset property: wrong at smallest sigma but not at largest = '
          f'{int(np.sum((M[:, 0] == 1) & (M[:, -1] == 0)))}, wrong at largest = {int(M[:, -1].sum())}')
# seed-by-seed vs agent-code rerun
try:
    agent = {}
    for l in open(R + 'data/review_order/raw/b4_sigma/runs.jsonl'):
        r = json.loads(l)
        for x in r['runs']:
            agent[(r['chi'], x['rho'], x['sigma'], r['seed'])] = x
    mine = {(r['chi'], x['rho'], x['sigma'], r['seed']): x for r in recs for x in r['runs']}
    common = sorted(set(agent) & set(mine))
    agree = sum(agent[k]['winner'] == mine[k]['winner'] for k in common)
    dt = max(abs(agent[k]['t1'] / mine[k]['t1'] - 1) for k in common)
    print(f'seed-by-seed vs agent-code rerun: {len(common)} runs, winner agreement {agree}/{len(common)}, max |t1 ratio - 1| = {dt:.2e}')
except FileNotFoundError:
    print('no agent-code b4_sigma rerun yet')
