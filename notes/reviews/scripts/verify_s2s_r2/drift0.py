"""Conservation drift from t = 0: max_e |Q_e(theta_end) - Q_e(theta_0)|_max, relative to max_e |Q_e(0)|_max
and to the library normalisation (|Q(0)| + max |G|^2 at the end)."""
import glob, json, sys
import numpy as np, torch, pandas as pd
torch.set_num_threads(1)
sys.path.insert(0, 'experiments/s2s_core'); sys.path.insert(0, '.')
import s2s
from ttn.network import conserved_quantities
rows = []
specs = {}
for sweep in ['main', 'large', 'k78', 'balanced']:
    for f in sorted(glob.glob(f'data/s2s_core/{sweep}/*.json')):
        s = json.load(open(f))
        d = np.load(f[:-5] + '.npz')
        if 'theta_end' not in d.files or 'theta0' not in d.files:
            rows.append(dict(sweep=sweep, name=s['name'], has=False)); continue
        key = (s['tree'], s['N'], s['chi'], tuple(s['weights']))
        if key not in specs:
            specs[key] = s2s.setup(s['tree'], s['N'], s['d'], s['chi'], s['weights'], s['teacher_seed'])[0]
        spec = specs[key]
        c0 = spec.unflatten(torch.as_tensor(d['theta0'])); c1 = spec.unflatten(torch.as_tensor(d['theta_end']))
        q0 = conserved_quantities(spec, c0); q1 = conserved_quantities(spec, c1)
        dq = max((q1[v] - q0[v]).abs().max().item() for v in q0)
        q0m = max(x.abs().max().item() for x in q0.values())
        g1 = max(c.pow(2).sum().item() for c in c1)
        rows.append(dict(sweep=sweep, name=s['name'], tree=f"{s['tree']}{s['N']}", sigma=s['sigma'], has=True, dq=dq, q0=q0m,
                         rel_q0=dq / q0m, rel_lib=dq / (q0m + g1), drift_chunk=s['max_conservation_drift']))
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], index=False)
pd.set_option('display.width', 250)
print('has theta0/theta_end:', df.has.sum(), '/', len(df))
h = df[df.has]
print(h.groupby(['sweep', 'tree']).agg(n=('dq', 'size'), rel_q0_max=('rel_q0', 'max'), rel_q0_med=('rel_q0', 'median'),
      rel_lib_max=('rel_lib', 'max'), chunk_max=('drift_chunk', 'max'), n_lib_gt1e6=('rel_lib', lambda x: int((x > 1e-6).sum())),
      n_q0_gt1e2=('rel_q0', lambda x: int((x > 1e-2).sum()))).to_string())
print(h.sort_values('rel_q0', ascending=False).head(12)[['sweep', 'name', 'rel_q0', 'rel_lib', 'drift_chunk']].to_string())
