import json, glob, numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 300)
rows = []
for f in sorted(glob.glob('data/s2s_core/first/*.json')):
    s = json.load(open(f))
    c = s.get('corrflow', {})
    rows.append(dict(tree=f"{s['tree']}{s['N']}", chi=s['chi'], teacher=s['teacher'], init=s['init'], seed=s['seed'],
                     T=s['T'], T_v1=s.get('T_v1', np.nan), r1=s['r1_pred'], r1_v1=s.get('r1_pred_v1', -1),
                     amp_stop=c.get('amp_stop', np.nan), ratio_stop=c.get('ratio_stop', np.nan), status=c.get('status'),
                     nruns=len(s['runs']), n_t1=sum('t1' in r for r in s['runs'])))
d = pd.DataFrame(rows)
d['dT_rel'] = (d['T'] - d['T_v1']) / d['T_v1']
d['capped'] = d.amp_stop > 1e29
d['conv'] = d.ratio_stop <= 1e-6
print(len(d), 'jobs; has T_v1:', d.T_v1.notna().sum())
g = d.groupby(['tree', 'chi', 'teacher', 'init'])
print(g.agg(n=('seed', 'size'), capped=('capped', 'sum'), conv=('conv', 'sum'), ratio_max=('ratio_stop', 'max'),
            dT_absmax=('dT_rel', lambda x: np.nanmax(np.abs(x))), r1_changed=('r1', 'size')).to_string())
d['r1chg'] = d.r1 != d.r1_v1
print('r1 changed:', d.r1chg.sum())
print(d[d.r1chg][['tree', 'chi', 'teacher', 'init', 'seed', 'r1_v1', 'r1', 'T_v1', 'T', 'dT_rel', 'ratio_stop']].to_string())
print('status values', d.status.value_counts().to_dict())
print('abs dT max overall', np.nanmax(np.abs(d.T - d.T_v1)))
x = d[(d.tree == 'binary4')]
print(x.groupby('teacher').apply(lambda y: pd.Series({'max_abs_dT': np.nanmax(np.abs(y.T - y.T_v1)), 'max_rel': np.nanmax(np.abs(y.dT_rel))})))
print('jobs with unresolved runs', (d.n_t1 < d.nruns).sum())
