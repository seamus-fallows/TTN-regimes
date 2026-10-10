import pandas as pd, numpy as np
from scipy.optimize import minimize_scalar, brentq
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
c = pd.read_csv(R+"p9_conditions.csv")
p = pd.read_csv(R+"p9_sgd.csv")
esc = c[c.escaped==1].copy()
laws = {'P9': lambda x: np.where(x<1, -np.log1p(-np.minimum(x,1-1e-15))/np.maximum(x,1e-15), np.inf),
        'pole': lambda x: np.where(x<1, 1/(1-np.minimum(x,1-1e-15)), np.inf),
        'linear': lambda x: 1 + x/2}
for name, f in laws.items():
    chi2 = lambda cg: np.sum(((esc.ratio - f(cg*esc.z))/esc.ratio_se)**2)
    r = minimize_scalar(chi2, bounds=(1, 200), method='bounded')
    print(f"{name}: c={r.x:.2f} chi2={r.fun:.1f} dof={len(esc)-1}")
# per-condition inversion of P9
def inv(R_):
    return brentq(lambda x: -np.log1p(-x)/x - R_, 1e-9, 1-1e-12)
esc['x_inv'] = [inv(r) for r in esc.ratio]; esc['c_eff'] = esc.x_inv/esc.z
print(esc[['b0','B','eta_over_B','z','ratio','ratio_se','c_eff']].sort_values('z').round(4).to_string())
# B=16 vs 64
m = c[c.B==16].merge(c[c.B==64], on=['b0','eta_over_B'], suffixes=('_16','_64'))
m['diff_sig'] = (m.ratio_64-m.ratio_16)/np.sqrt(m.ratio_se_64**2+m.ratio_se_16**2)
print(m[['b0','eta_over_B','ratio_16','ratio_se_16','ratio_64','ratio_se_64','diff_sig','escaped_16','escaped_64']].round(4).to_string())
# z >= 0.025 trapped runs
print("z>=0.025 runs:", (p.z>=0.025).sum(), "escaped among them:", p[p.z>=0.025].escaped.sum(), "; z>=0.0249:", (p.z>=0.0249).sum(), p[p.z>=0.0249].escaped.sum())
# P10 counts
p['dip'] = p.min_a1 < 0.5
p['f'] = p.eta_over_eta_ms.round(4)
print(p.groupby('f').agg(n=('dip','size'), dips=('dip','sum'), tmax=('t_max','max')).to_string())
lo = p[p.f<=0.076]; mid = p[(p.f>0.1)&(p.f<0.2)]; hi = p[p.f>0.2]
print(f"<=0.076: {lo.dip.sum()}/{len(lo)}; 0.12-0.15: {mid.dip.sum()}/{len(mid)}; 0.25-0.30: {hi.dip.sum()}/{len(hi)}")
q = p[np.isclose(p.eta_over_B, 2.5e-4)]
print("eta/B=2.5e-4 dips by b0:", q.groupby('b0').dip.agg(['sum','size']).to_dict())
print("dip run at 0.12-0.15:", mid[mid.dip][['b0','B','eta','seed','t_max','min_a1','escaped']].to_string())
