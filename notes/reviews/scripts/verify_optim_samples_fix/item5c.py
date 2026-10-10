import pandas as pd, numpy as np
from scipy.optimize import minimize_scalar
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results/"
c = pd.read_csv(R+"p9_conditions.csv")
esc = c[c.escaped==1]
zmax = esc.z.max()
laws = {'P9': lambda x: -np.log1p(-x)/x, 'pole': lambda x: 1/(1-x), 'linear': lambda x: 1 + x/2}
for name, f in laws.items():
    chi2 = lambda cg: np.sum(((esc.ratio - f(cg*esc.z))/esc.ratio_se)**2)
    ub = (1-1e-9)/zmax if name != 'linear' else 500
    r = minimize_scalar(chi2, bounds=(1, ub), method='bounded', options={'xatol':1e-6})
    print(f"{name}: c={r.x:.2f} chi2={r.fun:.1f} chi2/dof={r.fun/(len(esc)-1):.1f}")
    # also: chi2 with the 3 s.e.-floor? leave-out the two near-trap points
    sub = esc[esc.z < 0.02]
    chi2s = lambda cg: np.sum(((sub.ratio - f(cg*sub.z))/sub.ratio_se)**2)
    r2 = minimize_scalar(chi2s, bounds=(1, (1-1e-9)/sub.z.max() if name!='linear' else 500), method='bounded')
    print(f"    without the z=0.0208 points: c={r2.x:.2f} chi2={r2.fun:.1f} (dof {len(sub)-1})")
