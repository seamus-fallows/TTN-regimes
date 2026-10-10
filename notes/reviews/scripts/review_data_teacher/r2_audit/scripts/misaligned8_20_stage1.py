"""First-stage escape time (max_e s_2 reaching a fraction of the intermediate level 3.7e-4) at eps = 1e-2, 1e-3."""
import numpy as np
root = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/raw/audit_r2_saddle/"
res = {}
for eps in ("0.01", "0.001"):
    d = np.load(root + f"misaligned8_20_mps_s{eps}_seed0.npz")
    t, sp = d["t"], d["spectra"][:, :, 1]
    s2 = sp.max(axis=1)
    out = {}
    for lev in (5e-5, 1e-4, 1.5e-4, 2e-4):
        k = int(np.argmax(s2 >= lev))
        # log-interpolate the crossing
        tk = t[k - 1] + (np.log(lev) - np.log(s2[k - 1])) / (np.log(s2[k]) - np.log(s2[k - 1])) * (t[k] - t[k - 1])
        out[lev] = tk
    res[eps] = out
    print(eps, {k: round(v, 1) for k, v in out.items()})
for lev in res["0.01"]:
    dt = res["0.001"][lev] - res["0.01"][lev]
    print(f"level {lev:.0e}: dt per decade of eps = {dt:.1f}; predicted ln(10)/|lambda_min| = {np.log(10)/5.063764e-04:.1f}; ratio {dt/(np.log(10)/5.063764e-04):.3f}")
