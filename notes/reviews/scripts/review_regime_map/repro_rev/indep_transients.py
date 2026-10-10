"""How often do overlap transients exceed 1/2 (independent implementation), and do they move the
per-seed S2S boundary? Grid scan over sigma for chi in (9, 12, 16, 36, 64), seeds 0..11."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import indep_b4 as ib
out = sys.argv[1]
res = {}
sig = np.geomspace(0.004, 0.3, 16)
with open(os.path.join(out, "indep_transients.jsonl"), "w") as f:
    for chi in (9, 12, 16, 36, 64):
        for s in range(12):
            first, last, nup = [], [], []
            for x in sig:
                r = ib.run(chi, float(x), s)
                f.write(json.dumps(r) + "\n")
                first.append(r["min_rho"]); last.append(r["min_rho_last"]); nup.append(max(r["n_up"]))
            first, last = np.array(first), np.array(last)
            def bnd(v):
                a = np.where(np.isfinite(v), v, 0) > 3
                if not a[0]:
                    return np.nan
                i = np.argmax(~a) if (~a).any() else -1
                return float(np.sqrt(sig[i - 1] * sig[i])) if i > 0 else np.nan
            res[f"{chi}_{s}"] = dict(n_transient=int(sum(n > 1 for n in nup)), b_first=bnd(first), b_last=bnd(last))
            print(chi, s, res[f"{chi}_{s}"], flush=True)
json.dump(res, open(os.path.join(out, "indep_transients_summary.json"), "w"), indent=1)
