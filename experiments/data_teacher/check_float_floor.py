"""Float64-floor check (lead's note; review of round 1: cover the runs behind the claims).

A dormant direction whose signal falls below roundoff is re-seeded by roundoff, not by the init.
- CP-teacher runs with overlaps (pilot npz and dense library Trajectories): minimum over t between the
  first escape (L < 0.9 L(0)) and component r's first learning event (a_r = 1/2) of the sliding-window
  envelope (+-5 output points) of |a_r| (a sign change passes through 0 without losing the signal);
  flagged below 1e-12. (Before the first escape everything is ~sigma^K, small but without
  cancellation.)
- TTN-teacher dense runs (no overlaps): minimum of max_e s_2^e between the first escape (L < 0.9 L(0))
  and the second-escape event (s_2^e >= 0.03 sigma_2^e(T)); flagged below 1e-14 (QR accuracy ~1e-15).
Saddle-start runs (r2_saddle) start from noise eps on O(1) cores: the route amplitudes are ~eps and the
route gradient ~ |lambda| eps (k = 2) or c eps^2 (k = 3), above roundoff for eps >= 1e-8 (k = 2) and
eps >= 1e-5 (k = 3); the check there is that t_esc keeps its law down to the smallest eps
(analyze_r2_saddle.py).
Fix round 2: also the star random-init runs (a_q while it is forgotten, r and r' before they are
learned) and the correlated-input runs (s_2 across 01|23 before its rise). The star saddle-start runs
(theta_q + eps, eps >= 1e-6) are covered by the saddle-start argument above: the strict rotation is
seeded at ~eps and the polynomial r route at ~eps^2 >= 1e-12; their overlaps a_r ~ eps^m are tiny by
construction, not by cancellation.
Writes results/float_floor_check.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_ROOT, RESULTS

from ttn.dynamics import Trajectory

HERE = Path(__file__).parent


def envelope_min(a, k):
    seg = np.abs(a[1:max(k, 2)])
    env = np.array([seg[max(0, i - 5):i + 6].max() for i in range(len(seg))])
    return float(seg.min()), float(env.min())


rows = []
pilot = ["e1_order", "e1_scaling_sigma", "e1_scaling_lambda", "e2_misaligned", "e3_correlated",
         "e4a_nonorth"]
dense = ["e4a_sigma_dense", "e2_sigma_dense", "e4b_ttn_dense", "e4c_cross_dense", "r2_random_dense",
         "r2_cross_dense", "r2_cross_b_dense"]
for name in pilot + dense:
    for f in sorted((OUT_ROOT / name).glob("*.npz")):
        if name in pilot:
            d = np.load(f, allow_pickle=True)
            ov, t, loss, sp, tsp = d["overlaps"], d["t"], d["loss"], None, None
        else:
            tr = Trajectory.load(f)
            ov = tr.data.get("overlaps", np.zeros(0))
            t, loss = tr.data["t"], tr.data["loss"]
            sp, tsp = np.nan_to_num(tr.data["spectra"]), tr.data["teacher_spectra"]
        if ov.size:
            # the danger is a dormant component whose signal falls near roundoff AFTER the first
            # escape (O(1) learned terms present); before it, W and its gradient are ~sigma^K and
            # sigma^(K-1), small but computed without cancellation (at sigma = 3e-4, K = 4 the initial
            # overlaps are ~1e-15 by construction)
            k0 = int(np.argmax(loss < 0.9 * loss[0]))
            for r in range(ov.shape[1]):
                a = ov[:, r]
                k = np.argmax(a >= 0.5) if (a >= 0.5).any() else len(a)
                if k <= k0 + 1:
                    continue  # learned in the first escape
                mn, env = envelope_min(a[k0:], k - k0)
                rows.append(dict(sweep=name, run=f.stem, signal=f"a_{r}", min_abs=mn, min_signal=env,
                                 flag=env < 1e-12))
        elif sp is not None:
            k0 = int(np.argmax(loss < 0.9 * loss[0]))
            edges = [j for j in range(sp.shape[1]) if tsp[j, 1] > 1e-3]
            hit = [np.argmax(sp[:, j, 1] >= 0.03 * tsp[j, 1]) for j in edges
                   if (sp[k0:, j, 1] >= 0.03 * tsp[j, 1]).any()]
            k2 = min([h for h in hit if h > k0], default=len(t))
            s2 = sp[k0:k2, :, 1].max(axis=1) if k2 > k0 else np.array([np.nan])
            mn = float(np.nanmin(s2))
            rows.append(dict(sweep=name, run=f.stem, signal="max_e s2", min_abs=mn, min_signal=mn,
                             flag=mn < 1e-14))
# fix round 2 (review of round 2, LOW 11): the star and correlated-input runs
# star random init: a_q after its first peak until it is relearned (a_q >= 1/2 again; the relearning is
# seeded by what is left of a_q), and r, r' before they are learned (after the first escape)
for f in sorted((OUT_ROOT / "r2_star_random").glob("*.npz")):
    d = np.load(f)
    ov, loss = d["overlaps"], d["loss"]
    k0 = int(np.argmax(loss < 0.9 * loss[0]))
    aq = ov[:, 0]
    kp = int(np.argmax(aq)) if not (aq > 0.5).any() else int(np.argmax(aq > 0.5))
    while kp + 1 < len(aq) and aq[kp + 1] >= aq[kp]:
        kp += 1
    later = np.nonzero((np.arange(len(aq)) > kp) & (aq < 0.5))[0]
    if len(later):
        relearn = np.nonzero((np.arange(len(aq)) > later[0]) & (aq >= 0.5))[0]
        k = relearn[0] if len(relearn) else len(aq)
        mn, env = envelope_min(aq[kp:], k - kp)
        rows.append(dict(sweep="r2_star_random", run=f.stem, signal="a_q (forgotten)", min_abs=mn,
                         min_signal=env, flag=env < 1e-12))
    for r in (1, 2):
        a = ov[:, r]
        k = np.argmax(a >= 0.5) if (a >= 0.5).any() else len(a)
        if k > k0 + 1:
            mn, env = envelope_min(a[k0:], k - k0)
            rows.append(dict(sweep="r2_star_random", run=f.stem, signal=f"a_{r}", min_abs=mn, min_signal=env,
                             flag=env < 1e-12))
# correlated inputs: s_2 across the middle cut between the first escape and its final rise
for f in sorted((OUT_ROOT / "r2_corr").glob("*.npz")):
    d = np.load(f)
    t, loss, sp = d["t"], d["loss"], d["spectra_mid"]
    k0 = int(np.argmax(loss < 0.9 * loss[0]))
    fin = sp[-1, 1]
    k2 = int(np.argmax(sp[:, 1] >= 0.03 * fin)) if (sp[k0:, 1] >= 0.03 * fin).any() else len(t)
    s2 = sp[k0:max(k2, k0 + 1), 1]
    mn = float(np.nanmin(s2))
    rows.append(dict(sweep="r2_corr", run=f.stem, signal="s2 (01|23)", min_abs=mn, min_signal=mn,
                     flag=mn < 1e-14))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "float_floor_check.csv", index=False)
print(df.groupby("sweep").agg(n=("run", "size"), n_flag=("flag", "sum"),
                              min_signal=("min_signal", "min")).to_string())
