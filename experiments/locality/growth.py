"""Single-run estimate of the escape degree (theory_saddles.md R5).

On a plateau the overlap a of the next component grows like a k-core mode: eps' = lam eps^(k-1),
a ∝ eps^k, so da/dt ∝ a^(2 - 2/k). The local exponent g = d log(da/dt) / d log a is therefore 1 (k = 2,
exponential growth), 4/3, 3/2, 8/5 (k = 3, 4, 5), ... and k = 2 / (2 - g). We fit g by least squares on
the saved trajectory between the plateau midpoint and the point where a reaches `a_hi` (before
saturation), using only points where a and da/dt are positive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from bond_events import RAW, npz_path  # noqa: E402
from rules import get_tree  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402


def overlaps_trajectory(rec, raw_dir=RAW / "pilot3"):
    tree = get_tree(rec["kind"], rec.get("n"))
    spec = TTNSpec.uniform(tree, rec["d"], rec["chi"])
    teacher = pattern_teacher(rec["weights"], rec["patterns"], d=rec["d"],
                              seed=rec.get("teacher_seed", 1000))
    data = np.load(npz_path(rec, raw_dir))
    ys = torch.from_numpy(data["y"])
    a = np.array([teacher.component_overlaps(spec, spec.unflatten(ys[:, i])).numpy()
                  for i in range(ys.shape[1])])
    return data["t"], a


def growth_exponents(rec, raw_dir=RAW / "pilot3", a_hi=0.05, min_points=8):
    """Per plateau: (g, k_est = 2/(2-g), number of points used, a range)."""
    t, a = overlaps_trajectory(rec, raw_dir)
    out = []
    for p in rec.get("plateaus", []):
        if "t_mid" not in p:
            out.append(None)
            continue
        r = p["before"]
        sel = (t >= p["t_mid"]) & (t <= p["t_b"])
        tt, aa = t[sel], a[sel, r]
        # keep the growth phase below a_hi
        if len(aa) < min_points:
            out.append(None)
            continue
        stop = np.argmax(aa > a_hi) if np.any(aa > a_hi) else len(aa)
        tt, aa = tt[:stop], aa[:stop]
        if len(aa) < min_points:
            out.append(None)
            continue
        da = np.gradient(aa, tt)
        ok = (aa > 0) & (da > 0)
        if ok.sum() < min_points:
            out.append(None)
            continue
        x, y = np.log(aa[ok]), np.log(da[ok])
        g = np.polyfit(x, y, 1)[0]
        out.append(dict(g=g, k_est=2 / (2 - g) if g < 2 else np.inf, points=int(ok.sum()),
                        a_lo=float(aa[ok].min()), a_hi=float(aa[ok].max())))
    return out
