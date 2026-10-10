"""Audit analysis of the locality / MPS runs (summaries from the repo's summarize.py + route_amplitudes.py).

Usage: uv run python audit_analyze.py <summary.json> <routes.json or -> <runs_dir>
Prints per run: absolute escape time t_a, tau to reach loss levels, L(1e4), test acc, first rising
edges, k<=5 share and best k=3 amplitude (binary), and the growth exponent gamma = dlog(da/dt)/dlog(a)
of the first route's s_2/s_1 (theory_saddles P4: 1 for k = 2, 4/3 for k = 3).
"""

import json
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE))
from analysis import load  # noqa: E402


def gamma_fit(t, a, lo=3e-4, hi=3e-2):
    """Slope of log(da/dt) vs log(a) on the rising part with lo <= a <= hi (one-sided differences
    on the merged rows; only the monotone rise before a first exceeds hi)."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    ok = np.isfinite(a) & (a > 0)
    t, a = t[ok], a[ok]
    if not (a >= hi).any():
        return np.nan, 0
    end = int(np.flatnonzero(a >= hi)[0])
    # start: last time a was below lo before `end`
    below = np.flatnonzero(a[:end] < lo)
    start = int(below[-1]) if below.size else 0
    tt, aa = t[start : end + 1], a[start : end + 1]
    keep = np.concatenate([[True], np.diff(tt) > 0])
    tt, aa = tt[keep], aa[keep]
    rate = np.diff(aa) / np.diff(tt)
    amid = np.sqrt(aa[1:] * aa[:-1])
    m = (rate > 0) & (amid >= lo) & (amid <= hi)
    if m.sum() < 4:
        return np.nan, int(m.sum())
    p = np.polyfit(np.log(amid[m]), np.log(rate[m]), 1)[0]
    return float(p), int(m.sum())


def main(summary_json, routes_json, runs_dir):
    rows = [r for r in json.load(open(summary_json)) if "error" not in r]
    routes = {}
    if routes_json != "-":
        for r in json.load(open(routes_json)):
            if "pairs" in r:
                lam = np.array([p["lam"] for p in r["pairs"]])
                k = np.array([p["k"] for p in r["pairs"]])
                tot = (lam**2).sum()
                routes[r["name"]] = {
                    "share_k3": float((lam[k == 3] ** 2).sum() / tot),
                    "share_k5": float((lam[k == 5] ** 2).sum() / tot),
                    "best3": float(lam[k == 3].max()),
                    "max_all": float(lam.max()),
                    "rank1_err": r["rank1_rel_err"],
                    "plateau_L": r["train_loss"],
                    "best3_pair": [(p["u"], p["w"]) for p in r["pairs"] if p["k"] == 3][
                        int(np.argmax(lam[k == 3]))
                    ],
                }
    out = []
    for r in sorted(rows, key=lambda r: (r["tree"], r["ordering"])):
        d = {
            "name": r["name"],
            "tree": r["tree"],
            "ordering": r["ordering"],
            "t_a": r.get("t_a"),
            "t1": r.get("t1"),
            "L_end": r.get("loss_end"),
            "test_acc": r.get("test_acc_end"),
            "test_L": r.get("test_loss_end"),
            "n_drops": r.get("n_drops"),
            "first_rank2_edges": r.get("first_rank2_edges"),
            "gap1": (r["gaps"][0] if r.get("gaps") else None),
            "t_first_rank2_after_drop1": (r["t_first_rank2"] - r["drops"][0]["t"])
            if r.get("drops")
            else None,
        }
        for lvl in (0.3, 0.25, 0.2, 0.15, 0.12, 0.1):
            d[f"tau_L{lvl}"] = r.get(f"t_loss_{lvl}")
        d.update(routes.get(r["name"], {}))
        # growth exponent of the first rising edge(s)
        run = load(Path(runs_dir) / f"{r['name']}.npz")
        p2 = run["p2"]
        fs = p2["fspectra"]
        edges = r.get("first_rank2_edges") or []
        gam = []
        for e in edges[:2]:
            g, n = gamma_fit(p2["t"], fs[:, e, 1] / fs[:, e, 0])
            gam.append((e, round(g, 3) if np.isfinite(g) else None, n))
        d["gamma_first_edges"] = gam
        if r["tree"] == "mps":
            g4 = []
            for e in (4, 3, 6):
                g, n = gamma_fit(p2["t"], fs[:, e, 1] / fs[:, e, 0])
                g4.append((e, round(g, 3) if np.isfinite(g) else None, n))
            d["gamma_mps_edges_4_3_6"] = g4
        out.append(d)
    for d in out:
        print(json.dumps(d, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    return out


if __name__ == "__main__":
    res = main(*sys.argv[1:4])
    if len(sys.argv) > 4:
        json.dump(res, open(sys.argv[4], "w"), indent=1, default=str)
