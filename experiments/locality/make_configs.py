"""Write the YAML configs of the locality experiments (experiments/locality/configs/)."""
from pathlib import Path

import numpy as np
import yaml

from rules import get_tree, steiner_k

CFG = Path(__file__).parent / "configs"
SEEDS = list(range(8))


def logs(hi, lo, n=6):
    return [float(f"{x:.4g}") for x in np.logspace(np.log10(hi), np.log10(lo), n)]


def dump(name, cases, note):
    CFG.mkdir(exist_ok=True)
    dumper = type("NoAliases", (yaml.SafeDumper,), {"ignore_aliases": lambda self, data: True})
    text = yaml.dump({"cases": cases}, Dumper=dumper, sort_keys=False, default_flow_style=None,
                     width=110)
    (CFG / name).write_text(f"# {note}\n" + text)
    print(name, len(cases), "cases,", sum(len(c["seeds"]) * len(c["sigmas"]) for c in cases), "runs")


# ---- step 1/2: two-component teachers (weights 1, 0.5), every tree, two protocols ----
SETS = {
    ("mps", 4): [[0, 1], [1, 2], [0, 2], [0, 1, 2], [1, 2, 3], [0, 3], [0, 1, 2, 3]],
    ("binary", 4): [[0, 1], [2, 3], [1, 2], [0, 3], [0, 2], [0, 1, 2], [0, 1, 2, 3]],
    ("ht", 4): [[0, 1], [2, 3], [1, 2], [0, 3], [0, 2], [0, 1, 2], [0, 1, 2, 3]],
    ("mps", 6): [[0, 1], [2, 3], [0, 2], [2, 3, 4], [0, 3], [1, 4], [1, 3, 5], [0, 5], list(range(6))],
    ("mps", 8): [[0, 1], [3, 4], [2, 3, 4], [2, 5], [1, 5], [1, 6], [0, 7]],
    ("binary", 8): [[0, 1], [1, 2], [0, 2], [0, 1, 2, 3], [4, 5, 6], [3, 4], [0, 7], [2, 3, 4, 5],
                    [0, 2, 4]],
    ("ht", 8): [[0, 1], [6, 7], [1, 2], [0, 1, 2, 3], [3, 4]],
    ("star3", None): [[0, 1], [1, 2], [0, 5], [0, 2, 4], list(range(6))],
    ("midroot6", None): [[2, 3], [0, 2], [0, 3], [0, 5]],
}
# plateau ~ A sigma^-(k-2): keep the smallest sigma affordable; >= 1 decade wherever possible
RANGE = {1: (0.1, 1e-3), 2: (0.1, 1e-4), 3: (0.1, 1e-3), 4: (0.2, 0.01), 5: (0.3, 0.03), 6: (0.5, 0.08),
         7: (0.6, 0.15), 8: (0.6, 0.2)}
T1_MAX = 1e13  # first plateau ~ 3 sigma^-(K-2) (small init only)
cases = []
for (kind, n), sets in SETS.items():
    tree = get_tree(kind, n)
    K = tree.n_nodes
    for S in sets:
        k = steiner_k(tree, S)
        if k > 8:
            continue
        for start in ("small", "saddle"):
            hi, lo = RANGE[k]
            if start == "small":
                if kind == "ht" and n == 8:
                    continue
                lo = max(lo, (T1_MAX / 3) ** (-1 / (K - 2)))
            if hi / lo < 2.5:
                continue
            npts = 7 if hi / lo > 20 else 6
            extra = dict(seeds=SEEDS[:5], wall_limit=900) if k >= 6 else dict(seeds=SEEDS)
            # cost < 0: the runner starts the cheap (low-k) cases first
            cases.append(dict(kind=kind, n=n, S=S, start=start, sigmas=logs(hi, lo, npts), cost=-k, **extra))
dump("twocomp.yaml", cases, "Step 1/2: two-component pattern teachers (1, 0.5), d = 2, chi = 2")

# ---- step 3: multi-component teachers (theory_saddles.md section 5 + nested teachers) ----
T = {
    "B4": dict(patterns=[[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], weights=[1.0, 0.3, 0.6], chi=3),
    "N4": dict(patterns=[[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], weights=[1.0, 0.6, 0.4], chi=3),
    "TA": dict(patterns=[[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], weights=[1.0, 0.6, 0.35], chi=2),
    "TB": dict(patterns=[[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
               weights=[1.0, 0.6, 0.35, 0.2], chi=2),
    "E8": dict(patterns=[[0] * 8, [0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0]],
               weights=[1.0, 0.4, 0.7], chi=3),
    "TC": dict(patterns=[[0] * 8, [1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1] * 8],
               weights=[1.0, 0.6, 0.35, 0.2], chi=2),
    "TD": dict(patterns=[[0] * 8, [0, 0, 0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
               weights=[1.0, 0.6, 0.35], chi=2),
}
multi = []
for name in ("B4", "N4", "TA", "TB"):
    for kind, rng in (("mps", (0.2, 0.005)), ("binary", (0.1, 0.002)), ("ht", (0.25, 0.02))):
        if name == "B4" and kind == "mps":
            rng = (0.2, 0.01)
        if name == "B4" and kind == "ht":
            rng = (0.3, 0.05)
        for start in (("small", "small_balanced") if kind == "ht" else ("small",)):
            multi.append(dict(name=name, kind=kind, n=4, start=start, sigmas=logs(*rng), seeds=SEEDS,
                              cost=6, **T[name]))
for name, kind, rng in (("E8", "mps", (0.3, 0.06)), ("E8", "binary", (0.3, 0.04)),
                        ("TC", "mps", (0.25, 0.03)), ("TC", "binary", (0.25, 0.03)),
                        ("TD", "mps", (0.2, 0.02))):
    multi.append(dict(name=name, kind=kind, n=8, start="small", sigmas=logs(*rng), seeds=SEEDS, cost=8,
                      **T[name]))
dump("multicomp.yaml", multi, "Step 3: multi-component teachers on several trees")

# ---- step 5: robustness on N = 4 ----
base = [("mps", [1, 2], 2), ("mps", [0, 1, 2], 3), ("binary", [1, 2], 3), ("mps", [0, 3], 4),
        ("binary", [0, 1], 1), ("ht", [0, 1], 3)]
rob = []
for kind, S, k in base:
    hi, lo = RANGE[k]
    sig = logs(hi, max(lo, 1e-3) if kind != "ht" else max(lo, 0.004), 6)
    variants = [dict(d=2, chi=4)] + [dict(d=4, chi=c) for c in (2, 4, 8)] + \
        [dict(weights=[1.0, s2]) for s2 in (0.25, 0.75)]
    for v in variants:
        rob.append(dict(kind=kind, n=4, S=S, start="small", sigmas=sig, seeds=SEEDS, cost=k, **v))
for s2 in (0.25, 0.75):
    rob.append(dict(kind="mps", n=4, S=[1, 2], start="saddle", weights=[1.0, s2], sigmas=logs(0.1, 1e-4, 7),
                    seeds=SEEDS, cost=1))
dump("robust.yaml", rob, "Step 5: chi = 4, 8, d = 4, weight ratio 0.25 / 0.75 (N = 4)")

# ---- R12 check: gauge-balanced twins of the high-k two-component small-init cases ----
bal = [dict(c, start="small_balanced") for c in cases if c["start"] == "small"
       and steiner_k(get_tree(c["kind"], c["n"]), c["S"]) >= 4]
dump("twocomp_balanced.yaml", bal, "R12 check: gauge-balanced init (Q_e = 0) for two-component cases with k >= 4")

# ---- step 3 supplement: MPS (log plateaus) over more decades of sigma ----
wide = []
for name in ("B4", "N4", "TA", "TB"):
    # B4 has a k* = 4 step: below sigma ~ 1e-3 its escape signal (~ eps^3) nears the float64 floor
    lo, npts = (1e-3, 7) if name == "B4" else (1e-5, 9)
    wide.append(dict(name=name, kind="mps", n=4, start="small", sigmas=logs(0.1, lo, npts), seeds=SEEDS,
                     cost=3, **T[name]))
for name in ("TC", "TD"):
    wide.append(dict(name=name, kind="mps", n=8, start="saddle", sigmas=logs(0.1, 1e-4, 7), seeds=SEEDS,
                     cost=5, **T[name]))
dump("multicomp_wide.yaml", wide, "Step 3 supplement: MPS teachers with log plateaus, sigma over 4-5 decades")

# ---- round-2 fixes (review notes/reviews/locality_round1.md) ----
# chi = 4 twins of the k >= 4 cases (lead's request (b), round 1)
chi4 = [dict(c, chi=4, cost=c["cost"]) for c in cases if c["start"] == "small"
        and steiner_k(get_tree(c["kind"], c["n"]), c["S"]) >= 4]
dump("twocomp_chi4.yaml", chi4, "Spare bond channels (chi = 4, d = 2) for every two-component case with k >= 4, small init")

# k = 5 extension to smaller sigma for the two MPS rows above 3 (round 1)
k5 = [dict(kind=kind, n=n, S=S, start=st, sigmas=[0.03, 0.019, 0.012], seeds=SEEDS, cost=1)
      for kind, n, S in (("mps", 8, [1, 5]), ("mps", 6, [1, 3, 5])) for st in ("saddle", "small_balanced")]
dump("k5_extension.yaml", k5, "Pre-asymptotic check of the two MPS k = 5 rows: smaller sigma")

# k >= 6 rows over >= 1 decade with 8 seeds, gauge-balanced init (review issue 1)
khigh = [dict(kind="binary", n=8, S=[0, 2, 4], start="small_balanced", sigmas=logs(0.5, 0.05, 8), seeds=SEEDS,
              wall_limit=2400, cost=6),
         dict(kind="ht", n=4, S=[0, 1, 2], start="small_balanced", sigmas=logs(0.5, 0.05, 8), seeds=SEEDS,
              wall_limit=2400, cost=6),
         dict(kind="mps", n=8, S=[1, 6], start="small_balanced", sigmas=logs(0.5, 0.05, 8), seeds=SEEDS,
              wall_limit=2400, cost=6),
         dict(kind="ht", n=4, S=[0, 1, 2, 3], start="small_balanced", sigmas=logs(0.6, 0.06, 8), seeds=SEEDS,
              wall_limit=2400, cost=7)]
dump("khigh_extension.yaml", khigh, "k >= 6 rows over one decade of sigma, 8 seeds, gauge-balanced init")

# K-dependence of lambda_2 (s1 != 1): k = 2 from the exact saddle, weights (2, 1)
lam = [dict(kind="mps", n=n, S=S, start="saddle", weights=[2.0, 1.0], sigmas=logs(0.1, 1e-4, 7), seeds=SEEDS,
            cost=1) for n, S in ((4, [1, 2]), (6, [2, 3]), (8, [3, 4]))]
dump("lambda2_K.yaml", lam, "K-dependence of lambda_2 = s2 s1^((K-2)/K): weights (2, 1), saddle start, k = 2")
