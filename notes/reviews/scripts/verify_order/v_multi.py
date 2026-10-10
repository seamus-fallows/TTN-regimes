"""Verifier: R >= 3 recomputation from raw multi3 / multi_nostop / multi2 (independent code)."""

import json
import itertools
import numpy as np
from scipy.stats import norm, binomtest

RAW = "c:/Users/seamu/Iliad-ML/TTN_regimes/data/order"


def load(name):
    return [json.loads(l) for l in open(f"{RAW}/{name}/runs.jsonl")]


def greedy(times):
    t = np.array([np.inf if x is None else x for x in times])
    return bool(np.all(np.diff(t) > 0))


def iid_model(taus, w, n=400000, seed=1):
    rng = np.random.default_rng(seed)
    d = rng.choice(taus, size=(n, len(w))) / w
    return np.mean(np.all(np.diff(d, axis=1) > 0, axis=1))


print("=== multi3 (R = 3) ===")
runs = load("multi3")
groups = {}
for r in runs:
    groups.setdefault((r["tree"], r["rho"]), []).append(r)
zs = {}
paired = {}
for key in sorted(groups):
    g = groups[key]
    seeds = sorted(r["seed"] for r in g)
    missing = sorted(set(range(200)) - set(seeds))
    stat = {
        s: sum(r["status"] == s for r in g) for s in ("converged", "stalled", "t_max", "failed")
    }
    fin = [all(x is not None for x in r["times"]) for r in g]
    flag = [np.nanmin(np.array(r["min_slice"], float)) < 1e-12 for r in g]
    used = [r for r, f, fl in zip(g, fin, flag) if f and not fl]
    excl = [r for r, f, fl in zip(g, fin, flag) if not (f and not fl)]
    w = key[1] ** np.arange(3)
    p_gf = np.mean([greedy(r["times"]) for r in used])
    race_used = [
        greedy(np.array(r["tau"]) / w * 1.0)
        if False
        else bool(np.all(np.diff(np.array(r["tau"]) / w) > 0))
        for r in used
    ]
    p_race_own = np.mean(race_used)
    p_mod = iid_model(np.concatenate([r["tau"] for r in used]), w)
    z = (p_gf - p_mod) / np.sqrt(p_mod * (1 - p_mod) / len(used))
    # paired: GF greedy vs own race greedy (McNemar on discordant pairs)
    gfg = np.array([greedy(r["times"]) for r in used])
    rg = np.array(race_used)
    b, c = int(np.sum(gfg & ~rg)), int(np.sum(~gfg & rg))
    p_mc = binomtest(b, b + c).pvalue if b + c else 1.0
    # first-event / full-order agreement with own race
    first_ok = np.mean(
        [np.argmin([x for x in r["times"]]) == np.argmin(np.array(r["tau"]) / w) for r in used]
    )
    full_ok = np.mean(
        [tuple(np.argsort(r["times"])) == tuple(np.argsort(np.array(r["tau"]) / w)) for r in used]
    )
    # excluded: own-race first component and which component unlearned
    ex_first = (
        np.bincount([int(np.argmin(np.array(r["tau"]) / w)) for r in excl], minlength=3)
        if excl
        else None
    )
    us_first = np.bincount([int(np.argmin(np.array(r["tau"]) / w)) for r in used], minlength=3)
    unl = np.zeros(3, int)
    for r in excl:
        for i, x in enumerate(r["times"]):
            if x is None:
                unl[i] += 1
    race_g_ex = (
        np.mean([bool(np.all(np.diff(np.array(r["tau"]) / w) > 0)) for r in excl])
        if excl
        else np.nan
    )
    zs[key] = z
    paired[key] = (gfg, rg, [r["seed"] for r in used])
    # min slice of completed runs, of t_max runs' unlearned components
    ms_conv = min(np.nanmin(np.array(r["min_slice"], float)) for r in used)
    tmax_ms = [
        np.nanmin([m for m, t in zip(r["min_slice"], r["times"]) if t is None])
        for r in g
        if r["status"] == "t_max"
    ]
    walls = {
        s: np.median([r["wall"] for r in g if r["status"] == s]) if stat[s] else np.nan
        for s in stat
    }
    print(f"{key}: n={len(g)} missing={missing} {stat} used={len(used)} floor_flag={sum(flag)}")
    print(
        f"   P_gf={p_gf:.4f} P_model(iid,same runs)={p_mod:.4f} z={z:+.2f} | own race on used={p_race_own:.4f}"
        f" McNemar GF>race {b} vs race>GF {c} p={p_mc:.3f}"
    )
    print(
        f"   own race: first agree={first_ok:.3f} full order agree={full_ok:.3f}; race-greedy excluded={race_g_ex:.3f}"
    )
    print(
        f"   race-first used={us_first.tolist()} excluded={None if ex_first is None else ex_first.tolist()} unlearned(excl)={unl.tolist()}"
    )
    print(
        f"   min slice ratio over used runs before events={ms_conv:.2e}; t_max runs' unlearned min slice: "
        f"{sorted('%.0e' % x for x in tmax_ms)}; median wall by status {walls}"
    )

m3 = [k for k in zs if not k[0].startswith("binary:4")]
zz = np.array([zs[k] for k in m3])
print("m>=3 z:", np.round(zz, 2), "Stouffer (indep.)", round(zz.sum() / np.sqrt(len(zz)), 2))
# correlation of the GF-greedy indicator between rho = 0.8 and 0.9 for the same seed (shared inits)
cors = []
for tree in ("binary:8", "mps:4", "mps:6"):
    a, b = paired[(tree, 0.8)], paired[(tree, 0.9)]
    common = sorted(set(a[2]) & set(b[2]))
    ia = {s: i for i, s in enumerate(a[2])}
    ib = {s: i for i, s in enumerate(b[2])}
    x = np.array([a[0][ia[s]] for s in common], float)
    y = np.array([b[0][ib[s]] for s in common], float)
    cors.append(np.corrcoef(x, y)[0, 1])
print("corr(GF greedy at rho .8, .9) per tree:", np.round(cors, 2))
c = np.mean(cors)
var = len(zz) + 2 * 3 * c
print(f"Stouffer with within-tree correlation {c:.2f}: {zz.sum() / np.sqrt(var):.2f}")

print("\n=== multi_nostop (R = 4, no stall stop) ===")
for r in load("multi_nostop"):
    pass
rows = load("multi_nostop")
learned_ms, unlearned_ms = [], []
for r in rows:
    for t, m in zip(r["times"], r["min_slice"]):
        (learned_ms if t is not None else unlearned_ms).append(m)
print(
    "runs",
    len(rows),
    "statuses",
    {s: sum(r["status"] == s for r in rows) for s in ("t_max", "converged")},
    "t_end min",
    min(r["t_end"] for r in rows),
)
print(
    "learned components: n=%d, min slice ratio before event: min %.2e"
    % (len(learned_ms), min(learned_ms))
)
print(
    "unlearned components: n=%d, below 1e-14: %d, values:"
    % (len(unlearned_ms), sum(m < 1e-14 for m in unlearned_ms)),
    sorted("%.1e" % m for m in unlearned_ms),
)
for tree in ("mps:6", "binary:8"):
    rr = [r for r in rows if r["tree"] == tree]
    nl = [sum(t is not None for t in r["times"]) for r in rr]
    would_stall = sum(
        any(t is None and m < 1e-14 for t, m in zip(r["times"], r["min_slice"])) for r in rr
    )
    print(
        tree,
        "seeds",
        sorted(r["seed"] for r in rr),
        "learned counts",
        sorted(nl),
        "runs the slice stop would have stopped:",
        would_stall,
    )

print("\n=== multi2 (round 1) R = 4 ===")
for key in itertools.product(("binary:4", "mps:4", "mps:6", "binary:8"), (0.8, 0.9)):
    g = [r for r in load("multi2") if r["tree"] == key[0] and r["rho"] == key[1] and r["R"] == 4]
    inc = [r for r in g if any(t is None for t in r["times"])]
    nl = np.bincount([sum(t is not None for t in r["times"]) for r in inc], minlength=5)
    print(key, "n", len(g), "incomplete", len(inc), "learned-at-stop counts (0..4)", nl.tolist())
