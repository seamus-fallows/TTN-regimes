"""One run = gradient flow from small init on one target at one (sigma, seed), plus its per-run
analysis. Sweep = all runs of configs/targets.yaml in a process pool (resumable).

usage:
  uv run python experiments/route_nonstar/run.py sweep [--targets A,B] [--workers N] [--out DIR]
         [--sigmas -1,-2] [--seeds 0-39]
  uv run python experiments/route_nonstar/run.py one TARGET LOG10_SIGMA SEED      (prints the record)

Raw output per run (not in git): <out>/raw/<target>/s<log10 sigma>_seed<k>.json (record) and .npz
(every solver step: t, L, |grad|, amplitudes P, off-target mass; W at <= 400 steps in float32;
theta at every plateau's minimum-gradient point in float64). Definitions: notes/route_nonstar.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "locality"))
from nsdyn import Model, integrate, pattern_str  # noqa: E402
from rules import route_rule  # noqa: E402

from ttn.diagnostics import find_plateaus  # noqa: E402
from ttn.routes import AlignedManifold, effective_depth  # noqa: E402

torch.set_num_threads(1)
OUT = Path(os.environ.get("ROUTE_NONSTAR_OUT", "/root/outputs/route-nonstar"))
CFG = HERE / "configs" / "targets.yaml"

# registered analysis constants (notes/route_nonstar.md, section 3)
MIN_DROP = 0.02  # find_plateaus: a loss drop must lower L by >= 2 %
CANONICAL = 0.03  # a plateau is a canonical partial-target point if its relative distance < this
OFF_STRING = 0.01  # off-target strings listed when |W[x]| >= this
ESC = 1e-3  # escape from a plateau: L < L_plateau - ESC
NEG = -1e-9  # a Hessian eigenvalue below this counts as negative
FLOOR = 100.0  # floor flag: |grad| < FLOOR x its roundoff bound at the minimum-gradient point
ROUTE_SUB_TOL = 1e-2  # route_rule at a plateau point: singular directions below 1e-2 are dormant
ROUTE_TOL = 1e-2  # ... and a local residual counts if ||R_C|| > 1e-2 ||R|| (the point is only near
# the critical point, so tiny R_C on small sets are approach residue, not routes)


def eff_depth(t, y, s):
    """ttn.routes.effective_depth on a window, None if the window has too few distinct times."""
    t, y = np.asarray(t), np.asarray(y)
    ok = np.concatenate([[True], np.diff(t) > 0])
    if ok.sum() < 6:
        return None
    return effective_depth(t[ok], y[ok], s)


def load_targets(path=CFG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg, {t["name"]: t for t in cfg["targets"]}


def make_model(t: dict, spare: int) -> Model:
    pats = [[int(c) for c in p] for p in t["patterns"]]
    return Model(t["kind"], t["n"], pats, t["weights"], d=t.get("d", 2), spare=spare)


def crossings(t, y, level):
    """All crossings of y = level on the step values: (time by linear interpolation, +1 up / -1
    down, index of the step after the crossing)."""
    out = []
    above = y >= level
    for i in np.nonzero(above[1:] != above[:-1])[0]:
        frac = (level - y[i]) / (y[i + 1] - y[i])
        out.append((float(t[i] + frac * (t[i + 1] - t[i])), 1 if above[i + 1] else -1, int(i + 1)))
    return out


def recombinant_tag(model: Model, x: tuple) -> str:
    """'e<edge>:<a>|<b>' if string x takes the lower side of an edge cut from target a and the upper
    side from target b (a recombinant across one cut); '' otherwise."""
    tree = model.tree
    pats = model.patterns
    for e in tree.edges:
        below = tree.subtree_sites[e]
        above = [s for s in range(tree.n_sites) if s not in below]
        for a, b in itertools.permutations(range(len(pats)), 2):
            if all(x[s] == pats[a][s] for s in below) and all(x[s] == pats[b][s] for s in above):
                return f"e{e}:{pattern_str(pats[a])}|{pattern_str(pats[b])}"
    return ""


def partial_distances(model: Model, w: np.ndarray):
    """Relative distance ||W - sum_J s_a U_a|| / ||W*|| for every subset J; returns the nearest J
    and its distance, and the function itself."""
    norm_star = math.sqrt(2 * model.L_star)
    amps = w[model.flat_idx]
    tot2 = float((w**2).sum())
    off2 = max(tot2 - float((amps**2).sum()), 0.0)

    def dist(J):
        on = np.where(np.isin(np.arange(model.n_pat), list(J)), model.weights, 0.0)
        return math.sqrt(off2 + float(((amps - on) ** 2).sum())) / norm_star

    best = min((frozenset(J) for k in range(model.n_pat + 1)
                for J in itertools.combinations(range(model.n_pat), k)), key=dist)
    return best, dist(best), dist


def jset(model: Model, J) -> str:
    return "{" + ",".join(pattern_str(model.patterns[a]) for a in sorted(J)) + "}"


def plateau_point(model: Model, theta: np.ndarray, man: AlignedManifold | None) -> dict:
    """Everything measured at one plateau's minimum-gradient point."""
    th = torch.from_numpy(theta)
    s = model.summary(th)
    w, P = s["w"], s["P"]
    J_amp = frozenset(a for a in range(model.n_pat) if P[a] > model.weights[a] / 2)
    J_near, d_near, dist = partial_distances(model, w)
    rest = w.copy()
    rest[model.flat_idx] = 0.0
    order = np.argsort(-np.abs(rest))
    shape = (model.d,) * model.tree.n_sites
    strings = []
    for j in order[:8]:
        if abs(rest[j]) < OFF_STRING:
            break
        x = tuple(int(i) for i in np.unravel_index(int(j), shape))
        strings.append(dict(string=pattern_str(x), amp=float(rest[j]), tag=recombinant_tag(model, x)))
    H = model.hessian(th).numpy()
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    g, _ = model.grad(th)
    gnorm = float(g.norm())
    bound = float(np.linalg.norm(model.grad_roundoff(th)))
    k_star, hits = route_rule(model.tree, w.reshape(shape), model.w_star_np, tol=ROUTE_TOL,
                              sub_tol=ROUTE_SUB_TOL)
    route = None
    if hits:
        C, rel = max(hits, key=lambda h: h[1])
        route = dict(nodes=[int(v) for v in C], rel=float(rel))
        # which string the minimal route serves: the largest entry of its local residual
        rc = _local_residual(model, w.reshape(shape), C)
        j = int(np.argmax(np.abs(rc)))
        x = tuple(int(i) for i in np.unravel_index(j, shape))
        route["string"] = pattern_str(x)
        route["is_target"] = x in model.patterns
    out = dict(L=float(0.5 * ((w - model.w_star_np.reshape(-1)) ** 2).sum()), P=P.tolist(),
               off=s["off"], J_amp=jset(model, J_amp), J_amp_idx=sorted(J_amp),
               J_near=jset(model, J_near), J_near_idx=sorted(J_near), d_near=d_near,
               d_amp=dist(J_amp), canonical=bool(d_near < CANONICAL), off_strings=strings,
               lam_min=float(ev[0]), lam_2=float(ev[1]), n_neg=int((ev < NEG).sum()),
               lam_max=float(ev[-1]), gnorm=gnorm, grad_bound=bound,
               floor_flag=bool(gnorm < FLOOR * bound), route_k=k_star, route=route)
    if man is not None and J_amp and len(J_amp) < model.n_pat:
        out["entries"] = _route_entries(model, man, th, sorted(J_amp))
    return out


def _local_residual(model: Model, w: np.ndarray, C) -> np.ndarray:
    """R_C of experiments/locality/rules.route_rule (the residual with every subtree hanging off C
    projected onto the subspace the outside cores carry), flattened."""
    from rules import _project_sites, learned_subspace

    tree = model.tree
    rc = model.w_star_np - w
    for v in tree.edges:
        p = tree.parent[v]
        if (v in C) == (p in C):
            continue
        far = sorted(tree.subtree_sites[v]) if v not in C else \
            [i for i in range(tree.n_sites) if i not in tree.subtree_sites[v]]
        rc = _project_sites(rc, far, learned_subspace(w, far, tol=ROUTE_SUB_TOL))
    return rc.reshape(-1)


def _route_entries(model: Model, man: AlignedManifold, th, J) -> dict:
    """Star-condition controls only: route-entry norms of every missing pattern after `align`."""
    cores = model.spec.unflatten(th.clone())
    try:
        aligned, quality = man.align(cores, J)
    except Exception as exc:  # noqa: BLE001
        return dict(error=repr(exc))
    out = dict(quality=min(quality.values()), norms={})
    for r in range(model.n_pat):
        if r in J:
            continue
        try:
            norms = man.route_entry_norms(aligned, J, r)
            out["norms"][pattern_str(model.patterns[r])] = {int(k): v for k, v in norms.items()}
        except Exception as exc:  # noqa: BLE001
            out["norms"][pattern_str(model.patterns[r])] = repr(exc)
    return out


def relative_clock(res: dict, P: np.ndarray, weights: np.ndarray):
    """Times relative to T0 = the first step with some P_a > 0.05 s_a (the start of the first
    escape), formed in long double from the solver's offset and local clock: absolute times reach
    1e17, where float64 spacing (16) exceeds the steps of a fast transition. Steps whose relative
    time does not advance in float64 are dropped. Returns (t_rel, kept indices, T0)."""
    T = res["toff"].astype(np.longdouble) + res["tloc"].astype(np.longdouble)
    started = np.nonzero((P / weights > 0.05).any(axis=1))[0]
    T0 = T[started[0]] if len(started) else T[0]
    t = np.asarray(T - T0, dtype=np.float64)
    keep = [0]
    for i in range(1, len(t)):
        if t[i] > t[keep[-1]]:
            keep.append(i)
    keep = np.array(keep)
    return t[keep], keep, float(T0)


def analyse(model: Model, res: dict, man) -> tuple[dict, dict]:
    S = [model.summary(torch.from_numpy(y)) for y in res["y"]]
    P = np.array([s["P"] for s in S])
    t, idx, T0 = relative_clock(res, P, model.weights)
    S = [S[i] for i in idx]
    P, Y, L = P[idx], res["y"][idx], res["L"][idx]
    n = len(t)
    off = np.array([s["off"] for s in S])
    gnorm = res["gnorm"][idx]
    s_w = model.weights
    rec: dict = dict(T0=T0)
    # ---- amplitude crossings and learning events (brief section 3 definitions)
    cross, events, forgotten = {}, [], []
    for a in range(model.n_pat):
        c = crossings(t, P[:, a], s_w[a] / 2)
        cross[pattern_str(model.patterns[a])] = [(tc, dr) for tc, dr, _ in c]
        ups = [x for x in c if x[1] > 0]
        if ups and P[-1, a] > s_w[a] / 2:
            tc, _, i = ups[-1]
            events.append(dict(a=a, t=tc, i=i))
        for tc, dr, i in c:
            if dr > 0:
                down = [x for x in crossings(t[i:], P[i:, a], s_w[a] / 4) if x[1] < 0]
                if down:
                    forgotten.append(dict(pattern=pattern_str(model.patterns[a]), t_learned=tc,
                                          t_forgotten=down[0][0]))
    events.sort(key=lambda e: e["t"])
    rec["crossings"] = cross
    rec["forgotten"] = forgotten
    rec["order"] = [pattern_str(model.patterns[e["a"]]) for e in events]
    # ---- loss drops and the segments between them
    plats, drops = find_plateaus(t, L, min_drop=MIN_DROP)
    bounds = [float(t[0])] + [d.t_half for d in drops] + [float(t[-1])]
    segs = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        inside = np.nonzero((t > lo) & (t < hi))[0]
        if k == 0:
            inside = np.nonzero(t < hi)[0]
            lo = float(t[0])
        if len(inside) == 0:
            segs.append(None)
            continue
        i_mid = int(inside[np.argmin(gnorm[inside])])
        final = k == len(bounds) - 2
        resolved = any(p.t_start <= t[i_mid] <= p.t_end for p in plats)
        segs.append(dict(k=k, t_start=lo, t_end=hi, i_mid=i_mid, t_mid=float(t[i_mid]),
                         final=final, resolved=resolved))
    states = {}
    for seg in segs:
        if seg is None:
            continue
        converged = seg["final"] and res["status"] == "converged"
        if seg["k"] == 0 or converged:
            seg["L"] = float(L[seg["i_mid"]])
            seg["P"] = P[seg["i_mid"]].tolist()
            continue
        states[f"seg{seg['k']}"] = Y[seg["i_mid"]]
        seg.update(plateau_point(model, Y[seg["i_mid"]], man))
        later = np.nonzero((t > seg["t_mid"]) & (L < seg["L"] - ESC))[0]
        seg["t_escape"] = float(t[later[0]] - seg["t_start"]) if len(later) else None
    # ---- what each drop does (between the minimum-gradient points around it)
    trans = []
    for k, d in enumerate(drops):
        a_seg, b_seg = segs[k], segs[k + 1]
        if a_seg is None or b_seg is None:
            trans.append(None)
            continue
        i0, i1 = a_seg["i_mid"], b_seg["i_mid"]
        if i1 <= i0:
            i1 = min(i0 + 1, n - 1)
        J0 = {a for a in range(model.n_pat) if P[i0, a] > s_w[a] / 2}
        J1 = {a for a in range(model.n_pat) if P[i1, a] > s_w[a] / 2}
        win = slice(i0, i1 + 1)
        w0, w1 = S[i0]["w"], S[i1]["w"]
        j = int(np.argmax(np.abs(w1 - w0)))
        x = tuple(int(q) for q in np.unravel_index(j, (model.d,) * model.tree.n_sites))
        dL = L[i0] - L[win]
        eff_loss = eff_depth(t[win], dL, max(L[i0] - L[i1], 1e-300))
        eff_amp = {}
        for a in sorted(J1 - J0):
            e = eff_depth(t[win], P[win, a], s_w[a])
            eff_amp[pattern_str(model.patterns[a])] = e["depth"] if e else None
        trans.append(dict(k=k + 1, t_half=d.t_half, width=d.width, L_before=d.loss_before,
                          L_after=d.loss_after, learned=[pattern_str(model.patterns[a])
                                                         for a in sorted(J1 - J0)],
                          lost=[pattern_str(model.patterns[a]) for a in sorted(J0 - J1)],
                          top_change=pattern_str(x), top_change_is_target=x in model.patterns,
                          top_change_dw=float(w1[j] - w0[j]),
                          off_peak=float(off[win].max()), off_before=float(off[i0]),
                          off_after=float(off[i1]),
                          eff_depth_loss=eff_loss["depth"] if eff_loss else None,
                          eff_depth_amp=eff_amp))
    rec["segments"] = segs
    rec["drops"] = trans
    # ---- learning events with the drop that contains them
    ev_out = []
    prev_t = float(t[0])
    for e in events:
        a, i = e["a"], e["i"]
        J_before = sorted(b for b in range(model.n_pat) if P[i - 1, b] > s_w[b] / 2 and b != a)
        k_drop = int(np.argmin([abs(d.t_half - e["t"]) for d in drops])) + 1 if drops else None
        # amplitude estimator over the plateau before the event (from the previous event on)
        win = (t >= prev_t) & (t <= e["t"])
        eff = eff_depth(t[win], P[win, a], s_w[a])
        ev_out.append(dict(pattern=pattern_str(model.patterns[a]), a=a, t=e["t"],
                           duration=e["t"] - prev_t, J_before=J_before, drop=k_drop,
                           eff_depth=eff["depth"] if eff else None,
                           eff_points=eff["points"] if eff else 0))
        prev_t = e["t"]
    rec["events"] = ev_out
    # ---- what to save
    tl = t - t[0] + 1.0
    keep = np.unique(np.concatenate([np.linspace(0, n - 1, min(n, 200)).astype(int),
                                     np.searchsorted(tl, np.logspace(0, math.log10(tl[-1]),
                                                                     200)).clip(0, n - 1)]))
    arrays = dict(t=t, L=L, gnorm=gnorm, P=P, off=off, keep=keep,
                  W=np.array([S[i]["w"] for i in keep], dtype=np.float32))
    for k, v in states.items():
        arrays["theta_" + k] = v
    return rec, arrays


def run_one(name: str, log_sigma: float, seed: int, cfg: dict, targets: dict, save=True) -> dict:
    t0 = time.time()
    tgt = targets[name]
    model = make_model(tgt, cfg["spare"])
    man = None
    if tgt["star"]:
        man = AlignedManifold(model.target, model.d, chi=None)
        if man.spec.bond_dims != model.spec.bond_dims:
            man = _manifold_with_bonds(model, man)
    sigma = 10.0**log_sigma
    y0 = model.init(sigma, seed)
    with threadpool_limits(1):
        res = integrate(model, y0, t_max=cfg["t_max"], rtol=cfg["rtol"], atol=cfg["atol"],
                        wall=cfg["wall"])
        rec, arrays = analyse(model, res, man)
    rec.update(t_first_escape_abs=float(rec["T0"]) + (rec["drops"][0]["t_half"] if rec["drops"]
                                                         and rec["drops"][0] else 0.0),
               target=name, family=tgt["family"], tree=f"{tgt['kind']}{tgt['n']}", kind=tgt["kind"],
               n=tgt["n"], K=model.tree.n_nodes, d=model.d, chi=max(model.spec.bond_dims),
               bonds=list(model.spec.bond_dims), n_params=model.n_params, log10_sigma=log_sigma,
               sigma=sigma, seed=seed, status=res["status"], n_steps=len(res["t"]),
               t_end=float(res["t"][-1]), L_end=float(res["L"][-1]),
               P_end=arrays["P"][-1].tolist(), off_end=float(arrays["off"][-1]),
               n_restart=res["n_restart"], fails=res["fails"], secs_integrate=res["secs"],
               secs=time.time() - t0)
    if save:
        d = OUT / "raw" / name
        d.mkdir(parents=True, exist_ok=True)
        stem = f"s{log_sigma:+.2f}_seed{seed}"
        np.savez_compressed(d / f"{stem}.npz", **arrays)
        (d / f"{stem}.json").write_text(json.dumps(rec, default=_json))
    return rec


def _manifold_with_bonds(model, man):
    """AlignedManifold whose spec has the experiment's bond dimensions (blocks + 2 = rank + 2
    under the star condition)."""
    chi = max(model.spec.bond_dims)
    man2 = AlignedManifold(model.target, model.d, chi=chi)
    if man2.spec.bond_dims != model.spec.bond_dims:
        return None
    return man2


def _json(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, frozenset | set):
        return sorted(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _task(args):
    name, ls, seed = args
    cfg, targets = load_targets()
    stem = OUT / "raw" / name / f"s{ls:+.2f}_seed{seed}"
    if stem.with_suffix(".json").exists():
        return name, ls, seed, "skip", 0.0
    try:
        rec = run_one(name, ls, seed, cfg, targets)
        return name, ls, seed, rec["status"], rec["secs"]
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        (OUT / "errors").mkdir(parents=True, exist_ok=True)
        (OUT / "errors" / f"{name}_s{ls:+.2f}_seed{seed}.txt").write_text(err)
        return name, ls, seed, "error", 0.0


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "one"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigmas", default=None, help="comma-separated log10 sigmas (default: config)")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    if a.out:
        OUT = Path(a.out)
        os.environ["ROUTE_NONSTAR_OUT"] = str(OUT)
    cfg, targets = load_targets()
    if a.mode == "one":
        name, ls, seed = a.args[0], float(a.args[1]), int(a.args[2])
        rec = run_one(name, ls, seed, cfg, targets, save=False)
        print(json.dumps(rec, default=_json, indent=1))
        return
    names = a.targets.split(",") if a.targets else list(targets)
    seeds = parse_range(a.seeds) if a.seeds else cfg["seeds"]
    tasks = []
    for name in names:
        sig = [float(x) for x in a.sigmas.split(",")] if a.sigmas else targets[name]["log10_sigmas"]
        tasks += [(name, ls, s) for ls in sig for s in seeds]
    # expensive first (large trees, small sigma) so the pool drains evenly
    tasks.sort(key=lambda x: (-targets[x[0]]["K"], x[1]))
    print(f"{len(tasks)} runs, {a.workers} workers, out {OUT}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(a.workers, maxtasksperchild=50) as pool:
        for name, ls, seed, status, secs in pool.imap_unordered(_task, tasks):
            done += 1
            if status != "skip":
                print(f"[{done}/{len(tasks)} {time.time() - t0:.0f}s] {name} {ls:+.2f} {seed} "
                      f"{status} {secs:.1f}s", flush=True)
    print(f"finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
