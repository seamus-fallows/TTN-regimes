"""Second-escape law from the rank-one plateau, per seed (re-analysis after the round-1 review).

For the dense library runs of a random-init sigma sweep (configs e4a_sigma, e2_sigma, e4b_ttn,
r2_random), following theory_teachers.md v2 section 9:
- rank-one critical points of the loss on the student's tree (constrained.py): levels L_1 and, for
  whitened MPS-4 students, the predicted slope A = (1 - kappa_e*)/gamma_e* of each (R6);
- per run: k0 = first time L < 0.9 L(0); the second-escape event t_e(f) = first time after k0 that
  s_2^e >= f sigma_2^e(T) on some edge (diagnostics.md: f = 0.05, sensitivity f = 0.03, 0.1); the plateau point = minimum parameter speed
  between k0 and t_e(0.03); its loss is matched to the nearest rank-one level, and the run counts as
  RESOLVED only if it is within 5% of it (else it is flagged and excluded from the fits);
  t_1 = first time L <= (L(0) + L_visited)/2; gap_f = t_e(f) - t_1; first edge = argmax at t_e;
- gauge-projected Hessian lambda_min at the plateau point (strict saddle: sigma-independent < 0;
  higher order: -> 0 with sigma);
- per-seed fits of the log and the power law in the same (log) space (lawfit.py), runs grouped by the
  visited saddle; mean +- s.e.m. over seeds;
- asymptotic slopes (fix round 2): per seed through the 3 smallest sigma (A_loc, p_loc), compared with
  R6 for the edge that grows first at those sigma (A_pred_edge; seeds whose first edge changes are
  counted apart).
Usage: uv run python experiments/data_teacher/analyze_law_v2.py <config name> [--procs 8]
Writes results/<name>_v2_runs.csv, _v2_perseed.csv, _v2_perseed_local.csv, _v2_summary.csv, _v2.png.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, RESULTS  # noqa: E402
from constrained import constrained_minima, metric_loss_of_dense, ranked_spec  # noqa: E402
from lawfit import per_seed, per_seed_local  # noqa: E402
from run_generic import expand  # noqa: E402
from setups import build  # noqa: E402

from ttn.diagnostics import gauge_directions, hessian, project_out_gauge  # noqa: E402
from ttn.dynamics import Problem, Trajectory, merged_rows  # noqa: E402

HERE = Path(__file__).parent


def rank1_info(spec, teacher, metrics):
    """Rank-one critical levels; predicted R6 slopes for whitened MPS-4 students (else nan): A_pred for
    the predicted first edge e* = argmax s2(M_e), A_pred_edges for every edge."""
    spec1 = ranked_spec(spec, {v: 1 for v in spec.tree.edges})
    T = teacher.dense()
    mins = constrained_minima(spec1, metric_loss_of_dense(T, metrics), n_starts=12)
    out = []
    for m in mins:
        a_pred, a_edges = np.nan, [np.nan] * len(spec.tree.edges)
        if metrics is None and spec.tree.name.startswith("mps") and T.ndim == 4:
            n = 4
            X = m["W"]
            w = [torch.linalg.svd(torch.movedim(X, i, 0).reshape(T.shape[i], -1))[0][:, 0]
                 for i in range(n)]
            lam = float(torch.einsum("abcd,a,b,c,d->", T, *w))
            if lam < 0:
                w[0], lam = -w[0], -lam
            s2 = []
            for i in range(n - 1):
                M = T
                for k in sorted([k for k in range(n) if k not in (i, i + 1)], reverse=True):
                    M = torch.tensordot(M, w[k], dims=([k], [0]))
                s2.append(float(torch.linalg.svdvals(M)[1]))
            e = int(np.argmax(s2))
            # odeco teachers: no 2-core route (s2(M_e) = 0), no log-law prediction
            a_edges = [(1 - x / lam) / (lam ** 0.5 * x) if x > 1e-8 else np.nan for x in s2]
            a_pred = a_edges[e]
        out.append(dict(level=float(m["loss"]), A_pred=a_pred, A_pred_edges=a_edges))
    return out


def analyze_run(args):
    path, meta, levels, spec_args = args
    torch.set_num_threads(1)
    tr = Trajectory.load(path)
    # plateau point (speed minimum, parameters) from the log rows; event times from the merged
    # log + fine rows when the run has them (final library; older runs have log rows only)
    tl, speed, params = tr.data["t"], tr.data["speed"], tr.data["params"]
    if "fine_t" in tr.data:
        m = merged_rows(tr.data)
        t, loss, sp = m["t"], m["loss"], np.nan_to_num(m["spectra"])
    else:
        t, loss, sp = tl, tr.data["loss"], np.nan_to_num(tr.data["spectra"])
    tsp = tr.data["teacher_spectra"]
    edges = [j for j in range(sp.shape[1]) if tsp[j, 1] > 1e-3]
    k0 = int(np.argmax(loss < 0.9 * loss[0]))

    def event(f, start):
        hits = []
        for j in edges:
            h = np.nonzero((np.arange(len(t)) >= start) & (sp[:, j, 1] >= f * tsp[j, 1]))[0]
            if len(h):
                hits.append((h[0], j))
        return min(hits) if hits else (None, None)

    ke3, _ = event(0.03, k0)
    row = dict(meta)
    if ke3 is None or ke3 - k0 < 3:
        row.update(resolved=False, reason="no plateau between first escape and second event")
        return row
    # plateau point on the log grid between the first escape and the second-escape onset
    lk0, lke = np.searchsorted(tl, t[k0]), np.searchsorted(tl, t[ke3])
    if lke - lk0 < 3:
        row.update(resolved=False, reason="plateau shorter than 3 log rows")
        return row
    kpl = lk0 + int(np.argmin(speed[lk0:lke]))
    lp = float(tr.data["loss"][kpl])
    lv = np.array([x["level"] for x in levels])
    j = int(np.argmin(np.abs(lv - lp) / lv))
    rel = abs(lv[j] - lp) / lv[j]
    row.update(plateau_level=lp, saddle=j, level_pred=lv[j], level_rel_err=rel,
               A_pred=levels[j]["A_pred"], A_pred_edges=json.dumps(levels[j]["A_pred_edges"]), resolved=bool(rel < 0.05),
               reason="" if rel < 0.05 else "speed minimum not at a rank-one level")
    l_vis = lv[j]
    k1 = int(np.argmax(loss <= 0.5 * (loss[0] + l_vis)))
    row["t1"] = float(t[k1])
    for f in (0.03, 0.05, 0.1):
        ke, edge = event(f, k1)
        row[f"gap_{f}"] = float(t[ke] - t[k1]) if ke is not None else np.nan
        row[f"edge_{f}"] = edge
    # gauge-projected Hessian at the plateau point
    spec, teacher, metrics = spec_args
    prob = Problem(spec, teacher, site_metrics=metrics)
    th = torch.as_tensor(params[kpl])
    h = hessian(prob.loss, th)
    _, eigs = project_out_gauge(h, gauge_directions(spec, th))  # library default rtol (1e-3)
    row["lambda_min"] = float(eigs[0])
    row["lambda_min_raw"] = float(torch.linalg.eigvalsh(h)[0])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    name = a.config
    cfg = yaml.safe_load(open(HERE / "configs" / f"{name}.yaml"))
    jobs, levels_all = [], {}
    for case in expand(cfg):
        for tree in cfg["trees"]:
            built = build(dict(case, tree=tree), 0)
            if built is None:
                continue
            spec, teacher, metrics = built[:3]
            levels = rank1_info(spec, teacher, metrics)
            levels_all[(case["tag"], tree)] = levels
            for sigma in cfg["sigmas"][tree]:
                for seed in range(*cfg["seeds"]):
                    p = OUT_ROOT / f"{name}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
                    if p.exists():
                        meta = dict(case=case["tag"], tree=tree, sigma=sigma, seed=seed)
                        jobs.append((p, meta, levels, (spec, teacher, metrics)))
    # spawn: the parent has used autograd (constrained minima), which does not survive a fork
    with mp.get_context("spawn").Pool(a.procs) as pool:
        rows = pool.map(analyze_run, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}_v2_runs.csv", index=False)
    res = df[df.resolved == True].copy()  # noqa: E712
    out_ps, out_sum, out_psl = [], [], []
    for f in (0.03, 0.05, 0.1):
        r = res.rename(columns={f"gap_{f}": "gap"})
        ps, summ = per_seed(r, ["case", "tree", "saddle"])
        if len(ps):
            ps["f"], summ["f"] = f, f
            ap_ = r.groupby(["case", "tree", "saddle"])["A_pred"].first().reset_index()
            summ = summ.merge(ap_, on=["case", "tree", "saddle"], how="left")
            lam = r.groupby(["case", "tree", "saddle"]).apply(
                lambda g: pd.Series(dict(
                    lam_small_sigma=g[g.sigma == g.sigma.min()]["lambda_min"].median(),
                    lam_large_sigma=g[g.sigma == g.sigma.max()]["lambda_min"].median(),
                    lam_raw_small_sigma=g[g.sigma == g.sigma.min()]["lambda_min_raw"].median(),
                    lam_raw_large_sigma=g[g.sigma == g.sigma.max()]["lambda_min_raw"].median()))).reset_index()
            summ = summ.merge(lam, on=["case", "tree", "saddle"], how="left")
            # asymptotic slopes: per seed through the 3 smallest sigma (review of round 2, HIGH 2: the
            # global fits include pre-asymptotic large-sigma points); seeds whose first edge changes
            # between those sigma mix routes and are also reported separately
            psl, sl = per_seed_local(r, ["case", "tree", "saddle"], edge_col=f"edge_{f}")
            if len(psl):
                ape = r.groupby(["case", "tree", "saddle"])["A_pred_edges"].first()
                psl["A_pred_edge"] = [json.loads(ape[(c, t, sd)])[int(e)] if e >= 0 else np.nan
                                      for c, t, sd, e in zip(psl.case, psl.tree, psl.saddle, psl.edge_loc)]
                psl["f"] = f
                single = psl[psl.edge_loc >= 0].groupby(["case", "tree", "saddle"]).agg(
                    n_single_edge=("seed", "size"), A_loc_single_mean=("A_loc", "mean"),
                    A_loc_single_sem=("A_loc", "sem"), A_pred_edge_mean=("A_pred_edge", "mean")).reset_index()
                summ = summ.merge(sl, on=["case", "tree", "saddle"], how="left").merge(
                    single, on=["case", "tree", "saddle"], how="left")
                out_psl.append(psl)
            out_ps.append(ps)
            out_sum.append(summ)
    nres = df.groupby(["case", "tree"]).agg(n_runs=("seed", "size"),
                                            n_resolved=("resolved", "sum")).reset_index()
    summ = pd.concat(out_sum).merge(nres, on=["case", "tree"], how="left")
    pd.concat(out_ps).to_csv(RESULTS / f"{name}_v2_perseed.csv", index=False)
    if out_psl:
        pd.concat(out_psl).to_csv(RESULTS / f"{name}_v2_perseed_local.csv", index=False)
    summ.to_csv(RESULTS / f"{name}_v2_summary.csv", index=False)
    print(summ.round(4).to_string())
    # figure: per case, gap vs sigma (per seed, f = 0.03) and lambda_min vs sigma
    cases = list(dict.fromkeys(df["case"]))
    fig, axes = plt.subplots(2, len(cases), figsize=(4.2 * len(cases), 7.5), squeeze=False)
    for i, c in enumerate(cases):
        for tree, col in (("mps", "C3"), ("binary", "C0")):
            g = res[(res.case == c) & (res.tree == tree)]
            for (seed, sad), gg in g.groupby(["seed", "saddle"]):
                gg = gg.sort_values("sigma")
                axes[0, i].plot(gg["sigma"], gg["gap_0.05"], "-", color=col, alpha=0.5, lw=0.8,
                                marker="o" if sad == 0 else "x", ms=3)
            lam = g.groupby("sigma")["lambda_min"].median()
            axes[1, i].plot(lam.index, -lam.values, "o-", color=col, label=f"{tree}4")
        s = summ[(summ.case == c) & (summ.f == 0.05)]
        txt = "\n".join(f"{r.tree} saddle {r.saddle}: A={r.A_mean:.2f}+-{r.A_sem:.2f} (pred {r.A_pred:.2f}), "
                        f"p={r.p_mean:.2f}+-{r.p_sem:.2f}, log wins {r.log_wins}/{r.n_seeds}"
                        for r in s.itertuples())
        axes[0, i].set(xscale="log", yscale="log", xlabel="sigma", ylabel="gap (f = 0.05)", title=c)
        axes[0, i].text(0.02, 0.02, txt, transform=axes[0, i].transAxes, fontsize=5.5, va="bottom")
        axes[1, i].set(xscale="log", yscale="log", xlabel="sigma",
                       ylabel="-lambda_min at the plateau (gauge-projected)")
        axes[1, i].legend(fontsize=7)
    fig.suptitle(f"{name}: second escape per seed (o: major saddle, x: other), resolved runs only",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_v2.png", dpi=100)


if __name__ == "__main__":
    main()
