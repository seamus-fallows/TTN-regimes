"""Round 2, item 1: escape law from the first saddle per tree, same teacher (saddle-start runs).

Reads every saddle-start run (r2_saddle.yaml, the balanced N = 8 misaligned supplement
r2_saddle_b.yaml, the eps extension r2_saddle_ext.yaml and the misaligned8_20 supplement
r2_saddle_m20.yaml; all in <OUT_ROOT>/r2_saddle/).
Escape event (notes/diagnostics.md): first time some edge has s_2^e >= f s_2^{e,*}, f = 0.05
(sensitivity 0.03, 0.1), and at least 10x its starting value; the edge that fires is the realized edge.
misaligned8_20 (fix round 2): its escape ends at an intermediate saddle with s_2 ~ 3.6e-4 on every bond,
below f s_2^* (the f-rule then times the NEXT escape), so its event is max_e s_2^e >= S2_INT = 1.8e-4,
used where s_2(0) <= S2_INT / 10.
For every (teacher, tree), per seed (the same seed = the same noise direction at every eps):
- global fits over all eps of the log law t = A ln(1/eps) + B and of the power law t = C eps^-p, both
  by least squares on log t and scored by the rms of the log residuals (lawfit.py);
- asymptotic slopes (fix round 2, theory_teachers.md section 9): a line through the 5 smallest eps
  (MPS: eps <= 1e-4; binary / HT: eps <= 1e-3; A_loc, p_loc) and through the 3 smallest (A_loc3, p_loc3);
mean +- s.e.m. over seeds. Runs that did not escape before t_cap are counted and excluded.
Predictions: MPS (strict saddle) A = 1/gamma_e of the realized edge (gamma_e = -lambda_min of the
Hessian block of edge e at the exact saddle, results/r2_saddle_hessian.csv, r2_saddle_b_hessian.csv;
pred_A = 1/|lambda_min| is the fastest edge); binary / HT (non-strict; 3-core route) p = 1.
Usage: uv run python experiments/data_teacher/analyze_r2_saddle.py
Writes results/r2_saddle_runs_all.csv, r2_saddle_perseed.csv, r2_saddle_perseed_local.csv,
r2_saddle_fits.csv, r2_saddle_law.png, r2_saddle_collapse.png.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, RESULTS  # noqa: E402
from lawfit import per_seed, per_seed_local  # noqa: E402

HERE = Path(__file__).parent
TREES = ("mps", "binary", "ht")
COLORS = {"mps": "C3", "binary": "C0", "ht": "C2"}
S2_INT = 1.8e-4  # half the intermediate level of misaligned8_20
RETIMED = ("misaligned8_20",)


_TSP = {}


def teacher_s2(tname, n, tree):
    """Teacher's second singular value on every student edge (function-space for covariances)."""
    import yaml

    from r2_common import student, teacher_for
    from run_generic import dense_bond_spectra
    if not _TSP:
        for cfgname in ("r2_saddle.yaml", "r2_saddle_b.yaml", "r2_saddle_m20.yaml"):
            cfg = yaml.safe_load(open(HERE / "configs" / cfgname))
            for nn, block in cfg["teachers"].items():
                for tn, tc in block.items():
                    _TSP[tn] = (int(nn), tc)
    nn, tc = _TSP[tname]
    teacher, metrics = teacher_for(tc["kind"], nn, **tc.get("args", {}))
    spec = student(tree, nn)
    sp = dense_bond_spectra(teacher.dense(), spec.tree, metrics)
    return np.array([sp[v][1] if len(sp[v]) > 1 else 0.0 for v in spec.tree.edges])


def load_runs(fs=(0.03, 0.05, 0.1)):
    """Summaries plus escape times recomputed with the diagnostics.md onset rule: first time some edge
    has s_2^e >= f s_2^{e,*} (teacher value on that cut, edges with s_2^{e,*} > 1e-2) and at least 10x
    its value at the start (so that large-eps noise cannot fire it at t = 0); edge_f = the edge that
    fires. t_int: first time max_e s_2^e >= S2_INT (valid if s_2(0) <= S2_INT / 10)."""
    rows, cache = [], {}
    for f in sorted((OUT_ROOT / "r2_saddle").glob("*.npz")):
        d = np.load(f)
        r = json.loads(str(d["summary"]))
        key = (r["teacher"], r["tree"])
        if key not in cache:
            cache[key] = teacher_s2(r["teacher"], r["n"], r["tree"])
        s2t = cache[key]
        t, sp = d["t"], d["spectra"][:, :, 1]
        for fv in fs:
            hits = []
            for j in np.nonzero(s2t > 1e-2)[0]:
                thr = max(fv * s2t[j], 10 * sp[0, j])
                h = np.nonzero(sp[:, j] >= thr)[0]
                if len(h):
                    hits.append((t[h[0]], j))
            r[f"t_esc_{fv}"], r[f"edge_{fv}"] = min(hits) if hits else (np.nan, -1)
        m = sp.max(axis=1)
        h = np.nonzero(m >= S2_INT)[0]
        r["t_int"] = t[h[0]] if len(h) and m[0] <= S2_INT / 10 else np.nan
        r["edge_int"] = int(np.argmax(sp[h[0]])) if len(h) else -1
        rows.append(r)
    return pd.DataFrame(rows)


def fit_block(df, t_col, edge_col, label):
    """Global and asymptotic per-seed fits of one event column; returns (per_seed, local, summary)."""
    ok = df[np.isfinite(df[t_col])].assign(gap=lambda x: x[t_col], edge=lambda x: x[edge_col])
    ps, sf = per_seed(ok, ["teacher", "n", "tree"], min_points=4)
    pls = []
    for tree_set, n_small, tag in ((("mps",), 5, "loc"), (("binary", "ht"), 5, "loc"), (TREES, 3, "loc3")):
        g = ok[ok.tree.isin(tree_set)]
        psl, _ = per_seed_local(g, ["teacher", "n", "tree"], n_small=n_small, edge_col="edge")
        if len(psl):
            pls.append(psl.assign(fit=tag))
    psl = pd.concat(pls)
    agg = psl.groupby(["teacher", "n", "tree", "fit"]).agg(
        n_seeds=("seed", "size"), eps_max=("sigma_loc_max", "max"), A_mean=("A_loc", "mean"),
        A_sem=("A_loc", "sem"), p_mean=("p_loc", "mean"), p_sem=("p_loc", "sem"),
        n_single_edge=("edge_loc", lambda e: int((e >= 0).sum()))).reset_index()
    wide = agg.pivot_table(index=["teacher", "n", "tree"], columns="fit",
                           values=["A_mean", "A_sem", "p_mean", "p_sem", "eps_max", "n_single_edge"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    sf = sf.merge(wide.reset_index(), on=["teacher", "n", "tree"], how="left")
    sf["event"] = label
    return ok, ps.assign(event=label), psl.assign(event=label), sf


def main():
    runs = load_runs()
    runs.to_csv(RESULTS / "r2_saddle_runs_all.csv", index=False)
    hes = pd.concat([pd.read_csv(RESULTS / f) for f in ("r2_saddle_hessian.csv", "r2_saddle_b_hessian.csv")
                     if (RESULTS / f).exists()]).drop_duplicates(["teacher", "tree"])
    gam = {(r.teacher, r.tree): np.array(json.loads(r.edge_gammas)) for r in hes.itertuples()}
    runs["escaped"] = np.isfinite(runs["t_esc_0.05"])
    rest = runs[~runs.teacher.isin(RETIMED)]
    sums, pss, psls = [], [], []
    for fv in (0.03, 0.05, 0.1):
        okf, ps_, psl_, sf = fit_block(rest, f"t_esc_{fv}", f"edge_{fv}", f"onset f={fv}")
        sf["f"] = fv
        sums.append(sf), pss.append(ps_), psls.append(psl_)
        if fv == 0.05:
            ok = okf
    ok_i, ps_, psl_, sf = fit_block(runs[runs.teacher.isin(RETIMED)], "t_int", "edge_int",
                                    f"intermediate level s2 >= {S2_INT}")
    sf["f"] = np.nan
    sums.append(sf), pss.append(ps_), psls.append(psl_)
    ok = pd.concat([ok, ok_i])
    ps, psl, summ = pd.concat(pss), pd.concat(psls), pd.concat(sums)
    # predictions: fastest edge (1/|lambda_min|) and the realized edge of each seed (5 smallest eps)
    summ = summ.merge(hes[["teacher", "tree", "lambda_min", "l_saddle", "n_negative"]], on=["teacher", "tree"],
                      how="left")
    summ["pred_A"] = np.where(summ["lambda_min"] < -1e-8, 1 / summ["lambda_min"].abs(), np.nan)
    summ["pred_p"] = np.where(summ["tree"] == "mps", np.nan, 1.0)
    psl["pred_A_edge"] = [1 / gam[(t, tr)][int(e)] if tr == "mps" and e >= 0 else np.nan
                          for t, tr, e in zip(psl.teacher, psl.tree, psl.edge_loc)]
    pe = psl[psl.fit == "loc"].groupby(["teacher", "tree", "event"]).agg(
        pred_A_edge=("pred_A_edge", "mean"),
        edges_loc=("edge_loc", lambda e: " ".join(str(int(x)) for x in sorted(e)))).reset_index()
    summ = summ.merge(pe, on=["teacher", "tree", "event"], how="left")
    summ["dev_loc"] = summ["A_mean_loc"] / summ["pred_A_edge"] - 1
    cnt = runs.groupby(["teacher", "tree"]).agg(n_runs=("seed", "size"), n_escaped=("escaped", "sum"),
                                                eps_min=("sigma", "min"), eps_max=("sigma", "max")).reset_index()
    summ = summ.merge(cnt, on=["teacher", "tree"], how="left")
    ps.to_csv(RESULTS / "r2_saddle_perseed.csv", index=False)
    psl.to_csv(RESULTS / "r2_saddle_perseed_local.csv", index=False)
    summ.to_csv(RESULTS / "r2_saddle_fits.csv", index=False)
    main_rows = summ[(summ.f == 0.05) | summ.f.isna()]
    cols = ["teacher", "tree", "event", "n_seeds", "A_mean", "A_sem", "A_mean_loc", "A_sem_loc", "A_mean_loc3",
            "pred_A", "pred_A_edge", "dev_loc", "edges_loc", "p_mean", "p_mean_loc", "p_sem_loc", "p_mean_loc3",
            "log_wins"]
    print(main_rows[cols].round(4).to_string())
    print(summ[summ.tree == "mps"].pivot_table(index="teacher", columns="f", values="A_mean_loc").round(3).to_string())
    teachers = list(dict.fromkeys(runs.sort_values("n")["teacher"]))
    ncol = 7
    nrow = int(np.ceil(len(teachers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
    for i, tname in enumerate(teachers):
        ax = axes[i // ncol, i % ncol]
        for tree in TREES:
            g = ok[(ok.teacher == tname) & (ok.tree == tree)]
            if not len(g):
                continue
            m = g.groupby("sigma")["gap"].agg(["mean", "sem"])
            s = main_rows[(main_rows.teacher == tname) & (main_rows.tree == tree)]
            if len(s):
                s = s.iloc[0]
                lab = (f"{tree}: A_loc={s.A_mean_loc:.2f}+-{s.A_sem_loc:.2f} (pred {s.pred_A_edge:.2f})"
                       if tree == "mps" else f"{tree}: p_loc={s.p_mean_loc:.3f}+-{s.p_sem_loc:.3f}")
            else:
                lab = tree
            ax.errorbar(m.index, m["mean"], m["sem"], fmt="o-", ms=3, lw=0.8, color=COLORS[tree], label=lab)
            if tree == "mps" and len(s) and np.isfinite(s.pred_A_edge):
                xs = np.array(sorted(m.index))
                y0 = m["mean"].loc[xs[0]]
                ax.plot(xs, y0 + s.pred_A_edge * (np.log(1 / xs) - np.log(1 / xs[0])), "k--", lw=0.8)
        ax.set(xscale="log", yscale="log", title=tname, xlabel="eps (noise at the saddle)",
               ylabel="escape time")
        ax.legend(fontsize=5.5, loc="upper right")
    for j in range(len(teachers), nrow * ncol):
        axes[j // ncol, j % ncol].set_visible(False)
    fig.suptitle("Escape from the exact balanced rank-1 saddle + noise eps (library LSODA; mean +- s.e.m. over "
                 "5 seeds). Dashed: predicted MPS slope 1/gamma of the realized edge, through the smallest eps. "
                 "Labels: asymptotic per-seed slopes (5 smallest eps). misaligned8_20: intermediate-level event.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_saddle_law.png", dpi=100)
    # collapse figure: MPS gamma t_esc vs ln(1/eps), gamma of the realized edge (linear axes: a log law is
    # a line of slope 1 after this rescaling), both relative to the smallest eps; binary / HT eps * t_esc
    # relative to its value at the smallest eps (flat for p = 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    pa = main_rows.set_index(["teacher", "tree"])["pred_A_edge"]
    for i, tname in enumerate(teachers):
        for j, tree in enumerate(TREES):
            g = ok[(ok.teacher == tname) & (ok.tree == tree)]
            if not len(g):
                continue
            m = g.groupby("sigma")["gap"].mean()
            if tree == "mps":
                if not np.isfinite(pa.get((tname, tree), np.nan)):
                    continue
                x, y = np.log(1 / m.index.values), m.values / pa[(tname, tree)]
                k = np.argmax(x)
                axes[0].plot(x - x[k], y - y[k], "o-", ms=3, label=tname)
            else:
                z = m.index.values * m.values
                axes[j].plot(m.index, z / z[np.argmin(m.index.values)], "o-", ms=3, label=tname)
    x = np.linspace(-np.log(ok.sigma.max() / ok.sigma.min()), 0, 10)
    axes[0].plot(x, x, "k--", lw=1, label="slope 1 (prediction)")
    axes[0].set(xlabel="ln(1/eps) - ln(1/eps_min)", ylabel="gamma_e (t_esc - t_esc(eps_min))",
                title="MPS students: log law with slope 1/gamma (realized edge)")
    for j, tree in ((1, "binary"), (2, "ht")):
        axes[j].set(xscale="log", xlabel="eps", ylabel="eps t_esc / (eps t_esc at eps_min)", ylim=(0, 2),
                    title=f"{tree} students: flat = sigma^-1 (k = 3)")
    for ax in axes:
        ax.legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "r2_saddle_collapse.png", dpi=100)


if __name__ == "__main__":
    main()
