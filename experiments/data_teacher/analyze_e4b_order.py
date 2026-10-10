"""Step 4(b): order of bond-rank growth and the law of the second escape for random MPS teachers on
an MPS-4 student (dense library runs of configs/e4b_ttn.yaml).

Per teacher: the rank-1 critical points (constrained.py); at each, the site vectors w_i,
lambda = <T, (x) w_i>, M_e = T contracted with w_i off edge e, and theory_teachers.md R5/R6:
gamma_e = lambda^((N-2)/N) s2(M_e), A_e = (1 - s2(M_e)/lambda)/gamma_e, e* = argmax_e s2(M_e).
The naive rule picks argmax_e s_2^e(T) (teacher's second Schmidt value on the cut).
Per run: t_1 = first time L < 0.9 L(0); the plateau after it (minimum |dL/dt| with L > 1e-3 L(0));
t_2 = first time after it that max_e s_2^e exceeds 1e-2 (theory_teachers.md 3.3); the edge where
that happens. Fit Dt_2 = t_2 - t_1 vs log(1/sigma) per teacher and compare the slope with A_e*.
Usage: uv run python experiments/data_teacher/analyze_e4b_order.py
Writes results/e4b_order.csv, results/e4b_order_fits.csv, results/e4b_law.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, OUT_ROOT  # noqa: E402
from constrained import constrained_minima, metric_loss_of_dense, ranked_spec  # noqa: E402
from run_generic import dense_bond_spectra, expand  # noqa: E402
from setups import build  # noqa: E402

from ttn.dynamics import Trajectory  # noqa: E402

HERE = Path(__file__).parent
NAME = "e4b_ttn"


def rank1_theory(T, spec):
    out = []
    spec1 = ranked_spec(spec, {v: 1 for v in spec.tree.edges})
    n = T.ndim
    for m in constrained_minima(spec1, metric_loss_of_dense(T, None), n_starts=10):
        X = m["W"]
        w = [torch.linalg.svd(torch.movedim(X, i, 0).reshape(T.shape[i], -1))[0][:, 0]
             for i in range(n)]
        lam = float(torch.einsum("abcd,a,b,c,d->", T, *w))
        if lam < 0:
            w[0], lam = -w[0], -lam
        s2, a = [], []
        for i in range(n - 1):  # MPS edge between sites i and i+1
            M = T
            for k in sorted([k for k in range(n) if k not in (i, i + 1)], reverse=True):
                M = torch.tensordot(M, w[k], dims=([k], [0]))
            v = float(torch.linalg.svdvals(M)[1])
            s2.append(v)
            a.append((1 - v / lam) / (lam ** ((n - 2) / n) * v))
        e = int(np.argmax(s2))
        out.append(dict(loss=m["loss"], lam=lam, s2_M=s2, A=a, edge_star=e, A_star=a[e],
                        count=m["count"]))
    return out


def main():
    cfg = yaml.safe_load(open(HERE / "configs" / f"{NAME}.yaml"))
    rows, theory = [], []
    for case in expand(cfg):
        spec, teacher = build(dict(case, tree="mps"), 0)[:2]
        T = teacher.dense()
        tsp = dense_bond_spectra(T, spec.tree)
        naive = int(np.argmax([tsp[v][1] for v in spec.tree.edges]))
        th = rank1_theory(T, spec)
        for k, x in enumerate(th):
            theory.append(dict(teacher=case["tag"], saddle=k, naive_edge=naive,
                               teacher_s2=[round(float(tsp[v][1]), 4) for v in spec.tree.edges],
                               **{kk: vv for kk, vv in x.items()}))
        for sigma in cfg["sigmas"]["mps"]:
            for seed in range(*cfg["seeds"]):
                p = OUT_ROOT / f"{NAME}_dense" / f"{case['tag']}_mps_s{sigma}_seed{seed}.npz"
                if not p.exists():
                    continue
                tr = Trajectory.load(p)
                t, loss = tr.data["t"], tr.data["loss"]
                sp = np.nan_to_num(tr.data["spectra"])
                k1 = int(np.argmax(loss < 0.9 * loss[0]))
                win = np.nonzero((np.arange(len(t)) > k1) & (loss > 1e-3 * loss[0]))[0]
                speed = np.abs(np.gradient(loss, t))
                kp = win[int(np.argmin(speed[win]))] if len(win) else k1
                s2 = sp[:, :, 1]
                after = np.nonzero((np.arange(len(t)) > k1) & (s2.max(axis=1) > 1e-2))[0]
                k2 = after[0] if len(after) else len(t) - 1
                # match the rank-1 saddle by plateau level when several exist
                lev = float(loss[kp])
                j = int(np.argmin([abs(x["loss"] - lev) for x in th]))
                rows.append(dict(teacher=case["tag"], sigma=sigma, seed=seed, t1=t[k1], t2=t[k2],
                                 dt2=t[k2] - t[k1], first_edge=int(np.argmax(s2[k2])),
                                 s2_at_t1=float(s2[k1].max()), plateau_level=lev,
                                 saddle=j, edge_star=th[j]["edge_star"], naive_edge=naive,
                                 A_star=th[j]["A_star"]))
    df = pd.DataFrame(rows)
    df["theory_ok"] = df["first_edge"] == df["edge_star"]
    df["naive_ok"] = df["first_edge"] == df["naive_edge"]
    df.to_csv(RESULTS / "e4b_order.csv", index=False)
    pd.DataFrame(theory).to_csv(RESULTS / "e4b_theory.csv", index=False)
    fits = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (tag, g) in enumerate(df.groupby("teacher")):
        m = g.groupby("sigma")["dt2"].agg(["mean", "sem"])
        x = -np.log(m.index.values)
        a = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(a, m["mean"].values, rcond=None)
        res = m["mean"].values - a @ coef
        se = np.sqrt(res @ res / max(len(x) - 2, 1) * np.linalg.inv(a.T @ a)[0, 0])
        lg = np.polyfit(np.log(m.index.values), np.log(m["mean"].values), 1)
        fits.append(dict(teacher=tag, A_fit=coef[0], A_se=se, A_pred=g["A_star"].median(),
                         power_exponent=-lg[0], theory_edge_ok=g["theory_ok"].mean(),
                         naive_edge_ok=g["naive_ok"].mean(), n=len(g)))
        ax.errorbar(x, m["mean"], m["sem"], fmt="o", color=f"C{i}",
                    label=f"{tag}: A = {coef[0]:.1f} +- {se:.1f} (pred {g['A_star'].median():.1f})")
        x0, y0 = x.mean(), m["mean"].mean()
        ax.plot(x, y0 + g["A_star"].median() * (x - x0), "--", color=f"C{i}", lw=0.8)
    ax.set(xlabel="log(1/sigma)", ylabel="Dt_2 = t_2 - t_1 (mean +- s.e., 6 seeds)",
           title="random MPS teachers on MPS-4: second escape (dashed: predicted slope)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "e4b_law.png", dpi=110)
    f = pd.DataFrame(fits)
    f.to_csv(RESULTS / "e4b_order_fits.csv", index=False)
    print(pd.DataFrame(theory).to_string())
    print(f.to_string())


if __name__ == "__main__":
    main()
