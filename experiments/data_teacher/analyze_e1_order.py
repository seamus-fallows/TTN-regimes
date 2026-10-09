"""Analysis of the step-1 order sweep (configs/e1_order.yaml) from the saved runs.

Outputs (results/): e1_order_table.csv (order and swap statistics per setting), e1_paired.csv
(paired Sigma vs whitened comparison), e1_plateaus.csv (plateau levels vs prediction), figures
e1_order_probs.png, e1_paired_times.png, e1_examples.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, plateau_level  # noqa: E402

HERE = Path(__file__).parent
CFG = yaml.safe_load(open(HERE / "configs/e1_order.yaml"))
RUN_DIR = OUT_ROOT / CFG["name"]


def load(tree, sigma, teacher, variant, seed):
    return np.load(RUN_DIR / f"{tree}_s{sigma}_{teacher}_{variant}_seed{seed}.npz",
                   allow_pickle=True)


def binom(p, n):
    return np.sqrt(max(p * (1 - p), 1.0 / n) / n)


def main():
    rows = pd.read_pickle(RUN_DIR / "summary_rows.pkl")
    df = pd.DataFrame(rows)
    df["learned_all"] = df["loss_end"] < 1e-5 * df["loss0"]

    # 1. Order and swap statistics.
    tab = []
    for key, g in df.groupby(["tree", "teacher", "variant", "sigma"]):
        n = len(g)
        p1 = (g["first"] == 1).mean()
        ps = (g["n_unlearn"] > 0).mean()
        swap_given = {}
        for f in (0, 1):
            gg = g[g["first"] == f]
            swap_given[f] = (gg["n_unlearn"] > 0).mean() if len(gg) else np.nan
        tab.append(dict(zip(["tree", "teacher", "variant", "sigma"], key), n=n,
                        P_first_highvar=p1, se=binom(p1, n), P_swap=ps,
                        P_swap_given_first0=swap_given[0], P_swap_given_first1=swap_given[1],
                        all_learned=g["learned_all"].mean(), max_q_drift=g["q_drift"].max()))
    tab = pd.DataFrame(tab)
    tab.to_csv(HERE / "results/e1_order_table.csv", index=False)
    print(tab.to_string())

    # 2. Paired comparison: Sigma vs whitened with Lambda s (race) and sqrt(Lambda) s.
    paired = []
    piv = df.set_index(["tree", "teacher", "sigma", "seed", "variant"])
    for (tree, teacher, sigma, seed), _ in df.groupby(["tree", "teacher", "sigma", "seed"]):
        a = piv.loc[(tree, teacher, sigma, seed, "sigma")]
        for other in ("white_Ls", "white_sqrtLs", "white_sqrtLs_mapped"):
            b = piv.loc[(tree, teacher, sigma, seed, other)]
            ta = np.nanmin([a["t_half_0"], a["t_half_1"]])
            tb = np.nanmin([b["t_half_0"], b["t_half_1"]])
            # second learning event = the later first-crossing
            sa = np.nanmax([a["t_half_0"], a["t_half_1"]])
            sb = np.nanmax([b["t_half_0"], b["t_half_1"]])
            paired.append(dict(tree=tree, teacher=teacher, sigma=sigma, seed=seed, other=other,
                               same_first=a["first"] == b["first"], same_swaps=(
                                   (a["n_unlearn"] > 0) == (b["n_unlearn"] > 0)),
                               t1_sigma=ta, t1_other=tb, rel_dt1=(tb - ta) / ta,
                               t2_sigma=sa, t2_other=sb, rel_dt2=(sb - sa) / sa))
    paired = pd.DataFrame(paired)
    paired.to_csv(HERE / "results/e1_paired.csv", index=False)
    agg = paired.groupby(["tree", "teacher", "other", "sigma"]).agg(
        n=("seed", "size"), same_first=("same_first", "mean"), same_swaps=("same_swaps", "mean"),
        rel_dt1_mean=("rel_dt1", "mean"), rel_dt1_absmax=("rel_dt1", lambda x: x.abs().max()),
        rel_dt2_mean=("rel_dt2", "mean"), rel_dt2_sem=("rel_dt2", "sem"))
    agg.to_csv(HERE / "results/e1_paired_agg.csv")
    print(agg.to_string())

    # 3. Plateau levels after the first learning event vs 1/2 Lambda_r s_r^2 of the other comp.
    plat = []
    for _, r in df[df["variant"] == "sigma"].iterrows():
        d = load(r["tree"], r["sigma"], r["teacher"], "sigma", r["seed"])
        ev = [e for e in r["events"]]
        if len(ev) < 2:
            continue
        lam = np.array([r["Lambda_0"], r["Lambda_1"]])
        s = np.array([r["s_0"], r["s_1"]])
        for (ta, ca, sa), (tb, _, _) in zip(ev[:-1], ev[1:]):
            # learned set after event a: components whose overlap is > 1/2 just after ta
            k = np.searchsorted(d["t"], ta) + 1
            learned = d["overlaps"][min(k, len(d["t"]) - 1)] > 0.5
            pred = 0.5 * np.sum((lam * s**2)[~learned])
            meas = plateau_level(d["t"], d["loss"], ta, tb)
            plat.append(dict(tree=r["tree"], teacher=r["teacher"], sigma=r["sigma"],
                             seed=r["seed"], learned="".join(str(int(x)) for x in learned),
                             measured=meas, predicted=pred, rel_err=(meas - pred) / pred,
                             separation=(tb - ta)))
    plat = pd.DataFrame(plat)
    plat.to_csv(HERE / "results/e1_plateaus.csv", index=False)
    print(plat.groupby(["tree", "teacher", "sigma", "learned"]).agg(
        n=("seed", "size"), measured=("measured", "mean"), predicted=("predicted", "mean"),
        rel_err_med=("rel_err", "median")).to_string())

    plot_probs(tab)
    plot_paired(paired)
    plot_examples(df)


def plot_probs(tab):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    styles = {"sigma": ("C3", "o-"), "white_Ls": ("C0", "s--"), "white_sqrtLs": ("C2", "^:"),
              "white_sqrtLs_mapped": ("C4", "v-.")}
    labels = {"sigma": "(Sigma, s)", "white_Ls": "(I, Lambda s)", "white_sqrtLs": "(I, sqrt(Lambda) s)",
              "white_sqrtLs_mapped": "(I, sqrt(Lambda) s), mapped init"}
    for i, tree in enumerate(CFG["trees"]):
        for j, teacher in enumerate(CFG["teachers"]):
            ax = axes[i, j]
            for v in CFG["variants"]:
                g = tab[(tab.tree == tree) & (tab.teacher == teacher) & (tab.variant == v)]
                c, st = styles[v]
                ax.errorbar(g["sigma"], g["P_first_highvar"], g["se"], fmt=st, color=c,
                            label=f"first = comp 2, {labels[v]}", capsize=2)
                ax.plot(g["sigma"], g["P_swap"], st[0], color=c, mfc="none", ms=9, ls="none",
                        label=f"P(swap), {labels[v]}")
            ax.set_xscale("log")
            ax.set_title(f"{tree}-4, {teacher}", fontsize=9)
            ax.set_xlabel("sigma")
            ax.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=6)
    axes[0, 0].set_ylabel("probability (40 seeds)")
    axes[1, 0].set_ylabel("probability (40 seeds)")
    fig.suptitle("Comp 2 = smaller weight on the high-variance eigenvector. Filled: P(comp 2 learned "
                 "first); open: P(a learned component is later unlearned = swap)", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "results/e1_order_probs.png", dpi=110)


def plot_paired(paired):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in [(axes[0], "t1", "first learning event"),
                           (axes[1], "t2", "second component's first learning")]:
        for tree, mk in [("binary", "o"), ("mps", "s")]:
            for other, c in [("white_Ls", "C0"), ("white_sqrtLs_mapped", "C4")]:
                g = paired[(paired.tree == tree) & (paired.other == other)]
                ax.scatter(g[f"{col}_sigma"], g[f"{col}_other"], s=8, marker=mk, color=c,
                           alpha=0.6, label=f"{tree}: vs {other}")
        lim = [paired[[f"{col}_sigma", f"{col}_other"]].min().min() * 0.8,
               paired[[f"{col}_sigma", f"{col}_other"]].max().max() * 1.2]
        ax.plot(lim, lim, "k-", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("time under (Sigma, s)")
        ax.set_ylabel("time in paired whitened run (same init)")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "results/e1_paired_times.png", dpi=110)


def plot_examples(df):
    """Linear-time trajectories: a binary-4 swap (T2) and an MPS-4 run, Sigma vs paired runs."""
    picks = []
    g = df[(df.tree == "binary") & (df.teacher == "T2_race_vs_greedy") & (df.variant == "sigma")
           & (df.n_unlearn > 0)]
    if len(g):
        r = g.sort_values("sigma").iloc[0]
        picks.append(("binary", r["sigma"], "T2_race_vs_greedy", r["seed"]))
    g = df[(df.tree == "mps") & (df.teacher == "T2_race_vs_greedy") & (df.variant == "sigma")
           & (df["first"] == 1)]
    if len(g):
        r = g.sort_values("sigma").iloc[0]
        picks.append(("mps", r["sigma"], "T2_race_vs_greedy", r["seed"]))
    fig, axes = plt.subplots(len(picks), 4, figsize=(18, 3.6 * len(picks)), squeeze=False)
    for i, (tree, sigma, teacher, seed) in enumerate(picks):
        for j, v in enumerate(CFG["variants"]):
            d = load(tree, sigma, teacher, v, seed)
            summ = d["summary"].item()
            t = d["t"]
            ts = [e[0] for e in summ["events"]]
            m = (t > 0.7 * min(ts)) & (t < 1.3 * max(ts))
            ax = axes[i, j]
            ax.plot(t[m], d["overlaps"][m, 0], label="a_1 (s=1, low variance)")
            ax.plot(t[m], d["overlaps"][m, 1], label="a_2 (s=0.6, high variance)")
            ax.plot(t[m], d["loss"][m] / d["loss"][0], "k:", label="L / L(0)")
            if v == "sigma":
                lam = np.array([summ["Lambda_0"], summ["Lambda_1"]])
                s = np.array([summ["s_0"], summ["s_1"]])
                lv = 0.5 * lam * s**2 / d["loss"][0]
                for k, ls in zip((0, 1), ("--", "-.")):
                    ax.axhline(lv[k], color="gray", ls=ls, lw=0.8)
            ax.set_title(f"{tree}-4 sigma={sigma} seed {seed}: {v}", fontsize=9)
            ax.set_xlabel("t (linear)")
        axes[i, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "results/e1_examples.png", dpi=110)


if __name__ == "__main__":
    main()
