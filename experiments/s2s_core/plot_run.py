"""Figure (b): one representative saddle-to-saddle run: loss (log and linear time), overlaps, bond
spectra on every edge, effective bond ranks.

    uv run python experiments/s2s_core/plot_run.py <run name> [--raw DIR] [--out DIR]
e.g. mps4_chi3_w3-2-1_fanin_s0.01_seed0 (a run of configs/main.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import s2s  # noqa: E402

from ttn import diagnostics as dg  # noqa: E402
from ttn import plotting as tp  # noqa: E402

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--raw", default="/root/outputs/s2s_core/main")
    ap.add_argument("--out", required=True, help="output directory (e.g. experiments/s2s_core/results)")
    args = ap.parse_args()
    raw = Path(args.raw)
    summ = json.loads((raw / f"{args.name}.json").read_text())
    d = np.load(raw / f"{args.name}.npz")
    spec, teacher = s2s.setup(summ["tree"], summ["N"], summ["d"], summ["chi"], summ["weights"], summ["teacher_seed"])
    weights = np.asarray(summ["weights"])
    R = len(weights)

    # Coarse log grid plus the refined event windows, merged in time.
    t, loss, a = d["t"], d["loss"], d["overlaps"]
    ts, ls, as_ = [t], [loss], [a]
    for r in range(R):
        if f"fine_t_{r}" in d:
            ts.append(d[f"fine_t_{r}"])
            ls.append(d[f"fine_loss_{r}"])
            as_.append(d[f"fine_a_{r}"])
    tm = np.concatenate(ts)
    idx = np.argsort(tm, kind="stable")
    tm, lm, am = tm[idx], np.concatenate(ls)[idx], np.concatenate(as_)[idx]

    tp.set_style()
    fig = plt.figure(figsize=(tp.TEXT_WIDTH, 5.0), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    ax = top.subplots(1, 4)
    colors = tp.palette(R)
    ev = np.array([np.nan if x is None else x for x in summ["events"]])
    levels = sorted({0.5 * float(np.sum(weights[[q for q in range(R) if q not in summ["order"][:k]]] ** 2))
                     for k in range(R + 1)})
    tp.plot_loss(ax[0], tm, lm)
    tp.plot_loss_linear(ax[1], tm, lm)
    for lev in levels:
        for axx in ax[:2]:
            if lev > 0:
                axx.axhline(lev, color=tp.MUTED, lw=0.7, ls=":")
    ax[1].set_xlim(0, 1.15 * np.nanmax(ev))
    ax[0].set_title("loss, log time")
    ax[1].set_title("loss, linear time")
    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    tp.plot_overlaps(ax[2], tm, am, labels=[f"$a_{r + 1}$ ($s={w:g}$)" for r, w in enumerate(weights)])
    ax[2].legend(fontsize=6.5, loc="center left")
    ax[2].set_title("overlaps")
    # Effective bond rank r_e(t) = #{s_k > 0.1 min teacher value} on every edge (diagnostics.md).
    sp = d["spectra"]
    ranks = dg.effective_rank(np.nan_to_num(sp), 0.1 * weights.min())
    for j, v in enumerate(spec.tree.edges):
        ax[3].plot(t, ranks[:, j] + 0.04 * (j - len(spec.tree.edges) / 2), lw=1.0, color=tp.ordinal_colors(len(spec.tree.edges))[j])
    ax[3].set_xscale("log")
    ax[3].set_yticks(range(R + 1))
    ax[3].set_title("effective bond rank\n(one line per edge)")
    ax[3].set_xlabel("$t$")
    for x in ev[np.isfinite(ev)]:
        for axx in (ax[0], ax[2], ax[3]):
            axx.axvline(x, color=tp.MUTED, lw=0.6, ls="--", zorder=0)
    edges = list(spec.tree.edges)
    tp.plot_bond_spectra(bottom, t, {v: sp[:, j, :] for j, v in enumerate(edges)}, tree=spec.tree,
                         teacher=weights, ncols=min(len(edges), 4))
    fig.suptitle(f"{summ['tree']}{summ['N']} (K={summ['K']}), $\\chi$={summ['chi']}, "
                 f"s=({', '.join(f'{w:g}' for w in weights)}), $\\sigma$={summ['sigma']:g}, seed {summ['seed']}; "
                 f"order {[r + 1 for r in summ['order']]}, min $\\rho$ = {min(summ['separation_ratios']):.3g}",
                 fontsize=8)
    tp.savefig(fig, Path(args.out) / f"fig_b_run_{args.name}")


if __name__ == "__main__":
    main()
