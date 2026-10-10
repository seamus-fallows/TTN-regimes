"""Data panels (b, c) of the teaser figure of the report's introduction; panel (a) is TikZ
(report/tikz/teaser.tex).

(b) Loss in linear time with the partial-fit levels: the MPS4 run of the saddle-to-saddle figure of
    Sec. 4 (experiment s2s_core, sweep main): odeco teacher with weights (3, 2, 1), d = 4, chi = 3,
    fan-in initialization, sigma = 0.01, seed 0. Read from results/teaser_run.csv, which
    --raw DIR rewrites from the raw run in DIR (the sweep folder, e.g. data/s2s_core/main).
(c) Measured plateau exponent against k - 2 for the two-component pattern teachers of the locality
    figure of Sec. 4 (d = chi = 2, weights (1, 0.5)): the rows with k = 3-5 of
    experiments/locality/results/twocomp_table.csv (committed; written by experiments/locality/make_figures.py).

    uv run python experiments/report_figures/make_teaser.py [--raw data/s2s_core/main]

writes experiments/report_figures/results/teaser.pdf and .png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ttn import plotting as tp  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RUN_NAME = "mps4_chi3_w3-2-1_fanin_s0.01_seed0"
RUN_CSV = RESULTS / "teaser_run.csv"
LOCALITY_TABLE = HERE.parent / "locality" / "results" / "twocomp_table.csv"
WEIGHTS = np.array([3.0, 2.0, 1.0])  # teacher of the run in panel (b)

# Tree family colours of the report figures (MPS blue, binary orange, HT green); the two other trees
# of the locality sweep share one muted colour here.
FAMILY_COLOR = {"mps": "#2a78d6", "binary": "#eb6834", "ht": "#1baf7a", "other": tp.MUTED}
FAMILY_NAME = {"mps": "MPS", "binary": "binary", "ht": "HT", "other": "other trees"}


def family(tree: str) -> str:
    for k in ("mps", "binary", "ht"):
        if tree.startswith(k):
            return k
    return "other"


def write_run_csv(raw: Path) -> None:
    """Loss on the log grid merged with the refined event windows, as in the Sec. 4 figure."""
    summ = json.loads((raw / f"{RUN_NAME}.json").read_text())
    d = np.load(raw / f"{RUN_NAME}.npz")
    ts, ls = [d["t"]], [d["loss"]]
    for r in range(len(summ["weights"])):
        if f"fine_t_{r}" in d.files:
            ts.append(d[f"fine_t_{r}"])
            ls.append(d[f"fine_loss_{r}"])
    t, loss = np.concatenate(ts), np.concatenate(ls)
    idx = np.argsort(t, kind="stable")
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": t[idx], "loss": loss[idx]}).to_csv(
        RUN_CSV, index=False, float_format="%.10g"
    )
    events = ", ".join(f"{x:.4g}" for x in summ["events"])
    print(f"wrote {RUN_CSV} ({len(idx)} rows; events at {events})")


def panel_loss(ax) -> None:
    run = pd.read_csv(RUN_CSV)
    levels = [0.5 * float(np.sum(WEIGHTS[k:] ** 2)) for k in range(len(WEIGHTS))]
    for k, lev in enumerate(levels):
        ax.axhline(lev, color=tp.MUTED, lw=0.7, ls=":", zorder=1)
        ax.text(
            6.0e5,
            lev + 0.15,
            ["nothing learned", "1 learned", "2 learned"][k],
            ha="right",
            va="bottom",
            fontsize=7,
            color=tp.INK_SECONDARY,
        )
    ax.plot(run["t"], run["loss"], color=tp.PALETTE[0], lw=1.2, zorder=2)
    ax.set_xlim(0, 6.2e5)
    ax.set_ylim(-0.3, 7.6)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_xlabel("time $t$")
    ax.set_ylabel("loss $L(t)$")
    ax.set_title(r"MPS4, $\sigma=0.01$", fontsize=8)


def panel_exponents(ax) -> None:
    t = pd.read_csv(LOCALITY_TABLE)
    base = t[(t["d"] == 2) & (t["chi"] == 2) & (t["w"] == "(1.0, 0.5)")]
    base = base[base["k_rank"].between(3, 5) & base["p_fit"].notna()]
    for (fam, start), g in base.groupby([base["tree"].map(family), "start"]):
        jit = 0.09 * (list(FAMILY_COLOR).index(fam) - 1.5) + (0.03 if start == "saddle" else -0.03)
        ax.errorbar(
            g["k_rank"] - 2 + jit,
            g["p_fit"],
            yerr=g["p_se"],
            fmt="o" if start == "small" else "s",
            ms=3,
            color=FAMILY_COLOR[fam],
            mfc=FAMILY_COLOR[fam] if start == "small" else "white",
            lw=0.8,
            capsize=0,
        )
    ax.plot([0.6, 3.4], [0.6, 3.4], "k--", lw=0.8, zorder=0)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(0.6, 5.3)  # headroom for the legend above the k - 2 = 1 points
    ax.set_xlabel("predicted exponent $k-2$")
    ax.set_ylabel("measured exponent")
    ax.set_title("two-component teachers, 9 trees", fontsize=8)
    handles = [
        Line2D([], [], color=c, marker="o", ls="none", ms=3.5, label=FAMILY_NAME[f])
        for f, c in FAMILY_COLOR.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            color=tp.INK_SECONDARY,
            marker="s",
            mfc="white",
            ls="none",
            ms=3.5,
            label="saddle start",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        handletextpad=0.2,
        borderaxespad=0.2,
        frameon=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="rewrite results/teaser_run.csv from this sweep folder",
    )
    args = ap.parse_args()
    if args.raw is not None:
        write_run_csv(args.raw)
    tp.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(0.56 * tp.TEXT_WIDTH, 1.95), layout="constrained")
    panel_loss(axes[0])
    panel_exponents(axes[1])
    for ax, letter in zip(axes, "bc"):
        ax.text(
            -0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=9
        )
    out = tp.savefig(fig, RESULTS / "teaser")
    print("wrote", ", ".join(str(p) for p in out))


if __name__ == "__main__":
    main()
