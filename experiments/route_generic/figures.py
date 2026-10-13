"""Draft figures of route_generic, from the committed CSVs only.

usage: uv run python experiments/route_generic/figures.py <results_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttn.plotting import INK, MUTED, palette, savefig, set_style  # noqa: E402

FAMILIES = [
    ("pattern", "Pattern target\n(control, inside the theory)"),
    ("cp03", "Random CP teacher,\noverlap 0.3"),
    ("cp06", "Random CP teacher,\noverlap 0.6"),
    ("ttn", "Random TTN teacher"),
    ("mnist", "MNIST 3 vs 5\n(16 pixels)"),
]
TREE_COLOR = {"mps": palette(2)[0], "binary": palette(2)[1]}
TREE_LABEL = {"mps": "chain (MPS) tree", "binary": "binary tree"}
FAM_COLORS = dict(zip([f for f, _ in FAMILIES], palette(5)))
RNG = np.random.default_rng(0)


def tree_kind(name: str) -> str:
    return "mps" if name.startswith("mps") else "binary"


def later(ex: pd.DataFrame) -> pd.DataFrame:
    return ex[(ex["fitted"]) & (ex["event"] > 0)].copy()


def strip_panel(ax, df, ycol, ref, ylabel, ylim):
    for kind, off in (("mps", -0.15), ("binary", 0.15)):
        sub = df[df["tree"].map(tree_kind) == kind]
        if not len(sub):
            continue
        x = sub["k_region"].to_numpy() + off + RNG.uniform(-0.08, 0.08, len(sub))
        y = sub[ycol].to_numpy()
        clipped = np.clip(y, *ylim)
        ax.scatter(x, clipped, s=14, color=TREE_COLOR[kind], alpha=0.7, linewidths=0, label=TREE_LABEL[kind])
        out = y != clipped
        if out.any():
            ax.scatter(x[out], clipped[out], s=22, marker="^" if ylim[1] in clipped[out] else "v",
                       facecolors="none", edgecolors=TREE_COLOR[kind], linewidths=0.8)
    ks = np.arange(1.5, 17)
    ax.plot(ks, ref(ks), ls="--", color=MUTED, lw=1.2, zorder=0)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)


def fig_vs_region(ex, out, ycol, ref, ylabel, ylim, title, ref_label, name):
    df = later(ex)
    fams = [f for f in FAMILIES if (df["target"] == f[0]).any()]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.6), sharey=True, squeeze=False)
    for ax, (fam, label) in zip(axes[0], fams):
        sub = df[df["target"] == fam]
        strip_panel(ax, sub, ycol, ref, ylabel if fam == fams[0][0] else "", ylim)
        ax.set_title(label, fontsize=11)
        kmax = int(sub["k_region"].max()) if len(sub) else 7
        ax.set_xlim(1.4, kmax + 0.6)
        ax.set_xticks(range(2, kmax + 1))
        ax.set_xlabel("cores that change at the event\n(k_region)")
    axes[0][0].plot([], [], ls="--", color=MUTED, label=ref_label)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    savefig(fig, out / name)
    plt.close(fig)


def fig_exponent_vs_depth(ex, out, title):
    df = later(ex).dropna(subset=["eff_depth", "p"])
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for fam, label in FAMILIES:
        sub = df[df["target"] == fam]
        if not len(sub):
            continue
        ax.scatter(np.clip(sub["eff_depth"], 0.5, 8), np.clip(sub["p"], -0.5, 5), s=16, alpha=0.7, linewidths=0,
                   color=FAM_COLORS[fam], label=label.replace("\n", " "))
    d = np.linspace(1.5, 8, 50)
    ax.plot(d, d - 2, ls="--", color=MUTED, lw=1.2, label="prediction: exponent = depth - 2")
    ax.set_xlim(0.5, 8)
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel("effective depth from the growth law (d_eff)")
    ax.set_ylabel("plateau duration exponent p\n(duration ~ sigma^-p)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    savefig(fig, out / "exponent_vs_depth")
    plt.close(fig)


def fig_examples(ev, ex, out, title):
    df = later(ex)
    fams = [f for f in FAMILIES if (df["target"] == f[0]).any()]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.4), squeeze=False)
    keys = ["target", "tree", "teacher_seed", "seed", "prior", "rises"]
    for ax, (fam, label) in zip(axes[0], fams):
        sub = df[df["target"] == fam]
        # the most common k_region among the fitted series; then the series with the median exponent
        k = int(sub["k_region"].mode().iloc[0])
        cand = sub[sub["k_region"] == k].sort_values("p")
        row = cand.iloc[len(cand) // 2]
        pts = ev.merge(pd.DataFrame([row[keys]]), on=keys).sort_values("sigma")
        pts = pts[(pts["duration"] > 0) & (~pts["floor_flag"].astype(bool))]
        color = TREE_COLOR[tree_kind(row["tree"])]
        ax.loglog(pts["sigma"], pts["duration"], "o-", color=color, ms=5, lw=1.5, label="measured")
        s = np.array([pts["sigma"].min(), pts["sigma"].max()])
        mid = np.exp(np.mean(np.log(pts["duration"])))
        smid = np.exp(np.mean(np.log(pts["sigma"])))
        ax.loglog(s, mid * (s / smid) ** (-(k - 2)), ls="--", color=MUTED, lw=1.2,
                  label=f"prediction from cores that change: slope -{k - 2}")
        ax.set_title(f"{label.replace(chr(10), ' ')}\n{row['tree']}, event {int(row['event'])}, k_region = {k}",
                     fontsize=10)
        ax.set_xlabel("init scale sigma")
        if fam == fams[0][0]:
            ax.set_ylabel("plateau duration before the event\n(gradient-flow time)")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig(fig, out / "examples_duration_vs_sigma")
    plt.close(fig)


def fig_agreement(sm, out, title):
    rows = sm[(sm["events"] == "later") & (sm["k_class"] == "all")].copy()
    order = {f: i for i, (f, _) in enumerate(FAMILIES)}
    rows["o"] = rows["target"].map(order)
    rows = rows.sort_values(["o", "tree"])
    labels = [f"{FAMILIES[o][1].split(chr(10))[0]} - {t}" for o, t in zip(rows["o"], rows["tree"])]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(rows) + 1.6))
    c = palette(2)
    ax.scatter(rows["frac_dd_within_0.3"], y + 0.12, s=50, color=c[0],
               label="depth within 0.3 of cores that change")
    ax.scatter(rows["frac_dp_within_0.3"], y - 0.12, s=50, marker="s", color=c[1],
               label="exponent within 0.3 of (cores that change - 2)")
    ax.axvline(0.7, ls="--", color=MUTED, lw=1)
    ax.text(0.71, y.max() + 0.6, "70% = 'holds' threshold", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}  (n = {n})" for lab, n in zip(labels, rows["n_series"])], fontsize=9)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, y.max() + 1.2)
    ax.set_xlabel("fraction of series (seed x event) that agree with the picture")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    savefig(fig, out / "agreement_by_target")
    plt.close(fig)


def main():
    res = Path(sys.argv[1])
    set_style()
    ex = pd.read_csv(res / "exponents.csv")
    ev = pd.read_csv(res / "events.csv")
    sm = pd.read_csv(res / "summary.csv")
    titles = {}
    tpath = res / "figure_titles.txt"
    if tpath.exists():
        for line in tpath.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                titles[k.strip()] = v.strip()
    fig_vs_region(ex, res, "eff_depth", lambda k: k, "effective depth from the growth law", (0.5, 9),
                  titles.get("depth", "Growth-law depth against the number of cores that change"),
                  "prediction: depth = cores that change", "depth_vs_region")
    fig_vs_region(ex, res, "p", lambda k: k - 2, "plateau duration exponent p\n(duration ~ sigma^-p)", (-0.5, 6),
                  titles.get("exponent", "Plateau duration exponent against the number of cores that change"),
                  "prediction: p = cores that change - 2", "exponent_vs_region")
    fig_exponent_vs_depth(ex, res, titles.get("exp_depth", "Duration exponent against growth-law depth"))
    fig_examples(ev, ex, res, titles.get("examples", "Plateau duration against init scale: one typical series per target"))
    fig_agreement(sm, res, titles.get("agreement", "Where the picture holds"))


if __name__ == "__main__":
    main()
