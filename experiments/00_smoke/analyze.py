"""Figures and tables of the smoke experiment (README.md), from the outputs of ttn.runner.

    uv run python experiments/00_smoke/analyze.py [--data data/00_smoke]

Reads <data>/base/smoke.npz (the representative run) and <data>/grid_sigma/ (the sigma sweep:
summary.csv and the trajectories). Writes to experiments/00_smoke/results/: loss, overlaps,
bond_spectra, escape_time and separation (.pdf and .png), escape_times.csv (per sigma), seeds.csv
(per seed: fits across sigma) and runs.csv (per run); prints the numbers quoted in the README.

Statistics: the init stream depends on the seed only, so sigma rescales one fixed init direction
per seed, and the 40 runs are 8 independent samples (seeds) at 5 scales. Every mean +- SE pooled
over sigma is over the 8 seeds of a per-seed quantity, never over the 40 runs; exponents are fitted
per seed. Events are those of `ttn.runner.learning_events`, on the log rows and the fine rows
together (resolved in linear time).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ttn import plotting as tp
from ttn.config import RunConfig
from ttn.diagnostics import find_plateaus, plateau_scaling, separation_ratios
from ttn.dynamics import merged_rows
from ttn.runner import learning_events, load_run, make_spec
from ttn.theory import escape_time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"


def plateau_levels(weights) -> np.ndarray:
    """L_k = 1/2 sum_{r > k} s_r^2 (largest weights learned first), k = 0..R."""
    w = np.sort(np.asarray(weights))[::-1]
    return np.array([0.5 * np.sum(w[k:] ** 2) for k in range(len(w) + 1)])


def base_figures(base_dir: Path) -> None:
    traj = load_run(base_dir, "smoke")
    cfg = RunConfig.from_dict(traj.meta["config"])
    weights = cfg.teacher.weights
    rows = merged_rows(traj.data)
    t, loss = rows["t"], rows["loss"]
    levels = plateau_levels(weights)

    # Log-log (convergence) and linear (the staircase; plateaus are judged in linear time,
    # notes/diagnostics.md) with the saddle levels L_k and the plateaus found by find_plateaus.
    fig, axes = tp.subplots(1, 2, panel_aspect=0.7)
    tp.plot_loss(axes[0], t, loss, color=tp.PALETTE[0])
    tp.plot_loss_linear(axes[1], t, loss, color=tp.PALETTE[0])
    plateaus, _ = find_plateaus(t, loss)
    for p in plateaus:
        axes[1].axvspan(p.t_start, p.t_end, color=tp.GRID, lw=0, zorder=0)
    for ax in axes:
        for level in levels[:-1]:
            ax.axhline(level, color=tp.MUTED, ls="--", lw=0.8, zorder=1)
    for k, level in enumerate(levels[:-1]):
        axes[1].annotate(
            f"$L_{k}={level:g}$",
            (1.0, level),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=tp.INK_SECONDARY,
            fontsize=8,
        )
    axes[0].set_title("(a) log time")
    axes[1].set_title("(b) linear time")
    handles = [
        Line2D([], [], color=tp.MUTED, ls="--", lw=0.8, label=r"$L_k=\frac{1}{2}\sum_{r>k}s_r^2$"),
        Patch(color=tp.GRID, label="plateau (find_plateaus)"),
    ]
    axes[0].legend(handles=handles, loc="lower left")
    tp.savefig(fig, RESULTS / "loss")

    fig, ax = tp.subplots(1, 1, width=tp.TEXT_WIDTH / 2, panel_aspect=0.8)
    labels = [f"$a_{r + 1}$ ($s_{r + 1}={w:g}$)" for r, w in enumerate(weights)]
    tp.plot_overlaps(ax, t, rows["overlaps"], labels=labels)
    tp.savefig(fig, RESULTS / "overlaps")

    tree = make_spec(cfg).tree
    fig = plt.figure(figsize=(tp.TEXT_WIDTH, 0.33 * tp.TEXT_WIDTH))
    tp.plot_bond_spectra(fig, t, rows["spectra"], tree=tree, teacher=weights)
    tp.savefig(fig, RESULTS / "bond_spectra")

    events = learning_events(traj)
    rho = separation_ratios(events["time"], events["width"])
    meta = traj.meta
    print(
        f"base run: {meta['stop_reason']} at t = {meta['t_end']:.1f}, {meta['n_steps']} steps, "
        f"{meta['n_evals']} gradient evaluations, {meta['timing']['total']:.1f} s; "
        f"{len(traj['t'])} log rows + {len(traj['fine_t'])} fine rows; plateaus {len(plateaus)}"
    )
    print(
        f"  events t = {np.round(events['time'], 2).tolist()}, widths "
        f"{np.round(events['width'], 3).tolist()}, max sample spacing across the rises "
        f"{np.round(events['spacing'], 3).tolist()}, rho = {np.round(rho, 1).tolist()}"
    )


def _fit(x, y) -> tuple[float, float]:
    """Least squares y = a + b x: (b, a)."""
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def _mean_se(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def sweep_figures(grid_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(grid_dir / "summary.csv")
    # All runs share the config except sigma and the seed.
    cfg = RunConfig.from_dict(load_run(grid_dir, runs["name"][0]).meta["config"])
    k = make_spec(cfg).n_cores
    weights = np.array(cfg.teacher.weights)
    runs["sigma"] = runs["init.sigma"]
    # Separation ratios between consecutive events (sorted by time), from the trajectories.
    rhos = []
    for name in runs["name"]:
        ev = learning_events(load_run(grid_dir, name))
        rhos.append(separation_ratios(ev["time"], ev["width"]))
    runs["rho_1"], runs["rho_2"] = [r[0] for r in rhos], [r[1] for r in rhos]
    # Aligned balanced init with per-core amplitude a0 = sigma: independent Saxe-type modes.
    for r, s in enumerate(weights):
        runs[f"t_aligned_{r}"] = [escape_time(k, s, a0) for a0 in runs["sigma"]]
        runs[f"ratio_{r}"] = runs[f"t_cross_{r}"] / runs[f"t_aligned_{r}"]
    agg = {"n_runs": ("seed", "size")}
    for r in range(len(weights)):
        agg |= {
            f"t{r + 1}_mean": (f"t_cross_{r}", "mean"),
            f"t{r + 1}_se": (f"t_cross_{r}", "sem"),
            f"t{r + 1}_aligned": (f"t_aligned_{r}", "first"),
            f"ratio{r + 1}_mean": (f"ratio_{r}", "mean"),
            f"ratio{r + 1}_se": (f"ratio_{r}", "sem"),
        }
    agg |= {
        "rho1_median": ("rho_1", "median"),
        "rho2_median": ("rho_2", "median"),
        "rho_min_min": ("rho_min", "min"),
    }
    table = runs.groupby("sigma").agg(**agg).sort_index(ascending=False).reset_index()
    table.to_csv(RESULTS / "escape_times.csv", index=False, float_format="%.6g")

    # Per seed, across sigma: power-law exponent of t1; t1 = A sigma^-(K-2) + B with A relative to
    # the aligned leading coefficient 1 / ((K-2) s_1); mean t1 / t1_aligned; exponents of rho_1, rho_2.
    a_aligned = 1 / ((k - 2) * weights[0])
    per_seed = []
    for seed, g in runs.groupby("seed"):
        log_s = np.log(g["sigma"])
        slope, intercept = _fit(g["sigma"] ** -(k - 2), g["t_cross_0"])
        per_seed.append(
            {
                "seed": seed,
                "t1_exponent": _fit(log_s, np.log(g["t_cross_0"]))[0],
                "A_over_aligned": slope / a_aligned,
                "B": intercept,
                "ratio1_mean": g["ratio_0"].mean(),
                "rho1_exponent": _fit(log_s, np.log(g["rho_1"]))[0],
                "rho2_exponent": _fit(log_s, np.log(g["rho_2"]))[0],
                "rho1_sigma_median": float(np.median(g["rho_1"] * g["sigma"])),
            }
        )
    seeds = pd.DataFrame(per_seed)
    seeds.to_csv(RESULTS / "seeds.csv", index=False, float_format="%.6g")

    fig, axes = tp.subplots(1, 2, panel_aspect=0.75)
    ax = axes[0]
    tp.plot_scaling(
        ax, table["sigma"], table["t1_mean"], table["t1_se"], color=tp.PALETTE[0], label=None
    )
    p, p_se = _mean_se(seeds["t1_exponent"])
    fine = np.geomspace(table["sigma"].min(), table["sigma"].max(), 100)
    prefactor = np.exp(np.mean(np.log(table["t1_mean"]) - p * np.log(table["sigma"])))
    ax.plot(
        fine,
        prefactor * fine**p,
        color=tp.PALETTE[0],
        lw=1.0,
        label=rf"random init, exponent per seed ${p:.3f} \pm {p_se:.3f}$",
    )
    theory = [escape_time(k, weights[0], a0) for a0 in fine]
    ax.plot(fine, theory, color=tp.INK, ls="--", lw=1.0, label=r"aligned init (theory)")
    ax.set_ylim(top=8 * table["t1_mean"].max())  # room for the legend above the data
    ax.legend(loc="upper right")
    ax.set_xlabel(r"init scale $\sigma$")
    ax.set_ylabel(r"escape time $t_1$ ($a_1 = 1/2$)")
    ax.set_title(f"(a) first component ($s_1={weights[0]:g}$, $K={k}$)")

    ax = axes[1]
    for _, g in runs.groupby("seed"):
        g = g.sort_values("sigma")
        ax.plot(g["sigma"], g["ratio_0"], color=tp.MUTED, lw=0.6, marker="o", ms=2.5, mew=0)
    ax.errorbar(
        table["sigma"],
        table["ratio1_mean"],
        table["ratio1_se"],
        fmt="o",
        color=tp.PALETTE[0],
        mec=tp.SURFACE,
        mew=0.8,
        elinewidth=0.8,
        label="mean ± SE over 8 seeds",
    )
    ax.plot([], [], color=tp.MUTED, lw=0.6, marker="o", ms=2.5, mew=0, label="one seed")
    ax.axhline(1.0, color=tp.MUTED, ls="--", lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"init scale $\sigma$")
    ax.set_ylabel(r"$t_1$ (random init) / $t_1$ (aligned, $a_0=\sigma$)")
    ax.set_title("(b) the O(1) factor")
    ax.legend(loc="lower left")
    tp.savefig(fig, RESULTS / "escape_time")

    # Separation ratios vs sigma: theory_regimes R9, rho_i ~ R_i sigma^-(K-2) per seed.
    fig, axes = tp.subplots(1, 2, panel_aspect=0.75)
    for ax, col, title in zip(axes, ["rho_1", "rho_2"], ["(a) events 1-2", "(b) events 2-3"]):
        for _, g in runs.groupby("seed"):
            g = g.sort_values("sigma")
            ax.plot(g["sigma"], g[col], color=tp.MUTED, lw=0.6, marker="o", ms=2.5, mew=0)
        exponent, se = _mean_se(seeds[f"{col.replace('_', '')}_exponent"])
        ref = np.median(table[f"{col.replace('_', '')}_median"] * table["sigma"] ** (k - 2))
        ax.plot(
            fine,
            ref * fine ** -(k - 2),
            color=tp.INK,
            ls="--",
            lw=1.0,
            label=rf"$\propto\sigma^{{-{k - 2}}}$ (per-seed fits: ${exponent:.3f} \pm {se:.3f}$)",
        )
        ax.axhline(3.0, color=tp.PALETTE[1], lw=0.8, zorder=1, label=r"S2S threshold $\rho=3$")
        ax.plot([], [], color=tp.MUTED, lw=0.6, marker="o", ms=2.5, mew=0, label="one seed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1.0, 30 * runs[col].max())  # room for the legend above the data
        ax.set_xlabel(r"init scale $\sigma$")
        ax.set_ylabel(r"separation ratio $\rho$")
        ax.set_title(title)
        ax.legend(loc="upper right")
    tp.savefig(fig, RESULTS / "separation")

    columns = [
        "name",
        "sigma",
        "seed",
        "stop_reason",
        "t_end",
        "t_cross_0",
        "t_cross_1",
        "t_cross_2",
        "width_0",
        "width_1",
        "width_2",
        "order",
        "n_plateaus",
        "rho_1",
        "rho_2",
        "rho_min",
        "n_unresolved",
        "regime",
        "float64_floor",
        "conservation_drift",
        "n_evals",
        "n_fine_rows",
        "wall_time",
    ]
    runs[columns].sort_values(["sigma", "seed"], ascending=[False, True]).to_csv(
        RESULTS / "runs.csv", index=False, float_format="%.6g"
    )

    theory_fit = tp.fit_power_law(table["sigma"], table["t1_aligned"])
    scaling = plateau_scaling(runs["sigma"], runs["t_cross_0"])
    print(f"sweep: {len(runs)} runs; stop reasons {runs['stop_reason'].value_counts().to_dict()}")
    print(f"  learning orders {runs['order'].value_counts().to_dict()}")
    print(f"  regimes {runs['regime'].value_counts().to_dict()}")
    print(f"  plateaus per run {runs['n_plateaus'].value_counts().to_dict()}")
    print(
        f"  unresolved events {int(runs['n_unresolved'].sum())}; float64-floor runs "
        f"{int(runs['float64_floor'].sum())}; fine rows per run "
        f"{runs['n_fine_rows'].min()}-{runs['n_fine_rows'].max()}"
    )
    print(
        f"  rho_min over all runs {runs['rho_min'].min():.2f} "
        f"({runs.loc[runs['rho_min'].idxmin(), 'name']})"
    )
    for col in ("t1_exponent", "A_over_aligned", "B", "ratio1_mean"):
        mean, se = _mean_se(seeds[col])
        print(
            f"  per seed {col}: {mean:.4f} +- {se:.4f} (SE, 8 seeds; range "
            f"{seeds[col].min():.3f} to {seeds[col].max():.3f})"
        )
    for col in ("rho1_exponent", "rho2_exponent", "rho1_sigma_median"):
        mean, se = _mean_se(seeds[col])
        print(
            f"  per seed {col}: {mean:.3f} +- {se:.3f} (range {seeds[col].min():.3f} to "
            f"{seeds[col].max():.3f})"
        )
    print(
        f"  aligned theory over the same sigma range: exponent {theory_fit[0]:.3f};"
        f" asymptotically -(K-2) = {-(k - 2)}"
    )
    print(
        f"  plateau_scaling: grows = {scaling['grows']}, log slope"
        f" {scaling['log']['slope']:.1f} +- {scaling['log']['slope_se']:.1f}"
    )
    print(
        f"  max conservation drift {runs['conservation_drift'].max():.1e};"
        f" wall time per run {runs['wall_time'].min():.1f}-{runs['wall_time'].max():.1f} s"
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO / "data" / "00_smoke")
    args = parser.parse_args()
    tp.set_style()
    RESULTS.mkdir(exist_ok=True)
    base_figures(args.data / "base")
    sweep_figures(args.data / "grid_sigma")
    print(f"figures and tables in {RESULTS}")


if __name__ == "__main__":
    main()
