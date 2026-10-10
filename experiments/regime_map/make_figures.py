"""Regenerate every figure and summary table of the regime map from the saved runs.

Order: (1) run the sweeps (`sweep.py configs/*.yaml`), (2) the lazy predictions
(`lazy_predict.py`, `pilots/lazy_pred_b8.py`), (3) this script. Usage:
    uv run python experiments/regime_map/make_figures.py [--raw DIR] [--out DIR] [--force] [--only NAME ...]
--raw: raw-output root (default $REGIME_MAP_OUT, else /root/outputs/regime_map). Outputs go to
--out (default experiments/regime_map/results); existing files are not overwritten unless --force.
Fix round 1 (local, partial raw data): `fix_round1.py analyse` regenerates the subset of figures
that the local reruns support.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import plots  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")

FIGURES = {
    "sigma_chi_b4": lambda: [plots.fig_sigma_chi("sigma_chi_b4", m, f"{RES}/lazy_pred_b4.csv", f"binary N=4, {m}: ")
                             for m in ("centred", "plain", "twocopy")]
    + [plots.fig_sensitivity("sigma_chi_b4", m) for m in ("centred", "plain", "twocopy")],
    "chi_b4": lambda: plots.fig_chi_b4("chi_b4", f"{RES}/lazy_pred_b4.csv"),
    "sigma_chi_b8": lambda: plots.fig_sigma_chi_b8("sigma_chi_b8", f"{RES}/lazy_pred_b8.csv"),
    "sigma_chi_b8_d2": lambda: plots.fig_sigma_chi_b8("sigma_chi_b8_d2", None, {32: "8/32"}, floor=None),
    "sigma_K": lambda: [plots.fig_sigma_K("sigma_K", chi) for chi in (6, 3)],
    "sigma_K_d2": lambda: [plots.fig_sigma_K("sigma_K_d2", chi) for chi in (4, 2)],
    "sigma_K_centred_d2": lambda: plots.fig_sigma_K("sigma_K_centred_d2", 4),
    "summary": lambda: plots.fig_summary(),
    "sigma_c_vs_K": lambda: plots.fig_sigma_c_vs_K([
        ("centred, chi = 4 (2x rank)", "sigma_K_centred_d2", 4), ("plain, chi = 4 (2x rank)", "sigma_K_d2", 4),
        ("plain, full bonds", "sigma_K_full_d2", "full"), ("plain, chi = 2 (= rank)", "sigma_K_d2", 2)]),
    "stalls": lambda: plots.fig_stall_fraction([
        ("d = 2, chi = teacher rank (2)", "sigma_K_d2", 2), ("d = 2, chi = 2 x rank (4)", "sigma_K_d2", 4),
        ("d = 2, full bonds", "sigma_K_full_d2", "full"), ("d = 3, chi = teacher rank (3)", "sigma_K", 3)]),
    "imbalance_b4": lambda: plots.fig_imbalance("imbalance_b4"),
    "examples": lambda: plots.fig_examples("sigma_chi_b4", [
        ("saddle-to-saddle: chi=24, sigma=0.016", dict(model="centred", kind="binary", n=4, d=3, chi=24, sigma=0.01585, seed=0, linear_inset=True)),
        ("rich: chi=24, sigma=0.25", dict(model="centred", kind="binary", n=4, d=3, chi=24, sigma=0.2517, seed=0)),
        ("lazy: chi=24, sigma=4", dict(model="centred", kind="binary", n=4, d=3, chi=24, sigma=3.995, seed=0)),
        ("stalled (floor): chi=3, sigma=2.5", dict(model="centred", kind="binary", n=4, d=3, chi=3, sigma=2.52, seed=0)),
    ], "b4_centred"),
    "large_chi": lambda: [plots.fig_large_chi("large_chi_b4", m, s) for m in ("centred", "plain") for s in (1.0, 1.5)]
    + [plots.fig_large_chi("large_chi_b8", "plain", 1.0, chi_fit_min=8)],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=RES)
    ap.add_argument("--raw", default=None, help="raw-output root (default $REGIME_MAP_OUT)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(FIGURES))
    a = ap.parse_args()
    plots.RES, plots.FORCE = a.out, a.force
    if a.raw:
        plots.analysis.OUT = a.raw
    for name in a.only:
        print("==", name, flush=True)
        FIGURES[name]()
