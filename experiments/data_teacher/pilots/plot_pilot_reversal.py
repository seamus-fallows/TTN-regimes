"""Plot overlaps a_r(t) and loss for the reversal pilot (throwaway)."""
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
res = pickle.load(open("/root/outputs/data_teacher/pilots/pilot_reversal.pkl", "rb"))
trees = sorted({r["args"][0] for r in res})
variants = ["sigma", "white_Ls", "white_sqrtLs"]
for tree in trees:
    fig, axes = plt.subplots(3, 8, figsize=(24, 8), sharex=True)
    for r in res:
        kind, s2, alpha, sigma, seed, variant, t_max = r["args"]
        if kind != tree:
            continue
        ax = axes[variants.index(variant), seed]
        ax.plot(r["t"], r["overlaps"][:, 0], label="comp1 (s=1, low var)")
        ax.plot(r["t"], r["overlaps"][:, 1], label="comp2 (s=0.6, high var)")
        ax.plot(r["t"], r["loss"] / r["loss"][0], "k:", label="L/L0")
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.2)
        ax.set_title(f"{variant} seed {seed}", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"/root/outputs/data_teacher/pilots/pilot_reversal_{tree}.png", dpi=70)
    for r in res:
        if r["args"][0] == tree:
            print(r["args"][4], r["args"][5], np.round(r["t_half"], 1), "qdrift %.1e" % r["q_drift"],
                  "Lfin %.1e" % r["loss"][-1])
