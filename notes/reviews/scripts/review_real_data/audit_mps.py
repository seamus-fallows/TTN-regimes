"""MPS checks on one run: is the L ~ 0.23 plateau (where the Hessian was evaluated) before or after the
first route (edge 5) rose?  Usage: uv run python audit_mps.py <summary.json> <runs_dir> [name_filter]"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "code"))
from analysis import load  # noqa: E402


def main(summary_json, runs_dir, flt="mps"):
    for r in json.load(open(summary_json)):
        if flt not in r["name"] or "drops" not in r:
            continue
        run = load(Path(runs_dir) / f"{r['name']}.npz")
        lg = run["p2_log"]
        d0, d1 = r["drops"][0], r["drops"][1]
        inside = np.flatnonzero((lg["t"] > d0["t_hi"]) & (lg["t"] < d1["t_lo"]))
        i_mid = inside[np.argmin(lg["speed"][inside])] if inside.size else None
        fs = lg["fspectra"]
        ratio = fs[:, :, 1] / fs[:, :, 0]
        first = r.get("first_rank2_edges") or []
        print(
            f"== {r['name']}: t_a = {r['t_a']:.4g}, drops 0/1 at tau {d0['t']:.1f} / {d1['t']:.1f}, "
            f"first rank-2 edges {first} at tau {r['t_first_rank2']:.2f}"
        )
        for e in first[:2] + [4, 3]:
            print(
                f"   edge {e}: s2/s1 at tau=0: {ratio[0, e]:.3g}; at plateau midpoint "
                f"(tau {lg['t'][i_mid]:.1f}, L {lg['loss'][i_mid]:.4f}): {ratio[i_mid, e]:.3g}"
                if i_mid is not None
                else ""
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
