"""Which loss drop carries the first route? Compares the repo's first_route gap assignment (scaling.py)
with the drop whose widened window contains the route's rank crossing.

Usage: uv run python audit_attribution.py <summary.json> [name_filter]
"""

import json
import sys

import numpy as np


def main(summary_json, flt=""):
    for r in json.load(open(summary_json)):
        if "drops" not in r or flt not in r["name"]:
            continue
        drops = r["drops"]
        tr = r.get("t_first_rank2", np.nan)
        print(
            f"== {r['name']}  first_rank2_edges={r.get('first_rank2_edges')}  t_first_rank2={tr:.2f}"
        )
        for i, d in enumerate(drops):
            lo, hi = d["t_lo"] - d["width"], d["t_hi"] + d["width"]
            inside = lo <= tr <= hi
            print(
                f"   drop {i}: t={d['t']:.2f} fwhm={d['width']:.2f} window=[{lo:.1f},{hi:.1f}] "
                f"L {d['loss_before']:.4f}->{d['loss_after']:.4f} n={d['n_samples']} "
                f"edges={r['drop_edges'][i]} {'<- route crossing inside' if inside else ''}"
            )
        # repo rule (scaling.first_route)
        later = [d for d in drops[1:] if d["t_hi"] + d["width"] >= tr]
        repo_gap = later[0]["t"] - drops[0]["t"] if later else np.nan
        carrier = [
            i for i, d in enumerate(drops) if d["t_lo"] - d["width"] <= tr <= d["t_hi"] + d["width"]
        ]
        print(
            f"   repo first_route gap = {repo_gap:.2f}; drop(s) whose window contains the crossing: {carrier}; "
            f"rank crossing - drop0 = {tr - drops[0]['t']:.2f}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
