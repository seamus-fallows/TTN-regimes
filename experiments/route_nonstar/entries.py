"""Route-entry norms at the plateau points (post hoc, from the saved plateau states).

`run.py` meant to compute them for the star controls but skipped them: its AlignedManifold had other
bond dimensions than the network (rank + 2 per edge), so it was set to None (README, deviation 4 of
the note). Here the manifold's `align` / `route_entry_norms` run on the network's own spec (they only
read the bond dimensions and the target). For the non-star targets this works only at plateaus
where the learned set's blocks are stars; elsewhere the row records the error.

usage: uv run python experiments/route_nonstar/entries.py [--targets A,B] [--workers N]
Writes results/route_entries.csv: one row per (run, plateau, missing pattern).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import OUT, load_targets, make_model  # noqa: E402

from ttn.routes import AlignedManifold  # noqa: E402

torch.set_num_threads(1)


def one(path: str):
    rec = json.loads(Path(path).read_text())
    cfg, targets = load_targets()
    t = targets[rec["target"]]
    model = make_model(t, cfg["spare"])
    man = AlignedManifold(model.target, model.d, require_star=False)
    man.spec = model.spec  # align / route_entry_norms only read bond dims and the target
    z = np.load(path.replace(".json", ".npz"))
    rows = []
    for s in rec["segments"]:
        if not s or "J_amp_idx" not in s or f"theta_seg{s['k']}" not in z:
            continue
        J = s["J_amp_idx"]
        if not J or len(J) == model.n_pat:
            continue
        cores = model.spec.unflatten(torch.from_numpy(z[f"theta_seg{s['k']}"]))
        base = dict(target=rec["target"], tree=rec["tree"], d=rec["d"], chi=rec["chi"],
                    sigma=rec["sigma"], log10_sigma=rec["log10_sigma"], seed=rec["seed"],
                    segment=s["k"], J=s["J_amp"], canonical=s["canonical"], L=s["L"])
        try:
            aligned, quality = man.align(cores, J)
            q = float(min(quality.values()))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(base, missing="", error=f"align: {exc!r}"[:120]))
            continue
        for r in range(model.n_pat):
            if r in J:
                continue
            row = dict(base, missing="".join(map(str, model.patterns[r])), align_quality=q,
                       depth_paper=model.target.depth(J, r))
            try:
                norms = man.route_entry_norms(aligned, J, r)
                v = np.array(list(norms.values()))
                sig = rec["sigma"]
                row.update(entry_min=v.min() / sig, entry_max=v.max() / sig,
                           entry_geo=float(np.exp(np.log(np.maximum(v, 1e-300)).mean())) / sig,
                           n_nodes=len(v), error="")
            except Exception as exc:  # noqa: BLE001
                row.update(error=f"{exc!r}"[:120])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="B4_mps4,CTL1_mps6,CTL1_binary8,RND3_mps6")
    ap.add_argument("--workers", type=int, default=100)
    a = ap.parse_args()
    paths = []
    for name in a.targets.split(","):
        paths += sorted(str(p) for p in (OUT / "raw" / name).glob("*.json"))
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, paths, chunksize=4) for r in rs]
    df = pd.DataFrame(rows).sort_values(["target", "log10_sigma", "seed", "segment", "missing"])
    df.to_csv(HERE / "results" / "route_entries.csv", index=False, float_format="%.6g")
    ok = df[df.error == ""]
    print(f"{len(df)} rows, {len(ok)} with norms")
    print(ok.groupby(["target", "J", "missing"]).apply(
        lambda g: pd.Series(dict(n=len(g), slope_geo=np.polyfit(
            g.log10_sigma, np.log10(g.entry_geo * g.sigma), 1)[0] if g.log10_sigma.nunique() > 2
            else np.nan, quality=g.align_quality.median()))).to_string())


if __name__ == "__main__":
    main()
