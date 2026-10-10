"""Step 4(c) re-analysis (after the round-1 review): event rule of theory_teachers.md 3.3 / section 9.

Events (e, k): first time after the first escape (L < 0.9 L(0)) that s_k^e >= f sigma_k^e(T), with
f = 0.05 (diagnostics.md; sensitivity 0.03, 0.1) (the 50%-of-teacher rule of round 1 fires only after later escapes and gave
artefacts). For each config's runs (default e4c_cross; also r2_cross and r2_cross_b):
- binary student: are the first two k = 2 events a sibling pair (same parent)? Does the root
  bipartition get k = 2 only after a bottom bond on each half (P11)? Is P11 discriminating, i.e. is the
  root cut's teacher s_2 above both bottom bonds of at least one half (else a weight-order rule predicts
  the same order)? Is it above every bottom bond (then a weight-order rule predicts the root FIRST)?
  Pair-level naive rule: does the sibling pair containing argmax_e s_2^e(T) (bottom edges) go first?
- MPS student: first edge vs the naive argmax_e s_2^e(T) and vs e* = argmax_e s2(M_e)
  (results/e4c_theory.csv).
Usage: uv run python experiments/data_teacher/analyze_cross_v2.py [config name]
Writes results/<name>_v2_events.csv and results/<name>_v2_summary.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import OUT_ROOT, RESULTS
from run_generic import expand
from setups import build

from ttn.dynamics import Trajectory

HERE = Path(__file__).parent


def events(t, sp, tsp, f, start):
    out = []
    for j in range(sp.shape[1]):
        for k in range(1, sp.shape[2]):
            if tsp[j, k] < 1e-2:
                continue
            h = np.nonzero((np.arange(len(t)) >= start) & (sp[:, j, k] >= f * tsp[j, k]))[0]
            if len(h):
                out.append((float(t[h[0]]), j, k + 1))
    return sorted(out)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "e4c_cross"
    cfg = yaml.safe_load(open(HERE / "configs" / f"{name}.yaml"))
    theory = pd.read_csv(RESULTS / "e4c_theory.csv") if (RESULTS / "e4c_theory.csv").exists() else None
    ev_rows, rows = [], []
    for case in expand(cfg):
        for tree in cfg["trees"]:
            built = build(dict(case, tree=tree), 0)
            if built is None:
                continue
            spec = built[0]
            tr_ = spec.tree
            edges = list(tr_.edges)
            for sigma in cfg["sigmas"][tree]:
                for seed in range(*cfg["seeds"]):
                    p = OUT_ROOT / f"{name}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
                    if not p.exists():
                        continue
                    tr = Trajectory.load(p)
                    t, loss = tr.data["t"], tr.data["loss"]
                    sp, tsp = np.nan_to_num(tr.data["spectra"]), tr.data["teacher_spectra"]
                    start = int(np.argmax(loss < 0.9 * loss[0]))
                    for f in (0.03, 0.05, 0.1):
                        ev = events(t, sp, tsp, f, start)
                        for (te, j, k) in ev:
                            ev_rows.append(dict(case=case["tag"], tree=tree, sigma=sigma, seed=seed,
                                                f=f, t=te, edge=edges[j], k=k))
                        k2 = [(te, edges[j]) for te, j, k in ev if k == 2]
                        row = dict(case=case["tag"], teacher_seed=case.get("teacher_seed"),
                                   pair=case.get("pair"), tree=tree, sigma=sigma,
                                   seed=seed, f=f, first_edge=k2[0][1] if k2 else -1,
                                   order=" ".join(f"{edges[j]}:{k}" for _, j, k in ev))
                        s2t = {v: tsp[j, 1] for j, v in enumerate(edges)}
                        if tree == "binary":
                            par = tr_.parent
                            root_edges = [v for v in edges if par[v] == tr_.root]
                            bottoms = [v for v in edges if not tr_.children[v]]
                            if len(k2) >= 2:
                                a, b = k2[0][1], k2[1][1]
                                row["sibling_first"] = bool(a in bottoms and b in bottoms
                                                            and par[a] == par[b])
                            t_root = min([te for te, v in k2 if v in root_edges], default=np.nan)
                            halves = {}
                            for te, v in k2:
                                if v in bottoms:
                                    halves.setdefault(par[v], te)
                            t_halves = max(halves.values()) if len(halves) == 2 else np.nan
                            row["bottom_up"] = bool(t_root > t_halves) if np.isfinite(
                                t_root) and np.isfinite(t_halves) else np.nan
                            # P11 differs from a weight-order rule only if on SOME half both bottom bonds
                            # are weaker than the root cut (review of round 2, HIGH 1; the round-2 flag
                            # "root not the weakest cut" was too loose)
                            root_s2 = min(s2t[v] for v in root_edges)
                            halves_b = {}
                            for v in bottoms:
                                halves_b.setdefault(par[v], []).append(s2t[v])
                            row["p11_discriminating"] = bool(any(root_s2 > max(b) for b in halves_b.values()))
                            row["root_above_all_bottoms"] = bool(root_s2 > max(s2t[v] for v in bottoms))
                            row["root_margin"] = root_s2 / min(max(b) for b in halves_b.values())
                            naive = max(bottoms, key=lambda v: s2t[v])
                            row["naive_edge"] = naive
                            row["naive_pair_first"] = bool(k2 and k2[0][1] in bottoms
                                                           and par[k2[0][1]] == par[naive])
                        else:
                            row["naive_edge"] = max(edges, key=lambda v: s2t[v])
                            row["naive_ok"] = row["first_edge"] == row["naive_edge"]
                            if theory is not None and "teacher_seed" in case:
                                th = theory[theory.teacher_seed == case["teacher_seed"]]
                                if len(th):
                                    e_star = int(th.sort_values("saddle_loss").iloc[0]["edge_star"])
                                    row["e_star"] = edges[e_star]
                                    row["e_star_ok"] = row["first_edge"] == edges[e_star]
                        rows.append(row)
    pd.DataFrame(ev_rows).to_csv(RESULTS / f"{name}_v2_events.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}_v2_summary.csv", index=False)
    agg = {c: (c, "mean") for c in ("sibling_first", "bottom_up", "naive_pair_first", "naive_ok",
                                     "e_star_ok", "p11_discriminating", "root_above_all_bottoms")
           if c in df}
    print(df.groupby(["pair", "case", "sigma", "f"]).agg(n=("seed", "size"),
                                                          **agg).round(3).to_string())


if __name__ == "__main__":
    main()
