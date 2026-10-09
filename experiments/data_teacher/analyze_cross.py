"""Step 4(c): teacher on a different tree. Which student bonds grow first?

For every dense run of configs/e4c_cross.yaml: rank-increase events (edge e, level k >= 2) at the
first time s_k^e reaches half the teacher's value on that cut (diagnostics.md event rule), sorted.
Tests:
- naive rule: the first rank increase is on argmax_e s_2^e(W*) (teacher's second Schmidt value on
  the student's cut; theory_teachers.md 5.1 calls it an upper-bound proxy);
- P11 (binary student): the root bipartition {0123}|{4567} gets its second singular value only after
  both halves have a bottom bond with rank >= 2.
Usage: uv run python experiments/data_teacher/analyze_cross.py
Writes results/e4c_cross_events.csv and results/e4c_cross_summary.csv, figure e4c_cross.png.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, crossings  # noqa: E402
from run_generic import expand  # noqa: E402
from setups import build  # noqa: E402

from ttn.dynamics import Trajectory  # noqa: E402

HERE = Path(__file__).parent
NAME = "e4c_cross"


def main():
    cfg = yaml.safe_load(open(HERE / "configs" / f"{NAME}.yaml"))
    ev_rows, sum_rows = [], []
    examples = {}
    for case in expand(cfg):
        for tree in cfg["trees"]:
            built = build(dict(case, tree=tree), 0)
            if built is None:
                continue
            spec = built[0]
            edges = list(spec.tree.edges)
            below = {v: tuple(sorted(spec.tree.subtree_sites[v])) for v in edges}
            for sigma in cfg["sigmas"][tree]:
                for seed in range(*cfg["seeds"]):
                    p = OUT_ROOT / f"{NAME}_dense" / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
                    if not p.exists():
                        continue
                    tr = Trajectory.load(p)
                    t, sp = tr.data["t"], np.nan_to_num(tr.data["spectra"])
                    tsp = tr.data["teacher_spectra"]
                    evs = []
                    for j, v in enumerate(edges):
                        for k in range(1, sp.shape[2]):
                            if tsp[j, k] < 1e-2:
                                continue
                            up = [tc for tc, sg in crossings(t, sp[:, j, k] / tsp[j, k], 0.5)
                                  if sg > 0]
                            evs.append(dict(pair=case["pair"], teacher_seed=case["teacher_seed"],
                                            seed=seed, edge=v, cut=str(below[v]), k=k + 1,
                                            teacher_value=tsp[j, k],
                                            t=up[0] if up else np.nan))
                    ev_rows += evs
                    e = pd.DataFrame(evs).dropna(subset=["t"]).sort_values("t")
                    if not len(e):
                        continue
                    k2 = e[e.k == 2]
                    naive_edge = edges[int(np.argmax(tsp[:, 1]))]
                    row = dict(pair=case["pair"], teacher_seed=case["teacher_seed"], seed=seed,
                               first_edge=int(k2.iloc[0]["edge"]) if len(k2) else -1,
                               naive_edge=naive_edge, order=" ".join(
                                   f"{int(r.edge)}:{int(r.k)}" for r in e.itertuples()),
                               unresolved=int(sum(np.isnan(x["t"]) for x in evs)),
                               t_end=float(t[-1]), loss_end_rel=float(tr.data["loss"][-1] / tr.data["loss"][0]))
                    row["naive_ok"] = row["first_edge"] == naive_edge
                    if tree == "binary":
                        # root bipartition = the two edges into the root (same cut); halves = bottom
                        root_edges = [v for v in edges if spec.tree.parent[v] == spec.tree.root]
                        bottoms = [v for v in edges if not spec.tree.children[v]]
                        t_root = k2[k2.edge.isin(root_edges)]["t"].min()
                        left = [v for v in bottoms if max(below[v]) < 4]
                        right = [v for v in bottoms if min(below[v]) >= 4]
                        t_left = k2[k2.edge.isin(left)]["t"].min()
                        t_right = k2[k2.edge.isin(right)]["t"].min()
                        row.update(t_root=t_root, t_left=t_left, t_right=t_right,
                                   bottom_up=bool(t_root > max(t_left, t_right))
                                   if np.isfinite([t_root, t_left, t_right]).all() else np.nan)
                    sum_rows.append(row)
                    examples.setdefault(case["pair"], (tr, edges, below, case["tag"], seed))
    pd.DataFrame(ev_rows).to_csv(HERE / "results" / f"{NAME}_events.csv", index=False)
    s = pd.DataFrame(sum_rows)
    s.to_csv(HERE / "results" / f"{NAME}_summary.csv", index=False)
    print(s.to_string())
    print(s.groupby("pair").agg(n=("seed", "size"), naive_ok=("naive_ok", "mean")).to_string())
    if "bottom_up" in s:
        print("bottom-up (binary student):", s["bottom_up"].dropna().mean(),
              "of", s["bottom_up"].notna().sum())
    fig, axes = plt.subplots(1, len(examples), figsize=(7 * len(examples), 4.5), squeeze=False)
    for ax, (pair, (tr, edges, below, tag, seed)) in zip(axes[0], examples.items()):
        t, sp = tr.data["t"], np.nan_to_num(tr.data["spectra"])
        for j, v in enumerate(edges):
            for k in range(min(sp.shape[2], 4)):
                ax.plot(t, sp[:, j, k], color=f"C{j}", lw=1.2 if k == 0 else 0.8,
                        ls=["-", "--", ":", "-."][k], label=f"cut {below[v]}" if k == 0 else None)
        ax2 = ax.twinx()
        ax2.plot(t, tr.data["loss"] / tr.data["loss"][0], "k", lw=1.5)
        ax2.set_yscale("log")
        ax2.set_ylabel("L / L(0)")
        ta = tr.meta["t_a"]
        ax.set(xscale="symlog", xlim=(ta * 0.98, t[-1]), yscale="log", ylim=(1e-3, 1.5),
               xlabel="t", ylabel="bond singular values s_k^e (k = 1 solid, 2 dashed, 3 dotted)",
               title=f"{pair} ({tag}, seed {seed})")
        ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"{NAME}.png", dpi=110)


if __name__ == "__main__":
    main()
