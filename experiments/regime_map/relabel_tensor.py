"""Relabel the round-1 N = 4 maps with the tensor-space kernel (lead's request after round 2).

  uv run python experiments/regime_map/relabel_tensor.py compute --raw RAW --cache CACHE [--n-proc 120]
  uv run python experiments/regime_map/relabel_tensor.py compare --raw RAW --cache CACHE [--out DIR] [--force]

`compute` fills dK_tensor (`run_regime.tensor_kernel_change`, from theta_end and the rebuilt initial
point) for every run of sigma_chi_b4, chi_b4 and imbalance_b4 into CACHE/<exp>.csv. `compare` loads
each sweep twice (labels with the 128-probe kernel and with the tensor kernel; same thresholds
delta_K < 0.1, rho > 3) and writes, into results/tensor_relabel/:
  label_changes.csv        counts of label transitions per sweep and model
  lazy_boundary_<exp>.csv  per-seed lazy boundary (sigma where delta_K first drops below 0.1, coming
                           from large sigma; `analysis.seed_kernel_boundary`) with both kernels
The report tables and figures are then regenerated with REGIME_MAP_LABEL_KERNEL=tensor (README).
"""

from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXPS = ("sigma_chi_b4", "chi_b4", "imbalance_b4")
GROUP = {"sigma_chi_b4": ["model", "chi"], "chi_b4": ["model", "chi"], "imbalance_b4": ["chi", "root_scale"]}


def compute(n_proc: int) -> None:
    import analysis as an

    for exp in EXPS:
        df = an.load(exp)
        print(f"{exp}: {int(df.dK_tensor.isna().sum()) if 'dK_tensor' in df else len(df)} of {len(df)} rows to compute",
              flush=True)
        an.add_tensor_kernel(df, exp, n_proc=n_proc)


def load_both(exp: str):
    import analysis as an

    an.LABEL_KERNEL = "probe"
    a = an.load(exp)
    an.LABEL_KERNEL = "tensor"
    b = an.load(exp)
    an.LABEL_KERNEL = "probe"
    return a, b


def compare(out: str, force: bool) -> None:
    import analysis as an

    os.makedirs(out, exist_ok=True)

    def save(df, name):
        path = os.path.join(out, name)
        if os.path.exists(path) and not force:
            raise FileExistsError(path)
        df.to_csv(path, index=False, float_format="%.6g")

    rows = []
    for exp in EXPS:
        a, b = load_both(exp)
        by = "model" if "model" in a and a.model.nunique() > 1 else None
        for key, ga in (a.groupby(by) if by else [("all", a)]):
            gb = b.loc[ga.index]
            tr = pd.crosstab(ga.label, gb.label)
            changed = int((ga.label != gb.label).sum())
            row = dict(exp=exp, model=key, n_runs=len(ga), n_changed=changed,
                       missing_tensor_converged=int(((ga.l_end < 1e-3) & gb.dK.isna()).sum()))
            for i in tr.index:
                for j in tr.columns:
                    if i != j and tr.loc[i, j]:
                        row[f"{i}->{j}"] = int(tr.loc[i, j])
            rows.append(row)
        if exp in ("sigma_chi_b4", "chi_b4"):
            recs = []
            for (model, chi, seed), g in a.groupby(["model", "chi", "seed"]):
                gb = b.loc[g.index]
                recs.append(dict(model=model, chi=chi, seed=seed,
                                 probe=an.seed_kernel_boundary(g, 0.1, col="dK"),
                                 tensor=an.seed_kernel_boundary(gb, 0.1, col="dK")))
            lb = pd.DataFrame(recs)
            lb["tensor_over_probe"] = lb.tensor / lb.probe
            save(lb, f"lazy_boundary_{exp}.csv")
            summ = lb.groupby(["model", "chi"]).agg(n_probe=("probe", "count"), n_tensor=("tensor", "count"),
                                                    probe_med=("probe", "median"), tensor_med=("tensor", "median"),
                                                    ratio_med=("tensor_over_probe", "median"),
                                                    ratio_min=("tensor_over_probe", "min"),
                                                    ratio_max=("tensor_over_probe", "max")).reset_index()
            save(summ, f"lazy_boundary_summary_{exp}.csv")
            print(summ.to_string())
    ch = pd.DataFrame(rows).fillna(0)
    save(ch, "label_changes.csv")
    print(ch.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["compute", "compare"])
    ap.add_argument("--raw", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-proc", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "tensor_relabel"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import analysis as an

    if a.raw:
        an.OUT = a.raw
    an.TENSOR_CACHE = a.cache
    if a.cmd == "compute":
        compute(a.n_proc)
    else:
        compare(a.out, a.force)
