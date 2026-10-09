"""First escape and rank-1 plateau of the runs of a run_generic sweep vs the predicted rank-1
critical points (predict_rank1.py).

Per run: t_1 = first time any top bond singular value s_1^e reaches half its final value (first
escape); t_2 = first time after t_1 that max_e s_2^e exceeds 1e-2 (start of the second escape; the
small absolute threshold of theory_teachers.md 3.3); plateau level = loss at the time of minimum
|dL/dt| in [t_1, t_2]; matched to the nearest predicted rank-1 saddle level. Also the alignment of the
dense W(t) with the predicted rank-1 tensors: 'race' (best bond-rank-1 approximation of M W*, the
escape direction of the near-origin flow) and 'saddle' (rank-1 critical points of the loss), at the
first time ||W|| reaches 20% of its plateau norm (early) and at the plateau.

Usage: uv run python experiments/data_teacher/analyze_plateaus.py <config name> [<predictions tag>]
Writes results/<name>_plateaus.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from common import OUT_ROOT
from run_generic import expand
from setups import build

from ttn.network import to_dense

HERE = Path(__file__).parent


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


def analyze_run(path, spec, preds):
    d = np.load(path, allow_pickle=True)
    summ = d["summary"].item()
    t, loss, sp, final = d["t"], d["loss"], d["spectra"], d["teacher_spectra"]
    s1 = sp[:, :, 0] / np.maximum(final[None, :, 0], 1e-12)
    k1 = np.argmax((s1 >= 0.5).any(axis=1))
    s2max = sp[:, :, 1].max(axis=1) if sp.shape[2] > 1 else np.zeros(len(t))
    after = np.nonzero((np.arange(len(t)) > k1) & (s2max > 1e-2))[0]
    k2 = after[0] if len(after) else len(t) - 1
    out = dict(summ)
    out.update(t1=t[k1], t2=t[k2], plateau_duration=t[k2] - t[k1])
    if k2 - k1 >= 3:
        speed = np.abs(np.gradient(loss[k1:k2 + 1], t[k1:k2 + 1]))
        km = k1 + int(np.argmin(speed))
        out.update(plateau_level=loss[km], plateau_rel=loss[km] / loss[0], t_plateau=t[km])
    else:
        km = k1
        out.update(plateau_level=np.nan, plateau_rel=np.nan, t_plateau=np.nan)
    sad = [p for p in preds if p[0]["kind"] == "saddle"]
    if sad and np.isfinite(out["plateau_level"]):
        levels = np.array([p[0]["loss"] for p in sad])
        j = int(np.argmin(np.abs(levels - out["plateau_level"])))
        out.update(pred_level=levels[j], pred_index=j, n_pred=len(levels),
                   level_rel_err=(out["plateau_level"] - levels[j]) / levels[j])
    # alignment with predicted rank-1 tensors
    y = d["y"]
    w_pl = to_dense(spec, spec.unflatten(torch.from_numpy(y[km].copy())))
    idx = np.unique(np.r_[np.arange(0, km + 1, max(1, km // 200)), km])
    norms = np.array([float(to_dense(spec, spec.unflatten(torch.from_numpy(y[k].copy()))).norm())
                      for k in idx])
    early = idx[np.argmax(norms >= 0.2 * float(w_pl.norm()))]  # km itself always qualifies
    w_early = to_dense(spec, spec.unflatten(torch.from_numpy(y[early].copy())))
    for kind in ("race", "saddle"):
        cands = [p[1] for p in preds if p[0]["kind"] == kind]
        if cands:
            out[f"cos_{kind}_early"] = max(abs(cos(w_early, c)) for c in cands)
            out[f"cos_{kind}_plateau"] = max(abs(cos(w_pl, c)) for c in cands)
    return out


def main():
    name = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    cfg = yaml.safe_load(open(HERE / "configs" / f"{name}.yaml"))
    pred_file = OUT_ROOT / (f"rank1_predictions_{tag}.pt" if tag else "rank1_predictions.pt")
    preds_all = torch.load(pred_file, weights_only=False) if pred_file.exists() else []
    rows = []
    for case in expand(cfg):
        for tree in cfg["trees"]:
            # predictions are computed for the 'sigma' variant; the white_mapped variant shares the
            # critical values (Fact B), and its states are Sigma^{1/2} times those (not compared).
            ptag = case["tag"].replace("variantwhite_mapped", "variantsigma")
            preds = [p for p in preds_all if p[0]["exp"] == name and p[0]["tag"] == ptag
                     and p[0]["tree"] == tree]
            if case.get("variant") == "white_mapped":
                preds = [p for p in preds if p[0]["kind"] == "saddle"]
                preds = [({**r, "kind": "saddle_level_only"}, w) for r, w in preds]
            built = build(dict(case, tree=tree), 0)
            if built is None:
                continue
            spec = built[0]
            for sigma in cfg["sigmas"][tree]:
                for seed in range(*cfg["seeds"]):
                    path = OUT_ROOT / name / f"{case['tag']}_{tree}_s{sigma}_seed{seed}.npz"
                    if not path.exists():
                        continue
                    r = analyze_run(path, spec, preds)
                    if case.get("variant") == "white_mapped":
                        levels = np.array([p[0]["loss"] for p in preds])
                        if len(levels) and np.isfinite(r["plateau_level"]):
                            j = int(np.argmin(np.abs(levels - r["plateau_level"])))
                            r.update(pred_level=levels[j], pred_index=j, n_pred=len(levels),
                                     level_rel_err=(r["plateau_level"] - levels[j]) / levels[j])
                    r.pop("events", None)
                    r.pop("spectral_events", None)
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / f"{name}_plateaus.csv", index=False)
    keys = [k for k in ("alpha", "theta_deg", "variant", "s2", "overlap", "rho") if k in df]
    cols = [c for c in ("plateau_rel", "level_rel_err", "plateau_duration", "cos_race_early",
                        "cos_saddle_early", "cos_saddle_plateau") if c in df]
    print(df.groupby(keys + ["tree"])[cols].agg(["median"]).to_string())


if __name__ == "__main__":
    main()
