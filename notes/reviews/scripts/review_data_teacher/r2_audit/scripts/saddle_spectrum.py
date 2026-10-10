"""Negative Hessian eigenvalues at the exact balanced rank-1 saddle (MPS-8 students) for a few teachers,
using the agent's own saddle construction (r2_saddle_runs.saddle_for). Read-only w.r.t. the repo."""
import os, sys
os.environ["DATA_TEACHER_RESULTS"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/results"
os.environ["DATA_TEACHER_OUT"] = r"C:/Users/seamu/Iliad-ML/TTN_regimes/data/review_data_teacher/r2_claims_audit/raw"
sys.path.insert(0, r"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher")
import numpy as np, torch, yaml
torch.set_num_threads(1)
from r2_saddle_runs import saddle_for
from ttn.diagnostics import gauge_directions, hessian, project_out_gauge
from ttn.dynamics import Problem

cfg = yaml.safe_load(open(r"C:/Users/seamu/Iliad-ML/TTN_regimes/experiments/data_teacher/configs/r2_saddle.yaml"))
cases = [(n, t) for n, blk in cfg["teachers"].items() for t in blk]
want = sys.argv[1:] or ["randbin8_303", "misaligned8_20", "randmps8_303", "nonorth8_c0.45", "randbin_302"]
for n, tname in cases:
    if tname not in want:
        continue
    tcfg = cfg["teachers"][n][tname]
    spec, teacher, metrics, cores, l_sad, n_sad = saddle_for(tname, tcfg, n, "mps")
    prob = Problem(spec, teacher, site_metrics=metrics)
    th = spec.flatten(cores)
    h = hessian(prob.loss, th)
    ev, eigs = project_out_gauge(h, gauge_directions(spec, th))
    eigs = np.sort(np.asarray(eigs))
    neg = eigs[eigs < -1e-7]
    # per-edge attribution of the most negative eigenvectors (raw Hessian)
    w, v = torch.linalg.eigh(h)
    sizes = [int(np.prod(s)) for s in spec.shapes]
    offs = np.cumsum([0] + sizes)
    rows = []
    for k in range(min(8, len(neg))):
        vec = v[:, k].numpy() ** 2
        wt = [vec[offs[i]:offs[i + 1]].sum() for i in range(len(sizes))]
        rows.append((float(w[k]), [round(x, 2) for x in wt]))
    print(tname, "L_saddle", round(float(l_sad), 6), "n_rank1_min", n_sad, "grad", float(prob.grad(th).norm()))
    print("  negative projected eigenvalues:", np.round(neg, 6).tolist())
    for lam, wt in rows:
        print("  raw eig", round(lam, 6), "core weights", wt)
