"""Shared helpers for the data-teacher experiments: setups, one gradient-flow run, event extraction.

Gradient flow uses the pilot integrator (scipy DOP853 on the flat parameter vector, rtol 1e-9,
atol 1e-14) until ttn/dynamics.py lands; see README.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "pilots"))
from pilot_gf import run_gf  # noqa: E402

from ttn.losses import power_law_cov  # noqa: E402
from ttn.network import TTNSpec  # noqa: E402
from ttn.teachers import pattern_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

OUT_ROOT = Path(os.environ.get("DATA_TEACHER_OUT", "/root/outputs/data_teacher"))


def aligned_setup(n_sites, d, weights, eig_index, alpha, variant):
    """Odeco teacher on covariance eigenvectors (component r on eigenvector eig_index[r] at every
    site) and the site metrics for a variant: 'sigma' (Sigma, s), 'white' (I, s),
    'white_Ls' (I, Lambda s), 'white_sqrtLs' (I, sqrt(Lambda) s). Returns teacher, metrics, Lambda."""
    cov = power_law_cov(d, alpha)
    lam = torch.diagonal(cov)
    big_lam = torch.stack([lam[k] ** n_sites for k in eig_index])
    s = torch.as_tensor(weights, dtype=torch.float64)
    metrics = None
    if variant == "sigma":
        w, metrics = s, [cov] * n_sites
    elif variant == "white":
        w = s
    elif variant == "white_Ls":
        w = big_lam * s
    elif variant == "white_sqrtLs":
        w = big_lam.sqrt() * s
    else:
        raise ValueError(variant)
    patterns = [[k] * n_sites for k in eig_index]
    teacher = pattern_teacher(w, patterns, d, basis="standard")
    return teacher, metrics, big_lam.numpy()


def run_and_save(path: Path, spec, cores0, teacher, metrics, run_cfg, meta, loss_fn=None):
    """One gradient-flow run; saves an npz with the trajectory and returns the summary row."""
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return data["summary"].item()
    rec = run_gf(
        spec, cores0, teacher, metrics,
        # float(): PyYAML reads "1.0e-9" (no exponent sign) as a string.
        t_max=float(run_cfg["t_max"]), n_rec=int(run_cfg["n_rec"]), rtol=float(run_cfg["rtol"]),
        atol=float(run_cfg["atol"]),
        stop_rel_loss=None if run_cfg.get("stop_rel_loss") is None else float(run_cfg["stop_rel_loss"]),
        w_star=teacher.dense(), loss_fn=loss_fn, keep_y=run_cfg.get("keep_y", True),
    )
    edges = list(spec.tree.edges)
    chi_max = max(spec.bond_dims[v] for v in edges)
    spectra = np.zeros((len(rec["t"]), len(edges), chi_max))
    for k, sp in enumerate(rec["spectra"]):
        for j, v in enumerate(edges):
            spectra[k, j, : len(sp[v])] = sp[v]
    ev = component_events(rec["t"], rec["overlaps"])
    summary = dict(meta)
    summary.update(
        q_drift=rec["q_drift"], nfev=rec["nfev"], wall=rec["wall"], status=rec["status"],
        t_end=rec["t"][-1], loss0=rec["loss"][0], loss_end=rec["loss"][-1], **ev,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, t=rec["t"], loss=rec["loss"], overlaps=rec["overlaps"], spectra=spectra,
        edges=np.array(edges), y=rec["y"], summary=np.array(summary, dtype=object),
    )
    return summary


def crossings(t, a, level):
    """Times where a(t) crosses `level` upward (+1) or downward (-1), linearly interpolated."""
    above = a >= level
    out = []
    for k in np.nonzero(above[1:] != above[:-1])[0]:
        frac = (level - a[k]) / (a[k + 1] - a[k])
        out.append((t[k] + frac * (t[k + 1] - t[k]), 1 if above[k + 1] else -1))
    return out


def component_events(t, overlaps, level=0.5):
    """Per component: first learning time, and the ordered list of learn/unlearn events.

    `order` lists components in the order of their first up-crossing; `swaps` counts down-crossings
    (a learned component being unlearned, e.g. a piggyback rotation to a better component).
    """
    n_comp = overlaps.shape[1]
    events = []
    first = {}
    for r in range(n_comp):
        for tc, sign in crossings(t, overlaps[:, r], level):
            events.append((tc, r, sign))
            if sign > 0 and r not in first:
                first[r] = tc
    events.sort()
    order = sorted(first, key=first.get)
    out = {
        "order": "".join(str(r) for r in order),
        "first": order[0] if order else -1,
        "n_unlearn": sum(1 for e in events if e[2] < 0),
        "events": [(float(tc), int(r), int(sg)) for tc, r, sg in events],
    }
    for r in range(n_comp):
        out[f"t_half_{r}"] = first.get(r, np.nan)
    return out


def plateau_level(t, loss, t_a, t_b):
    """Loss at the point of slowest loss change between two events (linear time)."""
    m = (t > t_a) & (t < t_b)
    if m.sum() < 3:
        return np.nan
    tt, ll = t[m], loss[m]
    speed = np.abs(np.gradient(ll, tt))
    return ll[np.argmin(speed)]


def make_spec(tree_kind, n_sites, d, chi):
    return TTNSpec.uniform(make_tree(tree_kind, n_sites), d, chi)
