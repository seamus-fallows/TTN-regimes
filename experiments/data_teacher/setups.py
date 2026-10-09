"""Problem setups for steps 2-4: each returns the student spec, a teacher, and the loss metric.

Every setup function takes the config's `setup` dict plus the seed of the run and returns
(spec, teacher, metrics, loss_fn, meta): `metrics` is a list of site matrices (product covariance)
or None; `loss_fn` overrides the loss for non-product inputs (dense moment matrix) or is None.
"""

from __future__ import annotations

import math

import torch

from correlated import dense_metric_loss_fn, moment_matrix, site_cross_cov
from ttn.losses import power_law_cov
from ttn.network import TTNSpec, _sqrt_psd
from ttn.teachers import CPTeacher, TTNTeacher, random_ttn_teacher
from ttn.tree import make_tree


def rotation(d: int, theta: float) -> torch.Tensor:
    """Rotation by theta in the (e1, e2) plane of R^d."""
    r = torch.eye(d, dtype=torch.float64)
    c, s = math.cos(theta), math.sin(theta)
    r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    return r


def misaligned(cfg):
    """Step 2. Odeco teacher on e1, e2 (every site); Sigma_i = R diag(lambda) R^T with R a rotation
    by theta (degrees) in the (e1, e2) plane: theta = 0 is aligned."""
    n, d = cfg["n_sites"], cfg["d"]
    rot = rotation(d, math.radians(cfg["theta_deg"]))
    cov = rot @ power_law_cov(d, cfg["alpha"]) @ rot.T
    eye = torch.eye(d, dtype=torch.float64)
    factors = torch.stack([eye[k].expand(n, d) for k in range(len(cfg["weights"]))])
    teacher = CPTeacher(torch.as_tensor(cfg["weights"], dtype=torch.float64), factors.clone())
    return teacher, [cov] * n


def nonorth(cfg):
    """Step 4(a). Whitened inputs; component 1 = e1 at every site, component 2 = c e1 + sqrt(1-c^2)
    e2 at every site, so <U_1, U_2> = c^N."""
    n, d, c = cfg["n_sites"], cfg["d"], cfg["overlap"]
    weights = [1.0, cfg["s2"]] if "s2" in cfg else cfg["weights"]
    u1 = torch.zeros(d, dtype=torch.float64)
    u1[0] = 1.0
    u2 = torch.zeros(d, dtype=torch.float64)
    u2[0], u2[1] = c, math.sqrt(1 - c * c)
    factors = torch.stack([u1.expand(n, d), u2.expand(n, d)]).clone()
    return CPTeacher(torch.as_tensor(weights, dtype=torch.float64), factors), None


def absorbed(teacher: CPTeacher, metrics):
    """Fact B: the white problem with teacher (x)_i Sigma_i^{1/2} W* (a CP teacher with factors
    Sigma_i^{1/2} u_r / norm) and the init map G -> G x Sigma_i^{1/2} on physical legs."""
    roots = [_sqrt_psd(m) for m in metrics]
    f = torch.stack([teacher.factors[:, i, :] @ roots[i] for i in range(teacher.n_sites)], dim=1)
    norms = f.norm(dim=-1)
    return CPTeacher(teacher.weights * norms.prod(-1), f / norms[..., None]), roots


def build(cfg, seed):
    """Dispatch on cfg['kind']; returns spec, teacher, metrics, loss_fn, init_map.

    cfg['variant'] == 'white_mapped' (product covariances only) replaces the problem by the absorbed
    white problem started from the mapped init: same landscape and same start, different GF metric.
    """
    spec = TTNSpec.uniform(make_tree(cfg["tree"], cfg["n_sites"]), cfg["d"], cfg["chi"])
    kind = cfg["kind"]
    loss_fn = None
    init_map = None
    if kind == "misaligned":
        teacher, metrics = misaligned(cfg)
        if cfg.get("variant") == "white_mapped":
            teacher, init_map = absorbed(teacher, metrics)
            metrics = None
    elif kind == "nonorth":
        teacher, metrics = nonorth(cfg)
    elif kind == "correlated":
        teacher, metrics = nonorth(cfg) if "overlap" in cfg else odeco_std(cfg)
        metric = moment_matrix(cfg["n_sites"], cfg["d"], site_cross_cov(cfg["n_sites"], cfg["rho"]))
        loss_fn = dense_metric_loss_fn(spec, teacher.dense(), metric)
    elif kind == "ttn_teacher":
        teacher, metrics = ttn_teacher(cfg), None
    else:
        raise ValueError(kind)
    return spec, teacher, metrics, loss_fn, init_map


def odeco_std(cfg):
    n, d = cfg["n_sites"], cfg["d"]
    eye = torch.eye(d, dtype=torch.float64)
    factors = torch.stack([eye[k].expand(n, d) for k in range(len(cfg["weights"]))]).clone()
    return CPTeacher(torch.as_tensor(cfg["weights"], dtype=torch.float64), factors), None


def ttn_teacher(cfg):
    """Steps 4(b), 4(c). Random TTN teacher on cfg['teacher_tree'] (default: the student's tree)
    with bond ranks cfg['teacher_ranks'] (one per edge, in tree.edges order). Bond index k on every
    edge is weighted by q^k (q = cfg['decay']) so that bond spectra are spread out and plateaus are
    separated. Normalized to ||W*||_F = 1."""
    n, d = cfg["n_sites"], cfg["d"]
    tree = make_tree(cfg.get("teacher_tree", cfg["tree"]), n)
    if "teacher_perm" in cfg:  # relabel sites: teacher site j sits where the student has perm[j]
        from ttn.tree import Tree
        perm = cfg["teacher_perm"]
        tree = Tree(tree.parent, tuple(tuple(perm[s] for s in ss) for ss in tree.sites),
                    name=tree.name + "_perm")
    bonds = [1] * tree.n_nodes
    for v, r in zip(tree.edges, cfg["teacher_ranks"]):
        bonds[v] = r
    spec_t = TTNSpec(tree, (d,) * n, tuple(bonds))
    base = random_ttn_teacher(spec_t, seed=cfg["teacher_seed"], norm=1.0)
    q = cfg.get("decay", 1.0)
    cores = [c.clone() for c in base.cores]
    for v in tree.edges:  # scale v's parent leg
        w = torch.tensor([q**k for k in range(bonds[v])], dtype=torch.float64)
        cores[v] = cores[v] * w
    teacher = TTNTeacher(spec_t, cores)
    scale = teacher.norm2().sqrt() ** (-1.0 / spec_t.n_cores)
    return TTNTeacher(spec_t, [c * scale for c in cores])
