"""Lazy-limit predictions at initialization (notes/regime_map.md, sections 1.1-1.2).

Matrix-free, so it works for any tree size:
- u_lin = J^+ W*, the parameter change of the linearized (lazy) dynamics, by conjugate gradients
  on J^T J u = J^T W* from u = 0 (stays in the row space of J, so it converges to the min-norm
  solution);
- lazy residual ||P_perp W*||^2 / ||W*||^2 (loss fraction left by the lazy limit);
- kappa = ||D Theta[u_lin]||_F / ||Theta||_F for the probe tangent kernel, evaluated at the
  sigma = 1 init; the kernel change at scale sigma is then delta_K ~ kappa / sigma^K.

J^T y = grad_theta <W(theta), y>, and J^T J u = d/d phi [grad_theta <W(theta), W(phi)>] . u at
theta = phi = theta_0 (forward-over-reverse).
"""

from __future__ import annotations

import math

import torch

from ttn.network import TTNSpec, forward, inner, to_dense


def _jtj(spec: TTNSpec, flat0: torch.Tensor, mask: torch.Tensor | None = None):
    def grad_inner(phi):
        def f(theta):
            return inner(spec, spec.unflatten(theta), spec.unflatten(phi))

        return torch.func.grad(f)(flat0)

    def apply(u):
        if mask is not None:
            u = u * mask
        out = torch.func.jvp(grad_inner, (flat0,), (u,))[1]
        return out * mask if mask is not None else out

    return apply


def lazy_solution(spec: TTNSpec, flat0: torch.Tensor, teacher, mask=None, tol=1e-12, max_iter=None):
    """u_lin = J^+ W* by CG on the normal equations; returns (u_lin, residual fraction, n_iter)."""
    theta = flat0.clone().requires_grad_(True)
    g = torch.autograd.grad(teacher.inner_ttn(spec, spec.unflatten(theta)), theta)[0]  # J^T W*
    if mask is not None:
        g = g * mask
    apply = _jtj(spec, flat0, mask)
    w2 = teacher.norm2().item()
    u = torch.zeros_like(flat0)
    r = g.clone()
    p = r.clone()
    rr = r @ r
    max_iter = max_iter or 4 * flat0.numel()
    it = 0
    for it in range(1, max_iter + 1):
        ap = apply(p)
        step = rr / (p @ ap)
        u = u + step * p
        r = r - step * ap
        rr_new = r @ r
        if rr_new.sqrt() < tol * g.norm():
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    fit = u @ g  # <W*, J u>
    ju2 = u @ apply(u)  # ||J u||^2
    residual = (w2 - 2 * fit + ju2) / w2
    return u, residual.item(), it


def probe_jacobian(spec: TTNSpec, flat: torch.Tensor, probes: torch.Tensor, mask=None):
    jac = torch.func.jacrev(lambda p: forward(spec, spec.unflatten(p), probes))(flat)
    return jac * mask if mask is not None else jac


def kernel_change_rate(spec, flat0, u, probes, mask=None, eps=1e-5):
    """||D Theta[u]||_F / ||Theta||_F for the probe tangent kernel (central differences, exact up
    to O(eps^2) for this polynomial)."""
    j0 = probe_jacobian(spec, flat0, probes, mask)
    k0 = j0 @ j0.T
    scale = flat0.norm() / u.norm()
    jp = probe_jacobian(spec, flat0 + eps * scale * u, probes, mask)
    jm = probe_jacobian(spec, flat0 - eps * scale * u, probes, mask)
    dk = (jp @ jp.T - jm @ jm.T) / (2 * eps * scale)
    return (dk.norm() / k0.norm()).item()


def lazy_solution_dense(spec: TTNSpec, flat0: torch.Tensor, teacher, mask=None):
    """Same as lazy_solution with the dense D x P Jacobian and an SVD least-squares solve
    (robust for ill-conditioned J, e.g. square cores; only for small D * P)."""
    jac = torch.func.jacrev(lambda p: to_dense(spec, spec.unflatten(p)).reshape(-1))(flat0)
    if mask is not None:
        jac = jac * mask
    wstar = teacher.dense().reshape(-1)
    u = torch.linalg.lstsq(jac, wstar.unsqueeze(1), driver="gelsd").solution[:, 0]
    residual = ((wstar - jac @ u).norm() ** 2 / wstar.norm() ** 2).item()
    return u, residual, 0


def lazy_predictions(spec: TTNSpec, cores_sigma1, teacher, probes, mask=None, dense_limit=2e8):
    """All lazy-limit predictions for an init drawn at sigma = 1 (same seed as the run).

    Returns dict(residual, kappa, rel_dtheta1, cg_iter); at scale sigma: delta_K ~ kappa/sigma^K,
    delta_theta ~ rel_dtheta1 / sigma^K, final loss fraction -> residual. Uses the dense solver
    when D * P <= dense_limit, CG otherwise (cg_iter > 0).
    """
    flat0 = spec.flatten(cores_sigma1).detach()
    if math.prod(spec.phys_dims) * spec.n_params <= dense_limit:
        u, res, it = lazy_solution_dense(spec, flat0, teacher, mask)
    else:
        u, res, it = lazy_solution(spec, flat0, teacher, mask)
    kappa = kernel_change_rate(spec, flat0, u, probes, mask)
    return dict(residual=res, kappa=kappa, rel_dtheta1=(u.norm() / flat0.norm()).item(), cg_iter=it)
