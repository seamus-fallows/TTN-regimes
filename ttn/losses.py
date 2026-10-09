"""Population and empirical square losses, and input sampling."""

from __future__ import annotations

import torch

from ttn.network import Cores, TTNSpec, forward, norm2


def population_loss(
    spec: TTNSpec,
    cores: Cores,
    teacher,
    site_metrics: list[torch.Tensor] | None = None,
    teacher_norm2: torch.Tensor | None = None,
) -> torch.Tensor:
    """L = 1/2 E[(f - f*)^2] = 1/2 ||W - W*||_M^2 for x^(i) ~ N(0, M_i) independent across sites.

    Exact (no sampling). Pass `teacher_norm2` to skip recomputing the constant ||W*||_M^2.
    """
    t2 = teacher.norm2(site_metrics) if teacher_norm2 is None else teacher_norm2
    return 0.5 * (
        norm2(spec, cores, site_metrics) - 2 * teacher.inner_ttn(spec, cores, site_metrics) + t2
    )


def empirical_loss(spec: TTNSpec, cores: Cores, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1/(2n) sum_j (f(x_j) - y_j)^2."""
    return 0.5 * (forward(spec, cores, x) - y).pow(2).mean()


def sample_inputs(
    n: int,
    n_sites: int,
    d: int,
    generator: torch.Generator,
    site_covs: list[torch.Tensor] | None = None,
    dtype=torch.float64,
) -> torch.Tensor:
    """(n, N, d) Gaussian site features, independent across sites, x^(i) ~ N(0, site_covs[i])."""
    z = torch.randn(n, n_sites, d, generator=generator, dtype=dtype)
    if site_covs is None:
        return z
    chol = [torch.linalg.cholesky(c) for c in site_covs]
    return torch.stack([z[:, i, :] @ chol[i].T for i in range(n_sites)], dim=1)


def power_law_cov(d: int, exponent: float, dtype=torch.float64) -> torch.Tensor:
    """Diagonal covariance with eigenvalues k^-exponent (k = 1..d), normalized to trace d."""
    ev = torch.arange(1, d + 1, dtype=dtype) ** (-exponent)
    return torch.diag(ev * d / ev.sum())
