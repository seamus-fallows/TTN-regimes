"""Exact population loss for cross-site-correlated Gaussian inputs (small N only).

Input model ("shared latent factor"): x^(i) = sqrt(1 - rho) z^(i) + sqrt(rho) xi with z^(i), xi iid
N(0, I_d) and the same xi at every site. Marginals stay whitened (Cov x^(i) = I); sites are
correlated, Cov(x^(i), x^(j)) = rho I for i != j.

The features Phi(x) = x^(1) (x) ... (x) x^(N) are then not orthonormal and the metric
M = E[vec Phi vec Phi^T] is not a product, so the library's tree contraction does not apply. For
small N we build M exactly with Isserlis' theorem (sum over the (2N-1)!! pairings of the 2N Gaussian
factors) and use L = 1/2 (w - w*)^T M (w - w*) on the dense tensor.
"""

from __future__ import annotations

import string

import torch

from ttn.network import to_dense


def perfect_matchings(items: list[int]):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for k, partner in enumerate(rest):
        for m in perfect_matchings(rest[:k] + rest[k + 1 :]):
            yield [(first, partner)] + m


def site_cross_cov(n_sites: int, rho: float) -> torch.Tensor:
    """(N, N) correlation between sites; the d x d blocks are this times I_d."""
    c = torch.full((n_sites, n_sites), rho, dtype=torch.float64)
    c.fill_diagonal_(1.0)
    return c


def moment_matrix(n_sites: int, d: int, site_corr: torch.Tensor) -> torch.Tensor:
    """M[(a_1..a_N), (b_1..b_N)] = E[prod_i x^(i)_{a_i} prod_i x^(i)_{b_i}] for Gaussian inputs with
    Cov(x^(i)_a, x^(j)_b) = site_corr[i, j] delta_ab. Returned as a (d^N, d^N) matrix."""
    letters = string.ascii_letters
    slots = list(range(2 * n_sites))  # slot k < N: a_k at site k; slot N + k: b_k at site k
    eye = torch.eye(d, dtype=torch.float64)
    out = torch.zeros((d,) * (2 * n_sites), dtype=torch.float64)
    for matching in perfect_matchings(slots):
        coef = 1.0
        terms = []
        for p, q in matching:
            coef *= site_corr[p % n_sites, q % n_sites].item()
            terms.append(letters[p] + letters[q])
        if coef == 0.0:
            continue
        eq = ",".join(terms) + "->" + letters[: 2 * n_sites]
        out += coef * torch.einsum(eq, *([eye] * len(terms)))
    return out.reshape(d**n_sites, d**n_sites)


def sample_correlated(n: int, n_sites: int, d: int, rho: float, generator: torch.Generator):
    z = torch.randn(n, n_sites, d, generator=generator, dtype=torch.float64)
    xi = torch.randn(n, 1, d, generator=generator, dtype=torch.float64)
    return (1 - rho) ** 0.5 * z + rho**0.5 * xi


def dense_metric_loss_fn(spec, w_star: torch.Tensor, metric: torch.Tensor):
    """Loss on flat parameters: 1/2 (w - w*)^T M (w - w*) with the dense tensor W(theta)."""
    w_star_flat = w_star.reshape(-1)

    def loss(flat):
        r = to_dense(spec, spec.unflatten(flat)).reshape(-1) - w_star_flat
        return 0.5 * r @ (metric @ r)

    return loss
