"""Teacher tensors W* (realizable targets f*(x) = <W*, x_1 (x) ... (x) x_N>).

Two representations:
- `CPTeacher`: W* = sum_r s_r u_r^(1) (x) ... (x) u_r^(N). Inner products with any TTN are
  tree-agnostic (evaluate the TTN on the rank-one terms). "Odeco" teachers (u_r^(i) orthonormal over
  r at every site) have exactly the weights s_r as Schmidt values on every bond of every tree (for
  whitened inputs; the Sigma-weighted values differ under anisotropic site covariances).
- `TTNTeacher`: W* given as a TTN on some tree (the student's or another).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ttn.network import Cores, TTNSpec, forward, inner, to_dense


@dataclass
class CPTeacher:
    weights: torch.Tensor  # (R,)
    factors: torch.Tensor  # (R, N, d), unit vectors u_r^(i)

    @property
    def n_components(self) -> int:
        return self.factors.shape[0]

    @property
    def n_sites(self) -> int:
        return self.factors.shape[1]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """f*(x) for x of shape (B, N, d)."""
        proj = torch.einsum("rnd,bnd->brn", self.factors, x)
        return proj.prod(-1) @ self.weights

    def dense(self) -> torch.Tensor:
        out = 0
        for r in range(self.n_components):
            term = self.weights[r]
            for i in range(self.n_sites):
                term = term.unsqueeze(-1) * self.factors[r, i]
            out = out + term
        return out

    def _metric_vectors(self, site_metrics: list[torch.Tensor] | None) -> list[torch.Tensor]:
        """M_i u_r^(i) stacked over r, one (R, d) tensor per site."""
        if site_metrics is None:
            return [self.factors[:, i, :] for i in range(self.n_sites)]
        return [self.factors[:, i, :] @ site_metrics[i] for i in range(self.n_sites)]

    def gram(self, site_metrics: list[torch.Tensor] | None = None) -> torch.Tensor:
        """G[r, r'] = <U_r, M U_r'> for the rank-one terms U_r = (x)_i u_r^(i)."""
        vecs = self._metric_vectors(site_metrics)
        gram = torch.ones(
            self.n_components,
            self.n_components,
            dtype=self.weights.dtype,
            device=self.weights.device,
        )
        for i in range(self.n_sites):
            gram = gram * (self.factors[:, i, :] @ vecs[i].T)
        return gram

    def norm2(self, site_metrics: list[torch.Tensor] | None = None) -> torch.Tensor:
        return self.weights @ self.gram(site_metrics) @ self.weights

    def inner_ttn(
        self, spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """<W_student, W*>_M."""
        return self.weights @ forward(spec, cores, self._metric_vectors(site_metrics))

    def component_overlaps(
        self, spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """a_r = <W, M U_r> / <W*, M U_r>, U_r = (x)_i u_r^(i): 0 at small init, 1 at W = W* for
        any site metrics. For an odeco teacher with whitened inputs <W*, U_r> = s_r, so a_r is the
        fraction of component r learned. (Dividing by s_r instead would converge to
        prod_i <u_r, M_i u_r> != 1 under anisotropic inputs.)"""
        target = self.gram(site_metrics) @ self.weights
        return forward(spec, cores, self._metric_vectors(site_metrics)) / target


@dataclass
class TTNTeacher:
    spec: TTNSpec
    cores: Cores

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return forward(self.spec, self.cores, x)

    def dense(self) -> torch.Tensor:
        return to_dense(self.spec, self.cores)

    def norm2(self, site_metrics: list[torch.Tensor] | None = None) -> torch.Tensor:
        return inner(self.spec, self.cores, self.cores, site_metrics)

    def inner_ttn(
        self, spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        if spec.tree == self.spec.tree:
            return inner(spec, cores, self.cores, site_metrics, spec_b=self.spec)
        from ttn.general import inner_general

        return inner_general(spec, cores, self.spec, self.cores, site_metrics)


# ---- constructors ----------------------------------------------------------------------------


def teacher_generator(seed: int) -> torch.Generator:
    """Random stream for teacher construction, independent of `TTNSpec.init(seed=...)`.

    With a plain manual_seed(seed), teacher and student init draw the same Gaussians for equal
    seeds (both default to 0): random_ttn_teacher(spec) is then exactly proportional to
    spec.init(sigma), and an odeco/pattern teacher's first site basis is the Q factor of the first
    core's Gaussian matrix (the student's first core is triangular in the teacher basis)."""
    mixed = np.random.SeedSequence([0x7EAC, seed]).generate_state(1, dtype=np.uint64)[0]
    return torch.Generator().manual_seed(int(mixed))


def random_orthogonal(d: int, generator: torch.Generator, dtype=torch.float64) -> torch.Tensor:
    """Haar-random orthogonal d x d matrix."""
    q, r = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=dtype))
    return q * torch.sign(torch.diagonal(r))


def site_bases(
    n_sites: int, d: int, seed: int, basis: str = "random", dtype=torch.float64
) -> torch.Tensor:
    """(N, d, d): an orthonormal basis per site (columns). basis = 'random' (Haar) or 'standard'."""
    if basis == "standard":
        return torch.eye(d, dtype=dtype).expand(n_sites, d, d).clone()
    g = teacher_generator(seed)
    return torch.stack([random_orthogonal(d, g, dtype) for _ in range(n_sites)])


def pattern_teacher(
    weights: list[float] | torch.Tensor,
    patterns: list[list[int]],
    d: int,
    seed: int = 0,
    basis: str = "random",
) -> CPTeacher:
    """Component r uses basis vector patterns[r][i] at site i.

    patterns = [[0, 0, 0, 0], [1, 1, 1, 1]] is odeco; [[0, 0, 0, 0], [1, 1, 0, 0]] gives a second
    component that differs from the first only on sites 0 and 1 (a "local" component).
    """
    n_sites = len(patterns[0])
    bases = site_bases(n_sites, d, seed, basis)
    factors = torch.stack(
        [torch.stack([bases[i][:, p[i]] for i in range(n_sites)]) for p in patterns]
    )
    return CPTeacher(torch.as_tensor(weights, dtype=torch.float64), factors)


def odeco_teacher(
    n_sites: int, d: int, weights: list[float] | torch.Tensor, seed: int = 0, basis: str = "random"
) -> CPTeacher:
    """Orthogonally decomposable teacher: component r uses the r-th basis vector at every site."""
    n_comp = len(weights)
    if n_comp > d:
        raise ValueError("an odeco teacher needs at most d components")
    return pattern_teacher(weights, [[r] * n_sites for r in range(n_comp)], d, seed, basis)


def random_cp_teacher(
    n_sites: int, d: int, weights: list[float] | torch.Tensor, seed: int = 0
) -> CPTeacher:
    """Unit-norm Gaussian factors (generically non-orthogonal components)."""
    g = teacher_generator(seed)
    f = torch.randn(len(weights), n_sites, d, generator=g, dtype=torch.float64)
    f = f / f.norm(dim=-1, keepdim=True)
    return CPTeacher(torch.as_tensor(weights, dtype=torch.float64), f)


def random_ttn_teacher(spec: TTNSpec, seed: int = 0, norm: float = 1.0) -> TTNTeacher:
    """Gaussian cores (fan-in init), rescaled so that ||W*||_F = norm."""
    cores = spec.init(sigma=1.0, generator=teacher_generator(seed))
    current = inner(spec, cores, cores).sqrt()
    scale = (norm / current) ** (1.0 / spec.n_cores)
    return TTNTeacher(spec, [c * scale for c in cores])


@dataclass
class ShiftedTeacher:
    """Target W* + W_0, where W_0 is the student's initial tensor (Chizat et al.'s centred model).

    Training the TTN on this target is training f_theta - f_theta0 on W*: the initial function is
    removed exactly (in exact arithmetic), without changing the architecture. Numerically the loss
    is a difference of terms of size ||W_0||^2 ~ sigma^(2K): its absolute error is ~1e-16 ||W_0||^2
    (garbage for K = 7 at sigma >= 10). For large init scales use `antisymmetric_double` with
    `sum_difference_cores`, which is cancellation-free.
    """

    base: CPTeacher | TTNTeacher
    spec0: TTNSpec
    cores0: Cores

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + forward(self.spec0, self.cores0, x)

    def norm2(self, site_metrics: list[torch.Tensor] | None = None) -> torch.Tensor:
        w0 = inner(self.spec0, self.cores0, self.cores0, site_metrics)
        cross = self.base.inner_ttn(self.spec0, self.cores0, site_metrics)
        return self.base.norm2(site_metrics) + 2 * cross + w0

    def inner_ttn(
        self, spec: TTNSpec, cores: Cores, site_metrics: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        own = inner(spec, cores, self.cores0, site_metrics, spec_b=self.spec0)
        return self.base.inner_ttn(spec, cores, site_metrics) + own
