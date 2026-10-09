"""Closed-form and reduced-ODE predictions for decoupled (aligned) TTN dynamics.

Setting: odeco teacher W* = sum_r s_r u_r^(1) (x) ... (x) u_r^(N), whitened inputs, aligned init
(`ttn.network.aligned_init`: core v holds amplitude a_{v,r} on component r). Gradient flow stays in
this family and every component evolves on its own. For one component with weight s > 0 and
per-core amplitudes a_1..a_K (K = number of cores):

    P = prod_v a_v,   L = 1/2 (s - P)^2,   a_v' = (s - P) prod_{w != v} a_w = (s - P) P / a_v.

Conservation: d(a_v^2)/dt = 2 (s - P) P is the same for every v, so a_v^2 - a_w^2 is constant.
Balanced case (a_v = a for all v): a' = a^{K-1} (s - a^K). For the whole teacher,
L = 1/2 sum_r (s_r - P_r)^2 and P_r / s_r is `CPTeacher.component_overlaps`.

Conventions: time is gradient-flow time (d theta/dt = -grad L, unit learning rate on every core).
The escape time is the time at which P first reaches level * s (default level = 1/2). Amplitudes
are positive; flipping the signs of an even number of cores is a symmetry, so P0 > 0 is all that
matters (P0 < 0 is not covered). numpy float64 throughout; times t are increasing and >= 0.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import hyp2f1


def _solve(rhs, t0: float, y0: float, t, y_end: float, atol: float = 1e-12) -> np.ndarray:
    """y(t) for the scalar autonomous ODE y' = rhs(y) with y(t0) = y0 (DOP853, rtol = 1e-12),
    integrated backward from t0 for t < t0 and forward for t > t0. The timing error grows like
    ~1e-12 |t - t0|.

    y_end: the stable fixed point the solution approaches monotonically (the learned mode). Once
    within 10 times the tolerance of it, later times get y_end: an explicit solver is
    stability-limited there (step ~ 1 / (K s^{2-2/K})), so integrating on to t >> the escape
    time would take ~t K s^{2-2/K} steps.
    """
    t = np.asarray(t, dtype=float)
    out = np.full(t.shape, float(y0))

    def converged(_, y):
        return abs(y[0] - y_end) - 10 * (atol + 1e-12 * abs(y_end))

    converged.terminal = True
    for side, direction in ((t < t0, -1.0), (t > t0, 1.0)):
        if side.any():
            order = np.argsort(direction * t[side])
            ts = t[side][order]
            sol = solve_ivp(
                lambda _, y: rhs(y[0]),
                (t0, ts[-1]),
                [y0],
                t_eval=ts,
                method="DOP853",
                rtol=1e-12,
                atol=atol,
                events=converged if direction > 0 else None,
            )
            values = np.full(len(ts), float(y_end))
            if len(sol.t):  # none if the solution converged before the first requested time
                values[: len(sol.t)] = sol.y[0]
            out[np.flatnonzero(side)[order]] = values
    return out


def _solve_two_anchors(rhs, y0: float, t_esc: float, y_esc: float, t, y_end: float, atol=1e-12):
    """y(t) from y(0) = y0 for t <= t_esc / 2 and from the escape point y(t_esc) = y_esc (exact,
    from quadrature) after, so the timing error is ~1e-12 min(t, |t - t_esc|): the curve passes
    through the escape point however long the plateau before it (from t = 0 alone the overlap at
    t_esc = 2e9 was 0.4992 instead of 0.5), and starts exactly at y0 (a backward solve alone loses
    the relative accuracy of a tiny amplitude that grows from t = 0). y_end: see `_solve`."""
    t = np.asarray(t, dtype=float)
    early = t <= 0.5 * t_esc
    out = np.empty(t.shape)
    out[early] = _solve(rhs, 0.0, y0, t[early], y_end, atol)
    out[~early] = _solve(rhs, t_esc, y_esc, t[~early], y_end, atol)
    return out


# ---- balanced mode ------------------------------------------------------------------------------


def balanced_mode_solution(K: int, s: float, a0: float, t) -> np.ndarray:
    """Per-core amplitude a(t) of a balanced mode: a' = a^{K-1} (s - a^K), a(0) = a0 > 0.

    Integrated numerically (DOP853, rtol = atol = 1e-12) in u = log a, u' = a^{K-2} (s - a^K), so
    the error is relative at every scale and a0 can be tiny. If P0 = a0^K < s/2, times past T/2
    are integrated from the escape point a(T) = (s/2)^{1/K}, T = `escape_time` (exact): the timing
    error is ~1e-12 min(t, |t - T|) instead of ~1e-12 t. For K = 2 see `balanced_mode_solution_k2`.
    K = 1 (linear, no plateau) uses the closed form a = s + (a0 - s) exp(-t), since
    u' = (s - a)/a is unbounded as a0 -> 0.
    """
    if K == 1:
        return s + (a0 - s) * np.exp(-np.asarray(t, dtype=float))

    def rhs(u):
        a = np.exp(u)
        return a ** (K - 2) * (s - a**K)

    u_end = np.log(s) / K
    if a0**K < 0.5 * s:
        t_esc = escape_time(K, s, a0)
        return np.exp(_solve_two_anchors(rhs, np.log(a0), t_esc, np.log(0.5 * s) / K, t, u_end))
    return np.exp(_solve(rhs, 0.0, np.log(a0), t, u_end))


def balanced_mode_solution_k2(s: float, a0: float, t) -> np.ndarray:
    """Closed form for K = 2: P = a^2 obeys the logistic equation P' = 2 P (s - P), so

    P(t) = s P0 / (P0 + (s - P0) exp(-2 s t)),   a(t) = sqrt(P(t)),   P0 = a0^2.
    """
    t = np.asarray(t, dtype=float)
    p0 = a0**2
    return np.sqrt(s * p0 / (p0 + (s - p0) * np.exp(-2 * s * t)))


def escape_time(K: int, s: float, a0: float, level: float = 0.5) -> float:
    """Exact time for a balanced mode to go from P0 = a0^K to P = level * s (P0 < level s < s).

        t = int_{a0}^{a*} da / (a^{K-1} (s - a^K)),   a* = (level s)^{1/K}.

    Partial fractions, 1/(a^{K-1} (s - a^K)) = 1/(s a^{K-1}) + a / (s (s - a^K)), split off the
    part that is singular at a = 0 and integrate it exactly:

        t = (a0^{2-K} - a*^{2-K}) / ((K-2) s)      [log(a*/a0) / s for K = 2]
            + (1/s) int_{a0}^{a*} a / (s - a^K) da  [smooth and bounded on [0, a*]: quadrature],

    accurate to ~1e-13 relative for any a0 > 0 and any K >= 1 (K = 1: linear model, no plateau).
    """
    a_star = (level * s) ** (1.0 / K)
    if K == 2:
        singular = np.log(a_star / a0) / s
    else:
        singular = (a0 ** (2 - K) - a_star ** (2 - K)) / ((K - 2) * s)
    regular, _ = quad(lambda a: a / (s - a**K), a0, a_star, epsabs=0.0, epsrel=1e-13, limit=200)
    return float(singular + regular / s)


def escape_time_asymptotic(K: int, s, a0, level: float = 0.5, correction: bool = True):
    """Small-init expansion of `escape_time` (vectorized over s and a0).

    Derivation. With G(a) = int_0^a x dx / (s - x^K), the split in `escape_time` is exactly

        t = (a0^{2-K} - a*^{2-K}) / ((K-2) s) + (G(a*) - G(a0)) / s.

    Expanding 1/(s - x^K) in powers of x^K / s and integrating termwise gives
    G(a) = a^2 / (2s) * F(a^K / s) with F(z) = 2F1(1, 2/K; 1 + 2/K; z), so
    G(a0) = a0^2 / (2s) + O(a0^{K+2}). With a* = (lam s)^{1/K} (lam = level), for K >= 3

        t = a0^{-(K-2)} / ((K-2) s) + s^{-(2 - 2/K)} c_K(lam) + O(a0^2 / s^2),
        c_K(lam) = lam^{2/K} F(lam) / 2 - lam^{-(K-2)/K} / (K-2).

    The O(1) term scales as s^{-(2-2/K)} (rescale a = s^{1/K} b, t = s^{-(2-2/K)} tau);
    c_3(1/2) = -0.850, c_4(1/2) = -0.266, and c_K(lam) -> 1/2 as K -> infinity. The remainder is
    exactly -G(a0)/s < 0, so the corrected formula overestimates t by ~a0^2 / (2 s^2).
    For K = 2 (logistic, see `balanced_mode_solution_k2`):

        t = log(s / a0^2) / (2s) + log(lam / (1 - lam)) / (2s) + O(a0^2 / s^2).

    For K = 1 (no plateau) t -> -log(1 - lam). correction=False keeps only the leading term
    (a0^{-(K-2)} / ((K-2) s), log(s / a0^2) / (2s), or 0 for K = 1).
    """
    s, a0 = np.asarray(s, dtype=float), np.asarray(a0, dtype=float)
    if K == 1:
        lead, const = np.zeros_like(s * a0), -np.log(1 - level)
    elif K == 2:
        lead, const = np.log(s / a0**2) / (2 * s), np.log(level / (1 - level)) / (2 * s)
    else:
        lead = a0 ** (2.0 - K) / ((K - 2) * s)
        f_lam = hyp2f1(1, 2 / K, 1 + 2 / K, level)
        c_k = level ** (2 / K) * f_lam / 2 - level ** (-(K - 2) / K) / (K - 2)
        const = s ** (-(2 - 2 / K)) * c_k
    return lead + const if correction else lead


# ---- unbalanced mode ----------------------------------------------------------------------------


def unbalanced_mode_solution(s: float, a0, t) -> np.ndarray:
    """Per-core amplitudes a_v(t), shape (len(t), K), of one mode with any a_v(0) != 0, P0 > 0.

    Since d(a_v^2)/dt = 2 (s - P) P is the same for every v, a_v^2 = a_v(0)^2 + y with one common
    increment y(t), and a single ODE remains:

        y' = 2 P (s - P),   P = sqrt(prod_v (a_v(0)^2 + y)),   y(0) = 0.

    y' is bounded (no blow-up when one core starts tiny); DOP853 with rtol = 1e-12 and
    atol = 1e-12 min_v a_v(0)^2 keeps every a_v^2 accurate to ~1e-12 relative. As in
    `balanced_mode_solution`, if P0 < s/2 times past T/2 are integrated from the escape point
    (P = s/2 at T = `unbalanced_escape_time`), so the timing error is ~1e-12 min(t, |t - T|).
    Signs of a_v(0) are carried along (a_v never crosses 0 when P0 > 0).
    """
    a2 = np.asarray(a0, dtype=float) ** 2

    def rhs(y):
        p = np.sqrt(np.prod(a2 + y))
        return 2 * p * (s - p)

    atol = 1e-12 * a2.min()
    # P(y_end) = s at the end; P >= y^{K/2} bounds the root from above when P0 < s.
    p0 = np.sqrt(np.prod(a2))
    bracket = (0.0, s ** (2 / len(a2))) if p0 < s else (-a2.min(), 0.0)
    y_end = brentq(lambda y: np.sqrt(np.prod(a2 + y)) - s, *bracket, xtol=1e-300)
    if p0 < 0.5 * s:
        t_esc, x_star = _unbalanced_escape(s, a2, 0.5)
        y = _solve_two_anchors(rhs, 0.0, t_esc, x_star - a2.min(), t, y_end, atol=atol)
    else:
        y = _solve(rhs, 0.0, 0.0, t, y_end, atol=atol)
    return np.sign(a0) * np.sqrt(a2 + y[:, None])


def unbalanced_escape_time(s: float, a0, level: float = 0.5) -> float:
    """Time for P = prod_v a_v to go from P0 > 0 to level * s, for arbitrary per-core a_v(0).

    With x = min_v a_v^2 and constants c_v = a_v(0)^2 - x(0) >= 0 (so a_v^2 = x + c_v, see
    `unbalanced_mode_solution`), dx/dt = 2 P (s - P), hence

        t = int_{x0}^{x*} dx / (2 P(x) (s - P(x))),   P(x) = sqrt(prod_v (x + c_v)),

    with P(x*) = level s, by quadrature in log x (which resolves the long plateau at small x).
    Remark: if m cores share the smallest amplitude a_min -> 0 while the others stay finite, then
    P ~ sqrt(C) x^{m/2} with C = prod_{c_v > 0} c_v, and for m >= 3 the escape time is
    ~ a_min^{-(m-2)} / ((m-2) s sqrt(C)): a depth-m balanced mode with effective weight s sqrt(C).
    """
    return _unbalanced_escape(s, np.asarray(a0, dtype=float) ** 2, level)[0]


def _unbalanced_escape(s: float, a2: np.ndarray, level: float) -> tuple[float, float]:
    """(escape time, x*) of `unbalanced_escape_time`, from the squared amplitudes a2."""
    x0, c = a2.min(), a2 - a2.min()
    target = level * s

    def p(x):
        return np.sqrt(np.prod(x + c))

    # P(x) >= x^{K/2}, so the root lies below target^{2/K}.
    x_star = brentq(lambda x: p(x) - target, x0, target ** (2 / len(c)), xtol=1e-300)

    def integrand(u):
        x = np.exp(u)
        px = p(x)
        return x / (2 * px * (s - px))

    t, _ = quad(integrand, np.log(x0), np.log(x_star), epsabs=0.0, epsrel=1e-12, limit=500)
    return float(t), float(x_star)


# ---- whole odeco teacher ------------------------------------------------------------------------


def odeco_aligned_prediction(weights, K: int, a0, t, level: float = 0.5) -> dict[str, np.ndarray]:
    """Predicted gradient flow of a K-core TTN with aligned init on an odeco teacher.

    Whitened inputs. weights: (R,) teacher weights s_r > 0. a0: positive per-core initial
    amplitudes, either a scalar (same on every core and component), (R,) (balanced, one value per
    component) or (K, R) (per core and component, exactly as passed to `ttn.network.aligned_init`).
    Components with equal amplitudes on all cores use the balanced solution, the others the
    unbalanced reduction.

    Returns numpy arrays:
      overlaps (len(t), R)   P_r(t) / s_r, to compare with `CPTeacher.component_overlaps`;
      loss (len(t),)         L(t) = 1/2 sum_r (s_r - P_r(t))^2 (exact: the rank-one terms are
                             orthonormal);
      escape_times (R,)      time at which P_r = level * s_r;
      order (R,)             components sorted by escape time (the learning order);
      plateau_levels (R+1,)  L_k = 1/2 sum_{r not among the first k learned} s_r^2, k = 0..R.
                             For small a0 the loss sits at L_k between the k-th and (k+1)-th
                             escapes.
    """
    weights = np.asarray(weights, dtype=float)
    n_comp = len(weights)
    amps = np.broadcast_to(np.asarray(a0, dtype=float), (K, n_comp))
    overlaps = np.empty((len(np.atleast_1d(t)), n_comp))
    escape = np.empty(n_comp)
    for r, s in enumerate(weights):
        a_r = amps[:, r]
        if np.all(a_r == a_r[0]):
            p = balanced_mode_solution(K, s, a_r[0], t) ** K
            escape[r] = escape_time(K, s, a_r[0], level)
        else:
            p = np.prod(unbalanced_mode_solution(s, a_r, t), axis=1)
            escape[r] = unbalanced_escape_time(s, a_r, level)
        overlaps[:, r] = p / s
    order = np.argsort(escape)
    return {
        "overlaps": overlaps,
        "loss": 0.5 * np.sum((weights * (1 - overlaps)) ** 2, axis=1),
        "escape_times": escape,
        "order": order,
        "plateau_levels": np.array(
            [0.5 * np.sum(weights[order[k:]] ** 2) for k in range(n_comp + 1)]
        ),
    }


# ---- HEURISTIC: first-escape race from random small init ----------------------------------------


def winner_probability(
    weights, K: int, sampler, n_samples: int, seed: int = 0, level: float = 0.5
) -> np.ndarray:
    """HEURISTIC model: probability that component r is learned first, by Monte Carlo.

    Not exact for random init. It pretends the random init is decoupled ("decoupled diagonal
    model"): component r is a balanced mode with effective initial per-core amplitude c_r > 0 and
    escapes at the small-init time T_r = c_r^{-(K-2)} / ((K-2) s_r) + O(1)
    (`escape_time_asymptotic`; log form for K = 2). The winner is argmin_r T_r; for K >= 3 and
    c -> 0 this is argmax_r s_r c_r^{K-2}, so a lucky init can beat a larger teacher weight
    (hypothesis H1 in notes/prior_knowledge.md).

    sampler(rng, shape) returns positive effective amplitudes of shape (n_samples, R), e.g.
    `product_gaussian_sampler`, or |P_r(0)|^{1/K} measured on real random inits
    (P_r(0) = s_r * CPTeacher.component_overlaps). Returns (R,) frequencies, summing to 1.
    """
    weights = np.asarray(weights, dtype=float)
    c = sampler(np.random.default_rng(seed), (n_samples, len(weights)))
    winners = np.argmin(escape_time_asymptotic(K, weights, c, level), axis=1)
    return np.bincount(winners, minlength=len(weights)) / n_samples


def product_gaussian_sampler(K: int, scale: float):
    """Sampler for `winner_probability`: c = scale * |z_1 ... z_K|^{1/K}, z_v iid N(0, 1).

    The effective (geometric-mean) amplitude of a mode whose K per-core amplitudes are iid
    N(0, scale^2), e.g. the diagonal entries of a fan-in init (scale = sigma / sqrt(F) for equal
    fan-ins).
    """

    def sample(rng: np.random.Generator, shape) -> np.ndarray:
        z = rng.standard_normal((K, *shape))
        return scale * np.prod(np.abs(z), axis=0) ** (1.0 / K)

    return sample
