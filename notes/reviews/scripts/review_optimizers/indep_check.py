"""Independent check (reviewer): no code from experiments/optimizers is used.

Library pieces used: make_tree, TTNSpec (shapes, fan-in init, flatten/unflatten), to_dense,
odeco_teacher (factors, weights). Everything else is written here: dense loss 1/2||W - W*||^2,
overlaps a_r = <W, (x)_i u_r^(i)> / s_r, my own Adam / RMSProp updates (Kingma-Ba with bias
correction, as torch.optim), GF first escape located with scipy solve_ivp (LSODA) + an event
function a_max(theta) = 1/2 (exact event location, no output grid).

    uv run python indep_check.py MODE OUTCSV      (MODE: b4 | b8 | eps | cg)
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.integrate
import torch

sys.path.insert(0, "c:/Users/seamu/Iliad-ML/TTN_regimes")
from ttn.network import TTNSpec, to_dense  # noqa: E402
from ttn.teachers import odeco_teacher  # noqa: E402
from ttn.tree import make_tree  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


class Prob:
    def __init__(self, tree, n, d, chi, weights, teacher_seed=100):
        self.spec = TTNSpec.uniform(make_tree(tree, n), d, chi)
        t = odeco_teacher(n, d, weights, seed=teacher_seed)
        self.n = n
        self.s = torch.as_tensor(t.weights, dtype=torch.float64)
        self.u = torch.as_tensor(t.factors, dtype=torch.float64)  # (R, N, d)
        # dense teacher built here from factors (not the library's dense())
        w = 0
        for r in range(len(self.s)):
            term = self.s[r]
            for i in range(n):
                term = term.unsqueeze(-1) * self.u[r, i]
            w = w + term
        self.wstar = w
        self.K = len(self.spec.shapes)

    def W(self, theta):
        return to_dense(self.spec, self.spec.unflatten(theta))

    def loss(self, theta):
        return 0.5 * (self.W(theta) - self.wstar).pow(2).sum()

    def loss_grad(self, theta):
        th = theta.detach().clone().requires_grad_(True)
        L = self.loss(th)
        (g,) = torch.autograd.grad(L, th)
        return L.item(), g

    def overlaps(self, theta):
        with torch.no_grad():
            w = self.W(theta)
            out = []
            for r in range(len(self.s)):
                x = w
                for i in range(self.n):  # contract site 0 each time
                    x = torch.tensordot(self.u[r, i], x, dims=([0], [0]))
                out.append((x / self.s[r]).item())
        return np.array(out)

    def init(self, sigma, seed):
        return self.spec.flatten(self.spec.init(sigma, seed=seed)).detach().clone()


def first_escape_gf(p, theta0, t_max, rtol=1e-10, atol=1e-15):
    def f(t, y):
        _, g = p.loss_grad(torch.from_numpy(y))
        return -g.numpy()

    def ev(t, y):
        return p.overlaps(torch.from_numpy(y)).max() - 0.5

    ev.terminal, ev.direction = True, 1
    sol = scipy.integrate.solve_ivp(
        f, (0, t_max), theta0.numpy(), method="LSODA", rtol=rtol, atol=atol, events=ev
    )
    te = sol.t_events[0]
    comp = int(np.argmax(p.overlaps(torch.from_numpy(sol.y_events[0][0])))) if len(te) else -1
    return (float(te[0]) if len(te) else np.nan), comp, sol.nfev


def first_escape_adaptive(p, theta0, kind, lr, n_max, eps=1e-8, b1=0.9, b2=0.999, alpha=0.99):
    th = theta0.clone()
    m = torch.zeros_like(th)
    v = torch.zeros_like(th)
    a_prev = p.overlaps(th).max()
    for n in range(1, n_max + 1):
        _, g = p.loss_grad(th)
        if kind == "adam":
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1**n)
            denom = (v.sqrt() / math.sqrt(1 - b2**n)) + eps
            th = th - lr * mhat / denom
        elif kind == "rmsprop":
            v = alpha * v + (1 - alpha) * g * g
            th = th - lr * g / (v.sqrt() + eps)
        a = p.overlaps(th)
        if a.max() >= 0.5:  # interpolate linearly in n between n-1 and n
            frac = (0.5 - a_prev) / (a.max() - a_prev)
            return n - 1 + frac, int(np.argmax(a))
        a_prev = a.max()
    return np.nan, -1


def mode_b4(out):
    p = Prob("binary", 4, 3, 4, [3.0, 2.0, 1.0])
    rows = []
    for sigma in [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]:
        for seed in [0, 1, 2]:
            th0 = p.init(sigma, seed)
            t0 = time.time()
            tg, cg, nf = first_escape_gf(p, th0, 50.0 / sigma)
            rows.append(
                dict(
                    tree="binary4",
                    method="gf",
                    lr=np.nan,
                    sigma=sigma,
                    seed=seed,
                    t_first=tg,
                    comp=cg,
                )
            )
            for kind, lr, nmax in [
                ("adam", 1e-3, 20000),
                ("adam", 1e-4, 100000),
                ("rmsprop", 1e-3, 20000),
            ]:
                n1, c1 = first_escape_adaptive(p, th0, kind, lr, nmax)
                rows.append(
                    dict(
                        tree="binary4",
                        method=kind,
                        lr=lr,
                        sigma=sigma,
                        seed=seed,
                        t_first=n1,
                        comp=c1,
                    )
                )
            print(
                sigma,
                seed,
                f"gf {tg:.4g} ({nf} fev)",
                [f"{r['method']} {r['lr']:g} {r['t_first']:.5g}" for r in rows[-3:]],
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            write(rows, out)


def mode_b8(out):
    p = Prob("binary", 8, 3, 4, [3.0, 2.0, 1.0])
    rows = []
    for sigma in [0.25, 0.3, 0.4, 0.5, 0.8]:
        for seed in [0, 1, 2]:
            th0 = p.init(sigma, seed)
            t0 = time.time()
            tg, cg, nf = first_escape_gf(p, th0, 1e6)
            rows.append(
                dict(
                    tree="binary8_fanin",
                    method="gf",
                    lr=np.nan,
                    sigma=sigma,
                    seed=seed,
                    t_first=tg,
                    comp=cg,
                )
            )
            n1, c1 = first_escape_adaptive(p, th0, "adam", 1e-3, 20000)
            rows.append(
                dict(
                    tree="binary8_fanin",
                    method="adam",
                    lr=1e-3,
                    sigma=sigma,
                    seed=seed,
                    t_first=n1,
                    comp=c1,
                )
            )
            print(
                sigma,
                seed,
                f"gf {tg:.5g} c{cg} adam {n1:.5g} c{c1} {time.time() - t0:.1f}s",
                flush=True,
            )
            write(rows, out)


def mode_eps(out):
    """MPS8 (K = 8), fan-in init (independent of the agent's balancing): Adam eps law
    n_1 = (eps/lr) T_GF(sigma) + n_sign with T_GF measured directly at that sigma and also
    extrapolated from sigma = 0.3 as (0.3/sigma)^6."""
    p = Prob("mps", 8, 3, 4, [3.0, 2.0, 1.0])
    rows = []
    for seed in [0, 1]:
        for sigma in [0.3, 0.05, 0.03]:
            th0 = p.init(sigma, seed)
            t0 = time.time()
            tg, cg, nf = first_escape_gf(p, th0, 1e13)
            rows.append(
                dict(
                    tree="mps8_fanin",
                    method="gf",
                    lr=np.nan,
                    eps=np.nan,
                    sigma=sigma,
                    seed=seed,
                    t_first=tg,
                    comp=cg,
                )
            )
            print("gf", seed, sigma, tg, cg, nf, f"{time.time() - t0:.1f}s", flush=True)
            write(rows, out)
        for sigma, eps in [(0.03, 1e-30), (0.05, 1e-8), (0.03, 1e-8)]:
            th0 = p.init(sigma, seed)
            t0 = time.time()
            n1, c1 = first_escape_adaptive(p, th0, "adam", 1e-3, 200000, eps=eps)
            rows.append(
                dict(
                    tree="mps8_fanin",
                    method="adam",
                    lr=1e-3,
                    eps=eps,
                    sigma=sigma,
                    seed=seed,
                    t_first=n1,
                    comp=c1,
                )
            )
            print("adam", seed, sigma, eps, n1, c1, f"{time.time() - t0:.1f}s", flush=True)
            write(rows, out)


def mode_cg(out):
    """rms gradient at the init / sigma^(K-1) (the agent's c_g), fan-in init, 6 seeds."""
    rows = []
    for tree, n in [("binary", 4), ("binary", 8), ("mps", 8)]:
        p = Prob(tree, n, 3, 4, [3.0, 2.0, 1.0])
        for seed in range(6):
            sigma = 0.01
            _, g = p.loss_grad(p.init(sigma, seed))
            cg = g.pow(2).mean().sqrt().item() / sigma ** (p.K - 1)
            rows.append(
                dict(
                    tree=f"{tree}{n}",
                    K=p.K,
                    seed=seed,
                    c_g=cg,
                    sigma_eps=(1e-8 / cg) ** (1 / (p.K - 1)),
                )
            )
    write(rows, out)
    import pandas as pd

    print(pd.DataFrame(rows).groupby("tree")[["c_g", "sigma_eps"]].median())


def write(rows, out):
    import pandas as pd

    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    {"b4": mode_b4, "b8": mode_b8, "eps": mode_eps, "cg": mode_cg}[sys.argv[1]](sys.argv[2])
