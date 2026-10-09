"""Step 1, exact check: aligned (decoupled) init under an aligned anisotropic covariance.

Derived (notes/data_teacher.md 1.1): Pi_r(t; Sigma) = Pi_r^white(Lambda_r t; s_r), i.e. each
component follows the whitened trajectory with its own clock. Compares the TTN gradient flow
(pilot integrator) with the scalar ODE, and with the naive substitution s -> Lambda s.
Teacher: s = (1, 0.7, 0.5) placed on eigenvectors (e3, e2, e1) of a power-law site covariance
(alpha = 0.5, d = 4), so that Lambda s and Lambda s^2 both reverse the order of s.

Usage: uv run python experiments/data_teacher/e1_exact_aligned.py   (runs if needed, then plots)
"""

from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT_ROOT, aligned_setup, make_spec  # noqa: E402
from pilots.pilot_gf import run_gf  # noqa: E402

from ttn.network import aligned_init  # noqa: E402

HERE = Path(__file__).parent
N, D, CHI, ALPHA, A0 = 4, 4, 4, 0.5, 0.05
WEIGHTS, EIG = [1.0, 0.7, 0.5], [2, 1, 0]


def scalar_ode(s, lam, k, a0, t):
    """Balanced per-core amplitude: da/dt = Lambda (s - a^K) a^(K-1); returns Pi = a^K at t."""
    sol = solve_ivp(lambda _t, a: lam * (s - a**k) * a ** (k - 1), (0, t[-1]), [a0],
                    t_eval=t, rtol=1e-12, atol=1e-16, method="DOP853")
    return sol.y[0] ** k


def half_and_width(t, a):
    """t(a = 1/2) and the 10-90% width (linear interpolation)."""
    def cross(level):
        k = np.nonzero(a >= level)[0][0]
        return t[k - 1] + (level - a[k - 1]) * (t[k] - t[k - 1]) / (a[k] - a[k - 1])
    return cross(0.5), cross(0.9) - cross(0.1)


def run(tree, variant):
    path = OUT_ROOT / "e1_exact" / f"{tree}_{variant}.npz"
    if path.exists():
        return dict(np.load(path))
    torch.set_num_threads(1)
    spec = make_spec(tree, N, D, CHI)
    teacher, metrics, big_lam = aligned_setup(N, D, WEIGHTS, EIG, ALPHA, variant)
    amps = torch.full((spec.n_cores, len(WEIGHTS)), A0, dtype=torch.float64)
    cores0 = aligned_init(spec, teacher.factors, amps)
    t_max = 200.0 if spec.n_cores == 3 else 2000.0
    rec = run_gf(spec, cores0, teacher, metrics, t_max=t_max, n_rec=3000,
                 w_star=teacher.dense(), keep_y=False, record_spectra=False)
    out = dict(t=rec["t"], overlaps=rec["overlaps"], loss=rec["loss"], big_lam=big_lam,
               weights=teacher.weights.numpy(), K=spec.n_cores, q_drift=rec["q_drift"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    rows = []
    for i, tree in enumerate(["binary", "mps"]):
        res = run(tree, "sigma")
        sub = run(tree, "white_Ls")
        t, k, lam = res["t"], int(res["K"]), res["big_lam"]
        colors = ["C0", "C1", "C2"]
        for r, s in enumerate(WEIGHTS):
            theory = scalar_ode(s, lam[r], k, A0, t) / s
            err = np.max(np.abs(res["overlaps"][:, r] - theory))
            rows.append((tree, r, s, lam[r], err, res["q_drift"]))
            lab = f"s={s}, Lambda={lam[r]:.2f}"
            axes[i, 0].plot(t, res["overlaps"][:, r], color=colors[r], label=lab)
            axes[i, 0].plot(t, theory, "k:", lw=1)
            axes[i, 1].plot(lam[r] * t, res["overlaps"][:, r], color=colors[r], label=lab)
            white = scalar_ode(s, 1.0, k, A0, lam[r] * t) / s
            axes[i, 1].plot(lam[r] * t, white, "k:", lw=1)
            # Naive substitution s -> Lambda s: whitened run with weights Lambda s, as a fraction.
            # Compared in linear time around each event: widths differ by Lambda^(1-2/K)-type factors.
            th, w = half_and_width(t, res["overlaps"][:, r])
            th_sub, w_sub = half_and_width(sub["t"], sub["overlaps"][:, r])
            th_pred, w_pred = half_and_width(t, theory)
            rows[-1] += (th, th_pred, th_sub, w, w_pred, w_sub)
            axes[i, 2].plot(t - th, res["overlaps"][:, r], color=colors[r], label=f"Sigma, {lab}")
            axes[i, 2].plot(sub["t"] - th_sub, sub["overlaps"][:, r], "--", color=colors[r],
                            label="I, weight Lambda s")
        axes[i, 0].set_title(f"{tree}-4 (K={k}): a_r(t) (dotted: scalar ODE)")
        axes[i, 1].set_title("vs Lambda_r t (dotted: whitened ODE, weight s_r)")
        axes[i, 2].set_title("Sigma (solid) vs substitution s -> Lambda s (dashed)")
        for ax in axes[i][:2]:
            ax.set_xscale("log")
        for ax in axes[i]:
            ax.set_ylabel("a_r")
        axes[i, 2].set_xlim(-3 if k == 3 else -1, 3 if k == 3 else 1)
        axes[i, 0].legend(fontsize=7)
        axes[i, 2].legend(fontsize=6)
        axes[i, 0].set_xlabel("t")
        axes[i, 1].set_xlabel("Lambda_r t")
        axes[i, 2].set_xlabel("t - t_half (linear)")
    fig.tight_layout()
    fig.savefig(HERE / "results" / "e1_exact_aligned.png", dpi=110)
    with open(HERE / "results" / "e1_exact_aligned.csv", "w") as f:
        f.write("tree,component,s,Lambda,max_abs_dev_from_scalar_ode,q_drift,t_half,t_half_ode,"
                "t_half_subst,width,width_ode,width_subst\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
