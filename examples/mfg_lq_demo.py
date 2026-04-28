"""Linear-quadratic Mean Field Game demo.

Three panels:
    Top-left:  Riccati solution P(t) and the equilibrium control offset
               Q(t) = -P(t) m_t on twin y-axes.
    Top-right: Analytical variance V(t) (red dashed) overlaid with the
               empirical particle variance from the converged simulation
               (blue dots) with 2-sigma Monte Carlo error bars.
    Bottom:    A sample of particle trajectories X_i(t) under the
               equilibrium control, with the mean trajectory m_t in
               solid black.

Usage:
    pip install matplotlib scipy
    python examples/mfg_lq_demo.py
"""

from __future__ import annotations

import numpy as np

from mvkit.mfg import solve_lq_mfg


def main() -> None:
    import matplotlib.pyplot as plt

    q, q_T, sigma, T = 1.0, 0.5, 0.5, 1.0
    mu_0_mean, mu_0_var = 1.5, 0.5
    n_particles = 5000
    n_grid = 400

    sol = solve_lq_mfg(
        q=q,
        q_T=q_T,
        sigma=sigma,
        T=T,
        mu_0_mean=mu_0_mean,
        mu_0_var=mu_0_var,
        n_particles=n_particles,
        n_grid=n_grid,
        tol=1e-4,
        seed=2026,
    )

    print(
        f"Picard converged: {sol.converged}, "
        f"n_iterations: {sol.n_iterations}"
    )

    t = sol.t_grid
    Q = -sol.P * sol.m

    emp_var = sol.x_trajectory.var(axis=1, ddof=0)
    mc_2sigma = 2.0 * sol.V * np.sqrt(2.0 / n_particles)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    # Top-left: P(t) and Q(t) on twin axes.
    ax0 = fig.add_subplot(gs[0, 0])
    line_P = ax0.plot(t, sol.P, color="tab:blue", lw=1.8, label="P(t)")
    ax0.set_xlabel("t")
    ax0.set_ylabel("P(t)", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue")
    ax0.grid(alpha=0.3)
    ax0_r = ax0.twinx()
    line_Q = ax0_r.plot(
        t, Q, color="tab:orange", lw=1.5, ls="--", label="Q(t)"
    )
    ax0_r.set_ylabel("Q(t)", color="tab:orange")
    ax0_r.tick_params(axis="y", labelcolor="tab:orange")
    lines = line_P + line_Q
    ax0.legend(lines, [ln.get_label() for ln in lines], loc="best")
    ax0.set_title("Riccati P(t) and equilibrium offset Q(t) = -P(t) m(t)")

    # Top-right: V analytical vs empirical.
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(
        t, sol.V, color="tab:red", lw=1.8, ls="--",
        label="V(t) analytical",
    )
    sample_idx = np.linspace(0, len(t) - 1, 30, dtype=int)
    ax1.errorbar(
        t[sample_idx],
        emp_var[sample_idx],
        yerr=mc_2sigma[sample_idx],
        fmt="o",
        color="tab:blue",
        markersize=4,
        elinewidth=0.8,
        capsize=2,
        alpha=0.85,
        label="empirical (95% MC band)",
    )
    ax1.set_xlabel("t")
    ax1.set_ylabel("V(t)")
    ax1.set_title("Variance trajectory")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bottom: sample of trajectories with m(t) overlaid.
    ax2 = fig.add_subplot(gs[1, :])
    rng = np.random.default_rng(0)
    chosen = rng.choice(n_particles, size=30, replace=False)
    for j in chosen:
        ax2.plot(t, sol.x_trajectory[:, j], color="tab:gray", lw=0.6, alpha=0.6)
    ax2.plot(t, sol.m, color="black", lw=2.2, label="m(t) (empirical)")
    ax2.axhline(
        mu_0_mean, color="tab:red", lw=1.0, ls=":",
        label=f"m_0 = {mu_0_mean}",
    )
    ax2.set_xlabel("t")
    ax2.set_ylabel("X(t)")
    ax2.set_title("Particle trajectories under the equilibrium control")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"LQ-MFG: q={q}, q_T={q_T}, sigma={sigma}, T={T}, "
        f"mu_0=N({mu_0_mean}, {mu_0_var}), N={n_particles}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
