"""Picard vs Fictitious Play on the LQ Mean Field Game.

Two panels:
    Left:  log-y plot of the iterate increment ``||m^(k+1) - m^(k)||_inf``
           versus iteration ``k``, for both methods. Picard's geometric
           decay and the behavior of averaging can be compared. Small
           increments alone do not establish a small fixed-point residual.
    Right: distance to the analytical equilibrium ``||m^(k) - m_0||_inf``
           versus iteration ``k`` (recall that in the symmetric LQ case
           the equilibrium mean is the constant ``m_0``).

Usage:
    pip install matplotlib scipy
    python examples/mfg_lq_picard_vs_fp.py
"""

from __future__ import annotations

import numpy as np

from mvkit.mfg import solve_lq_mfg, solve_lq_mfg_fictitious_play


def main() -> None:
    import matplotlib.pyplot as plt

    q, q_T, sigma, T = 2.0, 1.0, 0.3, 2.0
    mu_0_mean, mu_0_var = 0.0, 0.5
    n_particles = 20000
    n_grid = 200
    seed = 0

    sol_p = solve_lq_mfg(
        q=q, q_T=q_T, sigma=sigma, T=T,
        mu_0_mean=mu_0_mean, mu_0_var=mu_0_var,
        n_particles=n_particles, n_grid=n_grid,
        n_iterations_max=30, tol=1e-5, seed=seed,
        method="picard",
    )
    sol_fp = solve_lq_mfg_fictitious_play(
        q=q, q_T=q_T, sigma=sigma, T=T,
        mu_0_mean=mu_0_mean, mu_0_var=mu_0_var,
        n_particles=n_particles, n_grid=n_grid,
        n_iterations_max=120, tol=1e-5, seed=seed,
    )

    print(
        f"Picard: converged={sol_p.converged}, "
        f"n_iterations={sol_p.n_iterations}, residual={sol_p.fixed_point_residual:.3e}"
    )
    print(
        f"FP:     converged={sol_fp.converged}, "
        f"n_iterations={sol_fp.n_iterations}, residual={sol_fp.fixed_point_residual:.3e}"
    )

    def increments(iterates: list[np.ndarray]) -> np.ndarray:
        return np.array([
            float(np.max(np.abs(iterates[k + 1] - iterates[k])))
            for k in range(len(iterates) - 1)
        ])

    def distances_to_equilibrium(iterates: list[np.ndarray]) -> np.ndarray:
        return np.array([
            float(np.max(np.abs(arr - mu_0_mean))) for arr in iterates
        ])

    inc_p = increments(sol_p.m_iterates)
    inc_fp = increments(sol_fp.m_iterates)
    dist_p = distances_to_equilibrium(sol_p.m_iterates)
    dist_fp = distances_to_equilibrium(sol_fp.m_iterates)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(inc_p) + 1), inc_p, "o-", color="tab:blue",
                label=f"Picard ({sol_p.n_iterations} iters)")
    ax.semilogy(np.arange(1, len(inc_fp) + 1), inc_fp, "s-", color="tab:orange",
                label=f"Fictitious Play ({sol_fp.n_iterations} iters)")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|m^{(k+1)} - m^{(k)}\|_\infty$")
    ax.set_title("Iterate increment (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.semilogy(np.arange(len(dist_p)), dist_p, "o-", color="tab:blue",
                label="Picard")
    ax.semilogy(np.arange(len(dist_fp)), dist_fp, "s-", color="tab:orange",
                label="Fictitious Play")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|m^{(k)} - m_0\|_\infty$")
    ax.set_title("Distance to analytical equilibrium")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.suptitle(
        f"LQ-MFG, Picard vs Fictitious Play. q={q}, q_T={q_T}, "
        f"sigma={sigma}, T={T}, N={n_particles}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
