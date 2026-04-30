"""Non-LQ Mean Field Game on a 1D periodic grid.

A congestion MFG: agents want to be near a target position, but pay a
cost proportional to the local density (it's crowded there). The
running cost is

.. math::
    F(x, m) = \\tfrac{1}{2}\\, q\\, (x - x_\\mathrm{target})^2 + \\lambda\\, m,

with :math:`\\lambda > 0`. The first term pulls agents toward the
target; the second pushes them apart wherever the density spikes.
This is the standard Lasry-Lions monotone case for which Picard is
contracting, so the grid solver converges quickly. The LQ closed form
does not apply here, which is the whole point.

Three panels:
    Top-left:  density at three time slices for weak congestion
               (lambda = 0.1). The mass concentrates near the target.
    Top-right: density at three time slices for strong congestion
               (lambda = 1.0). The mass spreads out.
    Bottom:    terminal density m(T, x) for both cases overlaid,
               showing how the congestion term reshapes the equilibrium.

Usage:
    pip install matplotlib scipy
    python examples/mfg_grid_demo.py
"""

from __future__ import annotations

import numpy as np

from mvkit.mfg import MFGProblem, solve_mfg


def main() -> None:
    import matplotlib.pyplot as plt

    a, b = -np.pi, np.pi
    L = b - a
    n_x = 200
    n_t = 100
    sigma = 0.4
    T = 1.0
    q_quad = 1.0
    x_target = 0.0

    # Initial density: shifted Gaussian-like bump on the periodic
    # domain (Von Mises-ish).
    def initial_density(x):
        return (1.0 + 0.5 * np.cos(x + 1.0)) / L

    def make_problem(lam: float) -> MFGProblem:
        def running_cost(t, x, m):
            return 0.5 * q_quad * (x - x_target) ** 2 + lam * m

        def terminal_cost(x, m):
            return np.zeros_like(x)

        return MFGProblem(
            sigma=sigma, T=T, domain=(a, b), n_x=n_x,
            initial_density=initial_density,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
        )

    # Solve under weak vs strong congestion.
    cases = {
        "weak (lambda = 0.1)": (0.1, "tab:blue"),
        "strong (lambda = 1.0)": (1.0, "tab:red"),
    }
    sols = {}
    for label, (lam, _color) in cases.items():
        sol = solve_mfg(
            make_problem(lam), n_t=n_t, method="picard",
            tol=1e-6, n_iterations_max=80,
        )
        sols[label] = sol
        print(
            f"{label}: converged={sol.converged}, "
            f"n_iter={sol.n_iterations}"
        )

    # Time slices to visualize.
    slice_indices = [0, n_t // 2, n_t]

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    for j, (label, (lam, color)) in enumerate(cases.items()):
        ax = fig.add_subplot(gs[0, j])
        sol = sols[label]
        for k, idx in enumerate(slice_indices):
            ax.plot(
                sol.x_grid, sol.m[idx],
                color=plt.cm.viridis(0.15 + 0.7 * k / len(slice_indices)),
                lw=1.6,
                label=f"t = {sol.t_grid[idx]:.2f}",
            )
        ax.axvline(x_target, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("m(t, x)")
        ax.set_title(f"Density evolution: {label}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    # Bottom row: terminal density overlay.
    ax = fig.add_subplot(gs[1, :])
    for label, (lam, color) in cases.items():
        sol = sols[label]
        ax.plot(
            sol.x_grid, sol.m[-1], color=color, lw=2.0,
            label=f"m(T, x), {label}",
        )
    initial_grid = initial_density(sols[next(iter(cases))].x_grid)
    initial_grid = initial_grid / (initial_grid.sum() * (L / n_x))
    ax.plot(
        sols[next(iter(cases))].x_grid, initial_grid,
        color="black", lw=1.2, ls=":", alpha=0.6, label="m(0, x)",
    )
    ax.axvline(x_target, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Terminal-time densities side by side")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Non-LQ MFG: F(x, m) = (1/2) q (x - x_target)^2 + lambda m, "
        f"sigma = {sigma}, T = {T}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
