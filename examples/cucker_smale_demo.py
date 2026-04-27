"""Cucker-Smale flocking demo: visualize convergence to consensus.

Usage:
    pip install matplotlib
    python examples/cucker_smale_demo.py
"""

from __future__ import annotations

import numpy as np

from mvkit import simulate_cucker_smale


def main() -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2026)
    n = 200
    d = 2

    x0 = np.empty((n, 2 * d))
    x0[:, :d] = rng.normal(scale=2.0, size=(n, d))
    x0[:, d:] = rng.normal(scale=1.5, size=(n, d))

    history = simulate_cucker_smale(
        x0,
        t_final=10.0,
        n_steps=2000,
        spatial_dim=d,
        beta=0.4,
        sigma=0.05,
        record_every=20,
        seed=0,
    )
    times = np.linspace(0.0, 10.0, history.shape[0])

    # Velocity dispersion over time.
    velocities = history[..., d:]
    var_t = velocities.var(axis=1).sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(times, var_t, lw=1.6)
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("velocity dispersion (sum of variances)")
    axes[0].set_title("Convergence to consensus")
    axes[0].grid(alpha=0.3)

    snapshots = [0, len(times) // 4, len(times) // 2, len(times) - 1]
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(snapshots)))
    for k, color in zip(snapshots, colors):
        axes[1].scatter(
            history[k, :, 0],
            history[k, :, 1],
            s=8,
            color=color,
            alpha=0.7,
            label=f"t = {times[k]:.1f}",
        )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Particle positions")
    axes[1].legend()
    axes[1].set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        f"Cucker-Smale flocking, N = {n}, beta = 0.4, sigma = 0.05",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
