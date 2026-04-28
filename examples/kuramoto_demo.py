"""Kuramoto demo: phase trajectories and order parameter on either side
of the critical coupling.

Usage:
    pip install matplotlib
    python examples/kuramoto_demo.py

Three panels:
    Top-left:  theta_i(t) mod 2*pi for ~50 sample particles, sub-critical K.
    Top-right: same but super-critical K (visible bunching of phases).
    Bottom:    order parameter r(t) for both regimes on the same axes,
               with the predicted asymptote sqrt(1 - K_c / K) for the
               super-critical run shown as a dashed line.
"""

from __future__ import annotations

import numpy as np

from mvkit import simulate_kuramoto


def main() -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2026)
    n = 200
    omega_std = 0.5
    sigma = 0.05
    t_final = 30.0
    n_steps = 6000
    record_every = 30

    k_critical = 2.0 * omega_std * np.sqrt(2.0 / np.pi)
    k_sub = 0.4 * k_critical
    k_super = 4.0 * k_critical

    omegas = rng.normal(0.0, omega_std, size=n)
    x0 = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))

    hist_sub = simulate_kuramoto(
        x0,
        t_final,
        n_steps,
        coupling_k=k_sub,
        omegas=omegas,
        sigma=sigma,
        record_every=record_every,
        seed=1,
    )
    hist_super = simulate_kuramoto(
        x0,
        t_final,
        n_steps,
        coupling_k=k_super,
        omegas=omegas,
        sigma=sigma,
        record_every=record_every,
        seed=1,
    )

    n_frames = hist_sub.shape[0]
    times = np.linspace(0.0, t_final, n_frames)

    def order_param(history: np.ndarray) -> np.ndarray:
        z = np.exp(1j * history[..., 0])
        return np.abs(z.mean(axis=1))

    r_sub = order_param(hist_sub)
    r_super = order_param(hist_super)

    sample = np.linspace(0, n - 1, 50, dtype=int)

    fig, axd = plt.subplot_mosaic(
        [["sub", "sup"], ["r", "r"]],
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [1.0, 0.8]},
    )

    for i in sample:
        axd["sub"].plot(
            times,
            hist_sub[:, i, 0] % (2.0 * np.pi),
            color="tab:blue",
            alpha=0.4,
            lw=0.6,
        )
    axd["sub"].set_title(
        f"Sub-critical K = {k_sub:.2f} ($K_c \\approx$ {k_critical:.2f})"
    )
    axd["sub"].set_xlabel("t")
    axd["sub"].set_ylabel(r"$\theta_i(t) \bmod 2\pi$")
    axd["sub"].set_ylim(0.0, 2.0 * np.pi)

    for i in sample:
        axd["sup"].plot(
            times,
            hist_super[:, i, 0] % (2.0 * np.pi),
            color="tab:red",
            alpha=0.4,
            lw=0.6,
        )
    axd["sup"].set_title(f"Super-critical K = {k_super:.2f}")
    axd["sup"].set_xlabel("t")
    axd["sup"].set_ylabel(r"$\theta_i(t) \bmod 2\pi$")
    axd["sup"].set_ylim(0.0, 2.0 * np.pi)

    axd["r"].plot(
        times, r_sub, color="tab:blue", label=f"K = {k_sub:.2f} (sub-critical)"
    )
    axd["r"].plot(
        times,
        r_super,
        color="tab:red",
        label=f"K = {k_super:.2f} (super-critical)",
    )
    r_predicted = np.sqrt(max(0.0, 1.0 - k_critical / k_super))
    axd["r"].axhline(
        r_predicted,
        color="tab:red",
        linestyle="--",
        alpha=0.5,
        label=fr"predicted $r_\infty = \sqrt{{1 - K_c/K}} \approx {r_predicted:.2f}$",
    )
    axd["r"].set_xlabel("t")
    axd["r"].set_ylabel("r(t)")
    axd["r"].set_ylim(0.0, 1.0)
    axd["r"].set_title(
        f"Order parameter (Gaussian $\\omega \\sim N(0, {omega_std}^2)$, "
        f"$K_c \\approx$ {k_critical:.2f})"
    )
    axd["r"].legend(loc="center right")
    axd["r"].grid(alpha=0.3)

    fig.suptitle(
        f"Kuramoto, N = {n}, $\\sigma_\\omega$ = {omega_std}, sigma = {sigma}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
