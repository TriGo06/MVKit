"""Strong-error rate of Euler-Maruyama vs Milstein on mean-field CIR.

The Brownian-increment hook on every ``simulate_*`` function lets you
drive coarse and fine simulations from the **same Brownian path**. Doing
so makes pathwise (strong) error well-defined, and recovers the textbook
rates: Euler is strong order 1/2 on multiplicative-noise SDEs, Milstein
is strong order 1.

Two panels:
    Left:  log-log plot of strong error vs ``dt`` for both schemes, with
           least-squares slope fits and reference slope lines.
    Right: ratio Milstein/Euler vs ``dt``, showing how much Milstein wins
           as the grid is refined.

Usage:
    pip install matplotlib
    python examples/strong_error_demo.py
"""

from __future__ import annotations

import numpy as np

from mvkit import simulate_mean_field_cir
from mvkit.brownian import coarsen, generate_increments


def main() -> None:
    import matplotlib.pyplot as plt

    N, T = 5000, 1.0
    kappa, theta, b, sigma = 1.0, 0.04, 0.0, 0.2
    x0 = np.full((N, 1), 0.04)
    n_fine_ref = 4096
    n_steps_grid = [32, 64, 128, 256, 512, 1024]

    Z_finest = generate_increments(
        n_steps=n_fine_ref, n_particles=N, dim=1, seed=0
    )

    def reference(scheme: str) -> np.ndarray:
        hist = simulate_mean_field_cir(
            x0, t_final=T, n_steps=n_fine_ref,
            kappa=kappa, theta=theta, b=b, sigma=sigma,
            increments=Z_finest, scheme=scheme,
        )
        return hist[-1, :, 0]

    X_ref_e = reference("euler")
    X_ref_m = reference("milstein")

    def strong_errors(scheme: str, X_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errs, dts = [], []
        for n_steps in n_steps_grid:
            factor = n_fine_ref // n_steps
            Z_n = coarsen(Z_finest, factor=factor)
            hist = simulate_mean_field_cir(
                x0, t_final=T, n_steps=n_steps,
                kappa=kappa, theta=theta, b=b, sigma=sigma,
                increments=Z_n, scheme=scheme,
            )
            err = float(np.sqrt(np.mean((hist[-1, :, 0] - X_ref) ** 2)))
            errs.append(err)
            dts.append(T / n_steps)
        return np.array(dts), np.array(errs)

    dts_e, errs_e = strong_errors("euler", X_ref_e)
    dts_m, errs_m = strong_errors("milstein", X_ref_m)

    slope_e, intercept_e = np.polyfit(np.log(dts_e), np.log(errs_e), 1)
    slope_m, intercept_m = np.polyfit(np.log(dts_m), np.log(errs_m), 1)

    print(f"Euler-Maruyama strong-order fit: slope = {slope_e:.3f} (theory 0.5)")
    print(f"Milstein         strong-order fit: slope = {slope_m:.3f} (theory 1.0)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.loglog(dts_e, errs_e, "o-", color="tab:blue",
              label=f"Euler-Maruyama (slope {slope_e:.2f})")
    ax.loglog(dts_m, errs_m, "s-", color="tab:orange",
              label=f"Milstein (slope {slope_m:.2f})")
    ref_half = errs_e[0] * (dts_e / dts_e[0]) ** 0.5
    ref_one = errs_m[0] * (dts_m / dts_m[0]) ** 1.0
    ax.loglog(dts_e, ref_half, "k--", alpha=0.4, label="slope 0.5")
    ax.loglog(dts_m, ref_one, "k:", alpha=0.5, label="slope 1.0")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("strong error")
    ax.set_title("Strong error vs grid spacing on mean-field CIR")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.loglog(dts_e, errs_m / errs_e, "d-", color="tab:purple")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel("Milstein / Euler error")
    ax.set_title("Milstein advantage over Euler")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"CIR(kappa={kappa}, theta={theta}, sigma={sigma}), N={N}, T={T}",
        fontsize=11,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
