"""Propagation-of-chaos rate demo on the linear-quadratic model.

Two panels:
    Left:  log-log W2 vs N with shaded IQR band, the least-squares fit in
           solid, and a slope -1/2 reference line in dashed.
    Right: empirical CDF of the terminal sample at the largest N, overlaid
           with the closed-form Gaussian CDF.

Usage:
    pip install matplotlib scipy
    python examples/poc_rate_lq.py

The -1/2 guide is an empirical Gaussian benchmark on this N range.
The general Fournier-Guillin moment theorem bounds E[W2**2] at that
order, not E[W2]. Other laws and interacting models can have slower rates.
"""

from __future__ import annotations

import numpy as np

from mvkit import simulate_linear_quadratic
from mvkit.poc import estimate_propagation_of_chaos_rate


def main() -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    a, b, sigma = -0.5, 1.0, 0.5
    t_final = 1.0
    n_steps = 1000
    m_0, v_0 = 0.0, 1.0

    m_T = m_0 * np.exp((a + b) * t_final)
    v_T = v_0 * np.exp(2 * a * t_final) + sigma**2 * (
        np.exp(2 * a * t_final) - 1
    ) / (2 * a)
    sd_T = np.sqrt(v_T)
    reference = norm(loc=m_T, scale=sd_T)

    def simulator(n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x0 = rng.normal(m_0, np.sqrt(v_0), size=(n, 1))
        history = simulate_linear_quadratic(
            x0, t_final, n_steps, a=a, b=b, sigma=sigma, seed=seed
        )
        return history[-1, :, 0]

    n_values = [100, 300, 1_000, 3_000, 10_000]
    result = estimate_propagation_of_chaos_rate(
        simulator=simulator,
        reference_inv_cdf=reference.ppf,
        n_values=n_values,
        n_seeds=16,
    )

    n_arr = result.n_values.astype(float)
    fit_line = np.exp(result.fitted_intercept) * n_arr ** result.fitted_slope
    ref_slope = -0.5
    ref_line = result.w2_median[0] * (n_arr / n_arr[0]) ** ref_slope

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.fill_between(
        n_arr,
        result.w2_q25,
        result.w2_q75,
        color="tab:blue",
        alpha=0.2,
        label="IQR over seeds",
    )
    ax.plot(
        n_arr, result.w2_median, "o", color="tab:blue", label="median W2"
    )
    ax.plot(
        n_arr,
        fit_line,
        "-",
        color="tab:blue",
        label=f"fit, slope = {result.fitted_slope:.3f}",
    )
    ax.plot(
        n_arr,
        ref_line,
        "--",
        color="tab:red",
        alpha=0.7,
        label="slope $-1/2$ reference",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of particles)")
    ax.set_ylabel(r"$W_2(\mu_N, \mu_T)$")
    ax.set_title(
        "Propagation of chaos rate on LinearQuadratic\n"
        f"(a={a}, b={b}, sigma={sigma}, T={t_final})"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    n_show = max(n_values)
    sample = simulator(n_show, seed=0)
    sorted_sample = np.sort(sample)
    empirical_cdf = (np.arange(n_show) + 0.5) / n_show
    grid = np.linspace(sorted_sample.min(), sorted_sample.max(), 400)
    ax.plot(
        sorted_sample,
        empirical_cdf,
        color="tab:blue",
        lw=1.0,
        label=f"empirical CDF (N={n_show})",
    )
    ax.plot(
        grid,
        reference.cdf(grid),
        "--",
        color="tab:red",
        label=f"closed-form $N({m_T:.2f}, {sd_T:.2f}^2)$",
    )
    ax.set_xlabel("X_T")
    ax.set_ylabel("CDF")
    ax.set_title("Empirical vs analytical CDF at the largest N")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
