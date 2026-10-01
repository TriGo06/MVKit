# Numerical guide

[Back to the README](../README.md)

Model equations, numerical assumptions and worked examples for `mvkit`.
For installation and the API overview, start with the README.

- [Linear-quadratic McKean-Vlasov](#linear-quadratic-mckean-vlasov)
- [Cucker-Smale](#cucker-smale)
- [Kuramoto](#kuramoto)
- [Mean-field CIR](#mean-field-cir)
- [Propagation of chaos](#propagation-of-chaos)
- [Shared Brownian paths](#strong-error-tests-via-shared-brownian-paths)
- [Mean Field Games](#mean-field-games)
- [References](#references)

The additive-noise particle models have the form

$$
\mathrm{d}X^{i,N}_t = b\!\left(X^{i,N}_t, \tfrac{1}{N}\sum_{j=1}^N \delta_{X^{j,N}_t}\right)\mathrm{d}t + \sigma\,\mathrm{d}W^i_t.
$$

Under suitable assumptions, propagation of chaos connects these interacting
systems to a McKean-Vlasov limit as the particle count grows. State-dependent
diagonal diffusion is also supported; the CIR model below is one example.
See the [random-stream contract](reproducibility.md) when comparing seeded
experiments.

## Linear-quadratic McKean-Vlasov

Each particle has scalar state with dynamics

$$
\mathrm{d}X_i = (a\, X_i + b\, \bar X)\,\mathrm{d}t + \sigma\,\mathrm{d}W_i,
$$

where $\bar X = (1/N)\sum_j X_j$. For i.i.d. Gaussian initial states, the mean is $m(t)=m_0\exp((a+b)t)$. Define $v_c(t)=v_0\exp(2ct)+\sigma^2(\exp(2ct)-1)/(2c)$, with $v_0+\sigma^2t$ when $c=0$. The McKean-Vlasov limit variance is $v_a(t)$. At finite $N$, the marginal particle variance is $(1-1/N)v_a(t)+v_{a+b}(t)/N$, while the expected empirical population variance is $(1-1/N)v_a(t)$. These distinctions matter when benchmarking discretization errors at finite particle counts.

## Cucker-Smale

Per particle the state is $(x_i, v_i) \in \mathbb{R}^{2d}$. The dynamics:

$$
\mathrm{d}x_i = v_i\,\mathrm{d}t,\qquad
\mathrm{d}v_i = \frac{1}{N}\sum_{j=1}^N K(|x_j - x_i|)(v_j - v_i)\,\mathrm{d}t + \sigma\,\mathrm{d}W_i,
$$

with kernel $K(r) = (1 + r^2)^{-\beta}$. The deterministic part conserves the mean velocity $\bar v = \frac{1}{N}\sum_i v_i$, used as a sanity check in the test suite. For $\sigma=0$ and $\beta < 1/2$, velocities concentrate around $\bar v$ without restrictions on the initial configuration (Cucker and Smale, 2007). This deterministic flocking statement does not apply unchanged with independent velocity noise.

## Kuramoto

Each particle has a scalar phase $\theta_i \in \mathbb{R}$ (left unwrapped during integration; the user can reduce mod $2\pi$ for visualization). The dynamics:

$$
\mathrm{d}\theta_i = \omega_i\,\mathrm{d}t + \frac{K}{N}\sum_{j=1}^N \sin(\theta_j - \theta_i)\,\mathrm{d}t + \sigma\,\mathrm{d}W_i,
$$

where $\omega_i$ are heterogeneous natural frequencies and $K$ is the coupling strength. Synchronization is captured by the Kuramoto order parameter

$$
r(t)\, e^{i\psi(t)} = \frac{1}{N}\sum_{j=1}^N e^{i\theta_j(t)},
$$

with $r \in [0, 1]$. For Gaussian $\omega_i \sim N(0, \sigma_\omega^2)$ without dynamic noise, the infinite-population critical coupling is $K_c = 2\sigma_\omega \sqrt{2/\pi}$. Below $K_c$ the population stays incoherent ($r \to 0$ as $N \to \infty$); above $K_c$ a fraction of oscillators lock and $r$ stabilizes between 0 and 1. The Lorentzian formula $r=\sqrt{1-K_c/K}$ does not apply to Gaussian frequencies. Nonzero dynamic noise also changes the synchronization threshold.

The drift is implemented in $O(N)$ per time step via the trig identity $\sum_j \sin(\theta_j - \theta_i) = S\cos(\theta_i) - C\sin(\theta_i)$ with $C = \sum_j \cos(\theta_j)$ and $S = \sum_j \sin(\theta_j)$: a sequential reduction over the phase column gives $C, S$, then a parallel-over-rows application sets each particle's drift in constant time.

Reference: Kuramoto, Y. (1975). *Self-entrainment of a population of coupled non-linear oscillators*. International Symposium on Mathematical Problems in Theoretical Physics.

See [`examples/kuramoto_demo.py`](../examples/kuramoto_demo.py) for a runnable visualization showing phase trajectories and $r(t)$ on either side of $K_c$.

## mean-field CIR

Each particle has scalar state $X_i \ge 0$ with dynamics

$$
\mathrm{d}X_i = \kappa(\theta - X_i)\,\mathrm{d}t + b\,(\bar X - X_i)\,\mathrm{d}t + \sigma\sqrt{\max(X_i, 0)}\,\mathrm{d}W_i,
$$

where $\bar X = (1/N)\sum_j X_j$. The first drift term is the standard CIR mean-reversion towards $\theta$ at rate $\kappa$; the second is the McKean-Vlasov interaction. The diffusion is square-root in the state, which is what makes Milstein non-trivial on this model: the diagonal Jacobian $(\sigma\sqrt{x})' = 0.5 \sigma / \sqrt{x}$ is non-zero, so the Milstein correction term $\tfrac{1}{2}\sigma\sigma'\,\mathrm{d}t (Z^2 - 1)$ fires and modifies trajectories pathwise.

The truncation $\max(X_i, 0)$ keeps the diffusion real if a discretization step crosses below zero. For non-negative interaction $b$ and strictly positive initial states, the Feller condition $2\kappa\theta \ge \sigma^2$ guarantees positivity of the continuous-time process. Euler and Milstein do not guarantee positivity of discrete states. The Python entry point requires finite non-negative initial states, and both APIs require finite non-negative $b$. For positive states the Milstein coefficient is evaluated directly as $\sigma^2\,dt/4$, without a derivative floor; it is zero for the truncated non-positive discrete extension.

For $b=0$, particles are independent classical CIR processes. For any admissible $b$, interactions cancel in the population sum, so $\mathbb{E}[\bar X_t] = \theta + (\bar X_0 - \theta)e^{-\kappa t}$. The mean-reversion rate of the population mean is $\kappa$, independent of $b$.

A note on Milstein's improvement. For compatible diffusion and sufficiently regular coefficients, Milstein has strong order 1 (vs Euler's strong order 1/2), but on weak error of smooth functionals of $X_T$ both schemes are order 1; the constants of the leading $O(\mathrm{d}t)$ terms can go either way depending on the functional, and on CIR the Milstein-only contribution to $E[X_{n+1}^2 \mid X_n]$ is $+\tfrac{1}{8}\sigma^4\,\mathrm{d}t^2$. This positive contribution may improve or worsen the absolute second-moment error, depending on the Euler bias. A higher strong order concerns pathwise approximation; it does not guarantee a smaller weak bias or a particular convergence rate for discontinuous path-dependent payoffs. CIR has a non-Lipschitz square-root coefficient, so its rate also depends on parameters and boundary behavior; the shared-increment tests below check selected regimes.

References: Cox, J. C., Ingersoll, J. E., and Ross, S. A. (1985). *A theory of the term structure of interest rates*. Econometrica 53, 385-407. McKean-Vlasov extensions are standard, see Carmona and Delarue (2018).

The Rust coordinatewise Milstein implementation requires `MeanFieldSDE::supports_milstein()` to opt in. This requires a correct self-derivative or direct `milstein_coefficient` override, and vanishing cross-noise derivatives across coordinates and particles. A diagonal diffusion matrix alone is insufficient; cross iterated stochastic integrals are not implemented. All four built-in models satisfy the structural condition.

## Propagation of chaos

Propagation of chaos connects an interacting particle system to its McKean-Vlasov limit under suitable assumptions on the dynamics. For **independent** 1D samples with a finite $(4+\varepsilon)$-th moment, Fournier and Guillin (2015) give $\mathbb{E}[W_2^2(\mu_N,\mu)]=O(N^{-1/2})$. Jensen's inequality yields the general bound $\mathbb{E}[W_2(\mu_N,\mu)]=O(N^{-1/4})$. A $-1/2$ slope for $W_2$ is not universal: Bernoulli samples have order $N^{-1/4}$. Interacting particles also require a model-specific coupling estimate.

`mvkit.poc` sweeps particle counts and fits the log-log slope of median W2 errors. Distances between two empirical measures use exact step-quantile integration, including unequal sample counts. Distances to a reference inverse CDF use adaptive quadrature with error tolerances on W2 squared. The squared transport cost on this reference-quadrature path must fit in float64. Known jumps or narrow features of the reference quantile must be supplied through `reference_breakpoints` (for example `[0.9999]` for a rare second atom of probability `1e-4`). An adaptive quadrature can otherwise miss a rare tail entirely, even while reporting zero estimated error; arbitrary inverse-CDF callbacks do not carry a certified error bound. A slope near $-1/2$ is an empirical benchmark for the Gaussian example below, not a consequence of the general moment bound.

```python
import numpy as np
from scipy.stats import norm

from mvkit import simulate_linear_quadratic
from mvkit.poc import estimate_propagation_of_chaos_rate

a, b, sigma, T = -0.5, 1.0, 0.5, 1.0
m_0, v_0 = 0.0, 1.0
m_T = m_0 * np.exp((a + b) * T)
v_T = v_0 * np.exp(2 * a * T) + sigma**2 * (np.exp(2 * a * T) - 1) / (2 * a)

def simulator(n, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(m_0, np.sqrt(v_0), size=(n, 1))
    h = simulate_linear_quadratic(x0, T, 1000, a=a, b=b, sigma=sigma, seed=seed)
    return h[-1, :, 0]

result = estimate_propagation_of_chaos_rate(
    simulator=simulator,
    reference_inv_cdf=norm(loc=m_T, scale=np.sqrt(v_T)).ppf,
    n_values=[100, 300, 1000, 3000, 10000],
    n_seeds=16,
)
print(result.fitted_slope)  # empirical fit; not a universal convergence rate
```

See [`examples/poc_rate_lq.py`](../examples/poc_rate_lq.py) for a runnable two-panel figure showing the log-log fit and the empirical-vs-analytical CDF overlay at the largest $N$.

Scope. Real-valued 1D samples only (for example LinearQuadratic and MeanFieldCIR). Phase angles require a separate circular-distance treatment. Cucker-Smale state has $2d$ coordinates (four with the default spatial dimension $d=2$). Full-state Wasserstein distances are not implemented here; coordinate projections measure a different quantity.

## Strong-error tests via shared Brownian paths

Every `simulate_*` function accepts an optional `increments` keyword: a 3D array of standard normals of shape `(n_steps, N, dim)` that the integrator uses in place of internal sampling. With it you can drive two simulations at different `n_steps` from the same Brownian path, which makes pathwise (strong) error well-defined. The `mvkit.brownian` module ships two helpers:

- `generate_increments(n_steps, n_particles, dim, seed)` for drawing a fresh fine grid.
- `coarsen(fine_increments, factor)` that aggregates onto a coarser grid via the variance-preserving bridge relation $Z^{\text{coarse}}_k = \tfrac{1}{\sqrt f}\sum_{j=0}^{f-1} Z^{\text{fine}}_{f k + j}$.

The following experiment estimates Euler's strong error in one CIR parameter regime, using a fine-grid Milstein trajectory as the reference:

```python
import numpy as np
from mvkit import simulate_mean_field_cir
from mvkit.brownian import generate_increments, coarsen

N, T, n_fine = 5000, 1.0, 4096
x0 = np.full((N, 1), 0.04)
Z_fine = generate_increments(n_steps=n_fine, n_particles=N, dim=1, seed=0)

ref = simulate_mean_field_cir(x0, T, n_fine, kappa=1.0, theta=0.04, b=0.0,
                              sigma=0.2, increments=Z_fine, scheme="milstein")
X_ref = ref[-1, :, 0]

errs, dts = [], []
for n_steps in [32, 64, 128, 256, 512, 1024]:
    Z_n = coarsen(Z_fine, factor=n_fine // n_steps)
    h = simulate_mean_field_cir(x0, T, n_steps, kappa=1.0, theta=0.04, b=0.0,
                                sigma=0.2, increments=Z_n, scheme="euler")
    errs.append(np.sqrt(np.mean((h[-1, :, 0] - X_ref) ** 2)))
    dts.append(T / n_steps)

slope, _ = np.polyfit(np.log(dts), np.log(errs), 1)
print(slope)   # about 0.5 in this experiment; parameter-dependent for CIR
```

See [`examples/strong_error_demo.py`](../examples/strong_error_demo.py) for the two-panel figure showing Euler vs Milstein on CIR.

## Mean Field Games

A Mean Field Game (Lasry and Lions, 2007) is a Cournot-Nash equilibrium for a continuum of identical agents: each agent chooses a control to minimize a personal cost that depends on the population's distribution, and at equilibrium the distribution generated by every agent's optimal response coincides with the input distribution. Numerically, finding an equilibrium amounts to solving a coupled forward-backward system: a Hamilton-Jacobi-Bellman PDE for the value function $u(t, x)$ backward from a terminal condition, and a Fokker-Planck PDE for the state distribution $\mu_t$ forward from the initial law.

`mvkit.mfg` provides scalar and vector linear-quadratic solvers and general-cost grid solvers in 1D and 2D. In the scalar LQ case, the HJB ansatz $u(t, x) = \tfrac{1}{2} P(t) x^2 + Q(t) x + R(t)$ reduces the problem to ODEs. Riccati and variance/covariance reference trajectories are integrated numerically with SciPy's Radau method, independently of the particle simulation. Closed-form scalar Riccati cases are used in regression tests.

All four equilibrium solvers accept the `method` keyword:

- `method="picard"` (default): $m^{(k+1)} = \mathrm{BR}(m^{(k)})$. Geometric convergence when the BR map is a contraction. The iteration count depends on the problem, discretization and tolerance.
- `method="fictitious_play"` (also exposed as `solve_lq_mfg_fictitious_play`): $m^{(k+1)}=\mathrm{BR}(\bar m^{(k)})$, where $\bar m^{(k)}$ is the historical average. It can be much slower than Picard. Cardaliaguet and Hadikhanloo (2017) prove convergence for potential MFGs under additional regularity assumptions; this is not an unconditional $O(1/k)$ sup-norm error bound. Lasry-Lions monotonicity alone does not make the best-response map contractive.

The `damping_burn_in` parameter for Fictitious Play runs leading Picard steps before starting to accumulate the historical average, which can improve convergence depending on the problem and initial guess.

### Picking an MFG solver

| Use | Function | Numerical approach |
|---|---|---|
| Scalar LQ costs | `solve_lq_mfg` | Particle response iteration; scalar Riccati and variance ODEs |
| Vector LQ costs | `solve_lq_mfg_vector` | Particle response iteration; matrix Riccati and covariance ODEs |
| General costs, 1D state | `solve_mfg` | HJB and Fokker-Planck grid |
| General costs, 2D state | `solve_mfg_2d` | HJB and Fokker-Planck grid |

All functions are exported by `mvkit.mfg`. The vector LQ solver accepts full
cost and diffusion matrices. Its covariance reference includes cross-coordinate
correlations from non-diagonal diffusion matrices. LQ particle solvers avoid
spatial grids, but retain time-discretization and sampling errors. Grid
regression tests compare against LQ ODE references under refinement.

```python
import numpy as np
from mvkit.mfg import solve_lq_mfg

sol = solve_lq_mfg(
    q=1.0, q_T=0.5, sigma=0.5, T=1.0,
    mu_0_mean=1.5, mu_0_var=1.0,
    n_particles=10_000, n_grid=200, seed=0,
)
print("converged:", sol.converged, "n_iterations:", sol.n_iterations)
print("max |m - m_0|:", np.max(np.abs(sol.m - 1.5)))   # equilibrium mean is constant

# Compare empirical terminal variance to the independent ODE reference V(T).
V_T_emp = sol.x_trajectory[-1].var()
print(f"V(T) empirical = {V_T_emp:.4f}, ODE reference = {sol.V[-1]:.4f}")
```

For an arbitrary input mean, the affine HJB coefficient must be solved backward: $s'=Ps+qm$, $s(T)=-q_Tm(T)$, and $\alpha^*=-Px-s$. In the vector case, $s'=PR^{-1}s+Qm$ and $\alpha^*=-R^{-1}(Px+s)$. The shortcut $s=-Pm$ only applies to constant input means.

All four MFG solvers report `fixed_point_residual = ||BR(m)-m||_inf` for the returned mean or density. `converged` is true only when this residual is below `tol`. Small changes between successive averaged responses are insufficient. Grid solutions evaluate the returned value function and control against the returned density. The explicit HJB and transport steps enforce CFL bounds of one; increase `n_t` if the guard rejects a step. Neumann grids use cell centers.

The LQ reduction follows Carmona and Delarue (2018), *Probabilistic Theory of Mean Field Games with Applications I*, Section 3.5. The Fictitious Play scheme follows Cardaliaguet and Hadikhanloo (2017). See [`examples/mfg_lq_demo.py`](../examples/mfg_lq_demo.py) for a three-panel figure ($P(t)$, ODE-reference versus empirical variance, particle trajectories with the equilibrium mean overlaid) and [`examples/mfg_lq_picard_vs_fp.py`](../examples/mfg_lq_picard_vs_fp.py) for a side-by-side log-y convergence trace of both methods.

### Grid methods and current limits

The HJB solver uses an Engquist-Osher upwind Hamiltonian and implicit diffusion.
The forward Fokker-Planck solver uses conservative upwind transport and implicit
diffusion. Both 1D and 2D grids support periodic boundaries and Neumann
boundaries: zero normal value-function derivative and zero probability flux.
The 2D diffusion may be isotropic or anisotropic via `sigma=(sigma_x, sigma_y)`.

The 1D `solve_hjb` and `MFGProblem` accept a `Hamiltonian` with vectorized
`H` and `dH` callables. It must be convex, minimized at zero, and satisfy
`H(0) = dH(0) = 0`. `power_hamiltonian(q)` supplies `H(p) = |p|^q / q`
for `q > 1`. The runtime checks sample gradients; they do not prove that
an arbitrary callback satisfies the contract.

The 2D grid solver currently uses a quadratic Hamiltonian. Higher-dimensional
grid solvers are not provided. The standalone `solve_hjb`,
`solve_fokker_planck`, `solve_hjb_2d` and `solve_fokker_planck_2d` functions
are also exported by `mvkit.mfg`.

Quadratic HJB tests use the Hopf-Cole transformation as a reference.
Fokker-Planck tests check Fourier-mode transport/diffusion and mass
conservation; coupled grid tests compare with LQ references in periodic
and Neumann settings, including anisotropic 2D diffusion.

Development priorities are tracked in the [performance roadmap](performance-roadmap.md).

## References

- Cucker, F. and Smale, S. (2007). *Emergent behavior in flocks*. IEEE Trans. Automatic Control.
- Sznitman, A.-S. (1991). *Topics in propagation of chaos*. Ecole d'Eté de Probabilités de Saint-Flour XIX.
- Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications I & II*. Springer.
- Lasry, J.-M. and Lions, P.-L. (2007). *Mean field games*. Japanese Journal of Mathematics 2, 229-260.
- Cardaliaguet, P. and Hadikhanloo, S. (2017). *Learning in mean field games: the fictitious play*. ESAIM: Control, Optimisation and Calculus of Variations 23, 569-591.
