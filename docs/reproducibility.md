# Random streams and reproducibility

A seed identifies an experiment together with its model, inputs, numerical
method, discretization, implementation and environment. It is not a universal
identifier for the same normal samples across different generators.

## Native particle simulations

The Rust integrators use `Xoshiro256PlusPlus::seed_from_u64(seed)` and
`rand_distr::StandardNormal`. Sampling follows time, particle and coordinate
order, including coordinates whose diffusion is zero. Euler and Milstein use
the same normals for identical seeds and dimensions.

`Kuramoto::with_gaussian_omegas` advances the seeded generator using its
`jump()` before sampling frequencies. The jump separates its starting state
from the integrator by 2^128 raw generator steps. Passing the same seed to
the constructor and integrator therefore does not reuse the first normals.
This separation assumes neither stream exhausts that block. It is not a
general stream-allocation API for arbitrary user generators.

Supplying `increments` bypasses internal generation; these values are
standard normals, scaled by `sqrt(dt)` inside the integrator. The seed is
ignored on that path. `brownian.generate_increments` uses NumPy, so equal
seeds do not reproduce Rust's internal normals. Supply the same array when
comparing numerical methods on the same noise realization.

## LQ mean-field solvers

The scalar and vector LQ solvers use explicit NumPy PCG64 generators with
`SeedSequence(master_seed, spawn_key=(role,))`. Role identifiers are fixed:

| Role | Identifier |
|---|---:|
| Scalar initial population | 0 |
| Scalar dynamic noise | 1 |
| Vector initial population | 2 |
| Vector dynamic noise | 3 |

The role is not folded into the seed using an XOR mask. Distinct master seeds
therefore do not exchange initial and dynamic streams through the previous
mask construction. SeedSequence provides probabilistic stream separation;
this is not a proof of mathematical independence of finite deterministic
sequences. See [NumPy's stream construction](https://numpy.org/doc/stable/reference/random/parallel.html).

Outer iterations intentionally reuse the same initial population and dynamic
noise. Changing the fixed-point method or stopping tolerance does not draw a
new random experiment at every iteration.

## Scope of guarantees

- With the same build, environment, inputs and seed, built-in simulations are
  repeatable. The native built-in models use a fixed reduction/sampling order
  independent of the Rayon thread count. Custom models must preserve their
  own reduction order to make the same claim.
- No bitwise guarantee is made across different platforms, compilers,
  dependency versions or floating-point instruction choices. Numerical or
  statistical equivalence must be checked at the relevant tolerance.
- Changing particle counts, time grids or splitting a simulation into calls
  does not preserve per-particle stream identity. Calls restart their RNG;
  an API for serializing and continuing generator state is not yet exposed.
- Corrections can change trajectories for an existing seed. Record the
  package revision and environment alongside experimental results. Consult
  the changelog for changes to seed mappings and numerical reductions.

The current corrections change Kuramoto's generated frequencies and LQ-MFG
streams. Native integrator normal generation is unchanged. Compensated mean
reductions and the corrected CIR Milstein coefficient can change trajectory
rounding, and the CIR correction changes mathematically below the old floor.
