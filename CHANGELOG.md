# Changelog

## Unreleased

### Correctness

- Separate Rust Kuramoto frequency generation from the integrator stream by
  a Xoshiro jump; identical constructor/integrator seeds no longer reuse noise.
- Replace LQ-MFG XOR seed masks with fixed SeedSequence role identifiers and
  explicit PCG64 streams. Initial/dynamic streams no longer swap between the
  master seeds implicated by the old masks.
- Evaluate the CIR Milstein coefficient directly for positive states instead
  of flooring its diffusion derivative. Keep the truncated coefficient zero
  for non-positive discrete states.
- Require non-negative finite CIR interaction and finite non-negative Python
  initial states. The numerical schemes still do not preserve positivity.
- Use deterministic block reductions for population means, with compensated
  cancellation and scaled overflow fallbacks in the linear and CIR models;
  skip unused means at zero coupling.
- Accept `reference_breakpoints` for reference-quantile Wasserstein integration
  and rate estimation. Known rare atoms can now be integrated explicitly;
  quadrature of an arbitrary opaque quantile remains uncertified.
- Scale empirical Wasserstein norms before squaring, reject multivariate
  samples, and report distances outside the float64 range explicitly.

### Compatibility

Kuramoto frequencies and LQ-MFG trajectories change for existing seeds.
Native integrator normal samples retain their existing mapping. Compensated
reductions and the CIR fix can change trajectory values. See
[the reproducibility contract](docs/reproducibility.md).

Custom Rust models keep the existing Milstein derivative path by default.
They may optionally implement `milstein_coefficient` to evaluate a regular
product without constructing a singular derivative. The cross-noise
restrictions of coordinatewise Milstein still apply.
