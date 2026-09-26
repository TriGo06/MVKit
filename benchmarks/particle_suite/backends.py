"""Optional Python adapters. All return initial and final states in float64."""

import time

import numpy as np

from .cases import numpy_solve


def prepare(name, case, x0, z):
    metadata = {"device": "cpu", "input_transfer_s": 0.0, "interface": "Python"}
    if name == "numpy":
        return lambda: numpy_solve(case, x0, z), {**metadata, "backend_version": np.__version__}
    if name == "mvkit-python":
        import mvkit

        if case.name == "multiplicative":
            raise NotImplementedError("Custom multidimensional diffusion is currently Rust-only")
        if case.name == "moments":
            solve = lambda: mvkit.simulate_linear_quadratic(
                x0, case.t_final, case.steps, a=case.a, b=case.b,
                sigma=case.sigma, increments=z, record_every=0,
            )
        else:
            solve = lambda: mvkit.simulate_cucker_smale(
                x0, case.t_final, case.steps, spatial_dim=case.dim,
                beta=case.beta, sigma=case.sigma, increments=z, record_every=0,
            )
        return solve, {**metadata, "backend_version": mvkit.__version__}
    if name == "numba":
        import numba

        solver = _numba_solver(case, numba)
        return lambda: solver(x0, z), {
            **metadata, "backend_version": numba.__version__, "threads": numba.get_num_threads(),
        }
    if name == "diffrax":
        import diffrax
        import jax
        import jax.numpy as jnp
        import lineax

        jax.config.update("jax_enable_x64", True)
        start = time.perf_counter()
        x_device, z_device = jnp.asarray(x0), jnp.asarray(z)
        jax.block_until_ready((x_device, z_device))
        transfer = time.perf_counter() - start
        dt = case.t_final / case.steps

        def drift(t, flat, args):
            x = flat.reshape(case.n, case.width)
            if case.name != "pairwise":
                value = case.a * x + case.b * jnp.mean(x, axis=0)
            else:
                d = case.dim

                def one(row):
                    r2 = jnp.sum((x[:, :d] - row[:d]) ** 2, axis=1)
                    weights = (1 + r2) ** (-case.beta)
                    dv = jnp.mean(weights[:, None] * (x[:, d:] - row[d:]), axis=0)
                    return jnp.concatenate((row[d:], dv))

                value = jax.vmap(one)(x)
            return value.reshape(-1)

        def diffusion(t, flat, args):
            if case.name == "multiplicative":
                diagonal = case.sigma * flat
            elif case.name == "pairwise":
                row = jnp.concatenate((jnp.zeros(case.dim), jnp.full(case.dim, case.sigma)))
                diagonal = jnp.tile(row, case.n)
            else:
                diagonal = jnp.full_like(flat, case.sigma)
            return lineax.DiagonalLinearOperator(diagonal)

        @jax.jit
        def solver(x, noise):
            # This control is intentionally restricted to the fixed benchmark grid.
            # StepTo guarantees that no interpolation or resampling is involved.
            def increment(t0, t1):
                index = jnp.rint(t0 / dt).astype(jnp.int32)
                return noise[index].reshape(-1) * jnp.sqrt(dt)

            terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, increment))
            sol = diffrax.diffeqsolve(
                terms, diffrax.Euler(), t0=0.0, t1=case.t_final, dt0=None,
                y0=x.reshape(-1), saveat=diffrax.SaveAt(t0=True, t1=True),
                stepsize_controller=diffrax.StepTo(jnp.linspace(0, case.t_final, case.steps + 1)),
                max_steps=case.steps,
            )
            return sol.ys.reshape(2, case.n, case.width)

        def solve():
            return jax.block_until_ready(solver(x_device, z_device))

        return solve, {
            **metadata, "backend_version": diffrax.__version__, "jax_version": jax.__version__,
            "device": str(jax.devices()[0]), "input_transfer_s": transfer,
            "float64": bool(jax.config.jax_enable_x64),
        }
    raise ValueError(f"Unknown Python backend: {name}")


def _numba_solver(case, numba):
    # A compiled fused-loop baseline, with the same all-pairs interaction.
    a, b, sigma, beta = case.a, case.b, case.sigma, case.beta
    d, steps, dt = case.dim, case.steps, case.t_final / case.steps
    pairwise, multiplicative = case.name == "pairwise", case.name == "multiplicative"

    @numba.njit(parallel=True, fastmath=False, cache=False)
    def solve(x0, z):
        n, width = x0.shape
        x = x0.copy()
        nxt = np.empty_like(x)
        mean = np.empty(width)
        for step in range(steps):
            if not pairwise:
                for k in range(width):
                    total = 0.0
                    for i in range(n):
                        total += x[i, k]
                    mean[k] = total / n
            for i in numba.prange(n):
                if pairwise:
                    for k in range(d):
                        nxt[i, k] = x[i, k] + dt * x[i, d + k]
                        nxt[i, d + k] = 0.0
                    for j in range(n):
                        r2 = 0.0
                        for k in range(d):
                            r2 += (x[j, k] - x[i, k]) ** 2
                        weight = (1 + r2) ** (-beta)
                        for k in range(d):
                            nxt[i, d + k] += weight * (x[j, d + k] - x[i, d + k])
                    for k in range(d):
                        nxt[i, d + k] = x[i, d + k] + dt * nxt[i, d + k] / n + sigma * np.sqrt(dt) * z[step, i, d + k]
                else:
                    for k in range(width):
                        g = sigma * x[i, k] if multiplicative else sigma
                        nxt[i, k] = x[i, k] + (a * x[i, k] + b * mean[k]) * dt + g * np.sqrt(dt) * z[step, i, k]
            x, nxt = nxt, x
        history = np.empty((2, n, width))
        history[0] = x0
        history[1] = x
        return history

    return solve
