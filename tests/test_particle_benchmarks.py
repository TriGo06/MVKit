"""Check benchmark equations independently before trusting timing comparisons."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from particle_suite.cases import Case, coarsen, drift, independent_geometric_exact, inputs, numpy_solve
from particle_suite.backends import prepare


def test_coarsening_preserves_brownian_endpoint():
    case = Case("multiplicative", 11, 3, 32)
    _, z = inputs(case, 123)
    coarse = coarsen(z, 4)
    np.testing.assert_allclose(np.sqrt(1 / 32) * z.sum(0), np.sqrt(1 / 8) * coarse.sum(0), atol=1e-15)


def test_pairwise_force_conserves_mean_velocity():
    case = Case("pairwise", 17, 3, 8)
    x, _ = inputs(case, 4)
    value = drift(case, x)
    np.testing.assert_allclose(value[:, :3], x[:, 3:])
    np.testing.assert_allclose(value[:, 3:].sum(0), 0, atol=3e-15)


def test_moments_deterministic_euler_has_closed_form():
    case = Case("moments", 9, 1, 16, sigma=0)
    x, z = inputs(case, 2)
    dt = case.t_final / case.steps
    m = x.mean(0)
    expected = (1 + case.a * dt)**case.steps * (x - m) + (1 + (case.a + case.b) * dt)**case.steps * m
    np.testing.assert_allclose(numpy_solve(case, x, z)[-1], expected, rtol=1e-14, atol=1e-14)


def test_multiplicative_refinement_against_exact_paths():
    fine = Case("multiplicative", 2048, 3, 256, b=0)
    x, z = inputs(fine, 14)
    exact = independent_geometric_exact(fine, x, z)
    errors = []
    for factor in (16, 4, 1):
        case = replace(fine, steps=fine.steps // factor)
        result = numpy_solve(case, x, coarsen(z, factor))[-1]
        errors.append(np.sqrt(np.mean((result - exact)**2)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[0] / errors[2] > 3


@pytest.mark.parametrize("name,dim", [("moments", 1), ("pairwise", 2), ("multiplicative", 4)])
@pytest.mark.parametrize("backend", ["mvkit-python", "numba", "diffrax"])
def test_adapter_agrees_on_shared_noise(name, dim, backend):
    if backend == "mvkit-python" and name == "multiplicative":
        pytest.skip("Model available through Rust trait only")
    module = "mvkit" if backend == "mvkit-python" else backend
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"Optional backend {backend} is not installed")
    case = Case(name, 7, dim, 8)
    x, z = inputs(case, 5)
    expected = numpy_solve(case, x, z)
    solve, _ = prepare(backend, case, x, z)
    np.testing.assert_allclose(np.asarray(solve()), expected, rtol=3e-12, atol=3e-12)
