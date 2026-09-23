"""Regressions for best responses, transport distances, and PDE stability."""

import numpy as np
import pytest
from scipy.stats import norm
from mvkit.poc import wasserstein2_between_samples, wasserstein2_to_reference
from mvkit.mfg import (
    MFGProblem, solve_mfg, solve_lq_mfg, solve_lq_mfg_vector,
    solve_fokker_planck, solve_fokker_planck_2d, solve_hjb, solve_hjb_2d,
    MFGProblem2D, solve_mfg_2d,
)
from mvkit.mfg.linear_quadratic import _lq_best_response


def test_lq_best_response_to_terminal_target():
    # Minimize 1/2 int alpha^2 dt + 1/2 (X_1-1)^2, X_0=0.
    # The unique optimal trajectory is X_t=t/2.
    s = solve_lq_mfg(q=0, q_T=1, sigma=0, T=1, mu_0_mean=0,
        mu_0_var=0, n_particles=2, n_grid=1000, n_iterations_max=1,
        m_initial=np.linspace(0, 1, 1001))
    assert s.m_iterates[1][-1] == pytest.approx(.5, abs=1e-3)


def test_lq_best_response_to_time_varying_running_target():
    # Euler-Lagrange equation: X'' = X-t, X(0)=0, X'(1)=0.
    # Its solution is X(t) = t - sinh(t)/cosh(1).
    t = np.linspace(0, 1, 1001)
    sol = solve_lq_mfg(q=1, q_T=0, sigma=0, T=1, mu_0_mean=0,
        mu_0_var=0, n_particles=2, n_grid=1000, n_iterations_max=1,
        m_initial=t)
    np.testing.assert_allclose(sol.m, t - np.sinh(t)/np.cosh(1), atol=2e-4)


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('boundary', ['periodic', 'neumann'])
def test_fp_returns_nonnegative_density_or_rejects_step(dim, boundary):
    x = np.arange(8, dtype=float)
    try:
        if dim == 1:
            initial = np.zeros(8); initial[3] = 1
            s = solve_fokker_planck(sigma=.01, T=1, x_grid=x, n_t=1,
                initial=initial, boundary=boundary,
                drift=lambda t, x: np.full_like(x, 1.5))
        else:
            initial = np.zeros((8, 8)); initial[3, 3] = 1
            s = solve_fokker_planck_2d(sigma=.01, T=1, x_grid=x,
                y_grid=x, n_t=1, initial=initial, boundary=boundary,
                drift=lambda t, X, Y: np.stack(
                    (np.full_like(X, 1.5), np.zeros_like(X)), axis=-1))
    except ValueError as exc:
        assert 'CFL' in str(exc)
        return
    assert s.m.min() >= -1e-12


@pytest.mark.parametrize('dim', [1, 2])
def test_hjb_preserves_comparison_or_rejects_step(dim):
    x = np.arange(8, dtype=float)
    g = np.zeros(8); g[3] = 1
    if dim == 2:
        g = np.broadcast_to(g[:, None], (8, 8)).copy()
    g2 = g.copy(); g2[3] += .01
    kw = dict(sigma=.01, T=.75, x_grid=x, n_t=1)
    try:
        if dim == 1:
            a = solve_hjb(**kw, terminal=g, running_cost=lambda t, x: np.zeros_like(x))
            b = solve_hjb(**kw, terminal=g2, running_cost=lambda t, x: np.zeros_like(x))
        else:
            kw.update(y_grid=x, running_cost=lambda t, X, Y: np.zeros_like(X))
            a = solve_hjb_2d(**kw, terminal=g)
            b = solve_hjb_2d(**kw, terminal=g2)
    except ValueError as exc:
        assert 'CFL' in str(exc)
        return
    assert np.min(b.u - a.u) >= -1e-12


def test_fictitious_play_converged_means_small_residual():
    s = solve_lq_mfg(q=1, q_T=1, sigma=0, T=1, mu_0_mean=0,
        mu_0_var=0, n_particles=2, n_grid=100, n_iterations_max=180,
        tol=1e-4, seed=0, method='fictitious_play', m_initial=np.ones(101))
    br = _lq_best_response(s.m, s.P, s.t_grid, 0, 0, 0, 2, 0, q=1, q_T=1)
    residual = np.max(np.abs(br - s.m))
    assert s.fixed_point_residual == pytest.approx(residual, abs=1e-12)
    assert not s.converged
    assert residual > 1e-4


def test_grid_fictitious_play_converged_means_small_residual():
    p = MFGProblem(sigma=.3, T=1, domain=(0, 2*np.pi), n_x=16,
        initial_density=lambda x: (1 + .5*np.cos(x))/(2*np.pi),
        running_cost=lambda t, x, m: 20*m, terminal_cost=lambda x, m: 5*m)
    s = solve_mfg(p, n_t=50, method='fictitious_play',
        n_iterations_max=500, tol=1e-5)
    br = solve_mfg(p, n_t=50, n_iterations_max=1, m_initial_iterate=s.m)
    residual = np.max(np.abs(br.m - s.m))
    assert s.fixed_point_residual == pytest.approx(residual, abs=1e-12)
    assert s.converged == (residual < 1e-5)
    np.testing.assert_allclose(s.u[-1], 5*s.m[-1], atol=1e-12)


def test_w2_invariant_under_sample_replication():
    # Both lists describe exactly 1/2 delta_0 + 1/2 delta_1.
    assert wasserstein2_between_samples([0, 1], [0, 0, 1, 1]) == pytest.approx(0)


def test_w2_dirac_to_standard_normal():
    # Only possible coupling: X=0 and Y~N(0,1); E[(X-Y)^2]=1.
    assert wasserstein2_to_reference([0], norm.ppf) == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize('cov', [np.zeros((1, 1)), np.ones((2, 2))])
def test_vector_accepts_singular_psd_covariance(cov):
    d = len(cov)
    s = solve_lq_mfg_vector(Q=np.eye(d), Q_T=np.eye(d), Sigma=np.zeros((d,d)),
        T=.1, mu_0_mean=np.zeros(d), mu_0_var=cov,
        n_particles=2, n_grid=4, n_iterations_max=2)
    assert np.isfinite(s.x_trajectory).all()


@pytest.mark.parametrize('which', ['Q', 'Q_T', 'R'])
def test_vector_rejects_negative_cost_matrices(which):
    kw = dict(Q=np.zeros((1, 1)), Q_T=np.zeros((1, 1)), R=np.eye(1),
        Sigma=np.zeros((1, 1)), T=.1, mu_0_mean=np.zeros(1), mu_0_var=np.eye(1),
        n_particles=2, n_grid=4, n_iterations_max=2)
    kw[which] = -np.eye(1)
    with pytest.raises(ValueError):
        solve_lq_mfg_vector(**kw)


def test_grid_2d_fictitious_play_checks_returned_density_residual():
    p = MFGProblem2D(
        sigma=.3, T=.5, domain=((0, 2*np.pi), (0, 2*np.pi)), n_x=(8, 8),
        initial_density=lambda X, Y: (1 + .5*np.cos(X))/(2*np.pi)**2,
        running_cost=lambda t, X, Y, m: 20*m,
        terminal_cost=lambda X, Y, m: 5*m,
    )
    sol = solve_mfg_2d(p, n_t=20, method="fictitious_play",
                       n_iterations_max=80, tol=1e-6)
    br = solve_mfg_2d(p, n_t=20, n_iterations_max=1, m_initial_iterate=sol.m)
    residual = np.max(np.abs(br.m - sol.m))
    assert sol.fixed_point_residual == pytest.approx(residual, abs=1e-12)
    assert sol.converged == (residual < 1e-6)
    np.testing.assert_allclose(sol.u[-1], 5*sol.m[-1], atol=1e-12)


def test_neumann_uniform_density_has_domain_midpoint_mean():
    p = MFGProblem(sigma=.1, T=1, domain=(0, 1), n_x=4, boundary='neumann',
        initial_density=lambda x: np.ones_like(x),
        running_cost=lambda t, x, m: np.zeros_like(x),
        terminal_cost=lambda x, m: np.zeros_like(x))
    s = solve_mfg(p, n_t=4)
    assert np.sum(s.x_grid * s.m[-1]) / 4 == pytest.approx(.5)


def test_vector_response_to_terminal_target_with_noncommuting_costs():
    # For zero running cost, the deterministic optimum is a straight line:
    # (R + Q_T) X(1) = Q_T m(1), starting from zero.
    R = np.array([[2.0, 0.5], [0.5, 1.0]])
    QT = np.array([[1.0, 0.2], [0.2, 3.0]])
    target = np.array([1.0, -2.0])
    flow = np.linspace(0, 1, 501)[:, None] * target
    sol = solve_lq_mfg_vector(
        Q=np.zeros((2, 2)), Q_T=QT, R=R, Sigma=np.zeros((2, 2)),
        T=1, mu_0_mean=np.zeros(2), mu_0_var=np.zeros((2, 2)),
        n_particles=2, n_grid=500, n_iterations_max=1, m_initial=flow,
    )
    expected = np.linalg.solve(R + QT, QT @ target)
    np.testing.assert_allclose(sol.m[-1], expected, atol=2e-5)


def test_vector_fictitious_play_checks_returned_mean_residual():
    kwargs = dict(Q=np.eye(2), Q_T=np.eye(2), Sigma=np.zeros((2, 2)),
                  T=1, mu_0_mean=np.zeros(2), mu_0_var=np.zeros((2, 2)),
                  n_particles=2, n_grid=20, tol=1e-3, seed=0)
    sol = solve_lq_mfg_vector(**kwargs, method="fictitious_play",
        n_iterations_max=100, m_initial=np.ones((21, 2)))
    br = solve_lq_mfg_vector(**kwargs, n_iterations_max=1, m_initial=sol.m)
    residual = np.max(np.abs(br.m - sol.m))
    assert sol.fixed_point_residual == pytest.approx(residual, abs=1e-12)
    assert not sol.converged
    assert residual > 1e-3


def test_unequal_empirical_w2_matches_exact_transport():
    # Send half the mass at zero to 0, then 1/6 from 2 to 0, 1/3 from 2 to 3.
    assert wasserstein2_between_samples([0, 2], [0, 0, 3])**2 == pytest.approx(1)


def test_reference_w2_uniform_includes_quantile_cell_variance():
    # Midpoint quantiles alone would return zero, but each cell contributes
    # its conditional uniform variance, 1/(12*n**2).
    n = 7
    samples = (np.arange(n) + 0.5) / n
    assert wasserstein2_to_reference(samples, lambda u: u)**2 == pytest.approx(1/(12*n*n))
