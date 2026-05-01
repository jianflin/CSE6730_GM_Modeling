"""
test_solver.py  –  pytest suite for solver.py
"""

from __future__ import annotations

import numpy as np
import pytest

from forces import ViscousParams, CoulombParams, BBParams
from solver import (
    SolverOptions,
    linear_viscous_body_velocity,
    total_wrench_residual,
    solve_body_velocity,
)


# ---------------------------------------------------------------------------
# Minimal single-segment fixture (N=2, no legs)
# ---------------------------------------------------------------------------

def _simple_args(n_alpha: int = 1, num_leg: int = 0, active: bool = True):
    """Return keyword args for a straight, stationary N-segment chain."""
    alpha = np.zeros(n_alpha)
    alpha_dot = np.zeros(n_alpha)
    beta = np.zeros(2 * num_leg)
    beta_dot = np.zeros(2 * num_leg)
    leg_posi: list[int] = list(range(1, num_leg + 1))
    L, L_leg = 1.0, 0.5
    N = n_alpha + 1
    activation = np.ones(N + 2 * num_leg) if active else np.zeros(N + 2 * num_leg)
    return dict(
        alpha=alpha, alpha_dot=alpha_dot,
        beta=beta, beta_dot=beta_dot,
        leg_posi=leg_posi, L=L, L_leg=L_leg,
        activation=activation,
    )


# ---------------------------------------------------------------------------
# solve_body_velocity: zero activation
# ---------------------------------------------------------------------------

def test_zero_activation_returns_zero_and_success():
    args = _simple_args(n_alpha=2, active=False)
    xi, ok = solve_body_velocity(**args, model="viscous")
    assert ok is True
    assert np.allclose(xi, 0.0, atol=1e-14)


def test_zero_activation_nonlinear_returns_zero():
    args = _simple_args(n_alpha=2, active=False)
    xi, ok = solve_body_velocity(**args, model="coulomb", params=CoulombParams())
    assert ok is True
    assert np.allclose(xi, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# total_wrench_residual: shape
# ---------------------------------------------------------------------------

def test_total_wrench_residual_shape():
    args = _simple_args(n_alpha=2, num_leg=1)
    xi = np.array([0.1, 0.05, 0.02])
    W = total_wrench_residual(xi, model="viscous", params=ViscousParams(), **args)
    assert W.shape == (3,)


def test_total_wrench_residual_finite():
    args = _simple_args(n_alpha=3, num_leg=2)
    xi = np.array([0.2, -0.1, 0.05])
    for model, params in [("viscous", ViscousParams()), ("coulomb", CoulombParams()), ("bb", BBParams())]:
        W = total_wrench_residual(xi, model=model, params=params, **args)
        assert np.all(np.isfinite(W)), f"{model} residual not finite"


# ---------------------------------------------------------------------------
# linear_viscous_body_velocity
# ---------------------------------------------------------------------------

def test_viscous_linear_solve_finite():
    alpha = np.array([0.3, -0.2])
    alpha_dot = np.array([0.1, -0.05])
    beta = np.zeros(2)
    beta_dot = np.array([0.2, -0.2])
    leg_posi = [1]
    N = 3
    activation = np.ones(N + 2)
    K = np.diag([1.0, 2.0, 0.1])
    xi = linear_viscous_body_velocity(
        alpha, alpha_dot, beta, beta_dot, leg_posi, 1.0, 0.5, activation, K
    )
    assert xi.shape == (3,)
    assert np.all(np.isfinite(xi))


def test_viscous_linear_solve_zero_qdot_gives_nonzero_xi():
    """Non-zero qdot must drive non-zero xi (active drag)."""
    alpha = np.array([0.2])
    alpha_dot = np.array([0.5])   # non-zero body rate
    beta = np.zeros(0)
    beta_dot = np.zeros(0)
    activation = np.ones(2)
    xi = linear_viscous_body_velocity(
        alpha, alpha_dot, beta, beta_dot, [], 1.0, 0.5, activation, np.eye(3)
    )
    assert np.linalg.norm(xi) > 1e-10


# ---------------------------------------------------------------------------
# solve_body_velocity: viscous succeeds
# ---------------------------------------------------------------------------

def test_solve_body_velocity_viscous_succeeds():
    alpha = np.array([0.3, -0.15])
    alpha_dot = np.array([0.2, 0.1])
    beta = np.array([np.pi / 2, -np.pi / 2])
    beta_dot = np.array([0.3, -0.3])
    leg_posi = [1]
    N = 3
    activation = np.ones(N + 2)
    xi, ok = solve_body_velocity(
        alpha, alpha_dot, beta, beta_dot, leg_posi, 1.0, 0.5,
        activation, model="viscous", params=ViscousParams(kx=1.0, ky=2.0),
        K=np.diag([1.0, 2.0, 0.1]),
    )
    assert ok is True
    assert xi.shape == (3,)
    assert np.all(np.isfinite(xi))


# ---------------------------------------------------------------------------
# solve_body_velocity: nonlinear models run without crashing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params", [
    ("coulomb", CoulombParams()),
    ("bb",      BBParams()),
])
def test_nonlinear_solver_runs(model, params):
    alpha = np.array([0.2])
    alpha_dot = np.array([0.3])
    beta = np.zeros(0)
    beta_dot = np.zeros(0)
    activation = np.ones(2)
    xi, ok = solve_body_velocity(
        alpha, alpha_dot, beta, beta_dot, [], 1.0, 0.5,
        activation, model=model, params=params,
        options=SolverOptions(tol=1e-6, max_iter=100),
    )
    assert xi.shape == (3,)
    assert np.all(np.isfinite(xi))


# ---------------------------------------------------------------------------
# solve_body_velocity: residual near zero at solution
# ---------------------------------------------------------------------------

def test_viscous_solution_satisfies_residual():
    alpha = np.array([0.3, -0.2])
    alpha_dot = np.array([0.1, 0.15])
    beta = np.array([np.pi / 2, -np.pi / 2])
    beta_dot = np.array([0.2, -0.2])
    leg_posi = [2]
    N = 3
    activation = np.ones(N + 2)
    xi, _ = solve_body_velocity(
        alpha, alpha_dot, beta, beta_dot, leg_posi, 1.0, 0.5,
        activation, model="viscous",
        K=np.diag([1.0, 1.0, 0.5]),
    )
    W = total_wrench_residual(
        xi, alpha, alpha_dot, beta, beta_dot,
        leg_posi, 1.0, 0.5, activation,
        model="viscous", params=ViscousParams(kx=1.0, ky=1.0),
    )
    assert np.linalg.norm(W) < 0.1
