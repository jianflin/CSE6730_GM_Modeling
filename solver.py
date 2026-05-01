"""
solver.py

Quasi-static force balance solver for undulatory locomotion.

Force-free condition: W_total(xi_body) = 0

Three entry points:
    linear_viscous_body_velocity  – closed-form linear solve (viscous only)
    total_wrench_residual         – residual for nonlinear models
    solve_body_velocity           – dispatches based on model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from kinematics import adjoint_se2, se2_inv, frames_in_head, spatial_jacobian, body_point_twist
from forces import ForceModel, CoulombParams, ViscousParams, BBParams, rft_element_wrench

Array = NDArray[np.float64]
ForceParams = CoulombParams | ViscousParams | BBParams | None


@dataclass
class SolverOptions:
    tol: float = 1e-8
    max_iter: int = 50
    scipy_method: str = "hybr"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _leg_twist_jacobian(
    J_body: Array,
    J_leg_col: Array,
    pos: int,
    alpha_dot: Array,
    beta_dot_k: float,
) -> tuple[Array, Array]:
    """
    Combined (J_partial, qdot_partial) for a leg at 1-based body segment pos.

    pos == 1 : only the leg's own joint contributes.
    pos > 1  : body joints 0..pos-2 plus the leg joint contribute.

    Matches MATLAB forceResidual_1.m / computeBodyVelocity.m leg handling.
    """
    if pos == 1:
        return J_leg_col.reshape(3, 1), np.array([beta_dot_k])
    J_part = np.hstack([J_body[:, : pos - 1], J_leg_col.reshape(3, 1)])
    qdot_part = np.concatenate([alpha_dot[: pos - 1], [beta_dot_k]])
    return J_part, qdot_part


# ---------------------------------------------------------------------------
# Viscous linear solve  (matches MATLAB computeBodyVelocity.m)
# ---------------------------------------------------------------------------

def linear_viscous_body_velocity(
    alpha: Array,
    alpha_dot: Array,
    beta: Array,
    beta_dot: Array,
    leg_posi: Sequence[int],
    L: float,
    L_leg: float,
    activation: Array,
    K: Array,
) -> Array:
    """
    Closed-form solve for viscous drag:  omega1 * xi = -omega2

        omega1 = sum_i  act_i * Ad_i^T K Ad_i
        omega2 = sum_i  act_i * Ad_i^T K Ad_i * J_i * qdot_i
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    alpha_dot = np.asarray(alpha_dot, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    beta_dot = np.asarray(beta_dot, dtype=float).ravel()
    activation = np.asarray(activation, dtype=float).ravel()

    if activation.sum() == 0.0:
        return np.zeros(3, dtype=float)

    N = alpha.size + 1
    g_body, g_legs = frames_in_head(alpha, beta, leg_posi, L, L_leg)
    J_body, J_leg = spatial_jacobian(alpha, leg_posi, L)

    omega1 = np.zeros((3, 3), dtype=float)
    omega2 = np.zeros(3, dtype=float)

    for i in range(N):
        w = activation[i]
        if w == 0.0:
            continue
        Ad_inv = adjoint_se2(se2_inv(g_body[i]))
        A = Ad_inv.T @ K @ Ad_inv
        omega1 += w * A
        if i > 0:
            omega2 += w * (A @ (J_body[:, :i] @ alpha_dot[:i]))

    num_leg = len(leg_posi)
    for k in range(num_leg):
        pos = leg_posi[k]
        for side in range(2):
            w = activation[N + 2 * k + side]
            if w == 0.0:
                continue
            Ad_inv = adjoint_se2(se2_inv(g_legs[k][side]))
            A = Ad_inv.T @ K @ Ad_inv
            omega1 += w * A
            J_part, qdot_part = _leg_twist_jacobian(
                J_body, J_leg[k], pos, alpha_dot, beta_dot[2 * k + side]
            )
            omega2 += w * (A @ (J_part @ qdot_part))

    return np.linalg.solve(omega1, -omega2)


# ---------------------------------------------------------------------------
# Nonlinear wrench residual  (matches MATLAB forceResidual_1.m)
# ---------------------------------------------------------------------------

def total_wrench_residual(
    xi_body: Array,
    alpha: Array,
    alpha_dot: Array,
    beta: Array,
    beta_dot: Array,
    leg_posi: Sequence[int],
    L: float,
    L_leg: float,
    activation: Array,
    model: ForceModel,
    params: ForceParams = None,
) -> Array:
    """
    Net wrench on the robot in the head frame.

    Force-free locomotion requires  total_wrench_residual(xi) = 0.
    """
    alpha = np.asarray(alpha, dtype=float).ravel()
    alpha_dot = np.asarray(alpha_dot, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    beta_dot = np.asarray(beta_dot, dtype=float).ravel()
    activation = np.asarray(activation, dtype=float).ravel()

    N = alpha.size + 1
    g_body, g_legs = frames_in_head(alpha, beta, leg_posi, L, L_leg)
    J_body, J_leg = spatial_jacobian(alpha, leg_posi, L)

    W = np.zeros(3, dtype=float)

    for i in range(N):
        w = activation[i]
        if w == 0.0:
            continue
        J_part = J_body[:, :i] if i > 0 else np.zeros((3, 0), dtype=float)
        qdot_part = alpha_dot[:i] if i > 0 else np.zeros(0, dtype=float)
        xi_local = body_point_twist(xi_body, g_body[i], J_part, qdot_part)
        wrench_local = rft_element_wrench(L, xi_local, model, params)
        Ad_inv = adjoint_se2(se2_inv(g_body[i]))
        W += w * (Ad_inv.T @ wrench_local)

    num_leg = len(leg_posi)
    for k in range(num_leg):
        pos = leg_posi[k]
        for side in range(2):
            w = activation[N + 2 * k + side]
            if w == 0.0:
                continue
            g_leg = g_legs[k][side]
            J_part, qdot_part = _leg_twist_jacobian(
                J_body, J_leg[k], pos, alpha_dot, beta_dot[2 * k + side]
            )
            xi_local = body_point_twist(xi_body, g_leg, J_part, qdot_part)
            wrench_local = rft_element_wrench(L_leg, xi_local, model, params)
            Ad_inv = adjoint_se2(se2_inv(g_leg))
            W += w * (Ad_inv.T @ wrench_local)

    return W


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_body_velocity(
    alpha: Array,
    alpha_dot: Array,
    beta: Array,
    beta_dot: Array,
    leg_posi: Sequence[int],
    L: float,
    L_leg: float,
    activation: Array,
    model: ForceModel,
    params: ForceParams = None,
    K: Array | None = None,
    options: SolverOptions | None = None,
    xi_guess: Array | None = None,
) -> tuple[Array, bool]:
    """
    Solve for xi_body such that total wrench = 0.

    Returns
    -------
    xi_body : (3,) [vx, vy, omega]
    success : bool
    """
    activation = np.asarray(activation, dtype=float).ravel()

    if activation.sum() == 0.0:
        return np.zeros(3, dtype=float), True

    opts = options if options is not None else SolverOptions()

    if model == "viscous":
        K_mat = K if K is not None else np.eye(3, dtype=float)
        xi = linear_viscous_body_velocity(
            alpha, alpha_dot, beta, beta_dot, leg_posi, L, L_leg, activation, K_mat
        )
        return xi, True

    # nonlinear: prefer caller-supplied warm-start, else viscous K=I guess
    if xi_guess is not None:
        xi0 = np.asarray(xi_guess, dtype=float).ravel()
    else:
        xi0 = linear_viscous_body_velocity(
            alpha, alpha_dot, beta, beta_dot, leg_posi, L, L_leg, activation,
            np.eye(3, dtype=float),
        )

    def residual(xi: Array) -> Array:
        return total_wrench_residual(
            xi, alpha, alpha_dot, beta, beta_dot,
            leg_posi, L, L_leg, activation, model, params,
        )

    result = root(
        residual, xi0,
        method=opts.scipy_method,
        tol=opts.tol,
        options={"maxfev": opts.max_iter * 100},
    )
    return result.x, bool(result.success)


# ---------------------------------------------------------------------------
# Legacy shim kept for backward compat with old simulation.py tests
# ---------------------------------------------------------------------------

@dataclass
class RobotGeom:
    L_body: float
    L_leg: float
    leg_posi: Sequence[int]


def compute_body_velocity(
    alpha: Array,
    beta: Array,
    alpha_dot: Array,
    beta_dot: Array,
    geom: RobotGeom,
    model: ForceModel,
    params: ForceParams = None,
    integrate_along_segment: bool = False,
    n_points: int = 3,
    xi_guess: Array | None = None,
    tol: float = 1e-9,
    max_iter: int = 30,
) -> Array:
    """Backward-compatible wrapper around solve_body_velocity."""
    N = np.asarray(alpha).size + 1
    num_leg = len(geom.leg_posi)
    activation = np.ones(N + 2 * num_leg, dtype=float)
    xi, _ = solve_body_velocity(
        alpha=alpha,
        alpha_dot=alpha_dot,
        beta=beta,
        beta_dot=beta_dot,
        leg_posi=geom.leg_posi,
        L=geom.L_body,
        L_leg=geom.L_leg,
        activation=activation,
        model=model,
        params=params,
    )
    return xi
