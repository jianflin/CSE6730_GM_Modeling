"""
solver.py

Assemble whole-body force/torque balance and solve for the unknown
body twist xi_body = [vx, vy, omega]^T under quasi-static force balance.

This version stays minimal because:
- kinematics.py already computes frames and local twists
- forces.py already computes local element wrenches
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from kinematics import frames_in_head, spatial_jacobian, body_point_twist
from forces import (
    ForceModel,
    CoulombParams,
    ViscousParams,
    BBParams,
    rft_element_wrench,
)

Array = NDArray[np.float64]


@dataclass
class RobotGeom:
    L_body: float
    L_leg: float
    leg_posi: Sequence[int]


def _partial_body_qdot(alpha_dot: Array, i: int) -> Array:
    """Body joint rates that affect segment i (0-based body segment index)."""
    if i <= 0:
        return np.zeros(0, dtype=float)
    return alpha_dot[:i]


def _net_wrench(
    xi_body: Array,
    alpha: Sequence[float],
    beta: Sequence[float],
    alpha_dot: Sequence[float],
    beta_dot: Sequence[float],
    geom: RobotGeom,
    model: ForceModel,
    params: CoulombParams | ViscousParams | BBParams | None = None,
    integrate_along_segment: bool = False,
    n_points: int = 3,
) -> Array:
    """
    Total wrench on the robot, expressed in the body/head frame.

    Force-free locomotion solves:
        W_net(xi_body) = 0
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    alpha_dot = np.asarray(alpha_dot, dtype=float).reshape(-1)
    beta_dot = np.asarray(beta_dot, dtype=float).reshape(-1)

    g_body, g_legs = frames_in_head(
        alpha=alpha,
        beta=beta,
        leg_posi=geom.leg_posi,
        L=geom.L_body,
        L_leg=geom.L_leg,
    )
    J_body, J_leg = spatial_jacobian(
        alpha=alpha,
        leg_posi=geom.leg_posi,
        L=geom.L_body,
    )

    W = np.zeros(3, dtype=float)

    # --- body segments ---
    for i, g_seg in enumerate(g_body):
        J_partial = J_body[:, :i] if i > 0 else np.zeros((3, 0), dtype=float)
        qdot_partial = _partial_body_qdot(alpha_dot, i)

        xi_local = body_point_twist(
            xi_body=xi_body,
            g_local=g_seg,
            J_partial=J_partial,
            qdot_partial=qdot_partial,
        )

        w_local = rft_element_wrench(
            length=geom.L_body,
            xi_local=xi_local,
            model=model,
            params=params,
            integrate_along_segment=integrate_along_segment,
            n_points=n_points,
        )

        # map local wrench back to head/body frame
        from kinematics import adjoint_se2, se2_inv
        Ad_inv = adjoint_se2(se2_inv(g_seg))
        W += Ad_inv.T @ w_local

    # --- legs ---
    for k, (g_left, g_right) in enumerate(g_legs):
        jcol = J_leg[k].reshape(3, 1)

        # left leg
        xi_left = body_point_twist(
            xi_body=xi_body,
            g_local=g_left,
            J_partial=jcol,
            qdot_partial=np.array([beta_dot[2 * k]], dtype=float),
        )
        w_left = rft_element_wrench(
            length=geom.L_leg,
            xi_local=xi_left,
            model=model,
            params=params,
            integrate_along_segment=integrate_along_segment,
            n_points=n_points,
        )
        from kinematics import adjoint_se2, se2_inv
        Ad_inv_left = adjoint_se2(se2_inv(g_left))
        W += Ad_inv_left.T @ w_left

        # right leg
        xi_right = body_point_twist(
            xi_body=xi_body,
            g_local=g_right,
            J_partial=jcol,
            qdot_partial=np.array([beta_dot[2 * k + 1]], dtype=float),
        )
        w_right = rft_element_wrench(
            length=geom.L_leg,
            xi_local=xi_right,
            model=model,
            params=params,
            integrate_along_segment=integrate_along_segment,
            n_points=n_points,
        )
        Ad_inv_right = adjoint_se2(se2_inv(g_right))
        W += Ad_inv_right.T @ w_right

    return W


def compute_body_velocity(
    alpha: Sequence[float],
    beta: Sequence[float],
    alpha_dot: Sequence[float],
    beta_dot: Sequence[float],
    geom: RobotGeom,
    model: ForceModel,
    params: CoulombParams | ViscousParams | BBParams | None = None,
    integrate_along_segment: bool = False,
    n_points: int = 3,
    xi_guess: Array | None = None,
    tol: float = 1e-9,
    max_iter: int = 30,
) -> Array:
    """
    Solve W_net(xi_body)=0 using a simple finite-difference Newton method.

    Works for viscous, Coulomb, and BB models.
    """
    xi = np.zeros(3, dtype=float) if xi_guess is None else np.asarray(xi_guess, dtype=float).reshape(3)

    for _ in range(max_iter):
        r = _net_wrench(
            xi_body=xi,
            alpha=alpha,
            beta=beta,
            alpha_dot=alpha_dot,
            beta_dot=beta_dot,
            geom=geom,
            model=model,
            params=params,
            integrate_along_segment=integrate_along_segment,
            n_points=n_points,
        )

        if np.linalg.norm(r) < tol:
            return xi

        # finite-difference Jacobian
        J = np.zeros((3, 3), dtype=float)
        eps = 1e-6
        for j in range(3):
            dxi = np.zeros(3, dtype=float)
            dxi[j] = eps
            rp = _net_wrench(
                xi_body=xi + dxi,
                alpha=alpha,
                beta=beta,
                alpha_dot=alpha_dot,
                beta_dot=beta_dot,
                geom=geom,
                model=model,
                params=params,
                integrate_along_segment=integrate_along_segment,
                n_points=n_points,
            )
            J[:, j] = (rp - r) / eps

        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(J, -r, rcond=None)

        xi = xi + step

        if np.linalg.norm(step) < tol:
            return xi

    raise RuntimeError("compute_body_velocity did not converge")