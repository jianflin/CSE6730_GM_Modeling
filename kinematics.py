"""
kinematics.py calculate the trans matrix for kinematics for series-connected robot

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def rot2(theta: float) -> Array:
    """2D rotation matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def se2(theta: float, x: float, y: float) -> Array:
    """SE(2) homogeneous transform."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, x],
            [s,  c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def se2_inv(g: Array) -> Array:
    """Inverse of an SE(2) homogeneous transform."""
    R = g[:2, :2]
    p = g[:2, 2]
    Rt = R.T
    out = np.eye(3, dtype=float)
    out[:2, :2] = Rt
    out[:2, 2] = -Rt @ p
    return out


def adjoint_se2(g: Array) -> Array:
    """
    Planar adjoint map,Twist xi = [vx, vy, omega]^T
    """
    return np.array(
        [
            [g[0, 0], g[0, 1],  g[1, 2]],
            [g[1, 0], g[1, 1], -g[0, 2]],
            [0.0,     0.0,      1.0],
        ],
        dtype=float,
    )


def translation_x(dist: float) -> Array:
    """Pure x-translation in SE(2)."""
    return np.array(
        [
            [1.0, 0.0, dist],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rotation_only(theta: float) -> Array:
    """Pure rotation in SE(2)."""
    return se2(theta, 0.0, 0.0)


def frames_in_head(
    alpha: Sequence[float],
    beta: Sequence[float],
    leg_posi: Sequence[int],
    L: float,
    L_leg: float,
) -> Tuple[List[Array], List[Tuple[Array, Array]]]:
    """
    Body and leg frames expressed in the head frame.

    Returns
    -------
    g_body :
        list of N body frames, g_body[i] is the frame of segment i+1
    g_legs :
        list with one entry per leg attachment site:
        [(left_leg_frame, right_leg_frame), ...]
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    N = alpha.size + 1

    F = translation_x(L)

    g_body: List[Array] = [np.eye(3, dtype=float)]
    for i in range(1, N):
        g_next = g_body[i - 1] @ F @ rotation_only(alpha[i - 1]) @ F
        g_body.append(g_next)

    g_legs: List[Tuple[Array, Array]] = []
    for ind, pos in enumerate(leg_posi):
        if pos < 1 or pos > N:
            raise ValueError(f"leg_posi contains invalid segment index {pos}; valid range is 1..{N}")
        base = g_body[pos - 1]
        g_left = base @ rotation_only(beta[2 * ind]) @ translation_x(L_leg)
        g_right = base @ rotation_only(beta[2 * ind + 1]) @ translation_x(L_leg)
        g_legs.append((g_left, g_right))

    return g_body, g_legs


def joints_in_head(alpha: Sequence[float], L: float) -> List[Array]:
    """
    Joint frames in the head frame.
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    N = alpha.size + 1

    F = translation_x(L)

    g_joint: List[Array] = [np.eye(3, dtype=float)]
    g_joint[0] = g_joint[0] @ se2(0.0, -L, 0.0)  # "virtual 0th joint"

    for i in range(1, N):
        g_joint.append(g_joint[i - 1] @ F @ F @ rotation_only(alpha[i - 1]))
    g_joint.append(g_joint[-1] @ F @ F)

    return g_joint


def spatial_jacobian(
    alpha: Sequence[float],
    leg_posi: Sequence[int],
    L: float,
) -> Tuple[Array, List[Array]]:
    """
    Spatial Jacobian for body joints and leg joints.

    Returns
    -------
    J_body : (3, N-1)
        body joint spatial Jacobian columns
    J_leg : list[(3,)]
        one 3x1 Jacobian column for each leg base
    """
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    N = alpha.size + 1
    g_joint = joints_in_head(alpha, L)
    F = translation_x(L)

    q = np.zeros((2, N - 1), dtype=float)
    for i in range(N - 1):
        q[:, i] = g_joint[i + 1][:2, 2]

    # planar revolute joint twist column = [y, -x, 1]^T
    J_body = np.vstack((q[1, :], -q[0, :], np.ones(N - 1, dtype=float)))

    J_leg: List[Array] = []
    for pos in leg_posi:
        q0 = (g_joint[pos - 1] @ F)[:2, 2]
        J_leg.append(np.array([q0[1], -q0[0], 1.0], dtype=float))

    return J_body, J_leg


def body_point_twist(
    xi_body: Array,
    g_local: Array,
    J_partial: Array | None = None,
    qdot_partial: Array | None = None,
) -> Array:
    """
    Local twist at a segment/leg frame.

    xi_local = Ad_{g^{-1}} xi_body + Ad_{g^{-1}} J qdot
    """
    Ad_inv = adjoint_se2(se2_inv(g_local))
    xi_local = Ad_inv @ xi_body
    if J_partial is not None and qdot_partial is not None and J_partial.size > 0:
        xi_local = xi_local + Ad_inv @ (J_partial @ qdot_partial)
    return xi_local