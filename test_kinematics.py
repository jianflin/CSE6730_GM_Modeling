"""
test_kinematics.py

Simple checks for kinematics.py
"""

from __future__ import annotations

import numpy as np

from kinematics import (
    se2,
    se2_inv,
    adjoint_se2,
    frames_in_head,
    joints_in_head,
    spatial_jacobian,
    body_point_twist,
)


def check_se2_inverse() -> None:
    g = se2(np.pi / 6.0, 1.2, -0.4)
    g_inv = se2_inv(g)
    I = g @ g_inv

    print("check_se2_inverse")
    print(I)
    print("close to identity:", np.allclose(I, np.eye(3), atol=1e-10))
    print()


def check_adjoint_identity() -> None:
    g = np.eye(3)
    Ad = adjoint_se2(g)

    print("check_adjoint_identity")
    print(Ad)
    print("close to identity:", np.allclose(Ad, np.eye(3), atol=1e-10))
    print()


def check_frames_in_head() -> None:
    alpha = [0.2, -0.1]
    beta = [0.3, -0.3, 0.2, -0.2]
    leg_posi = [1, 3]
    L = 1.0
    L_leg = 0.5

    g_body, g_legs = frames_in_head(
        alpha=alpha,
        beta=beta,
        leg_posi=leg_posi,
        L=L,
        L_leg=L_leg,
    )

    print("check_frames_in_head")
    print("number of body frames:", len(g_body))   # should be N = len(alpha)+1 = 3
    print("number of leg frame pairs:", len(g_legs))  # should be len(leg_posi) = 2
    print("body frame 0:\n", g_body[0])
    print("body frame 1:\n", g_body[1])
    print("first left leg frame:\n", g_legs[0][0])
    print()


def check_joints_in_head() -> None:
    alpha = [0.2, -0.1]
    L = 1.0

    g_joint = joints_in_head(alpha=alpha, L=L)

    print("check_joints_in_head")
    print("number of joint frames:", len(g_joint))
    for i, g in enumerate(g_joint):
        print(f"joint frame {i}:")
        print(g)
    print()


def check_spatial_jacobian() -> None:
    alpha = [0.2, -0.1]
    leg_posi = [1, 3]
    L = 1.0

    J_body, J_leg = spatial_jacobian(
        alpha=alpha,
        leg_posi=leg_posi,
        L=L,
    )

    print("check_spatial_jacobian")
    print("J_body shape:", J_body.shape)
    print(J_body)
    print("number of J_leg columns:", len(J_leg))
    for i, col in enumerate(J_leg):
        print(f"J_leg[{i}] =", col)
    print()


def check_body_point_twist_body_only() -> None:
    xi_body = np.array([1.0, 0.2, 0.1], dtype=float)
    g_local = se2(0.0, 1.0, 0.0)

    xi_local = body_point_twist(
        xi_body=xi_body,
        g_local=g_local,
    )

    print("check_body_point_twist_body_only")
    print("xi_body =", xi_body)
    print("xi_local =", xi_local)
    print()


def check_body_point_twist_with_joints() -> None:
    xi_body = np.array([1.0, 0.0, 0.1], dtype=float)
    g_local = se2(0.0, 1.0, 0.0)

    J_partial = np.array(
        [
            [0.0, 0.2],
            [-1.0, -2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    qdot_partial = np.array([0.5, -0.25], dtype=float)

    xi_local = body_point_twist(
        xi_body=xi_body,
        g_local=g_local,
        J_partial=J_partial,
        qdot_partial=qdot_partial,
    )

    print("check_body_point_twist_with_joints")
    print("xi_body =", xi_body)
    print("J_partial =\n", J_partial)
    print("qdot_partial =", qdot_partial)
    print("xi_local =", xi_local)
    print()


def main() -> None:
    check_se2_inverse()
    check_adjoint_identity()
    check_frames_in_head()
    check_joints_in_head()
    check_spatial_jacobian()
    check_body_point_twist_body_only()
    check_body_point_twist_with_joints()


if __name__ == "__main__":
    main()