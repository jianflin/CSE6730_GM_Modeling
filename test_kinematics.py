"""
test_kinematics.py  –  pytest suite for kinematics.py
"""

from __future__ import annotations

import numpy as np
import pytest

from kinematics import (
    rot2,
    se2,
    se2_inv,
    adjoint_se2,
    frames_in_head,
    joints_in_head,
    spatial_jacobian,
    body_point_twist,
)


# ---------------------------------------------------------------------------
# se2 / se2_inv
# ---------------------------------------------------------------------------

def test_se2_inverse_consistency():
    g = se2(np.pi / 6.0, 1.2, -0.4)
    assert np.allclose(g @ se2_inv(g), np.eye(3), atol=1e-12)
    assert np.allclose(se2_inv(g) @ g, np.eye(3), atol=1e-12)


def test_se2_inv_identity():
    assert np.allclose(se2_inv(np.eye(3)), np.eye(3), atol=1e-14)


# ---------------------------------------------------------------------------
# adjoint_se2
# ---------------------------------------------------------------------------

def test_adjoint_identity_frame():
    """Ad at identity must equal I."""
    assert np.allclose(adjoint_se2(np.eye(3)), np.eye(3), atol=1e-14)


def test_adjoint_pure_translation():
    """Ad(T(x,y)) has translation column [y, -x, 1]^T."""
    g = se2(0.0, 3.0, 4.0)
    Ad = adjoint_se2(g)
    assert np.allclose(Ad[:2, :2], np.eye(2), atol=1e-14)
    assert np.allclose(Ad[:, 2], [4.0, -3.0, 1.0], atol=1e-14)


# ---------------------------------------------------------------------------
# rot2
# ---------------------------------------------------------------------------

def test_rot2_orthogonality():
    R = rot2(np.pi / 5.0)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
    assert np.allclose(R.T @ R, np.eye(2), atol=1e-12)


def test_rot2_determinant():
    R = rot2(1.23)
    assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# frames_in_head
# ---------------------------------------------------------------------------

def test_frames_in_head_output_sizes():
    alpha = [0.2, -0.1, 0.05]
    beta = [0.3, -0.3, 0.2, -0.2]
    leg_posi = [1, 3]
    g_body, g_legs = frames_in_head(alpha, beta, leg_posi, L=1.0, L_leg=0.5)
    assert len(g_body) == 4       # N = len(alpha)+1 = 4
    assert len(g_legs) == 2       # one pair per leg site


def test_frames_in_head_first_frame_is_identity():
    g_body, _ = frames_in_head([0.1], [0.0, 0.0], [1], L=1.0, L_leg=0.5)
    assert np.allclose(g_body[0], np.eye(3), atol=1e-14)


def test_frames_in_head_zero_shape_straight_body():
    """All alpha=0 → body segments lie along +x, spaced 2L apart."""
    L = 1.5
    N = 4
    alpha = np.zeros(N - 1)
    beta = np.zeros(2)
    g_body, _ = frames_in_head(alpha, beta, [1], L=L, L_leg=0.5)
    for i, g in enumerate(g_body):
        assert np.allclose(g[:2, :2], np.eye(2), atol=1e-12), f"segment {i} rotation not I"
        assert abs(g[0, 2] - 2 * i * L) < 1e-12, f"segment {i} x-position wrong"
        assert abs(g[1, 2]) < 1e-12, f"segment {i} y-position not zero"


def test_frames_in_head_invalid_leg_posi():
    with pytest.raises(ValueError):
        frames_in_head([0.0], [0.0, 0.0], [5], L=1.0, L_leg=0.5)


# ---------------------------------------------------------------------------
# joints_in_head
# ---------------------------------------------------------------------------

def test_joints_in_head_count():
    """N-link chain should yield N+2 joint frames."""
    alpha = [0.1, -0.2, 0.05]   # N=4
    g_joint = joints_in_head(alpha, L=1.0)
    assert len(g_joint) == len(alpha) + 2   # N+2 = 5 for N=4


def test_joints_in_head_zero_shape_x_positions():
    """alpha=0 → joint frames evenly spaced, first at x=-L."""
    L = 1.0
    alpha = np.zeros(2)   # N=3
    g_joint = joints_in_head(alpha, L=L)
    expected_x = [-L, L, 3 * L, 5 * L]
    for i, (g, ex) in enumerate(zip(g_joint, expected_x)):
        assert abs(g[0, 2] - ex) < 1e-12, f"joint {i}: expected x={ex}, got {g[0,2]}"


# ---------------------------------------------------------------------------
# spatial_jacobian
# ---------------------------------------------------------------------------

def test_spatial_jacobian_shape():
    alpha = [0.1, -0.2, 0.0]   # N=4
    leg_posi = [1, 3]
    J_body, J_leg = spatial_jacobian(alpha, leg_posi, L=1.0)
    assert J_body.shape == (3, 3)     # (3, N-1)
    assert len(J_leg) == 2


def test_spatial_jacobian_last_row_ones():
    """Third row of J_body must be all ones (planar revolute joints)."""
    J_body, _ = spatial_jacobian([0.1, -0.2], [1], L=1.0)
    assert np.allclose(J_body[2, :], 1.0, atol=1e-14)


def test_spatial_jacobian_zero_shape_values():
    """For alpha=0, N=2, L=1: J_body[:,0] = [0, -1, 1]."""
    J_body, _ = spatial_jacobian([0.0], [1], L=1.0)
    assert J_body.shape == (3, 1)
    assert np.allclose(J_body[:, 0], [0.0, -1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# body_point_twist
# ---------------------------------------------------------------------------

def test_body_point_twist_head_frame():
    """At g=I the local twist equals the body twist."""
    xi = np.array([1.0, 0.5, 0.2])
    xi_local = body_point_twist(xi, np.eye(3))
    assert np.allclose(xi_local, xi, atol=1e-14)


def test_body_point_twist_with_joints():
    xi_body = np.array([1.0, 0.0, 0.1])
    g_local = se2(0.0, 1.0, 0.0)
    J_partial = np.array([[0.0, 0.2], [-1.0, -2.0], [1.0, 1.0]])
    qdot = np.array([0.5, -0.25])
    xi_local = body_point_twist(xi_body, g_local, J_partial, qdot)
    assert xi_local.shape == (3,)
    assert np.all(np.isfinite(xi_local))
