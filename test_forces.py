"""
test_forces.py  –  pytest suite for forces.py
"""

from __future__ import annotations

import numpy as np
import pytest

from forces import (
    CoulombParams,
    ViscousParams,
    BBParams,
    rft_local_wrench,
    rft_element_wrench,
)


# ---------------------------------------------------------------------------
# rft_local_wrench: zero velocity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params", [
    ("coulomb", CoulombParams()),
    ("viscous", ViscousParams()),
    ("bb",      BBParams()),
])
def test_zero_velocity_gives_zero_wrench(model, params):
    w = rft_local_wrench(np.zeros(3), model, params)
    assert np.allclose(w, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# rft_local_wrench: sign / direction checks
# ---------------------------------------------------------------------------

def test_viscous_opposes_motion_vx():
    p = ViscousParams(kx=2.0, ky=1.0)
    xi = np.array([1.0, 0.0, 0.0])
    w = rft_local_wrench(xi, "viscous", p)
    assert w[0] < 0.0, "x-force should oppose positive vx"


def test_viscous_opposes_motion_vy():
    p = ViscousParams(kx=1.0, ky=3.0)
    xi = np.array([0.0, 1.0, 0.0])
    w = rft_local_wrench(xi, "viscous", p)
    assert w[1] < 0.0, "y-force should oppose positive vy"


def test_viscous_no_torque():
    w = rft_local_wrench(np.array([0.5, -0.3, 0.9]), "viscous", ViscousParams())
    assert w[2] == 0.0


# ---------------------------------------------------------------------------
# rft_local_wrench: finite values for coulomb and bb
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xi", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.7, 0.7, 0.0]),
    np.array([-1.0, 0.5, 0.2]),
])
def test_coulomb_returns_finite(xi):
    w = rft_local_wrench(xi, "coulomb", CoulombParams())
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))


@pytest.mark.parametrize("xi", [
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.5, -0.5, 0.1]),
])
def test_bb_returns_finite(xi):
    w = rft_local_wrench(xi, "bb", BBParams())
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))


def test_bb_no_torque():
    w = rft_local_wrench(np.array([0.3, 0.8, 0.0]), "bb", BBParams())
    assert w[2] == 0.0


# ---------------------------------------------------------------------------
# rft_local_wrench: unknown model raises
# ---------------------------------------------------------------------------

def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown force model"):
        rft_local_wrench(np.array([1.0, 0.0, 0.0]), "mystery", None)  # type: ignore


# ---------------------------------------------------------------------------
# rft_element_wrench: pointwise mode matches rft_local_wrench
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params", [
    ("viscous", ViscousParams()),
    ("bb",      BBParams()),
    ("coulomb", CoulombParams()),
])
def test_rft_element_pointwise_matches_local(model, params):
    xi = np.array([0.6, 0.3, 0.0])
    w_local = rft_local_wrench(xi, model, params)
    w_elem = rft_element_wrench(1.0, xi, model, params, integrate_along_segment=False)
    assert np.allclose(w_elem, w_local, atol=1e-14)


# ---------------------------------------------------------------------------
# rft_element_wrench: integrated mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,params", [
    ("viscous", ViscousParams()),
    ("bb",      BBParams()),
    ("coulomb", CoulombParams()),
])
def test_rft_element_integrated_shape_and_finite(model, params):
    xi = np.array([0.4, 0.2, 0.0])
    w = rft_element_wrench(1.0, xi, model, params, integrate_along_segment=True, n_points=5)
    assert w.shape == (3,)
    assert np.all(np.isfinite(w))


def test_rft_element_integrated_zero_velocity():
    w = rft_element_wrench(1.0, np.zeros(3), "viscous", ViscousParams(), integrate_along_segment=True)
    assert np.allclose(w, 0.0, atol=1e-14)
