"""
forces.py considers the local forces when knowing the local velocity

Three different interaction models are included:
    -BB
    -Columb friction
    -Viscous fluid


"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from kinematics import adjoint_se2, se2, se2_inv


Array = NDArray[np.float64]
ForceModel = Literal["coulomb", "viscous", "bb"]


@dataclass
class CoulombParams:
    """
    Anisotropic Coulomb-style RFT:
    tangential force ~ cos(phi), normal force ~ A sin(phi) 
    A is related to the drag anistropic
    """
    A: float = 1.0
    sigmoid_gain: float = 100.0
    speed_scale: float = 2.0 * np.pi


@dataclass
class ViscousParams:
    """
    Linear viscous drag in the segment frame:
    f = -diag(kx, ky, 0) xi
    """
    kx: float = 1.0
    ky: float = 1.0


@dataclass
class BBParams:
    """
    BB force fit, from Ding et al.
    """
    Cs: float = 3.21
    Cf: float = 1.34
    Cl: float = -0.82
    gamma: float = 2.79
    sigmoid_gain: float = 5.0


def _smooth_sign(x: float, gain: float) -> float:
    return 2.0 / (1.0 + np.exp(np.clip(gain * x, -500.0, 500.0))) - 1.0


def _angle_from_velocity(vx: float, vy: float) -> float:
    if np.hypot(vx, vy) == 0.0:
        return 0.0
    return abs(np.arctan2(vy, vx))


def rft_local_wrench(
    xi_local: Array,
    model: ForceModel,
    params: CoulombParams | ViscousParams | BBParams | None = None,
) -> Array:
    """
    Local segment-frame wrench [Fx, Fy, M]^T.

    Three models are kept:
    - coulomb
    - viscous
    - bb
    """
    vx, vy, omega = map(float, xi_local)
    speed = np.hypot(vx, vy)

    if speed == 0.0:
        return np.zeros(3, dtype=float)

    if model == "coulomb":
        p = params if isinstance(params, CoulombParams) else CoulombParams()
        phi = _angle_from_velocity(vx, vy)
        sig1 = _smooth_sign(vx, p.sigmoid_gain)
        sig2 = _smooth_sign(vy, p.sigmoid_gain)
        mag = speed * p.speed_scale
        return np.array(
            [
                sig1 * np.cos(phi),
                sig2 * p.A * np.sin(phi),
                0.0,
            ],
            dtype=float,
        ) * mag

    if model == "viscous":
        p = params if isinstance(params, ViscousParams) else ViscousParams()
        return np.array(
            [
                -p.kx * vx,
                -p.ky * vy,
                0.0,
            ],
            dtype=float,
        )

    if model == "bb":
        p = params if isinstance(params, BBParams) else BBParams()
        phi = _angle_from_velocity(vx, vy)
        sig1 = _smooth_sign(vx, p.sigmoid_gain)
        sig2 = _smooth_sign(vy, p.sigmoid_gain)
        return np.array(
            [
                sig1 * (p.Cf * np.cos(phi) + p.Cl * (1.0 - np.sin(phi))),
                sig2 * p.Cs * np.sin(np.arctan(p.gamma * np.sin(phi))),
                0.0,
            ],
            dtype=float,
        )

    raise ValueError(f"Unknown force model: {model}")


def rft_element_wrench(
    length: float,
    xi_local: Array,
    model: ForceModel,
    params: CoulombParams | ViscousParams | BBParams | None = None,
    integrate_along_segment: bool = False,
    n_points: int = 3,
) -> Array:
    """
    Wrench contribution of one body/leg element in its own frame.

    - integrate_along_segment=False:
        just use local pointwise RFT
    - integrate_along_segment=True:
        sample along the segment axis and accumulate
    """
    if not integrate_along_segment:
        return rft_local_wrench(xi_local, model, params)

    if n_points < 1:
        raise ValueError("n_points must be >= 1")

    wrench = np.zeros(3, dtype=float)
    ys = np.linspace(-length / 2.0, length / 2.0, n_points)
    ds = length / max(n_points, 1)

    for y in ys:
        g_t = se2(0.0, 0.0, y)
        Ad_inv = adjoint_se2(se2_inv(g_t))
        xi_t = Ad_inv @ xi_local
        wrench += Ad_inv.T @ rft_local_wrench(xi_t, model, params) * ds

    return wrench