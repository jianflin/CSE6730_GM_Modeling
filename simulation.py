"""
simulation.py

Time integration and trajectory generation for the undulatory robot.

Responsibilities:
- prescribe gait variables alpha(t), beta(t)
- call solver.compute_body_velocity(...)
- integrate body pose
- store trajectory/results
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from solver import RobotGeom, compute_body_velocity
from forces import ForceModel, CoulombParams, ViscousParams, BBParams

Array = NDArray[np.float64]


@dataclass
class SimParams:
    dt: float
    T: float


@dataclass
class SimResult:
    t: Array
    pose: Array
    xi_body: Array
    alpha: Array
    beta: Array
    alpha_dot: Array
    beta_dot: Array


def integrate_pose_se2(pose: Array, xi_body: Array, dt: float) -> Array:
    """
    Integrate body pose using body-frame twist xi_body = [vx, vy, omega]^T.

    pose = [x, y, theta] in world coordinates
    """
    x, y, theta = map(float, pose)
    vx, vy, omega = map(float, xi_body)

    c = np.cos(theta)
    s = np.sin(theta)

    v_world = np.array(
        [
            c * vx - s * vy,
            s * vx + c * vy,
        ],
        dtype=float,
    )

    return np.array(
        [
            x + dt * v_world[0],
            y + dt * v_world[1],
            theta + dt * omega,
        ],
        dtype=float,
    )


def sinusoidal_body_gait(
    t: float,
    n_alpha: int,
    amplitude: float,
    omega: float,
    phase_lag: float,
) -> tuple[Array, Array]:
    """
    Example body gait:
        alpha_i(t) = A sin(omega t + i*phase_lag)
    """
    idx = np.arange(n_alpha, dtype=float)
    phase = omega * t + idx * phase_lag
    alpha = amplitude * np.sin(phase)
    alpha_dot = amplitude * omega * np.cos(phase)
    return alpha, alpha_dot


def sinusoidal_leg_gait(
    t: float,
    n_beta: int,
    amplitude: float,
    omega: float,
    phase_lag: float = 0.0,
) -> tuple[Array, Array]:
    """
    Example leg gait:
        beta_j(t) = A sin(omega t + j*phase_lag)
    """
    idx = np.arange(n_beta, dtype=float)
    phase = omega * t + idx * phase_lag
    beta = amplitude * np.sin(phase)
    beta_dot = amplitude * omega * np.cos(phase)
    return beta, beta_dot


def run_simulation(
    sim_params: SimParams,
    geom: RobotGeom,
    n_alpha: int,
    n_beta: int,
    body_gait_fn: Callable[[float], tuple[Array, Array]],
    leg_gait_fn: Callable[[float], tuple[Array, Array]],
    model: ForceModel,
    force_params: CoulombParams | ViscousParams | BBParams | None = None,
    integrate_along_segment: bool = False,
    n_points: int = 3,
    pose0: Sequence[float] = (0.0, 0.0, 0.0),
    xi_guess0: Sequence[float] = (0.0, 0.0, 0.0),
) -> SimResult:
    """
    Main simulation loop.
    """
    dt = sim_params.dt
    T = sim_params.T

    n_steps = int(np.floor(T / dt)) + 1
    t_arr = dt * np.arange(n_steps, dtype=float)

    pose_hist = np.zeros((n_steps, 3), dtype=float)
    xi_hist = np.zeros((n_steps, 3), dtype=float)
    alpha_hist = np.zeros((n_steps, n_alpha), dtype=float)
    beta_hist = np.zeros((n_steps, n_beta), dtype=float)
    alpha_dot_hist = np.zeros((n_steps, n_alpha), dtype=float)
    beta_dot_hist = np.zeros((n_steps, n_beta), dtype=float)

    pose_hist[0] = np.asarray(pose0, dtype=float).reshape(3)
    xi_guess = np.asarray(xi_guess0, dtype=float).reshape(3)

    for k, t in enumerate(t_arr):
        alpha, alpha_dot = body_gait_fn(t)
        beta, beta_dot = leg_gait_fn(t)

        alpha = np.asarray(alpha, dtype=float).reshape(n_alpha)
        alpha_dot = np.asarray(alpha_dot, dtype=float).reshape(n_alpha)
        beta = np.asarray(beta, dtype=float).reshape(n_beta)
        beta_dot = np.asarray(beta_dot, dtype=float).reshape(n_beta)

        xi_body = compute_body_velocity(
            alpha=alpha,
            beta=beta,
            alpha_dot=alpha_dot,
            beta_dot=beta_dot,
            geom=geom,
            model=model,
            params=force_params,
            integrate_along_segment=integrate_along_segment,
            n_points=n_points,
            xi_guess=xi_guess,
        )

        xi_hist[k] = xi_body
        alpha_hist[k] = alpha
        beta_hist[k] = beta
        alpha_dot_hist[k] = alpha_dot
        beta_dot_hist[k] = beta_dot

        if k < n_steps - 1:
            pose_hist[k + 1] = integrate_pose_se2(pose_hist[k], xi_body, dt)

        xi_guess = xi_body.copy()

    return SimResult(
        t=t_arr,
        pose=pose_hist,
        xi_body=xi_hist,
        alpha=alpha_hist,
        beta=beta_hist,
        alpha_dot=alpha_dot_hist,
        beta_dot=beta_dot_hist,
    )