"""
simulation.py

Time integration for undulatory robot locomotion.

Flow per time step:
    phase(t) -> alpha(t), d_alpha(t), beta(phase), d_beta(phase), activation(phase)
    -> solve_body_velocity -> xi_body
    -> integrate_body_twist_to_pose -> global pose
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from forces import ForceModel, CoulombParams, ViscousParams, BBParams
from solver import SolverOptions, solve_body_velocity

Array = NDArray[np.float64]
ForceParams = CoulombParams | ViscousParams | BBParams | None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    N: int
    leg_posi: Sequence[int]
    L: float
    L_leg: float
    model: ForceModel
    force_params: ForceParams = None
    K: Array | None = None
    solver_options: SolverOptions | None = None


@dataclass
class GaitFunctions:
    """
    Gait prescription matching MATLAB main_forward.m body_func structure.

    alpha     : t -> (N-1,) body joint angles
    d_alpha   : t -> (N-1,) body joint angle rates
    phase     : t -> float  gait phase
    beta      : phase -> (2*num_leg,) leg angles
    d_beta    : phase -> (2*num_leg,) leg angle rates
    activation: phase -> (N + 2*num_leg,) contact weights
    """
    alpha: Callable[[float], Array]
    d_alpha: Callable[[float], Array]
    phase: Callable[[float], float]
    beta: Callable[[float], Array]
    d_beta: Callable[[float], Array]
    activation: Callable[[float], Array]


@dataclass
class SimResult:
    t: Array
    pose: Array           # (n, 3)  [x, y, theta] world frame
    xi_body: Array        # (n, 3)  [vx, vy, omega] head frame
    alpha: Array          # (n, N-1)
    beta: Array           # (n, 2*num_leg)
    solver_success: Array # (n,) bool


# ---------------------------------------------------------------------------
# Pose integration
# ---------------------------------------------------------------------------

def integrate_body_twist_to_pose(
    t: Array,
    xi_hist: Array,
    g0: Array | None = None,
) -> Array:
    """
    Integrate body-frame twist history into world-frame pose.

    Matches MATLAB globalVelocity.m:
    1. integrate omega -> theta
    2. rotate body velocity to world frame
    3. integrate world velocity -> (x, y)

    Parameters
    ----------
    t        : (n,) monotonically increasing time array
    xi_hist  : (n, 3) body twist [vx, vy, omega]
    g0       : initial pose [x0, y0, theta0], default zeros

    Returns
    -------
    pose : (n, 3) [x, y, theta]
    """
    t = np.asarray(t, dtype=float)
    xi_hist = np.asarray(xi_hist, dtype=float)
    x0, y0, theta0 = (0.0, 0.0, 0.0) if g0 is None else map(float, g0)

    theta = theta0 + np.concatenate(
        [[0.0], cumulative_trapezoid(xi_hist[:, 2], t)]
    )

    vx_w = np.cos(theta) * xi_hist[:, 0] - np.sin(theta) * xi_hist[:, 1]
    vy_w = np.sin(theta) * xi_hist[:, 0] + np.cos(theta) * xi_hist[:, 1]

    x = x0 + np.concatenate([[0.0], cumulative_trapezoid(vx_w, t)])
    y = y0 + np.concatenate([[0.0], cumulative_trapezoid(vy_w, t)])

    return np.column_stack([x, y, theta])


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate(
    t: Array,
    gait: GaitFunctions,
    params: SimulationParams,
    store_frames: bool = False,
    store_residuals: bool = False,
) -> SimResult:
    """
    Run time simulation over the array t.

    At each step evaluates gait functions, solves for xi_body, then
    integrates all xi_body values into a global pose trajectory.
    """
    t = np.asarray(t, dtype=float)
    n = len(t)

    # probe sizes at t[0]
    ph0 = gait.phase(float(t[0]))
    n_alpha = np.asarray(gait.alpha(float(t[0]))).size
    n_beta = np.asarray(gait.beta(ph0)).size

    xi_hist = np.zeros((n, 3), dtype=float)
    alpha_hist = np.zeros((n, n_alpha), dtype=float)
    beta_hist = np.zeros((n, n_beta), dtype=float)
    success_hist = np.zeros(n, dtype=bool)
    xi_prev: np.ndarray | None = None      # warm-start for nonlinear solver

    for k, tk in enumerate(t):
        tk = float(tk)
        phase = gait.phase(tk)
        alpha = np.asarray(gait.alpha(tk), dtype=float).ravel()
        d_alpha = np.asarray(gait.d_alpha(tk), dtype=float).ravel()
        beta = np.asarray(gait.beta(phase), dtype=float).ravel()
        d_beta = np.asarray(gait.d_beta(phase), dtype=float).ravel()
        act = np.asarray(gait.activation(phase), dtype=float).ravel()

        xi, ok = solve_body_velocity(
            alpha=alpha,
            alpha_dot=d_alpha,
            beta=beta,
            beta_dot=d_beta,
            leg_posi=params.leg_posi,
            L=params.L,
            L_leg=params.L_leg,
            activation=act,
            model=params.model,
            params=params.force_params,
            K=params.K,
            options=params.solver_options,
            xi_guess=xi_prev,
        )
        if ok:
            xi_prev = xi

        xi_hist[k] = xi
        alpha_hist[k] = alpha
        beta_hist[k] = beta
        success_hist[k] = ok

    pose = integrate_body_twist_to_pose(t, xi_hist)

    return SimResult(
        t=t,
        pose=pose,
        xi_body=xi_hist,
        alpha=alpha_hist,
        beta=beta_hist,
        solver_success=success_hist,
    )


# ---------------------------------------------------------------------------
# Convenience gait builders (kept from previous version)
# ---------------------------------------------------------------------------

def sinusoidal_body_gait(
    t: float,
    n_alpha: int,
    amplitude: float,
    omega: float,
    phase_lag: float,
) -> tuple[Array, Array]:
    idx = np.arange(n_alpha, dtype=float)
    phase = omega * t + idx * phase_lag
    return amplitude * np.sin(phase), amplitude * omega * np.cos(phase)


def sinusoidal_leg_gait(
    t: float,
    n_beta: int,
    amplitude: float,
    omega: float,
    phase_lag: float = 0.0,
) -> tuple[Array, Array]:
    idx = np.arange(n_beta, dtype=float)
    phase = omega * t + idx * phase_lag
    return amplitude * np.sin(phase), amplitude * omega * np.cos(phase)


# ---------------------------------------------------------------------------
# Legacy run_simulation kept for backward compat
# ---------------------------------------------------------------------------

@dataclass
class SimParams:
    dt: float
    T: float


def run_simulation(
    sim_params: SimParams,
    geom,
    n_alpha: int,
    n_beta: int,
    body_gait_fn,
    leg_gait_fn,
    model: ForceModel,
    force_params: ForceParams = None,
    integrate_along_segment: bool = False,
    n_points: int = 3,
    pose0: Sequence[float] = (0.0, 0.0, 0.0),
    xi_guess0: Sequence[float] = (0.0, 0.0, 0.0),
):
    """Backward-compatible simulation loop."""
    dt = sim_params.dt
    n_steps = int(np.floor(sim_params.T / dt)) + 1
    t_arr = dt * np.arange(n_steps, dtype=float)

    N = n_alpha + 1
    num_leg = len(geom.leg_posi)
    activation = np.ones(N + 2 * num_leg, dtype=float)

    def _phase(tk): return tk
    def _alpha(tk): return body_gait_fn(tk)[0]
    def _d_alpha(tk): return body_gait_fn(tk)[1]
    def _beta(ph): return leg_gait_fn(ph)[0]
    def _d_beta(ph): return leg_gait_fn(ph)[1]
    def _act(ph): return activation

    gait = GaitFunctions(
        alpha=_alpha, d_alpha=_d_alpha,
        phase=_phase,
        beta=_beta, d_beta=_d_beta,
        activation=_act,
    )
    sim_p = SimulationParams(
        N=N, leg_posi=geom.leg_posi,
        L=geom.L_body, L_leg=geom.L_leg,
        model=model, force_params=force_params,
    )
    return simulate(t_arr, gait, sim_p)
