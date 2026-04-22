"""
test_simulation.py

Simple runnable example for simulation.py
"""

from __future__ import annotations

import numpy as np

from solver import RobotGeom
from simulation import (
    SimParams,
    run_simulation,
    sinusoidal_body_gait,
    sinusoidal_leg_gait,
)
from forces import ViscousParams


def main() -> None:
    geom = RobotGeom(
        L_body=1.0,
        L_leg=0.5,
        leg_posi=[1, 3],
    )

    sim_params = SimParams(
        dt=0.01,
        T=2.0,
    )

    n_alpha = 3
    n_beta = 2 * len(geom.leg_posi)

    body_gait = lambda t: sinusoidal_body_gait(
        t=t,
        n_alpha=n_alpha,
        amplitude=0.4,
        omega=2.0 * np.pi,
        phase_lag=np.pi / 4.0,
    )

    leg_gait = lambda t: sinusoidal_leg_gait(
        t=t,
        n_beta=n_beta,
        amplitude=0.3,
        omega=2.0 * np.pi,
        phase_lag=np.pi / 2.0,
    )

    result = run_simulation(
        sim_params=sim_params,
        geom=geom,
        n_alpha=n_alpha,
        n_beta=n_beta,
        body_gait_fn=body_gait,
        leg_gait_fn=leg_gait,
        model="viscous",
        force_params=ViscousParams(kx=1.0, ky=2.0),
        integrate_along_segment=False,
        n_points=3,
    )

    print("Final pose:", result.pose[-1])
    print("Final body velocity:", result.xi_body[-1])


if __name__ == "__main__":
    main()