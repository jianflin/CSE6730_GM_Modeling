"""
experiment_trajectories.py

For three body-wave amplitudes (30, 50, 70 deg):
  - Limbless undulatory in viscous fluid (gamma_y/gamma_x = 5)
  - Centipede in Coulomb friction (phi_lag = 180 deg, approx optimal)

Two figures:
  Figure 1 — Limbless:   x(t), y(t), x-y trajectory
  Figure 2 — Centipede:  x(t), y(t), x-y trajectory
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from forces import ViscousParams, CoulombParams
from simulation import SimulationParams, simulate
from test_simulation import build_undulatory_gait


# ---------------------------------------------------------------------------
# Shared parameters  (identical to experiment_amplitude.py)
# ---------------------------------------------------------------------------

N           = 6
L           = 1.3
L_leg       = 0.8
BL          = 2.0 * L * N

temporal_freq = 2.0 * np.pi / 5.0
n_cycles      = 5
T_total       = n_cycles * 2.0 * np.pi / temporal_freq
n_steps       = 501

t = np.linspace(0.0, T_total, n_steps)

amplitudes_deg = [30.0, 50.0, 70.0]
amplitudes_rad = [np.deg2rad(a) for a in amplitudes_deg]
colors         = ['tab:blue', 'tab:orange', 'tab:green']
linestyles     = ['-', '--', ':']

# Centipede settings
leg_posi_centipede = list(range(1, N + 1))
_COULOMB_PARAMS    = CoulombParams(A=1.0, sigmoid_gain=0.5)
_BODY_ACT          = 0.05
PHI_LAG_OPT        = np.pi          # ~180 deg (approx optimum from phase sweep)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_limbless(amplitude_rad: float) -> object:
    kx, ky = 1.0, 5.0
    K = np.diag([kx, ky, 0.0])
    params = SimulationParams(
        N=N, leg_posi=[], L=L, L_leg=L_leg,
        model="viscous",
        force_params=ViscousParams(kx=kx, ky=ky),
        K=K,
    )
    gait = build_undulatory_gait(
        N=N, num_leg=0,
        body_amplitude=amplitude_rad,
        temporal_freq=temporal_freq,
        spatial_freq=-(1.0 - 1.0 / N),
    )
    return simulate(t, gait, params)


def run_centipede(amplitude_rad: float) -> object:
    params = SimulationParams(
        N=N, leg_posi=leg_posi_centipede, L=L, L_leg=L_leg,
        model="coulomb",
        force_params=_COULOMB_PARAMS,
    )
    gait = build_undulatory_gait(
        N=N, num_leg=len(leg_posi_centipede),
        body_amplitude=amplitude_rad,
        temporal_freq=temporal_freq,
        phi_lag=PHI_LAG_OPT,
        body_activation=_BODY_ACT,
    )
    return simulate(t, gait, params)


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------

print("Running limbless viscous (gamma_y/gamma_x = 5) ...")
results_limbless = []
for A, A_deg in zip(amplitudes_rad, amplitudes_deg):
    res = run_limbless(A)
    results_limbless.append(res)
    print(f"  {A_deg:.0f} deg done")

print(f"Running centipede Coulomb (phi_lag = {np.rad2deg(PHI_LAG_OPT):.0f} deg) ...")
results_centipede = []
for A, A_deg in zip(amplitudes_rad, amplitudes_deg):
    res = run_centipede(A)
    n_fail = int((~res.solver_success).sum())
    results_centipede.append(res)
    fail_str = f"  [{n_fail} failures]" if n_fail else ""
    print(f"  {A_deg:.0f} deg done{fail_str}")


# ---------------------------------------------------------------------------
# Figure 1 — Limbless
# ---------------------------------------------------------------------------

fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
fig1.suptitle(r'Limbless Undulatory — Viscous Fluid ($\gamma_y/\gamma_x = 5$)')

for res, A_deg, c, ls in zip(results_limbless, amplitudes_deg, colors, linestyles):
    lbl = f'{A_deg:.0f} deg'
    axes1[0].plot(t, res.pose[:, 0], color=c, ls=ls, label=lbl)
    axes1[1].plot(t, res.pose[:, 1], color=c, ls=ls, label=lbl)
    axes1[2].plot(res.pose[:, 0], res.pose[:, 1], color=c, ls=ls, label=lbl)

axes1[0].set_xlabel('Time (s)')
axes1[0].set_ylabel('x (m)')
axes1[0].set_title('x vs t')

axes1[1].set_xlabel('Time (s)')
axes1[1].set_ylabel('y (m)')
axes1[1].set_title('y vs t')

axes1[2].set_xlabel('x (m)')
axes1[2].set_ylabel('y (m)')
axes1[2].set_title('Trajectory')
axes1[2].set_aspect('equal', adjustable='datalim')

for ax in axes1:
    ax.legend()
    ax.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig('trajectories_limbless.png', dpi=150)
print("\nFigure saved -> trajectories_limbless.png")
plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Centipede
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle(
    f'Centipede — Coulomb (A = 1.0),  '
    f'phi_lag = {np.rad2deg(PHI_LAG_OPT):.0f} deg'
)

for res, A_deg, c, ls in zip(results_centipede, amplitudes_deg, colors, linestyles):
    lbl = f'{A_deg:.0f} deg'
    axes2[0].plot(t, res.pose[:, 0], color=c, ls=ls, label=lbl)
    axes2[1].plot(t, res.pose[:, 1], color=c, ls=ls, label=lbl)
    axes2[2].plot(res.pose[:, 0], res.pose[:, 1], color=c, ls=ls, label=lbl)

axes2[0].set_xlabel('Time (s)')
axes2[0].set_ylabel('x (m)')
axes2[0].set_title('x vs t')

axes2[1].set_xlabel('Time (s)')
axes2[1].set_ylabel('y (m)')
axes2[1].set_title('y vs t')

axes2[2].set_xlabel('x (m)')
axes2[2].set_ylabel('y (m)')
axes2[2].set_title('Trajectory')
axes2[2].set_aspect('equal', adjustable='datalim')

for ax in axes2:
    ax.legend()
    ax.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig('trajectories_centipede.png', dpi=150)
print("Figure saved -> trajectories_centipede.png")
plt.show()
