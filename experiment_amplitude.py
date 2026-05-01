"""
experiment_amplitude.py

Sweep body-wave amplitude (0 - 80 deg) for two locomotion systems:

  Experiment 1 — Limbless undulatory locomotion, viscous fluid
      gamma_y / gamma_x = 1  (isotropic drag, no propulsion expected)
      gamma_y / gamma_x = 5  (anisotropic drag, forward propulsion)

  Experiment 2 — Centipede locomotion, Coulomb friction
      Legs at every body segment; body-wave amplitude varied.

Metric: net x-displacement per gait cycle, normalised by body length (BL/cycle).

Notes on parameters:
  - The viscous linear solver uses the K matrix (diag([kx, ky, 0])) to encode
    drag anisotropy.  K must be passed explicitly; force_params alone is ignored
    by the linear path.
  - CoulombParams(A=3.0): lateral leg force = 3x tangential, giving the
    directional anisotropy needed for centipede forward propulsion.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from forces import ViscousParams, CoulombParams
from simulation import SimulationParams, simulate
from test_simulation import build_undulatory_gait


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

N           = 6                         # body segments
L           = 1.3                       # half-segment length  (segment span = 2L)
L_leg       = 0.8                       # leg length
BL          = 2.0 * L * N              # total body length

temporal_freq = 2.0 * np.pi / 5.0      # angular frequency  (period = 5 s)
T_cycle       = 2.0 * np.pi / temporal_freq
n_cycles      = 5
T_total       = n_cycles * T_cycle
n_steps       = 501                    # ~100 pts per cycle

t = np.linspace(0.0, T_total, n_steps)

amplitudes_deg = np.linspace(0.0, 80.0, 17)   # 0, 5, 10, ..., 80
amplitudes_rad = np.deg2rad(amplitudes_deg)


# ---------------------------------------------------------------------------
# Displacement metric
# ---------------------------------------------------------------------------

def x_displacement_per_cycle(result, n_cyc: int, body_length: float) -> float:
    """Net x-displacement divided by (n_cycles * BL). Positive = forward."""
    dx = float(result.pose[-1, 0] - result.pose[0, 0])
    return dx / n_cyc / body_length


# ---------------------------------------------------------------------------
# Experiment 1 - Limbless + viscous
# ---------------------------------------------------------------------------
# The viscous linear solver uses K = diag([kx, ky, 0]) to encode drag
# anisotropy.  Without an explicit K, the solver defaults to K = I (isotropic),
# giving zero net locomotion for both gamma ratios.

def run_limbless_viscous(amplitude_rad: float, kx: float, ky: float) -> float:
    K = np.diag([kx, ky, 0.0])
    params = SimulationParams(
        N=N, leg_posi=[], L=L, L_leg=L_leg,
        model="viscous",
        force_params=ViscousParams(kx=kx, ky=ky),
        K=K,
    )
    # Negative spatial_freq → tailward wave → forward (+x) locomotion
    gait = build_undulatory_gait(N=N, num_leg=0, body_amplitude=amplitude_rad,
                                 temporal_freq=temporal_freq,
                                 spatial_freq=-(1.0 - 1.0 / N))
    result = simulate(t, gait, params)
    return x_displacement_per_cycle(result, n_cycles, BL)


print("Experiment 1: limbless viscous (gamma_y/gamma_x = 1) ...")
disp_gamma1 = []
for i, A in enumerate(amplitudes_rad):
    d = run_limbless_viscous(A, kx=1.0, ky=1.0)
    disp_gamma1.append(d)
    print(f"  {amplitudes_deg[i]:5.1f} deg  ->  {d:+.5f} BL/cycle")

print("Experiment 1: limbless viscous (gamma_y/gamma_x = 5) ...")
disp_gamma5 = []
for i, A in enumerate(amplitudes_rad):
    d = run_limbless_viscous(A, kx=1.0, ky=5.0)
    disp_gamma5.append(d)
    print(f"  {amplitudes_deg[i]:5.1f} deg  ->  {d:+.5f} BL/cycle")


# ---------------------------------------------------------------------------
# Centipede shared setup
# ---------------------------------------------------------------------------
# body_activation=0: centipede legs lift the body — only leg tips contact
# the ground.  Body segments have no Coulomb drag.

leg_posi_centipede = list(range(1, N + 1))   # one leg pair per segment
_COULOMB_A = 1.0                             # leg Coulomb friction anisotropy

# sigmoid_gain=10 instead of the default 100: smoother residual for
# scipy.optimize.root.  gain=100 is nearly discontinuous and causes the
# hybr solver to converge to wrong branches.
# sigmoid_gain=10: smoother than the default 100, avoids near-discontinuities
# that cause the hybr solver to miss the root.
_COULOMB_PARAMS = CoulombParams(A=_COULOMB_A, sigmoid_gain=0.5)
# body_activation=0.05: centipede body has minimal ground contact; a tiny
# regularisation weight stabilises the solver when all legs are in swing.
_BODY_ACT = 0.05


def run_centipede_coulomb(amplitude_rad: float, phi_lag: float = 0.0) -> tuple[float, int]:
    """Returns (displacement_BL_per_cycle, n_solver_failures)."""
    params = SimulationParams(
        N=N, leg_posi=leg_posi_centipede, L=L, L_leg=L_leg,
        model="coulomb",
        force_params=_COULOMB_PARAMS,
    )
    gait = build_undulatory_gait(
        N=N, num_leg=len(leg_posi_centipede),
        body_amplitude=amplitude_rad,
        temporal_freq=temporal_freq,
        phi_lag=phi_lag,
        body_activation=_BODY_ACT,
    )
    result = simulate(t, gait, params)
    n_fail = int((~result.solver_success).sum())
    return x_displacement_per_cycle(result, n_cycles, BL), n_fail


# ---------------------------------------------------------------------------
# Experiment 3 - phase lag sweep  (runs FIRST to find optimal lag)
# ---------------------------------------------------------------------------
# Body amplitude fixed at 60 deg.  phi_lag shifts the leg contact pattern
# relative to the body wave, from 0 to 2*pi.

AMP_FIXED    = np.deg2rad(60.0)
phi_lags_deg = np.linspace(0.0, 360.0, 37)   # every 10 deg (0 and 360 both included)
phi_lags_rad = np.deg2rad(phi_lags_deg)

print(f"Experiment 3: phase lag sweep, amplitude = {np.rad2deg(AMP_FIXED):.0f} deg ...")
disp_philag = []
for i, phi in enumerate(phi_lags_rad):
    d, nf = run_centipede_coulomb(AMP_FIXED, phi_lag=phi)
    disp_philag.append(d)
    fail_str = f"  [{nf} failures]" if nf else ""
    print(f"  phi_lag = {phi_lags_deg[i]:6.1f} deg  ->  {d:+.5f} BL/cycle{fail_str}")

disp_philag = np.array(disp_philag)
best_idx      = int(np.argmax(disp_philag))
phi_lag_opt   = phi_lags_rad[best_idx]
phi_lag_opt_d = phi_lags_deg[best_idx]
print(f"  => optimal phi_lag = {phi_lag_opt_d:.1f} deg  "
      f"({disp_philag[best_idx]:+.5f} BL/cycle)")


# ---------------------------------------------------------------------------
# Experiment 2 - amplitude sweep using optimal phase lag
# ---------------------------------------------------------------------------

print(f"Experiment 2: centipede Coulomb, phi_lag = {phi_lag_opt_d:.1f} deg ...")
disp_coulomb = []
for i, A in enumerate(amplitudes_rad):
    d, nf = run_centipede_coulomb(A, phi_lag=phi_lag_opt)
    disp_coulomb.append(d)
    fail_str = f"  [{nf} failures]" if nf else ""
    print(f"  {amplitudes_deg[i]:5.1f} deg  ->  {d:+.5f} BL/cycle{fail_str}")


# ---------------------------------------------------------------------------
# Figure 1 — Phase lag sweep  (shown first)
# ---------------------------------------------------------------------------

fig_phase, ax_phase = plt.subplots(figsize=(7, 5))
ax_phase.plot(phi_lags_deg, disp_philag, '^-', color='tab:purple',
              label=f'amp = {np.rad2deg(AMP_FIXED):.0f} deg')
ax_phase.axvline(phi_lag_opt_d, color='tab:red', lw=1.4, linestyle='--',
                 label=f'optimal = {phi_lag_opt_d:.0f} deg')
ax_phase.axhline(0, color='k', lw=0.7, linestyle='--')
ax_phase.set_xlabel('Body-leg phase lag (deg)')
ax_phase.set_ylabel('Displacement (BL / cycle)')
ax_phase.set_title(f'Centipede Coulomb — Phase Lag Sweep  (A = {_COULOMB_A})')
ax_phase.set_xlim(0, 360)
ax_phase.set_xticks([0, 90, 180, 270, 360])
ax_phase.legend()
ax_phase.grid(True, alpha=0.3)
fig_phase.tight_layout()
fig_phase.savefig('experiment_phase_lag.png', dpi=150)
print("\nFigure saved -> experiment_phase_lag.png")
plt.show()


# ---------------------------------------------------------------------------
# Figure 2 — Amplitude sweeps
# ---------------------------------------------------------------------------

fig_amp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Limbless viscous
ax1.plot(amplitudes_deg, disp_gamma1, 'o-', color='tab:blue',
         label=r'$\gamma_y / \gamma_x = 1$  (isotropic)')
ax1.plot(amplitudes_deg, disp_gamma5, 's-', color='tab:orange',
         label=r'$\gamma_y / \gamma_x = 5$  (anisotropic)')
ax1.axhline(0, color='k', lw=0.7, linestyle='--')
ax1.set_xlabel('Body wave amplitude (deg)')
ax1.set_ylabel('Displacement (BL / cycle)')
ax1.set_title('Limbless Undulatory — Viscous Fluid')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Centipede at optimal phase lag
ax2.plot(amplitudes_deg, disp_coulomb, 'D-', color='tab:red',
         label=f'phi_lag = {phi_lag_opt_d:.0f} deg  (optimal)')
ax2.axhline(0, color='k', lw=0.7, linestyle='--')
ax2.set_xlabel('Body wave amplitude (deg)')
ax2.set_ylabel('Displacement (BL / cycle)')
ax2.set_title(f'Centipede — Coulomb (A = {_COULOMB_A}), Optimal Phase Lag')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig_amp.tight_layout()
fig_amp.savefig('experiment_amplitude.png', dpi=150)
print("Figure saved -> experiment_amplitude.png")
plt.show()
