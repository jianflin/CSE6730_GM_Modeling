"""
experiment_snapshots.py

Body-configuration snapshots at evenly spaced time frames, amplitude = 60 deg.
  - Limbless undulatory, viscous fluid (gamma_y/gamma_x = 5)
  - Centipede, Coulomb friction (phi_lag = 180 deg)

Each subplot overlays all snapshots on one axes; color darkens with time.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from forces import ViscousParams, CoulombParams
from simulation import SimulationParams, simulate
from test_simulation import build_undulatory_gait
from kinematics import se2, frames_in_head, joints_in_head


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

N     = 6
L     = 1.3
L_leg = 0.8

temporal_freq = 2.0 * np.pi / 5.0
n_cycles      = 5
T_total       = n_cycles * 2.0 * np.pi / temporal_freq
n_steps       = 501

t = np.linspace(0.0, T_total, n_steps)

AMP            = np.deg2rad(60.0)
N_SNAP         = 12                          # evenly spaced snapshots
leg_posi_cent  = list(range(1, N + 1))
PHI_LAG_OPT    = np.pi                       # ~180 deg (approx optimum)
_COULOMB       = CoulombParams(A=1.0, sigmoid_gain=0.5)
_BODY_ACT      = 0.05


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

def run_limbless() -> object:
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
        body_amplitude=AMP,
        temporal_freq=temporal_freq,
        spatial_freq=-(1.0 - 1.0 / N),
    )
    return simulate(t, gait, params)


def run_centipede() -> object:
    params = SimulationParams(
        N=N, leg_posi=leg_posi_cent, L=L, L_leg=L_leg,
        model="coulomb",
        force_params=_COULOMB,
    )
    gait = build_undulatory_gait(
        N=N, num_leg=len(leg_posi_cent),
        body_amplitude=AMP,
        temporal_freq=temporal_freq,
        phi_lag=PHI_LAG_OPT,
        body_activation=_BODY_ACT,
    )
    return simulate(t, gait, params)


print("Running limbless viscous (60 deg) ...")
res_lb = run_limbless()

print("Running centipede Coulomb (60 deg) ...")
res_cp = run_centipede()
n_fail = int((~res_cp.solver_success).sum())
if n_fail:
    print(f"  [{n_fail} solver failures]")


# ---------------------------------------------------------------------------
# Body-shape helpers
# ---------------------------------------------------------------------------

def spine_world(pose_k: np.ndarray, alpha_k: np.ndarray) -> np.ndarray:
    """World-frame spine points (N+1, 2) using joints_in_head."""
    g_head = se2(pose_k[2], pose_k[0], pose_k[1])
    pts = [(g_head @ gj)[:2, 2] for gj in joints_in_head(alpha_k, L)]
    return np.array(pts)


def legs_world(
    pose_k: np.ndarray,
    alpha_k: np.ndarray,
    beta_k: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """World-frame leg endpoints: list of (base_xy, tip_left_xy, tip_right_xy)."""
    g_head = se2(pose_k[2], pose_k[0], pose_k[1])
    g_body, g_legs = frames_in_head(alpha_k, beta_k, leg_posi_cent, L, L_leg)
    out = []
    for k_l, (g_left, g_right) in enumerate(g_legs):
        pos = leg_posi_cent[k_l]
        base = (g_head @ g_body[pos - 1])[:2, 2]
        tl   = (g_head @ g_left)[:2, 2]
        tr   = (g_head @ g_right)[:2, 2]
        out.append((base, tl, tr))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

snap_idx = np.linspace(0, n_steps - 1, N_SNAP, dtype=int)
norm     = Normalize(vmin=0, vmax=N_SNAP - 1)
t_snap   = t[snap_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Limbless ---
cmap_lb = cm.Blues
for i, k in enumerate(snap_idx):
    c     = cmap_lb(0.25 + 0.75 * norm(i))
    spine = spine_world(res_lb.pose[k], res_lb.alpha[k])
    ax1.plot(spine[:, 0], spine[:, 1], 'o-', color=c, lw=2.0, ms=4, zorder=2)

sm1 = plt.cm.ScalarMappable(
    cmap=cmap_lb,
    norm=Normalize(vmin=t_snap[0], vmax=t_snap[-1]),
)
sm1.set_array([])
fig.colorbar(sm1, ax=ax1, label='Time (s)', fraction=0.04, pad=0.04)

ax1.set_aspect('equal', adjustable='datalim')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_title(r'Limbless — Viscous ($\gamma_y/\gamma_x = 5$),  A = 60 deg')
ax1.grid(True, alpha=0.3)

# --- Centipede ---
cmap_cp = cm.Oranges
for i, k in enumerate(snap_idx):
    c     = cmap_cp(0.25 + 0.75 * norm(i))
    spine = spine_world(res_cp.pose[k], res_cp.alpha[k])
    ax2.plot(spine[:, 0], spine[:, 1], 'o-', color=c, lw=2.0, ms=4, zorder=2)

    for base, tl, tr in legs_world(res_cp.pose[k], res_cp.alpha[k], res_cp.beta[k]):
        ax2.plot([base[0], tl[0]], [base[1], tl[1]], '-', color=c, lw=1.2, alpha=0.7)
        ax2.plot([base[0], tr[0]], [base[1], tr[1]], '-', color=c, lw=1.2, alpha=0.7)

sm2 = plt.cm.ScalarMappable(
    cmap=cmap_cp,
    norm=Normalize(vmin=t_snap[0], vmax=t_snap[-1]),
)
sm2.set_array([])
fig.colorbar(sm2, ax=ax2, label='Time (s)', fraction=0.04, pad=0.04)

ax2.set_aspect('equal', adjustable='datalim')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_title(
    f'Centipede — Coulomb (A=1.0),  phi_lag = {np.rad2deg(PHI_LAG_OPT):.0f} deg,  '
    f'amp = 60 deg'
)
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Body Snapshots  ({N_SNAP} evenly spaced frames,  T = {T_total:.1f} s)')
fig.tight_layout()
fig.savefig('snapshots_60deg.png', dpi=150)
print("\nFigure saved -> snapshots_60deg.png")
plt.show()
