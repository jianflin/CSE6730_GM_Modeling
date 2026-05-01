"""
visualization.py

Animate undulatory locomotion from simulation output.

Drawing conventions:
  - Body segments : thick dark lines
  - Joints        : white-filled circles, dark border
  - Leg contacts  : large circles — dark red (filled) = stance,
                                    white (open)       = swing
  - COM           : solid black circle; trajectory: blue line
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

from kinematics import frames_in_head, se2, translation_x
from simulation import SimResult, SimulationParams, GaitFunctions

Array = NDArray[np.float64]

# ── colour palette ─────────────────────────────────────────────────────────
_C_BODY   = np.array([0.20, 0.20, 0.20])
_C_STANCE = np.array([0.635, 0.078, 0.184])   # dark red
_C_SWING  = 'white'
_C_COM_TR = '#0072BD'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _initial_head_frame(alpha0: Array, N: int) -> Array:
    ave_body = -np.sum(np.cumsum(alpha0)) / N
    return se2(ave_body, 0.0, 0.0)


def _link_endpoints(g: Array, L: float) -> tuple[Array, Array]:
    return (g @ translation_x(-L))[:2, 2], (g @ translation_x(L))[:2, 2]


def _robot_extent(g_body_w: list[Array], g_legs_w: list, L: float) -> tuple[float, float, float, float]:
    """Bounding box of all body endpoints and leg tip positions."""
    xs, ys = [], []
    for g in g_body_w:
        p1, p2 = _link_endpoints(g, L)
        xs += [p1[0], p2[0]]
        ys += [p1[1], p2[1]]
    for gl, gr in g_legs_w:
        xs += [gl[0, 2], gr[0, 2]]
        ys += [gl[1, 2], gr[1, 2]]
    return min(xs), max(xs), min(ys), max(ys)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_locomotion(
    t: Array,
    result: SimResult,
    gait: GaitFunctions,
    params: SimulationParams,
    figsize: tuple[int, int] = (14, 7),
    view_radius: float | None = None,
    fps: int = 20,
    save_path: str | None = None,
    show: bool = True,
) -> FuncAnimation:
    """
    Animate robot locomotion.

    Camera follows the robot COM in both x and y.
    Legs are shown as contact-point dots only (no lines):
      dark red filled = stance, white open = swing.

    Parameters
    ----------
    view_radius : half-size of the square view window; None = auto
    fps         : playback / save frame rate
    save_path   : file path (.gif or .mp4); None = don't save
    show        : open interactive matplotlib window
    """
    N, num_leg = params.N, len(params.leg_posi)
    L, L_leg   = params.L, params.L_leg

    g_h0 = _initial_head_frame(result.alpha[0], N)

    # Pre-evaluate leg activation for contact colouring
    act_hist     = np.array([gait.activation(gait.phase(float(tk))) for tk in t])
    leg_act_hist = act_hist[:, N:]      # (n_steps, 2*num_leg)

    # Auto view radius: use robot body half-length + leg length + margin
    if view_radius is None:
        body_half = L * N
        view_radius = body_half + L_leg + L

    # ── figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.axis('off')

    # ── artists ─────────────────────────────────────────────────────────────
    body_lines = [
        ax.plot([], [], '-', color=_C_BODY, lw=6, solid_capstyle='round')[0]
        for _ in range(N)
    ]
    joint_dots = [
        ax.plot([], [], 'o', color='white', markersize=11,
                markeredgecolor=_C_BODY, markeredgewidth=2)[0]
        for _ in range(N - 1)
    ]
    # Leg contact points only — no lines
    leg_dots = [
        ax.plot([], [], 'o', markersize=16,
                markeredgecolor=_C_BODY, markeredgewidth=2)[0]
        for _ in range(2 * num_leg)
    ]
    com_dot,  = ax.plot([], [], 'o', color='k', markersize=14, zorder=6)
    com_traj, = ax.plot([], [], '-', color=_C_COM_TR, lw=2, zorder=5)
    time_txt  = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=11, va='top')

    com_xs: list[float] = []
    com_ys: list[float] = []

    # ── per-frame update ────────────────────────────────────────────────────
    def _update(k: int):
        alpha = result.alpha[k]
        beta  = result.beta[k]
        pose  = result.pose[k]
        act   = leg_act_hist[k]

        g_world  = g_h0 @ se2(pose[2], pose[0], pose[1])
        g_body_h, g_legs_h = frames_in_head(alpha, beta, params.leg_posi, L, L_leg)
        g_body_w = [g_world @ g for g in g_body_h]
        g_legs_w = [(g_world @ gl, g_world @ gr) for gl, gr in g_legs_h]

        # Body links
        for i, g in enumerate(g_body_w):
            p1, p2 = _link_endpoints(g, L)
            body_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        # Joints
        for i in range(N - 1):
            p = (g_body_w[i] @ translation_x(L))[:2, 2]
            joint_dots[i].set_data([p[0]], [p[1]])

        # Leg contact dots (no lines)
        for ki in range(num_leg):
            for side in range(2):
                idx = 2 * ki + side
                tip = g_legs_w[ki][side][:2, 2]
                in_stance = act[idx] > 0
                leg_dots[idx].set_data([tip[0]], [tip[1]])
                leg_dots[idx].set_markerfacecolor(_C_STANCE if in_stance else _C_SWING)

        # COM
        com = np.mean([g[:2, 2] for g in g_body_w], axis=0)
        com_xs.append(float(com[0]))
        com_ys.append(float(com[1]))
        com_dot.set_data([com[0]], [com[1]])
        com_traj.set_data(com_xs, com_ys)

        # Follow robot in both x and y
        ax.set_xlim(com[0] - view_radius, com[0] + view_radius)
        ax.set_ylim(com[1] - view_radius, com[1] + view_radius)

        time_txt.set_text(f't = {t[k]:.2f} s')

        return (*body_lines, *joint_dots, *leg_dots,
                com_dot, com_traj, time_txt)

    anim = FuncAnimation(
        fig, _update, frames=len(t),
        interval=1000.0 / fps,
        blit=False,
    )

    if save_path:
        _save(anim, save_path, fps)

    if show:
        plt.tight_layout()
        plt.show()

    return anim


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(anim: FuncAnimation, path: str, fps: int) -> None:
    if path.endswith('.gif'):
        anim.save(path, writer='pillow', fps=fps)
    else:
        try:
            anim.save(path, writer='ffmpeg', fps=fps)
        except Exception as exc:
            fallback = path.rsplit('.', 1)[0] + '.gif'
            print(f"ffmpeg unavailable ({exc}); saving as {fallback}")
            anim.save(fallback, writer='pillow', fps=fps)
    print(f"Animation saved to {path}")
