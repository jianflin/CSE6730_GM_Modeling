"""
test_simulation.py

Runnable example and pytest smoke tests for simulation.py.
Run directly (python test_simulation.py) or via pytest.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from forces import ViscousParams
from simulation import (
    GaitFunctions,
    SimulationParams,
    simulate,
    integrate_body_twist_to_pose,
)


# ---------------------------------------------------------------------------
# Leg gait helpers
# ---------------------------------------------------------------------------

def leg_angle_profile(amplitude: float, phase: np.ndarray, duty: float) -> np.ndarray:
    """
    Smooth periodic leg angle over one gait cycle [0, 2π).

    Swing phase  (0 to (1-duty)π):           rises 0 → +amplitude
    Stance phase ((1-duty)π to (2-(1-duty))π): swings +amplitude → -amplitude
    Recovery     ((2-(1-duty))π to 2π):       returns -amplitude → 0
    """
    c = np.asarray(phase, dtype=float) % (2 * np.pi)
    swing_end  = (1 - duty) * np.pi
    stance_end = 2 * np.pi - swing_end

    out = np.empty_like(c)
    swing_in  = c < swing_end
    stance    = (c >= swing_end) & (c < stance_end)
    swing_out = c >= stance_end

    out[swing_in]  = amplitude * np.sin(c[swing_in] / (2 * (1 - duty)))
    out[stance]    = amplitude * np.cos(
        (c[stance] - swing_end) * np.pi / (stance_end - swing_end)
    )
    out[swing_out] = -amplitude * np.cos(
        (c[swing_out] - stance_end) / (2 * (1 - duty))
    )
    return out


def leg_contact_weight(phase: np.ndarray, duty: float) -> np.ndarray:
    """Returns 2 during stance, 0 during swing."""
    c = np.asarray(phase, dtype=float) % (2 * np.pi)
    swing_end  = (1 - duty) * np.pi
    stance_end = 2 * np.pi - swing_end
    return np.where((c >= swing_end) & (c < stance_end), 2.0, 0.0)


# ---------------------------------------------------------------------------
# Gait builder: traveling body wave + alternating leg stance/swing
# ---------------------------------------------------------------------------

def build_undulatory_gait(
    N: int,
    num_leg: int,
    body_amplitude: float = np.deg2rad(40),
    temporal_freq: float = 2 * np.pi / 5,
    spatial_freq: float = None,
    leg_amplitude: float = np.deg2rad(10),
    leg_duty: float = 0.5,
    wave_tilt: float = np.deg2rad(45),
    phi_lag: float = 0.0,
    body_activation: float = 1.0,
) -> GaitFunctions:
    """
    Traveling sinusoidal body wave with alternating leg stance/swing cycles.

    Body shape traces a circle in (sin, cos) shape space, producing a
    wave that travels from head to tail.
    """
    if spatial_freq is None:
        spatial_freq = 1.0 - 1.0 / N

    idx      = np.arange(N - 1)
    basis_s  = np.sin(idx * spatial_freq * 2 * np.pi + wave_tilt)
    basis_c  = np.cos(idx * spatial_freq * 2 * np.pi + wave_tilt)
    A        = body_amplitude

    def alpha(t: float) -> np.ndarray:
        return A * np.sin(temporal_freq * t) * basis_s \
             - A * np.cos(temporal_freq * t) * basis_c

    def d_alpha(t: float) -> np.ndarray:
        return A * temporal_freq * np.cos(temporal_freq * t) * basis_s \
             + A * temporal_freq * np.sin(temporal_freq * t) * basis_c

    def phase(t: float) -> float:
        return float(np.arctan2(
            A * np.cos(temporal_freq * t),
            A * np.sin(temporal_freq * t),
        ))

    rest = np.tile([np.pi / 2, -np.pi / 2], num_leg)

    def beta(ph: float) -> np.ndarray:
        out = np.empty(2 * num_leg)
        for k in range(num_leg):
            off = k * spatial_freq * 2 * np.pi + phi_lag
            out[2 * k]     =  leg_angle_profile(leg_amplitude, np.array([ph + off]),          leg_duty)[0]
            out[2 * k + 1] = -leg_angle_profile(leg_amplitude, np.array([ph + off + np.pi]),  leg_duty)[0]
        return out + rest

    def d_beta(ph: float) -> np.ndarray:
        h = 1e-5
        return (beta(ph + h) - beta(ph)) / h

    def activation(ph: float) -> np.ndarray:
        body = np.full(N, body_activation)
        legs = np.empty(2 * num_leg)
        for k in range(num_leg):
            off = k * spatial_freq * 2 * np.pi + phi_lag
            legs[2 * k]     = leg_contact_weight(np.array([ph + off]),          leg_duty)[0]
            legs[2 * k + 1] = leg_contact_weight(np.array([ph + off + np.pi]),  leg_duty)[0]
        return np.concatenate([body, legs])

    return GaitFunctions(
        alpha=alpha, d_alpha=d_alpha,
        phase=phase,
        beta=beta, d_beta=d_beta,
        activation=activation,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_gait_overview(
    t: np.ndarray,
    result,
    gait: GaitFunctions,
    params: SimulationParams,
    figsize: tuple[int, int] = (14, 11),
) -> None:
    """
    Four-panel gait visualization.

    Panel 1 — Body joint angles (heatmap, joint × time)
    Panel 2 — Left leg shoulder angles  (heatmap, leg × time)
    Panel 3 — Right leg shoulder angles (heatmap, leg × time)
    Panel 4 — Contact pattern: left legs (top) and right legs (bottom),
               separated by a dashed line
    """
    N       = params.N
    num_leg = len(params.leg_posi)

    # Recompute activation at each time step
    act_hist    = np.array([gait.activation(gait.phase(float(tk))) for tk in t])
    leg_contact = act_hist[:, N:]           # (n_steps, 2*num_leg)

    # Subtract per-leg time-mean so the rest offset (±π/2) is removed;
    # what remains is the pure swing/stance oscillation around zero.
    beta_dev     = result.beta - result.beta.mean(axis=0)
    left_dev     = beta_dev[:, 0::2]        # (n, num_leg)  deviation only
    right_dev    = beta_dev[:, 1::2]
    left_contact  = leg_contact[:, 0::2]
    right_contact = leg_contact[:, 1::2]

    joint_labels = [str(i + 1) for i in range(N - 1)]
    left_labels  = [f"L{params.leg_posi[k]}" for k in range(num_leg)]
    right_labels = [f"R{params.leg_posi[k]}" for k in range(num_leg)]

    # Independent symmetric limits per angle panel
    body_lim = max(np.abs(result.alpha).max(), 1e-3)
    leg_lim  = max(np.abs(beta_dev).max(), 1e-3)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        4, 1,
        height_ratios=[N - 1, num_leg, num_leg, 2 * num_leg],
        hspace=0.55,
    )

    t_ext  = [t[0], t[-1]]
    cb_kw  = dict(fraction=0.025, pad=0.02)

    # ── Panel 1: body joint angles ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(result.alpha.T,
                     aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[*t_ext, 0.5, N - 0.5],
                     vmin=-body_lim, vmax=body_lim)
    ax1.set_yticks(range(1, N))
    ax1.set_yticklabels(joint_labels)
    ax1.set_ylabel('Body joint')
    ax1.set_title('Body joint angles (rad)')
    plt.colorbar(im1, ax=ax1, **cb_kw)

    # ── Panel 2: left leg shoulder angles (deviation from rest) ────────
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(left_dev.T,
                     aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[*t_ext, 0.5, num_leg + 0.5],
                     vmin=-leg_lim, vmax=leg_lim)
    ax2.set_yticks(range(1, num_leg + 1))
    ax2.set_yticklabels(left_labels)
    ax2.set_ylabel('Left leg')
    ax2.set_title('Left leg shoulder angles — deviation from rest (rad)')
    plt.colorbar(im2, ax=ax2, **cb_kw)

    # ── Panel 3: right leg shoulder angles (deviation from rest) ───────
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(right_dev.T,
                     aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[*t_ext, 0.5, num_leg + 0.5],
                     vmin=-leg_lim, vmax=leg_lim)
    ax3.set_yticks(range(1, num_leg + 1))
    ax3.set_yticklabels(right_labels)
    ax3.set_ylabel('Right leg')
    ax3.set_title('Right leg shoulder angles — deviation from rest (rad)')
    plt.colorbar(im3, ax=ax3, **cb_kw)

    # ── Panel 4: contact pattern (right at bottom, left on top) ────────
    # Stack: rows 1..num_leg = right, rows num_leg+1..2*num_leg = left
    contact_stacked = np.vstack([right_contact.T, left_contact.T])   # (2*num_leg, n)
    ax4 = fig.add_subplot(gs[3])
    ax4.imshow(
        (contact_stacked > 0).astype(float),
        aspect='auto', origin='lower',
        extent=[*t_ext, 0.5, 2 * num_leg + 0.5],
        cmap='Greys', vmin=0, vmax=1,
    )
    # Dashed separator between right (bottom) and left (top)
    ax4.axhline(y=num_leg + 0.5, color='tab:red', linewidth=1.2, linestyle='--')
    ax4.set_yticks(list(range(1, 2 * num_leg + 1)))
    ax4.set_yticklabels(right_labels + left_labels)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Leg')
    ax4.set_title('Contact pattern  (black = stance)  —  dashed: L/R boundary')

    plt.show()


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(
    t: np.ndarray,
    result,
    figsize: tuple[int, int] = (13, 4),
) -> None:
    """
    Two-panel trajectory figure.

    Left  — x(t) and y(t) vs time as line plots
    Right — x-y path in world frame
    """
    _, (ax_t, ax_xy) = plt.subplots(1, 2, figsize=figsize)

    ax_t.plot(t, result.pose[:, 0], color='tab:blue',   lw=2, label='x')
    ax_t.plot(t, result.pose[:, 1], color='tab:orange', lw=2, label='y')
    ax_t.axhline(0, color='k', lw=0.6, linestyle='--')
    ax_t.set_xlabel('Time (s)')
    ax_t.set_ylabel('Position')
    ax_t.set_title('Global position vs time')
    ax_t.legend(loc='upper right')
    ax_t.grid(True, alpha=0.3)

    ax_xy.plot(result.pose[:, 0], result.pose[:, 1], color='#0072BD', lw=2)
    ax_xy.plot(*result.pose[0, :2],  'go', markersize=8, label='start')
    ax_xy.plot(*result.pose[-1, :2], 'rs', markersize=8, label='end')
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xy.set_title('Trajectory (world frame)')
    ax_xy.set_aspect('equal')
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Runnable example
# ---------------------------------------------------------------------------

def run_example(
    show_gait: bool = True,
    show_trajectory: bool = True,
    show_animation: bool = True,
    save_animation: str | None = './animation_preview.gif',
) -> None:
    """
    Parameters
    ----------
    show_gait       : plot gait heatmaps (angles + contact)
    show_trajectory : plot x(t), y(t) and x-y path
    show_animation  : open interactive animation window
    save_animation  : path to save animation (.gif or .mp4); None = don't save
    """
    N       = 6
    leg_posi = list(range(1, N + 1))

    params = SimulationParams(
        N=N,
        leg_posi=leg_posi,
        L=1.3,
        L_leg=0.8,
        model="viscous",
        force_params=ViscousParams(kx=1.0, ky=5.0),
        K=np.eye(3),
    )

    t    = np.linspace(0, 5.0, 200)
    gait = build_undulatory_gait(N=N, num_leg=len(leg_posi))

    result = simulate(t, gait, params)

    print("Final pose   [x, y, theta]:", result.pose[-1])
    print("Final twist  [vx, vy, om] :", result.xi_body[-1])
    print("All solves succeeded       :", result.solver_success.all())

    if show_gait:
        plot_gait_overview(t, result, gait, params)

    if show_trajectory:
        plot_trajectory(t, result)

    if show_animation or save_animation:
        from visualization import animate_locomotion
        animate_locomotion(
            t, result, gait, params,
            show=show_animation,
            save_path=save_animation,
        )


# ---------------------------------------------------------------------------
# pytest smoke tests
# ---------------------------------------------------------------------------

def test_simulate_runs_and_returns_correct_shapes():
    N       = 3
    leg_posi = [1, 2]
    num_leg  = len(leg_posi)
    n_steps  = 20

    params = SimulationParams(
        N=N, leg_posi=leg_posi, L=1.0, L_leg=0.5,
        model="viscous", force_params=ViscousParams(),
    )

    freq = 2 * np.pi
    wave = np.sin(np.arange(N - 1) * 2 * np.pi / N)
    gait = GaitFunctions(
        alpha      = lambda t:  0.3 * np.sin(freq * t) * wave,
        d_alpha    = lambda t:  0.3 * freq * np.cos(freq * t) * wave,
        phase      = lambda t:  freq * t,
        beta       = lambda ph: np.tile([np.pi / 2, -np.pi / 2], num_leg)
                                + 0.1 * np.sin(ph + np.arange(2 * num_leg) * np.pi / num_leg),
        d_beta     = lambda ph: 0.1 * np.cos(ph + np.arange(2 * num_leg) * np.pi / num_leg),
        activation = lambda _:  np.ones(N + 2 * num_leg),
    )

    t      = np.linspace(0, 1.0, n_steps)
    result = simulate(t, gait, params)

    assert result.pose.shape    == (n_steps, 3)
    assert result.xi_body.shape == (n_steps, 3)
    assert result.alpha.shape   == (n_steps, N - 1)
    assert result.beta.shape    == (n_steps, 2 * num_leg)
    assert np.all(np.isfinite(result.pose))
    assert np.all(np.isfinite(result.xi_body))


def test_integrate_body_twist_to_pose_pure_forward():
    """Constant vx=1, vy=0, omega=0 → x reaches 1, y and theta stay 0."""
    n    = 50
    t    = np.linspace(0, 1.0, n)
    xi   = np.zeros((n, 3))
    xi[:, 0] = 1.0
    pose = integrate_body_twist_to_pose(t, xi)
    assert np.allclose(pose[:, 1], 0.0, atol=1e-10)
    assert np.allclose(pose[:, 2], 0.0, atol=1e-10)
    assert np.allclose(pose[-1, 0], 1.0, atol=1e-4)


def test_undulatory_gait_builder_output_shapes():
    N, num_leg = 4, 4
    gait = build_undulatory_gait(N=N, num_leg=num_leg)
    ph   = 0.5
    assert np.asarray(gait.alpha(0.0)).shape      == (N - 1,)
    assert np.asarray(gait.beta(ph)).shape         == (2 * num_leg,)
    assert np.asarray(gait.activation(ph)).shape   == (N + 2 * num_leg,)


if __name__ == "__main__":
    run_example()
