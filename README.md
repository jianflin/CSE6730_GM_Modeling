# CSE 6730 Final Project — Undulatory Locomotion Simulation

Quasi-static simulation of two locomotion systems on a planar surface:

1. **Limbless undulatory robot** in viscous fluid (RFT, linear drag)
2. **Centipede robot** with legs in Coulomb friction (nonlinear RFT)

## Files

| File | Description |
|------|-------------|
| `kinematics.py` | SE(2) Lie group kinematics — segment frames, spatial Jacobian |
| `forces.py` | Force models: viscous, Coulomb, BB (Ding et al.) |
| `solver.py` | Quasi-static force-balance solver (linear + nonlinear) |
| `simulation.py` | Time integration loop; integrates body twist to world pose |
| `test_simulation.py` | Gait builder (`build_undulatory_gait`) and unit tests |
| `experiment_amplitude.py` | Amplitude sweep (0–80 deg); phase-lag sweep for centipede |
| `experiment_trajectories.py` | x(t), y(t), and x-y trajectory at 30/50/70 deg |
| `experiment_snapshots.py` | Body-configuration snapshots at 60 deg amplitude |
| `visualization.py` | Shared plotting utilities |

## Running the Experiments

```bash
# Amplitude sweep + phase-lag sweep
python experiment_amplitude.py

# x-y trajectories for 30, 50, 70 deg
python experiment_trajectories.py

# Body-shape snapshots at 60 deg
python experiment_snapshots.py
```

## Dependencies

- Python 3.10+
- `numpy`, `scipy`, `matplotlib`

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 6 | Number of body segments |
| L | 13 cm | Half-segment length |
| L_leg | 8 cm | Leg length |
| T_cycle | 5 s | Gait period |
| n_cycles | 5 | Simulation duration |

## Output Figures

| File | Contents |
|------|----------|
| `experiment_phase_lag.png` | Phase-lag sweep for centipede |
| `experiment_amplitude.png` | Displacement vs amplitude (limbless + centipede) |
| `trajectories_limbless.png` | x(t), y(t), trajectory — limbless |
| `trajectories_centipede.png` | x(t), y(t), trajectory — centipede |
| `snapshots_60deg.png` | Body snapshots at 60 deg amplitude |

## AI Use Statement 
The authors acknowledge the use of ChatGPT (OpenAI) and Claude (Anthropic) for debugging, code refactoring, test development during programming, and language refinement in the final report.

