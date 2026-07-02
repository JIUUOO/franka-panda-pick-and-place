# Franka Panda Pick-and-Place

Minimal Franka Panda manipulation experiments using Isaac Lab built-in tasks.

## Environment

Isaac Lab path:

```bash
cd /path/to/IsaacLab
```

## 1. Run basic lift task

Default Franka **lift-cube environment**.

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --num_envs 1
```

## 2. Run keyboard teleoperation

Relative IK control for end-effector teleoperation.

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard
```

## 3. Record one keyboard demonstration

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --dataset_file ./datasets/lift_cube_keyboard.hdf5 \
  --num_demos 1
```

## Keyboard Controls

| Action | Key |
|---|---|
| Move X | `W` / `S` |
| Move Y | `A` / `D` |
| Move Z | `Q` / `E` |
| Toggle gripper | `K` |
| Roll | `Z` / `X` |
| Pitch | `T` / `G` |
| Yaw | `C` / `V` |

## Task Notes

- `Isaac-Lift-Cube-Franka-v0`: default joint-control lift task.
- `Isaac-Lift-Cube-Franka-IK-Rel-v0`: relative IK task, better for keyboard teleop and action-space experiments.
- Demo files are saved under `/path/to/IsaacLab/datasets/`.
