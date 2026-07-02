# Franka Panda Pick-and-Place

Minimal custom Isaac Lab task for one-cube Franka Panda tabletop pick-and-place.

## Environment

Set the project and Isaac Lab paths:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export PROJECT_PATH=/path/to/franka-panda-pick-and-place
```

The custom task is registered by the project wrapper before running Isaac Lab scripts:

```bash
cd $ISAACLAB_PATH
./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py <isaaclab-script.py> [args...]
```

## Task

Gym id:

```text
Isaac-PickPlace-Cube-Franka-IK-Rel-v0
```

This task reuses Isaac Lab's `Isaac-Lift-Cube-Franka-IK-Rel-v0` configuration and adds:

- fixed cube start at `(0.50, -0.12, 0.055)`
- visible tabletop target square centered at `(0.50, 0.18)`
- relative IK Franka control
- `terminations.success` for demo recording

## 1. Run Random Policy

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/random_agent.py \
  --task Isaac-PickPlace-Cube-Franka-IK-Rel-v0 \
  --num_envs 1
```

## 2. Run Keyboard Teleoperation

Relative IK control for end-effector teleoperation:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard
```

## 3. Record One Keyboard Demo

By default, `record_demos.py` requires the success condition to stay true for `10` consecutive steps before exporting a successful demo. For quick validation, set `--num_success_steps 1`.

Quick success-check recording:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/record_demos.py \
  --task Isaac-PickPlace-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --dataset_file ./datasets/pick_place_cube_keyboard.hdf5 \
  --num_demos 1 \
  --num_success_steps 1
```

Stable recording:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/record_demos.py \
  --task Isaac-PickPlace-Cube-Franka-IK-Rel-v0 \
  --teleop_device keyboard \
  --dataset_file ./datasets/pick_place_cube_keyboard.hdf5 \
  --num_demos 1
```

## 4. Replay Demo

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/replay_demos.py \
  --task Isaac-PickPlace-Cube-Franka-IK-Rel-v0 \
  --dataset_file ./datasets/pick_place_cube_keyboard.hdf5 \
  --num_envs 1
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

- Base task: `Isaac-Lift-Cube-Franka-IK-Rel-v0`.
- Target square size: `0.12 m x 0.12 m`.
- Success: cube root is within target x/y bounds and near resting height.
- Demo files are saved under `$ISAACLAB_PATH/datasets/`.
