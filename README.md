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
Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0
```

Backward-compatible alias:

```text
Isaac-PickPlace-Cube-Franka-IK-Rel-v0
```

Use the `Task1` id for new recordings so future datasets can be tied to a specific task preset.

This task reuses Isaac Lab's `Isaac-Lift-Cube-Franka-IK-Rel-v0` configuration and adds:

- cube start around `(0.50, -0.12, 0.055)` with small reset-time randomization
- visible tabletop target square centered at `(0.50, 0.18)`
- single fixed oblique RGB camera observation named `oblique_cam`
- relative IK Franka control
- `terminations.success` for demo recording

Camera setup:

- camera prim: `/World/envs/env_0/ObliqueCamera`
- observation group: `rgb_camera`
- observation key: `oblique_cam`
- resolution: `256 x 256`
- pose: fixed oblique table view from `(1.25, -0.9, 0.95)`

Domain randomization:

- cube reset position offset: `x ∈ [-0.04, 0.04] m`, `y ∈ [-0.05, 0.05] m`
- cube friction randomized per reset: static `0.7–1.2`, dynamic `0.5–1.0`
- cube mass randomized per reset: scale `0.85–1.15`
- camera pose and target square stay fixed for the initial ACT data collection pass

## 1. Run Random Policy

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/random_agent.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --num_envs 1 \
  --enable_cameras
```

## 2. Run Keyboard Teleoperation

Relative IK control for end-effector teleoperation:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras
```

To inspect the fixed oblique camera in Isaac Sim, launch the task with the command above, then select
`/World/envs/env_0/ObliqueCamera` in the Stage panel and set the viewport camera to that prim. If you only
need to confirm that the camera sensor exists, the scene should include the `ObliqueCamera` prim under `env_0`.

## 3. Run Gamepad Teleoperation

Connect the controller before launching Isaac Lab. Isaac Lab uses the first detected gamepad device.

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device gamepad \
  --enable_cameras
```

Optional sensitivity tuning:

```bash
./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device gamepad \
  --sensitivity 0.5 \
  --enable_cameras
```

## 4. Record One Keyboard Demo

By default, `record_demos.py` requires the success condition to stay true for `10` consecutive steps before exporting a successful demo. For quick validation, set `--num_success_steps 1`.
When launched through the project wrapper, camera frames are recorded to `obs/rgb_camera/oblique_cam` in the HDF5 dataset.

Quick success-check recording:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/record_demos.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --teleop_device keyboard \
  --dataset_file ./datasets/pick_place_cube_keyboard.hdf5 \
  --num_demos 1 \
  --num_success_steps 1 \
  --enable_cameras
```

Stable recording:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/record_demos.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --teleop_device keyboard \
  --dataset_file ./datasets/pick_place_cube_keyboard.hdf5 \
  --num_demos 1 \
  --enable_cameras
```

## 5. Record One Gamepad Demo

Recommended flow for gamepad demos is the project manual-start recorder. It launches the environment,
waits without recording, then starts a clean demo after the gamepad start input.

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/record_demos_manual_start.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --teleop_device gamepad \
  --dataset_file ./datasets/pick_place_cube_gamepad.hdf5 \
  --num_demos 1 \
  --num_success_steps 1 \
  --enable_cameras
```

Recording controls:

- press `A` to reset and start recording demo `#N`
- press `LB` + `RB` to discard/reset the current attempt
- terminal logs show `[WAIT]`, `[START]`, `[SUCCESS]`, `[EXPORT]`, and `[DONE]`

## 6. Replay Demo

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/replay_demos.py \
  --task Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0 \
  --dataset_file ./datasets/pick_place_cube_gamepad.hdf5 \
  --num_envs 1 \
  --enable_cameras
```

## 7. Inspect Recorded Dataset

Before converting demos to LeRobot format, verify that actions and camera frames were recorded with matching
timesteps:

```bash
cd $PROJECT_PATH

python scripts/inspect_isaaclab_hdf5.py \
  $ISAACLAB_PATH/datasets/pick_place_cube_gamepad.hdf5 \
  --list \
  --preview ./datasets/oblique_cam_preview.png
```

Expected camera path:

```text
data/demo_0/obs/rgb_camera/oblique_cam
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

## Gamepad Controls

The project gamepad mapping is tuned for tabletop pick-and-place: the right stick handles planar motion, and the d-pad handles vertical motion.

| Action | Control |
|---|---|
| Start recording | `A` button |
| Reset environment | `LB` + `RB` |
| Toggle gripper | `X` button |
| Move +X / -X | Right stick up / down |
| Move +Y / -Y | Right stick left / right |
| Move +Z / -Z | D-pad up / down |
| Roll | D-pad left / right |
| Pitch | Left stick up / down |
| Yaw | Left stick right / left |

## Task Notes

- Base task: `Isaac-Lift-Cube-Franka-IK-Rel-v0`.
- Target square size: `0.12 m x 0.12 m`.
- Success: cube footprint is inside the target square and cube root is not lifted above resting height.
- Domain randomization is intentionally conservative so manual gamepad demos remain recordable.
- Demo files are saved under `$ISAACLAB_PATH/datasets/`.
