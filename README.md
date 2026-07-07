# Franka Policy Isaac Lab

Simulation-only workspace for recording, training, and evaluating Franka manipulation policies in Isaac Lab.
The repository keeps task presets, recording, conversion, training, and closed-loop evaluation in one reusable pipeline.

## Environment

### Directory Layout

Expected local layout:

```text
/path/to/
├── IsaacLab/
└── franka-policy-isaac-lab/
```

Set the project and Isaac Lab paths manually:

```bash
export ISAACLAB_PATH=/path/to/IsaacLab
export PROJECT_PATH=/path/to/franka-policy-isaac-lab
```

### Conda Environments

Use two separate conda environments:

- `env_isaacsim`: Isaac Sim / Isaac Lab runtime
- `env_lerobot`: LeRobot dataset conversion, training, and policy serving

Do not install LeRobot into the Isaac Sim / Isaac Lab environment.

#### Isaac Lab Environment

Install Isaac Lab separately by following the official Isaac Lab installation guide. This project assumes Isaac Lab is
already cloned, installed, and runnable from `$ISAACLAB_PATH`.

Use `env_isaacsim` for Isaac Sim / Isaac Lab execution:

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH
```

Use this environment for:

- opening Isaac Lab tasks
- teleoperation
- recording Isaac Lab HDF5 demos
- replaying demos
- running Isaac Lab policy evaluation clients

Verify the Isaac Lab environment:

```bash
cd $ISAACLAB_PATH
./isaaclab.sh --help
```

#### LeRobot Environment

Create a separate environment for LeRobot tooling:

```bash
conda create -n env_lerobot python=3.10
conda activate env_lerobot
pip install lerobot
```

Use `env_lerobot` for LeRobot tooling:

```bash
conda activate env_lerobot
cd $PROJECT_PATH
```

Use this environment for:

- converting Isaac Lab HDF5 demos to LeRobot datasets
- inspecting LeRobot datasets
- training ACT / LeRobot policies
- serving trained LeRobot policies during Isaac Lab evaluation

Verify the LeRobot environment:

```bash
python -c "import lerobot; print(lerobot.__version__)"
```

The Isaac Lab evaluation script launches the LeRobot policy server with:

```bash
conda run --no-capture-output -n env_lerobot python -u $PROJECT_PATH/scripts/serve_lerobot_policy.py
```

If the LeRobot environment name is different, pass `--lerobot_env <name>` or
`--lerobot_python /path/to/python` to the evaluation script.

### Isaac Lab Wrapper

The custom task is registered by the project wrapper before running most built-in Isaac Lab scripts:

```bash
cd $ISAACLAB_PATH
./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py <isaaclab-script.py> [args...]
```

Project-owned scripts such as `record_demos_manual_start.py` and `eval_lerobot_act_isaaclab.py` import the custom
task package directly and should be launched with `./isaaclab.sh -p $PROJECT_PATH/scripts/<script>.py`.

### Data Locations

- Isaac Lab HDF5 demos: `$ISAACLAB_PATH/datasets/`
- Converted LeRobot datasets: `$PROJECT_PATH/datasets/lerobot/`
- LeRobot training outputs: `$PROJECT_PATH/outputs/train/`

## Task Presets

Use explicit task ids for new recordings and experiments so datasets, training runs, and evaluation logs can be tied to
a stable task preset.

| Preset | Gym Id | Robot | Task | Cameras | Randomization | Status |
|---|---|---|---|---|---|---|
| `Task1` | `Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0` | Franka Panda | One-cube tabletop pick-and-place | oblique: `/ObliqueCamera` | cube start, cube friction, cube mass | active |
| `Task2` | `Isaac-Stack-Cube-Franka-Task2-IK-Rel-v0` | Franka Panda | Three-cube tabletop stack, blue → red → green | front: `/camera_front`, wrist: `/camera_wrist` | cube start positions | active |
| `Task3` | `Isaac-Open-Drawer-Franka-Task3-IK-Rel-v0` | Franka Panda | Open the lower cabinet drawer | oblique: `/camera_oblique`, wrist: `/camera_wrist` | cabinet pose, robot arm joints, light intensity | active |

### Task1 Details

Task1 reuses Isaac Lab's `Isaac-Lift-Cube-Franka-IK-Rel-v0` configuration and adds:

- cube start around `(0.50, -0.12, 0.055)` with small reset-time randomization
- visible tabletop target square centered at `(0.50, 0.18)`
- single fixed oblique RGB camera observation named `oblique_cam`
- relative IK Franka control with one continuous analog gripper command
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

### Task2 Details

Task2 reuses Isaac Lab's Franka stack IK-relative configuration and adds two RGB camera observations.

- base config: Isaac Lab Franka stack IK-relative config
- task: stack the cubes in blue → red → green order
- action: relative IK Franka control with one continuous analog gripper command
- success: inherited `terminations.success` from the Isaac Lab stack task
- camera observations: `obs/rgb_camera/camera_front`, `obs/rgb_camera/camera_wrist`, `160 x 160`
- cube start positions are randomized by the upstream stack reset event

Task2 camera previews at recorded resolution:

![Task2 160x160 camera previews](docs/images/task2_camera_previews_160.png)

Task2 wrist camera note:

- The wrist camera is synced from the actual `panda_hand` body pose before rendering.
- This avoids stale USD/Fabric link transforms causing incorrect angles or choppy gripper motion.
- Use the debug script to save a wrist RGB preview and pose JSON while tuning camera offsets:

```bash
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  $PROJECT_PATH/scripts/debug_task2_wrist_camera.py \
  --enable_cameras \
  --headless
```

### Task3 Details

Task3 reuses Isaac Lab's Franka cabinet IK-relative configuration and adapts it for lower-drawer opening.

- base config: Isaac Lab Franka cabinet IK-relative config
- task: grasp and open the lower cabinet drawer
- action: relative IK Franka control with one continuous analog gripper command
- success: bottom drawer joint exceeds the configured open threshold
- camera observations: `obs/rgb_camera/camera_oblique`, `obs/rgb_camera/camera_wrist`, `480 x 480`
- reset randomization: cabinet farther `x`, one-sided lateral `y`, small Franka arm joint noise, and light intensity

Task3 camera previews at recorded resolution:

![Task3 480x480 camera previews](docs/images/task3_camera_previews_480.png)

Task3 is intentionally camera-only simple at this stage: fixed oblique camera plus wrist camera, no depth,
segmentation, language, or camera randomization.

## Workflow

The workflow is task-id driven. Set variables once, then reuse the same record/convert/train/eval commands for each
new task preset.

```bash
export TASK_ID=Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0
export RUN_NAME=franka_pick_place_task1_gamepad
export HDF5_DATASET=$ISAACLAB_PATH/datasets/${RUN_NAME}.hdf5
export LEROBOT_DATASET=$PROJECT_PATH/datasets/lerobot/${RUN_NAME}
export LEROBOT_REPO_ID=local/${RUN_NAME}
export ACT_RUN_NAME=act_${RUN_NAME}
export POLICY_DIR=$PROJECT_PATH/outputs/train/${ACT_RUN_NAME}/checkpoints/100000/pretrained_model
```

For Task2 stack experiments, switch only the task/run variables:

```bash
export TASK_ID=Isaac-Stack-Cube-Franka-Task2-IK-Rel-v0
export RUN_NAME=franka_stack_task2_gamepad
```

For Task3 drawer-opening experiments:

```bash
export TASK_ID=Isaac-Open-Drawer-Franka-Task3-IK-Rel-v0
export RUN_NAME=franka_open_drawer_task3_gamepad
```

Current end-to-end pipeline:

```text
open task in Isaac Lab
→ teleoperate
→ record Isaac Lab HDF5 demos
→ inspect HDF5 demos
→ convert HDF5 demos to LeRobot format
→ train an ACT policy with LeRobot
→ evaluate the trained policy in Isaac Lab
```

Unless noted otherwise, Isaac Lab commands run in `env_isaacsim`, and LeRobot commands run in `env_lerobot`.

## 1. Open a Task

Run a random policy to verify that the selected task loads, camera observations are available, and reset behavior is
working.

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/random_agent.py \
  --task $TASK_ID \
  --num_envs 1 \
  --enable_cameras
```

## 2. Teleoperate

### Gamepad

Gamepad teleoperation is the recommended control path for recording demos.

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task $TASK_ID \
  --num_envs 1 \
  --teleop_device gamepad \
  --enable_cameras
```

Optional sensitivity tuning:

```bash
./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task $TASK_ID \
  --num_envs 1 \
  --teleop_device gamepad \
  --sensitivity 0.5 \
  --enable_cameras
```

### Keyboard

Keyboard teleoperation remains available for quick debugging.

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/environments/teleoperation/teleop_se3_agent.py \
  --task $TASK_ID \
  --num_envs 1 \
  --teleop_device keyboard \
  --enable_cameras
```

To inspect the fixed oblique camera in Isaac Sim, launch a teleoperation command, select
`/World/envs/env_0/ObliqueCamera` in the Stage panel, and set the viewport camera to that prim.

## 3. Record Demonstrations

Use the project manual-start recorder for gamepad demonstrations. It launches the task, waits without recording, and
starts a clean demo only after the gamepad start input.

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/record_demos_manual_start.py \
  --task $TASK_ID \
  --teleop_device gamepad \
  --dataset_file $HDF5_DATASET \
  --num_demos 10 \
  --num_success_steps 1 \
  --enable_cameras
```

Recording controls:

- `A`: reset and start recording demo `#N`
- `LB` + `RB`: discard/reset the current attempt
- terminal logs: `[WAIT]`, `[START]`, `[SUCCESS]`, `[EXPORT]`, `[DONE]`

`--num_success_steps 1` exports a demo as soon as the success condition is detected once. Increase it if successful
placements should remain stable for multiple consecutive steps before export.

Keyboard recording can still use Isaac Lab's built-in recorder through the wrapper:

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/record_demos.py \
  --task $TASK_ID \
  --teleop_device keyboard \
  --dataset_file $ISAACLAB_PATH/datasets/${RUN_NAME}_keyboard.hdf5 \
  --num_demos 1 \
  --num_success_steps 1 \
  --enable_cameras
```

## 4. Inspect Isaac Lab HDF5 Demos

Before converting demos to LeRobot format, verify that actions and camera frames were recorded with matching timesteps.

```bash
cd $PROJECT_PATH

python scripts/inspect_isaaclab_hdf5.py \
  $HDF5_DATASET \
  --list \
  --camera_path obs/rgb_camera/oblique_cam \
  --preview ./datasets/${RUN_NAME}_oblique_cam_preview.png
```

For Task2 and Task3, set `--camera_path` to one of the task camera paths listed below.

Task1 camera path:

```text
data/demo_0/obs/rgb_camera/oblique_cam
```

Task2 camera paths:

```text
data/demo_0/obs/rgb_camera/camera_front
data/demo_0/obs/rgb_camera/camera_wrist
```

Task3 camera paths:

```text
data/demo_0/obs/rgb_camera/camera_oblique
data/demo_0/obs/rgb_camera/camera_wrist
```

Task1 expected key shapes:

```text
actions: (T, 7)
obs/joint_pos: (T, 9)
obs/rgb_camera/oblique_cam: (T, 256, 256, 3)
```

Task2 expected image shape is `(T, 160, 160, 3)` for each camera stream.
Task3 expected image shape is `(T, 480, 480, 3)` for each camera stream.

## 5. Convert to LeRobot

Convert the Isaac Lab HDF5 dataset into a local LeRobot dataset.

```bash
conda activate env_lerobot
cd $PROJECT_PATH

python scripts/convert_isaaclab_hdf5_to_lerobot.py \
  $HDF5_DATASET \
  --output_root $LEROBOT_DATASET \
  --repo_id $LEROBOT_REPO_ID \
  --overwrite \
  --no_videos
```

Without `--camera`, the converter uses the Task1 default `obs/rgb_camera/oblique_cam`.

For multi-camera datasets, pass each camera explicitly:

```bash
python scripts/convert_isaaclab_hdf5_to_lerobot.py \
  $HDF5_DATASET \
  --output_root $LEROBOT_DATASET \
  --repo_id $LEROBOT_REPO_ID \
  --overwrite \
  --no_videos \
  --camera obs/rgb_camera/camera_front:observation.images.camera_front \
  --camera obs/rgb_camera/camera_wrist:observation.images.camera_wrist
```

For Task3, use:

```bash
--camera obs/rgb_camera/camera_oblique:observation.images.camera_oblique \
--camera obs/rgb_camera/camera_wrist:observation.images.camera_wrist
```

Validate the converted dataset:

```bash
python - <<'PY'
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id=os.environ["LEROBOT_REPO_ID"],
    root=os.environ["LEROBOT_DATASET"],
)

print(ds)
sample = ds[0]
image_keys = [key for key in sample if key.startswith("observation.images.")]
for key in image_keys:
    print(key, sample[key].shape, sample[key].dtype)
print("state", sample["observation.state"].shape, sample["observation.state"].dtype)
print("action", sample["action"].shape, sample["action"].dtype)
PY
```

## 6. Train ACT

Train an ACT policy with LeRobot. Use `--policy.push_to_hub=false` for local-only experiments.

```bash
conda activate env_lerobot
cd $PROJECT_PATH

lerobot-train \
  --dataset.repo_id=$LEROBOT_REPO_ID \
  --dataset.root=$LEROBOT_DATASET \
  --policy.type=act \
  --policy.repo_id=local/${ACT_RUN_NAME} \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/${ACT_RUN_NAME} \
  --job_name=${ACT_RUN_NAME} \
  --policy.device=cuda \
  --steps=100000 \
  --wandb.enable=false
```

The trained policy is saved under:

```text
outputs/train/<ACT_RUN_NAME>/checkpoints/<step>/pretrained_model/
```

For a quick pipeline sanity check, reduce `--steps` before running a long training job.

## 7. Check Policy Loading

Verify that the trained ACT checkpoint loads before running closed-loop Isaac Lab evaluation.

```bash
conda activate env_lerobot
cd $PROJECT_PATH

python - <<'PY'
import os
from lerobot.policies.act.modeling_act import ACTPolicy

policy_path = os.environ["POLICY_DIR"]
policy = ACTPolicy.from_pretrained(policy_path, local_files_only=True)
policy.to("cuda")
policy.eval()
policy.reset()

print("loaded:", policy_path)
print("device:", next(policy.parameters()).device)
print("policy config:", policy.config.type)
PY
```

## 8. Evaluate ACT in Isaac Lab

The evaluation client runs in `env_isaacsim` and automatically launches the LeRobot policy server in `env_lerobot`.
Reset-time domain randomization is preserved during evaluation because each episode calls the selected task environment
reset normally.

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/eval_lerobot_act_isaaclab.py \
  --task $TASK_ID \
  --policy_path $POLICY_DIR \
  --num_episodes 10 \
  --max_steps 500 \
  --num_success_steps 1 \
  --enable_cameras
```

Useful evaluation options:

| Option | Meaning |
|---|---|
| `--num_episodes` | Number of evaluation episodes |
| `--max_steps` | Maximum rollout steps per episode |
| `--num_success_steps` | Consecutive success steps required before counting success |
| `--max_translation` | Clip policy translation actions |
| `--max_rotation` | Clip policy rotation actions |
| `--port` | Local TCP port for the LeRobot policy server |
| `--lerobot_env` | Conda environment used for the policy server |
| `--lerobot_python` | Explicit Python executable for the policy server |

## 9. Replay Demos

```bash
conda activate env_isaacsim
cd $ISAACLAB_PATH

./isaaclab.sh -p $PROJECT_PATH/scripts/run_isaaclab_with_tasks.py \
  scripts/tools/replay_demos.py \
  --task $TASK_ID \
  --dataset_file $HDF5_DATASET \
  --num_envs 1 \
  --enable_cameras
```

## Controls

### Gamepad Controls

The project gamepad mapping is tuned for tabletop manipulation: the right stick handles planar motion, the d-pad
handles pitch/yaw trim, the left stick handles vertical motion and roll trim, and RT controls the gripper analog value.

| Action | Control |
|---|---|
| Start recording | `A` button |
| Reset environment | `LB` + `RB` |
| Analog gripper close | `RT` trigger |
| Toggle gripper | `X` button fallback |
| Move +X / -X | Right stick up / down |
| Move +Y / -Y | Right stick left / right |
| Move +Z / -Z | Left stick up / down |
| Roll | Left stick left / right |
| Pitch | D-pad down / up |
| Yaw | D-pad left / right |

### Keyboard Controls

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

- Task presets should use explicit gym ids such as `Isaac-PickPlace-Cube-Franka-Task<N>-IK-Rel-v0`.
- Demo files are saved under `$ISAACLAB_PATH/datasets/`.
