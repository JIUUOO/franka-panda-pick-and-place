from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_vec(text: str, length: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != length:
        raise argparse.ArgumentTypeError(f"{name} must have {length} comma-separated values.")
    return values


def _save_rgb(path: Path, rgb):
    import numpy as np

    image = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else rgb
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] > 3:
        image = image[..., :3]
    image = image.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image).save(path)
    except Exception:
        ppm_path = path.with_suffix(".ppm")
        with ppm_path.open("wb") as file:
            height, width, _ = image.shape
            file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            file.write(image.tobytes())
        print(f"PIL unavailable; saved PPM preview to: {ppm_path}")


def _tensor_list(value):
    return [round(float(item), 6) for item in value.detach().cpu().flatten().tolist()]


def _quat_norm(value) -> float:
    return round(sum(float(item) * float(item) for item in value) ** 0.5, 6)


parser = argparse.ArgumentParser(description="Inspect Task2 wrist camera actual sensor pose and preview image.")
parser.add_argument("--task", default="Isaac-Stack-Cube-Franka-Task2-IK-Rel-v0")
parser.add_argument("--output", default="./datasets/debug/task2_wrist_camera_preview.png")
parser.add_argument("--json_output", default="./datasets/debug/task2_wrist_camera_pose.json")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--wrist_pos", type=lambda text: _parse_vec(text, 3, "wrist_pos"), default=None)
parser.add_argument("--wrist_rot", type=lambda text: _parse_vec(text, 4, "wrist_rot"), default=None)
parser.add_argument("--direct_child", action="store_true")
parser.add_argument("--root_camera", action="store_true")
parser.add_argument("--sync_from_hand", action="store_true")
parser.add_argument("--look_at", type=lambda text: _parse_vec(text, 3, "look_at"), default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# PLACEHOLDER: Extension template (do not remove this comment)

import gymnasium as gym
import torch
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
cfg.scene.camera_front = None
cfg.observations.rgb_camera.camera_front = None
cfg.image_obs_list = ["camera_wrist"]

if args.wrist_pos is not None:
    if args.direct_child or args.root_camera:
        cfg.scene.camera_wrist.offset.pos = args.wrist_pos
    else:
        cfg.scene.wrist_camera_mount.init_state.pos = args.wrist_pos
if args.wrist_rot is not None:
    if args.direct_child or args.root_camera:
        cfg.scene.camera_wrist.offset.rot = args.wrist_rot
    else:
        cfg.scene.wrist_camera_mount.init_state.rot = args.wrist_rot
if args.root_camera:
    cfg.scene.wrist_camera_mount = None
    cfg.scene.camera_wrist.prim_path = "{ENV_REGEX_NS}/camera_wrist"
elif args.direct_child:
    cfg.scene.wrist_camera_mount = None
    cfg.scene.camera_wrist.prim_path = "{ENV_REGEX_NS}/Robot/panda_hand/camera_wrist"

env = gym.make(args.task, cfg=cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene["robot"]
camera = env.unwrapped.scene.sensors["camera_wrist"]

hand_id = robot.find_bodies("panda_hand", preserve_order=True)[0][0]
link7_id = robot.find_bodies("panda_link7", preserve_order=True)[0][0]
synced_camera_position = None
synced_camera_orientation = None


def _sync_camera_from_hand():
    global synced_camera_position, synced_camera_orientation
    hand_position = robot.data.body_pos_w[:, hand_id]
    hand_orientation = robot.data.body_quat_w[:, hand_id]
    local_position = torch.tensor([cfg.scene.camera_wrist.offset.pos], device=hand_position.device)
    local_orientation = torch.tensor([cfg.scene.camera_wrist.offset.rot], device=hand_position.device)
    camera_position, camera_orientation = math_utils.combine_frame_transforms(
        hand_position,
        hand_orientation,
        local_position,
        local_orientation,
    )
    if args.look_at is not None:
        target = torch.tensor([args.look_at], device=hand_position.device)
        rotation_matrix = math_utils.create_rotation_matrix_from_view(
            camera_position,
            target,
            up_axis="Z",
            device=str(hand_position.device),
        )
        camera_orientation = math_utils.quat_from_matrix(rotation_matrix)
        synced_camera_orientation = math_utils.convert_camera_frame_orientation_convention(
            camera_orientation,
            origin="opengl",
            target="world",
        ).clone()
        camera.set_world_poses(camera_position, camera_orientation, convention="opengl")
        synced_camera_position = camera_position.clone()
        return
    synced_camera_position = camera_position.clone()
    synced_camera_orientation = camera_orientation.clone()
    camera.set_world_poses(camera_position, camera_orientation, convention="world")


for _ in range(args.steps):
    if args.sync_from_hand:
        _sync_camera_from_hand()
    zero_action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    obs, _, _, _, _ = env.step(zero_action)

if args.sync_from_hand:
    _sync_camera_from_hand()
    env.unwrapped.sim.render()
    camera.update(0.0, force_recompute=True)
    obs = env.unwrapped.observation_manager.compute()

hand_pos = robot.data.body_pos_w[0, hand_id]
hand_quat = robot.data.body_quat_w[0, hand_id]
link7_pos = robot.data.body_pos_w[0, link7_id]
link7_quat = robot.data.body_quat_w[0, link7_id]
cam_pos = camera.data.pos_w[0]
cam_quat_world = camera.data.quat_w_world[0]
view_pos, view_quat = camera._view.get_world_poses()

cam_pos_in_hand, cam_quat_in_hand = math_utils.subtract_frame_transforms(hand_pos, hand_quat, cam_pos, cam_quat_world)
suggested_local_pos = None
suggested_local_quat = None
if synced_camera_position is not None and synced_camera_orientation is not None:
    suggested_local_pos, suggested_local_quat = math_utils.subtract_frame_transforms(
        hand_pos.unsqueeze(0),
        hand_quat.unsqueeze(0),
        synced_camera_position,
        synced_camera_orientation,
    )
cam_forward_w = math_utils.quat_apply(cam_quat_world.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=cam_pos.device))[0]
cam_up_w = math_utils.quat_apply(cam_quat_world.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=cam_pos.device))[0]

result = {
    "direct_child": args.direct_child,
    "root_camera": args.root_camera,
    "sync_from_hand": args.sync_from_hand,
    "configured_wrist_mount_pos": None if not hasattr(cfg.scene, "wrist_camera_mount") or cfg.scene.wrist_camera_mount is None else list(cfg.scene.wrist_camera_mount.init_state.pos),
    "configured_wrist_mount_rot_wxyz": None if not hasattr(cfg.scene, "wrist_camera_mount") or cfg.scene.wrist_camera_mount is None else list(cfg.scene.wrist_camera_mount.init_state.rot),
    "configured_wrist_mount_rot_norm": None if not hasattr(cfg.scene, "wrist_camera_mount") or cfg.scene.wrist_camera_mount is None else _quat_norm(cfg.scene.wrist_camera_mount.init_state.rot),
    "configured_camera_offset_pos": list(cfg.scene.camera_wrist.offset.pos),
    "configured_camera_offset_rot_wxyz": list(cfg.scene.camera_wrist.offset.rot),
    "configured_camera_offset_rot_norm": _quat_norm(cfg.scene.camera_wrist.offset.rot),
    "camera_prim": camera.cfg.prim_path,
    "panda_hand_pos_w": _tensor_list(hand_pos),
    "panda_hand_quat_wxyz": _tensor_list(hand_quat),
    "panda_link7_pos_w": _tensor_list(link7_pos),
    "panda_link7_quat_wxyz": _tensor_list(link7_quat),
    "camera_pos_w": _tensor_list(cam_pos),
    "camera_quat_world_wxyz": _tensor_list(cam_quat_world),
    "camera_view_pos_after_sync": _tensor_list(view_pos[0]),
    "camera_view_quat_opengl_after_sync": _tensor_list(view_quat[0]),
    "synced_desired_camera_pos_w": None if synced_camera_position is None else _tensor_list(synced_camera_position[0]),
    "synced_desired_camera_quat_world_wxyz": None
    if synced_camera_orientation is None
    else _tensor_list(synced_camera_orientation[0]),
    "suggested_local_pos_in_panda_hand": None if suggested_local_pos is None else _tensor_list(suggested_local_pos[0]),
    "suggested_local_quat_in_panda_hand_wxyz": None
    if suggested_local_quat is None
    else _tensor_list(suggested_local_quat[0]),
    "camera_quat_ros_wxyz": _tensor_list(camera.data.quat_w_ros[0]),
    "camera_quat_opengl_wxyz": _tensor_list(camera.data.quat_w_opengl[0]),
    "camera_pos_in_panda_hand": _tensor_list(cam_pos_in_hand),
    "camera_quat_in_panda_hand_world_convention": _tensor_list(cam_quat_in_hand),
    "camera_forward_axis_w": _tensor_list(cam_forward_w),
    "camera_up_axis_w": _tensor_list(cam_up_w),
    "joint_pos": _tensor_list(robot.data.joint_pos[0]),
    "rgb_shape": list(obs["rgb_camera"]["camera_wrist"].shape),
}

json_path = Path(args.json_output)
json_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(result, indent=2))
_save_rgb(Path(args.output), obs["rgb_camera"]["camera_wrist"])

print(json.dumps(result, indent=2))
print(f"Saved wrist preview: {args.output}")
print(f"Saved pose debug: {args.json_output}")

env.close()
simulation_app.close()
