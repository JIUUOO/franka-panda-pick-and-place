from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sync_camera_to_robot_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    camera_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    body_name: str,
    local_pos: tuple[float, float, float],
    local_rot: tuple[float, float, float, float],
) -> None:
    robot = env.scene[robot_cfg.name]
    camera = env.scene.sensors[camera_cfg.name]
    body_id = robot.find_bodies(body_name, preserve_order=True)[0][0]

    body_pos = robot.data.body_pos_w[:, body_id]
    body_rot = robot.data.body_quat_w[:, body_id]
    offset_pos = torch.tensor(local_pos, device=env.device, dtype=body_pos.dtype).repeat(env.num_envs, 1)
    offset_rot = torch.tensor(local_rot, device=env.device, dtype=body_rot.dtype).repeat(env.num_envs, 1)

    camera_pos, camera_rot = math_utils.combine_frame_transforms(body_pos, body_rot, offset_pos, offset_rot)
    camera.set_world_poses(camera_pos, camera_rot, convention="world")
