from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_in_target_square(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.50, 0.18),
    half_extent: float = 0.06,
    cube_half_extent: float = 0.025,
    max_rest_z: float = 0.07,
    min_rest_z: float = -0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return true when the cube footprint is in the target and not lifted above table-resting height."""
    cube: RigidObject = env.scene[object_cfg.name]
    cube_pos_env = cube.data.root_pos_w - env.scene.env_origins
    allowed_half_extent = half_extent - cube_half_extent

    x_ok = torch.abs(cube_pos_env[:, 0] - target_xy[0]) < allowed_half_extent
    y_ok = torch.abs(cube_pos_env[:, 1] - target_xy[1]) < allowed_half_extent
    z_ok = (cube_pos_env[:, 2] <= max_rest_z) & (cube_pos_env[:, 2] >= min_rest_z)

    return x_ok & y_ok & z_ok


def cabinet_drawer_opened(
    env: ManagerBasedRLEnv,
    threshold: float = 0.35,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
) -> torch.Tensor:
    cabinet: Articulation = env.scene[cabinet_cfg.name]
    joint_ids = cabinet_cfg.joint_ids
    if joint_ids is None:
        joint_ids = cabinet.find_joints(cabinet_cfg.joint_names, preserve_order=True)[0]
    return cabinet.data.joint_pos[:, joint_ids[0]] > threshold
