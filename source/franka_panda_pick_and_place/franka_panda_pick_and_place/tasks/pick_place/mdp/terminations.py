from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_in_target_square(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.50, 0.18),
    half_extent: float = 0.06,
    cube_half_extent: float = 0.025,
    expected_rest_z: float = 0.055,
    z_threshold: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return true when the full cube footprint is inside the target square and resting on the table."""
    cube: RigidObject = env.scene[object_cfg.name]
    cube_pos_env = cube.data.root_pos_w - env.scene.env_origins
    allowed_half_extent = half_extent - cube_half_extent

    x_ok = torch.abs(cube_pos_env[:, 0] - target_xy[0]) < allowed_half_extent
    y_ok = torch.abs(cube_pos_env[:, 1] - target_xy[1]) < allowed_half_extent
    z_ok = torch.abs(cube_pos_env[:, 2] - expected_rest_z) < z_threshold

    return x_ok & y_ok & z_ok
