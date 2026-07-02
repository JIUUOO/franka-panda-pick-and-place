from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg

from . import mdp


TARGET_XY = (0.50, 0.18)
TARGET_SIZE = 0.12
CUBE_HALF_EXTENT = 0.025
CUBE_START_POS = (0.50, -0.12, 0.055)
CUBE_REST_Z = CUBE_START_POS[2]


@configclass
class PickPlaceTableSceneCfg(ObjectTableSceneCfg):
    """Lift scene plus a visible tabletop target square."""

    target_square = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetSquare",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TARGET_XY[0], TARGET_XY[1], 0.003]),
        spawn=sim_utils.CuboidCfg(
            size=(TARGET_SIZE, TARGET_SIZE, 0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.85),
        ),
    )


@configclass
class FrankaCubePickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    """Relative-IK Franka pick-and-place task built from Isaac Lab's lift task."""

    scene: PickPlaceTableSceneCfg = PickPlaceTableSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        self.scene.object.init_state = RigidObjectCfg.InitialStateCfg(pos=CUBE_START_POS, rot=[1, 0, 0, 0])
        self.events.reset_object_position.func = lift_mdp.reset_root_state_uniform
        self.events.reset_object_position.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        }

        self.commands.object_pose.ranges.pos_x = (TARGET_XY[0], TARGET_XY[0])
        self.commands.object_pose.ranges.pos_y = (TARGET_XY[1], TARGET_XY[1])
        self.commands.object_pose.ranges.pos_z = (CUBE_REST_Z, CUBE_REST_Z)

        self.terminations.success = DoneTerm(
            func=mdp.cube_in_target_square,
            params={
                "target_xy": TARGET_XY,
                "half_extent": TARGET_SIZE / 2.0,
                "cube_half_extent": CUBE_HALF_EXTENT,
                "expected_rest_z": CUBE_REST_Z,
                "z_threshold": 0.01,
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
