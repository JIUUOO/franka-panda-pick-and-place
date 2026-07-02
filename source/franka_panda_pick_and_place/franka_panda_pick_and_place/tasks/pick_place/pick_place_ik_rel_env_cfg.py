from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from franka_panda_pick_and_place.devices import FrankaPickPlaceGamepadCfg

from . import mdp


TASK1_TARGET_XY = (0.50, 0.18)
TASK1_TARGET_SIZE = 0.12
CUBE_HALF_EXTENT = 0.025
TASK1_CUBE_START_POS = (0.50, -0.12, 0.055)
TASK1_CUBE_REST_Z = TASK1_CUBE_START_POS[2]
TASK1_CUBE_START_X_RANGE = (-0.04, 0.04)
TASK1_CUBE_START_Y_RANGE = (-0.05, 0.05)
OBLIQUE_CAMERA_POS = (1.25, -0.9, 0.95)
OBLIQUE_CAMERA_ROT = (-0.41563, 0.84342, 0.30536, -0.15048)
OBLIQUE_CAMERA_RESOLUTION = (256, 256)
FRANKA_GRIPPER_DOWN_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.1894,
    "panda_joint3": 0.0,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.04,
}


@configclass
class PickPlaceTableSceneCfg(ObjectTableSceneCfg):
    """Task 1 lift scene plus a visible tabletop target square."""

    target_square = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetSquare",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[TASK1_TARGET_XY[0], TASK1_TARGET_XY[1], 0.003]),
        spawn=sim_utils.CuboidCfg(
            size=(TASK1_TARGET_SIZE, TASK1_TARGET_SIZE, 0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.85),
        ),
    )


@configclass
class PickPlaceObservationsCfg(ObservationsCfg):
    """Lift observations plus a fixed oblique RGB camera stream."""

    @configclass
    class RGBCameraCfg(ObsGroup):
        """RGB image observations for visuomotor imitation learning."""

        oblique_cam = ObsTerm(
            func=lift_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("oblique_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb_camera: RGBCameraCfg = RGBCameraCfg()


@configclass
class FrankaCubePickPlaceTask1EnvCfg(FrankaCubeLiftEnvCfg):
    """Task 1 preset: one-cube tabletop pick-and-place with conservative reset randomization."""

    scene: PickPlaceTableSceneCfg = PickPlaceTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PickPlaceObservationsCfg = PickPlaceObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.init_state.joint_pos.update(FRANKA_GRIPPER_DOWN_JOINT_POS)

        self.events.reset_all.params = {"reset_joint_targets": True}

        self.scene.object.init_state = RigidObjectCfg.InitialStateCfg(pos=TASK1_CUBE_START_POS, rot=[1, 0, 0, 0])
        self.events.reset_object_position.func = lift_mdp.reset_root_state_uniform
        self.events.reset_object_position.params = {
            "pose_range": {"x": TASK1_CUBE_START_X_RANGE, "y": TASK1_CUBE_START_Y_RANGE, "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        }
        self.events.randomize_object_material = EventTerm(
            func=lift_mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "static_friction_range": (0.7, 1.2),
                "dynamic_friction_range": (0.5, 1.0),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 16,
                "make_consistent": True,
            },
        )
        self.events.randomize_object_mass = EventTerm(
            func=lift_mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "mass_distribution_params": (0.85, 1.15),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        self.commands.object_pose.ranges.pos_x = (TASK1_TARGET_XY[0], TASK1_TARGET_XY[0])
        self.commands.object_pose.ranges.pos_y = (TASK1_TARGET_XY[1], TASK1_TARGET_XY[1])
        self.commands.object_pose.ranges.pos_z = (TASK1_CUBE_REST_Z, TASK1_CUBE_REST_Z)

        self.terminations.success = DoneTerm(
            func=mdp.cube_in_target_square,
            params={
                "target_xy": TASK1_TARGET_XY,
                "half_extent": TASK1_TARGET_SIZE / 2.0,
                "cube_half_extent": CUBE_HALF_EXTENT,
                "max_rest_z": TASK1_CUBE_REST_Z + 0.005,
                "min_rest_z": -0.02,
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        self.scene.oblique_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ObliqueCamera",
            update_period=0.0,
            height=OBLIQUE_CAMERA_RESOLUTION[0],
            width=OBLIQUE_CAMERA_RESOLUTION[1],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=OBLIQUE_CAMERA_POS, rot=OBLIQUE_CAMERA_ROT, convention="ros"),
        )

        self.scene.num_envs = 1
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.observations.policy.enable_corruption = False
        self.image_obs_list = ["oblique_cam"]
        self.teleop_devices = DevicesCfg(
            devices={
                "gamepad": FrankaPickPlaceGamepadCfg(
                    pos_sensitivity=0.1,
                    rot_sensitivity=0.1,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class FrankaCubePickPlaceEnvCfg(FrankaCubePickPlaceTask1EnvCfg):
    """Backward-compatible alias for the current Task 1 preset."""

    pass
