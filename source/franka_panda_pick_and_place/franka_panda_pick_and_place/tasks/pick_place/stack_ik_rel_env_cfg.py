from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_env_cfg import (
    FrankaCubeStackRedGreenEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from franka_panda_pick_and_place.devices import FrankaPickPlaceGamepadCfg

from .pick_place_ik_rel_env_cfg import (
    FRANKA_GRIPPER_DOWN_JOINT_POS,
    OBLIQUE_CAMERA_POS,
    OBLIQUE_CAMERA_RESOLUTION,
    OBLIQUE_CAMERA_ROT,
)


STACK_CUBE_SIZE = 0.0468
STACK_CUBE_REST_Z = STACK_CUBE_SIZE / 2.0
FRANKA_GRIPPER_DOWN_DEFAULT_POSE = [
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint1"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint2"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint3"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint4"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint5"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint6"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_joint7"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_finger_joint.*"],
    FRANKA_GRIPPER_DOWN_JOINT_POS["panda_finger_joint.*"],
]


def _stack_cube_cfg(prim_path: str, pos: tuple[float, float, float], color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(STACK_CUBE_SIZE, STACK_CUBE_SIZE, STACK_CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class StackTableSceneCfg(ObjectTableSceneCfg):
    """Stack scene plus the project-standard oblique RGB camera."""

    oblique_cam = CameraCfg(
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


@configclass
class StackObservationsCfg(ObservationsCfg):
    """Stack observations plus a fixed oblique RGB camera stream."""

    @configclass
    class RGBCameraCfg(ObsGroup):
        """RGB image observations for visuomotor imitation learning."""

        oblique_cam = ObsTerm(
            func=stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("oblique_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb_camera: RGBCameraCfg = RGBCameraCfg()


@configclass
class FrankaCubeStackTask2EnvCfg(FrankaCubeStackRedGreenEnvCfg):
    """Task 2 preset: single-Franka two-cube tabletop stacking."""

    scene: StackTableSceneCfg = StackTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: StackObservationsCfg = StackObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.cube_1 = _stack_cube_cfg(
            "{ENV_REGEX_NS}/Cube_1",
            (0.4, 0.0, STACK_CUBE_REST_Z),
            (0.05, 0.20, 0.95),
        )
        self.scene.cube_2 = _stack_cube_cfg(
            "{ENV_REGEX_NS}/Cube_2",
            (0.55, 0.05, STACK_CUBE_REST_Z),
            (0.95, 0.05, 0.05),
        )
        self.scene.cube_3 = _stack_cube_cfg(
            "{ENV_REGEX_NS}/Cube_3",
            (0.60, -0.1, STACK_CUBE_REST_Z),
            (0.05, 0.75, 0.05),
        )
        self.events.randomize_cube_positions.params["pose_range"]["z"] = (STACK_CUBE_REST_Z, STACK_CUBE_REST_Z)
        self.events.init_franka_arm_pose.params["default_pose"] = FRANKA_GRIPPER_DOWN_DEFAULT_POSE
        self.events.randomize_franka_joint_state.params["std"] = 0.0

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
