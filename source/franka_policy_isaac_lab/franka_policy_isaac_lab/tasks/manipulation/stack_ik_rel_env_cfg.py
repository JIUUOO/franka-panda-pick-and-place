from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.devices import DevicesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_env_cfg import (
    FrankaCubeStackEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObjectTableSceneCfg, ObservationsCfg

from franka_policy_isaac_lab.devices import FrankaPickPlaceGamepadCfg
from franka_policy_isaac_lab.tasks.manipulation import mdp as project_mdp

STACK_CUBE_SIZE = 0.0468
STACK_CUBE_REST_Z = STACK_CUBE_SIZE / 2.0
CAMERA_RESOLUTION = (160, 160)
WRIST_CAMERA_LOCAL_POS = (0.15, 0.0, -0.15)
WRIST_CAMERA_LOCAL_ROT = (0.000780, -0.627572, 0.001738, -0.778556)
TASK2_FRANKA_INITIAL_JOINT_POS = {
    "panda_joint1": -0.0100,
    "panda_joint2": -0.1813,
    "panda_joint3": 0.0064,
    "panda_joint4": -2.4974,
    "panda_joint5": 0.0016,
    "panda_joint6": 2.3162,
    "panda_joint7": 0.7805,
    "panda_finger_joint.*": 0.04,
}


def _fixed_camera_cfg(
    name: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    convention: str,
    clipping_range: tuple[float, float] = (0.1, 3.0),
) -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        update_period=0.0,
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention),
    )


def _wrist_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_wrist",
        update_period=0.0,
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=35.0,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


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
    """Stack scene with front and wrist RGB cameras."""

    camera_front = _fixed_camera_cfg(
        "camera_front",
        pos=(1.45, 0.0, 0.40),
        rot=(0.43046, -0.56099, -0.56099, 0.43046),
        convention="ros",
    )
    camera_wrist = _wrist_camera_cfg()


@configclass
class StackObservationsCfg(ObservationsCfg):
    """Stack observations plus front and wrist RGB camera streams."""

    @configclass
    class RGBCameraCfg(ObsGroup):
        """RGB image observations for visuomotor imitation learning."""

        camera_front = ObsTerm(
            func=stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "rgb", "normalize": False},
        )
        camera_wrist = ObsTerm(
            func=stack_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb_camera: RGBCameraCfg = RGBCameraCfg()


@configclass
class FrankaCubeStackTask2EnvCfg(FrankaCubeStackEnvCfg):
    """Task 2 preset: single-Franka three-cube tabletop stacking with IK-Rel control."""

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
        self.scene.robot.init_state.joint_pos.update(TASK2_FRANKA_INITIAL_JOINT_POS)

        self.events.randomize_cube_positions.params["pose_range"]["z"] = (STACK_CUBE_REST_Z, STACK_CUBE_REST_Z)
        self.events.randomize_cube_positions.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.init_franka_arm_pose = None
        self.events.randomize_franka_joint_state.params["std"] = 0.0
        self.events.sync_wrist_camera = EventTerm(
            func=project_mdp.sync_camera_to_robot_body,
            mode="reset",
            params={
                "camera_cfg": SceneEntityCfg("camera_wrist"),
                "robot_cfg": SceneEntityCfg("robot"),
                "body_name": "panda_hand",
                "local_pos": WRIST_CAMERA_LOCAL_POS,
                "local_rot": WRIST_CAMERA_LOCAL_ROT,
            },
        )

        self.scene.num_envs = 1
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.observations.policy.enable_corruption = False
        self.image_obs_list = ["camera_front", "camera_wrist"]
        self.teleop_devices = DevicesCfg(
            devices={
                "gamepad": FrankaPickPlaceGamepadCfg(
                    pos_sensitivity=0.1,
                    rot_sensitivity=0.1,
                    sim_device=self.sim.device,
                ),
            }
        )
