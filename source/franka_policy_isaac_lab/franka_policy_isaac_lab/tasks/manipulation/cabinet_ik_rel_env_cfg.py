from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.devices import DevicesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    CabinetSceneCfg,
    FRAME_MARKER_SMALL_CFG,
    ObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.ik_rel_env_cfg import (
    FrankaCabinetEnvCfg,
)

from franka_policy_isaac_lab.devices import FrankaPickPlaceGamepadCfg
from franka_policy_isaac_lab.tasks.manipulation import mdp as project_mdp

CAMERA_RESOLUTION = (480, 480)
OBLIQUE_CAMERA_POS = (-0.10, 1.10, 0.75)
OBLIQUE_CAMERA_ROT = (-0.188189, 0.220363, -0.727809, 0.621546)
WRIST_CAMERA_LOCAL_POS = (0.15, 0.0, -0.15)
WRIST_CAMERA_LOCAL_ROT = (0.000780, -0.627572, 0.001738, -0.778556)
TASK3_CABINET_X_FARTHER_RANGE = 0.315
TASK3_CABINET_Y_RANGE = 0.500
TASK3_CABINET_YAW_RANGE = 0.0
TASK3_ROBOT_ARM_JOINT_NOISE = 0.03
TASK3_LIGHT_INTENSITY_RANGE = (2400.0, 3600.0)
TASK3_FRANKA_BASE_POS = (0.0, 0.0, 0.0)
TASK3_FRANKA_INITIAL_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -1.2,
    "panda_joint3": 0.0,
    "panda_joint4": -3.0,
    "panda_joint5": 0.0,
    "panda_joint6": 3.3,
    "panda_joint7": -0.785,
    "panda_finger_joint.*": 0.04,
}


def _fixed_camera_cfg(
    name: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    convention: str,
    clipping_range: tuple[float, float] = (0.1, 4.0),
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


@configclass
class CabinetTask3SceneCfg(CabinetSceneCfg):
    """Cabinet scene with oblique and wrist RGB cameras."""

    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.222, 0.0, 0.005),
                    rot=(0.5, 0.5, -0.5, -0.5),
                ),
            ),
        ],
    )
    camera_oblique = _fixed_camera_cfg(
        "camera_oblique",
        pos=OBLIQUE_CAMERA_POS,
        rot=OBLIQUE_CAMERA_ROT,
        convention="ros",
    )
    camera_wrist = _wrist_camera_cfg()


@configclass
class CabinetTask3ObservationsCfg(ObservationsCfg):
    """Cabinet observations plus oblique and wrist RGB camera streams."""

    @configclass
    class RGBCameraCfg(ObsGroup):
        camera_oblique = ObsTerm(
            func=cabinet_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_oblique"), "data_type": "rgb", "normalize": False},
        )
        camera_wrist = ObsTerm(
            func=cabinet_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    rgb_camera: RGBCameraCfg = RGBCameraCfg()


@configclass
class FrankaOpenDrawerTask3EnvCfg(FrankaCabinetEnvCfg):
    """Task 3 preset: single-Franka top-drawer opening with IK-Rel control."""

    scene: CabinetTask3SceneCfg = CabinetTask3SceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: CabinetTask3ObservationsCfg = CabinetTask3ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        project_mdp.use_analog_parallel_gripper(self)
        self.scene.num_envs = 1
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.observations.policy.enable_corruption = False
        self.image_obs_list = ["camera_oblique", "camera_wrist"]
        self.scene.robot.init_state.pos = TASK3_FRANKA_BASE_POS
        self.scene.robot.init_state.joint_pos.update(TASK3_FRANKA_INITIAL_JOINT_POS)
        self.events.reset_robot_joints.params["position_range"] = (
            -TASK3_ROBOT_ARM_JOINT_NOISE,
            TASK3_ROBOT_ARM_JOINT_NOISE,
        )
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        self.events.robot_physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.robot_physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
        self.events.robot_physics_material.params["num_buckets"] = 1
        self.events.cabinet_physics_material.params["asset_cfg"].body_names = "drawer_handle_top"
        self.events.cabinet_physics_material.params["static_friction_range"] = (1.25, 1.25)
        self.events.cabinet_physics_material.params["dynamic_friction_range"] = (1.25, 1.25)
        self.events.cabinet_physics_material.params["num_buckets"] = 1
        self.observations.policy.cabinet_joint_pos.params["asset_cfg"].joint_names = ["drawer_top_joint"]
        self.observations.policy.cabinet_joint_vel.params["asset_cfg"].joint_names = ["drawer_top_joint"]
        self.rewards.open_drawer_bonus.params["asset_cfg"].joint_names = ["drawer_top_joint"]
        self.rewards.multi_stage_open_drawer.params["asset_cfg"].joint_names = ["drawer_top_joint"]
        self.events.randomize_cabinet_pose = EventTerm(
            func=cabinet_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, TASK3_CABINET_X_FARTHER_RANGE),
                    "y": (0.0, TASK3_CABINET_Y_RANGE),
                    "yaw": (-TASK3_CABINET_YAW_RANGE, TASK3_CABINET_YAW_RANGE),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cabinet"),
            },
        )
        self.events.randomize_light = EventTerm(
            func=project_mdp.randomize_light_intensity,
            mode="reset",
            params={"intensity_range": TASK3_LIGHT_INTENSITY_RANGE},
        )

        self.terminations.success = DoneTerm(
            func=project_mdp.cabinet_drawer_opened,
            params={
                "threshold": 0.30,
                "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            },
        )
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
        self.teleop_devices = DevicesCfg(
            devices={
                "gamepad": FrankaPickPlaceGamepadCfg(
                    pos_sensitivity=0.1,
                    rot_sensitivity=0.1,
                    sim_device=self.sim.device,
                ),
            }
        )
