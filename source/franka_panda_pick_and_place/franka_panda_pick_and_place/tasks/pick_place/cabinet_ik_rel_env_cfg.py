from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.devices import DevicesCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    CabinetSceneCfg,
    ObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.ik_rel_env_cfg import (
    FrankaCabinetEnvCfg,
)

from franka_panda_pick_and_place.devices import FrankaPickPlaceGamepadCfg
from franka_panda_pick_and_place.tasks.pick_place import mdp as project_mdp

CAMERA_RESOLUTION = (160, 160)
WRIST_CAMERA_LOCAL_POS = (0.15, 0.0, -0.15)
WRIST_CAMERA_LOCAL_ROT = (0.000780, -0.627572, 0.001738, -0.778556)


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
    """Cabinet scene with front and wrist RGB cameras."""

    camera_front = _fixed_camera_cfg(
        "camera_front",
        pos=(1.65, 0.0, 0.75),
        rot=(0.43046, -0.56099, -0.56099, 0.43046),
        convention="ros",
    )
    camera_wrist = _wrist_camera_cfg()


@configclass
class CabinetTask3ObservationsCfg(ObservationsCfg):
    """Cabinet observations plus front and wrist RGB camera streams."""

    @configclass
    class RGBCameraCfg(ObsGroup):
        camera_front = ObsTerm(
            func=cabinet_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "rgb", "normalize": False},
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

        self.scene.num_envs = 1
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.observations.policy.enable_corruption = False
        self.image_obs_list = ["camera_front", "camera_wrist"]

        self.terminations.success = DoneTerm(
            func=project_mdp.cabinet_drawer_opened,
            params={
                "threshold": 0.35,
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
