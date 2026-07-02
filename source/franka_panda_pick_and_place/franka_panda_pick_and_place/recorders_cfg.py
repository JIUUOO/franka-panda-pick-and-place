from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class PreStepRGBCameraObservationsRecorder(RecorderTerm):
    """Recorder term that stores RGB camera observation groups before each action step."""

    def record_pre_step(self):
        rgb_camera_obs = self._env.obs_buf.get("rgb_camera")
        if rgb_camera_obs is None:
            return None, None
        return "obs/rgb_camera", rgb_camera_obs


@configclass
class PreStepRGBCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for RGB camera observation recording."""

    class_type: type[RecorderTerm] = PreStepRGBCameraObservationsRecorder


@configclass
class ActionStateRGBRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action-state recorder with additional RGB camera observations."""

    record_pre_step_rgb_camera_observations = PreStepRGBCameraObservationsRecorderCfg()
