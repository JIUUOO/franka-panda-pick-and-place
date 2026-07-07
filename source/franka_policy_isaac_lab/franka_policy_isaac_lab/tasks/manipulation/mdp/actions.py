from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


class AnalogParallelGripperAction(ActionTerm):
    """One-dimensional continuous position command for a parallel gripper."""

    cfg: "AnalogParallelGripperActionCfg"

    def __init__(self, cfg: "AnalogParallelGripperActionCfg", env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        command = torch.clamp(actions[:, :1], -1.0, 1.0)
        gripper_pos = self.cfg.close_position + (command + 1.0) * 0.5 * (
            self.cfg.open_position - self.cfg.close_position
        )
        self._processed_actions[:] = gripper_pos.repeat(1, len(self._joint_ids))

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0.0


@configclass
class AnalogParallelGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AnalogParallelGripperAction
    joint_names: list[str] = ["panda_finger.*"]
    open_position: float = 0.04
    close_position: float = 0.0


def use_analog_parallel_gripper(env_cfg, asset_name: str = "robot"):
    env_cfg.actions.gripper_action = AnalogParallelGripperActionCfg(
        asset_name=asset_name,
        joint_names=["panda_finger.*"],
        open_position=0.04,
        close_position=0.0,
    )
