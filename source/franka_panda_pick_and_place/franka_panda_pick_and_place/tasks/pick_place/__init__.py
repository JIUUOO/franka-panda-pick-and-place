"""Franka Panda tabletop manipulation task presets."""

import gymnasium as gym


_TASKS = {
    "Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0": "FrankaCubePickPlaceTask1EnvCfg",
    "Isaac-PickPlace-Cube-Franka-IK-Rel-v0": "FrankaCubePickPlaceEnvCfg",
    "Isaac-Stack-Cube-Franka-Task2-IK-Rel-v0": "FrankaCubeStackTask2EnvCfg",
    "Isaac-Open-Drawer-Franka-Task3-IK-Rel-v0": "FrankaOpenDrawerTask3EnvCfg",
}

_CFG_MODULES = {
    "FrankaCubePickPlaceTask1EnvCfg": "pick_place_ik_rel_env_cfg",
    "FrankaCubePickPlaceEnvCfg": "pick_place_ik_rel_env_cfg",
    "FrankaCubeStackTask2EnvCfg": "stack_ik_rel_env_cfg",
    "FrankaOpenDrawerTask3EnvCfg": "cabinet_ik_rel_env_cfg",
}

for task_id, cfg_name in _TASKS.items():
    entry_point = "franka_panda_pick_and_place.envs:WristCameraSyncedManagerBasedRLEnv"
    if task_id not in {
        "Isaac-Stack-Cube-Franka-Task2-IK-Rel-v0",
        "Isaac-Open-Drawer-Franka-Task3-IK-Rel-v0",
    }:
        entry_point = "isaaclab.envs:ManagerBasedRLEnv"
    gym.register(
        id=task_id,
        entry_point=entry_point,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_CFG_MODULES[cfg_name]}:{cfg_name}",
        },
        disable_env_checker=True,
    )
