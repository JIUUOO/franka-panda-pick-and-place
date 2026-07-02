"""One-cube Franka Panda tabletop pick-and-place task."""

import gymnasium as gym


_TASKS = {
    "Isaac-PickPlace-Cube-Franka-Task1-IK-Rel-v0": "FrankaCubePickPlaceTask1EnvCfg",
    "Isaac-PickPlace-Cube-Franka-IK-Rel-v0": "FrankaCubePickPlaceEnvCfg",
}

for task_id, cfg_name in _TASKS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pick_place_ik_rel_env_cfg:{cfg_name}",
        },
        disable_env_checker=True,
    )
