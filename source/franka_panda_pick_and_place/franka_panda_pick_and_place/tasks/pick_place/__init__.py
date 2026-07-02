"""One-cube Franka Panda tabletop pick-and-place task."""

import gymnasium as gym


gym.register(
    id="Isaac-PickPlace-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_place_ik_rel_env_cfg:FrankaCubePickPlaceEnvCfg",
    },
    disable_env_checker=True,
)
