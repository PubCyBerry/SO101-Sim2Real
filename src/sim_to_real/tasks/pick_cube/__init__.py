"""SO-101 cube Pick-and-Place task registration."""

import gymnasium as gym

gym.register(
    id="SimToReal-SO101-PickCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cube_env_cfg:PickCubeEnvCfg",
    },
)
