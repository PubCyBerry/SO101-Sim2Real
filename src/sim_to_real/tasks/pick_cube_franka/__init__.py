"""Franka Panda cube Pick-and-Place task registration (cube_desk scene)."""

import gymnasium as gym

gym.register(
    id="SimToReal-Franka-PickCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cube_franka_env_cfg:PickCubeFrankaEnvCfg",
    },
)
