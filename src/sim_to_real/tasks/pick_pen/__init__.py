"""SO-101 pen Pick-and-Place task registration."""

import gymnasium as gym

gym.register(
    id="SimToReal-SO101-PickPen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_pen_env_cfg:PickPenEnvCfg",
    },
)

gym.register(
    id="SimToReal-SO101-PickPen-Direct-v0",
    entry_point=f"{__name__}.direct.pick_pen_env:PickPenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.pick_pen_env:PickPenEnvCfg",
    },
)
