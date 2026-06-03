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

# SimToReal-SO101-PickPen-Direct-v0 is deferred until a pure-Isaac-Lab
# DirectRLEnv base is implemented (post T0.3).
