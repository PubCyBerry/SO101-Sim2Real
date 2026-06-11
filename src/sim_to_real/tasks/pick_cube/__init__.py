"""SO-101 cube Pick-and-Place task registration."""

import gymnasium as gym

gym.register(
    id="SimToReal-SO101-PickCube-v0",
    entry_point=f"{__name__}.pick_cube_env:PickCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cube_env_cfg:PickCubeEnvCfg",
    },
)

# scripted state machine 데모 전용 — in-sim DifferentialIK(arm) + binary gripper.
# (학습/teleop 은 위 SimToReal-SO101-PickCube-v0 의 6-dim joint-space 계약을 쓴다.)
gym.register(
    id="SimToReal-SO101-PickCube-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_cube_so101_ik_env_cfg:PickCubeSo101IkEnvCfg",
    },
)
