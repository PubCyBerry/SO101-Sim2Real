"""Isaac Sim 6 / Isaac Lab 3 전용 parity 실행 경로."""

import gymnasium as gym


gym.register(
    id="SimToReal-SO101-PickCube-Isaac6Parity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "sim_to_real.isaac6.pick_cube_parity_env_cfg:PickCubeIsaac6ParityEnvCfg"
        ),
    },
)
