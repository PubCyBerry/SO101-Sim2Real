"""SO-101 cube Pick-and-Place task registration.

leisaac Workshop 사다리(base teleop → task leaf + DR/Eval 변형) 대응:
- Teleop-v0        : base substrate(로봇+책상+조명, 성공/보상 없음) — teleop·씬 author.
- PickCube-v0      : **DR-off 기본** (고정 실측 배치, 순간 성공 종료).
- PickCube-DR-v0   : DR-on **full 모드**(큐브 좌우대칭 종모양 scatter + 그릇 arc + 물리·시각 DR).
- PickCube-DRBase-v0 : DR-on **base 모드**(큐브 스폰을 nominal 주변 좁은 사각형으로 제한, 그 외 full 동일).
- PickCube-Eval-v0 : DR-off + 디바운스 성공 종료 — 재현성 평가.
- PickCube-DR-Eval-v0 : DR-on + 디바운스 성공 종료.

Isaac Lab Mimic / SkillGen 변형(`ManagerBasedRLMimicEnv` + subtask 계약):
- PickCube-Mimic-v0    : DR-off — source 데모 주석(annotate)·재현 검증용.
- PickCube-Mimic-DR-v0 : DR-on  — 증강 데이터 생성 본 경로.
"""

import gymnasium as gym

# base teleop substrate (태스크 오브젝트/성공 없음) — stock ManagerBasedRLEnv.
gym.register(
    id="SimToReal-SO101-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "sim_to_real.tasks.so101_base_env_cfg:SO101TeleopEnvCfg",
    },
)

# pick_cube leaf 변형 4종 — 모두 PickCubeEnv(동적 gripper effort 배선).
_PICK_CUBE_VARIANTS = {
    "SimToReal-SO101-PickCube-v0": "PickCubeEnvCfg",
    "SimToReal-SO101-PickCube-DR-v0": "PickCubeDREnvCfg",
    "SimToReal-SO101-PickCube-DRBase-v0": "PickCubeDRBaseEnvCfg",
    "SimToReal-SO101-PickCube-Eval-v0": "PickCubeEvalEnvCfg",
    "SimToReal-SO101-PickCube-DR-Eval-v0": "PickCubeEvalDREnvCfg",
}
for _env_id, _cfg_cls in _PICK_CUBE_VARIANTS.items():
    gym.register(
        id=_env_id,
        entry_point=f"{__name__}.pick_cube_env:PickCubeEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.pick_cube_env_cfg:{_cfg_cls}",
        },
    )

# Mimic/SkillGen 변형 — entry_point 가 `ManagerBasedRLMimicEnv` 자손이라야 공식 드라이버
# (annotate·generate)가 받아준다.
_PICK_CUBE_MIMIC_VARIANTS = {
    "SimToReal-SO101-PickCube-Mimic-v0": "SO101PickCubeMimicEnvCfg",
    "SimToReal-SO101-PickCube-Mimic-DR-v0": "SO101PickCubeMimicDREnvCfg",
}
for _env_id, _cfg_cls in _PICK_CUBE_MIMIC_VARIANTS.items():
    gym.register(
        id=_env_id,
        entry_point=f"{__name__}.mimic_env:SO101PickCubeMimicEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mimic_env_cfg:{_cfg_cls}",
        },
    )
