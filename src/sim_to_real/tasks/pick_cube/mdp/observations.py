"""Cube Pick-and-Place 전용 관측 — contact-sensor 기반 grasp 신호.

leisaac Workshop ``mdp/terms.py:any_vial_grasped`` 를 우리 cube_desk 씬에 이식했다.
차이 두 가지:

1. **양 손가락 envelope**: Workshop 은 jaw 센서 1개만 봤지만, 우리 씬은 jaw(가동)+
   gripper(고정) 두 ContactSensor 를 이미 authored(pick_cube_env_cfg 의 ``contact_jaw``·
   ``contact_gripper``). 두 손가락이 **같은 큐브**에 동시 접촉해야 grasp 로 본다 —
   책상에 놓인 큐브를 손가락 하나가 스쳐도 오탐하지 않는 실제 envelop 신호.
2. **상태를 env 에 저장**: Workshop 은 함수 속성(``any_vial_grasped._is_holding``)에
   상태를 뒀는데, 이는 한 프로세스에 env 인스턴스가 2개면 충돌한다. 여기선 ``env`` 에
   버퍼를 달아(``env._cube_is_holding`` 등) 인스턴스별로 격리한다. (Workshop 의
   ``vial_placed_on_rack_termination`` 자신도 ``env._rack_*`` 패턴을 쓴다.)

hysteresis 규칙은 Workshop 과 동일: 접촉+들림 이면 grasp 개시, 접촉 유지되는 한 유지,
접촉 끊기면 해제. warmup 동안은 항상 False.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # 런타임 isaaclab 의존 없이 import 가능(호스트 self-check 용)
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor
    from isaaclab.assets import RigidObject
    from so101_contract.cube_specs import CubeSpec


def _per_filter_contact(sensor, force_threshold: float) -> torch.Tensor:
    """ContactSensor 의 filter(=큐브)별 접촉력 크기 → (num_envs, num_filters) bool.

    force_matrix_w: (num_envs, num_bodies, num_filters, 3). body 축 합산으로 센서 prim
    전체의 큐브별 접촉을 얻는다(우리 센서는 body 1개라 합산은 no-op 이지만 일반형 유지).
    """
    forces = sensor.data.force_matrix_w  # (E, B, F, 3)
    norm = torch.linalg.vector_norm(forces, dim=-1)  # (E, B, F)
    per_filter = norm.sum(dim=1)  # (E, F)
    return per_filter > force_threshold


#: 꼭짓점으로 선 큐브를 "들림"으로 오탐하지 않기 위한 여유(m).
CORNER_TILT_MARGIN_M: float = 0.0054

#: DR 사다리 상한(m) — `env.cube_size_m` 이 없을 때 쓰는 보수 기준.
_MAX_DR_CUBE_SIZE_M: float = 0.040


def min_lift_for_cube(size_m: float) -> float:
    """큐브 한 변 → "들림" 임계(m, 큐브 **중심** 기준).

    ★큐브가 **꼭짓점으로 서면** 중심이 상판 위 ``s·√3/2`` 다(body diagonal ``s√3`` 의 절반).
    손가락이 조이다 큐브를 세워 놓은 것이 "들림" 으로 새면 파지 오탐이 된다. 그래서 임계는
    반드시 그보다 위여야 한다.

    ==========  ==============  ==========
    s (m)       ``s·√3/2`` (m)  임계 (m)
    ==========  ==============  ==========
    0.025       0.02165         0.0271
    0.030       0.02598         0.0314
    0.035       0.03031         0.0357
    0.040       0.03464         0.0400
    ==========  ==============  ==========

    ⚠ ``s/2·√3/2``(= ``s·√3/4``) 로 잘못 쓰면 값이 **절반**이 돼 게이트가 무력해진다.
    이 함수가 단일 소스다 — 수식을 다른 파일에 복제하지 말 것.
    """
    return float(size_m) * (3.0 ** 0.5 / 2.0) + CORNER_TILT_MARGIN_M


def any_cube_grasped(
    env: "ManagerBasedRLEnv",
    jaw_sensor_cfg: "SceneEntityCfg",
    gripper_sensor_cfg: "SceneEntityCfg",
    cubes: list[str],
    desk_top_z: float = 0.705,
    min_lift: float | None = min_lift_for_cube(_MAX_DR_CUBE_SIZE_M),
    warmup_steps: int = 15,
    force_threshold: float = 0.5,
    require_both_fingers: bool = True,
    hold_steps: int = 3,
) -> torch.Tensor:
    """큐브 중 하나라도 그리퍼에 파지 중이면 True (contact-sensor + dwell + hysteresis).

    grasp 개시 조건: (양)손가락이 같은 큐브에 접촉 **AND** 그 큐브가 책상 위로 들림
    (``desk_top_z + min_lift`` 초과) 인 상태가 ``hold_steps`` 연속 프레임 유지 **AND** 아직
    holding 아님. 개시 후엔 접촉만 유지되면 (높이 무관) 계속 holding, 접촉 끊기면 해제.

    Args:
        jaw_sensor_cfg: 가동 손가락(jaw) ContactSensor cfg (filter=큐브들).
        gripper_sensor_cfg: 고정 손가락(gripper) ContactSensor cfg (filter=큐브들, 동일 순서).
        cubes: 큐브 asset 이름 리스트(센서 filter_prim_paths_expr 순서와 일치해야 함).
        desk_top_z: 책상 상판 world z (= common ``_geometry.DESK_TOP_Z``. 자족 규칙상
            이 모듈은 리터럴 기본값을 두고 cfg 가 단일 소스에서 주입한다).
        min_lift: 상판 위로 이만큼(m) 큐브 중심이 올라오면 "들림". 초기 grasp 에만 요구.
            **None** 이면 들림을 보지 않고 **접촉만으로** 판정한다(mimic subtask0 종료를
            파지 개시에 맞추는 설정).
            기본값 = :func:`min_lift_for_cube` 를 DR 상한(40 mm)에 적용한 보수값. 런타임에
            ``env.cube_size_m`` 가 있으면 **per-env 로 파생**해 덮어쓴다(작은 큐브에서 실제
            들림을 놓치지 않는다).
        warmup_steps: 리셋 직후 이 스텝 동안은 grasp 무시(초기 접촉 노이즈 차단).
        force_threshold: 손가락별 접촉력(N) 임계.
        require_both_fingers: True 면 jaw·gripper 둘 다 같은 큐브 접촉 요구(envelope).
            False 면 한쪽만으로도 접촉 인정.
        hold_steps: 접촉+들림이 이만큼 **연속** 프레임 유지돼야 개시(0 = 즉시 = 옛 동작).
            물리 튐 1프레임으로 latch 되는 오탐을 막는다. 3 = 100 ms @30 Hz.

    Returns:
        (num_envs, 1) float 텐서. 관측 그룹용.

    Note:
        dwell 은 카운터가 아니라 **개시 시점(episode step) 래치**로 구현한다 — 같은 step 에서
        관측이 두 번 계산돼도 결과가 동일하다(카운터면 두 번 증가해 dwell 이 반토막 난다).
    """
    num_envs = env.num_envs
    device = env.device
    _NEVER = -1

    if not hasattr(env, "_cube_is_holding"):
        env._cube_is_holding = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_cube_grasp_since"):
        env._cube_grasp_since = torch.full((num_envs,), _NEVER, dtype=torch.long, device=device)

    # 리셋된 env 는 holding·dwell 상태 초기화
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._cube_is_holding[just_reset] = False
        env._cube_grasp_since[just_reset] = _NEVER

    in_warmup = env.episode_length_buf < warmup_steps

    jaw: "ContactSensor" = env.scene[jaw_sensor_cfg.name]
    jaw_contact = _per_filter_contact(jaw, force_threshold)  # (E, F)
    if require_both_fingers:
        grip: "ContactSensor" = env.scene[gripper_sensor_cfg.name]
        grip_contact = _per_filter_contact(grip, force_threshold)  # (E, F)
        finger_contact = jaw_contact & grip_contact  # 같은 큐브 양손가락 접촉
    else:
        finger_contact = jaw_contact

    any_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    candidate = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 들림 임계: 크기 DR 이 켜져 있으면 **per-env 로 파생**한다(`min_lift_for_cube` 단일 소스).
    if min_lift is None:
        lift_thresholds = torch.full((num_envs,), -1.0, device=device)  # 접촉만으로 판정
    else:
        size = getattr(env, "cube_size_m", None)
        if size is not None:
            s = torch.as_tensor(size, dtype=torch.float32, device=device)
            if s.dim() > 1:  # (E, C) — 큐브별 크기면 가장 큰 것으로 보수 판정
                s = s.max(dim=-1).values
            min_lift = s * (3.0 ** 0.5 / 2.0) + CORNER_TILT_MARGIN_M
        lift_thresholds = desk_top_z + min_lift

    for cube_idx, cube_name in enumerate(cubes):
        cube: "RigidObject" = env.scene[cube_name]
        cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        contact_i = finger_contact[:, cube_idx]
        lifted_i = cube_z > lift_thresholds

        any_contact = any_contact | contact_i
        candidate = candidate | (contact_i & lifted_i)

    # dwell: 후보(접촉+들림)가 처음 성립한 episode step 을 래치, 끊기면 지운다.
    ep = env.episode_length_buf
    never = torch.full_like(env._cube_grasp_since, _NEVER)
    started = torch.where(env._cube_grasp_since < 0, ep, env._cube_grasp_since)
    env._cube_grasp_since = torch.where(candidate, started, never)
    dwell_ok = (env._cube_grasp_since >= 0) & ((ep - env._cube_grasp_since) >= hold_steps)

    new_grasp = dwell_ok & (~env._cube_is_holding)

    # hysteresis: 개시(new_grasp) | (유지: 이전 holding & 접촉 유지)
    env._cube_is_holding = (env._cube_is_holding & any_contact) | new_grasp

    is_grasped = env._cube_is_holding & (~in_warmup)
    return is_grasped.float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# self-check — fake env/sensor 로 hysteresis·warmup·envelope·dwell 검증 (torch 만 필요)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import types

    class _FakeData:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _FakeSensor:
        def __init__(self, force_matrix_w):
            self.data = _FakeData(force_matrix_w=force_matrix_w)

    class _FakeCube:
        def __init__(self, z):  # z = world-frame 큐브 중심 z (E,)
            self.data = _FakeData(root_pos_w=torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=-1))

    class _FakeScene(dict):
        def __init__(self, env_origins, **kw):
            super().__init__(**kw)
            self.env_origins = env_origins

    def make_env(step, jaw_f, grip_f, cube_z):
        E = jaw_f.shape[0]
        env = types.SimpleNamespace()
        env.num_envs = E
        env.device = torch.device("cpu")
        env.episode_length_buf = torch.full((E,), step, dtype=torch.long)
        env.scene = _FakeScene(
            env_origins=torch.zeros(E, 3),
            jaw=_FakeSensor(jaw_f),
            grip=_FakeSensor(grip_f),
            Cube1=_FakeCube(cube_z),
        )
        return env

    jaw_cfg = types.SimpleNamespace(name="jaw")
    grip_cfg = types.SimpleNamespace(name="grip")

    def contact(mag):  # (1 env, 1 body, 1 filter, 3)
        return torch.tensor([[[[mag, 0.0, 0.0]]]])

    DESK = 0.705
    sqrt3_2 = 3.0 ** 0.5 / 2.0  # 모서리로 선 큐브 중심높이 계수

    # 25mm 큐브: 꼭짓점 서기(최악) = 25mm·√3/2 = 0.02165
    CUBE_25_S = 0.025
    CORNER_LIFT_25 = CUBE_25_S * sqrt3_2  # 0.02165
    MIN_LIFT_25 = CORNER_LIFT_25 + 0.0054  # margin ≈ 0.0271
    REST_Z_25 = torch.tensor([DESK + 0.015])  # 책상 위 정지(들림 아님)
    UP_Z_25 = torch.tensor([DESK + MIN_LIFT_25 + 0.01])  # min_lift 보다 높음
    CORNER_Z_25 = torch.tensor([DESK + CORNER_LIFT_25])

    # 40mm 큐브: 꼭짓점 서기 = 40mm·√3/2 = 0.03464
    CUBE_40_S = 0.040
    CORNER_LIFT_40 = CUBE_40_S * sqrt3_2  # 0.03464
    MIN_LIFT_40 = CORNER_LIFT_40 + 0.0054  # margin ≈ 0.0400
    REST_Z_40 = torch.tensor([DESK + 0.021])
    UP_Z_40 = torch.tensor([DESK + MIN_LIFT_40 + 0.01])  # min_lift 보다 높음
    CORNER_Z_40 = torch.tensor([DESK + CORNER_LIFT_40])

    # 1~6 은 dwell 없는(hold_steps=0) 원래 hysteresis 계약 — 회귀 방지용으로 그대로 유지.

    # 1) warmup 중엔 접촉+들림이어도 False
    env = make_env(2, contact(5.0), contact(5.0), UP_Z_40)
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=0)
    assert r.item() == 0.0, f"warmup 이어야 0, got {r.item()}"

    # 2) warmup 지나고 양손가락 접촉+들림 → grasp 개시
    env.episode_length_buf = torch.tensor([20])
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=0)
    assert r.item() == 1.0, f"grasp 개시 1 이어야, got {r.item()}"

    # 3) 개시 후 큐브가 다시 내려가도(들림 아님) 접촉 유지되면 holding 유지
    env.scene["Cube1"] = _FakeCube(REST_Z_40)
    env.episode_length_buf = torch.tensor([21])
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=0)
    assert r.item() == 1.0, f"접촉 유지 시 holding 유지여야, got {r.item()}"

    # 4) 접촉 끊기면 해제
    env.scene["jaw"] = _FakeSensor(contact(0.0))
    env.scene["grip"] = _FakeSensor(contact(0.0))
    env.episode_length_buf = torch.tensor([22])
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=0)
    assert r.item() == 0.0, f"접촉 끊기면 해제여야, got {r.item()}"

    # 5) envelope: 한 손가락만 접촉 → grasp 아님(require_both_fingers=True)
    env2 = make_env(20, contact(5.0), contact(0.0), UP_Z_40)
    r = any_cube_grasped(env2, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=0)
    assert r.item() == 0.0, f"한 손가락만이면 0 이어야, got {r.item()}"

    # 6) 리셋(step<=1)이면 holding 클리어
    env2.episode_length_buf = torch.tensor([0])
    env2.scene["jaw"] = _FakeSensor(contact(5.0))
    env2.scene["grip"] = _FakeSensor(contact(5.0))
    r = any_cube_grasped(env2, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=0)
    assert r.item() == 0.0, f"리셋 직후(warmup) 0 이어야, got {r.item()}"

    # ---- dwell(hold_steps) ----

    # 7) hold_steps=3: 접촉+들림 3 프레임 미달이면 0, 도달하면 1
    env3 = make_env(20, contact(5.0), contact(5.0), UP_Z_40)
    for step, want in ((20, 0.0), (21, 0.0), (22, 0.0), (23, 1.0)):
        env3.episode_length_buf = torch.tensor([step])
        r = any_cube_grasped(env3, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=MIN_LIFT_40)
        assert r.item() == want, f"dwell step {step}: {want} 이어야, got {r.item()}"

    # 8) dwell 중 접촉 끊기면 래치 리셋 → 다시 3 프레임 필요
    env4 = make_env(20, contact(5.0), contact(5.0), UP_Z_40)
    for step in (20, 21):
        env4.episode_length_buf = torch.tensor([step])
        any_cube_grasped(env4, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=MIN_LIFT_40)
    env4.scene["jaw"] = _FakeSensor(contact(0.0))          # 2 프레임 만에 놓침
    env4.episode_length_buf = torch.tensor([22])
    any_cube_grasped(env4, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=MIN_LIFT_40)
    env4.scene["jaw"] = _FakeSensor(contact(5.0))          # 다시 접촉
    for step, want in ((23, 0.0), (24, 0.0), (25, 0.0), (26, 1.0)):
        env4.episode_length_buf = torch.tensor([step])
        r = any_cube_grasped(env4, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=MIN_LIFT_40)
        assert r.item() == want, f"재개시 step {step}: {want} 이어야, got {r.item()}"

    # 9) 같은 step 에서 2회 계산해도 dwell 이 앞당겨지지 않음(카운터 아닌 step 래치)
    env5 = make_env(20, contact(5.0), contact(5.0), UP_Z_40)
    for step in (20, 21, 22):
        env5.episode_length_buf = torch.tensor([step])
        for _ in range(2):  # 관측 중복 계산 시뮬
            r = any_cube_grasped(env5, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=MIN_LIFT_40)
        assert r.item() == 0.0, f"중복 계산이 dwell 을 앞당김: step {step} → {r.item()}"

    # ---- 큐브 크기 파생 임계 (25mm vs 40mm) ----

    # 10) 25mm: corner tilt 오탐 방지 (min_lift=0.035)
    env6 = make_env(20, contact(5.0), contact(5.0), CORNER_Z_25)
    for step in (20, 21, 22, 23, 24):
        env6.episode_length_buf = torch.tensor([step])
        r = any_cube_grasped(env6, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, min_lift=MIN_LIFT_25)
    assert r.item() == 0.0, f"25mm corner tilt 는 들림 아님(0) 이어야, got {r.item()}"

    # 11) 40mm: corner tilt 오탐 방지 (min_lift=0.04)
    env7 = make_env(20, contact(5.0), contact(5.0), CORNER_Z_40)
    for step in (20, 21, 22, 23, 24):
        env7.episode_length_buf = torch.tensor([step])
        r = any_cube_grasped(env7, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, min_lift=MIN_LIFT_40)
    assert r.item() == 0.0, f"40mm corner tilt 는 들림 아님(0) 이어야, got {r.item()}"

    # 12) min_lift=None: 접촉만으로 판정 (높이 무관, hold_steps 적용)
    env8 = make_env(20, contact(5.0), contact(5.0), REST_Z_40)
    for step in (20, 21, 22):
        env8.episode_length_buf = torch.tensor([step])
        r = any_cube_grasped(env8, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=None)
    assert r.item() == 0.0, f"dwell 3 미달, got {r.item()}"
    env8.episode_length_buf = torch.tensor([23])
    r = any_cube_grasped(env8, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15, hold_steps=3, min_lift=None)
    assert r.item() == 1.0, f"min_lift=None, dwell 도달해 1 이어야, got {r.item()}"

    print("any_cube_grasped self-check PASSED (12 cases)")
