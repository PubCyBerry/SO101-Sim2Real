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


def _per_filter_contact(sensor, force_threshold: float) -> torch.Tensor:
    """ContactSensor 의 filter(=큐브)별 접촉력 크기 → (num_envs, num_filters) bool.

    force_matrix_w: (num_envs, num_bodies, num_filters, 3). body 축 합산으로 센서 prim
    전체의 큐브별 접촉을 얻는다(우리 센서는 body 1개라 합산은 no-op 이지만 일반형 유지).
    """
    forces = sensor.data.force_matrix_w  # (E, B, F, 3)
    norm = torch.linalg.vector_norm(forces, dim=-1)  # (E, B, F)
    per_filter = norm.sum(dim=1)  # (E, F)
    return per_filter > force_threshold


def any_cube_grasped(
    env: "ManagerBasedRLEnv",
    jaw_sensor_cfg: "SceneEntityCfg",
    gripper_sensor_cfg: "SceneEntityCfg",
    cubes: list[str],
    desk_top_z: float = 0.705,
    min_lift: float = 0.03,
    warmup_steps: int = 15,
    force_threshold: float = 0.5,
    require_both_fingers: bool = True,
) -> torch.Tensor:
    """큐브 중 하나라도 그리퍼에 파지 중이면 True (contact-sensor + hysteresis).

    grasp 개시 조건: (양)손가락이 같은 큐브에 접촉 **AND** 그 큐브가 책상 위로 들림
    (``desk_top_z + min_lift`` 초과) **AND** 아직 holding 아님. 개시 후엔 접촉만 유지되면
    (높이 무관) 계속 holding, 접촉 끊기면 해제.

    Args:
        jaw_sensor_cfg: 가동 손가락(jaw) ContactSensor cfg (filter=큐브들).
        gripper_sensor_cfg: 고정 손가락(gripper) ContactSensor cfg (filter=큐브들, 동일 순서).
        cubes: 큐브 asset 이름 리스트(센서 filter_prim_paths_expr 순서와 일치해야 함).
        desk_top_z: 책상 상판 world z. lift 판정 기준(pick_cube=0.705; common
            ``_geometry.DESK_TOP_Z`` 0.76 은 pen 잔재라 쓰지 않음).
        min_lift: 상판 위로 이만큼(m) 큐브 중심이 올라오면 "들림". 초기 grasp 에만 요구.
        warmup_steps: 리셋 직후 이 스텝 동안은 grasp 무시(초기 접촉 노이즈 차단).
        force_threshold: 손가락별 접촉력(N) 임계.
        require_both_fingers: True 면 jaw·gripper 둘 다 같은 큐브 접촉 요구(envelope).
            False 면 한쪽만으로도 접촉 인정.

    Returns:
        (num_envs, 1) float 텐서. 관측 그룹용.
    """
    num_envs = env.num_envs
    device = env.device

    if not hasattr(env, "_cube_is_holding"):
        env._cube_is_holding = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 리셋된 env 는 holding 상태 초기화
    just_reset = env.episode_length_buf <= 1
    if just_reset.any():
        env._cube_is_holding[just_reset] = False

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
    new_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
    lift_threshold = desk_top_z + min_lift

    for cube_idx, cube_name in enumerate(cubes):
        cube: "RigidObject" = env.scene[cube_name]
        cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        contact_i = finger_contact[:, cube_idx]
        lifted_i = cube_z > lift_threshold

        any_contact = any_contact | contact_i
        new_grasp = new_grasp | (contact_i & lifted_i & (~env._cube_is_holding))

    # hysteresis: 개시(new_grasp) | (유지: 이전 holding & 접촉 유지)
    env._cube_is_holding = (env._cube_is_holding & any_contact) | new_grasp

    is_grasped = env._cube_is_holding & (~in_warmup)
    return is_grasped.float().unsqueeze(-1)


# ---------------------------------------------------------------------------
# self-check — fake env/sensor 로 hysteresis·warmup·envelope 로직 검증 (torch 만 필요)
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

    LIFT_Z = 0.705 + 0.05  # 들린 높이
    REST_Z = torch.tensor([0.705 + 0.021])  # 책상 위 정지(들림 아님)
    UP_Z = torch.tensor([LIFT_Z])

    # 1) warmup 중엔 접촉+들림이어도 False
    env = make_env(2, contact(5.0), contact(5.0), UP_Z)
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15)
    assert r.item() == 0.0, f"warmup 이어야 0, got {r.item()}"

    # 2) warmup 지나고 양손가락 접촉+들림 → grasp 개시
    env.episode_length_buf = torch.tensor([20])
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15)
    assert r.item() == 1.0, f"grasp 개시 1 이어야, got {r.item()}"

    # 3) 개시 후 큐브가 다시 내려가도(들림 아님) 접촉 유지되면 holding 유지
    env.scene["Cube1"] = _FakeCube(REST_Z)
    env.episode_length_buf = torch.tensor([21])
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15)
    assert r.item() == 1.0, f"접촉 유지 시 holding 유지여야, got {r.item()}"

    # 4) 접촉 끊기면 해제
    env.scene["jaw"] = _FakeSensor(contact(0.0))
    env.scene["grip"] = _FakeSensor(contact(0.0))
    env.episode_length_buf = torch.tensor([22])
    r = any_cube_grasped(env, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15)
    assert r.item() == 0.0, f"접촉 끊기면 해제여야, got {r.item()}"

    # 5) envelope: 한 손가락만 접촉 → grasp 아님(require_both_fingers=True)
    env2 = make_env(20, contact(5.0), contact(0.0), UP_Z)
    r = any_cube_grasped(env2, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15)
    assert r.item() == 0.0, f"한 손가락만이면 0 이어야, got {r.item()}"

    # 6) 리셋(step<=1)이면 holding 클리어
    env2.episode_length_buf = torch.tensor([0])
    env2.scene["jaw"] = _FakeSensor(contact(5.0))
    env2.scene["grip"] = _FakeSensor(contact(5.0))
    r = any_cube_grasped(env2, jaw_cfg, grip_cfg, ["Cube1"], warmup_steps=15)
    assert r.item() == 0.0, f"리셋 직후(warmup) 0 이어야, got {r.item()}"

    print("any_cube_grasped self-check PASSED (6 cases)")
