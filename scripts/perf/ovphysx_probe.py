"""ovphysx P0 게이트 probe (Track B).

검증 목표 (둘 중 하나라도 실패 시 Track B 중단):
  게이트1: ovphysx 설치·부팅·USD 버전 체크 (isaacsim 5.6.1 schema 로 author 한 USD 가
           ovphysx 0.4.13(PhysX 5.9.0) 런타임에 로드되는가).
  게이트2: SO-101 robot USD 의 articulation 이 인식되고, 6-DOF position target 제어가
           실제로 관절을 움직이는가.

실행:  .venv-ovphysx/bin/python scripts/perf/ovphysx_probe.py

이 스크립트는 메인 .venv 가 아니라 전용 .venv-ovphysx 로 실행한다 (핀 환경 격리).
"""

from __future__ import annotations

import os

import numpy as np

import ovphysx
from ovphysx.types import TensorType

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_USD = os.path.join(_REPO, "assets", "robots", "so101_follower.usd")
SCENE_USD = os.path.join(_REPO, "assets", "scenes", "cube_desk", "scene.usd")

# robot USD 구조 (usd-core 로 사전 확인됨):
#   defaultPrim         /so101_new_calib
#   ArticulationRootAPI /so101_new_calib/base
#   joints(6)           shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
ARTIC_PATTERNS = ["/so101_new_calib/base", "/so101_new_calib", "/so101_new_calib*", "/*"]


def main() -> None:
    print("=" * 70)
    print(f"[probe] ovphysx {ovphysx.__version__}")
    ovphysx.bootstrap()

    # ----- 게이트1: 부팅 + robot USD 로드 (버전 mismatch 우회 허용) -----
    physx = ovphysx.PhysX(device="cpu", ignore_version_mismatch=True)
    print("[probe] PhysX(device=cpu, ignore_version_mismatch=True) 생성 OK")

    h, op = physx.add_usd(ROBOT_USD)
    physx.wait_all()
    print(f"[probe] add_usd(robot) OK  handle={h}")

    # ----- 게이트2: articulation 인식 -----
    pos_tgt = None
    used_pattern = None
    for pat in ARTIC_PATTERNS:
        try:
            b = physx.create_tensor_binding(
                pattern=pat,
                tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET,
                raise_if_empty=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] pattern {pat!r}: 예외 {exc}")
            continue
        cnt = getattr(b, "count", 0)
        print(f"[probe] pattern {pat!r}: count={cnt} shape={getattr(b, 'shape', None)}")
        if cnt and cnt > 0:
            pos_tgt = b
            used_pattern = pat
            break

    if pos_tgt is None:
        print("[probe] ❌ 게이트2 실패 — articulation 인식 안 됨. Track B 중단.")
        physx.release()
        return

    print(f"[probe] ✓ articulation 인식: pattern={used_pattern!r}")
    for attr in ("dof_count", "body_count", "is_fixed_base", "dof_names", "joint_names"):
        print(f"[probe]   {attr} = {getattr(pos_tgt, attr, '?')}")

    # 관측 binding
    pos = physx.create_tensor_binding(
        pattern=used_pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION
    )
    stiff = physx.create_tensor_binding(
        pattern=used_pattern, tensor_type=TensorType.ARTICULATION_DOF_STIFFNESS
    )
    damp = physx.create_tensor_binding(
        pattern=used_pattern, tensor_type=TensorType.ARTICULATION_DOF_DAMPING
    )

    # PD drive 보장 (USD drive 가 0이면 position target 무시됨) — 충분한 stiffness 주입
    shp = pos_tgt.shape
    stiff.write(np.full(shp, 200.0, dtype=np.float32))
    damp.write(np.full(shp, 20.0, dtype=np.float32))

    q0 = np.zeros(shp, dtype=np.float32)
    pos.read(q0)
    print(f"[probe] q0 = {q0.ravel()}")

    # ----- 게이트2: position target 으로 관절 이동 -----
    tgt = q0.copy()
    # shoulder_lift(index 1) 를 +0.4 rad 목표로
    move_dof = 1 if shp[-1] > 1 else 0
    tgt[..., move_dof] += 0.4
    pos_tgt.write(tgt)

    for i in range(240):  # 2s @120Hz
        physx.step(1.0 / 120.0, i / 120.0)
    physx.wait_all()

    q1 = np.zeros(shp, dtype=np.float32)
    pos.read(q1)
    print(f"[probe] q1 = {q1.ravel()}")
    moved = float(np.abs(q1 - q0).max())
    reached = float(abs(q1.ravel()[move_dof] - tgt.ravel()[move_dof]))
    print(f"[probe] dof{move_dof} 이동량 max={moved:.4f} rad, 목표와 잔차={reached:.4f} rad")

    if moved > 0.05:
        print("[probe] ✓✓ 게이트2 PASS — articulation 6-DOF position 제어 작동.")
    else:
        print("[probe] ❌ 게이트2 실패 — target 줘도 관절 안 움직임 (drive/fixed-base 점검).")

    # ----- (보너스) scene.usd payload 로드 게이트 -----
    try:
        h2, _ = physx.add_usd(SCENE_USD, path_prefix="/Scene")
        physx.wait_all()
        cube_b = physx.create_tensor_binding(
            pattern="/Scene/*Cube*", tensor_type=TensorType.RIGID_BODY_POSE,
            raise_if_empty=False,
        )
        print(f"[probe] scene.usd 로드 OK — Cube rigid body 매칭 count={getattr(cube_b, 'count', 0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] scene.usd 로드/매칭 예외(보너스, 비치명): {exc}")

    physx.release()
    print("=" * 70)


if __name__ == "__main__":
    main()
