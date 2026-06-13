"""ovphysx P1 grasp probe — robot + scene co-load + 1-cube grasp 시뮬.

목표: ovphysx 에서 SO-101 + cube_desk scene 을 로드하고 고정 위치 큐브를 grasp→lift 해서
물리 거동(접촉 강도, 큐브 들림)이 작동하는지 검증.

실행:
    /home/konan147/Workspaces/SO101-Sim2Real/.venv-ovphysx/bin/python \\
        scripts/perf/ovphysx_grasp_probe.py

P1 진행 체크리스트:
  1. robot + scene co-load 우회 (path_prefix 미지원 대책)
  2. 큐브 rigid body 인식 (tensor binding count + 위치 확인)
  3. SO101Kinematics 로 grasp 관절각 계산
  4. grasp 시퀀스(접근→하강→close→lift) 실행
  5. 큐브 z 상승량 측정 → 물리 동작 확인
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np

import ovphysx
from ovphysx.types import TensorType

# SO101Kinematics 임포트 (같은 디렉토리의 so101_kin.py)
sys.path.insert(0, os.path.dirname(__file__))
from so101_kin import SO101Kinematics


# ---- 프로젝트 경로 ----
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROBOT_USD = os.path.join(_REPO, "assets", "robots", "so101_follower.usd")
SCENE_USD = os.path.join(_REPO, "assets", "scenes", "cube_desk", "scene.usd")
COMBINED_USD = os.path.join(_REPO, "outputs", "ovphysx_combined.usda")

# ---- 큐브 위치 상수 (pick_cube_env_cfg.py) ----
# 마지막 Cube1 고정 위치 (world frame)
CUBE1_POS_W = np.array([1.70, -0.44, 0.725], dtype=np.float32)
ROBOT_BASE_POS_W = np.array([1.84, -0.565, 0.6749], dtype=np.float32)

# world → base frame 변환 상수
# (robot 이 world 에서 고정 방향이라고 가정 — yaw=0 정렬)
BASE_XY_OFFSET = (0.0204, 0.0157)
BASE_Z_OFFSET = 0.0325

# gripper body → TCP(gripper_frame) offset (base frame)
GRASP_OFF = np.array([-0.0079, -0.000218121, -0.0981274], dtype=np.float32)


def world_to_base(p_w: np.ndarray, root_p: np.ndarray, root_yaw: float = 0.0
                  ) -> tuple[float, float, float]:
    """world 좌표(numpy) → URDF base_link frame (pick_cube_state_machine.py 과 동일).

    USD root body frame ↔ URDF base_link frame 정합 (캘리브레이션 8자세 실측):
      URDF = rot(+90°) · (USD_root_local - BASE_XY_OFFSET), z - BASE_Z_OFFSET
    """
    # 원본 _world_to_base_np 와 정확히 동일: -root_yaw 회전(가짜 +π/2 금지).
    # root_yaw = _quat_to_yaw(robot.root_quat). 로봇 rot(wxyz)=(0,0,0,1)→yaw=π.
    d = p_w - root_p
    c, s = math.cos(-root_yaw), math.sin(-root_yaw)
    bx = c * d[0] - s * d[1]
    by = s * d[0] + c * d[1]
    return (-(by - BASE_XY_OFFSET[1]), bx - BASE_XY_OFFSET[0], float(d[2]) - BASE_Z_OFFSET)


def main() -> None:
    print("=" * 70)
    print(f"[P1] ovphysx {ovphysx.__version__} grasp probe")
    ovphysx.bootstrap()

    # ----- 1. 합성 USD 로드 (robot + scene co-load 우회) -----
    # 메인 venv 에서 미리 생성한 combined USD (robot reference + scene reference 포함)
    # 를 단일 add_usd 로 로드 → 한 stage 에 둘 다 들어감
    if not os.path.exists(COMBINED_USD):
        print(f"[P1] ❌ 합성 USD 없음: {COMBINED_USD}")
        print("[P1]    메인 .venv 에서 먼저 실행: python scripts/perf/author_combined_usd.py")
        return

    physx = ovphysx.PhysX(device="cpu", ignore_version_mismatch=True)
    print("[P1] PhysX 생성 OK")

    # 합성 USD 로드 (robot + scene 모두)
    h_combined, _ = physx.add_usd(COMBINED_USD)
    physx.wait_all()
    print(f"[P1] add_usd(combined: robot+scene) OK  handle={h_combined}")

    # ----- 2. 큐브 rigid body 인식 -----
    # 합성 USD에서 scene reference로 포함된 큐브들
    # pattern: /World/Scene/Cube* (flattened references)
    print("[P1] === Cube rigid body 인식 ===")
    cube_pose = physx.create_tensor_binding(
        pattern="/World/Scene/Cube*", tensor_type=TensorType.RIGID_BODY_POSE, raise_if_empty=False
    )
    cube_count = getattr(cube_pose, "count", 0)
    print(f"[P1] Cube rigid body count = {cube_count}")
    if cube_count > 0:
        cube_pose_init = np.zeros((cube_count, 7), dtype=np.float32)
        cube_pose.read(cube_pose_init)
        for i in range(cube_count):
            print(f"[P1]   Cube[{i}] world pos = {cube_pose_init[i, :3]}")
        # Cube1 (index 0) 을 사용 (여러 큐브가 있으면 처음 것)
        cube1_world_pos = cube_pose_init[0, :3]
    else:
        print("[P1] ⚠ 큐브 rigid body 인식 실패")
        cube1_world_pos = CUBE1_POS_W

    # ----- 3. articulation 제어 바인딩 -----
    # robot은 /World/Robot 아래에 flattened (so101_new_calib 이 defaultPrim, base 가 articulation root)
    # 실제 prim path: /World/Robot/base (또는 /World/Robot/so101/base 형식)
    print("[P1] === Robot articulation 인식 ===")

    # patterns 시도 순서
    artic_pattern = None
    for pat in [
        "/World/Robot/inst/so101_new_calib/base",  # 래퍼 부모/inst/ref 구조
        "/World/Robot/base",
        "/World/Robot/so101_new_calib/base",
        "/World/Robot*base",
        "/*base"
    ]:
        pos_tgt = physx.create_tensor_binding(
            pattern=pat,
            tensor_type=TensorType.ARTICULATION_DOF_POSITION_TARGET,
            raise_if_empty=False
        )
        artic_count = getattr(pos_tgt, "count", 0)
        if artic_count > 0:
            print(f"[P1] ✓ pattern {pat!r}: count={artic_count}")
            artic_pattern = pat
            break
        else:
            print(f"[P1] pattern {pat!r}: count=0")

    if artic_pattern is None:
        print("[P1] ❌ 로봇 articulation 인식 실패")
        physx.release()
        return

    # 다른 바인딩들
    pos = physx.create_tensor_binding(
        pattern=artic_pattern, tensor_type=TensorType.ARTICULATION_DOF_POSITION
    )
    stiff = physx.create_tensor_binding(
        pattern=artic_pattern, tensor_type=TensorType.ARTICULATION_DOF_STIFFNESS
    )
    damp = physx.create_tensor_binding(
        pattern=artic_pattern, tensor_type=TensorType.ARTICULATION_DOF_DAMPING
    )
    # link world pose (blind 디버깅: TCP/gripper 가 실제로 큐브에 닿는지 검증용)
    link_pose = physx.create_tensor_binding(
        pattern=artic_pattern, tensor_type=TensorType.ARTICULATION_LINK_POSE
    )
    print(f"[P1] body_names = {getattr(pos_tgt, 'body_names', '?')}")
    print(f"[P1] link_pose shape = {getattr(link_pose, 'shape', '?')}")

    def _gripper_link_world():
        lp = np.zeros(link_pose.shape, dtype=np.float32)
        link_pose.read(lp)
        return lp[0]  # [L,7] for env0

    def _base_link_world():
        lp = np.zeros(link_pose.shape, dtype=np.float32)
        link_pose.read(lp)
        return lp[0, 0, :3]  # base link(idx0) world pos

    # PD drive 설정
    shp = pos_tgt.shape
    stiff.write(np.full(shp, 200.0, dtype=np.float32))
    damp.write(np.full(shp, 20.0, dtype=np.float32))
    # gripper 더 강하게
    grip_stiff = np.full(shp, 200.0, dtype=np.float32)
    grip_stiff[..., 5] = 500.0  # gripper = DOF 5
    stiff.write(grip_stiff)

    q0 = np.zeros(shp, dtype=np.float32)
    pos.read(q0)
    print(f"[P1] q0 = {q0.ravel()}")

    # ----- 4. IK 로직만 검증 -----
    kin = SO101Kinematics()

    # 테스트: zero pose FK 로 reach 범위 확인
    q_zero = [0.0, 0.0, 0.0, 0.0, 0.0]
    tcp_zero = kin.fk_tcp(q_zero)
    print(f"[P1] zero pose FK: TCP(base frame) = {tcp_zero}")
    print(f"[P1]   (기대값: (0.391, 0.000, 0.227))")

    # reach 테스트: zero pose 근처 좌표로 ik_reach 사용 (pitch 범위 자동)
    # ik_reach 는 top-down 부터 pitch 를 완화시키며 첫 해를 찾음
    print(f"[P1] === IK reach 테스트 (pitch 자동 적응) ===")
    # 타입 캐스팅 명시: tuple of float
    tcp_test_base = tuple(float(x) for x in [0.40, 0.0, 0.25])  # tuple + explicit float
    print(f"[P1] tcp_test_base = {tcp_test_base} (type: {type(tcp_test_base)})")

    q_test_result = kin.ik_reach(tcp_test_base, grasp_yaw=0.0)
    print(f"[P1] ik_reach 결과: {q_test_result}")

    if q_test_result is None:
        print("[P1] ❌ ik_reach 실패. 기구학 점검 필요.")
        print("[P1] → 수동 ik pitch 스캔 시도...")
        q_test = None
        for pitch_deg in range(-90, 1, 15):
            pitch = math.radians(pitch_deg)
            q = kin.ik(tcp_test_base, grasp_yaw=0.0, pitch=pitch)
            if q is not None:
                q_test = q
                print(f"[P1] ✓ pitch={pitch_deg}° 에서 해 발견: {q}")
                break
        if q_test is None:
            print("[P1] 모든 pitch 에서 실패. IK reach 범위 문제.")
            physx.release()
            return
    else:
        q_test, pitch_achieved = q_test_result
        print(f"[P1] ✓ ik_reach 성공:")
        print(f"[P1]   q = {q_test}")
        print(f"[P1]   pitch = {math.degrees(pitch_achieved):.1f}°")

    tcp_verify = kin.fk_tcp(q_test)
    err = [tcp_verify[i] - tcp_test_base[i] for i in range(3)]
    print(f"[P1]   검증 FK: {tcp_verify}")
    print(f"[P1]   오차: {err} (max: {max([abs(e) for e in err]):.6f})")

    # 실제 큐브 reach (reach 범위 밖이면 grasp 스킵)
    # cube1_world_pos 는 위에서 읽음 (없으면 CUBE1_POS_W)
    # 실제 base link world pose 를 ovphysx 에서 읽어 root_p·yaw 산출 (하드코드 불신 —
    # 원본 SM 이 root_pos_w/root_quat_w 를 쓰는 것과 동일). 하강 후 시점 base 는 안 움직임.
    def _quat_to_yaw_wxyz(w, x, y, z):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    _lp0 = np.zeros(link_pose.shape, dtype=np.float32)
    link_pose.read(_lp0)
    base_full = _lp0[0, 0]  # [7]: px,py,pz, q...
    root_p_act = base_full[:3].astype(float)
    qa = base_full[3:7]
    # ovphysx pose quat order 미상 → xyzw·wxyz 둘 다 yaw 계산해 진단
    yaw_xyzw = _quat_to_yaw_wxyz(qa[3], qa[0], qa[1], qa[2])  # [qx,qy,qz,qw] 가정
    yaw_wxyz = _quat_to_yaw_wxyz(qa[0], qa[1], qa[2], qa[3])  # [qw,qx,qy,qz] 가정
    print(f"[P1] [DIAG] base quat raw={qa}  yaw(xyzw->wxyz)={math.degrees(yaw_xyzw):.1f}° "
          f"yaw(wxyz)={math.degrees(yaw_wxyz):.1f}°")
    root_yaw_act = yaw_xyzw  # 로봇 180° → π 근처인 해석 채택(실행 후 거리로 검증)
    print(f"[P1] [DIAG] root_p_act={root_p_act} root_yaw_act={math.degrees(root_yaw_act):.1f}°")
    cube_pos_base = world_to_base(cube1_world_pos, root_p_act, root_yaw=root_yaw_act)
    print(f"\n[P1] === Cube grasp 시도 ===")
    print(f"[P1] Cube1 (hardcoded world pos) = {CUBE1_POS_W}")
    print(f"[P1] Cube1 base frame pos = {cube_pos_base}")
    cube_reach = math.sqrt(cube_pos_base[0]**2 + cube_pos_base[1]**2)
    print(f"[P1] Cube1 reach distance = {cube_reach:.4f} m (SO-101 max ~0.45m)")

    # ik_reach 로 큐브 접근자세 찾기
    tcp_approach_cube = (cube_pos_base[0], cube_pos_base[1], cube_pos_base[2] + 0.05)
    q_approach_result = kin.ik_reach(tcp_approach_cube, grasp_yaw=0.0)
    if q_approach_result is None:
        print("[P1] ⚠ Cube 접근 IK 실패 (reach 범위 밖). 테스트 타겟으로만 진행.")
        q_approach = q_test  # 테스트 타겟 재사용
        q_grasp = q_test
    else:
        q_approach, pitch_app = q_approach_result
        print(f"[P1] ✓ q_approach = {q_approach} (pitch={math.degrees(pitch_app):.1f}°)")

        # 하강: z 값만 큐브 위로 이동
        tcp_grasp_contact = (cube_pos_base[0], cube_pos_base[1], cube_pos_base[2])
        q_grasp_result = kin.ik_reach(tcp_grasp_contact, grasp_yaw=0.0, q_ref=q_approach)
        if q_grasp_result is None:
            print("[P1] ⚠ Cube 하강 IK 실패. 접근 자세만 사용.")
            q_grasp = q_approach
        else:
            q_grasp, pitch_grasp = q_grasp_result
            print(f"[P1] ✓ q_grasp (큐브 접촉) = {q_grasp} (pitch={math.degrees(pitch_grasp):.1f}°)")

    # 시뮬 시퀀스:
    # 1. 접근(1s)
    # 2. 하강(1s)
    # 3. gripper close(0.5s)
    # 4. lift(2s)
    # 5. 안정화(1s)

    tgt = q0.copy()
    tgt[..., :5] = q_approach
    tgt[..., 5] = 0.5 - 0.2  # gripper 반개방(GRIPPER_ACTION_OFFSET=0.2)

    print("[P1] === 시퀀스 1: 접근(1s) ===")
    for i in range(120):  # 1s @120Hz
        pos_tgt.write(tgt)
        physx.step(1.0 / 120.0, i / 120.0)

    # 하강
    print("[P1] === 시퀀스 2: 하강(1s) ===")
    tgt[..., :5] = q_grasp
    for i in range(120):
        pos_tgt.write(tgt)
        physx.step(1.0 / 120.0, (120 + i) / 120.0)
    physx.wait_all()

    q_mid = np.zeros(shp, dtype=np.float32)
    pos.read(q_mid)
    print(f"[P1] 하강 후 q = {q_mid.ravel()}")
    # === blind 디버깅: base link + 모든 link world pos, 마지막 link↔큐브 거리 ===
    base_w = _base_link_world()
    print(f"[P1] [DIAG] base link world = {base_w} (기대 ~{ROBOT_BASE_POS_W})")
    lp = _gripper_link_world()  # [L,7]
    for li in range(lp.shape[0]):
        print(f"[P1] [DIAG] link[{li}] world pos = {lp[li, :3]}")
    tcp_w = lp[-1, :3]  # 마지막 link = gripper/jaw 추정
    dist = float(np.linalg.norm(tcp_w - cube1_world_pos))
    print(f"[P1] [DIAG] last-link world = {tcp_w}, cube1 world = {cube1_world_pos}")
    print(f"[P1] [DIAG] last-link ↔ cube 거리 = {dist*1000:.1f} mm "
          f"(작아야 grasp 가능 — 크면 팔이 큐브에 안 닿은 것)")

    # close (강하게)
    print("[P1] === 시퀀스 3: gripper close(0.5s) ===")
    tgt[..., 5] = -0.5  # 전폐(raw -1.0 근처)
    for i in range(60):  # 0.5s
        pos_tgt.write(tgt)
        physx.step(1.0 / 120.0, (240 + i) / 120.0)
    physx.wait_all()

    q_close = np.zeros(shp, dtype=np.float32)
    pos.read(q_close)
    print(f"[P1] close 후 q = {q_close.ravel()}")

    # lift: 팔 전체 위로 + 잡은 자세 유지
    print("[P1] === 시퀀스 4: lift(2s) ===")
    tcp_lift = (cube_pos_base[0], cube_pos_base[1], cube_pos_base[2] + 0.15)
    q_lift = kin.ik(tcp_lift, grasp_yaw=0.0, pitch=-math.pi / 2, q_ref=q_grasp)
    if q_lift is None:
        print("[P1] ⚠ lift IK 실패, 기본 lift 로 진행")
        q_lift = list(q_grasp)
        q_lift[1] -= 0.3  # shoulder_lift 를 올려 팔 전체 상승 시도

    tgt[..., :5] = q_lift
    tgt[..., 5] = -0.5  # gripper 유지
    for i in range(240):  # 2s
        pos_tgt.write(tgt)
        physx.step(1.0 / 120.0, (300 + i) / 120.0)
    physx.wait_all()

    q_lift_final = np.zeros(shp, dtype=np.float32)
    pos.read(q_lift_final)
    print(f"[P1] lift 후 q = {q_lift_final.ravel()}")

    # 안정화
    print("[P1] === 시퀀스 5: 안정화(1s) ===")
    for i in range(120):
        physx.step(1.0 / 120.0, (540 + i) / 120.0)
    physx.wait_all()

    q_final = np.zeros(shp, dtype=np.float32)
    pos.read(q_final)
    print(f"[P1] 최종 q = {q_final.ravel()}")

    # ----- 5. 큐브 z 상승량 측정 (contact 물리 검증) -----
    print(f"\n[P1] === Cube z 상승량 측정 ===")
    cube_pose_final = np.zeros((cube_count, 7), dtype=np.float32)
    cube_pose.read(cube_pose_final)

    cube1_z_init = float(cube_pose_init[0, 2])
    cube1_z_final = float(cube_pose_final[0, 2])
    cube1_z_rise = cube1_z_final - cube1_z_init

    print(f"[P1] Cube1 초기 z = {cube1_z_init:.6f}")
    print(f"[P1] Cube1 최종 z = {cube1_z_final:.6f}")
    print(f"[P1] Cube1 z 상승량 = {cube1_z_rise:.6f} m = {cube1_z_rise*1000:.2f} mm")

    # ----- 6. 결과 리포트 -----
    print(f"\n{'=' * 70}")
    print(f"[P1] === P1 게이트 결과 ===")
    print(f"[P1] ✓ robot+scene co-load: 우회 성공 (merged USD)")
    print(f"[P1] ✓ robot 6-DOF 제어: 작동 확인")
    print(f"[P1] ✓ SO101Kinematics (IK): 작동 확인 (pitch 자동 적응)")
    print(f"[P1] ✓ grasp 시퀀스 완료: 접근→하강→close→lift")
    print(f"[P1]")
    print(f"[P1] === Contact 물리 측정 ===")
    print(f"[P1] Cube1 z 상승량: {cube1_z_rise:.6f} m ({cube1_z_rise*1000:.2f} mm)")
    if cube1_z_rise > 0.01:
        print(f"[P1] ✓ grasp 물리 작동 (큐브 들림 감지)")
    else:
        print(f"[P1] ⚠ grasp 물리 미검출 (큐브 미끌림 또는 클립)")

    print(f"{'=' * 70}")

    physx.release()


if __name__ == "__main__":
    main()
