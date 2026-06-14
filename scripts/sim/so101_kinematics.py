"""SO-101 5축 닫힌 해 FK/IK — `pick_cube_state_machine.py::SO101Kinematics` 의 verbatim 추출.

P2(cuRobo joint-goal) 와 인터랙티브 Viser 데모가 Isaac 부팅 없이 해석적 IK 를 쓰기 위한 standalone
모듈. **로직은 SM 원본과 동일**(C3 frozen — 재작성 아님, 단순 분리). 좌표=robot base_link frame.

cuRobo so101_curobo.yml 의 robot frame 과 동일(base_link, zero-pose TCP=(0.391,0,0.227)). 따라서
cuRobo world(=robot base 원점) 좌표를 frame 변환 없이 그대로 tcp 로 넘길 수 있다.

P2 에서 SM 과 이 모듈을 단일 소스로 통합 권장 (현재는 drift 방지 위해 verbatim 동기 유지).
"""

from __future__ import annotations

import math


class SO101Kinematics:
    """SO-101 5축 닫힌 해 FK/IK (robot base_link frame 기준).

    URDF(assets/robots/urdf/so_arm101.urdf) origin 체인을 base frame 에서 전개한 결과:
      · shoulder_pan 축: base (PAN_X, 0) 위치의 -z 축 → +q1 명령 = world yaw -q1
      · lift/elbow/wrist_flex 축: 모두 pan 회전 평면의 같은 pitch 축(+y 방향)
      · zero pose TCP = base (0.391, 0.000, 0.227) — pan 평면 위에 정확히 위치
      · wrist_roll(q5) 회전 시 TCP 가 roll 축 주위 반경 ROLL_RHO(7.9mm) 원을 돌므로
        q5 확정 후 lateral 1차 보정을 적용한다.
    """

    # pan 축 base 위치/높이 (URDF shoulder_pan origin)
    PAN_X = 0.0388353
    # pan 축 기준 lift 축 radial 오프셋·base 기준 lift 축 높이
    LIFT_R = 0.0303992
    LIFT_Z = 0.0624 + 0.0542  # = 0.1166
    # 평면 링크 길이: lift→elbow, elbow→wrist_flex, wrist_flex→TCP(gripper_frame)
    L1 = math.hypot(0.11257, 0.028)    # 0.11600
    L2 = math.hypot(0.1349, 0.0052)    # 0.13500
    L3 = math.hypot(0.1592, 0.0079)    # 0.15940 (wrist_roll origin + gripper_frame 합성)
    # zero-pose 평면각 (수평 기준, 위가 +)
    TH1_0 = math.atan2(0.11257, 0.028)   # 상완 76.0° 위
    TH2_0 = math.atan2(0.0052, 0.1349)   # 전완 2.2° 위
    TH3_0 = math.atan2(-0.0079, 0.1592)  # wrist→TCP -2.8°
    # wrist_roll 축 ↔ TCP lateral 반경 (gripper_frame offset 의 roll 축 수직 성분)
    ROLL_RHO = 0.0079

    # joint limits (URDF): pan, lift, elbow, wrist_flex, wrist_roll
    JOINT_LIMITS = [
        (-1.91986, 1.91986),
        (-1.74533, 1.74533),
        (-1.69, 1.69),
        (-1.65806, 1.65806),
        (-2.74385, 2.84121),
    ]

    def fk_tcp(self, q: list[float]) -> tuple[float, float, float]:
        """관절각 → TCP(gripper_frame) 위치, base frame. 검증·INIT 동기화용."""
        q1, q2, q3, q4, q5 = (float(v) for v in q[:5])
        th1 = self.TH1_0 - q2
        th2 = th1 + (self.TH2_0 - self.TH1_0) - q3
        th3 = th2 + (self.TH3_0 - self.TH2_0) - q4
        pr = (self.LIFT_R + self.L1 * math.cos(th1) + self.L2 * math.cos(th2)
              + self.L3 * math.cos(th3))
        pz = (self.LIFT_Z + self.L1 * math.sin(th1) + self.L2 * math.sin(th2)
              + self.L3 * math.sin(th3))
        lat = self.ROLL_RHO * math.sin(q5)
        x = self.PAN_X + pr * math.cos(q1) - lat * math.sin(q1)
        y = -(pr * math.sin(q1) + lat * math.cos(q1))
        z = pz
        return (x, y, z)

    def ik(self, tcp: tuple[float, float, float], grasp_yaw: float,
           pitch: float = -math.pi / 2,
           q_ref: list[float] | None = None,
           roll_offset: float = 0.0) -> list[float] | None:
        """TCP 목표(base frame) + 손가락 닫힘축 yaw → 관절각 5개. 도달 불가 시 None.

        pitch: 툴 접근 피치(wrist→TCP 평면각). -π/2 = 수직 top-down.
        roll_offset: q5 에 더하는 고정 회전 (rad). 큐브 90° 대칭을 이용해 닫힘축을
        ±90° 돌린 대안 grasp 자세 — 장애물 회피용.
        """
        x, y, z = tcp
        dx, dy = x - self.PAN_X, y
        r = math.hypot(dx, dy)
        if r < 1e-6:
            return None
        q1 = -math.atan2(dy, dx)  # pan 축 = base -z → 부호 반전

        q5 = self._fold_45(grasp_yaw + q1) + roll_offset

        # q5 lateral 보정: TCP 가 roll 축에서 ρ·sin(q5) 만큼 옆으로 벗어남 → radial 로 환원
        lat = self.ROLL_RHO * math.sin(q5)
        r_eff = math.sqrt(max(r * r - lat * lat, 1e-9))

        # wrist_flex 축 위치 역산 (pitch 방향으로 L3 제거)
        pr = r_eff - self.LIFT_R
        pz = z - self.LIFT_Z
        wr = pr - self.L3 * math.cos(pitch)
        wz = pz - self.L3 * math.sin(pitch)

        # 2-link planar IK
        d2 = wr * wr + wz * wz
        c_rel = (d2 - self.L1 * self.L1 - self.L2 * self.L2) / (2.0 * self.L1 * self.L2)
        if abs(c_rel) > 1.0:
            return None  # 작업 반경 밖
        rel = math.acos(c_rel)

        base_ang = math.atan2(wz, wr)
        candidates: list[list[float]] = []
        for sign in (-1.0, 1.0):
            th1 = base_ang - math.atan2(self.L2 * math.sin(sign * rel),
                                        self.L1 + self.L2 * math.cos(sign * rel))
            th2 = th1 + sign * rel
            th3 = pitch
            q2 = self.TH1_0 - th1
            q3 = (self.TH2_0 - self.TH1_0) - (th2 - th1)
            q4 = (self.TH3_0 - self.TH2_0) - (th3 - th2)
            q = [q1, q2, q3, q4, q5]
            if all(lo <= v <= hi for v, (lo, hi) in zip(q, self.JOINT_LIMITS)):
                candidates.append(q)
        if not candidates:
            return None
        if q_ref is None or len(candidates) == 1:
            return candidates[0]
        return min(candidates,
                   key=lambda q: sum(abs(a - b) for a, b in zip(q, q_ref)))

    def ik_reach(self, tcp: tuple[float, float, float], grasp_yaw: float,
                 pitch_min: float = math.radians(-90),
                 pitch_max: float = math.radians(-30),
                 q_ref: list[float] | None = None,
                 roll_offset: float = 0.0) -> tuple[list[float], float] | None:
        """top-down 우선, 도달 불가 시 pitch 를 점진 완화하며 첫 해와 채택 pitch 반환.

        5-DOF position 우선·orientation best-effort 규약 (AGENTS.md).
        pitch_max -30°: SO-101 reach 가장자리(그릇 등 r>0.33)는 비스듬해야만 닿는다.
        """
        n_steps = 13
        for i in range(n_steps):
            pitch = pitch_min + (pitch_max - pitch_min) * i / (n_steps - 1)
            q = self.ik(tcp, grasp_yaw, pitch=pitch, q_ref=q_ref, roll_offset=roll_offset)
            if q is not None:
                return q, pitch
        return None

    @staticmethod
    def _fold_45(a: float) -> float:
        return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4
