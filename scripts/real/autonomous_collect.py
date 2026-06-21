"""실기기 SO-101 자율 pick-place 데이터 수집 (LeRobot v3, 무인).

Windows 네이티브 LeRobot 0.4.4 `SOFollower`(COM 직결) + 고전 비전(HSV) + 해석적 IK
scripts-expert 로 큐브를 그릇에 담고(record) 그릇에서 꺼내 흩어(reset) 자율 반복한다.
학습 정책(SmolVLA/GR00T) 미사용 — 사용자 확인상 큐브 1개도 못 집음.

스테이지(--stage):
  check        무모션. 연결+joint 읽기+3캠 프레임 저장 → 카메라 index·색·calib 검증
  touchcalib   사용자 터치 hand-eye(detect→touch ×N→finish) → calibration.json
  collect      [bowl] test(1 grasp) → verify(5ep) → collect → finalize (+옵션 업로드)
  stack        [ECE4560 assignment8] 스택 1회 검증(비기록, cubic spline). offset/사이즈/그립 튜닝
  stackcollect [ECE4560 assignment8] 스택 VLA 수집: test→verify→record N→finalize (+업로드)
               좌→우 큐브를 최우측 바닥에 적재. reset=unstack 산포(또는 --manual-reset)

단위: SOFollower 가 네이티브 LeRobot 단위 입출력(arm degrees, gripper [0,100]).
  · IK(rad) → 명령: arm deg=rad*180/pi, gripper 는 [0,100] 직접 지정(OPEN/CLOSE 튜닝값).
  · 기록: get_observation(state)·send_action 반환(clamped action) 둘 다 이미 lerobot 단위 → 그대로.
  · 이미지: get_observation 는 RGB → recorder(imageio RGB)·검출(vision 내부 BGR 변환) 양쪽 호환.

실행:
  uv run --group teleop python scripts/real/autonomous_collect.py --stage check
  uv run --group teleop python scripts/real/autonomous_collect.py --stage calibrate
  uv run --group teleop python scripts/real/autonomous_collect.py --stage collect --target 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── 경로: scripts/real(이 디렉터리) + scripts/sim(recorder) ───────────────────
_HERE = Path(__file__).resolve().parent
_SIM = _HERE.parent / "sim"
for _p in (str(_HERE), str(_SIM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import trajectory as TRAJ  # noqa: E402
import vision as V  # noqa: E402
from so101_kinematics import SO101Kinematics  # noqa: E402

# ── 상수 / 기본 설정 ──────────────────────────────────────────────────────────
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS6 = ARM + ["gripper"]
TASK_NAME = "pick up the cube and place it in the bowl"
STACK_TASK_NAME = "stack the cubes"  # ECE4560 assignment8 block stacking
_RAD2DEG = 180.0 / math.pi
_DEG2RAD = math.pi / 180.0

log = logging.getLogger("auto")


@dataclass
class Cfg:
    # 하드웨어 (.env)
    robot_port: str = os.getenv("ROBOT_PORT", "COM8")
    robot_id: str = os.getenv("ROBOT_ID", "so101_robot")
    top_idx: int = int(os.getenv("TOP_CAM_PORT", "2"))
    wrist_idx: int = int(os.getenv("WRIST_CAM_PORT", "1"))
    front_idx: int = int(os.getenv("FRONT_CAM_PORT", "0"))
    cam_w: int = 640
    cam_h: int = 480
    cam_fps: int = 25
    cam_fourcc: str = "MJPG"
    cam_warmup_s: int = 3
    # 안전
    max_rel_target: float = 5.0   # per-step joint Δ 상한(deg/[0,100]) — 잘못된 IK 점프 차단
    fps: float = 30.0
    r_min: float = 0.12
    r_max: float = 0.42
    z_floor: float = 0.005
    # 기하 (base_link frame, m) — base_link ≈ desk 레벨
    safe_z: float = 0.15
    observe_tcp: tuple = (0.20, 0.0, 0.26)  # 검출 전 팔 접는 pose = fold 가 실제 도달한 candidate(워크스페이스 가림 제거+3큐브 검출 확인)
    grasp_z: float = 0.02         # grasp TCP z (튜닝)
    bowl_clear_z: float = 0.18
    release_z: float = 0.08
    calib_z: float = 0.06         # 캘리브 높이. 큰 ArUco 시트가 책상 클리어하도록(parallax는 ArUco robust로 감수)
    side_offset: float = 0.035    # side-approach 횡오프셋
    calib_offset_x: float = 0.0   # tip≠TCP 보정(test서 튜닝)
    calib_offset_y: float = 0.0
    # 그리퍼 [0,100] (test서 튜닝)
    grip_open: float = 70.0
    grip_close: float = 6.0
    # waypoint
    joint_tol: float = 0.05       # rad
    move_timeout: float = 12.0
    stuck_secs: float = 2.5
    close_dwell_s: float = 0.6
    max_ep_steps: int = 1200
    # ── 스택(ECE4560 assignment8) ──────────────────────────────────────────
    # 큐브 한 변(m). 적재 z·grasp_z 계산. 이종이면 base/pick 따로 지정(rightmost=base).
    cube_size: float = 0.04       # 기본 변 길이(--cube-size 가 base/pick 양쪽 set)
    base_size: float | None = None  # 바닥(최우측) 큐브 변. None=cube_size
    pick_size: float | None = None  # 들어올릴 큐브 변. None=cube_size
    num_blocks: int = 2           # 스택 총 블록 수(part2=2, part3=3). 검출 큐브로 제한
    stack_clear: float = 0.03     # 접근/이탈 시 타겟 위 여유 높이(assignment8=0.03)
    grasp_z_margin: float = 0.0   # grasp TCP z = pick_size/2 + margin (음수면 더 깊게)
    seg_dur_move: float = 2.0     # 큰 이동 cubic 구간 길이(s) — assignment9
    seg_dur_act: float = 1.0      # 하강/그립토글/이탈 cubic 구간 길이(s)
    manual_reset: bool = False    # True=리셋 시 사용자에게 재배치 요청(Enter), False=자동 unstack 산포
    # 캘리브 그리드 — 신뢰 reach 안(far stall 회피) + 가림 적은 영역
    calib_xs: tuple = (0.16, 0.20, 0.24, 0.27)
    calib_ys: tuple = (-0.08, -0.03, 0.03, 0.08)
    calib_settle_s: float = 2.0
    calib_resid_max_m: float = 0.02
    use_aruco: bool = False       # 그리퍼 부착 ArUco 마커로 tip 검출(저각 카메라 robust)
    aruco_id: int = -1            # 특정 id만(-1=첫 마커)
    # scatter (reset 산포 reachable 범위)
    scatter_x: tuple = (0.19, 0.30)
    scatter_y: tuple = (-0.11, 0.11)
    scatter_sep: float = 0.06
    # 게이트
    verify_eps: int = 5
    verify_min_rate: float = 0.6
    grasp_attempts: int = 3
    # 경로
    dataset_root: Path = Path("datasets/pick_cube_real")          # calibration.json 위치(읽기)
    stack_dataset_root: Path = Path("datasets/stack_cubes_real")  # 스택 LeRobot 데이터(쓰기, 분리 — calib rmtree 방지)
    debug_root: Path = Path("outputs/real_debug")


class _ClampFilter(logging.Filter):
    """lerobot 의 max_relative_target clamp WARNING(root logger, 스텝마다 1건) 억제."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "clamped to be safe" not in record.getMessage()


def setup_logging(stage: str, cfg: Cfg) -> None:
    cfg.debug_root.mkdir(parents=True, exist_ok=True)
    # Windows 콘솔 cp949 가 한글/°/≈ 인코딩 실패 → stdout/err UTF-8 강제(로그 에러 방지).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.debug_root / "autonomous_loop.log", encoding="utf-8"),
        ],
    )
    for h in logging.getLogger().handlers:
        h.addFilter(_ClampFilter())
    log.info("=== stage=%s ===", stage)


# ── 로봇 IO ───────────────────────────────────────────────────────────────────
class ArmIO:
    """SOFollower wrapper: 연결/관측/명령 + 단위 glue + 안전."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.robot = None

    def connect(self, retries: int = 3) -> None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        def cam(idx):
            return OpenCVCameraConfig(
                index_or_path=idx, width=self.cfg.cam_w, height=self.cfg.cam_h,
                fps=self.cfg.cam_fps, fourcc=self.cfg.cam_fourcc, warmup_s=self.cfg.cam_warmup_s,
            )
        cams = {"top": cam(self.cfg.top_idx), "wrist": cam(self.cfg.wrist_idx),
                "front": cam(self.cfg.front_idx)}
        rc = SOFollowerRobotConfig(
            port=self.cfg.robot_port, id=self.cfg.robot_id,
            max_relative_target=self.cfg.max_rel_target, cameras=cams, use_degrees=True,
        )
        # feetech serial 은 connect 시 간헐 "no status packet"(특히 gripper id=6) → 재시도.
        last = None
        for attempt in range(1, retries + 1):
            self.robot = SOFollower(rc)
            log.info("connecting SOFollower port=%s id=%s (try %d/%d) ...",
                     self.cfg.robot_port, self.cfg.robot_id, attempt, retries)
            try:
                self.robot.connect(calibrate=False)  # 무인: input() 프롬프트 회피, 기존 calib 사용
                log.info("robot connected. is_connected=%s", self.robot.is_connected)
                return
            except Exception as e:
                last = e
                log.warning("connect attempt %d failed: %s", attempt, e)
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
                time.sleep(2.0)
        raise RuntimeError(f"connect failed after {retries} tries: {last}")

    def disconnect(self) -> None:
        try:
            if self.robot is not None:
                self.robot.disconnect()
                log.info("robot disconnected")
        except Exception as e:
            log.warning("disconnect error: %s", e)

    def observe(self) -> dict:
        return self.robot.get_observation()

    @staticmethod
    def state_vec(obs: dict) -> np.ndarray:
        """obs → [pan,lift,elbow,wrist_flex,wrist_roll,gripper] (arm deg, grip [0,100])."""
        return np.array([obs[f"{j}.pos"] for j in JOINTS6], dtype=np.float32)

    @staticmethod
    def arm_rad(obs: dict) -> list[float]:
        return [obs[f"{j}.pos"] * _DEG2RAD for j in ARM]

    @staticmethod
    def images(obs: dict) -> dict:
        return {c: np.ascontiguousarray(obs[c]) for c in ("top", "wrist", "front")}

    def action_dict(self, q_rad: list[float], grip_0_100: float) -> dict:
        d = {f"{ARM[i]}.pos": float(q_rad[i] * _RAD2DEG) for i in range(5)}
        d["gripper.pos"] = float(max(0.0, min(100.0, grip_0_100)))
        return d

    def action_dict_deg6(self, vec6: list[float]) -> dict:
        """[arm5 deg, gripper 0-100] 6-vec → action dict (cubic 궤적 재생용)."""
        d = {f"{ARM[i]}.pos": float(vec6[i]) for i in range(5)}
        d["gripper.pos"] = float(max(0.0, min(100.0, vec6[5])))
        return d

    def send(self, action: dict) -> np.ndarray:
        """send_action → 실제 전송(clamped) action 6-vec 반환."""
        sent = self.robot.send_action(action)
        return np.array([sent.get(f"{j}.pos", action[f"{j}.pos"]) for j in JOINTS6], dtype=np.float32)


# ── 컨트롤러 (waypoint executor + grasp/place/reset SM + 기록) ────────────────
class Controller:
    def __init__(self, io: ArmIO, cfg: Cfg):
        self.io = io
        self.cfg = cfg
        self.kin = SO101Kinematics()
        self.H = None
        self.writer = None
        self.recording = False
        self.dt = 1.0 / cfg.fps

    # 좌표 헬퍼
    def in_workspace(self, tcp) -> bool:
        x, y, z = tcp
        r = math.hypot(x - self.kin.PAN_X, y)
        return (self.cfg.r_min <= r <= self.cfg.r_max) and (z >= self.cfg.z_floor)

    def cube_xy(self, pixel) -> tuple[float, float]:
        x, y = V.pixel_to_base_xy(self.H, pixel)
        return (x + self.cfg.calib_offset_x, y + self.cfg.calib_offset_y)

    # 기록 1 tick
    def _record(self, action_vec: np.ndarray, obs: dict) -> None:
        if self.recording and self.writer is not None:
            self.writer.add_frame(action_vec, self.io.state_vec(obs), self.io.images(obs))

    def move_to(self, tcp, grip, *, pitch_min=math.radians(-90), pitch_max=math.radians(-30),
                grasp_yaw=0.0, roll_offset=0.0, timeout=None, tol=None, stuck_secs=None) -> bool:
        """30Hz: waypoint 1개 = 고정 joint target. IK 를 진입 시 **1회만** 풀어(q_ref=시작자세)
        그 target 으로 수렴까지 명령(하드웨어 max_rel_target 가 step 클램프). 매 tick 재해(再解)는
        pitch 분기가 바뀌어 target 이 진동→false stuck 유발하므로 금지."""
        if not self.in_workspace(tcp):
            log.warning("target out of workspace: %s", [round(v, 3) for v in tcp])
            return False
        timeout = timeout or self.cfg.move_timeout
        tol = tol or self.cfg.joint_tol
        stuck_secs = stuck_secs or self.cfg.stuck_secs
        try:
            q_ref = self.io.arm_rad(self.io.observe())
        except Exception as e:
            log.error("observe failed (watchdog): %s", e)
            return False
        sol = self.kin.ik_reach(tcp, grasp_yaw, pitch_min=pitch_min, pitch_max=pitch_max,
                                q_ref=q_ref, roll_offset=roll_offset)
        if sol is None:
            log.warning("IK fail at %s", [round(v, 3) for v in tcp])
            return False
        q_t = sol[0]  # 고정 target
        t0 = time.perf_counter()
        best = float("inf")
        best_t = t0
        while True:
            now = time.perf_counter()
            try:
                obs = self.io.observe()
            except Exception as e:
                log.error("observe failed (watchdog): %s", e)
                return False
            q_now = self.io.arm_rad(obs)
            err = max(abs(q_now[i] - q_t[i]) for i in range(5))
            if err < tol:
                self.io.send(self.io.action_dict(q_t, grip))
                return True
            if err < best - 1e-3:
                best, best_t = err, now
            elif now - best_t > stuck_secs:
                log.warning("stuck at %s (err=%.3f)", [round(v, 3) for v in tcp], err)
                return False
            if now - t0 > timeout:
                log.warning("timeout to %s (err=%.3f)", [round(v, 3) for v in tcp], err)
                return False
            sent = self.io.send(self.io.action_dict(q_t, grip))
            self._record(sent, obs)
            time.sleep(max(0.0, self.dt - (time.perf_counter() - now)))

    def move_cubic(self, tcp, grip, duration=None, *,
                   pitch_min=math.radians(-90), pitch_max=math.radians(-30),
                   grasp_yaw=0.0, roll_offset=0.0) -> bool:
        """ECE4560 assignment9 cubic spline 궤적을 open-loop 재생(스택 SM 용).

        세그먼트 시작마다 **측정** 6-vec 을 a0 로(이전 세그먼트 랙 흡수), 목표 TCP 의 IK 를
        1회 풀어 goal 6-vec(arm deg+grip) 산출 → 6축 독립 cubic(zero-vel 양끝)을 fps 로
        샘플해 실시간 송신+record. arm Δ=0·grip 만 변하면 부드러운 그립 토글이 된다.
        목표 grip 은 구간 끝값으로 spline(중간 보간) → soft 큐브에 충격 없이 파지/해제.
        """
        duration = self.cfg.seg_dur_move if duration is None else duration
        if not self.in_workspace(tcp):
            log.warning("cubic target out of workspace: %s", [round(v, 3) for v in tcp])
            return False
        try:
            obs0 = self.io.observe()
        except Exception as e:
            log.error("observe failed (move_cubic start): %s", e)
            return False
        q_ref = self.io.arm_rad(obs0)
        sol = self.kin.ik_reach(tcp, grasp_yaw, pitch_min=pitch_min, pitch_max=pitch_max,
                                q_ref=q_ref, roll_offset=roll_offset)
        if sol is None:
            log.warning("cubic IK fail at %s", [round(v, 3) for v in tcp])
            return False
        q_goal_deg = [v * _RAD2DEG for v in sol[0]]
        start6 = [float(v) for v in self.io.state_vec(obs0)]   # 측정 arm deg + grip
        goal6 = q_goal_deg + [float(grip)]
        coeffs = TRAJ.cubic_coeffs(start6, goal6, duration)
        t0 = time.perf_counter()
        while True:
            now = time.perf_counter()
            t = now - t0
            vec6 = TRAJ.cubic_eval(coeffs, t, duration)
            try:
                obs = self.io.observe()
            except Exception as e:
                log.error("observe failed (move_cubic): %s", e)
                return False
            sent = self.io.send(self.io.action_dict_deg6(vec6))
            self._record(sent, obs)
            if t >= duration:
                break
            time.sleep(max(0.0, self.dt - (time.perf_counter() - now)))
        return True

    def set_gripper(self, grip, hold_pose_tcp, *, dwell_s=None, **mv) -> None:
        """현 pose 유지하며 그리퍼만 grip 으로(dwell). hold_pose_tcp 로 IK 재명령."""
        dwell_s = dwell_s if dwell_s is not None else self.cfg.close_dwell_s
        try:
            q_ref = self.io.arm_rad(self.io.observe())
        except Exception as e:
            log.error("observe failed in set_gripper: %s", e)
            return
        sol = self.kin.ik_reach(hold_pose_tcp, mv.get("grasp_yaw", 0.0),
                                q_ref=q_ref, roll_offset=mv.get("roll_offset", 0.0))
        q_t = sol[0] if sol else q_ref  # 고정 hold target
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < dwell_s:
            now = time.perf_counter()
            try:
                obs = self.io.observe()
            except Exception as e:
                log.error("observe failed in set_gripper: %s", e)
                return
            sent = self.io.send(self.io.action_dict(q_t, grip))
            self._record(sent, obs)
            time.sleep(max(0.0, self.dt - (time.perf_counter() - now)))

    def home(self, grip=None) -> None:
        grip = self.cfg.grip_open if grip is None else grip
        self.move_to((0.24, 0.0, self.cfg.safe_z), grip, timeout=14.0)

    def go_observe(self) -> None:
        """검출 전 팔을 top 카메라 밖(높이 retract)으로 접어 워크스페이스 가림 제거."""
        self.move_to(self.cfg.observe_tcp, self.cfg.grip_open, timeout=14.0)

    # ── pick → place in bowl ──────────────────────────────────────────────
    def pick(self, cube_xy, grasp_capture_dir=None) -> bool:
        """큐브 집기: hover→side-approach 하강→slide→close→lift. 성공 시 큐브 든 채 safe_z.

        grasp_capture_dir 지정 시 close 직후(lift 전) top/wrist/front 저장(offset 튜닝용)."""
        cx, cy = cube_xy
        ang = math.atan2(cy, cx - self.kin.PAN_X)
        ox = -self.cfg.side_offset * math.cos(ang)
        oy = -self.cfg.side_offset * math.sin(ang)
        seq = [
            ("hover", (cx, cy, self.cfg.safe_z), self.cfg.grip_open),
            ("pre", (cx + ox, cy + oy, self.cfg.grasp_z + 0.02), self.cfg.grip_open),
            ("descend", (cx + ox, cy + oy, self.cfg.grasp_z), self.cfg.grip_open),
            ("slide", (cx, cy, self.cfg.grasp_z), self.cfg.grip_open),
        ]
        for name, tcp, grip in seq:
            if not self.move_to(tcp, grip):
                log.warning("pick abort at %s", name)
                return False
        if grasp_capture_dir is not None:
            obs = self.io.observe()
            for c in ("top", "wrist", "front"):
                V.save_rgb(Path(grasp_capture_dir) / f"grasp_{c}.png", np.ascontiguousarray(obs[c]))
        self.set_gripper(self.cfg.grip_close, (cx, cy, self.cfg.grasp_z), dwell_s=self.cfg.close_dwell_s)
        if not self.move_to((cx, cy, self.cfg.safe_z), self.cfg.grip_close):
            return False
        return True

    def place_in_bowl(self, bowl_xy) -> bool:
        bx, by = bowl_xy
        if not self.move_to((bx, by, self.cfg.bowl_clear_z), self.cfg.grip_close):
            return False
        if not self.move_to((bx, by, self.cfg.release_z), self.cfg.grip_close):
            return False
        self.set_gripper(self.cfg.grip_open, (bx, by, self.cfg.release_z), dwell_s=self.cfg.close_dwell_s)
        self.move_to((bx, by, self.cfg.bowl_clear_z), self.cfg.grip_open)
        self.home()
        return True

    def grasp_and_place(self, cube_xy, bowl_xy) -> bool:
        if not self.pick(cube_xy):
            return False
        return self.place_in_bowl(bowl_xy)

    # ── 스택(ECE4560 assignment8) — 수직 pick/place 프리미티브 (cubic spline) ──
    def _pick_at(self, xy, grasp_z, capture_dir=None) -> bool:
        """수직 grasp: above(open)→descend(open)→close→lift(closed). assignment8 pick_up_block."""
        cx, cy = xy
        gz = max(grasp_z, self.cfg.z_floor)
        up = gz + self.cfg.stack_clear
        seq = [
            ("above", (cx, cy, up), self.cfg.grip_open, self.cfg.seg_dur_move),
            ("descend", (cx, cy, gz), self.cfg.grip_open, self.cfg.seg_dur_act),
        ]
        for name, tcp, grip, dur in seq:
            if not self.move_cubic(tcp, grip, dur):
                log.warning("pick abort at %s", name)
                return False
        if capture_dir is not None:
            obs = self.io.observe()
            for c in ("top", "wrist", "front"):
                V.save_rgb(Path(capture_dir) / f"grasp_{c}.png", np.ascontiguousarray(obs[c]))
        # close(arm 정지·grip 만 spline) → lift
        if not self.move_cubic((cx, cy, gz), self.cfg.grip_close, self.cfg.seg_dur_act):
            return False
        return self.move_cubic((cx, cy, up), self.cfg.grip_close, self.cfg.seg_dur_act)

    def _place_at(self, xy, place_z) -> bool:
        """수직 place: above(closed)→descend(closed)→release(open)→retract(open). assignment8 place_block."""
        tx, ty = xy
        pz = max(place_z, self.cfg.z_floor)
        up = pz + self.cfg.stack_clear
        if not self.move_cubic((tx, ty, up), self.cfg.grip_close, self.cfg.seg_dur_move):
            return False
        if not self.move_cubic((tx, ty, pz), self.cfg.grip_close, self.cfg.seg_dur_act):
            return False
        if not self.move_cubic((tx, ty, pz), self.cfg.grip_open, self.cfg.seg_dur_act):
            return False
        return self.move_cubic((tx, ty, up), self.cfg.grip_open, self.cfg.seg_dur_act)

    def stack_sequence(self, ordered, base_size, pick_size, n_blocks, capture_dir=None) -> bool:
        """좌→우 정렬 큐브 ordered=[(Blob,(x,y))] (전체 검출). **전역 최우측=바닥 고정**,
        좌측 (n_blocks-1)개를 좌→우 순으로 그 위에 적재.

        place 중심 z = (현 스택 top 표면) + pick_size/2. 바닥은 table 위 → top0=base_size."""
        base_xy = ordered[-1][1]                                   # 전체 최우측 = 바닥
        picks = [c[1] for c in ordered[: max(0, n_blocks - 1)]]    # 좌측 n-1 = 들어올릴 것
        running_top = base_size  # 현 스택 최상면 높이(table=0)
        for i, cube_xy in enumerate(picks):
            log.info("STACK %d/%d pick=(%.3f,%.3f) → base=(%.3f,%.3f) top=%.3f",
                     i + 1, len(picks), cube_xy[0], cube_xy[1], base_xy[0], base_xy[1], running_top)
            if not self._pick_at(cube_xy, pick_size / 2 + self.cfg.grasp_z_margin,
                                 capture_dir=capture_dir if i == 0 else None):
                return False
            if not self._place_at(base_xy, running_top + pick_size / 2):
                return False
            running_top += pick_size
        return True

    def unstack_sequence(self, base_xy, base_size, pick_size, n_placed, targets) -> bool:
        """스택 위에서부터 n_placed 개를 꺼내 targets 로 산포(reset). 바닥(base)은 남김."""
        running_top = base_size + n_placed * pick_size
        for i in range(n_placed):
            top_center = running_top - pick_size / 2
            tgt = targets[i] if i < len(targets) else targets[-1]
            log.info("UNSTACK %d/%d from top_z=%.3f → (%.3f,%.3f)",
                     i + 1, n_placed, top_center, tgt[0], tgt[1])
            if not self._pick_at(base_xy, top_center):
                return False
            if not self._place_at(tgt, pick_size / 2):
                return False
            running_top -= pick_size
        return True

    # ── reset: 그릇에서 큐브 추출 → 책상 무작위 산포 ──────────────────────
    def extract_one(self, bowl_xy, target_xy) -> bool:
        bx, by = bowl_xy
        tx, ty = target_xy
        if not self.move_to((bx, by, self.cfg.bowl_clear_z), self.cfg.grip_open):
            return False
        if not self.move_to((bx, by, self.cfg.grasp_z + 0.01), self.cfg.grip_open):
            return False
        self.set_gripper(self.cfg.grip_close, (bx, by, self.cfg.grasp_z + 0.01))
        if not self.move_to((bx, by, self.cfg.bowl_clear_z), self.cfg.grip_close):
            return False
        if not self.move_to((tx, ty, self.cfg.safe_z), self.cfg.grip_close):
            return False
        if not self.move_to((tx, ty, self.cfg.grasp_z), self.cfg.grip_close):
            return False
        self.set_gripper(self.cfg.grip_open, (tx, ty, self.cfg.grasp_z))
        self.move_to((tx, ty, self.cfg.safe_z), self.cfg.grip_open)
        return True

    def random_scatter_targets(self, n: int) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        tries = 0
        while len(out) < n and tries < 500:
            tries += 1
            x = random.uniform(*self.cfg.scatter_x)
            y = random.uniform(*self.cfg.scatter_y)
            if not self.in_workspace((x, y, self.cfg.grasp_z)):
                continue
            if all(math.hypot(x - px, y - py) >= self.cfg.scatter_sep for px, py in out):
                out.append((x, y))
        return out


# ── 비전 헬퍼 (성공/상태 판정) ───────────────────────────────────────────────
def snapshot(io: ArmIO):
    obs = io.observe()
    top = np.ascontiguousarray(obs["top"])
    cubes = V.detect_gray_cubes(top)
    bowl = V.detect_blue_bowl(top)
    in_bowl = V.detect_cubes_in_bowl(top, bowl) if bowl else []
    return obs, top, cubes, bowl, in_bowl


# ── 스테이지 ──────────────────────────────────────────────────────────────────
def stage_check(cfg: Cfg) -> int:
    io = ArmIO(cfg)
    io.connect()
    try:
        obs = io.observe()
        state = io.state_vec(obs)
        log.info("joint state [pan,lift,elbow,wflex,wroll,grip]=%s",
                 [round(float(v), 2) for v in state])
        d = cfg.debug_root / "check"
        for c in ("top", "wrist", "front"):
            V.save_rgb(d / f"{c}.png", np.ascontiguousarray(obs[c]))
        # top 검출 오버레이
        top = np.ascontiguousarray(obs["top"])
        cubes = V.detect_gray_cubes(top)
        bowl = V.detect_blue_bowl(top)
        V.save_debug_overlay(d / "top_overlay.png", top, cubes=cubes, bowl=bowl,
                             label=f"cubes={len(cubes)} bowl={'Y' if bowl else 'N'}")
        log.info("CHECK: cubes=%d bowl=%s  frames saved → %s",
                 len(cubes), bool(bowl), d)
        return 0
    finally:
        io.disconnect()


def _move_joints_deg(io: ArmIO, target6: list[float], tol: float = 3.0, timeout: float = 9.0):
    """joint-space 이동(IK 없이). 각 관절 deg 목표로 수렴까지 명령. (ok, 측정6vec) 반환."""
    dt = 1.0 / 30.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        now = time.perf_counter()
        obs = io.observe()
        cur = io.state_vec(obs)
        if max(abs(float(cur[i]) - target6[i]) for i in range(5)) < tol:
            return True, cur
        act = {f"{ARM[i]}.pos": float(target6[i]) for i in range(5)}
        act["gripper.pos"] = float(target6[5])
        io.send(act)
        time.sleep(max(0.0, dt - (time.perf_counter() - now)))
    return False, io.state_vec(io.observe())


def stage_jointcheck(cfg: Cfg) -> int:
    """조명 확인 → 밝으면 각 관절 ±30° wiggle 로 추종 점검, 어두우면 토크 풀고 대기."""
    io = ArmIO(cfg)
    io.connect()
    try:
        obs = io.observe()
        top = np.ascontiguousarray(obs["top"])
        vmean, vmed = V.mean_brightness(top)
        V.save_rgb(cfg.debug_root / "jointcheck" / "top.png", top)
        log.info("brightness Vmean=%.0f Vmedian=%.0f (bright≈120 / dark≈25)", vmean, vmed)
        if vmean < 60:
            log.info("DARK (Vmean %.0f < 60) — 불 꺼짐 판정. 토크 풀고 대기.", vmean)
            return 0  # finally 의 disconnect 가 disable_torque_on_disconnect=True 로 토크 해제
        # 밝음: 안전하게 elevated observe pose 로 올린 뒤 관절별 ±30° 점검
        ctrl = Controller(io, cfg)
        ctrl.go_observe()
        start = io.state_vec(io.observe())
        arm0 = [float(v) for v in start[:5]]
        grip = float(start[5])
        log.info("wiggle 시작자세(deg)=%s", [round(v, 1) for v in arm0])
        lim = [(math.degrees(lo), math.degrees(hi)) for lo, hi in SO101Kinematics.JOINT_LIMITS]
        all_ok = True
        for i in range(5):
            for delta in (-30.0, 30.0, 0.0):
                lo, hi = lim[i]
                val = max(lo + 2, min(hi - 2, arm0[i] + delta))
                tgt = arm0 + [grip]
                tgt[i] = val
                ok, meas = _move_joints_deg(io, tgt)
                err = abs(float(meas[i]) - val)
                good = ok and err < 5.0
                all_ok = all_ok and good
                log.info("  %-13s %+5.0f° → cmd=%.1f meas=%.1f err=%.1f %s",
                         ARM[i], delta, val, float(meas[i]), err, "OK" if good else "SLOW/OFF")
        ctrl.go_observe()
        log.info("JOINTCHECK %s", "ALL JOINTS OK" if all_ok else "일부 관절 추종 불량")
        return 0 if all_ok else 7
    finally:
        io.disconnect()


def stage_touchcalib(cfg: Cfg, mode: str) -> int:
    """사용자 2-step 터치 캘리브 (마커 책상에 정적, 점당 detect→touch):
      detect : 그리퍼 비킨 채 마커 픽셀 검출 → pending 저장 (가림 없음).
      touch  : 그리퍼 grasp-point 를 마커 중심에 댄 채 FK 읽어 pending 픽셀과 1쌍 기록.
      finish : 수집 쌍으로 homography 계산.
    마커는 두 step 사이 움직이지 않게(책상 고정). 점마다 위치 바꿔 ~6회."""
    pairs_path = cfg.dataset_root / "touch_pairs.json"
    pend_path = cfg.dataset_root / "touch_pending.json"
    cfg.dataset_root.mkdir(parents=True, exist_ok=True)
    pairs = json.loads(pairs_path.read_text(encoding="utf-8")) if pairs_path.exists() else []

    if mode == "finish":
        if len(pairs) < 4:
            log.error("need >=4 pairs, have %d", len(pairs))
            return 2
        pix = [tuple(p["pixel"]) for p in pairs]
        xy = [tuple(p["xy"]) for p in pairs]
        H, resid = V.compute_homography(pix, xy)
        out = cfg.dataset_root / "calibration.json"
        V.save_homography(out, H, pairs, resid, extra={"method": "touch", "n_pairs": len(pairs)})
        med = float(np.median(resid)) if resid else 9.9
        inl = sum(1 for r in resid if r < 0.012)
        log.info("TOUCH CALIB done: %d pairs median=%.4fm max=%.4fm inliers(<1.2cm)=%d → %s",
                 len(pairs), med, max(resid) if resid else 0, inl, out)
        return 0 if (med < 0.015 and H is not None) else 3

    cfg.cam_warmup_s = 1
    io = ArmIO(cfg)
    io.connect()
    try:
        d = cfg.debug_root / "touch"
        obs = io.observe()
        top = np.ascontiguousarray(obs["top"])
        if mode == "detect":
            pix = V.detect_aruco(top, cfg.aruco_id if cfg.aruco_id >= 0 else None)
            V.save_debug_overlay(d / f"detect{len(pairs):02d}.png", top, gripper_tip=pix,
                                 label=f"detect #{len(pairs)} px={pix}")
            if pix is None:
                log.warning("NO marker — 그리퍼 비키고 마커 평평·전체 보이게. 재시도.")
                return 4
            pend_path.write_text(json.dumps({"pixel": [int(pix[0]), int(pix[1])]}), encoding="utf-8")
            log.info("DETECT ok: marker_px=%s → 이제 그리퍼를 마커 중심에 대고 touch 실행", pix)
            return 0
        # touch
        if not pend_path.exists():
            log.error("pending 없음 — 먼저 --detect 실행")
            return 2
        pix = json.loads(pend_path.read_text(encoding="utf-8"))["pixel"]
        fk = SO101Kinematics().fk_tcp(io.arm_rad(obs))
        pairs.append({"pixel": [int(pix[0]), int(pix[1])],
                      "xy": [round(float(fk[0]), 4), round(float(fk[1]), 4)],
                      "fk": [round(float(v), 4) for v in fk]})
        pairs_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
        pend_path.unlink()
        log.info("PAIR %d recorded: marker_px=%s gripper_fk_xy=(%.3f,%.3f) z=%.3f",
                 len(pairs), pix, fk[0], fk[1], fk[2])
        return 0
    finally:
        io.disconnect()


def stage_arucosweep(cfg: Cfg) -> int:
    """고정 XY 에서 wrist pitch 를 쓸어 top 카메라가 ArUco 마커를 읽는 자세 탐색."""
    import cv2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    kin = ctrl.kin
    d = cfg.debug_root / "arucosweep"
    try:
        found = []
        # forward-extended pose 들: 큰 x + 얕은(가능한 가장 얕은) pitch → gripper 가 전방(카메라) 향함
        poses = []
        for x in (0.22, 0.25, 0.28, 0.30):
            for pdeg in (-10, -25, -40, -60):  # 얕은 순(전방) 우선
                q = kin.ik((x, 0.0, 0.10), 0.0, pitch=math.radians(pdeg))
                if q is not None:
                    poses.append((f"x{x:.2f}_p{pdeg}", q))
                    break  # 그 x 에서 가장 얕은 reachable
        for label, q in poses:
            tgt = [math.degrees(v) for v in q] + [cfg.grip_close]
            ok, _m = _move_joints_deg(io, tgt, tol=5.0, timeout=14.0)
            time.sleep(0.8)
            top = np.ascontiguousarray(io.observe()["top"])
            gray = cv2.cvtColor(cv2.cvtColor(top, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = V._aruco_detector().detectMarkers(gray)
            det = ids is not None and len(ids) > 0
            sz = int(max(corners[0][0][:, 0].ptp(), corners[0][0][:, 1].ptp())) if det else 0
            V.save_rgb(d / f"{label}.png", top)
            log.info("%s move_ok=%s aruco=%s size=%dpx", label, ok, "YES" if det else "no", sz)
            if det:
                found.append((label, sz))
        log.info("SWEEP detected: %s", found)
        return 0
    finally:
        io.disconnect()


def stage_arucotest(cfg: Cfg) -> int:
    """그리퍼 ArUco 마커 검출 확인: 가시 pose 로 이동→top 캡처→마커 검출·크기·위치 로그+오버레이."""
    import cv2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    try:
        ctrl.move_to((0.21, 0.0, 0.12), cfg.grip_close, timeout=12.0)  # 중앙 mid-height, 책상 클리어
        time.sleep(0.8)
        obs = io.observe()
        top = np.ascontiguousarray(obs["top"])
        d = cfg.debug_root / "aruco"
        V.save_rgb(d / "raw.png", top)
        bgr = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = V._aruco_detector().detectMarkers(gray)
        vis = bgr.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            for i, mid in enumerate(ids.flatten()):
                c = corners[i][0]
                w = float(c[:, 0].max() - c[:, 0].min())
                h = float(c[:, 1].max() - c[:, 1].min())
                cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
                log.info("ArUco id=%d center=(%.0f,%.0f) size=%.0fx%.0f px (frame 640x480)",
                         int(mid), cx, cy, w, h)
        else:
            log.warning("NO ArUco detected (check print/dictionary DICT_4X4_50/lighting)")
        cv2.imwrite(str(d / "overlay.png"), vis)
        log.info("aruco test frames → %s", d)
        return 0
    finally:
        io.disconnect()


def stage_jointreach(cfg: Cfg, bowl_xy: tuple) -> int:
    """IK 의 top-down pitch 제약 무시. joint-space 로 그릇 방위각 향해 **최대 reach** 자세를
    탐색해 직접 명령 → 실제로 닿는지 사진 확인(사용자: 수동 조작하면 닿음)."""
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    kin = ctrl.kin
    PAN = kin.PAN_X
    lim = kin.JOINT_LIMITS
    try:
        # 그릇 방위각 → pan (q1 = -atan2(dy,dx), base -z 규약)
        q1 = -math.atan2(bowl_xy[1], bowl_xy[0] - PAN)
        q1 = max(lim[0][0] + 0.03, min(lim[0][1] - 0.03, q1))
        # q2,q3,q4 격자 탐색: table 높이(0~0.18m)서 base 반경 r 최대화
        best = None
        st = math.radians(6)
        q2 = lim[1][0]
        while q2 <= lim[1][1]:
            q3 = lim[2][0]
            while q3 <= lim[2][1]:
                q4 = lim[3][0]
                while q4 <= lim[3][1]:
                    x, y, z = kin.fk_tcp([q1, q2, q3, q4, 0.0])
                    if 0.0 <= z <= 0.18:
                        r = math.hypot(x - PAN, y)
                        if best is None or r > best[0]:
                            best = (r, [q1, q2, q3, q4, 0.0], (x, y, z))
                    q4 += st
                q3 += st
            q2 += st
        if best is None:
            log.error("no table-height config found")
            return 1
        r, q, tcp = best
        log.info("MAX-REACH toward bowl: r=%.3f TCP=(%.3f,%.3f,%.3f) (bowl r=%.3f)",
                 r, tcp[0], tcp[1], tcp[2], math.hypot(bowl_xy[0] - PAN, bowl_xy[1]))
        log.info("  joint target deg=%s", [round(math.degrees(v), 1) for v in q])
        q_deg = [math.degrees(v) for v in q]
        target6 = q_deg + [cfg.grip_open]
        ok, meas = _move_joints_deg(io, target6, tol=4.0, timeout=22.0)
        obs = io.observe()
        mnow = io.state_vec(obs)
        for i in range(5):
            log.info("  joint %-13s target=%6.1f meas=%6.1f diff=%6.1f", ARM[i], q_deg[i],
                     float(mnow[i]), float(mnow[i]) - q_deg[i])
        fk = kin.fk_tcp(io.arm_rad(obs))
        gap = math.hypot(bowl_xy[0] - fk[0], bowl_xy[1] - fk[1])
        log.info("JOINTREACH ok=%s achieved TCP=(%.3f,%.3f,%.3f) gap_to_bowl=%.1fcm",
                 ok, fk[0], fk[1], fk[2], gap * 100)
        d = cfg.debug_root / "jointreach"
        for c in ("top", "wrist", "front"):
            V.save_rgb(d / f"{c}.png", np.ascontiguousarray(obs[c]))
        top = np.ascontiguousarray(obs["top"])
        V.save_debug_overlay(d / "top_overlay.png", top, gripper_tip=V.detect_gripper_tip(top),
                             label=f"maxreach r={r:.2f} gap={gap*100:.0f}cm")
        log.info("JOINTREACH photos → %s", d)
        return 0
    finally:
        io.disconnect()


def stage_reachbowl(cfg: Cfg, bowl_xy: tuple) -> int:
    """팔을 그릇 방향으로 최대한 뻗어(reach 한계) 그릇에 닿는지 사진으로 확인(사용자 검증용)."""
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    cal = cfg.dataset_root / "calibration.json"
    ctrl.H = V.load_homography(cal) if cal.exists() else None
    kin = ctrl.kin
    PAN = kin.PAN_X
    try:
        dx, dy = bowl_xy[0] - PAN, bowl_xy[1]
        rxy = math.hypot(dx, dy)
        ux, uy = dx / rxy, dy / rxy
        log.info("bowl base=(%.3f,%.3f) r=%.3f → reach toward it", bowl_xy[0], bowl_xy[1], rxy)
        reached = None
        for rr in [0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30]:
            tcp = (PAN + rr * ux, rr * uy, 0.10)
            if kin.ik_reach(tcp, 0.0) is None:
                log.info("  r=%.2f IK unreachable", rr)
                continue
            log.info("  r=%.2f reachable → moving", rr)
            if ctrl.move_to(tcp, cfg.grip_open, timeout=16.0):
                reached = (tcp, rr)
                break
        time.sleep(1.0)
        obs = io.observe()
        fk = kin.fk_tcp(io.arm_rad(obs))
        gap = math.hypot(bowl_xy[0] - fk[0], bowl_xy[1] - fk[1])
        log.info("REACH: achieved TCP=(%.3f,%.3f,%.3f)  bowl=(%.3f,%.3f)  gap=%.3f m (%.1f cm)",
                 fk[0], fk[1], fk[2], bowl_xy[0], bowl_xy[1], gap, gap * 100)
        d = cfg.debug_root / "reachbowl"
        for c in ("top", "wrist", "front"):
            V.save_rgb(d / f"{c}.png", np.ascontiguousarray(obs[c]))
        # top 오버레이: 그리퍼 tip + 그릇 추정 픽셀(역투영) 표시
        top = np.ascontiguousarray(obs["top"])
        tip = V.detect_gripper_tip(top)
        V.save_debug_overlay(d / "top_overlay.png", top, gripper_tip=tip,
                             label=f"reach r={reached[1] if reached else 'NA'} gap={gap*100:.0f}cm")
        log.info("REACH photos → %s (gripper tip=%s)", d, tip)
        return 0
    finally:
        io.disconnect()


def stage_fold(cfg: Cfg) -> int:
    """팔을 높이 접어(park) 워크스페이스에서 비킨 뒤 3캠 전부 캡처 → top 뷰·카메라 매핑 재확인.

    사용자 요청: 팔이 top 카메라를 가려 워크스페이스가 안 보임 → 접고 다시 본다.
    """
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    try:
        obs0 = io.observe()
        log.info("pre-fold joint=%s", [round(float(v), 1) for v in io.state_vec(obs0)])
        # 높이 retract 후보 — 첫 도달해 채택 (팔을 위/뒤로 접어 책상에서 비킴)
        candidates = [(0.14, 0.0, 0.30), (0.17, 0.0, 0.28), (0.20, 0.0, 0.26),
                      (0.22, 0.0, 0.24), (0.16, 0.0, 0.26)]
        parked = False
        for tcp in candidates:
            sol = ctrl.kin.ik_reach(tcp, 0.0, q_ref=io.arm_rad(io.observe()))
            if sol is None:
                continue
            log.info("folding → TCP=%s", [round(v, 3) for v in tcp])
            if ctrl.move_to(tcp, cfg.grip_open, timeout=16.0):
                parked = True
                break
        if not parked:
            log.warning("fold: no parked pose reached; capturing as-is")
        time.sleep(1.0)
        # 3캠 전부 저장 + top/front 검출 오버레이
        obs = io.observe()
        d = cfg.debug_root / "fold"
        for c in ("top", "wrist", "front"):
            V.save_rgb(d / f"{c}.png", np.ascontiguousarray(obs[c]))
        for c in ("top", "front"):
            rgb = np.ascontiguousarray(obs[c])
            cubes = V.detect_gray_cubes(rgb)
            bowl = V.detect_blue_bowl(rgb)
            V.save_debug_overlay(d / f"{c}_overlay.png", rgb, cubes=cubes, bowl=bowl,
                                 label=f"{c} cubes={len(cubes)} bowl={'Y' if bowl else 'N'}")
        log.info("FOLD done. frames → %s (arm parked, workspace should be clear)", d)
        return 0
    finally:
        io.disconnect()


def stage_calibrate(cfg: Cfg) -> int:
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    kin = ctrl.kin
    try:
        # snake 그리드
        grid = []
        for i, x in enumerate(cfg.calib_xs):
            ys = cfg.calib_ys if i % 2 == 0 else tuple(reversed(cfg.calib_ys))
            for y in ys:
                grid.append((x, y))
        pairs_pixel, pairs_xy, pairs_log = [], [], []
        d = cfg.debug_root / "calibrate"
        for idx, (x, y) in enumerate(grid):
            tcp = (x, y, cfg.calib_z)
            # tol 완화(0.07≈4°)+stuck 여유: z=0.06 far pose 가 torque로 수° 못 미쳐도 pair 채택(마커 robust)
            ok = ctrl.move_to(tcp, cfg.grip_close, timeout=10.0, tol=0.07, stuck_secs=4.0)
            if not ok:
                pairs_log.append({"idx": idx, "target": [x, y], "status": "move_fail"})
                continue
            time.sleep(cfg.calib_settle_s)
            obs = io.observe()
            top = np.ascontiguousarray(obs["top"])
            tip = (V.detect_aruco(top, cfg.aruco_id if cfg.aruco_id >= 0 else None)
                   if cfg.use_aruco else V.detect_gripper_tip(top))
            q_now = io.arm_rad(obs)
            fk = kin.fk_tcp(q_now)
            V.save_debug_overlay(d / f"pose{idx:02d}.png", top, gripper_tip=tip,
                                 label=f"{idx} tip={tip} fk=({fk[0]:.3f},{fk[1]:.3f})")
            if tip is None:
                pairs_log.append({"idx": idx, "target": [x, y], "status": "no_tip"})
                continue
            pairs_pixel.append((float(tip[0]), float(tip[1])))
            pairs_xy.append((float(fk[0]), float(fk[1])))
            pairs_log.append({"idx": idx, "tip": list(tip),
                              "fk_xy": [round(fk[0], 4), round(fk[1], 4)], "status": "ok"})
            log.info("calib %d/%d tip=%s fk_xy=(%.3f,%.3f)", idx + 1, len(grid), tip, fk[0], fk[1])
        ctrl.home()
        H, resid = V.compute_homography(pairs_pixel, pairs_xy)
        out = cfg.dataset_root / "calibration.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        V.save_homography(out, H, pairs_log, resid,
                          extra={"calib_z": cfg.calib_z, "n_grid": len(grid)})
        if H is None:
            log.error("CALIB FAIL: pairs=%d (<4). H not computed.", len(pairs_pixel))
            return 2
        rms = float(np.sqrt(np.mean(np.square(resid))))
        med = float(np.median(resid))  # RANSAC outlier 에 강건 — H 자체는 inlier 적합
        inliers = sum(1 for r in resid if r < 0.01)
        log.info("CALIB done: pairs=%d  median=%.4f m  RMS=%.4f m  max=%.4f m  inliers(<1cm)=%d → %s",
                 len(pairs_pixel), med, rms, max(resid), inliers, out)
        if med > cfg.calib_resid_max_m or inliers < 6:
            log.error("CALIB quality low (median %.4f > %.4f or inliers %d < 6). Inspect/tune.",
                      med, cfg.calib_resid_max_m, inliers)
            return 3
        return 0
    finally:
        io.disconnect()


def _run_episode(ctrl: Controller, io: ArmIO, cfg: Cfg, tag: str, record: bool = False) -> bool:
    """1 에피소드: 관측 pose→큐브 검출→pick→bowl. 비전 성공판정. before/after 저장.

    record=True 면 grasp_and_place 구간만 기록(관측/검출 이동은 기록 제외 → 깨끗한 demo).
    """
    d = cfg.debug_root / "episodes"
    ctrl.recording = False
    ctrl.go_observe()  # 팔을 top 카메라 밖으로
    obs, top, cubes, bowl, in_bowl = snapshot(io)
    n_before, b_before = len(cubes), len(in_bowl)
    # 진단: 가드 전에 항상 관측 프레임(raw+overlay) 저장(검출 실패 원인 추적용)
    V.save_rgb(d / f"{tag}_raw.png", top)
    V.save_debug_overlay(d / f"{tag}_before.png", top, cubes=cubes, bowl=bowl,
                         label=f"{tag} table={n_before} bowl={'Y' if bowl else 'N'}")
    if bowl is None:
        log.warning("[%s] no bowl detected (see %s_before.png)", tag, tag)
        return False
    if not cubes:
        log.warning("[%s] no table cube detected (see %s_before.png)", tag, tag)
        return False
    target = cubes[0]
    cube_xy = ctrl.cube_xy((target.px, target.py))
    bowl_xy = ctrl.cube_xy((bowl.px, bowl.py))
    log.info("[%s] pick cube px=%s→base=(%.3f,%.3f) bowl=(%.3f,%.3f)",
             tag, (target.px, target.py), cube_xy[0], cube_xy[1], bowl_xy[0], bowl_xy[1])
    ctrl.recording = record
    ok_motion = ctrl.grasp_and_place(cube_xy, bowl_xy)
    ctrl.recording = False
    # 성공 판정 (vision)
    ctrl.go_observe()
    obs2, top2, cubes2, bowl2, in_bowl2 = snapshot(io)
    n_after, b_after = len(cubes2), len(in_bowl2)
    success = ok_motion and (n_after <= n_before - 1) and (b_after >= b_before + 1)
    V.save_debug_overlay(d / f"{tag}_after.png", top2, cubes=cubes2, bowl=bowl2,
                         label=f"{tag} table={n_after} bowl={b_after} {'OK' if success else 'FAIL'}")
    log.info("[%s] motion=%s table %d→%d bowl %d→%d ⇒ %s",
             tag, ok_motion, n_before, n_after, b_before, b_after, "SUCCESS" if success else "FAIL")
    return success


def _maybe_reset(ctrl: Controller, io: ArmIO, cfg: Cfg) -> None:
    """책상 큐브가 비면 그릇에서 추출해 무작위 산포(reset). 기록하지 않음."""
    rec = ctrl.recording
    ctrl.recording = False
    try:
        _, top, cubes, bowl, in_bowl = snapshot(io)
        if len(cubes) >= 1 or bowl is None or not in_bowl:
            return
        log.info("RESET: table empty, extracting %d cube(s) from bowl", len(in_bowl))
        targets = ctrl.random_scatter_targets(len(in_bowl))
        bowl_xy = ctrl.cube_xy((bowl.px, bowl.py))
        for i, tgt in enumerate(targets):
            # 매 추출 전 그릇 내부 재검출(움직임으로 위치 변함)
            _, _, _, bowl_now, in_bowl_now = snapshot(io)
            if bowl_now is None or not in_bowl_now:
                break
            bxy = ctrl.cube_xy((bowl_now.px, bowl_now.py))
            ok = ctrl.extract_one(bxy, tgt)
            log.info("RESET extract %d→(%.3f,%.3f) %s", i, tgt[0], tgt[1], "ok" if ok else "FAIL")
        ctrl.home()
    finally:
        ctrl.recording = rec


# ── 스택(ECE4560 assignment8) 에피소드 / 리셋 ───────────────────────────────────
def _stack_sizes(cfg: Cfg) -> tuple[float, float]:
    base = cfg.base_size if cfg.base_size is not None else cfg.cube_size
    pick = cfg.pick_size if cfg.pick_size is not None else cfg.cube_size
    return base, pick


def _detect_ordered_cubes(ctrl: Controller, io: ArmIO, cfg: Cfg):
    """관측 pose 에서 top 큐브 검출 → 좌→우 정렬 [(Blob,(x,y))] + (top RGB, raw cubes)."""
    obs = io.observe()
    top = np.ascontiguousarray(obs["top"])
    cubes = V.detect_gray_cubes(top)
    ordered = V.order_cubes_left_to_right(
        cubes, ctrl.H, offset_xy=(cfg.calib_offset_x, cfg.calib_offset_y))
    return ordered, top, cubes


def _run_stack_episode(ctrl: Controller, io: ArmIO, cfg: Cfg, tag: str, record: bool = False) -> dict:
    """1 스택 에피소드: 관측→좌→우 큐브 검출→최우측=바닥 고정·나머지 적재. 기록 옵션.

    반환 {success, base_xy, n_placed} — reset(unstack)에서 사용."""
    d = cfg.debug_root / "episodes"
    base_size, pick_size = _stack_sizes(cfg)
    ctrl.recording = False
    ctrl.go_observe()
    ordered, top, cubes = _detect_ordered_cubes(ctrl, io, cfg)
    n_before = len(cubes)
    V.save_rgb(d / f"{tag}_raw.png", top)
    V.save_debug_overlay(d / f"{tag}_before.png", top, cubes=cubes,
                         label=f"{tag} cubes={n_before} need>={cfg.num_blocks}")
    if n_before < cfg.num_blocks:
        log.warning("[%s] cubes=%d < num_blocks=%d — 스택 불가(검출/배치 점검)", tag, n_before, cfg.num_blocks)
        return {"success": False, "base_xy": None, "n_placed": 0}
    # 전역 최우측 = 바닥(고정), 좌측 (num_blocks-1)개 = 들어올릴 것
    base_xy = ordered[-1][1]
    n_placed = cfg.num_blocks - 1
    log.info("[%s] stack %d blocks; base(rightmost)=(%.3f,%.3f) picks=%s",
             tag, cfg.num_blocks, base_xy[0], base_xy[1],
             [(round(x, 3), round(y, 3)) for _, (x, y) in ordered[:n_placed]])
    ctrl.recording = record
    ok_motion = ctrl.stack_sequence(ordered, base_size, pick_size, cfg.num_blocks,
                                    capture_dir=(None if record else d))
    ctrl.recording = False
    ctrl.home()
    # 성공 판정(약): 모션 OK + table 큐브 수가 픽 수만큼 감소(적재로 blob 병합/사라짐).
    _, top2, cubes2 = _detect_ordered_cubes(ctrl, io, cfg)
    n_after = len(cubes2)
    success = ok_motion and (n_after <= n_before - n_placed)
    V.save_debug_overlay(d / f"{tag}_after.png", top2, cubes=cubes2,
                         label=f"{tag} cubes {n_before}->{n_after} {'OK' if success else 'FAIL'}")
    log.info("[%s] motion=%s table %d→%d ⇒ %s (※ count proxy — hold/after 프레임 병행 확인)",
             tag, ok_motion, n_before, n_after, "SUCCESS" if success else "FAIL")
    return {"success": success, "base_xy": base_xy, "n_placed": n_placed}


def _maybe_reset_stack(ctrl: Controller, io: ArmIO, cfg: Cfg, ep: dict) -> None:
    """다음 에피소드용 리셋: 쌓인 큐브를 풀어 산포. manual_reset 이면 사용자에게 재배치 요청."""
    if cfg.manual_reset:
        log.info("MANUAL RESET: 큐브를 좌→우로 다시 흩어 놓고 Enter (스택 %d개)...", cfg.num_blocks)
        try:
            input()
        except EOFError:
            time.sleep(2.0)
        return
    base_xy, n_placed = ep.get("base_xy"), ep.get("n_placed", 0)
    if not base_xy or n_placed <= 0:
        return
    base_size, pick_size = _stack_sizes(cfg)
    targets = ctrl.random_scatter_targets(n_placed)
    if len(targets) < n_placed:
        log.warning("RESET: scatter 타겟 부족(%d<%d) — 가능한 만큼만", len(targets), n_placed)
    rec = ctrl.recording
    ctrl.recording = False
    try:
        ok = ctrl.unstack_sequence(base_xy, base_size, pick_size, min(n_placed, len(targets)), targets)
        log.info("RESET unstack %s", "ok" if ok else "FAIL")
        ctrl.home()
    finally:
        ctrl.recording = rec


def stage_test(cfg: Cfg) -> int:
    """단일 grasp 테스트(비기록): 검출→pick→bowl→성공판정. calib_offset/grasp_z/grip 튜닝용."""
    cal = cfg.dataset_root / "calibration.json"
    if not cal.exists():
        log.error("calibration.json 없음 (%s). --stage calibrate 먼저.", cal)
        return 2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    ctrl.H = V.load_homography(cal)
    if ctrl.H is None:
        log.error("호모그래피 로드 실패")
        io.disconnect()
        return 2
    try:
        ok = _run_episode(ctrl, io, cfg, "test", record=False)
        log.info("TEST grasp %s (frames: outputs/real_debug/episodes/test_*.png)",
                 "SUCCESS" if ok else "FAIL")
        return 0 if ok else 4
    finally:
        io.disconnect()


def stage_pick(cfg: Cfg) -> int:
    """큐브 집기만 검증(그릇 place 없음): 검출→pick→lift→테이블 큐브 수 감소 확인 + 프레임."""
    cal = cfg.dataset_root / "calibration.json"
    if not cal.exists():
        log.error("calibration.json 없음. --stage calibrate 먼저.")
        return 2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    ctrl.H = V.load_homography(cal)
    try:
        d = cfg.debug_root / "pick"
        ctrl.go_observe()
        obs, top, cubes, bowl, in_bowl = snapshot(io)
        V.save_debug_overlay(d / "before.png", top, cubes=cubes, bowl=bowl,
                             label=f"table={len(cubes)}")
        if not cubes:
            log.warning("no cube detected")
            return 4
        n_before = len(cubes)
        # 신뢰 zone(r≤0.28) 내 **가장 가까운** 큐브 선택(먼 큐브 stall 회피)
        scored = []
        for c in cubes:
            xy = ctrl.cube_xy((c.px, c.py))
            r = math.hypot(xy[0] - ctrl.kin.PAN_X, xy[1])
            scored.append((r, c, xy))
        scored.sort(key=lambda t: t[0])
        log.info("cubes by r: %s", [(round(r, 3), (c.px, c.py)) for r, c, _ in scored])
        reachable = [t for t in scored if t[0] <= 0.28]
        if not reachable:
            log.warning("no cube within reliable reach (r<=0.28); nearest r=%.3f", scored[0][0])
            return 6
        r, target, cube_xy = reachable[0]
        log.info("pick NEAREST cube px=(%d,%d)→base=(%.3f,%.3f) r=%.3f",
                 target.px, target.py, cube_xy[0], cube_xy[1], r)
        ok = ctrl.pick(cube_xy, grasp_capture_dir=d)
        # 든 채 프레임(wrist 로 파지 확인) — grip 유지
        obs2 = io.observe()
        for c in ("top", "wrist", "front"):
            V.save_rgb(d / f"hold_{c}.png", np.ascontiguousarray(obs2[c]))
        # 테이블서 비켜 들어올린 뒤 테이블 큐브 수 재검출(grip 유지)
        ctrl.move_to(cfg.observe_tcp, cfg.grip_close)
        _, top3, cubes3, _, _ = snapshot(io)
        n_after = len(cubes3)
        V.save_debug_overlay(d / "after.png", top3, cubes=cubes3, label=f"table={n_after}")
        lifted = ok and n_after <= n_before - 1
        log.info("PICK motion=%s table %d→%d ⇒ %s (frames %s)",
                 ok, n_before, n_after, "LIFTED" if lifted else "MISS", d)
        # 큐브 놓아주기(아무데나 안전 release)
        ctrl.set_gripper(cfg.grip_open, cfg.observe_tcp, dwell_s=0.5)
        return 0 if lifted else 6
    finally:
        io.disconnect()


def stage_collect(cfg: Cfg, target: int, do_upload: bool) -> int:
    cal = cfg.dataset_root / "calibration.json"
    if not cal.exists():
        log.error("calibration.json 없음 (%s). --stage calibrate 먼저.", cal)
        return 2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    ctrl.H = V.load_homography(cal)
    if ctrl.H is None:
        log.error("호모그래피 로드 실패")
        io.disconnect()
        return 2
    try:
        ctrl.home()
        # ── test: 1 grasp (비기록) ──
        log.info(">>> TEST grasp")
        if not _run_episode(ctrl, io, cfg, "test"):
            log.error("TEST grasp FAILED — 캘리브/HSV/그리퍼값 점검 필요. 중단.")
            return 4
        _maybe_reset(ctrl, io, cfg)

        # ── verify + collect (기록) ──
        from lerobot_recorder import LeRobotV3DatasetWriter
        ctrl.writer = LeRobotV3DatasetWriter(cfg.dataset_root, overwrite=True, enable_videos=True)

        successes = 0
        attempts = 0
        max_attempts = max(target * 4, cfg.verify_eps * 4)
        verify_done = False
        verify_succ = 0
        while successes < target and attempts < max_attempts:
            attempts += 1
            tag = f"ep{attempts:03d}"
            ok = False
            for k in range(cfg.grasp_attempts):
                ok = _run_episode(ctrl, io, cfg, f"{tag}.{k}", record=True)
                if ok:
                    break
                ctrl.writer.commit_episode(success=False, task_name=TASK_NAME)  # 실패 시도 프레임 폐기
                ctrl.home()
            committed = ctrl.writer.commit_episode(success=ok, task_name=TASK_NAME)
            if committed:
                successes += 1
            log.info("attempt %d: %s (successes=%d/%d)", attempts,
                     "commit" if committed else "discard", successes, target)
            _maybe_reset(ctrl, io, cfg)

            # verify 게이트 (첫 verify_eps 시도 후)
            if not verify_done and attempts >= cfg.verify_eps:
                verify_succ = successes
                rate = verify_succ / attempts
                log.info("VERIFY gate: %d/%d = %.0f%%", verify_succ, attempts, rate * 100)
                if rate < cfg.verify_min_rate:
                    log.error("VERIFY 성공률 %.0f%% < %.0f%% — 중단+보고.",
                              rate * 100, cfg.verify_min_rate * 100)
                    summary = ctrl.writer.finalize(TASK_NAME)
                    log.info("partial dataset: %s", summary)
                    return 5
                verify_done = True

        summary = ctrl.writer.finalize(TASK_NAME)
        log.info("COLLECT done: successes=%d/%d attempts=%d  %s",
                 successes, target, attempts, summary)
        if do_upload and successes > 0:
            _upload(cfg)
        return 0 if successes >= target else 6
    finally:
        io.disconnect()


def stage_stack(cfg: Cfg) -> int:
    """스택 1회 검증(비기록): 좌→우 검출→최우측=바닥·나머지 적재→프레임. offset/사이즈/그립 튜닝용."""
    cal = cfg.dataset_root / "calibration.json"
    if not cal.exists():
        log.error("calibration.json 없음. --stage touchcalib 먼저.")
        return 2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    ctrl.H = V.load_homography(cal)
    if ctrl.H is None:
        log.error("호모그래피 로드 실패")
        io.disconnect()
        return 2
    try:
        base_size, pick_size = _stack_sizes(cfg)
        log.info("STACK test: num_blocks=%d base_size=%.3f pick_size=%.3f grip(o/c)=%.0f/%.0f",
                 cfg.num_blocks, base_size, pick_size, cfg.grip_open, cfg.grip_close)
        ep = _run_stack_episode(ctrl, io, cfg, "stacktest", record=False)
        # hold/after 프레임은 _run_stack_episode 가 episodes/ 에 저장(grasp_*, *_after)
        log.info("STACK test %s (frames: outputs/real_debug/episodes/stacktest_*.png)",
                 "SUCCESS" if ep["success"] else "FAIL")
        return 0 if ep["success"] else 4
    finally:
        io.disconnect()


def stage_stackcollect(cfg: Cfg, target: int, do_upload: bool) -> int:
    """ECE4560 assignment8 스택 데이터 수집: test→verify(게이트)→record N→finalize(+업로드).

    각 에피소드 = 좌→우 큐브를 최우측 바닥에 적재(cubic spline 궤적). reset=unstack 산포(또는 manual)."""
    cal = cfg.dataset_root / "calibration.json"
    if not cal.exists():
        log.error("calibration.json 없음. --stage touchcalib 먼저.")
        return 2
    io = ArmIO(cfg)
    io.connect()
    ctrl = Controller(io, cfg)
    ctrl.H = V.load_homography(cal)
    if ctrl.H is None:
        log.error("호모그래피 로드 실패")
        io.disconnect()
        return 2
    try:
        ctrl.home()
        log.info(">>> STACK TEST (비기록)")
        test_ep = _run_stack_episode(ctrl, io, cfg, "stacktest", record=False)
        if not test_ep["success"]:
            log.error("STACK TEST FAILED — 캘리브/사이즈/그립/offset 점검. 중단.")
            return 4
        _maybe_reset_stack(ctrl, io, cfg, test_ep)  # test 스택 풀고 다음 에피소드 준비

        # 스택 LeRobot 데이터는 calib(dataset_root)과 분리된 dir 에 기록(overwrite rmtree 로 calib 삭제 방지)
        from lerobot_recorder import LeRobotV3DatasetWriter
        ctrl.writer = LeRobotV3DatasetWriter(cfg.stack_dataset_root, overwrite=True, enable_videos=True)

        successes, attempts = 0, 0
        max_attempts = max(target * 4, cfg.verify_eps * 4)
        verify_done = False
        while successes < target and attempts < max_attempts:
            attempts += 1
            tag = f"ep{attempts:03d}"
            ep = {"success": False, "base_xy": None, "n_placed": 0}
            for k in range(cfg.grasp_attempts):
                ep = _run_stack_episode(ctrl, io, cfg, f"{tag}.{k}", record=True)
                if ep["success"]:
                    break
                ctrl.writer.commit_episode(success=False, task_name=STACK_TASK_NAME)  # 실패 프레임 폐기
                ctrl.home()
            committed = ctrl.writer.commit_episode(success=ep["success"], task_name=STACK_TASK_NAME)
            if committed:
                successes += 1
            log.info("attempt %d: %s (successes=%d/%d)", attempts,
                     "commit" if committed else "discard", successes, target)
            _maybe_reset_stack(ctrl, io, cfg, ep)

            if not verify_done and attempts >= cfg.verify_eps:
                rate = successes / attempts
                log.info("VERIFY gate: %d/%d = %.0f%%", successes, attempts, rate * 100)
                if rate < cfg.verify_min_rate:
                    log.error("VERIFY 성공률 %.0f%% < %.0f%% — 중단+보고.",
                              rate * 100, cfg.verify_min_rate * 100)
                    log.info("partial dataset: %s", ctrl.writer.finalize(STACK_TASK_NAME))
                    return 5
                verify_done = True

        summary = ctrl.writer.finalize(STACK_TASK_NAME)
        log.info("STACKCOLLECT done: successes=%d/%d attempts=%d  %s",
                 successes, target, attempts, summary)
        if do_upload and successes > 0:
            _upload(cfg, dataset_dir=cfg.stack_dataset_root, repo_name="so101_real_stack_cubes")
        return 0 if successes >= target else 6
    finally:
        io.disconnect()


def _upload(cfg: Cfg, dataset_dir: Path | None = None, repo_name: str = "so101_real_pick_cube") -> None:
    import subprocess
    hf_user = os.getenv("HF_USER", "")
    repo = f"{hf_user}/{repo_name}"
    ddir = dataset_dir or cfg.dataset_root
    cmd = [sys.executable, str(_SIM / "upload_to_huggingface.py"),
           "--dataset_dir", str(ddir), "--repo_id", repo]
    log.info("uploading %s → %s", ddir, repo)
    subprocess.run(cmd, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="SO-101 자율 pick-place 수집")
    ap.add_argument("--stage", choices=["check", "jointcheck", "arucotest", "arucosweep", "touchcalib", "reachbowl", "jointreach", "fold", "calibrate", "pick", "test", "collect", "stack", "stackcollect"], required=True)
    ap.add_argument("--finish", action="store_true", help="touchcalib: 수집된 쌍으로 homography 계산")
    ap.add_argument("--detect", action="store_true", help="touchcalib: 마커 픽셀 검출 step(그리퍼 비킴)")
    ap.add_argument("--bowl-x", type=float, default=0.357)
    ap.add_argument("--bowl-y", type=float, default=0.335)
    ap.add_argument("--aruco", action="store_true", help="그리퍼 ArUco 마커로 calib tip 검출")
    ap.add_argument("--aruco-id", type=int, default=-1, help="ArUco marker id(-1=첫 마커)")
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--dataset-root", type=Path, default=None, help="calibration.json 위치")
    ap.add_argument("--stack-dataset-root", type=Path, default=None, help="스택 LeRobot 데이터 출력 dir")
    ap.add_argument("--upload", action="store_true")
    # 자주 튜닝하는 값 오버라이드
    ap.add_argument("--grasp-z", type=float, default=None)
    ap.add_argument("--side-offset", type=float, default=None, help="side-approach 횡오프셋(0=직하강)")
    ap.add_argument("--grip-open", type=float, default=None)
    ap.add_argument("--grip-close", type=float, default=None)
    ap.add_argument("--calib-offset-x", type=float, default=None)
    ap.add_argument("--calib-offset-y", type=float, default=None)
    ap.add_argument("--max-rel-target", type=float, default=None)
    # 스택(assignment8) 오버라이드
    ap.add_argument("--cube-size", type=float, default=None, help="큐브 한 변(m). base/pick 양쪽 set")
    ap.add_argument("--base-size", type=float, default=None, help="바닥(최우측) 큐브 변(m). 이종 대응")
    ap.add_argument("--pick-size", type=float, default=None, help="들어올릴 큐브 변(m). 이종 대응")
    ap.add_argument("--num-blocks", type=int, default=None, help="스택 블록 수(part2=2/part3=3)")
    ap.add_argument("--stack-clear", type=float, default=None, help="접근/이탈 여유 높이(m)")
    ap.add_argument("--grasp-z-margin", type=float, default=None, help="grasp z = size/2 + margin")
    ap.add_argument("--seg-dur-move", type=float, default=None, help="큰 이동 cubic 길이(s)")
    ap.add_argument("--seg-dur-act", type=float, default=None, help="하강/그립/이탈 cubic 길이(s)")
    ap.add_argument("--manual-reset", action="store_true", help="리셋 시 사용자 재배치(Enter) 대기")
    args = ap.parse_args()

    cfg = Cfg()
    if args.dataset_root:
        cfg.dataset_root = args.dataset_root
    if args.stack_dataset_root:
        cfg.stack_dataset_root = args.stack_dataset_root
    for a, c in (("grasp_z", "grasp_z"), ("grip_open", "grip_open"), ("grip_close", "grip_close"),
                 ("calib_offset_x", "calib_offset_x"), ("calib_offset_y", "calib_offset_y"),
                 ("side_offset", "side_offset"), ("max_rel_target", "max_rel_target"),
                 ("cube_size", "cube_size"), ("base_size", "base_size"), ("pick_size", "pick_size"),
                 ("num_blocks", "num_blocks"), ("stack_clear", "stack_clear"),
                 ("grasp_z_margin", "grasp_z_margin"),
                 ("seg_dur_move", "seg_dur_move"), ("seg_dur_act", "seg_dur_act")):
        v = getattr(args, a)
        if v is not None:
            setattr(cfg, c, v)
    if args.manual_reset:
        cfg.manual_reset = True
    if args.aruco:
        cfg.use_aruco = True
    if args.aruco_id >= 0:
        cfg.aruco_id = args.aruco_id
    setup_logging(args.stage, cfg)
    log.info("cfg port=%s cams(top/wrist/front)=%d/%d/%d max_rel=%.1f grasp_z=%.3f grip(o/c)=%.0f/%.0f",
             cfg.robot_port, cfg.top_idx, cfg.wrist_idx, cfg.front_idx,
             cfg.max_rel_target, cfg.grasp_z, cfg.grip_open, cfg.grip_close)
    try:
        if args.stage == "check":
            return stage_check(cfg)
        if args.stage == "jointcheck":
            return stage_jointcheck(cfg)
        if args.stage == "arucotest":
            return stage_arucotest(cfg)
        if args.stage == "arucosweep":
            return stage_arucosweep(cfg)
        if args.stage == "touchcalib":
            mode = "finish" if args.finish else ("detect" if args.detect else "touch")
            return stage_touchcalib(cfg, mode)
        if args.stage == "reachbowl":
            return stage_reachbowl(cfg, (args.bowl_x, args.bowl_y))
        if args.stage == "jointreach":
            return stage_jointreach(cfg, (args.bowl_x, args.bowl_y))
        if args.stage == "fold":
            return stage_fold(cfg)
        if args.stage == "calibrate":
            return stage_calibrate(cfg)
        if args.stage == "pick":
            return stage_pick(cfg)
        if args.stage == "test":
            return stage_test(cfg)
        if args.stage == "stack":
            return stage_stack(cfg)
        if args.stage == "stackcollect":
            return stage_stackcollect(cfg, args.target, args.upload)
        return stage_collect(cfg, args.target, args.upload)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as e:
        log.exception("FATAL: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
