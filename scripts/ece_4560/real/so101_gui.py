#!/usr/bin/env python
# so101_gui.py
"""SO-101 실기기 실시간 제어 GUI (Windows native uv).

같은 폴더의 `so101_utils.py`(모터 버스)·`so101_kinematics.py`(FK/IK) 만 재사용한다.
mujoco/isaac 의존 없음. tkinter(표준 라이브러리) + cv2(lerobot[feetech] 번들) + lerobot
네이티브 데이터셋만 쓴다.

기능 (요구사항 1~5):
  1. 슬라이더로 각 joint·gripper 조작 (Live jog, slew-limit 안전).
  2. 현재 자세 read-out: 각 joint·gripper degree + ee-pose 6D euler [x,y,z, r,p,y].
  3. Pick & Place phase 시퀀스: phase 단위 입력(Joint-space 또는 Cartesian IK) + 실행.
  4. 종료 시 모터 토크 해제 (try/finally 로 보장).
  5. lerobot-record 식 카메라 동반 에피소드 녹화 (LeRobot v3 데이터셋).

스레드: Main(tkinter) / RobotWorker(버스 단독 소유, ~50Hz) / Camera(카메라당 1).
버스는 thread-safe 하지 않으므로 모든 bus.* 호출은 worker 스레드에서만 한다.

실행:
  set -a; source .env; set +a
  uv run python scripts/ece_4560/real/so101_gui.py --port COM8
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

# 같은 real 폴더 모듈 (수정 없이 재사용)
from so101_utils import load_calibration, setup_motors
import so101_kinematics as kin

# lerobot 네이티브 (공용 의존성) — 데이터셋 writer + 카메라
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)
from lerobot.utils.constants import ACTION, OBS_STR

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

JOINT_ORDER = kin.JOINT_ORDER  # 6 (arm 5 + gripper)
ARM_JOINTS = kin.ARM_JOINTS
LIMITS = kin.LIMITS

# 이미지/카메라 계약 (출처: src/so101_contract/feature_codec.py)
IMG_H, IMG_W, IMG_C = 480, 640, 3

# real_pick_place.py 의 6-phase 프리셋 (move_time, hold_time 포함).
PICK_PLACE_PRESET = [
    # (pan, lift, elbow, wrist_flex, wrist_roll, gripper, move_time, hold_time)
    (-45.0, 45.0, -45.0, 90.0, 0.0, 50.0, 2.0, 1.0),   # pick 위 그리퍼 열림
    (-45.0, 45.0, -45.0, 90.0, 0.0, 5.0, 1.5, 1.0),    # 파지(닫음)
    (-45.0, 0.0, 0.0, 90.0, 0.0, 5.0, 2.0, 0.0),       # 들어올림
    (45.0, 0.0, 0.0, 90.0, 0.0, 5.0, 2.5, 0.0),        # place 쪽 수평 이동
    (45.0, 45.0, -45.0, 90.0, 0.0, 5.0, 2.0, 1.0),     # place 하강
    (45.0, 45.0, -45.0, 90.0, 0.0, 50.0, 1.5, 1.0),    # 놓음(열림)
]


def _slew(cur: float, tgt: float, step: float) -> float:
    """cur 를 tgt 쪽으로 한 틱당 최대 step 만큼만 이동."""
    d = tgt - cur
    if d > step:
        d = step
    elif d < -step:
        d = -step
    return cur + d


# ---------------------------------------------------------------------------
# phase 시퀀스 JSON 직렬화 — 사람·기계 모두 읽기 쉬운 flat 포맷:
#   {"name": ..., "phases": [{joint6..., "move_time", "hold_time"}, ...]}
# GUI 저장/불러오기 + 헤드리스 --run-sequence 가 같은 포맷을 공유한다.
# ---------------------------------------------------------------------------
def phase_to_flat(p: dict) -> dict:
    d = {j: round(float(p["target"][j]), 3) for j in JOINT_ORDER}
    d["move_time"] = float(p["move_time"])
    d["hold_time"] = float(p["hold_time"])
    return d


def flat_to_phase(d: dict) -> dict:
    return {
        "target": {j: float(d[j]) for j in JOINT_ORDER},
        "move_time": float(d.get("move_time", 2.0)),
        "hold_time": float(d.get("hold_time", 0.0)),
    }


def load_sequence_file(path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["phases"] if isinstance(data, dict) else data
    return [flat_to_phase(r) for r in rows]


def save_sequence_file(path, phases, name=None) -> None:
    out = {"name": name or Path(path).stem, "phases": [phase_to_flat(p) for p in phases]}
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


# ===========================================================================
# Camera 스레드 — cv2.VideoCapture, 최신 프레임을 RGB uint8 (H,W,3) 로 버퍼링
# ===========================================================================
class CameraThread(threading.Thread):
    def __init__(self, name: str, index: int, w: int = IMG_W, h: int = IMG_H, fps: int = 30):
        super().__init__(daemon=True)
        self.name = name
        self.index = index
        self.w, self.h, self.fps = w, h, fps
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = np.zeros((h, w, 3), np.uint8)
        self.opened = False

    def run(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.opened = cap.isOpened()
        if not self.opened:
            cap.release()
            return
        period = 1.0 / max(self.fps, 1)
        while not self._stop.is_set():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.w or frame.shape[0] != self.h:
                    frame = cv2.resize(frame, (self.w, self.h))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest = rgb
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
        cap.release()

    def get_latest(self) -> np.ndarray:
        with self._lock:
            return self._latest.copy()

    def stop(self):
        self._stop.set()


# ===========================================================================
# Robot worker 스레드 — 버스 단독 소유. GUI 는 cmd_q 로 명령, snapshot 으로 read.
# ===========================================================================
class RobotWorker(threading.Thread):
    def __init__(self, port, robot_name, hz=50.0, slew_deg=2.0, gripper_slew=3.0):
        super().__init__(daemon=True)
        self.port = port
        self.robot_name = robot_name
        self.hz = hz
        self.slew_deg = slew_deg
        self.gripper_slew = gripper_slew

        self.cmd_q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.snapshot: dict | None = None
        self._quit = threading.Event()

        self.bus = None
        self._cmd: dict = {}          # 마지막으로 명령한 goal
        self._jog_target: dict = {}   # 슬라이더 목표
        self._home: dict = {}         # 시작 시 자세 (Home 복귀용)
        self._mode = "JOG"            # JOG | PHASE
        self._torque = True
        self._status = "starting"

        # phase stepper
        self._phase_plan: list = []
        self._phase_idx = 0
        self._phase_sub = "move"
        self._phase_t0 = 0.0
        self._phase_start: dict = {}

        # recording
        self.cameras: dict[str, CameraThread] = {}
        self.dataset = None
        self.ds_features: dict = {}
        self.recording = False
        self.task = ""
        self.rec_fps = 30
        self._last_rec = 0.0
        self.ep_count = 0
        self.frame_count = 0

    # ---- 명령 enqueue (GUI 스레드에서 호출) ----
    def send(self, kind, payload=None):
        self.cmd_q.put((kind, payload))

    def get_snapshot(self):
        with self._lock:
            return dict(self.snapshot) if self.snapshot else None

    def _set_status(self, msg):
        self._status = msg

    # ---- 메인 루프 ----
    def run(self):
        try:
            calib = load_calibration(self.robot_name)
            self.bus = setup_motors(calib, self.port)  # torque ON, POSITION mode, P=16
            present = self.bus.sync_read("Present_Position")
            self._cmd = dict(present)
            self._jog_target = dict(present)
            self._home = dict(present)
            self._set_status("connected")

            period = 1.0 / self.hz
            while not self._quit.is_set():
                t0 = time.monotonic()
                self._drain_commands()

                present = self.bus.sync_read("Present_Position")
                try:
                    ee = kin.ee_pose_from_joints(present)
                except Exception:
                    ee = np.zeros(6)

                if self._torque:
                    if self._mode == "JOG":
                        self._step_jog()
                    elif self._mode == "PHASE":
                        self._step_phase()
                    self.bus.sync_write("Goal_Position", self._cmd, normalize=True)

                if self.recording:
                    self._maybe_record(present)

                self._publish(present, ee)

                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        except Exception as e:  # noqa: BLE001
            self._set_status(f"ERROR: {e}")
            traceback.print_exc()
            self._publish(self._cmd or {}, np.zeros(6))
        finally:
            self._safe_shutdown()

    # ---- 명령 처리 ----
    def _drain_commands(self):
        while True:
            try:
                kind, payload = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                self._handle(kind, payload)
            except Exception as e:  # noqa: BLE001
                self._set_status(f"cmd '{kind}' 실패: {e}")
                traceback.print_exc()

    def _handle(self, kind, payload):
        if kind == "quit":
            self._quit.set()
        elif kind == "jog":
            self._jog_target = dict(payload)
        elif kind == "abort":
            self._mode = "JOG"
            self._jog_target = dict(self._cmd)
            self._phase_plan = []
            self._set_status("aborted")
        elif kind == "run_phases":
            self._start_phases(payload)
        elif kind == "home":
            self._start_phases([{"target": dict(self._home), "move_time": 2.5, "hold_time": 0.0}])
            self._set_status("homing")
        elif kind == "torque":
            self._set_torque(bool(payload))
        elif kind == "start_episode":
            self._start_episode(payload)
        elif kind == "stop_episode":
            self._stop_episode(save=True)

    def _set_torque(self, on: bool):
        if self.bus is None:
            return
        if on and not self._torque:
            present = self.bus.sync_read("Present_Position")
            self._cmd = dict(present)
            self._jog_target = dict(present)
            try:
                self.bus.enable_torque()
            except Exception:  # noqa: BLE001
                pass
            self._torque = True
            self._mode = "JOG"
            self._set_status("torque ON")
        elif not on and self._torque:
            self.bus.disable_torque()
            self._torque = False
            self._mode = "JOG"
            self._set_status("torque OFF (자유 이동)")

    # ---- phase stepper ----
    def _start_phases(self, plan):
        if not self._torque:
            self._set_status("phase 실행 불가: torque OFF")
            return
        self._phase_plan = [dict(p) for p in plan]
        if not self._phase_plan:
            return
        self._phase_idx = 0
        self._phase_sub = "move"
        self._phase_t0 = time.monotonic()
        self._phase_start = dict(self._cmd)
        self._mode = "PHASE"
        self._set_status(f"phase 1/{len(self._phase_plan)}")

    def _step_phase(self):
        ph = self._phase_plan[self._phase_idx]
        target = ph["target"]
        now = time.monotonic()
        elapsed = now - self._phase_t0
        if self._phase_sub == "move":
            mt = max(float(ph.get("move_time", 2.0)), 1e-3)
            a = min(elapsed / mt, 1.0)
            for j in JOINT_ORDER:
                self._cmd[j] = (1 - a) * self._phase_start[j] + a * float(target[j])
            if a >= 1.0:
                self._phase_sub = "hold"
                self._phase_t0 = now
        else:  # hold
            for j in JOINT_ORDER:
                self._cmd[j] = float(target[j])
            if elapsed >= float(ph.get("hold_time", 0.0)):
                self._phase_idx += 1
                if self._phase_idx >= len(self._phase_plan):
                    self._mode = "JOG"
                    self._jog_target = dict(self._cmd)
                    self._set_status("sequence done")
                else:
                    self._phase_sub = "move"
                    self._phase_t0 = now
                    self._phase_start = dict(self._cmd)
                    self._set_status(f"phase {self._phase_idx + 1}/{len(self._phase_plan)}")

    def _step_jog(self):
        for j in ARM_JOINTS:
            self._cmd[j] = _slew(self._cmd[j], self._jog_target.get(j, self._cmd[j]), self.slew_deg)
        self._cmd["gripper"] = _slew(
            self._cmd["gripper"], self._jog_target.get("gripper", self._cmd["gripper"]), self.gripper_slew
        )

    # ---- recording ----
    def _start_episode(self, cfg):
        self.task = cfg.get("task", "")  # task 는 에피소드별로 바뀌어도 됨
        if self.dataset is None:
            # 카메라·fps·feature schema 는 데이터셋 생성 시점에 고정 (이후 에피소드 동일)
            self.cameras = cfg.get("cameras", {})
            self.rec_fps = int(cfg.get("fps", 30))
            motor_ft = {f"{m}.pos": float for m in JOINT_ORDER}
            cam_ft = {name: (IMG_H, IMG_W, IMG_C) for name in self.cameras}
            obs_features = hw_to_dataset_features({**motor_ft, **cam_ft}, OBS_STR, use_video=True)
            action_features = hw_to_dataset_features(motor_ft, ACTION, use_video=True)
            self.ds_features = combine_feature_dicts(obs_features, action_features)
            self.dataset = LeRobotDataset.create(
                repo_id=cfg["repo_id"],
                fps=self.rec_fps,
                features=self.ds_features,
                root=cfg["root"],
                robot_type="so101_follower",
                use_videos=True,
                batch_encoding_size=1,
                vcodec="h264",  # Windows ffmpeg 호환 (libsvtav1 미보장; h264=software libx264)
            )
        self.frame_count = 0
        self._last_rec = 0.0
        self.recording = True
        self._set_status(f"REC ep {self.ep_count}")

    def _maybe_record(self, present):
        now = time.monotonic()
        if now - self._last_rec < 1.0 / max(self.rec_fps, 1):
            return
        self._last_rec = now
        obs_values = {f"{m}.pos": float(present[m]) for m in JOINT_ORDER}
        for name, cam in self.cameras.items():
            obs_values[name] = cam.get_latest()
        action_values = {f"{m}.pos": float(self._cmd[m]) for m in JOINT_ORDER}
        obs_frame = build_dataset_frame(self.ds_features, obs_values, OBS_STR)
        action_frame = build_dataset_frame(self.ds_features, action_values, ACTION)
        frame = {**obs_frame, **action_frame, "task": self.task}
        self.dataset.add_frame(frame)
        self.frame_count += 1

    def _stop_episode(self, save=True):
        if not self.recording:
            return
        self.recording = False
        if save and self.dataset is not None and self.frame_count > 0:
            try:
                self.dataset.save_episode(parallel_encoding=False)
                self.ep_count += 1
                self._set_status(f"episode {self.ep_count - 1} 저장 ({self.frame_count} frames)")
            except Exception as e:  # noqa: BLE001
                self._set_status(f"save_episode 실패: {e}")
                traceback.print_exc()

    # ---- snapshot publish ----
    def _publish(self, present, ee):
        with self._lock:
            self.snapshot = {
                "present": {k: float(v) for k, v in present.items()} if present else {},
                "cmd": dict(self._cmd),
                "ee": [float(x) for x in ee],
                "status": self._status,
                "mode": self._mode,
                "torque": self._torque,
                "recording": self.recording,
                "episodes": self.ep_count,
                "frames": self.frame_count,
            }

    # ---- 종료: 토크 해제 보장 ----
    def _safe_shutdown(self):
        try:
            if self.recording:
                self._stop_episode(save=True)
        finally:
            try:
                if self.bus is not None:
                    self.bus.disable_torque()
                    self._set_status("torque released (종료)")
            except Exception:  # noqa: BLE001
                traceback.print_exc()


# ===========================================================================
# Tkinter GUI
# ===========================================================================
class App:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.worker = RobotWorker(
            port=args.port,
            robot_name=args.robot_name,
            hz=args.hz,
            slew_deg=args.slew,
            gripper_slew=args.gripper_slew,
        )
        self.cameras: dict[str, CameraThread] = {}
        self.phases: list[dict] = []  # {target:{6}, move_time, hold_time}
        self._syncing = False
        self._sliders_initialized = False
        self._last_mode = None  # PHASE→JOG 전환 감지(슬라이더 재동기화)용

        root.title("SO-101 실시간 제어")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self._build_ui()

        self.worker.start()
        self.root.after(50, self._poll)

    # ---------- UI 빌드 ----------
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")

        # 좌: 슬라이더 + read-out + 토크/홈
        left = ttk.LabelFrame(outer, text="① 슬라이더 jog  /  ② read-out", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._build_sliders(left)

        # 우상: phase
        right = ttk.LabelFrame(outer, text="③ Pick & Place — phase 시퀀스", padding=6)
        right.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self._build_phase_panel(right)

        # 우하: 녹화
        rec = ttk.LabelFrame(outer, text="⑤ 카메라 동반 에피소드 녹화", padding=6)
        rec.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self._build_record_panel(rec)

        # 하단 좌: 제어 버튼
        ctrl = ttk.Frame(outer, padding=2)
        ctrl.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._build_control_panel(ctrl)

        # 상태바
        self.status_var = tk.StringVar(value="시작 중…")
        bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=4)
        bar.grid(row=1, column=0, sticky="ew")
        self.root.columnconfigure(0, weight=1)

    def _build_sliders(self, parent):
        self.present_vars = {}
        for i, j in enumerate(JOINT_ORDER):
            lo, hi = LIMITS[j]
            ttk.Label(parent, text=j, width=14).grid(row=i, column=0, sticky="w", pady=1)
            var = tk.DoubleVar(value=0.0)
            self.slider_vars[j] = var
            res = 1.0 if j == "gripper" else 0.5
            sc = tk.Scale(
                parent, from_=lo, to=hi, resolution=res, orient="horizontal",
                length=240, variable=var, command=lambda _v: self._on_slider(),
                showvalue=True,
            )
            sc.grid(row=i, column=1, sticky="ew", padx=4)
            pv = tk.StringVar(value="—")
            self.present_vars[j] = pv
            ttk.Label(parent, textvariable=pv, width=9, anchor="e").grid(row=i, column=2, sticky="e")

        ttk.Label(parent, text="현재값 (degree / gripper 0-100)", foreground="#555").grid(
            row=len(JOINT_ORDER), column=2, sticky="e"
        )
        # ee-pose read-out
        ee_fr = ttk.LabelFrame(parent, text="ee-pose 6D euler  [x y z (m)  roll pitch yaw (deg)]", padding=4)
        ee_fr.grid(row=len(JOINT_ORDER) + 1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.ee_var = tk.StringVar(value="—")
        ttk.Label(ee_fr, textvariable=self.ee_var, font=("Consolas", 10)).grid(row=0, column=0, sticky="w")

        ttk.Button(parent, text="슬라이더 ← 현재자세 동기화", command=self.sync_sliders).grid(
            row=len(JOINT_ORDER) + 2, column=0, columnspan=3, sticky="ew", pady=(6, 0)
        )

    def _build_control_panel(self, parent):
        self.torque_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="Torque ON (끄면 자유 이동)", variable=self.torque_var, command=self.on_torque
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Button(parent, text="Home (시작자세 복귀)", command=self.on_home).grid(
            row=1, column=0, sticky="ew", pady=2
        )
        ttk.Button(parent, text="종료 (토크 해제)", command=self.on_close).grid(
            row=2, column=0, sticky="ew", pady=2
        )
        self.mode_var = tk.StringVar(value="mode: —")
        ttk.Label(parent, textvariable=self.mode_var).grid(row=3, column=0, sticky="w", pady=(6, 0))

    def _build_phase_panel(self, parent):
        cols = ("idx", "pan", "lift", "elbow", "wflex", "wroll", "grip", "move", "hold")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", height=8)
        widths = (32, 48, 48, 48, 48, 48, 44, 44, 44)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.grid(row=0, column=0, columnspan=6, sticky="nsew", pady=2)
        # 선택 → 입력칸에 로드(편집), 더블클릭 → 해당 각도로 로봇 이동
        self.tree.bind("<<TreeviewSelect>>", self._on_phase_select)
        self.tree.bind("<Double-1>", self._on_phase_doubleclick)
        ttk.Label(
            parent, text="행 클릭=입력칸 로드·편집 / 더블클릭=해당 각도로 이동", foreground="#555"
        ).grid(row=4, column=0, columnspan=6, sticky="w")

        # Joint-space 입력
        jf = ttk.LabelFrame(parent, text="Joint-space 입력 (degree / gripper 0-100)", padding=4)
        jf.grid(row=1, column=0, columnspan=6, sticky="ew", pady=2)
        self.joint_entries = {}
        for i, j in enumerate(JOINT_ORDER):
            ttk.Label(jf, text=j, width=12).grid(row=i // 3, column=(i % 3) * 2, sticky="e")
            e = ttk.Entry(jf, width=8)
            e.insert(0, "0.0")
            e.grid(row=i // 3, column=(i % 3) * 2 + 1, sticky="w", padx=2, pady=1)
            self.joint_entries[j] = e
        ttk.Label(jf, text="move_t").grid(row=2, column=0, sticky="e")
        self.move_entry = ttk.Entry(jf, width=6)
        self.move_entry.insert(0, "2.0")
        self.move_entry.grid(row=2, column=1, sticky="w")
        ttk.Label(jf, text="hold_t").grid(row=2, column=2, sticky="e")
        self.hold_entry = ttk.Entry(jf, width=6)
        self.hold_entry.insert(0, "0.5")
        self.hold_entry.grid(row=2, column=3, sticky="w")
        ttk.Button(jf, text="현재자세→입력", command=self.fill_joints_from_current).grid(row=2, column=4)
        ttk.Button(jf, text="Phase 추가 (joints)", command=self.add_phase_joints).grid(row=2, column=5)
        ttk.Button(jf, text="선택 갱신", command=self.update_selected_phase).grid(row=2, column=6)

        # Cartesian IK 입력
        cf = ttk.LabelFrame(parent, text="Cartesian IK 입력 (x y z [m]  roll pitch yaw [deg]  grip)", padding=4)
        cf.grid(row=2, column=0, columnspan=6, sticky="ew", pady=2)
        self.cart_entries = {}
        labels = [("x", "0.20"), ("y", "0.00"), ("z", "0.10"), ("roll", "180"), ("pitch", "0"), ("yaw", "0"), ("grip", "5")]
        for i, (name, dv) in enumerate(labels):
            ttk.Label(cf, text=name, width=5).grid(row=i // 4, column=(i % 4) * 2, sticky="e")
            e = ttk.Entry(cf, width=8)
            e.insert(0, dv)
            e.grid(row=i // 4, column=(i % 4) * 2 + 1, sticky="w", padx=2, pady=1)
            self.cart_entries[name] = e
        ttk.Button(cf, text="IK 풀어 Phase 추가", command=self.add_phase_ik).grid(row=1, column=6, padx=4)

        # phase 제어 버튼
        bf = ttk.Frame(parent)
        bf.grid(row=3, column=0, columnspan=6, sticky="ew", pady=4)
        ttk.Button(bf, text="프리셋 로드", command=self.load_preset).grid(row=0, column=0, padx=2)
        ttk.Button(bf, text="선택 삭제", command=self.delete_phase).grid(row=0, column=1, padx=2)
        ttk.Button(bf, text="전체 삭제", command=self.clear_phases).grid(row=0, column=2, padx=2)
        ttk.Button(bf, text="▶ 선택 위치로 이동", command=self.move_to_selected_phase).grid(row=0, column=3, padx=2)
        ttk.Button(bf, text="▶ 시퀀스 실행", command=self.run_phases).grid(row=0, column=4, padx=2)
        ttk.Button(bf, text="■ Abort", command=self.abort_phases).grid(row=0, column=5, padx=2)
        ttk.Button(bf, text="시퀀스 저장…", command=self.save_sequence).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(bf, text="시퀀스 불러오기…", command=self.load_sequence).grid(row=1, column=1, padx=2, pady=2)

    def _build_record_panel(self, parent):
        self.rec_entries = {}
        rows = [
            ("저장 dir", "root", self.args.dataset_dir),
            ("repo_id", "repo_id", "local/so101_gui"),
            ("task", "task", "Pick the cube and place it in the bowl"),
            ("fps", "fps", str(self.args.fps)),
            ("cameras", "cameras", self.args.cameras),
        ]
        for i, (label, key, default) in enumerate(rows):
            ttk.Label(parent, text=label, width=10).grid(row=i, column=0, sticky="e", pady=1)
            e = ttk.Entry(parent, width=34)
            e.insert(0, default)
            e.grid(row=i, column=1, sticky="w", padx=2)
            self.rec_entries[key] = e
        ttk.Button(parent, text="dir…", command=self._browse_dir).grid(row=0, column=2, padx=2)
        ttk.Button(parent, text="카메라 열기", command=self.open_cameras).grid(row=5, column=0, pady=4)
        self.cam_status = tk.StringVar(value="카메라: 미연결")
        ttk.Label(parent, textvariable=self.cam_status, foreground="#555").grid(row=5, column=1, sticky="w")
        ttk.Button(parent, text="● Start Episode", command=self.start_episode).grid(row=6, column=0, pady=2)
        ttk.Button(parent, text="■ Stop Episode", command=self.stop_episode).grid(row=6, column=1, sticky="w")
        self.rec_status = tk.StringVar(value="episodes: 0   frames: 0")
        ttk.Label(parent, textvariable=self.rec_status).grid(row=7, column=0, columnspan=2, sticky="w")

    # ---------- 슬라이더 / read-out ----------
    def _on_slider(self):
        # 첫 동기화 전에는 슬라이더 기본값(0)으로 jog 보내지 않음 (급이동 방지)
        if self._syncing or not self._sliders_initialized:
            return
        target = {j: float(self.slider_vars[j].get()) for j in JOINT_ORDER}
        self.worker.send("jog", target)

    def sync_sliders(self):
        snap = self.worker.get_snapshot()
        if not snap or not snap.get("present"):
            return
        self._syncing = True
        try:
            for j in JOINT_ORDER:
                if j in snap["present"]:
                    self.slider_vars[j].set(round(snap["present"][j], 2))
        finally:
            self._syncing = False
        # 슬라이더=현재값으로 jog target 갱신(무동작)
        self.worker.send("jog", {j: float(self.slider_vars[j].get()) for j in JOINT_ORDER})

    # ---------- 제어 ----------
    def on_torque(self):
        self.worker.send("torque", self.torque_var.get())

    def on_home(self):
        if not messagebox.askokcancel("Home", "시작 자세로 복귀합니다. 주변 정리됐나요?"):
            return
        self.worker.send("home")

    # ---------- phase ----------
    def _read_joint_entries(self):
        return {j: float(self.joint_entries[j].get()) for j in JOINT_ORDER}

    def fill_joints_from_current(self):
        snap = self.worker.get_snapshot()
        if not snap or not snap.get("present"):
            return
        for j in JOINT_ORDER:
            self.joint_entries[j].delete(0, tk.END)
            self.joint_entries[j].insert(0, f"{snap['present'].get(j, 0.0):.2f}")

    def _append_phase(self, target, move_t, hold_t):
        self.phases.append({"target": dict(target), "move_time": move_t, "hold_time": hold_t})
        self._refresh_tree()

    def add_phase_joints(self):
        try:
            target = self._read_joint_entries()
            mt = float(self.move_entry.get())
            ht = float(self.hold_entry.get())
        except ValueError:
            messagebox.showerror("입력 오류", "joint/시간 값이 숫자가 아닙니다.")
            return
        self._append_phase(target, mt, ht)

    def add_phase_ik(self):
        try:
            xyz = [float(self.cart_entries[k].get()) for k in ("x", "y", "z")]
            rpy = [float(self.cart_entries[k].get()) for k in ("roll", "pitch", "yaw")]
            grip = float(self.cart_entries["grip"].get())
            mt = float(self.move_entry.get())
            ht = float(self.hold_entry.get())
        except ValueError:
            messagebox.showerror("입력 오류", "Cartesian 값이 숫자가 아닙니다.")
            return
        snap = self.worker.get_snapshot()
        seed = snap["present"] if snap and snap.get("present") else {j: 0.0 for j in JOINT_ORDER}
        joint, res_pos, res_rot, reachable = kin.solve_ik_dls(xyz, rpy, seed, gripper=grip)
        msg = f"IK 잔차: pos {res_pos:.1f} mm, orient {res_rot:.1f}° (5-DOF best-effort)"
        if not reachable:
            if not messagebox.askokcancel("도달 어려움", msg + "\n\nposition 잔차가 큽니다. 그래도 추가할까요?"):
                return
        # joint 입력칸에도 채워 확인 가능
        for j in JOINT_ORDER:
            self.joint_entries[j].delete(0, tk.END)
            self.joint_entries[j].insert(0, f"{joint[j]:.2f}")
        self._append_phase(joint, mt, ht)
        self.status_var.set(msg)

    def load_preset(self):
        self.phases = []
        for row in PICK_PLACE_PRESET:
            target = {j: row[i] for i, j in enumerate(JOINT_ORDER)}
            self._append_phase(target, row[6], row[7])

    def delete_phase(self):
        sel = self.tree.selection()
        if not sel:
            return
        idxs = sorted((self.tree.index(s) for s in sel), reverse=True)
        for i in idxs:
            del self.phases[i]
        self._refresh_tree()

    def clear_phases(self):
        self.phases = []
        self._refresh_tree()

    def save_sequence(self):
        if not self.phases:
            messagebox.showinfo("phase 없음", "저장할 phase 가 없습니다.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="sequence.json"
        )
        if not path:
            return
        save_sequence_file(path, self.phases)
        self.status_var.set(f"시퀀스 저장: {path}")

    def load_sequence(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.phases = load_sequence_file(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("불러오기 실패", str(e))
            return
        self._refresh_tree()
        self.status_var.set(f"시퀀스 불러옴 ({len(self.phases)} phase): {path}")

    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, p in enumerate(self.phases):
            t = p["target"]
            self.tree.insert(
                "", "end",
                values=(
                    i,
                    f"{t['shoulder_pan']:.1f}", f"{t['shoulder_lift']:.1f}", f"{t['elbow_flex']:.1f}",
                    f"{t['wrist_flex']:.1f}", f"{t['wrist_roll']:.1f}", f"{t['gripper']:.0f}",
                    f"{p['move_time']:.1f}", f"{p['hold_time']:.1f}",
                ),
            )

    def _selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        idx = self.tree.index(sel[0])
        return idx if 0 <= idx < len(self.phases) else None

    def _on_phase_select(self, event=None):
        # 선택한 phase 값을 입력칸에 로드 (편집용)
        idx = self._selected_index()
        if idx is None:
            return
        p = self.phases[idx]
        for j in JOINT_ORDER:
            self.joint_entries[j].delete(0, tk.END)
            self.joint_entries[j].insert(0, f"{p['target'][j]:.2f}")
        self.move_entry.delete(0, tk.END)
        self.move_entry.insert(0, f"{p['move_time']:.2f}")
        self.hold_entry.delete(0, tk.END)
        self.hold_entry.insert(0, f"{p['hold_time']:.2f}")

    def update_selected_phase(self):
        # 입력칸 값으로 선택 phase 덮어쓰기 (수정 반영)
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("선택 없음", "수정할 phase 를 먼저 선택하세요.")
            return
        try:
            target = self._read_joint_entries()
            mt = float(self.move_entry.get())
            ht = float(self.hold_entry.get())
        except ValueError:
            messagebox.showerror("입력 오류", "joint/시간 값이 숫자가 아닙니다.")
            return
        self.phases[idx] = {"target": dict(target), "move_time": mt, "hold_time": ht}
        self._refresh_tree()
        kids = self.tree.get_children()
        if idx < len(kids):
            self.tree.selection_set(kids[idx])

    def move_to_selected_phase(self):
        # 선택 phase 의 joint 각도로 로봇 이동 (단일 phase run)
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("선택 없음", "이동할 phase 를 먼저 선택하세요.")
            return
        if not self.torque_var.get():
            messagebox.showwarning("torque OFF", "torque 를 켜야 이동합니다.")
            return
        self.worker.send("run_phases", [dict(self.phases[idx])])

    def _on_phase_doubleclick(self, event=None):
        # 더블클릭 = 해당 phase 각도로 로봇 이동
        self.move_to_selected_phase()

    def run_phases(self):
        if not self.phases:
            messagebox.showinfo("phase 없음", "먼저 phase 를 추가하세요 (프리셋 로드 가능).")
            return
        if not self.torque_var.get():
            messagebox.showwarning("torque OFF", "torque 를 켜야 실행됩니다.")
            return
        if not messagebox.askokcancel("시퀀스 실행", f"{len(self.phases)} phase 를 실행합니다. 시작할까요?"):
            return
        self.worker.send("run_phases", [dict(p) for p in self.phases])

    def abort_phases(self):
        self.worker.send("abort")

    # ---------- 녹화 ----------
    def _browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.rec_entries["root"].delete(0, tk.END)
            self.rec_entries["root"].insert(0, d)

    def _parse_cameras(self):
        text = self.rec_entries["cameras"].get().strip()
        out = {}
        if not text:
            return out
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            name, _, idx = token.partition(":")
            out[name.strip()] = int(idx.strip())
        return out

    def open_cameras(self):
        for cam in self.cameras.values():
            cam.stop()
        self.cameras = {}
        try:
            spec = self._parse_cameras()
        except ValueError:
            messagebox.showerror("카메라 설정 오류", "형식: top:0,wrist:1,front:2")
            return
        opened, failed = [], []
        for name, idx in spec.items():
            cam = CameraThread(name, idx, fps=int(self.rec_entries["fps"].get() or 30))
            cam.start()
            time.sleep(0.3)
            self.cameras[name] = cam
            (opened if cam.opened else failed).append(f"{name}:{idx}")
        self.cam_status.set(f"열림 {opened}" + (f" / 실패 {failed}" if failed else ""))

    def start_episode(self):
        snap = self.worker.get_snapshot()
        if snap and snap.get("recording"):
            messagebox.showinfo("이미 녹화 중", "Stop Episode 후 다시 시작하세요.")
            return
        root = self.rec_entries["root"].get().strip()
        repo_id = self.rec_entries["repo_id"].get().strip()
        task = self.rec_entries["task"].get().strip()
        try:
            fps = int(self.rec_entries["fps"].get())
        except ValueError:
            messagebox.showerror("입력 오류", "fps 가 정수가 아닙니다.")
            return
        if not root or not repo_id:
            messagebox.showerror("입력 오류", "저장 dir 와 repo_id 가 필요합니다.")
            return
        if self.worker.dataset is None and Path(root).exists() and any(Path(root).iterdir()):
            messagebox.showerror("디렉터리 충돌", f"{root} 가 비어있지 않습니다. 새 경로를 쓰세요.")
            return
        self.worker.send("start_episode", {
            "root": root, "repo_id": repo_id, "task": task, "fps": fps,
            "cameras": dict(self.cameras),
        })

    def stop_episode(self):
        self.worker.send("stop_episode")

    # ---------- 폴링 ----------
    def _poll(self):
        snap = self.worker.get_snapshot()
        if snap:
            if not self._sliders_initialized and snap.get("present"):
                self.sync_sliders()
                self._sliders_initialized = True
            for j in JOINT_ORDER:
                v = snap["present"].get(j)
                self.present_vars[j].set("—" if v is None else f"{v:.2f}")
            ee = snap.get("ee", [0] * 6)
            self.ee_var.set(
                f"x={ee[0]:+.3f} y={ee[1]:+.3f} z={ee[2]:+.3f} | "
                f"r={ee[3]:+.1f} p={ee[4]:+.1f} y={ee[5]:+.1f}"
            )
            # PHASE→JOG 전환 시 슬라이더를 현재자세로 재동기화 (이동 후 stale jog 급이동 방지)
            mode = snap.get("mode")
            if self._last_mode == "PHASE" and mode == "JOG" and self._sliders_initialized:
                self.sync_sliders()
            self._last_mode = mode
            self.status_var.set(snap.get("status", ""))
            self.mode_var.set(f"mode: {mode}   torque: {'ON' if snap.get('torque') else 'OFF'}")
            self.rec_status.set(
                f"episodes: {snap.get('episodes', 0)}   frames: {snap.get('frames', 0)}"
                + ("   ● REC" if snap.get("recording") else "")
            )
            # GUI 토크 토글을 worker 상태와 동기화 (외부 변경 반영)
            if snap.get("torque") != self.torque_var.get():
                self.torque_var.set(bool(snap.get("torque")))
        self.root.after(50, self._poll)

    # ---------- 종료 ----------
    def on_close(self):
        self.worker.send("quit")
        for cam in self.cameras.values():
            cam.stop()
        # worker 가 토크 해제·에피소드 저장을 마치도록 대기
        self.worker.join(timeout=8.0)
        self.root.destroy()


def run_sequence_headless(args, phases):
    """GUI 없이 phase 시퀀스를 1회 실행하고 토크 해제 후 종료 (Claude/CLI 용)."""
    worker = RobotWorker(
        port=args.port, robot_name=args.robot_name, hz=args.hz,
        slew_deg=args.slew, gripper_slew=args.gripper_slew,
    )
    worker.start()
    # 연결 대기
    while True:
        snap = worker.get_snapshot()
        st = snap.get("status", "") if snap else ""
        if st.startswith("connected") or st.startswith("ERROR"):
            break
        time.sleep(0.1)
    snap = worker.get_snapshot()
    if snap and snap.get("status", "").startswith("ERROR"):
        print(f"[run-sequence] 연결 실패: {snap['status']}")
        worker.send("quit")
        worker.join(timeout=5.0)
        return
    print(f"[run-sequence] {len(phases)} phase 실행 시작")
    worker.send("run_phases", [dict(p) for p in phases])
    # PHASE 진입 후 JOG 복귀 = 완료
    seen_phase = False
    while True:
        snap = worker.get_snapshot()
        if snap:
            mode = snap.get("mode")
            if mode == "PHASE":
                seen_phase = True
            elif seen_phase and mode == "JOG":
                break
        time.sleep(0.1)
    print("[run-sequence] 완료 → 토크 해제")
    worker.send("quit")
    worker.join(timeout=8.0)


def main():
    p = argparse.ArgumentParser(description="SO-101 실시간 제어 GUI")
    p.add_argument("--port", default="COM8", help="SO-101 직렬 포트 (예: COM8)")
    p.add_argument("--robot-name", default="so101_robot", help="calibration JSON 이름")
    p.add_argument("--hz", type=float, default=50.0, help="제어 루프 주파수")
    p.add_argument("--slew", type=float, default=2.0, help="arm jog slew limit (deg/tick)")
    p.add_argument("--gripper-slew", type=float, default=3.0, help="gripper jog slew limit (unit/tick)")
    p.add_argument("--fps", type=int, default=30, help="녹화 fps")
    p.add_argument("--dataset-dir", default=str(Path.cwd() / "datasets" / "so101_gui_record"))
    p.add_argument("--cameras", default="top:0,wrist:1,front:2", help="name:index 콤마 구분")
    p.add_argument("--run-sequence", default=None, help="JSON 시퀀스 파일을 GUI 없이 1회 실행 후 종료")
    args = p.parse_args()

    if args.run_sequence:
        phases = load_sequence_file(args.run_sequence)
        run_sequence_headless(args, phases)
        return

    root = tk.Tk()
    App(root, args)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
