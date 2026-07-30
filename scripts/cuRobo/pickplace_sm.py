"""SO-101 pick-place state machine — Isaac side (ZMQ → cuRobo).

Minimal 2-process, pure-ZMQ pick-place. This is the **isaac-sim** half; the planner
half is ``curobo_batch_planner.py`` (curobo-datagen container).

    ┌─ curobo-datagen (Docker) ────┐         ┌─ isaac-sim (Docker) ─────────────┐
    │ curobo_batch_planner.py      │   ZMQ   │ pickplace_sm.py  (this file)     │
    │  cuRobo v0.8 collision-free  │◀──REQ───│  IsaacLab pick_cube env variant  │
    │  full pick-place planner     │───REP──▶│  reset→read cube/bowl→plan→       │
    │  REP  tcp://*:5599           │  :5599  │  env.step replay→success check   │
    └──────────────────────────────┘         └──────────────────────────────────┘

No ROS, no separate executor: we read cube/bowl poses (``env.scene``) and drive joints
(``env.step``) directly — the cuRobo planner is the only remote piece.

Multi-env (``--num_envs N``, 기본 1): N envs reset/plan/replay in **lockstep**. One batch
ZMQ request carries per-env cube/bowl/start; the planner returns per-env trajectories
(null = plan fail → that env holds init). Shorter trajectories pad with their last row;
success is judged per env. Viewer camera follows env 0.

## 서브커맨드 (mode)

    random  — 통상 랜덤 DR 배치. 인터랙티브(livestream 키) 또는 --auto_trials N 자동.
              +--record_hdf5 PATH = leisaac 방식 IsaacLab HDF5 녹화(auto 전용,
              --enable_cameras 필수; 에피소드 = 정지 2s→pick-place→init 복귀→정지 1s,
              termination 자동 종료·multi-env env당 1 demo). README §record 모드.
              +--record_lerobot DIR = LeRobot v3 직기록(leisaac --use_lerobot_recorder
              동형, single-env·성공만·스트리밍 — LeRobotV3DatasetWriter 백엔드).
    fail    — sweep 결과(--results)의 place/plan-fail 셀 좌표만 재현(인터랙티브).
    sweep   — DR 스폰영역 전체 grid+boundary 정량 평가 → JSON(--out). 자동/headless.

인터랙티브 키(livestream 필수, WebRTC 클라이언트로 전달):
  N = random: 새 DR 레이아웃 · fail: 다음 fail batch     (로봇 → init)
  R = random: 같은 레이아웃 · fail: 같은 batch           (로봇 → init)
  B = cuRobo plan 요청(ZMQ) + pick-place 실행
R/N (조작 중 포함) = 남은 로봇 동작 취소 + 재배치. Ctrl-C / 스트림 닫기로 종료.

실행(터미널 2개; 둘 다 network_mode: host 라 ZMQ localhost):

    # 1) planner  (curobo-datagen)
    docker compose -f docker/docker-compose.yaml run --rm curobo-datagen \
        python /workspace/scripts/cuRobo/curobo_batch_planner.py

    # 2) SM  (isaac-sim) — 예: 통상 랜덤, livestream 관전
    docker compose -f docker/docker-compose.yaml run --rm isaac-sim \
        python /workspace/scripts/cuRobo/pickplace_sm.py random \
        --task SimToReal-SO101-PickCube-DR-v0 --livestream 2

관전: WebRTC :49100 (원격 relay = .env 의 LIVESTREAM=1 + PUBLIC_IP).
Env variants(--task): PickCube-v0(고정) · -DR-v0 · -DRBase-v0 · -Eval-v0 · -DR-Eval-v0
(src/sim_to_real/tasks/pick_cube/__init__.py).
"""
import argparse

from isaaclab.app import AppLauncher

# ── CLI (subcommands) — AppLauncher needs a SimulationApp before ANY isaac import ──
# 공용 인자 + AppLauncher 인자는 부모 파서 하나에 정의하고, 각 서브커맨드가 상속한다.
_common = argparse.ArgumentParser(add_help=False)
_common.add_argument("--task", default="SimToReal-SO101-PickCube-DR-v0",
                     help="registered pick_cube env variant (tasks/pick_cube/__init__.py)")
_common.add_argument("--planner", default="tcp://127.0.0.1:5599", help="cuRobo planner ZMQ REQ endpoint")
_common.add_argument("--num_envs", type=int, default=1, help="parallel envs (lockstep plan+replay)")
_common.add_argument("--grasp_z", type=float, default=None,
                     help="grasp 조준 z 를 robot-base 프레임에서 직접 지정(m). 기본 None = "
                          "`TABLE_TOP_BASE + cube_half` 로 유도(grasp_geometry 단일 소스). "
                          "튜닝 실험용 override 이며 상시 사용하는 값이 아니다")
_common.add_argument("--cube_sizes", default=None,
                     help="큐브 크기 DR 후보를 이 목록으로 **덮어쓴다**(콤마 구분, m). "
                          "예 `0.025` = 전 env 25 mm 고정(크기별 성공률 진단용), "
                          "`0.025,0.040` = 두 크기만. 기본 None = env cfg 사다리(25~40 mm) 그대로. "
                          "크기 DR 이 없는 env(-DR 아닌 변형)에서는 사용 불가")
_common.add_argument("--grasp_retries", type=int, default=1,
                     help="한 번에 실패한 env 를 몇 번 더 재계획·재시도할지(기본 1). "
                          "grasp 실패로 큐브가 밀려나면 **입력 pose 가 바뀌므로** 결정적 planner 도 "
                          "다른 궤적을 낸다. 이미 성공한 env 는 init hold 로 고정해 건드리지 않는다. "
                          "0 = 옛 동작(1회 시도). record 모드는 에피소드 규격상 재시도하지 않는다")
_common.add_argument("--settle", type=int, default=5, help="physics steps to settle after each reset")
_common.add_argument("--bowl_tol", type=float, default=0.06,
                     help="success = cube-center within this xy radius of bowl center (m)")
_common.add_argument("--seed", type=int, default=0, help="base seed for reset/plan")
_common.add_argument("--plan_timeout_s", type=float, default=900.0,
                     help="planner 응답 대기 상한(초). 초과 시 해당 batch 를 plan-fail 로 기록하고 "
                          "소켓 재연결 후 진행한다(planner 사망 시 headless sweep 무한 정지 방지). "
                          "0 = 무제한")
_common.add_argument("--planner_knobs_json", default=None,
                     help="JSON object forwarded to cuRobo planner request.knobs")
_common.add_argument("--log_every", type=int, default=0,
                     help="print EE/cube state every N env steps. 0(기본) = 끔 — headless "
                          "sweep 은 step 마다 3줄이 수십만 줄 stdout 이 돼 스텝률을 갉아먹는다")
_common.add_argument("--cam_eye", type=float, nargs=3, default=[0.2, 0.8, 1.2],
                     help="viewport/livestream camera eye (env-relative)")
_common.add_argument("--cam_target", type=float, nargs=3, default=[0.0, 0.1, 0.7],
                     help="viewport/livestream camera lookat (env-relative)")
AppLauncher.add_app_launcher_args(_common)

parser = argparse.ArgumentParser(description="SO-101 pick-place SM (Isaac side, ZMQ→cuRobo)")
_sub = parser.add_subparsers(dest="mode", required=True)

_p_random = _sub.add_parser("random", parents=[_common], help="통상 랜덤 DR 배치(인터랙티브/자동)")
_p_random.add_argument("--auto_trials", type=int, default=0,
                       help="키 입력 없이 이만큼 랜덤 trial 실행 후 종료(0 = 인터랙티브)")
_p_random.add_argument("--record_viewport_dir", default=None,
                       help="auto-trial viewport MP4 디렉터리. 'none' = 비활성")
_p_random.add_argument("--summary_dir", default=None,
                       help="auto-trial summary.json 디렉터리(기본 record_viewport_dir 또는 scratch)")
_p_random.add_argument("--record_fps", type=int, default=30, help="viewport MP4 FPS")
_p_random.add_argument("--record_every", type=int, default=1, help="N step 마다 1 프레임 기록")
_p_random.add_argument("--record_hdf5", default=None,
                       help="IsaacLab HDF5 데이터셋 경로(datagen record 모드). --auto_trials>0 + "
                            "--enable_cameras 필수. 에피소드 = [정지 preroll_s] 이동 pick-place "
                            "init 복귀 [정지 posthold_s] 자동 종료(termination)")
_p_random.add_argument("--record_lerobot", default=None,
                       help="LeRobot v3 직기록 디렉터리(leisaac --use_lerobot_recorder 동형, "
                            "LeRobotV3DatasetWriter 스트리밍·성공만 저장). **single-env 전용** "
                            "(--num_envs 1) + --auto_trials + --enable_cameras. "
                            "⚠ 기존 디렉터리는 덮어씀(overwrite)")
_p_random.add_argument("--task_description",
                       default="pick up the cube and place it in the bowl",
                       help="record_lerobot: LeRobot task 문자열(계약 canonical 기본값)")
_p_random.add_argument("--preroll_s", type=float, default=1.0,
                       help="record: 이동 시작 전 정지 구간(초)")
_p_random.add_argument("--posthold_s", type=float, default=0.5,
                       help="record: init 복귀 후 정지 유지(초) — 종료 term 의 hold 길이")

_p_fail = _sub.add_parser("fail", parents=[_common], help="sweep 결과의 fail 좌표만 재현")
_p_fail.add_argument("--results", required=True,
                     help="sweep JSON 경로. place/plan-fail 셀 좌표만 batch 로 로드")
_p_fail.add_argument("--auto", action="store_true",
                     help="키 입력 없이 전 fail batch 자동 재현 + planned/placed 집계(headless)")

_p_sweep = _sub.add_parser("sweep", parents=[_common], help="스폰영역 grid+boundary 정량 평가")
_p_sweep.add_argument("--nx", type=int, default=15, help="interior grid x 해상도")
_p_sweep.add_argument("--ny", type=int, default=8, help="interior grid y 해상도")
_p_sweep.add_argument("--boundary_n", type=int, default=20,
                      help="경계(최외곽 bell/y·최내곽 base/bowl/exclude) 곡선당 샘플 수")
_p_sweep.add_argument("--trials", type=int, default=1,
                      help="셀당 반복 횟수(판정 노이즈 평균). >1 이면 --yaw random 권장")
_p_sweep.add_argument("--yaw", default="0", help="큐브 yaw(도) 고정값 또는 'random'")
_p_sweep.add_argument("--out", default="/workspace/outputs/curobo_sweep/sweep_results.json",
                      help="sweep 결과 JSON 경로(호스트 ./outputs 마운트)")
args = parser.parse_args()

# ⚠ AppLauncher gets a WHITELIST only (AGENTS.md): passing vars(args) whole feeds custom
#   args (--task/--cam_eye/…) into _prepare_ui and breaks livestream viewport docking.
_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

# ── isaac / project imports (only valid after SimulationApp exists) ──────────────
import faulthandler  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

faulthandler.enable()  # C-레벨 크래시 시 파이썬 스택 덤프

import carb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402
import gymnasium as gym  # noqa: E402
import sim_to_real  # noqa: F401,E402  (gym.register side effect)
from isaaclab.utils.math import (  # noqa: E402
    euler_xyz_from_quat, quat_from_euler_xyz, quat_mul, subtract_frame_transforms,
)
from isaaclab.managers import DatasetExportMode, SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from so101_contract.feature_codec import SO101_JOINT_ORDER, policy_feature_to_sim_joint_radians  # noqa: E402
from so101_contract.grasp_geometry import FIXED_INNER_CENTER as _FIXED_INNER_CENTER  # noqa: E402
from so101_contract.grasp_geometry import PAD_LOW_OFF, TABLE_TOP_BASE  # noqa: E402
from sim_to_real.utils.cube_specs import CUBE_HALF_EXTENTS  # noqa: E402
from sim_to_real.utils.domain_randomization import CUBE_SIZE_ATTR  # noqa: E402
from sim_to_real.tasks.common.mdp.recorders import SO101DatagenRecorderManagerCfg  # noqa: E402
from sim_to_real.tasks.pick_cube import spawn_area as SA  # noqa: E402
from sim_to_real.tasks.pick_cube.mdp import terminations as pc_term  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras  # noqa: E402

CUBE, BOWL = "Cube1", "Bowl"  # pick_cube leaf scene entity names (single cube + bowl)


def _force_done_term(env):
    """record 모드 전용 driver-제어 종료 — env._force_done_mask 를 그대로 반환.

    plan-fail env(이동 없음 → returned_home 미발화)를 트라이얼 끝에 강제 종료해
    auto-reset(새 DR 레이아웃 + episode_length_buf 리셋)시키는 용도. 드라이버가 mask 를
    세팅하고 1 step 후 해제한다.
    """
    mask = getattr(env, "_force_done_mask", None)
    if mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return mask
# jaw pad 기하 = so101_contract.grasp_geometry 단일 소스(planner 와 같은 값을 공유).
FIXED_INNER_CENTER = np.array(_FIXED_INNER_CENTER, dtype=np.float64)  # 진단 로그용
_CUBE_Z_DRIFT_TOL = 0.005  # 측정 z vs 유도 z 경고 문턱(m) — settle 잔진동(0.1 mm)보다 크게

# record 트라이얼의 종료 대기 상한. 정상 경로는 posthold 뒤 termination 이 즉시 발화하므로
# 여기까지 오면 이미 이상 상태 — env time_out(30 s) 에 넘기고 다음 트라이얼로 간다.
_RECORD_DRAIN_MAX_STEPS = 1200

# Robot start pose (degrees) — arm holds this while the cube settles, and it is the plan's
# start joint state. wrist_roll -90 = top-down-tilt-ready; gripper -10° = feature 0 (open).
INIT_POSE_DEG = {
    "shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0,
    "wrist_flex": 50.0, "wrist_roll": -90.0, "gripper": -10.0,
}
INIT_RAD = {j: math.radians(d) for j, d in INIT_POSE_DEG.items()}
INIT_ACTION = [INIT_RAD[j] for j in SO101_JOINT_ORDER]  # env action / planner start order (rad)


# ══ 포맷/수학 헬퍼 ════════════════════════════════════════════════════════════════
def _wrap180(rad):
    return (math.degrees(rad) + 180.0) % 360.0 - 180.0


def _quat_rotate_np(q, v):
    """wxyz quaternion 으로 vector 를 회전한다."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
    qv = np.array([x, y, z], dtype=np.float64)
    return v + 2.0 * np.cross(qv, w * v + np.cross(qv, v))


def _fmt_vec(v):
    return "(" + ",".join(f"{float(x):+.3f}" for x in v) + ")"


def _fmt_quat(q):
    return "(" + ",".join(f"{float(x):+.3f}" for x in q) + ")"


def _euler_deg_from_wxyz_np(q):
    qt = torch.tensor(np.asarray(q, dtype=np.float64), dtype=torch.float32).unsqueeze(0)
    return tuple(_wrap180(a[0].item()) for a in euler_xyz_from_quat(qt))


class ViewportVideoRecorder:
    """Active viewport RGB 를 step 마다 MP4 로 기록한다(random --auto_trials 전용)."""

    def __init__(self, out_dir, fps=30, every=1):
        self.out_dir = Path(out_dir) if out_dir else None
        self.fps = int(fps)
        self.every = max(1, int(every))
        self.writer = None
        self.path = None
        self.step_i = 0
        self._imageio = None
        self._annotator = None
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio  # noqa: PLC0415
        import omni.kit.viewport.utility as viewport_utils  # noqa: PLC0415
        import omni.replicator.core as rep  # noqa: PLC0415

        viewport = viewport_utils.get_active_viewport()
        if viewport is None or not viewport.render_product_path:
            raise RuntimeError(
                "active viewport render product 없음. --record_viewport_dir 사용 시 GUI 또는 --livestream 2로 실행하세요."
            )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(viewport.render_product_path)
        self._imageio = imageio
        self._annotator = annotator
        self.render_product_path = str(viewport.render_product_path)

    @property
    def enabled(self):
        return self.out_dir is not None

    def start(self, trial_idx):
        if not self.enabled:
            return None
        self.close()
        self.path = self.out_dir / f"trial_{trial_idx:03d}.mp4"
        self.step_i = 0
        self.writer = self._imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p"],
        )
        return str(self.path)

    def capture(self):
        if self.writer is None:
            return
        self.step_i += 1
        if self.step_i % self.every:
            return
        arr = np.asarray(self._annotator.get_data())
        if arr.size == 0:
            return
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            self.writer.append_data(arr.astype(np.uint8))

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


# ══ task-space reads (robot base frame, per-env) ══════════════════════════════════
def _cube_halves(env):
    """Per-env 큐브 반변(m) — 크기 DR(`randomize_cube_sizes`)이 기록한 **실제** 값.

    크기 DR 이 없는 env 변형에는 attribute 자체가 없으므로 authored 상수로 폴백한다.
    grasp 조준 z(`TABLE_TOP_BASE + half`)와 planner 의 face-center 계산이 같은 값을 써야
    하므로, 이 함수가 SM 쪽 단일 소스다.
    """
    sizes = (getattr(env, CUBE_SIZE_ATTR, None) or {}).get(CUBE)
    if sizes is None:
        return [CUBE_HALF_EXTENTS[CUBE]] * env.num_envs
    return [0.5 * float(v) for v in sizes.tolist()]


def _cubes_bowls_in_base(env):
    """Per-env cube (6D) + bowl (xy) in each robot's base_link frame — the planner input
    (it applies Rz(90)+BASE_T internally, see curobo_batch_planner.usd_to_urdf).
    Returns (cubes [N][x,y,grasp_z,qw,qx,qy,qz], bowls [N][x,y], halves [N]).
    planner extracts cube face normals directly from the quat.

    z 는 측정값이 아니라 **유도값**을 보낸다(안착 직후 측정 z 는 settle 잔진동으로 흔들린다).
    유도 규칙 = ``TABLE_TOP_BASE + cube_half`` — 책상 상판 위에 큐브가 놓인다는 기하 그대로다.
    상판은 `grasp_geometry.TABLE_TOP_BASE`(실측) 단일 소스이고 planner 의 urdf-프레임
    ``TABLE_TOP`` 도 같은 값에서 파생한다. 큐브 크기가 바뀌면 half 를 통해 자동으로 따라간다.

    2026-07-29 이전에는 ``--grasp_z 0.060``(경험 튜닝) + (half − 0.020) 이었고 실측 안착
    0.04976 과 **10.24 mm** 어긋나 있었다. planner 의 ``GRASP_Z_OFF = -0.008`` 이 그걸 부분
    상쇄해 우연히 동작하던 상태다 — 두 값을 함께 재측정해 정리했다(§Goal B).
    """
    robot = env.scene["robot"].data
    cp, cq = subtract_frame_transforms(robot.root_pos_w, robot.root_quat_w,
                                       env.scene[CUBE].data.root_pos_w,
                                       env.scene[CUBE].data.root_quat_w)
    wp, _ = subtract_frame_transforms(robot.root_pos_w, robot.root_quat_w,
                                      env.scene[BOWL].data.root_pos_w,
                                      env.scene[BOWL].data.root_quat_w)
    # 크기 DR 로 env 마다 큐브가 다르면 조준 z 도 env 마다 다르다(상판 + 각자의 반변).
    halves = _cube_halves(env)
    grasp_zs = [args.grasp_z if args.grasp_z is not None else TABLE_TOP_BASE + h for h in halves]
    _warn_cube_z_drift(cp[:, 2], grasp_zs)
    cubes = [[cp[i, 0].item(), cp[i, 1].item(), grasp_zs[i], *cq[i].tolist()]
             for i in range(env.num_envs)]
    bowls = [[wp[i, 0].item(), wp[i, 1].item()] for i in range(env.num_envs)]
    return cubes, bowls, halves


def _warn_cube_z_drift(measured_z, assumed_z, tol=_CUBE_Z_DRIFT_TOL):
    """측정 큐브 z 와 planner 로 보내는 유도 z 가 tol 이상 벌어지면 1회 경고.

    책상 높이 변경·큐브 교체·안착 실패(굴러떨어짐/겹침) 같은 "조용히 조준만 틀어지는" 사고를
    잡는 저비용 감시. 판정에는 개입하지 않는다(경고만) — 유도값이 여전히 진실 소스다.
    """
    assumed = torch.as_tensor(assumed_z, device=measured_z.device, dtype=measured_z.dtype)
    dz = float((measured_z - assumed).abs().max().item())
    if dz > tol and not getattr(_warn_cube_z_drift, "warned", False):
        _warn_cube_z_drift.warned = True
        print(f"[sm] ⚠ cube z drift {dz * 1000:.1f} mm (measured vs assumed "
              f"{assumed.min().item():.4f}~{assumed.max().item():.4f}) "
              f"— 책상 높이·큐브 크기·안착 상태 확인. grasp 조준이 그만큼 어긋난다", flush=True)


def _cubes_in_bowl(env):
    """Per-env success proxy: cube center within --bowl_tol (xy) of its bowl. → [bool ×N]."""
    c = env.scene[CUBE].data.root_pos_w
    b = env.scene[BOWL].data.root_pos_w
    return (torch.linalg.norm(c[:, :2] - b[:, :2], dim=1) < args.bowl_tol).tolist()


def _log_state(env):
    """Print env-0 EE/cube 6D pose + bowl xy each step, world frame (multi-env 은 env 0 만).
    pose = position (m), Euler-XYZ (deg), quaternion (wxyz)."""
    ee, cube, bowl = env.scene["ee_frame"].data, env.scene[CUBE].data, env.scene[BOWL].data
    ep = ee.target_pos_w[0, 0].detach().cpu().numpy()
    eq = ee.target_quat_w[0, 0].detach().cpu().numpy()
    cp = cube.root_pos_w[0].detach().cpu().numpy()
    cq = cube.root_quat_w[0].detach().cpu().numpy()
    wp = bowl.root_pos_w[0].detach().cpu().numpy()
    ee_e = _euler_deg_from_wxyz_np(eq)
    cu_e = _euler_deg_from_wxyz_np(cq)
    xax = _quat_rotate_np(eq, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    yax = _quat_rotate_np(eq, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    zax = _quat_rotate_np(eq, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    dx, dy, dz = FIXED_INNER_CENTER
    fixed_inner = ep + dx * xax + dy * yax + dz * zax
    fixed_tip = ep + PAD_LOW_OFF * zax
    print(f"[world ee]   pos={_fmt_vec(ep)} "
          f"eul=({ee_e[0]:+6.1f},{ee_e[1]:+6.1f},{ee_e[2]:+6.1f}) "
          f"quat={_fmt_quat(eq)}", flush=True)
    print(f"[world cube] pos={_fmt_vec(cp)} "
          f"eul=({cu_e[0]:+6.1f},{cu_e[1]:+6.1f},{cu_e[2]:+6.1f}) "
          f"quat={_fmt_quat(cq)}  bowl={_fmt_vec(wp)}", flush=True)
    print(f"[world jaw]  fixed_tip={_fmt_vec(fixed_tip)} fixed_inner={_fmt_vec(fixed_inner)}",
          flush=True)


def _step(env, action, recorder=None):
    """env.step + per-step EE/cube/bowl task-space state print (--log_every)."""
    out = env.step(action)
    _step.i = getattr(_step, "i", 0) + 1
    if args.log_every > 0 and _step.i % args.log_every == 0:
        _log_state(env)
    if recorder is not None:
        recorder.capture()
    return out


# ══ livestream keyboard (R/N/B one-shot flag) ═════════════════════════════════════
def _key_listener():
    """Subscribe to carb keyboard (comes through the WebRTC livestream client). R/N/B set a
    one-shot flag consumed by the loop. Returns a dict holding the flag + the subscription."""
    # ⚠ 지연 import — `omni.appwindow` 는 headless + 카메라 없음(=sweep 의 조합)에서 로드되지
    #   않아 모듈 최상단 import 면 ModuleNotFoundError 로 sweep 이 부팅 중 죽는다.
    #   이 함수는 interactive 모드에서만 호출되므로 여기서 들여온다.
    import omni.appwindow  # noqa: PLC0415

    state = {"key": None}
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    inp = carb.input.acquire_input_interface()

    def _on_kbd(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                state["key"] = "R"
            elif event.input == carb.input.KeyboardInput.N:
                state["key"] = "N"
            elif event.input == carb.input.KeyboardInput.B:
                state["key"] = "B"
        return True

    state["_sub"] = inp.subscribe_to_keyboard_events(keyboard, _on_kbd)  # keep ref alive
    return state


def _poll_key(state):
    """Consume and return a pending R/N/B key press (or None)."""
    if state is None:
        return None
    k, state["key"] = state["key"], None
    return k


def _wait_key(state):
    """Pump the app (livestream stays live) until a key is pressed. Returns the key, or None
    if the window closed. Honors an already-pending press (doesn't drop it)."""
    while simulation_app.is_running():
        k = _poll_key(state)
        if k is not None:
            return k
        simulation_app.update()
    return None


# ══ DR layout snapshot / restore (so R replays the exact same scene, all envs) ════
def _capture_layout(env):
    """Snapshot per-env DR cube + bowl world poses so R can restore this exact layout."""
    c, b = env.scene[CUBE].data, env.scene[BOWL].data
    return {"cube": (c.root_pos_w.clone(), c.root_quat_w.clone()),
            "bowl": (b.root_pos_w.clone(), b.root_quat_w.clone())}


def _restore_layout(env, layout):
    """Overwrite the freshly-DR'd cube + bowl with the saved poses (zero velocity), all envs."""
    for name, k in ((CUBE, "cube"), (BOWL, "bowl")):
        pos, quat = layout[k]
        obj = env.scene[name]
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))


def _sweep_summary(cells, n_targets, cube_halves=None):
    """sweep 셀 집계 + 스폰영역 기하 메타 → plot_sweep 가 읽는 JSON dict.

    ``cube_halves`` = per-env 큐브 반변(크기 DR). env 당 크기가 런 내내 고정이므로
    셀↔크기 대응은 ``cells[].fails[].env`` 인덱스로 되짚는다. 크기별 성공률을 깨끗이
    보려면 ``--cube_sizes <하나>`` 로 고정해 sweep 을 크기마다 돌린다.
    """
    return {
        "task": args.task, "num_envs": args.num_envs, "bowl_tol": args.bowl_tol,
        "cube_sizes_arg": args.cube_sizes,
        "grasp_retries": int(args.grasp_retries),
        "cube_halves": [float(h) for h in cube_halves] if cube_halves else None,
        "yaw": args.yaw, "trials": args.trials, "seed": args.seed,
        "grid": {"nx": args.nx, "ny": args.ny, "boundary_n": args.boundary_n},
        "spawn": {
            "bell": [list(p) for p in SA.CUBE_SCATTER_BELL],
            "x_range": list(SA.CUBE_SCATTER_X_RANGE), "y_range": list(SA.CUBE_SCATTER_Y_RANGE),
            "exclude_box": list(SA.CUBE_ARM_EXCLUDE),
            "bowl_center_xy": list(SA.BOWL_CENTER_XY), "bowl_sep": SA.MIN_BOWL_SEP,
            "base_xy": list(SA.BASE_XY), "base_sep": SA.MIN_BASE_SEP,
        },
        "n_targets": int(n_targets),
        "cells": list(cells.values()),
    }


def _build_env():
    """pick_cube env 를 SM 규약대로 생성한다(카메라 제거·init 자세 고정·조기종료 해제·뷰포트).

    ``--record_hdf5`` 면 leisaac 방식 datagen record 로 전환: 카메라·시각 DR 유지 +
    IsaacLab RecorderManager 배선 + 에피소드 규격 termination(자동 종료·success attr).
    """
    record_hdf5 = getattr(args, "record_hdf5", None)
    record_lerobot = getattr(args, "record_lerobot", None)
    if record_hdf5 and record_lerobot:
        raise SystemExit("--record_hdf5 와 --record_lerobot 은 동시 사용 불가 (하나만)")
    record = record_hdf5 or record_lerobot
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if record:
        # 카메라 = 기록 대상이므로 유지. AppLauncher 렌더 파이프라인 필수.
        if not getattr(args, "enable_cameras", False):
            raise SystemExit("--record_hdf5/--record_lerobot requires --enable_cameras")
        if record_lerobot and args.num_envs != 1:
            raise SystemExit("--record_lerobot 은 --num_envs 1 전용 (multi-env 는 --record_hdf5)")
    else:
        remove_pick_cube_cameras(env_cfg)  # SM plans on state only → skip camera spawn/render
        # SM 은 렌더 없는 state-only → **시각** DR 만 제거. -DR 은 robot-color(Replicator) 용으로
        # replicate_physics=False + robot-color/lights/focal 이벤트를 켜는데, 이는 렌더가 있어야
        # 의미 있고 headless 에선 physx view(get_dof_velocities)를 깨뜨린다. datagen(카메라 경로)은 유지.
        for _ev in ("randomize_robot_color", "randomize_lights", "randomize_camera_focal"):
            if hasattr(env_cfg.events, _ev):
                setattr(env_cfg.events, _ev, None)
        # 큐브 **크기** DR 은 물리(=grasp 기하)라 state-only 에서도 유지한다. 다만 prestartup
        # USD 편집이라 EventManager 가 replicate_physics=True 를 금지 → 켜져 있으면 False 로 둔다.
        # (시각 DR 만 있었을 때는 True 로 되돌려 physx view 크래시를 피하던 자리다.)
        env_cfg.scene.replicate_physics = getattr(env_cfg.events, "randomize_cube_sizes", None) is None
    if args.cube_sizes:
        # 크기별 성공률 진단·검증용 override — DR 사다리를 이 목록으로 좁힌다(전 env 고정도 가능).
        # ⚠ record/state-only 분기 **뒤**에 둔다(분기 안에 끼면 카메라 제거 else 를 가로챈다).
        ev = getattr(env_cfg.events, "randomize_cube_sizes", None)
        if ev is None:
            raise SystemExit(f"--cube_sizes 는 큐브 크기 DR 이 있는 env 에서만 사용 가능 "
                             f"(task={args.task} 에 randomize_cube_sizes 이벤트 없음)")
        ev.params["sizes"] = [float(v) for v in str(args.cube_sizes).split(",") if v.strip()]
        print(f"[sm] cube size DR override → {ev.params['sizes']}", flush=True)
    # Robot spawns AT the start pose from frame 0 (no neutral→init transient), reset jitter zeroed.
    env_cfg.scene.robot.init_state.joint_pos = dict(INIT_RAD)
    if hasattr(env_cfg.events, "reset_robot_joints"):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    # Replay the WHOLE planned trajectory: drop early-cut terminations. `success` fires while the
    # held cube merely passes over the bowl mid-transit → env would auto-reset before release runs.
    # Keep only time_out (30 s = 900 steps ≫ ~442-row traj); judge success at end via _cubes_in_bowl.
    for _term in ("success", "cube_lost"):
        if hasattr(env_cfg.terminations, _term):
            setattr(env_cfg.terminations, _term, None)
    if record:
        # 에피소드 규격 종료: 이동 후 init 복귀 + posthold_s 연속 정지 → auto-reset → export.
        # success(placed_and_returned) 는 같은 스텝 발화 — stock record_pre_reset 이 attr 로 기록.
        step_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
        hold_steps = max(1, int(round(args.posthold_s * step_hz)))
        env_cfg.terminations.episode_done = DoneTerm(
            func=pc_term.returned_home_after_motion, params={"hold_steps": hold_steps})
        env_cfg.terminations.success = DoneTerm(
            func=pc_term.placed_and_returned,
            params={"cube_cfg": SceneEntityCfg(CUBE), "bowl_cfg": SceneEntityCfg(BOWL),
                    "bowl_tol": args.bowl_tol, "hold_steps": hold_steps})
        env_cfg.terminations.force_done = DoneTerm(func=_force_done_term)
        if record_hdf5:
            # leisaac 방식 recorder: stock RecorderManager (multi-env = env당 demo_N 분리) + datagen term.
            out = Path(args.record_hdf5)
            env_cfg.recorders = SO101DatagenRecorderManagerCfg()
            env_cfg.recorders.dataset_export_dir_path = str(out.parent)
            env_cfg.recorders.dataset_filename = out.stem
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL  # 실패도 저장(attr 구분)
            env_cfg.recorders.export_in_close = False  # close 시 잔여(꼬리 쓰레기) 버퍼 export 금지
        else:
            # record_lerobot: stock term 만(활성 term ≥1 이어야 success attr 경로가 돎).
            # EXPORT_NONE = env 생성 시 stock manager 가 파일을 만들지 않게. 이후 main() 이
            # SO101LeRobotRecorderManager 로 교체(leisaac use_lerobot_recorder 동형).
            from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
            env_cfg.recorders = ActionStateRecorderManagerCfg()
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env_cfg.viewer.origin_type = "env"  # env-relative camera (env0 = world origin here)
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args.cam_eye)
    env_cfg.viewer.lookat = tuple(args.cam_target)
    env_cfg.seed = args.seed
    return gym.make(args.task, cfg=env_cfg).unwrapped


class PickPlaceSM:
    """env + planner 소켓 + 트라이얼 상태를 들고 있는 SM 본체.

    모드 드라이버(``run_random``/``run_fail``/``run_sweep``)가 이 인스턴스를 받아 쓴다.
    (예전에는 ``main()`` 안 15개 클로저 + ``nonlocal``/함수 attribute 로 흩어져 있었다 —
    상태 소유자를 한 곳으로 모아 replay/판정 로직의 교차 오염을 구조적으로 막는다.)
    """

    def __init__(self):
        self.env = _build_env()
        env = self.env
        if getattr(args, "record_lerobot", None):
            # leisaac use_lerobot_recorder 동형: stock manager 를 v3 직기록 manager 로 교체.
            from sim_to_real.data.lerobot_recorder import LeRobotV3DatasetWriter
            from sim_to_real.data.lerobot_recorder_manager import SO101LeRobotRecorderManager
            writer = LeRobotV3DatasetWriter(args.record_lerobot, overwrite=True,
                                            enable_videos=True, robot_type="so_follower")
            del env.recorder_manager
            env.recorder_manager = SO101LeRobotRecorderManager(
                env.cfg.recorders, env, writer, args.task_description)
            print(f"[sm] record_lerobot → {args.record_lerobot}", flush=True)
        print(f"[sm] mode={args.mode} env={args.task} action_dim={env.action_space.shape} "
              f"device={env.device}", flush=True)
        self.robot = env.scene["robot"]
        # articulation → SO101 order
        self.so101_idx = [self.robot.joint_names.index(j) for j in SO101_JOINT_ORDER]
        self.hold = torch.tensor(INIT_ACTION, device=env.device,
                                 dtype=torch.float32).repeat(env.num_envs, 1)  # (N,6) init hold

        self.zmq_ctx = zmq.Context()
        self.sock = None
        self.poller = zmq.Poller()
        self._connect_planner()
        print(f"[sm] planner {args.planner} (plan_timeout={args.plan_timeout_s}s)", flush=True)

        interactive = (args.mode == "fail" and not getattr(args, "auto", False)) \
            or (args.mode == "random" and args.auto_trials == 0)
        self.key = _key_listener() if interactive else None
        self.layout = None
        self.last_manip = {}
        self.request_i = 0
        self.planner_knobs = json.loads(args.planner_knobs_json) if args.planner_knobs_json else None
        if self.planner_knobs is not None and not isinstance(self.planner_knobs, dict):
            raise ValueError("--planner_knobs_json must decode to a JSON object")

    # ── planner 소켓 ─────────────────────────────────────────────────────────────
    def _connect_planner(self):
        """REQ 소켓 (재)생성. REQ 는 send→recv 를 엄격히 번갈아야 해서, 응답을 못 받은 소켓은
        재사용이 불가능하다 → 타임아웃 뒤에는 버리고 새로 만든다."""
        if self.sock is not None:
            self.poller.unregister(self.sock)
            self.sock.close(linger=0)
        self.sock = self.zmq_ctx.socket(zmq.REQ)
        self.sock.connect(args.planner)
        self.poller.register(self.sock, zmq.POLLIN)

    def _recv_plan(self):
        """planner 응답 대기 — Kit 앱을 계속 pump 하면서(안 그러면 plan 동안 앱이 얼고
        WebRTC livestream 이 입력을 못 받는다). 2 ms poll 이라 대기는 render-bound(~30 FPS).

        → ("ok", reply) · ("closed", None) 창 닫힘 · ("timeout", None) deadline 초과.
        타임아웃이 없으면 planner 가 죽었을 때 headless sweep 이 **영원히** 멈춘다(무인 실행
        최대 리스크). --plan_timeout_s 0 = 무제한(옛 동작).
        """
        t0 = time.monotonic()
        while not self.poller.poll(2):
            if not simulation_app.is_running():
                return "closed", None
            if args.plan_timeout_s > 0 and (time.monotonic() - t0) > args.plan_timeout_s:
                return "timeout", None
            simulation_app.update()
        return "ok", json.loads(self.sock.recv())

    # ── 씬 조작 primitives ────────────────────────────────────────────────────────
    def reset(self, which, seed=None, recorder=None):
        """Scene reset: robot → init (env.reset), cube → NEW DR (N) or SAVED layout (R)."""
        env = self.env
        if seed is None:
            env.reset()
        else:
            env.reset(seed=int(seed))
        if which == "R" and self.layout is not None:
            _restore_layout(env, self.layout)  # overwrite the fresh DR with the saved layout
        for _ in range(args.settle):           # settle (hold init; zeros would drift the arm)
            _step(env, self.hold, recorder)
        if which == "N":
            self.layout = _capture_layout(env)  # remember this new layout for a later R

    def reset_to_targets(self, batch_xy, yaws_rad, seed):
        """env.reset(robot→init) 후 Cube1 을 batch_xy(env-local) 로, Bowl 을 nominal 고정으로
        덮어써 settle. DR 배치를 통제된 타깃으로 치환(bowl 고정 = 성공맵 교란 제거). fail/sweep 공용.
        batch_xy/yaws_rad 길이 = num_envs (부분 batch 는 호출 측에서 첫 타깃으로 패딩)."""
        env = self.env
        env.reset(seed=int(seed))
        cube, bowl = env.scene[CUBE], env.scene[BOWL]
        origins = env.scene.env_origins
        cdef = cube.data.default_root_state.clone()   # (N,13) env-local (origin 미포함)
        bdef = bowl.data.default_root_state.clone()
        tx = torch.tensor([p[0] for p in batch_xy], device=env.device, dtype=torch.float32)
        ty = torch.tensor([p[1] for p in batch_xy], device=env.device, dtype=torch.float32)
        th = torch.tensor(list(yaws_rad), device=env.device, dtype=torch.float32)
        z0 = torch.zeros_like(th)
        cq = quat_mul(cdef[:, 3:7], quat_from_euler_xyz(z0, z0, th))  # 안착 자세 + yaw
        cpos = torch.stack([tx, ty, cdef[:, 2]], dim=-1) + origins
        cube.write_root_pose_to_sim(torch.cat([cpos, cq], dim=-1))
        cube.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
        bpos = bdef[:, :3] + origins                                 # bowl → nominal 고정
        bowl.write_root_pose_to_sim(torch.cat([bpos, bdef[:, 3:7]], dim=-1))
        bowl.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
        for _ in range(args.settle):
            _step(env, self.hold)

    # ── planner ──────────────────────────────────────────────────────────────────
    def plan_batch(self):
        """batch plan 요청 + lockstep 텐서 조립. manipulate/manipulate_record 공용.

        → (status, plan): "closed"(소켓/앱 종료) · "allfail"(전 env plan 실패) ·
        "ok" + {"planned": [bool×N], "tgt": (T,N,6) np}. last_manip 갱신.

        ※ planner 는 결정적이다(cuRobo v0.8 `reset_seed()` 는 인자를 받지 않아 요청 seed 로
          해를 흔들 수 없다) — 같은 (cube, bowl, start) 는 항상 같은 궤적. 그래서 seed 는
          knob 으로 보내지 않는다(옛 `knobs.seed` 는 진단 문자열에만 쓰이던 no-op).
        """
        env = self.env
        cubes, bowls, cube_halves = _cubes_bowls_in_base(env)
        starts = [self.robot.data.joint_pos[i][self.so101_idx].tolist() for i in range(env.num_envs)]
        # cube_half = 큐브 기하가 pose 와 **함께** 실려 간다(cube_specs 단일 소스).
        # planner 가 자체 상수로 40 mm 를 가정하던 것을 대체 — 크기 변경이 한 곳만 고치면 끝난다.
        # 크기 DR 이 켜지면 env 마다 다르므로 **per-env 리스트**로 보낸다(planner 는 스칼라도 수용).
        req = {"cmd": "plan_pickplace", "cubes": cubes, "bowl": bowls, "start": starts,
               "cube_half": [float(h) for h in cube_halves]}
        if self.planner_knobs:
            req["knobs"] = dict(self.planner_knobs)
        self.request_i += 1
        request_i = self.request_i
        print(f"[sm] send plan_request #{request_i}: envs={env.num_envs} "
              f"cube0={_fmt_vec(cubes[0][:3])} bowl0={_fmt_vec(bowls[0])} "
              f"cube_mm={[round(2000.0 * h) for h in cube_halves]}", flush=True)
        self.sock.send_string(json.dumps(req))
        status, rep = self._recv_plan()
        if status == "closed":
            return "closed", None
        if status == "timeout":
            print(f"[sm] ⚠ planner TIMEOUT ({args.plan_timeout_s}s) — 요청 #{request_i} 포기, "
                  f"소켓 재연결 후 다음 batch 로 진행 (planner 로그 확인)", flush=True)
            self._connect_planner()
            self.last_manip = {
                "request": {"cubes": cubes, "bowls": bowls, "starts": starts,
                            "cube_halves": [float(h) for h in cube_halves]},
                "planner_ok": False, "planner_err": "timeout", "diagnostics": [],
                "planned": [False] * env.num_envs, "placed": [False] * env.num_envs,
                "n_steps": 0,
            }
            return "allfail", None
        print(f"[sm] recv plan_reply #{request_i}: ok={bool(rep.get('ok'))}", flush=True)
        if not rep.get("ok"):
            print(f"[sm] planner ERROR: {rep.get('err')}", flush=True)
        trajs = rep.get("trajectories") if rep.get("ok") else None
        self.last_manip = {
            "request": {"cubes": cubes, "bowls": bowls, "starts": starts,
                            "cube_halves": [float(h) for h in cube_halves]},
            "planner_ok": bool(rep.get("ok")),
            "planner_err": rep.get("err"),
            "diagnostics": rep.get("diagnostics", []),
            "planned": [],
            "placed": [],
            "n_steps": 0,
        }
        for i, diag in enumerate(self.last_manip["diagnostics"]):
            print(f"[sm]   diag env{i}: ok={diag.get('ok')} fail={diag.get('fail')} "
                  f"phases={diag.get('phases')} candidates={diag.get('num_candidates')}",
                  flush=True)
        if not trajs or all(t is None for t in trajs):
            print(f"[sm] plan FAIL (all {env.num_envs} envs)", flush=True)
            self.last_manip["planned"] = [False] * env.num_envs
            self.last_manip["placed"] = [False] * env.num_envs
            return "allfail", None
        planned = [t is not None for t in trajs]
        n_steps = max(len(t) for t in trajs if t is not None)
        self.last_manip["planned"] = [bool(v) for v in planned]
        self.last_manip["n_steps"] = int(n_steps)
        # planner row = (arm deg ×5, gripper feature) → sim joint radians (SO101 order).
        # plan-fail env 는 init hold, 짧은 궤적은 last-row 패딩 → (T, N, 6) lockstep 텐서.
        per_env = []
        for t in trajs:
            if t is None:
                per_env.append(np.tile(np.asarray(INIT_ACTION, np.float32), (n_steps, 1)))
                continue
            rows = np.stack([policy_feature_to_sim_joint_radians(np.asarray(r, np.float32))
                             for r in t])
            if len(rows) < n_steps:
                rows = np.concatenate([rows, np.tile(rows[-1:], (n_steps - len(rows), 1))])
            per_env.append(rows)
        tgt = np.stack(per_env, axis=1)  # (T, N, 6)
        return "ok", {"planned": planned, "tgt": tgt}

    # ── replay ───────────────────────────────────────────────────────────────────
    def manipulate(self, recorder=None):
        """batch-plan + lockstep replay, 실패 env 는 ``--grasp_retries`` 회 재시도.

        재시도가 의미 있는 이유: grasp 실패는 큐브를 **밀어낸다**(측정상 15~48 mm). planner 는
        결정적이지만 입력 cube pose 가 바뀌었으므로 재계획하면 다른 후보·다른 궤적이 나온다.
        사람이 놓친 물건을 다시 집는 것과 같다. 이미 성공한 env 는 init hold 로 고정해
        (그릇 안 큐브를 다시 집으려다 쳐내는 사고를 막는다) 성공 판정을 누적한다.

        → (status, val): ("closed",_) · ("abort","N"/"R") · ("done", (n_attempt, n_placed)).
        """
        env = self.env
        status, val = self._manipulate_once(recorder)
        if status != "done":
            return status, val
        retries = max(0, int(args.grasp_retries))
        for attempt in range(1, retries + 1):
            placed = list(self.last_manip.get("placed", []))
            fail_ids = [i for i, ok in enumerate(placed) if not ok]
            if not fail_ids:
                break
            print(f"[sm] retry {attempt}/{retries}: 실패 env {fail_ids} 재계획 "
                  f"(성공 env 는 init hold)", flush=True)
            hold_ids = [i for i in range(env.num_envs) if i not in set(fail_ids)]
            status, val = self._manipulate_once(recorder, hold_ids=hold_ids)
            if status != "done":
                return status, val
            merged = [bool(a or b) for a, b in zip(placed, self.last_manip.get("placed", []))]
            self.last_manip["placed"] = merged
            self.last_manip["retry_attempts"] = attempt
            self.last_manip["retry_env_ids"] = fail_ids
            val = (env.num_envs, sum(merged))
        return "done", val

    def _manipulate_once(self, recorder=None, hold_ids=()):
        """1회 batch-plan + lockstep replay. ``hold_ids`` env 는 궤적 대신 init hold.

        plan-fail env 는 init hold(실패 처리), 짧은 궤적은 마지막 row 로 패딩 —
        모든 env 가 같은 step 수를 소화한다.
        """
        env = self.env
        status, plan = self.plan_batch()
        if status == "closed":
            return "closed", None
        if status == "allfail":
            return "done", (env.num_envs, 0)
        planned, tgt = plan["planned"], plan["tgt"]
        for i in hold_ids:   # 재시도: 이미 성공한 env 는 움직이지 않는다
            tgt[:, i, :] = np.asarray(INIT_ACTION, np.float32)
        cube_asset = env.scene[CUBE]
        max_cube_z = (cube_asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).clone()
        truncated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for step_rows in tgt:
            k = _poll_key(self.key)
            if k in ("N", "R"):              # cancel remaining actions → next reset command
                return "abort", k
            action = torch.as_tensor(step_rows, device=env.device, dtype=torch.float32)
            _obs, _rew, term, trunc, _info = _step(env, action, recorder)
            cube_z = cube_asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            max_cube_z = torch.maximum(max_cube_z, cube_z)
            if bool(term.any()) or bool(trunc.any()):
                # ⚠ replay 도중 종료 = 궤적이 에피소드 예산(time_out 900 step)을 넘겼다는 뜻.
                #   IsaacLab 은 env.step **안에서** auto-reset 하므로(manager_based_rl_env :216)
                #   종료 env 의 씬 상태는 이미 새 레이아웃이라 성공 판정이 불가능하다.
                #   옛 코드는 여기서 조용히 break 하고 post-reset 상태로 전 env 를 판정했다
                #   (한 env 종료가 나머지 env 궤적까지 잘라먹는 교차 오염) → 크게 알리고
                #   종료 env 만 판정 불가(False)로 표시한다.
                truncated = term | trunc
                print(f"[sm] ⚠ replay truncated by termination: envs={truncated.tolist()} "
                      f"— 궤적이 time_out 예산을 초과했다. 해당 env 는 판정 불가(False)로 처리",
                      flush=True)
                break
        success = torch.as_tensor(_cubes_in_bowl(env), device=env.device, dtype=torch.bool)
        success = success & ~truncated
        final_cube_xyz = cube_asset.data.root_pos_w[:, :3] - env.scene.env_origins
        self.last_manip["max_cube_z"] = [float(v) for v in max_cube_z.tolist()]
        self.last_manip["final_cube_xyz"] = [[float(v) for v in row] for row in final_cube_xyz.tolist()]
        self.last_manip["truncated"] = [bool(v) for v in truncated.tolist()]
        self.last_manip["placed"] = [bool(v) for v in success.tolist()]
        n_placed = sum(1 for i, ok in enumerate(success.tolist()) if planned[i] and ok)
        for i, (p, ok) in enumerate(zip(planned, success.tolist())):
            print(f"[sm]   env{i}: plan={'ok' if p else 'FAIL'} placed={bool(ok)}", flush=True)
        return "done", (env.num_envs, n_placed)

    # ══ record 모드 (--record_hdf5): termination-driven 에피소드 + RecorderManager ═══
    def force_reset_envs(self, ids):
        """이동 없던 env 를 force_done term 으로 1-step 강제 종료 — auto-reset 이 새 DR
        레이아웃과 episode_length_buf 리셋을 제공한다(time_out 대기·수동 reset 불필요)."""
        env = self.env
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[ids] = True
        env._force_done_mask = mask
        _step(env, self.hold)
        env._force_done_mask = None

    def manipulate_record(self, preroll_steps):
        """record 트라이얼 1회: plan(step 없음=미기록) → 버퍼 폐기+initial_state →
        pre-roll 정지 → 궤적 replay → 전 env 종료 대기(auto-reset=export). → (status, val)."""
        env = self.env
        rm = env.recorder_manager
        tm = env.termination_manager
        status, plan = self.plan_batch()
        if status == "closed":
            return "closed", None
        # 기록 시작점 — settle/직전 트라이얼 꼬리/cold-start 프레임 폐기 후 initial_state 재기록
        # (settle 뒤 스냅샷이라 stock post-reset 시점보다 실제 시작 상태에 더 정확).
        rm.reset()
        rm.add_to_episodes("initial_state", env.scene.get_state(is_relative=True))
        if status == "allfail":
            self.force_reset_envs(list(range(env.num_envs)))  # 1-프레임 잔여는 변환기 최소길이 필터가 제거
            return "done", (env.num_envs, 0)
        planned, tgt = plan["planned"], plan["tgt"]
        planned_t = torch.tensor(planned, dtype=torch.bool, device=env.device)
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        placed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        def _step_track(action):
            nonlocal done, placed
            _o, _r, term, trunc, _i = _step(env, action)
            # TerminationManager 캐시는 auto-reset **전에** 계산된 값이라 종료 스텝에도 유효하다
            # (씬 상태는 이미 리셋됨 → 씬에서 읽으면 안 된다).
            placed = placed | tm.get_term("success")
            done = done | term | trunc

        for _ in range(preroll_steps):   # 정지 pre-roll — moved 래치 False 라 종료 미발화
            _step_track(self.hold)
        for step_rows in tgt:            # 이동 → pick-place → retreat(init 복귀) → init hold
            _step_track(torch.as_tensor(step_rows, device=env.device, dtype=torch.float32))
        guard = 0
        while not bool((done | ~planned_t).all()) and simulation_app.is_running():
            _step_track(self.hold)       # posthold 종료 대기 (안전망 = time_out 30 s)
            guard += 1
            if guard > _RECORD_DRAIN_MAX_STEPS:
                print(f"[sm] record: 종료 대기 초과({_RECORD_DRAIN_MAX_STEPS} steps) — "
                      f"time_out 에 위임", flush=True)
                break
        fail_ids = [i for i, p in enumerate(planned) if not p]
        if fail_ids:
            rm.reset(fail_ids)           # 이동 없던 버퍼 폐기 후 강제 종료(새 레이아웃)
            self.force_reset_envs(fail_ids)
        self.last_manip["placed"] = [bool(v) for v in placed.tolist()]
        n_placed = int((placed & planned_t).sum().item())
        for i, p in enumerate(planned):
            print(f"[sm]   env{i}: plan={'ok' if p else 'FAIL'} placed={bool(placed[i])} "
                  f"done={bool(done[i])}", flush=True)
        return "done", (env.num_envs, n_placed)


# ══ mode: random ═════════════════════════════════════════════════════════════════
def run_random(sm):
    if getattr(args, "record_hdf5", None) or getattr(args, "record_lerobot", None):
        if args.auto_trials <= 0:
            raise SystemExit("--record_hdf5/--record_lerobot 은 --auto_trials N(자동 모드) 필수 — "
                             "인터랙티브는 에피소드 규격(2s pre-roll) 보장 불가")
        _run_random_auto_record(sm)
    elif args.auto_trials > 0:
        _run_random_auto(sm)
    else:
        _run_random_interactive(sm)


def _run_random_auto_record(sm):
    """record 자동 루프 — 트라이얼 경계 = termination auto-reset(export). env.reset 은 최초 1회만."""
    env = sm.env
    preroll_steps = max(0, int(round(args.preroll_s / env.step_dt)))
    summary_dir = args.summary_dir or "/workspace/scratch/curobo-auto-trials"
    record_out = args.record_hdf5 or args.record_lerobot
    print(f"[sm] record={record_out} auto_trials={args.auto_trials} "
          f"preroll={preroll_steps} steps posthold={args.posthold_s}s", flush=True)
    env.reset()                      # 초기 DR 레이아웃 — 이후 경계는 auto-reset
    trials = []
    try:
        for trial_i in range(1, args.auto_trials + 1):
            if not simulation_app.is_running():
                break
            for _ in range(args.settle):   # 레이아웃 안정(다음 rm.reset 이 폐기)
                _step(env, sm.hold)
            status, val = sm.manipulate_record(preroll_steps)
            n_attempt, n_placed = val if status == "done" else (env.num_envs, 0)
            trials.append({"trial": trial_i, "status": status,
                           "n_attempt": int(n_attempt), "n_placed": int(n_placed),
                           **sm.last_manip})
            print(f"[sm] record trial {trial_i}/{args.auto_trials}: status={status} "
                  f"placed={n_placed}/{n_attempt}", flush=True)
            if status == "closed":
                break
    finally:
        summary = {
            "task": args.task, "num_envs": args.num_envs, "base_seed": args.seed,
            "record_hdf5": args.record_hdf5, "record_lerobot": args.record_lerobot,
            "preroll_s": args.preroll_s, "posthold_s": args.posthold_s, "trials": trials,
        }
        summary_path = Path(summary_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[sm] record summary → {summary_path}", flush=True)
        env.close()                  # recorder/HDF5 파일 정상 close (export_in_close=False)


def _run_random_auto(sm):
    """키 입력 없이 랜덤 DR trial 을 --auto_trials 회 실행 + (선택) viewport MP4 기록."""
    env = sm.env
    record_dir = args.record_viewport_dir
    if record_dir is not None and record_dir.lower() in {"", "none", "off", "false", "0"}:
        record_dir = None
    summary_dir = args.summary_dir or record_dir or "/workspace/scratch/curobo-auto-trials"
    recorder = ViewportVideoRecorder(record_dir, fps=args.record_fps, every=args.record_every)
    trials = []
    print(f"[sm] auto_trials={args.auto_trials} seed={args.seed} "
          f"record_dir={record_dir} summary_dir={summary_dir}", flush=True)
    try:
        for trial_i in range(1, args.auto_trials + 1):
            if not simulation_app.is_running():
                break
            trial_seed = int(args.seed + trial_i - 1)
            video_path = recorder.start(trial_i) if recorder.enabled else None
            print(f"[sm] auto trial {trial_i}/{args.auto_trials} seed={trial_seed}", flush=True)
            status, val = "unknown", None
            try:
                sm.reset("N", seed=trial_seed, recorder=recorder)
                status, val = sm.manipulate(recorder=recorder)
            finally:
                recorder.close()
            n_attempt, n_placed = val if status == "done" else (env.num_envs, 0)
            trials.append({
                "trial": trial_i, "seed": trial_seed, "status": status, "video": video_path,
                "n_attempt": int(n_attempt), "n_placed": int(n_placed), **sm.last_manip,
            })
            print(f"[sm] auto trial {trial_i}: status={status} placed={n_placed}/{n_attempt} "
                  f"video={video_path}", flush=True)
            if status == "closed":
                break
    finally:
        summary = {
            "task": args.task, "num_envs": args.num_envs, "base_seed": args.seed,
            "record_fps": args.record_fps, "record_every": args.record_every,
            "viewport_render_product": getattr(recorder, "render_product_path", None),
            "trials": trials,
        }
        summary_path = Path(summary_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[sm] auto summary → {summary_path}", flush=True)


def _run_random_interactive(sm):
    """livestream 키: N=새 DR · R=같은 레이아웃 · B=plan+manipulate."""
    env = sm.env
    sm.reset("N")  # initial scene = new DR layout
    print(f"[sm] ready — B=plan+manipulate · N=new layout · R=same layout "
          f"(livestream keys, {env.num_envs} envs)", flush=True)
    n_ok = n_run = 0
    while simulation_app.is_running():
        k = _wait_key(sm.key)
        if k is None:
            break                            # window closed
        if k in ("N", "R"):
            sm.reset(k)
            print(f"[sm] reset ({k})", flush=True)
            continue
        status, val = sm.manipulate()        # k == "B"
        if status == "closed":
            break
        if status == "abort":                # R/N mid-run → cancel + reset robot pose + scene
            sm.reset(val)
            print(f"[sm] ABORT → reset ({val})", flush=True)
            continue
        n_attempt, n_placed = val
        n_run += n_attempt
        n_ok += n_placed
        print(f"[sm] manipulate {n_placed}/{n_attempt} ({n_ok}/{n_run} placed total)", flush=True)


# ══ mode: fail — sweep 결과의 fail 좌표만 재현(인터랙티브) ══════════════════════════
def run_fail(sm):
    env = sm.env
    fdata = json.loads(Path(args.results).read_text(encoding="utf-8"))
    fails = [c for c in fdata["cells"] if c["n_placed"] < c["n"]]  # place+plan fail 전부
    fails.sort(key=lambda c: (c["y"], c["x"]))
    Nf = env.num_envs
    fbatches = [fails[i:i + Nf] for i in range(0, len(fails), Nf)] or [[]]

    if args.auto:  # headless: 전 batch 재현 + planned/placed 집계(sweep 루프 미러, 키 無)
        # 최신 sweep 는 실패 당시 yaw/seed 를 fails[] 에 기록한다. 셀당 실패가 여러 번이면
        # 각각을 독립 케이스로 재현하고, 구형 결과처럼 상세가 없을 때만 yaw=0 으로 폴백한다.
        cases = []
        for cell in fails:
            records = [f for f in cell.get("fails", []) if isinstance(f, dict)]
            if records:
                cases.extend({"cell": cell, "failure": f} for f in records)
            else:
                cases.append({"cell": cell, "failure": {}})
        case_batches = [cases[i:i + Nf] for i in range(0, len(cases), Nf)] or [[]]
        tot = t_pl = t_ok = 0
        for bi, batch in enumerate(case_batches):
            if not simulation_app.is_running() or not batch:
                break
            padded = batch + [batch[0]] * (Nf - len(batch))
            seed = args.seed + bi
            sm.reset_to_targets(
                [(case["cell"]["x"], case["cell"]["y"]) for case in padded],
                [math.radians(float(case["failure"].get("yaw_deg", 0.0))) for case in padded],
                seed,
            )
            status, _ = sm.manipulate()
            if status == "closed":
                break
            planned = sm.last_manip.get("planned", [False] * Nf)
            placed = sm.last_manip.get("placed", [False] * Nf)
            diags = sm.last_manip.get("diagnostics", [])
            max_z = sm.last_manip.get("max_cube_z", [])
            final_xyz = sm.last_manip.get("final_cube_xyz", [])
            for i, case in enumerate(batch):  # 실 케이스만(패딩 제외)
                c, failure = case["cell"], case["failure"]
                pl = bool(planned[i]) if i < len(planned) else False
                ok = pl and (bool(placed[i]) if i < len(placed) else False)
                cand = (diags[i].get("candidate") or {}) if i < len(diags) else {}
                fk = cand.get("fk_face_error", {})
                tot += 1; t_pl += int(pl); t_ok += int(ok)
                print(f"[fail-auto] ({c['x']:+.3f},{c['y']:+.3f}) {c['kind']:12s} "
                      f"yaw={float(failure.get('yaw_deg', 0.0)):+.2f} "
                      f"source_seed={failure.get('scene_seed', failure.get('plan_seed'))} "
                      f"planned={pl} placed={ok} "
                      f"alpha={cand.get('alpha_deg')} rho={cand.get('rho_deg')} "
                      f"face={fk.get('face_angle')} h={fk.get('h')} t={fk.get('t')} "
                      f"max_z={max_z[i] if i < len(max_z) else None} "
                      f"final={final_xyz[i] if i < len(final_xyz) else None}", flush=True)
            print(f"[fail-auto] batch {bi + 1}/{len(case_batches)} cumulative "
                  f"planned={t_pl}/{tot} placed={t_ok}/{tot}", flush=True)
        print(f"[fail-auto] DONE planned={t_pl}/{tot} placed={t_ok}/{tot}", flush=True)
        return

    fb = {"i": 0}

    def _load_fail(advance):
        if advance:
            fb["i"] = (fb["i"] + 1) % len(fbatches)
        real = list(fbatches[fb["i"]])
        if not real:
            print("[fail] fail 셀 없음(전부 성공)", flush=True)
            return
        padded = real + [real[0]] * (Nf - len(real))  # 남는 env 는 첫 fail 좌표 복제
        sm.reset_to_targets([(c["x"], c["y"]) for c in padded], [0.0] * Nf, args.seed + fb["i"])
        print(f"[fail] batch {fb['i'] + 1}/{len(fbatches)} — {len(real)} fail 좌표:", flush=True)
        for c in real:
            typ = "plan-fail " if c["n_planned"] == 0 else "place-fail"
            print(f"[fail]   ({c['x']:+.3f}, {c['y']:+.3f})  {c['kind']:12s} {typ} "
                  f"fails={c.get('fails')}", flush=True)

    _load_fail(advance=False)
    print(f"[sm] FAIL-REPLAY ready — N=다음 batch · R=같은 batch · B=plan+run "
          f"({len(fails)} fail cells, {len(fbatches)} batches, {Nf} envs)", flush=True)
    while simulation_app.is_running():
        k = _wait_key(sm.key)
        if k is None:
            break
        if k == "N":
            _load_fail(advance=True)
            continue
        if k == "R":
            _load_fail(advance=False)
            continue
        status, val = sm.manipulate()        # B
        if status == "closed":
            break
        if status == "abort":
            _load_fail(advance=(val == "N"))
            continue
        n_attempt, n_placed = val
        print(f"[sm] manipulate {n_placed}/{n_attempt}", flush=True)


# ══ mode: sweep — 스폰영역 grid+boundary 정량 평가 → JSON ═══════════════════════════
def run_sweep(sm):
    env = sm.env
    targets_all = SA.sweep_targets(args.nx, args.ny, args.boundary_n)
    N = env.num_envs
    halves = _cube_halves(env)   # env 당 고정(크기 DR) — 결과 JSON 메타로 남긴다
    yaw_mode = str(args.yaw).strip().lower()
    rng = np.random.default_rng(args.seed)
    cells = {}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [targets_all[i:i + N] for i in range(0, len(targets_all), N)]
    n_bnd = sum(1 for _, _, k in targets_all if k != "interior")
    print(f"[sweep] targets={len(targets_all)} (boundary={n_bnd}) chunks={len(chunks)} "
          f"num_envs={N} trials={args.trials} yaw={args.yaw} out={out_path}", flush=True)

    def _record_cell(x, y, kind, planned, placed, fail):
        ckey = f"{round(x, 4)}_{round(y, 4)}_{kind}"
        c = cells.setdefault(ckey, {"x": float(x), "y": float(y), "kind": kind,
                                    "n": 0, "n_planned": 0, "n_placed": 0, "fails": []})
        c["n"] += 1
        c["n_planned"] += int(planned)
        c["n_placed"] += int(placed)
        if fail:
            c["fails"].append(fail)

    counter = 0
    try:
        for trial in range(args.trials):
            for ci, chunk in enumerate(chunks):
                if not simulation_app.is_running():
                    break
                real = list(chunk)
                batch = real + [real[0]] * (N - len(real))       # 부분 chunk 패딩
                if yaw_mode == "random":
                    yaws = [float(rng.uniform(0.0, 2.0 * math.pi)) for _ in batch]
                else:
                    yaws = [math.radians(float(args.yaw))] * N
                seed = int(args.seed + counter)
                counter += 1
                try:
                    sm.reset_to_targets(batch, yaws, seed)
                    status, _val = sm.manipulate()
                except Exception:  # 한 chunk 예외로 전체 sweep 죽지 않게(+traceback 노출)
                    print(f"[sweep] chunk {ci + 1} EXCEPTION — cells 를 error 처리:", flush=True)
                    traceback.print_exc()
                    for x, y, kind in real:
                        _record_cell(x, y, kind, planned=False, placed=False, fail="exception")
                    continue
                if status == "closed":
                    break
                planned = sm.last_manip.get("planned", [False] * N)
                placed = sm.last_manip.get("placed", [False] * N)
                diags = sm.last_manip.get("diagnostics", [])
                for i, (x, y, kind) in enumerate(real):
                    pl = bool(planned[i]) if i < len(planned) else False
                    ok = pl and (bool(placed[i]) if i < len(placed) else False)
                    fail = None
                    if not ok:
                        fail = {
                            "type": "place" if pl else "plan",
                            "trial": int(trial),
                            "chunk": int(ci),
                            "env": int(i),
                            "yaw_deg": float(math.degrees(yaws[i])),
                            # 씬(reset) seed. planner 는 결정적이라 plan seed 는 존재하지 않는다.
                            "scene_seed": int(seed),
                        }
                        if i < len(diags) and isinstance(diags[i], dict) and diags[i].get("fail"):
                            fail["diagnostic"] = diags[i]["fail"]
                        # 실패 유형 판별용 물리 흔적. 이게 없으면 place-fail 이 "못 잡았다"인지
                        # "잡았는데 그릇 밖에 떨어뜨렸다"인지 사후에 알 방법이 없어, 재현 실행을
                        # 한 번 더 돌려야 했다(확률적 실패는 재현도 안 된다).
                        mz = sm.last_manip.get("max_cube_z", [])
                        fx = sm.last_manip.get("final_cube_xyz", [])
                        if i < len(mz):
                            fail["max_cube_z"] = mz[i]
                        if i < len(fx):
                            fail["final_cube_xyz"] = fx[i]
                        # 채택된 grasp 후보의 FK 오차 — place-fail 이 "조준이 나빴나"를 본다.
                        cand = (diags[i].get("candidate") or {}) if i < len(diags) else {}
                        if cand.get("fk_face_error"):
                            fail["fk_face_error"] = cand["fk_face_error"]
                            fail["alpha_deg"] = cand.get("alpha_deg")
                            fail["score"] = cand.get("score")
                    _record_cell(x, y, kind, pl, ok, fail)
                done = sum(v["n"] for v in cells.values())
                placed_tot = sum(v["n_placed"] for v in cells.values())
                print(f"[sweep] trial {trial} chunk {ci + 1}/{len(chunks)} "
                      f"→ cumulative placed {placed_tot}/{done}", flush=True)
                with out_path.open("w", encoding="utf-8") as fp:  # 증분 저장(중단 안전)
                    json.dump(_sweep_summary(cells, len(targets_all), halves), fp, indent=2)
    finally:
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(_sweep_summary(cells, len(targets_all), halves), fp, indent=2)
        print(f"[sweep] done → {out_path}", flush=True)


def main():
    sm = PickPlaceSM()
    try:
        {"random": run_random, "fail": run_fail, "sweep": run_sweep}[args.mode](sm)
    finally:
        sm.env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
