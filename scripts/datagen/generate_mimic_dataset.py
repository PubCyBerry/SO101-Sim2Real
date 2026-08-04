"""Isaac Lab Mimic / **SkillGen** 증강 데이터 생성 — SO-101, cuRobo v0.8.

공식 `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py` 와 같은 흐름이지만 두 곳이
다르다:

1. **warp 선점** — 아래 `import warp` 한 줄. 이게 없으면 env 생성이 kit 확장
   `omni.warp.core`(warp 1.8.2)를 먼저 물려 cuRobo v0.8 커널 임포트가
   ``TypeError: func() got an unexpected keyword argument 'module'`` 로 죽는다.
   ★2026-07-01 "cuRobo v0.8 은 Isaac 과 in-process 공존 불가" 판정의 실제 원인이 이것이었다
   (CUDA 컨텍스트 문제가 아니다). 선점하면 6단계 전부 통과한다.
2. **planner** — 공식 `isaaclab_mimic.motion_planners.curobo.CuroboPlanner` 는 cuRobo v0.7 API
   라 우리 이미지에서 임포트 자체가 불가하다. 대신
   `sim_to_real.datagen.skillgen_planner.SO101SkillGenPlanner` 를 주입한다.

datagen 코어(`isaaclab_mimic.datagen.*`)는 cuRobo 무관해서 **이미지 설치본을 그대로 쓴다**.

## 파이프라인

```
① source 데모 (cuRobo pick-place SM)
   scripts/cuRobo/pickplace_sm.py random --record_hdf5 ...
② subtask 주석 (공식 스크립트 그대로)
   /workspace/isaaclab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
       --task SimToReal-SO101-PickCube-Mimic-v0 --auto \
       --input_file <src.hdf5> --output_file <annotated.hdf5> --headless
③ 증강 생성 (이 스크립트)
   python generate_mimic_dataset.py --task SimToReal-SO101-PickCube-Mimic-DR-v0 \
       --input_file <annotated.hdf5> --output_file <gen.hdf5> \
       --generation_num_trials 100 --headless
```

`--self_test` 는 kit 을 띄우고 실제 씬에서 planner 를 1회 돌려 프레임 왕복·IK 투영·
plan_cspace·retiming 을 검증한다(source 데이터 불요).
"""

# ── ★ warp 선점: AppLauncher 보다 먼저. 위 docstring §1 참조. ───────────────────────
import warp as _warp  # noqa: F401  isort:skip

import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="SO-101 Isaac Lab Mimic/SkillGen 데이터 생성")
parser.add_argument("--task", type=str, default="SimToReal-SO101-PickCube-Mimic-DR-v0",
                    help="mimic env id (ManagerBasedRLMimicEnv 자손이어야 한다)")
parser.add_argument("--input_file", type=str, default=None,
                    help="주석된 source 데이터셋(hdf5). --self_test 가 아니면 필수")
parser.add_argument("--output_file", type=str, default="./datasets/mimic_generated.hdf5",
                    help="생성 결과 hdf5 경로")
parser.add_argument("--generation_num_trials", type=int, default=None,
                    help="생성 목표 수(기본 = env cfg 값)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="병렬 env 수. env 당 planner 1개가 생성된다(VRAM 주의)")
parser.add_argument("--pause_subtask", action="store_true",
                    help="subtask 마다 멈춤(GUI 디버그용)")
parser.add_argument("--no_skillgen", action="store_true",
                    help="planner 없이 순수 Mimic(보간 전이)으로 생성 — 대조군")
parser.add_argument("--planner_step_size", type=float, default=None,
                    help="전이 계획 linear retiming step(rad). 작을수록 프레임이 늘어 같은 경로를 "
                         "더 천천히 지난다(후려치기 완화의 올바른 레버). 미지정 = env cfg "
                         "`mimic_planner_step_size`. 0 이면 retiming 끔")
parser.add_argument("--motion_noise_scale", type=float, default=0.0,
                    help="계획 waypoint 에 더할 gaussian 노이즈 스케일")
parser.add_argument("--debug_planner", action="store_true", help="planner 진단 로그")
parser.add_argument("--record_segments", type=str, default=None,
                    help="생성 에피소드의 **프레임별 구간 출처**를 이 JSON 에 남긴다. 증강본은 "
                         "[cuRobo 전이 | 보간 | 합성 place] 가 이어붙은 결과인데 HDF5 엔 그 "
                         "경계가 안 남아 사후 역산이 불가능하다. `--num_envs 1` 전용")
parser.add_argument("--self_test", action="store_true",
                    help="source 없이 planner 1회 왕복 검증만 하고 종료")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.self_test and not args_cli.input_file:
    parser.error("--input_file 은 필수다(--self_test 제외)")

# AppLauncher 에는 그가 실제로 쓰는 키만 넘긴다(AGENTS.md §AppLauncher 인자 필터).
# 커스텀 인자를 통째로 넘기면 Windows 에서 `_prepare_ui` access violation, Linux 에선
# livestream viewport docking 이 조용히 실패한다. C-레벨 크래시 추적용 faulthandler 도 켠다.
_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}

import faulthandler  # noqa: E402
import os  # noqa: E402

_fault_dir = os.environ.get("SO101_FAULT_DIR", "/workspace/outputs")
try:
    os.makedirs(_fault_dir, exist_ok=True)
    faulthandler.enable(file=open(os.path.join(_fault_dir, "mimic_datagen_fault.txt"), "w"))
except OSError:
    faulthandler.enable()

app_launcher = AppLauncher({k: v for k, v in vars(args_cli).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

"""이하 kit 부팅 후."""

import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab_mimic.datagen.generation import (  # noqa: E402
    env_loop,
    setup_async_generation,
    setup_env_config,
)
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths  # noqa: E402

import sim_to_real.tasks  # noqa: F401,E402  gym 등록
from sim_to_real.datagen.skillgen_planner import (  # noqa: E402
    SO101SkillGenPlanner,
    SO101SkillGenPlannerCfg,
)
from sim_to_real.tasks.common.mdp.recorders import SO101DatagenRecorderManagerCfg  # noqa: E402
from sim_to_real.tasks.pick_cube.mimic_env import EEF_NAME  # noqa: E402
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras  # noqa: E402
from sim_to_real.utils.constant import BOWL_NAME, CUBE_NAMES  # noqa: E402
from so101_contract.grasp_geometry import FIXED_INNER_CENTER  # noqa: E402
from so101_contract.grasp_manifold import project_pose_best_pan  # noqa: E402


#: 합성 place 구간 프레임 배분 — [정착 | 개방 | retreat | hold].
_SYNTH_SETTLE, _SYNTH_OPEN, _SYNTH_RETREAT, _SYNTH_HOLD = 6, 10, 12, 12
#: 그릇 중심 위 **큐브(pad)** 높이(m, 그릇 원점 기준). source 투하 높이 실측 중앙 135.4 mm
#: (범위 129~165, n=8)에 맞춘다. rim 은 75 mm 라 그 위에서 놓는다.
_SYNTH_HOVER_M = 0.135
#: 개방 후 물러나는 높이(m).
_SYNTH_RETREAT_M = 0.060


def _synth_place_waypoints(env, env_id: int, source_traj):
    """마지막 subtask 를 source 재생 대신 **합성**한다.

    ★왜 — source 의 place 구간은 그릇 기준으로 SE(3) 변환되는데, 그 구간이 담고 있는 동작
    (들어올리기·운반·홈 복귀)은 물리적으로 큐브·로봇 기준이다. 그래서 변환이 어긋나 전이가
    엉뚱한 방향으로 가고(사용자 관측: "그릇과 멀어지는 방향으로 이동했다가"), 홈 복귀는
    그릇 변위만큼 밀린다(실측 INIT 대비 14.8°, 최대 26.2°).

    여기서는 그 구간을 통째로 버리고 **그릇 pose 에서 직접** 만든다:

        cuRobo(접근) + subtask0(파지) + cuRobo(들어올리기+운반) + [합성: 투하·retreat·hold]

    첫 waypoint 가 곧 cuRobo 전이의 목표라, "그릇 위 hover" 를 첫 점으로 두면 운반을 planner
    가 담당한다. 그 뒤는 그리퍼 개방 → 상승 → 정지뿐이라 source 가 필요 없다.

    ★첫 waypoint 의 그리퍼는 **닫힌 값**이어야 한다. cuRobo 전이는 다음 구간 첫 waypoint 의
    그리퍼 명령을 그대로 쓰므로, 열린 값이면 운반 내내 열린 채라 큐브를 도중에 떨군다.
    """
    from isaaclab_mimic.datagen.waypoint import Waypoint

    # ★`list(source_traj)` 로 쓰면 안 된다. `WaypointTrajectory.__getitem__` 은 범위를 넘으면
    #   `IndexError` 가 아니라 **`AssertionError`** 를 던져서, 옛 반복 규약을 쓰는 `list()` 가
    #   종료하지 못하고 그대로 터진다(실측: generate 가 AssertionError 로 죽어 run 전멸).
    points = [source_traj[i] for i in range(len(source_traj))]
    closed_action = points[0].gripper_action     # 쥔 상태(구간 시작)
    open_action = points[-1].gripper_action      # 놓은 상태(구간 끝 = 꼬리 hold)
    reference = points[0].pose                   # dtype/device 를 그대로 물려받는다

    bowl = env.get_object_poses(env_ids=[env_id])[BOWL_NAME][0]
    with torch.no_grad():
        bowl_np = np.asarray(bowl.detach().cpu(), dtype=np.float64).reshape(4, 4)
        seed = env._canonical_joint_pos([env_id])[0, :5].detach().cpu().numpy().astype(np.float64)

    pad_offset = np.asarray(FIXED_INNER_CENTER, dtype=np.float64)
    base_rotation = np.asarray(points[0].pose.detach().cpu(), dtype=np.float64)[:3, :3]

    def pose_at(height: float) -> torch.Tensor:
        """**pad**(=큐브가 물려 있는 점)를 그릇 중심 위 `height` 에 놓는 tool pose.

        ★TCP 를 겨누면 안 된다. 큐브는 TCP 가 아니라 pad 에 있고, pad 은 top-down 자세에서
        TCP 로부터 world 기준 (14.7, 21.5, **−46.3**) mm 떨어져 있다. TCP 를 그릇 중심 위
        115 mm 에 두면 큐브는 수평 26 mm 어긋난 채(성공반경 60 mm 의 43%) 68.7 mm 에 놓여
        rim(75 mm)보다 낮다 — 그리퍼가 그릇 안으로 들어가 rim 을 친다.
        실측 실패: 최소 xy 20.5 mm 까지 갔다가 최종 178 mm 로 튕겨나감.

        pose = pad_target − R @ pad_offset 인데 R 은 투영 결과라 p 에 의존한다. 2회 반복이면
        수렴한다(manifold 투영은 위치를 보존하고 회전만 스냅하므로 변화가 작다).
        """
        pad_target = bowl_np[:3, 3] + np.array([0.0, 0.0, height])
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = base_rotation
        target[:3, 3] = pad_target - base_rotation @ pad_offset
        for _ in range(2):
            # 그릇 위 자세도 5-DOF 도달 manifold 로 투영한다 — env·planner 와 같은 함수.
            snapped = project_pose_best_pan(env.eef_ik, target, seed)
            target = snapped.copy()
            target[:3, 3] = pad_target - snapped[:3, :3] @ pad_offset
        return torch.as_tensor(target, dtype=reference.dtype, device=reference.device)

    hover, lifted = pose_at(_SYNTH_HOVER_M), pose_at(_SYNTH_HOVER_M + _SYNTH_RETREAT_M)
    plan: list[Waypoint] = []
    plan += [Waypoint(hover, closed_action, 0.0) for _ in range(_SYNTH_SETTLE)]
    plan += [Waypoint(hover, open_action, 0.0) for _ in range(_SYNTH_OPEN)]
    # ★retreat 은 **ease-in/out**(cosine)으로 낸다. 선형 램프는 정지 블록(settle·open) 직후
    #   첫 프레임에 속도가 계단으로 서서 "멈췄다 확 튄다"는 체감을 만든다. 정지 길이 자체는
    #   기능적(투하 안정화)이라 줄이지 않고, 재출발만 부드럽게 한다.
    for step in range(1, _SYNTH_RETREAT + 1):
        ratio = 0.5 * (1.0 - math.cos(math.pi * step / _SYNTH_RETREAT))
        blend = hover.clone()
        blend[:3, 3] = (1.0 - ratio) * hover[:3, 3] + ratio * lifted[:3, 3]
        plan.append(Waypoint(blend, open_action, 0.0))
    plan += [Waypoint(lifted, open_action, 0.0) for _ in range(_SYNTH_HOLD)]
    return plan


def _as_trajectory(waypoints: list):
    """`Waypoint` 리스트 → `WaypointTrajectory`.

    `DataGenerator.merge_eef_subtask_trajectory` 가 받는 타입이다. 합성 구간도 이걸 통해야
    진입 보간(`num_interpolation_steps`)이 붙는다.
    """
    from isaaclab_mimic.datagen.waypoint import WaypointSequence, WaypointTrajectory

    trajectory = WaypointTrajectory()
    trajectory.add_waypoint_sequence(WaypointSequence(sequence=list(waypoints)))
    return trajectory


def _install_generation_hooks(out_path: str | None) -> list[dict]:
    """`DataGenerator` 훅 3종을 **항상** 건다. `out_path` 가 있으면 구간 로그도 남긴다.

    두 가지를 한 곳에서 한다 — 둘 다 같은 지점을 감싸야 하기 때문이다:

    1. **관절 해 동승**(항상). `_convert_planned_trajectory_to_waypoints` 가 만든 전이
       waypoint 에 planner 의 관절 해를 붙이고, `MultiWaypoint.execute` 가 실행 직전 env 로
       넘긴다. upstream 은 `waypoint.pose` 만 env 에 주므로 이 우회가 없으면 env 가 pose 를
       다시 IK 로 풀고, 그 왕복이 `wrist_roll` 을 흔든다.
    2. **구간 계측**(`out_path` 있을 때만 JSON 기록). 증강본은 [전이 | 보간 | 합성 place] 가
       이어붙은 결과인데 HDF5 엔 경계가 안 남아 사후 역산이 불가능하다.

    Isaac Lab 소스는 건드리지 않는다.
    """
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from isaaclab_mimic.datagen.waypoint import MultiWaypoint

    episodes: list[dict] = []
    frames: list[str] = []
    cube_track: list[list[float]] = []   # 프레임별 [cube xyz, bowl xyz] (mm, world)

    def tag(waypoints, label: str, start: int = 0) -> None:
        for wp in list(waypoints)[start:]:
            wp._so101_segment = label

    picks: list[dict] = []   # subtask 별로 고른 source demo 인덱스

    orig_plan = DataGenerator._convert_planned_trajectory_to_waypoints
    orig_merge = DataGenerator.merge_eef_subtask_trajectory
    orig_exec = MultiWaypoint.execute
    orig_generate = DataGenerator.generate
    orig_select = DataGenerator.select_source_demo

    # ★호출부가 **키워드 인자**를 쓴다 — 파라미터 이름을 원본과 한 글자도 다르게 두면
    #   TypeError 로 생성이 통째로 죽는다.
    def select_wrapped(self, eef_name, eef_pose, object_pose,
                       src_demo_current_subtask_boundaries, subtask_object_name,
                       selection_strategy_name, selection_strategy_kwargs=None):
        index = orig_select(self, eef_name, eef_pose, object_pose,
                            src_demo_current_subtask_boundaries, subtask_object_name,
                            selection_strategy_name, selection_strategy_kwargs)
        # ★`generation_select_src_per_subtask=True` 라 subtask 마다 source 를 **따로** 고른다.
        #   grasp 를 A 데모에서, place 를 B 데모에서 가져오면 큐브가 그리퍼 안에 물린 위치가
        #   서로 달라 투하 지점이 그만큼 어긋난다("완주했는데 그릇에 없음" 가설). 어느 데모를
        #   골랐는지 남겨 실패와의 상관을 본다.
        picks.append({"ref": subtask_object_name, "src": int(index)})
        return index

    def plan_wrapped(self, motion_planner, gripper_action):
        result = orig_plan(self, motion_planner, gripper_action)
        tag(result, "curobo_transition")
        # ★planner 가 **이미 푼 관절 해**를 waypoint 에 실어 보낸다. upstream 은 `waypoint.pose`
        #   만 env 로 넘기므로(`waypoint.py`), env 는 그 pose 를 다시 IK 로 풀어야 한다 —
        #   5-DOF 에서 그 왕복은 항등이 아니고 매 프레임 pan 재스캔으로 wrist_roll 이 따라
        #   움직인다. 해를 같이 보내면 왕복 자체가 사라진다(`exec_wrapped` → env 가 소비).
        joints = motion_planner.get_planned_joint_positions()
        for waypoint, arm_q in zip(result, joints):
            waypoint._so101_plan_q = arm_q
        return result

    def merge_wrapped(self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_traj):
        label = f"subtask{subtask_index}"
        if subtask_index == len(self.env_cfg.subtask_configs[eef_name]) - 1:
            # ★마지막 subtask(place)는 **항상 합성**한다 — source 재생 분기는 없앴다.
            #   source 의 place 구간은 `object_ref=Bowl` 로 변환되는데 담고 있는 동작(들어올리기·
            #   운반·홈 복귀)은 물리적으로 큐브·로봇 기준이라 변환이 어긋난다. 합성 도입 후
            #   **투하 실패 0건**(이전엔 실패의 40 %)이라 분기를 유지할 이유가 없다.
            #
            # ★합성분도 **`orig_merge` 를 통과시킨다.** 예전엔 여기서 리스트를 그대로 반환해
            #   진입 보간이 **0** 이었다 — planner 도착점(IK 잔차 ≤6 mm + 투영 잔차)에서 첫
            #   `hover` 로 한 스텝에 점프해 슬루 상한을 후려쳤다. `merge_eef_subtask_trajectory`
            #   는 `WaypointTrajectory` 를 받으므로 감싸 주기만 하면 보간이 공짜로 붙는다.
            subtask_traj = _as_trajectory(_synth_place_waypoints(self.env, env_id, subtask_traj))
            label = "synth_place"
        n_src = len(subtask_traj)  # merge 가 소비할 수 있으니 호출 **전에** 잰다
        result = orig_merge(self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_traj)
        # 앞쪽은 진입 보간(num_interpolation_steps + num_fixed_steps), 뒤쪽 n_src 개가
        # 구간 본체다. 보간 길이는 pop_first 때문에 계산보다 실제 길이로 자른다.
        tag(result[: max(len(result) - n_src, 0)], "interp")
        tag(result, label, start=max(len(result) - n_src, 0))
        return result

    async def exec_wrapped(self, env, success_term, env_id=0, env_action_queue=None):
        waypoint = next(iter(self.waypoints.values()))
        frames.append(getattr(waypoint, "_so101_segment", "unknown"))
        # cuRobo 전이 waypoint 면 planner 관절 해를 env 로 넘긴다(`take_pending_plan_joints`).
        # 다른 구간에는 없으므로 env 가 기존 pose→IK 경로를 그대로 탄다.
        env.put_pending_plan_joints(env_id, getattr(waypoint, "_so101_plan_q", None))
        # ★생성 HDF5 에는 프레임별 물체 pose 가 없다(actions·applied_target·obs_x 뿐).
        #   물리 실패(완주했는데 그릇에 없음)를 수치로 짚으려면 큐브 궤적이 필요해서 여기서
        #   같이 적는다. 계측이 생성을 죽이면 안 되므로 통째로 감싼다.
        try:
            cube = env.scene[CUBE_NAMES[0]].data.root_pos_w[env_id]
            bowl = env.scene[BOWL_NAME].data.root_pos_w[env_id]
            # ★pad↔큐브 거리도 같이 잰다 — 실패 12/13 이 "한 번도 안 들림"이라 파지 자체가
            #   범인이고, 그 직접 지표가 이것이다. env 의 mimic 훅은 `datagen_info` 와 **같은
            #   base 프레임** pose 를 주므로 source(주석본)의 24.5 mm 와 바로 비교된다.
            #   world 좌표로 손수 변환하면 base↔USD affine 때문에 조용히 틀린다(실측 300 mm).
            eef = env.get_robot_eef_pose(EEF_NAME, env_ids=[env_id])[0]
            obj = env.get_object_poses(env_ids=[env_id])[CUBE_NAMES[0]][0]
            pad = eef[:3, 3] + eef[:3, :3] @ torch.as_tensor(
                FIXED_INNER_CENTER, dtype=eef.dtype, device=eef.device)
            gap = float(torch.linalg.vector_norm(pad - obj[:3, 3]) * 1000)
            cube_track.append([round(float(v) * 1000, 1) for v in (*cube, *bowl)] + [round(gap, 1)])
        except Exception:  # noqa: BLE001
            cube_track.append([])
        return await orig_exec(self, env, success_term, env_id, env_action_queue)

    async def generate_wrapped(self, *a, **kw):
        frames.clear()
        cube_track.clear()
        picks.clear()
        result = await orig_generate(self, *a, **kw)
        runs: list[list] = []
        for label in frames:
            if runs and runs[-1][0] == label:
                runs[-1][2] += 1
            else:
                runs.append([label, sum(r[2] for r in runs), 1])
        episode = {"success": bool(result.get("success")), "frames": len(frames),
                   "segments": [{"label": l, "start": s, "length": n} for l, s, n in runs],
                   "source_picks": list(picks),
                   # subtask 끼리 다른 source 를 골랐나(파지↔투하 기하 불일치 후보)
                   "src_mismatch": len({p["src"] for p in picks}) > 1}
        track = [row for row in cube_track if len(row) == 7]
        if track:
            arr = np.asarray(track)                       # (T, 7) mm — cube xyz, bowl xyz, pad gap
            xy = np.linalg.norm(arr[:, :2] - arr[:, 3:5], axis=1)
            episode["cube"] = {
                "final_xy_to_bowl_mm": round(float(xy[-1]), 1),
                "min_xy_to_bowl_mm": round(float(xy.min()), 1),
                "final_z_mm": round(float(arr[-1, 2]), 1),
                "max_z_mm": round(float(arr[:, 2].max()), 1),
                "max_z_frame": int(arr[:, 2].argmax()),
                "start_z_mm": round(float(arr[0, 2]), 1),
                "bowl_z_mm": round(float(arr[-1, 5]), 1),
                # subtask0(파지 구간) 안에서의 pad↔큐브 최소 거리 = 집게가 얼마나 잘 물었나.
                # source 성공 파지 실측 24.5 mm 가 기준선이다.
                "pad_gap_min_mm": round(float(arr[:, 6].min()), 1),
                "pad_gap_min_frame": int(arr[:, 6].argmin()),
                "pad_gap_at_subtask0_end_mm": round(float(arr[
                    min(next((s["start"] + s["length"] - 1 for s in episode["segments"]
                              if s["label"] == "subtask0"), len(arr) - 1), len(arr) - 1), 6]), 1),
            }
        episodes.append(episode)
        if out_path:
            with open(out_path, "w") as fp:
                json.dump({"episodes": episodes}, fp, indent=1)
        return result

    DataGenerator._convert_planned_trajectory_to_waypoints = plan_wrapped
    DataGenerator.merge_eef_subtask_trajectory = merge_wrapped
    DataGenerator.generate = generate_wrapped
    MultiWaypoint.execute = exec_wrapped
    DataGenerator.select_source_demo = select_wrapped
    return episodes


def _planner_step_size(env) -> float | None:
    """전이 retiming step — CLI 우선, 없으면 env cfg 값. 0 이면 retiming 끔(None)."""
    value = args_cli.planner_step_size
    if value is None:
        value = float(getattr(env.cfg, "mimic_planner_step_size", 0.05))
    return value if value > 0 else None


def _planner_cfg(env) -> SO101SkillGenPlannerCfg:
    """env cfg 의 로봇 경로·씬 물체를 planner cfg 로 옮긴다(값은 한 곳에서만 정의)."""
    return SO101SkillGenPlannerCfg(
        robot_yaml=env.cfg.mimic_robot_yaml,
        urdf_path=env.cfg.mimic_urdf_path,
        cube_names=tuple(CUBE_NAMES),
        bowl_name=BOWL_NAME,
        motion_step_size=_planner_step_size(env),
        motion_noise_scale=args_cli.motion_noise_scale,
        debug=args_cli.debug_planner,
    )


def _make_env(env_name: str, output_dir: str, output_file_name: str):
    # ★`--enable_cameras` 면 **SM datagen 과 같은 recorder** 를 쓴다. 그래야 생성본이
    # `scripts/convert/isaaclab2lerobotv3.py` 계약(`obs_x/joint_pos`·`obs_x/images/*`·
    # `applied_target`)을 그대로 만족해 변환기를 손대지 않고 LeRobot v3 로 넘어간다.
    # stock `ActionStateRecorderManagerCfg` 는 `actions`/`obs`/`states` 만 남겨 변환 불가다.
    # (대신 생성본에는 `obs/datagen_info` 가 없어 **다시 mimic source 로는 못 쓴다** —
    #  학습 데이터셋 용도다. source 가 필요하면 §5.6 SM 녹화본을 쓴다.)
    # 카메라 없이 도는 스모크는 이미지 obs 자체가 없으므로 stock 을 그대로 둔다.
    with_cameras = bool(getattr(args_cli, "enable_cameras", False))
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
        recorder_cfg=SO101DatagenRecorderManagerCfg() if with_cameras else None,
    )
    # 카메라는 `--enable_cameras` 일 때만 스폰한다(SM 과 같은 규약). 이미지 없는 스모크·self-test
    # 는 카메라 제거로 부팅/스텝을 크게 줄인다. 학습 데이터 생성은 --enable_cameras 로 켠다.
    if not getattr(args_cli, "enable_cameras", False):
        remove_pick_cube_cameras(env_cfg)
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError(f"{env_name} 은 ManagerBasedRLMimicEnv 자손이 아니다")
    return env, success_term


def _self_test(env) -> int:
    """실제 씬으로 planner 를 1회 왕복 검증한다. 0=PASS, 1=FAIL."""
    planner = SO101SkillGenPlanner(env=env, robot=env.scene["robot"],
                                   config=_planner_cfg(env), env_id=0)
    failures: list[str] = []

    # ① 프레임 정합: env 어댑터의 tool FK ↔ planner 의 cuRobo FK 가 같은 프레임인가.
    arm = env.scene["robot"].data.joint_pos[0, env.canonical_joint_idx][:5]
    adapter_pose = env.get_robot_eef_pose(EEF_NAME, env_ids=[0])[0]
    planner_pose = planner._fk_matrices(arm.reshape(1, 5).to("cuda"))[0]
    frame_err = float(torch.linalg.matrix_norm(adapter_pose - planner_pose).item())
    print(f"[self-test] adapter FK ↔ cuRobo FK 차이 = {frame_err:.6f}")
    if frame_err > 2e-3:
        failures.append(f"FK 프레임 불일치 {frame_err:.6f} (>2e-3) — URDF/tcp_grasp 정합 확인")

    # ② 물체 pose 가 EEF 와 같은 프레임인지 — 큐브가 상판 높이(solver z)에 있어야 한다.
    cube_z = float(env.get_object_poses(env_ids=[0])[CUBE_NAMES[0]][0, 2, 3].item())
    print(f"[self-test] 큐브 solver z = {cube_z:.4f} m")
    if not -0.02 < cube_z < 0.12:
        failures.append(f"큐브 solver z {cube_z:.4f} 가 상판 근처가 아니다 — 프레임 변환 확인")

    # ③ 홈(parked) 자세가 계획 start 로 성립하는가 — 제약 항목별로 확인.
    #    parked 자세는 sphere 모델상 4쌍이 sub-mm 겹쳐(false positive) 완화 없이는 전 계획이
    #    전멸한다. 여기서 이름으로 잡아둔다 — 안 그러면 "그냥 계획 실패"로만 보인다.
    planner.update_world()
    start_arm = planner._clamp_to_urdf_limits(planner._current_arm_rad())
    diagnosis = planner.diagnose_state(start_arm)
    print(f"[self-test] {diagnosis}")
    if any(part.split("=")[1] != "0.0000" for part in diagnosis.split(": ")[-1].split()):
        failures.append(f"start 상태가 이미 infeasible — {diagnosis}")

    # ④ 도달 가능한 목표(현재 자세에서 +3cm 위)로 전이 계획이 되는가.
    target = adapter_pose.clone()
    target[2, 3] += 0.03
    ok = planner.update_world_and_plan_motion(target_pose=target, expected_attached_object=None,
                                              env_id=0, step_size=planner.step_size)
    poses = planner.get_planned_poses()
    print(f"[self-test] plan_cspace ok={ok} waypoints={len(poses)}")
    if not ok or len(poses) < 2:
        failures.append("전이 계획 실패 — 목표가 도달 가능한데도 계획이 안 나왔다")
    else:
        reached = float(torch.linalg.vector_norm(poses[-1][:3, 3] - target[:3, 3]).item())
        print(f"[self-test] 도착 위치 오차 = {reached * 1000:.1f} mm")
        if reached > 0.02:
            failures.append(f"도착 위치 오차 {reached * 1000:.1f}mm > 20mm")

    # ⑤ waypoint 이터레이터가 계획 길이와 맞는가.
    planner.reset_plan()
    if planner.has_next_waypoint():
        failures.append("reset_plan 후에도 waypoint 가 남아 있다")

    for line in failures:
        print(f"[self-test] FAIL: {line}")
    print("[self-test] " + ("PASS" if not failures else f"FAIL ({len(failures)})"))
    return 1 if failures else 0


def main() -> None:
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    if args_cli.record_segments and args_cli.num_envs != 1:
        parser.error("--record_segments 는 --num_envs 1 전용(구간 로그가 env 간 섞인다)")
    # 훅은 **항상** 건다 — 관절 해 동승이 여기 들어 있다(구간 JSON 만 opt-in).
    _install_generation_hooks(args_cli.record_segments)

    env, success_term = _make_env(env_name, output_dir, output_file_name)

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)
    env.reset()

    if args_cli.self_test:
        code = _self_test(env)
        env.close()
        simulation_app.close()
        raise SystemExit(code)

    motion_planners = None
    if args_cli.no_skillgen:
        env.cfg.datagen_config.use_skillgen = False
        print("[mimic] SkillGen 끔 — 순수 Mimic 보간 전이로 생성")
    else:
        env.cfg.datagen_config.use_skillgen = True
        motion_planners = {}
        for env_id in range(args_cli.num_envs):
            print(f"[mimic] env {env_id} planner 초기화")
            motion_planners[env_id] = SO101SkillGenPlanner(
                env=env, robot=env.scene["robot"], config=_planner_cfg(env), env_id=env_id)

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,
    )

    data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
    try:
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("[mimic] tasks cancelled")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[mimic] async 정리 중 오류: {exc}")
        if motion_planners is not None:
            motion_planners.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[mimic] 사용자 중단")
    simulation_app.close()
