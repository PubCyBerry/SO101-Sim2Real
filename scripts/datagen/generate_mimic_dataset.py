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
                         "[cuRobo 전이 | 보간 | subtask source 재생] 이 이어붙은 결과인데 "
                         "HDF5 엔 그 경계가 안 남아 사후 역산이 불가능하다. `--num_envs 1` 전용")
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


def _install_segment_recorder(out_path: str) -> list[dict]:
    """생성 프레임마다 어느 구간에서 나왔는지 기록한다(관찰용, opt-in).

    `DataGenerator.generate` 는 매 루프 1 waypoint 를 실행하고, 그 waypoint 는 두 곳
    중 하나에서 온다 — `_convert_planned_trajectory_to_waypoints`(cuRobo 전이) 또는
    `merge_eef_subtask_trajectory`(진입 보간 + source subtask 구간). 그래서 두 생성지에
    waypoint 를 태깅하고 `MultiWaypoint.execute` 에서 태그를 읽어 프레임 순서대로 쌓는다.
    Isaac Lab 소스는 건드리지 않는다.
    """
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from isaaclab_mimic.datagen.waypoint import MultiWaypoint

    episodes: list[dict] = []
    frames: list[str] = []

    def tag(waypoints, label: str, start: int = 0) -> None:
        for wp in list(waypoints)[start:]:
            wp._so101_segment = label

    orig_plan = DataGenerator._convert_planned_trajectory_to_waypoints
    orig_merge = DataGenerator.merge_eef_subtask_trajectory
    orig_exec = MultiWaypoint.execute
    orig_generate = DataGenerator.generate

    def plan_wrapped(self, motion_planner, gripper_action):
        result = orig_plan(self, motion_planner, gripper_action)
        tag(result, "curobo_transition")
        return result

    def merge_wrapped(self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_traj):
        n_src = len(subtask_traj)  # merge 가 소비할 수 있으니 호출 **전에** 잰다
        result = orig_merge(self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_traj)
        # 앞쪽은 진입 보간(num_interpolation_steps + num_fixed_steps), 뒤쪽 n_src 개가
        # source 구간이다. 보간 길이는 pop_first 때문에 계산보다 실제 길이로 자른다.
        tag(result[: max(len(result) - n_src, 0)], "interp")
        tag(result, f"subtask{subtask_index}", start=max(len(result) - n_src, 0))
        return result

    async def exec_wrapped(self, env, success_term, env_id=0, env_action_queue=None):
        frames.append(getattr(next(iter(self.waypoints.values())), "_so101_segment", "unknown"))
        return await orig_exec(self, env, success_term, env_id, env_action_queue)

    async def generate_wrapped(self, *a, **kw):
        frames.clear()
        result = await orig_generate(self, *a, **kw)
        runs: list[list] = []
        for label in frames:
            if runs and runs[-1][0] == label:
                runs[-1][2] += 1
            else:
                runs.append([label, sum(r[2] for r in runs), 1])
        episodes.append({"success": bool(result.get("success")), "frames": len(frames),
                         "segments": [{"label": l, "start": s, "length": n} for l, s, n in runs]})
        with open(out_path, "w") as fp:
            json.dump({"episodes": episodes}, fp, indent=1)
        return result

    DataGenerator._convert_planned_trajectory_to_waypoints = plan_wrapped
    DataGenerator.merge_eef_subtask_trajectory = merge_wrapped
    DataGenerator.generate = generate_wrapped
    MultiWaypoint.execute = exec_wrapped
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

    if args_cli.record_segments:
        if args_cli.num_envs != 1:
            parser.error("--record_segments 는 --num_envs 1 전용(구간 로그가 env 간 섞인다)")
        _install_segment_recorder(args_cli.record_segments)

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
