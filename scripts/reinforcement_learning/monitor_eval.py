"""PickCube 모니터 — 체크포인트의 단계별 성공률을 env-type 별로 집계 + 그리드 비디오.

- env-type: scratch(정상 시작) / full-grasp 부트스트랩 / pre-grasp 부트스트랩 을 따로 집계.
  (eval 환경의 grasp_bootstrap_prob 를 고정값으로 두고, PickCubeEnv.bootstrap_kind 로 분류.)
- 단계: reach → grasp → lift → over-bowl → placed(그릇 안) → success(termination).
- 비디오: num_envs(기본 16) 그리드를 한 화면에 담는 뷰포트로 녹화. multi-env 조명 과노출은
  DomeLight/DistantLight intensity 를 1/num_envs 로 스케일해 해결(무한 광원이라 합이 원래값 유지).

사용법:
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$(pwd)/src .venv/bin/python \
      scripts/reinforcement_learning/monitor_eval.py --recurrent --obs_normalization \
      --checkpoint outputs/.../model_300.pt --num_envs 16 --num_episodes 48 \
      --active_objects 1 --bootstrap_prob 0.5 --video --device cuda:0
"""

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PickCube 단계별 모니터 + 비디오")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=48, help="집계 최소 에피소드 수")
parser.add_argument("--max_steps", type=int, default=4000, help="안전 상한(스텝)")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--rl_device", default=None)
parser.add_argument("--obs_group", default="rl_policy")
parser.add_argument("--clip_actions", type=float, default=1.0)
# 분류용 부트스트랩 확률(고정, annealing 없음). 0=scratch만, 1=부트스트랩만.
parser.add_argument("--bootstrap_prob", type=float, default=0.6,
                    help="부트스트랩 비율(고정). scratch/부트스트랩 분류용.")
parser.add_argument("--bootstrap_close", type=float, default=-0.15)
parser.add_argument("--pregrasp_frac", type=float, default=0.5,
                    help="부트스트랩 env 중 pre-grasp 비율(나머지 full-grasp). 세 그룹 동시 측정.")
parser.add_argument("--force_kind", type=int, default=0, choices=[0, 1, 2],
                    help="비디오 데모: 0=혼합(기본), 1=전부 full-grasp, 2=전부 pre-grasp. "
                         "설정 시 grasp_offset 캐시 후 전체 강제 부트스트랩 → step0부터 해당 상태 녹화.")
# 카메라 뷰(그리드 span 배수). eye=중심+(side·-back·height), target=중심+(-0.1·0.05·look)·span.
parser.add_argument("--cam_side", type=float, default=0.85,
                    help="카메라 좌우 위치(+오른쪽/-왼쪽). span 배수.")
parser.add_argument("--cam_back", type=float, default=1.05,
                    help="카메라 뒤(-y) 거리. 클수록 멀리. span 배수.")
parser.add_argument("--cam_height", type=float, default=0.40,
                    help="카메라 높이. 작을수록 낮은(수평) 뷰. span 배수.")
parser.add_argument("--cam_look", type=float, default=0.18,
                    help="목표점 높이(고개 들기). 클수록 위를 봄(하늘↑), 작을수록 책상 집중. span 배수.")
# 절대 좌표 카메라(world). 지정 시 span 기반 자동 프레이밍 대신 이 eye/target 고정 사용.
# 기본값 = GUI 에서 보정한 뷰(num_envs=16 그리드 기준 한 env 클로즈업).
parser.add_argument("--cam_eye", type=float, nargs=3, default=[-2.739, 5.835, 1.811],
                    help="카메라 위치(world xyz). span 자동프레이밍 쓰려면 --no_cam_abs.")
parser.add_argument("--cam_target", type=float, nargs=3, default=[-2.206, 5.025, 1.565],
                    help="카메라가 보는 점(world xyz).")
parser.add_argument("--no_cam_abs", action="store_true", default=False,
                    help="절대좌표 카메라 끄고 span 기반 자동 프레이밍(--cam_side/back/height/look) 사용.")
parser.add_argument("--gui", action="store_true", default=False,
                    help="Isaac Sim GUI 로 추론 실행(headless 해제, 녹화·집계 없음). 뷰포트에서 "
                         "직접 카메라 이동, 5초마다 현재 eye/target 출력. 원하는 뷰 값 확인용.")
# 비디오
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=450, help="녹화 길이(step). 450≈15s.")
parser.add_argument("--video_dir", default=None)
# 단계 임계값
parser.add_argument("--reach_thr", type=float, default=0.06)
parser.add_argument("--grasp_thr", type=float, default=0.06)
parser.add_argument("--close_thr", type=float, default=0.50)
parser.add_argument("--lift_min", type=float, default=0.02)
parser.add_argument("--lift_stage", type=float, default=0.05)
# 정책 아키텍처(체크포인트와 일치)
parser.add_argument("--recurrent", action="store_true", default=False)
parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"])
parser.add_argument("--rnn_hidden_dim", type=int, default=256)
parser.add_argument("--rnn_num_layers", type=int, default=1)
parser.add_argument("--obs_normalization", action="store_true", default=False)
parser.add_argument("--init_noise_std", type=float, default=0.5)
parser.add_argument("--active_objects", type=int, default=1, choices=[1, 2, 3, 4])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui  # GUI 모드면 headless 해제(뷰포트 표시)
if args.video:
    args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import sim_to_real  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    apply_curriculum as apply_cube_curriculum,
    BOWL_CENTER_XY, BOWL_SUCCESS_RADIUS, BOWL_HEIGHT_RANGE,
)
from sim_to_real.tasks.pick_pen.mdp.rewards import (  # noqa: E402
    _get_gripper_pos, _pen_inside_cup_mask, _DESK_TOP_Z,
)
from sim_to_real.utils.constant import CUBE_NAMES  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


def _build_eval_cfg() -> dict:
    rl_device = args.rl_device if args.rl_device is not None else args.device
    policy_cfg = {
        "class_name": "ActorCritic",
        "init_noise_std": args.init_noise_std,
        "actor_hidden_dims": [256, 128] if args.recurrent else [128, 128],
        "critic_hidden_dims": [256, 128] if args.recurrent else [128, 128],
        "activation": "elu",
        "actor_obs_normalization": args.obs_normalization,
        "critic_obs_normalization": args.obs_normalization,
    }
    if args.recurrent:
        policy_cfg.update({
            "class_name": "ActorCriticRecurrent",
            "rnn_type": args.rnn_type,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "rnn_num_layers": args.rnn_num_layers,
        })
    return {
        "seed": args.seed, "device": rl_device, "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1, "experiment_name": "monitor",
        "run_name": "", "resume": False, "load_run": ".*",
        "load_checkpoint": "model_.*.pt", "logger": "tensorboard",
        "obs_groups": {"policy": [args.obs_group], "critic": [args.obs_group]},
        "policy": policy_cfg,
        "algorithm": {
            "class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1,
            "learning_rate": 3e-4, "schedule": "fixed", "gamma": 0.99, "lam": 0.95,
            "entropy_coef": 0.0, "desired_kl": 0.01, "max_grad_norm": 1.0,
            "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
        },
    }


def _normalize_lights() -> None:
    """무한 광원(Dome/Distant) 과노출 방지 — 타입별 개수로 정규화(각 light intensity /= 개수).

    광원이 /World 단일(권장 구조)이면 개수 1 → no-op. 혹시 per-env 복제(구 scene.usd)면
    개수 N → 1/N 로 자동 정규화돼 단일-light 등가 노출 유지. scene 버전에 무관하게 robust.
    """
    try:
        import omni.usd
        from pxr import UsdLux
        stage = omni.usd.get_context().get_stage()
        domes, distants = [], []
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                domes.append(UsdLux.DomeLight(prim))
            elif prim.IsA(UsdLux.DistantLight):
                distants.append(UsdLux.DistantLight(prim))
        for group in (domes, distants):
            k = len(group)
            if k <= 1:
                continue
            for light in group:
                attr = light.GetIntensityAttr()
                cur = attr.Get()
                if cur is not None:
                    attr.Set(float(cur) / float(k))
        print(json.dumps({"dome_lights": len(domes), "distant_lights": len(distants)}), flush=True)
    except Exception as exc:  # 조명 정규화 실패는 치명적 아님(녹화만 과노출)
        print(json.dumps({"lights_normalize_error": str(exc)}), flush=True)


def _hide_ceiling() -> None:
    """천장 prim 을 비가시화 — 부감(top-down/oblique) 녹화 시 책상/로봇을 가리지 않게."""
    try:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        n = 0
        for prim in stage.Traverse():
            if "Ceiling" in prim.GetPath().pathString:
                UsdGeom.Imageable(prim).MakeInvisible()
                n += 1
        print(json.dumps({"ceiling_hidden": n}), flush=True)
    except Exception as exc:
        print(json.dumps({"ceiling_hide_error": str(exc)}), flush=True)


def _print_viewport_camera() -> None:
    """GUI 활성 뷰포트의 현재 카메라 eye/target 을 set_camera_view 형식으로 출력."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        import omni.usd
        from pxr import UsdGeom, Gf
        vp = get_active_viewport()
        stage = omni.usd.get_context().get_stage()
        cam_path = str(vp.camera_path)
        cam = stage.GetPrimAtPath(cam_path)
        xf = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0.0)
        eye = xf.ExtractTranslation()
        fwd = xf.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))  # 카메라 -Z = 보는 방향
        tgt = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
        print(json.dumps({
            "viewport_camera": cam_path,
            "eye": [round(eye[0], 3), round(eye[1], 3), round(eye[2], 3)],
            "target": [round(tgt[0], 3), round(tgt[1], 3), round(tgt[2], 3)],
        }), flush=True)
    except Exception as exc:
        print(json.dumps({"viewport_cam_error": str(exc)}), flush=True)


def main() -> None:
    device = args.device
    rl_device = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        apply_cube_curriculum(
            env_cfg, active_objects=args.active_objects,
            object_radius_scale=1.0, container_angle_scale=1.0, container_radius_scale=1.0,
        )
        # 고정 부트스트랩(annealing off) — env-type 분류용
        if hasattr(env_cfg, "grasp_bootstrap_prob"):
            env_cfg.grasp_bootstrap_prob = args.bootstrap_prob
            env_cfg.grasp_bootstrap_close = args.bootstrap_close
            env_cfg.grasp_bootstrap_anneal_steps = 0.0
            if args.force_kind > 0:
                # 데모: 전부 한 종류로 강제(full-grasp=1, pre-grasp=2)
                env_cfg.grasp_bootstrap_prob = 1.0
                env_cfg.grasp_bootstrap_prob_final = 1.0
                env_cfg.grasp_bootstrap_pregrasp_frac = 1.0 if args.force_kind == 2 else 0.0
            else:
                env_cfg.grasp_bootstrap_prob_final = args.bootstrap_prob  # 감쇠 없음
                env_cfg.grasp_bootstrap_pregrasp_frac = args.pregrasp_frac  # 고정 pre-grasp 비율

        env = gym.make(args.task, cfg=env_cfg,
                       render_mode="rgb_array" if args.video else None)

        if args.video:
            vdir = args.video_dir or os.path.join(
                os.path.dirname(os.path.abspath(args.checkpoint)), "videos", "monitor")
            os.makedirs(vdir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env, video_folder=vdir,
                step_trigger=lambda step: step == 0,
                video_length=args.video_length, disable_logger=True,
            )
            print(json.dumps({"video_dir": vdir, "length": args.video_length}), flush=True)

        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)
        torch.manual_seed(args.seed)

        runner = OnPolicyRunner(env, _build_eval_cfg(), log_dir=None, device=rl_device)
        runner.load(args.checkpoint, load_optimizer=False, map_location=rl_device)
        policy = runner.get_inference_policy(device=rl_device)

        base = env.unwrapped
        n = base.num_envs
        cube_cfg = SceneEntityCfg(CUBE_NAMES[0])
        robot_cfg = SceneEntityCfg("robot", body_names=["gripper"])
        cup_cfg = SceneEntityCfg("Bowl")

        # 그리드를 한 화면에 담는 뷰포트 (조명 스케일도 여기서)
        obs = env.get_observations()
        if args.gui:
            _normalize_lights()
            _hide_ceiling()  # 자유 시점에서 책상 가림 방지(원하면 GUI Stage 에서 다시 켜기)
        if args.video:
            _normalize_lights()
            _hide_ceiling()
            if not args.no_cam_abs:
                # 절대좌표 고정 뷰(GUI 보정값). num_envs=16 그리드 기준.
                eye = tuple(args.cam_eye)
                tgt = tuple(args.cam_target)
            else:
                # span 기반 자동 프레이밍(실제 큐브 world 좌표로 그리드 전체).
                cw = base.scene[CUBE_NAMES[0]].data.root_pos_w  # (N,3)
                ctr = cw.mean(dim=0)
                span_x = float(cw[:, 0].max() - cw[:, 0].min())
                span_y = float(cw[:, 1].max() - cw[:, 1].min())
                span = max(span_x, span_y, 1.5)
                cx, cy, cz = float(ctr[0]), float(ctr[1]), float(ctr[2])
                eye = (cx + span * args.cam_side, cy - span * args.cam_back, cz + span * args.cam_height)
                tgt = (cx - span * 0.10, cy + span * 0.05, cz + span * args.cam_look)
            try:
                base.sim.set_camera_view(eye=eye, target=tgt)
                print(json.dumps({"cam_eye": [round(e, 3) for e in eye],
                                  "cam_target": [round(t, 3) for t in tgt]}), flush=True)
            except Exception as exc:
                print(json.dumps({"camera_view_error": str(exc)}), flush=True)

        # 데모: 첫 에피소드(=초기 reset, grasp_offset 미캐시라 항상 scratch)를 건너뛰고
        # 전체를 강제 부트스트랩 → step0 부터 full/pre-grasp 상태가 녹화되게 한다.
        if args.force_kind > 0:
            base._cache_grasp_geom()
            base._reset_idx(torch.arange(n, device=device))
            obs = env.get_observations()
            print(json.dumps({"force_kind": args.force_kind,
                              "kind_counts": base.bootstrap_kind.bincount(minlength=3).tolist()}),
                  flush=True)

        # GUI 모드: 무한 추론 루프 + 5초마다 현재 뷰포트 카메라 출력(집계·녹화 없음).
        if args.gui:
            import time
            print("[GUI] 뷰포트 카메라 조작: Alt+좌클릭=회전, Alt+중클릭=이동(pan), 스크롤=줌. "
                  "원하는 뷰를 잡으면 아래 출력되는 eye/target 값을 알려주세요. (창 닫으면 종료)",
                  flush=True)
            last = time.time()
            with torch.inference_mode():
                while simulation_app.is_running():
                    actions = policy(obs)
                    obs, _rew, _dones, _ = env.step(actions)
                    if time.time() - last > 5.0:
                        _print_viewport_camera()
                        last = time.time()
            return

        # env-type 별 단계 카운터: 0=scratch,1=full-grasp,2=pre-grasp
        stages = ["reach", "grasp", "lift", "over_bowl", "placed", "success"]
        counts = {k: {s: 0 for s in stages} for k in (0, 1, 2)}
        ep_count = {0: 0, 1: 0, 2: 0}

        # 에피소드 내 "단계 도달" 누적 플래그
        ever = {s: torch.zeros(n, dtype=torch.bool, device=device) for s in stages[:-1]}

        def cube_pos():
            return base.scene[CUBE_NAMES[0]].data.root_pos_w  # (N,3)

        max_steps = args.max_steps
        step = 0
        total_eps = 0
        with torch.inference_mode():
            while total_eps < args.num_episodes and step < max_steps:
                kind_snap = base.bootstrap_kind.clone()  # 이번 에피소드 종류(리셋 전)
                actions = policy(obs)
                obs, _rew, dones, extras = env.step(actions)
                step += 1

                # 현재 상태 단계 판정
                ee = _get_gripper_pos(base, robot_cfg)             # (N,3)
                cp = cube_pos()
                dist = torch.linalg.vector_norm(cp - ee, dim=1)
                gripper = base.scene["robot"].data.joint_pos[:, -1]
                closed = gripper < args.close_thr
                local_z = cp[:, 2] - base.scene.env_origins[:, 2]
                lifted_min = local_z > (_DESK_TOP_Z + args.lift_min)
                lifted_stage = local_z > (_DESK_TOP_Z + args.lift_stage)
                placed = _pen_inside_cup_mask(
                    base, cp, BOWL_CENTER_XY, BOWL_SUCCESS_RADIUS, BOWL_HEIGHT_RANGE, cup_cfg)
                cxw, cyw = BOWL_CENTER_XY
                over_bowl = (torch.hypot(
                    (cp[:, 0] - base.scene.env_origins[:, 0]) - cxw,
                    (cp[:, 1] - base.scene.env_origins[:, 1]) - cyw) < 0.10) & lifted_min

                reached = dist < args.reach_thr
                grasped = (dist < args.grasp_thr) & closed & lifted_min

                done_mask = dones.bool() if dones.dtype != torch.bool else dones
                cont = ~done_mask
                # 연속 중인 env 만 OR (done env 의 post-reset 상태 오염 방지)
                ever["reach"] |= reached & cont
                ever["grasp"] |= grasped & cont
                ever["lift"] |= lifted_stage & cont
                ever["over_bowl"] |= over_bowl & cont
                ever["placed"] |= placed & cont

                if done_mask.any():
                    time_outs = extras.get("time_outs", None)
                    if time_outs is None:
                        time_outs = torch.zeros_like(done_mask)
                    success = done_mask & (~time_outs.bool())
                    idxs = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
                    for i in idxs:
                        k = int(kind_snap[i].item())
                        ep_count[k] += 1
                        total_eps += 1
                        # success termination 이 발화한 step 은 done=True 라
                        # cont(~done) 게이트에 막혀 그 step 의 placed/over_bowl 가
                        # ever[] 에 누적되지 않는다. 성공은 정의상 모든 선행 단계를
                        # 통과한 것이므로 함의 처리(단조성 보장: success⊂placed⊂...).
                        succ_i = bool(success[i])
                        for s in stages[:-1]:
                            if bool(ever[s][i]) or succ_i:
                                counts[k][s] += 1
                        if succ_i:
                            counts[k]["success"] += 1
                    # 종료 env 플래그 리셋
                    for s in stages[:-1]:
                        ever[s][done_mask] = False

        def rates(k):
            e = ep_count[k]
            return {s: (round(counts[k][s] / e, 3) if e else None) for s in stages}

        result = {
            "status": "ok",
            "checkpoint": args.checkpoint,
            "num_envs": n,
            "episodes_total": total_eps,
            "episodes": {"scratch": ep_count[0], "full_grasp": ep_count[1], "pre_grasp": ep_count[2]},
            "stage_success": {
                "scratch": rates(0),
                "full_grasp": rates(1),
                "pre_grasp": rates(2),
            },
            "note": "scratch=정상시작(진짜 실력). full/pre_grasp=부트스트랩(하류 역량).",
        }
        print(json.dumps(result), flush=True)

    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "traceback": traceback.format_exc()}), flush=True)
        sys.exit(1)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
