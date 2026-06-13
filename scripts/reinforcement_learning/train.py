"""TB.2/TB.3 PPO 학습 스크립트 — SO-101 PickCube/PickPen.

사용법:
    # PickCube 상태 기반 (기본값 — rl_policy 그룹 사용)
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 \
        --num_envs 64 --device cuda:0 --max_iterations 200

    # North Star 6-dim 정책만 사용
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 --obs_group policy \
        --num_envs 64 --device cuda:0 --max_iterations 200

    # 비대칭 AC: actor=rl_policy, critic=rl_policy
    uv run python scripts/reinforcement_learning/train.py \
        --task SimToReal-SO101-PickCube-v0 \
        --obs_group rl_policy --critic_obs_group rl_policy \
        --num_envs 64 --device cuda:0 --max_iterations 200
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TB.2 PPO training")
parser.add_argument("--task", default="SimToReal-SO101-PickCube-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--rl_device", default=None, help="RL 연산 디바이스 (기본값: --device와 동일)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--num_learning_epochs", type=int, default=20,
                    help="PPO update당 learning epoch 수. contact-rich grasp 학습을 위해 기본 20.")
parser.add_argument("--num_mini_batches", type=int, default=4,
                    help="PPO minibatch 수")
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--experiment_name", default="so101_pick_cube_ppo")
parser.add_argument("--run_name", default="")
parser.add_argument("--log_root_path", default="outputs/rl/rsl_rl")
parser.add_argument("--checkpoint_dir", default=None, help="체크포인트 저장 경로 (스모크 테스트용)")
parser.add_argument("--clip_actions", type=float, default=1.0)
parser.add_argument(
    "--obs_group",
    default="rl_policy",
    help="actor 에 사용할 obs 그룹 이름 (기본값: rl_policy)",
)
parser.add_argument(
    "--critic_obs_group",
    default=None,
    help="critic 에 사용할 obs 그룹 이름 (기본값: --obs_group 과 동일)",
)
# 커리큘럼 파라미터 — gym.make() 이전에 env_cfg 에 적용
parser.add_argument("--active_objects", "--active_pens", dest="active_objects",
                    type=int, default=4, choices=[1, 2, 3, 4],
                    help="학습에 사용할 대상 수 (1~4, 기본값: 4). --active_pens는 호환 alias.")
parser.add_argument("--object_radius_scale", "--pen_radius_scale", dest="object_radius_scale",
                    type=float, default=1.0,
                    help="대상 reset ellipse 반경 배율. --pen_radius_scale은 호환 alias.")
parser.add_argument("--container_angle_scale", "--cup_angle_scale", dest="container_angle_scale",
                    type=float, default=1.0,
                    help="그릇/컵 reset 각도 범위 배율. --cup_angle_scale은 호환 alias.")
parser.add_argument("--container_radius_scale", "--cup_radius_scale", dest="container_radius_scale",
                    type=float, default=1.0,
                    help="그릇/컵 안 판정 반경 배율. --cup_radius_scale은 호환 alias.")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="에피소드 길이(초) override (기본값: env 설정값 30.0)")
parser.add_argument("--skill", default="full", choices=["full", "acquire", "place", "full_bc"],
                    help="skill chaining 프리셋(PickCube): acquire(=skill1, '그릇 위 grasp' "
                         "도달 즉시 종료·grasp 점화 dense) / place(=skill2, lower+open·단기 "
                         "horizon 5s·grasp_close 제거) / full(기존 전체 task) / full_bc(BC "
                         "warmstart 의 단일 end-to-end finetune·camp-free·require_open). full_bc 는 "
                         "--resume_checkpoint <bc> --demo_reset_prob <r> --demo_dataset_dir <SM demos> 와 쓴다.")
parser.add_argument("--grasp_bootstrap_prob", type=float, default=0.0,
                    help="초기상태 grasp 부트스트랩 비율(0~1). reset 시 이 비율의 env 를 큐브-인-그리퍼로 시작.")
parser.add_argument("--grasp_bootstrap_close", type=float, default=-0.15,
                    help="부트스트랩 시 gripper 닫힘 각(rad). -0.15 가 30mm 큐브 held 0.94.")
parser.add_argument("--grasp_bootstrap_prob_final", type=float, default=0.0,
                    help="부트스트랩 prob 를 이 값으로 선형 감쇠(annealing). 정상-env grasp 학습 압력↑.")
parser.add_argument("--grasp_bootstrap_anneal_iters", type=int, default=0,
                    help="부트스트랩 감쇠 구간(iteration). 0=감쇠 없음. steps=iters×num_steps_per_env.")
parser.add_argument("--grasp_bootstrap_pregrasp_frac", type=float, default=-1.0,
                    help="grasp 부트스트랩 중 pre-grasp(큐브 옆 그리퍼 open→닫기 연습) 비율 고정값. "
                         "-1=anneal 진행도 p(초반 full→후반 pre). 0.5=절반 close 연습(grasp 점화용).")
parser.add_argument("--place_bootstrap_prob", type=float, default=0.0,
                    help="place 부트스트랩 비율(0~1). grasp 부트스트랩 후 남은 scratch env 중 이 비율을 큐브-그릇위로 시작.")
# demo-state reset (RFCL reverse curriculum) — SM 성공 궤적 상태를 reset 분포로 주입
parser.add_argument("--demo_reset_prob", type=float, default=0.0,
                    help="reset 시 SM 데모 상태로 시작할 env 비율(0~1). place 탐색 valley 우회.")
parser.add_argument("--demo_dataset_dir", default=None,
                    help="demo_*.pt 디렉터리(pick_cube_state_machine --record_demos 산출).")
parser.add_argument("--demo_anneal_iters", type=int, default=0,
                    help="reverse curriculum 구간(iteration). 초반 success 근처→시작쪽 확장. 0=전구간 uniform.")
parser.add_argument("--demo_subsample", type=int, default=2, help="데모 궤적 매 k step 만 적재(메모리).")
parser.add_argument("--demo_max_files", type=int, default=4000, help="적재할 demo 파일 상한.")
# 진행 모니터링용 주기적 에피소드 비디오 녹화
parser.add_argument("--video", action="store_true", default=False,
                    help="학습 중 주기적으로 에피소드 비디오 녹화(headless offscreen). enable_cameras 자동 on.")
parser.add_argument("--video_length", type=int, default=450,
                    help="녹화 길이(policy step 수, 30Hz 기준 450≈15s)")
parser.add_argument("--video_interval", type=int, default=1500,
                    help="녹화 간격(policy step 수). 이 step 마다 1회 녹화 시작.")
parser.add_argument("--resume_checkpoint", default=None,
                    help="이어학습 체크포인트 경로 (.pt). 설정 시 learn() 전 로드.")
parser.add_argument("--resume_without_optimizer", action="store_true",
                    help="체크포인트에서 policy/value만 로드하고 optimizer state는 새 설정으로 시작")
parser.add_argument("--init_noise_std", type=float, default=0.5,
                    help="ActorCritic 초기 action noise std")
parser.add_argument("--override_policy_std", type=float, default=None,
                    help="resume checkpoint 로드 후 policy action std를 이 값으로 덮어씀")
parser.add_argument("--entropy_coef", type=float, default=0.005,
                    help="PPO entropy coefficient")
parser.add_argument("--learning_rate", type=float, default=3e-4,
                    help="PPO learning rate")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="PPO 할인율. 긴 지평(900 step/30s)엔 0.997 권장.")
parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--grasp_close_weight", type=float, default=None,
                    help="grasp_close_cube reward weight override(미지정 시 cfg 기본 3.0). "
                         "점화(3.0) 후 resume 시 낮춰(예 1.0) camp hold income↓ → carry 유도.")
parser.add_argument("--grasp_align_weight", type=float, default=None,
                    help="grasp_align_cube reward weight override(미지정 시 cfg 기본 1.0). "
                         "carry phase 서 0 으로 두면 per-step 상태보상 camp(align hover) 차단.")
parser.add_argument("--carry_rc_anneal_iters", type=int, default=0,
                    help="carry 역커리큘럼: full-grasp env 그릇을 든 큐브 밑(f=0)→정상 arc(f=1) 로 "
                         "이 iters 동안 이동. 0=비활성. release→short→long carry backward 학습.")
parser.add_argument("--rnd", action="store_true", default=False,
                    help="Random Network Distillation(내재 탐색 보상) 사용 — grasp 탐색 벽 공략.")
parser.add_argument("--rnd_weight", type=float, default=0.5,
                    help="RND 내재 보상 weight(초당; 내부에서 step_dt 곱해짐).")
parser.add_argument("--rnd_state_group", default=None,
                    help="RND novelty 계산에 쓸 obs 그룹. 미지정 시 --obs_group. "
                         "단일 큐브 스테이지에선 비활성 큐브 마스킹으로 rl_policy 가 사실상 grasp 집중.")
# LSTM(recurrent) 정책 옵션 — 설정 시 ActorCriticRecurrent 사용
parser.add_argument("--recurrent", action="store_true", default=False,
                    help="ActorCriticRecurrent(LSTM) 정책 사용. 미설정 시 기존 feedforward ActorCritic.")
parser.add_argument("--rnn_type", default="lstm", choices=["lstm", "gru"],
                    help="recurrent 정책 RNN 종류 (--recurrent 일 때만)")
parser.add_argument("--rnn_hidden_dim", type=int, default=256,
                    help="RNN hidden state 차원 (--recurrent 일 때만)")
parser.add_argument("--rnn_num_layers", type=int, default=1,
                    help="RNN 층 수 (--recurrent 일 때만)")
parser.add_argument("--obs_normalization", action="store_true", default=False,
                    help="actor/critic 관측 정규화(empirical running stats) 사용. 43-dim rl_state 권장.")
parser.add_argument("--schedule", default="fixed", choices=["fixed", "adaptive"],
                    help="PPO learning rate schedule")
# --device / --headless 는 AppLauncher 가 등록
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# headless 기본값 강제 (명시적으로 --no-headless 를 전달하지 않은 경우)
args.headless = True
# 비디오 녹화 시 offscreen 렌더를 위해 카메라 활성화 강제
if args.video:
    args.enable_cameras = True

launcher = AppLauncher(args)
simulation_app = launcher.app

# ---- Isaac Sim 기동 후 임포트 ----
import glob  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import sim_to_real  # noqa: E402  # SimToReal-SO101-PickCube/PickPen-v0 등록

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import (  # noqa: E402
    apply_curriculum as apply_cube_curriculum,
    apply_skill_acquire,
    apply_skill_full_bc,
    apply_skill_place,
)
from sim_to_real.tasks.pick_pen.pick_pen_env_cfg import apply_curriculum as apply_pen_curriculum  # noqa: E402


TASK_ID = "TB.3"


def _checkpoint_index(path: str) -> int:
    """model_<step>.pt 체크포인트를 숫자 기준으로 정렬하기 위한 키."""
    match = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
    if match is None:
        return -1
    return int(match.group(1))


def _build_train_cfg(args: argparse.Namespace) -> dict:
    """OnPolicyRunner 에 전달할 학습 설정 딕셔너리 반환."""
    rl_device = args.rl_device if args.rl_device is not None else args.device
    obs_group = args.obs_group
    critic_group = args.critic_obs_group if args.critic_obs_group is not None else obs_group
    obs_groups = {"policy": [obs_group], "critic": [critic_group]}
    # RND novelty 입력 그룹을 명시(미지정 시 rsl_rl 가 policy 로 폴백하며 경고).
    if args.rnd:
        rnd_group = args.rnd_state_group if args.rnd_state_group is not None else obs_group
        obs_groups["rnd_state"] = [rnd_group]

    # 정책 설정 — --recurrent 시 LSTM/GRU(ActorCriticRecurrent), 아니면 feedforward.
    # OnPolicyRunner 가 policy["class_name"] 을 eval() 로 rsl_rl.modules 에서 찾는다.
    policy_cfg = {
        "class_name": "ActorCritic",
        "init_noise_std": args.init_noise_std,
        # MLP/LSTM 공통 [256,128] — near-MDP 87dim obs 에 충분 용량(MLP 도 LSTM 과 동등 비교).
        "actor_hidden_dims": [256, 128],
        "critic_hidden_dims": [256, 128],
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

    algorithm_cfg = {
        "class_name": "PPO",
        "num_learning_epochs": args.num_learning_epochs,
        "num_mini_batches": args.num_mini_batches,
        "learning_rate": args.learning_rate,
        "schedule": args.schedule,
        "gamma": args.gamma,
        "lam": args.lam,
        "entropy_coef": args.entropy_coef,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
    }
    # RND(내재 탐색 보상). num_states/obs_groups 는 OnPolicyRunner 가 rnd_state(=policy obs)
    # 로 자동 채움. weight 는 내부에서 step_dt 곱해짐. 후반 과탐색 방지 위해 선형 감쇠.
    if args.rnd:
        algorithm_cfg["rnd_cfg"] = {
            "weight": args.rnd_weight,
            "num_outputs": 64,
            "predictor_hidden_dims": [256, 128],
            "target_hidden_dims": [256, 128],
            "activation": "elu",
            "learning_rate": 1e-3,
            "state_normalization": True,
            "reward_normalization": True,
            "weight_schedule": {
                "mode": "linear", "initial_step": 0,
                "final_step": args.max_iterations, "final_value": 0.0,
            },
        }

    return {
        "seed": args.seed,
        "device": rl_device,
        "num_steps_per_env": args.num_steps_per_env,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "logger": "tensorboard",
        "obs_groups": obs_groups,
        "policy": policy_cfg,
        "algorithm": algorithm_cfg,
    }


def _apply_task_curriculum(env_cfg, args: argparse.Namespace) -> None:
    """task 이름에 맞는 curriculum을 적용한다."""

    params = {
        "active_objects": args.active_objects,
        "object_radius_scale": args.object_radius_scale,
        "container_angle_scale": args.container_angle_scale,
        "container_radius_scale": args.container_radius_scale,
    }
    if args.task and "PickCube" in args.task:
        apply_cube_curriculum(env_cfg, **params)
    else:
        apply_pen_curriculum(
            env_cfg,
            active_pens=params["active_objects"],
            pen_radius_scale=params["object_radius_scale"],
            cup_angle_scale=params["container_angle_scale"],
            cup_radius_scale=params["container_radius_scale"],
        )


def _resolve_log_dir(args: argparse.Namespace) -> str:
    if args.checkpoint_dir is not None:
        log_dir = args.checkpoint_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f"_{args.run_name}" if args.run_name else ""
        log_dir = os.path.join(
            args.log_root_path,
            args.experiment_name,
            f"{timestamp}{suffix}",
        )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _override_policy_std(policy, value: float | None) -> None:
    if value is None:
        return
    if not hasattr(policy, "std"):
        raise AttributeError("ActorCritic policy does not expose a 'std' parameter")
    with torch.no_grad():
        policy.std.fill_(float(value))


def main() -> None:
    device: str = args.device
    rl_device: str = args.rl_device if args.rl_device is not None else device
    env = None
    try:
        # 환경 설정 파싱 및 생성
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = args.seed
        # 커리큘럼 파라미터 적용
        _apply_task_curriculum(env_cfg, args)
        # skill chaining 프리셋 (PickCube 한정; apply_curriculum 이후 = 활성 큐브 cfg 회수 가능)
        if args.task and "PickCube" in args.task:
            if args.skill == "acquire":
                apply_skill_acquire(env_cfg)
            elif args.skill == "place":
                apply_skill_place(env_cfg)
            elif args.skill == "full_bc":
                apply_skill_full_bc(env_cfg)
        # --episode_length_s 는 skill 프리셋(place=5s)보다 우선(명시 CLI override)
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
        # grasp_close weight override (camp 탈출 — 점화 후 hold income↓ 로 carry 유도).
        # 점화는 weight 3.0 으로 달성(scratch), resume 후 이 arg 로 낮춰 camp 깨고 carry 연결.
        if args.grasp_close_weight is not None and hasattr(env_cfg.rewards, "grasp_close_cube"):
            env_cfg.rewards.grasp_close_cube.weight = float(args.grasp_close_weight)
            print(f"[reward] grasp_close_cube.weight override → {args.grasp_close_weight}")
        if args.grasp_align_weight is not None and hasattr(env_cfg.rewards, "grasp_align_cube"):
            env_cfg.rewards.grasp_align_cube.weight = float(args.grasp_align_weight)
            print(f"[reward] grasp_align_cube.weight override → {args.grasp_align_weight}")
        # grasp 부트스트랩(backward curriculum) — PickCubeEnv 가 읽는다.
        if hasattr(env_cfg, "grasp_bootstrap_prob"):
            env_cfg.grasp_bootstrap_prob = args.grasp_bootstrap_prob
            env_cfg.grasp_bootstrap_close = args.grasp_bootstrap_close
            # annealing: anneal_iters → common_step_counter 단위(=iters×num_steps_per_env)
            env_cfg.grasp_bootstrap_prob_final = args.grasp_bootstrap_prob_final
            env_cfg.grasp_bootstrap_anneal_steps = float(
                args.grasp_bootstrap_anneal_iters * args.num_steps_per_env
            )
            if hasattr(env_cfg, "grasp_bootstrap_pregrasp_frac"):
                env_cfg.grasp_bootstrap_pregrasp_frac = args.grasp_bootstrap_pregrasp_frac
        # carry 역커리큘럼 — PickCubeEnv 가 읽는다(iters → common_step_counter 단위).
        if args.carry_rc_anneal_iters > 0:
            env_cfg.carry_rc_anneal_steps = float(args.carry_rc_anneal_iters * args.num_steps_per_env)
        # place 부트스트랩 — PickCubeEnv 가 읽는다.
        if hasattr(env_cfg, "place_bootstrap_prob"):
            env_cfg.place_bootstrap_prob = args.place_bootstrap_prob
        # demo-state reset (RFCL) — PickCubeEnv 가 읽는다.
        if hasattr(env_cfg, "demo_reset_prob"):
            env_cfg.demo_reset_prob = args.demo_reset_prob
            env_cfg.demo_dataset_dir = args.demo_dataset_dir
            env_cfg.demo_anneal_steps = float(args.demo_anneal_iters * args.num_steps_per_env)
            env_cfg.demo_subsample = args.demo_subsample
            env_cfg.demo_max_files = args.demo_max_files

        # 로그 디렉터리(비디오 폴더가 필요해 env 생성 전에 결정)
        log_dir = _resolve_log_dir(args)

        env = gym.make(args.task, cfg=env_cfg,
                       render_mode="rgb_array" if args.video else None)

        # 주기적 에피소드 비디오 녹화 (RL 래퍼보다 먼저 감싼다)
        if args.video:
            video_dir = os.path.join(log_dir, "videos", "train")
            os.makedirs(video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                step_trigger=lambda step: step % args.video_interval == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
            print(json.dumps({"video": True, "video_dir": video_dir,
                              "interval": args.video_interval, "length": args.video_length}),
                  flush=True)

        # rsl_rl VecEnv 래퍼
        env = RslRlVecEnvWrapper(env, clip_actions=args.clip_actions)

        # 재현성 시드
        torch.manual_seed(args.seed)
        if hasattr(env, "seed"):
            env.seed(args.seed)

        # 학습 설정
        train_cfg = _build_train_cfg(args)

        # OnPolicyRunner 생성 및 학습
        run_start_time = time.time()
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=rl_device)
        # 이어학습: resume_checkpoint 지정 시 optimizer 포함 로드
        if args.resume_checkpoint is not None:
            load_optimizer = not args.resume_without_optimizer
            if args.resume_without_optimizer:
                # BC warmstart 등 — policy state_dict 만 로드(optimizer/RND 는 fresh).
                # BC ckpt 엔 rnd_state_dict 가 없어 runner.load 의 RND 로드가 KeyError → 수동 로드로 우회.
                import torch as _torch
                _loaded = _torch.load(args.resume_checkpoint, map_location=rl_device, weights_only=False)
                runner.alg.policy.load_state_dict(_loaded["model_state_dict"])
                print(f"[resume] policy-only load from {args.resume_checkpoint} (optimizer/RND fresh)", flush=True)
            else:
                try:
                    runner.load(args.resume_checkpoint, load_optimizer=load_optimizer, map_location=rl_device)
                except TypeError:
                    runner.load(args.resume_checkpoint)
            _override_policy_std(runner.alg.policy, args.override_policy_std)
        runner.learn(
            num_learning_iterations=args.max_iterations,
            init_at_random_ep_len=True,
        )

        # 체크포인트 목록 수집
        checkpoints = sorted(
            (
                c for c in glob.glob(os.path.join(log_dir, "model_*.pt"))
                if os.path.getmtime(c) >= run_start_time - 1.0
            ),
            key=_checkpoint_index,
        )
        if not checkpoints:
            print(
                json.dumps({
                    "task_id": TASK_ID,
                    "status": "failed",
                    "error": f"체크포인트 없음: {log_dir}",
                }),
                flush=True,
            )
            sys.exit(1)

        total_steps = args.num_envs * args.num_steps_per_env * args.max_iterations
        print(
            json.dumps({
                "task_id": TASK_ID,
                "status": "passed",
                "task": args.task,
                "num_envs": args.num_envs,
                "num_steps_per_env": args.num_steps_per_env,
                "max_iterations": args.max_iterations,
                "num_learning_epochs": args.num_learning_epochs,
                "num_mini_batches": args.num_mini_batches,
                "total_steps": total_steps,
                "log_dir": log_dir,
                "checkpoints": [os.path.basename(c) for c in checkpoints],
                "latest_checkpoint": checkpoints[-1],
                "curriculum": {
                    "skill": args.skill,
                    "active_objects": args.active_objects,
                    "object_radius_scale": args.object_radius_scale,
                    "container_angle_scale": args.container_angle_scale,
                    "container_radius_scale": args.container_radius_scale,
                    "episode_length_s": args.episode_length_s,
                    "resume_checkpoint": args.resume_checkpoint,
                    "resume_without_optimizer": args.resume_without_optimizer,
                    "init_noise_std": args.init_noise_std,
                    "override_policy_std": args.override_policy_std,
                    "entropy_coef": args.entropy_coef,
                    "learning_rate": args.learning_rate,
                },
            }),
            flush=True,
        )

    except Exception as exc:
        tb = traceback.format_exc()
        print(
            json.dumps({
                "task_id": TASK_ID,
                "status": "failed",
                "error": str(exc),
                "traceback": tb,
            }),
            flush=True,
        )
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
