"""source 데모에 Isaac Lab Mimic **subtask 주석**을 단다 — SO-101 (auto 모드 전용).

이미지의 공식 `annotate_demos.py` 를 그대로 쓸 수 없는 이유는 하나다: 그 스크립트는
`isaaclab_mimic.envs` 와 `isaaclab_tasks` 만 import 해서 **우리 gym 등록
(`import sim_to_real.tasks`)을 못 본다** → `NameNotFound: SimToReal-SO101-PickCube-Mimic`.
그래서 공식 `--auto` 경로만 최소 이식했다(대화형 수동 주석 UI 는 이식하지 않았다).

하는 일: 각 에피소드를 `initial_state` 로 되돌려 기록된 action 을 재생하고, 매 스텝
`obs/datagen_info/{object_pose, eef_pose, target_eef_pose, subtask_term_signals,
subtask_start_signals}` 를 기록한다. 성공 + 모든 subtask 신호가 발화한 에피소드만 export 한다.

```bash
python scripts/datagen/annotate_mimic_demos.py \
    --task SimToReal-SO101-PickCube-Mimic-v0 \
    --input_file /workspace/datasets/src.hdf5 \
    --output_file /workspace/datasets/src_annotated.hdf5 --headless
```

`--enable_cameras` 없으면 카메라를 떼고 부팅한다(주석에 이미지는 불필요 — §5.6 과 같은 규약).
SkillGen(`use_skillgen=True`)은 **subtask start 신호가 필수**라 기본으로 기록한다.
"""

# planner 를 쓰지 않지만 generate 와 같은 규약을 유지한다(cuRobo 를 나중에 붙여도 안전).
import warp as _warp  # noqa: F401  isort:skip

import argparse  # noqa: E402
import os  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="SO-101 Mimic subtask 자동 주석")
parser.add_argument("--task", type=str, default="SimToReal-SO101-PickCube-Mimic-v0",
                    help="mimic env id")
parser.add_argument("--input_file", type=str, required=True, help="source 데이터셋(hdf5)")
parser.add_argument("--output_file", type=str, default="./datasets/src_annotated.hdf5",
                    help="주석 결과 hdf5")
parser.add_argument("--no_start_signals", action="store_true",
                    help="subtask start 신호 기록 생략(순수 Mimic 전용 — SkillGen 은 필수다)")
parser.add_argument("--trace_divergence", action="store_true",
                    help="재생 중 source 의 applied_target·joint_pos 와 프레임 단위 대조(진단)")
parser.add_argument("--include_failed", action="store_true",
                    help="source 의 success=False 에피소드도 주석 시도(기본은 건너뜀). "
                         "SM --record_hdf5 는 실패도 저장하므로 기본 필터가 맞다")
parser.add_argument("--cube_sizes", default=None,
                    help="큐브 크기 DR 후보를 이 목록으로 덮어쓴다(콤마 구분, m). "
                         "★source 녹화·증강 생성과 **같은 값**이어야 한다 — 개루프 재생이라 "
                         "크기가 다르면 파지 기하가 어긋난다. 크기 DR 이 있는 `-DR` 변형 전용")
parser.add_argument("--retries", type=int, default=3,
                    help="에피소드별 재생 재시도 횟수. 개루프 재생은 접촉 물리가 비결정적이라 "
                         "marginal grasp 가 시도마다 뒤집힌다(같은 입력·같은 코드로 실패자가 "
                         "{8,9,10,11} → {8,9,10} 로 바뀐 것을 실측). 0 = 재시도 없음")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args_cli).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

"""이하 kit 부팅 후."""

import contextlib  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLMimicEnv  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import sim_to_real.tasks  # noqa: F401,E402  gym 등록 — 공식 스크립트가 못 하는 바로 그 부분
from sim_to_real.tasks.pick_cube.pick_cube_env_cfg import remove_pick_cube_cameras  # noqa: E402
from so101_contract.feature_codec import SO101_JOINT_ORDER  # noqa: E402


class PreStepDatagenInfoRecorder(RecorderTerm):
    """매 스텝 mimic datagen info 기록(공식 `annotate_demos.py` 와 동일 키·형식)."""

    def record_pre_step(self):
        eef_pose_dict = {
            eef_name: self._env.get_robot_eef_pose(eef_name=eef_name)
            for eef_name in self._env.cfg.subtask_configs
        }
        return "obs/datagen_info", {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsRecorder


class PreStepSubtaskStartsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskStartsRecorder


@configclass
class MimicAnnotationRecorderCfg(ActionStateRecorderManagerCfg):
    """action/state 기본 + mimic 주석 3종.

    ★stock `record_pre_step_flat_policy_observations` 는 **끈다.** 그 term 은 키 `"obs"` 에
    `obs_buf["policy"]` 를 그대로 넣는데, 우리 policy 관측은 **concat 된 텐서**다(6-dim joint).
    그러면 `obs` 가 스텝별 텐서 리스트가 되고, 뒤이어 `obs/datagen_info/...` 를 쓸 때
    `EpisodeData.add` 가 리스트를 dict 처럼 인덱싱해 터진다
    (`TypeError: list indices must be integers or slices, not str`).
    공식 franka mimic env 는 `concatenate_terms=False` 라 `obs` 가 dict → 충돌이 없다.
    mimic 이 소비하는 것은 `actions`·`initial_state`·`obs/datagen_info` 뿐이라 손실도 없다.
    """

    record_pre_step_flat_policy_observations: RecorderTermCfg | None = None

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsRecorderCfg()


def replay_episode(env: ManagerBasedRLMimicEnv, episode: EpisodeData, success_term) -> bool:
    """기록된 action 을 그대로 재생한다. 성공 판정까지 통과하면 True."""
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(episode.data["initial_state"], None, is_relative=True)
    # ★`reset_to` 는 `_reset_idx()`(= action_manager.reset) 를 **먼저** 부르고 그 다음에 관절을
    #   덮어쓴다(isaaclab manager_based_env.py). 그래서 슬루 리미터 내부 기준
    #   (`SlewLimitedJointPositionAction._limited_actions`) 이 **복원 전** 자세로 굳은 채 관절만
    #   텔레포트된다 → 첫 스텝부터 엉뚱한 기준에서 램프해 궤적이 어긋난다. 값이 직전 에피소드의
    #   종료 자세에 의존하므로 **결정적이지만 데모마다 다르게** 틀어진다.
    #   복원이 끝난 지금 다시 reset 해서 기준을 실제 시작 자세로 맞춘다.
    env.action_manager.reset()

    # 진단: source 가 남긴 `applied_target`(슬루 후 명령)·`obs_x/joint_pos`(실측)와 프레임 단위 대조.
    # 같은 action 을 같은 초기상태에서 재생했는데 결과가 다르면, 갈라지는 지점이 원인을 특정한다:
    #   `applied_target` 부터 어긋남 → 액션 파이프라인(슬루 기준·offset) 문제
    #   명령은 같고 `joint_pos` 만 어긋남 → 접촉/솔버 물리 문제
    reference_target = episode.data.get("applied_target") if args_cli.trace_divergence else None
    reference_state = episode.data.get("obs_x", {}).get("joint_pos") if reference_target is not None else None
    joint_index = env.canonical_joint_idx
    first_cmd = first_state = None
    max_cmd = max_state = 0.0
    worst_joint, worst_detail = None, ""

    for step, action in enumerate(episode.data["actions"]):
        # ★비교는 **스텝 전** 상태로 한다. source 의 `obs_x/joint_pos` 는 `record_pre_step`,
        #   즉 그 스텝을 밟기 **전** 값이다. 스텝 후에 읽으면 1프레임 밀려, 슬루 상한으로 움직이는
        #   축에서 가짜 발산이 나온다(그리퍼 2.5 rad/s ÷ 30 Hz = 0.0833 rad = 4.8° — 실제로
        #   전 데모에서 0.0825 가 관측돼 상수 발산으로 오독했다).
        if reference_state is not None and step < len(reference_state):
            measured = env.scene["robot"].data.joint_pos[0, joint_index].detach().cpu().numpy()
            want = reference_state[step]
            want = want.detach().cpu().numpy() if hasattr(want, "detach") else np.asarray(want)
            per_joint = np.abs(measured - want)
            error_state = float(per_joint.max())
            if error_state > max_state:
                max_state = error_state
                worst_joint = SO101_JOINT_ORDER[int(np.argmax(per_joint))]
                worst_detail = " ".join(f"{n}={v:+.4f}" for n, v in zip(SO101_JOINT_ORDER, per_joint))
            if first_state is None and error_state > 1e-2:
                first_state = (step, error_state, SO101_JOINT_ORDER[int(np.argmax(per_joint))])

        env.step(torch.Tensor(action).reshape(1, action.shape[0]))
        if reference_target is None or step >= len(reference_target):
            continue
        term = env.action_manager.get_term("arm")
        command = term.processed_actions[0].detach().cpu().numpy()
        # processed_actions 컬럼은 asset 순서 → canonical 로 재배열
        order = [term._joint_names.index(j) for j in SO101_JOINT_ORDER]
        command = command[order]
        expected = reference_target[step]
        expected = expected.detach().cpu().numpy() if hasattr(expected, "detach") else np.asarray(expected)
        error_cmd = float(np.abs(command - expected).max())
        max_cmd = max(max_cmd, error_cmd)
        if first_cmd is None and error_cmd > 1e-3:
            first_cmd = (step, error_cmd)

    if reference_target is not None:
        print(f"\t  발산: 명령 first@{first_cmd} max={max_cmd:.5f} rad · "
              f"실측 first@{first_state} max={max_state:.5f} rad (최악축 {worst_joint})")
        print(f"\t         최악 프레임 축별: {worst_detail}")

    if success_term is not None:
        return bool(success_term.func(env, **success_term.params)[0])
    return True


def report_success_gates(env: ManagerBasedRLMimicEnv, success_term) -> None:
    """성공 판정이 왜 실패했는지 **게이트별로** 출력한다.

    `task_done` 은 물체마다 (xy 거리 < radius) AND (그릇 대비 상대높이 ∈ height_range) 를 본다.
    어느 쪽이 걸렸는지 모르면 "재생이 실패했다"만 남고 원인 추적이 막힌다 — 실측에서 물리 DR·
    파지 마진 가설을 둘 다 기각시킨 뒤 이 로그가 필요해졌다.
    """
    params = dict(success_term.params)
    container_cfg = params.get("container_cfg")
    radius = float(params.get("radius", 0.06))
    height_range = tuple(params.get("height_range", (0.005, 0.12)))
    origins = env.scene.env_origins[0]
    if container_cfg is not None:
        container = env.scene[container_cfg.name].data.root_pos_w[0] - origins
    else:
        center = params.get("container_center_xy", (0.0, 0.0))
        container = torch.tensor([center[0], center[1], 0.76], device=env.device)

    for object_cfg in params.get("objects_cfg", []):
        position = env.scene[object_cfg.name].data.root_pos_w[0] - origins
        distance = float(torch.linalg.vector_norm(position[:2] - container[:2]).item())
        relative_height = float((position[2] - container[2]).item())
        gates = []
        if distance >= radius:
            gates.append(f"xy {distance * 1000:.1f}mm ≥ {radius * 1000:.0f}mm")
        if relative_height <= height_range[0]:
            gates.append(f"높이 {relative_height * 1000:+.1f}mm ≤ {height_range[0] * 1000:.1f}mm")
        if relative_height >= height_range[1]:
            gates.append(f"높이 {relative_height * 1000:+.1f}mm ≥ {height_range[1] * 1000:.1f}mm")
        verdict = " · ".join(gates) if gates else "통과"
        print(f"\t  {object_cfg.name}: xy {distance * 1000:6.1f}mm 상대높이 "
              f"{relative_height * 1000:+7.1f}mm → {verdict}")


def annotate_episode(env: ManagerBasedRLMimicEnv, episode: EpisodeData, success_term,
                     want_start_signals: bool) -> bool:
    """재생 + 신호 발화 검사. 하나라도 발화 안 하면 그 에피소드는 버린다."""
    if not replay_episode(env, episode, success_term):
        print("\t재생 결과가 성공 판정을 통과하지 못했다 — 제외")
        report_success_gates(env, success_term)
        return False

    annotated = env.recorder_manager.get_episode(0)
    groups = [("term", annotated.data["obs"]["datagen_info"]["subtask_term_signals"])]
    if want_start_signals:
        groups.append(("start", annotated.data["obs"]["datagen_info"]["subtask_start_signals"]))

    ok = True
    for kind, signal_dict in groups:
        for name, flags in signal_dict.items():
            values = torch.as_tensor(flags, device=env.device).reshape(-1).float()
            if torch.any(values > 0):
                continue
            # ★"한 번도 발화 안 함"만 남기면 원인 추적이 막힌다. 신호가 **얼마나 근접했는지**와
            #   종료 시점 기하를 같이 남긴다 — 실측에서 이 로그가 없어 재생 자체를 의심하느라
            #   프로브를 따로 짜야 했다(2026-08-04).
            print(f'\tsubtask "{name}" 의 {kind} 신호가 한 번도 발화하지 않았다 — 제외 '
                  f'(T={values.numel()} max={float(values.max()):.3f})')
            report_signal_gates(env, name)
            ok = False
    return ok


def report_signal_gates(env: ManagerBasedRLMimicEnv, signal_name: str) -> None:
    """subtask 신호가 안 뜬 이유 — 그 term 의 **하위 조건**을 종료 시점 값으로 찍는다.

    `place_cube1`(= `object_in_container`)은 4조건 AND 다: xy < radius · 상대높이 ∈ height_range ·
    그리퍼 열림. 어느 하나만 걸려도 0 이 나오는데 그걸 구분 못 하면 "재생이 틀렸나"부터 의심하게 된다.
    """
    term_cfg = getattr(env.cfg.observations.subtask_terms, signal_name, None)
    params = dict(getattr(term_cfg, "params", {}) or {})
    object_cfg = params.get("object_cfg")
    container_cfg = params.get("container_cfg")
    if object_cfg is None or container_cfg is None:
        return

    origins = env.scene.env_origins[0]
    obj = env.scene[object_cfg.name].data.root_pos_w[0] - origins
    box = env.scene[container_cfg.name].data.root_pos_w[0] - origins
    radius = float(params.get("radius", 0.06))
    low, high = tuple(params.get("height_range", (0.005, 0.06)))
    threshold = float(params.get("grasp_threshold", 0.60))

    robot_cfg = params.get("robot_cfg")
    joint_ids = getattr(robot_cfg, "joint_ids", None)
    index = joint_ids[0] if isinstance(joint_ids, list) and joint_ids else -1
    gripper = float(env.scene["robot"].data.joint_pos[0, index])

    xy = float(torch.linalg.vector_norm(obj[:2] - box[:2]))
    dz = float(obj[2] - box[2])
    print(f"\t  종료 시점: xy={xy * 1000:.1f}mm(<{radius * 1000:.0f}) "
          f"dz={dz * 1000:.1f}mm(∈{low * 1000:.0f}~{high * 1000:.0f}) "
          f"gripper={gripper:+.3f}(>{threshold:.2f})")
    blocked = [n for n, good in (("xy", xy < radius), ("dz", low < dz < high),
                                 ("gripper_open", gripper > threshold)) if not good]
    print(f"\t  걸린 조건: {', '.join(blocked) if blocked else '없음(중간 프레임에서만 만족)'}")

    # 기하가 멀쩡한데 신호가 0 이면 **신호 경로**가 범인이다 — obs_buf 와 hook 을 같이 본다.
    live_obs = env.obs_buf.get("subtask_terms", {}) if isinstance(env.obs_buf, dict) else {}
    raw = live_obs.get(signal_name)
    hook = env.get_subtask_term_signals().get(signal_name)
    print(f"\t  live obs_buf[{signal_name}]={None if raw is None else raw.reshape(-1).tolist()} "
          f"· hook={None if hook is None else hook.reshape(-1).tolist()}")


def main() -> int:
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"입력 데이터셋이 없다: {args_cli.input_file}")

    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.input_file)
    if handler.get_num_episodes() == 0:
        print("에피소드가 없다.")
        return 0

    output_dir = os.path.dirname(os.path.abspath(args_cli.output_file))
    os.makedirs(output_dir, exist_ok=True)

    env_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name

    if not hasattr(env_cfg.terminations, "success"):
        raise NotImplementedError("env 에 success termination 이 없다")
    success_term = env_cfg.terminations.success
    env_cfg.terminations = None

    want_start_signals = not args_cli.no_start_signals
    env_cfg.recorders = MimicAnnotationRecorderCfg()
    if not want_start_signals:
        env_cfg.recorders.record_pre_step_subtask_start_signals = None
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.output_file))[0]

    if not getattr(args_cli, "enable_cameras", False):
        remove_pick_cube_cameras(env_cfg)
    if args_cli.cube_sizes:
        # ★주석은 source 를 **개루프 재생**한다 — 큐브 크기가 녹화 때와 다르면 파지 기하가
        #   어긋나 재생이 실패하거나(신호 미발화) 조용히 다른 궤적이 된다. 녹화·주석·증강
        #   세 단계가 같은 크기여야 한다(`pickplace_sm.py` / 여기 / `generate_mimic_dataset.py`).
        event = getattr(env_cfg.events, "randomize_cube_sizes", None)
        if event is None:
            raise SystemExit(f"--cube_sizes 는 크기 DR 이 있는 env 에서만 쓴다 "
                             f"(task={env_name} 에 randomize_cube_sizes 없음 — `-DR` 변형을 쓰라)")
        event.params["sizes"] = [float(v) for v in str(args_cli.cube_sizes).split(",") if v.strip()]
        print(f"[annotate] cube size DR override → {event.params['sizes']}", flush=True)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError(f"{env_name} 은 ManagerBasedRLMimicEnv 자손이 아니다")
    for hook in ("get_subtask_term_signals", *(["get_subtask_start_signals"] if want_start_signals else [])):
        if getattr(env, hook).__func__ is getattr(ManagerBasedRLMimicEnv, hook):
            raise NotImplementedError(f"env 가 {hook} 을 구현하지 않았다 — auto 주석 불가")

    env.reset()

    exported = processed = retried = skipped = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for index, episode_name in enumerate(handler.get_episode_names()):
            episode = handler.load_episode(episode_name, env.device)
            # SM 의 --record_hdf5 는 실패 에피소드도 저장한다(success attr 로 구분). 태스크를
            # 완수하지 못한 데모는 재생해도 성공 판정을 통과할 수 없으므로 주석 대상이 아니다 —
            # 분모에 넣으면 주석 성공률이 source 실패율만큼 깎여 보인다.
            # isaaclab2lerobotv3.py 와 같은 규약(--include_failed 로 해제).
            # attr 은 h5py 가 numpy.bool_ 로 준다 — `is False` 는 항상 거짓이라 쓰면 안 된다.
            # attr 자체가 없으면 None 이고, 그때는 판정 불가라 주석을 시도한다.
            if episode.success is not None and not episode.success and not args_cli.include_failed:
                skipped += 1
                print(f"\n에피소드 #{index} ({episode_name}) 건너뜀 — source success=False")
                continue
            processed += 1
            print(f"\n에피소드 #{index} ({episode_name}) 주석")
            for attempt in range(args_cli.retries + 1):
                if attempt:
                    retried += 1
                    print(f"\t재시도 {attempt}/{args_cli.retries}")
                if annotate_episode(env, episode, success_term, want_start_signals):
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device))
                    env.recorder_manager.export_episodes()
                    exported += 1
                    print(f"\t주석 완료 — export (시도 {attempt + 1}회)")
                    break

    print(f"\n주석 export {exported}/{processed} 에피소드 (재시도 {retried}회, "
          f"source 실패 {skipped}건 제외) → {args_cli.output_file}")
    env.close()
    return exported


if __name__ == "__main__":
    count = 0
    try:
        count = main()
    except KeyboardInterrupt:
        print("\n사용자 중단")
    simulation_app.close()
    raise SystemExit(0 if count > 0 else 1)
