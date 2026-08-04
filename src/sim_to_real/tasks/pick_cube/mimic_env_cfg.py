"""SO-101 pick-place **Mimic / SkillGen** env cfg.

`PickCubeEnvCfg` 계열 + `isaaclab.envs.mimic_env_cfg.MimicEnvCfg` 를 합쳐, mimic 데이터 생성이
읽는 `datagen_config` 와 `subtask_configs` 를 붙인다. 씬·액션·관측·성공 판정은 **기존 pick_cube
그대로** 다 — mimic 은 그 위에 데이터 생성 메타만 얹는다.

## subtask 2개

======  ==============  ===================  =========================================
index   object_ref      subtask_term_signal  의미
======  ==============  ===================  =========================================
0       Cube1           `cube_grasped`       큐브 파지(양 손가락 접촉 + 상판 위로 들림)
1       Bowl            `place_cube1`        그릇 안에 놓음(마지막 subtask)
======  ==============  ===================  =========================================

각 구간은 `object_ref` 프레임 기준으로 변환된다: grasp 구간은 큐브가 어디로 스폰되든 그 큐브
기준 상대 궤적을 재사용하고, place 구간은 그릇 기준 상대 궤적을 재사용한다.

## SkillGen 을 쓸 때

`datagen_config.use_skillgen = True` 이며, ``[이전 종료 → 이번 시작]`` 전이는 cuRobo v0.8
planner(`sim_to_real.datagen.skillgen_planner`)가 충돌회피로 다시 계획한다. 그래서
`get_subtask_start_signals` 가 **필수**이고, 그 임계값 knob 들이 여기 `mimic_*` 필드다.
"""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import DataGenConfig, MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from sim_to_real.tasks.common.mdp._geometry import DESK_TOP_Z

from .mimic_env import EEF_NAME
from .pick_cube_env_cfg import PickCubeDREnvCfg, PickCubeEnvCfg

#: `observations.subtask_terms` 의 신호 이름 = subtask 종료 신호 이름(순서 = subtask 순서).
MIMIC_TERM_SIGNALS: tuple[str, str] = ("cube_grasped", "place_cube1")
#: grasp 대상 큐브(1-cube 씬). 다중 큐브로 늘리면 subtask 를 쌍으로 추가한다.
MIMIC_GRASP_OBJECT = "Cube1"
MIMIC_PLACE_OBJECT = "Bowl"


def mimic_datagen_config() -> DataGenConfig:
    """데이터 생성 계약 — DR/non-DR 변형이 **같은 값**을 쓴다."""
    config = DataGenConfig()
    config.name = "so101_pick_cube_mimic_D0"
    config.generation_guarantee = True
    config.generation_keep_failed = True
    config.generation_num_trials = 10
    config.generation_select_src_per_subtask = True
    config.generation_transform_first_robot_pose = False
    config.generation_interpolate_from_last_target_pose = True
    config.generation_relative = True
    config.max_num_failures = 25
    config.seed = 1
    # SkillGen(전이를 cuRobo 로 재계획) 기본 on — 드라이버가 planner 를 주입한다.
    config.use_skillgen = True
    return config


def mimic_subtask_configs() -> dict[str, list[SubTaskConfig]]:
    """subtask 2개(grasp → place)."""
    common = dict(
        # SkillGen 은 시작 경계도 흔들 수 있다. 다만 시작을 뒤로 밀면 그리퍼 폐합을 넘길
        # 위험이 있어 0 으로 둔다 — 다양성은 planner 재계획이 만든다.
        subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        # ★결정적 최근접(nn_k=1) + **원본 거리 지표**. 상위 k 중 무작위로 고르면 부적합
        # source 를 뽑는다. 그리고 "위치를 가깝게" 는 **틀린 목적함수**다 — 세 번 반증됐다
        # (`rot_weight` 0.05 → 76.1 % → 52~57 % · Δψ 인지 선택 · 대칭 접기 → 81.6 % → 22.2 %).
        # **어느 face 를 무느냐**(회전)가 5-DOF 도달성을 가르고 그게 성공률 절벽을 만든다.
        # 상세 = `docs/spec/09_TACIT_KNOWLEDGE.md`. **재시도 금지.**
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 1},
        action_noise=0.0,
        # ★구간 **진입** 보간 = 전이 도착 자세 ↔ source 구간 시작 자세 차이를 소화하는 프레임 수.
        # clamp 이 아니라 **시간을 늘리는** 레버다. 0/0 으로 지웠다가 영상에서 경계 튐이 그대로
        # 드러나 되돌렸다(투영 잔차는 0 이 아니고, 한 스텝에 소화하면 슬루 상한을 후려친다).
        #
        # 10 → 15. 실측(명령 포화율 = slew cap 에 붙은 프레임 비율): 0 → 1.78~1.89 %,
        # 15 → 0.29~0.38 %(6.5배 개선). **40 은 0.26 % 로 정체**라 더 늘릴 이유가 없다.
        # 비용 = 경계당 +5 프레임.
        num_interpolation_steps=15,
        num_fixed_steps=3,
        apply_noise_during_interpolation=False,
    )
    return {EEF_NAME: [
        SubTaskConfig(
            object_ref=MIMIC_GRASP_OBJECT,
            subtask_term_signal=MIMIC_TERM_SIGNALS[0],
            description="Pick up the cube",
            next_subtask_description="Place the cube in the bowl",
            **common,
        ),
        SubTaskConfig(
            object_ref=MIMIC_PLACE_OBJECT,
            subtask_term_signal=MIMIC_TERM_SIGNALS[1],
            description="Place the cube in the bowl",
            **common,
        ),
    ]}


@configclass
class _MimicKnobs:
    """env 어댑터(`mimic_env.py`)가 읽는 knob — 두 변형이 공유하는 필드 정의."""

    mimic_robot_yaml: str = "/workspace/assets/robots/so101.yml"
    mimic_urdf_path: str = "/workspace/assets/robots/urdf/so_arm101.urdf"
    mimic_term_signals: tuple[str, str] = MIMIC_TERM_SIGNALS
    mimic_grasp_object: str = MIMIC_GRASP_OBJECT
    #: place subtask 의 기준 물체(그릇). 시작 신호가 "그릇 상공 도달"을 재는 데도 쓴다.
    mimic_place_object: str = MIMIC_PLACE_OBJECT
    mimic_desk_top_world_z: float = DESK_TOP_Z

    #: grasp subtask **시작** 판정: tool(`tcp_grasp`) ↔ 큐브 중심 거리 상한(m).
    #:
    #: ★`tcp_grasp` 원점은 손가락 사이 pinch 점이 **아니다**. `grasp_geometry.FIXED_INNER_CENTER
    #: =(0.0215, 0.0147, 0.0463)` 만큼 pad 접촉면에서 떨어져 있어, 파지 순간에도 tcp↔큐브 중심
    #: 거리가 상당히 남는다. source 데모 4개 실측(40 mm 큐브, 그리퍼 열린 구간 최소 거리):
    #: **62.5 · 62.7 · 75.0 · 76.1 mm**. 기울어진(α≠0) grasp 일수록 크다.
    #: → 0.060 은 **한 번도 발화 못 한다**(초기값이 그랬다). 0.075 도 기울어진 셀을 놓친다.
    #: 0.090 은 4/4 에서 그리퍼가 열리는 시점 부근(step 144~147)에 발화한다.
    #:
    #: 크면 planner 재계획 구간이 짧아지고(증강 다양성 ↓), 너무 작으면 발화가 그리퍼 폐합 뒤로
    #: 밀려 planner 가 **닫힌 그리퍼로 접근**을 계획한다(치명적).
    #: ★큐브 크기·grasp 튜닝을 바꾸면 `scratch/approach_calib.py` 류로 다시 잰다.
    mimic_approach_radius_m: float = 0.090
    #: 위 판정의 "그리퍼가 아직 열려 있다" 기준(rad). sim gripper: 닫힘 ≈ -0.17, 열림 ≈ +1.4.
    mimic_gripper_open_rad: float = 0.6
    #: place subtask **시작** 판정 상승고(m, 상판 기준). grasp 종료 판정보다 커야 한다 —
    #: `datagen_info_pool` 이 subtask 경계 단조성을 assert 한다.
    #:
    #: ★0.05 → 0.09. subtask1 은 `object_ref="Bowl"` 이라 **그릇 기준으로 SE(3) 변환**되는데,
    #: 0.05 에서는 신호가 켜지는 순간 팔이 아직 **큐브에 붙어 있다**(실측 eef↔큐브 63~65 mm,
    #: eef↔그릇 207~398 mm). 그 앞 15~21 프레임은 수직 들어올리기 — 큐브 좌표에 종속인 동작인데
    #: 그릇 델타(산포 14 mm)로만 보정돼, 최근접 짝지어도 남는 큐브 오차(중앙 11.4 mm ·
    #: p90 23.3 mm)가 그대로 실린다. 그래서 전이가 파지 지점에서 **source 큐브 자리로 갔다가**
    #: 다시 그릇으로 향한다(사용자 영상 관측: "그릇과 멀어지는 방향으로 이동했다가").
    #:
    #: 지금은 **단조성 확보용 하한**으로만 쓴다(실제 경계는 아래 `bowl_xy` 가 정한다).
    #: 파지 확정과 같은 프레임에 place 가 켜지면 `datagen_info_pool` 의 단조성 assert 가 터진다.
    mimic_place_start_lift_m: float = 0.05

    #: place subtask **시작** 판정: 큐브가 그릇 중심에서 이 xy 거리 안(m).
    #:
    #: ★상승고가 아니라 **그릇 상공 도달**로 끊는다. subtask1 은 그릇 기준으로 변환되므로
    #: 그 구간은 전부 그릇 상대 동작이어야 한다. 이 값이면 남는 구간이
    #: **하강+투하+retreat+hold**(실측 중앙 55프레임)뿐이고, 들어올리기·운반은 앞쪽 cuRobo
    #: 전이가 큐브를 든 채 담당한다:
    #:
    #:     cuRobo(접근) + subtask0(파지) + cuRobo(들어올리기+운반) + subtask1(하강·투하·retreat·hold)
    #:
    #: 0.08 m 근거: 실측 이 시점의 그리퍼가 **아직 닫혀 있다**(+0.33~0.48 rad). cuRobo 전이의
    #: 그리퍼 명령은 다음 구간 첫 waypoint 에서 오므로 투하 이후로 잡으면 전이 내내 열린 채라
    #: 큐브를 도중에 떨군다. 스폰 최소 큐브↔그릇 거리는 140 mm 라 파지 전 오발화도 없다.
    mimic_place_start_bowl_xy_m: float = 0.08

    #: `target_eef_pose_to_action` 의 IK step bound = 슬루 상한 × `step_dt` × 이 계수.
    #: 1.0 = 물리적으로 정확히 따라갈 수 있는 최대. 1.0 초과는 추종오차를 다시 만든다.
    #: 낮추면 더 부드럽지만 목표 추종이 느려져 접촉 타이밍이 밀린다.
    mimic_step_bound_scale: float = 1.0

    #: 전이 계획 waypoint 간격(rad, planner `motion_step_size`). 작을수록 프레임이 늘어 같은
    #: 경로를 더 천천히 지난다. 0.05 rad/step ≈ 1.5 rad/s @30 Hz.
    #: ★0.025 로 내려봤지만 되돌렸다 — wrist_roll 후려치기는 **전이 경로 속도**가 아니라
    #: **전이↔source 경계 불연속**에서 왔고 그건 `num_interpolation_steps` 가 담당한다.
    #: 여기를 내리면 전이가 2배 길어져 "중간에 멈췄다 간다"는 체감만 악화된다(목표 ①과 상충).
    mimic_planner_step_size: float = 0.05


def strip_physics_dr(events) -> list[str]:
    """event cfg 에서 **물리 DR**(큐브 질량·마찰) term 을 떼어낸다. 뗀 이름 목록 반환.

    ★왜 필요한가 — mimic 은 **같은 물리에서 3단계**를 돌려야 한다:
    ① source 녹화 → ② 개루프 재생(주석) → ③ 변환 궤적 실행(증강).
    그런데 `PickCubeDREventCfg` 는 `randomize_object_{material,mass}` 를 **`mode="startup"`**
    으로 건다(질량 ±10% scale · static 마찰 1.4~2.0). startup 난수는 **프로세스마다 새로** 뽑히고
    `reset_to` 는 **pose 만** 복원하므로:

    * ②는 authored 물리로 재생 → source 녹화 당시와 다름. 마진 얇은 grasp 이 미끄러진다
      (실측: 주석 export 8/10, 새 source 로도 같은 비율 재현).
    * ③은 또 다른 startup 난수 → source 데모가 학습한 접촉과 다른 물리에서 재생.

    pose DR(큐브 scatter·그릇 arc)과 시각 DR 은 **그대로 둔다** — mimic 의 다양성은 물체 배치에서
    오고, 그건 `initial_state` 로 정확히 복원된다.
    ⚠ 물리 다양성을 잃는 건 sim2real 관점의 손실이다. 되살리려면 startup 값을 HDF5 에 기록하고
    재생 시 복원해야 한다(recorder+replay 양쪽 작업) — 그때까지는 정합이 우선이다.
    """
    removed = [name for name in list(vars(events))
               if name.startswith("randomize_") and (name.endswith("_material")
                                                     or name.endswith("_mass"))]
    for name in removed:
        setattr(events, name, None)
    return removed


def use_grasp_onset_signal(observations) -> None:
    """subtask0 종료 신호를 "들림"이 아니라 **파지 개시**(접촉만)에 맞춘다.

    ★기본값을 바꾸는 게 아니라 **mimic 에서만** 덮어쓴다. `cube_grasped` 는 등록 env 전체가
    쓰는 관측 term 이고, 거기서 들림 요구를 빼면 스치기만 해도 파지로 센다.

    mimic 에서만 다른 이유: SkillGen 은 ``[이전 종료 → 이번 시작]`` 을 planner 로 다시 계획한다.
    종료를 "들림"에 두면 들어올리기가 source 재생 구간에 남아 큐브 좌표에 종속인 동작이
    그릇 델타로만 보정된다. 파지 개시에서 끊으면 들어올리기·운반을 앞쪽 cuRobo 전이가 큐브를
    든 채 충돌회피로 담당한다(실측 발화 시점이 12~16 프레임 당겨진다).
    """
    observations.subtask_terms.cube_grasped.params["min_lift"] = None


@configclass
class SO101PickCubeMimicEnvCfg(PickCubeEnvCfg, MimicEnvCfg, _MimicKnobs):
    """pick-place mimic env cfg — DR-off(source 데모 주석·재현 검증용).

    ⚠ `configclass` 다중 상속은 `__post_init__` 을 자동 연쇄하지 않는다(MRO 첫 부모만 호출).
    그래서 `MimicEnvCfg` 쪽 필드는 여기서 직접 채운다 — 공식 mimic env cfg 들도 같은 방식이다.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.datagen_config = mimic_datagen_config()
        self.subtask_configs = mimic_subtask_configs()
        use_grasp_onset_signal(self.observations)
        strip_physics_dr(self.events)
        # ★녹화 env(`PickCubeDREnvCfg`)와 **물리 authoring 을 맞춘다.** 이 값이 다르면 PhysX 가
        # 씬을 instanced ↔ de-instanced 로 다르게 구성한다 — 같은 pose 를 복원해도 접촉 해석이
        # 달라져, 착지 직후의 준안정 자세(바닥 코너 1~2개만 접촉)가 재생에서만 무너져 큐브가
        # 굴러버렸다. 두 cfg 의 scene 차이는 실측상 이 필드 **하나뿐**이었다.
        self.scene.replicate_physics = False


@configclass
class SO101PickCubeMimicDREnvCfg(PickCubeDREnvCfg, MimicEnvCfg, _MimicKnobs):
    """DR-on 변형 — 증강 생성·source 녹화 경로(큐브 scatter + 그릇 arc + 시각 DR).

    ★**물리 DR 은 뗀다** — 이유는 :func:`strip_physics_dr`. source 녹화도 이 env 로 해야
    3단계 물리가 일치한다(`pickplace_sm.py --task SimToReal-SO101-PickCube-Mimic-DR-v0`).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.datagen_config = mimic_datagen_config()
        self.subtask_configs = mimic_subtask_configs()
        use_grasp_onset_signal(self.observations)
        removed = strip_physics_dr(self.events)
        print(f"[mimic-cfg] 물리 DR term {len(removed)}개 제거: {removed}", flush=True)
