"""SO-101 cube_desk **teleop/데이터 substrate** — 태스크 중립 base 층.

leisaac Workshop 의 3층 상속 사다리(``so101_env_cfg`` → ``task_env_cfg`` →
``vials_to_rack_env_cfg``)를 우리 저장소에 맞춰 이식한 것의 **1층**이다.

Workshop 은 중간 ``task_env_cfg`` 층에서 카메라·라이트박스를 씬에 static 으로 얹어 여러
태스크가 공유했지만, 우리는 cube_desk 가 단일 USD 로 책상·조명을 이미 포함하고, 카메라는
leaf 의 ``PickCubeSceneCfg`` 에 **static** 으로 둔다(Workshop VisualCfg 패턴 이식). 그래서
사다리는 **base(이 파일, 무카메라 substrate) → pick_cube leaf(+카메라·VisualCfg 관측)** 2층
으로 접힌다. 무카메라 실행(teleop/smoke)은 base ``Teleop-v0`` 를 쓰거나 leaf 에서
``remove_pick_cube_cameras`` 로 카메라·image 관측을 떼어낸다. 새 태스크는 이 base 를 상속해
씬 오브젝트·성공 판정·카메라만 얹으면 된다(Workshop leaf 패턴).

이 층이 담는 것: 로봇(+contact 리포트) · 책상 USD · 조명 · slew joint 액션 ·
6-DOF joint 관측 · 리셋 이벤트 · sim/physx 설정. **성공/보상 없음**(teleop 은 무종료).
"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from sim_to_real.assets.robots.lerobot import SO101_FOLLOWER_CFG
from sim_to_real.assets.scenes.cube_desk import CUBE_DESK_CFG
from so101_contract import SO101_JOINT_ORDER
from sim_to_real.tasks.common.mdp import SlewLimitedJointPositionActionCfg
from sim_to_real.tasks.common.utils import SO101_JOINT_TARGET_MAX_VELOCITY


# Robot base position — recenter 로 world 원점(XY)에 배치.
# x: 0.0 (desk_left_edge=-0.44 + 440mm 장착), y: 0.0 (책상 앞 모서리 world y=-0.045 약간 뒤)
# z: desk_top(0.705) - base_min_z(0.0301) = 0.6749 (z 불변)
_ROBOT_POS = (0.0, 0.0, 0.6749)
# Identity rotation; articulation USD already faces the desk objects.
_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)


@configclass
class SO101BaseSceneCfg(InteractiveSceneCfg):
    """cube_desk + SO-101 follower(contact 리포트 on) + 조명. 태스크 오브젝트 없음."""

    # shared world assets (not per-env)
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # 조명: /World 계층(env 밖) 단일 배치 → InteractiveScene 이 복제하지 않음.
    # USD 광원은 scope 격리가 없어 {ENV_REGEX_NS}/Scene 안에 두면 env 수만큼 복제돼
    # N배 과노출(IsaacLab #4340/#1729). 그래서 scene.usd 에서 광원을 빼고 여기서
    # /World/Light(dome)·/World/KeyLight(distant) 1개씩만 author 한다. env=1·N-env
    # 모두 동일 노출(스케일링 불필요). 강도/색은 기존 scene.usd 값(2000/1800) 이식.
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.98, 0.95), angle=1.0),
        # RotateXYZ(-50,0,-35)° 등가 quat(wxyz) — 위에서 비스듬히 내리쬐어 입체감.
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.8644, -0.4031, -0.1271, -0.2725)),
    )

    # cube desk USD (책상·정적 지오메트리·큐브/그릇 prim 포함; 큐브/그릇은 leaf 가 wrap).
    # +y 0.01 shift: 정적 지오메트리를 로봇 기준 1cm 뒤로. 큐브/그릇 rigid body 는
    # leaf init_state(env-frame)로 동일 shift 반영(독립) — Scene translate 와 이중이동 없음.
    scene: AssetBaseCfg = CUBE_DESK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Scene",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.01, 0.0)),
    )

    # SO-101 follower articulation — 단일소스 SO101_FOLLOWER_CFG(assets/robots/lerobot.py)
    # 에서 actuator/solver/rigid_props 를 가져오고 씬 특화만 override.
    #  - spawn: activate_contact_sensors=True (jaw/gripper ↔ 큐브 접촉 리포트 활성; leaf 의
    #    contact_jaw/contact_gripper 센서가 소비. teleop 단독으로는 무해). Workshop
    #    ``S0101_CONTACT_GRASP_CFG`` 동치.
    #  - init_state: base pose + gripper 0(중립). elbow_flex 요청 +100°는 USD 상한 90° 로 캡.
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_FOLLOWER_CFG.spawn.replace(activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_ROBOT_POS,
            rot=_ROBOT_ROT,
            joint_pos={
                "shoulder_pan": math.radians(0.0),
                "shoulder_lift": math.radians(-100.0),
                "elbow_flex": math.radians(90.0),    # 요청 +100°, USD 상한 90° 로 캡
                "wrist_flex": math.radians(70.0),
                "wrist_roll": math.radians(-100.0),
                "gripper": 0.0,
            },
        ),
    )

    # EE frame transformer — Workshop ``ee_frame`` 이식. base 링크 기준 gripper/jaw world pose 를
    # 계산해 common mdp(``ee_frame_state``·``object_grasped``·``ee_near_object``)에 공급한다.
    # kinematic 센서라 렌더/카메라와 무관(``--enable_cameras`` 불요). target 순서 고정:
    #   idx0 = gripper (ee_frame_state 기준 프레임) · idx1 = jaw (object_grasped/ee_near_object jaw_pos).
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", name="gripper"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/jaw", name="jaw"),
        ],
    )


@configclass
class SO101ActionsCfg:
    """6-dim slew-limited joint position action (North Star joint order).

    offset 전면 제거(VLA-only): action = 절대 joint target. gripper velocity 는 태스크가
    필요 시 override(pick_cube 는 grasp 튜닝으로 gripper 2.5 로 낮춤).
    """

    arm: SlewLimitedJointPositionActionCfg = SlewLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=SO101_JOINT_ORDER,
        scale=1.0,
        use_default_offset=False,
        max_velocity=dict(SO101_JOINT_TARGET_MAX_VELOCITY),
    )


@configclass
class SO101PolicyObservationsCfg:
    """policy 관측 = 6-dim joint position (North Star 계약, 불변). 태스크가 subtask 그룹 추가."""

    @configclass
    class PolicyCfg(ObsGroup):
        """6-dim joint position in North Star order."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SO101_JOINT_ORDER)},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101BaseEventCfg:
    """리셋 이벤트 base — 씬 리셋 + 시작 포즈 jitter. 태스크가 오브젝트/시각 DR 추가."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 시작 포즈 jitter (DR): ±0.05 rad(~3°) — 실기 reset 편차 모사. velocity 0 유지.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class SO101TeleopEnvCfg(ManagerBasedRLEnvCfg):
    """SO-101 cube_desk teleop/데이터 substrate — 성공/보상 없음(무종료).

    pick_cube leaf 가 이걸 상속해 씬 오브젝트(큐브/그릇)·contact 센서·subtask 관측·
    성공 종료·DR 이벤트를 얹는다.
    """

    scene: SO101BaseSceneCfg = SO101BaseSceneCfg(num_envs=1, env_spacing=2.5)
    observations: SO101PolicyObservationsCfg = SO101PolicyObservationsCfg()
    actions: SO101ActionsCfg = SO101ActionsCfg()
    events: SO101BaseEventCfg = SO101BaseEventCfg()
    rewards = None       # substrate — RL 보상 없음
    terminations = None  # teleop — 무종료(leaf 가 성공/실패 종료 주입)

    # teleop/datagen device 배선(substrate 레벨 — teleop·record·replay 스크립트가 공용 사용).
    dynamic_reset_gripper_effort_limit: bool = True
    task_type: str = "so101leader"  # 활성 teleop/datagen device (use_teleop_device 가 설정)

    def use_teleop_device(self, teleop_device: str) -> None:
        """teleop/datagen device 선택. leisaac task 템플릿 등가.

        task_type 저장 + 직접 joint 제어(키보드/게임패드/state-machine) 시 중력 off
        (떨림 없는 결정적 제어). action term 구성(init_action_cfg)은 teleop/SM 드라이버가
        필요 시 별도 호출한다(기본 action = VLA용 slew joint target).
        """
        self.task_type = teleop_device
        if teleop_device in ["keyboard", "gamepad", "so101_state_machine"]:
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict, teleop_device) -> "object":
        """device 출력 → action tensor. vendored devices.action_process 에 위임.

        ``Device.advance()`` 가 호출한다. devices 는 isaac-sim 런타임서만 import 되므로
        지연 import(serial/lerobot 미설치 컨테이너서 패키지 import 안전 — Docker 정합 규칙1).
        """
        from sim_to_real.devices.action_process import preprocess_device_action

        return preprocess_device_action(action, teleop_device)

    def __post_init__(self) -> None:
        # Physics: 120 Hz simulation, 30 Hz policy (decimation=4)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        # GPU pipeline — contact/aggregate 버퍼(대규모 env 대비 상향).
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 2**16
        self.sim.physx.gpu_collision_stack_size = 2**29
