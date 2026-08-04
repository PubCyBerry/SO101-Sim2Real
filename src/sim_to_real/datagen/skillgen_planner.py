"""Isaac Lab **SkillGen** motion planner — cuRobo **v0.8(v2)** 재구현 (SO-101).

공식 `isaaclab_mimic.motion_planners.curobo.curobo_planner.CuroboPlanner` 는 cuRobo **v0.7**
API(`MotionGen`·`MotionGenConfig`·`WorldConfig`·`UsdHelper`)로 쓰여 있어 우리 이미지의
cuRobo 0.8 에서는 임포트 자체가 실패한다(`curobo.wrap`·`curobo.cuda_robot_model` 부재).
이 모듈은 **같은 역할·같은 인터페이스**를 v0.8 API 로 다시 구현한 것이다.

`isaaclab_mimic.datagen.data_generator.DataGenerator` 가 요구하는 표면(duck-typed):

===============================================  ====================================
`update_world_and_plan_motion(target_pose, ...)` subtask 시작 pose 로 충돌회피 전이 계획
`get_planned_poses() -> list[(4,4) tensor]`      계획을 EEF pose 열로 환원
`has_next_waypoint()` / `get_next_waypoint_ee_pose()` / `reset_plan()`
`step_size`                                     linear retiming step(None=끔)
`visualize_spheres`                             sphere viz 토글(여기선 항상 False)
`config.motion_noise_scale`                     waypoint 노이즈 스케일
===============================================  ====================================

## 공식본과 의도적으로 다른 3가지

1. **pose-goal 대신 IK→cspace.** SO-101 은 팔 5축이라 임의 6-DOF pose goal 이 대체로 불가다
   (`AGENTS.md` §5-DOF 공통 원칙). 공식 SkillGen 은 v0.7 `PoseCostMetric(reach_partial_pose=…)`
   로 **position-only** 전이를 계획하지만, 그러면 도착 orientation 이 방임돼 이어지는 subtask
   구간이 엉뚱한 손목 자세에서 시작한다. 여기서는 목표 pose 를 `SO101BoundedIK`(position 우선·
   orientation best-effort) 로 **도달 가능 manifold 에 투영**한 뒤 그 관절 배치로 `plan_cspace`
   한다 → 계획 자체가 항상 실현 가능하고, 도착 자세가 이어질 subtask 구간의 시작 자세와 **동일**
   하다. 투영에 쓰는 IK 는 env 어댑터(`target_eef_pose_to_action`)와 **같은 인스턴스 계약**이다.
2. **USD stage 자동 추출 대신 선언적 씬.** v0.8 은 obstacle **집합**이 planner 생성 시 굳는다
   (런타임엔 pose/enable 만 바뀐다). 그리고 우리 씬은 stage 를 그대로 넣으면 안 된다 —
   책상은 obstacle 로 넣으면 로봇 base 구가 상판 안이라 전 plan 이 start-collision 이고,
   그릇은 오목이라 solid convex 로 넣으면 내부가 허위충돌이다(`so101_contract.curobo_frames`).
   그래서 큐브 N개 + 그릇 rim ring 8개를 명시 선언하고 매 계획마다 pose 만 동기화한다.
3. **sphere/plan 시각화 미이식.** 진단 전용이고 v0.8 에 대응 API 가 없다.
   `visualize_spheres=False` 고정, 훅은 no-op 으로 남긴다.

## 프레임

mimic 이 주고받는 모든 pose 는 **cuRobo URDF solver 프레임**이다(tool frame = `tcp_grasp`).
env 어댑터(`sim_to_real.tasks.pick_cube.mimic_env`)가 sim USD ↔ URDF 변환을 **정확히 1회**
담당한다(`so101_contract.curobo_frames`). 이 planner 는 target pose 를 변환하지 않는다 —
두 곳에서 변환하면 프레임 오차가 상쇄돼 조용히 어긋난다(과거 이식본의 결정적 결함이 이것이었다).
obstacle 좌표만은 sim 씬에서 직접 읽어야 하므로 이 안에서 변환한다(:meth:`_object_pose_in_solver`).

self-check: `scripts/datagen/generate_mimic_dataset.py --self_test` (kit 부팅 + 실제 씬으로
프레임 왕복·IK 투영·plan_cspace·retiming 을 한 번에 검증한다 — 이 모듈은 isaaclab 의존이라
단독 `-m` 실행이 불가하다).
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.sphere_fit import SphereFitType
from curobo._src.geom.types import Cuboid
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase

from so101_contract.curobo_frames import CUBE_DIMS, bowl_ring, usd_to_urdf
from so101_contract.grasp_manifold import (
    ALPHA_SCAN_DEG,
    decompose,
    manifold_rotation,
    pan_from_position,
    partial_pose_axes_weight,
    project_pose_best_pan,
)
from so101_contract.eef_ik import SO101BoundedIK
from so101_contract.eef_kinematics import encode_rotation_matrices
from so101_contract.feature_codec import SO101_JOINT_ORDER

from sim_to_real.utils.cube_specs import CUBE_SIZES

ARM_JOINT_NAMES = tuple(SO101_JOINT_ORDER[:5])

# descend/contact 구간에서 collision 을 끄는 링크 — SM planner 와 같은 집합.
# 잡은 큐브를 물고 있는 자세에서 wrist sphere 가 큐브 obstacle 과 모델상 겹친다(짧은 wrist,
# 물리 접촉 없음 = sphere 보수 근사).
CONTACT_LINKS = ("gripper_link", "moving_jaw_so101_v1_link", "wrist_link", "wrist_cam_mount_link")

#: URDF joint limit clamp 여유(rad) — 경계에 정확히 붙으면 cuRobo 가 위반으로 볼 수 있다.
_LIMIT_MARGIN_RAD = 0.005

#: 전이 목표에서 이 거리 안의 큐브는 obstacle 을 끈다(m). 근거는
#: :meth:`SO101SkillGenPlanner._mute_cubes_near_target`. `tcp_grasp` 는 pinch 점이 아니라
#: 손가락 안쪽 중심에서 46 mm 떨어져 있어, 파지 자세에서도 tcp↔큐브가 60~90 mm 다.
#: 100 mm 면 그걸 덮으면서 옆 큐브(최소 이격 > 100 mm)는 회피 대상으로 남긴다.
_GRASP_MUTE_RADIUS_M = 0.10


def _diagnosis_is_clean(diagnosis: str) -> bool:
    """:meth:`SO101SkillGenPlanner.diagnose_state` 한 줄에 제약 위반이 하나도 없나.

    형식은 ``"start 제약: cspace=0.0000 self_collision=0.0000 scene_collision=0.7602"``.
    항목 이름을 하드코딩하지 않고 `이름=값` 을 전부 훑는다 — cuRobo 가 제약 항목을 늘려도
    조용히 통과하지 않게 한다. 파싱할 값이 하나도 없으면(진단 불가 문자열) 깨끗하다고
    보지 않는다.
    """
    values = re.findall(r"=\s*([-\d.eE+]+)", diagnosis)
    return bool(values) and all(abs(float(v)) < 1e-9 for v in values)

#: ★parked(홈) 자세에서만 만나는 sphere 오겹침 쌍 — self_collision_ignore 에 더한다.
#:
#: SkillGen 의 **첫 전이는 반드시 홈 자세에서 출발**하는데, 그 자세에서 cuRobo sphere 모델이
#: 4쌍을 충돌로 본다. 공식 `RobotDebugger` 실측 관통량은 **0.13~0.65 mm**(max_penetration
#: 0.000645 m) — 팔이 접혀 그리퍼가 어깨·전면 카메라 마운트 옆을 스치는 자세의 보수적 근사이고,
#: sim 은 매 에피소드 이 자세로 접촉 없이 동작한다. 즉 **false positive** 다.
#: 이걸 두면 start-in-collision 으로 `plan_cspace`·`plan_pose` 가 **전멸**한다(obstacle 을 전부
#: 꺼도 실패해 원인 추적이 어렵다 — 진단 경로는 `RobotDebugger.check_collision_at_config`).
#:
#: 같은 부류의 결함을 프로젝트가 이미 한 번 고쳤다(jaw↔wrist_cam_mount, `so101.yml` 에 baked).
#: 다만 여기서는 공유 `so101.yml` 을 **건드리지 않고** 런타임 파생본에만 넣는다 —
#: SM planner(124/124·372/372 A/B 검증본)에 회귀를 만들지 않기 위해서다. SM 은 홈 자세를
#: start 로 hard-check 하지 않아 이 완화가 필요 없다. 승격은 SM 재검증 후 별건.
PARKED_POSE_SELF_COLLISION_IGNORE: dict[str, tuple[str, ...]] = {
    "shoulder_link": ("wrist_link", "gripper_link"),
    "front_cam_mount_link": ("gripper_link", "moving_jaw_so101_v1_link"),
}


@dataclass
class SO101SkillGenPlannerCfg:
    """SkillGen cuRobo v0.8 planner 설정.

    `isaaclab.utils.configclass` 대신 평범한 dataclass 다 — env cfg 트리에 들어가지 않고
    드라이버가 직접 만들어 넘기는 런타임 객체라서 configclass 의 이점이 없다.
    """

    robot_yaml: str = "/workspace/assets/robots/so101.yml"
    urdf_path: str = "/workspace/assets/robots/urdf/so_arm101.urdf"

    #: obstacle 로 올릴 IsaacLab rigid object 이름(큐브). attach 대상도 이 목록에서 고른다.
    cube_names: tuple[str, ...] = ("Cube1",)
    #: 그릇 rigid object 이름. rim ring 8개 obstacle 의 중심을 여기서 읽는다.
    bowl_name: str = "Bowl"

    # ── planning 예산 ────────────────────────────────────────────────────────────
    num_ik_seeds: int = 64          # SM planner 와 동일(기본 32 는 5-DOF 에서 얇다)
    num_trajopt_seeds: int = 8
    # 실측(증강 run 1회): plan_cspace 실패 103 / 307. 그중 71 건은 start 제약이 **전부 0**
    # (joint bound 위반은 0 건) — 즉 start 무죄이고 경로 탐색이 예산 안에 못 찾은 것이다.
    # cuRobo 기본값이 5 인데 3 으로 깎여 있었다. 8 로 올린다.
    max_planning_attempts: int = 8
    enable_graph_attempt: int = 1
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    #: 0.01 → 0.02. gripper 추종 오차가 상시 **mean 7.2 mm**, 교란 직전 **13.3 mm** 로 실측됐다
    #: — 계획 여유가 추종 오차보다 작으면 계획은 무충돌인데 실행이 물체를 친다.
    #: 올리면 계획 실패가 늘 수 있으므로 성공률과 **함께** 잰다.
    collision_activation_distance: float = 0.02
    #: retiming 이후 사후 충돌 검사에서 **무시할 꼬리 프레임 수**. 파지 접근의 마지막 구간은
    #: 정의상 손가락이 큐브를 감싸므로 위반이 정상이다(그래서 `_mute_cubes_near_target` 이 있다).
    #: 그 앞 구간의 관통은 정상이 아니다 — 사용자 영상 관측 "cuRobo 구간에서 큐브를 굴림".
    post_check_tail_skip: int = 12
    #: 사후 검사 위반 임계(cuRobo 제약 단위). 0 이면 어떤 접촉도 불허.
    post_check_scene_tolerance: float = 0.0

    #: 목표 pose 투영 IK 의 position 허용치(m). orientation 은 5-DOF 라 게이트하지 않는다.
    ik_position_tolerance_m: float = 0.006
    #: position-only 폴백에서 받아올 후보 해 개수. 이 중 **DLS 최선해에 관절공간으로 가장 가까운**
    #: 해를 고른다 — 도착 자세를 이어질 source 구간과 맞춰 경계 wrist_roll 후려치기를 줄인다.
    ik_return_seeds: int = 16
    # `plan_cspace` 가 양끝 무충돌인데 실패했을 때 시도할 **대체 목표 배치** 수. 실측
    # (56 trial) 계획 실패 6건 중 5건이 이 부류라 예산이 아니라 목표가 문제였다.
    # 첫 후보가 성공하면 실행되지 않는 경로다(기존 성공 케이스 동작 불변).
    max_goal_alternates: int = 4

    # ── 계획 후처리 ──────────────────────────────────────────────────────────────
    #: linear retiming step(관절공간 rad). None = retiming 끔.
    motion_step_size: float | None = 0.05
    #: 마지막 waypoint 반복 횟수(도착 안정화). None = 반복 없음.
    n_repeat: int | None = None
    #: DataGenerator 가 waypoint 에 실어주는 gaussian 노이즈 스케일.
    motion_noise_scale: float = 0.0

    # ── attach ───────────────────────────────────────────────────────────────────
    attached_object_link: str = "attached_object"
    attach_num_spheres: int = 10
    #: 그리퍼 관절(rad)이 이 값보다 **작으면** 큐브를 잡은 것으로 본다.
    #: sim gripper: 닫힘 ≈ -0.17 rad, 열림 ≈ +1.4 rad.
    grasp_gripper_closed_rad: float = 0.6

    # ── 진단 ─────────────────────────────────────────────────────────────────────
    debug: bool = False
    #: v0.8 미이식 — 항상 False. DataGenerator 가 이 속성을 읽는다.
    visualize_spheres: bool = False
    visualize_plan: bool = False

    #: 큐브 크기를 sim 에서 못 읽었을 때의 **최후 폴백**(m). 평소엔 쓰이지 않는다 —
    #: 실제 값은 :meth:`SO101SkillGenPlanner._cube_size_m` 이 매번 env 에서 해석한다.
    cube_dims: float = CUBE_DIMS
    #: **world obstacle** 한 변에 더할 여유(m). obstacle 은 planner 생성 시 dims 가 굳어
    #: 나중에 못 바꾸므로(v0.8 은 pose/enable 만 가능), 큐브 크기 DR(25~40 mm 예정) 상한을
    #: 덮도록 여유를 준다. **obstacle 과대근사는 안전한 방향**이다 — 더 크게 피할 뿐이다.
    cube_obstacle_margin_m: float = 0.012
    #: attach blob 은 매 attach 마다 새로 만들므로 **여유 없이 실측 크기 그대로** 쓴다.
    #: 여기를 과대하게 잡으면 잡은 큐브가 그릇 rim 과 허위 충돌해 계획이 거부된다
    #: (실측: 0.05 blob vs 실제 0.040 = 부피 1.95배, run 당 START 충돌 1건).
    attach_dims_margin_m: float = 0.0

    #: parked 자세 sphere 오겹침 완화 쌍(:data:`PARKED_POSE_SELF_COLLISION_IGNORE` 참조).
    #: `{}` 로 두면 완화를 끈다 — 그러면 홈 자세에서 첫 전이가 전멸한다.
    extra_self_collision_ignore: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(PARKED_POSE_SELF_COLLISION_IGNORE))

    _diag: list[str] = field(default_factory=list, repr=False)


class SO101SkillGenPlanner(MotionPlannerBase):
    """SO-101 SkillGen 전이 planner (cuRobo v0.8, env 1개당 1 인스턴스)."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: SO101SkillGenPlannerCfg,
        env_id: int = 0,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=config.debug)
        self.config = config
        # `_write_scene_model()` 이 곧바로 `_cube_size_m()` 을 부르므로 **그 전에** 준비한다.
        self._logged_cube_size: dict[str, float] = {}

        # DataGenerator 가 직접 읽는 속성 3개
        self.step_size: float | None = config.motion_step_size
        self.visualize_spheres: bool = False
        self.visualize_plan: bool = False
        self.plan_visualizer = None

        # sim articulation joint 순서 → canonical(SO101_JOINT_ORDER) 인덱스
        self._joint_idx = [robot.joint_names.index(j) for j in SO101_JOINT_ORDER]

        # 목표 pose 투영 IK = env 어댑터와 같은 계약(so101_contract 단일 소스)
        self.ik = SO101BoundedIK.from_files(config.urdf_path, config.robot_yaml)

        self._planner = MotionPlanner(MotionPlannerCfg.create(
            robot=self._write_robot_model(),
            scene_model=self._write_scene_model(),
            num_ik_seeds=config.num_ik_seeds,
            num_trajopt_seeds=config.num_trajopt_seeds,
            position_tolerance=config.position_tolerance,
            orientation_tolerance=config.orientation_tolerance,
            optimizer_collision_activation_distance=config.collision_activation_distance,
            use_cuda_graph=False,  # 가변 batch/goalset — cuda graph 는 재사용 불가
        ))
        self._planner.warmup(enable_graph=False, num_warmup_iterations=2)

        if tuple(self._planner.joint_names) != ARM_JOINT_NAMES:
            raise RuntimeError(
                f"cuRobo planning joints {self._planner.joint_names} != {ARM_JOINT_NAMES} — "
                f"{config.robot_yaml} 의 lock_joints/cspace 를 확인하라"
            )
        self._tool_frame = self._planner.tool_frames[0]

        # 계획 상태
        self._plan_q: torch.Tensor | None = None   # (T, 5) cuRobo device
        self._plan_index = 0
        self._attached_cube: str | None = None

    # ══ robot / 씬 구성 ════════════════════════════════════════════════════════════
    def _write_robot_model(self) -> str:
        """`so101.yml` + parked 자세 self-collision 완화 → 임시 yml 경로.

        원본은 **읽기만** 한다(단일 소스 유지). 완화 근거 =
        :data:`PARKED_POSE_SELF_COLLISION_IGNORE`.
        ★dict 를 `MotionPlannerCfg.create(robot=...)` 에 직접 주입하면 과거 `RobotCfg.create`
        중복 kwarg 버그를 재현하므로 **반드시 경로**로 넘긴다.
        """
        spec = yaml.safe_load(open(self.config.robot_yaml, encoding="utf-8"))
        if not self.config.extra_self_collision_ignore:
            return self.config.robot_yaml
        ignore = spec["kinematics"].setdefault("self_collision_ignore", {})
        for link, others in self.config.extra_self_collision_ignore.items():
            ignore[link] = sorted(set(ignore.get(link) or []) | set(others))
            for other in others:  # 로더가 한쪽만 볼 수 있어 대칭으로 넣는다
                ignore[other] = sorted(set(ignore.get(other) or []) | {link})
        handle = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
        yaml.safe_dump(spec, handle)
        handle.close()
        return handle.name

    def _cube_obstacle(self, cube_name: str) -> str:
        return f"cube_{cube_name}"

    def _cube_size_m(self, cube_name: str) -> float:
        """이 큐브의 **현재** 한 변(m). 매 호출마다 sim 에서 다시 읽는다.

        큐브 크기는 DR 대상이 될 예정이라(25~40 mm) 상수로 굳히면 조용히 틀린다. 우선순위:

        1. ``env.cube_size_m`` — 크기 DR 훅이 쓰는 per-env 텐서(스칼라/(E,)/(E,C) 모두 허용)
        2. ``env.scene[name]`` prim 의 world scale × 저작 크기 — DR 이 USD scale 로 걸린 경우
        3. :data:`CUBE_SIZES` 정적 표(cube_specs 단일 소스)
        4. ``config.cube_dims`` 최후 폴백

        해석 결과는 값이 바뀔 때만 로그로 남긴다(매 프레임 도배 방지).
        """
        size = None
        table = CUBE_SIZES.get(cube_name)
        raw = getattr(self.env, "cube_size_m", None)
        if raw is not None:
            try:
                with torch.no_grad():
                    # float64 로 받는다 — float32 로 두면 0.025 가 0.0250000004 로 새어
                    # 로그·비교가 지저분해진다.
                    tensor = torch.as_tensor(raw, dtype=torch.float64).reshape(-1)
                index = self.env_id if tensor.numel() > 1 else 0
                if tensor.numel() > 1 and tensor.numel() % max(len(self.config.cube_names), 1) == 0:
                    # (E, C) 레이아웃이면 이 큐브 열을 고른다.
                    per_env = tensor.reshape(-1, len(self.config.cube_names))
                    index = None
                    size = float(per_env[self.env_id, self.config.cube_names.index(cube_name)])
                else:
                    size = float(tensor[index])
            except Exception as exc:  # noqa: BLE001 — 해석 실패는 폴백으로
                self._log(f"cube_size_m 해석 실패({type(exc).__name__}) — 정적 표로 폴백")
                size = None
        if size is None:
            size = table if table is not None else self.config.cube_dims
        if self._logged_cube_size.get(cube_name) != size:
            self._logged_cube_size[cube_name] = size
            self._log(f"{cube_name} 크기 {size * 1000:.1f} mm "
                      f"(정적표 {(table or 0) * 1000:.1f} mm)")
        return float(size)

    def _write_scene_model(self) -> str:
        """obstacle 집합(그릇 rim ring + 큐브 N개)을 임시 yml 로 굳힌다.

        v0.8 은 생성 후 obstacle 을 **추가**할 수 없다(pose/enable 만 가능) → 여기서 전부 선언하고
        실제 좌표는 :meth:`_sync_world` 가 매 계획마다 주입한다. placeholder 는 먼 좌표.
        """
        # ★obstacle dims 는 여기서 굳으면 못 바꾼다 → 크기 DR 상한을 덮도록 여유를 더한다.
        #   피하는 쪽 과대근사는 안전하다. 반대로 **attach blob 은 여유 0** 이다(허위 충돌).
        world = {"cuboid": {
            **bowl_ring(9.0, 9.0),
            **{self._cube_obstacle(name): {
                "dims": [self._cube_size_m(name) + self.config.cube_obstacle_margin_m] * 3,
                "pose": [9.0, 9.0, 0.02, 1.0, 0.0, 0.0, 0.0]}
               for name in self.config.cube_names},
        }}
        handle = tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False)
        yaml.safe_dump(json.loads(json.dumps(world)), handle)
        handle.close()
        return handle.name

    # ══ 진단 ═══════════════════════════════════════════════════════════════════════
    def _log(self, msg: str) -> None:
        if self.config.debug:
            print(f"[skillgen-planner env{self.env_id}] {msg}", flush=True)

    # ══ sim → solver 프레임 읽기 ═══════════════════════════════════════════════════
    def _object_pose_in_solver(self, asset_name: str) -> tuple[np.ndarray, np.ndarray]:
        """rigid object 의 (위치, quat wxyz) 를 URDF solver 프레임으로 읽는다.

        ★robot **root full SE(3)** 를 뺀다 — env origin 만 빼면 안 된다. 로봇이 책상 위
        (env-local z≈0.675)에 장착돼 있어 env origin 기준 좌표는 base_link 기준이 아니다.
        (공식 planner 는 env origin 만 빼는데 그건 로봇이 env 원점에 있는 Franka 전제다.)

        sim 텐서는 `env_loop` 의 inference_mode 에서 만들어진 **inference tensor** 라 autograd
        문맥에서 만지면 안 된다 → `no_grad` 안에서 읽고 즉시 numpy 로 내린다.
        """
        obj = self.env.scene[asset_name]
        with torch.no_grad():
            pos_b, quat_b = PoseUtils.subtract_frame_transforms(
                self.robot.data.root_pos_w[self.env_id: self.env_id + 1],
                self.robot.data.root_quat_w[self.env_id: self.env_id + 1],
                obj.data.root_pos_w[self.env_id: self.env_id + 1],
                obj.data.root_quat_w[self.env_id: self.env_id + 1],
            )
            pos_usd = pos_b[0].detach().cpu().numpy().astype(np.float64)
            quat_usd = quat_b[0].detach().cpu().numpy().astype(np.float64)
        # USD base_link → URDF solver: 위치는 Rz(90)+BASE_T, 회전은 Rz(90) 좌곱.
        pos_solver = np.asarray(usd_to_urdf(pos_usd), dtype=np.float64)
        quat_solver = _quat_mul(_quat_rot_z(math.pi / 2.0), quat_usd)
        return pos_solver, quat_solver

    def _current_arm_rad(self) -> np.ndarray:
        """현재 팔 관절 5축 radian (canonical 순서). **측정값 그대로** — clamp 없음."""
        q = self.robot.data.joint_pos[self.env_id, self._joint_idx]
        return q[:5].detach().cpu().numpy().astype(np.float64)

    def _clamp_to_urdf_limits(self, arm_rad: np.ndarray) -> np.ndarray:
        """계획 start 를 URDF joint limit 안으로 밀어넣는다.

        ★sim USD 의 joint limit(±105°)이 cuRobo URDF(±100°)보다 넓어서, 캘리브 마진만큼
        **홈 자세가 URDF 밖**일 수 있다(실측: `shoulder_lift` −1.7926 rad < URDF −1.74533).
        그 상태를 그대로 start 로 주면 cuRobo 가 bounds 위반을 `Start or End state in collision`
        으로 보고해 **모든** 계획이 실패한다 — obstacle 을 전부 꺼도 실패하므로 원인 추적이 어렵다.
        SM planner(`curobo_batch_planner._start_state`)도 같은 이유로 같은 clamp 를 한다.
        """
        limits = self.ik.joint_limits_rad
        clamped = np.clip(arm_rad, limits[:, 0] + _LIMIT_MARGIN_RAD, limits[:, 1] - _LIMIT_MARGIN_RAD)
        shifted = float(np.max(np.abs(clamped - arm_rad)))
        if shifted > 1e-6:
            self._log(f"start q 를 URDF limit 안으로 {math.degrees(shifted):.2f}° clamp")
        return clamped

    def _gripper_rad(self) -> float:
        return float(self.robot.data.joint_pos[self.env_id, self._joint_idx[5]].item())

    # ══ world 동기화 ═══════════════════════════════════════════════════════════════
    @property
    def _scene_checker(self):
        checker = getattr(self._planner, "scene_collision_checker", None)
        if checker is not None:
            return checker
        return self._planner.trajopt_solver.scene_collision_checker

    def _set_obstacle_pose(self, name: str, pos, quat_wxyz) -> None:
        pose = Pose(
            position=torch.tensor([list(pos)], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([list(quat_wxyz)], device="cuda", dtype=torch.float32),
        )
        self._scene_checker.update_obstacle_pose(name, pose, env_idx=0)

    def update_world(self) -> None:
        """IsaacLab 씬의 실좌표를 cuRobo obstacle 에 주입한다(매 계획 1회)."""
        for cube_name in self.config.cube_names:
            pos, quat = self._object_pose_in_solver(cube_name)
            self._set_obstacle_pose(self._cube_obstacle(cube_name), pos, quat)
            self._scene_checker.enable_obstacle(self._cube_obstacle(cube_name), True, env_idx=0)

        bowl_pos, _ = self._object_pose_in_solver(self.config.bowl_name)
        for name, entry in bowl_ring(float(bowl_pos[0]), float(bowl_pos[1])).items():
            x, y, z, qw, qx, qy, qz = entry["pose"]
            self._set_obstacle_pose(name, (x, y, z), (qw, qx, qy, qz))

    # ══ attach / detach ════════════════════════════════════════════════════════════
    @property
    def _attachment_manager(self):
        """설치본에 따라 `MotionPlanner.attachment_manager` 또는 `trajopt_solver.core` 경유."""
        manager = getattr(self._planner, "attachment_manager", None)
        if manager is not None:
            return manager
        solver = self._planner.trajopt_solver
        manager = getattr(solver, "attachment_manager", None)
        return manager if manager is not None else solver.core.attachment_manager

    def _attach_cube(self, cube_name: str, arm_rad: np.ndarray) -> bool:
        """잡은 큐브 blob 을 `attached_object` 링크에 붙이고 world 큐브는 끈다.

        `world_objects_pose_offset=None`(identity) — so101.yml 의 `attached_object` 가
        `tcp_grasp` 와 **같은 transform** 이라 blob 이 정확히 pinch 점에 놓인다.
        """
        # ★blob 은 **실측 크기 그대로**(여유 0). attach 는 매번 새로 만드니 크기 DR 도 따라간다.
        #   과대하게 잡으면 잡은 큐브가 그릇 rim 과 허위 충돌해 `plan_cspace` 가 START 충돌로
        #   거부한다(실측: 0.05 blob vs 실제 0.040 = 부피 1.95배).
        blob = self._cube_size_m(cube_name) + self.config.attach_dims_margin_m
        try:
            self._attachment_manager.attach(
                self._joint_state(arm_rad),
                [Cuboid(name=f"attached_{cube_name}", pose=[0, 0, 0, 1, 0, 0, 0],
                        dims=[blob] * 3)],
                link_name=self.config.attached_object_link,
                num_spheres=self.config.attach_num_spheres,
                sphere_fit_type=SphereFitType.VOXEL,
                world_objects_pose_offset=None,
                disable_obstacle_names=[self._cube_obstacle(cube_name)],
            )
        except Exception as exc:  # noqa: BLE001 — attach 실패는 무부착 계획으로 폴백
            self._log(f"attach FAIL {type(exc).__name__}: {exc} — 무부착 계획")
            return False
        self._attached_cube = cube_name
        self._log(f"attach ok: {cube_name}")
        return True

    def _detach(self) -> None:
        """detach + stale 큐브 obstacle 재-disable.

        cuRobo detach() 는 attach 때 끈 world 큐브를 **원래 pickup 좌표에** 되살린다. 그 자리는
        이미 빈 공간이라(큐브는 그리퍼/그릇에 있다) 유령 box 가 다음 계획을 start-collision 으로
        전멸시킨다. 즉시 다시 끄고, 다음 :meth:`update_world` 가 재배치+재활성한다.
        """
        if self._attached_cube is None:
            return
        try:
            self._attachment_manager.detach()
            self._scene_checker.enable_obstacle(
                self._cube_obstacle(self._attached_cube), False, env_idx=0)
        except Exception as exc:  # noqa: BLE001
            self._log(f"detach FAIL {type(exc).__name__}: {exc}")
        self._attached_cube = None

    def get_attached_objects(self) -> list[str]:
        return [] if self._attached_cube is None else [self._attached_cube]

    def has_attached_objects(self) -> bool:
        return self._attached_cube is not None

    def _is_grasped(self, cube_name: str) -> bool:
        """그리퍼 폐합 여부로 파지 판정(공식본과 같은 기준)."""
        grasped = self._gripper_rad() < self.config.grasp_gripper_closed_rad
        self._log(f"{cube_name} grasped={grasped} (gripper {self._gripper_rad():.3f} rad)")
        return grasped

    # ══ cuRobo 형변환 ══════════════════════════════════════════════════════════════
    def _joint_state(self, arm_rad) -> JointState:
        arm = torch.as_tensor(np.asarray(arm_rad, dtype=np.float32).reshape(1, 5), device="cuda")
        return JointState.from_position(arm, joint_names=list(self._planner.joint_names))

    def _mute_cubes_near_target(self, target_pose) -> list[str]:
        """전이 **목표가 파지 자세일 때** 그 큐브 obstacle 을 끈다. 끈 이름 목록 반환.

        ★왜 필요한가 — 파지 목표는 정의상 **손가락이 큐브를 감싼 자세**다. 그런데 attach 는
        파지 *후*에만 걸리므로, 파지로 가는 전이에서는 큐브가 여전히 obstacle 이다.
        그래서 cuRobo 는 목표를 `scene_collision` 으로 보고 계획 자체를 거부한다 —
        **구조적으로 통과 불가능한 목표**다.

        실측(증강 run, 실패 64건 귀속): START 는 깨끗한데 GOAL 이 `scene_collision` 인 경우가
        **45건(70%)**, 그 전부가 `Expected attached object: None`(=파지 전 approach 전이)이었다.
        `max_attempts` 를 3→8 로 올려도 68% 에 머문 이유가 이것이다 — 예산이 아니라 목표가
        불가능했다.

        범위를 좁게 잡는다: 목표 위치에서 :data:`_GRASP_MUTE_RADIUS_M` 안에 있는 큐브만 끈다.
        멀리 있는 큐브는 계속 회피 대상이다.
        """
        with torch.no_grad():
            target = np.asarray(target_pose.detach().cpu(), dtype=np.float64).reshape(4, 4)[:3, 3]
        muted = []
        for cube_name in self.config.cube_names:
            if cube_name == self._attached_cube:
                continue  # attach 가 이미 disable_obstacle_names 로 처리했다
            pos, _ = self._object_pose_in_solver(cube_name)
            if float(np.linalg.norm(pos - target)) <= _GRASP_MUTE_RADIUS_M:
                self._scene_checker.enable_obstacle(self._cube_obstacle(cube_name), False, env_idx=0)
                muted.append(cube_name)
        if muted:
            self._log(f"파지 목표 근방 큐브 obstacle mute: {muted}")
        return muted

    def _fk_matrices(self, arm_q: torch.Tensor) -> torch.Tensor:
        """(T,5) 관절 → (T,4,4) tool pose (solver 프레임, env device)."""
        state = self._planner.compute_kinematics(
            JointState.from_position(arm_q, joint_names=list(self._planner.joint_names)))
        tool = state.tool_poses.get_link_pose(self._tool_frame)
        pos = tool.position.detach().reshape(-1, 3).to(self.env.device)
        quat = tool.quaternion.detach().reshape(-1, 4).to(self.env.device)
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

    # ══ 계획 ═══════════════════════════════════════════════════════════════════════
    def _project_target_to_manifold(self, target_pose: torch.Tensor,
                                    seed_arm_rad: np.ndarray) -> np.ndarray | None:
        """목표 4×4(solver 프레임) → 도달 가능 관절 배치. 실패면 None.

        ★**position-only IK 를 쓰지 않는다.** 회전 weight 를 전부 0 으로 두면 도착 관절 배치의
        손목이 방임되는데, 곧이어 재생될 source 구간은 특정 `wrist_roll` 을 요구하므로 경계에서
        그 차이를 슬루 상한으로 쓸어야 한다(실측: 경계마다 8~14 프레임 5.0 rad/s 포화, 순변화
        −76°~−134°). 물리적으론 실행 가능하지만 source 엔 없는 bang-bang 이라 BC 학습에 해롭다.

        **main SM(99.9 %)은 position-only 를 어느 구간에서도 쓰지 않는다** — `plan_pose` 를
        기본 full-pose 로만 부르고, 대신 목표를 **처음부터 도달 가능 manifold 위에서 생성**한다.
        여기도 같은 순서다:

        ① **full-pose DLS** — 위치 잔차 ≤ tol 이면 채택. 이 해는 곧 `target_eef_pose_to_action`
           이 명령할 바로 그 자세라 경계가 **구조적으로 연속**이다.
        ② **`(α, ρ)` 후보 스캔** — 목표 **위치는 보존**하고 회전만 manifold 후보로 갈아끼우며
           full-pose DLS 를 다시 푼다. `|α − α₀|` 오름차순이라 원래 의도에 가까운 것부터 본다.
           회전을 **버리지 않는다**.
        ③ **축별 부분 pose** — 그래도 없으면 SO-101 이 못 만드는 **1축만** 푼다
           (`grasp_manifold.partial_pose_axes_weight`). 못 만드는 방향은 manifold 접선공간의
           여축 하나뿐이라(캐노니컬 top-down 에서 tool ŷ) 3축을 다 푸는 `track_position()` 과
           다르다 — roll/pitch 가 남아 `wrist_roll` 이 자유변수가 되지 않는다.
        ④ 그래도 없으면 **전이 실패**로 버린다. main SM 의 99.9 %는 오차를 줄여서가 아니라
           **오차 큰 시도를 버려서** 나온 값이다.
        """
        matrix = target_pose.detach().cpu().numpy().astype(np.float64).reshape(4, 4)
        position, rotation = matrix[:3, 3], matrix[:3, :3]
        tolerance = self.config.ik_position_tolerance_m

        # ① full-pose DLS
        dls = self._solve_full_pose(position, rotation, seed_arm_rad)
        reference_q = np.asarray(dls.joint_radians, dtype=np.float64)
        if dls.position_residual_m <= tolerance:
            self._log(f"IK 투영 ok(① DLS full-pose): pos {dls.position_residual_m * 1000:.2f}mm "
                      f"rot {math.degrees(dls.orientation_residual_rad):.1f}°")
            return reference_q

        # ② (α, ρ) 후보 스캔 — 위치 보존, 회전만 manifold 후보로
        best = self._scan_alpha_candidates(position, rotation, seed_arm_rad, tolerance)
        if best is not None:
            candidate_q, alpha_deg, residual = best
            self._log(f"IK 투영 ok(② α 스캔 α={alpha_deg:+.0f}°): pos {residual * 1000:.2f}mm")
            return candidate_q

        # ③ 축별 부분 pose (못 만드는 1축만 해제)
        solution = self._curobo_partial_pose_ik(position, rotation, reference_q)
        if solution is None:
            self._log(f"IK 투영 실패(④ 전이 폐기): DLS pos {dls.position_residual_m * 1000:.1f}mm "
                      f"({dls.reason}) · α 스캔 {len(ALPHA_SCAN_DEG)}개 · 부분 pose 도 미달")
        return solution

    def _solve_full_pose(self, position: np.ndarray, rotation: np.ndarray,
                         seed_arm_rad: np.ndarray):
        """full-pose DLS 1회 — 회전 weight 를 절대 0 으로 두지 않는다."""
        pose_vec = np.concatenate([position, encode_rotation_matrices(rotation[None], "rot6d")[0]])
        return self.ik.solve(pose_vec, seed_arm_rad, representation="rot6d")

    def _scan_alpha_candidates(self, position: np.ndarray, rotation: np.ndarray,
                               seed_arm_rad: np.ndarray, tolerance: float):
        """`(α, ρ)` manifold 후보를 훑어 full-pose DLS 가 통과하는 첫 해를 돌려준다.

        `pan` 은 위치가 고정하므로(팔의 pan 축은 solver z) 자유 파라미터는 α·ρ 둘이다.
        `decompose` 가 목표 회전을 그 두 값으로 닫힌 해로 쪼개 주므로, α 만 `ALPHA_SCAN_DEG`
        로 훑고 ρ 는 그대로 물려준다. 순서는 **원래 α 에 가까운 것부터** — SM 의 top-down 우선과
        달리 여기 목표는 source 궤적에서 온 것이라 의도 보존이 먼저다.

        Returns:
            `(관절 (5,), 채택 α(도), 위치 잔차 m)` 또는 None.
        """
        pan = pan_from_position(position)
        alpha0, rho = decompose(rotation, pan)
        order = sorted(ALPHA_SCAN_DEG, key=lambda deg: abs(math.radians(deg) - alpha0))
        for alpha_deg in order:
            candidate_rot = manifold_rotation(pan, math.radians(alpha_deg), rho)
            result = self._solve_full_pose(position, candidate_rot, seed_arm_rad)
            if result.position_residual_m <= tolerance:
                return (np.asarray(result.joint_radians, dtype=np.float64),
                        alpha_deg, float(result.position_residual_m))
        return None

    def _curobo_partial_pose_ik(self, position: np.ndarray, rotation: np.ndarray,
                                reference_q: np.ndarray) -> np.ndarray | None:
        """cuRobo 멀티시드 IK, **못 만드는 1축만 해제**한 부분 pose. 실패면 None.

        ★`ToolPoseCriteria.track_position()`(회전 3축 weight 0) 을 **쓰지 않는다**. SO-101 이
        만들 수 없는 방향은 manifold 접선공간의 여축 **하나뿐**이라 3축을 다 풀 이유가 없고,
        다 풀면 `wrist_roll` 이 자유변수가 돼 경계에서 슬루 상한으로 쓸리게 된다.
        `partial_pose_axes_weight` 가 `(x,y,z,roll,pitch,yaw)` 6-vector 를 만들고,
        `project_distance_to_goal=True` 로 **goal 프레임**에서 그 축을 잰다(축은 α 에 따라
        회전하므로 목표마다 다시 계산한다 — 하드코딩 금지).

        `reference_q`(full-pose DLS 최선해)에 **관절공간으로 가장 가까운** 해를 고른다.
        가중치는 균일하다 — 손목 불연속이 곧 가장 큰 관절 델타라 균일 L2 로도 자연히 잡힌다.
        """
        quaternion = _mat2quat(rotation)
        goal = GoalToolPose(
            tool_frames=self._planner.tool_frames,
            position=torch.tensor(position, dtype=torch.float32, device="cuda").view(1, 1, 1, 1, 3),
            quaternion=torch.tensor(quaternion, dtype=torch.float32, device="cuda").view(1, 1, 1, 1, 4),
        )
        pan = pan_from_position(position)
        alpha, rho = decompose(rotation, pan)
        weights = partial_pose_axes_weight(pan, alpha, rho)
        partial = {frame: ToolPoseCriteria(terminal_pose_axes_weight_factor=list(weights),
                                           non_terminal_pose_axes_weight_factor=[
                                               0.1 * w for w in weights],
                                           project_distance_to_goal=True)
                   for frame in self._planner.tool_frames}
        neutral = {frame: ToolPoseCriteria() for frame in self._planner.tool_frames}
        self._log(f"부분 pose weight (x,y,z,r,p,y) = "
                  f"{[round(float(w), 3) for w in weights]} @ α={math.degrees(alpha):+.1f}°")
        try:
            self._planner.update_tool_pose_criteria(partial)
            result = self._planner.ik_solver.solve_pose(
                goal, return_seeds=self.config.ik_return_seeds)
        except Exception as exc:  # noqa: BLE001
            self._log(f"cuRobo IK 예외 {type(exc).__name__}: {exc}")
            return None
        finally:
            self._planner.update_tool_pose_criteria(neutral)
        if result is None:
            return None

        # ★`js_solution` 은 **lock 된 gripper 까지 포함**한 full joint state 로 올 수 있다
        #   (실측 dim 6 = arm 5 + gripper). planning 관절만 **이름으로** 골라낸다 —
        #   앞 5개 슬라이스는 순서가 바뀌면 조용히 틀린다.
        state = result.js_solution
        rows = state.position.detach().reshape(-1, state.position.shape[-1])
        names = list(getattr(state, "joint_names", None) or self._planner.joint_names)
        if len(names) == rows.shape[-1]:
            index = [names.index(joint) for joint in self._planner.joint_names]
        else:
            index = list(range(len(self._planner.joint_names)))
        candidates = rows[:, index].cpu().numpy().astype(np.float64)

        success = result.success.detach().reshape(-1).cpu().numpy()
        errors = result.position_error.detach().reshape(-1).cpu().numpy()
        # 후보 수와 성공/오차 벡터 길이가 어긋나면(레이아웃 변화) 필터를 포기하고 전부 본다.
        if len(success) != len(candidates) or len(errors) != len(candidates):
            success = np.ones(len(candidates), dtype=bool)
            errors = np.zeros(len(candidates))
        keep = np.flatnonzero(success.astype(bool)
                              & (errors <= self.config.ik_position_tolerance_m))
        if keep.size == 0:
            best = float(np.min(errors)) if errors.size else float("inf")
            self._log(f"cuRobo IK position 최소 오차 {best * 1000:.1f}mm 초과 — 후보 없음")
            return None

        distances = np.linalg.norm(candidates[keep] - reference_q[None, :], axis=1)
        order = keep[np.argsort(distances)]
        # 나머지 후보는 버리지 않고 남긴다 — `_plan_cspace` 가 **양끝 무충돌인데** 실패하면
        # 목표 배치 자체가 경로상 곤란한 것이므로 다음 후보로 다시 시도한다(`_alternate_goals`).
        self._goal_alternates = [candidates[i] for i in order[1:]]
        pick = int(order[0])
        self._log(f"IK 투영 ok(cuRobo position-only, {keep.size}/{len(candidates)} 후보): "
                  f"pos {errors[pick] * 1000:.2f}mm · DLS 대비 관절거리 "
                  f"{distances.min():.3f} rad (최악 후보 {distances.max():.3f})")
        return candidates[pick]

    def _alternate_goals(self, position: np.ndarray, rotation: np.ndarray,
                         reference_q: np.ndarray) -> list[np.ndarray]:
        """1차 목표로 계획이 실패했을 때 쓸 **대체 목표 관절배치** 목록.

        실측(증강 56 trial): `plan_cspace` 실패 6건 중 **5건이 START·GOAL 둘 다 제약 위반 0**
        이었다. 충돌이 아니라 그 목표 배치로 가는 경로를 못 찾은 것이다. `max_attempts` 를
        3→8 로 올려도 남은 부류라 예산 문제가 아니다 — **목표를 바꿔야** 한다.

        같은 tool position 을 만족하는 IK 해는 여럿이라(5-DOF 라 더 많다) 그중 하나는 경로가
        열려 있을 수 있다. `_curobo_partial_pose_ik` 가 이미 계산해두고 버리던 후보를 재사용하며,
        없으면 그때 한 번 더 푼다.

        **첫 후보가 성공하면 이 경로는 아예 실행되지 않는다** — 기존 성공 케이스의 동작은
        비트 단위로 같다.
        """
        alternates = list(getattr(self, "_goal_alternates", []))
        if not alternates:
            # ①/② 가 바로 통과해 부분 pose IK 를 안 푼 경우 — 여기서 한 번 푼다.
            self._curobo_partial_pose_ik(position, rotation, reference_q)
            alternates = list(getattr(self, "_goal_alternates", []))
        return alternates[: self.config.max_goal_alternates]

    def _plan_cspace(self, goal_arm_rad: np.ndarray, start_arm_rad: np.ndarray) -> bool:
        """관절 목표까지 충돌회피 계획 → `self._plan_q` 채움.

        ★**링크 collision 을 끄지 않는다.** 예전엔 큐브를 든 전이에서 `CONTACT_LINKS`
        (gripper·jaw·wrist·wrist_cam) 를 `disable_link_collision` 했는데, 그 API 는 해당 링크의
        collision 을 **씬 전체에 대해** 끈다 → 전이 경로가 **그릇 rim ring 을 통과**할 수 있었다
        (사용자 영상 관찰: "grasp 후 전이 중에 그릇을 친다").

        그 코드의 명분 두 개가 모두 무효였다:
        * "손가락 sphere 가 attach 된 큐브 blob 과 겹친다" → `so101.yml` 의
          `self_collision_ignore.attached_object` 가 이미 그 4개 링크를 무시한다.
        * "wrist sphere 가 큐브 obstacle 과 겹친다" → attach 시 `disable_obstacle_names` 로
          world 큐브를 이미 끈다.
        """
        try:
            result = self._planner.plan_cspace(
                self._joint_state(goal_arm_rad),
                self._joint_state(start_arm_rad),
                max_attempts=self.config.max_planning_attempts,
                enable_graph_attempt=self.config.enable_graph_attempt,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"plan_cspace 예외 {type(exc).__name__}: {exc}")
            return False

        if result is None or not bool(result.success.detach().reshape(-1)[0].item()):
            # start 와 goal 을 **둘 다** 진단한다. 어느 쪽도 위반이 없으면 남는 건 경로 탐색
            # 예산 부족이므로 `max_planning_attempts` 를 올리는 게 맞고, 한쪽이 위반이면
            # 그 배치를 고쳐야 한다 — 로그만으로 갈리도록 이름을 붙여 남긴다.
            self._last_start_diagnosis = self.diagnose_state(start_arm_rad)
            self._log(f"plan_cspace 실패 — START {self._last_start_diagnosis}"
                      f" | GOAL {self.diagnose_state(goal_arm_rad)}")
            return False

        position = result.interpolated_trajectory.position.detach()
        while position.dim() > 2:
            position = position[0]
        last = result.interpolated_last_tstep
        n_step = int(last.detach().reshape(-1)[0].item()) if last is not None else position.shape[0]
        self._plan_q = position[: max(n_step, 2), :5].clone()
        self._plan_index = 0
        self._log(f"plan_cspace ok: {self._plan_q.shape[0]} waypoints")
        return True

    def scene_collision_per_waypoint(self, plan_q: torch.Tensor) -> np.ndarray | None:
        """계획 waypoint 별 `scene_collision` 위반량 (T,). 진단기가 없으면 None.

        `diagnose_state` 와 같은 rollout 을 쓰되 **horizon 축에 서로 다른 waypoint 를 채워**
        한 번에 T 개를 잰다(자세 하나를 horizon 전체에 복제하는 대신).
        """
        graph = getattr(self._planner, "graph_planner", None)
        if graph is None or plan_q is None or plan_q.shape[0] == 0:
            return None
        buffer = graph._max_act_buffer
        horizon = int(buffer.shape[1])
        out: list[float] = []
        for start in range(0, int(plan_q.shape[0]), horizon):
            chunk = plan_q[start: start + horizon].to(buffer.device, dtype=buffer.dtype)
            buffer.zero_()
            # 남는 슬롯은 마지막 waypoint 로 채운다(0 자세는 그 자체로 충돌일 수 있다).
            buffer[0, : chunk.shape[0], :] = chunk
            if chunk.shape[0] < horizon:
                buffer[0, chunk.shape[0]:, :] = chunk[-1]
            terms = graph.feasibility_rollout.compute_metrics_from_action(
                buffer).costs_and_constraints
            values = dict(zip(terms.constraints.names, terms.constraints.values))
            scene = values.get("scene_collision")
            if scene is None:
                return None
            out.extend(scene[0].detach().reshape(-1)[: chunk.shape[0]].cpu().numpy().tolist())
        return np.asarray(out, dtype=np.float64)

    def diagnose_state(self, arm_rad) -> str:
        """관절 배치의 **제약 항목별 위반량**을 한 줄로 돌려준다.

        계획 실패의 거의 모든 원인이 여기서 이름으로 드러난다:
        `cspace`>0 = joint bound 위반(→ :meth:`_clamp_to_urdf_limits` 확인),
        `self_collision`>0 = sphere 오겹침(→ :data:`PARKED_POSE_SELF_COLLISION_IGNORE`),
        `scene_collision`>0 = 실제 씬 충돌(→ :meth:`update_world` 동기화·attach 확인).
        전부 0 이면 start 는 무죄이고 goal 또는 경로 문제다.
        """
        graph = getattr(self._planner, "graph_planner", None)
        if graph is None:
            return "graph planner 없음 — 진단 불가"
        buffer = graph._max_act_buffer
        buffer.zero_()
        q = torch.as_tensor(np.asarray(arm_rad, dtype=np.float32).reshape(1, 1, 5),
                            device=buffer.device)
        buffer[:1, :, :] = q.expand(1, buffer.shape[1], 5)
        terms = graph.feasibility_rollout.compute_metrics_from_action(buffer).costs_and_constraints
        parts = [f"{name}={float(value[0].sum().item()):.4f}"
                 for name, value in zip(terms.constraints.names, terms.constraints.values)]
        return "start 제약: " + " ".join(parts)

    def _linearly_retime(self, step_size: float) -> None:
        """관절공간 등간격 재샘플 — 실행 속도를 균일하게 만든다(공식본 이식, 버전 무관)."""
        path = self._plan_q
        if path is None or path.shape[0] <= 1:
            return
        keep = [path[0]]
        for delta, waypoint in zip(torch.norm(path[1:] - path[:-1], dim=-1), path[1:]):
            if float(delta.item()) > 1e-6:
                keep.append(waypoint)
        if len(keep) <= 1:
            return
        waypoints = torch.stack(keep)
        distances = torch.norm(waypoints[1:] - waypoints[:-1], dim=-1)
        cumulative = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, 0)])
        total = cumulative[-1]
        if float(total.item()) < 1e-6:
            return
        num = int(torch.ceil(total / step_size).item()) + 1
        sampled = torch.linspace(0.0, float(total.item()), num, device=cumulative.device)
        idx = torch.clamp(torch.searchsorted(cumulative, sampled), 1, len(cumulative) - 1)
        weight = ((sampled - cumulative[idx - 1]) / (cumulative[idx] - cumulative[idx - 1])).unsqueeze(-1)
        self._plan_q = (1 - weight) * waypoints[idx - 1] + weight * waypoints[idx]

    # ══ MotionPlannerBase 인터페이스 ═══════════════════════════════════════════════
    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        **_kwargs,
    ) -> bool:
        """world 동기화 → attach → 전이 계획 → detach. 성공 여부 반환.

        Args:
            target_pose: 다음 subtask 시작 EEF pose, 4×4 (**URDF solver 프레임**).
            expected_attached_object: 이 구간에서 들고 있어야 하는 큐브 이름(없으면 None).
            env_id: DataGenerator 가 넘기는 env 인덱스(인스턴스 env_id 와 같아야 한다).
            step_size / enable_retiming: linear retiming 설정.
        """
        if env_id != self.env_id:
            raise ValueError(f"planner env_id {self.env_id} != 요청 env_id {env_id}")
        # ★`isaaclab_mimic.datagen.generation.env_loop` 는 전체를 `torch.inference_mode()` 로
        # 감싼다. cuRobo v0.8 의 LBFGS/TrajOpt 는 **autograd 가 필요**해서 그 안에서 돌리면
        # `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
        # 로 모든 계획이 죽는다(실측 2692건). 여기서만 inference_mode 를 풀고 grad 를 켠다.
        # sim 텐서(inference tensor)는 이 블록에 들이지 않는다 — 읽기는 no_grad 안에서 numpy 로.
        with torch.inference_mode(False), torch.enable_grad():
            return self._plan_transition(target_pose, expected_attached_object, step_size,
                                         enable_retiming)

    def _plan_transition(self, target_pose, expected_attached_object, step_size,
                         enable_retiming) -> bool:
        self.reset_plan()
        self.update_world()

        measured_arm = self._current_arm_rad()
        if expected_attached_object is not None:
            if expected_attached_object not in self.config.cube_names:
                self._log(f"{expected_attached_object} 는 obstacle 선언에 없다 — attach 생략")
            elif self._is_grasped(expected_attached_object):
                # attach blob 배치는 **측정 자세**로 — clamp 된 자세를 쓰면 blob 이 살짝 어긋난다.
                self._attach_cube(expected_attached_object, measured_arm)
            else:
                # 공식본과 동일: 미파지면 attach 를 건너뛴다(유령 큐브와 충돌할 수 있음).
                self._log(f"{expected_attached_object} 미파지 — attach 생략")

        start_arm = self._clamp_to_urdf_limits(measured_arm)
        # ★env 와 **같은 함수**로 목표를 manifold 에 투영한다. 여기서 투영하지 않으면 planner 가
        #   팔을 A 로 데려다 놓고 구간 실행이 B 로 명령해, 그 간극이 그대로 추종오차가 된다.
        with torch.no_grad():
            raw = np.asarray(target_pose.detach().cpu(), dtype=np.float64).reshape(4, 4)
        snapped = project_pose_best_pan(self.ik, raw, measured_arm)
        target_pose = torch.as_tensor(snapped, dtype=target_pose.dtype, device=target_pose.device)
        # ★큐브 obstacle mute 는 **마지막 수단**이다. mute 는 목표뿐 아니라 **경로 전체**에
        #   걸려서 계획이 큐브를 관통해도 막지 못한다 — 사용자 영상 관측으로 확인했다
        #   ("첫 cuRobo 구간에서 큐브와 충돌해서 큐브를 굴림", 실패 5건 중 3건).
        #   그래서 순서를 뒤집는다: 큐브를 **켠 채로** 먼저 풀고, 그게 안 될 때만 끈다.
        muted: list[str] = []
        try:
            self._goal_alternates = []
            goal_arm = self._project_target_to_manifold(target_pose, start_arm)
            if goal_arm is None:
                return False
            if not self._plan_cspace(goal_arm, start_arm):
                # ★START 가 제약 위반이면 목표를 바꾸든 큐브를 끄든 안 풀린다 — 실측에서
                #   그 케이스가 대체 목표 4개를 전부 태우고 같은 START 진단으로 4번 실패했다.
                if not _diagnosis_is_clean(getattr(self, "_last_start_diagnosis", "")):
                    self._log("START 가 제약 위반 상태 — 재시도 무의미하므로 생략")
                    return False
                # ① 같은 tool position 의 다른 IK 해로 재시도(큐브는 켜둔 채).
                alternates = self._alternate_goals(snapped[:3, 3], snapped[:3, :3], goal_arm)
                for index, alternate in enumerate(alternates, 1):
                    self._log(f"대체 목표 {index}/{len(alternates)} 로 재계획 "
                              f"(관절거리 {float(np.linalg.norm(alternate - goal_arm)):.3f} rad)")
                    if self._plan_cspace(alternate, start_arm):
                        break
                else:
                    # ② 그래도 안 되면 그때 큐브를 끈다(목표가 정말 큐브에 막힌 경우).
                    muted = self._mute_cubes_near_target(target_pose)
                    if not muted or not self._plan_cspace(goal_arm, start_arm):
                        return False
                    self._log("큐브 mute 후에야 계획 성공 — 이 경로는 큐브를 통과할 수 있다")
        finally:
            for name in muted:
                self._scene_checker.enable_obstacle(self._cube_obstacle(name), True, env_idx=0)
            self._detach()

        if enable_retiming is None:
            enable_retiming = step_size is not None
        effective_step = step_size if step_size is not None else self.step_size
        if enable_retiming and effective_step is not None:
            before = int(self._plan_q.shape[0])
            self._linearly_retime(effective_step)
            self._log(f"retime {before} → {int(self._plan_q.shape[0])} waypoints")

        # ★사후 충돌 검사는 **retiming 이후·큐브를 켠 상태**에서 한다. 두 구멍을 동시에 막는다:
        #   ① `_mute_cubes_near_target` 은 파지 목표를 통과시키려고 큐브를 끄는데, 그 mute 가
        #      **경로 전체**에 걸려 접근 중간에 큐브를 관통해도 안 막혔다(사용자 영상 관측).
        #   ② `_linearly_retime` 은 등간격 재샘플만 하고 충돌을 다시 보지 않는다 — 직선 보간이
        #      볼록하지 않은 여유를 가로지를 수 있다.
        #   꼬리 `post_check_tail_skip` 프레임은 파지 접근이라 위반이 정상이므로 뺀다.
        if not self._post_check_plan():
            return False
        return True

    def _post_check_plan(self) -> bool:
        """계획 경로가 (꼬리 제외) 씬을 관통하지 않는지 확인한다. 위반이면 False."""
        violations = self.scene_collision_per_waypoint(self._plan_q)
        if violations is None:
            return True
        body = violations[: max(len(violations) - self.config.post_check_tail_skip, 0)]
        if body.size == 0:
            return True
        worst = float(body.max())
        if worst <= self.config.post_check_scene_tolerance:
            return True
        self._log(f"사후 충돌 검사 실패 — 경로 {int(np.argmax(body))}/{len(violations)} 프레임에서 "
                  f"scene_collision {worst:.4f} (꼬리 {self.config.post_check_tail_skip} 제외). "
                  f"전이 폐기")
        self.reset_plan()   # 폐기한 계획이 남아 다음 호출에 소비되지 않도록 비운다
        return False

    def has_next_waypoint(self) -> bool:
        return self._plan_q is not None and self._plan_index < self._plan_q.shape[0]

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        """다음 waypoint 의 EEF pose 4×4(solver 프레임). 인덱스를 1 전진시킨다."""
        if not self.has_next_waypoint():
            raise IndexError("No more waypoints in the plan.")
        with torch.inference_mode(False), torch.enable_grad():
            pose = self._fk_matrices(self._plan_q[self._plan_index: self._plan_index + 1])[0]
        self._plan_index += 1
        return pose

    def reset_plan(self) -> None:
        self._plan_q = None
        self._plan_index = 0

    def get_planned_poses(self) -> list[torch.Tensor]:
        """계획 전체를 EEF pose 4×4 리스트로. 실행 인덱스를 건드리지 않는다."""
        if self._plan_q is None:
            return []
        with torch.inference_mode(False), torch.enable_grad():
            poses = [pose.detach().clone() for pose in self._fk_matrices(self._plan_q)]
        if self.config.n_repeat:
            poses.extend([poses[-1]] * self.config.n_repeat)
        return poses

    def get_planned_joint_positions(self) -> list[np.ndarray]:
        """계획 전체를 arm 관절 (5,) 리스트로 — :meth:`get_planned_poses` 와 **같은 순서·길이**.

        ★왜 필요한가 — planner 는 관절공간에서 해를 풀어 놓고, 공식 SkillGen 은 그걸 FK 로 pose
        화해 waypoint 에 담는다. env 가 그 pose 를 **다시 IK 로 푸는** 왕복은 5-DOF 에서 항등이
        아니고, 매 프레임 pan 이 재스캔되며 ρ(=`wrist_roll`)가 따라 움직인다 — 증강본에서 손목이
        일그러지는 직접 원인이다. 이미 푼 해를 같이 실어 보내면 왕복 자체가 사라진다.

        소비 = `generate_mimic_dataset` 이 waypoint 에 태깅 → env 의
        `take_pending_plan_joints()`. source 재생·보간·합성 구간에는 관절 해가 없으므로 기존
        pose→IK 경로가 그대로 쓰인다(회귀 표면이 전이 구간에 한정된다).
        """
        if self._plan_q is None:
            return []
        rows = [np.asarray(q.detach().cpu(), dtype=np.float64).reshape(5)
                for q in self._plan_q]
        if self.config.n_repeat:
            rows.extend([rows[-1]] * self.config.n_repeat)
        return rows

    def _update_visualization_at_joint_positions(self, joint_positions: torch.Tensor) -> None:
        """sphere viz 미이식(no-op). `visualize_spheres=False` 라 호출되지 않는다."""

    def get_planner_info(self) -> dict:
        info = super().get_planner_info()
        info.update(curobo_api="v0.8", transition_strategy="ik_projection+plan_cspace")
        return info


# ══ quaternion helper (numpy, wxyz) ═══════════════════════════════════════════════
def _quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_rot_z(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float64)


def _mat2quat(rotation: np.ndarray) -> np.ndarray:
    """3×3 회전행렬 → quat wxyz (Shepperd branch)."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] >= r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    quaternion = np.asarray(q, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)
