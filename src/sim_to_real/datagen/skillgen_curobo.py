"""공식 Isaac Lab SkillGen cuRobo planner의 SO-101 어댑터.

Isaac Lab/isaaclab_mimic 소스는 수정하지 않는다. 기존 cuRobo v0.8용 robot YAML을
공식 SkillGen pinned commit(v0.7 계열)이 읽는 ``robot_cfg`` schema로 런타임 변환하고,
URDF↔USD base frame의 고정 변환을 가상 root link로 흡수한다.
"""

from __future__ import annotations

import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.util_file import get_task_configs_path
import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from so101_contract.leader_calibration import SO101_FOLLOWER_USD_JOINT_LIMITS


_V07_KINEMATICS_KEYS = {
    "use_usd_kinematics",
    "isaac_usd_path",
    "usd_path",
    "usd_robot_root",
    "usd_flip_joints",
    "usd_flip_joint_limits",
    "urdf_path",
    "asset_root_path",
    "base_link",
    "ee_link",
    "collision_link_names",
    "collision_spheres",
    "collision_sphere_buffer",
    "extra_collision_spheres",
    "use_global_cumul",
    "self_collision_ignore",
    "self_collision_buffer",
    "mesh_link_names",
    "lock_joints",
    "extra_links",
    "cspace",
}

# scripts/cuRobo/curobo_batch_planner.py에서 물리 검증한 좌표 정합:
# p_urdf = Rz(+90°) p_usd + (0.01576, -0.02079, -0.03248).
# 따라서 USD root에서 URDF base로 가는 transform은 아래 역변환이다.
_USD_ROOT_TO_URDF_BASE_XYZ = (0.02079, 0.01576, 0.03248)
_USD_ROOT_TO_URDF_BASE_RPY = (0.0, 0.0, -1.5707963267948966)
_SKILLGEN_ROOT_LINK = "skillgen_usd_base_link"
_PLANNER_INIT_LOCK = threading.Lock()


def _write_aligned_urdf(source: Path, output_dir: Path) -> Path:
    """기존 URDF에 USD-frame 가상 root를 추가해 planner pose frame을 env와 맞춘다."""
    tree = ET.parse(source)
    robot = tree.getroot()
    if robot.find(f"./link[@name='{_SKILLGEN_ROOT_LINK}']") is not None:
        raise ValueError(f"source URDF already contains reserved link {_SKILLGEN_ROOT_LINK}")

    ET.SubElement(robot, "link", {"name": _SKILLGEN_ROOT_LINK})
    joint = ET.SubElement(robot, "joint", {"name": "skillgen_usd_base_fixed", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": _SKILLGEN_ROOT_LINK})
    ET.SubElement(joint, "child", {"link": "base_link"})
    ET.SubElement(
        joint,
        "origin",
        {
            "xyz": " ".join(str(value) for value in _USD_ROOT_TO_URDF_BASE_XYZ),
            "rpy": " ".join(str(value) for value in _USD_ROOT_TO_URDF_BASE_RPY),
        },
    )

    # planner joint bounds도 시뮬/실기기 계약의 단일 소스와 일치시킨다. 원본
    # editable URDF는 구형 ±95°/±100° 한계를 담고 있어, 실제 USD가 허용하는
    # shoulder_lift -105° 및 wrist_flex +105° source state를 invalid로 거부한다.
    for joint_name, (lower_deg, upper_deg) in SO101_FOLLOWER_USD_JOINT_LIMITS.items():
        joint_element = robot.find(f"./joint[@name='{joint_name}']")
        if joint_element is None:
            raise KeyError(f"URDF joint missing while syncing limits: {joint_name}")
        limit = joint_element.find("limit")
        if limit is None:
            raise KeyError(f"URDF limit missing for joint: {joint_name}")
        limit.set("lower", str(lower_deg * 3.141592653589793 / 180.0))
        limit.set("upper", str(upper_deg * 3.141592653589793 / 180.0))

    path = output_dir / "so_arm101_skillgen_aligned.urdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _write_safe_solver_configs(output_dir: Path) -> dict[str, str]:
    """5-DOF SO-101에서 안전한 공식 LBFGS solver 설정을 만든다.

    pinned cuRobo의 fused LBFGS CUDA kernel은 ``v_dim < 32``일 때 block의
    일부 thread가 ``__syncthreads`` 전에 return한다. Franka의 7x32 trajectory
    dimension에서는 드러나지 않지만, gripper를 lock한 SO-101 IK dimension 5에서는
    illegal memory access가 발생한다. 알고리즘/가중치는 그대로 두고 세 fused
    custom kernel만 cuRobo가 제공하는 PyTorch fallback으로 전환한다.
    """
    task_configs = Path(get_task_configs_path())
    outputs: dict[str, str] = {}
    for filename in ("gradient_ik_autotune.yml", "gradient_trajopt.yml"):
        source = task_configs / filename
        solver_cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
        lbfgs_cfg = solver_cfg["lbfgs"]
        lbfgs_cfg["use_cuda_kernel"] = False
        lbfgs_cfg["use_cuda_line_search_kernel"] = False
        lbfgs_cfg["use_cuda_update_best_kernel"] = False
        output = output_dir / filename
        output.write_text(yaml.safe_dump(solver_cfg, sort_keys=False), encoding="utf-8")
        outputs[filename] = str(output)
    return outputs


def build_so101_robot_config(
    source_yaml: str = "/workspace/assets/robots/so101.yml",
) -> tuple[str, dict, dict[str, str]]:
    """v0.8 robot model을 pinned official SkillGen cuRobo schema로 격리 변환한다."""
    source_path = Path(source_yaml).resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw_kinematics = raw.get("robot_cfg", raw).get("kinematics")
    if not isinstance(raw_kinematics, dict):
        raise ValueError(f"kinematics mapping missing in {source_path}")

    temp_dir = Path(tempfile.mkdtemp(prefix="so101_skillgen_curobo_"))
    source_urdf = Path(raw_kinematics["urdf_path"]).resolve()
    aligned_urdf = _write_aligned_urdf(source_urdf, temp_dir)

    kinematics = {
        key: value for key, value in raw_kinematics.items() if key in _V07_KINEMATICS_KEYS
    }
    kinematics["urdf_path"] = str(aligned_urdf)
    kinematics["base_link"] = _SKILLGEN_ROOT_LINK
    # Mimic 환경의 8D target pose는 USD gripper link origin 기준이다.
    kinematics["ee_link"] = "gripper_link"
    kinematics["lock_joints"] = {"gripper": 1.0}
    # 고정 base는 책상 상판에 끼워져 장착되므로 base collision spheres가 상판과
    # 의도적으로 겹친다. 움직이는 링크와 scene collision은 모두 유지하되 planner의
    # robot-vs-world query에서 고정 mount만 제외한다.
    kinematics["collision_link_names"] = [
        name for name in kinematics["collision_link_names"] if name != "base_link"
    ]
    collision_spheres = dict(kinematics["collision_spheres"])
    collision_spheres.pop("base_link", None)
    kinematics["collision_spheres"] = collision_spheres
    cspace = dict(kinematics["cspace"])
    # v0.8 builder 명칭을 official pinned v0.7 CSpaceConfig 계약으로 변환한다.
    cspace["retract_config"] = cspace.pop("default_joint_position")
    cspace.pop("null_space_maximum_distance", None)
    kinematics["cspace"] = cspace

    output = temp_dir / "so101_skillgen_curobo.yml"
    wrapped = {"robot_cfg": {"kinematics": kinematics}}
    output.write_text(yaml.safe_dump(wrapped, sort_keys=False), encoding="utf-8")
    solver_configs = _write_safe_solver_configs(temp_dir)
    return str(output), wrapped, solver_configs


def make_so101_planner_cfg(
    *,
    debug: bool = False,
    visualize_spheres: bool = False,
    visualize_plan: bool = False,
) -> CuroboPlannerCfg:
    """SO-101 PickCube용 공식 CuroboPlannerCfg를 만든다."""
    robot_config_file, _, solver_configs = build_so101_robot_config()
    cfg = CuroboPlannerCfg(
        robot_config_file=robot_config_file,
        robot_name="so101",
        ee_link_name="gripper_link",
        gripper_joint_names=["gripper"],
        gripper_open_positions={"gripper": 1.0},
        gripper_closed_positions={"gripper": 0.01},
        hand_link_names=["gripper_link", "moving_jaw_so101_v1_link"],
        attached_object_link_name="attached_object",
        collision_checker_type=CollisionCheckerType.MESH,
        # SO-101은 arm 5-DOF라 SkillGen이 물체 pose를 재배치한 뒤의 full 6D
        # target에 대해 IK 해 공간이 6/7-DOF 로봇보다 훨씬 좁다. 공식 planner의
        # seed 기본값(4/4)보다 넓게 탐색하되 48 GB GPU의 single-env baseline 안에서
        # 메모리 사용을 제한한다. num_ik_seeds는 아래 loader override에서 64로 둔다.
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=1.0 / 30.0,
        collision_cache_size={"obb": 200, "mesh": 300},
        trajopt_tsteps=32,
        collision_activation_distance=0.005,
        approach_distance=0.0,
        retreat_distance=0.0,
        grasp_gripper_open_val=0.60,
        enable_graph=True,
        enable_graph_attempt=5,
        max_planning_attempts=10,
        # SO-101 collision sphere 근사가 실제 source 경로에도 링크 간 false-positive를
        # 만든다. Scene/object/attached-object collision은 유지하고 self-collision만
        # MotionGen loader에서 비활성화한다. low-DOF finetune도 coarse plan을 반복
        # 거부하므로 기본 trajectory optimization 결과를 사용한다.
        enable_finetune_trajopt=False,
        time_dilation_factor=1.0,
        surface_sphere_radius=0.008,
        n_repeat=5,
        motion_step_size=0.05,
        visualize_spheres=visualize_spheres,
        visualize_plan=visualize_plan,
        debug_planner=debug,
        motion_noise_scale=0.0,
        collision_spheres_file=None,
        extra_collision_spheres={"attached_object": 100},
        position_threshold=0.005,
        # Mimic의 8D xyz+quat action/SE(3) 변환은 그대로 보존한다. 다만 5-DOF arm은
        # 재배치된 xyz에서 원본 quaternion을 정확히 동시에 만족시킬 수 없으므로,
        rotation_threshold=0.05,
        world_ignore_substrings=[
            "/World/defaultGroundPlane",
            "/curobo",
            "/Robot",
            # UsdHelper는 PhysicsCollisionAPI 여부와 무관하게 모든 geometry를 읽는다.
            # 각 asset의 render geometry를 제외해 실제 Collision prim과 중복 obstacle이
            # 생기지 않게 한다. 그렇지 않으면 Cube1 attach 시 Visual만 disable되고
            # 같은 위치의 Collision ghost가 남아 attached spheres와 즉시 충돌한다.
            "/Visual",
            "/Wall",
            "/Bottom",
            "DeskTopSurface",
            "top_camera",
            "wrist_camera",
            "front_camera",
        ],
    )
    # CuroboPlanner는 USD world 추출보다 먼저 warmup한다. 완전 빈 world는 pinned
    # cuRobo의 graph collision kernel에서 유효하지 않으므로 멀리 떨어진 cache seed를 둔다.
    cfg.get_world_config = lambda: WorldConfig.from_dict(
        {
            "cuboid": {
                "skillgen_cache_seed": {
                    "dims": [0.01, 0.01, 0.01],
                    "pose": [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0],
                }
            }
        }
    )
    # 공식 CuroboPlannerCfg에는 solver 파일 override 필드가 없으므로 SO-101
    # subclass가 초기화 구간에만 읽는 격리 속성으로 전달한다.
    cfg.skillgen_gradient_ik_file = solver_configs["gradient_ik_autotune.yml"]
    cfg.skillgen_gradient_trajopt_file = solver_configs["gradient_trajopt.yml"]
    return cfg


class SO101CuroboPlanner(CuroboPlanner):
    """single-gripper SO-101 보정과 실행 증거 카운터를 추가한 공식 planner."""

    def __init__(self, *args, **kwargs) -> None:
        self.skillgen_stats = {
            "planning_attempts": 0,
            "planning_successes": 0,
            "planning_failures": 0,
            "plans_with_attached_cube": 0,
            "attachment_checks": 0,
            "attachment_checks_passed": 0,
            "failure_statuses": {},
        }
        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            config = args[2]
        if config is None:
            raise ValueError("SO101CuroboPlanner requires a planner config")

        # CuroboPlanner가 MotionGenConfig에 solver 파일을 전달하지 않는 공식 2.3.2
        # API를 유지하면서, 해당 초기화 호출 하나에만 SO-101 fallback을 주입한다.
        import isaaclab_mimic.motion_planners.curobo.curobo_planner as official_planner

        official_motion_gen_config = official_planner.MotionGenConfig

        class _SO101MotionGenConfig:
            @staticmethod
            def load_from_robot_config(*loader_args, **loader_kwargs):
                loader_kwargs["gradient_ik_file"] = config.skillgen_gradient_ik_file
                loader_kwargs["gradient_trajopt_file"] = config.skillgen_gradient_trajopt_file
                loader_kwargs["self_collision_check"] = False
                loader_kwargs["self_collision_opt"] = False
                loader_kwargs["num_ik_seeds"] = 64
                return official_motion_gen_config.load_from_robot_config(*loader_args, **loader_kwargs)

        with _PLANNER_INIT_LOCK:
            official_planner.MotionGenConfig = _SO101MotionGenConfig
            try:
                super().__init__(*args, **kwargs)
            finally:
                official_planner.MotionGenConfig = official_motion_gen_config

        # SO-101 arm은 5-DOF라 xyz(3)+full orientation(3)을 독립적으로 만족할 수 없다.
        # 동일 query A/B에서 full pose와 단일 orientation 축 해제는 생성 분포 전반의
        # IK_FAIL을 제거하지 못했다. 따라서 cuRobo transition은 xyz collision-free path를
        # 계획하고 orientation은 이후 Mimic action 구간의 원본 8D target과 DLS controller가
        # 계속 추종한다. 이 분리는 underactuated arm에서 공식 partial-pose metric으로
        # 표현되며 source action이나 object-centric SE(3) 변환 계약은 바꾸지 않는다.
        self.plan_config.pose_cost_metric = PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=self.tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            project_to_goal_frame=True,
        )
        self._world_collision_diagnosed = False

    def _rigid_object_pose_in_planner_frame(self, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """동적 물체 pose를 static USD world와 같은 Robot root frame으로 반환한다.

        공식 planner의 static world extractor는 ``reference_prim_path=Robot``을
        사용하지만 dynamic sync는 env-origin을 빼는 데 그친다. Robot이 env origin에
        놓이는 Franka 예제에서는 두 frame이 같지만, SO-101은 책상 위에 translation과
        yaw가 적용되어 있으므로 반드시 Robot root의 full SE(3)를 제거해야 한다.
        """
        obj = self.env.scene.rigid_objects[object_name]
        object_pos, object_quat = PoseUtils.subtract_frame_transforms(
            self.robot.data.root_pos_w[self.env_id],
            self.robot.data.root_quat_w[self.env_id],
            obj.data.root_pos_w[self.env_id],
            obj.data.root_quat_w[self.env_id],
        )
        return self._to_curobo_device(object_pos), self._to_curobo_device(object_quat)

    def _sync_object_poses_with_isaaclab(self) -> None:
        """공식 object discovery를 유지하며 dynamic pose만 Robot frame으로 동기화한다."""
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects
        static_objects = getattr(self.config, "static_objects", [])
        updated: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []

        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue
            if any(static_name in object_name.lower() for static_name in static_objects):
                continue

            current_pos, current_quat = self._rigid_object_pose_in_planner_frame(object_name)
            pose_list = [
                float(current_pos[0].item()),
                float(current_pos[1].item()),
                float(current_pos[2].item()),
                float(current_quat[0].item()),
                float(current_quat[1].item()),
                float(current_quat[2].item()),
                float(current_quat[3].item()),
            ]
            if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                updated.append((object_name, object_path, current_pos, current_quat))

        for _, object_path, current_pos, current_quat in updated:
            self.motion_gen.world_coll_checker.update_obstacle_pose(
                object_path,
                self._make_pose(position=current_pos, quaternion=current_quat),
                update_cpu_reference=True,
            )
        self.logger.debug(f"SYNC: Updated {len(updated)} object poses in Robot-root planner frame")

    def _diagnose_start_world_collision(self) -> None:
        """debug 모드에서 현재 state와 충돌하는 obstacle을 한 번만 식별한다."""
        if self._world_collision_diagnosed or not self.config.debug_planner:
            return
        self._world_collision_diagnosed = True
        checker = self.motion_gen.world_coll_checker
        names = list(checker.get_obstacle_names(self.env_id))
        start_state = self._get_current_joint_state_for_curobo()
        baseline = self.motion_gen.check_start_state(start_state)
        self.logger.debug(f"START WORLD COLLISION baseline={baseline}, obstacles={names}")

        for name in names:
            checker.enable_obstacle(name=name, enable=False, env_idx=self.env_id)
        no_world = self.motion_gen.check_start_state(start_state)
        self.logger.debug(f"START WORLD COLLISION all-disabled={no_world}")

        colliding: list[str] = []
        for name in names:
            checker.enable_obstacle(name=name, enable=True, env_idx=self.env_id)
            result = self.motion_gen.check_start_state(start_state)
            if not result[0]:
                colliding.append(name)
                self.logger.debug(f"START WORLD COLLISION obstacle={name}, result={result}")
            checker.enable_obstacle(name=name, enable=False, env_idx=self.env_id)
        for name in names:
            checker.enable_obstacle(name=name, enable=True, env_idx=self.env_id)
        self.logger.debug(f"START WORLD COLLISION colliding={colliding}")

    def _check_object_grasped(self, gripper_pos, object_name: str) -> bool:
        """공식 Franka의 마지막 2-joint 가정 대신 SO-101 단일 gripper를 검사한다."""
        del gripper_pos
        self.skillgen_stats["attachment_checks"] += 1
        gripper_joint = float(self.robot.data.joint_pos[self.env_id, -1].item())
        grasped = gripper_joint < self.config.grasp_gripper_open_val
        if grasped:
            self.skillgen_stats["attachment_checks_passed"] += 1
        self.logger.info(
            f"Object {object_name} is grasped: {grasped} (SO-101 gripper={gripper_joint:.5f})"
        )
        return grasped

    def plan_motion(
        self,
        target_pose: torch.Tensor,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        if self.has_attached_objects():
            self.skillgen_stats["plans_with_attached_cube"] += 1
        # official generation.env_loop는 시뮬 전체를 torch.inference_mode()로 감싼다.
        # fused solver를 끈 5-DOF 안전 fallback은 autograd 기반이므로 planner 경계에서만
        # inference mode를 해제한다. target도 normal tensor로 복제해 backward 저장을 허용한다.
        # 공식 wrapper는 실패 status를 DEBUG 로그에만 남겨 대규모 headless 실행의
        # failure taxonomy를 만들 수 없다. MotionGen 호출을 이 planner 인스턴스 안에서만
        # 감싸 마지막 result를 수집하고, 공식 planning/execution 코드는 그대로 호출한다.
        original_plan_single = self.motion_gen.plan_single
        self._last_motion_gen_result = None
        self._last_motion_gen_call = None

        def _capture_result(*plan_args, **plan_kwargs):
            self._last_motion_gen_call = (plan_args, plan_kwargs)
            result = original_plan_single(*plan_args, **plan_kwargs)
            self._last_motion_gen_result = result
            return result

        self.motion_gen.plan_single = _capture_result
        try:
            with torch.inference_mode(False), torch.enable_grad():
                target_pose_normal = target_pose.detach().clone()
                success = super().plan_motion(target_pose_normal, step_size, enable_retiming)
        finally:
            self.motion_gen.plan_single = original_plan_single

        if not success:
            result = self._last_motion_gen_result
            status = str(getattr(result, "status", "missing_result"))
            statuses = self.skillgen_stats["failure_statuses"]
            statuses[status] = statuses.get(status, 0) + 1
            position_error = getattr(result, "position_error", None)
            rotation_error = getattr(result, "rotation_error", None)

            def _finite_min(value) -> float | None:
                if not isinstance(value, torch.Tensor) or value.numel() == 0:
                    return None
                finite = value.detach()[torch.isfinite(value)]
                return float(finite.min().item()) if finite.numel() else None

            self.logger.info(
                "MotionGen failure: "
                f"status={status}, attached={self.has_attached_objects()}, "
                f"min_position_error={_finite_min(position_error)}, "
                f"min_rotation_error={_finite_min(rotation_error)}"
            )
            if self.config.debug_planner and self._last_motion_gen_call is not None:
                # 실패가 kinematic reachability인지 world collision인지 한 번의 같은
                # query로 분리한다. 진단 뒤 모든 obstacle enable 상태를 즉시 복구한다.
                checker = self.motion_gen.world_coll_checker
                obstacle_names = list(checker.get_obstacle_names(self.env_id))
                for name in obstacle_names:
                    checker.enable_obstacle(name=name, enable=False, env_idx=self.env_id)
                try:
                    plan_args, plan_kwargs = self._last_motion_gen_call
                    with torch.inference_mode(False), torch.enable_grad():
                        no_world_result = original_plan_single(*plan_args, **plan_kwargs)
                    self.logger.info(
                        "MotionGen no-world diagnostic: "
                        f"status={getattr(no_world_result, 'status', 'missing_result')}, "
                        f"success={bool(no_world_result.success.item())}"
                    )
                finally:
                    for name in obstacle_names:
                        checker.enable_obstacle(name=name, enable=True, env_idx=self.env_id)
        return success

    def update_world_and_plan_motion(self, *args, **kwargs) -> bool:
        self.skillgen_stats["planning_attempts"] += 1
        self.update_world()
        self._diagnose_start_world_collision()
        # 공식 wrapper도 plan 직전에 update_world()를 호출한다. 두 번째 sync는
        # 멱등이며 object validation까지 그대로 보존한다.
        success = super().update_world_and_plan_motion(*args, **kwargs)
        key = "planning_successes" if success else "planning_failures"
        self.skillgen_stats[key] += 1
        return success
