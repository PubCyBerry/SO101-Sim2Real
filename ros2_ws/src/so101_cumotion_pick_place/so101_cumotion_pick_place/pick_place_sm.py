#!/usr/bin/env python3
"""SO-101 cuMotion + ROS 2 pick-and-place state machine (PATH E, cube_desk).

ground-truth 큐브/그릇 포즈(/cube_poses, /bowl_pose, base_link frame)를 받아 MoveItPy +
cuMotion planner 로 명시 단계 pick-and-place 를 수행한다. 기존 in-process Lula SM 의 8단계
구조를 ROS 노드로 이식한 것. cuMotion 이 articulation frame 에서 직접 계획하므로 Lula↔USD
정합 잔차가 없다.

단계(큐브 1개):
  open → approach(상공) → descend(grasp z, tilt) → grasp-close → lift → check →
  transport(그릇 상공) → place-descend → release → retreat

5DOF 처리: set_from_ik 에 여러 grasp 자세(수직→앞으로 tilt) 후보를 순차 시도해 첫 성공
자세로 joint-space 목표를 잡고 cuMotion 으로 계획한다(완전 top-down 불가 회피).

전제: ros2 launch so101_cumotion_pick_place pick_place.launch.py
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
import rclpy.logging
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

EE_FRAME = "gripper_frame_link"
PLANNING_GROUP = "manipulator"
GRIPPER_GROUP = "gripper"
JAW_LINK = "moving_jaw_so101_v1_link"  # 모터 jaw (Isaac USD body "jaw" 대응)
FIX_LINK = "gripper_link"              # 고정 finger 마운트
# grasp 접점(두 손가락 사이) = JAW_LINK·FIX_LINK 의 FK 중점 + grasp_z_offset(손가락 끝 보정).
# (이전 JAW_GRASP_OFFSET=Isaac jaw-frame 추측값은 URDF frame 과 7cm 어긋나 폐기 — 실측 두 link 중점으로 대체.)
BASE_FRAME = "base_link"  # bridge 가 base_link 기준 TF 로 물체 포즈 publish
GRIPPER_ACTION = "/follower/gripper_controller/gripper_cmd"
JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"  # bridge 입력(TopicBasedSystem cmd) 진단용
GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = -0.16
CUBE_FRAMES = ["Cube1", "Cube2", "Cube3", "Cube4"]
BOWL_FRAME = "Bowl"


class ObjectPoseStore(Node):
    """bridge 가 publish 하는 TF(base_link→CubeN/Bowl)로 물체 ground-truth 포즈를 조회한다.

    `.cubes` / `.bowl` 는 매 접근마다 최신 TF 를 lookup 하는 property (grasp 후 z 상승 판정에
    실시간 값 필요). 해소되는 큐브 프레임만 리스트에 포함한다.
    """

    def __init__(self) -> None:
        super().__init__("pick_place_object_store")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # bridge 입력 command 진단용: gripper close 실패가 controller→bridge 미전달인지(command 1.5
        # 유지) vs 물리 반력으로 안 닫히는지(command -0.16 인데 state 1.5) 구분.
        self.last_grip_cmd: float | None = None
        self.create_subscription(JointState, JOINT_COMMANDS_TOPIC, self._grip_cmd_cb, 10)

    def _grip_cmd_cb(self, msg: JointState) -> None:
        try:
            if "gripper" in msg.name:
                self.last_grip_cmd = float(msg.position[list(msg.name).index("gripper")])
            elif len(msg.position) >= 6:
                self.last_grip_cmd = float(msg.position[5])  # joint order 마지막 = gripper
        except (IndexError, ValueError):
            pass

    def _xyz(self, frame: str) -> tuple[float, float, float] | None:
        try:
            t = self.tf_buffer.lookup_transform(BASE_FRAME, frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        tr = t.transform.translation
        return (tr.x, tr.y, tr.z)

    @property
    def cubes(self) -> list[tuple[float, float, float]]:
        out = []
        for f in CUBE_FRAMES:
            p = self._xyz(f)
            if p is not None:
                out.append(p)
        return out

    @property
    def bowl(self) -> tuple[float, float, float] | None:
        return self._xyz(BOWL_FRAME)


class PickPlaceSM:
    def __init__(self, robot: MoveItPy, store: ObjectPoseStore, params: dict) -> None:
        self.robot = robot
        self.arm = robot.get_planning_component(PLANNING_GROUP)
        self.store = store
        self.p = params
        self.logger = rclpy.logging.get_logger("pick_place_sm")
        self.cumotion_params = MultiPipelinePlanRequestParameters(robot, ["cumotion"])
        self.ompl_params = MultiPipelinePlanRequestParameters(robot, ["ompl_rrtc"])
        # ⚠ 5-DOF 핵심: MoveIt/cuMotion 의 goal 샘플러는 pose/position goal 을 IK("랜덤 orientation
        # +IK")로 풀어 5-DOF 에선 거의 모든 랜덤 orientation 이 도달 불가 → thin achievable manifold
        # 를 못 찾고 "Unable to sample valid states for goal tree"/INVERSE_KINEMATICS_FAILURE 로
        # 실패한다(position-only 도 동일). → goal 을 JOINT config 로 준다: FK 랜덤 샘플링(5-DOF-aware,
        # in-process joint_fk SM 과 동일 원리)으로 target 에 도달하는 config 를 찾고 set_from_ik 로
        # 정밀화한 뒤, planner 는 joint→joint collision-free 모션만 푼다(이건 cuMotion/OMPL 잘 함).
        # 라우팅: grasp 는 OMPL 우선(XRDF-sphere start-validity 거부 회피), transport 는 cuMotion 우선.
        self.grasp_order = ((self.ompl_params, "OMPL"), (self.cumotion_params, "cuMotion"))
        self.transport_order = ((self.cumotion_params, "cuMotion"), (self.ompl_params, "OMPL"))
        self.sample_rs = RobotState(robot.get_robot_model())  # FK 샘플링용 재사용 RobotState
        # FK 샘플 self-collision 배제용. set_to_random_positions 는 joint bounds 만 지키고 self-collision
        # (shoulder↔lower_arm 등)은 무시 → colliding goal config 면 OMPL "goal tree 샘플 실패"/cuMotion
        # INVALID_INITIAL_CSPACE_POSITION 으로 planning 실패. is_state_colliding 으로 거른다.
        self.psm = robot.get_planning_scene_monitor()
        self.gripper = ActionClient(store, ParallelGripperCommand, GRIPPER_ACTION)

    def _colliding(self, rs) -> bool:
        """현재 rs config 가 self-collision 인지(planning scene). world collision object 는 미등록이라
        self-collision 만 본다 — grasp goal/start 가 colliding 이면 planner 가 못 쓰므로 사전 배제."""
        with self.psm.read_only() as scene:
            return scene.is_state_colliding(robot_state=rs, joint_model_group_name=PLANNING_GROUP)

    def _diag_grasp(self, cx: float, cy: float, cz: float) -> None:
        """grasp close 직후 실제 로봇 상태 진단: 그리퍼 닫힘(gripper joint) + 모터 jaw·고정 finger·
        계산 grasp_pt 위치 vs 큐브. 두 손가락이 큐브를 사이에 두는지 + JAW_GRASP_OFFSET 정합 확인."""
        try:
            with self.psm.read_only() as scene:
                cur = scene.current_state
                cur.update()
                grip = float(cur.get_joint_group_positions(GRIPPER_GROUP)[0])
                jaw = cur.get_pose(JAW_LINK).position           # 모터 jaw
                fix = cur.get_pose("gripper_link").position      # 고정 finger 마운트
                ee = cur.get_pose(EE_FRAME).position             # TCP (arm 추종 확인용 — FK ee_z 와 비교)
                gpt = self._grasp_point(cur)                     # 계산 grasp 접점
                axv = self._grasp_axis_vert(cur)                 # grasp axis 수직 비율(0 수평=좋음)
            cmd = self.store.last_grip_cmd
            self.logger.info(
                f"DIAG grasp: grip={grip:.3f} grip_cmd={cmd if cmd is None else round(cmd, 3)} axis_vert={axv:.2f} "
                f"jaw=({jaw.x:.3f},{jaw.y:.3f},{jaw.z:.3f}) "
                f"fix=({fix.x:.3f},{fix.y:.3f},{fix.z:.3f}) gpt=({gpt[0]:.3f},{gpt[1]:.3f},{gpt[2]:.3f}) "
                f"cube=({cx:.3f},{cy:.3f},{cz:.3f})"
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"DIAG grasp 진단 실패: {e}")

    # ── 저수준 ────────────────────────────────────────────────────────
    def _plan_exec(self, plan_params) -> bool:
        result = self.arm.plan(multi_plan_parameters=plan_params)
        if result:
            self.robot.execute(result.trajectory, controllers=[])
            return True
        return False

    @staticmethod
    def _tool_tilt(q) -> float:
        """tool z 축이 수직 아래(-z_base)에서 기운 각(deg). q=geometry_msgs Quaternion. 0=완전 down."""
        zz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)  # R[2,2] = tool z 의 base-z 성분
        return math.degrees(math.acos(max(-1.0, min(1.0, -zz))))

    @staticmethod
    def _quat_to_R(q) -> np.ndarray:
        """geometry_msgs Quaternion(x,y,z,w) → 3×3 rotation matrix."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _grasp_point(self, rs) -> np.ndarray:
        """현재 config 의 두 손가락 사이 grasp 접점(base_link frame) = moving_jaw·gripper_link FK 중점.

        ROS SM 이 gripper_frame_link(TCP)을 큐브에 맞추면 실제 손가락은 TCP 보다 위라 못 감싼다.
        실측(DIAG)상 두 손가락 사이는 JAW_LINK·FIX_LINK 원점 중점에 가깝다 → 이 중점을 큐브에 맞춘다.
        link 원점이 손가락 끝보다 위라, grasp_z_offset(양수)으로 중점을 큐브보다 약간 위에 둬 손가락
        끝이 큐브 높이에 오게 한다(_move_to grasp target z = cz + grasp_z_offset).
        """
        jaw = rs.get_pose(JAW_LINK).position
        fix = rs.get_pose(FIX_LINK).position
        return np.array([(jaw.x + fix.x) / 2.0, (jaw.y + fix.y) / 2.0, (jaw.z + fix.z) / 2.0])

    def _grasp_axis_vert(self, rs) -> float:
        """grasp axis(두 손가락 jaw→fix 벡터)의 수직 성분 비율 [0,1]. 0=수평(큐브 옆면 잡음, 좋음),
        1=수직(큐브 위아래 누름, grasp 실패). 중점만 맞추면 axis 방향이 random 이라 이 제약이 필요."""
        jaw = rs.get_pose(JAW_LINK).position
        fix = rs.get_pose(FIX_LINK).position
        axis = np.array([fix.x - jaw.x, fix.y - jaw.y, fix.z - jaw.z])
        n = float(np.linalg.norm(axis))
        return abs(axis[2]) / n if n > 1e-6 else 1.0

    def _fk_sample_goal(self, x: float, y: float, z: float, *, tilt_max_deg: float, stage: str = "transport"):
        """random FK 샘플링으로 (x,y,z) 도달 manipulator joint config 를 찾고 set_from_ik 로 정밀화.

        5-DOF 라 pose/position goal 을 planner IK 샘플러가 못 푼다(랜덤 orientation 이 거의 도달
        불가) → 도달 가능 config 자체를 FK 로 찾는다(in-process joint_fk SM 과 동일 원리). set_to_random
        _positions 는 joint bounds 내에서만 샘플 → 한계 자동 준수.

        stage:
          - "grasp"/"approach": **grasp 접점(두 손가락 midpoint)** 을 (x,y,z)에 맞춘다(TCP 아님).
            tilt 를 [grasp_tilt_min, tilt_max] 로 hard-filter 해 약tilt(수직 top-down, jaw 가 큐브 옆/
            아래로 안 감)를 배제하고, 통과한 config 중 **접점-큐브 거리(d) 최소**를 골라 xy 정합을 확보한다.
            정밀화는 grasp_pt→target shift 로 EE target 보정.
          - "transport": 기존대로 TCP(gripper_frame_link)를 (x,y,z)에 맞춘다(잡은 후 운반/배치).
        """
        target = np.array([x, y, z])
        rs = self.sample_rs
        n = int(self.p.get("fk_samples", 12000))
        grasp_stage = stage in ("approach", "grasp")
        # grasp 는 좁은 gate(xy 정합), transport(lift/운반/배치)는 넓은 gate(도달성). 좁은 gate 를
        # transport 에 쓰면 먼 target 도달 config 를 못 찾아 "FK 도달 config 없음" 으로 lift 실패한다.
        gate = float(self.p.get("fk_pos_gate", 0.05)) if grasp_stage \
            else max(float(self.p.get("fk_pos_gate", 0.05)), float(self.p.get("fk_pos_gate_transport", 0.04)))
        tilt_min = float(self.p.get("grasp_tilt_min", 45.0)) if grasp_stage else 0.0
        best_q = None
        best_ee = None
        best_pt = None
        best_d = 1e9
        best_score = 1e9
        for _ in range(n):
            rs.set_to_random_positions()
            if grasp_stage:
                # ⚠ CLOSED 로 중점 계산: gripper 를 닫으면 moving_jaw 가 revolute 축으로 회전해 중점이
                # z 로 ~3cm 이동한다. OPEN 중점을 큐브에 맞추면 close 후 손가락이 큐브 위로 떠 못 잡는다
                # → close 후 중점이 큐브에 오도록 처음부터 CLOSED 기준으로 정합(실측 open0.055→close0.087).
                rs.set_joint_group_positions(GRIPPER_GROUP, [GRIPPER_CLOSED])
            rs.update()
            ee = rs.get_pose(EE_FRAME)
            if grasp_stage:
                pt = self._grasp_point(rs)
            else:
                pt = np.array([ee.position.x, ee.position.y, ee.position.z])
            d = float(np.linalg.norm(pt - target))
            if d > gate:
                continue
            tilt_deg = self._tool_tilt(ee.orientation)
            if tilt_deg > tilt_max_deg:
                continue
            # grasp/approach: 약tilt(수직 top-down) hard-filter. 강tilt 라야 jaw 가 큐브 옆/아래로 가
            # 감싼다(수직은 고정 finger 가 책상에 닿아 모터 jaw 가 큐브 위에 남음, in-process SM 검증).
            # 통과한 config 중 d(접점-큐브 거리) 최소 → xy 정합 확보(tilt penalty 가 위치 압도하던 문제 해결).
            if tilt_deg < tilt_min:
                continue
            # grasp axis(두 손가락 벌어짐 방향)가 수평이라야 큐브 옆면을 잡는다. 수직이면 큐브 위아래를
            # 눌러 못 잡음(중점만 맞추면 axis 방향이 random — grasp 분산의 핵심). grasp 단계만 적용.
            if stage == "grasp" and self._grasp_axis_vert(rs) > float(self.p.get("grasp_axis_vert_max", 0.4)):
                continue
            score = d
            if score < best_score:
                if self._colliding(rs):  # self-colliding goal 은 planner 가 못 씀 → 배제(후보일 때만 체크)
                    continue
                best_score = score
                best_d = d
                best_q = np.array(rs.get_joint_group_positions(PLANNING_GROUP))
                best_ee = ee
                best_pt = pt
        if best_q is None:
            return None
        if grasp_stage:
            self.logger.info(
                f"FK[{stage}] d={best_d:.4f} grasp_pt=({best_pt[0]:.3f},{best_pt[1]:.3f},{best_pt[2]:.3f}) "
                f"ee_z={best_ee.position.z:.3f} target=({x:.3f},{y:.3f},{z:.3f}) tilt={self._tool_tilt(best_ee.orientation):.0f}°"
            )
        # 정밀화: grasp 단계는 grasp_pt→target shift 로 EE target 보정, transport 는 EE 직접.
        rs.set_joint_group_positions(PLANNING_GROUP, best_q)
        if grasp_stage:
            rs.set_joint_group_positions(GRIPPER_GROUP, [GRIPPER_CLOSED])  # 중점 정합(CLOSED 기준)
        rs.update()
        refine = Pose()
        if grasp_stage:
            shift = target - best_pt  # grasp_pt 를 target 으로 옮기는 EE 이동(국소 근사)
            refine.position.x = best_ee.position.x + float(shift[0])
            refine.position.y = best_ee.position.y + float(shift[1])
            refine.position.z = best_ee.position.z + float(shift[2])
        else:
            refine.position.x, refine.position.y, refine.position.z = x, y, z
        refine.orientation = best_ee.orientation
        if rs.set_from_ik(PLANNING_GROUP, refine, EE_FRAME, float(self.p.get("ik_timeout", 0.2))):
            rq = np.array(rs.get_joint_group_positions(PLANNING_GROUP))
            rs.update()
            if grasp_stage:
                new_d = float(np.linalg.norm(self._grasp_point(rs) - target))
            else:
                rp = rs.get_pose(EE_FRAME)
                new_d = math.dist((rp.position.x, rp.position.y, rp.position.z), (x, y, z))
            # set_from_ik 는 collision 무시 → 정밀화 config 가 colliding 이면 coarse(free) 로 폴백.
            if new_d <= best_d + 1e-3 and not self._colliding(rs):  # 악화 안 됐고 free 면 정밀화 채택
                return rq
        return best_q  # 정밀화 실패/악화/collision 시 coarse config(루프에서 free 보장)

    def _move_to(
        self, x: float, y: float, z: float, *, tilt_candidates: list[float], planner_order=None,
        stage: str = "transport",
    ) -> bool:
        """JOINT-space goal 계획. FK 샘플링으로 (x,y,z) 도달 config 를 찾아 joint goal 로 주고,
        planner 는 joint→joint collision-free 모션만 푼다(5-DOF 에서 pose-goal 비가능 회피).

        tilt_candidates 는 허용 tool z tilt 상한 결정에 쓴다(grasp 는 더 큰 tilt 허용, transport 는
        가능한 수직). planner_order 로 grasp(OMPL 우선)/transport(cuMotion 우선)를 분리한다.
        stage("grasp"/"approach"/"transport")는 FK 샘플의 정합 기준(grasp 접점 vs TCP)을 정한다.
        """
        if planner_order is None:
            planner_order = self.transport_order
        tilt_max = max(tilt_candidates) + 20.0
        goal_q = self._fk_sample_goal(x, y, z, tilt_max_deg=tilt_max, stage=stage)
        if goal_q is None:
            self.logger.error(f"FK-sample 도달 config 없음 ({x:.3f},{y:.3f},{z:.3f}) tilt≤{tilt_max:.0f}°")
            return False
        goal_rs = RobotState(self.robot.get_robot_model())
        goal_rs.set_joint_group_positions(PLANNING_GROUP, goal_q)
        goal_rs.update()
        for params, label in planner_order:
            self.arm.set_start_state_to_current_state()
            self.arm.set_goal_state(robot_state=goal_rs)
            if self._plan_exec(params):
                self.logger.info(
                    f"{label} OK → ({x:.3f},{y:.3f},{z:.3f}) q={[round(float(v), 3) for v in goal_q]}"
                )
                return True
        self.logger.error(f"joint-goal plan 실패 ({x:.3f},{y:.3f},{z:.3f})")
        return False

    def _set_gripper(self, position: float) -> None:
        if not self.gripper.wait_for_server(timeout_sec=5.0):
            self.logger.error("gripper action server 없음")
            return
        goal = ParallelGripperCommand.Goal()
        js = JointState()
        js.name = ["gripper"]
        js.position = [position]
        goal.command = js
        self.logger.info(f"gripper → {position:.3f}")
        self.gripper.send_goal_async(goal)
        # 느린 SO-101 그리퍼 모터 + 접촉/마찰 정착 대기. close 는 더 길게(큐브 감싸고 압착될 시간).
        dwell = float(self.p["gripper_dwell_s"])
        if position < 0:
            dwell = max(dwell, float(self.p.get("gripper_close_dwell_s", 3.0)))
        time.sleep(dwell)

    # ── 단계 ──────────────────────────────────────────────────────────
    def pick_and_place(self, idx: int, n_placed: int) -> bool:
        if idx >= len(self.store.cubes) or self.store.bowl is None:
            return False
        cx, cy, cz = self.store.cubes[idx]
        approach = [self.p["grasp_tilt_deg"]]            # grasp 자세 후보(단계적 tilt)
        approach += [self.p["grasp_tilt_deg"] - 15.0, self.p["grasp_tilt_deg"] + 15.0, 0.0]
        vert = [0.0, 15.0, 30.0]                          # 운반/배치는 가능한 수직

        # grasp 접근/하강 = OMPL+pick_ik 우선(self.grasp_order). grasp 접점(두 손가락 midpoint) 정합.
        self._set_gripper(GRIPPER_OPEN)
        if not self._move_to(
            cx, cy, cz + self.p["approach_height"], tilt_candidates=approach,
            planner_order=self.grasp_order, stage="approach",
        ):
            return False
        if not self._move_to(
            cx, cy, cz + self.p["grasp_z_offset"], tilt_candidates=approach,
            planner_order=self.grasp_order, stage="grasp",
        ):
            return False
        pre_z = self.store.cubes[idx][2]
        self._set_gripper(GRIPPER_CLOSED)
        self._diag_grasp(cx, cy, cz)  # 그리퍼 실제 닫힘·jaw/cube 상대 위치 진단

        # 들림은 큐브를 잡은 채 TCP 수직 상승(grasp 접점 보정 불필요) → stage="transport".
        lift_z = cz + self.p["lift_height"]
        self._move_to(cx, cy, lift_z, tilt_candidates=approach, planner_order=self.grasp_order, stage="transport")
        if self.store.cubes[idx][2] - pre_z < self.p["grasped_dz"]:
            self.logger.warning(f"cube{idx}: grasp 실패(안 들림) — 건너뜀")
            self._set_gripper(GRIPPER_OPEN)
            return False

        # 운반/배치 = cuMotion 우선(self.transport_order, collision-free).
        bx, by, bz = self.store.bowl
        if not self._move_to(
            bx, by, bz + self.p["transport_height"], tilt_candidates=vert, planner_order=self.transport_order
        ):
            return False
        place_z = bz + self.p["place_height"] + n_placed * self.p["stack_increment"]
        self._move_to(bx, by, place_z, tilt_candidates=vert, planner_order=self.transport_order)
        self._set_gripper(GRIPPER_OPEN)
        self._move_to(bx, by, bz + self.p["transport_height"], tilt_candidates=vert, planner_order=self.transport_order)
        return True

    def _placed(self, idx: int) -> bool:
        if idx >= len(self.store.cubes) or self.store.bowl is None:
            return False
        cx, cy, cz = self.store.cubes[idx]
        bx, by, bz = self.store.bowl
        in_xy = math.hypot(cx - bx, cy - by) < self.p["bowl_success_radius"]
        dz = cz - bz
        in_z = self.p["bowl_z_lo"] <= dz <= self.p["bowl_z_hi"]
        return in_xy and in_z

    def run(self, num_cubes: int) -> None:
        # 근접순(로봇 base = base_link 원점)으로 정렬.
        order = sorted(range(min(num_cubes, len(self.store.cubes))),
                       key=lambda i: math.hypot(self.store.cubes[i][0], self.store.cubes[i][1]))
        self.logger.info(f"pick order(근접순): {order}")
        for n, idx in enumerate(order):
            self.logger.info(f"pick-and-place cube[{idx}] (placed so far={n})")
            self.pick_and_place(idx, n)
        n_ok = sum(self._placed(i) for i in order)
        self.logger.info(f"RESULT: {n_ok}/{len(order)} cubes in bowl.")


def main() -> None:
    rclpy.init()
    robot = MoveItPy(
        node_name="pick_place_moveit",
        # MoveItCpp 가 joint_states 토픽을 const 로 박아둠 → namespaced 토픽으로 remap.
        remappings={"joint_states": "/follower/joint_states"},
    )
    store = ObjectPoseStore()

    # 파라미터(launch 의 pick_place_params.yaml, top-key `/**`). store 노드에서 declare/get.
    defaults = {
        "num_cubes": 4, "approach_height": 0.06, "grasp_z_offset": 0.02,
        # grasp 강tilt(60°): jaw 가 큐브 옆/아래로 내려가 감싸야 grip 성립(in-process SM 검증).
        "grasp_tilt_deg": 60.0, "lift_height": 0.07, "transport_height": 0.12,
        "place_height": 0.08, "stack_increment": 0.022, "grasped_dz": 0.025,
        "gripper_dwell_s": 1.5, "gripper_close_dwell_s": 3.0, "bowl_success_radius": 0.06,
        "bowl_z_lo": 0.005, "bowl_z_hi": 0.22,
        # JOINT-goal FK 샘플링(5-DOF): pose/position goal 비가능 → FK 로 도달 config 직접 탐색.
        # grasp 단계는 TCP 대신 grasp 접점(두 손가락 midpoint)을 큐브에 맞추고, tilt 를 [grasp_tilt_min,
        # max] 로 hard-filter 한 뒤 접점-큐브 거리(d) 최소를 골라 강tilt + xy 정합을 동시 확보.
        "fk_samples": 15000, "fk_pos_gate": 0.03, "fk_pos_gate_transport": 0.04, "ik_timeout": 0.5,
        "grasp_tilt_min": 45.0, "grasp_axis_vert_max": 0.4,
    }
    params = {k: store.declare_parameter(k, v).value for k, v in defaults.items()}
    rclpy.logging.get_logger("pick_place_sm").info(
        f"loaded params: grasp_tilt={params['grasp_tilt_deg']} approach_h={params['approach_height']} "
        f"fk_samples={params['fk_samples']} fk_pos_gate={params['fk_pos_gate']}"
    )

    # 구독 spin 은 별도 스레드(MoveItPy 호출은 메인에서 블로킹).
    executor = MultiThreadedExecutor()
    executor.add_node(store)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    logger = rclpy.logging.get_logger("pick_place_sm")
    logger.info("물체 포즈 수신 대기…")
    t0 = time.time()
    while store.bowl is None or not store.cubes:
        if time.time() - t0 > 30.0:
            logger.error("/cube_poses · /bowl_pose 미수신 — bridge 실행 확인")
            rclpy.shutdown()
            return
        time.sleep(0.2)

    sm = PickPlaceSM(robot, store, params)
    sm.run(int(params["num_cubes"]))

    rclpy.shutdown()


if __name__ == "__main__":
    main()
