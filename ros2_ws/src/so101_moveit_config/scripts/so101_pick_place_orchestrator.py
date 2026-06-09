#!/usr/bin/env python3
"""SO-101 cube_desk Pick & Place 상태기계 (MoveItPy + cuMotion/OMPL/Pilz).

Isaac Sim(Windows)이 cube_desk 장면을 띄우고 ROS 2 브릿지로:
  - /isaac_joint_states 발행 (topic_based_ros2_control 이 상태로 소비)
  - /isaac_joint_commands 구독 (arm_trajectory_controller + gripper_controller 출력)
  - /cube_desk/object_poses (geometry_msgs/PoseArray, frame_id="base_link",
    순서 [Cube1,Cube2,Cube3,Cube4,Bowl]) 발행
하면, 이 노드가 MoveIt2 path planning 으로 큐브를 그릇에 담는다.

planner 분담:
  - free-space (APPROACH/TRANSPORT/RETREAT) : cuMotion (실패 시 OMPL 폴백)
  - cartesian 직선 (DESCEND/LIFT/PLACE)      : Pilz LIN
  - gripper                                   : gripper_controller GripperCommand action

PlanningScene: 미처리 큐브 + 그릇을 collision object 로 등록, 잡은 큐브는 gripper 에 attach.
SO-101 은 5-DOF 라 임의 6-DOF pose 도달 불가 → grasp 자세는 top-down tilt + pick_ik approximate.

선행: WSL2 에서  ros2 launch so101_bringup isaac_pick_place.launch.py use_cumotion:=true
실행 :          ros2 launch so101_moveit_config pick_place_orchestrator.launch.py
"""
from __future__ import annotations

import math
import time
from enum import Enum

import numpy as np
import rclpy
import rclpy.logging
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped
from control_msgs.action import GripperCommand
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from tf_transformations import quaternion_from_euler, quaternion_from_matrix

# ── 프레임/그룹 ────────────────────────────────────────────────────────────────
BASE_FRAME = "base_link"          # SRDF virtual_joint world→base_link, MoveIt 기준
EE_FRAME = "gripper_frame_link"   # manipulator tip (TCP)
PLANNING_GROUP = "manipulator"

# ── 객체 (pick_cube_env_cfg.py / constant.py 와 동기) ────────────────────────────
CUBE_NAMES = ["Cube1", "Cube2", "Cube3", "Cube4"]
BOWL_NAME = "Bowl"
OBJECT_ORDER = CUBE_NAMES + [BOWL_NAME]   # PoseArray 순서 계약

# 큐브 크기 (한 변, m). Cube1/2=30mm, Cube3/4=40mm.
CUBE_SIZE = {"Cube1": 0.030, "Cube2": 0.030, "Cube3": 0.040, "Cube4": 0.040}
BOWL_RADIUS = 0.06                # BOWL_SUCCESS_RADIUS
BOWL_HEIGHT = 0.05                # 충돌용 근사 높이

# ── 동작 높이/오프셋 (m) — pick_cube_state_machine.py 값 재사용 ───────────────────
APPROACH_HEIGHT = 0.06            # 큐브 위 pre-pick (도달영역 z<=0.15 고려)
GRASP_Z_OFFSET = 0.005           # grasp 목표 z = 큐브 중심 + 값
LIFT_HEIGHT = 0.08
TRANSPORT_HEIGHT = 0.08
PLACE_HEIGHT = 0.04
RETREAT_HEIGHT = 0.08

# ── grasp 자세 (gripper_frame_link RPY). so101_moveit_test.py 의 검증된 DOWN 자세
#   RPY=[0, pi, 0](EE 를 아래 -Z 로) 를 기준 — pick_ik 가 이 자세를 도달함(검증됨).
#   5-DOF 라 yaw 는 pick_ik 가 자동 결정. tilt 는 0 에서 시작(도달 우선), 이후 튜닝.
GRASP_RPY = (0.0, math.pi, 0.0)
GRASP_TILT_RAD = math.radians(0.0)

GRIPPER_OPEN = 1.4               # SRDF open=1.5
GRIPPER_CLOSED = -0.1            # SRDF closed=-0.16
GRIPPER_MAX_EFFORT = 5.0

IK_TIMEOUT = 0.3

# Isaac Sim 없이 OMPL+RViz+mock 으로 돌릴 때의 객체 pose (base_link frame).
# cube_desk world→base 정확 변환은 로봇 base_link 의 world 회전(USD 내부 프레임)에 의존해
# 모호하므로, so101_moveit_test.py 의 검증된 도달 envelope(전방 +x≈0.13~0.39, y≈±0.1,
# z≈0.05~0.22) 안에 cube_desk 레이아웃(큐브 4 + 그릇 1)을 근사 배치한 mock 좌표.
# (kinematic 데모용 — 정확한 cube_desk world 좌표는 Linux 서버 Isaac 연동에서 사용.)
# grid probe 실측: SO-101 도달영역은 y≈0 선상 (x 0.20~0.38, z 0.05~0.15).
# y=±0.10 은 도달 불가 → 큐브/그릇을 y=0 에 x 를 달리해 배치(kinematic 데모).
MOCK_POSES_BASE = {
    "Cube1": (0.22, 0.0, 0.05),
    "Cube2": (0.28, 0.0, 0.05),
    "Cube3": (0.33, 0.0, 0.05),
    "Cube4": (0.38, 0.0, 0.05),
    "Bowl": (0.20, 0.0, 0.06),
}


class FSMState(str, Enum):
    HOME = "HOME"
    OPEN = "OPEN"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    DONE = "DONE"


# ═════════════════════════════════════════════════════════════════════════════
#  객체 pose 취득 (Isaac OmniGraph 가 base_link→객체 TF 발행 → tf2 조회)
#  Isaac 의 ROS2PublishTransformTree: parent=base_link, target=Cube1..4,Bowl prim.
# ═════════════════════════════════════════════════════════════════════════════

class ObjectFrames(Node):
    def __init__(self, mock: bool = False):
        super().__init__("so101_object_frames")
        self._mock = mock
        if not mock:
            self._buf = Buffer()
            self._listener = TransformListener(self._buf, self)

    def get(self, name: str) -> Pose | None:
        if self._mock:
            xyz = MOCK_POSES_BASE.get(name)
            if xyz is None:
                return None
            p = Pose()
            p.position.x, p.position.y, p.position.z = xyz
            p.orientation.w = 1.0
            return p
        try:
            t = self._buf.lookup_transform(BASE_FRAME, name, rclpy.time.Time())
        except Exception:
            return None
        p = Pose()
        p.position.x = t.transform.translation.x
        p.position.y = t.transform.translation.y
        p.position.z = t.transform.translation.z
        p.orientation = t.transform.rotation
        return p

    def ready(self) -> bool:
        return all(self.get(n) is not None for n in OBJECT_ORDER)


# ═════════════════════════════════════════════════════════════════════════════
#  Gripper action client
# ═════════════════════════════════════════════════════════════════════════════

class Gripper:
    def __init__(self, node: Node):
        self._node = node
        self._client = ActionClient(
            node, GripperCommand, "/follower/gripper_controller/gripper_cmd"
        )

    @staticmethod
    def _wait(future, timeout=10.0):
        # 백그라운드 executor 가 node 를 스핀하므로 future 를 폴링만 한다.
        t0 = time.time()
        while not future.done() and time.time() - t0 < timeout:
            time.sleep(0.01)
        return future.result() if future.done() else None

    def command(self, position: float, logger) -> bool:
        if not self._client.wait_for_server(timeout_sec=5.0):
            logger.error("gripper action server 없음")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = GRIPPER_MAX_EFFORT
        handle = self._wait(self._client.send_goal_async(goal))
        if handle is None or not handle.accepted:
            logger.error("gripper goal 거부")
            return False
        self._wait(handle.get_result_async())
        time.sleep(0.3)  # 물리 정착
        return True


# ═════════════════════════════════════════════════════════════════════════════
#  PlanningScene helpers (collision objects + attach)
#  주의: moveit_py PlanningScene 메서드명은 설치 버전에 따라 다를 수 있다(M3 검증).
# ═════════════════════════════════════════════════════════════════════════════

def _make_box(name: str, pose: Pose, size: float, frame: str = BASE_FRAME) -> CollisionObject:
    co = CollisionObject()
    co.header.frame_id = frame
    co.id = name
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [size, size, size]
    co.primitives = [prim]
    co.primitive_poses = [pose]
    co.operation = CollisionObject.ADD
    return co


def _make_bowl(name: str, pose: Pose, frame: str = BASE_FRAME) -> CollisionObject:
    co = CollisionObject()
    co.header.frame_id = frame
    co.id = name
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.CYLINDER
    prim.dimensions = [BOWL_HEIGHT, BOWL_RADIUS]  # [height, radius]
    co.primitives = [prim]
    co.primitive_poses = [pose]
    co.operation = CollisionObject.ADD
    return co


def refresh_scene(robot, listener: ObjectPoseListener, skip: set[str], logger) -> None:
    """미처리 큐브 + 그릇을 collision object 로 갱신. skip 에 든 객체(처리완료/대상)는 제외."""
    psm = robot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        for name in CUBE_NAMES:
            pose = listener.get(name)
            if pose is None:
                continue
            if name in skip:
                # 제거 (REMOVE)
                co = CollisionObject()
                co.header.frame_id = BASE_FRAME
                co.id = name
                co.operation = CollisionObject.REMOVE
                scene.apply_collision_object(co)
            else:
                scene.apply_collision_object(_make_box(name, pose, CUBE_SIZE[name]))
        bowl = listener.get(BOWL_NAME)
        if bowl is not None:
            scene.apply_collision_object(_make_bowl(BOWL_NAME, bowl))
        scene.current_state.update()


def attach_cube(robot, name: str, pose: Pose) -> None:
    psm = robot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        aco = AttachedCollisionObject()
        aco.link_name = EE_FRAME
        aco.object = _make_box(name, pose, CUBE_SIZE[name])
        aco.object.operation = CollisionObject.ADD
        aco.touch_links = ["gripper_link", "moving_jaw_so101_v1_link", "gripper_frame_link"]
        scene.process_attached_collision_object(aco)
        scene.current_state.update()


def detach_cube(robot, name: str) -> None:
    psm = robot.get_planning_scene_monitor()
    with psm.read_write() as scene:
        # 1) gripper 에서 detach (MoveIt 은 detach 시 world 로 되돌림)
        aco = AttachedCollisionObject()
        aco.link_name = EE_FRAME
        aco.object.id = name
        aco.object.operation = CollisionObject.REMOVE
        scene.process_attached_collision_object(aco)
        # 2) world 에서도 제거 — 안 그러면 놓인 큐브가 다음 큐브 start-state 충돌을 일으킴.
        co = CollisionObject()
        co.header.frame_id = BASE_FRAME
        co.id = name
        co.operation = CollisionObject.REMOVE
        scene.apply_collision_object(co)
        scene.current_state.update()


# ═════════════════════════════════════════════════════════════════════════════
#  Planning (so101_moveit_test.py 패턴 재사용)
# ═════════════════════════════════════════════════════════════════════════════

def _grasp_pose(obj_pose: Pose, dz: float) -> PoseStamped:
    """큐브 pose + z 오프셋 → top-down tilt grasp 의 gripper_frame_link 목표."""
    ps = PoseStamped()
    ps.header.frame_id = BASE_FRAME
    ps.pose.position.x = obj_pose.position.x
    ps.pose.position.y = obj_pose.position.y
    ps.pose.position.z = obj_pose.position.z + dz
    q = quaternion_from_euler(GRASP_RPY[0], GRASP_RPY[1] + GRASP_TILT_RAD, GRASP_RPY[2])
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
    return ps


def _level_pose(x: float, y: float, z: float) -> PoseStamped:
    """straight-down(tilt 0) — 운반/배치용 (grasp 와 동일 자세)."""
    ps = PoseStamped()
    ps.header.frame_id = BASE_FRAME
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
    q = quaternion_from_euler(GRASP_RPY[0], GRASP_RPY[1], GRASP_RPY[2])
    ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
    return ps


def plan_and_execute(robot, arm, logger, params=None) -> bool:
    result = arm.plan(multi_plan_parameters=params) if params else arm.plan()
    if result:
        robot.execute(result.trajectory, controllers=[])
        return True
    logger.error("Planning 실패")
    return False


# SO-101 5-DOF: 임의 6-DOF 자세 도달 불가. FK 측정상 전방-저위치의 도달 자세는
# Y축 pitch ~133°(rest EE quat). 목표 위치는 고정하고 도달 가능한 자세를 sweep.
_PITCH_SWEEP = (2.32, 2.5, 2.1, 2.7, 1.9, 2.9, math.pi)
_YAW_SWEEP = (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2)


def _relaxed_ik(robot, logger, pos) -> RobotState | None:
    """위치 고정 + 자세 sweep 으로 5-DOF 도달 가능한 IK 해 탐색. 첫 성공 반환."""
    model = robot.get_robot_model()
    with robot.get_planning_scene_monitor().read_only() as scene:
        seed = scene.current_state.get_joint_group_positions(PLANNING_GROUP)
    tp = Pose()
    tp.position.x, tp.position.y, tp.position.z = pos
    for pitch in _PITCH_SWEEP:
        for yaw in _YAW_SWEEP:
            q = quaternion_from_euler(0.0, pitch, yaw)
            tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = q
            rs = RobotState(model)
            rs.set_joint_group_positions(PLANNING_GROUP, seed)
            rs.update()
            if rs.set_from_ik(PLANNING_GROUP, tp, EE_FRAME, 0.1):
                rs.update()
                return rs
    logger.error(f"IK 실패(자세 sweep 후) — pos={tuple(round(v,3) for v in pos)} 도달 불가")
    return None


def plan_to_pose(robot, arm, logger, pose: PoseStamped, params, *, use_ik: bool) -> bool:
    """pose goal 로 이동. use_ik=True 면 위치 고정+자세 sweep IK→joint goal(free-space),
    False 면 pose goal 직접(Pilz LIN cartesian)."""
    arm.set_start_state_to_current_state()
    if use_ik:
        pos = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
        rs = _relaxed_ik(robot, logger, pos)
        if rs is None:
            return False
        arm.set_goal_state(robot_state=rs)
    else:
        arm.set_goal_state(pose_stamped_msg=pose, pose_link=EE_FRAME)
    return plan_and_execute(robot, arm, logger, params)


# ═════════════════════════════════════════════════════════════════════════════
#  FSM
# ═════════════════════════════════════════════════════════════════════════════

def run_pick_place(robot, arm, listener, gripper, logger,
                   free_params, lin_params, mock: bool = False) -> dict:
    results: dict[str, bool] = {}
    placed = 0
    # mock: LIN(정확 직선+자세) 대신 relaxed free-space IK. real: Pilz LIN cartesian.
    cart_params = free_params if mock else lin_params
    cart_ik = mock

    # 처리 순서: 로봇 base 에서 가까운 큐브부터(near_robot)
    def dist(name):
        p = listener.get(name)
        return math.hypot(p.position.x, p.position.y) if p else 9e9
    order = sorted(CUBE_NAMES, key=dist)

    bowl = listener.get(BOWL_NAME)
    bx, by = bowl.position.x, bowl.position.y
    bz = bowl.position.z

    # HOME (named state)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="rest")
    plan_and_execute(robot, arm, logger)
    gripper.command(GRIPPER_OPEN, logger)

    for cube in order:
        logger.info(f"═══ {cube} ═══")
        if not mock:
            # 정식 충돌 회피(Isaac/실물): 그릇+미처리 큐브를 obstacle 로. 대상 큐브는 제외.
            # mock 은 좁은 도달영역에서 obstacle 가 start-state 충돌을 일으켜 생략(kinematic 데모).
            skip = {cube} | {n for n in CUBE_NAMES if results.get(n)}
            refresh_scene(robot, listener, skip, logger)

        cpose = listener.get(cube)
        ok = True

        # APPROACH (free-space)
        ok = plan_to_pose(robot, arm, logger, _grasp_pose(cpose, APPROACH_HEIGHT),
                          free_params, use_ik=True)
        if not ok:
            results[cube] = False
            continue
        # DESCEND (cartesian LIN, 실패 시 IK 폴백)
        gripper.command(GRIPPER_OPEN, logger)
        if not plan_to_pose(robot, arm, logger, _grasp_pose(cpose, GRASP_Z_OFFSET),
                            cart_params, use_ik=cart_ik):
            logger.warning("DESCEND LIN 실패 → IK 폴백")
            ok = plan_to_pose(robot, arm, logger, _grasp_pose(cpose, GRASP_Z_OFFSET),
                              free_params, use_ik=True) and ok
        # CLOSE + attach (RViz 에서 큐브가 그리퍼에 붙어 함께 이동)
        gripper.command(GRIPPER_CLOSED, logger)
        attach_cube(robot, cube, listener.get(cube))
        # LIFT (LIN)
        plan_to_pose(robot, arm, logger, _grasp_pose(cpose, LIFT_HEIGHT), cart_params, use_ik=cart_ik)
        # TRANSPORT (free-space) → 그릇 위
        ok = plan_to_pose(robot, arm, logger, _level_pose(bx, by, bz + TRANSPORT_HEIGHT),
                          free_params, use_ik=True) and ok
        # PLACE (LIN)
        ok = plan_to_pose(robot, arm, logger, _level_pose(bx, by, bz + PLACE_HEIGHT),
                          cart_params, use_ik=cart_ik) and ok
        # RELEASE + detach
        gripper.command(GRIPPER_OPEN, logger)
        detach_cube(robot, cube)
        # RETREAT (LIN 위로)
        plan_to_pose(robot, arm, logger, _level_pose(bx, by, bz + RETREAT_HEIGHT), cart_params, use_ik=cart_ik)

        if mock:
            # mock(물리 없음): 모든 핵심 phase 가 계획·실행됐는지로 판정(kinematic 데모).
            results[cube] = bool(ok)
        else:
            # 물리 모드: 큐브 xy 가 그릇 반경 안 + 높이.
            time.sleep(0.5)
            cnow = listener.get(cube)
            in_xy = math.hypot(cnow.position.x - bx, cnow.position.y - by) <= BOWL_RADIUS
            in_z = 0.0 <= (cnow.position.z - bz) <= 0.15
            results[cube] = bool(in_xy and in_z and ok)
        if results[cube]:
            placed += 1
        logger.info(f"  {cube}: {'planned' if mock else 'placed'}={results[cube]}")

    return {"placed": placed, "total": len(CUBE_NAMES), "results": results}


def main() -> None:
    rclpy.init()
    logger = rclpy.logging.get_logger("so101_pick_place")

    import os
    import threading
    from rclpy.executors import MultiThreadedExecutor

    # 플래그는 env var 로 (moveit_py 노드 param 은 노드명 매칭이 까다로움).
    use_cumotion = os.environ.get("SO101_USE_CUMOTION", "0") == "1"
    mock = os.environ.get("SO101_MOCK_POSES", "0") == "1"

    listener = ObjectFrames(mock=mock)
    # 백그라운드 executor 로 TF 구독 + gripper action 을 계속 스핀.
    executor = MultiThreadedExecutor()
    executor.add_node(listener)
    threading.Thread(target=executor.spin, daemon=True).start()

    if mock:
        logger.info("mock_poses 모드 — MOCK_POSES_BASE 사용 (Isaac Sim 불필요, kinematic 데모).")
    else:
        logger.info("객체 TF 대기 (base_link→Cube/Bowl) …")
        t0 = time.time()
        while rclpy.ok() and not listener.ready():
            time.sleep(0.2)
            if time.time() - t0 > 30.0:
                logger.error("객체 TF 미수신 — Isaac Sim TF 브릿지(ROS2PublishTransformTree) 확인. 종료.")
                rclpy.shutdown()
                return
        logger.info("객체 TF 수신 완료.")

    robot = MoveItPy(
        node_name="so101_orchestrator",
        remappings={"joint_states": "/follower/joint_states"},
    )
    arm = robot.get_planning_component(PLANNING_GROUP)
    gripper = Gripper(listener)

    # ── FK 진단: named config 의 gripper_frame_link 위치(base frame). 도달 영역 파악용. ──
    try:
        rs_fk = RobotState(robot.get_robot_model())
        for nm, jv in (("rest", [0.0, -1.57, 1.57, 0.75, 0.0]),
                       ("extended", [0.0, 1.57, -1.57, 0.0, 0.0]),
                       ("zero", [0.0, 0.0, 0.0, 0.0, 0.0])):
            rs_fk.set_joint_group_positions(PLANNING_GROUP, jv)
            rs_fk.update()
            tf = rs_fk.get_global_link_transform(EE_FRAME)
            logger.info(f"FK[{nm}] {EE_FRAME} xyz = "
                        f"({tf[0, 3]:.3f}, {tf[1, 3]:.3f}, {tf[2, 3]:.3f})")
            if nm == "rest":
                # round-trip: rest EE 의 실제 pose 로 IK 가 풀리는지(메커니즘 확증).
                q = quaternion_from_matrix(tf)
                tp = Pose()
                tp.position.x, tp.position.y, tp.position.z = tf[0, 3], tf[1, 3], tf[2, 3]
                tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w = q
                rs_rt = RobotState(robot.get_robot_model())
                rs_rt.set_joint_group_positions(PLANNING_GROUP, [0.0, 0.0, 0.0, 0.0, 0.0])
                rs_rt.update()
                ok_rt = rs_rt.set_from_ik(PLANNING_GROUP, tp, EE_FRAME, 0.5)
                logger.info(f"IK round-trip(rest pose) = {ok_rt}  quat={[round(v,3) for v in q]}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"FK 진단 스킵: {e}")

    # 도달 영역 grid probe (SO101_PROBE_GRID=1) — 도달 가능한 base-frame 셀을 찾아 로그.
    if os.environ.get("SO101_PROBE_GRID") == "1":
        model = robot.get_robot_model()
        reach = []
        for gx in (0.20, 0.26, 0.32, 0.38):
            for gy in (-0.10, 0.0, 0.10):
                for gz in (0.05, 0.10, 0.15):
                    found = False
                    for pitch in (2.0, 2.32, 2.6, 2.9):
                        for yaw in (0.0, 0.5, -0.5, 1.0, -1.0):
                            qq = quaternion_from_euler(0.0, pitch, yaw)
                            tpp = Pose()
                            tpp.position.x, tpp.position.y, tpp.position.z = gx, gy, gz
                            (tpp.orientation.x, tpp.orientation.y,
                             tpp.orientation.z, tpp.orientation.w) = qq
                            rsp = RobotState(model)
                            rsp.set_joint_group_positions(PLANNING_GROUP, [0.0, 0.0, 0.0, 0.0, 0.0])
                            rsp.update()
                            if rsp.set_from_ik(PLANNING_GROUP, tpp, EE_FRAME, 0.05):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        reach.append((gx, gy, gz))
        logger.info(f"REACHABLE_CELLS({len(reach)}): {reach}")
        rclpy.shutdown()
        return

    # planner 파라미터 (이름은 moveit_py_config.yaml 정의).
    # use_cumotion=True(Linux 서버, isaac_ros_cumotion 설치 시) → cuMotion primary + OMPL 폴백.
    # 기본(WSL2) → OMPL 단독 (cuMotion 미설치).
    free_sets = ["cumotion", "ompl_rrtc"] if use_cumotion else ["ompl_rrtc"]
    free_params = MultiPipelinePlanRequestParameters(robot, free_sets)
    lin_params = MultiPipelinePlanRequestParameters(robot, ["pilz_lin"])
    logger.info(f"free-space planner sets: {free_sets}")

    out = run_pick_place(robot, arm, listener, gripper, logger, free_params, lin_params, mock=mock)
    verb = "planned" if mock else "placed"
    logger.info(f"완료: {out['placed']}/{out['total']} {verb} — {out['results']}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
