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

import rclpy
import rclpy.logging
from control_msgs.action import ParallelGripperCommand
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, MultiPipelinePlanRequestParameters
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_from_euler

EE_FRAME = "gripper_frame_link"
PLANNING_GROUP = "manipulator"
BASE_FRAME = "base_link"  # bridge 가 base_link 기준 TF 로 물체 포즈 publish
GRIPPER_ACTION = "/follower/gripper_controller/gripper_cmd"
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
        # cuMotion 우선, 실패 시 OMPL fallback.
        self.cumotion_params = MultiPipelinePlanRequestParameters(robot, ["cumotion"])
        self.ompl_params = MultiPipelinePlanRequestParameters(robot, ["ompl_rrtc"])
        self.gripper = ActionClient(store, ParallelGripperCommand, GRIPPER_ACTION)

    # ── 저수준 ────────────────────────────────────────────────────────
    def _pose(self, x: float, y: float, z: float, yaw: float, tilt_deg: float) -> PoseStamped:
        """tool 을 아래로 향하고 yaw 정렬 + 앞으로 tilt 한 grasp pose (base_link frame)."""
        # 기준 down = rpy(0, π, 0) (so101_moveit_test 의 DOWN). tilt 는 pitch 를 줄여 앞으로 기울임.
        q = quaternion_from_euler(0.0, math.pi - math.radians(tilt_deg), yaw)
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = (
            q[0], q[1], q[2], q[3],
        )
        return ps

    def _yaw_to(self, x: float, y: float) -> float:
        return math.atan2(y, x)  # base_link 원점 기준(로봇 base)

    def _plan_exec(self, plan_params) -> bool:
        result = self.arm.plan(multi_plan_parameters=plan_params)
        if result:
            self.robot.execute(result.trajectory, controllers=[])
            return True
        return False

    def _ik_solve(self, x: float, y: float, z: float, *, tilt_candidates: list[float]) -> RobotState | None:
        """여러 tilt 후보로 set_from_ik 시도 → 첫 성공 RobotState. 5DOF 미도달 회피."""
        model = self.robot.get_robot_model()
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            cur = scene.current_state.get_joint_group_positions(PLANNING_GROUP)
        yaw = self._yaw_to(x, y)
        for tilt in tilt_candidates:
            rs = RobotState(model)
            rs.set_joint_group_positions(PLANNING_GROUP, cur)
            rs.update()
            pose = self._pose(x, y, z, yaw, tilt)
            if rs.set_from_ik(PLANNING_GROUP, pose.pose, EE_FRAME, float(self.p["ik_timeout"])):
                rs.update()
                return rs
        return None

    def _move_to(self, x: float, y: float, z: float, *, tilt_candidates: list[float]) -> bool:
        rs = self._ik_solve(x, y, z, tilt_candidates=tilt_candidates)
        if rs is None:
            self.logger.error(f"IK 실패 ({x:.3f},{y:.3f},{z:.3f}) tilts={tilt_candidates}")
            return False
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=rs)
        if self._plan_exec(self.cumotion_params):
            return True
        self.logger.warning("cuMotion 계획 실패 → OMPL fallback")
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=rs)
        return self._plan_exec(self.ompl_params)

    def _set_gripper(self, position: float) -> None:
        if not self.gripper.wait_for_server(timeout_sec=5.0):
            self.logger.error("gripper action server 없음")
            return
        goal = ParallelGripperCommand.Goal()
        js = JointState()
        js.name = ["gripper"]
        js.position = [position]
        goal.command = js
        self.gripper.send_goal_async(goal)
        time.sleep(self.p["gripper_dwell_s"])

    # ── 단계 ──────────────────────────────────────────────────────────
    def pick_and_place(self, idx: int, n_placed: int) -> bool:
        if idx >= len(self.store.cubes) or self.store.bowl is None:
            return False
        cx, cy, cz = self.store.cubes[idx]
        approach = [self.p["grasp_tilt_deg"]]            # grasp 자세 후보(단계적 tilt)
        approach += [self.p["grasp_tilt_deg"] - 15.0, self.p["grasp_tilt_deg"] + 15.0, 0.0]
        vert = [0.0, 15.0, 30.0]                          # 운반/배치는 가능한 수직

        self._set_gripper(GRIPPER_OPEN)
        if not self._move_to(cx, cy, cz + self.p["approach_height"], tilt_candidates=approach):
            return False
        if not self._move_to(cx, cy, cz + self.p["grasp_z_offset"], tilt_candidates=approach):
            return False
        pre_z = self.store.cubes[idx][2]
        self._set_gripper(GRIPPER_CLOSED)

        lift_z = cz + self.p["lift_height"]
        self._move_to(cx, cy, lift_z, tilt_candidates=approach)
        if self.store.cubes[idx][2] - pre_z < self.p["grasped_dz"]:
            self.logger.warning(f"cube{idx}: grasp 실패(안 들림) — 건너뜀")
            self._set_gripper(GRIPPER_OPEN)
            return False

        bx, by, bz = self.store.bowl
        if not self._move_to(bx, by, bz + self.p["transport_height"], tilt_candidates=vert):
            return False
        place_z = bz + self.p["place_height"] + n_placed * self.p["stack_increment"]
        self._move_to(bx, by, place_z, tilt_candidates=vert)
        self._set_gripper(GRIPPER_OPEN)
        self._move_to(bx, by, bz + self.p["transport_height"], tilt_candidates=vert)
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

    # 파라미터(launch 의 pick_place_params.yaml). store 노드에서 declare/get.
    defaults = {
        "num_cubes": 4, "approach_height": 0.12, "grasp_z_offset": -0.005,
        "grasp_tilt_deg": 60.0, "lift_height": 0.12, "transport_height": 0.15,
        "place_height": 0.06, "stack_increment": 0.022, "grasped_dz": 0.03,
        "ik_timeout": 0.2, "gripper_dwell_s": 1.5, "bowl_success_radius": 0.06,
        "bowl_z_lo": 0.005, "bowl_z_hi": 0.22,
    }
    params = {k: store.declare_parameter(k, v).value for k, v in defaults.items()}

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
