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
        self.gripper = ActionClient(store, ParallelGripperCommand, GRIPPER_ACTION)

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

    def _fk_sample_goal(self, x: float, y: float, z: float, *, tilt_max_deg: float):
        """random FK 샘플링으로 (x,y,z) 근처에 down-ish(tool z tilt≤tilt_max) tip 을 두는 manipulator
        joint config 를 찾고 set_from_ik 로 위치를 정밀화한다(5-DOF-aware). 실패 시 None.

        5-DOF 라 pose/position goal 을 planner IK 샘플러가 못 푼다(랜덤 orientation 이 거의 도달
        불가) → 도달 가능 config 자체를 FK 로 찾는다(in-process joint_fk SM 과 동일 원리). set_to_random
        _positions 는 joint bounds 내에서만 샘플 → 한계 자동 준수. 거리 gate 안에서 최근접을 고른다.
        """
        target = np.array([x, y, z])
        rs = self.sample_rs
        n = int(self.p.get("fk_samples", 12000))
        gate = float(self.p.get("fk_pos_gate", 0.05))
        best_q = None
        best_pose = None
        best_d = 1e9
        for _ in range(n):
            rs.set_to_random_positions()
            rs.update()
            p = rs.get_pose(EE_FRAME)
            d = math.dist((p.position.x, p.position.y, p.position.z), (x, y, z))
            if d > gate:
                continue
            if self._tool_tilt(p.orientation) > tilt_max_deg:
                continue
            if d < best_d:
                best_d = d
                best_q = np.array(rs.get_joint_group_positions(PLANNING_GROUP))
                best_pose = p
        if best_q is None:
            return None
        # 정밀화: best config 의 (도달 가능) orientation + 목표 위치로 set_from_ik(seed=best config).
        rs.set_joint_group_positions(PLANNING_GROUP, best_q)
        rs.update()
        refine = Pose()
        refine.position.x, refine.position.y, refine.position.z = x, y, z
        refine.orientation = best_pose.orientation
        if rs.set_from_ik(PLANNING_GROUP, refine, EE_FRAME, float(self.p.get("ik_timeout", 0.2))):
            rq = np.array(rs.get_joint_group_positions(PLANNING_GROUP))
            rp = rs.get_pose(EE_FRAME)
            if math.dist((rp.position.x, rp.position.y, rp.position.z), (x, y, z)) <= best_d + 1e-3:
                return rq
        return best_q  # 정밀화 실패/악화 시 coarse config

    def _move_to(
        self, x: float, y: float, z: float, *, tilt_candidates: list[float], planner_order=None
    ) -> bool:
        """JOINT-space goal 계획. FK 샘플링으로 (x,y,z) 도달 config 를 찾아 joint goal 로 주고,
        planner 는 joint→joint collision-free 모션만 푼다(5-DOF 에서 pose-goal 비가능 회피).

        tilt_candidates 는 허용 tool z tilt 상한 결정에 쓴다(grasp 는 더 큰 tilt 허용, transport 는
        가능한 수직). planner_order 로 grasp(OMPL 우선)/transport(cuMotion 우선)를 분리한다.
        """
        if planner_order is None:
            planner_order = self.transport_order
        tilt_max = max(tilt_candidates) + 20.0
        goal_q = self._fk_sample_goal(x, y, z, tilt_max_deg=tilt_max)
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

        # grasp 접근/하강/들림 = OMPL+pick_ik 우선(self.grasp_order).
        self._set_gripper(GRIPPER_OPEN)
        if not self._move_to(
            cx, cy, cz + self.p["approach_height"], tilt_candidates=approach, planner_order=self.grasp_order
        ):
            return False
        if not self._move_to(
            cx, cy, cz + self.p["grasp_z_offset"], tilt_candidates=approach, planner_order=self.grasp_order
        ):
            return False
        pre_z = self.store.cubes[idx][2]
        self._set_gripper(GRIPPER_CLOSED)

        lift_z = cz + self.p["lift_height"]
        self._move_to(cx, cy, lift_z, tilt_candidates=approach, planner_order=self.grasp_order)
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
        "num_cubes": 4, "approach_height": 0.06, "grasp_z_offset": -0.005,
        "grasp_tilt_deg": 30.0, "lift_height": 0.07, "transport_height": 0.12,
        "place_height": 0.08, "stack_increment": 0.022, "grasped_dz": 0.025,
        "gripper_dwell_s": 1.5, "bowl_success_radius": 0.06,
        "bowl_z_lo": 0.005, "bowl_z_hi": 0.22,
        # JOINT-goal FK 샘플링(5-DOF): pose/position goal 비가능 → FK 로 도달 config 직접 탐색.
        "fk_samples": 15000, "fk_pos_gate": 0.04, "ik_timeout": 0.2,
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
