"""action sink shim — /isaac_joint_commands(JointState) → 실기기 controller 액션.

sim 은 bridge 의 IsaacArticulationController 가 /isaac_joint_commands(JointState) 를
직접 물리 적용하므로 이 shim 이 필요 없다. **실기기**는 5축 arm 이 FollowJointTrajectory
액션, gripper 가 GripperCommand 액션으로 구동되므로, VLA 노드가 내는 동일한 6관절 target
JointState 를 받아 해당 액션 goal 로 변환해 재사용할 수 있게 한다.

이 노드 덕분에 VLA 추론 노드(`vla_policy_node`)는 sim/real 양쪽에서 코드 변경 없이
같은 `/isaac_joint_commands` 토픽으로 publish 하면 된다(launch arg 로 shim on/off).

⚠ 실기기 controller 이름/타입은 so101_bringup 설정에 맞춰 param 으로 지정해야 한다.
   기본값은 PATH D/E 관례(`/follower/arm_trajectory_controller`, gripper GripperCommand).
   gripper_action 이 비어 있으면 gripper 변환을 건너뛴다.
"""

from __future__ import annotations

import math

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_GRIPPER_JOINT = "gripper"


class JointCommandToTrajectory(Node):
    def __init__(self) -> None:
        super().__init__("joint_command_to_trajectory")
        p = self.declare_parameter
        self.cmd_topic = p("joint_commands_topic", "/isaac_joint_commands").value
        self.arm_action_name = p(
            "arm_action", "/follower/arm_trajectory_controller/follow_joint_trajectory"
        ).value
        self.gripper_action_name = p("gripper_action", "/follower/gripper_controller/gripper_cmd").value
        self.time_from_start = float(p("time_from_start", 0.10).value)
        self.min_period = 1.0 / float(p("max_rate", 20.0).value)
        self.gripper_max_effort = float(p("gripper_max_effort", 10.0).value)

        self._arm_client = ActionClient(self, FollowJointTrajectory, self.arm_action_name)
        self._gripper_client = (
            ActionClient(self, GripperCommand, self.gripper_action_name)
            if self.gripper_action_name else None
        )
        self._last_sent = 0.0
        self.create_subscription(JointState, self.cmd_topic, self._cb, 10)
        self.get_logger().info(
            f"shim up: {self.cmd_topic} → arm={self.arm_action_name} "
            f"gripper={self.gripper_action_name or '(skip)'}"
        )

    def _cb(self, msg: JointState) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_sent < self.min_period:
            return  # throttle — trajectory controller 에 goal 폭주 방지
        self._last_sent = now

        pos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}

        # arm: 단일 point FollowJointTrajectory.
        if all(j in pos for j in _ARM_JOINTS) and self._arm_client.server_is_ready():
            traj = JointTrajectory()
            traj.joint_names = list(_ARM_JOINTS)
            point = JointTrajectoryPoint()
            point.positions = [float(pos[j]) for j in _ARM_JOINTS]
            sec = int(self.time_from_start)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = int((self.time_from_start - sec) * 1e9)
            traj.points = [point]
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = traj
            self._arm_client.send_goal_async(goal)

        # gripper: GripperCommand(position[rad], max_effort).
        if self._gripper_client and _GRIPPER_JOINT in pos and self._gripper_client.server_is_ready():
            ggoal = GripperCommand.Goal()
            ggoal.command.position = float(pos[_GRIPPER_JOINT])
            ggoal.command.max_effort = self.gripper_max_effort
            self._gripper_client.send_goal_async(ggoal)


def main() -> None:
    rclpy.init()
    node = JointCommandToTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
