"""so101_vla_policy launch — VLA 추론 노드.

sim: vla_policy_node 가 /isaac_joint_commands 로 직접 publish → bridge(IsaacArticulationController)가 적용.

예:
  ros2 launch so101_vla_policy vla_policy.launch.py
  ros2 launch so101_vla_policy vla_policy.launch.py joint_states_topic:=/isaac_joint_states
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("so101_vla_policy")
    default_params = os.path.join(pkg_share, "config", "vla_policy.yaml")

    params_file = LaunchConfiguration("params_file")
    joint_states_topic = LaunchConfiguration("joint_states_topic")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("joint_states_topic", default_value="/isaac_joint_states"),
        Node(
            package="so101_vla_policy",
            executable="vla_policy_node",
            name="vla_policy_node",
            output="screen",
            parameters=[params_file, {"joint_states_topic": joint_states_topic}],
        ),
    ])
