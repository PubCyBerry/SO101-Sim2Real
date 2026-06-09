"""move_group + cuMotion planner (PATH E, cuMotion+ROS).

so101_moveit_config 의 SRDF/kinematics/joint_limits 를 그대로 쓰고 planning pipeline 에
cuMotion 을 추가한다(OMPL 은 fallback 으로 병행). 실제 GPU 계획은 isaac_ros_cumotion 의
cuMotion action server(상류 isaac_ros_cumotion.launch.py)가 XRDF/URDF 를 로드해 수행하고,
MoveIt plugin(isaac_ros_cumotion_moveit/CumotionPlanner)이 move_group ↔ action server 를 잇는다.

인자:
  xrdf_path : SO-101 XRDF (assets/robots/so101.xrdf, 절대경로)
  urdf_path : cuMotion action server 용 flat URDF (assets/robots/urdf/so_arm101.urdf)
  use_sim_time : Isaac Sim /clock 사용 시 true

nvblox(ESDF) 미사용 — read_esdf_world=false. 충돌은 MoveIt planning scene 에서 받는다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    xrdf_path = LaunchConfiguration("xrdf_path")
    urdf_path = LaunchConfiguration("urdf_path")

    desc_share = get_package_share_directory("so101_description")
    xacro_path = os.path.join(desc_share, "urdf", "so101_arm.urdf.xacro")

    # SRDF/kinematics/joint_limits 는 so101_moveit_config 재사용.
    moveit_config = (
        MoveItConfigsBuilder("so101_arm", package_name="so101_moveit_config")
        .robot_description(file_path=xacro_path, mappings={"variant": "follower", "use_ros2_control": "false"})
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines(
            pipelines=["isaac_ros_cumotion", "ompl"],
            default_planning_pipeline="isaac_ros_cumotion",
        )
        .joint_limits()
        .trajectory_execution(file_path="config/moveit_controllers.yaml", moveit_manage_controllers=False)
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
        remappings=[("joint_states", ["/", namespace, "/joint_states"])],
    )

    # cuMotion action server (상류 launch). XRDF/URDF 주입, nvblox(ESDF) 미사용.
    cumotion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("isaac_ros_cumotion"), "launch", "isaac_ros_cumotion.launch.py")
        ),
        launch_arguments={
            "cumotion_action_server.urdf_file_path": urdf_path,
            "cumotion_action_server.xrdf_file_path": xrdf_path,
            "cumotion_action_server.read_esdf_world": "false",
            "cumotion_action_server.update_esdf_on_request": "false",
            "cumotion_action_server.joint_states_topic": ["/", namespace, "/joint_states"],
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="follower"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("xrdf_path", default_value="", description="assets/robots/so101.xrdf 절대경로"),
            DeclareLaunchArgument("urdf_path", default_value="", description="assets/robots/urdf/so_arm101.urdf 절대경로"),
            move_group_node,
            cumotion,
        ]
    )
