"""so101_pick_place_orchestrator.py 를 MoveItPy 설정과 함께 실행.

MoveItPy 는 노드 파라미터에서 moveit 설정(robot_description, SRDF, kinematics,
planning_pipelines, plan_request_params 명명 세트)을 읽는다. move_group 과 별개의
MoveItCpp 인스턴스를 띄우므로 동일 설정 + moveit_py_config.yaml 을 주입해야 한다.

선행: isaac_pick_place.launch.py (ros2_control + move_group + cumotion_action_server) 기동.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _launch_setup(context, *args, **kwargs):
    use_cumotion = LaunchConfiguration("use_cumotion").perform(context).lower() in ("true", "1")
    mock_poses = LaunchConfiguration("mock_poses").perform(context).lower() in ("true", "1")

    desc_share = get_package_share_directory("so101_description")
    moveit_share = get_package_share_directory("so101_moveit_config")
    xacro_path = os.path.join(desc_share, "urdf", "so101_arm.urdf.xacro")

    pipelines = ["ompl", "pilz_industrial_motion_planner"]
    if use_cumotion:
        pipelines.append("isaac_ros_cumotion")

    moveit_config = (
        MoveItConfigsBuilder("so101_arm", package_name="so101_moveit_config")
        .robot_description(
            file_path=xacro_path,
            mappings={"variant": "follower", "use_ros2_control": "false"},
        )
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines(pipelines=pipelines)
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .joint_limits()
        # moveit_py 가 컨트롤러로 trajectory 를 실행하려면 필요 (mock/Isaac 공통).
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # moveit_py_config.yaml 은 ROS params 파일 형식(node/ros__parameters)이 아니므로
    # --params-file 로 넘기면 파싱 실패한다. dict 로 읽어 node param 으로 병합.
    moveit_py_yaml = os.path.join(moveit_share, "config", "moveit_py_config.yaml")
    with open(moveit_py_yaml) as f:
        moveit_py_params = yaml.safe_load(f)

    orchestrator = Node(
        package="so101_moveit_config",
        executable="so101_pick_place_orchestrator.py",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_py_params,
            {"use_sim_time": not mock_poses},
        ],
        # 플래그는 env var 로 전달 (moveit_py 노드 param 노드명 매칭 회피).
        additional_env={
            "SO101_USE_CUMOTION": "1" if use_cumotion else "0",
            "SO101_MOCK_POSES": "1" if mock_poses else "0",
        },
    )
    return [orchestrator]


def generate_launch_description():
    return LaunchDescription(
        [
            # 기본 false = OMPL/Pilz (WSL2). Linux 서버에서 isaac_ros_cumotion 설치 후 true.
            DeclareLaunchArgument("use_cumotion", default_value="false"),
            # mock_poses=true → Isaac Sim 없이 OMPL+RViz+mock (kinematic 데모). use_sim_time 도 끔.
            DeclareLaunchArgument("mock_poses", default_value="false"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
