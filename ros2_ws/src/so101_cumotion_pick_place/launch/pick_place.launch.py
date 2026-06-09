"""SO-101 cuMotion + ROS 2 pick-and-place 오케스트레이션 (PATH E).

기동 순서:
  1) follower_split.launch.py (hardware_type:=isaac, TopicBasedSystem)
       → ros2_control_node + rsp + joint_state_broadcaster/arm_trajectory_controller/gripper_controller
  2) move_group_cumotion.launch.py (cuMotion planner + move_group, use_sim_time:=true)
  3) pick_place_sm 노드 (MoveItPy + cuMotion)
  (+선택 RViz)

Isaac Sim bridge(scripts/sim/run_cube_desk_ros_bridge.py)는 **별도 프로세스**로 먼저 실행한다.
처리할 큐브 수는 bridge 의 --num_cubes 로 정한다(SM 은 수신한 큐브 전부 처리).

프레임: MoveIt virtual_joint(world→base_link)는 identity, bridge 가 base_link 기준 포즈를
publish → 별도 static TF 불필요. RViz fixed frame = base_link.

XRDF/URDF 경로: 환경변수 SO101_REPO(레포 루트) 로 해결(미설정 시 launch 인자로 전달).
    export SO101_REPO=/path/to/SO101-Sim2Real
    ros2 launch so101_cumotion_pick_place pick_place.launch.py use_rviz:=true

MoveItPy 파라미터는 node_name 으로 매칭되므로 SM Node 의 name 을 "pick_place_moveit" 로 둔다.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    use_rviz = LaunchConfiguration("use_rviz")
    xrdf_path = LaunchConfiguration("xrdf_path")
    urdf_path = LaunchConfiguration("urdf_path")

    bringup_share = get_package_share_directory("so101_bringup")
    moveit_share = get_package_share_directory("so101_moveit_config")
    cumotion_share = get_package_share_directory("so101_cumotion_moveit_config")
    sm_share = get_package_share_directory("so101_cumotion_pick_place")
    desc_share = get_package_share_directory("so101_description")

    repo = EnvironmentVariable("SO101_REPO", default_value="")

    # MoveItPy 용 moveit config (cuMotion + OMPL). SRDF/kinematics 는 so101_moveit_config.
    xacro_path = os.path.join(desc_share, "urdf", "so101_arm.urdf.xacro")
    moveit_config = (
        MoveItConfigsBuilder("so101_arm", package_name="so101_moveit_config")
        .robot_description(file_path=xacro_path, mappings={"variant": "follower", "use_ros2_control": "false"})
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines(pipelines=["isaac_ros_cumotion", "ompl"], default_planning_pipeline="isaac_ros_cumotion")
        .joint_limits()
        .to_moveit_configs()
    )
    with open(os.path.join(cumotion_share, "config", "moveit_py_cumotion.yaml")) as f:
        moveit_py_params = yaml.safe_load(f)

    # 1) ros2_control + rsp + controllers (Isaac Sim topic bridge)
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_share, "launch", "follower_split.launch.py")),
        launch_arguments={
            "hardware_type": "isaac",
            "controller_config_file": os.path.join(
                bringup_share, "config", "ros2_control", "follower_isaac_controllers.yaml"
            ),
            "use_rviz": "false",
        }.items(),
    )

    # 2) move_group + cuMotion planner
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(cumotion_share, "launch", "move_group_cumotion.launch.py")),
        launch_arguments={"use_sim_time": "true", "xrdf_path": xrdf_path, "urdf_path": urdf_path}.items(),
    )

    # 3) state machine — MoveItPy 노드명 매칭을 위해 name="pick_place_moveit".
    #    moveit config + moveit_py named set 은 dict 로(이 노드명에 매칭),
    #    pick_place_params.yaml 은 file 로(내부 키 pick_place_object_store 가 store 노드에 매칭).
    sm_node = Node(
        package="so101_cumotion_pick_place",
        executable="pick_place_sm",
        name="pick_place_moveit",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_py_params,
            {"use_sim_time": True},
            os.path.join(sm_share, "config", "pick_place_params.yaml"),
        ],
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_share, "launch", "moveit_rviz.launch.py")),
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument(
                "xrdf_path",
                default_value=PathJoinSubstitution([repo, "assets", "robots", "so101.xrdf"]),
                description="SO-101 XRDF 절대경로(기본: $SO101_REPO/assets/robots/so101.xrdf)",
            ),
            DeclareLaunchArgument(
                "urdf_path",
                default_value=PathJoinSubstitution([repo, "assets", "robots", "urdf", "so_arm101.urdf"]),
                description="cumotion 용 flat URDF(기본: $SO101_REPO/assets/robots/urdf/so_arm101.urdf)",
            ),
            bringup,
            move_group,
            sm_node,
            rviz,
        ]
    )
