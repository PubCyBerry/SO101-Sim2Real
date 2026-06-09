"""Isaac Sim cube_desk pick & place 통합 launch (WSL2 측).

스택:
  robot_state_publisher (hardware_type:=isaac)
  + controller_manager (topic_based hardware, isaac_controllers.yaml)
  + spawners (joint_state_broadcaster, gripper_controller, arm_trajectory_controller)
  + move_group (use_cumotion:=true → ompl/pilz/isaac_ros_cumotion pipelines)
  + cumotion_action_server (xrdf+urdf, tool_frame=gripper_frame_link)  [use_cumotion 시]
  + MoveIt RViz (옵션)
  + orchestrator FSM 노드 (run_orchestrator:=true 시 자동 실행)

Windows 쪽에서  `uv run scripts/ros2/cube_desk_ros2_sim.py` 로 Isaac Sim 을 띄워
/isaac_joint_states 발행 + /isaac_joint_commands 구독 + 큐브/그릇 pose 발행해야 한다.

예:
  ros2 launch so101_bringup isaac_pick_place.launch.py use_cumotion:=true run_orchestrator:=false
"""
import os
import subprocess
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _xacro(xacro_file: str, **mappings) -> str:
    """xacro 파일을 인자와 함께 펼쳐 URDF 문자열 반환."""
    args = ["xacro", xacro_file] + [f"{k}:={v}" for k, v in mappings.items()]
    return subprocess.check_output(args, text=True)


def _launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace").perform(context)
    use_rviz = LaunchConfiguration("use_rviz").perform(context)
    use_cumotion = LaunchConfiguration("use_cumotion").perform(context)
    run_orchestrator = LaunchConfiguration("run_orchestrator").perform(context)
    use_cumotion_bool = use_cumotion.lower() in ("true", "1")

    desc_share = get_package_share_directory("so101_description")
    bringup_share = get_package_share_directory("so101_bringup")
    moveit_share = get_package_share_directory("so101_moveit_config")
    xacro_file = os.path.join(desc_share, "urdf", "so101_arm.urdf.xacro")

    # robot_description (isaac topic_based hardware)
    robot_description_xml = _xacro(
        xacro_file, variant="follower", use_ros2_control="true", hardware_type="isaac"
    )
    robot_description = {"robot_description": ParameterValue(robot_description_xml, value_type=str)}

    controllers_yaml = os.path.join(
        bringup_share, "config", "ros2_control", "isaac_controllers.yaml"
    )

    nodes = []

    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            parameters=[robot_description, {"use_sim_time": True}],
            output="screen",
        )
    )

    nodes.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            namespace=namespace,
            parameters=[robot_description, controllers_yaml, {"use_sim_time": True}],
            output="screen",
            emulate_tty=True,
        )
    )

    for controller in ("joint_state_broadcaster", "gripper_controller", "arm_trajectory_controller"):
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                namespace=namespace,
                arguments=[controller],
                output="screen",
            )
        )

    # move_group (cuMotion 포함 여부 전달)
    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_share, "launch", "move_group.launch.py")
            ),
            launch_arguments={
                "namespace": namespace,
                "variant": "follower",
                "use_sim_time": "true",
                "use_cumotion": use_cumotion,
            }.items(),
        )
    )

    # cuMotion action server (xrdf + plain urdf)
    if use_cumotion_bool:
        xrdf_path = os.path.join(moveit_share, "config", "so101_arm.xrdf")
        # cuRobo 는 plain URDF 를 원하므로 ros2_control 없는 버전을 /tmp 에 생성.
        plain_urdf = _xacro(xacro_file, variant="follower", use_ros2_control="false")
        urdf_path = os.path.join(tempfile.gettempdir(), "so101_follower_plain.urdf")
        with open(urdf_path, "w") as f:
            f.write(plain_urdf)

        cumotion_launch = os.path.join(
            get_package_share_directory("isaac_ros_cumotion"),
            "launch",
            "isaac_ros_cumotion.launch.py",
        )
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(cumotion_launch),
                launch_arguments={
                    "cumotion_planner.urdf_path": urdf_path,
                    "cumotion_planner.xrdf_path": xrdf_path,
                    "cumotion_planner.tool_frame": "gripper_frame_link",
                    # joint_states remap (namespaced)
                    "cumotion_planner.joint_states_topic": "/" + namespace + "/joint_states",
                }.items(),
            )
        )

    # MoveIt RViz
    nodes.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit_share, "launch", "moveit_rviz.launch.py")
            ),
            launch_arguments={"namespace": namespace, "variant": "follower"}.items(),
            condition=IfCondition(use_rviz),
        )
    )

    # orchestrator FSM (옵션 자동 실행 — 보통은 스택 기동 후 수동 실행 권장)
    if run_orchestrator.lower() in ("true", "1"):
        nodes.append(
            Node(
                package="so101_moveit_config",
                executable="so101_pick_place_orchestrator.py",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="follower"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "use_cumotion",
                default_value="false",
                description="cuMotion(cuRobo) planner 활성(Linux 서버, isaac_ros_cumotion 설치 시). "
                "기본 false=OMPL/Pilz (WSL2 는 cuMotion 의존성 미충족).",
            ),
            DeclareLaunchArgument(
                "run_orchestrator",
                default_value="false",
                description="pick&place FSM 노드 자동 실행 (디버깅 시 false 후 수동 실행 권장)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
