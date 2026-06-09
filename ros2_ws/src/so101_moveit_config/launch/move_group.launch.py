import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    variant = LaunchConfiguration("variant").perform(context)
    # use_cumotion=true 면 isaac_ros_cumotion pipeline 을 추가한다.
    # (cuMotion 미설치 환경에서 기존 PATH D 가 깨지지 않도록 기본 false)
    use_cumotion = LaunchConfiguration("use_cumotion").perform(context).lower() in ("true", "1")

    xacro_path = os.path.join(
        get_package_share_directory("so101_description"),
        "urdf",
        "so101_arm.urdf.xacro",
    )

    pipelines = ["ompl", "pilz_industrial_motion_planner"]
    if use_cumotion:
        # MoveItConfigsBuilder 는 config/<pipeline>_planning.yaml 을 찾는다.
        # isaac_ros_cumotion_moveit 의 isaac_ros_cumotion_planning.yaml 을
        # so101_moveit_config/config/ 로 복사해 둘 것 (05_install_cumotion.sh 안내).
        pipelines.append("isaac_ros_cumotion")

    moveit_config = (
        MoveItConfigsBuilder("so101_arm", package_name="so101_moveit_config")
        .robot_description(
            file_path=xacro_path,
            mappings={
                "variant": variant,
                "use_ros2_control": "false",
            },
        )
        .robot_description_semantic()  # uses your SRDF in so101_moveit_config
        .robot_description_kinematics()
        .planning_pipelines(pipelines=pipelines)
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .joint_limits()
        .trajectory_execution(
            file_path="config/moveit_controllers.yaml",
            moveit_manage_controllers=False,  # don't let MoveIt switch controllers
        )
        .to_moveit_configs()
    )

    # Run in root namespace (MoveIt + namespaces is buggy, all reference
    # projects do this).  Remap joint_states so move_group finds the
    # topic published inside the controller namespace.
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time == "true"}],
        remappings=[("joint_states", "/" + namespace + "/joint_states")],
    )

    return [move_group_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="follower"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("variant", default_value="follower"),
            DeclareLaunchArgument(
                "use_cumotion",
                default_value="false",
                description="isaac_ros_cumotion 플래닝 파이프라인 추가 (Isaac Sim pick&place 경로)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
