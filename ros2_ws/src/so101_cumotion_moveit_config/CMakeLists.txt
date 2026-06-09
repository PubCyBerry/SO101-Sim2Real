cmake_minimum_required(VERSION 3.8)
project(so101_cumotion_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
