cmake_minimum_required(VERSION 3.8)
project(so101_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(so101_description REQUIRED)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

# WSL2 USB-IP + FastDDS 카메라 publisher (gscam/v4l2_camera 가 USB-IP 에서
# 동작하지 않아 OpenCV MJPG 캡처로 대체). ros2 run so101_bringup cv2_camera_publisher.py
install(
  PROGRAMS scripts/cv2_camera_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
