"""WSL2 USB-IP 환경용 cv2 카메라 publisher launch.

gscam/v4l2_camera 가 usbipd-win 가상 V4L2 디바이스에서 동작하지 않아
(MMAP/포맷 협상 실패), OpenCV(MJPG) 캡처 노드로 카메라 이미지를 발행한다.
cam_wrist/cam_overhead/cam_front 3캠을 단일 노드에서 라운드로빈 캡처한다.

RMW 는 환경(env.sh)을 따른다. mirrored 네트워킹에서는 CycloneDDS 가
Image cross-process 전달에 실패하므로 env.sh 가 FastDDS 를 설정한다.

사용:
  source <repo>/ros2_ws/setup/env.sh
  ros2 launch so101_bringup cameras_cv2.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="so101_bringup",
                executable="cv2_camera_publisher.py",
                name="cv2_camera_publisher",
                output="screen",
            )
        ]
    )
