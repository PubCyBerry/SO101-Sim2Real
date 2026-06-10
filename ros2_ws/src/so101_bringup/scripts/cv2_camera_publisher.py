#!/usr/bin/env python3
"""WSL2 USB-IP 환경용 cv2 기반 멀티카메라 publisher.

gscam(appsink sample pull 실패)·v4l2_camera(YUYV DQBUF 행 / MJPG 디코드 미지원)가
usbipd-win 가상 V4L2 디바이스에서 동작하지 않아, OpenCV(MJPG) 캡처로 대체한다.

USB-IP 가상 디바이스는 (1) 동시 open 경합, (2) 다중 스레드 동시 블로킹 read 를
모두 견디지 못한다. 따라서 단일 스레드에서 3캠을 라운드로빈으로 순차 read 후
각각 sensor_msgs/Image(bgr8) + CameraInfo 를 발행한다(검증: 3캠 합산 ~18fps).

토픽 네이밍은 North Star 계약(observation.images.{top,wrist,front})에 맞춰
/camera/<name>/image_raw 와 /camera/<name>/camera_info 로 통일한다.
"""
import cv2
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

# (device, name) - 토픽: /camera/<name>/image_raw, frame_id: <name>_camera_optical_frame
CAMERAS = [
    ("/dev/cam_top", "top"),
    ("/dev/cam_wrist", "wrist"),
    ("/dev/cam_front", "front"),
]
WIDTH, HEIGHT, FPS = 640, 480, 25


def open_cap(device):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if cap.isOpened() and cap.read()[0]:
        return cap
    cap.release()
    return None


def make_camera_info(frame_id):
    # 미보정 기본값(보정 시 camera_info_url 로 대체). width/height 만 채운다.
    info = CameraInfo()
    info.width = WIDTH
    info.height = HEIGHT
    info.header.frame_id = frame_id
    return info


def main():
    rclpy.init()
    node = rclpy.create_node("cv2_camera_publisher")
    bridge = CvBridge()
    log = node.get_logger()

    cams = []  # (cap, img_pub, info_pub, info_msg, frame_id)
    for device, name in CAMERAS:
        cap = open_cap(device)  # 순차 open - 동시 open 경합 회피
        if cap is None:
            log.error(device + ": open 실패")
            continue
        ns = "/camera/" + name
        frame_id = name + "_camera_optical_frame"
        img_pub = node.create_publisher(Image, ns + "/image_raw", 10)
        info_pub = node.create_publisher(CameraInfo, ns + "/camera_info", 10)
        cams.append((cap, img_pub, info_pub, make_camera_info(frame_id), frame_id))
        log.info(device + " -> " + ns + "/image_raw 스트리밍 시작")

    try:
        while rclpy.ok():
            for cap, img_pub, info_pub, info_msg, frame_id in cams:  # 단일 스레드 라운드로빈
                ret, frame = cap.read()
                if not ret:
                    continue
                stamp = node.get_clock().now().to_msg()
                msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = stamp
                msg.header.frame_id = frame_id
                img_pub.publish(msg)
                info_msg.header.stamp = stamp
                info_pub.publish(info_msg)
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        for cap, *_ in cams:
            cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
