#!/usr/bin/env python3
"""SO-101 top-camera hand-eye 호모그래피 캘리브 수집 (native rclpy).

고정 orientation + 고정 z 로 base XY 격자를 이동하며,
각 점에서 (top 이미지 그리퍼 tip 픽셀) <-> (이동 후 FK gripper_frame_link base XY) 페어를 모은다.

modes:
  plan : 격자 각 점 IK 도달성 + 현재 자세 대비 joint delta 만 출력 (모션 없음)
  run  : 도달가능+안전 점만 이동·캡처·검출·페어 기록 -> JSON + 주석 이미지

env.sh(FastDDS) source 한 셸에서:
  python3 so101_calib.py plan
  python3 so101_calib.py run
"""
import sys, time, json
import numpy as np
import cv2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, Image
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
EE = 'gripper_frame_link'
GROUP = 'manipulator'
TRAJ_ACTION = '/follower/arm_trajectory_controller/follow_joint_trajectory'
JS_TOPIC = '/follower/joint_states'
TOP_TOPIC = '/camera/top/image_raw'
TMP = '/mnt/c/Users/taehunkim/AppData/Local/Temp'

# 캘리브 격자 (base_link frame). 큐브 영역(top 상단-중앙)을 덮도록 전방으로 확장.
XS = [0.16, 0.21, 0.26]
YS = [-0.08, 0.0, 0.08]
Z_CAL = 0.13              # 큐브(2.5cm) 위로 그리퍼 통과 — cube-knock 방지(parallax 는 wrist servo 가 보정)
MAX_JOINT_DELTA = 1.0     # rad, flip/대형 swing 가드
MOVE_DUR = 3.5


def get_js(node, timeout=15.0):
    box = {}
    sub = node.create_subscription(JointState, JS_TOPIC,
                                   lambda m: box.__setitem__('m', m), qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < timeout and 'm' not in box:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return box.get('m')


def js_arm(js):
    d = dict(zip(js.name, js.position))
    return [d[j] for j in ARM]


def fk(node, js):
    cli = node.create_client(GetPositionFK, '/compute_fk')
    if not cli.wait_for_service(timeout_sec=5):
        return None
    req = GetPositionFK.Request()
    req.header.frame_id = 'base_link'
    req.fk_link_names = [EE]
    rs = RobotState(); rs.joint_state = js
    req.robot_state = rs
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5)
    res = fut.result()
    return res.pose_stamped[0].pose if res and res.pose_stamped else None


def ik(node, js, x, y, z, q, timeout=2.0):
    cli = node.create_client(GetPositionIK, '/compute_ik')
    if not cli.wait_for_service(timeout_sec=5):
        return None, 'no_service'
    req = GetPositionIK.Request()
    r = PositionIKRequest()
    r.group_name = GROUP
    rs = RobotState(); rs.joint_state = js
    r.robot_state = rs
    r.ik_link_name = EE
    r.pose_stamped.header.frame_id = 'base_link'
    r.pose_stamped.pose.position.x = float(x)
    r.pose_stamped.pose.position.y = float(y)
    r.pose_stamped.pose.position.z = float(z)
    r.pose_stamped.pose.orientation.x = float(q[0])
    r.pose_stamped.pose.orientation.y = float(q[1])
    r.pose_stamped.pose.orientation.z = float(q[2])
    r.pose_stamped.pose.orientation.w = float(q[3])
    r.timeout.sec = int(timeout); r.timeout.nanosec = int((timeout % 1) * 1e9)
    req.ik_request = r
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout + 6)
    res = fut.result()
    if not res:
        return None, 'no_response'
    if res.error_code.val != 1:
        return None, 'ik_err_%d' % res.error_code.val
    d = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
    return [float(d[j]) for j in ARM], 'ok'


def move_arm(node, arm_pos, dur=MOVE_DUR):
    ac = ActionClient(node, FollowJointTrajectory, TRAJ_ACTION)
    if not ac.wait_for_server(timeout_sec=6):
        return 'no_server'
    goal = FollowJointTrajectory.Goal()
    jt = JointTrajectory(); jt.joint_names = ARM
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in arm_pos]
    pt.velocities = [0.0] * 5
    pt.time_from_start.sec = int(dur); pt.time_from_start.nanosec = int((dur % 1) * 1e9)
    jt.points = [pt]
    goal.trajectory = jt
    fut = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=6)
    gh = fut.result()
    if not gh or not gh.accepted:
        return 'rejected'
    rf = gh.get_result_async()
    rclpy.spin_until_future_complete(node, rf, timeout_sec=dur + 8)
    r = rf.result()
    return 'ok' if (r and r.result.error_code == 0) else 'err'


def capture_top(node, bridge, timeout=6.0):
    box = {}
    sub = node.create_subscription(Image, TOP_TOPIC,
                                   lambda m: box.__setitem__('m', m), 10)
    t0 = time.time()
    while time.time() - t0 < timeout and 'm' not in box:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if 'm' not in box:
        return None
    return bridge.imgmsg_to_cv2(box['m'], 'bgr8')


def detect_gripper_tip(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (115, 60, 40), (165, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None
    pts = c.reshape(-1, 2)
    tip = pts[np.argmin(pts[:, 1])]   # 화면 위쪽(=전방) 극점
    M = cv2.moments(c)
    cen = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return dict(tip=[int(tip[0]), int(tip[1])], centroid=list(cen),
                area=int(cv2.contourArea(c)))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plan'
    rclpy.init()
    node = rclpy.create_node('so101_calib')
    bridge = CvBridge()
    out = {'mode': mode, 'z_cal': Z_CAL, 'poses': []}
    try:
        js0 = get_js(node)
        if js0 is None:
            print(json.dumps({'error': 'no_joint_states'})); return
        cur = fk(node, js0)
        q = [cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w]
        cur_arm = js_arm(js0)
        out['fixed_quat'] = [round(v, 4) for v in q]
        out['start_pose'] = dict(x=round(cur.position.x, 4), y=round(cur.position.y, 4),
                                 z=round(cur.position.z, 4))

        # snake order 로 격자 생성 (인접 이동 최소화)
        grid = []
        for i, x in enumerate(XS):
            ys = YS if i % 2 == 0 else list(reversed(YS))
            for y in ys:
                grid.append((x, y))

        if mode == 'probe':
            probe_pts = [
                (0.188, -0.023, 0.051), (0.188, -0.023, 0.07), (0.188, -0.023, 0.09),
                (0.22, -0.023, 0.05), (0.25, 0.0, 0.05), (0.25, 0.0, 0.03),
                (0.20, 0.06, 0.05), (0.20, -0.06, 0.05), (0.16, 0.0, 0.05),
                (0.22, 0.0, 0.04), (0.28, 0.0, 0.04),
            ]
            for px, py, pz in probe_pts:
                sol, status = ik(node, js0, px, py, pz, q)
                rec = dict(target=[px, py, pz], ik=status)
                if sol:
                    rec['max_delta'] = round(max(abs(s - c) for s, c in zip(sol, cur_arm)), 3)
                out['poses'].append(rec)
            return

        seed = js0
        for idx, (x, y) in enumerate(grid):
            sol, status = ik(node, seed, x, y, Z_CAL, q)
            rec = dict(idx=idx, target=[x, y, Z_CAL], ik=status)
            if sol:
                delta = max(abs(s - c) for s, c in zip(sol, cur_arm))
                rec['max_delta'] = round(delta, 3)
                rec['safe'] = delta <= MAX_JOINT_DELTA
            else:
                rec['safe'] = False
            if mode == 'run' and sol and rec['safe']:
                mv = move_arm(node, sol)
                rec['move'] = mv
                if mv == 'ok':
                    time.sleep(0.6)
                    js1 = get_js(node)
                    p1 = fk(node, js1) if js1 else None
                    if p1:
                        rec['fk_xy'] = [round(p1.position.x, 4), round(p1.position.y, 4),
                                        round(p1.position.z, 4)]
                        cur_arm = js_arm(js1)   # 다음 점 delta 기준 갱신
                        seed = js1
                    img = capture_top(node, bridge)
                    if img is not None:
                        det = detect_gripper_tip(img)
                        rec['det'] = det
                        ann = img.copy()
                        if det:
                            cv2.circle(ann, tuple(det['tip']), 7, (0, 0, 255), -1)
                            cv2.circle(ann, tuple(det['centroid']), 7, (255, 0, 0), -1)
                        cv2.imwrite('%s/calib_%02d.jpg' % (TMP, idx), ann)
            out['poses'].append(rec)
    finally:
        print(json.dumps(out))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
