#!/usr/bin/env python3
"""SO-101 top-camera hand-eye 캘리브 — JOINT-SPACE 방식 (IK 불필요).

5-DOF + MoveIt IK service 는 full-pose 검증으로 orientation 자유도를 못 줘서
대부분 XY 에서 -31. 대신 현재 joints 에 작은 섭동을 가해 그리퍼를 sweep 하고
각 자세에서 FK(gripper_frame_link) base XY <-> top 이미지 그리퍼 tip 픽셀 페어를 모은다.

jplan : 각 섭동 target joints 의 FK 를 이동 없이 미리 계산(안전 z 확인 + joint→cartesian 매핑 파악)
jrun  : 안전(FK z>=floor)·도달 자세만 이동→FK→top 캡처→tip 검출→페어 기록 + 주석 이미지

env.sh source 셸에서:  python3 so101_jcalib.py jplan   /   python3 so101_jcalib.py jrun
"""
import sys, time, json
import numpy as np
import cv2
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState, Image
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
EE = 'gripper_frame_link'
TRAJ_ACTION = '/follower/arm_trajectory_controller/follow_joint_trajectory'
JS_TOPIC = '/follower/joint_states'
TOP_TOPIC = '/camera/top/image_raw'
TMP = '/mnt/c/Users/taehunkim/AppData/Local/Temp'
Z_FLOOR = 0.02
MOVE_DUR = 3.0

# (dpan, dlift, delbow) 섭동 — 현재 joints 기준. pan=좌우 sweep, lift+elbow=reach/height.
# delbow 로 reach 변화 시 그리퍼가 desk 로 처박히지 않게 보정.
OFFSETS = [
    (0.0, 0.0, 0.0),
    (-0.20, 0.0, 0.0), (0.20, 0.0, 0.0),
    (-0.35, 0.0, 0.0), (0.35, 0.0, 0.0),
    (0.0, -0.15, 0.15), (0.0, 0.15, -0.15),
    (-0.20, -0.15, 0.15), (0.20, -0.15, 0.15),
    (-0.20, 0.15, -0.15), (0.20, 0.15, -0.15),
    (-0.35, 0.15, -0.15), (0.35, 0.15, -0.15),
]


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


def make_js(arm_pos, gripper=0.0):
    js = JointState()
    js.name = ARM + ['gripper']
    js.position = [float(v) for v in arm_pos] + [float(gripper)]
    return js


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
    sub = node.create_subscription(Image, TOP_TOPIC, lambda m: box.__setitem__('m', m), 10)
    t0 = time.time()
    while time.time() - t0 < timeout and 'm' not in box:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return bridge.imgmsg_to_cv2(box['m'], 'bgr8') if 'm' in box else None


def detect_tip(img):
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
    tip = pts[np.argmin(pts[:, 1])]
    M = cv2.moments(c)
    cen = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return dict(tip=[int(tip[0]), int(tip[1])], centroid=list(cen), area=int(cv2.contourArea(c)))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'jplan'
    rclpy.init()
    node = rclpy.create_node('so101_jcalib')
    bridge = CvBridge()
    out = {'mode': mode, 'poses': []}
    try:
        js0 = get_js(node)
        if js0 is None:
            print(json.dumps({'error': 'no_joint_states'})); return
        base_arm = js_arm(js0)
        out['base_arm'] = [round(v, 4) for v in base_arm]
        p0 = fk(node, js0)
        out['base_fk'] = [round(p0.position.x, 4), round(p0.position.y, 4), round(p0.position.z, 4)]
        cur_arm = list(base_arm)
        for idx, (dp, dl, de) in enumerate(OFFSETS):
            tgt = [base_arm[0] + dp, base_arm[1] + dl, base_arm[2] + de, base_arm[3], base_arm[4]]
            pf = fk(node, make_js(tgt))
            rec = dict(idx=idx, off=[dp, dl, de])
            if pf is None:
                rec['fk'] = 'fail'; out['poses'].append(rec); continue
            rec['fk_xyz'] = [round(pf.position.x, 4), round(pf.position.y, 4), round(pf.position.z, 4)]
            rec['safe'] = pf.position.z >= Z_FLOOR
            if mode == 'jrun' and rec['safe']:
                mv = move_arm(node, tgt); rec['move'] = mv
                if mv == 'ok':
                    time.sleep(0.6)
                    js1 = get_js(node); pa = fk(node, js1) if js1 else None
                    if pa:
                        rec['fk_act'] = [round(pa.position.x, 4), round(pa.position.y, 4), round(pa.position.z, 4)]
                        cur_arm = js_arm(js1)
                    img = capture_top(node, bridge)
                    if img is not None:
                        det = detect_tip(img); rec['det'] = det
                        ann = img.copy()
                        if det:
                            cv2.circle(ann, tuple(det['tip']), 7, (0, 0, 255), -1)
                            cv2.circle(ann, tuple(det['centroid']), 7, (255, 0, 0), -1)
                        cv2.imwrite('%s/jcal_%02d.jpg' % (TMP, idx), ann)
            out['poses'].append(rec)
        # jrun 끝나면 base 로 복귀
        if mode == 'jrun':
            out['return'] = move_arm(node, base_arm, dur=3.5)
    finally:
        print(json.dumps(out))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
