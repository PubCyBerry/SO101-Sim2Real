#!/usr/bin/env python3
"""SO-101 follower 캘리브/제어 헬퍼 (native rclpy).

MoveIt /compute_ik·/compute_fk + arm_trajectory_controller FollowJointTrajectory 액션.
실기기 blind 조작 안전장치: IK 실패 시 모션 없음, z 하한 가드, 결과 JSON 출력.

사용 (env.sh source 한 셸에서):
  python3 so101_cal.py state
  python3 so101_cal.py iktest X Y Z [QX QY QZ QW]
  python3 so101_cal.py move   X Y Z [QX QY QZ QW] [DUR]
  python3 so101_cal.py nudgez DZ [DUR]
  python3 so101_cal.py movej  j0 j1 j2 j3 j4 [DUR]     # 5 arm joints (canonical order)
"""
import sys, time, json
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
EE = 'gripper_frame_link'
GROUP = 'manipulator'
TRAJ_ACTION = '/follower/arm_trajectory_controller/follow_joint_trajectory'
JS_TOPIC = '/follower/joint_states'
Z_FLOOR = 0.005   # base_link 기준 안전 z 하한 (이 아래로는 IK 타겟 거부)


def get_js(node, timeout=25.0):
    # FastDDS+WSL mirrored 에서 fresh 노드 discovery 가 느려 warmup 후 대기
    box = {}
    sub = node.create_subscription(JointState, JS_TOPIC,
                                   lambda m: box.__setitem__('m', m),
                                   qos_profile_sensor_data)
    t0 = time.time()
    # discovery warmup: publisher 가 보일 때까지(또는 3s) 먼저 spin
    while time.time() - t0 < 3.0 and sub.get_publisher_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    while time.time() - t0 < timeout and 'm' not in box:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return box.get('m')


def fk(node, js, link=EE):
    cli = node.create_client(GetPositionFK, '/compute_fk')
    if not cli.wait_for_service(timeout_sec=5):
        return None
    req = GetPositionFK.Request()
    req.header.frame_id = 'base_link'
    req.fk_link_names = [link]
    rs = RobotState(); rs.joint_state = js
    req.robot_state = rs
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5)
    res = fut.result()
    if not res or not res.pose_stamped:
        return None
    return res.pose_stamped[0].pose


def ik(node, js, x, y, z, quat, timeout=2.0):
    cli = node.create_client(GetPositionIK, '/compute_ik')
    if not cli.wait_for_service(timeout_sec=5):
        return None, 'no_ik_service'
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
    r.pose_stamped.pose.orientation.x = float(quat[0])
    r.pose_stamped.pose.orientation.y = float(quat[1])
    r.pose_stamped.pose.orientation.z = float(quat[2])
    r.pose_stamped.pose.orientation.w = float(quat[3])
    r.timeout.sec = int(timeout)
    r.timeout.nanosec = int((timeout % 1) * 1e9)
    req.ik_request = r
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout + 6)
    res = fut.result()
    if not res:
        return None, 'no_response'
    if res.error_code.val != 1:
        return None, 'ik_err_%d' % res.error_code.val
    name2pos = dict(zip(res.solution.joint_state.name,
                        res.solution.joint_state.position))
    try:
        return [float(name2pos[j]) for j in ARM], 'ok'
    except KeyError as e:
        return None, 'missing_joint_%s' % e


def move_arm(node, arm_pos, dur=4.0):
    ac = ActionClient(node, FollowJointTrajectory, TRAJ_ACTION)
    if not ac.wait_for_server(timeout_sec=6):
        return 'no_action_server'
    goal = FollowJointTrajectory.Goal()
    jt = JointTrajectory(); jt.joint_names = ARM
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in arm_pos]
    pt.velocities = [0.0] * 5
    pt.time_from_start.sec = int(dur)
    pt.time_from_start.nanosec = int((dur % 1) * 1e9)
    jt.points = [pt]
    goal.trajectory = jt
    fut = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=6)
    gh = fut.result()
    if not gh or not gh.accepted:
        return 'goal_rejected'
    rf = gh.get_result_async()
    rclpy.spin_until_future_complete(node, rf, timeout_sec=dur + 8)
    r = rf.result()
    return 'err_%d' % r.result.error_code if r else 'no_result'


def pose_xyzq(p):
    return dict(x=round(p.position.x, 4), y=round(p.position.y, 4),
                z=round(p.position.z, 4),
                q=[round(p.orientation.x, 4), round(p.orientation.y, 4),
                   round(p.orientation.z, 4), round(p.orientation.w, 4)])


def main():
    a = sys.argv[1:]
    mode = a[0] if a else 'state'
    rclpy.init()
    node = rclpy.create_node('so101_cal')
    out = {'mode': mode}
    try:
        js = get_js(node)
        if js is None:
            print(json.dumps({'error': 'no_joint_states'})); return
        cur = fk(node, js)
        if cur is None:
            print(json.dumps({'error': 'fk_failed'})); return
        cur_q = [cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w]
        out['cur_pose'] = pose_xyzq(cur)
        out['cur_joints'] = {n: round(p, 4) for n, p in zip(js.name, js.position)}

        if mode == 'state':
            pass

        elif mode in ('iktest', 'move'):
            x, y, z = float(a[1]), float(a[2]), float(a[3])
            rest = a[4:]
            if len(rest) >= 4:
                quat = [float(v) for v in rest[:4]]
                dur = float(rest[4]) if len(rest) >= 5 else 4.0
            else:
                quat = cur_q
                dur = float(rest[0]) if len(rest) >= 1 else 4.0
            out['target'] = dict(x=x, y=y, z=z, q=[round(v, 4) for v in quat], dur=dur)
            if z < Z_FLOOR:
                out['result'] = 'z_below_floor'; print(json.dumps(out)); return
            sol, status = ik(node, js, x, y, z, quat)
            out['ik_status'] = status
            out['ik_solution'] = [round(v, 4) for v in sol] if sol else None
            if mode == 'move' and sol:
                out['move_result'] = move_arm(node, sol, dur)
                time.sleep(0.5)
                js2 = get_js(node)
                p2 = fk(node, js2) if js2 else None
                out['achieved_pose'] = pose_xyzq(p2) if p2 else None

        elif mode == 'nudgez':
            dz = float(a[1]); dur = float(a[2]) if len(a) > 2 else 3.0
            z = cur.position.z + dz
            out['target'] = dict(x=round(cur.position.x, 4), y=round(cur.position.y, 4),
                                 z=round(z, 4), dz=dz, dur=dur)
            if z < Z_FLOOR:
                out['result'] = 'z_below_floor'; print(json.dumps(out)); return
            sol, status = ik(node, js, cur.position.x, cur.position.y, z, cur_q)
            out['ik_status'] = status
            if sol:
                out['move_result'] = move_arm(node, sol, dur)
                time.sleep(0.5)
                js2 = get_js(node); p2 = fk(node, js2) if js2 else None
                out['achieved_pose'] = pose_xyzq(p2) if p2 else None

        elif mode == 'movej':
            tgt = [float(v) for v in a[1:6]]
            dur = float(a[6]) if len(a) > 6 else 4.0
            out['target_joints'] = tgt
            out['move_result'] = move_arm(node, tgt, dur)
            time.sleep(0.5)
            js2 = get_js(node); p2 = fk(node, js2) if js2 else None
            out['achieved_pose'] = pose_xyzq(p2) if p2 else None
        else:
            out['error'] = 'unknown_mode'
    finally:
        print(json.dumps(out))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
