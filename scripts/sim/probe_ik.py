#!/usr/bin/env python3
"""SO-101 grasp reachability 프로브 (PATH E, cuMotion+ROS) — FK 워크스페이스 샘플링.

실행 중인 move_group 의 `/compute_fk`(MoveIt) 서비스로 랜덤 joint config 를 FK 해서
도달 가능 워크스페이스 cloud 를 만들고, 질의 위치 근처에 팁이 가는 config 와 그때의
approach tilt(tool z 가 수직 아래에서 기운 각)를 보고한다.

왜 FK 샘플링인가: SO-101 은 5-DOF 라 임의 6-DOF pose 의 IK(/compute_ik)는 거의 항상
NO_IK_SOLUTION(-31)을 낸다 — orientation 을 정확히 못 맞추기 때문. 따라서 "위치가 도달
가능한가 + 거기서 어떤 자세가 나오는가"는 FK 로 샘플링해야 정확히 안다.

사용(컨테이너, bridge + pick_place.launch.py 가 떠 있어야 함):
    source /opt/ros/jazzy/setup.bash && source /build/install/setup.bash
    python3 /workspace/scripts/sim/probe_ik.py --samples 3000
"""
from __future__ import annotations

import argparse
import math
import random

import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionFK
from rclpy.node import Node
from sensor_msgs.msg import JointState

EE = "gripper_frame_link"
BASE = "base_link"
# URDF arm joint limits (so_arm101.urdf).
LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}
ARM = list(LIMITS.keys())

# 질의: cube1 컬럼·bowl 컬럼의 z-sweep(도달 가능 높이 범위 파악 — approach/lift/transport 예산).
QUERIES = {f"cube_z{z:.2f}(0.14,-0.125)": (0.140, -0.125, z)
           for z in (0.05, 0.08, 0.11, 0.14, 0.17)}
QUERIES.update({f"bowl_z{z:.2f}(0.22,-0.305)": (0.220, -0.305, z)
                for z in (0.04, 0.08, 0.12, 0.16)})


def tool_z_tilt(qx, qy, qz, qw):
    """tool z 축이 수직 아래(-z_base)에서 기운 각(deg). 0=완전 down, 90=수평."""
    # R * [0,0,1] 의 base-frame 성분.
    zx = 2 * (qx * qz + qw * qy)
    zy = 2 * (qy * qz - qw * qx)
    zz = 1 - 2 * (qx * qx + qy * qy)
    down = -zz  # +1 이면 완전 아래
    down = max(-1.0, min(1.0, down))
    return math.degrees(math.acos(down)), (zx, zy, zz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--near", type=float, default=0.035, help="질의 위치 매칭 반경(m)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("probe_fk")
    fk_cli = node.create_client(GetPositionFK, "/compute_fk")
    fk_cli.wait_for_service(timeout_sec=10.0)

    def fk(qvals):
        req = GetPositionFK.Request()
        req.header.frame_id = BASE
        req.fk_link_names = [EE]
        rs = RobotState()
        js = JointState()
        js.name = ARM
        js.position = qvals
        rs.joint_state = js
        req.robot_state = rs
        f = fk_cli.call_async(req)
        rclpy.spin_until_future_complete(node, f, timeout_sec=5.0)
        res = f.result()
        if res and res.error_code.val == 1 and res.pose_stamped:
            return res.pose_stamped[0].pose
        return None

    cloud = []  # (x,y,z, tilt_deg)
    for _ in range(args.samples):
        qvals = [random.uniform(*LIMITS[j]) for j in ARM]
        p = fk(qvals)
        if p is None:
            continue
        tilt, _ = tool_z_tilt(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        cloud.append((p.position.x, p.position.y, p.position.z, tilt))

    print(f"[FK cloud] {len(cloud)}/{args.samples} samples")
    if cloud:
        xs = [c[0] for c in cloud]; ys = [c[1] for c in cloud]; zs = [c[2] for c in cloud]
        rs = [math.hypot(c[0], c[1]) for c in cloud]
        print(f"  x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}] "
              f"z[{min(zs):.3f},{max(zs):.3f}] radius[{min(rs):.3f},{max(rs):.3f}]")

    # grasp-height 도달 가능 영역: 데스크 grasp 밴드 z∈[zlo,zhi]에서 down-graspable(tilt≤tmax)
    # 한 sample 의 (x,y) 분포 → scatter box 결정용. base_link frame.
    zlo, zhi, tmax = 0.02, 0.09, 50.0
    band = [(c[0], c[1]) for c in cloud if zlo <= c[2] <= zhi and c[3] <= tmax]
    print(f"\n[grasp-height 영역] z∈[{zlo},{zhi}] & tilt≤{tmax}° (down-graspable) sample={len(band)}")
    if band:
        bx = [p[0] for p in band]; by = [p[1] for p in band]
        print(f"  base_link x[{min(bx):.3f},{max(bx):.3f}] y[{min(by):.3f},{max(by):.3f}]")
        # 5cm 그리드 density (x: -0.1..0.45, y: -0.45..0.05)
        import collections
        grid = collections.Counter()
        for x, y in band:
            grid[(round(x / 0.05) * 0.05, round(y / 0.05) * 0.05)] += 1
        print("  density grid(5cm, 값=samples), x↓ y→:")
        ys = sorted({k[1] for k in grid})
        xs = sorted({k[0] for k in grid})
        print("        " + " ".join(f"{y:+.2f}" for y in ys))
        for x in xs:
            row = " ".join(f"{grid.get((x, y), 0):5d}" for y in ys)
            print(f"  x{x:+.2f} {row}")

    print(f"\n[질의별 도달성] near={args.near}m 안에 든 sample 수 + 그때 tilt(0=down,90=수평):")
    for name, (qx, qy, qz) in QUERIES.items():
        near = [c for c in cloud if math.dist((c[0], c[1], c[2]), (qx, qy, qz)) <= args.near]
        if near:
            tilts = sorted(c[3] for c in near)
            tmin, tmax = tilts[0], tilts[-1]
            print(f"  {name:36} hits={len(near):4}  tilt[{tmin:.0f}°~{tmax:.0f}°]  "
                  f"(down에 가장 가까운={tmin:.0f}°)")
        else:
            # 가장 가까운 sample 거리
            if cloud:
                d = min(math.dist((c[0], c[1], c[2]), (qx, qy, qz)) for c in cloud)
                print(f"  {name:36} hits=   0  ⚠도달불가 (최근접 sample {d*100:.1f}cm)")
            else:
                print(f"  {name:36} hits=   0  (cloud 비어있음)")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
