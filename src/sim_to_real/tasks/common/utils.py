"""도메인 중립 robot/camera cfg helper (pick_pen_env_cfg 에서 relocate, VLA-only 리팩토링)."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg


SO101_JOINT_TARGET_MAX_VELOCITY: dict[str, float] = {
    "shoulder_pan": 5.00,
    "shoulder_lift": 5.00,
    "elbow_flex": 5.00,
    "wrist_flex": 5.00,
    "wrist_roll": 5.00,
    # 그리퍼도 5.0 rad/s 상한(사용자 지시). 이건 *상한*이지 항상 이 속도로 닫는 게
    # 아니다 — teleop 은 leader 입력 속도를, RL/SM 은 명령 속도를 따른다. (과거 SM 이
    # cap 5.0 에서 큐브를 snap-튕긴 건 명령 속도를 너무 키운 탓이라, 접촉 시 명령 속도
    # 자체를 줄이는 쪽으로 푼다.)
    "gripper": 5.00,
}
"""Processed joint-position target speed cap in rad/s (sim time).

팔 joint = 5.0 rad/s 상한(사용자 요청, teleop·RL·state-machine 공용). 그리퍼 = 1.0 rad/s
(grasp 접촉 안정). actuator ``velocity_limit_sim`` 은 ≥5 rad/s 헤드룸을 유지해야 명령 속도를
실제로 추종한다. 주의: 1.0 rad/s 로 학습된 RL checkpoint 는 팔 변경으로 dynamics 가 바뀌어
재학습이 필요하다."""


def _yaw_quat(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(_dot3(v, v))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {v!r}")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _quat_from_matrix(
    m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float, float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return (
        (m[1][0] - m[0][1]) / s,
        (m[0][2] + m[2][0]) / s,
        (m[1][2] + m[2][1]) / s,
        0.25 * s,
    )


def _look_at_quat_world(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Quaternion for Isaac Lab camera world convention: forward +X, up +Z."""

    forward = _normalize3((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    up_hint = _normalize3(up)
    up_axis_raw = (
        up_hint[0] - _dot3(up_hint, forward) * forward[0],
        up_hint[1] - _dot3(up_hint, forward) * forward[1],
        up_hint[2] - _dot3(up_hint, forward) * forward[2],
    )
    if _dot3(up_axis_raw, up_axis_raw) < 1e-9:
        up_axis_raw = (0.0, 1.0, 0.0)
    up_axis = _normalize3(up_axis_raw)
    right_axis = _cross3(up_axis, forward)
    matrix = (
        (forward[0], right_axis[0], up_axis[0]),
        (forward[1], right_axis[1], up_axis[1]),
        (forward[2], right_axis[2], up_axis[2]),
    )
    return _quat_from_matrix(matrix)


def _pinhole_camera_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    focal_length: float,
    *,
    focus_distance: float,
    clipping_range: tuple[float, float],
) -> TiledCameraCfg:
    """640×480 RGB TiledCamera. offset 은 prim_path 부모 프레임 기준."""

    return TiledCameraCfg(
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        width=640,
        height=480,
        update_period=1.0 / 30.0,
        update_latest_camera_pose=True,
    )
