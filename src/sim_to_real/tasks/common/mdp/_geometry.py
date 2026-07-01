"""공유 상수 — common MDP 내부 전용."""

from __future__ import annotations

# world frame 책상 상판 z — scene authoring 상수와 동기화 유지
DESK_TOP_Z: float = 0.76

# jaw body 기준 실제 접촉 중심까지의 로컬 오프셋 (m)
JAW_GRASP_OFFSET: tuple[float, float, float] = (-0.021, -0.070, 0.020)

# 컨테이너(컵/그릇) 기본 위치 (m, world XY). env_cfg 에서 명시 주입 시 무시.
# recenter(robot base→원점, delta=(-1.84,+0.565)) 반영.
CONTAINER_DEFAULT_CENTER_XY: tuple[float, float] = (0.36, 0.395)
