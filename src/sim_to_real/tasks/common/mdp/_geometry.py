"""공유 상수 — common MDP 내부 전용."""

from __future__ import annotations

# world frame 책상 상판 z — scene authoring 상수와 동기화 유지.
#   저작: SCENE_OFFSET[2] = 0.705 (author_pick_cube_scene.py, 상판 local z = 0)
#   실측: 로봇 base world z 0.6749 + so101_contract.grasp_geometry.TABLE_TOP_BASE 0.02976
#         = 0.7047 (차이 0.3 mm = collider 침투)
# 이전 값 0.76 은 pen 태스크 잔재였고 현 씬에는 존재하지 않는 높이였다. 그 탓에
# ``task_done``/``cube_lost``/``object_in_container`` 의 z 판정이 실물보다 5.5 cm 높은 창을
# 보고 있었다(그릇 안 안착 큐브 0.743 < 옛 하한 0.765 → 성공 판정 불가).
DESK_TOP_Z: float = 0.705

# jaw body 기준 실제 접촉 중심까지의 로컬 오프셋 (m)
JAW_GRASP_OFFSET: tuple[float, float, float] = (-0.021, -0.070, 0.020)

# 컨테이너(컵/그릇) 기본 위치 (m, world XY). env_cfg 에서 명시 주입 시 무시.
# recenter(robot base→원점, delta=(-1.84,+0.565)) 반영.
CONTAINER_DEFAULT_CENTER_XY: tuple[float, float] = (0.36, 0.395)
