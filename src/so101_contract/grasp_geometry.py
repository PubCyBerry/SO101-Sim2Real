"""SO-101 그리퍼 pad 기하 — **단일 진실 소스**.

fixed jaw pad 의 tcp-frame 오프셋은 grasp 조준(planner)과 진단 로그(SM)가 **같은 값**을
써야 한다. 예전엔 ``scripts/cuRobo/curobo_batch_planner.py`` 와 ``scripts/cuRobo/pickplace_sm.py``
에 리터럴로 복제돼 있어, 한쪽만 갱신되면 "로그는 맞는데 조준이 틀린"(또는 그 반대) 디버깅
지옥이 열린다.

값의 출처 = ``assets/robots/so101.yml`` (gripper_link collision sphere 실측) + ``tcp_grasp``
extra_link. **so101.yml 을 다시 빌드하면 여기도 같이 갱신**해야 한다.

⚠ 자족 규칙: stdlib 만 쓴다. isaac-sim 컨테이너와 curobo-datagen 컨테이너 양쪽에서
``PYTHONPATH=/workspace/src`` 로 import 되므로 무거운 의존을 들이면 안 된다.
"""

# fixed jaw pad 접촉면 center 의 tcp-frame 오프셋 (closing, lateral, jaw 아래방향), m.
# pad 이 tcp 서 46 mm 아래·15 mm 옆 — tcp 를 큐브 중심에 조준하면 pad 이 face 를 크게
# 벗어나 모서리를 잡는다(그래서 조준 타깃은 tcp 가 아니라 이 pad center proxy).
FIXED_INNER_CENTER: tuple[float, float, float] = (0.0215, 0.0147, 0.0463)

# tcp → 물리 pad 최저점(fixed jaw tip)의 approach 축 거리 (m).
# so101.yml 실측(tip -0.100 − tcp -0.025). moving jaw 는 12 mm 얕아 fixed 기준이 지배한다.
# table clearance clamp 와 진단 로그가 이 값을 공유한다.
PAD_LOW_OFF: float = 0.075

# 책상 상판 z, **robot base_link 프레임** (m).
#
# 저작값과 실측이 둘 다 있다(2026-07-29):
#   저작 — author_pick_cube_scene.py: 상판 = SCENE_OFFSET[2](0.705) + _DESK_TOP_LOCAL(0.0),
#          로봇 base world z = 0.6749  →  base_link 프레임 상판 = 0.0301
#   실측 — settle 후 큐브 중심 z = 0.04976 ± 0.0001 (96 샘플, 추가 30 스텝 변화 0.09 mm
#          = 잔진동 아닌 정착값)  →  상판 = 0.04976 − half(0.020) = 0.02976
#   차이 0.34 mm = 큐브 collider 접촉 침투(CUBE_CONTACT_OFFSET 0.002). 물리가 보는 값이
#   조준에 맞으므로 **실측 쪽**을 쓴다.
#
# ★ 임의 큐브 크기에 유효하다: 씬은 큐브를 `상판 + half + slack` 에 저작하고 물리는
#   `상판 + half − 침투` 로 정착시킨다. 둘 다 half 에 선형이고 상수항이 같아, 40/50 mm
#   어느 쪽이든 `TABLE_TOP_BASE + half` 가 안착 중심이다(50 mm 별도 실측 불요).
# SM 은 여기에 half 를 더해 grasp 조준 z 를 유도하고, planner 는 BASE_T[2] 를 더해 urdf
# 프레임 TABLE_TOP 으로 쓴다(descend clamp). **두 곳이 같은 물리량을 쓰므로 여기가 단일 소스다** —
# 예전엔 SM 의 --grasp_z 0.060(경험 튜닝)과 planner 의 0.035(리터럴)가 따로 놀아 각각
# +10.24 mm / +5.24 mm 어긋나 있었다.
TABLE_TOP_BASE: float = 0.0298


def _self_check() -> None:
    assert len(FIXED_INNER_CENTER) == 3
    assert all(isinstance(v, float) for v in FIXED_INNER_CENTER)
    # pad 은 tcp 아래(+z, approach 방향)에 있고 tip 은 그보다 더 아래여야 한다.
    assert 0.0 < FIXED_INNER_CENTER[2] < PAD_LOW_OFF, "pad center 는 tcp 와 jaw tip 사이"
    # 상판은 base_link 원점보다 위이고 팔 길이 안이다(프레임 뒤집힘 감지).
    assert 0.0 < TABLE_TOP_BASE < 0.1, f"TABLE_TOP_BASE 가 base_link 프레임 값이 아니다: {TABLE_TOP_BASE}"
    # 40 mm 큐브 안착 중심 = 상판 + half. 실측 0.04976 과 1 mm 안에서 맞아야 한다.
    assert abs((TABLE_TOP_BASE + 0.020) - 0.04976) < 0.001, "실측 안착 z 와 불일치"
    print("[grasp_geometry] OK "
          f"fixed_inner={FIXED_INNER_CENTER} pad_low_off={PAD_LOW_OFF} "
          f"table_top_base={TABLE_TOP_BASE}")


if __name__ == "__main__":
    _self_check()
