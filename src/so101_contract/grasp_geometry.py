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


def _self_check() -> None:
    assert len(FIXED_INNER_CENTER) == 3
    assert all(isinstance(v, float) for v in FIXED_INNER_CENTER)
    # pad 은 tcp 아래(+z, approach 방향)에 있고 tip 은 그보다 더 아래여야 한다.
    assert 0.0 < FIXED_INNER_CENTER[2] < PAD_LOW_OFF, "pad center 는 tcp 와 jaw tip 사이"
    print("[grasp_geometry] OK "
          f"fixed_inner={FIXED_INNER_CENTER} pad_low_off={PAD_LOW_OFF}")


if __name__ == "__main__":
    _self_check()
