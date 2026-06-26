"""환경 유틸 — leisaac ``utils/env_utils.py`` 에서 우리가 아직 보유하지 않은 함수만 vendor.

``dynamic_reset_gripper_effort_limit_sim`` / ``write_gripper_effort_limit_sim`` 는 이미
``sim_to_real.utils.gripper_effort`` 에 포팅돼 있으므로 여기서는 다루지 않는다(중복 금지).
"""

from __future__ import annotations


def get_task_type(task: str, task_type: str | None = None) -> str:
    """task id 에서 지원 teleop device 타입을 추론한다.

    명시적 ``task_type`` 이 주어지면 그대로 사용. 아니면 task 이름으로 판별.
    """
    if task_type is not None:
        return task_type
    if "BiArm" in task:
        return "bi-so101leader"
    elif "LeKiwi" in task:
        return "lekiwi-leader"
    else:
        return "so101leader"


def delete_attribute(obj, attr_name: str) -> None:
    """객체에 속성이 있으면 삭제(없으면 no-op)."""
    if hasattr(obj, attr_name):
        delattr(obj, attr_name)
