from pathlib import Path


def _extend_isaaclab_pip_namespace() -> None:
    """Isaac Lab pip 패키지의 core package 경로를 보강한다."""
    try:
        import isaaclab
    except Exception:
        return

    search_locations = getattr(isaaclab, "__path__", None)
    if search_locations is None:
        return

    for location in list(search_locations):
        candidate = Path(location) / "source" / "isaaclab" / "isaaclab"
        if candidate.exists() and str(candidate) not in search_locations:
            search_locations.append(str(candidate))


_extend_isaaclab_pip_namespace()

# IsaacLab 임시 패치(TerminationManager) 적용 — best-effort.
# isaacsim 앱 컨텍스트 밖(host schema 검증 등)에선 isaaclab.managers import 가 실패할 수
# 있으므로 어떤 예외든 삼키고 스킵한다(패치는 순수 최적화·버그수정이라 없어도 import 는 진행).
try:
    from .utils.monkey_patch import monkey_patch

    monkey_patch()
except Exception:
    pass

try:
    from .tasks import *
except ModuleNotFoundError as exc:
    # T0.3 이후: 외부 task wrapper 의존성 제거 완료.
    # isaaclab/isaacsim 없이 import 시(예: schema 검증 스크립트) omni.* 오류는 허용.
    missing = exc.name or ""
    if not missing.startswith("omni.") and not missing.startswith("isaacsim"):
        raise
