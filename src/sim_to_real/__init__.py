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

try:
    from .tasks import *
except ModuleNotFoundError as exc:
    # T0.2는 의존성 전환 단계다. Isaac 런타임/환경 재작성은 T0.3에서 검증한다.
    missing = exc.name or ""
    if missing != "leisaac" and not missing.startswith("omni."):
        raise
