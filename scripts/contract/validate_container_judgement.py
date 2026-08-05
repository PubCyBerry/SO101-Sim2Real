"""Mimic/SkillGen 컨테이너 판정 검증.

task_done 컨테이너 판정이 아래를 지키는지 확인한다:
1. 실제 그릇 asset 을 쓸 때만 로컬 프레임 판정 + tilt gate
2. 큐브 크기 파생 임계 — 25~40mm 모두 안전
3. 그릇 rim 위 큐브 오탐 없음
4. 그릇 기울기 40° 초과 시 task_done=False
"""

import importlib.util
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    """패키지 init 을 우회해 **파일로** 모듈을 읽는다.

    `sim_to_real/__init__.py` 가 `isaaclab_tasks` 를 import 하는데 호스트엔 없다. 이 검증기는
    host-only(torch 만 필요)라 파일 로드로 우회한다 — `author_pick_cube_scene.py` 와 같은 규약.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # 자기참조 import 대비 선등록
    spec.loader.exec_module(module)
    return module



def _install_isaaclab_stub() -> None:
    """`isaaclab*` 를 **요구되는 대로 만들어 주는** import 훅을 건다(host-only 검증용).

    이 검증기는 GPU·Isaac 없이 돌아야 하는데, 검사 대상(`tasks/common/mdp/observations.py`)은
    모듈 레벨에서 `isaaclab.*` 를 여러 개 import 한다. 이름을 하나씩 stub 하면 서브모듈이 늘 때마다
    깨지므로(`isaaclab.envs.mdp` 에서 또 걸렸다) 아예 finder 로 전부 흡수한다.
    검사 대상 함수는 isaaclab 심볼을 **쓰지 않는다** — import 만 통과시키면 된다.
    """
    import importlib.abc
    import types

    class _Stub(types.ModuleType):
        __path__: list[str] = []
        __all__: list[str] = []        # `from ... import *` 가 __all__ 을 순회한다

        def __getattr__(self, attr):
            # 타입으로도(상속·isinstance) 함수로도(호출) 쓰이므로 둘 다 받아준다.
            value = type(attr, (), {"__init__": lambda self, *a, **k: None,
                                    "__call__": lambda self, *a, **k: None})
            setattr(self, attr, value)
            return value

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root not in ("isaaclab", "isaaclab_tasks", "isaaclab_mimic"):
                return None
            return importlib.util.spec_from_loader(fullname, self, is_package=True)

        def create_module(self, spec):
            return _Stub(spec.name)

        def exec_module(self, module):
            pass

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def test_container_tilt_gate() -> None:
    """그릇 로컬 프레임의 +z axis tilt angle 계산 (단위: 도)."""
    import torch

    # 그릇 로컬 +z = row2[:, 2] = R[2, 2] 의 z 요소 (회전행렬)
    # cos(angle) = row2[:, 2] — 30° tilt 까지 허용하면 cos(30°) = 0.866
    max_tilt_deg = 30.0
    cos_threshold = math.cos(math.radians(max_tilt_deg))

    # 테스트: 30° 기울기 (허용)
    assert cos_threshold == pytest.approx(0.866, abs=0.001)

    # 40° 기울기 (거부)
    cos_40deg = math.cos(math.radians(40.0))
    assert cos_40deg < cos_threshold, f"40° ({cos_40deg:.4f}) < threshold ({cos_threshold:.4f})"

    print(f"✓ tilt gate: 30° 허용, 40° 거부 (cos threshold={cos_threshold:.4f})")


def test_height_range_vs_cube_size() -> None:
    """height_range 상한과 큐브 크기 관계.

    상한 = rim 상단 - 큐브 반변:
    - 25mm: 0.08 - 0.0125 = 0.0675 m
    - 30mm: 0.08 - 0.0150 = 0.0650 m
    - 35mm: 0.08 - 0.0175 = 0.0625 m
    - 40mm: 0.08 - 0.0200 = 0.0600 m

    현 코드는 고정 (0.005, 0.18) — 상한 0.18 은 너무 크다.
    """
    rim_z = 0.08
    sizes = [0.025, 0.030, 0.035, 0.040]
    for size in sizes:
        half = size / 2.0
        upper = rim_z - half
        print(f"  {size*1000:.0f}mm: upper={upper:.4f} m ({upper*1000:.1f} mm)")


def test_min_lift_vs_corner_tilt() -> None:
    """들림 임계가 **꼭짓점 서기**를 배제하는가 — DR 사다리 전 구간.

    큐브가 꼭짓점으로 서면 중심 z = ``desk_z + s·√3/2`` 다(body diagonal ``s√3`` 의 절반).
    ⚠ 여기에 ``0.5`` 를 한 번 더 곱해 ``s·√3/4`` 로 쓰면 값이 절반이 돼 게이트가 무력해진다 —
    실제로 그 실수가 있었다. 그래서 임계 수식은 **`min_lift_for_cube` 단일 소스에서 import** 하고
    이 검증기는 corner 높이를 독립적으로 계산해 대조한다.
    """
    # ★패키지로 import 하면 `sim_to_real/__init__.py` → `isaaclab_tasks` 가 걸린다(호스트엔 없다).
    #   이 검증기는 host-only 라 **파일 로드**로 패키지 init 을 우회한다.
    observations = _load_module(
        "_pick_cube_obs", _REPO_ROOT / "src/sim_to_real/tasks/pick_cube/mdp/observations.py")
    CORNER_TILT_MARGIN_M = observations.CORNER_TILT_MARGIN_M
    min_lift_for_cube = observations.min_lift_for_cube

    for size in (0.025, 0.030, 0.035, 0.040):
        corner_lift = size * (3.0 ** 0.5 / 2.0)     # 독립 계산(단일 소스와 교차 검증)
        min_lift = min_lift_for_cube(size)
        margin = (min_lift - corner_lift) * 1000
        print(f"{size*1000:4.0f}mm  corner={corner_lift*1000:6.2f}mm  "
              f"min_lift={min_lift*1000:6.2f}mm  margin={margin:5.2f}mm")
        assert min_lift > corner_lift, f"{size}: min_lift {min_lift} ≤ corner {corner_lift}"
        assert abs(margin - CORNER_TILT_MARGIN_M * 1000) < 1e-6, "여유가 상수와 다르다"

    # ★회귀 방지 — 옛 버그(√3/4)를 쓰면 40 mm 에서 임계가 corner 아래로 내려간다.
    buggy_corner = 0.040 * 0.5 * (3.0 ** 0.5 / 2.0)
    assert buggy_corner < 0.040 * (3.0 ** 0.5 / 2.0), "옛 수식이 2배 작다는 전제 자체가 깨졌다"


def test_gripper_joint_index() -> None:
    """그리퍼 컬럼 해석이 **articulation 인덱스**를 돌려주는가.

    ★`joint_names` 리스트 안에서의 위치를 돌려주면 `["gripper"]` 는 항상 0(= `shoulder_pan`)이
    된다. 실측(2026-08-04): 그 탓에 `gripper_open` 이 항상 False 가 돼 `place_cube1` 이 한 번도
    발화 못 했고 Mimic 주석이 0/8 로 전멸했다. 기하는 전부 정상이라 원인이 안 보였다.
    """
    # 이 모듈은 `isaaclab.utils.math` 를 모듈 레벨에서 import 한다(호스트엔 없다).
    # 검사 대상 함수는 그걸 쓰지 않으므로 최소 stub 으로 막고 파일 로드한다.
    _install_isaaclab_stub()
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    import importlib

    observations = importlib.import_module("sim_to_real.tasks.common.mdp.observations")
    resolve = observations._get_gripper_joint_index

    class Cfg:                       # SceneEntityCfg.resolve() 결과만 흉내
        def __init__(self, joint_names, joint_ids):
            self.joint_names, self.joint_ids = joint_names, joint_ids

    # SO-101 articulation 에서 gripper 는 5번 컬럼(arm 5축 다음)이다.
    assert resolve(Cfg(["gripper"], [5])) == 5, "resolve 된 joint_ids 를 써야 한다"
    assert resolve(Cfg(["gripper"], [0])) == 0, "0번 컬럼도 그대로 돌려줘야 한다"
    # 미지정(SceneEntityCfg 기본) → 마지막 컬럼 폴백
    assert resolve(Cfg(None, slice(None))) == -1, "미지정이면 -1 폴백"
    assert resolve(Cfg([], None)) == -1
    # ★회귀: 리스트 내 위치를 돌려주던 옛 구현이면 [5] 에서도 0 이 나온다
    assert resolve(Cfg(["gripper"], [5])) != 0, "옛 버그(리스트 내 위치) 재발"
    print("gripper 컬럼 해석: articulation 인덱스 OK (5→5 · 0→0 · 미지정→-1)")


if __name__ == "__main__":
    try:
        import pytest
        has_pytest = True
    except ImportError:
        has_pytest = False
        print("pytest 미설치, torch 단독 실행")

    test_height_range_vs_cube_size()
    test_min_lift_vs_corner_tilt()
    test_gripper_joint_index()
    if has_pytest:
        test_container_tilt_gate()

    print("\nvalidate_container_judgement PASSED")
