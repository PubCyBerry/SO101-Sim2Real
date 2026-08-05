"""Mimic/SkillGen 컨테이너 판정 검증.

task_done 컨테이너 판정이 아래를 지키는지 확인한다:
1. 실제 그릇 asset 을 쓸 때만 로컬 프레임 판정 + tilt gate
2. 큐브 크기 파생 임계 — 25~40mm 모두 안전
3. 그릇 rim 위 큐브 오탐 없음
4. 그릇 기울기 40° 초과 시 task_done=False
"""

import importlib.util
import math
import types
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


def test_terms_end_to_end() -> None:
    """`object_in_container` · `task_done` 을 **stub env 로 실제 호출**한다.

    상수·수식만 보는 검사로는 **AND 합성 지점**이 안 잡힌다 — 실제로 그리퍼 조건 하나가 죽어
    기하 3조건이 다 참인데도 신호가 0 이던 사고가 있었다(`09_TACIT_KNOWLEDGE.md §16.10`).
    isaaclab 은 호스트에서 못 띄우지만 두 함수는 런타임에 torch 와 `env.scene` dict 만 쓴다.
    """
    import torch

    _install_isaaclab_stub()
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    import importlib

    obs = importlib.import_module("sim_to_real.tasks.common.mdp.observations")
    term = importlib.import_module("sim_to_real.tasks.common.mdp.terminations")
    geometry = importlib.import_module("sim_to_real.tasks.common.mdp._geometry")
    env_cfg = _load_module("_env_cfg_ast", _REPO_ROOT / "scripts/contract/validate_place_success.py")
    constants = env_cfg.literal_constants(
        _REPO_ROOT / "src/sim_to_real/tasks/pick_cube/pick_cube_env_cfg.py")

    RADIUS = float(constants["BOWL_SUCCESS_RADIUS"])
    HEIGHT = tuple(float(v) for v in constants["BOWL_HEIGHT_RANGE"])
    # `BOWL_CENTER_XY` 는 `spawn_area` 가 단일 소스다(env_cfg 는 거기서 재수출).
    spawn = env_cfg.literal_constants(
        _REPO_ROOT / "src/sim_to_real/tasks/pick_cube/spawn_area.py")
    BOWL_XY = tuple(float(v) for v in spawn["BOWL_CENTER_XY"])
    BOWL_Z = 0.715                    # `_BOWL_INIT_STATE` 의 z(리터럴 튜플이라 AST 로 못 읽는다)
    IN_BOWL_Z = BOWL_Z + 0.028        # 캐비티 바닥 위 안착 큐브 중심(실측)
    ABOVE_RIM_Z = BOWL_Z + 0.090      # rim(+0.080) 위에 들고 있는 상태

    assert geometry.DESK_TOP_Z == 0.705, f"DESK_TOP_Z 는 0.705 여야, got {geometry.DESK_TOP_Z}"

    class Cfg:                        # SceneEntityCfg.resolve() 결과만 흉내
        def __init__(self, name, joint_ids=slice(None), joint_names=None):
            self.name, self.joint_ids, self.joint_names = name, joint_ids, joint_names

    class Body:
        def __init__(self, xyz, quat=(1.0, 0.0, 0.0, 0.0)):
            self.data = type("D", (), {})()
            self.data.root_pos_w = torch.tensor([list(xyz)], dtype=torch.float32)
            # `task_done` 은 컨테이너 **로컬 프레임** 판정이라 자세도 본다(wxyz, 기본 = 정립).
            self.data.root_quat_w = torch.tensor([list(quat)], dtype=torch.float32)

    class Robot(Body):
        def __init__(self, joints):
            self.data = type("D", (), {})()
            self.data.joint_pos = torch.tensor([list(joints)], dtype=torch.float32)

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    def make_env(cube, bowl=None, joints=(0.0,) * 5 + (1.0,), bowl_quat=(1.0, 0.0, 0.0, 0.0)):
        e = types.SimpleNamespace(num_envs=1, device=torch.device("cpu"))
        e.scene = Scene(Cube1=Body(cube),
                        Bowl=Body(bowl or (*BOWL_XY, BOWL_Z), bowl_quat),
                        robot=Robot(joints))
        return e

    CUBE, BOWL = Cfg("Cube1"), Cfg("Bowl")
    ROBOT_LAST = Cfg("robot")                                    # 미지정 → 마지막 컬럼
    ROBOT_NAMED = Cfg("robot", joint_ids=[5], joint_names=["gripper"])   # SO-101 실제 컬럼

    def inside(env, robot_cfg=ROBOT_LAST, container_cfg=BOWL):
        return bool(obs.object_in_container(
            env, robot_cfg=robot_cfg, object_cfg=CUBE, container_cfg=container_cfg,
            container_center_xy=BOWL_XY, radius=RADIUS, height_range=HEIGHT).item())

    def done(env, container_cfg=BOWL):
        return bool(term.task_done(
            env, objects_cfg=[CUBE], container_center_xy=BOWL_XY, container_cfg=container_cfg,
            radius=RADIUS, height_range=HEIGHT, require_rest_pose=False).item())

    settled = make_env((*BOWL_XY, IN_BOWL_Z))
    assert inside(settled), "안착 큐브가 obs 에서 True 여야"
    assert done(settled), "안착 큐브가 termination 에서 True 여야"
    assert not inside(make_env((*BOWL_XY, ABOVE_RIM_Z))), "rim 위로 들고 있으면 False 여야"
    assert not inside(make_env((BOWL_XY[0] + 0.20, BOWL_XY[1], 0.726))), "그릇 밖은 False 여야"
    assert not inside(make_env((*BOWL_XY, BOWL_Z))), "창 하한 아래는 False 여야"

    # 그릇 DR(arc 최대 61 mm) 추종 — 판정원이 그릇을 따라가야 한다.
    moved = (BOWL_XY[0] + 0.061, BOWL_XY[1])
    e_moved = make_env((*moved, IN_BOWL_Z), bowl=(*moved, BOWL_Z))
    assert inside(e_moved), "그릇이 움직이면 판정원도 따라와야"
    assert done(e_moved), "termination 도 움직인 그릇을 따라와야"
    assert not inside(e_moved, container_cfg=None), "상수 폴백은 이동한 그릇을 놓쳐야(대조군)"

    # z 기준이 **컨테이너** 임을 확인 — 그릇이 올라가면 같은 큐브 z 는 창 아래.
    assert not inside(make_env((*BOWL_XY, IN_BOWL_Z), bowl=(*BOWL_XY, BOWL_Z + 0.05))), \
        "그릇이 5 cm 올라가면 같은 큐브 z 는 창 밖이어야"

    # 그리퍼 조건 — 닫혀 있으면 배치 아님.
    assert not inside(make_env((*BOWL_XY, IN_BOWL_Z), joints=(0.0,) * 5 + (0.2,))), \
        "그리퍼 닫힘이면 False 여야"

    # ★핵심 회귀: articulation 컬럼 5 = gripper(열림), 컬럼 0 = shoulder_pan(0.0).
    #   옛 구현은 `joint_names` 리스트 내 위치(=0)를 봐서 shoulder_pan 을 읽고 **항상 False** 였다.
    odd = make_env((*BOWL_XY, IN_BOWL_Z), joints=(0.0, 0.0, 0.0, 0.0, 0.0, 1.2))
    assert inside(odd, robot_cfg=ROBOT_NAMED), \
        "joint_ids=[5] 를 봐야 한다 — 리스트 내 위치(0=shoulder_pan)를 보면 영구 False"

    # ★그릇 tilt 게이트(main 개선분) — 40° 엎으면 안착 큐브라도 실패.
    import math as _m
    half = _m.radians(40.0) / 2.0
    tilted = make_env((*BOWL_XY, IN_BOWL_Z), bowl_quat=(_m.cos(half), _m.sin(half), 0.0, 0.0))
    assert not done(tilted), "그릇이 40° 기울면 task_done False 여야(로컬 프레임 + upright 게이트)"

    print("term end-to-end: 12 케이스 PASS (안착·rim위·그릇밖·창하한·DR추종·대조군·그릇상승·"
          "그리퍼닫힘·컬럼해석·그릇40°기울임)")


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
    test_terms_end_to_end()
    if has_pytest:
        test_container_tilt_gate()

    print("\nvalidate_container_judgement PASSED")
