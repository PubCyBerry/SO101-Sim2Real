"""큐브 오브젝트 사양 — **단일 진실 소스(single source of truth)**.

큐브 한 변 길이·질량을 여기 한 곳에서만 정의한다. author USD 생성, env_cfg
관측(half-extent)·DR 이격, cuRobo demo/batch planner, ROS bridge eval 이 모두
이 모듈을 import 해 파생값(scale·half-extent·footprint 반경)을 가져간다.

배경(2026-06-25): 예전엔 같은 크기 상수가 5곳에 흩어져 있었고, 40/50mm 확대
(06-18) 때 obs ``object_half_extents`` 만 갱신 누락돼 정책이 큐브를 실제보다
작게(30/40) 관측하던 잠복 정합성 결함이 있었다. 이 모듈로 일원화해 drift 를
구조적으로 차단한다.

⚠ 자족 규칙: 이 파일은 **stdlib 만** 쓴다(상대 import 금지). author 스크립트가
``isaaclab.app.AppLauncher`` 부팅 **전에** importlib 파일 로드로 이 모듈만 직접
읽기 때문(패키지 ``sim_to_real/__init__`` 의 isaac 의존 side-effect 우회).
"""

from dataclasses import dataclass

_SQRT2 = 2.0 ** 0.5


@dataclass(frozen=True)
class CubeSpec:
    """큐브 1종의 물리·기하 사양. 크기/질량만 1차값, 나머지는 파생."""

    name: str
    size: float  # 한 변 길이 (m)
    mass: float  # kg

    @property
    def half_extent(self) -> float:
        """반높이 (m) — obs 크기 채널·spawn z 계산."""
        return self.size * 0.5

    @property
    def scale(self) -> tuple[float, float, float]:
        """USD xform scale (정육면체)."""
        return (self.size, self.size, self.size)

    @property
    def footprint_radius(self) -> float:
        """정사각 바닥 외접원 반경 ((s/2)·√2) — DR 이격·volume inset."""
        return self.size * 0.5 * _SQRT2


# ── 단일 진실 소스 — 큐브 크기/질량 변경은 여기 한 곳만 고친다 ──────────────
#   mass: 의자다리 커버 폼이라 부피 완전비례보다 가볍게, 쉘(표면적 ∝ 변²)비례.
#         40mm(Cube1/2): 35 g, 50mm(Cube3/4): 35×(50/40)²≈54.7 → 55 g.
CUBE_SPECS: dict[str, CubeSpec] = {
    "Cube1": CubeSpec("Cube1", 0.040, 0.035),
    "Cube2": CubeSpec("Cube2", 0.040, 0.035),
    "Cube3": CubeSpec("Cube3", 0.050, 0.055),
    "Cube4": CubeSpec("Cube4", 0.050, 0.055),
}

# 편의 파생 매핑 — consumer 가 기존 dict 형태를 그대로 쓰도록.
CUBE_SIZES: dict[str, float] = {n: s.size for n, s in CUBE_SPECS.items()}
CUBE_MASSES: dict[str, float] = {n: s.mass for n, s in CUBE_SPECS.items()}
CUBE_HALF_EXTENTS: dict[str, float] = {n: s.half_extent for n, s in CUBE_SPECS.items()}
MAX_CUBE_SIZE: float = max(s.size for s in CUBE_SPECS.values())
# 최대 큐브 footprint 반경 = volume inset (사각 spawn 영역 안쪽 마진).
MAX_CUBE_FOOTPRINT_RADIUS: float = max(s.footprint_radius for s in CUBE_SPECS.values())
