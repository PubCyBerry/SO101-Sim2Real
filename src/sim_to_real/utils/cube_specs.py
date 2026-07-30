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


# 쉘(표면적 ∝ 변²) 비례 질량 기준점 — 40 mm = 35 g(실물 의자다리 커버 폼 실측).
_MASS_REF_SIZE, _MASS_REF_MASS = 0.040, 0.035


def mass_for_size(size: float) -> float:
    """한 변 ``size``(m) 큐브의 질량(kg) — 쉘 비례(∝ 변²).

    크기 DR 이 런타임에 큐브를 스케일할 때 질량도 같은 규칙으로 따라가야 grasp 물리가
    일관된다(부피 비례로 하면 25 mm 가 8.5 g 로 너무 가벼워 jaw 에 튕긴다).
    """
    return _MASS_REF_MASS * (size / _MASS_REF_SIZE) ** 2


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

# ── 크기 DR 사다리 (2026-07-29) ────────────────────────────────────────────────
# 런타임 큐브 크기 무작위화가 뽑는 이산 후보. **authored 크기(40 mm)가 상한**이라
# USD scale 은 항상 ≤1 이다 — 키우는 방향은 spawn z·이격·planner obstacle blob 이 모두
# 40 mm 기준으로 굳어 있어 별도 재조정이 필요하다(줄이는 쪽은 전부 안전측).
CUBE_SIZE_CHOICES: tuple[float, ...] = (0.025, 0.030, 0.035, 0.040)


def _self_check() -> None:
    assert max(CUBE_SIZE_CHOICES) <= MAX_CUBE_SIZE, "DR 상한이 authored 큐브보다 크다(scale>1)"
    assert abs(mass_for_size(0.040) - 0.035) < 1e-9
    assert abs(mass_for_size(0.050) - 0.0546875) < 1e-9  # CUBE_SPECS 55 g 반올림과 정합
    # 5 mm 등간격
    steps = {round(b - a, 6) for a, b in zip(CUBE_SIZE_CHOICES, CUBE_SIZE_CHOICES[1:])}
    assert steps == {0.005}, f"5 mm 간격 아님: {steps}"
    print(f"[cube_specs] OK sizes={CUBE_SIZE_CHOICES} "
          f"masses={[round(mass_for_size(s), 4) for s in CUBE_SIZE_CHOICES]}")


if __name__ == "__main__":
    _self_check()
