"""Author the SO-101 cube Pick-and-Place USD scene.

Layout follows the kitchen_with_orange pattern:

  assets/scenes/cube_desk/
  ├── scene.usd                         # references object USDs and places them
  ├── scene.usda
  └── objects/
      ├── Cube1/Cube1.usd       + .usda
      ├── Cube2/Cube2.usd       + .usda
      ├── Cube3/Cube3.usd       + .usda
      ├── Cube4/Cube4.usd       + .usda
      └── Bowl/Bowl.usd         + .usda

Coordinates are shifted to match the SO-101 follower init_state.pos = (1.84,
-0.555, 0.6749).
  SCENE_OFFSET.z = 다리(0.68) + 상판(0.025) = 0.705.
  SCENE_OFFSET.y = -0.52: 책상 앞 모서리(scene-local y=-0.09) = world y=-0.61.
  로봇 x = desk_left_edge(1.40) + 440mm = 1.84.
  로봇 y = -0.565: 책상 앞 모서리(-0.61)에서 10mm 뒤로 당겨 장착.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from pxr import Gf, Sdf


SCENE_DIR = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "cube_desk"
SCENE_USD_PATH = SCENE_DIR / "scene.usd"
SCENE_USDA_PATH = SCENE_DIR / "scene.usda"
OBJECTS_DIR = SCENE_DIR / "objects"

# offset.y = -0.52: 책상 앞 모서리(scene-local y=-0.09)가 world y=-0.61에
# 오도록 맞춰 로봇 베이스 마운트와 정확히 맞닿음.
# offset.z = 다리(0.68) + 상판(0.025) = 0.705: leg bottoms on Isaac ground(z=0).
SCENE_OFFSET: tuple[float, float, float] = (2.2, -0.52, 0.705)


MATERIALS = {
    "DeskWood": ((0.67, 0.51, 0.32), 0.78, 0.0),
    "DeskMat": ((0.025, 0.026, 0.032), 0.93, 0.0),
    "GrayFoam": ((0.45, 0.46, 0.47), 0.92, 0.0),
    "BowlBlue": ((0.65, 0.83, 0.96), 0.28, 0.0),
    "Ceiling": ((0.88, 0.86, 0.82), 0.95, 0.0),
}

# 큐브 4개 scene-local 배치 — 로봇팔(x≈-0.36) 아래 매트 앞쪽에 흩뿌림.
# 매트 x 범위 [-0.70, 0.16], y 범위 [0.00, 0.40].
# z = 매트 윗면(0.004) + 큐브 반높이 + slack 0.001
# 작은 큐브(40mm): 0.004 + 0.020 + 0.001 = 0.025
# 큰  큐브(50mm): 0.004 + 0.025 + 0.001 = 0.030
CUBES = (
    ("Cube1", (-0.50, 0.08, 0.025), 20.0),   # 작은 큐브 40mm
    ("Cube2", (-0.22, 0.06, 0.025), -35.0),  # 작은 큐브 40mm
    ("Cube3", (-0.46, 0.17, 0.030), 50.0),   # 큰  큐브 50mm
    ("Cube4", (-0.27, 0.14, 0.030), -20.0),  # 큰  큐브 50mm
)

# 그릇 scene-local (=바닥 중심): 매트 왼쪽 모서리(-0.70)에서 +x 120mm, 위쪽(0.40)에서 -y 140mm.
# 바닥 중심 x = -0.70 + 0.12 = -0.58 → 매트 왼쪽까지 r_top(0.075) 여백 = 45mm ≈ 40mm ✓.
# y = 0.40 - 0.14 = 0.26.
BOWL_LOCAL: tuple[float, float, float] = (-0.58, 0.26, 0.010)

# 큐브별 scale: Cube1/2 = 40mm 작은 큐브, Cube3/4 = 50mm 큰 큐브
CUBE_SCALES: dict[str, tuple[float, float, float]] = {
    "Cube1": (0.04, 0.04, 0.04),
    "Cube2": (0.04, 0.04, 0.04),
    "Cube3": (0.05, 0.05, 0.05),
    "Cube4": (0.05, 0.05, 0.05),
}

# 큐브와 그릇 물리 상수
# mass: 35 g — 폼보다 무겁게 잡아 grasp 가 안정적이면서 SO-101 gripper 토크
#   안에서 충분히 들어올려진다. 6 g 처럼 너무 가벼우면 빠른 가속 시 contact 가 끊겨
#   잘 잡아도 떨어진다.
CUBE_MASS: float = 0.035  # kg

# 정적·두꺼운 콜라이더 기본 contactOffset: 빠른 접근에서 한 step 관통을 막도록
#   contact 를 4 mm 일찍 생성한다. 책상/매트/그릇처럼 두꺼운 면에 적용.
CONTACT_OFFSET_DEFAULT = 0.004

# 큐브(grasp 대상) 전용 contactOffset. 4 mm 는 그리퍼 손가락 콜라이더 offset 과
#   합산되어 큐브가 표면에서 ~1 cm 떨어진 채 contact 가 trigger 되는 "거리 두고
#   잡힘"을 유발한다. 2 mm 로 줄여 실제 표면 근처에서 접촉이 생기게 한다
#   (큐브 CCD + solverPositionIterationCount 32 가 관통을 방어).
CUBE_CONTACT_OFFSET = 0.002

# 큐브 모서리 챔퍼(bevel) 크기: 3 mm.
# 시각적 라운딩을 위해 Box 시각 메시 대신 _bevel_mesh_visual() 로 생성된
# 26-face chamfered 메시를 사용하고, Box prim 은 충돌 전용(invisible) 으로 남긴다.
CUBE_BEVEL: float = 0.003

# 150 mm plastic bowl on a rubber desk mat. In early 4-cube scripted runs the
# bowl slid 10+ cm from light robot/cube contacts, so model it as a heavier
# high-friction container rather than a toy-light shell.
BOWL_MASS: float = 0.25  # kg, 약 250 g 플라스틱 그릇

# 그릇 곡면 프로파일 — 시각 Wall mesh 와 명시적 충돌 패널이 공유한다(드리프트 방지).
#   r(t) = r_bottom + (r_top - r_bottom) * t^0.2, z(t) = z_base + depth * t
# 그릇 위 지름 150mm(r_top=0.075), 바닥 지름 65mm(r_bottom=0.0325), 벽 높이 58mm.
BOWL_R_BOTTOM: float = 0.0325
BOWL_R_TOP: float = 0.075
BOWL_Z_BASE: float = 0.012
BOWL_DEPTH: float = 0.058
BOWL_LATS: int = 20   # 시각 mesh 위도 밴드 수
BOWL_LONS: int = 24   # 시각 mesh 경도 패널 수

# 충돌 패널 해상도. 패널을 자오선 경사만큼 기울여(아래 _bowl_collision_walls) ledge 를
#   없애므로 시각 mesh 보다 성겨도 containment 가 안정적이다(구 480 패널 → 144).
BOWL_COLLISION_LATS: int = 6
BOWL_COLLISION_LONS: int = 24
BOWL_COLLISION_THICKNESS: float = 0.003  # 패널 반경 방향 두께 3mm


# ---------------------------------------------------------------------------
# Low-level emitters
# ---------------------------------------------------------------------------

def _num(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _tuple(values: Iterable[float]) -> str:
    return "(" + ", ".join(_num(float(value)) for value in values) + ")"


def _block(lines: list[str], level: int, text: str = "") -> None:
    lines.append(f"{'    ' * level}{text}".rstrip())


def _shift(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Apply SCENE_OFFSET to a top-level position."""
    return (pos[0] + SCENE_OFFSET[0], pos[1] + SCENE_OFFSET[1], pos[2] + SCENE_OFFSET[2])


def _xform_ops(
    lines: list[str],
    level: int,
    *,
    translate: tuple[float, float, float] | None = None,
    rotate_x: float | None = None,
    rotate_y: float | None = None,
    rotate_z: float | None = None,
    scale: tuple[float, float, float] | None = None,
) -> None:
    order: list[str] = []
    if translate is not None:
        _block(lines, level, f"double3 xformOp:translate = {_tuple(translate)}")
        order.append("xformOp:translate")
    if rotate_x is not None:
        _block(lines, level, f"float xformOp:rotateX = {_num(rotate_x)}")
        order.append("xformOp:rotateX")
    if rotate_y is not None:
        _block(lines, level, f"float xformOp:rotateY = {_num(rotate_y)}")
        order.append("xformOp:rotateY")
    if rotate_z is not None:
        _block(lines, level, f"float xformOp:rotateZ = {_num(rotate_z)}")
        order.append("xformOp:rotateZ")
    if scale is not None:
        _block(lines, level, f"float3 xformOp:scale = {_tuple(scale)}")
        order.append("xformOp:scale")
    if order:
        tokens = ", ".join(f'"{item}"' for item in order)
        _block(lines, level, f"uniform token[] xformOpOrder = [{tokens}]")


def _material_binding(lines: list[str], level: int, prim_path: str) -> None:
    """prim_path is a plain absolute path like '/Scene/Looks/DeskWood' (no angle brackets)."""
    _block(lines, level, f"rel material:binding = <{prim_path}>")


def _physics_material_binding(lines: list[str], level: int, prim_path: str) -> None:
    _block(lines, level, f"rel material:binding:physics = <{prim_path}>")


def _material(lines: list[str], level: int, name: str, parent_path: str, color: tuple[float, float, float], roughness: float, metallic: float) -> None:
    """parent_path is the absolute prim path of the enclosing Looks scope, e.g. '/Scene/Looks'."""
    _block(lines, level, f'def Material "{name}"')
    _block(lines, level, "{")
    _block(lines, level + 1, f"token outputs:surface.connect = <{parent_path}/{name}/Preview.outputs:surface>")
    _block(lines, level + 1, 'def Shader "Preview"')
    _block(lines, level + 1, "{")
    _block(lines, level + 2, 'uniform token info:id = "UsdPreviewSurface"')
    _block(lines, level + 2, f"color3f inputs:diffuseColor = {_tuple(color)}")
    _block(lines, level + 2, f"float inputs:metallic = {_num(metallic)}")
    _block(lines, level + 2, f"float inputs:roughness = {_num(roughness)}")
    _block(lines, level + 2, "token outputs:surface")
    _block(lines, level + 1, "}")
    _block(lines, level, "}")


def _physics_material(lines: list[str], level: int, name: str = "PenGripPhysics") -> None:
    _block(lines, level, f'def Material "{name}" (')
    _block(lines, level + 1, 'prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]')
    _block(lines, level, ")")
    _block(lines, level, "{")
    _block(lines, level + 1, "float physics:staticFriction = 1.8")
    _block(lines, level + 1, "float physics:dynamicFriction = 1.5")
    _block(lines, level + 1, "float physics:restitution = 0")
    _block(lines, level + 1, 'uniform token physxMaterial:frictionCombineMode = "max"')
    _block(lines, level + 1, 'uniform token physxMaterial:restitutionCombineMode = "min"')
    _block(lines, level, "}")


def _collision_api(level: int, *, contact_tuning: bool) -> str:
    schemas = ['"PhysicsCollisionAPI"']
    if contact_tuning:
        schemas.append('"PhysxCollisionAPI"')
    schemas_text = ", ".join(schemas)
    return ' (\n' + f"{'    ' * (level + 1)}prepend apiSchemas = [{schemas_text}]\n{'    ' * level})"


def _collision_attrs(
    lines: list[str],
    level: int,
    *,
    contact_tuning: bool,
    enabled: bool = True,
    contact_offset: float = CONTACT_OFFSET_DEFAULT,
) -> None:
    _block(lines, level, f"bool physics:collisionEnabled = {1 if enabled else 0}")
    if not contact_tuning:
        return
    _block(lines, level, f"float physxCollision:contactOffset = {_num(contact_offset)}")
    _block(lines, level, "float physxCollision:restOffset = 0")
    _block(lines, level, f"float physxCollision:torsionalPatchRadius = {_num(0.004)}")
    _block(lines, level, "float physxCollision:minTorsionalPatchRadius = 0.001")


def _cube(
    lines: list[str],
    level: int,
    name: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    material_path: str | None,
    rotate_x: float | None = None,
    rotate_z: float | None = None,
    collision: bool = False,
    visible: bool = True,
    physics_material_path: str | None = None,
    contact_tuning: bool = False,
    collision_enabled: bool = True,
    contact_offset: float = CONTACT_OFFSET_DEFAULT,
) -> None:
    _block(lines, level, f'def Cube "{name}"{_collision_api(level, contact_tuning=contact_tuning) if collision else ""}')
    _block(lines, level, "{")
    if not visible:
        _block(lines, level + 1, 'token visibility = "invisible"')
    if collision:
        _collision_attrs(
            lines,
            level + 1,
            contact_tuning=contact_tuning,
            enabled=collision_enabled,
            contact_offset=contact_offset,
        )
    _block(lines, level + 1, "double size = 1")
    if material_path is not None:
        _material_binding(lines, level + 1, material_path)
    if physics_material_path is not None:
        _physics_material_binding(lines, level + 1, physics_material_path)
    _xform_ops(lines, level + 1, translate=translate, rotate_x=rotate_x, rotate_z=rotate_z, scale=scale)
    _block(lines, level, "}")


def _bevel_mesh_visual(
    lines: list[str],
    level: int,
    name: str,
    *,
    sx: float,
    sy: float,
    sz: float,
    bevel: float,
    material_path: str | None,
) -> None:
    """시각 전용 26-face chamfered box 메시 (충돌 없음).

    각 모서리를 bevel 크기만큼 잘라낸다:
      8 main quads (각 면) + 12 edge bevels (모서리 띠) + 8 corner tris (꼭지점)
    모든 face normal 이 바깥을 향하도록 winding 검증 완료.
    doubleSided=1 로 설정해 렌더러 face-culling 문제를 방지한다.
    """
    ax, ay, az = sx / 2.0, sy / 2.0, sz / 2.0
    c = bevel

    # 24 vertices: corner i=(qx,qy,qz) → 3i=v_x, 3i+1=v_y, 3i+2=v_z
    corner_signs = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    pts: list[tuple[float, float, float]] = []
    for (qx, qy, qz) in corner_signs:
        pts.append((qx * ax, qy * (ay - c), qz * (az - c)))      # v_x
        pts.append((qx * (ax - c), qy * ay, qz * (az - c)))      # v_y
        pts.append((qx * (ax - c), qy * (ay - c), qz * az))      # v_z

    # 26-face connectivity (외향 normal, CCW winding 검증 완료)
    faces: list[list[int]] = [
        # 6 main quads
        [0, 3, 9, 6],      # +x
        [12, 18, 21, 15],  # -x
        [1, 13, 16, 4],    # +y
        [7, 10, 22, 19],   # -y
        [2, 8, 20, 14],    # +z
        [5, 17, 23, 11],   # -z
        # 4 edge bevels (z-parallel)
        [0, 3, 4, 1],      # +x+y
        [6, 7, 10, 9],     # +x-y
        [12, 13, 16, 15],  # -x+y
        [18, 21, 22, 19],  # -x-y
        # 4 edge bevels (y-parallel)
        [0, 2, 8, 6],      # +x+z
        [3, 9, 11, 5],     # +x-z
        [12, 18, 20, 14],  # -x+z
        [15, 17, 23, 21],  # -x-z
        # 4 edge bevels (x-parallel)
        [1, 13, 14, 2],    # +y+z
        [4, 5, 17, 16],    # +y-z
        [7, 8, 20, 19],    # -y+z
        [10, 22, 23, 11],  # -y-z
        # 8 corner triangles
        [0, 1, 2],         # +++
        [3, 5, 4],         # ++-
        [6, 8, 7],         # +-+
        [9, 10, 11],       # +--
        [12, 14, 13],      # -++
        [15, 16, 17],      # -+-
        [18, 19, 20],      # --+
        [21, 23, 22],      # ---
    ]

    counts = [len(f) for f in faces]
    indices = [i for f in faces for i in f]
    pts_str = ", ".join(f"({_num(x)}, {_num(y)}, {_num(z)})" for x, y, z in pts)

    _block(lines, level, f'def Mesh "{name}"')
    _block(lines, level, "{")
    _block(lines, level + 1, "uniform int doubleSided = 1")
    _block(lines, level + 1, f"int[] faceVertexCounts = [{', '.join(str(x) for x in counts)}]")
    _block(lines, level + 1, f"int[] faceVertexIndices = [{', '.join(str(x) for x in indices)}]")
    _block(lines, level + 1, f"point3f[] points = [{pts_str}]")
    _block(lines, level + 1, 'uniform token subdivisionScheme = "none"')
    if material_path is not None:
        _material_binding(lines, level + 1, material_path)
    _block(lines, level, "}")


def _cylinder(
    lines: list[str],
    level: int,
    name: str,
    *,
    radius: float,
    height: float,
    material_path: str,
    axis: str = "Z",
    translate: tuple[float, float, float] | None = None,
    rotate_y: float | None = None,
    rotate_z: float | None = None,
    collision: bool = False,
    visible: bool = True,
    physics_material_path: str | None = None,
    contact_tuning: bool = False,
    collision_enabled: bool = True,
) -> None:
    _block(
        lines,
        level,
        f'def Cylinder "{name}"{_collision_api(level, contact_tuning=contact_tuning) if collision else ""}',
    )
    _block(lines, level, "{")
    _block(lines, level + 1, f'token axis = "{axis}"')
    _block(lines, level + 1, f"double height = {_num(height)}")
    if collision:
        _collision_attrs(lines, level + 1, contact_tuning=contact_tuning, enabled=collision_enabled)
    _block(lines, level + 1, f"double radius = {_num(radius)}")
    _material_binding(lines, level + 1, material_path)
    if physics_material_path is not None:
        _physics_material_binding(lines, level + 1, physics_material_path)
    if not visible:
        _block(lines, level + 1, 'token visibility = "invisible"')
    _xform_ops(lines, level + 1, translate=translate, rotate_y=rotate_y, rotate_z=rotate_z)
    _block(lines, level, "}")


# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------

def _object_header(prim_name: str) -> list[str]:
    return [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{prim_name}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
    ]


def _scene_header() -> list[str]:
    return [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Scene"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
    ]


# ---------------------------------------------------------------------------
# Standalone object USDAs (per-prim files referenced by scene.usd)
# ---------------------------------------------------------------------------

def author_cube_usda(name: str) -> str:
    """Author a cube USDA at origin with self-contained materials.

    Cube1/2: 4cm × 4cm × 4cm (scale=0.04), Cube3/4: 5cm × 5cm × 5cm (scale=0.05).
    GrayFoam 머티리얼과 CubeFriction 물리 머티리얼 자체 포함.
    """
    lines = _object_header(name)
    _block(lines, 0, f'def Xform "{name}" (')
    _block(lines, 1, 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]')
    _block(lines, 0, ")")
    _block(lines, 0, "{")
    _block(lines, 1, f"float physics:mass = {_num(CUBE_MASS)}")
    _block(lines, 1, "bool physics:kinematicEnabled = 0")
    _block(lines, 1, "bool physics:rigidBodyEnabled = 1")
    _block(lines, 1, "bool physxRigidBody:disableGravity = 0")
    _block(lines, 1, "bool physxRigidBody:enableCCD = 1")
    # grasp 시 흔들림/회전 억제는 적당히 — 5.0 처럼 과한 damping 은 부자연스럽다.
    _block(lines, 1, "float physxRigidBody:angularDamping = 1.5")
    _block(lines, 1, "float physxRigidBody:linearDamping = 1.5")
    _block(lines, 1, "float physxRigidBody:sleepThreshold = 0.0005")
    _block(lines, 1, "float physxRigidBody:stabilizationThreshold = 0.0005")
    # 분리 속도 상한 1.0 m/s (Isaac Lab 표준 기본값). 0.3 은 튕김 억제엔 좋았으나
    # 그리퍼 손가락(convexDecomposition)이 큐브를 관통했을 때 분리가 너무 느려
    # "꽂힌 채 유지"되는 원인이었다. 1.0 은 관통을 빠르게 풀면서도 과한 사출은 막는다.
    _block(lines, 1, "float physxRigidBody:maxDepenetrationVelocity = 1.0")
    # grasp contact 안정: position iteration 을 높여 미끄러짐/관통을 줄인다.
    _block(lines, 1, "int physxRigidBody:solverPositionIterationCount = 32")
    _block(lines, 1, "int physxRigidBody:solverVelocityIterationCount = 8")

    # Local Looks scope.
    looks_parent = f"/{name}/Looks"
    _block(lines, 1, 'def Scope "Looks"')
    _block(lines, 1, "{")
    color, roughness, metallic = MATERIALS["GrayFoam"]
    _material(lines, 2, "GrayFoam", looks_parent, color, roughness, metallic)
    # 큐브용 friction 물리 머티리얼 — grasp 미끄러짐을 막도록 높은 마찰.
    # frictionCombineMode "max" 라 그리퍼 머티리얼 마찰이 낮아도 이 값이 적용된다.
    _block(lines, 2, 'def Material "CubeFriction" (')
    _block(lines, 3, 'prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]')
    _block(lines, 2, ")")
    _block(lines, 2, "{")
    _block(lines, 3, "float physics:staticFriction = 1.8")
    _block(lines, 3, "float physics:dynamicFriction = 1.5")
    _block(lines, 3, "float physics:restitution = 0")
    _block(lines, 3, 'uniform token physxMaterial:frictionCombineMode = "max"')
    _block(lines, 3, 'uniform token physxMaterial:restitutionCombineMode = "min"')
    _block(lines, 2, "}")
    _block(lines, 1, "}")

    gray_foam_path = f"{looks_parent}/GrayFoam"
    cube_friction_path = f"{looks_parent}/CubeFriction"

    # 시각 메시: 3mm 챔퍼 bevel 로 모서리·꼭지점 라운딩 (충돌 없음)
    sx, sy, sz = CUBE_SCALES[name]
    _bevel_mesh_visual(
        lines, 1, "Visual",
        sx=sx, sy=sy, sz=sz,
        bevel=CUBE_BEVEL,
        material_path=gray_foam_path,
    )

    # 충돌 전용 Box — invisible, physics material 적용
    _cube(
        lines,
        1,
        "Box",
        translate=(0, 0, 0),
        scale=CUBE_SCALES[name],
        material_path=None,
        visible=False,
        collision=True,
        physics_material_path=cube_friction_path,
        contact_tuning=True,
        contact_offset=CUBE_CONTACT_OFFSET,
    )
    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


def _bowl_wall_mesh(
    lines: list[str],
    level: int,
    *,
    r_bottom: float,
    r_top: float,
    z_base: float,
    depth: float,
    lats: int,
    lons: int,
    material_path: str,
    physics_material_path: str | None = None,
    collision: bool = False,
) -> None:
    """단일 회전체 Mesh prim — 기본은 시각 전용.

    lats×lons 격자 Mesh 1개로 그릇 벽을 렌더링한다.
    profile: r(t) = r_bottom + (r_top - r_bottom) * t^0.2 (U자 곡선).
    face winding: 바깥쪽 법선 기준 CCW (doubleSided=1 로 내부도 렌더링).

    충돌(collision)은 기본 False. 두께 0 열린 회전면에 convexDecomposition 을
    부여하면 PhysX 가 오목한 안쪽 캐비티를 convex hull 로 메워 충돌 바닥을
    실제보다 높이는 문제가 있어, 충돌은 _bowl_collision_walls() 의 명시적 box
    패널로 따로 만들고 이 Mesh 는 렌더링만 담당한다. collision=True 는 정적
    씬에서 convexDecomposition 겸용이 필요한 경우를 위해 남겨 둔다.
    """
    def _profile_r(t: float) -> float:
        return r_bottom + (r_top - r_bottom) * (t ** 0.2)

    pts: list[tuple[float, float, float]] = []
    for lat in range(lats + 1):
        t = lat / lats
        r = _profile_r(t)
        z = z_base + depth * t
        for lon in range(lons):
            angle = lon * math.tau / lons
            pts.append((r * math.cos(angle), r * math.sin(angle), z))

    # quad face winding (바깥 법선 CCW): bottom-lon, bottom-lon+1, top-lon+1, top-lon
    faces: list[list[int]] = []
    for lat in range(lats):
        for lon in range(lons):
            v0 = lat * lons + lon
            v1 = lat * lons + (lon + 1) % lons
            v2 = (lat + 1) * lons + (lon + 1) % lons
            v3 = (lat + 1) * lons + lon
            faces.append([v0, v1, v2, v3])

    indices = [i for f in faces for i in f]
    pts_str = ", ".join(f"({_num(x)}, {_num(y)}, {_num(z)})" for x, y, z in pts)

    if collision:
        schemas = '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI"'
        _block(lines, level, 'def Mesh "Wall" (')
        _block(lines, level + 1, f"prepend apiSchemas = [{schemas}]")
        _block(lines, level, ")")
    else:
        _block(lines, level, 'def Mesh "Wall"')
    _block(lines, level, "{")
    _block(lines, level + 1, "uniform int doubleSided = 1")
    _block(lines, level + 1, f"int[] faceVertexCounts = [{', '.join('4' for _ in faces)}]")
    _block(lines, level + 1, f"int[] faceVertexIndices = [{', '.join(str(i) for i in indices)}]")
    _block(lines, level + 1, f"point3f[] points = [{pts_str}]")
    _block(lines, level + 1, 'uniform token subdivisionScheme = "none"')
    if collision:
        _block(lines, level + 1, "bool physics:collisionEnabled = 1")
        _block(lines, level + 1, 'uniform token physics:approximation = "convexDecomposition"')
        _block(lines, level + 1, f"float physxCollision:contactOffset = {_num(CONTACT_OFFSET_DEFAULT)}")
        _block(lines, level + 1, "float physxCollision:restOffset = 0")
    _material_binding(lines, level + 1, material_path)
    if collision and physics_material_path is not None:
        _physics_material_binding(lines, level + 1, physics_material_path)
    _block(lines, level, "}")


def _oriented_box(
    lines: list[str],
    level: int,
    name: str,
    *,
    matrix: Gf.Matrix4d,
    physics_material_path: str | None = None,
    contact_offset: float = CONTACT_OFFSET_DEFAULT,
) -> None:
    """단위 Cube(size=1)에 baked 4×4 transform 을 부여한 invisible 충돌 전용 box.

    Euler op-order/행벡터 관례 모호성을 피하려고 회전·스케일·이동을 Gf.Matrix4d 로
    합성해 matrix4d xformOp:transform 하나로 출력한다.
    """
    _block(lines, level, f'def Cube "{name}"{_collision_api(level, contact_tuning=True)}')
    _block(lines, level, "{")
    _block(lines, level + 1, 'token visibility = "invisible"')
    _collision_attrs(lines, level + 1, contact_tuning=True, enabled=True, contact_offset=contact_offset)
    _block(lines, level + 1, "double size = 1")
    if physics_material_path is not None:
        _physics_material_binding(lines, level + 1, physics_material_path)
    rows = ", ".join(
        "(" + ", ".join(_num(matrix.GetRow(i)[j]) for j in range(4)) + ")"
        for i in range(4)
    )
    _block(lines, level + 1, f"matrix4d xformOp:transform = ({rows})")
    _block(lines, level + 1, 'uniform token[] xformOpOrder = ["xformOp:transform"]')
    _block(lines, level, "}")


def _bowl_collision_walls(
    lines: list[str],
    level: int,
    *,
    r_bottom: float,
    r_top: float,
    z_base: float,
    depth: float,
    c_lats: int,
    c_lons: int,
    physics_material_path: str,
    thickness: float = BOWL_COLLISION_THICKNESS,
) -> None:
    """그릇 안쪽 충돌을 명시적 box 패널 링으로 구성 (시각 Wall 과 동일 프로파일).

    convexDecomposition 이 오목 캐비티를 메워 충돌 바닥을 높이는 문제를 피한다.
    각 latitude band 를 자오선 경사각(alpha)만큼 기울인 invisible box 패널로 근사해
    안쪽 면이 연속 램프가 되도록 한다(연직 패널이면 band 경계마다 ledge 가 생겨
    큐브가 거기 얹힘). 패널은 baked 4×4 transform 으로 배치한다.
    """
    def _profile_r(t: float) -> float:
        return r_bottom + (r_top - r_bottom) * (t ** 0.2)

    _block(lines, level, 'def Xform "CollisionWalls"')
    _block(lines, level, "{")
    _block(lines, level + 1, 'token visibility = "invisible"')

    panel_index = 0
    for band in range(c_lats):
        t_lo = band / c_lats
        t_hi = (band + 1) / c_lats
        r_lo, r_hi = _profile_r(t_lo), _profile_r(t_hi)
        z_lo = z_base + depth * t_lo
        z_hi = z_base + depth * t_hi
        seg_len = math.hypot(r_hi - r_lo, z_hi - z_lo)
        # 연직(+Z)에서 바깥(+r)으로 기운 각. 바닥 band 는 거의 수평(넓은 플레어).
        alpha = math.atan2(r_hi - r_lo, z_hi - z_lo)
        r_mid = 0.5 * (r_lo + r_hi)
        z_mid = 0.5 * (z_lo + z_hi)
        # 충돌 안쪽 면이 시각 프로파일에 닿도록 중심을 법선 바깥으로 thickness/2 이동.
        r_ctr = r_mid + 0.5 * thickness * math.cos(alpha)
        chord = 2.0 * r_ctr * math.sin(math.pi / c_lons) * 1.10  # 인접 패널 틈 제거
        height = seg_len * 1.05  # 인접 band 와 살짝 겹쳐 자오선 틈 제거
        for lon in range(c_lons):
            phi = lon * math.tau / c_lons
            center = (r_ctr * math.cos(phi), r_ctr * math.sin(phi), z_mid)
            # row-vector 관례: p' = p · (S · Ry · Rz · T) → Scale 먼저, 그다음 Ry, Rz, Translate.
            scale = Gf.Matrix4d().SetScale(Gf.Vec3d(thickness, chord, height))
            rot_y = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(alpha)))
            rot_z = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(phi)))
            trans = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*center))
            matrix = scale * rot_y * rot_z * trans
            _oriented_box(
                lines,
                level + 1,
                f"Wall{panel_index:03d}",
                matrix=matrix,
                physics_material_path=physics_material_path,
            )
            panel_index += 1

    _block(lines, level, "}")


def author_bowl_usda() -> str:
    """Author the Bowl USDA at origin with self-contained materials.

    Bowl은 동적 rigid body. 구성:
      - Bottom: 평평한 바닥 disk (Cylinder 충돌).
      - Wall: 시각 전용 단일 회전체 Mesh (충돌 없음).
      - CollisionWalls: 자오선 경사를 따라 기울인 명시적 box 패널 링(안쪽 충돌).
    벽 충돌을 명시적 프리미티브로 두는 이유는 convexDecomposition 이 오목한
    그릇 안쪽을 메워 큐브가 바닥까지 가라앉지 못하게 하기 때문이다.
    """
    lines = _object_header("Bowl")
    _block(lines, 0, 'def Xform "Bowl" (')
    _block(lines, 1, 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]')
    _block(lines, 0, ")")
    _block(lines, 0, "{")
    _block(lines, 1, f"float physics:mass = {_num(BOWL_MASS)}")
    _block(lines, 1, "bool physics:kinematicEnabled = 0")
    _block(lines, 1, "bool physics:rigidBodyEnabled = 1")
    _block(lines, 1, "bool physxRigidBody:disableGravity = 0")
    _block(lines, 1, "bool physxRigidBody:enableCCD = 1")
    _block(lines, 1, "float physxRigidBody:angularDamping = 8.0")
    _block(lines, 1, "float physxRigidBody:linearDamping = 2.0")
    _block(lines, 1, "float physxRigidBody:sleepThreshold = 0.0005")
    _block(lines, 1, "float physxRigidBody:stabilizationThreshold = 0.0005")
    _block(lines, 1, "int physxRigidBody:solverPositionIterationCount = 16")
    _block(lines, 1, "int physxRigidBody:solverVelocityIterationCount = 4")

    # Local materials.
    looks_parent = "/Bowl/Looks"
    _block(lines, 1, 'def Scope "Looks"')
    _block(lines, 1, "{")
    color, roughness, metallic = MATERIALS["BowlBlue"]
    _material(lines, 2, "BowlBlue", looks_parent, color, roughness, metallic)
    # 그릇용 friction 물리 머티리얼
    _block(lines, 2, 'def Material "BowlFriction" (')
    _block(lines, 3, 'prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]')
    _block(lines, 2, ")")
    _block(lines, 2, "{")
    _block(lines, 3, "float physics:staticFriction = 1.8")
    _block(lines, 3, "float physics:dynamicFriction = 1.5")
    _block(lines, 3, "float physics:restitution = 0.3")
    _block(lines, 3, 'uniform token physxMaterial:frictionCombineMode = "max"')
    _block(lines, 3, 'uniform token physxMaterial:restitutionCombineMode = "min"')
    _block(lines, 2, "}")
    _block(lines, 1, "}")

    bowl_blue_path = f"{looks_parent}/BowlBlue"
    bowl_friction_path = f"{looks_parent}/BowlFriction"

    # Bottom: 곡면 벽 최하단 반경과 이어지는 평평한 바닥 disk.
    # 그릇 위 지름 150mm(r_top=0.075), 바닥 지름 65mm(r_bottom=0.0325).
    # 높이 = BOWL_Z_BASE 라 윗면(z=0.012)이 벽 최하단과 정확히 이어진다.
    _cylinder(
        lines,
        1,
        "Bottom",
        radius=BOWL_R_BOTTOM,
        height=BOWL_Z_BASE,
        material_path=bowl_blue_path,
        translate=(0, 0, 0.5 * BOWL_Z_BASE),
        collision=True,
        physics_material_path=bowl_friction_path,
        contact_tuning=True,
    )

    # Wall: 시각 전용 단일 회전체 Mesh (충돌 없음).
    _bowl_wall_mesh(
        lines,
        1,
        r_bottom=BOWL_R_BOTTOM,
        r_top=BOWL_R_TOP,
        z_base=BOWL_Z_BASE,
        depth=BOWL_DEPTH,
        lats=BOWL_LATS,
        lons=BOWL_LONS,
        material_path=bowl_blue_path,
        collision=False,
    )

    # CollisionWalls: 안쪽 충돌은 경사 따라 기울인 명시적 box 패널 링으로 구성.
    _bowl_collision_walls(
        lines,
        1,
        r_bottom=BOWL_R_BOTTOM,
        r_top=BOWL_R_TOP,
        z_base=BOWL_Z_BASE,
        depth=BOWL_DEPTH,
        c_lats=BOWL_COLLISION_LATS,
        c_lons=BOWL_COLLISION_LONS,
        physics_material_path=bowl_friction_path,
    )

    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# scene.usd authoring (references the object USDs above)
# ---------------------------------------------------------------------------

def _scene_desk(lines: list[str]) -> None:
    desk_mat = "/Scene/Looks/DeskWood"
    mat_mat = "/Scene/Looks/DeskMat"
    desk_phys = "/Scene/Looks/DeskFriction"

    # 상판: 1600×800×25mm. 윗면이 scene local z=0 (= world z=SCENE_OFFSET.z=0.705).
    # 상판 중심 z = -두께/2 = -0.0125.
    _block(lines, 1, "# 상판 윗면 = world z=0.705. 다리 바닥 = world z=0.")
    _cube(
        lines,
        1,
        "DeskTop",
        translate=_shift((0.0, 0.31, -0.0125)),
        scale=(1.60, 0.80, 0.025),
        material_path=desk_mat,
        collision=True,
        contact_tuning=True,
        physics_material_path=desk_phys,
    )
    # 다리: 25×25×680mm. 중심 z = -0.705 + 0.34 = -0.365 (scene local).
    for name, pos in (
        ("DeskLegBackLeft",  (-0.72, 0.64, -0.365)),
        ("DeskLegBackRight", (0.72, 0.64, -0.365)),
        ("DeskLegFrontLeft",  (-0.72, -0.02, -0.365)),
        ("DeskLegFrontRight", (0.72, -0.02, -0.365)),
    ):
        _cube(lines, 1, name, translate=_shift(pos), scale=(0.025, 0.025, 0.68), material_path=desk_mat)
    # 매트: 860×400×4mm.
    # 기준: 책상 앞-왼 모서리(scene-local -0.80, -0.09)에서 가로 100mm, 세로 90mm.
    # 매트 좌하단 모서리: (-0.80+0.10, -0.09+0.09) = (-0.70, 0.00)
    # 매트 중심: (-0.70+0.43, 0.00+0.20) = (-0.27, 0.20)
    _cube(
        lines,
        1,
        "DeskMat",
        translate=_shift((-0.27, 0.20, 0.002)),
        scale=(0.86, 0.40, 0.004),
        material_path=mat_mat,
        collision=True,
        contact_tuning=True,
        physics_material_path=desk_phys,
    )


def _scene_ceiling(lines: list[str]) -> None:
    # 천장 높이 지상 2500mm = world z 2.5.
    # scene local z = 2.5 - SCENE_OFFSET.z(0.705) = 1.795.
    _cube(
        lines,
        1,
        "Ceiling",
        translate=_shift((0.0, 0.31, 1.795)),
        scale=(5.0, 4.0, 0.05),
        material_path="/Scene/Looks/Ceiling",
    )


def _scene_reference(lines: list[str], name: str, payload_rel: str, *, translate: tuple[float, float, float], rotate_z: float | None = None) -> None:
    _block(lines, 1, f'def Xform "{name}" (')
    _block(lines, 2, f"prepend payload = @{payload_rel}@")
    _block(lines, 1, ")")
    _block(lines, 1, "{")
    _xform_ops(lines, 2, translate=translate, rotate_z=rotate_z)
    _block(lines, 1, "}")


def author_scene_usda() -> str:
    """Author scene.usd as a thin layout that references each object USD."""
    lines = _scene_header()
    _block(lines, 0, 'def Xform "Scene"')
    _block(lines, 0, "{")
    _block(lines, 1, "# Generated by scripts/author_pick_cube_scene.py.")

    # Static-scene-only materials live here. Per-object materials (cubes, bowl)
    # are inside each referenced USD so each object remains self-contained.
    _block(lines, 1, 'def Scope "Looks"')
    _block(lines, 1, "{")
    for mat_name in ("DeskWood", "DeskMat", "Ceiling"):
        color, roughness, metallic = MATERIALS[mat_name]
        _material(lines, 2, mat_name, "/Scene/Looks", color, roughness, metallic)
    # 책상 상판/매트용 friction 물리 머티리얼. 미지정 시 PhysX 기본(마찰 ~0.5)이라
    #   큐브가 면 위에서 미끄러지거나 그릇이 밀릴 수 있다. combine=max 라 상대편
    #   머티리얼 마찰이 낮아도 이 값이 적용된다(restitution 은 min → 0 유지).
    _block(lines, 2, 'def Material "DeskFriction" (')
    _block(lines, 3, 'prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]')
    _block(lines, 2, ")")
    _block(lines, 2, "{")
    _block(lines, 3, "float physics:staticFriction = 0.9")
    _block(lines, 3, "float physics:dynamicFriction = 0.8")
    _block(lines, 3, "float physics:restitution = 0")
    _block(lines, 3, 'uniform token physxMaterial:frictionCombineMode = "max"')
    _block(lines, 3, 'uniform token physxMaterial:restitutionCombineMode = "min"')
    _block(lines, 2, "}")
    _block(lines, 1, "}")

    _scene_ceiling(lines)
    _scene_desk(lines)

    # Bowl is also a referenced asset.
    _scene_reference(
        lines,
        "Bowl",
        "./objects/Bowl/Bowl.usd",
        translate=_shift(BOWL_LOCAL),
    )

    _block(lines, 1, "# 큐브 4개는 매트 위에 흩어져 배치됨.")
    for name, pos, yaw in CUBES:
        _scene_reference(
            lines,
            name,
            f"./objects/{name}/{name}.usd",
            translate=_shift(pos),
            rotate_z=yaw,
        )

    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _write_pair(path_no_ext: Path, text: str) -> None:
    """Write USDA text and produce the binary .usd companion via usd-core API.

    Both files share the same prim layout — .usda is the human-readable source
    of truth and .usd is the runtime-loaded binary (faster Isaac Sim load).
    """
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    usda_path = path_no_ext.with_suffix(".usda")
    usd_path = path_no_ext.with_suffix(".usd")
    usda_path.write_text(text, encoding="utf-8")
    layer = Sdf.Layer.FindOrOpen(str(usda_path))
    if layer is None:
        raise RuntimeError(f"Failed to open {usda_path} as a USD layer")
    # `format=usdc` is equivalent to `usdcat --usdFormat usdc`: writes the
    # binary crate header (`PXR-USDC`) regardless of the .usd extension.
    if not layer.Export(str(usd_path), args={"format": "usdc"}):
        raise RuntimeError(f"Failed to export binary USD: {usda_path} -> {usd_path}")


def main() -> None:
    # 큐브 4개 파일
    for name, _pos, _yaw in CUBES:
        _write_pair(OBJECTS_DIR / name / name, author_cube_usda(name))
        print(f"[INFO]: Authored {OBJECTS_DIR / name / (name + '.usd')}")

    # 그릇
    _write_pair(OBJECTS_DIR / "Bowl" / "Bowl", author_bowl_usda())
    print(f"[INFO]: Authored {OBJECTS_DIR / 'Bowl' / 'Bowl.usd'}")

    # Top-level scene
    _write_pair(SCENE_DIR / "scene", author_scene_usda())
    print(f"[INFO]: Authored {SCENE_USD_PATH}")


if __name__ == "__main__":
    main()
