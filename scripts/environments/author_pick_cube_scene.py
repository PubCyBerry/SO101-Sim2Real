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

Coordinates are shifted to match the SO-101 follower init_state.pos = (2.2,
-0.61, 0.7299). The desk top right-front corner lands at the robot base.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from pxr import Sdf


SCENE_DIR = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "cube_desk"
SCENE_USD_PATH = SCENE_DIR / "scene.usd"
SCENE_USDA_PATH = SCENE_DIR / "scene.usda"
OBJECTS_DIR = SCENE_DIR / "objects"

# Shift the whole scene so the robot base sits at the front-edge clamp of the
# desk (not at the center). With offset.y = -0.55 the desk front edge lands at
# world y ≈ -0.64, putting the robot mount on the edge just like the real
# clamp setup. offset.z = 0.76 places the desk leg bottoms on the Isaac ground
# plane (z=0) and keeps the tabletop at a plausible real desk height.
SCENE_OFFSET: tuple[float, float, float] = (2.2, -0.57, 0.76)


MATERIALS = {
    "DeskWood": ((0.67, 0.51, 0.32), 0.78, 0.0),
    "DeskMat": ((0.025, 0.026, 0.032), 0.93, 0.0),
    "GrayFoam": ((0.45, 0.46, 0.47), 0.92, 0.0),
    "BowlBlue": ((0.72, 0.82, 0.90), 0.45, 0.0),
    "Ceiling": ((0.88, 0.86, 0.82), 0.95, 0.0),
}

# 큐브 4개 scene-local 배치 (매트 중앙).
# z = 매트윗면 0.006 + 큐브 반높이 0.0125 + slack 0.001 = 0.0195
CUBES = (
    ("Cube1", (-0.15, 0.22, 0.0195), 25.0),
    ("Cube2", (0.15, 0.22, 0.0195), -30.0),
    ("Cube3", (0.05, 0.26, 0.0195), 60.0),
    ("Cube4", (-0.05, 0.26, 0.0195), -10.0),
)

# 그릇 scene-local: (0.0, 0.40, 0.006) — 펜컵 위치와 동일(전방 호 정점)
BOWL_LOCAL: tuple[float, float, float] = (0.0, 0.40, 0.006)

# 큐브와 그릇 물리 상수
# mass: 35 g — 폼보다 무겁게 잡아 grasp 가 안정적이면서 SO-101 gripper 토크(1.5 Nm)
#   안에서 충분히 들어올려진다. 6 g 처럼 너무 가벼우면 빠른 가속 시 contact 가 끊겨
#   잘 잡아도 떨어진다.
CUBE_MASS: float = 0.035  # kg
# contactOffset: 그리퍼가 빠르게 접근할 때 한 step 안에 큐브를 관통하는 문제를 막기
#   위해 contact 를 더 일찍(4 mm) 생성한다. 0.0015 는 빠른 동작에서 관통이 발생.
CUBE_CONTACT_OFFSET = 0.004
CUBE_COLLISION_MARGIN = 0.001

BOWL_MASS: float = 0.15  # kg, 동적 그릇
BOWL_CONTACT_OFFSET = 0.0015
BOWL_COLLISION_MARGIN = 0.001


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


def _collision_attrs(lines: list[str], level: int, *, contact_tuning: bool, enabled: bool = True) -> None:
    _block(lines, level, f"bool physics:collisionEnabled = {1 if enabled else 0}")
    if not contact_tuning:
        return
    _block(lines, level, f"float physxCollision:contactOffset = {_num(CUBE_CONTACT_OFFSET)}")
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
) -> None:
    _block(lines, level, f'def Cube "{name}"{_collision_api(level, contact_tuning=contact_tuning) if collision else ""}')
    _block(lines, level, "{")
    if not visible:
        _block(lines, level + 1, 'token visibility = "invisible"')
    if collision:
        _collision_attrs(lines, level + 1, contact_tuning=contact_tuning, enabled=collision_enabled)
    _block(lines, level + 1, "double size = 1")
    if material_path is not None:
        _material_binding(lines, level + 1, material_path)
    if physics_material_path is not None:
        _physics_material_binding(lines, level + 1, physics_material_path)
    _xform_ops(lines, level + 1, translate=translate, rotate_x=rotate_x, rotate_z=rotate_z, scale=scale)
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

    큐브는 2.5cm × 2.5cm × 2.5cm 정육면체 (scale=0.025, USD Cube size=1)
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
    _block(lines, 1, "float physxRigidBody:linearDamping = 0.2")
    _block(lines, 1, "float physxRigidBody:sleepThreshold = 0.0005")
    _block(lines, 1, "float physxRigidBody:stabilizationThreshold = 0.0005")
    # 관통이 생겨도 분리 속도를 1 m/s 로 제한해 큐브가 튕겨 날아가지 않게 한다.
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

    # 큐브: Box (scale=0.025 → 2.5cm × 2.5cm × 2.5cm)
    _cube(
        lines,
        1,
        "Box",
        translate=(0, 0, 0),
        scale=(0.025, 0.025, 0.025),
        material_path=gray_foam_path,
        collision=True,
        physics_material_path=cube_friction_path,
        contact_tuning=True,
    )
    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


def _bowl_panel(
    lines: list[str],
    level: int,
    index: int,
    *,
    center: tuple[float, float, float],
    tangent_deg: float,
    tilt_deg: float,
    size: tuple[float, float, float],
    material_path: str,
    physics_material_path: str,
) -> None:
    """곡면 bowl 벽 세그먼트 하나 — 바깥 Xform(원주 배치)에 경사 Cube 를 담는다.

    바깥 Xform 의 rotateZ 가 panel 을 원주 접선 방향으로 정렬하고, 그 local
    frame 에서 안쪽 Cube 의 rotateX(tilt) 가 길이축(local +z)을 바깥·위로 눕혀
    bowl 곡면을 만든다. 중첩 Xform 으로 회전 순서를 명확히 한다(바깥 rotateZ
    먼저 → 안쪽 rotateX 는 회전된 local x=접선 축 기준).
    """
    _block(lines, level, f'def Xform "Wall{index:03d}"')
    _block(lines, level, "{")
    _xform_ops(lines, level + 1, translate=center, rotate_z=tangent_deg)
    _cube(
        lines,
        level + 1,
        "Seg",
        translate=(0.0, 0.0, 0.0),
        scale=size,
        material_path=material_path,
        rotate_x=tilt_deg,
        collision=True,
        visible=True,
        physics_material_path=physics_material_path,
        contact_tuning=True,
    )
    _block(lines, level, "}")


def author_bowl_usda() -> str:
    """Author the Bowl USDA at origin with self-contained materials.

    Bowl은 동적 rigid body. 밑바닥(Cylinder) + 반구형 곡면 벽(여러 밴드의 경사
    panel 로 근사). 각 panel 은 visible + collision 을 겸하며, 위로 갈수록
    바깥으로 벌어져 사진의 곡면 그릇 형상을 만든다.
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
    _block(lines, 1, "float physxRigidBody:angularDamping = 4.0")
    _block(lines, 1, "float physxRigidBody:linearDamping = 0.6")
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
    _block(lines, 3, "float physics:staticFriction = 1.0")
    _block(lines, 3, "float physics:dynamicFriction = 0.8")
    _block(lines, 3, "float physics:restitution = 0")
    _block(lines, 3, 'uniform token physxMaterial:frictionCombineMode = "max"')
    _block(lines, 3, 'uniform token physxMaterial:restitutionCombineMode = "min"')
    _block(lines, 2, "}")
    _block(lines, 1, "}")

    bowl_blue_path = f"{looks_parent}/BowlBlue"
    bowl_friction_path = f"{looks_parent}/BowlFriction"

    # Bottom: 곡면 벽 최하단 반경과 이어지는 평평한 바닥 disk.
    _cylinder(
        lines,
        1,
        "Bottom",
        radius=0.037,
        height=0.012,
        material_path=bowl_blue_path,
        translate=(0, 0, 0.006),
        collision=True,
        physics_material_path=bowl_friction_path,
        contact_tuning=True,
    )

    # Walls: 반구형 곡면을 8개 밴드 × 24 panel 로 근사.
    #   r(t) = r_bottom + (r_top - r_bottom) * t^0.6  (바닥은 좁고 위로 벌어짐)
    #   각 밴드는 아래/위 레벨을 잇는 경사 panel 24개. tilt = 수직→바깥 경사각.
    #   panel 길이를 1.25배로 늘려 인접 밴드와 겹쳐 이음매를 없앤다.
    panel_count = 24
    bands = 8
    r_bottom = 0.035
    r_top = 0.065
    depth = 0.045
    z_base = 0.012  # Bottom disk 윗면

    def _profile_r(t: float) -> float:
        return r_bottom + (r_top - r_bottom) * (t ** 0.6)

    wall_index = 0
    for band in range(bands):
        t0 = band / bands
        t1 = (band + 1) / bands
        z0 = z_base + depth * t0
        z1 = z_base + depth * t1
        r0 = _profile_r(t0)
        r1 = _profile_r(t1)
        z_mid = 0.5 * (z0 + z1)
        r_mid = 0.5 * (r0 + r1)
        dr = r1 - r0
        dz = z1 - z0
        wall_len = math.hypot(dr, dz) * 1.25
        tilt_deg = math.degrees(math.atan2(dr, dz))  # 수직(0)→바깥 위로 벌어짐(+)
        width = (2.0 * math.pi * r_mid / panel_count) * 1.15
        for j in range(panel_count):
            angle = j * math.tau / panel_count
            _bowl_panel(
                lines,
                1,
                wall_index,
                center=(r_mid * math.cos(angle), r_mid * math.sin(angle), z_mid),
                tangent_deg=math.degrees(angle) + 90.0,
                tilt_deg=tilt_deg,
                size=(width, 0.004, wall_len),
                material_path=bowl_blue_path,
                physics_material_path=bowl_friction_path,
            )
            wall_index += 1

    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# scene.usd authoring (references the object USDs above)
# ---------------------------------------------------------------------------

def _scene_desk(lines: list[str]) -> None:
    desk_mat = "/Scene/Looks/DeskWood"
    mat_mat = "/Scene/Looks/DeskMat"

    _block(lines, 1, "# The desk top is shifted so its top face sits at z=0.76 and the legs touch z=0.")
    _cube(
        lines,
        1,
        "DeskTop",
        translate=_shift((0.0, 0.31, -0.02)),
        scale=(1.20, 0.78, 0.04),
        material_path=desk_mat,
        collision=True,
        contact_tuning=True,
    )
    for name, pos in (
        ("DeskLegBackLeft", (-0.52, 0.64, -0.40)),
        ("DeskLegBackRight", (0.52, 0.64, -0.40)),
        ("DeskLegFrontLeft", (-0.52, -0.02, -0.40)),
        ("DeskLegFrontRight", (0.52, -0.02, -0.40)),
    ):
        _cube(lines, 1, name, translate=_shift(pos), scale=(0.06, 0.06, 0.72), material_path=desk_mat)
    _cube(
        lines,
        1,
        "DeskMat",
        translate=_shift((-0.02, 0.35, 0.003)),
        scale=(1.04, 0.57, 0.006),
        material_path=mat_mat,
        collision=True,
        contact_tuning=True,
    )


def _scene_ceiling(lines: list[str]) -> None:
    _cube(
        lines,
        1,
        "Ceiling",
        translate=_shift((0.0, 0.31, 1.2)),
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
