"""Author the SO-101 pen Pick-and-Place USD scene.

Layout follows the kitchen_with_orange pattern:

  assets/scenes/pen_desk/
  ├── scene.usd                         # references object USDs and places them
  ├── scene.usda
  └── objects/
      ├── PenWhite/PenWhite.usd  + .usda
      ├── PenGray/PenGray.usd    + .usda
      ├── PenBlack/PenBlack.usd  + .usda
      ├── PenBlue/PenBlue.usd    + .usda
      └── PenCup/PenCup.usd      + .usda

Coordinates are shifted to match the SO-101 follower init_state.pos = (2.2,
-0.61, 0.7299). The desk top right-front corner lands at the robot base.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from pxr import Sdf


SCENE_DIR = Path(__file__).resolve().parents[1] / "assets" / "scenes" / "pen_desk"
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
    "CupMetal": ((0.018, 0.019, 0.023), 0.34, 0.18),
    "CupMesh": ((0.031, 0.033, 0.038), 0.38, 0.22),
    "WhitePlastic": ((0.82, 0.80, 0.72), 0.41, 0.0),
    "OffWhitePlastic": ((0.93, 0.91, 0.82), 0.46, 0.0),
    "GrayPlastic": ((0.34, 0.35, 0.38), 0.39, 0.0),
    "BluePlastic": ((0.030, 0.19, 0.69), 0.33, 0.0),
    "BlackPlastic": ((0.016, 0.017, 0.021), 0.35, 0.0),
    "Steel": ((0.24, 0.25, 0.27), 0.20, 0.72),
    "Ceiling": ((0.88, 0.86, 0.82), 0.95, 0.0),
}

# Pen seed positions (scattered on the mat around the cup for Pick & Place).
# Coordinates are in scene-local frame; SCENE_OFFSET is applied when emitted
# into scene.usd. z = matTop(0.006) + collider half-thickness(0.0077) + 0.001
# of slack so the rigid body sits on the mat without interpenetrating.
PENS = (
    # Mid-mat cluster ("green ellipse" in docs/pics annotated 펜통_펜_배치_3.jpg).
    # Pens occupy the mat center while the cup lives further forward on the
    # ±30° arc, so the pen sampling region and the cup arc never overlap in xy.
    #
    # Layout (scene-local, mat extent x ∈ [-0.52, 0.50], y ∈ [0.065, 0.635]):
    #   - pen-cluster center ≈ (0, 0.24), the "green ellipse" centroid
    #   - 4 pens at the corners of a 0.30 × 0.04 rectangle inside the ellipse
    #   - pairwise yaw differs ≥ 35° so capsules cross instead of running
    #     parallel; closest end-to-end approach stays > 4 cm after the
    #     ellipse-jitter randomization (x_radius=0.05, y_radius=0.02).
    ("PenWhite", (-0.15, 0.22, 0.0147), 25.0, "OffWhitePlastic", "BlackPlastic"),
    ("PenGray", (0.15, 0.22, 0.0147), -30.0, "GrayPlastic", "BlackPlastic"),
    ("PenBlack", (0.05, 0.26, 0.0147), 60.0, "BlackPlastic", "Steel"),
    ("PenBlue", (-0.05, 0.26, 0.0147), -10.0, "BluePlastic", "BlackPlastic"),
)

# Pen cup default — apex of the forward-facing ±30° arc that the cup is
# sampled along ("orange arc" in docs/pics 펜통_펜_배치_3.jpg). Sitting at
# scene-local y=0.40 puts the cup ≈ 0.44 m forward of the robot base, right
# at the SO-101 reach perimeter. Lifting it deep into the mat also separates
# it in y from the pen cluster (y ≤ 0.28 after randomization) so pens can
# never spawn inside the cup like the failure case in docs/pics/펜통_펜_배치_1.jpg.
PEN_CUP_LOCAL: tuple[float, float, float] = (0.0, 0.40, 0.006)

# Pen cup as a dynamic rigid body (it can be nudged like a real cup).
PEN_CUP_MASS: float = 0.12  # kg, light steel mesh cup

PEN_CONTACT_OFFSET = 0.0015
PEN_TORSIONAL_PATCH_RADIUS = 0.004
# Visual barrel: Capsule(radius=0.0077, height=0.118). Each visual primitive
# (Barrel/Grip/BackPlug/Clip) carries its own analytic collider so the contact
# surface matches the rendered surface exactly — same approach the SO-101
# robot USD takes (visual meshes are reused as colliders).
PEN_BARREL_RADIUS = 0.0077
PEN_BARREL_HEIGHT = 0.118


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
    _block(lines, level, f"float physxCollision:contactOffset = {_num(PEN_CONTACT_OFFSET)}")
    _block(lines, level, "float physxCollision:restOffset = 0")
    _block(lines, level, f"float physxCollision:torsionalPatchRadius = {_num(PEN_TORSIONAL_PATCH_RADIUS)}")
    _block(lines, level, "float physxCollision:minTorsionalPatchRadius = 0.001")


def _cube(
    lines: list[str],
    level: int,
    name: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    material_path: str | None,
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
    _xform_ops(lines, level + 1, translate=translate, rotate_z=rotate_z, scale=scale)
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
    _xform_ops(lines, level + 1, translate=translate, rotate_y=rotate_y, rotate_z=rotate_z)
    _block(lines, level, "}")


def _capsule(
    lines: list[str],
    level: int,
    name: str,
    *,
    radius: float,
    height: float,
    material_path: str,
    axis: str = "Y",
    translate: tuple[float, float, float] | None = None,
    collision: bool = False,
    visible: bool = True,
    physics_material_path: str | None = None,
    contact_tuning: bool = False,
    collision_enabled: bool = True,
) -> None:
    _block(
        lines,
        level,
        f'def Capsule "{name}"{_collision_api(level, contact_tuning=contact_tuning) if collision else ""}',
    )
    _block(lines, level, "{")
    _block(lines, level + 1, f'token axis = "{axis}"')
    if not visible:
        _block(lines, level + 1, 'token visibility = "invisible"')
    _block(lines, level + 1, f"double height = {_num(height)}")
    if collision:
        _collision_attrs(lines, level + 1, contact_tuning=contact_tuning, enabled=collision_enabled)
    _block(lines, level + 1, f"double radius = {_num(radius)}")
    _material_binding(lines, level + 1, material_path)
    if physics_material_path is not None:
        _physics_material_binding(lines, level + 1, physics_material_path)
    _xform_ops(lines, level + 1, translate=translate)
    _block(lines, level, "}")


def _cone(
    lines: list[str],
    level: int,
    name: str,
    *,
    radius: float,
    height: float,
    material_path: str,
    translate: tuple[float, float, float],
    rotate_x: float | None = None,
) -> None:
    _block(lines, level, f'def Cone "{name}"')
    _block(lines, level, "{")
    _block(lines, level + 1, 'token axis = "Y"')
    _block(lines, level + 1, f"double height = {_num(height)}")
    _block(lines, level + 1, f"double radius = {_num(radius)}")
    _material_binding(lines, level + 1, material_path)
    _xform_ops(lines, level + 1, translate=translate, rotate_x=rotate_x)
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

def author_pen_usda(name: str, barrel: str, trim: str) -> str:
    """Author a pen USDA at origin with self-contained materials.

    Material lookups inside the file use </name/Looks/...> so the file can be
    referenced from any scene without dangling absolute paths.
    """
    lines = _object_header(name)
    _block(lines, 0, f'def Xform "{name}" (')
    _block(lines, 1, 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]')
    _block(lines, 0, ")")
    _block(lines, 0, "{")
    _block(lines, 1, "float physics:mass = 0.02")
    _block(lines, 1, "bool physics:kinematicEnabled = 0")
    _block(lines, 1, "bool physics:rigidBodyEnabled = 1")
    _block(lines, 1, "bool physxRigidBody:disableGravity = 0")
    _block(lines, 1, "bool physxRigidBody:enableCCD = 1")
    _block(lines, 1, "float physxRigidBody:angularDamping = 100.0")
    _block(lines, 1, "float physxRigidBody:linearDamping = 5.0")
    _block(lines, 1, "float physxRigidBody:sleepThreshold = 0.05")
    _block(lines, 1, "float physxRigidBody:stabilizationThreshold = 0.05")
    _block(lines, 1, "int physxRigidBody:solverPositionIterationCount = 16")
    _block(lines, 1, "int physxRigidBody:solverVelocityIterationCount = 4")

    # Local Looks scope (absolute paths used in bindings below).
    looks_parent = f"/{name}/Looks"
    _block(lines, 1, 'def Scope "Looks"')
    _block(lines, 1, "{")
    used = {barrel, trim, "Steel"}
    for mat_name in sorted(used):
        color, roughness, metallic = MATERIALS[mat_name]
        _material(lines, 2, mat_name, looks_parent, color, roughness, metallic)
    _physics_material(lines, 2)
    _block(lines, 1, "}")

    barrel_path = f"{looks_parent}/{barrel}"
    trim_path = f"{looks_parent}/{trim}"
    steel_path = f"{looks_parent}/Steel"
    grip_phys_path = f"{looks_parent}/PenGripPhysics"

    # Stable sim collider: visual primitives keep collision schemas for editing
    # visibility, but only the invisible rectangular CollisionBox participates
    # in physics. This matches the tracked assets used by TA.2/TC.4 smoke tests.
    _capsule(
        lines,
        1,
        "Barrel",
        radius=PEN_BARREL_RADIUS,
        height=PEN_BARREL_HEIGHT,
        material_path=barrel_path,
        collision=True,
        physics_material_path=grip_phys_path,
        contact_tuning=True,
        collision_enabled=False,
    )
    _cube(
        lines,
        1,
        "CollisionBox",
        translate=(0, 0, 0),
        scale=(0.0154, 0.118, 0.0154),
        material_path=None,
        collision=True,
        visible=False,
        physics_material_path=grip_phys_path,
        contact_tuning=True,
    )
    _cylinder(
        lines,
        1,
        "Grip",
        axis="Y",
        radius=0.0081,
        height=0.025,
        material_path=trim_path,
        translate=(0, 0.045, 0),
        collision=True,
        physics_material_path=grip_phys_path,
        contact_tuning=True,
        collision_enabled=False,
    )
    _cylinder(
        lines,
        1,
        "BackPlug",
        axis="Y",
        radius=0.0079,
        height=0.009,
        material_path=trim_path,
        translate=(0, -0.067, 0),
        collision=True,
        physics_material_path=grip_phys_path,
        contact_tuning=True,
        collision_enabled=False,
    )
    _cylinder(lines, 1, "AccentRing", axis="Y", radius=0.0083, height=0.003, material_path=steel_path, translate=(0, 0.031, 0))
    _cone(lines, 1, "TipSleeve", radius=0.0070, height=0.020, material_path=barrel_path, translate=(0, 0.072, 0))
    _cone(lines, 1, "Nib", radius=0.0022, height=0.007, material_path=steel_path, translate=(0, 0.083, 0))
    _cube(
        lines,
        1,
        "Clip",
        translate=(0.0065, -0.020, 0.0065),
        scale=(0.0020, 0.040, 0.0014),
        material_path=trim_path,
        collision=True,
        physics_material_path=grip_phys_path,
        contact_tuning=True,
        collision_enabled=False,
    )
    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


def _cup_collision_wall(lines: list[str], level: int, index: int, angle: float, radius: float, panel_length: float, mat_path: str, physics_mat_path: str | None = None) -> None:
    point = (radius * math.cos(angle), radius * math.sin(angle), 0.059)
    tangent_degrees = math.degrees(angle) + 90.0
    _cube(
        lines,
        level,
        f"Wall{index:03d}",
        translate=point,
        scale=(panel_length, 0.005, 0.11),
        material_path=mat_path,
        rotate_z=tangent_degrees,
        collision=True,
        visible=False,
        physics_material_path=physics_mat_path,
    )


def _cup_mesh_panel(lines: list[str], level: int, index: int, angle: float, radius: float, z: float, mesh_mat: str) -> None:
    _block(lines, level, f'def Xform "MeshPanel{index:03d}Z{int(round(z * 1000)):03d}"')
    _block(lines, level, "{")
    _xform_ops(
        lines,
        level + 1,
        translate=(radius * math.cos(angle), radius * math.sin(angle), z),
        rotate_z=math.degrees(angle) + 90.0,
    )
    _cylinder(lines, level + 1, "Rise", axis="X", radius=0.00115, height=0.034, material_path=mesh_mat, rotate_y=-58.0)
    _cylinder(lines, level + 1, "Fall", axis="X", radius=0.00115, height=0.034, material_path=mesh_mat, rotate_y=58.0)
    _block(lines, level, "}")


def author_cup_usda() -> str:
    """Author the PenCup USDA at origin with self-contained materials.

    PenCup is a rigid body so the robot can nudge it like a real cup. Damping
    is set high enough that small contacts do not send it sliding off the mat.
    """
    lines = _object_header("PenCup")
    _block(lines, 0, 'def Xform "PenCup" (')
    _block(lines, 1, 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]')
    _block(lines, 0, ")")
    _block(lines, 0, "{")
    _block(lines, 1, f"float physics:mass = {_num(PEN_CUP_MASS)}")
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
    looks_parent = "/PenCup/Looks"
    _block(lines, 1, 'def Scope "Looks"')
    _block(lines, 1, "{")
    for mat_name in ("CupMetal", "CupMesh"):
        color, roughness, metallic = MATERIALS[mat_name]
        _material(lines, 2, mat_name, looks_parent, color, roughness, metallic)
    # Friction so the cup doesn't slide on the desk mat from light contact.
    _physics_material(lines, 2, name="CupFriction")
    _block(lines, 1, "}")

    cup_metal = f"{looks_parent}/CupMetal"
    cup_mesh = f"{looks_parent}/CupMesh"
    cup_friction = f"{looks_parent}/CupFriction"

    panel_count = 24
    radius = 0.052
    panel_length = 2.0 * radius * math.sin(math.pi / panel_count) * 1.14

    _cylinder(
        lines,
        1,
        "Bottom",
        radius=0.056,
        height=0.008,
        material_path=cup_metal,
        translate=(0, 0, 0.004),
        collision=True,
        physics_material_path=cup_friction,
    )

    _block(lines, 1, "# Keep pen containment stable while the rendered wall remains perforated.")
    _block(lines, 1, 'def Xform "CollisionWalls"')
    _block(lines, 1, "{")
    _block(lines, 2, 'token visibility = "invisible"')
    for index in range(panel_count):
        _cup_collision_wall(
            lines, 2, index, index * math.tau / panel_count, radius, panel_length, cup_metal, cup_friction
        )
    _block(lines, 1, "}")

    _block(lines, 1, 'def Xform "WireMesh"')
    _block(lines, 1, "{")
    for index in range(panel_count):
        angle = index * math.tau / panel_count
        tangent_degrees = math.degrees(angle) + 90.0
        point = (radius * math.cos(angle), radius * math.sin(angle), 0.060)
        _cylinder(
            lines,
            2,
            f"Post{index:03d}",
            axis="Z",
            radius=0.00125,
            height=0.104,
            material_path=cup_mesh,
            translate=point,
        )
        for ring_index, ring_z in enumerate((0.010, 0.031, 0.052, 0.073, 0.094, 0.112)):
            _cylinder(
                lines,
                2,
                f"Ring{ring_index:02d}_{index:03d}",
                axis="X",
                radius=0.00135 if ring_index not in (0, 5) else 0.0022,
                height=panel_length * 1.04,
                material_path=cup_metal if ring_index in (0, 5) else cup_mesh,
                translate=(point[0], point[1], ring_z),
                rotate_z=tangent_degrees,
            )
        for z in (0.024, 0.046, 0.068, 0.090):
            _cup_mesh_panel(lines, 2, index, angle, radius + 0.0002, z, cup_mesh)
    _block(lines, 1, "}")
    _block(lines, 0, "}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# scene.usd authoring (references the object USDs above)
# ---------------------------------------------------------------------------

def _scene_desk(lines: list[str]) -> None:
    desk_mat = "/Scene/Looks/DeskWood"
    mat_mat = "/Scene/Looks/DeskMat"

    _block(lines, 1, "# The desk top is shifted so its top face sits at z=0.76 and the legs touch z=0.")
    # contact_tuning=True 로 PhysxCollisionAPI 를 붙여 contactOffset 을 펜과 동일한 0.0015m 로 맞춘다.
    # 디폴트 (0.02m) 인 채로 두면 매트(두께 6mm) 위에서 펜이 contact margin 안쪽에 박힌 채로 reset 되어
    # 첫 step 에서 강한 분리 impulse 가 발생하고, 빠른 그리퍼 접근 시 매트/책상을 한 step 안에 통과한다.
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
    _block(lines, 1, "# Generated by scripts/author_pick_pen_scene.py.")

    # Static-scene-only materials live here. Per-object materials (pens, cup)
    # are inside each referenced USD so each object remains self-contained.
    _block(lines, 1, 'def Scope "Looks"')
    _block(lines, 1, "{")
    for mat_name in ("DeskWood", "DeskMat", "Ceiling"):
        color, roughness, metallic = MATERIALS[mat_name]
        _material(lines, 2, mat_name, "/Scene/Looks", color, roughness, metallic)
    _block(lines, 1, "}")

    _scene_ceiling(lines)
    _scene_desk(lines)

    # PenCup is also a referenced asset so it can be reused / swapped easily.
    _scene_reference(
        lines,
        "PenCup",
        "./objects/PenCup/PenCup.usd",
        translate=_shift(PEN_CUP_LOCAL),
    )

    _block(lines, 1, "# Pens are seeded scattered on the mat for Pick-and-Place. Each pen is a")
    _block(lines, 1, "# referenced rigid-body asset so domain randomization can re-pose freely.")
    for name, pos, yaw, _, _ in PENS:
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
    # Per-pen files
    for name, _pos, _yaw, barrel, trim in PENS:
        _write_pair(OBJECTS_DIR / name / name, author_pen_usda(name, barrel, trim))
        print(f"[INFO]: Authored {OBJECTS_DIR / name / (name + '.usd')}")

    # Pen cup
    _write_pair(OBJECTS_DIR / "PenCup" / "PenCup", author_cup_usda())
    print(f"[INFO]: Authored {OBJECTS_DIR / 'PenCup' / 'PenCup.usd'}")

    # Top-level scene
    _write_pair(SCENE_DIR / "scene", author_scene_usda())
    print(f"[INFO]: Authored {SCENE_USD_PATH}")


if __name__ == "__main__":
    main()
