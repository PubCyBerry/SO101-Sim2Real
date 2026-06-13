"""Author the SO-101 cube Pick-and-Place USD scene (pxr/PhysX 스키마 API).

기존 문자열 조립 방식을 공식 USD 스키마 API 로 전면 재작성한 버전.
``UsdGeom`` / ``UsdPhysics`` / ``UsdLux`` / ``UsdShade`` / ``PhysxSchema`` 를 직접 사용하므로
스키마 오타·구조 오류가 author 시점에 검증된다.

PhysxSchema 는 isaacsim 번들의 pxr 플러그인이라 ``isaaclab.app.AppLauncher`` 로
isaacsim 을 headless 부팅한 뒤에만 import 된다. 따라서 ``main()`` 이 먼저 앱을
띄우고, 그 후 pxr 심볼을 모듈 전역에 주입한 다음 author 함수들을 호출한다.

실행:
    OMNI_KIT_ACCEPT_EULA=YES uv run --group isaac \
        python scripts/environments/author_pick_cube_scene.py

생성물 (kitchen_with_orange 패턴):
  assets/scenes/cube_desk/
  ├── scene.usd + .usda                 # 책상/조명 + 객체 5개 payload 참조
  └── objects/
      ├── Cube1..4/Cube{N}.usd + .usda  # 시각 bevel mesh + invisible 충돌 Box
      └── Bowl/Bowl.usd + .usda         # 시각 회전체 Mesh + watertight SDF 충돌 Mesh

좌표는 SO-101 follower init_state 와 맞춘 SCENE_OFFSET 으로 시프트한다.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# pxr / PhysxSchema 심볼은 isaacsim 부팅 후 main() 에서 주입된다(아래 _inject_pxr).
# 부팅 전에 import 하면 PhysxSchema 가 없어 ImportError 가 난다.
Usd = UsdGeom = UsdPhysics = UsdLux = UsdShade = Sdf = Gf = Vt = PhysxSchema = None  # type: ignore


SCENE_DIR = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "cube_desk"
SCENE_USD_PATH = SCENE_DIR / "scene.usd"
OBJECTS_DIR = SCENE_DIR / "objects"

# 로봇 base 를 world 원점(XY)에 두기 위한 offset (recenter delta=(-1.84,+0.565,0) 반영).
# offset.x = 0.36: 책상 중심(scene-local x=0)이 world x=0.36 → 로봇 발치 x=0 이 책상 위.
# offset.y = 0.045: 책상 앞 모서리(scene-local y=-0.09)가 world y=-0.045 에 오도록.
# offset.z = 다리(0.68) + 상판(0.025) = 0.705: leg bottoms on Isaac ground(z=0).
SCENE_OFFSET: tuple[float, float, float] = (0.36, 0.045, 0.705)

# (diffuseColor, roughness, metallic)
MATERIALS = {
    "DeskWood": ((0.67, 0.51, 0.32), 0.78, 0.0),
    "DeskMat": ((0.025, 0.026, 0.032), 0.93, 0.0),
    "GrayFoam": ((0.45, 0.46, 0.47), 0.92, 0.0),
    "BowlBlue": ((0.65, 0.83, 0.96), 0.28, 0.0),
    "Ceiling": ((0.88, 0.86, 0.82), 0.95, 0.0),
}

# 큐브 4개 scene-local 배치 (이름, translate, yaw°). 매트 앞쪽에 흩뿌림.
# z = 매트 윗면(scene-local 0.004) + 큐브 반높이 + slack(0.001).
#   30mm: 0.004 + 0.015 + 0.001 = 0.020,  40mm: 0.004 + 0.020 + 0.001 = 0.025
CUBES = (
    ("Cube1", (-0.50, 0.08, 0.020), 20.0),   # 작은 큐브 30mm
    ("Cube2", (-0.22, 0.06, 0.020), -35.0),  # 작은 큐브 30mm
    ("Cube3", (-0.46, 0.17, 0.025), 50.0),   # 큰  큐브 40mm
    ("Cube4", (-0.27, 0.14, 0.025), -20.0),  # 큰  큐브 40mm
)

# 그릇 scene-local (바닥 중심).
BOWL_LOCAL: tuple[float, float, float] = (-0.58, 0.26, 0.010)

CUBE_SCALES: dict[str, tuple[float, float, float]] = {
    "Cube1": (0.03, 0.03, 0.03),
    "Cube2": (0.03, 0.03, 0.03),
    "Cube3": (0.04, 0.04, 0.04),
    "Cube4": (0.04, 0.04, 0.04),
}

# ── 물리 상수 (docs/GRASP_PHYSICS.md 근거 — 임의 변경 금지) ──────────────────
# 큐브 질량 — 크기별 차등. 의자다리 커버 폼은 속이 약간 비어 부피 완전비례보다
#   가볍게, 쉘(표면적 ∝ 변²)비례로 잡는다. 30mm(Cube1/2): 20 g, 40mm(Cube3/4): 35 g.
CUBE_MASSES: dict[str, float] = {
    "Cube1": 0.020, "Cube2": 0.020,
    "Cube3": 0.035, "Cube4": 0.035,
}
CONTACT_OFFSET_DEFAULT = 0.004      # 정적·두꺼운 면(책상/매트/그릇)
CUBE_CONTACT_OFFSET = 0.002         # grasp 대상 큐브 전용
CUBE_BEVEL: float = 0.003           # 큐브 모서리 챔퍼 3mm (시각 전용)
BOWL_MASS: float = 0.25             # kg, 약 250 g 플라스틱 그릇

# 그릇 곡면 프로파일 — spherical cap(_profile_r). z(t) = z_base + depth * t.
#   외형 높이(바닥 외면~윗면) = z_base + depth = 0.070 = 7cm.
BOWL_R_BOTTOM: float = 0.0325       # 바닥 반경 (바닥 지름 65mm)
BOWL_R_TOP: float = 0.075           # 상단 반경 (위 지름 150mm)
BOWL_Z_BASE: float = 0.005          # 바닥(외면) 두께 5mm — 실물처럼 얇게
BOWL_DEPTH: float = 0.065           # 벽 높이 (z_base+depth=0.070=7cm 유지)
BOWL_LATS: int = 20                 # 시각 mesh 위도 밴드 수
BOWL_LONS: int = 24                 # 시각/충돌 mesh 경도 분할 수

# watertight 충돌 mesh 파라미터.
BOWL_WALL_THICKNESS: float = 0.004  # 벽 두께 4mm (외벽-내벽 간격)
BOWL_FLOOR_THICKNESS: float = 0.003 # 캐비티 바닥이 z_base 위로 올라오는 높이 3mm
# 그릇 충돌 = convexDecomposition. SDF triangle mesh 는 num_envs>1 에서 per-instance
#   cooking 비용·불안정(crash)으로 부적합(Isaac Lab RL 표준은 convex 계열). watertight
#   두께 shell 을 여러 convex hull 로 분해하되, shrinkWrap + 충분한 maxConvexHulls/
#   voxelResolution 으로 오목 캐비티를 보존해 큐브가 바닥까지 가라앉게 한다.
BOWL_MAX_CONVEX_HULLS: int = 64        # 캐비티 보존 위해 충분히 분해
BOWL_HULL_VERTEX_LIMIT: int = 64
BOWL_VOXEL_RESOLUTION: int = 500000    # 분해 정밀도(기본 500k)

# 물리 머티리얼 (static_friction, dynamic_friction, restitution).
# (static, dynamic, restitution). combineMode 는 _physics_material 인자로 지정.
#   큐브=average, 그릇=min, 매트/책상=max 조합으로 그릇 내부만 미끌게 하고
#   그릇-매트·큐브-매트 접촉(max 우세)은 마찰을 유지한다(밀림 방지 + 큐브 안정).
FRICTION_CUBE = (1.8, 1.5, 0.0)     # combine=average → 그릇(min)과 접촉 시 낮은 쪽
FRICTION_BOWL = (0.12, 0.10, 0.3)   # 미끌 — 매끈한 플라스틱 내부. combine=min
FRICTION_DESK = (0.9, 0.8, 0.0)     # combine=max (그릇·큐브 안착)


# ---------------------------------------------------------------------------
# 좌표 헬퍼
# ---------------------------------------------------------------------------

def _shift(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """top-level position 에 SCENE_OFFSET 적용."""
    return (pos[0] + SCENE_OFFSET[0], pos[1] + SCENE_OFFSET[1], pos[2] + SCENE_OFFSET[2])


def _profile_r(t: float, r_bottom: float, r_top: float, depth: float) -> float:
    """그릇 자오선 반경 프로파일 — spherical cap(반구형 사발 곡면).

    바닥(t=0)에서 r_bottom, 위(t=1)에서 r_top 을 지나는 원호. 바닥 근처는
    완만하고 위로 갈수록 가팔라져 실제 반구 사발 곡면을 재현한다(t^0.2 의
    '중화냄비형 급벽'을 대체). 구 중심 z_c·반지름 R 은 두 끝점으로 결정.
    """
    z_c = (r_top ** 2 + depth ** 2 - r_bottom ** 2) / (2.0 * depth)
    rr = (r_bottom ** 2 + z_c ** 2) - (depth * t - z_c) ** 2
    return math.sqrt(max(0.0, rr))


# ---------------------------------------------------------------------------
# Stage 생성 / export
# ---------------------------------------------------------------------------

def _new_stage(default_prim_name: str, usda_path: Path) -> "Usd.Stage":
    """루트 Xform + 메타데이터를 갖춘 새 USD stage 생성."""
    usda_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, f"/{default_prim_name}")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def _export_pair(stage: "Usd.Stage", path_no_ext: Path) -> None:
    """.usda(텍스트) 저장 + .usd(usdc 바이너리) export."""
    usda_path = path_no_ext.with_suffix(".usda")
    usd_path = path_no_ext.with_suffix(".usd")
    stage.GetRootLayer().Save()  # 이미 .usda 로 CreateNew 했으므로 텍스트 저장
    if not stage.GetRootLayer().Export(str(usd_path), args={"format": "usdc"}):
        raise RuntimeError(f"Failed to export binary USD: {usda_path} -> {usd_path}")
    print(f"[INFO]: Authored {usda_path}  +  {usd_path}")


# ---------------------------------------------------------------------------
# Transform / 머티리얼 / 물리 헬퍼
# ---------------------------------------------------------------------------

def _set_xform(
    xformable: "UsdGeom.Xformable",
    *,
    translate: tuple[float, float, float] | None = None,
    rotate_z: float | None = None,
    scale: tuple[float, float, float] | None = None,
) -> None:
    """translate → rotateZ → scale 순서로 xformOp 부여."""
    if translate is not None:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_z is not None:
        xformable.AddRotateZOp().Set(float(rotate_z))
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _visual_material(
    stage: "Usd.Stage",
    parent_path: str,
    name: str,
    color: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> str:
    """UsdPreviewSurface 시각 머티리얼 생성, prim path 반환."""
    mat_path = f"{parent_path}/{name}"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return mat_path


def _physics_material(
    stage: "Usd.Stage",
    parent_path: str,
    name: str,
    friction: tuple[float, float, float],
    *,
    friction_combine: str = "max",
) -> str:
    """PhysicsMaterialAPI + PhysxMaterialAPI 머티리얼 생성, prim path 반환.

    friction_combine: PhysX 마찰 결합 모드(우선순위 max>multiply>min>average).
      그릇 내부를 미끌게 하려면 그릇=min·큐브=average 로 두어, 매트(max)와의
      접촉은 max 가 이겨 밀림/큐브 안정이 보존되도록 한다.
    """
    static_friction, dynamic_friction, restitution = friction
    mat_path = f"{parent_path}/{name}"
    material = UsdShade.Material.Define(stage, mat_path)
    prim = material.GetPrim()
    phys = UsdPhysics.MaterialAPI.Apply(prim)
    phys.CreateStaticFrictionAttr().Set(static_friction)
    phys.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys.CreateRestitutionAttr().Set(restitution)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    physx.CreateFrictionCombineModeAttr().Set(friction_combine)
    physx.CreateRestitutionCombineModeAttr().Set("min")
    return mat_path


def _bind_visual(prim: "Usd.Prim", mat_path: str) -> None:
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(UsdShade.Material.Get(prim.GetStage(), mat_path))


def _bind_physics(prim: "Usd.Prim", mat_path: str) -> None:
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        UsdShade.Material.Get(prim.GetStage(), mat_path),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def _apply_rigid_body(
    prim: "Usd.Prim",
    *,
    mass: float,
    angular_damping: float,
    linear_damping: float,
    solver_position_iterations: int,
    solver_velocity_iterations: int,
    max_depenetration_velocity: float = 1.0,
    enable_ccd: bool = True,
    sleep_threshold: float = 0.0005,
    stabilization_threshold: float = 0.0005,
) -> None:
    """PhysicsRigidBodyAPI + MassAPI + PhysxRigidBodyAPI 일괄 적용."""
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    rb.CreateKinematicEnabledAttr().Set(False)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)

    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateDisableGravityAttr().Set(False)
    physx.CreateEnableCCDAttr().Set(enable_ccd)
    physx.CreateAngularDampingAttr().Set(angular_damping)
    physx.CreateLinearDampingAttr().Set(linear_damping)
    physx.CreateSleepThresholdAttr().Set(sleep_threshold)
    physx.CreateStabilizationThresholdAttr().Set(stabilization_threshold)
    physx.CreateMaxDepenetrationVelocityAttr().Set(max_depenetration_velocity)
    physx.CreateSolverPositionIterationCountAttr().Set(solver_position_iterations)
    physx.CreateSolverVelocityIterationCountAttr().Set(solver_velocity_iterations)


def _apply_collision(
    prim: "Usd.Prim",
    *,
    contact_tuning: bool = True,
    contact_offset: float = CONTACT_OFFSET_DEFAULT,
    rest_offset: float = 0.0,
) -> None:
    """PhysicsCollisionAPI (+ contact tuning 시 PhysxCollisionAPI) 적용."""
    col = UsdPhysics.CollisionAPI.Apply(prim)
    col.CreateCollisionEnabledAttr().Set(True)
    if not contact_tuning:
        return
    physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx.CreateContactOffsetAttr().Set(contact_offset)
    physx.CreateRestOffsetAttr().Set(rest_offset)
    physx.CreateTorsionalPatchRadiusAttr().Set(0.004)
    physx.CreateMinTorsionalPatchRadiusAttr().Set(0.001)


def _set_mesh(
    mesh: "UsdGeom.Mesh",
    points: list[tuple[float, float, float]],
    faces: list[list[int]],
    *,
    double_sided: bool = False,
) -> None:
    """Mesh prim 에 points / faceVertexCounts / faceVertexIndices / extent 설정."""
    counts = [len(f) for f in faces]
    indices = [i for f in faces for i in f]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if double_sided:
        mesh.CreateDoubleSidedAttr().Set(True)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mesh.CreateExtentAttr(
        [Gf.Vec3f(min(xs), min(ys), min(zs)), Gf.Vec3f(max(xs), max(ys), max(zs))]
    )


# ---------------------------------------------------------------------------
# Mesh 기하 생성 (순수 함수)
# ---------------------------------------------------------------------------

def _bevel_box_geometry(
    sx: float, sy: float, sz: float, bevel: float
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """26-face chamfered box (8 main quads + 12 edge bevels + 8 corner tris).

    모든 face normal 이 바깥을 향하도록 winding 검증 완료. 정점은 실제 크기.
    """
    ax, ay, az = sx / 2.0, sy / 2.0, sz / 2.0
    c = bevel
    corner_signs = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    pts: list[tuple[float, float, float]] = []
    for (qx, qy, qz) in corner_signs:
        pts.append((qx * ax, qy * (ay - c), qz * (az - c)))      # v_x
        pts.append((qx * (ax - c), qy * ay, qz * (az - c)))      # v_y
        pts.append((qx * (ax - c), qy * (ay - c), qz * az))      # v_z
    faces: list[list[int]] = [
        # 6 main quads
        [0, 3, 9, 6], [12, 18, 21, 15], [1, 13, 16, 4],
        [7, 10, 22, 19], [2, 8, 20, 14], [5, 17, 23, 11],
        # 4 edge bevels (z-parallel)
        [0, 3, 4, 1], [6, 7, 10, 9], [12, 13, 16, 15], [18, 21, 22, 19],
        # 4 edge bevels (y-parallel)
        [0, 2, 8, 6], [3, 9, 11, 5], [12, 18, 20, 14], [15, 17, 23, 21],
        # 4 edge bevels (x-parallel)
        [1, 13, 14, 2], [4, 5, 17, 16], [7, 8, 20, 19], [10, 22, 23, 11],
        # 8 corner triangles
        [0, 1, 2], [3, 5, 4], [6, 8, 7], [9, 10, 11],
        [12, 14, 13], [15, 16, 17], [18, 19, 20], [21, 23, 22],
    ]
    return pts, faces


def _bowl_wall_geometry(
    r_bottom: float, r_top: float, z_base: float, depth: float, lats: int, lons: int
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """시각 전용 단일 회전체 표면 (single surface, doubleSided 로 양면 렌더)."""
    pts: list[tuple[float, float, float]] = []
    for lat in range(lats + 1):
        t = lat / lats
        r = _profile_r(t, r_bottom, r_top, depth)
        z = z_base + depth * t
        for lon in range(lons):
            a = lon * math.tau / lons
            pts.append((r * math.cos(a), r * math.sin(a), z))
    faces: list[list[int]] = []
    for lat in range(lats):
        for lon in range(lons):
            v0 = lat * lons + lon
            v1 = lat * lons + (lon + 1) % lons
            v2 = (lat + 1) * lons + (lon + 1) % lons
            v3 = (lat + 1) * lons + lon
            faces.append([v0, v1, v2, v3])
    return pts, faces


def _bowl_collision_geometry(
    r_bottom: float,
    r_top: float,
    z_base: float,
    depth: float,
    lats: int,
    lons: int,
    wall_thickness: float,
    floor_thickness: float,
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """watertight(닫힌 manifold) 그릇 충돌 shell — SDF collision 용.

    외벽 + 내벽(캐비티 표면) + 상단 림 + 외곽 바닥 + 캐비티 바닥을 모두 닫아
    벽·바닥이 solid 가 되고 안쪽 빈 공간(캐비티)은 outside 가 된다. SDF 가
    오목한 그릇 내부를 정확히 표현해 큐브가 캐비티 바닥까지 가라앉는다.
    """
    pts: list[tuple[float, float, float]] = []

    # 외벽 ring: lat 0..lats, z = z_base + depth*t
    outer_base = 0
    for lat in range(lats + 1):
        t = lat / lats
        r = _profile_r(t, r_bottom, r_top, depth)
        z = z_base + depth * t
        for lon in range(lons):
            a = lon * math.tau / lons
            pts.append((r * math.cos(a), r * math.sin(a), z))

    # 내벽(캐비티) ring: 반경은 외벽 - thickness, 바닥은 floor_thickness 만큼 올림.
    inner_base = len(pts)
    cavity_z0 = z_base + floor_thickness
    cavity_depth = depth - floor_thickness  # 캐비티 top = z_base + depth (외벽 top 과 동일)
    for lat in range(lats + 1):
        t = lat / lats
        r = max(0.0005, _profile_r(t, r_bottom, r_top, depth) - wall_thickness)
        z = cavity_z0 + cavity_depth * t
        for lon in range(lons):
            a = lon * math.tau / lons
            pts.append((r * math.cos(a), r * math.sin(a), z))

    outer_center = len(pts); pts.append((0.0, 0.0, z_base))
    cavity_center = len(pts); pts.append((0.0, 0.0, cavity_z0))

    def oidx(lat: int, lon: int) -> int:
        return outer_base + lat * lons + (lon % lons)

    def iidx(lat: int, lon: int) -> int:
        return inner_base + lat * lons + (lon % lons)

    faces: list[list[int]] = []
    # 외벽 (바깥 법선)
    for lat in range(lats):
        for lon in range(lons):
            faces.append([oidx(lat, lon), oidx(lat, lon + 1), oidx(lat + 1, lon + 1), oidx(lat + 1, lon)])
    # 내벽 (캐비티 안쪽 = winding 반전)
    for lat in range(lats):
        for lon in range(lons):
            faces.append([iidx(lat, lon), iidx(lat + 1, lon), iidx(lat + 1, lon + 1), iidx(lat, lon + 1)])
    # 상단 림 (외벽 top ↔ 내벽 top, 위 법선)
    top = lats
    for lon in range(lons):
        faces.append([oidx(top, lon), iidx(top, lon), iidx(top, lon + 1), oidx(top, lon + 1)])
    # 외곽 바닥 (아래 법선)
    for lon in range(lons):
        faces.append([outer_center, oidx(0, lon + 1), oidx(0, lon)])
    # 캐비티 바닥 (위 법선)
    for lon in range(lons):
        faces.append([cavity_center, iidx(0, lon), iidx(0, lon + 1)])
    return pts, faces


# ---------------------------------------------------------------------------
# 객체 author 함수
# ---------------------------------------------------------------------------

def author_cube(name: str) -> "Usd.Stage":
    """큐브 1개 stage author — 시각 bevel mesh + invisible 충돌 Box."""
    stage = _new_stage(name, OBJECTS_DIR / name / f"{name}.usda")
    root_prim = stage.GetPrimAtPath(f"/{name}")

    _apply_rigid_body(
        root_prim,
        mass=CUBE_MASSES[name],
        angular_damping=1.5,
        linear_damping=1.5,
        solver_position_iterations=32,
        solver_velocity_iterations=8,
    )

    looks = f"/{name}/Looks"
    UsdGeom.Scope.Define(stage, looks)
    color, roughness, metallic = MATERIALS["GrayFoam"]
    gray_foam = _visual_material(stage, looks, "GrayFoam", color, roughness, metallic)
    # 물리 friction 머티리얼은 큐브 USD 에 두지 않는다 — PhysX 64K 머티리얼 한도 때문에
    # env 당 4개(큐브별) 복제를 막기 위해 scene.usd 가 단일 공유 CubeFriction 을 over-bind
    # 한다(값 동일, 인스턴스만 4→1). 16384 env 가능(8192×6=49K → 16384×3=49K).

    # 시각 메시: 3mm 챔퍼 bevel (충돌 없음).
    sx, sy, sz = CUBE_SCALES[name]
    visual = UsdGeom.Mesh.Define(stage, f"/{name}/Visual")
    pts, faces = _bevel_box_geometry(sx, sy, sz, CUBE_BEVEL)
    _set_mesh(visual, pts, faces, double_sided=True)
    _bind_visual(visual.GetPrim(), gray_foam)

    # 충돌 전용 Box: invisible 해석적 Cube(size=1) + scale. 완전 평면 grasp 면.
    box = UsdGeom.Cube.Define(stage, f"/{name}/Box")
    box.CreateSizeAttr(1.0)
    box.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    box.MakeInvisible()
    _set_xform(box, scale=CUBE_SCALES[name])
    _apply_collision(box.GetPrim(), contact_tuning=True, contact_offset=CUBE_CONTACT_OFFSET)
    # friction 머티리얼 바인딩은 scene.usd 가 공유 /Scene/Looks/CubeFriction 으로 over-bind.

    return stage


def author_bowl() -> "Usd.Stage":
    """그릇 stage author — 시각 회전체 Mesh + watertight SDF 충돌 Mesh."""
    stage = _new_stage("Bowl", OBJECTS_DIR / "Bowl" / "Bowl.usda")
    root_prim = stage.GetPrimAtPath("/Bowl")

    _apply_rigid_body(
        root_prim,
        mass=BOWL_MASS,
        angular_damping=8.0,
        linear_damping=2.0,
        solver_position_iterations=16,
        solver_velocity_iterations=4,
    )

    looks = "/Bowl/Looks"
    UsdGeom.Scope.Define(stage, looks)
    color, roughness, metallic = MATERIALS["BowlBlue"]
    bowl_blue = _visual_material(stage, looks, "BowlBlue", color, roughness, metallic)
    bowl_friction = _physics_material(stage, looks, "BowlFriction", FRICTION_BOWL, friction_combine="min")

    # 시각: 단일 회전체 표면 (얇은 벽 룩, 충돌 없음).
    wall = UsdGeom.Mesh.Define(stage, "/Bowl/Wall")
    pts, faces = _bowl_wall_geometry(
        BOWL_R_BOTTOM, BOWL_R_TOP, BOWL_Z_BASE, BOWL_DEPTH, BOWL_LATS, BOWL_LONS
    )
    _set_mesh(wall, pts, faces, double_sided=True)
    _bind_visual(wall.GetPrim(), bowl_blue)

    # 시각 바닥 disk — 벽 mesh 는 바닥이 뚫려 있으므로 Cylinder 로 막는다(충돌 없음).
    #   벽이 z_base 바로 위에서 곡면으로 벌어져 r_bottom 짜리 disk 로는 그 안쪽 띠가
    #   안 덮여 테두리 틈이 보인다. 반경을 키우고(BOWL_R_BOTTOM*1.6) top 을 캐비티
    #   바닥(z_base+floor)까지 올려 벽 안쪽과 겹쳐 틈을 가린다(double-sided 벽이 가림).
    bottom_r = BOWL_R_BOTTOM * 1.6
    bottom_h = BOWL_Z_BASE + BOWL_FLOOR_THICKNESS
    bottom = UsdGeom.Cylinder.Define(stage, "/Bowl/Bottom")
    bottom.CreateAxisAttr(UsdGeom.Tokens.z)
    bottom.CreateRadiusAttr(bottom_r)
    bottom.CreateHeightAttr(bottom_h)
    bottom.CreateExtentAttr(
        [Gf.Vec3f(-bottom_r, -bottom_r, -bottom_h * 0.5),
         Gf.Vec3f(bottom_r, bottom_r, bottom_h * 0.5)]
    )
    _set_xform(bottom, translate=(0.0, 0.0, bottom_h * 0.5))
    _bind_visual(bottom.GetPrim(), bowl_blue)

    # 충돌: watertight shell + convexDecomposition (num_envs>1 안정; SDF 는 멀티 env
    #   cooking 비용·crash 로 부적합). PhysX 전용 속성은 검증된 속성명을 직접 author
    #   한다(schema Create*Attr 메서드명 의존 회피). invisible.
    col = UsdGeom.Mesh.Define(stage, "/Bowl/Collision")
    cpts, cfaces = _bowl_collision_geometry(
        BOWL_R_BOTTOM, BOWL_R_TOP, BOWL_Z_BASE, BOWL_DEPTH,
        BOWL_LATS, BOWL_LONS, BOWL_WALL_THICKNESS, BOWL_FLOOR_THICKNESS,
    )
    _set_mesh(col, cpts, cfaces, double_sided=False)
    col.MakeInvisible()
    col_prim = col.GetPrim()
    _apply_collision(col_prim, contact_tuning=True, contact_offset=CONTACT_OFFSET_DEFAULT)
    UsdPhysics.MeshCollisionAPI.Apply(col_prim).CreateApproximationAttr().Set("convexDecomposition")
    col_prim.AddAppliedSchema("PhysxConvexDecompositionCollisionAPI")
    col_prim.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(BOWL_MAX_CONVEX_HULLS)
    col_prim.CreateAttribute("physxConvexDecompositionCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(BOWL_HULL_VERTEX_LIMIT)
    col_prim.CreateAttribute("physxConvexDecompositionCollision:voxelResolution", Sdf.ValueTypeNames.Int).Set(BOWL_VOXEL_RESOLUTION)
    col_prim.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(True)
    _bind_physics(col_prim, bowl_friction)

    return stage


# ---------------------------------------------------------------------------
# scene.usd author
# ---------------------------------------------------------------------------

def _static_cube(
    stage: "Usd.Stage",
    path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    visual_mat: str,
    collision: bool = False,
    physics_mat: str | None = None,
) -> None:
    """씬 정적 박스(책상/다리/매트/천장) — 해석적 Cube(size=1) + scale."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    _set_xform(cube, translate=translate, scale=scale)
    _bind_visual(cube.GetPrim(), visual_mat)
    if collision:
        _apply_collision(cube.GetPrim(), contact_tuning=True)
        if physics_mat is not None:
            _bind_physics(cube.GetPrim(), physics_mat)


def _add_payload_ref(
    stage: "Usd.Stage",
    name: str,
    payload_rel: str,
    *,
    translate: tuple[float, float, float],
    rotate_z: float | None = None,
) -> None:
    """객체 USD 를 payload 로 참조하는 Xform 배치."""
    xform = UsdGeom.Xform.Define(stage, f"/Scene/{name}")
    xform.GetPrim().GetPayloads().AddPayload(Sdf.Payload(payload_rel))
    _set_xform(xform, translate=translate, rotate_z=rotate_z)


def author_scene() -> "Usd.Stage":
    """scene.usd author — 책상/조명 + 객체 5개 payload 참조."""
    stage = _new_stage("Scene", SCENE_USD_PATH.with_suffix(".usda"))

    # 정적 씬 머티리얼.
    looks = "/Scene/Looks"
    UsdGeom.Scope.Define(stage, looks)
    mats: dict[str, str] = {}
    for mat_name in ("DeskWood", "DeskMat", "Ceiling"):
        color, roughness, metallic = MATERIALS[mat_name]
        mats[mat_name] = _visual_material(stage, looks, mat_name, color, roughness, metallic)
    desk_friction = _physics_material(stage, looks, "DeskFriction", FRICTION_DESK)
    # 공유 CubeFriction — 4큐브가 각자 가지면 env 당 머티리얼 4개(PhysX 64K 한도로 16384 env
    # 불가). scene 레벨 단일 머티리얼을 큐브 collider 에 over-bind(값 동일, 인스턴스 4→1).
    shared_cube_friction = _physics_material(stage, looks, "CubeFriction", FRICTION_CUBE, friction_combine="average")

    # ── 조명 ──────────────────────────────────────────────────────────────
    # 광원은 scene.usd(=per-env {ENV_REGEX_NS}/Scene 로 마운트)에 두지 않는다.
    # USD 광원은 scope 격리가 없어 env 수만큼 복제되면 N배 과노출(IsaacLab #4340/#1729).
    # → 광원은 PickCubeSceneCfg 가 /World/Light·/World/KeyLight(env 계층 밖, 복제 안 됨)에
    #   단일로 author 한다. usdview 단독 검증 시엔 뷰어 기본 조명/헤드라이트를 쓴다.

    # 천장.
    _static_cube(
        stage, "/Scene/Ceiling",
        translate=_shift((0.0, 0.31, 1.795)), scale=(5.0, 4.0, 0.05),
        visual_mat=mats["Ceiling"],
    )
    # 상판: 1600×800×25mm, 윗면 = world z=0.705.
    _static_cube(
        stage, "/Scene/DeskTop",
        translate=_shift((0.0, 0.31, -0.0125)), scale=(1.60, 0.80, 0.025),
        visual_mat=mats["DeskWood"], collision=True, physics_mat=desk_friction,
    )
    # 다리 4개 (충돌 없음).
    for leg_name, pos in (
        ("DeskLegBackLeft", (-0.72, 0.64, -0.365)),
        ("DeskLegBackRight", (0.72, 0.64, -0.365)),
        ("DeskLegFrontLeft", (-0.72, -0.02, -0.365)),
        ("DeskLegFrontRight", (0.72, -0.02, -0.365)),
    ):
        _static_cube(
            stage, f"/Scene/{leg_name}",
            translate=_shift(pos), scale=(0.025, 0.025, 0.68),
            visual_mat=mats["DeskWood"],
        )
    # 매트: 860×400×4mm.
    _static_cube(
        stage, "/Scene/DeskMat",
        translate=_shift((-0.27, 0.20, 0.002)), scale=(0.86, 0.40, 0.004),
        visual_mat=mats["DeskMat"], collision=True, physics_mat=desk_friction,
    )

    # 객체 payload 참조.
    _add_payload_ref(stage, "Bowl", "./objects/Bowl/Bowl.usd", translate=_shift(BOWL_LOCAL))
    for name, pos, yaw in CUBES:
        _add_payload_ref(
            stage, name, f"./objects/{name}/{name}.usd",
            translate=_shift(pos), rotate_z=yaw,
        )
        # 공유 CubeFriction 을 큐브 collider(payload 의 /Box)에 over-bind.
        box_over = stage.OverridePrim(f"/Scene/{name}/Box")
        _bind_physics(box_over, shared_cube_friction)

    return stage


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _inject_pxr() -> None:
    """isaacsim 부팅 후 pxr / PhysxSchema 심볼을 모듈 전역에 주입."""
    from pxr import Usd as _Usd, UsdGeom as _UsdGeom, UsdPhysics as _UsdPhysics
    from pxr import UsdLux as _UsdLux, UsdShade as _UsdShade
    from pxr import Sdf as _Sdf, Gf as _Gf, Vt as _Vt, PhysxSchema as _PhysxSchema
    globals().update(
        Usd=_Usd, UsdGeom=_UsdGeom, UsdPhysics=_UsdPhysics, UsdLux=_UsdLux,
        UsdShade=_UsdShade, Sdf=_Sdf, Gf=_Gf, Vt=_Vt, PhysxSchema=_PhysxSchema,
    )


def main() -> None:
    # PhysxSchema 활성화를 위해 isaacsim 을 headless 부팅한다.
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        _inject_pxr()

        for name, _pos, _yaw in CUBES:
            _export_pair(author_cube(name), OBJECTS_DIR / name / name)
        _export_pair(author_bowl(), OBJECTS_DIR / "Bowl" / "Bowl")
        _export_pair(author_scene(), SCENE_DIR / "scene")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
