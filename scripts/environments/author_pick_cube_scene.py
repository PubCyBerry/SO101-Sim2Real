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
  ├── scene.usd + .usda                 # 책상/조명 + 객체 2개(Cube1·Bowl) payload 참조
  └── objects/
      ├── Cube1/Cube1.usd + .usda       # 시각 라운드 mesh + invisible convexHull 충돌 mesh
      └── Bowl/Bowl.usd + .usda         # 시각 회전체 Mesh + watertight SDF 충돌 Mesh

좌표는 SO-101 follower init_state 와 맞춘 SCENE_OFFSET 으로 시프트한다.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

# pxr / PhysxSchema 심볼은 isaacsim 부팅 후 main() 에서 주입된다(아래 _inject_pxr).
# 부팅 전에 import 하면 PhysxSchema 가 없어 ImportError 가 난다.
Usd = UsdGeom = UsdPhysics = UsdLux = UsdShade = Sdf = Gf = Vt = PhysxSchema = None  # type: ignore


SCENE_DIR = Path(__file__).resolve().parents[2] / "assets" / "scenes" / "cube_desk"
SCENE_USD_PATH = SCENE_DIR / "scene.usd"
OBJECTS_DIR = SCENE_DIR / "objects"

# 큐브 크기 단일 진실 소스(cube_specs)를 importlib 파일 로드로 직접 읽는다.
# `from sim_to_real...` 는 패키지 __init__ 의 isaaclab_tasks 의존 때문에 AppLauncher
# 부팅 전엔 크래시 → 자족 leaf 파일만 직접 exec(패키지 init 우회).
def _load_cube_specs():
    import sys

    path = Path(__file__).resolve().parents[2] / "src" / "sim_to_real" / "utils" / "cube_specs.py"
    spec = importlib.util.spec_from_file_location("_cube_specs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass 가 cls.__module__ 를 sys.modules 에서 찾음 → 선등록 필수
    spec.loader.exec_module(mod)
    return mod


_CUBE_SPECS = _load_cube_specs().CUBE_SPECS

# 로봇 base 를 world 원점(XY)에 두기 위한 offset (recenter delta=(-1.84,+0.565,0) 반영).
# offset.x = 0.36: 책상 중심(scene-local x=0)이 world x=0.36 → 로봇 발치 x=0 이 책상 위.
# offset.y = 0.045: 책상 앞 모서리(scene-local y=-0.09)가 world y=-0.045 에 오도록.
# offset.z = 다리(0.68) + 상판(0.025) = 0.705: leg bottoms on Isaac ground(z=0).
SCENE_OFFSET: tuple[float, float, float] = (0.36, 0.045, 0.705)

# (diffuseColor, roughness, metallic)
MATERIALS = {
    "DeskWood": ((0.72, 0.64, 0.54), 0.72, 0.0),  # 밝은 자작 합판(다리·상판 측면). 윗면은 DeskTopTex 텍스처.
    "BowlBlue": ((0.16, 0.34, 0.78), 0.28, 0.0),  # 진한 푸른색에서 살짝 연하게(2026-06-26)
    "Ceiling": ((0.88, 0.86, 0.82), 0.95, 0.0),
}

# 큐브 4개 scene-local 평면 배치 (이름 → x, y, yaw°). 매트 앞쪽에 흩뿌림.
# z 와 scale·mass 는 cube_specs(단일 진실 소스)에서 파생 — 크기 변경은 cube_specs 만.
# 단일 큐브 씬(2026-06-26): Cube1 1개만. 매트 제거.
# 크기는 cube_specs 가 정한다 — 2026-07-30 부터 25mm(크기 DR 사다리의 하한).
_CUBE_LAYOUT: dict[str, tuple[float, float, float]] = {
    "Cube1": (-0.50, 0.08, 20.0),
}
_DESK_TOP_LOCAL: float = 0.0    # 책상 상판 윗면 scene-local z (매트 제거됨)
_CUBE_Z_SLACK: float = 0.001    # spawn 침투 방지 여유
# z 중심 = 책상 상판 + 큐브 반높이 + slack (25mm→0.0135 · 40mm→0.021).
CUBES = tuple(
    (name, (x, y, _DESK_TOP_LOCAL + _CUBE_SPECS[name].half_extent + _CUBE_Z_SLACK), yaw)
    for name, (x, y, yaw) in _CUBE_LAYOUT.items()
)

# 그릇 scene-local (바닥 중심).
BOWL_LOCAL: tuple[float, float, float] = (-0.58, 0.26, 0.010)

CUBE_SCALES: dict[str, tuple[float, float, float]] = {n: s.scale for n, s in _CUBE_SPECS.items()}

# ── 물리 상수 (docs/GRASP_PHYSICS.md 근거 — 임의 변경 금지) ──────────────────
# 큐브 질량 — cube_specs 에서 파생(크기별 차등, 쉘 ∝ 변² 비례. 40mm=35g·50mm=55g).
CUBE_MASSES: dict[str, float] = {n: s.mass for n, s in _CUBE_SPECS.items()}
CONTACT_OFFSET_DEFAULT = 0.004      # 정적·두꺼운 면(책상/매트/그릇)
CUBE_CONTACT_OFFSET = 0.002         # grasp 대상 큐브 전용(convexHull 접촉이 안정적이라 좁은 margin OK)
# 큐브 시각 형태 — 실물은 회색 펠트로 감싼 쿠션형(코너 반경 큼). 라운드 박스로 author.
CUBE_ROUND_RADIUS_FRAC: float = 0.22  # 변 대비 코너 반경 비율(0.040→8.8mm, 0.050→11mm)
CUBE_ROUND_SEGS: int = 10             # 면당 격자 분할(라운딩 매끈도)
# 큐브 충돌 = convexHull Mesh. 시각(라운드 펠트)과 동일 형상의 invisible mesh.
#   ⚠ 2026-06-22: SDF → convexHull 교체. 큐브는 **볼록**이라 convexHull 이 라운드 표면을
#   정확히 표현(오목 형상만 SDF 필요 — bowl/jaw 전용). SDF 는 평평한 책상 접촉서 normal 이
#   매 step 뒤집혀 큐브 제자리 회전 버즈(~2.9 rad/s, "덜그럭")를 냈고, 그 불안정이 grasp 도
#   망가뜨렸다(고정 spawn SM 3/16). convexHull 로 jitter 해소(0.056 rad/s 정지, 50배↓) +
#   grasp 복원(13/16, 81%). 측정: scripts/test/measure_cube_jitter.py.
CUBE_COLLISION_SEGS: int = 6          # 충돌 mesh 면당 분할(convexHull source — 시각보다 거칠어도 무방)
CUBE_FELT_ROUGHNESS: float = 0.95     # 펠트 천 — 거의 완전 확산
# 흰 시접 무늬는 geometry 가 아니라 albedo 텍스처에 그린다(평면 무늬). UV 는 큐브
# 전개도(net): 앞(+X)·윗(+Z)·뒤(-X)·밑(-Z) 4면을 세로(v)로 연속 적층(밴드 컬럼 u<0.5),
# 옆면(±Y)은 우측 컬럼(u>0.5). 텍스처의 밴드 컬럼에 흰 둥근사각 윤곽선을 그리면
# 앞→윗→뒤 3면에 걸친 길쭉한 사각 무늬가 모서리 넘어 연속으로 이어진다.
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
FRICTION_BOWL = (0.12, 0.10, 0.0)   # 미끌 — 매끈한 플라스틱 내부. combine=min.
#   restitution 0 (옛 0.3 = 큐브가 그릇서 튕겨 서로 충돌→솔버 폭발"팝콘". 반발 제거)
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


def _relativize_asset_paths(usd_path: Path) -> None:
    """export 된 .usd 레이어의 절대 asset path(텍스처 등)를 레이어 디렉터리 기준 상대경로로 정규화.

    ⚠ USD 의 layer Export 는 일부 플랫폼/버전(특히 리눅스)에서 작성한 relative asset path 를
    author 머신의 절대경로로 re-anchor 한다. 그 절대경로(`/home/<user>/.../textures/...`)가
    바이너리 .usd 에 박히면 다른 머신(Windows)에서 텍스처가 resolve 안 돼 로드 실패 → 큐브가
    검정으로 렌더된다(docs/TROUBLESHOOTING.md §"cube_desk 큐브가 GUI 에서 검은색"). committed
    .usd 가 항상 포터블한 상대경로를 갖도록, export 직후 절대 asset path 를 상대로 되돌린다.
    레이어 밖을 가리켜 상대화 불가한 절대경로가 남으면 author 시점에 RuntimeError(트립와이어).
    """
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        raise RuntimeError(f"cannot reopen exported layer for asset-path check: {usd_path}")
    layer_dir = str(usd_path.resolve().parent)

    def _is_abs(p: str) -> bool:
        return p.startswith("/") or (len(p) > 2 and p[1] == ":")

    remaining: list[str] = []

    def _walk(prim_spec) -> None:
        for prop in prim_spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec) or prop.typeName != Sdf.ValueTypeNames.Asset:
                continue
            val = prop.default
            if val is None:
                continue
            p = val.path
            if not p or not _is_abs(p):
                continue
            rel = os.path.relpath(os.path.abspath(p), layer_dir).replace(os.sep, "/")
            if _is_abs(rel):
                remaining.append(p)  # 레이어 밖(다른 드라이브 등) — 상대화 불가
            else:
                prop.default = Sdf.AssetPath(rel)
        for child in prim_spec.nameChildren:
            _walk(child)

    for root in layer.rootPrims:
        _walk(root)
    if remaining:
        raise RuntimeError(
            f"absolute asset paths could not be relativized in {usd_path}: {remaining}"
        )
    layer.Save()


def _export_pair(stage: "Usd.Stage", path_no_ext: Path) -> None:
    """.usda(텍스트) 저장 + .usd(usdc 바이너리) export. 텍스처 경로는 항상 상대로 정규화."""
    usda_path = path_no_ext.with_suffix(".usda")
    usd_path = path_no_ext.with_suffix(".usd")
    stage.GetRootLayer().Save()  # 이미 .usda 로 CreateNew 했으므로 텍스트 저장
    if not stage.GetRootLayer().Export(str(usd_path), args={"format": "usdc"}):
        raise RuntimeError(f"Failed to export binary USD: {usda_path} -> {usd_path}")
    # Export 가 절대경로로 re-anchor 한 텍스처 경로를 상대로 되돌린다(큐브 검정 재발 방지).
    _relativize_asset_paths(usd_path)
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


def _textured_material(
    stage: "Usd.Stage",
    parent_path: str,
    name: str,
    texture_rel: str,
    *,
    roughness: float,
    normal_rel: str | None = None,
    wrap: str = "clamp",
) -> str:
    """UsdPreviewSurface + UsdUVTexture 텍스처 머티리얼 생성, prim path 반환.

    texture_rel: 머티리얼이 author 되는 레이어 기준 상대 에셋 경로
      (scene.usda → "./textures/desk_mat.png", 객체 USD → "../../textures/cube_felt_albedo.png").
    diffuseColor 를 텍스처 rgb 출력에 연결하고, st 는 PrimvarReader_float2("st") 로 읽는다.
    normal_rel 지정 시 tangent-space normal map 을 surface.normal 에 연결한다
      (raw 컬러스페이스 + scale (2,2,2) / bias (-1,-1,-1) 로 [0,1]→[-1,1] 디코드).
    wrap: UsdUVTexture wrapS/T 모드("clamp" 단일 이미지 / "repeat" 타일링 펠트).
    """
    mat_path = f"{parent_path}/{name}"
    material = UsdShade.Material.Define(stage, mat_path)

    st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, f"{mat_path}/DiffuseTex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_rel)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    if normal_rel is not None:
        ntex = UsdShade.Shader.Define(stage, f"{mat_path}/NormalTex")
        ntex.CreateIdAttr("UsdUVTexture")
        ntex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_rel)
        ntex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
        ntex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        ntex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        ntex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        ntex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        ntex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        ntex_rgb = ntex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(ntex_rgb)

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

def _rounded_box_geometry(
    sx: float, sy: float, sz: float, radius: float, segs: int
) -> tuple[
    list[tuple[float, float, float]],
    list[list[int]],
    list[tuple[float, float]],
    list[tuple[float, float, float]],
]:
    """매끈한 라운드 박스(쿠션형) — clamp-core 라운딩. 실물 펠트 큐브 재현.

    6면을 각각 (segs+1)² 격자로 만들고, full-box 면 위 점 p 에 대해
    core = clamp(p, -a, +a) (a = 반치수 - radius) 로 잡아 v = core + radius·dir̂
    로 모서리/코너를 매끈하게 굴린다. 면 중심부(|p_inplane| ≤ a)는 평면 유지.

    normal = dir̂ (라운딩 방향) 을 per-vertex 로 author → 매끈 셰이딩. UV 는 큐브
    전개도(net) — `_net_uv` 로 앞·윗·뒤·밑을 세로 연속 적층(시접 무늬용). 면 경계
    정점은 위치·노멀이 위치기반 결정이라 인접 면과 동일 좌표로 산출돼 crack 이
    없다(double-sided 라 winding 영향도 없음).

    반환: (points, faces, uvs, normals)
    """
    h = (sx / 2.0, sy / 2.0, sz / 2.0)
    a = tuple(max(1e-6, hi - radius) for hi in h)  # core 반치수
    # (법선축, 부호, u축, v축)
    faces_def = [
        (0, +1, 1, 2), (0, -1, 2, 1),
        (1, +1, 2, 0), (1, -1, 0, 2),
        (2, +1, 0, 1), (2, -1, 1, 0),
    ]
    points: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for (ax, sign, ua, va) in faces_def:
        base = len(points)
        su, sv = 2.0 * h[ua], 2.0 * h[va]
        for j in range(segs + 1):
            for i in range(segs + 1):
                fu, fv = i / segs, j / segs
                p = [0.0, 0.0, 0.0]
                p[ax] = sign * h[ax]
                p[ua] = (fu - 0.5) * su
                p[va] = (fv - 0.5) * sv
                core = [min(max(p[k], -a[k]), a[k]) for k in range(3)]
                d = [p[k] - core[k] for k in range(3)]
                dl = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                if dl > 1e-9:
                    nrm = (d[0] / dl, d[1] / dl, d[2] / dl)
                    v = (core[0] + radius * nrm[0], core[1] + radius * nrm[1], core[2] + radius * nrm[2])
                else:  # radius=0 등 퇴화 — 면 법선
                    nrm = (0.0, 0.0, 0.0)
                    nrm = tuple(sign if k == ax else 0.0 for k in range(3))  # type: ignore[assignment]
                    v = tuple(core)  # type: ignore[assignment]
                points.append(v)
                normals.append(nrm)
                uvs.append(_net_uv(ax, sign, fu, fv))
        for j in range(segs):
            for i in range(segs):
                v00 = base + j * (segs + 1) + i
                v10 = v00 + 1
                v01 = v00 + (segs + 1)
                v11 = v01 + 1
                faces.append([v00, v10, v11, v01] if sign > 0 else [v00, v01, v11, v10])
    return points, faces, uvs, normals


def _net_uv(
    ax: int, sign: int, fu: float, fv: float
) -> tuple[float, float]:
    """라운드박스 면(ax,sign) 의 격자 파라미터(fu,fv)→큐브 전개도 UV.

    밴드 컬럼(u∈[0,0.5]): 앞(+X) v[0,.25] → 윗(+Z) v[.25,.5] → 뒤(-X) v[.5,.75]
    → 밑(-Z) v[.75,1]. Y(밴드 폭)→u. 인접면 공유 모서리에서 u·v 가 정확히 일치해
    무늬가 모서리 넘어 연속된다. 옆면(±Y): 우측 컬럼 u∈[0.5,1], 무늬 없음.
    """
    if ax == 0 and sign > 0:        # +X 앞 (fu=Y, fv=Z)
        return (0.5 * fu, 0.25 * fv)
    if ax == 2 and sign > 0:        # +Z 윗 (fu=X, fv=Y)
        return (0.5 * fv, 0.25 + 0.25 * (1.0 - fu))
    if ax == 0 and sign < 0:        # -X 뒤 (fu=Z, fv=Y)
        return (0.5 * fv, 0.50 + 0.25 * (1.0 - fu))
    if ax == 2 and sign < 0:        # -Z 밑 (fu=Y, fv=X)
        return (0.5 * fu, 0.75 + 0.25 * fv)
    if ax == 1 and sign > 0:        # +Y 옆 (fu=Z, fv=X)
        return (0.5 + 0.5 * fv, 0.5 * fu)
    return (0.5 + 0.5 * fu, 0.5 + 0.5 * fv)  # -Y 옆 (fu=X, fv=Z)


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
    """큐브 1개 stage author — 시각 라운드 mesh + invisible SDF 충돌 mesh."""
    stage = _new_stage(name, OBJECTS_DIR / name / f"{name}.usda")
    root_prim = stage.GetPrimAtPath(f"/{name}")

    _apply_rigid_body(
        root_prim,
        mass=CUBE_MASSES[name],
        angular_damping=3.0,                   # resting jitter 해소(2026-06-22): 1.5→3.0, 미세 회전 진동 흡수
        linear_damping=2.5,                    # resting jitter 해소(2026-06-22): 1.5→2.5, 미세 병진 진동 흡수
        solver_position_iterations=32,
        solver_velocity_iterations=8,          # 원복(16 시도→grasp 회귀 격리, maxDepen 이 주범)
        max_depenetration_velocity=1.0,        # 원복(0.5 가 grasp grip 약화 92.8→77% 회귀)
        stabilization_threshold=0.002,         # resting jitter 해소(2026-06-22): 0.0005→0.002, 정지 근처 솔버 안정화 강화(sleep_threshold 는 default 0.0005 유지)
    )

    looks = f"/{name}/Looks"
    UsdGeom.Scope.Define(stage, looks)
    # 회색 펠트 천 머티리얼 — 실사 기반 절차적 albedo + normal map (보풀감). 객체 USD
    #   기준 상대 경로(objects/<name>/<name>.usda → ../../textures/). repeat 타일링.
    felt = _textured_material(
        stage, looks, "GrayFelt",
        "../../textures/cube_felt_albedo.png",
        roughness=CUBE_FELT_ROUGHNESS,
        normal_rel="../../textures/cube_felt_normal.png",
        wrap="repeat",
    )
    # 물리 friction 머티리얼은 큐브 USD 에 두지 않는다 — PhysX 64K 머티리얼 한도 때문에
    # env 당 4개(큐브별) 복제를 막기 위해 scene.usd 가 단일 공유 CubeFriction 을 over-bind
    # 한다(값 동일, 인스턴스만 4→1). 16384 env 가능(8192×6=49K → 16384×3=49K).

    # 시각 메시: 라운드 박스(쿠션형 펠트 큐브) — per-vertex UV + 매끈 normal. 충돌 없음.
    sx, sy, sz = CUBE_SCALES[name]
    radius = min(sx, sy, sz) * CUBE_ROUND_RADIUS_FRAC
    visual = UsdGeom.Mesh.Define(stage, f"/{name}/Visual")
    pts, faces, uvs, normals = _rounded_box_geometry(sx, sy, sz, radius, CUBE_ROUND_SEGS)
    _set_mesh(visual, pts, faces, double_sided=True)
    st = UsdGeom.PrimvarsAPI(visual).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(*uv) for uv in uvs])
    visual.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
    visual.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    _bind_visual(visual.GetPrim(), felt)
    # 흰 시접 무늬는 GrayFelt albedo(전개도 net UV)에 그려져 있어 별도 mesh 불필요.

    # 충돌 전용 convexHull Mesh: 시각(라운드 펠트)과 동일 형상의 invisible mesh.
    #   ⚠ 2026-06-22: 기존 SDF → convexHull 로 교체(jitter 해소). 큐브는 **볼록(convex)**
    #   형상이라 convexHull 이 라운드 표면을 정확히 표현한다(오목면이 없어 SDF 불필요 —
    #   SDF 는 bowl/jaw 같은 오목 형상 전용). SDF 는 평평한 책상 접촉에서 normal 이 매 step
    #   뒤집혀 큐브가 제자리 회전 버즈(측정 각속도 ~2.9 rad/s 격렬 진동, "덜그럭")를 냈다.
    #   convexHull 은 동일 grasp 표면 + 안정 접촉(~0.056 rad/s 정지, 50배↓) + 저비용.
    #   측정·재현: scripts/test/measure_cube_jitter.py. grasp 회귀 없음(고정 spawn SM 동일).
    col_pts, col_faces, _cuv, _cn = _rounded_box_geometry(sx, sy, sz, radius, CUBE_COLLISION_SEGS)
    col = UsdGeom.Mesh.Define(stage, f"/{name}/Collision")
    _set_mesh(col, col_pts, col_faces, double_sided=False)
    col.MakeInvisible()
    _apply_collision(col.GetPrim(), contact_tuning=True, contact_offset=CUBE_CONTACT_OFFSET)
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr().Set("convexHull")
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
        max_depenetration_velocity=1.0,        # 원복(grasp 회귀 격리)
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


def _textured_quad(
    stage: "Usd.Stage",
    path: str,
    *,
    translate: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat_path: str,
) -> None:
    """텍스처용 평면 quad Mesh — +Z 향, primvars:st(vertex) 부여. 충돌 없음.

    단위 quad(XY ±0.5, z=0)를 scale 로 키운다. st: u→+X, v→+Y 로 매핑해
    텍스처 좌상단(꽃)이 매트 (-X,+Y) 쪽에 오도록 한다.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, -0.5, 0.0),
        Gf.Vec3f(0.5, 0.5, 0.0), Gf.Vec3f(-0.5, 0.5, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0)])
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr(True)
    normals = mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    _set_xform(mesh, translate=translate, scale=scale)
    _bind_visual(mesh.GetPrim(), mat_path)


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
    for mat_name in ("DeskWood",):   # Ceiling·DeskMat 제거
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

    # 천장 제거(사용자) — 불필요·거슬림.
    # 상판: 1600×800×25mm, 윗면 = world z=0.705.
    _static_cube(
        stage, "/Scene/DeskTop",
        translate=_shift((0.0, 0.31, -0.0125)), scale=(1.60, 0.80, 0.025),
        visual_mat=mats["DeskWood"], collision=True, physics_mat=desk_friction,
    )
    # 상판 윗면 나무 텍스처 (실사 자작 합판). Cube 는 UV 없어 윗면(scene-local z=0) 위 0.3mm 에
    #   UV quad 를 얹는다. 매트가 덮는 부분은 매트 Cube(불투명)가 위에서 가린다.
    desk_top_tex = _textured_material(
        stage, looks, "DeskTopTex", "./textures/desk_top.png", roughness=0.7
    )
    _textured_quad(
        stage, "/Scene/DeskTopSurface",
        translate=_shift((0.0, 0.31, 0.0003)), scale=(1.60, 0.80, 1.0),
        mat_path=desk_top_tex,
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
    # 매트 제거(2026-06-26 사용자) — 큐브는 책상 상판 위에 직접 안착(DeskTop 충돌 담당).

    # 객체 payload 참조.
    _add_payload_ref(stage, "Bowl", "./objects/Bowl/Bowl.usd", translate=_shift(BOWL_LOCAL))
    for name, pos, yaw in CUBES:
        _add_payload_ref(
            stage, name, f"./objects/{name}/{name}.usd",
            translate=_shift(pos), rotate_z=yaw,
        )
        # 공유 CubeFriction 을 큐브 collider(payload 의 /Collision)에 over-bind.
        box_over = stage.OverridePrim(f"/Scene/{name}/Collision")
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
