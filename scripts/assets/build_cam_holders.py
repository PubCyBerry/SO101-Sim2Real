"""SO-101 손목/전면(belly) 카메라 홀더 procedural 생성 + USD 베이크.

사진(docs/pics/cube_desk) 참고로 두 홀더를 box/wedge/fork 조합으로 부모 링크
프레임(미터)에 직접 빌드한다. 카메라 mount 면은 sim 튜닝 pose(WristCamera/
FrontCamera)에 맞춰 holder 모양과 카메라 위치가 동시에 정합되게 한다.

산출:
  - assets/robots/urdf/assets/wrist_cam_holder_so101.stl  (gripper_link frame, m)
  - assets/robots/urdf/assets/front_cam_holder_so101.stl  (shoulder_link frame, m)
  - so101_follower.usd 에 holder(보라)+카메라 모듈(검정) 메시 베이크
      /so101_new_calib/gripper/WristCamMount
      /so101_new_calib/shoulder/BellyCamMount

실행: uv run python scripts/assets/build_cam_holders.py
"""
from __future__ import annotations
import os, struct, math
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Vt, Sdf

REPO = "/home/konan147/Workspaces/SO101-Sim2Real"
URDF_ASSETS = os.path.join(REPO, "assets", "robots", "urdf", "assets")
FOLLOWER = os.path.join(REPO, "assets", "robots", "so101_follower.usd")

# ---------------------------------------------------------------- mesh builder
def _box(center, axes, half):
    """축정렬 아닌 box → (verts, faces). axes: 3x3 (열=x,y,z 단위벡터), half: 3."""
    c = np.asarray(center, float); A = np.asarray(axes, float); h = np.asarray(half, float)
    signs = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    V = [c + A[:,0]*s[0]*h[0] + A[:,1]*s[1]*h[1] + A[:,2]*s[2]*h[2] for s in signs]
    F = [(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),
         (1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0)]
    return V, F

def _cyl(center, axis, radius, length, nseg=24):
    """axis 방향 원기둥 (verts, faces). center=바닥면 중심."""
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    tmp = np.array([1.0,0,0]) if abs(a[0]) < 0.9 else np.array([0,1.0,0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u); v = np.cross(a, u)
    c0 = np.asarray(center, float); c1 = c0 + a*length
    V=[];
    for i in range(nseg):
        th = 2*math.pi*i/nseg
        r = u*math.cos(th)*radius + v*math.sin(th)*radius
        V.append(c0+r)
    for i in range(nseg):
        th = 2*math.pi*i/nseg
        r = u*math.cos(th)*radius + v*math.sin(th)*radius
        V.append(c1+r)
    V.append(c0); V.append(c1)
    b0=2*nseg; b1=2*nseg+1; F=[]
    for i in range(nseg):
        j=(i+1)%nseg
        F.append((i,j,nseg+j)); F.append((i,nseg+j,nseg+i))      # side
        F.append((b0,j,i)); F.append((b1,nseg+i,nseg+j))          # caps
    return V, F

class Mesh:
    def __init__(self): self.V=[]; self.F=[]
    def add(self, vf):
        v,f = vf; o=len(self.V); self.V.extend(v)
        self.F.extend([(a+o,b+o,c+o) for (a,b,c) in f])
    def tris(self):
        return [(self.V[a],self.V[b],self.V[c]) for (a,b,c) in self.F]
    def write_stl(self, path):
        tris=self.tris()
        with open(path,'wb') as fp:
            fp.write(b'\0'*80); fp.write(struct.pack('<I',len(tris)))
            for v0,v1,v2 in tris:
                n=np.cross(np.array(v1)-np.array(v0),np.array(v2)-np.array(v0))
                nn=np.linalg.norm(n); n=n/nn if nn>1e-12 else n
                fp.write(struct.pack('<3f',*n))
                for v in (v0,v1,v2): fp.write(struct.pack('<3f',*[float(x) for x in v]))
                fp.write(b'\0\0')

I3 = np.eye(3)

# 공용 UVC 카메라 목업(원통 렌즈 + 네모판 PCB) — wrist/belly 동일.
# pcam=광학중심(렌즈 베이스), Rcam 열=[right,up,fwd]. fwd=시선(장면 방향).
CAM_HALF = 0.016    # PCB 네모판 32x32 → half
CAM_TH = 0.004      # 카메라 두께(half 0.002)
LENS_R = 0.006; LENS_LEN = 0.0104   # 렌즈 길이(이전 0.013 의 80%)
def add_camera(cam, pcam, Rcam):
    fwd = Rcam[:,2]
    cam.add(_box(center=tuple(pcam - fwd*0.003), axes=Rcam, half=(CAM_HALF, CAM_HALF, CAM_TH/2)))
    cam.add(_cyl(center=pcam - fwd*0.001, axis=fwd, radius=LENS_R, length=LENS_LEN))

# ---------------------------------------------------------------- WRIST holder
# gripper_link frame. 나사판(wrist_roll +y면 hex-nut 2구멍) → 직각 얇은 융기 →
# 45° 숙인 두꺼운 융기(카메라 enclosure, 렌즈 주둥이 ~1cm만 노출).
def build_wrist():
    holder = Mesh(); cam = Mesh()
    right=np.array([-1,0,0.0]); up=np.array([0,0.883,-0.469]); fwd=np.array([0,-0.469,-0.883])
    up/=np.linalg.norm(up); fwd/=np.linalg.norm(fwd); right/=np.linalg.norm(right)
    Rcam=np.column_stack([right,up,fwd]); pcam=np.array([0,0.045,-0.04])
    # 1) 나사 박는 베이스판 (wrist_roll +y 외측면, 2구멍 덮음)
    holder.add(_box(center=(-0.001, 0.0255, -0.0234), axes=I3, half=(0.013, 0.0016, 0.013)))
    # 2) 직각 얇은 사각 융기 (베이스→enclosure 브리지하는 얇은 벽, 두께=카메라 몸통 4mm)
    holder.add(_box(center=(-0.001, 0.036, -0.030), axes=I3, half=(0.012, 0.010, CAM_TH/2)))
    # 3) 45° 숙인 사각 융기 = 카메라 enclosure(감쌈). 두께≈카메라 몸통 2.5배(10mm).
    holder.add(_box(center=tuple(pcam - fwd*0.004), axes=Rcam, half=(0.019, 0.019, 0.005)))
    # 4) 카메라 목업 (enclosure 안 — 렌즈 주둥이만 노출)
    add_camera(cam, pcam, Rcam)
    return holder, cam

# ---------------------------------------------------------------- BELLY holder
# shoulder_link frame. 앞면=local -x(world -Y, 작업공간). 카메라 fwd=local -x.
# 사진(전면 카메라 크롭): 카메라 뒤에 같은 두께 얇은 backing 판 + shoulder→판 얇은 기둥.
def build_belly():
    holder = Mesh(); cam = Mesh()
    xf = -0.050           # shoulder front face (local -x)
    ZC = 0.0226           # 4 나사구멍 중심(0.0126) + 1cm 아래(local +z=world 아래)
    plate_th = CAM_TH*1.5 # backing 판 두께 = 카메라 1.5배 (6mm)
    post_len = 0.019      # 좁은 기둥 길이 — 렌즈 끝이 숄더 표면(≈-0.0476)에서 ~4cm 앞에 오게.
    # right/up/fwd (카메라가 -x 향함)
    right=np.array([0,-1,0.0]); up=np.array([0,0,1.0]); fwd=np.array([-1,0,0.0])
    Rcam=np.column_stack([right,up,fwd])
    # 1) 좁은 네모 기둥: shoulder 앞면에서 -x 로 융기
    holder.add(_box(center=(xf-post_len/2, 0.0, ZC), axes=I3, half=(post_len/2+0.001, 0.006, 0.006)))
    # 2) backing 판: 카메라보다 살짝 크게(+1.5mm) → 동일 크기 coplanar z-fighting 방지.
    plate_x = xf - post_len - plate_th/2
    holder.add(_box(center=(plate_x, 0.0, ZC), axes=I3, half=(plate_th/2, CAM_HALF+0.0015, CAM_HALF+0.0015)))
    # 3) 카메라 목업: PCB 뒷면을 판 속으로 묻어 coplanar 면 제거(앞면만 노출).
    plate_front = plate_x - plate_th/2
    pcam = np.array([plate_front - 0.003, 0.0, ZC])
    add_camera(cam, pcam, Rcam)
    return holder, cam, pcam, Rcam

# ---------------------------------------------------------------- USD authoring
def author_mesh(stage, path, mesh, color, bind_mat=None):
    m = UsdGeom.Mesh.Define(stage, path)
    V = mesh.V; F = mesh.F
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in V]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(F)))
    idx=[]; [idx.extend(f) for f in F]
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    P=np.array(V); m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*P.min(0).tolist()),Gf.Vec3f(*P.max(0).tolist())]))
    if bind_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(bind_mat)
    return m

def apply_collision(prim, approximation="convexHull", contact_offset=0.002, rest_offset=0.0):
    """Mesh prim 에 PhysicsCollisionAPI + PhysicsMeshCollisionAPI 부여."""
    col_api = UsdPhysics.CollisionAPI.Apply(prim)
    col_api.CreateCollisionEnabledAttr(True)
    mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col_api.CreateApproximationAttr(approximation)
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact_offset)
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest_offset)


def ensure_black_material(stage):
    p="/so101_new_calib/Looks/material_black"
    if stage.GetPrimAtPath(p): return UsdShade.Material(stage.GetPrimAtPath(p))
    mat=UsdShade.Material.Define(stage,p)
    sh=UsdShade.Shader.Define(stage,p+"/Shader")
    sh.SetSourceAsset("OmniPBR.mdl","mdl"); sh.SetSourceAssetSubIdentifier("OmniPBR","mdl")
    sh.CreateInput("diffuse_color_constant",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04,0.04,0.045))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(),"out")
    return mat

def main():
    PURPLE=(0.4,0.03,0.75); BLACK=(0.04,0.04,0.045)
    wrist_h, wrist_c = build_wrist()
    belly_h, belly_c, bpcam, bRcam = build_belly()
    os.makedirs(URDF_ASSETS, exist_ok=True)
    wrist_h.write_stl(os.path.join(URDF_ASSETS,"wrist_cam_holder_so101.stl"))
    belly_h.write_stl(os.path.join(URDF_ASSETS,"front_cam_holder_so101.stl"))
    print("STL written (m, parent-link frame)")
    for nm,mh in [("wrist",wrist_h),("belly",belly_h)]:
        P=np.array(mh.V); print(f"  {nm} holder tris={len(mh.F)} bbox min{P.min(0).round(4)} max{P.max(0).round(4)}")

    st=Usd.Stage.Open(FOLLOWER)
    matp=UsdShade.Material(st.GetPrimAtPath("/so101_new_calib/Looks/material_a_3d_printed"))
    matk=ensure_black_material(st)
    # ---- wrist : replace old WristCamMount ----
    for old in ["/so101_new_calib/gripper/WristCamMount"]:
        if st.GetPrimAtPath(old): st.RemovePrim(old)
    UsdGeom.Xform.Define(st,"/so101_new_calib/gripper/WristCamMount")
    wrist_holder_m = author_mesh(st,"/so101_new_calib/gripper/WristCamMount/holder",wrist_h,PURPLE,matp)
    apply_collision(wrist_holder_m.GetPrim())
    author_mesh(st,"/so101_new_calib/gripper/WristCamMount/camera",wrist_c,BLACK,matk)
    # ---- belly : shoulder front ----
    for old in ["/so101_new_calib/shoulder/BellyCamMount"]:
        if st.GetPrimAtPath(old): st.RemovePrim(old)
    UsdGeom.Xform.Define(st,"/so101_new_calib/shoulder/BellyCamMount")
    belly_holder_m = author_mesh(st,"/so101_new_calib/shoulder/BellyCamMount/holder",belly_h,PURPLE,matp)
    apply_collision(belly_holder_m.GetPrim())
    author_mesh(st,"/so101_new_calib/shoulder/BellyCamMount/camera",belly_c,BLACK,matk)
    st.GetRootLayer().Save()
    print("baked WristCamMount + BellyCamMount into follower USD")
    print(f"  belly camera optical pt (shoulder frame) = {bpcam.round(4)} fwd={bRcam[:,2].round(3)}")

if __name__=="__main__":
    main()
