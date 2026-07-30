"""카메라 extrinsic DR(6-DoF) 상태기계 계약 검증 — CPU 텐서, 씬·렌더 없음.

검증 대상 = ``sim_to_real.utils.domain_randomization.CameraExtrinsicDR`` 의 **실물 코드**.
카메라 sensor 는 stub view(로컬 pose 를 들고 있다가 write 를 기록)로 대신하고, nominal 은
실측값(probe 로 읽은 prim local pose)을 그대로 쓴다. 렌더가 필요한 항목(Fabric 반영·시야)은
GPU 스모크가 담당한다.

7 항목:
  C1 zero-range identity + 해석적 합성 (범위 0 → nominal 과 동일, 단축 delta → 예상 pose)
  C2 frame-wise 갱신 (연속 프레임에서 delta 가 실제로 변한다)
  C3 update 게이팅 (update() 를 부르지 않은 프레임엔 상태·pose 불변)
  C4 bounds (10,000 프레임 × 3 cam × 16 env 전부 설정 범위 안)
  C5 temporal correlation (smooth 가 iid 보다 프레임간 변화 작고 autocorr 높다)
  C6 reset isolation (리셋한 env 만 초기화, 나머지 bit-identical)
  C7 no accumulation + quaternion norm (10k 프레임 후에도 nominal 기준 범위 내, |q|=1)

⚠ ``isaaclab.utils.math`` 는 pxr 을 요구해서 SimulationApp 부팅이 필요하다(호스트 venv 에
isaaclab 없음). 그래서 이 검증기는 isaac-sim 컨테이너에서 ``--headless`` 로 돈다:

  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --name camdr-validate \\
      isaac-sim python /workspace/scripts/contract/validate_camera_extrinsic_dr.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16, help="검증용 가상 env 수")
parser.add_argument("--frames", type=int, default=10_000, help="bounds/누적 검사 프레임 수")
parser.add_argument("--csv", default=None, help="첫 200 프레임 delta 궤적 CSV 경로(옵션)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

import math  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

from sim_to_real.utils.domain_randomization import (  # noqa: E402
    CAMERA_DR_KEYS,
    CameraExtrinsicDR,
    CameraExtrinsicDRCfg,
    CameraExtrinsicDRTermCfg,
)

# probe(2026-07-30) 로 읽은 실제 prim local pose — OpenGL 카메라 규약, quat wxyz.
NOMINAL = {
    "top": ((-0.17, 0.77, 1.05), (-0.085197, -0.052698, 0.523579, 0.846067)),
    "wrist": ((0.0, 0.045, -0.04), (0.967043, -0.254611, 0.0, 0.0)),
    "front": ((-0.045, 0.0, 0.025), (-0.5, 0.5, -0.5, 0.5)),
}


class _StubView:
    """XFormPrim view 대역 — local pose 를 들고 있고 write 를 기록한다."""

    def __init__(self, pos: torch.Tensor, quat: torch.Tensor):
        self.pos, self.quat = pos.clone(), quat.clone()
        self.writes = 0

    def get_local_poses(self):
        return self.pos.clone(), self.quat.clone()

    def set_local_poses(self, translations=None, orientations=None, indices=None):
        self.writes += 1
        idx = slice(None) if indices is None else torch.as_tensor(indices, dtype=torch.long)
        if translations is not None:
            self.pos[idx] = translations.to(self.pos.dtype)
        if orientations is not None:
            self.quat[idx] = orientations.to(self.quat.dtype)


def make_env(num_envs: int, cfg: CameraExtrinsicDRCfg, keys=CAMERA_DR_KEYS):
    """CameraExtrinsicDR 이 실제로 만지는 최소 표면만 갖춘 가상 env."""
    sensors = {}
    for key in keys:
        pos, quat = NOMINAL[key]
        sensors[f"{key}_camera"] = SimpleNamespace(_view=_StubView(
            torch.tensor(pos).repeat(num_envs, 1),
            torch.tensor(quat).repeat(num_envs, 1),
        ))
    return SimpleNamespace(
        device="cpu",
        num_envs=num_envs,
        cfg=SimpleNamespace(camera_extrinsic_dr=cfg),
        scene=SimpleNamespace(sensors=sensors),
    )


def term(trans_m=0.0, rot_deg=0.0, jit_m=0.0, jit_deg=0.0, alpha=0.9) -> CameraExtrinsicDRTermCfg:
    return CameraExtrinsicDRTermCfg(
        bias_trans_m=(trans_m,) * 3,
        bias_rot_deg=(rot_deg,) * 3,
        jitter_trans_m=(jit_m,) * 3,
        jitter_rot_deg=(jit_deg,) * 3,
        jitter_trans_alpha=alpha,
        jitter_rot_alpha=alpha,
    )


def cfg_uniform(**kw) -> CameraExtrinsicDRCfg:
    """세 카메라에 같은 범위를 주는 cfg(항목별 독립성 검사를 단순하게)."""
    mode = kw.pop("temporal_mode", "smooth_correlated")
    base = kw.pop("base", {})
    return CameraExtrinsicDRCfg(
        temporal_mode=mode,
        top=term(**kw), wrist=term(**kw), front=term(**kw), **base,
    )


results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}", flush=True)


# ---------------------------------------------------------------------------
# C1 — zero range identity + 해석적 합성
# ---------------------------------------------------------------------------
cfg = cfg_uniform()                                   # 전 범위 0
env = make_env(4, cfg)
dr = CameraExtrinsicDR(env, cfg)
dr.reset(None)
for _ in range(50):
    dr.update()
pos, quat = dr.randomized_local_poses()
max_dp = max_dq = 0.0
for c, key in enumerate(dr.keys):
    p0, q0 = NOMINAL[key]
    max_dp = max(max_dp, float((pos[c] - torch.tensor(p0)).abs().max()))
    max_dq = max(max_dq, float((quat[c] - torch.tensor(q0)).abs().max()))
check("C1a zero-range identity", max_dp < 1e-6 and max_dq < 1e-6,
      f"max |Δpos|={max_dp:.2e} m, max |Δquat|={max_dq:.2e}")

# 해석적: nominal = identity, delta = +x 5mm & y 축 90° → pose 를 손으로 계산한 값과 대조.
cfg = cfg_uniform()
env = make_env(1, cfg, keys=("top",))
env.scene.sensors["top_camera"]._view = _StubView(torch.zeros(1, 3),
                                                  torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
dr = CameraExtrinsicDR(env, cfg)
dr._delta[:] = 0.0
dr._bias[:] = 0.0
dr._delta[0, 0, 0] = 0.005                             # +x 5 mm
dr._delta[0, 0, 4] = math.pi / 2                       # y 축(=up) 90°
pos, quat = dr.randomized_local_poses()
exp_pos = torch.tensor([0.005, 0.0, 0.0])
exp_quat = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
check("C1b analytic composition",
      float((pos[0, 0] - exp_pos).abs().max()) < 1e-6
      and float((quat[0, 0] - exp_quat).abs().max()) < 1e-6,
      f"pos={pos[0, 0].tolist()} quat={quat[0, 0].tolist()}")

# ---------------------------------------------------------------------------
# C2 — frame-wise 갱신
# ---------------------------------------------------------------------------
cfg = cfg_uniform(jit_m=0.003, jit_deg=0.4, alpha=0.9)
env = make_env(8, cfg)
dr = CameraExtrinsicDR(env, cfg)
dr.reset(None)
seq = []
for _ in range(20):
    dr.update()
    seq.append(dr._delta.clone())
diffs = torch.stack([(seq[i + 1] - seq[i]).abs().max() for i in range(len(seq) - 1)])
check("C2 frame-wise update", bool((diffs > 0).all()),
      f"연속 프레임 최소 변화={float(diffs.min()):.2e}")

# ---------------------------------------------------------------------------
# C3 — update 게이팅 (부르지 않으면 아무것도 안 바뀐다)
# ---------------------------------------------------------------------------
# ★이 프로젝트는 physics 120 Hz / control·render·camera 가 모두 30 Hz 같은 tick 이라
#   "카메라 프레임이 아닌 physics step" 이 존재하지 않는다. 따라서 원본 요구(카메라 프레임이
#   아닌 step 에선 갱신 금지)의 등가 검사 = update() 호출 없이는 상태·pose 가 불변인지.
before_delta = dr._delta.clone()
before_pos = dr._cams[0]._view.pos.clone()
writes_before = dr._cams[0]._view.writes
frame_before = dr._frame
for _ in range(4):                    # obs manager 가 pose 를 여러 번 읽어도 상태 불변이어야
    dr.randomized_local_poses()
idle_ok = (torch.equal(before_delta, dr._delta)
           and torch.equal(before_pos, dr._cams[0]._view.pos)
           and writes_before == dr._cams[0]._view.writes
           and frame_before == dr._frame)
dr.update()                           # update 1회 = 프레임 1개 = 카메라당 write 1회
step_ok = (dr._frame == frame_before + 1
           and dr._cams[0]._view.writes == writes_before + 1
           and not torch.equal(before_delta, dr._delta))
check("C3 update gating", idle_ok and step_ok,
      f"read-only 호출 불변={idle_ok}, update 1회당 write 1회={step_ok}")

# ---------------------------------------------------------------------------
# C4 / C7 — bounds & no accumulation (10k 프레임)
# ---------------------------------------------------------------------------
cfg = cfg_uniform(trans_m=0.015, rot_deg=1.5, jit_m=0.003, jit_deg=0.4, alpha=0.9)
env = make_env(args.num_envs, cfg)
dr = CameraExtrinsicDR(env, cfg)
dr.reset(None)
half_bias = dr._half_bias
half_jit = dr._half_jit
worst = torch.zeros(6)
quat_norm_err = 0.0
for f in range(args.frames):
    dr.update()
    total = (dr._bias + dr._delta).abs().amax(dim=(0, 1))
    worst = torch.maximum(worst, total)
    if f % 500 == 0:
        _, q = dr.randomized_local_poses()
        quat_norm_err = max(quat_norm_err, float((q.norm(dim=-1) - 1.0).abs().max()))
limit = (half_bias + half_jit).amax(dim=(0, 1))
check("C4 bounds (10k frames)", bool((worst <= limit + 1e-9).all()),
      f"worst={[round(float(v), 6) for v in worst]} limit={[round(float(v), 6) for v in limit]}")
# 누적이 있으면 위 bound 를 반드시 깬다(delta 가 bounded 인데 pose 가 발산할 수는 없다).
# pose 쪽도 직접 확인: nominal 대비 이동량이 (bias+jit) translation 한계 이내.
pos, quat = dr.randomized_local_poses()
max_shift = float((pos - dr._pos_nom).norm(dim=-1).max())
trans_limit = float((half_bias + half_jit)[..., :3].norm(dim=-1).max())
check("C7 no accumulation + |q|=1",
      max_shift <= trans_limit + 1e-9 and quat_norm_err < 1e-6,
      f"max |Δpos|={max_shift * 1000:.3f} mm ≤ {trans_limit * 1000:.3f} mm, "
      f"quat norm err={quat_norm_err:.2e}")

# ---------------------------------------------------------------------------
# C5 — temporal correlation: smooth vs iid
# ---------------------------------------------------------------------------
def rollout(mode: str, frames: int = 2000):
    cfg = cfg_uniform(jit_m=0.003, jit_deg=0.4, alpha=0.9, temporal_mode=mode,
                      base={"use_episode_bias": False})
    env = make_env(8, cfg)
    dr = CameraExtrinsicDR(env, cfg)
    dr.reset(None)
    out = []
    for _ in range(frames):
        dr.update()
        out.append(dr._delta.clone())
    return torch.stack(out)          # (T, C, N, 6)


def step_norm(seq: torch.Tensor, sl: slice) -> float:
    return float((seq[1:, ..., sl] - seq[:-1, ..., sl]).norm(dim=-1).mean())


def autocorr(seq: torch.Tensor) -> float:
    x = seq.reshape(seq.shape[0], -1)
    x = x - x.mean(dim=0, keepdim=True)
    num = (x[1:] * x[:-1]).mean()
    den = (x * x).mean()
    return float(num / den)


smooth, iid = rollout("smooth_correlated"), rollout("iid")
s_t, i_t = step_norm(smooth, slice(0, 3)), step_norm(iid, slice(0, 3))
s_r, i_r = step_norm(smooth, slice(3, 6)), step_norm(iid, slice(3, 6))
s_a, i_a = autocorr(smooth), autocorr(iid)
check("C5 temporal correlation", s_t < i_t and s_r < i_r and s_a > i_a,
      f"frame-to-frame trans {s_t * 1000:.4f} vs {i_t * 1000:.4f} mm · "
      f"rot {math.degrees(s_r):.4f} vs {math.degrees(i_r):.4f} deg · "
      f"autocorr {s_a:.3f} vs {i_a:.3f}")

# ---------------------------------------------------------------------------
# C6 — reset isolation
# ---------------------------------------------------------------------------
cfg = cfg_uniform(trans_m=0.015, rot_deg=1.5, jit_m=0.003, jit_deg=0.4)
env = make_env(8, cfg)
dr = CameraExtrinsicDR(env, cfg)
dr.reset(None)
for _ in range(10):
    dr.update()
bias_before, delta_before = dr._bias.clone(), dr._delta.clone()
reset_ids = torch.tensor([1, 5])
keep = torch.tensor([0, 2, 3, 4, 6, 7])
dr.reset(reset_ids)
untouched = torch.equal(dr._bias[:, keep], bias_before[:, keep]) and \
    torch.equal(dr._delta[:, keep], delta_before[:, keep])
changed = bool((dr._bias[:, reset_ids] != bias_before[:, reset_ids]).any())
check("C6 reset isolation", untouched and changed,
      f"미리셋 env 불변={untouched}, 리셋 env bias 재추첨={changed}")

# 카메라·env 독립성: 같은 프레임에서 카메라간·env간 값이 서로 다른가(공유 샘플 금지).
cross_cam = float((smooth[:, 0] - smooth[:, 1]).abs().max())
cross_env = float((smooth[:, :, 0] - smooth[:, :, 1]).abs().max())
check("C6b camera/env independence", cross_cam > 0 and cross_env > 0,
      f"cam0 vs cam1 max diff={cross_cam:.2e}, env0 vs env1 max diff={cross_env:.2e}")

# ---------------------------------------------------------------------------
# 분포 통계 (§43) + 궤적 CSV
# ---------------------------------------------------------------------------
# ★프로덕션 기본 cfg(카메라별 실제 범위)로 따로 롤아웃한다. 위 검사들은 항목을 단순화하려고
# 세 카메라에 같은 범위를 줬으므로 그 통계·궤적을 문서 근거로 쓰면 오해를 부른다.
prod_cfg = CameraExtrinsicDRCfg()                     # = 03_ENV_SPEC §11.6 표
prod = CameraExtrinsicDR(make_env(8, prod_cfg), prod_cfg)
prod.reset(None)
traj = []
for _ in range(1000):
    prod.update()
    traj.append((prod._bias + prod._delta)[:, 0].clone())      # env 0, bias + jitter
prod_seq = torch.stack(traj)                                    # (T, C, 6)

print("\n[stats] 프로덕션 기본 cfg — bias + jitter, env 0, T=1000", flush=True)
for c, key in enumerate(prod.keys):
    s = prod_seq[:, c]
    lim = (prod._half_bias + prod._half_jit)[c, 0]
    print(f"  {key:<6} trans mm mean={[round(float(v) * 1000, 3) for v in s[:, :3].mean(0)]} "
          f"std={[round(float(v) * 1000, 3) for v in s[:, :3].std(0)]} "
          f"|max|={float(s[:, :3].abs().max()) * 1000:.3f} ≤ {float(lim[:3].max()) * 1000:.3f} · "
          f"rot deg std={[round(math.degrees(float(v)), 4) for v in s[:, 3:].std(0)]} "
          f"|max|={math.degrees(float(s[:, 3:].abs().max())):.4f} ≤ "
          f"{math.degrees(float(lim[3:].max())):.4f}", flush=True)

print("\n[stats] smooth vs iid (동일 범위 대조군, C=3, N=8, T=2000)", flush=True)
for c, key in enumerate(("top", "wrist", "front")):
    s = smooth[:, c].reshape(-1, 6)
    print(f"  {key:<6} jitter trans mm std={[round(float(v) * 1000, 4) for v in s[:, :3].std(0)]} "
          f"rot deg std={[round(math.degrees(float(v)), 4) for v in s[:, 3:].std(0)]}", flush=True)

if args.csv:
    import csv as _csv
    with open(args.csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["frame", "camera", "env", "dx_m", "dy_m", "dz_m",
                    "drx_deg", "dry_deg", "drz_deg"])
        for f, row in enumerate(traj[:200]):
            for c, key in enumerate(prod.keys):
                v = row[c].tolist()
                w.writerow([f, key, 0, *[round(x, 8) for x in v[:3]],
                            *[round(math.degrees(x), 6) for x in v[3:]]])
    print(f"[csv] {args.csv} (200 frames, env 0, 프로덕션 기본 cfg)", flush=True)

passed = sum(results.values())
print(f"\n[result] {passed}/{len(results)} PASS", flush=True)
simulation_app.close()
raise SystemExit(0 if passed == len(results) else 1)
