"""카메라 extrinsic DR **in-sim** 검증 — 실제 씬·렌더에서 동작·유계·부착 유지 + 성능.

상태기계 계약(수식·유계·리셋 격리)은 CPU 검증기
``validate_camera_extrinsic_dr.py`` 가 본다. 여기서만 볼 수 있는 것:

  S1 활성화       : DR-on 에서 prim local pose 가 nominal 에서 실제로 벗어난다(프레임마다 갱신).
                    = USD xform write 가 Fabric 렌더 파이프라인에 도달하는지(09_TACIT §15.2).
  S2 유계         : 전 프레임 |Δpos| ≤ (bias+jitter) 한계, 회전각 ≤ 축별 합, |q|=1.
  S3 부착 유지    : wrist/front 의 **world** 카메라 위치 == T_world_parent(t) ⊗ local(t)
                    (articulation 링크를 계속 따라간다 = parent-child 안 깨짐, 09_TACIT §15.3)
  S4 텔레포트 없음: 프레임간 픽셀 변화가 DR-off 대비 폭증하지 않는다.
  S5 성능         : step 당 벽시계 시간 (DR-on vs DR-off 를 따로 돌려 비교).
  + 카메라별 PNG(첫/마지막 프레임) — 시야가 살아 있는지 눈으로 확인.

IsaacLab·Isaac Sim 업그레이드 후 재실행할 회귀 검사다(pose write 경로가 조용히 무반영으로
바뀌는 종류의 파손을 잡는다).

실행 (DR-on / DR-off 각 1회, 같은 --num_envs):
  docker compose --env-file .env -f docker/docker-compose.yaml run --rm --name camdr-sim isaac-sim \\
    python /workspace/scripts/contract/validate_camera_extrinsic_dr_sim.py \\
    --headless --enable_cameras --num_envs 16 --camera_dr on
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SimToReal-SO101-PickCube-DR-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--camera_dr", choices=("on", "off"), default="on")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--out", default="/workspace/scratch/camera-dr-sim",
                    help="PNG·JSON 산출물 디렉터리(임시물 → scratch)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_LAUNCHER_KEYS = {"headless", "livestream", "enable_cameras", "device", "kit_args",
                  "experience", "rendering_mode"}
app_launcher = AppLauncher({k: v for k, v in vars(args).items() if k in _LAUNCHER_KEYS})
simulation_app = app_launcher.app

import faulthandler  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import sim_to_real.tasks  # noqa: F401,E402

faulthandler.enable()

# wrist/front 의 부모 링크 (prim path 의 부모 = articulation body 이름)
PARENT_BODY = {"wrist": "gripper", "front": "shoulder"}


def save_png(arr: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(arr.astype(np.uint8)).save(path)
    except Exception as exc:                                     # PIL 없으면 npy 로
        np.save(path.with_suffix(".npy"), arr.astype(np.uint8))
        print(f"[camdr-sim] PNG 저장 실패({exc}) → npy", flush=True)


def main() -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.camera_dr == "off":
        env_cfg.camera_extrinsic_dr.enabled = False
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    hold = env.action_manager.action.clone()
    robot = env.scene["robot"]
    keys = ("top", "wrist", "front")

    for _ in range(args.warmup):
        env.step(hold)

    dr = getattr(env, "_camera_extrinsic_dr", None)
    if args.camera_dr == "on" and dr is None:
        print("[camdr-sim] FAIL — camera_extrinsic_dr 상태가 생성되지 않았다", flush=True)
        return 1

    stats = {k: {"max_dpos_m": 0.0, "max_drot_deg": 0.0, "max_qnorm_err": 0.0,
                 "max_attach_err_m": 0.0, "pix_step_diff": []} for k in keys}
    prev_img: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()

    for step in range(args.steps):
        env.step(hold)
        for c, key in enumerate(keys):
            cam = env.scene.sensors[f"{key}_camera"]
            pos_l, quat_l = cam._view.get_local_poses()
            s = stats[key]
            if dr is not None:
                nom_p, nom_q = dr._pos_nom[c], dr._quat_nom[c]
                s["max_dpos_m"] = max(s["max_dpos_m"], float((pos_l - nom_p).norm(dim=-1).max()))
                # ★componentwise diff 로 재면 안 된다: USD 는 xformOp:orient 를 w>0 정규형으로
                # 돌려줄 수 있어(front nominal 은 w=-0.5) 같은 회전인데 diff≈1 로 보인다.
                # 회전 오차는 두 quat 사이 각도로 잰다(부호 무시).
                dot = (quat_l * nom_q).sum(dim=-1).abs().clamp(max=1.0)
                s["max_drot_deg"] = max(s["max_drot_deg"],
                                        float(torch.rad2deg(2.0 * torch.acos(dot)).max()))
                s["max_qnorm_err"] = max(s["max_qnorm_err"],
                                         float((quat_l.norm(dim=-1) - 1).abs().max()))
            # S3 부착: world 카메라 위치 == parent world ⊗ local (position only — 회전은
            # opengl↔world convention 변환이 끼어 비교가 지저분해진다)
            body = PARENT_BODY.get(key)
            if body is not None:
                bi = robot.body_names.index(body)
                p_w = robot.data.body_pos_w[:, bi]
                q_w = robot.data.body_quat_w[:, bi]
                expect = p_w + math_utils.quat_apply(q_w, pos_l.to(p_w.device))
                got = cam.data.pos_w
                s["max_attach_err_m"] = max(s["max_attach_err_m"],
                                            float((expect - got).norm(dim=-1).max()))
            img = env.obs_buf["images"][key][0].float().cpu().numpy()
            if key in prev_img:
                s["pix_step_diff"].append(float(np.abs(img - prev_img[key]).mean()))
            prev_img[key] = img
            if step in (0, args.steps - 1):
                save_png(img, out / f"camdr_{key}_{args.camera_dr}_step{step:03d}.png")

    dt = (time.perf_counter() - t0) / args.steps

    limit = {}
    if dr is not None:
        half = (dr._half_bias + dr._half_jit)                    # (C,1,6)
        for c, key in enumerate(keys):
            limit[key] = {
                "trans_m": float(half[c, 0, :3].norm()),
                "rot_deg_sum": float(torch.rad2deg(half[c, 0, 3:]).sum()),
            }

    ok = True
    print(f"\n[camdr-sim] camera_dr={args.camera_dr} num_envs={args.num_envs} "
          f"steps={args.steps} → {dt * 1000:.1f} ms/step ({1 / dt:.2f} step/s)", flush=True)
    for key in keys:
        s = stats[key]
        d = np.array(s["pix_step_diff"])
        line = (f"[camdr-sim] {key:<6} max|Δpos|={s['max_dpos_m'] * 1000:7.3f} mm "
                f"max|Δrot|={s['max_drot_deg']:6.3f}° qnorm_err={s['max_qnorm_err']:.2e} "
                f"attach_err={s['max_attach_err_m'] * 1000:6.3f} mm "
                f"pixΔ mean={d.mean():.4f} max={d.max():.4f}")
        if key in limit:
            # 회전 한계: 축별 half-range 3개가 동시에 최대일 때의 합성각 상한(보수적으로 합).
            rot_limit = limit[key]["rot_deg_sum"]
            within = (s["max_dpos_m"] <= limit[key]["trans_m"] + 1e-6
                      and s["max_drot_deg"] <= rot_limit + 1e-3)
            ok &= within
            line += (f" | 한계 {limit[key]['trans_m'] * 1000:.3f} mm / {rot_limit:.3f}° "
                     f"{'OK' if within else '초과!'}")
        if s["max_attach_err_m"] > 0.002:                        # 2 mm 초과 = 부착 깨짐 의심
            ok = False
            line += " | ★attach FAIL"
        print(line, flush=True)
        if args.camera_dr == "on" and s["max_dpos_m"] <= 0.0:
            ok = False
            print(f"[camdr-sim] {key} — DR-on 인데 pose 가 nominal 그대로다", flush=True)

    (out / f"camdr_{args.camera_dr}_{args.num_envs}env.json").write_text(json.dumps(
        {"ms_per_step": dt * 1000, "num_envs": args.num_envs, "camera_dr": args.camera_dr,
         "steps": args.steps, "limit": limit,
         "stats": {k: {kk: (vv if not isinstance(vv, list) else
                            {"mean": float(np.mean(vv)), "max": float(np.max(vv))})
                       for kk, vv in v.items()} for k, v in stats.items()}}, indent=2))
    print(f"[camdr-sim] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    env.close()
    return 0 if ok else 1


try:
    code = main()
finally:
    simulation_app.close()
raise SystemExit(code)
